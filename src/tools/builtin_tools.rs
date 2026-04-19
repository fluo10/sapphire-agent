use crate::provider::ToolSpec;
use crate::tools::{Tool, ToolSet};
use anyhow::{Context, Result};
use async_trait::async_trait;
use sapphire_workspace::WorkspaceState;
use serde_json::json;
use std::path::PathBuf;
use std::sync::{Arc, Mutex, Weak};

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

fn expand_path(path_str: &str) -> PathBuf {
    PathBuf::from(shellexpand::tilde(path_str).as_ref())
}

/// Truncate output to at most 50 000 chars, keeping head + tail.
fn truncate_output(s: &str) -> String {
    const MAX: usize = 50_000;
    const HEAD: usize = 20_000;
    const TAIL: usize = 30_000;

    if s.len() <= MAX {
        return s.to_string();
    }
    let head_end = s.floor_char_boundary(HEAD);
    let tail_start = s.floor_char_boundary(s.len() - TAIL);
    format!(
        "{}\n\n[... {} chars truncated ...]\n\n{}",
        &s[..head_end],
        s.len() - HEAD - TAIL,
        &s[tail_start..]
    )
}

// ---------------------------------------------------------------------------
// file_read
// ---------------------------------------------------------------------------

pub struct FileReadTool {
    state: Arc<Mutex<WorkspaceState>>,
    spec: ToolSpec,
}

impl FileReadTool {
    pub fn new(state: Arc<Mutex<WorkspaceState>>) -> Self {
        Self {
            state,
            spec: ToolSpec {
                name: "file_read".into(),
                description:
                    "Read a file with optional line-based pagination. \
                    Accepts absolute paths, ~/... paths, or workspace-relative paths \
                    (resolved against the workspace root). \
                    Returns lines prefixed with their 1-indexed line number in 'N|content' format. \
                    Use offset and limit for large files. \
                    Cannot read binary files or device paths (/dev/, /proc/)."
                        .into(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "File path — absolute, ~/..., or relative to the workspace root."
                        },
                        "offset": {
                            "type": "integer",
                            "description": "1-indexed line number to start reading from (default: 1).",
                            "default": 1,
                            "minimum": 1
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of lines to read (default: 500, max: 2000).",
                            "default": 500,
                            "maximum": 2000
                        }
                    },
                    "required": ["path"]
                }),
            },
        }
    }
}

#[async_trait]
impl Tool for FileReadTool {
    fn spec(&self) -> &ToolSpec {
        &self.spec
    }

    async fn execute(&self, input: &serde_json::Value) -> Result<String> {
        let path_str = input["path"].as_str().context("missing 'path'")?;
        let offset = input["offset"].as_u64().unwrap_or(1).max(1) as usize;
        let limit = input["limit"].as_u64().unwrap_or(500).min(2000) as usize;

        let path = expand_path(path_str);
        let path_abs = path.to_string_lossy();

        if path_abs.starts_with("/dev/") || path_abs.starts_with("/proc/") {
            anyhow::bail!("Reading device or proc paths is not allowed.");
        }

        let content = self
            .state
            .lock()
            .expect("WorkspaceState mutex poisoned")
            .read_file(&path)
            .with_context(|| format!("Failed to read '{}'", path.display()))?;

        let lines: Vec<&str> = content.lines().collect();
        let total_lines = lines.len();
        let start = offset.saturating_sub(1);

        if start >= total_lines && total_lines > 0 {
            anyhow::bail!(
                "offset {} exceeds file length ({} lines)",
                offset,
                total_lines
            );
        }

        let end = (start + limit).min(total_lines);
        let mut result = lines[start..end]
            .iter()
            .enumerate()
            .map(|(i, l)| format!("{}|{}", start + i + 1, l))
            .collect::<Vec<_>>()
            .join("\n");

        if end < total_lines {
            result.push_str(&format!(
                "\n[{} more lines — use offset={} to continue]",
                total_lines - end,
                end + 1
            ));
        }

        Ok(result)
    }
}

// ---------------------------------------------------------------------------
// file_write
// ---------------------------------------------------------------------------

static SENSITIVE_PREFIXES: &[&str] = &[
    "/etc/",
    "/boot/",
    "/usr/lib/",
    "/usr/bin/",
    "/usr/sbin/",
    "/bin/",
    "/sbin/",
    "/sys/",
    "/proc/",
    "/run/docker.sock",
    "/var/run/docker.sock",
];

pub struct FileWriteTool {
    state: Arc<Mutex<WorkspaceState>>,
    spec: ToolSpec,
}

impl FileWriteTool {
    pub fn new(state: Arc<Mutex<WorkspaceState>>) -> Self {
        Self {
            state,
            spec: ToolSpec {
                name: "file_write".into(),
                description: "Write content to a file, completely replacing its existing content. \
                    Accepts absolute paths, ~/... paths, or workspace-relative paths \
                    (resolved against the workspace root). \
                    Creates the file and any missing parent directories automatically. \
                    When the target file is inside the workspace, the search index and git sync \
                    are updated automatically. \
                    Refuses writes to sensitive system paths (/etc, /boot, /bin, etc.).".into(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "File path — absolute, ~/..., or relative to the workspace root."
                        },
                        "content": {
                            "type": "string",
                            "description": "Complete content to write to the file (overwrites entirely)."
                        }
                    },
                    "required": ["path", "content"]
                }),
            },
        }
    }
}

#[async_trait]
impl Tool for FileWriteTool {
    fn spec(&self) -> &ToolSpec {
        &self.spec
    }

    async fn execute(&self, input: &serde_json::Value) -> Result<String> {
        let path_str = input["path"].as_str().context("missing 'path'")?;
        let content = input["content"].as_str().context("missing 'content'")?;

        let path = expand_path(path_str);
        let path_abs = path.to_string_lossy().to_string();

        for prefix in SENSITIVE_PREFIXES {
            if path_abs.starts_with(prefix) || &path_abs == prefix {
                anyhow::bail!(
                    "Writing to '{}' is not allowed (sensitive system path).",
                    path_abs
                );
            }
        }

        self.state
            .lock()
            .expect("WorkspaceState mutex poisoned")
            .write_file(&path, content)
            .with_context(|| format!("Failed to write '{}'", path.display()))?;

        Ok(format!(
            "Written: {} ({} bytes)",
            path.display(),
            content.len()
        ))
    }
}

// ---------------------------------------------------------------------------
// file_delete
// ---------------------------------------------------------------------------

pub struct FileDeleteTool {
    state: Arc<Mutex<WorkspaceState>>,
    spec: ToolSpec,
}

impl FileDeleteTool {
    pub fn new(state: Arc<Mutex<WorkspaceState>>) -> Self {
        Self {
            state,
            spec: ToolSpec {
                name: "file_delete".into(),
                description: "Delete a file from the filesystem. \
                    Accepts absolute paths, ~/... paths, or workspace-relative paths \
                    (resolved against the workspace root). \
                    When the file is inside the workspace, it is also removed from the search index and git sync \
                    automatically. \
                    Cannot delete directories.".into(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "File path — absolute, ~/..., or relative to the workspace root."
                        }
                    },
                    "required": ["path"]
                }),
            },
        }
    }
}

#[async_trait]
impl Tool for FileDeleteTool {
    fn spec(&self) -> &ToolSpec {
        &self.spec
    }

    async fn execute(&self, input: &serde_json::Value) -> Result<String> {
        let path_str = input["path"].as_str().context("missing 'path'")?;
        let path = expand_path(path_str);

        self.state
            .lock()
            .expect("WorkspaceState mutex poisoned")
            .delete_file(&path)
            .with_context(|| format!("Failed to delete '{}'", path.display()))?;

        Ok(format!("Deleted: {}", path.display()))
    }
}

// ---------------------------------------------------------------------------
// file_append
// ---------------------------------------------------------------------------

pub struct FileAppendTool {
    state: Arc<Mutex<WorkspaceState>>,
    spec: ToolSpec,
}

impl FileAppendTool {
    pub fn new(state: Arc<Mutex<WorkspaceState>>) -> Self {
        Self {
            state,
            spec: ToolSpec {
                name: "file_append".into(),
                description: "Append content to the end of a file, creating it if missing. \
                    Accepts absolute paths, ~/... paths, or workspace-relative paths \
                    (resolved against the workspace root). \
                    Creates any missing parent directories automatically. \
                    When the target file is inside the workspace, the search index and git sync \
                    are updated automatically. \
                    Refuses writes to sensitive system paths (/etc, /boot, /bin, etc.).".into(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "File path — absolute, ~/..., or relative to the workspace root."
                        },
                        "content": {
                            "type": "string",
                            "description": "Content to append to the end of the file."
                        }
                    },
                    "required": ["path", "content"]
                }),
            },
        }
    }
}

#[async_trait]
impl Tool for FileAppendTool {
    fn spec(&self) -> &ToolSpec {
        &self.spec
    }

    async fn execute(&self, input: &serde_json::Value) -> Result<String> {
        let path_str = input["path"].as_str().context("missing 'path'")?;
        let content = input["content"].as_str().context("missing 'content'")?;

        let path = expand_path(path_str);
        let path_abs = path.to_string_lossy().to_string();

        for prefix in SENSITIVE_PREFIXES {
            if path_abs.starts_with(prefix) || &path_abs == prefix {
                anyhow::bail!(
                    "Writing to '{}' is not allowed (sensitive system path).",
                    path_abs
                );
            }
        }

        self.state
            .lock()
            .expect("WorkspaceState mutex poisoned")
            .append_file(&path, content)
            .with_context(|| format!("Failed to append to '{}'", path.display()))?;

        Ok(format!(
            "Appended: {} (+{} bytes)",
            path.display(),
            content.len()
        ))
    }
}

// ---------------------------------------------------------------------------
// dir_list
// ---------------------------------------------------------------------------

pub struct DirListTool {
    state: Arc<Mutex<WorkspaceState>>,
    spec: ToolSpec,
}

impl DirListTool {
    pub fn new(state: Arc<Mutex<WorkspaceState>>) -> Self {
        Self {
            state,
            spec: ToolSpec {
                name: "dir_list".into(),
                description: "List the direct children of a directory (non-recursive). \
                    Accepts absolute paths, ~/... paths, or workspace-relative paths. \
                    Entries are sorted alphabetically. Directories are shown with a \
                    trailing slash. For deeper exploration, use dir_walk."
                    .into(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Directory path — absolute, ~/..., or relative to the workspace root."
                        }
                    },
                    "required": ["path"]
                }),
            },
        }
    }
}

#[async_trait]
impl Tool for DirListTool {
    fn spec(&self) -> &ToolSpec {
        &self.spec
    }

    async fn execute(&self, input: &serde_json::Value) -> Result<String> {
        let path_str = input["path"].as_str().context("missing 'path'")?;
        let path = expand_path(path_str);

        let entries = self
            .state
            .lock()
            .expect("WorkspaceState mutex poisoned")
            .list_dir(&path)
            .with_context(|| format!("Failed to list '{}'", path.display()))?;

        if entries.is_empty() {
            return Ok(format!("(empty) {}", path.display()));
        }

        let lines: Vec<String> = entries
            .iter()
            .map(|(p, is_dir)| {
                if *is_dir {
                    format!("{}/", p.display())
                } else {
                    p.display().to_string()
                }
            })
            .collect();
        Ok(lines.join("\n"))
    }
}

// ---------------------------------------------------------------------------
// dir_walk
// ---------------------------------------------------------------------------

pub struct DirWalkTool {
    state: Arc<Mutex<WorkspaceState>>,
    spec: ToolSpec,
}

impl DirWalkTool {
    pub fn new(state: Arc<Mutex<WorkspaceState>>) -> Self {
        Self {
            state,
            spec: ToolSpec {
                name: "dir_walk".into(),
                description: "Recursively list all files and directories under a path. \
                    Accepts absolute paths, ~/... paths, or workspace-relative paths. \
                    Output is a sorted flat list; directories carry a trailing slash. \
                    Bounded by max_depth (default 5) and max_entries (default 500) to \
                    avoid runaway walks into large trees."
                    .into(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Directory path — absolute, ~/..., or relative to the workspace root."
                        },
                        "max_depth": {
                            "type": "integer",
                            "description": "Maximum recursion depth (default 5, max 20). 0 = only direct children.",
                            "default": 5,
                            "minimum": 0,
                            "maximum": 20
                        },
                        "max_entries": {
                            "type": "integer",
                            "description": "Maximum number of entries to return before truncating (default 500, max 5000).",
                            "default": 500,
                            "minimum": 1,
                            "maximum": 5000
                        }
                    },
                    "required": ["path"]
                }),
            },
        }
    }
}

fn walk_recurse(
    state: &WorkspaceState,
    workspace_root: &std::path::Path,
    abs_path: &std::path::Path,
    depth: usize,
    max_depth: usize,
    max_entries: usize,
    results: &mut Vec<(PathBuf, bool)>,
) -> Result<bool> {
    let entries = state.list_dir(abs_path)?;
    for (entry_path, is_dir) in entries {
        if results.len() >= max_entries {
            return Ok(true);
        }
        results.push((entry_path.clone(), is_dir));
        if is_dir && depth < max_depth {
            let abs_next = if entry_path.is_absolute() {
                entry_path
            } else {
                workspace_root.join(&entry_path)
            };
            if walk_recurse(
                state,
                workspace_root,
                &abs_next,
                depth + 1,
                max_depth,
                max_entries,
                results,
            )? {
                return Ok(true);
            }
        }
    }
    Ok(false)
}

#[async_trait]
impl Tool for DirWalkTool {
    fn spec(&self) -> &ToolSpec {
        &self.spec
    }

    async fn execute(&self, input: &serde_json::Value) -> Result<String> {
        let path_str = input["path"].as_str().context("missing 'path'")?;
        let max_depth = input["max_depth"].as_u64().unwrap_or(5).min(20) as usize;
        let max_entries = input["max_entries"]
            .as_u64()
            .unwrap_or(500)
            .clamp(1, 5000) as usize;

        let path = expand_path(path_str);

        let state = self.state.lock().expect("WorkspaceState mutex poisoned");
        let workspace_root = state.workspace.root.clone();
        let mut results: Vec<(PathBuf, bool)> = Vec::new();
        let truncated = walk_recurse(
            &state,
            &workspace_root,
            &path,
            0,
            max_depth,
            max_entries,
            &mut results,
        )
        .with_context(|| format!("Failed to walk '{}'", path.display()))?;
        drop(state);

        if results.is_empty() {
            return Ok(format!("(empty) {}", path.display()));
        }

        let mut out: Vec<String> = results
            .iter()
            .map(|(p, is_dir)| {
                if *is_dir {
                    format!("{}/", p.display())
                } else {
                    p.display().to_string()
                }
            })
            .collect();
        if truncated {
            out.push(format!(
                "[truncated — more than {max_entries} entries; raise max_entries or narrow path]"
            ));
        }
        Ok(out.join("\n"))
    }
}

// ---------------------------------------------------------------------------
// shell
// ---------------------------------------------------------------------------

/// Default shell used when neither the `shell` parameter nor `$SHELL` is set.
const FALLBACK_SHELL: &str = "/bin/sh";

pub struct ShellTool {
    workspace_root: PathBuf,
    spec: ToolSpec,
}

impl ShellTool {
    pub fn new(workspace_root: PathBuf) -> Self {
        Self {
            workspace_root,
            spec: ToolSpec {
                name: "shell".into(),
                description: "Execute a shell command and return its output. \
                    Returns stdout, stderr, and exit code. \
                    The default working directory is the workspace root. \
                    By default the command runs under the shell named by the \
                    `$SHELL` environment variable (falling back to `/bin/sh`); \
                    override per call with the `shell` parameter (e.g. `bash`, \
                    `zsh`, `fish`, or an absolute path). \
                    Use the timeout parameter for long-running commands (default 60 s, max 600 s). \
                    Not suitable for interactive commands or persistent daemons."
                    .into(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "The shell command to execute (run via `<shell> -c`)."
                        },
                        "shell": {
                            "type": "string",
                            "description": "Shell executable to run the command with — a name resolved via PATH (e.g. `bash`, `zsh`, `fish`) or an absolute path. Defaults to `$SHELL`, or `/bin/sh` if unset."
                        },
                        "timeout": {
                            "type": "integer",
                            "description": "Max seconds to wait before killing the process (default: 60, max: 600).",
                            "default": 60,
                            "minimum": 1,
                            "maximum": 600
                        },
                        "workdir": {
                            "type": "string",
                            "description": "Working directory (absolute or ~/... path). Defaults to the workspace root."
                        }
                    },
                    "required": ["command"]
                }),
            },
        }
    }
}

#[async_trait]
impl Tool for ShellTool {
    fn spec(&self) -> &ToolSpec {
        &self.spec
    }

    async fn execute(&self, input: &serde_json::Value) -> Result<String> {
        use std::time::Duration;
        use tokio::process::Command;

        let command = input["command"].as_str().context("missing 'command'")?;
        let timeout_secs = input["timeout"].as_u64().unwrap_or(60).min(600);
        let workdir = input["workdir"]
            .as_str()
            .map(expand_path)
            .unwrap_or_else(|| self.workspace_root.clone());

        let shell = input["shell"]
            .as_str()
            .map(|s| s.to_string())
            .or_else(|| std::env::var("SHELL").ok())
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| FALLBACK_SHELL.to_string());

        let mut cmd = Command::new(&shell);
        cmd.arg("-c")
            .arg(command)
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .current_dir(&workdir);

        let child = cmd
            .spawn()
            .with_context(|| format!("Failed to spawn shell '{shell}'"))?;
        let pid = child.id();

        let result =
            tokio::time::timeout(Duration::from_secs(timeout_secs), child.wait_with_output()).await;

        match result {
            Ok(Ok(output)) => {
                let stdout = truncate_output(&String::from_utf8_lossy(&output.stdout));
                let stderr = truncate_output(&String::from_utf8_lossy(&output.stderr));
                let exit_code = output.status.code().unwrap_or(-1);

                if stderr.is_empty() {
                    Ok(format!("[exit: {exit_code}]\n{stdout}"))
                } else {
                    Ok(format!(
                        "[exit: {exit_code}]\nstdout:\n{stdout}\nstderr:\n{stderr}"
                    ))
                }
            }
            Ok(Err(e)) => Err(e.into()),
            Err(_) => {
                if let Some(pid) = pid {
                    let _ = std::process::Command::new("kill")
                        .args(["-9", &pid.to_string()])
                        .output();
                }
                Ok(format!(
                    "[exit: 124]\nCommand timed out after {timeout_secs}s"
                ))
            }
        }
    }
}

// ---------------------------------------------------------------------------
// weather  (Open-Meteo — no API key required)
// ---------------------------------------------------------------------------

pub struct WeatherTool {
    spec: ToolSpec,
}

impl WeatherTool {
    pub fn new() -> Self {
        Self {
            spec: ToolSpec {
                name: "weather".into(),
                description: "Fetch a weather forecast via the Open-Meteo API \
                    (no API key required). Resolve a place by passing `location` \
                    (e.g. \"Tokyo\", \"渋谷\", \"Paris, FR\") — it is geocoded \
                    into coordinates — or specify `latitude` and `longitude` \
                    directly to skip geocoding. Returns the current conditions \
                    and a daily forecast (min/max temperature, precipitation, \
                    weather code) for the next `days` days (default 3, max 7). \
                    Temperatures are in Celsius; timezone auto-detected from \
                    the resolved coordinates."
                    .into(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "Place name to geocode (e.g. \"Tokyo\", \"New York\"). Ignored if latitude and longitude are provided."
                        },
                        "latitude": {
                            "type": "number",
                            "description": "Latitude in decimal degrees (-90..90). If set, longitude is required and location is ignored."
                        },
                        "longitude": {
                            "type": "number",
                            "description": "Longitude in decimal degrees (-180..180). If set, latitude is required and location is ignored."
                        },
                        "days": {
                            "type": "integer",
                            "description": "Number of forecast days to return, starting today (default 3, max 7).",
                            "default": 3,
                            "minimum": 1,
                            "maximum": 7
                        }
                    }
                }),
            },
        }
    }
}

impl Default for WeatherTool {
    fn default() -> Self {
        Self::new()
    }
}

/// Translate a WMO weather interpretation code into a short description.
/// Reference: https://open-meteo.com/en/docs (WMO Weather interpretation codes).
fn wmo_code_description(code: i64) -> &'static str {
    match code {
        0 => "clear sky",
        1 => "mainly clear",
        2 => "partly cloudy",
        3 => "overcast",
        45 => "fog",
        48 => "depositing rime fog",
        51 => "light drizzle",
        53 => "moderate drizzle",
        55 => "dense drizzle",
        56 => "light freezing drizzle",
        57 => "dense freezing drizzle",
        61 => "slight rain",
        63 => "moderate rain",
        65 => "heavy rain",
        66 => "light freezing rain",
        67 => "heavy freezing rain",
        71 => "slight snow fall",
        73 => "moderate snow fall",
        75 => "heavy snow fall",
        77 => "snow grains",
        80 => "slight rain showers",
        81 => "moderate rain showers",
        82 => "violent rain showers",
        85 => "slight snow showers",
        86 => "heavy snow showers",
        95 => "thunderstorm",
        96 => "thunderstorm with slight hail",
        99 => "thunderstorm with heavy hail",
        _ => "unknown conditions",
    }
}

#[async_trait]
impl Tool for WeatherTool {
    fn spec(&self) -> &ToolSpec {
        &self.spec
    }

    async fn execute(&self, input: &serde_json::Value) -> Result<String> {
        let client = reqwest::Client::new();
        let days = input["days"].as_u64().unwrap_or(3).clamp(1, 7);

        // Resolve coordinates. Explicit lat/lon wins; otherwise geocode `location`.
        let (latitude, longitude, resolved_name) = match (
            input["latitude"].as_f64(),
            input["longitude"].as_f64(),
        ) {
            (Some(lat), Some(lon)) => {
                if !(-90.0..=90.0).contains(&lat) {
                    anyhow::bail!("latitude {lat} out of range (-90..90)");
                }
                if !(-180.0..=180.0).contains(&lon) {
                    anyhow::bail!("longitude {lon} out of range (-180..180)");
                }
                (lat, lon, format!("{lat:.4}, {lon:.4}"))
            }
            (Some(_), None) | (None, Some(_)) => {
                anyhow::bail!("latitude and longitude must be provided together");
            }
            (None, None) => {
                let location = input["location"]
                    .as_str()
                    .context("provide either 'location' or both 'latitude' and 'longitude'")?;

                let geo_resp = client
                    .get("https://geocoding-api.open-meteo.com/v1/search")
                    .query(&[
                        ("name", location),
                        ("count", "1"),
                        ("language", "en"),
                        ("format", "json"),
                    ])
                    .send()
                    .await
                    .context("Open-Meteo geocoding request failed")?;

                if !geo_resp.status().is_success() {
                    let status = geo_resp.status();
                    let body = geo_resp.text().await.unwrap_or_default();
                    anyhow::bail!("Open-Meteo geocoding error {status}: {body}");
                }

                let geo: serde_json::Value = geo_resp
                    .json()
                    .await
                    .context("Failed to parse Open-Meteo geocoding response")?;

                let result = geo["results"]
                    .as_array()
                    .and_then(|arr| arr.first())
                    .with_context(|| format!("No matches for location '{location}'"))?;

                let lat = result["latitude"]
                    .as_f64()
                    .context("geocoding result missing 'latitude'")?;
                let lon = result["longitude"]
                    .as_f64()
                    .context("geocoding result missing 'longitude'")?;
                let name = result["name"].as_str().unwrap_or(location);
                let admin = result["admin1"].as_str().unwrap_or("");
                let country = result["country"].as_str().unwrap_or("");
                let pretty = [name, admin, country]
                    .iter()
                    .filter(|s| !s.is_empty())
                    .copied()
                    .collect::<Vec<_>>()
                    .join(", ");
                (lat, lon, pretty)
            }
        };

        let lat_s = latitude.to_string();
        let lon_s = longitude.to_string();
        let days_s = days.to_string();

        let forecast_resp = client
            .get("https://api.open-meteo.com/v1/forecast")
            .query(&[
                ("latitude", lat_s.as_str()),
                ("longitude", lon_s.as_str()),
                ("current", "temperature_2m,relative_humidity_2m,apparent_temperature,precipitation,weather_code,wind_speed_10m"),
                (
                    "daily",
                    "weather_code,temperature_2m_max,temperature_2m_min,precipitation_sum,precipitation_probability_max",
                ),
                ("timezone", "auto"),
                ("forecast_days", days_s.as_str()),
            ])
            .send()
            .await
            .context("Open-Meteo forecast request failed")?;

        if !forecast_resp.status().is_success() {
            let status = forecast_resp.status();
            let body = forecast_resp.text().await.unwrap_or_default();
            anyhow::bail!("Open-Meteo forecast error {status}: {body}");
        }

        let data: serde_json::Value = forecast_resp
            .json()
            .await
            .context("Failed to parse Open-Meteo forecast response")?;

        let timezone = data["timezone"].as_str().unwrap_or("UTC");

        let mut out = format!("Weather for {resolved_name} ({timezone})\n");

        if let Some(current) = data.get("current") {
            let temp = current["temperature_2m"].as_f64();
            let feels = current["apparent_temperature"].as_f64();
            let humidity = current["relative_humidity_2m"].as_f64();
            let precip = current["precipitation"].as_f64();
            let wind = current["wind_speed_10m"].as_f64();
            let code = current["weather_code"].as_i64().unwrap_or(-1);
            let time = current["time"].as_str().unwrap_or("");
            out.push_str(&format!("\nCurrent ({time}): {}\n", wmo_code_description(code)));
            if let Some(t) = temp {
                out.push_str(&format!("  temp: {t:.1}°C"));
                if let Some(f) = feels {
                    out.push_str(&format!(" (feels {f:.1}°C)"));
                }
                out.push('\n');
            }
            if let Some(h) = humidity {
                out.push_str(&format!("  humidity: {h:.0}%\n"));
            }
            if let Some(p) = precip {
                out.push_str(&format!("  precipitation: {p:.1} mm\n"));
            }
            if let Some(w) = wind {
                out.push_str(&format!("  wind: {w:.1} km/h\n"));
            }
        }

        if let Some(daily) = data.get("daily") {
            let dates = daily["time"].as_array();
            let codes = daily["weather_code"].as_array();
            let tmax = daily["temperature_2m_max"].as_array();
            let tmin = daily["temperature_2m_min"].as_array();
            let psum = daily["precipitation_sum"].as_array();
            let pprob = daily["precipitation_probability_max"].as_array();
            if let Some(dates) = dates {
                out.push_str("\nForecast:\n");
                for (i, date) in dates.iter().enumerate() {
                    let date = date.as_str().unwrap_or("");
                    let code = codes
                        .and_then(|a| a.get(i))
                        .and_then(|v| v.as_i64())
                        .unwrap_or(-1);
                    let hi = tmax.and_then(|a| a.get(i)).and_then(|v| v.as_f64());
                    let lo = tmin.and_then(|a| a.get(i)).and_then(|v| v.as_f64());
                    let pp = psum.and_then(|a| a.get(i)).and_then(|v| v.as_f64());
                    let pr = pprob.and_then(|a| a.get(i)).and_then(|v| v.as_f64());
                    out.push_str(&format!("  {date}: {}", wmo_code_description(code)));
                    if let (Some(hi), Some(lo)) = (hi, lo) {
                        out.push_str(&format!(", {lo:.1}°C / {hi:.1}°C"));
                    }
                    if let Some(pp) = pp {
                        out.push_str(&format!(", precip {pp:.1} mm"));
                    }
                    if let Some(pr) = pr {
                        out.push_str(&format!(" ({pr:.0}%)"));
                    }
                    out.push('\n');
                }
            }
        }

        Ok(out)
    }
}

// ---------------------------------------------------------------------------
// web_search  (Tavily)
// ---------------------------------------------------------------------------

pub struct WebSearchTool {
    api_key: String,
    spec: ToolSpec,
}

impl WebSearchTool {
    pub fn new(api_key: String) -> Self {
        Self {
            api_key,
            spec: ToolSpec {
                name: "web_search".into(),
                description: "Search the web for up-to-date information using Tavily. \
                    Returns titles, URLs, and short content excerpts for the top results."
                    .into(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query."
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Number of results to return (default: 5, max: 10).",
                            "default": 5,
                            "maximum": 10
                        }
                    },
                    "required": ["query"]
                }),
            },
        }
    }
}

#[async_trait]
impl Tool for WebSearchTool {
    fn spec(&self) -> &ToolSpec {
        &self.spec
    }

    async fn execute(&self, input: &serde_json::Value) -> Result<String> {
        let query = input["query"].as_str().context("missing 'query'")?;
        let limit = input["limit"].as_u64().unwrap_or(5).min(10) as usize;

        let client = reqwest::Client::new();
        let resp = client
            .post("https://api.tavily.com/search")
            .json(&json!({
                "api_key": self.api_key,
                "query": query,
                "max_results": limit,
            }))
            .send()
            .await
            .context("Tavily API request failed")?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            anyhow::bail!("Tavily API error {status}: {body}");
        }

        let data: serde_json::Value = resp
            .json()
            .await
            .context("Failed to parse Tavily response")?;
        let results = data["results"]
            .as_array()
            .context("Unexpected Tavily response format (missing 'results')")?;

        if results.is_empty() {
            return Ok("No results found.".to_string());
        }

        let lines: Vec<String> = results
            .iter()
            .map(|r| {
                let title = r["title"].as_str().unwrap_or("(no title)");
                let url = r["url"].as_str().unwrap_or("");
                let content = r["content"].as_str().unwrap_or("");
                let snippet = if content.len() > 300 {
                    &content[..content.floor_char_boundary(300)]
                } else {
                    content
                };
                format!("**{title}**\n{url}\n{snippet}")
            })
            .collect();

        Ok(lines.join("\n\n"))
    }
}

// ---------------------------------------------------------------------------
// mcp_reconnect — tear down and re-establish an MCP server connection
// ---------------------------------------------------------------------------

pub struct McpReconnectTool {
    spec: ToolSpec,
    tool_set: Weak<ToolSet>,
}

impl McpReconnectTool {
    pub fn new(tool_set: Weak<ToolSet>) -> Self {
        Self {
            spec: ToolSpec {
                name: "mcp_reconnect".into(),
                description:
                    "Reconnect to a configured MCP server (stdio or HTTP) and refresh its tool list. \
                     Use this when an MCP server has crashed, disconnected, or is being restarted \
                     during testing — tools registered under `mcp__<server>__*` become usable again \
                     without restarting the agent."
                        .into(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "server": {
                            "type": "string",
                            "description": "Name of the MCP server to reconnect (as configured in tools.mcp_servers)."
                        }
                    },
                    "required": ["server"]
                }),
            },
            tool_set,
        }
    }
}

#[async_trait]
impl Tool for McpReconnectTool {
    fn spec(&self) -> &ToolSpec {
        &self.spec
    }

    async fn execute(&self, input: &serde_json::Value) -> Result<String> {
        let server = input
            .get("server")
            .and_then(|v| v.as_str())
            .context("Missing required field: server")?;

        let tool_set = self
            .tool_set
            .upgrade()
            .context("ToolSet has been dropped; cannot reconnect")?;

        let known = tool_set.mcp_server_names();
        if !known.iter().any(|n| n == server) {
            anyhow::bail!(
                "unknown MCP server '{server}'. Configured servers: [{}]",
                known.join(", ")
            );
        }

        tool_set.reconnect_mcp_server(server).await
    }
}
