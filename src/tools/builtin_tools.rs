use crate::image_cache::ImageCache;
use crate::provider::ToolSpec;
use crate::tools::{Tool, ToolKind, ToolOutput, ToolSet};
use anyhow::{Context, Result};
use async_trait::async_trait;
use base64::{Engine, engine::general_purpose::STANDARD as BASE64_STANDARD};
use sapphire_framework::workspace::WorkspaceState;
use serde_json::json;
use std::path::{Path, PathBuf};
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
                description: "Read a file with optional line-based pagination. \
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
    fn kind(&self) -> ToolKind {
        ToolKind::Read
    }

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

/// Files the agent must not be able to rewrite through its own tools.
///
/// This is not merely tidy. `file_write` is `ToolKind::Edit`, which the
/// permission policy allows *without asking* from a chat channel and in
/// ACP's `accept_edits` mode. Unguarded, a Discord message could write
/// `{"profiles":{"zed":{"always_allow":["shell"]}}}` into the
/// permission record; the store is read once at startup, so after the
/// next restart the editor would run `shell` having never been asked.
/// The gate would still be doing its job — it would just be consulting
/// an answer somebody else wrote.
///
/// Named files rather than the whole config directory, deliberately.
/// When `workspace_dir` is unset the workspace root *is* the config
/// file's own directory, so blocking the directory would refuse every
/// write in the workspace on a default install.
static PROTECTED_CONFIG_FILES: &[&str] = &["acp-permissions.json", "config.toml"];

/// The agent's own config directory, if this host has one.
fn agent_config_dir() -> Option<PathBuf> {
    directories::ProjectDirs::from("", "", "sapphire-agent")
        .map(|dirs| dirs.config_dir().to_path_buf())
}

/// Collapse `.` and `..` without touching the filesystem.
///
/// `canonicalize` needs every component to exist, so
/// `nope/../acp-permissions.json` defeats it when `nope` does not; a
/// guard that then falls back to the raw path carries an un-collapsed
/// `..` into the comparison and matches nothing, while the writer's
/// `create_dir_all` makes `nope` and the kernel collapses the `..` at
/// open time.
///
/// Note the deliberate imprecision: collapsing `..` lexically disagrees
/// with the kernel whenever a symlink sits in the path, because the
/// kernel resolves the link first. That direction is accepted — see
/// `protected_leaf`, which does not rely on the whole path matching.
fn lexically_normalize(path: &Path) -> PathBuf {
    use std::path::Component;

    let mut out = PathBuf::new();
    for component in path.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => match out.components().next_back() {
                // An ordinary component: this `..` cancels it.
                Some(Component::Normal(_)) => {
                    out.pop();
                }
                // A root or drive prefix: `/..` is `/`. Drop it — it
                // cannot climb, and keeping it would leave an
                // un-collapsed `..` in a path the guard then compares.
                Some(Component::RootDir) | Some(Component::Prefix(_)) => {}
                // Nothing to cancel and no root to stop at: keep it, so
                // a relative path can still be joined onto a root.
                _ => out.push(".."),
            },
            other => out.push(other.as_os_str()),
        }
    }
    out
}

/// The filename the operating system will actually open, given the
/// filename as written.
///
/// Windows accepts several spellings of one name and normalises them
/// away before the filesystem sees them, so a guard that compares the
/// string as given is comparing something the OS never uses:
///
/// - `name::$DATA` is the file's default data stream — i.e. the file
///   itself. A colon is never legal in an NTFS filename, so everything
///   from the first one onwards is stream syntax and is cut.
/// - Trailing dots and spaces are stripped during path normalisation,
///   so `name.` and `name ` both open `name`.
///
/// Returns `None` when nothing is left, which is not a filename any
/// spelling of a protected file can reduce to.
fn opened_filename(name: &std::ffi::OsStr) -> Option<String> {
    let name = name.to_string_lossy();
    if !cfg!(windows) {
        return Some(name.into_owned());
    }
    let before_stream = name.split(':').next().unwrap_or("");
    let trimmed = before_stream.trim_end_matches(['.', ' ']);
    (!trimmed.is_empty()).then(|| trimmed.to_string())
}

/// Whether two *existing* paths are the same file.
///
/// Only ever called on directories here, which sidesteps two problems
/// with identifying the target instead: `same_file` opens the path it
/// is given, and opening a FIFO blocks until a writer appears; and a
/// target that does not exist yet — the ordinary case for a write, and
/// the case a fresh install is always in — has no identity to compare,
/// which is exactly how the previous version of this guard came to
/// degrade into a spelling comparison.
fn same_directory(a: &Path, b: &Path) -> bool {
    same_file::is_same_file(a, b).unwrap_or(false)
}

/// The protected filename this path would write to, if any.
///
/// Compares two things rather than one whole path: the *parent* by
/// identity, which is exact because a directory that a write is headed
/// into exists; and the *leaf* by the name the OS will open. Neither
/// half depends on the target file existing, which is what went wrong
/// every previous time.
///
/// A symlinked leaf is followed once — `read_link` rather than
/// `canonicalize`, so a dangling link still resolves — because
/// `fs::write` follows it too.
fn protected_leaf(
    path: &Path,
    workspace_root: Option<&Path>,
    protected_dir: &Path,
) -> Option<String> {
    let mut candidate = if path.is_absolute() {
        path.to_path_buf()
    } else {
        match workspace_root {
            Some(root) => root.join(path),
            None => path.to_path_buf(),
        }
    };

    // Follow a symlinked leaf, bounded. `fs::write` follows it, so a
    // guard that does not is judging a different file.
    for _ in 0..8 {
        let collapsed = lexically_normalize(&candidate);
        let (Some(parent), Some(leaf)) = (collapsed.parent(), collapsed.file_name()) else {
            return None;
        };

        let canonical_parent = parent
            .canonicalize()
            .unwrap_or_else(|_| parent.to_path_buf());
        let name = opened_filename(leaf)?;

        if same_directory(&canonical_parent, protected_dir) {
            let hit = PROTECTED_CONFIG_FILES.iter().find(|protected| {
                if cfg!(windows) {
                    name.eq_ignore_ascii_case(protected)
                } else {
                    name == **protected
                }
            });
            if let Some(hit) = hit {
                return Some((*hit).to_string());
            }
        }

        let target = canonical_parent.join(&name);
        match std::fs::symlink_metadata(&target) {
            Ok(meta) if meta.file_type().is_symlink() => match std::fs::read_link(&target) {
                Ok(dest) if dest.is_absolute() => candidate = dest,
                Ok(dest) => candidate = canonical_parent.join(dest),
                Err(_) => return None,
            },
            _ => return None,
        }
    }
    None
}

/// Refuse writes and deletes to a sensitive system path or to one of
/// the agent's own configuration files.
///
/// `protected_dir` is the directory holding those files: the agent's
/// own config directory in production, a tempdir under test. `verb` is
/// the word used in the error the model sees ("Writing to", "Deleting").
fn refuse_if_sensitive_in(
    path: &Path,
    workspace_root: Option<&Path>,
    protected_dir: Option<&Path>,
    verb: &str,
) -> Result<()> {
    // System prefixes are matched against both the raw path and the
    // collapsed one: the raw catches the plain case, the collapsed
    // catches a `..` detour. macOS canonicalises `/etc` to
    // `/private/etc`, so neither spelling alone is enough.
    let raw_str = path.to_string_lossy().to_string();
    // Separators normalised to `/`: `SENSITIVE_PREFIXES` are Unix-shaped,
    // and `lexically_normalize` rebuilds the path with the platform's
    // separator, so on Windows the collapsed form would never match one
    // and a `..` detour would slip through the check.
    let collapsed_str = lexically_normalize(path)
        .to_string_lossy()
        .replace('\\', "/");
    if SENSITIVE_PREFIXES.iter().any(|prefix| {
        [&raw_str, &collapsed_str]
            .iter()
            .any(|candidate| candidate.starts_with(prefix) || *candidate == prefix)
    }) {
        // Echo the path the model supplied rather than a canonicalised
        // absolute one, which would put the real home directory into a
        // reply that may reach a chat channel.
        anyhow::bail!("{verb} '{raw_str}' is not allowed (sensitive system path).");
    }

    if let Some(dir) = protected_dir
        && let Some(name) = protected_leaf(path, workspace_root, dir)
    {
        anyhow::bail!(
            "{verb} '{raw_str}' is not allowed: that is this agent's own \
             '{name}', which records what you have been given permission to do."
        );
    }

    Ok(())
}

/// Production entry point: protect the agent's real config directory.
fn refuse_if_sensitive(path: &Path, workspace_root: Option<&Path>, verb: &str) -> Result<()> {
    refuse_if_sensitive_in(path, workspace_root, agent_config_dir().as_deref(), verb)
}

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
                    When the target file is inside the workspace, the search index \
                    is updated automatically. \
                    Refuses writes to sensitive system paths (/etc, /boot, /bin, etc.) \
                    and to this agent's own config.toml and acp-permissions.json."
                    .into(),
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
    fn kind(&self) -> ToolKind {
        ToolKind::Edit
    }

    fn spec(&self) -> &ToolSpec {
        &self.spec
    }

    async fn execute(&self, input: &serde_json::Value) -> Result<String> {
        let path_str = input["path"].as_str().context("missing 'path'")?;
        let content = input["content"].as_str().context("missing 'content'")?;

        let path = expand_path(path_str);
        let workspace_root = self
            .state
            .lock()
            .expect("WorkspaceState mutex poisoned")
            .workspace
            .root
            .clone();
        refuse_if_sensitive(&path, Some(&workspace_root), "Writing to")?;

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
                    When the file is inside the workspace, it is also removed from the search index \
                    automatically. \
                    Refuses deletes of sensitive system paths (/etc, /boot, /bin, etc.) \
                    and of this agent's own config.toml and acp-permissions.json. \
                    Cannot delete directories."
                    .into(),
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
    fn kind(&self) -> ToolKind {
        ToolKind::Delete
    }

    fn spec(&self) -> &ToolSpec {
        &self.spec
    }

    async fn execute(&self, input: &serde_json::Value) -> Result<String> {
        let path_str = input["path"].as_str().context("missing 'path'")?;
        let path = expand_path(path_str);

        let workspace_root = self
            .state
            .lock()
            .expect("WorkspaceState mutex poisoned")
            .workspace
            .root
            .clone();
        refuse_if_sensitive(&path, Some(&workspace_root), "Deleting")?;

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
                    When the target file is inside the workspace, the search index \
                    is updated automatically. \
                    Refuses writes to sensitive system paths (/etc, /boot, /bin, etc.) \
                    and to this agent's own config.toml and acp-permissions.json."
                    .into(),
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
    fn kind(&self) -> ToolKind {
        ToolKind::Edit
    }

    fn spec(&self) -> &ToolSpec {
        &self.spec
    }

    async fn execute(&self, input: &serde_json::Value) -> Result<String> {
        let path_str = input["path"].as_str().context("missing 'path'")?;
        let content = input["content"].as_str().context("missing 'content'")?;

        let path = expand_path(path_str);
        let workspace_root = self
            .state
            .lock()
            .expect("WorkspaceState mutex poisoned")
            .workspace
            .root
            .clone();
        refuse_if_sensitive(&path, Some(&workspace_root), "Writing to")?;

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
    fn kind(&self) -> ToolKind {
        ToolKind::Search
    }

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
    fn kind(&self) -> ToolKind {
        ToolKind::Search
    }

    fn spec(&self) -> &ToolSpec {
        &self.spec
    }

    async fn execute(&self, input: &serde_json::Value) -> Result<String> {
        let path_str = input["path"].as_str().context("missing 'path'")?;
        let max_depth = input["max_depth"].as_u64().unwrap_or(5).min(20) as usize;
        let max_entries = input["max_entries"].as_u64().unwrap_or(500).clamp(1, 5000) as usize;

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
    fn kind(&self) -> ToolKind {
        ToolKind::Execute
    }

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
            .current_dir(&workdir)
            // The turn running this tool can be cancelled: the ACP endpoint
            // drops `run_llm_turn`'s future on `session/cancel` and on a
            // vanished client, which drops the `wait_with_output` future and
            // with it the `Child`. Without this, that drop merely *disowns*
            // the process — it keeps running, keeps writing to the
            // workspace, and never reaches the timeout branch below, because
            // the timeout is gone too. `shell` is the only tool that can
            // outlive its future this way; the file tools use blocking
            // `std::fs` and so cannot be interrupted part-way.
            //
            // Nothing else changes shape: `/rpc` and the voice/heartbeat
            // paths run their turn in a detached `tokio::spawn`, so it
            // finishes whether or not anyone is still listening, and this
            // never fires for them. `/a2a` awaits its turn inside the axum
            // handler, whose future hyper can drop when the HTTP client
            // vanishes mid-request — there this quietly stops leaking a
            // process it was already leaking.
            .kill_on_drop(true);

        let child = cmd
            .spawn()
            .with_context(|| format!("Failed to spawn shell '{shell}'"))?;

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
                // Nothing to kill here any more. `timeout` drops the
                // `wait_with_output` future at the end of the statement
                // above, which drops the `Child`, which `kill_on_drop`
                // turns into a SIGKILL (a `TerminateProcess` on Windows) —
                // so by the time this branch runs the shell is already
                // dead. What used to stand here, `kill -9 <pid>`, did
                // exactly that and no more: a positive pid signals one
                // process, not its group, so neither form reaches
                // grandchildren the shell left behind. It was also a
                // blocking `std::process::Command` on an async thread, and
                // relied on a `kill` binary being on PATH, which is not the
                // case on Windows.
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
    fn kind(&self) -> ToolKind {
        ToolKind::Fetch
    }

    fn spec(&self) -> &ToolSpec {
        &self.spec
    }

    async fn execute(&self, input: &serde_json::Value) -> Result<String> {
        let client = reqwest::Client::new();
        let days = input["days"].as_u64().unwrap_or(3).clamp(1, 7);

        // Resolve coordinates. Explicit lat/lon wins; otherwise geocode `location`.
        let (latitude, longitude, resolved_name) =
            match (input["latitude"].as_f64(), input["longitude"].as_f64()) {
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
            out.push_str(&format!(
                "\nCurrent ({time}): {}\n",
                wmo_code_description(code)
            ));
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
    fn kind(&self) -> ToolKind {
        ToolKind::Fetch
    }

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

// ---------------------------------------------------------------------------
// recall_image
// ---------------------------------------------------------------------------

/// Re-fetch a past image from the workspace-external image cache and
/// attach it to the tool_result so the model can look at it again.
///
/// Older turns in conversation history appear to the model as text
/// markers like `[image: image/png sha256=<hex>]` — the raw bytes
/// aren't re-sent every turn to keep input-token cost down. When the
/// user asks the model to look at a past image, the model is expected
/// to call this tool with the marker's sha256 + media_type; the cache
/// is then queried and the actual bytes get appended to the user
/// message carrying this tool's result.
pub struct RecallImageTool {
    cache: Arc<ImageCache>,
    spec: ToolSpec,
}

impl RecallImageTool {
    pub fn new(cache: Arc<ImageCache>) -> Self {
        let spec = ToolSpec {
            name: "recall_image".into(),
            description: "Re-fetch a past image from the conversation by its sha256 hash. \
                Use this when the user references an image that now appears in history as a \
                `[image: <media_type> sha256=<hex>]` text marker (only images attached in the \
                CURRENT turn are visible inline; older ones are shown as markers to save \
                tokens). Pass the marker's `sha256` and `media_type` verbatim; the actual \
                image bytes will be returned as an attachment for you to view."
                .into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "sha256": {
                        "type": "string",
                        "description": "Hex sha256 of the image to recall (copy from the `[image: ... sha256=<hex>]` marker)."
                    },
                    "media_type": {
                        "type": "string",
                        "description": "MIME type of the image, e.g. `image/png`, `image/jpeg`, `image/gif`, `image/webp` (copy from the marker)."
                    }
                },
                "required": ["sha256", "media_type"]
            }),
        };
        Self { cache, spec }
    }
}

#[async_trait]
impl Tool for RecallImageTool {
    fn kind(&self) -> ToolKind {
        ToolKind::Read
    }

    fn spec(&self) -> &ToolSpec {
        &self.spec
    }

    async fn execute(&self, _input: &serde_json::Value) -> Result<String> {
        // Should never be reached: `execute_full` is overridden below.
        // If something does call this path, surface a clear marker text
        // rather than a panic — the model can recover.
        Ok("recall_image returned no image (text-only path).".to_string())
    }

    async fn execute_full(&self, input: &serde_json::Value) -> Result<ToolOutput> {
        let sha256 = input
            .get("sha256")
            .and_then(|v| v.as_str())
            .context("Missing required field: sha256")?;
        let media_type = input
            .get("media_type")
            .and_then(|v| v.as_str())
            .context("Missing required field: media_type")?;

        if !is_hex_sha256(sha256) {
            anyhow::bail!(
                "sha256 must be a 64-char lowercase hex string; got {} char(s)",
                sha256.len()
            );
        }

        let bytes = self
            .cache
            .get(sha256)
            .with_context(|| format!("image not in cache (sha256={sha256})"))?;

        let data_base64 = BASE64_STANDARD.encode(&bytes);
        let byte_len = bytes.len();
        Ok(ToolOutput {
            text: format!(
                "Recalled image (sha256={sha256}, media_type={media_type}, {byte_len} bytes). \
                 Image attached to this tool result — refer to it directly."
            ),
            images: vec![(media_type.to_string(), data_base64)],
        })
    }
}

fn is_hex_sha256(s: &str) -> bool {
    s.len() == 64
        && s.chars()
            .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase())
}

#[cfg(test)]
mod sensitive_path_tests {
    use super::*;
    use tempfile::TempDir;

    /// A stand-in for the agent's config directory.
    ///
    /// `present` decides whether the protected files already exist.
    /// **Both regimes matter, and the absent one matters more**: on a
    /// fresh install nothing has recorded a permission answer yet, so
    /// `acp-permissions.json` is not there — and an earlier version of
    /// this guard leaned on file identity, which can only answer when
    /// the target exists. Its tests created the files every time and so
    /// never exercised the regime production actually starts in. That
    /// is how an NTFS alternate-data-stream bypass shipped.
    fn protected_dir(present: bool) -> (TempDir, PathBuf) {
        let dir = TempDir::new().unwrap();
        let path = dir.path().to_path_buf();
        if present {
            std::fs::write(path.join("acp-permissions.json"), b"{}").unwrap();
            std::fs::write(path.join("config.toml"), b"").unwrap();
        }
        (dir, path)
    }

    fn refused(path: &str, workspace_root: Option<&Path>, protected: &Path) -> bool {
        refuse_if_sensitive_in(
            Path::new(path),
            workspace_root,
            Some(protected),
            "Writing to",
        )
        .is_err()
    }

    #[test]
    fn system_paths_are_refused() {
        for path in [
            "/etc/passwd",
            "/bin/sh",
            "/proc/self/mem",
            "/var/run/docker.sock",
        ] {
            assert!(
                refuse_if_sensitive_in(Path::new(path), None, None, "Writing to").is_err(),
                "{path} should be refused"
            );
            // A `..` detour to the same place.
            let detour = format!("/tmp/../{}", path.trim_start_matches('/'));
            assert!(
                refuse_if_sensitive_in(Path::new(&detour), None, None, "Writing to").is_err(),
                "{detour} should be refused"
            );
        }
    }

    /// Every spelling that reaches a protected file must be refused —
    /// run twice, once with the files present and once absent, because
    /// the guard must not depend on the target existing.
    #[test]
    fn no_spelling_reaches_a_protected_file() {
        for present in [true, false] {
            let (_guard, dir) = protected_dir(present);
            std::fs::create_dir_all(dir.join("sub")).unwrap();

            let mut spellings: Vec<String> = vec![
                "acp-permissions.json".into(),
                "config.toml".into(),
                // A `..` through a directory that does NOT exist: the
                // writer's create_dir_all makes it and the kernel
                // collapses the `..` at open time.
                "nope/../acp-permissions.json".into(),
                "a/b/c/../../../acp-permissions.json".into(),
                // ...and through one that does.
                "sub/../acp-permissions.json".into(),
                "./acp-permissions.json".into(),
            ];
            // Absolute spellings of the same.
            spellings.push(
                dir.join("acp-permissions.json")
                    .to_string_lossy()
                    .into_owned(),
            );
            spellings.push(
                dir.join("nope/../acp-permissions.json")
                    .to_string_lossy()
                    .into_owned(),
            );

            if cfg!(windows) {
                // `name::$DATA` is the file's own default data stream.
                // A colon is never legal in an NTFS filename, so this
                // is stream syntax and the OS opens the plain file.
                spellings.push("acp-permissions.json::$DATA".into());
                spellings.push("config.toml::$DATA".into());
                // Trailing dots and spaces are stripped when Windows
                // normalises a path.
                spellings.push("acp-permissions.json.".into());
                spellings.push("acp-permissions.json  ".into());
                spellings.push("acp-permissions.json. . ".into());
                spellings.push("ACP-PERMISSIONS.JSON".into());
            }

            for spelling in &spellings {
                assert!(
                    refused(spelling, Some(&dir), &dir),
                    "'{spelling}' must not reach a protected file (present={present})"
                );
            }

            // A climb out of a subdirectory workspace.
            assert!(
                refused("../acp-permissions.json", Some(&dir.join("sub")), &dir),
                "a climb out of the workspace must be refused (present={present})"
            );
        }
    }

    /// A symlink is a different path that opens the same file, and
    /// `fs::write` follows it. The guard follows it too — with
    /// `read_link` rather than `canonicalize`, so a *dangling* link
    /// pointing at a record that does not exist yet still resolves.
    #[cfg(unix)]
    #[test]
    fn a_symlink_to_a_protected_file_is_refused() {
        for present in [true, false] {
            let (_guard, dir) = protected_dir(present);
            let workspace = TempDir::new().unwrap();

            let link = workspace.path().join("notes.json");
            std::os::unix::fs::symlink(dir.join("acp-permissions.json"), &link).unwrap();
            assert!(
                refuse_if_sensitive_in(&link, None, Some(&dir), "Writing to").is_err(),
                "a symlink to the record must be refused (present={present})"
            );

            // A relative link, resolved against its own directory.
            let rel = dir.join("innocent.json");
            std::os::unix::fs::symlink("acp-permissions.json", &rel).unwrap();
            assert!(
                refuse_if_sensitive_in(&rel, None, Some(&dir), "Writing to").is_err(),
                "a relative symlink must be refused (present={present})"
            );
        }
    }

    /// A hard link is the same inode under another name. It can only be
    /// made to a file that exists, so this is the present-file regime
    /// by construction.
    #[cfg(unix)]
    #[test]
    fn a_hard_link_to_the_record_is_refused() {
        let (_guard, dir) = protected_dir(true);
        let link = dir.join("innocent.json");
        std::fs::hard_link(dir.join("acp-permissions.json"), &link).unwrap();

        assert!(
            refuse_if_sensitive_in(&link, None, Some(&dir), "Writing to").is_err(),
            "a hard link to the record must be refused"
        );
    }

    /// The guard must not swallow ordinary paths. In particular a
    /// workspace that IS the protected directory — which is what an
    /// unset `workspace_dir` gives you — stays writable for everything
    /// except the named files.
    #[test]
    fn ordinary_paths_are_allowed() {
        for present in [true, false] {
            let (_guard, dir) = protected_dir(present);

            for path in ["/tmp/notes.md", "/home/someone/todo.txt"] {
                assert!(!refused(path, None, &dir), "{path} should be allowed");
            }
            for name in [
                "notes.md",
                "memory/thing.md",
                "heartbeat/daily.md",
                "acp-permissions.json.backup",
                "not-config.toml",
            ] {
                assert!(
                    !refused(name, Some(&dir), &dir),
                    "{name} must stay writable (present={present})"
                );
            }
        }
    }

    /// The same filename in a lookalike directory is a different file.
    #[test]
    fn a_sibling_directory_is_not_the_protected_one() {
        let (_guard, dir) = protected_dir(true);
        let sibling = TempDir::new().unwrap();
        std::fs::write(sibling.path().join("acp-permissions.json"), b"{}").unwrap();

        assert!(
            !refused(
                &sibling
                    .path()
                    .join("acp-permissions.json")
                    .to_string_lossy(),
                None,
                &dir
            ),
            "a different directory's file of the same name should be allowed"
        );
    }

    /// Deleting is guarded on the same terms as writing. Nothing else
    /// exercises the other verb.
    #[test]
    fn the_delete_verb_is_guarded_too() {
        let (_guard, dir) = protected_dir(true);
        let err = refuse_if_sensitive_in(
            &dir.join("acp-permissions.json"),
            None,
            Some(&dir),
            "Deleting",
        )
        .expect_err("deleting the record must be refused");
        assert!(err.to_string().starts_with("Deleting"), "got {err}");
    }

    /// The production wrapper must actually consult the agent's own
    /// config directory. Nothing else covers that wiring, and a guard
    /// pointed at the wrong directory would pass every other test here.
    #[test]
    fn the_production_wrapper_protects_the_real_config_dir() {
        let Some(config_dir) = agent_config_dir() else {
            panic!("no config dir: PermissionStore falls back to the cwd, which nothing guards");
        };
        assert!(
            refuse_if_sensitive(&config_dir.join("acp-permissions.json"), None, "Writing to")
                .is_err(),
            "the real permission record must be refused"
        );
        assert!(
            refuse_if_sensitive(&config_dir.join("notes.md"), None, "Writing to").is_ok(),
            "an ordinary file in the config dir must stay writable"
        );
    }

    #[test]
    fn lexical_normalisation_collapses_without_touching_the_disk() {
        assert_eq!(
            lexically_normalize(Path::new("/a/b/../c/./d")),
            PathBuf::from("/a/c/d")
        );
        // Never pop past the root, and never leave a `..` behind that a
        // later comparison would have to reason about.
        assert_eq!(
            lexically_normalize(Path::new("/../..")),
            PathBuf::from(std::path::MAIN_SEPARATOR_STR)
        );
        // A leading `..` on a relative path has nothing to cancel and
        // is kept, so it can still be joined onto a workspace root.
        assert_eq!(
            lexically_normalize(Path::new("../x")),
            PathBuf::from("../x")
        );
        assert_eq!(lexically_normalize(Path::new("")), PathBuf::from(""));
        assert_eq!(lexically_normalize(Path::new("x")), PathBuf::from("x"));
    }

    #[test]
    fn the_opened_filename_is_what_windows_will_open() {
        let name = |s: &str| opened_filename(std::ffi::OsStr::new(s));

        assert_eq!(
            name("acp-permissions.json").as_deref(),
            Some("acp-permissions.json")
        );
        if cfg!(windows) {
            assert_eq!(name("x.json::$DATA").as_deref(), Some("x.json"));
            assert_eq!(name("x.json:stream").as_deref(), Some("x.json"));
            assert_eq!(name("x.json. . ").as_deref(), Some("x.json"));
            // Nothing left is not a spelling of any real file.
            assert_eq!(name("::$DATA"), None);
            assert_eq!(name(". "), None);
        } else {
            // Elsewhere these are ordinary filename characters.
            assert_eq!(name("x.json::$DATA").as_deref(), Some("x.json::$DATA"));
            assert_eq!(name("x.json.").as_deref(), Some("x.json."));
        }
    }
}

#[cfg(test)]
mod recall_image_tests {
    use super::*;
    use tempfile::TempDir;

    fn open_cache() -> (TempDir, Arc<ImageCache>) {
        let tmp = TempDir::new().unwrap();
        let cache = ImageCache::open(tmp.path().to_path_buf()).unwrap();
        (tmp, cache)
    }

    #[tokio::test]
    async fn returns_image_on_cache_hit() {
        let (_tmp, cache) = open_cache();
        let bytes = b"\xff\xd8\xfffake-jpeg".to_vec();
        let sha = crate::image_cache::sha256_hex(&bytes);
        cache.put(&sha, &bytes).unwrap();

        let tool = RecallImageTool::new(cache);
        let out = tool
            .execute_full(&json!({"sha256": sha, "media_type": "image/jpeg"}))
            .await
            .unwrap();
        assert_eq!(out.images.len(), 1);
        assert_eq!(out.images[0].0, "image/jpeg");
        assert_eq!(BASE64_STANDARD.decode(&out.images[0].1).unwrap(), bytes);
        assert!(out.text.contains(&sha));
    }

    #[tokio::test]
    async fn errors_on_cache_miss() {
        let (_tmp, cache) = open_cache();
        let tool = RecallImageTool::new(cache);
        let sha = "0".repeat(64);
        let err = tool
            .execute_full(&json!({"sha256": sha, "media_type": "image/png"}))
            .await
            .unwrap_err();
        assert!(format!("{err:#}").contains("not in cache"));
    }

    #[tokio::test]
    async fn rejects_malformed_sha() {
        let (_tmp, cache) = open_cache();
        let tool = RecallImageTool::new(cache);
        let err = tool
            .execute_full(&json!({"sha256": "abc", "media_type": "image/png"}))
            .await
            .unwrap_err();
        assert!(format!("{err:#}").contains("64-char"));
    }
}
