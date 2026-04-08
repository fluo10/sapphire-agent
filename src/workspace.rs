use std::path::{Path, PathBuf};
use std::time::SystemTime;
use tokio::sync::Mutex;
use tracing::{debug, info};

/// Maximum characters loaded from a single workspace file (matches openclaw default).
const MAX_FILE_CHARS: usize = 20_000;

// ---------------------------------------------------------------------------
// Workspace file definitions
// ---------------------------------------------------------------------------

/// A single workspace file entry: one or more candidate filenames to try (in
/// order), plus the Markdown heading to use when injecting into the system
/// prompt.
struct WorkspaceFileDef {
    /// Candidate filenames tried in order; the first one found is used.
    candidates: &'static [&'static str],
    /// Heading inserted above the file content (e.g. "# Agent Instructions").
    heading: &'static str,
}

/// Ordered list of workspace files, following openclaw's convention.
/// Files that don't exist are silently skipped.
/// See: https://github.com/openclaw/openclaw (src/agents/workspace.ts)
static WORKSPACE_FILES: &[WorkspaceFileDef] = &[
    // openclaw uses "AGENTS.md" (plural); we also accept "AGENT.md" for
    // users who create the file without the trailing 's'.
    WorkspaceFileDef {
        candidates: &["AGENTS.md", "AGENT.md"],
        heading: "# Agent Instructions",
    },
    WorkspaceFileDef {
        candidates: &["SOUL.md"],
        heading: "# Soul",
    },
    WorkspaceFileDef {
        candidates: &["IDENTITY.md"],
        heading: "# Identity",
    },
    WorkspaceFileDef {
        candidates: &["USER.md"],
        heading: "# User",
    },
    WorkspaceFileDef {
        candidates: &["TOOLS.md"],
        heading: "# Tools",
    },
    WorkspaceFileDef {
        candidates: &["BOOTSTRAP.md"],
        heading: "# Bootstrap",
    },
    // openclaw tries "MEMORY.md" first, falls back to lowercase "memory.md".
    WorkspaceFileDef {
        candidates: &["MEMORY.md", "memory.md"],
        heading: "# Memory",
    },
];

// ---------------------------------------------------------------------------
// File cache
// ---------------------------------------------------------------------------

struct CachedFile {
    content: String,
    mtime: SystemTime,
}

// ---------------------------------------------------------------------------
// Workspace
// ---------------------------------------------------------------------------

/// Reads workspace files (AGENTS.md, SOUL.md, IDENTITY.md, USER.md, TOOLS.md,
/// HEARTBEAT.md, BOOTSTRAP.md, MEMORY.md) and assembles them into the system
/// prompt on every turn.
///
/// Files are cached by mtime so edits take effect immediately on the next
/// message without restarting the agent.
pub struct Workspace {
    dir: PathBuf,
    cache: Mutex<std::collections::HashMap<PathBuf, CachedFile>>,
}

impl Workspace {
    pub fn new(dir: PathBuf) -> Self {
        info!("Workspace dir: {}", dir.display());
        Self {
            dir,
            cache: Mutex::new(std::collections::HashMap::new()),
        }
    }

    /// Build the full system prompt:
    /// 1. Base system_prompt from config (if any)
    /// 2. Each workspace file that exists, in openclaw order
    pub async fn build_system_prompt(&self, base: Option<&str>) -> String {
        let mut parts: Vec<String> = Vec::new();

        if let Some(b) = base.filter(|s| !s.is_empty()) {
            parts.push(b.to_string());
        }

        // Inject current date/time so the agent has temporal awareness
        // (needed for tools like `memory` which write date-stamped files).
        let now_local = chrono::Local::now();
        parts.push(format!(
            "# Current Date and Time\n\n{} ({})",
            now_local.format("%Y-%m-%d %H:%M:%S %z"),
            now_local.format("%A")
        ));

        for def in WORKSPACE_FILES {
            if let Some((filename, content)) = self.read_first_existing(def.candidates).await {
                debug!("Injecting workspace file: {filename}");
                parts.push(format!("{}\n\n{content}", def.heading));
            }
        }

        parts.join("\n\n---\n\n")
    }

    /// Try each candidate filename in order; return the first one found.
    async fn read_first_existing(
        &self,
        candidates: &[&str],
    ) -> Option<(String, String)> {
        for &filename in candidates {
            if let Some(content) = self.read_file(filename).await {
                return Some((filename.to_string(), content));
            }
        }
        None
    }

    /// Read a workspace file with mtime caching. Returns `None` if not found.
    async fn read_file(&self, filename: &str) -> Option<String> {
        let path = self.dir.join(filename);

        match Self::file_mtime(&path) {
            None => {
                self.cache.lock().await.remove(&path);
                None
            }
            Some(mtime) => {
                {
                    let cache = self.cache.lock().await;
                    if let Some(entry) = cache.get(&path) {
                        if entry.mtime == mtime {
                            debug!("Workspace cache hit: {filename}");
                            return Some(entry.content.clone());
                        }
                    }
                }

                match std::fs::read_to_string(&path) {
                    Ok(raw) => {
                        let content = truncate_chars(&raw, MAX_FILE_CHARS);
                        info!(
                            "Loaded workspace file: {filename} ({} chars)",
                            content.len()
                        );
                        self.cache.lock().await.insert(
                            path,
                            CachedFile { content: content.clone(), mtime },
                        );
                        Some(content)
                    }
                    Err(e) => {
                        tracing::warn!("Failed to read {filename}: {e}");
                        None
                    }
                }
            }
        }
    }

    fn file_mtime(path: &Path) -> Option<SystemTime> {
        std::fs::metadata(path).ok()?.modified().ok()
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Truncate `s` to at most `max_chars` Unicode scalar values.
fn truncate_chars(s: &str, max_chars: usize) -> String {
    let mut chars = s.chars();
    let truncated: String = (&mut chars).take(max_chars).collect();
    if chars.next().is_some() {
        format!("{truncated}\n\n[... truncated to {max_chars} characters ...]")
    } else {
        truncated
    }
}
