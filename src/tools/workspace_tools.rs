use crate::provider::ToolSpec;
use crate::tools::Tool;
use anyhow::{Context, Result};
use async_trait::async_trait;
use sapphire_workspace::{RetrieveDb, WorkspaceState};
use serde_json::json;
use std::path::{Component, Path, PathBuf};
use std::sync::{Arc, Mutex};

// ---------------------------------------------------------------------------
// Shared helper
// ---------------------------------------------------------------------------

fn lock(state: &Mutex<WorkspaceState>) -> std::sync::MutexGuard<'_, WorkspaceState> {
    state.lock().expect("WorkspaceState mutex poisoned")
}

// ---------------------------------------------------------------------------
// memory
// ---------------------------------------------------------------------------

const ENTRY_SEP: &str = "\n\n---\n\n";

/// Validate and resolve a workspace-relative path, rejecting traversal.
fn resolve_workspace_path(workspace_root: &Path, rel: &str) -> Result<PathBuf> {
    let rel_path = Path::new(rel);
    for component in rel_path.components() {
        if component == Component::ParentDir {
            anyhow::bail!("Path traversal not allowed: {}", rel);
        }
    }
    Ok(workspace_root.join(rel_path))
}

/// Split file content into entries (separator: `\n\n---\n\n`).
fn split_entries(content: &str) -> Vec<&str> {
    if content.trim().is_empty() {
        vec![]
    } else {
        content.split(ENTRY_SEP).collect()
    }
}

/// Join entries back into file content.
fn join_entries(entries: &[&str]) -> String {
    entries.join(ENTRY_SEP)
}

/// Entry-based persistent memory management for workspace markdown files.
pub struct MemoryTool {
    state: Arc<Mutex<WorkspaceState>>,
    spec: ToolSpec,
}

impl MemoryTool {
    pub fn new(state: Arc<Mutex<WorkspaceState>>) -> Self {
        Self {
            state,
            spec: ToolSpec {
                name: "memory",
                description: "Add, replace, or remove an entry in a workspace markdown file. \
                    Files are entry-based, separated by horizontal rules (---). \
                    Use MEMORY.md for agent notes, USER.md for user profile, \
                    memory/daily/YYYY-MM-DD.md for daily logs. \
                    Parent directories are created automatically.",
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": ["add", "replace", "remove"],
                            "description": "Operation to perform."
                        },
                        "target": {
                            "type": "string",
                            "description": "Workspace-relative file path, e.g. \"MEMORY.md\" or \"memory/daily/2026-04-07.md\"."
                        },
                        "content": {
                            "type": "string",
                            "description": "Entry content (required for add and replace)."
                        },
                        "old_text": {
                            "type": "string",
                            "description": "Substring that uniquely identifies the entry to replace or remove (required for replace and remove)."
                        }
                    },
                    "required": ["action", "target"]
                }),
            },
        }
    }
}

#[async_trait]
impl Tool for MemoryTool {
    fn spec(&self) -> &ToolSpec {
        &self.spec
    }

    async fn execute(&self, input: &serde_json::Value) -> Result<String> {
        let action = input["action"].as_str().context("missing 'action'")?;
        let target = input["target"].as_str().context("missing 'target'")?;

        // Validate (no path traversal) and compute absolute path for reads.
        let workspace_root = lock(&self.state).workspace.root.clone();
        let abs_path = resolve_workspace_path(&workspace_root, target)?;
        let rel_path = Path::new(target);

        match action {
            "add" => {
                let content = input["content"].as_str().context("missing 'content' for add")?;
                let existing = std::fs::read_to_string(&abs_path).unwrap_or_default();
                let new_content = if existing.trim().is_empty() {
                    content.to_string()
                } else {
                    format!("{}{}{}", existing.trim_end(), ENTRY_SEP, content)
                };
                lock(&self.state)
                    .write_file(rel_path, &new_content)
                    .with_context(|| format!("Failed to write {target}"))?;
                Ok(format!("Added entry to {target}"))
            }

            "replace" => {
                let content = input["content"].as_str().context("missing 'content' for replace")?;
                let old_text = input["old_text"].as_str().context("missing 'old_text' for replace")?;
                let existing = std::fs::read_to_string(&abs_path)
                    .with_context(|| format!("Failed to read {target}"))?;
                let entries: Vec<&str> = split_entries(&existing);
                let idx = entries
                    .iter()
                    .position(|e| e.contains(old_text))
                    .with_context(|| format!("No entry containing {:?} found in {target}", old_text))?;
                let mut new_entries = entries.clone();
                new_entries[idx] = content;
                let joined = join_entries(&new_entries);
                lock(&self.state)
                    .write_file(rel_path, &joined)
                    .with_context(|| format!("Failed to write {target}"))?;
                Ok(format!("Replaced entry in {target}"))
            }

            "remove" => {
                let old_text = input["old_text"].as_str().context("missing 'old_text' for remove")?;
                let existing = std::fs::read_to_string(&abs_path)
                    .with_context(|| format!("Failed to read {target}"))?;
                let entries: Vec<&str> = split_entries(&existing);
                let idx = entries
                    .iter()
                    .position(|e| e.contains(old_text))
                    .with_context(|| format!("No entry containing {:?} found in {target}", old_text))?;
                let new_entries: Vec<&str> =
                    entries.iter().enumerate().filter(|(i, _)| *i != idx).map(|(_, e)| *e).collect();
                let joined = join_entries(&new_entries);
                lock(&self.state)
                    .write_file(rel_path, &joined)
                    .with_context(|| format!("Failed to write {target}"))?;
                Ok(format!("Removed entry from {target}"))
            }

            other => anyhow::bail!("Unknown action: {other}"),
        }
    }
}

// ---------------------------------------------------------------------------
// workspace_read
// ---------------------------------------------------------------------------

/// Read a file from the workspace.
pub struct WorkspaceReadTool {
    state: Arc<Mutex<WorkspaceState>>,
    spec: ToolSpec,
}

impl WorkspaceReadTool {
    pub fn new(state: Arc<Mutex<WorkspaceState>>) -> Self {
        Self {
            state,
            spec: ToolSpec {
                name: "workspace_read",
                description: "Read the contents of a file in the workspace \
                    (path relative to workspace root, e.g. \"notes/2025-01.md\").",
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Relative file path within the workspace."
                        }
                    },
                    "required": ["path"]
                }),
            },
        }
    }
}

#[async_trait]
impl Tool for WorkspaceReadTool {
    fn spec(&self) -> &ToolSpec { &self.spec }

    async fn execute(&self, input: &serde_json::Value) -> Result<String> {
        let rel = input["path"].as_str().context("missing 'path'")?;
        let state = lock(&self.state);
        let abs = state.workspace.root.join(rel);
        std::fs::read_to_string(&abs)
            .with_context(|| format!("Failed to read {rel}"))
    }
}

// ---------------------------------------------------------------------------
// workspace_write
// ---------------------------------------------------------------------------

/// Write a file in the workspace (create or overwrite).
pub struct WorkspaceWriteTool {
    state: Arc<Mutex<WorkspaceState>>,
    spec: ToolSpec,
}

impl WorkspaceWriteTool {
    pub fn new(state: Arc<Mutex<WorkspaceState>>) -> Self {
        Self {
            state,
            spec: ToolSpec {
                name: "workspace_write",
                description: "Write content to a file in the workspace \
                    (creates or overwrites). Path is relative to workspace root.",
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Relative file path."
                        },
                        "content": {
                            "type": "string",
                            "description": "Content to write."
                        }
                    },
                    "required": ["path", "content"]
                }),
            },
        }
    }
}

#[async_trait]
impl Tool for WorkspaceWriteTool {
    fn spec(&self) -> &ToolSpec { &self.spec }

    async fn execute(&self, input: &serde_json::Value) -> Result<String> {
        let rel = input["path"].as_str().context("missing 'path'")?;
        let content = input["content"].as_str().context("missing 'content'")?;
        let state = lock(&self.state);
        state
            .write_file(Path::new(rel), content)
            .with_context(|| format!("Failed to write {rel}"))?;
        Ok(format!("Written: {rel}"))
    }
}

// ---------------------------------------------------------------------------
// workspace_search
// ---------------------------------------------------------------------------

/// Full-text and semantic search across workspace files.
pub struct WorkspaceSearchTool {
    state: Arc<Mutex<WorkspaceState>>,
    spec: ToolSpec,
}

impl WorkspaceSearchTool {
    pub fn new(state: Arc<Mutex<WorkspaceState>>) -> Self {
        Self {
            state,
            spec: ToolSpec {
                name: "workspace_search",
                description: "Search across all indexed files in the workspace. \
                    Two modes are available: \
                    'fts' (full-text / BM25, always available) and \
                    'semantic' (vector similarity, requires an embedder to be configured — falls back to fts if unavailable). \
                    Returns matching file titles and paths.",
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query."
                        },
                        "mode": {
                            "type": "string",
                            "description": "Search mode: 'fts' (full-text, default) or 'semantic' (vector similarity).",
                            "enum": ["fts", "semantic"],
                            "default": "fts"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of results (default 10).",
                            "default": 10
                        }
                    },
                    "required": ["query"]
                }),
            },
        }
    }
}

#[async_trait]
impl Tool for WorkspaceSearchTool {
    fn spec(&self) -> &ToolSpec {
        &self.spec
    }

    async fn execute(&self, input: &serde_json::Value) -> Result<String> {
        let query = input["query"].as_str().context("missing 'query'")?;
        let limit = input["limit"].as_u64().unwrap_or(10) as usize;
        let mode = input["mode"].as_str().unwrap_or("fts");

        let state = lock(&self.state);

        if mode == "semantic" {
            if let Some(embedder) = state.embedder() {
                let vecs = embedder
                    .embed_texts(&[query])
                    .context("Failed to embed query")?;
                let query_vec: Vec<f32> = vecs
                    .into_iter()
                    .next()
                    .context("Embedder returned no vectors")?;
                let chunk_results = state
                    .retrieve_db()
                    .search_similar(&query_vec, limit * 3)
                    .context("Vector similarity search failed")?;
                let results = RetrieveDb::dedup_chunk_results(chunk_results, limit);

                if results.is_empty() {
                    return Ok("No results found.".to_string());
                }
                let lines: Vec<String> = results
                    .iter()
                    .map(|r| format!("- {} ({}) [score: {:.4}]", r.title, r.path, r.score))
                    .collect();
                return Ok(format!("[semantic]\n{}", lines.join("\n")));
            }
        }

        // FTS (default or fallback)
        let results = state
            .retrieve_db()
            .search_fts(query, limit)
            .context("FTS search failed")?;

        if results.is_empty() {
            return Ok("No results found.".to_string());
        }

        let header = if mode == "semantic" {
            "[fts — semantic fallback: no embedder configured]\n"
        } else {
            "[fts]\n"
        };
        let lines: Vec<String> = results
            .iter()
            .map(|r| format!("- {} ({})", r.title, r.path))
            .collect();
        Ok(format!("{}{}", header, lines.join("\n")))
    }
}

// ---------------------------------------------------------------------------
// workspace_sync
// ---------------------------------------------------------------------------

/// Sync the workspace via the configured backend (git commit + push).
pub struct WorkspaceSyncTool {
    state: Arc<Mutex<WorkspaceState>>,
    spec: ToolSpec,
}

impl WorkspaceSyncTool {
    pub fn new(state: Arc<Mutex<WorkspaceState>>) -> Self {
        Self {
            state,
            spec: ToolSpec {
                name: "workspace_sync",
                description: "Sync the workspace: index all files and, if a git \
                    remote is configured, commit and push changes.",
                input_schema: json!({
                    "type": "object",
                    "properties": {}
                }),
            },
        }
    }
}

#[async_trait]
impl Tool for WorkspaceSyncTool {
    fn spec(&self) -> &ToolSpec { &self.spec }

    async fn execute(&self, _input: &serde_json::Value) -> Result<String> {
        let state = lock(&self.state);

        let (upserted, removed) = state.sync().context("Failed to sync workspace index")?;

        if let Some(backend) = state.sync_backend() {
            backend.sync().context("Git sync failed")?;
            Ok(format!(
                "Synced: {upserted} files indexed, {removed} removed, git commit+push done."
            ))
        } else {
            Ok(format!(
                "Indexed: {upserted} files upserted, {removed} removed. \
                 No git remote configured."
            ))
        }
    }
}
