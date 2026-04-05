use crate::provider::ToolSpec;
use crate::tools::Tool;
use anyhow::{Context, Result};
use sapphire_workspace::WorkspaceState;
use serde_json::json;
use std::path::Path;
use std::sync::{Arc, Mutex};

// ---------------------------------------------------------------------------
// Shared helper
// ---------------------------------------------------------------------------

fn lock(state: &Mutex<WorkspaceState>) -> std::sync::MutexGuard<'_, WorkspaceState> {
    state.lock().expect("WorkspaceState mutex poisoned")
}

// ---------------------------------------------------------------------------
// memory_append
// ---------------------------------------------------------------------------

/// Append text to MEMORY.md (creates it if it doesn't exist).
pub struct MemoryAppendTool {
    state: Arc<Mutex<WorkspaceState>>,
    spec: ToolSpec,
}

impl MemoryAppendTool {
    pub fn new(state: Arc<Mutex<WorkspaceState>>) -> Self {
        Self {
            state,
            spec: ToolSpec {
                name: "memory_append",
                description: "Append text to MEMORY.md in the workspace. \
                    Use this to persist important information across conversations.",
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "text": {
                            "type": "string",
                            "description": "The text to append to MEMORY.md (Markdown is fine)."
                        }
                    },
                    "required": ["text"]
                }),
            },
        }
    }
}

impl Tool for MemoryAppendTool {
    fn spec(&self) -> &ToolSpec { &self.spec }

    fn execute(&self, input: &serde_json::Value) -> Result<String> {
        let text = input["text"].as_str().context("missing 'text'")?;
        let state = lock(&self.state);
        let content = format!("\n{text}\n");
        state
            .append_file(Path::new("MEMORY.md"), &content)
            .context("Failed to append to MEMORY.md")?;
        Ok("Appended to MEMORY.md".to_string())
    }
}

// ---------------------------------------------------------------------------
// memory_write
// ---------------------------------------------------------------------------

/// Overwrite MEMORY.md with new content.
pub struct MemoryWriteTool {
    state: Arc<Mutex<WorkspaceState>>,
    spec: ToolSpec,
}

impl MemoryWriteTool {
    pub fn new(state: Arc<Mutex<WorkspaceState>>) -> Self {
        Self {
            state,
            spec: ToolSpec {
                name: "memory_write",
                description: "Overwrite MEMORY.md with new content. \
                    Use this to reorganize or rewrite the entire long-term memory file.",
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "The new content for MEMORY.md."
                        }
                    },
                    "required": ["content"]
                }),
            },
        }
    }
}

impl Tool for MemoryWriteTool {
    fn spec(&self) -> &ToolSpec { &self.spec }

    fn execute(&self, input: &serde_json::Value) -> Result<String> {
        let content = input["content"].as_str().context("missing 'content'")?;
        let state = lock(&self.state);
        state
            .write_file(Path::new("MEMORY.md"), content)
            .context("Failed to write MEMORY.md")?;
        Ok("MEMORY.md updated".to_string())
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

impl Tool for WorkspaceReadTool {
    fn spec(&self) -> &ToolSpec { &self.spec }

    fn execute(&self, input: &serde_json::Value) -> Result<String> {
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

impl Tool for WorkspaceWriteTool {
    fn spec(&self) -> &ToolSpec { &self.spec }

    fn execute(&self, input: &serde_json::Value) -> Result<String> {
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

/// Full-text search across workspace files.
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
                description: "Full-text search across all indexed files in the workspace. \
                    Returns matching file titles and paths.",
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query."
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

impl Tool for WorkspaceSearchTool {
    fn spec(&self) -> &ToolSpec { &self.spec }

    fn execute(&self, input: &serde_json::Value) -> Result<String> {
        let query = input["query"].as_str().context("missing 'query'")?;
        let limit = input["limit"].as_u64().unwrap_or(10) as usize;
        let state = lock(&self.state);
        let results = state
            .retrieve_db()
            .search_fts(query, limit)
            .context("FTS search failed")?;

        if results.is_empty() {
            return Ok("No results found.".to_string());
        }

        let lines: Vec<String> = results
            .iter()
            .map(|r| format!("- {} ({})", r.title, r.path))
            .collect();
        Ok(lines.join("\n"))
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

impl Tool for WorkspaceSyncTool {
    fn spec(&self) -> &ToolSpec { &self.spec }

    fn execute(&self, _input: &serde_json::Value) -> Result<String> {
        let state = lock(&self.state);

        // Sync index
        let (upserted, removed) = state.sync().context("Failed to sync workspace index")?;

        // Git commit + push if backend is configured
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
