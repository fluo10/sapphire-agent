//! MEMORY.md compaction.
//!
//! Reads `<workspace>/MEMORY.md`, asks the provider to merge duplicates and
//! summarise stale entries, and writes the result back. The entry separator
//! `\n\n---\n\n` (used by `MemoryTool`) is preserved by instructing the model.

use crate::provider::{ChatMessage, Provider};
use std::path::Path;
use tracing::{info, warn};

const SYSTEM_PROMPT: &str = "You are reorganising an AI agent's persistent memory file (MEMORY.md). \
The file contains independent entries separated by `\\n\\n---\\n\\n` (a Markdown horizontal rule on its own line, blank lines around it). \
\
Your task: \
1. Merge entries that record the same fact (keep the most informative version). \
2. Summarise or compress entries that are clearly stale, verbose, or redundant. \
3. Preserve all unique, useful information. Do NOT invent new facts. \
4. Keep the same `\\n\\n---\\n\\n` separator between entries. \
5. Write in the same language as the original. \
\
Output ONLY the new contents of MEMORY.md, with no extra commentary, no code fences, and no preamble.";

/// Compact `<workspace_dir>/MEMORY.md` in place. No-op if the file doesn't
/// exist or is effectively empty. Errors are logged but never returned.
pub async fn compact_memory(provider: &dyn Provider, workspace_dir: &Path) {
    let path = workspace_dir.join("MEMORY.md");
    let original = match std::fs::read_to_string(&path) {
        Ok(s) => s,
        Err(_) => {
            // Try lowercase fallback (matches workspace.rs candidate order)
            let lower = workspace_dir.join("memory.md");
            match std::fs::read_to_string(&lower) {
                Ok(s) => s,
                Err(_) => {
                    info!("memory_compaction: no MEMORY.md found, skipping");
                    return;
                }
            }
        }
    };

    if original.trim().is_empty() {
        info!("memory_compaction: MEMORY.md is empty, skipping");
        return;
    }

    let user_msg = ChatMessage::user(&original);
    let response = match provider.chat(Some(SYSTEM_PROMPT), &[user_msg], None).await {
        Ok(r) => r,
        Err(e) => {
            warn!("memory_compaction: provider error, leaving file unchanged: {e:#}");
            return;
        }
    };

    let new_content = match response.text {
        Some(t) if !t.trim().is_empty() => t,
        _ => {
            warn!("memory_compaction: empty response, leaving file unchanged");
            return;
        }
    };

    if let Err(e) = std::fs::write(&path, &new_content) {
        warn!("memory_compaction: failed to write MEMORY.md: {e}");
        return;
    }

    info!(
        "memory_compaction: MEMORY.md updated ({} -> {} bytes)",
        original.len(),
        new_content.len()
    );
}
