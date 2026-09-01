//! Tools that reach the editor's machine over ACP, rather than this
//! agent's own filesystem.
//!
//! `file_read`/`file_write` (`src/tools/builtin_tools.rs`) touch the
//! machine this agent runs on. These two touch the machine the *editor*
//! runs on, via `fs/read_text_file` and `fs/write_text_file` — see
//! `crate::tools::acp_client`. Both machines can be present in the same
//! turn's tool list only when the wrong one is impossible to reach for:
//! outside an ACP session there is no client to ask, so both refuse
//! rather than silently doing nothing.

use crate::provider::ToolSpec;
use crate::tools::acp_client::current_acp_client;
use crate::tools::{Tool, ToolKind};
use anyhow::{Context, Result};
use async_trait::async_trait;
use serde_json::json;

/// The editor is not reachable: no ACP client is scoped to this call.
///
/// Shared by both tools below so the wording — and the substring the
/// tests and the model both key on — cannot drift between them.
fn no_editor_error() -> anyhow::Error {
    anyhow::anyhow!("no editor is connected to this session; this tool only works over ACP")
}

// ---------------------------------------------------------------------------
// client_file_read
// ---------------------------------------------------------------------------

/// Read a file on the machine the editor is running on.
///
/// Distinct from `file_read`, which reads the machine the *agent* runs
/// on. In an ACP session only this one is offered, so the model cannot
/// pick the wrong machine.
pub struct ClientFileRead {
    spec: ToolSpec,
}

impl ClientFileRead {
    pub fn new() -> Self {
        Self {
            spec: ToolSpec {
                name: "client_file_read".into(),
                description: "Read a file on the machine the connected editor is \
                    running on — NOT this agent's own machine. Use `file_read` \
                    instead for files on the agent's machine. \
                    Only available inside an ACP session whose editor supports \
                    `fs/read_text_file`; refuses otherwise. \
                    For large files, pass `line` and `limit` to read a range \
                    instead of the whole file: ACP sends the requested content \
                    over the wire in full, so reading an entire large file at \
                    once is expensive."
                    .into(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Absolute path on the editor's machine."
                        },
                        "line": {
                            "type": "integer",
                            "description": "1-indexed line number to start reading from. \
                                Pair with `limit` to read a large file in pieces.",
                            "minimum": 1
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of lines to read starting at `line`.",
                            "minimum": 1
                        }
                    },
                    "required": ["path"]
                }),
            },
        }
    }
}

impl Default for ClientFileRead {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Tool for ClientFileRead {
    fn kind(&self) -> ToolKind {
        ToolKind::Read
    }

    fn spec(&self) -> &ToolSpec {
        &self.spec
    }

    async fn execute(&self, input: &serde_json::Value) -> Result<String> {
        let path = input["path"].as_str().context("missing 'path'")?;
        let line = input["line"].as_u64().map(|v| v as u32);
        let limit = input["limit"].as_u64().map(|v| v as u32);

        let client = current_acp_client().ok_or_else(no_editor_error)?;
        client.read_text_file(path, line, limit).await
    }
}

// ---------------------------------------------------------------------------
// client_file_write
// ---------------------------------------------------------------------------

/// Write a file on the machine the editor is running on.
///
/// Distinct from `file_write`, which writes the machine the *agent*
/// runs on. In an ACP session only this one is offered, so the model
/// cannot pick the wrong machine.
pub struct ClientFileWrite {
    spec: ToolSpec,
}

impl ClientFileWrite {
    pub fn new() -> Self {
        Self {
            spec: ToolSpec {
                name: "client_file_write".into(),
                description: "Write content to a file on the machine the connected \
                    editor is running on — NOT this agent's own machine. Use \
                    `file_write` instead for files on the agent's machine. \
                    Completely replaces the file's existing content. \
                    Only available inside an ACP session whose editor supports \
                    `fs/write_text_file`; refuses otherwise."
                    .into(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Absolute path on the editor's machine."
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

impl Default for ClientFileWrite {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Tool for ClientFileWrite {
    fn kind(&self) -> ToolKind {
        ToolKind::Edit
    }

    fn spec(&self) -> &ToolSpec {
        &self.spec
    }

    async fn execute(&self, input: &serde_json::Value) -> Result<String> {
        let path = input["path"].as_str().context("missing 'path'")?;
        let content = input["content"].as_str().context("missing 'content'")?;

        let client = current_acp_client().ok_or_else(no_editor_error)?;
        client.write_text_file(path, content).await?;
        Ok(format!("Written: {path} ({} bytes)", content.len()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::acp_client::tests::FakeClient;
    use crate::tools::acp_client::{AcpClient, scope_acp_client};
    use serde_json::json;
    use std::sync::Arc;

    /// `line` and `limit` exist in ACP because a coding agent reads big
    /// files in pieces. Passing them through is the reason to prefer
    /// this over shelling out to `sed`.
    #[tokio::test]
    async fn read_passes_line_and_limit_through() {
        let fake = Arc::new(FakeClient::default());
        let client: Arc<dyn AcpClient> = Arc::clone(&fake) as Arc<dyn AcpClient>;
        scope_acp_client(client, async {
            ClientFileRead::new()
                .execute(&json!({"path": "/p/a.rs", "line": 10, "limit": 40}))
                .await
                .unwrap();
        })
        .await;
        assert_eq!(
            fake.reads.lock().unwrap().as_slice(),
            &[("/p/a.rs".to_string(), Some(10), Some(40))]
        );
    }

    /// Outside an ACP turn there is no editor. Refusing here is what
    /// keeps a Discord message from reaching a tool that would have
    /// nowhere to go.
    #[tokio::test]
    async fn a_client_tool_refuses_without_a_client() {
        let err = ClientFileRead::new()
            .execute(&json!({"path": "/p/a.rs"}))
            .await
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("no editor"),
            "the message should say why, got: {err}"
        );
    }

    /// The editor's refusal is information, not a failure to swallow:
    /// the model can read it and try something else.
    #[tokio::test]
    async fn the_clients_error_reaches_the_model() {
        let fake = Arc::new(FakeClient::default());
        *fake.read_answer.lock().unwrap() = Some(Err("permission denied".to_string()));
        let client: Arc<dyn AcpClient> = Arc::clone(&fake) as Arc<dyn AcpClient>;
        scope_acp_client(client, async {
            let err = ClientFileRead::new()
                .execute(&json!({"path": "/p/secret"}))
                .await
                .unwrap_err()
                .to_string();
            assert!(err.contains("permission denied"), "got: {err}");
        })
        .await;
    }

    #[tokio::test]
    async fn write_sends_the_path_and_content() {
        let fake = Arc::new(FakeClient::default());
        let client: Arc<dyn AcpClient> = Arc::clone(&fake) as Arc<dyn AcpClient>;
        scope_acp_client(client, async {
            ClientFileWrite::new()
                .execute(&json!({"path": "/p/b.rs", "content": "fn main() {}"}))
                .await
                .unwrap();
        })
        .await;
        assert_eq!(
            fake.writes.lock().unwrap().as_slice(),
            &[("/p/b.rs".to_string(), "fn main() {}".to_string())]
        );
    }

    #[test]
    fn the_kinds_match_what_the_permission_table_expects() {
        assert_eq!(ClientFileRead::new().kind(), ToolKind::Read);
        assert_eq!(ClientFileWrite::new().kind(), ToolKind::Edit);
    }
}
