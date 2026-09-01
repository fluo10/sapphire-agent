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
use crate::tools::acp_client::{ExitStatus, TerminalOutput, current_acp_client};
use crate::tools::{OUTPUT_CAP_BYTES, Tool, ToolKind};
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

// ---------------------------------------------------------------------------
// client_shell
// ---------------------------------------------------------------------------

/// The cap on how long the one-shot form waits. Past this the command
/// keeps running and the caller gets its handle back instead of a
/// result — see [`ClientShell`]'s doc for why releasing (which the ACP
/// schema defines as killing) is wrong here.
const DEFAULT_TIMEOUT_SECS: u64 = 120;
const MAX_TIMEOUT_SECS: u64 = 600;

/// Clamp a requested timeout (seconds) to `(0, MAX_TIMEOUT_SECS]`,
/// defaulting to `DEFAULT_TIMEOUT_SECS` when the caller didn't ask for
/// one. Pulled out of `execute` so the cap can be tested directly
/// instead of a test waiting out a real timeout.
fn clamp_timeout(requested: Option<u64>) -> std::time::Duration {
    std::time::Duration::from_secs(
        requested
            .unwrap_or(DEFAULT_TIMEOUT_SECS)
            .min(MAX_TIMEOUT_SECS),
    )
}

/// Render a finished command's output for the model: the (possibly
/// truncated) text plus how it ended.
fn format_finished(output: &TerminalOutput, status: &ExitStatus) -> String {
    let mut out = output.output.clone();
    if output.truncated {
        out.push_str("\n[output truncated]");
    }
    match (&status.exit_code, &status.signal) {
        (Some(code), _) => out.push_str(&format!("\n[exit code: {code}]")),
        (None, Some(signal)) => out.push_str(&format!("\n[terminated by signal: {signal}]")),
        (None, None) => out.push_str("\n[exit status unknown]"),
    }
    out
}

/// Run a command on the machine the editor is running on, and wait for
/// it to finish — up to a timeout.
///
/// # A timed-out command is not killed
///
/// ACP's `terminal/release` kills the command it releases — the schema
/// says so explicitly, the same way `terminal/kill` does. Releasing on
/// timeout would therefore throw away a build that has already run for
/// however long the timeout allowed, and for a non-idempotent command
/// (`git push`, a migration, a script that writes files) a retry after
/// that would run it a second time.
///
/// So on timeout this tool releases nothing. The terminal keeps
/// running and the handle is handed back in the result text, so the
/// model can poll it with `client_shell_output` or stop it with
/// `client_shell_kill` (both land in a later task) — the decision to
/// kill is left to the model or the human, never made here on their
/// behalf. This is a deliberate departure from what the protocol's own
/// `terminal/kill` doc suggests (kill on timeout and collect the
/// output).
///
/// The one new risk this creates is the model reading a timeout as a
/// failure and re-running the command — which for a non-idempotent
/// command is exactly the outcome not-releasing was meant to avoid.
/// The result text is worded so that misreading is not possible, and
/// ends with an explicit instruction not to re-run.
pub struct ClientShell {
    spec: ToolSpec,
}

impl ClientShell {
    pub fn new() -> Self {
        Self {
            spec: ToolSpec {
                name: "client_shell".into(),
                description: "Run a command on the machine the connected editor is \
                    running on — NOT this agent's own machine. Use `shell` instead \
                    for commands on the agent's machine. \
                    Waits up to `timeout_secs` (default 120, max 600) for the \
                    command to finish. If it finishes in time, returns its output, \
                    exit status, and whether the output was truncated. If it does \
                    NOT finish in time, the command is left running rather than \
                    killed — the result names the terminal handle so it can be \
                    checked or stopped later; do not re-run the command just \
                    because this call timed out. \
                    Only available inside an ACP session whose editor supports \
                    `terminal/*`; refuses otherwise."
                    .into(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "The command to run (not a shell string — no pipes or redirection)."
                        },
                        "args": {
                            "type": "array",
                            "items": { "type": "string" },
                            "description": "Arguments to pass to the command."
                        },
                        "cwd": {
                            "type": "string",
                            "description": "Working directory on the editor's machine. Defaults to the session's cwd."
                        },
                        "timeout_secs": {
                            "type": "integer",
                            "description": "Max seconds to wait for the command to finish before handing back a running handle instead (default: 120, max: 600).",
                            "minimum": 1,
                            "maximum": 600
                        }
                    },
                    "required": ["command"]
                }),
            },
        }
    }
}

impl Default for ClientShell {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Tool for ClientShell {
    fn kind(&self) -> ToolKind {
        ToolKind::Execute
    }

    fn spec(&self) -> &ToolSpec {
        &self.spec
    }

    async fn execute(&self, input: &serde_json::Value) -> Result<String> {
        let command = input["command"].as_str().context("missing 'command'")?;
        let args: Vec<String> = input["args"]
            .as_array()
            .map(|a| {
                a.iter()
                    .filter_map(|v| v.as_str().map(String::from))
                    .collect()
            })
            .unwrap_or_default();
        let cwd = input["cwd"].as_str();
        let timeout = clamp_timeout(input["timeout_secs"].as_u64());

        let client = current_acp_client().ok_or_else(no_editor_error)?;
        let handle = client
            .create_terminal(command, &args, cwd, Some(OUTPUT_CAP_BYTES as u64))
            .await?;

        match tokio::time::timeout(timeout, client.wait_for_terminal_exit(&handle)).await {
            Ok(status) => {
                let status = status?;
                let output = client.terminal_output(&handle).await?;
                client.release_terminal(&handle).await?;
                Ok(format_finished(&output, &status))
            }
            Err(_elapsed) => Ok(format!(
                "[timed out after {}s — the command is still running as terminal {handle}. \
                 It was not killed. Use client_shell_output to check on it, or \
                 client_shell_kill to stop it. Do not re-run the command.]",
                timeout.as_secs()
            )),
        }
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
        assert_eq!(ClientShell::new().kind(), ToolKind::Execute);
    }

    /// Outside an ACP turn there is no editor to run a command on,
    /// exactly as for the two file tools.
    #[tokio::test]
    async fn client_shell_refuses_without_a_client() {
        let err = ClientShell::new()
            .execute(&json!({"command": "ls", "args": []}))
            .await
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("no editor"),
            "the message should say why, got: {err}"
        );
    }

    /// The whole point of the timeout: a build that outruns it keeps
    /// running, and the model is handed the handle instead of a
    /// corpse. Killing here would throw away the work and, for a
    /// non-idempotent command, run it twice.
    #[tokio::test]
    async fn a_timed_out_command_is_not_killed_and_hands_back_its_handle() {
        let fake = Arc::new(FakeClient::default());
        fake.make_exit_never_return();
        let client: Arc<dyn AcpClient> = Arc::clone(&fake) as Arc<dyn AcpClient>;

        let out = scope_acp_client(client, async {
            ClientShell::new()
                .execute(&json!({"command": "cargo", "args": ["test"], "timeout_secs": 1}))
                .await
                .unwrap()
        })
        .await;

        assert!(out.contains("still running"), "got: {out}");
        assert!(
            out.contains("t1"),
            "the handle must be in the result: {out}"
        );
        assert!(
            fake.released.lock().unwrap().is_empty(),
            "release kills the command — it must not be called on a timeout"
        );
        assert!(fake.killed.lock().unwrap().is_empty(), "nor kill");
    }

    #[tokio::test]
    async fn a_command_that_finishes_in_time_is_released() {
        let fake = Arc::new(FakeClient::default());
        let client: Arc<dyn AcpClient> = Arc::clone(&fake) as Arc<dyn AcpClient>;
        scope_acp_client(client, async {
            ClientShell::new()
                .execute(&json!({"command": "ls", "args": []}))
                .await
                .unwrap();
        })
        .await;
        assert_eq!(
            fake.released.lock().unwrap().len(),
            1,
            "the handle is freed"
        );
    }

    /// The cap is handed to the client so the output is cut at the
    /// source rather than shipped across the wire and cut here.
    #[tokio::test]
    async fn the_output_cap_is_passed_to_the_client() {
        let fake = Arc::new(FakeClient::default());
        let client: Arc<dyn AcpClient> = Arc::clone(&fake) as Arc<dyn AcpClient>;
        scope_acp_client(client, async {
            ClientShell::new()
                .execute(&json!({"command": "ls", "args": []}))
                .await
                .unwrap();
        })
        .await;
        let (_, _, _, limit) = fake.creates.lock().unwrap()[0].clone();
        assert_eq!(limit, Some(crate::tools::OUTPUT_CAP_BYTES as u64));
    }

    /// `clamp_timeout` is what `execute` calls to turn a requested
    /// timeout into the duration it actually waits — testing it
    /// directly avoids a test that waits out a real 600s timeout.
    #[test]
    fn the_timeout_is_capped_at_ten_minutes() {
        assert_eq!(
            clamp_timeout(Some(9999)),
            std::time::Duration::from_secs(MAX_TIMEOUT_SECS)
        );
        assert_eq!(
            clamp_timeout(None),
            std::time::Duration::from_secs(DEFAULT_TIMEOUT_SECS)
        );
        assert_eq!(clamp_timeout(Some(30)), std::time::Duration::from_secs(30));
    }
}
