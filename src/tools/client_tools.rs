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
use crate::tools::acp_client::{ExitStatus, TerminalHandle, TerminalOutput, current_acp_client};
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

/// Render an exit status the same way regardless of caller —
/// `format_finished` (the one-shot path) and `ClientShellOutput` (the
/// long-running path) both need this tail, and duplicating the
/// three-way match would let the two drift apart.
fn format_exit_status(status: &ExitStatus) -> String {
    match (&status.exit_code, &status.signal) {
        (Some(code), _) => format!("\n[exit code: {code}]"),
        (None, Some(signal)) => format!("\n[terminated by signal: {signal}]"),
        (None, None) => "\n[exit status unknown]".to_string(),
    }
}

/// Render a finished command's output for the model: the (possibly
/// truncated) text plus how it ended.
fn format_finished(output: &TerminalOutput, status: &ExitStatus) -> String {
    let mut out = output.output.clone();
    if output.truncated {
        out.push_str("\n[output truncated]");
    }
    out.push_str(&format_exit_status(status));
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
/// running and the handle is handed back in the result text — and
/// tracked in `ServeState.acp_terminals`, the same registry
/// `ClientShellStart` uses, so it counts against the session's cap and
/// shows up if the model has to list what it is holding — so the
/// model can poll it with `client_shell_output` or stop it with
/// `client_shell_kill`. The decision to kill is left to the model or
/// the human, never made here on their behalf. This is a deliberate
/// departure from what the protocol's own `terminal/kill` doc suggests
/// (kill on timeout and collect the output).
///
/// The one new risk this creates is the model reading a timeout as a
/// failure and re-running the command — which for a non-idempotent
/// command is exactly the outcome not-releasing was meant to avoid.
/// The result text is worded so that misreading is not possible, and
/// ends with an explicit instruction not to re-run.
///
/// # Also subject to the session's terminal cap
///
/// A timed-out call leaves a handle tracked (see above), so without a
/// cap check here a model could loop `client_shell` with a short
/// `timeout_secs` and accumulate unbounded live processes on the
/// user's machine — exactly what `MAX_TERMINALS_PER_SESSION` exists to
/// prevent. So `execute` checks the same cap `ClientShellStart` does,
/// before calling `create_terminal`, and refuses the same way.
pub struct ClientShell {
    spec: ToolSpec,
}

impl ClientShell {
    pub fn new() -> Self {
        Self {
            spec: ToolSpec {
                name: "client_shell".into(),
                description: format!(
                    "Run a command on the machine the connected editor is \
                    running on — NOT this agent's own machine. Use `shell` instead \
                    for commands on the agent's machine. \
                    Waits up to `timeout_secs` (default 120, max 600) for the \
                    command to finish. If it finishes in time, returns its output, \
                    exit status, and whether the output was truncated. If it does \
                    NOT finish in time, the command is left running rather than \
                    killed — the result names the terminal handle so it can be \
                    checked or stopped later; do not re-run the command just \
                    because this call timed out. A session may hold at most \
                    {MAX_TERMINALS_PER_SESSION} terminals at once, counting both \
                    this tool's timed-out handles and client_shell_start's; \
                    starting one past that is refused. \
                    Only available inside an ACP session whose editor supports \
                    `terminal/*`; refuses otherwise."
                )
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

        // Same cap, same check, as `ClientShellStart`: a timed-out call
        // below leaves a handle tracked, so without this a model
        // looping `client_shell` with a short `timeout_secs` could
        // accumulate unbounded live processes — see this tool's doc.
        let held = client.tracked_terminals().await;
        if held.len() >= MAX_TERMINALS_PER_SESSION {
            return Err(cap_error(&held));
        }

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
            Err(_elapsed) => {
                // The handle escapes this call still running, so it has
                // to enter the same session-keyed tracking
                // `client_shell_start` uses — otherwise it would count
                // against nothing, `client_shell_start`'s cap would
                // never see it, and the model would have no way to
                // list it in order to clean it up.
                client.track_terminal(handle.clone()).await;
                Ok(format!(
                    "[timed out after {}s — the command is still running as terminal {handle}. \
                     It was not killed. Use client_shell_output to check on it, or \
                     client_shell_kill to stop it. Do not re-run the command.]",
                    timeout.as_secs()
                ))
            }
        }
    }
}

// ---------------------------------------------------------------------------
// client_shell_start / client_shell_output / client_shell_kill
// ---------------------------------------------------------------------------

/// How many terminals one session may hold at once.
///
/// A ceiling rather than a cleanup: nothing here is released on
/// disconnect (see `ServeState::acp_terminals`'s doc), so without a
/// cap a model that keeps starting commands and never killing them
/// would accumulate processes on the user's machine indefinitely.
/// Refusing the ninth — and naming the eight it is holding — makes the
/// model clean up rather than the agent guess which one is safe to
/// kill.
pub(crate) const MAX_TERMINALS_PER_SESSION: usize = 8;

/// The refusal `ClientShellStart` returns when the session is already
/// at the cap. Pulled out so the wording — and the handle substrings
/// (`t1`, `t{MAX_TERMINALS_PER_SESSION}`) the cap test keys on — lives
/// in one place.
fn cap_error(held: &[TerminalHandle]) -> anyhow::Error {
    let ids: Vec<String> = held.iter().map(TerminalHandle::to_string).collect();
    anyhow::anyhow!(
        "already holding the maximum of {MAX_TERMINALS_PER_SESSION} terminals for this \
         session: {}. Use client_shell_kill to free one before starting another.",
        ids.join(", ")
    )
}

/// Start a long-running command on the editor's machine and return
/// immediately with a terminal handle, rather than waiting for it to
/// finish the way `client_shell` does.
///
/// The handle is tracked against the session
/// (`ServeState.acp_terminals`) the moment the client hands it back —
/// before that, `client_shell_output`/`client_shell_kill` would have
/// nothing to check the model's handle against, and the cap below
/// would never see it.
///
/// The cap is checked *before* calling `create_terminal`: a session
/// already holding `MAX_TERMINALS_PER_SESSION` is refused without a
/// round trip to the editor, and the refusal names every handle held
/// so the model knows what to free.
pub struct ClientShellStart {
    spec: ToolSpec,
}

impl ClientShellStart {
    pub fn new() -> Self {
        Self {
            spec: ToolSpec {
                name: "client_shell_start".into(),
                description: format!(
                    "Start a long-running command on the machine the connected \
                    editor is running on — NOT this agent's own machine — and return \
                    immediately with a terminal handle instead of waiting for it to finish. \
                    Prefer this over `client_shell` for a command expected to keep running \
                    (a dev server, a watch task) or that may outlast a reasonable wait. \
                    Check on it with client_shell_output and stop it with client_shell_kill. \
                    A session may hold at most {MAX_TERMINALS_PER_SESSION} terminals at \
                    once; starting one more than that is refused, naming the handles \
                    already held, until one is freed. \
                    Only available inside an ACP session whose editor supports \
                    `terminal/*`; refuses otherwise."
                )
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
                        }
                    },
                    "required": ["command"]
                }),
            },
        }
    }
}

impl Default for ClientShellStart {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Tool for ClientShellStart {
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

        let client = current_acp_client().ok_or_else(no_editor_error)?;

        let held = client.tracked_terminals().await;
        if held.len() >= MAX_TERMINALS_PER_SESSION {
            return Err(cap_error(&held));
        }

        let handle = client
            .create_terminal(command, &args, cwd, Some(OUTPUT_CAP_BYTES as u64))
            .await?;
        client.track_terminal(handle.clone()).await;
        Ok(format!(
            "Started terminal {handle}. Use client_shell_output to check on it, or \
             client_shell_kill to stop it."
        ))
    }
}

/// Check on a command started by `client_shell_start` (or left running
/// by a `client_shell` timeout): its output so far, whether it has
/// finished, and its exit status if it has.
///
/// # An output error does NOT untrack the handle
///
/// The sanctioned reason to drop tracking without a release is "the
/// client says this handle doesn't exist" — but a `terminal_output`
/// error is not proof of that. A transient failure (the client
/// mid-reconnect, a timed-out request) surfaces through the same `Err`
/// as a genuinely unknown handle, and `AcpClient` gives no way to tell
/// the two apart. Untracking on every error would silently drop a
/// terminal that is still running: it stops counting against the cap
/// and disappears from the listing the model is told to clean up from,
/// with nothing left for the model to do about it.
///
/// So this tool leaves the handle tracked on any error and just
/// reports the client's message. Over-counting is the recoverable
/// direction — `client_shell_kill` untracks unconditionally (see its
/// doc), so the model can always clear a handle it no longer needs by
/// killing it, even if this tool keeps failing on it.
pub struct ClientShellOutput {
    spec: ToolSpec,
}

impl ClientShellOutput {
    pub fn new() -> Self {
        Self {
            spec: ToolSpec {
                name: "client_shell_output".into(),
                description: "Check on a command started with client_shell_start (or left \
                    running by a client_shell call that timed out): its output so far, \
                    whether it has finished, and its exit status if it has. Does not stop \
                    the command. Only available inside an ACP session whose editor supports \
                    `terminal/*`; refuses otherwise."
                    .into(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "terminal": {
                            "type": "string",
                            "description": "The terminal handle returned by client_shell_start \
                                (or by a client_shell call that timed out)."
                        }
                    },
                    "required": ["terminal"]
                }),
            },
        }
    }
}

impl Default for ClientShellOutput {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Tool for ClientShellOutput {
    fn kind(&self) -> ToolKind {
        ToolKind::Read
    }

    fn spec(&self) -> &ToolSpec {
        &self.spec
    }

    async fn execute(&self, input: &serde_json::Value) -> Result<String> {
        let terminal = input["terminal"].as_str().context("missing 'terminal'")?;
        let handle = TerminalHandle(terminal.to_string());

        let client = current_acp_client().ok_or_else(no_editor_error)?;
        // Deliberately does NOT untrack on error — see this tool's doc.
        // An error here could be transient rather than "this handle is
        // truly gone," and untracking a terminal that is still running
        // would be unrecoverable; leaving it tracked is not.
        let output = client.terminal_output(&handle).await?;

        let mut out = output.output.clone();
        if output.truncated {
            out.push_str("\n[output truncated]");
        }
        match &output.exit_status {
            Some(status) => out.push_str(&format_exit_status(status)),
            None => out.push_str("\n[still running]"),
        }
        Ok(out)
    }
}

/// Stop a command started by `client_shell_start` (or left running by
/// a `client_shell` timeout) and free its terminal handle.
///
/// Kills, then releases: ACP's `terminal/kill` alone leaves the handle
/// valid, so a caller that stopped there would leak it against the cap
/// forever. `release_terminal` — which the ACP schema defines as also
/// killing the command, redundantly with the kill just sent — is what
/// invalidates the handle on the client's side.
///
/// # Untracks unconditionally, even if `kill`/`release` error
///
/// This is the other half of `ClientShellOutput`'s asymmetry (see its
/// doc): an output error leaves a handle tracked because untracking a
/// terminal that might still be running is unrecoverable. A kill is
/// the opposite case — it is the action the cap's own refusal message
/// tells the model to take to free a slot, and a handle the client has
/// genuinely forgotten would otherwise error here forever and jam that
/// slot for good. So both wire calls are attempted and the handle is
/// dropped from tracking regardless of whether either succeeded; only
/// then does a real error from either call propagate to the model.
/// Over-counting from here is recoverable (the model can call this
/// tool again, or check with `client_shell_output`); a permanently
/// stuck slot is not.
pub struct ClientShellKill {
    spec: ToolSpec,
}

impl ClientShellKill {
    pub fn new() -> Self {
        Self {
            spec: ToolSpec {
                name: "client_shell_kill".into(),
                description: format!(
                    "Stop a command started with client_shell_start (or left \
                    running by a client_shell call that timed out) and free its terminal \
                    handle. Use this to make room under the {MAX_TERMINALS_PER_SESSION}-\
                    terminal cap, or to give up on a command that is no longer needed. \
                    Only available inside an ACP session whose editor supports \
                    `terminal/*`; refuses otherwise."
                )
                .into(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "terminal": {
                            "type": "string",
                            "description": "The terminal handle to stop and free."
                        }
                    },
                    "required": ["terminal"]
                }),
            },
        }
    }
}

impl Default for ClientShellKill {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl Tool for ClientShellKill {
    fn kind(&self) -> ToolKind {
        ToolKind::Execute
    }

    fn spec(&self) -> &ToolSpec {
        &self.spec
    }

    async fn execute(&self, input: &serde_json::Value) -> Result<String> {
        let terminal = input["terminal"].as_str().context("missing 'terminal'")?;
        let handle = TerminalHandle(terminal.to_string());

        let client = current_acp_client().ok_or_else(no_editor_error)?;

        // Both attempted before either `?` — see this tool's doc for
        // why: a kill failure must not skip the release attempt, and
        // the handle must come out of tracking even if both failed.
        let kill = client.kill_terminal(&handle).await;
        let release = client.release_terminal(&handle).await;
        client.untrack_terminal(&handle).await;
        kill?;
        release?;

        Ok(format!("Stopped and released terminal {handle}."))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::serve::ServeState;
    use crate::tools::acp_client::tests::FakeClient;
    use crate::tools::acp_client::{AcpClient, scope_acp_client};
    use serde_json::json;
    use std::sync::Arc;

    /// The session id [`shell_test_state`]'s `FakeClient` tracks
    /// terminals under, and the key the tests below read back off
    /// `ServeState.acp_terminals`.
    const TEST_SESSION_ID: &str = "client-tools-test-session";

    /// A `ServeState` and a `FakeClient` wired to share one terminal
    /// registry under [`TEST_SESSION_ID`] — the same relationship
    /// `AcpProgress`/`AcpClientHandle` have to a real `ServeState` in
    /// production (`src/serve/acp.rs`), so driving a tool purely
    /// through the fake is visible on `state.acp_terminals` exactly the
    /// way it would be for a real session.
    async fn shell_test_state() -> (Arc<ServeState>, Arc<FakeClient>) {
        let state = ServeState::for_test(true);
        // Field assignment rather than `FakeClient { .., ..Default::default() }`:
        // struct-update syntax requires every field to be visible from
        // the call site, including the ones left untouched, and most of
        // `FakeClient`'s fields are private to `acp_client`'s own test
        // module.
        let mut fake = FakeClient::default();
        fake.terminal_session = TEST_SESSION_ID.to_string();
        fake.terminals = Arc::clone(&state.acp_terminals);
        (state, Arc::new(fake))
    }

    /// Drives the real connection-teardown path
    /// (`crate::serve::acp::release_connection_sessions`) for one
    /// session id, rather than reimplementing "a connection closed"
    /// locally. The property under test is *what does not happen* —
    /// `acp_terminals` is untouched — so a test that only exercised a
    /// stand-in would not stop a future release from being added to
    /// the real path.
    async fn simulate_connection_teardown(state: &Arc<ServeState>, session_id: &str) {
        crate::serve::acp::release_connection_sessions(state, vec![session_id.to_string()]).await;
    }

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
        assert_eq!(ClientShellStart::new().kind(), ToolKind::Execute);
        assert_eq!(ClientShellOutput::new().kind(), ToolKind::Read);
        assert_eq!(ClientShellKill::new().kind(), ToolKind::Execute);
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

    // -----------------------------------------------------------------
    // client_shell_start / client_shell_output / client_shell_kill
    // -----------------------------------------------------------------

    /// Outside an ACP turn there is no editor, exactly as for the
    /// other client-side tools.
    #[tokio::test]
    async fn the_long_running_tools_refuse_without_a_client() {
        assert!(
            ClientShellStart::new()
                .execute(&json!({"command": "ls", "args": []}))
                .await
                .unwrap_err()
                .to_string()
                .contains("no editor")
        );
        assert!(
            ClientShellOutput::new()
                .execute(&json!({"terminal": "t1"}))
                .await
                .unwrap_err()
                .to_string()
                .contains("no editor")
        );
        assert!(
            ClientShellKill::new()
                .execute(&json!({"terminal": "t1"}))
                .await
                .unwrap_err()
                .to_string()
                .contains("no editor")
        );
    }

    /// The handle has to be recorded against the session, because that
    /// is what a reconnecting client's next turn will look it up by.
    #[tokio::test]
    async fn start_records_the_handle_against_the_session() {
        let (state, fake) = shell_test_state().await;
        let client: Arc<dyn AcpClient> = Arc::clone(&fake) as Arc<dyn AcpClient>;
        scope_acp_client(client, async {
            ClientShellStart::new()
                .execute(&json!({"command": "cargo", "args": ["test"]}))
                .await
                .unwrap();
        })
        .await;

        let held = state.acp_terminals.lock().await;
        assert_eq!(held.get(TEST_SESSION_ID).map(Vec::len), Some(1));
    }

    /// `kill` alone leaves the handle valid — the protocol says so, and
    /// says to release it afterwards. Doing only half would leak a
    /// handle against the cap forever.
    #[tokio::test]
    async fn kill_stops_the_command_and_then_frees_the_handle() {
        let (state, fake) = shell_test_state().await;
        let client: Arc<dyn AcpClient> = Arc::clone(&fake) as Arc<dyn AcpClient>;
        scope_acp_client(client, async {
            ClientShellStart::new()
                .execute(&json!({"command": "sleep", "args": ["999"]}))
                .await
                .unwrap();
            ClientShellKill::new()
                .execute(&json!({"terminal": "t1"}))
                .await
                .unwrap();
        })
        .await;

        assert_eq!(
            fake.killed.lock().unwrap().len(),
            1,
            "the command is stopped"
        );
        assert_eq!(
            fake.released.lock().unwrap().len(),
            1,
            "and the handle freed"
        );
        assert!(
            state
                .acp_terminals
                .lock()
                .await
                .get(TEST_SESSION_ID)
                .is_none_or(Vec::is_empty),
            "and it is no longer tracked"
        );
    }

    /// The cap has to name what is holding it. A bare "too many
    /// terminals" leaves the model with nothing to act on.
    #[tokio::test]
    async fn the_cap_refuses_and_lists_what_is_held() {
        let (_state, fake) = shell_test_state().await;
        fake.hand_out_distinct_handles();
        let client: Arc<dyn AcpClient> = Arc::clone(&fake) as Arc<dyn AcpClient>;

        let refusal = scope_acp_client(client, async {
            for _ in 0..MAX_TERMINALS_PER_SESSION {
                ClientShellStart::new()
                    .execute(&json!({"command": "sleep", "args": ["999"]}))
                    .await
                    .unwrap();
            }
            ClientShellStart::new()
                .execute(&json!({"command": "one", "args": ["too", "many"]}))
                .await
                .unwrap_err()
                .to_string()
        })
        .await;

        assert!(refusal.contains("t1"), "names a held handle: {refusal}");
        assert!(
            refusal.contains(&format!("t{MAX_TERMINALS_PER_SESSION}")),
            "names the last one too: {refusal}"
        );
        assert_eq!(
            fake.creates.lock().unwrap().len(),
            MAX_TERMINALS_PER_SESSION,
            "the refused call must not have reached the client"
        );
    }

    /// Review round 1, Finding 2: an output error alone is not proof the
    /// client has forgotten the handle — it could be transient (a
    /// reconnect, a timed-out request) — so it must not untrack a
    /// terminal that might still be running. Only `client_shell_kill`
    /// untracks unconditionally, because a handle the model explicitly
    /// asked to kill is exactly the case the cap's refusal message
    /// points the model at. This is a stronger pair than "an unknown
    /// handle is dropped from tracking" was: it proves the asymmetry in
    /// both directions, not just that *something* untracks eventually.
    #[tokio::test]
    async fn an_unknown_handle_is_dropped_from_tracking() {
        let (state, fake) = shell_test_state().await;
        let client: Arc<dyn AcpClient> = Arc::clone(&fake) as Arc<dyn AcpClient>;

        let output_err = scope_acp_client(Arc::clone(&client), async {
            ClientShellStart::new()
                .execute(&json!({"command": "sleep", "args": ["999"]}))
                .await
                .unwrap();

            fake.make_output_fail_with("no such terminal");
            ClientShellOutput::new()
                .execute(&json!({"terminal": "t1"}))
                .await
                .unwrap_err()
                .to_string()
        })
        .await;

        assert!(
            output_err.contains("no such terminal"),
            "the client's words reach the model: {output_err}"
        );
        assert_eq!(
            state
                .acp_terminals
                .lock()
                .await
                .get(TEST_SESSION_ID)
                .map(Vec::len),
            Some(1),
            "an output error alone must not drop tracking — the terminal could still be \
             running, and untracking it would lose it for good"
        );
        assert!(
            fake.released.lock().unwrap().is_empty(),
            "an output error must not release either — only a kill does that"
        );

        // Even a client that keeps saying "no such terminal" on kill
        // and release must not leave the slot stuck forever.
        fake.make_kill_fail_with("no such terminal");
        let kill_err = scope_acp_client(client, async {
            ClientShellKill::new()
                .execute(&json!({"terminal": "t1"}))
                .await
                .unwrap_err()
                .to_string()
        })
        .await;

        assert!(
            kill_err.contains("no such terminal"),
            "the failure is still reported: {kill_err}"
        );
        assert!(
            state
                .acp_terminals
                .lock()
                .await
                .get(TEST_SESSION_ID)
                .is_none_or(Vec::is_empty),
            "but a kill frees the handle from tracking regardless of whether the client \
             errors on it — otherwise the slot the cap points the model at could never \
             be freed"
        );
    }

    /// The property this whole task is shaped around. `terminal/release`
    /// kills the command, so a dropped socket must not trigger one — a
    /// network blip would otherwise kill the user's build.
    ///
    /// What this does and does not fence: `release_connection_sessions`
    /// takes only `&Arc<ServeState>` and a list of session ids — it has
    /// no route to any `AcpClient` — so `fake.released`/`fake.killed`
    /// staying empty is true by construction here and can never fail.
    /// The load-bearing assertion is the last one: the handle is still
    /// tracked after teardown. A release added to `serve_connection`'s
    /// teardown *outside* `release_connection_sessions` (rather than
    /// inside it) would not be caught by this test at all — only by
    /// code review of that call site, or a wire-level test in
    /// `src/serve/acp.rs`.
    #[tokio::test]
    async fn a_connection_ending_releases_nothing() {
        let (state, fake) = shell_test_state().await;
        let client: Arc<dyn AcpClient> = Arc::clone(&fake) as Arc<dyn AcpClient>;
        scope_acp_client(client, async {
            ClientShellStart::new()
                .execute(&json!({"command": "cargo", "args": ["test"]}))
                .await
                .unwrap();
        })
        .await;

        simulate_connection_teardown(&state, TEST_SESSION_ID).await;

        assert!(
            fake.released.lock().unwrap().is_empty(),
            "release kills the command; a lost socket is not a reason to"
        );
        assert!(fake.killed.lock().unwrap().is_empty());
        assert_eq!(
            state
                .acp_terminals
                .lock()
                .await
                .get(TEST_SESSION_ID)
                .map(Vec::len),
            Some(1),
            "and the handle stays addressable for a client that reconnects"
        );
    }

    /// Task 6's ruling that the plan itself does not state: a one-shot
    /// `client_shell` call that outruns its timeout must be tracked too
    /// — not just handed back in the result text. Otherwise the handle
    /// escapes both the cap and the "what is holding this session"
    /// listing, and the model is told to clean up while the very thing
    /// it needs to clean up stays invisible.
    #[tokio::test]
    async fn a_timed_out_one_shot_is_tracked_against_the_session() {
        let (state, fake) = shell_test_state().await;
        fake.make_exit_never_return();
        let client: Arc<dyn AcpClient> = Arc::clone(&fake) as Arc<dyn AcpClient>;

        scope_acp_client(client, async {
            ClientShell::new()
                .execute(&json!({"command": "cargo", "args": ["test"], "timeout_secs": 1}))
                .await
                .unwrap();
        })
        .await;

        assert_eq!(
            state
                .acp_terminals
                .lock()
                .await
                .get(TEST_SESSION_ID)
                .map(Vec::len),
            Some(1),
            "a timed-out one-shot's handle must be tracked, or it escapes the cap and \
             the model can never see it to clean it up"
        );
    }

    /// Review round 1, Finding 1: the one-shot path must respect the
    /// same cap `client_shell_start` does. `ClientShell`'s timeout
    /// branch tracks a handle (previous test), so without a cap check
    /// on this path too, a model looping `client_shell` with a short
    /// `timeout_secs` could accumulate live processes past the cap the
    /// same way looping `client_shell_start` would — exactly what
    /// `MAX_TERMINALS_PER_SESSION` exists to prevent.
    #[tokio::test]
    async fn the_one_shot_path_is_also_capped() {
        let (_state, fake) = shell_test_state().await;
        fake.hand_out_distinct_handles();
        let client: Arc<dyn AcpClient> = Arc::clone(&fake) as Arc<dyn AcpClient>;

        let refusal = scope_acp_client(client, async {
            for _ in 0..MAX_TERMINALS_PER_SESSION {
                ClientShellStart::new()
                    .execute(&json!({"command": "sleep", "args": ["999"]}))
                    .await
                    .unwrap();
            }
            ClientShell::new()
                .execute(&json!({"command": "one", "args": ["too", "many"]}))
                .await
                .unwrap_err()
                .to_string()
        })
        .await;

        assert!(refusal.contains("t1"), "names a held handle: {refusal}");
        assert_eq!(
            fake.creates.lock().unwrap().len(),
            MAX_TERMINALS_PER_SESSION,
            "the refused one-shot call must not have reached the client either"
        );
    }
}
