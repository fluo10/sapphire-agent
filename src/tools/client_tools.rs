//! Tools that reach the editor's machine over ACP, rather than this
//! agent's own filesystem.
//!
//! `file_read`/`file_write` (`src/tools/builtin_tools.rs`) touch the
//! machine this agent runs on. These two touch the machine the *editor*
//! runs on, via `fs/read_text_file` and `fs/write_text_file` — see
//! `crate::tools::acp_client`. Both sets can be offered in the same
//! turn's tool list at once — with host access on, an ACP turn sees
//! both `file_read` and `client_file_read`, both `shell` and
//! `client_shell` — the tool descriptions are what disambiguate which
//! machine each one reaches. Outside an ACP session there is no client
//! to ask, so the client-side tools refuse rather than silently doing
//! nothing.

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
/// on. With host access on, an ACP turn may be offered both; the tool
/// descriptions are what tell the model which machine each one reaches.
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
/// runs on. With host access on, an ACP turn may be offered both; the
/// tool descriptions are what tell the model which machine each one
/// reaches.
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
            .clamp(1, MAX_TIMEOUT_SECS),
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
        //
        // Reserve-then-create, not read-then-write: `run_llm_turn` runs
        // a turn's permitted calls concurrently, so one assistant
        // message with several `client_shell`/`client_shell_start`
        // calls must not let them all read the count before any of
        // them wrote it back. `try_reserve_terminal_slot` does the
        // check and the reservation in one lock span so that cannot
        // happen. See its doc.
        let reservation = client
            .try_reserve_terminal_slot()
            .await
            .map_err(|held| cap_error(&held))?;

        // If `create_terminal` errors, or this whole call is cancelled
        // before it returns — the turn's future dropped mid-RPC, which
        // is exactly the cancellation ACP treats as routine (Escape in
        // an editor, a dropped socket) — `reservation` is dropped here
        // without ever reaching `track_terminal` below. Its `Drop`
        // frees the slot itself; nothing further to do on this path.
        // See `TerminalReservation`'s doc.
        let handle = client
            .create_terminal(command, &args, cwd, Some(OUTPUT_CAP_BYTES as u64))
            .await?;
        // Tracked immediately, before anything that can fail or be
        // dropped: a `wait_for_terminal_exit` error below, or this
        // whole future being dropped mid-wait (turn cancellation), must
        // not lose a command that is genuinely still running on the
        // user's machine. This also consumes `reservation`, resolving
        // it into the real handle so its `Drop` won't also try to free
        // the slot out from under the now-tracked handle.
        client.track_terminal(reservation, handle.clone()).await;

        match tokio::time::timeout(timeout, client.wait_for_terminal_exit(&handle)).await {
            Ok(status) => {
                // On error, the handle stays tracked (already tracked
                // above) rather than being lost — the command may still
                // be running and a transient RPC error here is not
                // proof otherwise. See `ClientShellOutput`'s doc for the
                // same reasoning applied to polling.
                let status = status?;
                let output = client.terminal_output(&handle).await?;
                match client.release_terminal(&handle).await {
                    Ok(()) => {
                        client.untrack_terminal(&handle).await;
                        Ok(format_finished(&output, &status))
                    }
                    Err(e) => {
                        // The command finished and its output was
                        // already collected successfully — that must
                        // not be thrown away just because the release
                        // that follows failed. The handle is left
                        // tracked (over-counting is recoverable; losing
                        // a finished build's output is not), so the
                        // model can retry `client_shell_kill` to free it.
                        Ok(format!(
                            "{}\n[warning: the command finished, but releasing terminal \
                             {handle} failed: {e}. It may still be tracked; use \
                             client_shell_kill to free it.]",
                            format_finished(&output, &status)
                        ))
                    }
                }
            }
            Err(_elapsed) => {
                // Already tracked above — the handle escapes this call
                // still running, so it has to stay in the same
                // session-keyed tracking `client_shell_start` uses,
                // otherwise it would count against nothing, the cap
                // would never see it, and the model would have no way
                // to list it in order to clean it up.
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
///
/// `held.handles` alone can be shorter than `MAX_TERMINALS_PER_SESSION`
/// even though the session really is at the cap: a concurrent call
/// elsewhere in this same turn may be mid-`create_terminal`, holding a
/// reservation with no handle yet. Naming only `held.handles` in that
/// case would tell the model it holds fewer terminals than the cap it
/// is being refused against, with no way to reconcile the two — so any
/// in-flight reservations are appended as a count instead of being
/// silently dropped from the listing.
pub(crate) fn cap_error(held: &crate::tools::acp_client::CapHeld) -> anyhow::Error {
    let mut parts: Vec<String> = held.handles.iter().map(TerminalHandle::to_string).collect();
    if held.reservations > 0 {
        parts.push(format!(
            "{} more still starting (no handle yet)",
            held.reservations
        ));
    }
    anyhow::anyhow!(
        "already holding the maximum of {MAX_TERMINALS_PER_SESSION} terminals for this \
         session: {}. Use client_shell_kill to free one before starting another.",
        parts.join(", ")
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

        // Reserve-then-create, not read-then-write: see
        // `AcpClient::try_reserve_terminal_slot`'s doc. `run_llm_turn`
        // runs a turn's permitted calls concurrently, so one assistant
        // message containing several `client_shell_start` blocks must
        // not let them all read the count before any of them wrote it
        // back — a real, reachable way to bypass the cap within one
        // turn, not just across concurrent prompts.
        let reservation = client
            .try_reserve_terminal_slot()
            .await
            .map_err(|held| cap_error(&held))?;

        // As in `ClientShell`: if `create_terminal` errors or this call
        // is cancelled before it returns, `reservation` is dropped
        // without reaching `track_terminal`, and its `Drop` frees the
        // slot on its own.
        let handle = client
            .create_terminal(command, &args, cwd, Some(OUTPUT_CAP_BYTES as u64))
            .await?;
        client.track_terminal(reservation, handle.clone()).await;
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

        let held = state.acp_terminals.lock().unwrap();
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
                .unwrap()
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
                .unwrap()
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
                .unwrap()
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
                .unwrap()
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
                .unwrap()
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

    // -----------------------------------------------------------------
    // Final review, Fix 1 & Fix 2
    // -----------------------------------------------------------------

    /// Fix 1, item 1: a `wait_for_terminal_exit` error (the client
    /// mid-reconnect, an RPC timeout) is not proof the command has
    /// stopped — it is still running on the user's machine. The handle
    /// must stay tracked so the model can still poll or kill it later;
    /// losing it here is exactly the "under-counting loses a live
    /// process" outcome the design rules out.
    #[tokio::test]
    async fn a_wait_for_exit_error_leaves_the_handle_tracked() {
        let (state, fake) = shell_test_state().await;
        fake.make_wait_fail_with("connection reset");
        let client: Arc<dyn AcpClient> = Arc::clone(&fake) as Arc<dyn AcpClient>;

        let err = scope_acp_client(client, async {
            ClientShell::new()
                .execute(&json!({"command": "cargo", "args": ["build"]}))
                .await
                .unwrap_err()
                .to_string()
        })
        .await;

        assert!(err.contains("connection reset"), "got: {err}");
        assert_eq!(
            state
                .acp_terminals
                .lock()
                .unwrap()
                .get(TEST_SESSION_ID)
                .map(Vec::len),
            Some(1),
            "a wait-for-exit error must not drop tracking — the command may still be running"
        );
    }

    /// Fix 1, item 3: a `release_terminal` failure must not discard
    /// output that was already collected successfully. A finished
    /// build's output is real work; throwing it away because the
    /// unrelated release call that follows failed would be worse than
    /// reporting both.
    #[tokio::test]
    async fn a_release_error_still_returns_the_output_and_leaves_the_handle_tracked() {
        let (state, fake) = shell_test_state().await;
        // Also fails `kill_terminal`, but this path never calls it.
        fake.make_kill_fail_with("no such terminal");
        let client: Arc<dyn AcpClient> = Arc::clone(&fake) as Arc<dyn AcpClient>;

        let out = scope_acp_client(client, async {
            ClientShell::new()
                .execute(&json!({"command": "cargo", "args": ["build"]}))
                .await
                .unwrap()
        })
        .await;

        assert!(
            out.contains("[exit status unknown]"),
            "the finished command's output must still be reported: {out}"
        );
        assert!(
            out.contains("no such terminal"),
            "the release failure must be surfaced too, not swallowed: {out}"
        );
        assert_eq!(
            state
                .acp_terminals
                .lock()
                .unwrap()
                .get(TEST_SESSION_ID)
                .map(Vec::len),
            Some(1),
            "the handle stays tracked — a release failure is not proof it actually freed"
        );
    }

    /// Fix 2: `run_llm_turn` executes a turn's permitted tool calls
    /// concurrently (`futures_util::future::join_all`, `src/serve/mod.rs`),
    /// so one assistant message containing several `client_shell_start`
    /// blocks runs them all at once against the same session. The old
    /// code read `tracked_terminals()` and wrote `track_terminal()` as
    /// two separate steps, so every concurrent call could read the
    /// count before any of them wrote it back — bypassing the cap
    /// within a single turn, not just across the (unsupported)
    /// concurrent-prompts case. `try_reserve_terminal_slot` closes that
    /// by making the check and the reservation one atomic step.
    ///
    /// `FakeClient::create_terminal` yields once (simulating the real
    /// RPC round trip `AcpClientHandle::create_terminal` makes) so the
    /// concurrent calls below actually interleave between their reserve
    /// and their track step — without that, this test would pass even
    /// against the old, buggy read-then-write code.
    #[tokio::test]
    async fn concurrent_starts_cannot_exceed_the_cap() {
        let (state, fake) = shell_test_state().await;
        fake.hand_out_distinct_handles();
        let client: Arc<dyn AcpClient> = Arc::clone(&fake) as Arc<dyn AcpClient>;

        let attempts = MAX_TERMINALS_PER_SESSION + 4;
        let inputs: Vec<serde_json::Value> = (0..attempts)
            .map(|_| json!({"command": "sleep", "args": ["999"]}))
            .collect();
        let tool = ClientShellStart::new();

        let outcomes = scope_acp_client(client, async {
            futures_util::future::join_all(inputs.iter().map(|input| tool.execute(input))).await
        })
        .await;

        let succeeded = outcomes.iter().filter(|r| r.is_ok()).count();
        assert_eq!(
            succeeded, MAX_TERMINALS_PER_SESSION,
            "one turn's concurrent client_shell_start calls must not exceed the cap: {outcomes:?}"
        );
        assert_eq!(
            state
                .acp_terminals
                .lock()
                .unwrap()
                .get(TEST_SESSION_ID)
                .map(Vec::len),
            Some(MAX_TERMINALS_PER_SESSION),
            "the registry itself must never hold more than the cap, even transiently"
        );
    }

    // -----------------------------------------------------------------
    // Reservation-leak fix: cancellation between reserve and track
    // -----------------------------------------------------------------

    /// The regression this whole task exists to close, reproduced and
    /// then disproved.
    ///
    /// Before `TerminalReservation` existed, the placeholder
    /// `try_reserve_terminal_slot` pushed was freed only by
    /// `track_terminal` or an explicit `untrack_terminal` on
    /// `create_terminal` failure — neither of which runs when the
    /// call's future is dropped in between (a cancelled turn: Escape in
    /// an editor, a dropped socket firing `connection_cancel`, exactly
    /// what `create_terminal`'s RPC being in flight is the routine case
    /// for). The placeholder stayed in the registry forever, and after
    /// `MAX_TERMINALS_PER_SESSION` such cancellations the session could
    /// never start another terminal again, on this connection or any
    /// reconnect.
    ///
    /// Cancelling `MAX_TERMINALS_PER_SESSION + 1` times — one more than
    /// the cap — and then still succeeding is the sharpest version of
    /// this: if even a single cancellation leaked its slot, the cap
    /// would already be hit and the final start below would be refused
    /// instead of succeeding.
    #[tokio::test]
    async fn a_dropped_reservation_does_not_leak_the_slot() {
        let (state, fake) = shell_test_state().await;
        // Parks every `create_terminal` call after it records itself
        // and after the reservation above it has already landed in the
        // registry — the same window a real cancellation drops the
        // turn's future in.
        fake.make_create_terminal_hang();
        let client: Arc<dyn AcpClient> = Arc::clone(&fake) as Arc<dyn AcpClient>;

        for _ in 0..(MAX_TERMINALS_PER_SESSION + 1) {
            let task_client = Arc::clone(&client);
            let join = tokio::spawn(scope_acp_client(task_client, async move {
                ClientShellStart::new()
                    .execute(&json!({"command": "sleep", "args": ["999"]}))
                    .await
            }));

            // Let the spawned task actually reserve its slot and reach
            // the hang inside `create_terminal` before cancelling it —
            // otherwise this loop would abort a task that never got far
            // enough to exercise the drop path at all.
            for _ in 0..200 {
                if state
                    .acp_terminals
                    .lock()
                    .unwrap()
                    .get(TEST_SESSION_ID)
                    .map(Vec::len)
                    == Some(1)
                {
                    break;
                }
                tokio::task::yield_now().await;
            }
            assert_eq!(
                state
                    .acp_terminals
                    .lock()
                    .unwrap()
                    .get(TEST_SESSION_ID)
                    .map(Vec::len),
                Some(1),
                "the reservation must be visible before it is cancelled, or this test does \
                 not exercise the drop path this fix is about"
            );

            // The cancellation this fix is about: the turn's future is
            // dropped mid-RPC. `abort` reproduces exactly that for a
            // spawned task.
            join.abort();
            let _ = join.await;

            assert!(
                state
                    .acp_terminals
                    .lock()
                    .unwrap()
                    .get(TEST_SESSION_ID)
                    .is_none_or(Vec::is_empty),
                "a reservation whose future is dropped before track_terminal must free its \
                 slot immediately, not leak it"
            );
        }

        // The proof that matters: after MAX_TERMINALS_PER_SESSION + 1
        // cancellations, a fresh start still succeeds. Before this fix,
        // each cancellation above would have permanently cost the
        // session a slot, and it would already be stuck at the cap.
        fake.let_create_terminal_finish();
        scope_acp_client(client, async {
            ClientShellStart::new()
                .execute(&json!({"command": "sleep", "args": ["999"]}))
                .await
                .expect(
                    "the session must still be able to start a terminal after repeated \
                     cancellations",
                )
        })
        .await;
    }

    // -----------------------------------------------------------------
    // Reservation-leak fix: cap_error accounts for in-flight reservations
    // -----------------------------------------------------------------

    /// The second half of the trap this task closes: a model refused at
    /// the cap while some of what it holds is still an in-flight
    /// reservation (no handle yet, because `create_terminal` hasn't
    /// returned) must not be shown a listing shorter than the maximum
    /// it is told it hit. `try_reserve_terminal_slot` cannot name a
    /// reservation as a handle — there isn't one yet — so `cap_error`
    /// has to account for it some other way instead of just omitting
    /// it.
    #[test]
    fn cap_error_accounts_for_in_flight_reservations() {
        let held = crate::tools::acp_client::CapHeld {
            handles: vec![
                TerminalHandle("t1".to_string()),
                TerminalHandle("t2".to_string()),
                TerminalHandle("t3".to_string()),
            ],
            reservations: 5,
        };
        let message = cap_error(&held).to_string();

        assert!(
            message.contains(&MAX_TERMINALS_PER_SESSION.to_string()),
            "still names the cap itself: {message}"
        );
        for id in ["t1", "t2", "t3"] {
            assert!(
                message.contains(id),
                "still names every real handle held: {message}"
            );
        }
        // The bug this guards against: a refusal that claims "the
        // maximum of 8" while naming only 3 handles and saying nothing
        // about the other 5 gives the model nothing to reconcile the
        // two numbers with.
        assert!(
            message.contains('5'),
            "the 5 in-flight reservations must be accounted for, not silently dropped from \
             a listing shorter than the 8 the message claims: {message}"
        );
    }

    /// End-to-end version of the same fix: eight concurrent
    /// `client_shell_start` calls that are all still mid-`create_terminal`
    /// (parked, so none of them has resolved into a real handle yet)
    /// must still be reflected in the refusal a ninth concurrent call
    /// gets — as a count, since none of the eight has a handle for the
    /// message to name.
    #[tokio::test]
    async fn the_cap_refusal_accounts_for_reservations_still_in_flight() {
        let (state, fake) = shell_test_state().await;
        fake.make_create_terminal_hang();
        let client: Arc<dyn AcpClient> = Arc::clone(&fake) as Arc<dyn AcpClient>;

        let mut holders = Vec::new();
        for _ in 0..MAX_TERMINALS_PER_SESSION {
            let task_client = Arc::clone(&client);
            holders.push(tokio::spawn(scope_acp_client(task_client, async move {
                ClientShellStart::new()
                    .execute(&json!({"command": "sleep", "args": ["999"]}))
                    .await
            })));
        }

        for _ in 0..2000 {
            if state
                .acp_terminals
                .lock()
                .unwrap()
                .get(TEST_SESSION_ID)
                .map(Vec::len)
                == Some(MAX_TERMINALS_PER_SESSION)
            {
                break;
            }
            tokio::task::yield_now().await;
        }
        assert_eq!(
            state
                .acp_terminals
                .lock()
                .unwrap()
                .get(TEST_SESSION_ID)
                .map(Vec::len),
            Some(MAX_TERMINALS_PER_SESSION),
            "all eight must be reserved (none resolved — create_terminal is parked) before \
             the refusal below is meaningful"
        );

        let refusal = scope_acp_client(Arc::clone(&client), async {
            ClientShellStart::new()
                .execute(&json!({"command": "one", "args": ["too", "many"]}))
                .await
                .unwrap_err()
                .to_string()
        })
        .await;

        assert!(
            refusal.contains(&MAX_TERMINALS_PER_SESSION.to_string()),
            "the refusal must account for every in-flight reservation, not present a \
             listing shorter than the {MAX_TERMINALS_PER_SESSION} it claims: {refusal}"
        );

        for h in holders {
            h.abort();
            let _ = h.await;
        }
    }
}
