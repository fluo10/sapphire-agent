//! The agent's way of reaching the editor's machine.
//!
//! A tool's `execute` takes only its JSON input — it has no idea which
//! session called it, and the ACP connection lives per-connection in
//! `serve::acp`. Threading a session through `Tool` would touch every
//! tool for the benefit of six, so the handle is carried in a
//! `tokio::task_local` scoped around tool execution instead.
//!
//! `src/timer.rs` solves the same problem the same way: the turn loops
//! wrap `tools.execute(...)` in `scope_timer_origin` so the timer tool
//! can read where its call came from.
//!
//! This is a trait rather than the SDK's connection type so the tools
//! can be driven by a fake in tests, and so everything that knows about
//! ACP's wire types stays inside `serve::acp`.

use std::collections::HashMap;
use std::sync::Arc;

/// An opaque handle to a command running on the client's machine.
///
/// Opaque on purpose: the value is the client's, and nothing here may
/// parse or construct one except from what the client returned. The
/// inner `String` is public only so tests (the fake client) can build
/// one directly — a real handle is only ever the client's own value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TerminalHandle(pub String);

impl std::fmt::Display for TerminalHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ExitStatus {
    pub exit_code: Option<u32>,
    pub signal: Option<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TerminalOutput {
    pub output: String,
    /// The client hit the byte limit we asked for and cut the output.
    pub truncated: bool,
    /// `None` while the command is still running.
    pub exit_status: Option<ExitStatus>,
}

/// The session-keyed store of terminals a session has started and not
/// yet released.
///
/// `Arc`-wrapped so it can be shared between `ServeState.acp_terminals`
/// (`src/serve/mod.rs`), the authoritative copy that outlives any one
/// connection, and whichever [`AcpClient`] is tracking against it —
/// `AcpClientHandle` (`src/serve/acp.rs`) for a real connection,
/// `FakeClient` below for tests. `client_tools::ClientShellStart`'s cap
/// check has to see every handle a session holds, including ones left
/// behind by a connection other than the one asking, without this
/// module depending on `ServeState` itself — hence the type living
/// here rather than in `serve`.
pub(crate) type TerminalRegistry = Arc<tokio::sync::Mutex<HashMap<String, Vec<TerminalHandle>>>>;

/// What the editor can be asked to do on its own machine.
///
/// Every method maps 1:1 onto one ACP `agent → client` request. The
/// full set is exactly this — ACP has no directory listing, delete,
/// stat or rename, which is why there are no client-side tools for
/// those.
#[async_trait::async_trait]
pub trait AcpClient: Send + Sync {
    async fn read_text_file(
        &self,
        path: &str,
        line: Option<u32>,
        limit: Option<u32>,
    ) -> anyhow::Result<String>;

    async fn write_text_file(&self, path: &str, content: &str) -> anyhow::Result<()>;

    async fn create_terminal(
        &self,
        command: &str,
        args: &[String],
        cwd: Option<&str>,
        output_byte_limit: Option<u64>,
    ) -> anyhow::Result<TerminalHandle>;

    async fn terminal_output(&self, terminal: &TerminalHandle) -> anyhow::Result<TerminalOutput>;

    async fn wait_for_terminal_exit(&self, terminal: &TerminalHandle)
    -> anyhow::Result<ExitStatus>;

    /// Ends the command but keeps the handle usable, so the output can
    /// still be collected afterwards.
    async fn kill_terminal(&self, terminal: &TerminalHandle) -> anyhow::Result<()>;

    /// Frees the handle — **and kills the command if it is still
    /// running.** The ACP schema says so explicitly: `terminal/release`
    /// kills the command just like `terminal/kill` does, it just also
    /// invalidates the `TerminalId` afterwards. Nothing may call this
    /// just because a connection went away — releasing on a dropped
    /// socket would kill a user's build over a network blip.
    async fn release_terminal(&self, terminal: &TerminalHandle) -> anyhow::Result<()>;

    /// Terminal handles this client's session currently has open —
    /// started but not yet released — in the order they were tracked.
    ///
    /// `client_tools::ClientShellStart` reads this to decide whether
    /// the session is already at the cap *before* asking the client to
    /// create anything, which is what keeps a refused call from
    /// reaching the wire at all.
    async fn tracked_terminals(&self) -> Vec<TerminalHandle>;

    /// Record that `handle` now belongs to this client's session, so
    /// it counts against the cap and shows up in `tracked_terminals`
    /// until [`untrack_terminal`](AcpClient::untrack_terminal) drops
    /// it.
    async fn track_terminal(&self, handle: TerminalHandle);

    /// Drop `handle` from this client's session tracking.
    ///
    /// Not a release: called after a real `release_terminal`/
    /// `kill_terminal`+`release_terminal` succeeds, and also when the
    /// client reports the handle unknown — in that second case there
    /// is nothing left to release, only bookkeeping to correct so a
    /// handle the client has already forgotten stops counting against
    /// the cap.
    async fn untrack_terminal(&self, handle: &TerminalHandle);
}

tokio::task_local! {
    static ACP_CLIENT_TL: Arc<dyn AcpClient>;
}

/// Run `fut` with a client reachable from `current_acp_client`.
pub fn scope_acp_client<F: std::future::Future>(
    client: Arc<dyn AcpClient>,
    fut: F,
) -> impl std::future::Future<Output = F::Output> {
    ACP_CLIENT_TL.scope(client, fut)
}

/// The client for the turn currently executing, if it has one.
///
/// `None` on every non-ACP transport — `/rpc`, Matrix, Discord, voice —
/// which is what makes the client tools refuse there rather than
/// reaching for a connection that does not exist.
pub fn current_acp_client() -> Option<Arc<dyn AcpClient>> {
    ACP_CLIENT_TL.try_with(Arc::clone).ok()
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    /// One recorded call to `read_text_file`: `(path, line, limit)`.
    pub(crate) type ReadCall = (String, Option<u32>, Option<u32>);

    /// One recorded call to `create_terminal`: `(command, args, cwd,
    /// output_byte_limit)`.
    pub(crate) type CreateCall = (String, Vec<String>, Option<String>, Option<u64>);

    /// A stand-in for the editor. Records what it was asked and answers
    /// from a script, so the tools can be driven without a socket.
    #[derive(Default)]
    pub(crate) struct FakeClient {
        pub reads: Mutex<Vec<ReadCall>>,
        pub writes: Mutex<Vec<(String, String)>>,
        pub read_answer: Mutex<Option<Result<String, String>>>,
        pub creates: Mutex<Vec<CreateCall>>,
        pub released: Mutex<Vec<TerminalHandle>>,
        pub killed: Mutex<Vec<TerminalHandle>>,
        /// Set by [`FakeClient::make_exit_never_return`]. When true,
        /// `wait_for_terminal_exit` blocks forever instead of resolving
        /// — the only way to make `ClientShell`'s
        /// `tokio::time::timeout` race actually win on the timeout arm
        /// without a test sleeping out a real wall-clock timeout.
        exit_never_returns: Mutex<bool>,
        /// Set by [`FakeClient::hand_out_distinct_handles`]. When true,
        /// `create_terminal` hands back `t1`, `t2`, … in order instead
        /// of always `t1` — needed by the cap test, which has to tell
        /// the eight handles it holds apart.
        distinct_handles: Mutex<bool>,
        next_handle: Mutex<u32>,
        /// Set by [`FakeClient::make_output_fail_with`]:
        /// `terminal_output` returns this as an error instead of a
        /// result, simulating a client that no longer recognises the
        /// handle (e.g. after its own restart).
        output_error: Mutex<Option<String>>,
        /// The session id this fake's terminal tracking is keyed
        /// under, and the registry it tracks in. Plain fields rather
        /// than `Mutex`-wrapped: both are set once, by struct-update
        /// syntax, before the value is wrapped in an `Arc` for use.
        /// `client_tools`'s `shell_test_state` points `terminals` at
        /// the same `Arc` a `ServeState` under test uses, so tracking
        /// driven purely by calling this fake is visible on
        /// `ServeState.acp_terminals` too. Every other test builds a
        /// `FakeClient` with `..Default::default()`, which leaves an
        /// empty, private registry — correct, self-contained cap
        /// bookkeeping for tests that never look at a `ServeState`.
        pub(crate) terminal_session: String,
        pub(crate) terminals: TerminalRegistry,
    }

    impl FakeClient {
        /// Make every future `wait_for_terminal_exit` call hang forever,
        /// so a caller racing it against a timeout always sees the
        /// timeout branch.
        pub(crate) fn make_exit_never_return(&self) {
            *self.exit_never_returns.lock().unwrap() = true;
        }

        /// Make `create_terminal` hand back sequential ids (`t1`, `t2`,
        /// …) instead of always `t1`.
        pub(crate) fn hand_out_distinct_handles(&self) {
            *self.distinct_handles.lock().unwrap() = true;
        }

        /// Make `terminal_output` fail with `message`, as if the client
        /// no longer recognised the handle.
        pub(crate) fn make_output_fail_with(&self, message: &str) {
            *self.output_error.lock().unwrap() = Some(message.to_string());
        }
    }

    #[async_trait::async_trait]
    impl AcpClient for FakeClient {
        async fn read_text_file(
            &self,
            path: &str,
            line: Option<u32>,
            limit: Option<u32>,
        ) -> anyhow::Result<String> {
            self.reads
                .lock()
                .unwrap()
                .push((path.to_string(), line, limit));
            match self.read_answer.lock().unwrap().take() {
                Some(Ok(s)) => Ok(s),
                Some(Err(e)) => Err(anyhow::anyhow!(e)),
                None => Ok(String::new()),
            }
        }
        async fn write_text_file(&self, path: &str, content: &str) -> anyhow::Result<()> {
            self.writes
                .lock()
                .unwrap()
                .push((path.to_string(), content.to_string()));
            Ok(())
        }
        async fn create_terminal(
            &self,
            command: &str,
            args: &[String],
            cwd: Option<&str>,
            output_byte_limit: Option<u64>,
        ) -> anyhow::Result<TerminalHandle> {
            self.creates.lock().unwrap().push((
                command.to_string(),
                args.to_vec(),
                cwd.map(|s| s.to_string()),
                output_byte_limit,
            ));
            let id = if *self.distinct_handles.lock().unwrap() {
                let mut next = self.next_handle.lock().unwrap();
                *next += 1;
                format!("t{next}")
            } else {
                "t1".to_string()
            };
            Ok(TerminalHandle(id))
        }
        async fn terminal_output(&self, _t: &TerminalHandle) -> anyhow::Result<TerminalOutput> {
            if let Some(message) = self.output_error.lock().unwrap().clone() {
                return Err(anyhow::anyhow!(message));
            }
            Ok(TerminalOutput::default())
        }
        async fn wait_for_terminal_exit(&self, _t: &TerminalHandle) -> anyhow::Result<ExitStatus> {
            if *self.exit_never_returns.lock().unwrap() {
                // Never resolves — see `make_exit_never_return`.
                std::future::pending::<()>().await;
            }
            Ok(ExitStatus::default())
        }
        async fn kill_terminal(&self, t: &TerminalHandle) -> anyhow::Result<()> {
            self.killed.lock().unwrap().push(t.clone());
            Ok(())
        }
        async fn release_terminal(&self, t: &TerminalHandle) -> anyhow::Result<()> {
            self.released.lock().unwrap().push(t.clone());
            Ok(())
        }
        async fn tracked_terminals(&self) -> Vec<TerminalHandle> {
            self.terminals
                .lock()
                .await
                .get(&self.terminal_session)
                .cloned()
                .unwrap_or_default()
        }
        async fn track_terminal(&self, handle: TerminalHandle) {
            self.terminals
                .lock()
                .await
                .entry(self.terminal_session.clone())
                .or_default()
                .push(handle);
        }
        async fn untrack_terminal(&self, handle: &TerminalHandle) {
            let mut registry = self.terminals.lock().await;
            if let Some(held) = registry.get_mut(&self.terminal_session) {
                held.retain(|h| h != handle);
                if held.is_empty() {
                    registry.remove(&self.terminal_session);
                }
            }
        }
    }

    /// Outside a scope there is no client — a channel or `/rpc` turn
    /// must not find one lying around from an earlier ACP turn.
    #[tokio::test]
    async fn there_is_no_client_outside_a_scope() {
        assert!(current_acp_client().is_none());
    }

    #[tokio::test]
    async fn a_scoped_client_is_visible_inside_and_gone_after() {
        let fake: Arc<dyn AcpClient> = Arc::new(FakeClient::default());
        scope_acp_client(Arc::clone(&fake), async {
            let seen = current_acp_client().expect("inside the scope");
            seen.write_text_file("/p/a.txt", "hi").await.unwrap();
        })
        .await;
        assert!(
            current_acp_client().is_none(),
            "the scope must not leak past its future"
        );
    }

    /// The scope has to survive being handed to a spawned task's await
    /// points, since a turn awaits the model between tool calls.
    #[tokio::test]
    async fn the_scope_survives_an_await() {
        let fake: Arc<dyn AcpClient> = Arc::new(FakeClient::default());
        scope_acp_client(fake, async {
            tokio::task::yield_now().await;
            assert!(current_acp_client().is_some(), "still scoped after a yield");
        })
        .await;
    }
}
