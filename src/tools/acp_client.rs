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

use std::sync::Arc;

/// An opaque handle to a command running on the client's machine.
///
/// Opaque on purpose: the value is the client's, and nothing here may
/// parse or construct one except from what the client returned. The
/// inner `String` is public only so tests (the fake client) can build
/// one directly — a real handle is only ever the client's own value.
// Nothing constructs one from live code yet — `FakeClient::create_terminal`
// is the only constructor today, and it is itself unreachable until
// Task 5's `ClientShell` calls `AcpClient::create_terminal`. Remove this
// allow there.
#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TerminalHandle(pub String);

impl std::fmt::Display for TerminalHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

// Not constructed by any live call site until Task 5's `ClientShell` calls
// `AcpClient::wait_for_terminal_exit`. Remove this allow there.
#[allow(dead_code)]
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ExitStatus {
    pub exit_code: Option<u32>,
    pub signal: Option<String>,
}

// Not constructed by any live call site until Task 5's `ClientShell` calls
// `AcpClient::terminal_output`. Remove this allow there.
#[allow(dead_code)]
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TerminalOutput {
    pub output: String,
    /// The client hit the byte limit we asked for and cut the output.
    pub truncated: bool,
    /// `None` while the command is still running.
    pub exit_status: Option<ExitStatus>,
}

/// What the editor can be asked to do on its own machine.
///
/// Every method maps 1:1 onto one ACP `agent → client` request. The
/// full set is exactly this — ACP has no directory listing, delete,
/// stat or rename, which is why there are no client-side tools for
/// those.
#[async_trait::async_trait]
pub trait AcpClient: Send + Sync {
    // No call site until Task 4's `ClientFileRead` calls this. Remove this
    // allow there.
    #[allow(dead_code)]
    async fn read_text_file(
        &self,
        path: &str,
        line: Option<u32>,
        limit: Option<u32>,
    ) -> anyhow::Result<String>;

    // No call site until Task 4's `ClientFileWrite` calls this. Remove this
    // allow there.
    #[allow(dead_code)]
    async fn write_text_file(&self, path: &str, content: &str) -> anyhow::Result<()>;

    // No call site until Task 5's `ClientShell` calls this. Remove this
    // allow there.
    #[allow(dead_code)]
    async fn create_terminal(
        &self,
        command: &str,
        args: &[String],
        cwd: Option<&str>,
        output_byte_limit: Option<u64>,
    ) -> anyhow::Result<TerminalHandle>;

    // No call site until Task 5's `ClientShell` calls this. Remove this
    // allow there.
    #[allow(dead_code)]
    async fn terminal_output(&self, terminal: &TerminalHandle) -> anyhow::Result<TerminalOutput>;

    // No call site until Task 5's `ClientShell` calls this. Remove this
    // allow there.
    #[allow(dead_code)]
    async fn wait_for_terminal_exit(&self, terminal: &TerminalHandle)
    -> anyhow::Result<ExitStatus>;

    /// Ends the command but keeps the handle usable, so the output can
    /// still be collected afterwards.
    // No call site until Task 6's `ClientShellKill` calls this. Remove
    // this allow there.
    #[allow(dead_code)]
    async fn kill_terminal(&self, terminal: &TerminalHandle) -> anyhow::Result<()>;

    /// Frees the handle — **and kills the command if it is still
    /// running.** The ACP schema says so explicitly: `terminal/release`
    /// kills the command just like `terminal/kill` does, it just also
    /// invalidates the `TerminalId` afterwards. Nothing may call this
    /// just because a connection went away — releasing on a dropped
    /// socket would kill a user's build over a network blip.
    // No call site until Task 5's `ClientShell` releases on the happy
    // path. Remove this allow there.
    #[allow(dead_code)]
    async fn release_terminal(&self, terminal: &TerminalHandle) -> anyhow::Result<()>;
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
// No call site outside this module's own tests until Task 4's
// `ClientFileRead`/`ClientFileWrite` call this. Remove this allow there.
#[allow(dead_code)]
pub fn current_acp_client() -> Option<Arc<dyn AcpClient>> {
    ACP_CLIENT_TL.try_with(Arc::clone).ok()
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    /// A stand-in for the editor. Records what it was asked and answers
    /// from a script, so the tools can be driven without a socket.
    #[derive(Default)]
    pub(crate) struct FakeClient {
        // Written by `read_text_file` but nothing reads it back until
        // Task 4's tests assert on `fake.reads`. Remove this allow there.
        #[allow(dead_code)]
        pub reads: Mutex<Vec<(String, Option<u32>, Option<u32>)>>,
        pub writes: Mutex<Vec<(String, String)>>,
        // Set by Task 4's tests to script an error/answer from
        // `read_text_file`; nothing reads it back today because nothing
        // calls `read_text_file` yet. Remove this allow once Task 4
        // lands.
        #[allow(dead_code)]
        pub read_answer: Mutex<Option<Result<String, String>>>,
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
            _command: &str,
            _args: &[String],
            _cwd: Option<&str>,
            _output_byte_limit: Option<u64>,
        ) -> anyhow::Result<TerminalHandle> {
            Ok(TerminalHandle("t1".to_string()))
        }
        async fn terminal_output(&self, _t: &TerminalHandle) -> anyhow::Result<TerminalOutput> {
            Ok(TerminalOutput::default())
        }
        async fn wait_for_terminal_exit(&self, _t: &TerminalHandle) -> anyhow::Result<ExitStatus> {
            Ok(ExitStatus::default())
        }
        async fn kill_terminal(&self, _t: &TerminalHandle) -> anyhow::Result<()> {
            Ok(())
        }
        async fn release_terminal(&self, _t: &TerminalHandle) -> anyhow::Result<()> {
            Ok(())
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
