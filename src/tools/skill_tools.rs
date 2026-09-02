//! `skill`: load a written procedure — planning, TDD, debugging, code
//! review, finishing a branch — from the checkout on the editor's
//! machine that `crate::skills` resolves and indexes.
//!
//! Two things make this different from the other `client_*` tools:
//!
//! 1. **There is no ACP call to list a directory.** `create_terminal`
//!    is the only way to ask the editor's machine anything about its
//!    filesystem shape, so resolving and indexing the skills directory
//!    always goes through a shell script (`crate::skills::RESOLVE_AND_INDEX_SH`)
//!    run over `client_exec::run_client_command`, never `fs/read_text_file`.
//! 2. **Reading one skill's body prefers `fs/read_text_file` and falls
//!    back to the terminal.** An editor may scope `fs/read_text_file`
//!    to the open project, and the skills checkout is deliberately
//!    outside it — so the fallback (`cat` over the terminal) is the
//!    path expected to run in practice, not defensive padding.
//!
//! The resolved index is cached per session (see [`SkillTool::cache`]'s
//! doc) so a session's second and later `skill()` calls don't re-run
//! the resolver script.

use crate::provider::ToolSpec;
use crate::skills::{RESOLVE_AND_INDEX_SH, SkillEntry, SkillIndex, parse_index};
use crate::tools::acp_client::{AcpClient, ExitStatus, current_acp_client};
use crate::tools::client_exec::run_client_command;
use crate::tools::client_tools::format_exit_status;
use crate::tools::{Tool, ToolKind};
use anyhow::Result;
use async_trait::async_trait;
use serde_json::json;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// The editor is not reachable: no ACP client is scoped to this call.
/// Worded the same as `client_tools::no_editor_error` on purpose — same
/// substring, same situation — but kept as its own copy rather than
/// made `pub(crate)` there, since this is the only other module that
/// needs it.
fn no_editor_error() -> anyhow::Error {
    anyhow::anyhow!("no editor is connected to this session; this tool only works over ACP")
}

/// How long to wait for the client's shell to resolve and index the
/// skills directory, or to `cat` one skill's body when the
/// `fs/read_text_file` fallback is needed. Both run on the editor's
/// machine over a round trip this crate does not control the latency
/// of, so this is generous relative to what either operation actually
/// costs (a directory walk over a few files, or reading one of them).
const CLIENT_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30);

/// Crude cap on how many sessions' resolved indexes stay cached at
/// once. One entry costs a directory string plus a handful of skill
/// descriptions — a few hundred bytes — and losing one on eviction
/// costs only the next `skill()` call in that session a single
/// resolver round trip, so clearing the whole map past this size is
/// enough; nothing here needs LRU precision.
const MAX_CACHED_SESSIONS: usize = 128;

/// Load a skill: a written procedure for a kind of work — planning,
/// TDD, debugging, code review, finishing a branch — from a checkout
/// that lives on the editor's machine, not this agent's.
pub struct SkillTool {
    spec: ToolSpec,
    /// The resolved index, cached per session id.
    ///
    /// `SkillTool` is registered once into the `ToolSet` shared by
    /// every connection through `ServeState` (`src/main.rs`), and
    /// `/acp` accepts many concurrent editor connections against that
    /// one state. A single cache slot on the tool — what an earlier
    /// version of this code did, reasoning that "a session belongs to
    /// exactly one connection" made it safe — was wrong: that sentence
    /// says nothing about whether *this tool* is scoped per session,
    /// and it is not. Editor A calling `skill()` would resolve and
    /// cache machine A's directory; editor B, a different connection
    /// against the same shared `SkillTool`, would then be served A's
    /// listing and A's absolute paths — including A's home directory
    /// name — without the resolver ever running on B's machine. So
    /// this is a map keyed by session id, not one slot.
    ///
    /// The key comes from `TurnContext::session_id`
    /// (`src/serve/mod.rs`), reached via `current_turn_context()` in
    /// `execute` below. A turn with no session to key on — a
    /// subagent's nested tool call, where `TurnContext::session_id` is
    /// `None` because `TurnPersistence` is `None` for a subagent — is
    /// never cached: `resolve_index` always re-resolves and never
    /// writes to the map when its key is `None`. That is the safe
    /// default for an identity this code cannot establish: caching
    /// under some shared placeholder key would reopen the same
    /// cross-editor collision this map exists to close, if two
    /// different sessions' subagents both hit it.
    cache: Mutex<HashMap<String, Arc<SkillIndex>>>,
}

impl SkillTool {
    pub fn new() -> Self {
        Self {
            spec: build_spec(),
            cache: Mutex::new(HashMap::new()),
        }
    }

    /// Force the next `skill()` call *in `session_key`'s session* to
    /// re-resolve and re-index the directory instead of reusing the
    /// cached copy. Other sessions' cache entries are untouched.
    ///
    /// This task has no caller for it: `skill` never changes what's on
    /// disk. It exists for `skill_install`/`skill_update`/
    /// `skill_uninstall` (a later task), which do — without this, a
    /// skill installed mid-session would stay invisible to `skill()`
    /// until the process restarted.
    ///
    /// `#[allow(dead_code)]`: this crate builds as a binary, so an
    /// unused `pub` method is still dead code to clippy's default
    /// (non-`--all-targets`) profile — nothing outside `#[cfg(test)]`
    /// calls this until `skill_install`/`skill_update`/
    /// `skill_uninstall` (a later task) do. It is exercised by
    /// `invalidating_the_cache_forces_a_re_resolve` in this file's own
    /// `mod tests` in the meantime.
    #[allow(dead_code)]
    pub fn invalidate_cache(&self, session_key: &str) {
        self.cache.lock().unwrap().remove(session_key);
    }

    /// Whether a finished client command's exit status counts as a
    /// failure worth reporting as one, rather than the command's
    /// output being trusted.
    ///
    /// Only a *known* bad outcome — an explicit non-zero exit code, or
    /// termination by signal — counts. `(exit_code: None, signal:
    /// None)` is treated as success rather than failure: it is the
    /// same "nothing to go on" state `client_tools::format_exit_status`
    /// already renders as `[exit status unknown]` rather than as an
    /// error elsewhere in this crate, and failing a call outright on
    /// an editor that simply didn't report a status would be a worse
    /// failure mode than trusting output that, in every case actually
    /// seen, was fine.
    fn command_failed(status: &ExitStatus) -> bool {
        status.signal.is_some() || matches!(status.exit_code, Some(code) if code != 0)
    }

    /// Resolve and index the skills directory for `session_key`'s
    /// session, or return that session's cached result from an earlier
    /// call. `session_key` of `None` (no session reachable — see
    /// `cache`'s doc) always re-resolves and is never written back.
    async fn resolve_index(
        &self,
        client: &Arc<dyn AcpClient>,
        session_key: Option<&str>,
    ) -> Result<Arc<SkillIndex>> {
        if let Some(key) = session_key
            && let Some(index) = self.cache.lock().unwrap().get(key).cloned()
        {
            return Ok(index);
        }

        let run = run_client_command(
            client,
            "sh",
            &["-c".to_string(), RESOLVE_AND_INDEX_SH.to_string()],
            None,
            CLIENT_TIMEOUT,
        )
        .await?;
        if run.timed_out_handle.is_some() {
            anyhow::bail!(
                "timed out resolving the skills directory on the editor's machine \
                 after {}s",
                CLIENT_TIMEOUT.as_secs()
            );
        }
        let status = run
            .status
            .expect("run_client_command always sets `status` when it does not time out");
        if Self::command_failed(&status) {
            anyhow::bail!(
                "resolving the skills directory failed on the editor's machine: {}",
                format_exit_status(&status).trim()
            );
        }
        let index = Arc::new(parse_index(&run.output.output)?);

        if let Some(key) = session_key {
            let mut cache = self.cache.lock().unwrap();
            if !cache.contains_key(key) && cache.len() >= MAX_CACHED_SESSIONS {
                cache.clear();
            }
            cache.insert(key.to_string(), Arc::clone(&index));
        }
        Ok(index)
    }

    /// Read one skill's body: `fs/read_text_file` first, falling back
    /// to `cat` over the terminal on *any* failure — not just ones
    /// that look like a scoping refusal. ACP gives no reliable way to
    /// tell "the editor refused because the path is out of project
    /// scope" apart from "the file doesn't exist" or a transient RPC
    /// error at this layer, and the cost of trying anyway is one
    /// wasted `cat` round trip. What makes trying unconditionally safe
    /// is the exit-status check below: `cat` failing on a path that is
    /// genuinely absent must be reported as a failure, not returned as
    /// an empty success.
    async fn load_body(client: &Arc<dyn AcpClient>, entry: &SkillEntry) -> Result<String> {
        if let Ok(body) = client.read_text_file(&entry.path, None, None).await {
            return Ok(body);
        }
        let run = run_client_command(
            client,
            "cat",
            std::slice::from_ref(&entry.path),
            None,
            CLIENT_TIMEOUT,
        )
        .await?;
        if run.timed_out_handle.is_some() {
            anyhow::bail!(
                "timed out reading {} on the editor's machine after {}s",
                entry.path,
                CLIENT_TIMEOUT.as_secs()
            );
        }
        let status = run
            .status
            .expect("run_client_command always sets `status` when it does not time out");
        if Self::command_failed(&status) {
            anyhow::bail!(
                "reading {} failed on the editor's machine: {}",
                entry.path,
                format_exit_status(&status).trim()
            );
        }
        Ok(run.output.output)
    }
}

impl Default for SkillTool {
    fn default() -> Self {
        Self::new()
    }
}

/// The directory a skill's `SKILL.md` lives in — everything up to the
/// last `/`. Skills reference siblings by relative path
/// (`./implementer-prompt.md`, `references/codex-tools.md`,
/// `scripts/task-brief`), so a response with a body but no directory
/// header leaves every one of those dead.
///
/// String splitting rather than `std::path::Path`: every path here
/// comes from a POSIX shell running on the client
/// (`RESOLVE_AND_INDEX_SH`), so it is always `/`-separated regardless
/// of what platform this agent process itself is compiled for.
fn skill_dir(path: &str) -> &str {
    path.rsplit_once('/').map_or(path, |(dir, _)| dir)
}

/// Render the index for a bare `skill()` call: every skill's name and
/// description, so the model can pick one without a second round trip.
fn format_list(index: &SkillIndex) -> String {
    if index.skills.is_empty() {
        return format!("No skills found in {}.", index.dir);
    }
    let mut out = format!("Skills available in {}:\n", index.dir);
    for skill in &index.skills {
        out.push_str(&format!("- {}: {}\n", skill.name, skill.description));
    }
    out
}

fn build_spec() -> ToolSpec {
    ToolSpec {
        name: "skill".into(),
        description: "Load a skill: a written procedure for a kind of work — planning, TDD, \
            debugging, code review, finishing a branch. Call with no arguments to list \
            what is available; call with a name to load one.\n\n\
            Before any engineering or creative work — writing a plan, changing code, \
            debugging, reviewing — check this list first and follow a skill if one \
            applies. Skills reference sibling files by relative path; the response \
            names the skill's directory so those can be read."
            .into(),
        input_schema: json!({
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "The skill to load. Omit to list what is available."
                }
            }
        }),
    }
}

#[async_trait]
impl Tool for SkillTool {
    fn kind(&self) -> ToolKind {
        ToolKind::Read
    }

    fn spec(&self) -> &ToolSpec {
        &self.spec
    }

    async fn execute(&self, input: &serde_json::Value) -> Result<String> {
        let client = current_acp_client().ok_or_else(no_editor_error)?;
        // See `cache`'s doc: `None` here (no turn context reachable, or
        // a subagent's nested call, which has a turn context but no
        // session behind it) means `resolve_index` re-resolves every
        // time rather than risking a shared cache slot.
        let session_key =
            crate::serve::current_turn_context().and_then(|ctx| ctx.session_id.clone());
        let index = self.resolve_index(&client, session_key.as_deref()).await?;

        match input.get("name").and_then(|v| v.as_str()) {
            None | Some("") => Ok(format_list(&index)),
            Some(name) => {
                let Some(entry) = index.skills.iter().find(|s| s.name == name) else {
                    let known: Vec<&str> = index.skills.iter().map(|s| s.name.as_str()).collect();
                    anyhow::bail!("no skill named '{name}'. Available: {}", known.join(", "));
                };
                let body = Self::load_body(&client, entry).await?;
                Ok(format!(
                    "Skill directory: {}\n\n{body}",
                    skill_dir(&entry.path)
                ))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::acp_client::scope_acp_client;
    use crate::tools::acp_client::tests::FakeClient;

    /// One skill (`brainstorming`), in the bundled-source layout
    /// (`<dir>/<source>/skills/<name>/SKILL.md`) `RESOLVE_AND_INDEX_SH`
    /// also indexes.
    const INDEX_STDOUT: &str = "SKILLS_DIR\t/home/user/.local/share/sapphire-agent/skills\n\
        SKILL\t/home/user/.local/share/sapphire-agent/skills/bundle/skills/brainstorming/SKILL.md\n\
        FM\tname: brainstorming\n\
        FM\tdescription: You MUST use this before any creative work\n";

    fn fake_client_returning(stdout: &str) -> Arc<FakeClient> {
        let client = Arc::new(FakeClient::default());
        client.queue_terminal_stdout(stdout);
        client
    }

    /// A client whose `fs/read_text_file` always fails — as an editor
    /// that scopes it away from the skills checkout would — with the
    /// index resolver and the `cat` fallback each queued their own
    /// terminal stdout, since they are two separate `create_terminal`
    /// round trips.
    fn fake_client_where_fs_read_fails(index_stdout: &str, body: &str) -> Arc<FakeClient> {
        let client = Arc::new(FakeClient::default());
        client.queue_terminal_stdout(index_stdout);
        client.queue_terminal_stdout(body);
        *client.read_answer.lock().unwrap() =
            Some(Err("fs/read_text_file: out of scope".to_string()));
        client
    }

    async fn load_skill_with(client: Arc<FakeClient>, name: &str) -> Result<String> {
        let tool = SkillTool::new();
        scope_acp_client(client, async { tool.execute(&json!({"name": name})).await }).await
    }

    async fn load_skill(name: &str) -> Result<String> {
        load_skill_with(fake_client_returning(INDEX_STDOUT), name).await
    }

    /// Run a bare `skill()` call (list, or the index-resolution error)
    /// against `client`.
    async fn index_with(client: Arc<FakeClient>) -> Result<String> {
        let tool = SkillTool::new();
        scope_acp_client(client, async { tool.execute(&json!({})).await }).await
    }

    /// Two calls in the *same* session, one resolver invocation: the
    /// directory does not move during a session, and re-running a
    /// shell script per skill load would cost a round trip each time.
    /// Exercised directly against `resolve_index` (rather than through
    /// `execute`, which has no `TurnContext` to read a session id from
    /// in a bare unit test) — see `two_sessions_get_two_different_resolutions`
    /// for the regression test that pins the fix for Finding 1 (the
    /// cache used to have no session key at all).
    #[tokio::test]
    async fn the_index_is_resolved_once_and_reused_within_one_session() {
        let client = fake_client_returning(INDEX_STDOUT);
        let dyn_client: Arc<dyn AcpClient> = client.clone();
        let tool = SkillTool::new();

        tool.resolve_index(&dyn_client, Some("session-a"))
            .await
            .unwrap();
        tool.resolve_index(&dyn_client, Some("session-a"))
            .await
            .unwrap();

        assert_eq!(client.terminal_count(), 1);
    }

    /// The regression test for Finding 1: `SkillTool` is registered
    /// once into the `ToolSet` shared by every `/acp` connection, so
    /// two different editors — two different sessions — must each get
    /// their own resolver run and their own directory, never one
    /// another's.
    #[tokio::test]
    async fn two_sessions_get_two_different_resolutions() {
        let client = Arc::new(FakeClient::default());
        client
            .queue_terminal_stdout("SKILLS_DIR\t/home/alice/.local/share/sapphire-agent/skills\n");
        client.queue_terminal_stdout("SKILLS_DIR\t/home/bob/.local/share/sapphire-agent/skills\n");
        let dyn_client: Arc<dyn AcpClient> = client.clone();
        let tool = SkillTool::new();

        let alice = tool
            .resolve_index(&dyn_client, Some("session-alice"))
            .await
            .unwrap();
        let bob = tool
            .resolve_index(&dyn_client, Some("session-bob"))
            .await
            .unwrap();

        assert_eq!(alice.dir, "/home/alice/.local/share/sapphire-agent/skills");
        assert_eq!(bob.dir, "/home/bob/.local/share/sapphire-agent/skills");
        assert_ne!(alice.dir, bob.dir);
        assert_eq!(
            client.terminal_count(),
            2,
            "each session must trigger its own resolver run"
        );

        // And each session still gets its own cache: re-asking for
        // alice's session must not cost a third resolver run.
        let alice_again = tool
            .resolve_index(&dyn_client, Some("session-alice"))
            .await
            .unwrap();
        assert_eq!(alice_again.dir, alice.dir);
        assert_eq!(client.terminal_count(), 2);
    }

    /// A session id that never resolves to `Some` — the bare `execute`
    /// path outside any `TurnContext`, standing in for a subagent's
    /// nested call, where `TurnContext::session_id` is `None` — must
    /// never be cached: caching it under some shared placeholder key
    /// would reopen the same cross-session collision Finding 1 was
    /// about.
    #[tokio::test]
    async fn no_session_key_is_never_cached() {
        let client = fake_client_returning(INDEX_STDOUT);
        client.queue_terminal_stdout(INDEX_STDOUT);
        let dyn_client: Arc<dyn AcpClient> = client.clone();
        let tool = SkillTool::new();

        tool.resolve_index(&dyn_client, None).await.unwrap();
        tool.resolve_index(&dyn_client, None).await.unwrap();

        assert_eq!(client.terminal_count(), 2);
    }

    /// Skills reference siblings by relative path — `./implementer-prompt.md`,
    /// `scripts/task-brief`. Without this header every one of those is dead.
    #[tokio::test]
    async fn loading_a_skill_prefixes_its_absolute_directory() {
        let out = load_skill("brainstorming").await.unwrap();
        assert!(
            out.contains("/skills/bundle/skills/brainstorming"),
            "no directory header in: {out}"
        );
    }

    /// An editor may scope `fs/read_text_file` to the open project, and
    /// the skills checkout is deliberately outside it. The fallback is
    /// the path we expect to take, not a belt-and-braces extra.
    #[tokio::test]
    async fn a_refused_fs_read_falls_back_to_the_terminal() {
        let client = fake_client_where_fs_read_fails(INDEX_STDOUT, "body from cat");
        let out = load_skill_with(client, "brainstorming").await.unwrap();
        assert!(out.contains("body from cat"), "got: {out}");
    }

    #[tokio::test]
    async fn an_unknown_name_lists_what_exists() {
        let out = load_skill("nope").await.unwrap_err().to_string();
        assert!(out.contains("brainstorming"), "got: {out}");
    }

    #[tokio::test]
    async fn no_directory_is_reported_with_the_fix() {
        let err = index_with(fake_client_returning("NO_SKILLS_DIR\n"))
            .await
            .unwrap_err();
        assert!(err.to_string().contains("skill_install"));
    }

    /// A bare call with no editor's skill found still lists an empty
    /// result rather than panicking or erroring, when the directory
    /// resolves but has nothing in it.
    #[tokio::test]
    async fn an_empty_directory_lists_as_empty_not_an_error() {
        let out = index_with(fake_client_returning(
            "SKILLS_DIR\t/home/user/.local/share/sapphire-agent/skills\n",
        ))
        .await
        .unwrap();
        assert!(out.contains("No skills found"), "got: {out}");
    }

    #[tokio::test]
    async fn no_editor_refuses_rather_than_hanging() {
        let err = SkillTool::new()
            .execute(&json!({}))
            .await
            .unwrap_err()
            .to_string();
        assert!(err.contains("no editor"), "got: {err}");
    }

    #[test]
    fn the_kind_is_read() {
        assert_eq!(SkillTool::new().kind(), ToolKind::Read);
    }

    /// `invalidate_cache` is Task 4's hook for making a freshly
    /// installed skill visible without a process restart — pinned here
    /// so a later task's use of it is against a proven contract: after
    /// invalidation, the next call for *that session* re-resolves
    /// rather than reusing the stale index.
    #[tokio::test]
    async fn invalidating_the_cache_forces_a_re_resolve() {
        let client = fake_client_returning(INDEX_STDOUT);
        client.queue_terminal_stdout(INDEX_STDOUT);
        let dyn_client: Arc<dyn AcpClient> = client.clone();
        let tool = SkillTool::new();

        tool.resolve_index(&dyn_client, Some("session-a"))
            .await
            .unwrap();
        tool.invalidate_cache("session-a");
        tool.resolve_index(&dyn_client, Some("session-a"))
            .await
            .unwrap();

        assert_eq!(client.terminal_count(), 2);
    }

    /// `invalidate_cache` only clears the session named — a sibling
    /// session's cached entry (and resolver-call count) is untouched.
    #[tokio::test]
    async fn invalidating_one_session_leaves_another_cached() {
        let client = Arc::new(FakeClient::default());
        client.queue_terminal_stdout(INDEX_STDOUT);
        client.queue_terminal_stdout(INDEX_STDOUT);
        let dyn_client: Arc<dyn AcpClient> = client.clone();
        let tool = SkillTool::new();

        tool.resolve_index(&dyn_client, Some("session-a"))
            .await
            .unwrap();
        tool.resolve_index(&dyn_client, Some("session-b"))
            .await
            .unwrap();
        tool.invalidate_cache("session-a");
        // session-b's cache entry must still be warm.
        tool.resolve_index(&dyn_client, Some("session-b"))
            .await
            .unwrap();

        assert_eq!(client.terminal_count(), 2, "session-b must not re-resolve");
    }

    /// Finding 2: a `cat` fallback that runs but fails (the file
    /// genuinely doesn't exist, as opposed to `fs/read_text_file`
    /// merely being out of the editor's declared scope) must be
    /// reported as an error, not returned as an empty success.
    #[tokio::test]
    async fn a_failed_terminal_fallback_is_reported_not_swallowed() {
        let client = Arc::new(FakeClient::default());
        client.queue_terminal_stdout(INDEX_STDOUT);
        client.queue_terminal_result("", Some(1));
        *client.read_answer.lock().unwrap() =
            Some(Err("fs/read_text_file: out of scope".to_string()));
        let dyn_client: Arc<dyn AcpClient> = client.clone();

        let tool = SkillTool::new();
        let err = scope_acp_client(dyn_client, async {
            tool.execute(&json!({"name": "brainstorming"})).await
        })
        .await
        .unwrap_err()
        .to_string();

        assert!(err.contains("brainstorming"), "got: {err}");
        assert!(err.contains("exit code: 1"), "got: {err}");
    }

    /// Finding 3: the index resolver script itself can fail on the
    /// editor's machine (permission error, a broken `awk`, whatever) —
    /// that must be reported as the script having failed, not
    /// misattributed to `parse_index` finding no `SKILLS_DIR` line.
    #[tokio::test]
    async fn a_failed_resolver_script_is_reported_as_a_script_failure() {
        let client = Arc::new(FakeClient::default());
        client.queue_terminal_result("some stderr leaked to stdout", Some(2));
        let dyn_client: Arc<dyn AcpClient> = client.clone();

        let tool = SkillTool::new();
        let err = tool
            .resolve_index(&dyn_client, Some("session-a"))
            .await
            .unwrap_err()
            .to_string();

        assert!(err.contains("failed"), "got: {err}");
        assert!(err.contains("exit code: 2"), "got: {err}");
    }

    /// An exit status the client simply didn't report
    /// (`exit_code: None, signal: None`, `FakeClient`'s default when
    /// nothing is queued) is treated as success, not failure — the
    /// same "nothing to go on" state `client_tools::format_exit_status`
    /// renders as `[exit status unknown]` elsewhere rather than as an
    /// error. `queue_terminal_stdout` already defaults to
    /// `exit_code: Some(0)`, so this pins the *other* non-failing case
    /// (no status reported at all) as still not an error.
    #[test]
    fn an_unreported_exit_status_does_not_count_as_failed() {
        assert!(!SkillTool::command_failed(&ExitStatus {
            exit_code: None,
            signal: None,
        }));
    }
}
