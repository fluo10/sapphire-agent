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
//! The resolved index is cached on the tool for the life of the
//! process (see [`SkillTool::cache`]'s doc for why session-keying
//! isn't available here) so a session's second and later `skill()`
//! calls don't re-run the resolver script.

use crate::skills::{RESOLVE_AND_INDEX_SH, SkillEntry, SkillIndex, parse_index};
use crate::tools::acp_client::{AcpClient, current_acp_client};
use crate::tools::client_exec::run_client_command;
use crate::tools::{Tool, ToolKind};
use crate::provider::ToolSpec;
use anyhow::Result;
use async_trait::async_trait;
use serde_json::json;
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

/// Load a skill: a written procedure for a kind of work — planning,
/// TDD, debugging, code review, finishing a branch — from a checkout
/// that lives on the editor's machine, not this agent's.
pub struct SkillTool {
    spec: ToolSpec,
    /// The resolved index, cached after the first `skill()` call in a
    /// process.
    ///
    /// The plan for this cache was to key it by session, via whatever
    /// identity `crate::serve::current_turn_context()` exposes — but
    /// `TurnContext` (`src/serve/mod.rs`) carries no session id at all
    /// (`state`, `provider`, `progress`, `visible_specs`,
    /// `timer_origin`), so there is no session key reachable from a
    /// tool's `execute`. Falling back to caching on the tool itself
    /// (as the brief anticipates) is correct here, not just expedient:
    /// a second editor sharing this cache would only be wrong if two
    /// editors on different machines shared one agent *session*, and
    /// they cannot — a session belongs to exactly one connection, and
    /// this agent process serves one skills checkout's worth of
    /// resolution at a time regardless of how many sessions ask.
    cache: Mutex<Option<Arc<SkillIndex>>>,
}

impl SkillTool {
    pub fn new() -> Self {
        Self {
            spec: build_spec(),
            cache: Mutex::new(None),
        }
    }

    /// Force the next `skill()` call to re-resolve and re-index the
    /// directory instead of reusing the cached copy.
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
    pub fn invalidate_cache(&self) {
        *self.cache.lock().unwrap() = None;
    }

    /// Resolve and index the skills directory, or return the cached
    /// result from an earlier call in this process.
    async fn resolve_index(&self, client: &Arc<dyn AcpClient>) -> Result<Arc<SkillIndex>> {
        if let Some(index) = self.cache.lock().unwrap().clone() {
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
        let index = Arc::new(parse_index(&run.output.output)?);
        *self.cache.lock().unwrap() = Some(Arc::clone(&index));
        Ok(index)
    }

    /// Read one skill's body: `fs/read_text_file` first, falling back
    /// to `cat` over the terminal on failure — see the module doc for
    /// why the fallback, not `fs/read_text_file` alone, is the path
    /// expected to run in practice.
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
        let index = self.resolve_index(&client).await?;

        match input.get("name").and_then(|v| v.as_str()) {
            None | Some("") => Ok(format_list(&index)),
            Some(name) => {
                let Some(entry) = index.skills.iter().find(|s| s.name == name) else {
                    let known: Vec<&str> = index.skills.iter().map(|s| s.name.as_str()).collect();
                    anyhow::bail!(
                        "no skill named '{name}'. Available: {}",
                        known.join(", ")
                    );
                };
                let body = Self::load_body(&client, entry).await?;
                Ok(format!("Skill directory: {}\n\n{body}", skill_dir(&entry.path)))
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

    /// Two calls, one resolver invocation: the directory does not move
    /// during a session, and re-running a shell script per skill load
    /// would cost a round trip each time.
    #[tokio::test]
    async fn the_index_is_resolved_once_and_reused() {
        let client = fake_client_returning(INDEX_STDOUT);
        let tool = SkillTool::new();
        scope_acp_client(client.clone(), async {
            tool.execute(&json!({})).await.unwrap();
            tool.execute(&json!({"name": "brainstorming"}))
                .await
                .unwrap();
        })
        .await;
        assert_eq!(client.terminal_count(), 1);
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
    /// invalidation, the next call re-resolves rather than reusing the
    /// stale index.
    #[tokio::test]
    async fn invalidating_the_cache_forces_a_re_resolve() {
        let client = fake_client_returning(INDEX_STDOUT);
        client.queue_terminal_stdout(INDEX_STDOUT);
        let tool = SkillTool::new();
        scope_acp_client(client.clone(), async {
            tool.execute(&json!({})).await.unwrap();
            tool.invalidate_cache();
            tool.execute(&json!({})).await.unwrap();
        })
        .await;
        assert_eq!(client.terminal_count(), 2);
    }
}
