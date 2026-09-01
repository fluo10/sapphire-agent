//! `subagent`: delegate a task to a specialised agent.
//!
//! A subagent runs a whole nested conversation — its own system prompt,
//! its own tool-calling loop, its own history — and hands back only its
//! final answer. Three properties make that safe rather than a loophole:
//!
//! 1. **Judged by the parent's `Origin`, through the parent's
//!    `TurnHost`.** The nested loop (`crate::serve::TurnLoop`) is handed
//!    the caller's own `progress` unchanged (see [`TurnContext`] in
//!    `src/serve/mod.rs`). If delegation ran under its own host or its
//!    own origin, "ask a subagent" would become a way to get done by
//!    proxy what the model was refused directly. (One exception, scoped
//!    narrowly: `turn_error` is *not* forwarded unchanged — see
//!    `SubagentTool::execute`'s host wrapper, `ParentHostSansTurnError`,
//!    below.)
//! 2. **The system prompt is the definition and nothing else.** See
//!    [`subagent_system_prompt`].
//! 3. **The tool list actually offered is enforced, not just built
//!    restricted.** `subagent_tool_specs` removes `subagent` from a
//!    nested turn's own list, but that alone is a hint to the model, not
//!    a bound: `ToolSet::execute` dispatches by name across every tool
//!    the shared `ToolSet` has registered, `subagent` included, so
//!    nothing here would stop a nested turn from calling `subagent` by
//!    name even though its own list never offered it. What actually
//!    closes recursion — and a definition's `tools:` restriction, and
//!    the same hallucinated-name gap on the parent's own turn — is
//!    `TurnLoop::run`'s permission gate refusing any call whose name is
//!    not in *that round's own* `tool_specs`, checked ahead of
//!    everything else including the host-access gate. See
//!    `Refusal::NotOffered` in `crate::tools::policy`.
//!
//! `Tool::execute` receives only its JSON input — no session, no host,
//! no model. What a subagent needs to run is threaded through instead
//! via a `tokio::task_local` (`crate::serve::TurnContext`,
//! `scope_turn_context`/`current_turn_context`), the same vehicle
//! `crate::tools::acp_client` uses for the ACP connection.
//!
//! **What "isolation" does not cover.** `TurnHost::tool_start`/`tool_end`
//! fire on the *parent's* host for a subagent's own tool calls too (they
//! run inside the same `scope_turn_context`/`scope_memory_namespace`
//! wrapping every call in the parent's round), so a subagent's tool
//! activity is visible in the parent's ACP session stream as
//! notifications. That is necessary — it is what makes a permission
//! prompt for a subagent's call legible as coming from *this*
//! conversation — but it means the notification channel is not part of
//! what stays isolated. Nothing from it reaches the parent's stored
//! history or the ACP session store; only the returned final answer
//! does, as this tool's own result.
//!
//! **Lock re-entrancy is not a hazard here, deliberately.**
//! `SubagentTool::execute` runs *inside* `ToolSet::execute` (it is
//! itself one of the tools that set owns), and the nested loop it
//! drives calls back into the very same `ToolSet::execute` for its own
//! tool calls — `ToolSet::execute` is therefore entered twice,
//! re-entrantly, on the same task before the outer call returns. That
//! used to be a real hazard: `ToolSet::execute` held its read guard
//! across the whole call to `Tool::execute_full`, so for `subagent` the
//! guard was held across an entire nested conversation — up to
//! `MAX_TOOL_ROUNDS` provider calls plus however long a human takes to
//! answer an `AcpProgress::approve` prompt. `tokio::sync::RwLock` is
//! task-fair: a reader blocks as soon as a writer is queued, so a
//! concurrent write (an MCP server's `tools/list_changed` refresh via
//! `ToolSet::refresh_if_needed`, or `mcp_reconnect`) queued behind that
//! held guard would then block the nested re-entrant read behind
//! *itself* — and every later call on every transport, since the
//! writer stays queued in front of them too. Nothing releases; the
//! agent stops answering until restart. `ToolSet::execute` now clones
//! the matched `Arc<dyn Tool>` under a short-lived read guard and drops
//! the guard before calling `execute_full`, so no execution — nested or
//! not — ever holds the lock. That removes the class of hazard rather
//! than this one instance of it, and as a side effect it is also what
//! stops `mcp_reconnect` from deadlocking on its own write lock while
//! its own call's read guard was still held (#201) — the same guard was
//! the cause of both.

use crate::agents::AgentDef;
use crate::provider::{ChatMessage, ToolSpec};
use crate::tools::{Tool, ToolKind};
use anyhow::Context;
use async_trait::async_trait;
use serde_json::json;
use tracing::warn;

pub(crate) const SUBAGENT_TOOL_NAME: &str = "subagent";

/// A subagent's whole system prompt.
///
/// The definition's body, plus the date — and nothing else is baked
/// into the *prompt text*: not the workspace files (`SOUL.md`,
/// `IDENTITY.md`, `USER.md`, `AGENTS.md`, `TOOLS.md`), not a MEMORY.md
/// digest, not the day's cross-session digest, not the room metadata,
/// not the configured base prompt.
///
/// Dropping those is not an oversight, it is the feature: the main
/// agent carries them deliberately — it is someone to work *with* — and
/// a code review does not need yesterday's conversation. Inheriting
/// them by default would defeat the reason this exists.
///
/// The date is the one exception, because an agent that does not know
/// today's date cannot use a tool that writes one, and that is a fact
/// rather than a personality.
///
/// This is a statement about the prompt, not about reach: an
/// unrestricted definition (`tools: None`) still inherits whichever
/// `memory_*` tools the parent can see, and — because
/// `SubagentTool::execute` reads `current_memory_namespace()` at call
/// time — those calls land in the same namespace the delegating
/// conversation is already in. A subagent is not told what is in
/// memory; it is not prevented from asking, unless its own `tools:`
/// list says so.
pub(crate) fn subagent_system_prompt(def: &AgentDef) -> String {
    let now = chrono::Local::now();
    format!(
        "{}\n\n# Current Date and Time\n\n{} ({})",
        def.prompt,
        now.format("%Y-%m-%d %H:%M:%S %z"),
        now.format("%A")
    )
}

/// The tools a subagent may use.
///
/// `None` in the definition inherits the parent's visible set; a list
/// selects from it. Either way `subagent` itself is removed from the
/// list this function returns — but that is a hint to the model about
/// what to call, not the thing that actually caps delegation depth.
/// `ToolSet::execute` dispatches by name across every tool the shared
/// `ToolSet` has registered, `subagent` included, so a nested turn could
/// still call `subagent` by name even with it missing from this list.
/// What actually enforces the cap is `TurnLoop::run`'s permission gate,
/// which refuses any call whose name is not in that round's own
/// `tool_specs` — i.e. not in what this function returned. See
/// `Refusal::NotOffered` in `crate::tools::policy`.
///
/// `ToolSpec.name` is `Cow<'static, str>`, so every comparison below
/// goes through `.as_ref()` rather than relying on a direct `Cow` vs
/// `&str` comparison.
pub(crate) fn subagent_tool_specs(def: &AgentDef, parent_visible: &[ToolSpec]) -> Vec<ToolSpec> {
    parent_visible
        .iter()
        .filter(|s| s.name.as_ref() != SUBAGENT_TOOL_NAME)
        .filter(|s| match &def.tools {
            Some(allowed) => allowed.iter().any(|a| a == s.name.as_ref()),
            None => true,
        })
        .cloned()
        .collect()
}

/// Build the tool's spec: a fixed preamble plus one line per agent, so
/// the parent model's only basis for choosing — each definition's own
/// `description` — actually reaches it.
fn build_spec(agents: &[AgentDef]) -> ToolSpec {
    let mut description = String::from(
        "Delegate a task to a specialised agent. The agent runs with its own \
         system prompt and its own conversation, and only its final answer \
         comes back — use this to keep a large investigation out of this \
         conversation.\n\nAvailable agents:\n",
    );
    for agent in agents {
        description.push_str(&format!("- {}: {}\n", agent.name, agent.description));
    }

    ToolSpec {
        name: SUBAGENT_TOOL_NAME.into(),
        description: description.into(),
        input_schema: json!({
            "type": "object",
            "properties": {
                "agent": {
                    "type": "string",
                    "description": "Which agent to delegate to — one of the names \
                        listed in this tool's own description."
                },
                "prompt": {
                    "type": "string",
                    "description": "The task to hand the agent. It sees only this \
                        and its own definition — nothing else from the current \
                        conversation."
                }
            },
            "required": ["agent", "prompt"]
        }),
    }
}

/// Forwards every `TurnHost` method to the parent's host except
/// `turn_error`, which it swallows.
///
/// A subagent's nested `TurnLoop` runs with `progress: &ctx.progress` —
/// the parent's own host, by design (see the module doc's first
/// property). But `progress.turn_error` is not part of that judgement;
/// it is how a turn reports itself terminally failed to whoever is
/// waiting on *this* request id. For `/rpc` and voice, `SseProgress`'s
/// impl sends a terminal JSON-RPC error carrying the parent's own
/// `req_id` the instant it is called — so a subagent's own provider
/// failure, left unwrapped, would fire that mid-turn while the parent
/// turn is still running, and the parent would go on to send its own
/// terminal response for the same id: two terminal frames for one
/// request, which `run_turn` (`src/serve/mod.rs`) assumes cannot
/// happen. On ACP the effect is milder (`AcpProgress::turn_error` only
/// records a message, read back solely when the *parent's* own turn
/// ends with no reply) but still wrong: a subagent's failure has no
/// business overwriting what the parent's own failure, if any, would
/// have said. Swallowing it here keeps a subagent's provider failure on
/// the one channel that is actually correct for it — surfacing as this
/// tool's own result, which the parent model reads like any other tool
/// output and can act on.
///
/// Every other method is forwarded completely unchanged: `origin()`,
/// `approve()`, `acp_client()`, `client_fs_caps()`,
/// `client_terminal_cap()`, `tool_start`/`tool_end`/`tool_allowed` all
/// still resolve to the parent's own host, because those are what keep
/// delegation inside the same permission gate — only `turn_error` is
/// special-cased.
struct ParentHostSansTurnError(std::sync::Arc<dyn crate::serve::TurnHost>);

#[async_trait]
impl crate::serve::TurnHost for ParentHostSansTurnError {
    async fn tool_start(&self, id: &str, name: &str) {
        self.0.tool_start(id, name).await;
    }

    async fn tool_end(&self, id: &str, name: &str) {
        self.0.tool_end(id, name).await;
    }

    /// Swallowed — see the type doc. The subagent's own tool result
    /// still carries the failure to the model; nothing is lost, it just
    /// does not also masquerade as *this* turn's terminal outcome.
    async fn turn_error(&self, _message: &str) {}

    fn origin(&self) -> crate::tools::policy::Origin {
        self.0.origin()
    }

    fn acp_client(&self) -> Option<std::sync::Arc<dyn crate::tools::acp_client::AcpClient>> {
        self.0.acp_client()
    }

    fn client_fs_caps(&self) -> (bool, bool) {
        self.0.client_fs_caps()
    }

    fn client_terminal_cap(&self) -> bool {
        self.0.client_terminal_cap()
    }

    async fn tool_allowed(&self, id: &str) {
        self.0.tool_allowed(id).await;
    }

    async fn approve(
        &self,
        call: &crate::provider::ToolCall,
        kind: ToolKind,
    ) -> crate::tools::policy::Approval {
        self.0.approve(call, kind).await
    }
}

/// Delegates a task to a specialised agent and returns only its final
/// answer. See the module docs for the three properties this exists to
/// establish.
pub struct SubagentTool {
    agents: Vec<AgentDef>,
    spec: ToolSpec,
    /// `(agent name, tool name)` pairs already warned about by
    /// [`Self::newly_unknown_tools`], so a typo in one definition's
    /// `tools:` list is logged once rather than once per delegation.
    warned_unknown_tools: std::sync::Mutex<std::collections::HashSet<(String, String)>>,
}

impl SubagentTool {
    pub fn new(agents: Vec<AgentDef>) -> Self {
        let spec = build_spec(&agents);
        Self {
            agents,
            spec,
            warned_unknown_tools: std::sync::Mutex::new(std::collections::HashSet::new()),
        }
    }

    /// Which of `def.tools`' names resolve to nothing the parent can
    /// currently see (`parent_visible`) — i.e. which of them
    /// `subagent_tool_specs` would silently drop — that have not been
    /// reported for this `def` before. Records them as reported and
    /// returns them, rather than logging directly, so the dedup logic
    /// is testable without capturing tracing output; the caller logs.
    ///
    /// `subagent` itself is excluded: `subagent_tool_specs` always
    /// removes it regardless of what a definition asks for, so naming
    /// it is not a typo, it is a no-op the model cannot exploit — see
    /// this module's doc.
    ///
    /// Nothing here removes the name or disables the definition. A
    /// typo in a `tools:` list is a mistake in one line, not a reason
    /// to silently take the rest of that agent's tools with it — still
    /// less the whole definition it belongs to.
    fn newly_unknown_tools(&self, def: &AgentDef, parent_visible: &[ToolSpec]) -> Vec<String> {
        let Some(allowed) = &def.tools else {
            return Vec::new();
        };
        let known: std::collections::HashSet<&str> =
            parent_visible.iter().map(|s| s.name.as_ref()).collect();
        let mut warned = self.warned_unknown_tools.lock().unwrap();
        let mut newly = Vec::new();
        for name in allowed {
            if name.as_str() == SUBAGENT_TOOL_NAME {
                continue;
            }
            if known.contains(name.as_str()) {
                continue;
            }
            if warned.insert((def.name.clone(), name.clone())) {
                newly.push(name.clone());
            }
        }
        newly
    }
}

#[async_trait]
impl Tool for SubagentTool {
    fn spec(&self) -> &ToolSpec {
        &self.spec
    }

    fn kind(&self) -> ToolKind {
        ToolKind::Other
    }

    async fn execute(&self, input: &serde_json::Value) -> anyhow::Result<String> {
        let name = input["agent"].as_str().context("missing 'agent'")?;
        let prompt = input["prompt"].as_str().context("missing 'prompt'")?;

        let Some(def) = self.agents.iter().find(|a| a.name == name) else {
            // Recoverable: the parent picked a name that does not
            // exist, and can pick again if it is told what does.
            let known: Vec<&str> = self.agents.iter().map(|a| a.name.as_str()).collect();
            anyhow::bail!("no agent named '{name}'. Available: {}", known.join(", "));
        };

        // Only a live turn has a model, a permission host and a visible
        // tool set to lend. Nothing else can delegate.
        let ctx = crate::serve::current_turn_context().context("no turn to delegate from")?;

        // A typo, a renamed tool, or a name that was never registered
        // yields a subagent silently missing it — no warning at load,
        // no error at call time, it is simply never offered. Warn
        // rather than fail: one bad name in the list must not take the
        // rest of it, or the definition, down.
        for name in self.newly_unknown_tools(def, &ctx.visible_specs) {
            warn!(
                "agent '{}': tools list names '{name}', which this delegation \
                 cannot currently see — it is not a registered tool, or not \
                 one visible on this transport right now. The definition \
                 still runs; that name is simply never offered to it.",
                def.name
            );
        }

        let system = subagent_system_prompt(def);
        let specs = subagent_tool_specs(def, &ctx.visible_specs);
        let mut history = vec![ChatMessage::user(prompt)];

        // The subagent's own memory-tool calls (if it has any) write
        // under the same namespace the delegating conversation is in —
        // a fact about where this deployment's memory lives, not a
        // personality trait, so it travels through like the date does
        // rather than being stripped like the workspace files are.
        let namespace = crate::tools::workspace_tools::current_memory_namespace();

        // The parent's host, deliberately: a permission request from a
        // subagent must reach the same person, judged by the same
        // origin. A different host here would make delegation a way
        // around the gate. `turn_error` is the one method NOT forwarded
        // unchanged — see `ParentHostSansTurnError`'s doc for why a
        // subagent's own provider failure must not report itself as
        // *this request's* terminal outcome.
        let progress: std::sync::Arc<dyn crate::serve::TurnHost> = std::sync::Arc::new(
            ParentHostSansTurnError(std::sync::Arc::clone(&ctx.progress)),
        );

        let (text, stop) = crate::serve::TurnLoop {
            state: &ctx.state,
            provider: &ctx.provider,
            system: Some(&system),
            tool_specs: &specs,
            progress: &progress,
            timer_origin: ctx.timer_origin.clone(),
            namespace,
            // No session behind it. The conversation exists for the
            // length of this call and is then dropped — that is what
            // "context isolation" means here.
            persistence: None,
        }
        .run(&mut history)
        .await;

        Ok(match stop {
            crate::serve::TurnStop::BudgetExhausted { partial_text } => format!(
                "[the subagent used its whole tool budget without finishing]\n\n{partial_text}"
            ),
            _ => text.unwrap_or_else(|| "[the subagent produced no answer]".to_string()),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn defs() -> Vec<crate::agents::AgentDef> {
        vec![crate::agents::AgentDef {
            name: "reviewer".to_string(),
            description: "Reviews a diff.".to_string(),
            tools: Some(vec!["client_file_read".to_string()]),
            prompt: "You are a reviewer.".to_string(),
        }]
    }

    /// The description is the parent model's only basis for choosing,
    /// so every agent's own description has to reach it.
    #[test]
    fn the_tool_description_lists_every_agent() {
        let spec = SubagentTool::new(defs()).spec().clone();
        assert!(
            spec.description.contains("reviewer"),
            "{}",
            spec.description
        );
        assert!(
            spec.description.contains("Reviews a diff."),
            "{}",
            spec.description
        );
    }

    #[test]
    fn the_kind_is_other() {
        assert_eq!(SubagentTool::new(defs()).kind(), ToolKind::Other);
    }

    /// A name the operator never defined is a mistake the parent can
    /// recover from — list what exists rather than just refusing.
    #[tokio::test]
    async fn an_unknown_agent_names_the_ones_that_exist() {
        let err = SubagentTool::new(defs())
            .execute(&serde_json::json!({"agent": "nope", "prompt": "x"}))
            .await
            .unwrap_err()
            .to_string();
        assert!(err.contains("reviewer"), "{err}");
    }

    /// Outside a turn there is nothing to delegate with. Refusing here
    /// is what keeps the tool honest on any path that is not a live
    /// turn.
    #[tokio::test]
    async fn delegating_outside_a_turn_refuses() {
        let err = SubagentTool::new(defs())
            .execute(&serde_json::json!({"agent": "reviewer", "prompt": "x"}))
            .await
            .unwrap_err()
            .to_string();
        assert!(err.contains("no turn"), "{err}");
    }

    /// The point of the feature: a subagent's system prompt is its own
    /// definition and nothing else. A `SOUL.md` in the workspace must
    /// not reach it.
    #[test]
    fn the_system_prompt_is_the_definition_body_plus_the_date() {
        let sys = subagent_system_prompt(&defs()[0]);
        assert!(sys.contains("You are a reviewer."));
        assert!(
            sys.contains("Current Date and Time"),
            "the date is the one inherited fact: {sys}"
        );
        for absent in ["# Soul", "# Identity", "# User", "# Agent Instructions"] {
            assert!(
                !sys.contains(absent),
                "{absent} must not be inherited: {sys}"
            );
        }
    }

    /// `subagent_tool_specs` never lists `subagent` — the parent
    /// model's basis for choosing what to call — but the list alone is
    /// only a hint. What actually caps delegation depth is
    /// `TurnLoop::run`'s permission gate refusing any call outside a
    /// round's own `tool_specs`; see
    /// `a_subagent_cannot_invoke_subagent_by_name` in `src/serve/mod.rs`
    /// for the test that pins the gate itself, not just the list this
    /// function builds.
    #[test]
    fn a_subagents_tool_list_never_contains_subagent() {
        let parent_visible = [
            spec_named("client_file_read"),
            spec_named(SUBAGENT_TOOL_NAME),
        ];
        let inherited = subagent_tool_specs(&defs()[0], &parent_visible);
        assert!(!inherited.iter().any(|s| s.name == SUBAGENT_TOOL_NAME));

        let unrestricted = crate::agents::AgentDef {
            tools: None,
            ..defs()[0].clone()
        };
        let inherited = subagent_tool_specs(&unrestricted, &parent_visible);
        assert!(!inherited.iter().any(|s| s.name == SUBAGENT_TOOL_NAME));
        assert!(inherited.iter().any(|s| s.name == "client_file_read"));
    }

    /// Even when the definition asks for it by name.
    #[test]
    fn a_definition_cannot_grant_itself_subagent() {
        let greedy = crate::agents::AgentDef {
            tools: Some(vec![
                SUBAGENT_TOOL_NAME.to_string(),
                "client_file_read".to_string(),
            ]),
            ..defs()[0].clone()
        };
        let parent_visible = [
            spec_named("client_file_read"),
            spec_named(SUBAGENT_TOOL_NAME),
        ];
        let inherited = subagent_tool_specs(&greedy, &parent_visible);
        assert!(!inherited.iter().any(|s| s.name == SUBAGENT_TOOL_NAME));
    }

    /// An empty list is a definition, not an omission.
    #[test]
    fn an_empty_tools_list_yields_no_tools() {
        let toolless = crate::agents::AgentDef {
            tools: Some(vec![]),
            ..defs()[0].clone()
        };
        let parent_visible = [spec_named("client_file_read")];
        assert!(subagent_tool_specs(&toolless, &parent_visible).is_empty());
    }

    fn spec_named(name: &str) -> crate::provider::ToolSpec {
        crate::provider::ToolSpec {
            name: name.to_string().into(),
            description: "…".into(),
            input_schema: serde_json::json!({"type": "object"}),
        }
    }

    /// A name in `tools:` that the parent cannot see is reported once —
    /// not on every delegation — and a name that *is* visible is never
    /// reported at all.
    #[test]
    fn an_unknown_tool_name_is_reported_once() {
        let tool = SubagentTool::new(defs());
        let def = &tool.agents[0]; // tools: Some(["client_file_read"])
        let parent_visible = [spec_named("client_file_read")];

        let unknown = crate::agents::AgentDef {
            tools: Some(vec!["client_file_read".to_string(), "retrieve".to_string()]),
            ..def.clone()
        };

        let first = tool.newly_unknown_tools(&unknown, &parent_visible);
        assert_eq!(first, vec!["retrieve".to_string()], "{first:?}");

        let second = tool.newly_unknown_tools(&unknown, &parent_visible);
        assert!(
            second.is_empty(),
            "the same (agent, name) pair must not be reported twice: {second:?}"
        );
    }

    /// `subagent` named in a definition's own `tools:` is not a typo —
    /// `subagent_tool_specs` always drops it on purpose — so it must
    /// never show up as an "unknown tool" warning.
    #[test]
    fn subagent_itself_is_never_reported_as_unknown() {
        let tool = SubagentTool::new(vec![crate::agents::AgentDef {
            tools: Some(vec![SUBAGENT_TOOL_NAME.to_string()]),
            ..defs()[0].clone()
        }]);
        let def = &tool.agents[0];
        assert!(tool.newly_unknown_tools(def, &[]).is_empty());
    }

    /// An unrestricted definition (`tools: None`) has nothing to check
    /// against the parent's visible set — there is no list to contain a
    /// typo.
    #[test]
    fn an_unrestricted_definition_has_nothing_to_warn_about() {
        let tool = SubagentTool::new(vec![crate::agents::AgentDef {
            tools: None,
            ..defs()[0].clone()
        }]);
        let def = &tool.agents[0];
        assert!(tool.newly_unknown_tools(def, &[]).is_empty());
    }

    /// A minimal `TurnHost` that records every call it receives, so a
    /// test can tell `ParentHostSansTurnError` actually forwards to the
    /// wrapped host rather than silently no-op'ing everything.
    #[derive(Default)]
    struct RecordingHost {
        turn_errors: std::sync::Mutex<Vec<String>>,
        tool_starts: std::sync::Mutex<Vec<String>>,
    }

    #[async_trait]
    impl crate::serve::TurnHost for RecordingHost {
        async fn tool_start(&self, id: &str, _name: &str) {
            self.tool_starts.lock().unwrap().push(id.to_string());
        }
        async fn tool_end(&self, _id: &str, _name: &str) {}
        async fn turn_error(&self, message: &str) {
            self.turn_errors.lock().unwrap().push(message.to_string());
        }
        fn origin(&self) -> crate::tools::policy::Origin {
            crate::tools::policy::Origin::Channel
        }
    }

    /// The one behaviour this wrapper exists to change: `turn_error`
    /// never reaches the wrapped host, so a subagent's provider failure
    /// cannot masquerade as the parent turn's own terminal outcome (see
    /// the type doc — this is Fix 2's regression test).
    #[tokio::test]
    async fn turn_error_is_swallowed_not_forwarded() {
        let inner = std::sync::Arc::new(RecordingHost::default());
        let wrapped = ParentHostSansTurnError(inner.clone());

        crate::serve::TurnHost::turn_error(&wrapped, "the subagent's provider broke").await;

        assert!(
            inner.turn_errors.lock().unwrap().is_empty(),
            "turn_error must not reach the parent's host"
        );
    }

    /// Every other method forwards unchanged — delegation must stay
    /// inside the same permission gate the parent itself is judged by.
    #[tokio::test]
    async fn every_other_method_forwards_to_the_parent_host() {
        let inner = std::sync::Arc::new(RecordingHost::default());
        let wrapped = ParentHostSansTurnError(inner.clone());

        assert_eq!(
            crate::serve::TurnHost::origin(&wrapped),
            crate::tools::policy::Origin::Channel,
            "origin() must be the parent's own, unchanged"
        );

        crate::serve::TurnHost::tool_start(&wrapped, "call-1", "some_tool").await;
        assert_eq!(
            inner.tool_starts.lock().unwrap().as_slice(),
            ["call-1".to_string()],
            "tool_start must still reach the parent's host"
        );
    }
}
