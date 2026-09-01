//! `subagent`: delegate a task to a specialised agent.
//!
//! A subagent runs a whole nested conversation — its own system prompt,
//! its own tool-calling loop, its own history — and hands back only its
//! final answer. Two properties make that safe rather than a loophole:
//!
//! 1. **Judged by the parent's `Origin`, through the parent's
//!    `TurnHost`.** The nested loop (`crate::serve::TurnLoop`) is handed
//!    the caller's own `progress` unchanged (see [`TurnContext`] in
//!    `src/serve/mod.rs`). If delegation ran under its own host or its
//!    own origin, "ask a subagent" would become a way to get done by
//!    proxy what the model was refused directly.
//! 2. **The system prompt is the definition and nothing else.** See
//!    [`subagent_system_prompt`].
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
//! **Lock re-entrancy.** Because `SubagentTool::execute` runs *inside*
//! `ToolSet::execute` (it is itself one of the tools that set owns), and
//! the nested loop it drives calls back into the very same
//! `ToolSet::execute` for its own tool calls, `ToolSet::execute`'s
//! `self.inner.read().await` is acquired twice, re-entrantly, on the
//! same task before the outer acquisition is released.
//! `tokio::sync::RwLock` only blocks a new reader behind a writer that
//! is already queued, so this is safe as long as nothing requests the
//! write lock while a turn's tools are running — which is *not* quite
//! true: `mcp_reconnect` (`crate::tools::builtin_tools::McpReconnectTool`
//! → `ToolSet::reconnect_mcp_server` → `inner.write().await`) is
//! model-callable from inside `ToolSet::execute`, exactly like this
//! tool. It already deadlocks on its own today, though — it asks for
//! the write lock while `ToolSet::execute`'s own read guard for *that
//! very call* is still held, with nothing nested involved — so every
//! new hang scenario a subagent's re-entrant read could reach already
//! contains that existing one, and this feature introduces no new
//! deadlock. See the `mcp_reconnect` self-deadlock issue for that bug
//! itself; it is not fixed here.

use crate::agents::AgentDef;
use crate::provider::{ChatMessage, ToolSpec};
use crate::tools::{Tool, ToolKind};
use anyhow::Context;
use async_trait::async_trait;
use serde_json::json;

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
/// selects from it. Either way `subagent` itself is removed, which is
/// what caps delegation depth at one — a bound by construction rather
/// than a counter that has to be threaded through.
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

/// Delegates a task to a specialised agent and returns only its final
/// answer. See the module docs for the two properties this exists to
/// establish.
pub struct SubagentTool {
    agents: Vec<AgentDef>,
    spec: ToolSpec,
}

impl SubagentTool {
    pub fn new(agents: Vec<AgentDef>) -> Self {
        let spec = build_spec(&agents);
        Self { agents, spec }
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

        let system = subagent_system_prompt(def);
        let specs = subagent_tool_specs(def, &ctx.visible_specs);
        let mut history = vec![ChatMessage::user(prompt)];

        // The subagent's own memory-tool calls (if it has any) write
        // under the same namespace the delegating conversation is in —
        // a fact about where this deployment's memory lives, not a
        // personality trait, so it travels through like the date does
        // rather than being stripped like the workspace files are.
        let namespace = crate::tools::workspace_tools::current_memory_namespace();

        let (text, stop) = crate::serve::TurnLoop {
            state: &ctx.state,
            provider: &ctx.provider,
            system: Some(&system),
            tool_specs: &specs,
            // The parent's host, deliberately: a permission request
            // from a subagent must reach the same person, judged by the
            // same origin. A different host here would make delegation
            // a way around the gate.
            progress: &ctx.progress,
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

    /// Depth is capped at one by construction: a subagent cannot see
    /// the tool that would let it delegate again.
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
}
