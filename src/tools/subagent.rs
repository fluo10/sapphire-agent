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
///
/// A dispatched agent's answer is prefixed with a resumable handle (see
/// [`prefixed`]); passing that handle back as `resume` continues the
/// same child conversation — its own history, its own system prompt —
/// instead of starting a fresh one. `agent` and `resume` are mutually
/// exclusive: `SubagentTool::execute` rejects a call giving both, or
/// neither, naming the rule rather than picking a default.
fn build_spec(agents: &[AgentDef]) -> ToolSpec {
    let mut description = String::from(
        "Delegate a task to a specialised agent, or continue one you already \
         started. A dispatched agent's own system prompt and its own \
         conversation are its own — only its final answer comes back — use \
         this to keep a large investigation out of this conversation. Its \
         answer is prefixed with a handle; pass that back as `resume` (with \
         a new `prompt`) to continue that same conversation instead of \
         starting a fresh one.\n\nAvailable agents:\n",
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
                        listed in this tool's own description. Mutually \
                        exclusive with `resume`."
                },
                "resume": {
                    "type": "string",
                    "description": "A handle from an earlier delegation, to \
                        continue that subagent's own conversation. Mutually \
                        exclusive with `agent`."
                },
                "prompt": {
                    "type": "string",
                    "description": "The task, or the next instruction for a \
                        resumed subagent. It sees only this (plus its own \
                        prior history, when resuming) and its own \
                        definition — nothing else from the current \
                        conversation."
                }
            },
            "required": ["prompt"]
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
/// have said. Swallowing it here keeps a subagent's provider failure off
/// that channel — but it does not surface the cause anywhere the model
/// can read. `TurnStop::ProviderError` yields `text: None`, so
/// `SubagentTool::execute`'s match on `stop` falls through to a generic
/// `"[the subagent produced no answer]"` result (`src/tools/subagent.rs`,
/// below) — the parent model sees only that, never the specific error.
/// The cause is not lost operationally: `run_llm_turn`
/// (`src/serve/mod.rs`) logs it (`error!("Provider error: {e:#}")`)
/// before calling `turn_error`, so it is visible there. Whether it
/// should also reach the parent model is a separate design question,
/// not settled by this wrapper.
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

    /// Swallowed — see the type doc. Nothing is lost operationally (the
    /// cause is logged before this would have fired); it just does not
    /// also masquerade as *this* turn's terminal outcome, and — per the
    /// type doc — it does not reach the parent model either.
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
    /// Handles currently being resumed, for the duration of one
    /// `resume` call each. Claimed and released through [`ResumeGuard`]
    /// — a `Drop` guard, not a remember-to-remove-the-entry pattern, for
    /// the same reason `TerminalReservation`
    /// (`src/tools/acp_client.rs`) is one: two turns resuming the same
    /// handle concurrently would interleave writes into one history, and
    /// a resume that errors out — or whose turn is simply cancelled
    /// mid-flight, which ACP treats as routine — must still release the
    /// handle rather than leaving it refused forever.
    busy_handles: std::sync::Mutex<std::collections::HashSet<String>>,
}

impl SubagentTool {
    pub fn new(agents: Vec<AgentDef>) -> Self {
        let spec = build_spec(&agents);
        Self {
            agents,
            spec,
            warned_unknown_tools: std::sync::Mutex::new(std::collections::HashSet::new()),
            busy_handles: std::sync::Mutex::new(std::collections::HashSet::new()),
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
        let prompt = input["prompt"].as_str().context("missing 'prompt'")?;
        let agent = input.get("agent").and_then(|v| v.as_str());
        let resume = input.get("resume").and_then(|v| v.as_str());

        // Neither given, or both given, is a recoverable error naming
        // the rule rather than guessing which one was meant.
        match (agent, resume) {
            (Some(_), Some(_)) => anyhow::bail!(
                "'agent' and 'resume' are mutually exclusive: dispatch a new \
                 agent by name, or continue an existing one by its handle — \
                 not both in the same call"
            ),
            (None, None) => anyhow::bail!(
                "either 'agent' (to dispatch a new subagent) or 'resume' \
                 (to continue one by handle) is required"
            ),
            (Some(name), None) => self.dispatch(name, prompt).await,
            (None, Some(handle)) => self.resume(handle, prompt).await,
        }
    }
}

impl SubagentTool {
    /// Start a fresh child conversation with `def`, run it to
    /// completion, and store it under a freshly generated handle.
    async fn dispatch(&self, name: &str, prompt: &str) -> anyhow::Result<String> {
        let Some(def) = self.agents.iter().find(|a| a.name == name) else {
            // Recoverable: the parent picked a name that does not
            // exist, and can pick again if it is told what does.
            let known: Vec<&str> = self.agents.iter().map(|a| a.name.as_str()).collect();
            anyhow::bail!("no agent named '{name}'. Available: {}", known.join(", "));
        };

        // Only a live turn has a model, a permission host and a visible
        // tool set to lend. Nothing else can delegate.
        let ctx = crate::serve::current_turn_context().context("no turn to delegate from")?;

        let mut history = vec![ChatMessage::user(prompt)];
        // `Uuid::now_v7`'s hyphenated `Display` form is ASCII hex plus
        // `-`, which `SubagentCache::path_for`'s filename guard accepts
        // outright — no reserved-name collision is possible, and
        // nothing here needs the shorter `simple` rendering.
        let handle = uuid::Uuid::now_v7().to_string();
        let created_at = chrono::Utc::now();

        Ok(self
            .run_and_store(&ctx, def, &handle, created_at, &mut history)
            .await)
    }

    /// Continue the child conversation stored under `handle`.
    ///
    /// The definition is reloaded by its stored *name* from `self.agents`
    /// — the current, live list — never from anything stored alongside
    /// the history. That is what makes `subagent_tool_specs` recompute
    /// the offered tool list on resume instead of restoring a stale one:
    /// see the module doc's third property. A stored tool list would
    /// reopen the hole `TurnLoop::run`'s offer gate exists to close, by
    /// letting a resumed child carry forward a list wider than its
    /// current definition allows (or than the current parent turn can
    /// even see).
    async fn resume(&self, handle: &str, prompt: &str) -> anyhow::Result<String> {
        let ctx = crate::serve::current_turn_context().context("no turn to delegate from")?;

        // Claimed before anything else in this method can `.await`, so
        // a concurrent resume of the same handle always observes it
        // already held — see `ResumeGuard`'s doc. Released on every
        // exit path via `Drop`, success, error, or this whole call
        // being cancelled mid-flight.
        let _guard = ResumeGuard::claim(&self.busy_handles, handle)?;

        let Some(cache) = &ctx.state.subagent_cache else {
            anyhow::bail!(
                "no subagent is stored under handle '{handle}' — dispatch a \
                 new one instead, with `agent` set"
            );
        };
        let Some(stored) = cache.get(handle) else {
            anyhow::bail!(
                "no subagent is stored under handle '{handle}' — dispatch a \
                 new one instead, with `agent` set"
            );
        };

        let Some(def) = self.agents.iter().find(|a| a.name == stored.agent) else {
            // The definition this handle belongs to is gone (renamed,
            // deleted, or the operator's `.md` edit dropped it) — the
            // entry can never be resumed again, so it is not worth
            // keeping around; drop it rather than leaving a permanent
            // orphan for `prune_before` to eventually find.
            cache.remove(handle);
            anyhow::bail!(
                "the '{}' agent definition that handle '{handle}' belongs \
                 to no longer exists; dispatch a new agent instead",
                stored.agent
            );
        };

        let mut history = stored.history;
        history.push(ChatMessage::user(prompt));

        Ok(self
            .run_and_store(&ctx, def, handle, stored.created_at, &mut history)
            .await)
    }

    /// Run `def`'s nested `TurnLoop` to completion on `history`, then
    /// store the result under `handle` and prefix the answer with what
    /// that store attempt means for resumability. Shared by
    /// [`Self::dispatch`] (a freshly generated handle) and
    /// [`Self::resume`] (the same handle it was given), so this
    /// run-then-persist sequence — and the "not resumable" fallback
    /// when `put` refuses an over-cap entry — is written once.
    async fn run_and_store(
        &self,
        ctx: &crate::serve::TurnContext,
        def: &AgentDef,
        handle: &str,
        created_at: chrono::DateTime<chrono::Utc>,
        history: &mut Vec<ChatMessage>,
    ) -> String {
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
            // "context isolation" means here. Resumability is a
            // separate, workspace-external mechanism (`SubagentCache`),
            // not a promise that this history reaches the store.
            persistence: None,
        }
        .run(history)
        .await;

        let answer = answer_text(text, stop);
        let history = std::mem::take(history);
        let resumability = persist(
            ctx.state.subagent_cache.as_deref(),
            handle,
            &def.name,
            history,
            created_at,
        );
        prefixed(&def.name, resumability, &answer)
    }
}

/// How a dispatched or resumed child's answer should describe its own
/// resumability to the model — see [`prefixed`].
enum Resumability {
    Resumable(String),
    NotResumable(&'static str),
}

/// Persist `history` under `handle` if a cache is configured, choosing
/// how the caller should describe resumability.
///
/// Never truncates: `SubagentCache::put` refuses an over-cap entry
/// wholesale rather than dropping old messages to fit (see its own doc
/// for why — a `tool_use` cut loose from its matching `tool_result`
/// produces a history the provider API rejects outright, making the
/// entry unloadable rather than merely shorter). So `Ok(false)` here
/// still returns the answer normally; only the resumability marker
/// changes.
fn persist(
    cache: Option<&crate::subagent_cache::SubagentCache>,
    handle: &str,
    agent_name: &str,
    history: Vec<ChatMessage>,
    created_at: chrono::DateTime<chrono::Utc>,
) -> Resumability {
    let Some(cache) = cache else {
        return Resumability::NotResumable("no resume cache is configured on this deployment");
    };
    let child = crate::subagent_cache::StoredChild {
        agent: agent_name.to_string(),
        history,
        created_at,
        updated_at: chrono::Utc::now(),
    };
    match cache.put(handle, &child) {
        Ok(true) => Resumability::Resumable(handle.to_string()),
        Ok(false) => Resumability::NotResumable("history exceeded the cache limit"),
        Err(e) => {
            warn!("subagent cache: failed to store handle '{handle}': {e:#}");
            Resumability::NotResumable("could not be saved to the resume cache")
        }
    }
}

/// Prefix a child's answer with its handle, or with why it has none.
fn prefixed(agent_name: &str, resumability: Resumability, answer: &str) -> String {
    match resumability {
        Resumability::Resumable(handle) => {
            format!("[subagent {agent_name} · handle {handle}]\n{answer}")
        }
        Resumability::NotResumable(reason) => {
            format!("[subagent {agent_name} · not resumable: {reason}]\n{answer}")
        }
    }
}

/// What the parent model is told for a nested turn's own outcome.
fn answer_text(text: Option<String>, stop: crate::serve::TurnStop) -> String {
    match stop {
        crate::serve::TurnStop::BudgetExhausted { partial_text } => {
            format!("[the subagent used its whole tool budget without finishing]\n\n{partial_text}")
        }
        _ => text.unwrap_or_else(|| "[the subagent produced no answer]".to_string()),
    }
}

/// Holds one resume's exclusive claim on a handle for the duration of
/// the call.
///
/// A `Drop` guard rather than a remember-to-remove-the-entry pattern,
/// for the same reason `TerminalReservation` (`src/tools/acp_client.rs`)
/// is one: a resume that returns an error, or whose whole call is
/// cancelled mid-flight (the turn's future simply dropped — ACP treats
/// that as routine, no different from an Escape in the editor or a
/// dropped socket), must still release the handle, or it stays refused
/// forever. `claim` does all its work synchronously — a `Mutex` lock,
/// an insert — before this method's first `.await`, so two concurrent
/// resumes of the same handle can never both observe it free.
struct ResumeGuard<'a> {
    busy: &'a std::sync::Mutex<std::collections::HashSet<String>>,
    handle: String,
}

impl<'a> ResumeGuard<'a> {
    /// Claim `handle`, refusing if another resume already holds it.
    fn claim(
        busy: &'a std::sync::Mutex<std::collections::HashSet<String>>,
        handle: &str,
    ) -> anyhow::Result<Self> {
        let mut set = busy.lock().unwrap();
        if !set.insert(handle.to_string()) {
            anyhow::bail!(
                "subagent handle '{handle}' is already in use by another \
                 resume — wait for it to finish before resuming again"
            );
        }
        Ok(Self {
            busy,
            handle: handle.to_string(),
        })
    }
}

impl Drop for ResumeGuard<'_> {
    fn drop(&mut self) {
        self.busy.lock().unwrap().remove(&self.handle);
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

    // -----------------------------------------------------------------
    // Resume (Task 6)
    // -----------------------------------------------------------------

    /// Definitions carrying two tools, so the depth-cap test below can
    /// tell "the offered list was recomputed" apart from "the offered
    /// list happened to be empty".
    fn resumable_defs() -> Vec<crate::agents::AgentDef> {
        vec![crate::agents::AgentDef {
            name: "impl".to_string(),
            description: "Implements a task.".to_string(),
            tools: Some(vec![
                "client_file_read".to_string(),
                "client_shell".to_string(),
            ]),
            prompt: "You are impl.".to_string(),
        }]
    }

    fn text_response(text: &str) -> crate::provider::ChatResponse {
        crate::provider::ChatResponse {
            text: Some(text.to_string()),
            tool_calls: Vec::new(),
            stop_reason: None,
        }
    }

    /// Pulls the handle out of a `[subagent <name> · handle <handle>]`
    /// prefix — the exact format `prefixed` produces for a resumable
    /// answer.
    fn extract_handle(answer: &str) -> String {
        let after = answer
            .split("handle ")
            .nth(1)
            .unwrap_or_else(|| panic!("no handle in: {answer}"));
        after
            .split(']')
            .next()
            .unwrap_or_else(|| panic!("unterminated handle in: {answer}"))
            .to_string()
    }

    /// A provider double that records the full `(system, messages,
    /// tool_specs)` of every `chat()` call, not just the scripted
    /// outcome.
    ///
    /// `serve::StubProvider`'s own `ChatLog` (used by the depth-cap test
    /// in `src/serve/mod.rs`) only records tool *names*, which is enough
    /// to prove a call was offered a given set of tools but says
    /// nothing about message *content* — so it cannot answer "did the
    /// resumed turn's provider call actually carry its own stored
    /// history, and none of whatever else was in scope". This exists
    /// for that, kept local to this module rather than extending
    /// `ChatLog` for one property only these tests need.
    ///
    /// `chat()` yields once before answering (`tokio::task::yield_now`),
    /// the same technique `acp_client::tests::FakeClient::create_terminal`
    /// uses: a fake that resolved synchronously would let one resume's
    /// entire call — claim, run, persist, release — complete within a
    /// single poll, never giving a concurrently-started second resume a
    /// chance to observe the busy guard still held.
    #[derive(Default)]
    struct ScriptedProvider {
        script: std::sync::Mutex<std::collections::VecDeque<crate::provider::ChatResponse>>,
        calls: std::sync::Mutex<Vec<(Option<String>, Vec<ChatMessage>, Vec<ToolSpec>)>>,
    }

    impl ScriptedProvider {
        fn new(script: Vec<crate::provider::ChatResponse>) -> std::sync::Arc<Self> {
            std::sync::Arc::new(Self {
                script: std::sync::Mutex::new(script.into()),
                calls: std::sync::Mutex::new(Vec::new()),
            })
        }

        /// The `messages` the most recent `chat()` call received.
        fn last_messages(&self) -> Vec<ChatMessage> {
            self.calls
                .lock()
                .unwrap()
                .last()
                .expect("a call was recorded")
                .1
                .clone()
        }

        /// The `tool_specs` the most recent `chat()` call received.
        fn last_specs(&self) -> Vec<ToolSpec> {
            self.calls
                .lock()
                .unwrap()
                .last()
                .expect("a call was recorded")
                .2
                .clone()
        }
    }

    #[async_trait]
    impl crate::provider::Provider for ScriptedProvider {
        fn name(&self) -> &str {
            "scripted"
        }

        async fn chat(
            &self,
            system: Option<&str>,
            messages: &[ChatMessage],
            tools: Option<&[ToolSpec]>,
        ) -> anyhow::Result<crate::provider::ChatResponse> {
            self.calls.lock().unwrap().push((
                system.map(|s| s.to_string()),
                messages.to_vec(),
                tools.map(|t| t.to_vec()).unwrap_or_default(),
            ));
            tokio::task::yield_now().await;
            self.script
                .lock()
                .unwrap()
                .pop_front()
                .ok_or_else(|| anyhow::anyhow!("ScriptedProvider script exhausted"))
        }
    }

    /// A `TurnContext` built by hand rather than through a real parent
    /// turn — `SubagentTool::execute` only ever reads it through
    /// `current_turn_context()`, so scoping one directly around the call
    /// under test is enough, and keeps these tests from having to drive
    /// a whole outer `run_llm_turn` just to reach the resume path.
    fn turn_context(
        state: std::sync::Arc<crate::serve::ServeState>,
        provider: std::sync::Arc<dyn crate::provider::Provider>,
        visible_specs: Vec<ToolSpec>,
    ) -> std::sync::Arc<crate::serve::TurnContext> {
        std::sync::Arc::new(crate::serve::TurnContext {
            state,
            provider,
            progress: std::sync::Arc::new(crate::serve::NullProgress),
            visible_specs: visible_specs.into(),
            timer_origin: None,
            // `None` here matches what a subagent's own nested `TurnLoop`
            // sees by construction (see `TurnContext::session_id`'s
            // doc) — nothing under test reads it, but a stray unwrap
            // added later should fail loudly on `None`, not silently
            // pass because a test handed it a `Some`.
            session_id: None,
        })
    }

    /// A dispatched child's next resume must see its own prior history —
    /// the dispatch prompt and the model's own reply to it — appended
    /// to, not replacing, whatever it is asked next. Nothing here routes
    /// a parent's own conversation through this path at all (a
    /// subagent's history is always built from scratch on dispatch and
    /// `stored.history` alone on resume — see `dispatch`/`resume`), so
    /// this also stands as the regression test for that: if resume ever
    /// started pulling from anything other than the stored child, this
    /// assertion is where it would first show up as unexpected text.
    #[tokio::test]
    async fn a_resumed_child_continues_its_own_history() {
        let tool = SubagentTool::new(resumable_defs());
        let state = crate::serve::ServeState::for_test(false);

        let dispatch_provider = ScriptedProvider::new(vec![text_response("dispatch answer")]);
        let dispatch_input = serde_json::json!({"agent": "impl", "prompt": "first task"});
        let dispatched = crate::serve::scope_turn_context(
            turn_context(
                std::sync::Arc::clone(&state),
                std::sync::Arc::clone(&dispatch_provider)
                    as std::sync::Arc<dyn crate::provider::Provider>,
                Vec::new(),
            ),
            tool.execute(&dispatch_input),
        )
        .await
        .unwrap();
        let handle = extract_handle(&dispatched);

        let resume_provider = ScriptedProvider::new(vec![text_response("resume answer")]);
        let resume_input = serde_json::json!({"resume": handle, "prompt": "second instruction"});
        crate::serve::scope_turn_context(
            turn_context(
                std::sync::Arc::clone(&state),
                std::sync::Arc::clone(&resume_provider)
                    as std::sync::Arc<dyn crate::provider::Provider>,
                Vec::new(),
            ),
            tool.execute(&resume_input),
        )
        .await
        .unwrap();

        let texts: Vec<String> = resume_provider
            .last_messages()
            .iter()
            .filter_map(|m| m.text())
            .collect();
        assert!(
            texts.iter().any(|t| t.contains("first task")),
            "the resumed call must see the dispatch prompt: {texts:?}"
        );
        assert!(
            texts.iter().any(|t| t.contains("dispatch answer")),
            "the resumed call must see the dispatch reply: {texts:?}"
        );
        assert!(
            texts.iter().any(|t| t.contains("second instruction")),
            "the resumed call must see its own new prompt: {texts:?}"
        );
        assert!(
            !texts.iter().any(|t| t.contains("PARENT ONLY")),
            "nothing from outside the stored child history may reach a \
             resumed call: {texts:?}"
        );
    }

    /// Restoring a stored tool list on resume would make resume the
    /// hole `subagent_tool_specs` and `TurnLoop::run`'s offer gate exist
    /// to close (see the module doc's third property, and `resume`'s
    /// own doc). Asserted by full equality against exactly the two
    /// tools `resumable_defs`' `tools:` list names — not merely "no
    /// `subagent` in the list" — because a resume that happened to
    /// widen or narrow the list to anything else would pass a weaker
    /// check just as easily.
    #[tokio::test]
    async fn resume_recomputes_the_tool_list_so_the_depth_cap_still_holds() {
        let tool = SubagentTool::new(resumable_defs());
        let state = crate::serve::ServeState::for_test(false);

        // Wider than `resumable_defs`' own `tools:` list, and including
        // `subagent` itself, so a resumed list that was merely *not
        // empty* (rather than exactly recomputed) would still fail this
        // test.
        let visible = vec![
            spec_named("client_file_read"),
            spec_named("client_shell"),
            spec_named("some_other_tool"),
            spec_named(SUBAGENT_TOOL_NAME),
        ];

        let dispatch_provider = ScriptedProvider::new(vec![text_response("ok")]);
        let dispatch_input = serde_json::json!({"agent": "impl", "prompt": "go"});
        let dispatched = crate::serve::scope_turn_context(
            turn_context(
                std::sync::Arc::clone(&state),
                std::sync::Arc::clone(&dispatch_provider)
                    as std::sync::Arc<dyn crate::provider::Provider>,
                visible.clone(),
            ),
            tool.execute(&dispatch_input),
        )
        .await
        .unwrap();
        let handle = extract_handle(&dispatched);

        let resume_provider = ScriptedProvider::new(vec![text_response("ok again")]);
        let resume_input = serde_json::json!({"resume": handle, "prompt": "next"});
        crate::serve::scope_turn_context(
            turn_context(
                std::sync::Arc::clone(&state),
                std::sync::Arc::clone(&resume_provider)
                    as std::sync::Arc<dyn crate::provider::Provider>,
                visible,
            ),
            tool.execute(&resume_input),
        )
        .await
        .unwrap();

        let specs = resume_provider.last_specs();
        let names: Vec<&str> = specs.iter().map(|s| s.name.as_ref()).collect();
        assert_eq!(names, vec!["client_file_read", "client_shell"]);
    }

    /// A handle nobody ever stored (typo'd, expired, from a different
    /// process) is recoverable the same way an unknown agent name is:
    /// told what to do instead, not just refused.
    #[tokio::test]
    async fn an_unknown_handle_is_recoverable_and_says_what_to_do() {
        let tool = SubagentTool::new(defs());
        let state = crate::serve::ServeState::for_test(false);
        let provider = ScriptedProvider::new(Vec::new());
        let input = serde_json::json!({"resume": "nosuchhandle", "prompt": "x"});

        let err = crate::serve::scope_turn_context(
            turn_context(
                state,
                std::sync::Arc::clone(&provider) as std::sync::Arc<dyn crate::provider::Provider>,
                Vec::new(),
            ),
            tool.execute(&input),
        )
        .await
        .unwrap_err()
        .to_string();
        assert!(err.contains("dispatch"), "got: {err}");
    }

    /// Two turns resuming one handle at once would interleave writes
    /// into a single stored history. `ScriptedProvider::chat`'s forced
    /// yield (see its doc) is what makes the second attempt
    /// deterministically observe the first still holding the guard,
    /// rather than depending on scheduler luck.
    #[tokio::test]
    async fn a_busy_handle_is_refused() {
        let tool = SubagentTool::new(defs());
        let state = crate::serve::ServeState::for_test(false);

        let dispatch_provider = ScriptedProvider::new(vec![text_response("dispatch answer")]);
        let dispatch_input = serde_json::json!({"agent": "reviewer", "prompt": "go"});
        let dispatched = crate::serve::scope_turn_context(
            turn_context(
                std::sync::Arc::clone(&state),
                std::sync::Arc::clone(&dispatch_provider)
                    as std::sync::Arc<dyn crate::provider::Provider>,
                Vec::new(),
            ),
            tool.execute(&dispatch_input),
        )
        .await
        .unwrap();
        let handle = extract_handle(&dispatched);

        let provider_a = ScriptedProvider::new(vec![text_response("a")]);
        let provider_b = ScriptedProvider::new(vec![text_response("b")]);
        let input_a = serde_json::json!({"resume": handle, "prompt": "x"});
        let input_b = serde_json::json!({"resume": handle, "prompt": "y"});

        let fut_a = crate::serve::scope_turn_context(
            turn_context(
                std::sync::Arc::clone(&state),
                std::sync::Arc::clone(&provider_a) as std::sync::Arc<dyn crate::provider::Provider>,
                Vec::new(),
            ),
            tool.execute(&input_a),
        );
        let fut_b = crate::serve::scope_turn_context(
            turn_context(
                std::sync::Arc::clone(&state),
                std::sync::Arc::clone(&provider_b) as std::sync::Arc<dyn crate::provider::Provider>,
                Vec::new(),
            ),
            tool.execute(&input_b),
        );

        let (first, second) = tokio::join!(fut_a, fut_b);
        assert!(first.is_ok(), "{first:?}");
        assert!(
            second.unwrap_err().to_string().contains("in use"),
            "the second concurrent resume of the same handle must be refused"
        );
    }

    /// A definition reload (the operator edited the `.md`, or dropped
    /// the agent) is not the same failure as an unknown handle — the
    /// handle is real, its child answered before, but the definition it
    /// belongs to is gone. Simulated with two separate `SubagentTool`
    /// instances sharing one cache-backed `ServeState`, standing in for
    /// "the process reloaded its agent definitions between dispatch and
    /// resume" without needing a mutable, shared agents list.
    #[tokio::test]
    async fn an_agent_definition_that_disappeared_is_reported() {
        let state = crate::serve::ServeState::for_test(false);

        let dispatching_tool = SubagentTool::new(vec![crate::agents::AgentDef {
            name: "impl".to_string(),
            description: "Implements a task.".to_string(),
            tools: None,
            prompt: "You are impl.".to_string(),
        }]);
        let dispatch_provider = ScriptedProvider::new(vec![text_response("dispatch answer")]);
        let dispatch_input = serde_json::json!({"agent": "impl", "prompt": "go"});
        let dispatched = crate::serve::scope_turn_context(
            turn_context(
                std::sync::Arc::clone(&state),
                std::sync::Arc::clone(&dispatch_provider)
                    as std::sync::Arc<dyn crate::provider::Provider>,
                Vec::new(),
            ),
            dispatching_tool.execute(&dispatch_input),
        )
        .await
        .unwrap();
        let handle = extract_handle(&dispatched);

        // A `SubagentTool` whose agent list no longer has "impl" —
        // reloaded, per the module doc, from the current list, not from
        // anything stored.
        let resuming_tool = SubagentTool::new(Vec::new());
        let resume_provider = ScriptedProvider::new(Vec::new());
        let resume_input = serde_json::json!({"resume": handle, "prompt": "x"});
        let err = crate::serve::scope_turn_context(
            turn_context(
                state,
                std::sync::Arc::clone(&resume_provider)
                    as std::sync::Arc<dyn crate::provider::Provider>,
                Vec::new(),
            ),
            resuming_tool.execute(&resume_input),
        )
        .await
        .unwrap_err()
        .to_string();
        assert!(err.contains("impl"), "got: {err}");
    }

    /// `SubagentCache::put` refuses an over-cap history wholesale rather
    /// than truncating it (see `persist`'s doc for why truncation is
    /// not an option). The answer must still come back normally — only
    /// the resumability marker changes — so a huge answer is never
    /// silently lost just because it made the child unresumable.
    #[tokio::test]
    async fn an_over_cap_child_still_answers_but_says_it_is_not_resumable() {
        let mut state = crate::serve::ServeState::for_test(false);
        let cache_dir = tempfile::tempdir().unwrap();
        let tiny_cache =
            crate::subagent_cache::SubagentCache::open(cache_dir.path().to_path_buf(), 64).unwrap();
        std::sync::Arc::get_mut(&mut state)
            .expect("uniquely owned immediately after construction")
            .subagent_cache = Some(tiny_cache);

        let tool = SubagentTool::new(defs());
        let big_answer = format!("the answer is {}", "x".repeat(10_000));
        let provider = ScriptedProvider::new(vec![text_response(&big_answer)]);
        let input = serde_json::json!({"agent": "reviewer", "prompt": "go"});

        let out = crate::serve::scope_turn_context(
            turn_context(
                state,
                std::sync::Arc::clone(&provider) as std::sync::Arc<dyn crate::provider::Provider>,
                Vec::new(),
            ),
            tool.execute(&input),
        )
        .await
        .unwrap();

        assert!(out.contains("not resumable"), "got: {out}");
        assert!(out.contains("the answer"), "answer was lost: {out}");
    }
}
