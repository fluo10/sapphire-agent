//! Who may run which tool.
//!
//! One pure function, deliberately. There are two tool-calling loops in
//! this crate — `run_llm_turn` (`src/serve/mod.rs`) and the one inside
//! `Agent::handle_message` (`src/agent.rs`) — and the thing that must
//! not be duplicated between them is the *decision*. Merging the loops
//! themselves is a separate job, tracked in the design spec.
//!
//! Nothing here does I/O or knows about persistence: an `AllowAlways` is
//! recorded by the caller, not by `decide`.

use crate::tools::ToolKind;

/// The ACP session modes this agent offers.
///
/// Three, deliberately. `plan` is not "ask or don't ask" but "don't act
/// at all, produce a plan", which needs a different system prompt and a
/// way to present the plan — a separate feature.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionMode {
    Default,
    AcceptEdits,
    Bypass,
}

impl SessionMode {
    /// Every mode, in the order they are advertised to the client.
    pub const ALL: [SessionMode; 3] = [
        SessionMode::Default,
        SessionMode::AcceptEdits,
        SessionMode::Bypass,
    ];

    /// The wire id, as it appears in `session/set_mode`.
    pub fn id(self) -> &'static str {
        match self {
            SessionMode::Default => "default",
            SessionMode::AcceptEdits => "accept_edits",
            SessionMode::Bypass => "bypass",
        }
    }

    /// Human-readable name for the client's mode picker.
    pub fn name(self) -> &'static str {
        match self {
            SessionMode::Default => "Ask before editing",
            SessionMode::AcceptEdits => "Accept edits",
            SessionMode::Bypass => "Bypass permissions",
        }
    }

    /// One line under the name, saying what the mode actually changes.
    pub fn description(self) -> &'static str {
        match self {
            SessionMode::Default => "Ask before writing files or running commands.",
            SessionMode::AcceptEdits => {
                "Write files without asking; still ask before running commands."
            }
            SessionMode::Bypass => "Run everything without asking.",
        }
    }

    /// `None` for an id this agent does not implement — `plan`, notably.
    pub fn from_id(id: &str) -> Option<SessionMode> {
        SessionMode::ALL.into_iter().find(|m| m.id() == id)
    }
}

/// Which transport asked for this tool call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Origin {
    /// An editor over `/acp`, in the session's current mode.
    Acp(SessionMode),
    /// Matrix or Discord. Never asked — a channel turn is asynchronous,
    /// so blocking it on a human reply could hang for hours, and routing
    /// the question through the LLM turn would let the model broker its
    /// own permission request.
    ///
    /// The heartbeat's chat leg arrives here too, because it shares
    /// `Agent::handle_message`. That is the right answer rather than an
    /// accident: heartbeat tasks are workspace files, and `file_write`
    /// is an `Edit`, which this origin allows unasked — so a trusted
    /// heartbeat would let a chat message write itself a task that runs
    /// a command on the next tick.
    Channel,
    /// `/rpc`, voice and `/a2a`: authenticated before the turn began,
    /// with no UI to ask through. Behaviour must not change for these.
    ///
    /// `/a2a` is a peer agent rather than a local device, and a peer is
    /// promptable in ways a device is not. It sits here because it
    /// presents an operator-issued token from the same device registry
    /// as `/rpc` — the trust is in the token, not in the locality.
    Trusted,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Decision {
    Allow,
    Ask,
    Deny,
}

/// The outcome of asking. Maps 1:1 onto ACP's `PermissionOptionKind`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Approval {
    AllowOnce,
    AllowAlways,
    RejectOnce,
    RejectAlways,
}

impl Approval {
    /// Whether the call runs.
    pub fn allows(self) -> bool {
        matches!(self, Approval::AllowOnce | Approval::AllowAlways)
    }

    /// Whether this answer should outlive the call that prompted it.
    pub fn is_sticky(self) -> bool {
        matches!(self, Approval::AllowAlways | Approval::RejectAlways)
    }
}

/// Why a call was refused. The model reads the difference: one is
/// worth rephrasing around, the other is not worth retrying at all.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Refusal {
    /// A human was asked and said no.
    UserDeclined,
    /// The policy refuses it outright on this transport; nobody was asked.
    Unavailable,
}

/// Look a tool's kind up by name.
///
/// An unregistered name yields `Other`. It will come back from
/// `ToolSet::execute` as "Unknown tool" anyway, but it must not travel
/// there classified as safe on the way.
pub fn kind_of(name: &str, kinds: &[(String, ToolKind)]) -> ToolKind {
    kinds
        .iter()
        .find(|(known, _)| known == name)
        .map(|(_, kind)| *kind)
        .unwrap_or(ToolKind::Other)
}

/// What the model is told in place of a result.
///
/// One function rather than a literal at each call site: both tool
/// loops refuse calls, and two copies of this wording would drift.
pub fn refusal_message(tool: &str, why: Refusal) -> String {
    match why {
        Refusal::UserDeclined => format!(
            "Permission denied: the user declined the '{tool}' call. \
             Do not retry it without being asked to."
        ),
        Refusal::Unavailable => format!(
            "Permission denied: the '{tool}' tool is not available on this \
             transport. Try another approach, or ask the user to run it."
        ),
    }
}

/// The tools that operate on the agent's own filesystem and shell.
///
/// Listed by name rather than derived from `ToolKind`, because the
/// distinction is *which machine*, not how dangerous the operation is:
/// `memory_add` is also an `Edit`, and it is never in question.
pub const HOST_TOOLS: &[&str] = &[
    "file_read",
    "file_write",
    "file_append",
    "file_delete",
    "dir_list",
    "dir_walk",
    "shell",
];

/// Whether this call is refused before the policy table is consulted.
///
/// A gate in front of `decide` rather than a row inside it: `decide` is
/// a pure function of origin and kind, and this is a fact about the
/// deployment. Keeping them apart means the permission table still
/// reads as one thing.
pub fn host_tool_denied(name: &str, host_access_enabled: bool) -> bool {
    !host_access_enabled && HOST_TOOLS.contains(&name)
}

/// The whole policy. The table in the design spec is this function.
pub fn decide(origin: Origin, kind: ToolKind) -> Decision {
    // Group first, so that a `ToolKind` variant added upstream lands in
    // the strict bucket rather than silently becoming safe. `ToolKind`
    // is `#[non_exhaustive]`, so that is not hypothetical.
    //
    // Note what is in neither group: `Edit`, `Delete`, `Move` — the
    // middle row of the table — and `SwitchMode`, which upstream defines
    // but no tool here declares. `SwitchMode` therefore lands in that
    // middle row: asked in `default`, allowed once edits are accepted.
    // That is the right side to err on for something that changes the
    // session's own mode.
    let risky = matches!(kind, ToolKind::Execute | ToolKind::Other);
    let safe = matches!(
        kind,
        ToolKind::Read | ToolKind::Search | ToolKind::Fetch | ToolKind::Think
    );

    match origin {
        Origin::Trusted => Decision::Allow,
        Origin::Channel => {
            if risky {
                Decision::Deny
            } else {
                Decision::Allow
            }
        }
        Origin::Acp(SessionMode::Bypass) => Decision::Allow,
        Origin::Acp(_) if safe => Decision::Allow,
        Origin::Acp(SessionMode::AcceptEdits) => {
            if risky {
                Decision::Ask
            } else {
                Decision::Allow
            }
        }
        Origin::Acp(SessionMode::Default) => Decision::Ask,
    }
}

/// Split calls into the ones that may run and the ones that may not,
/// for an origin that has nobody to ask.
///
/// `Ask` is treated as a refusal, not as an allowance. Today only
/// `Origin::Channel` reaches this, and `decide` never returns `Ask` for
/// it — but a later policy change that did must not silently open the
/// channel path, so the unreachable case fails closed.
pub fn partition_without_asking(
    origin: Origin,
    calls: &[crate::provider::ToolCall],
    kinds: &[(String, ToolKind)],
) -> (Vec<crate::provider::ToolCall>, Vec<(String, String)>) {
    let mut permitted = Vec::with_capacity(calls.len());
    let mut refused = Vec::new();

    for call in calls {
        match decide(origin, kind_of(&call.name, kinds)) {
            Decision::Allow => permitted.push(call.clone()),
            Decision::Deny | Decision::Ask => refused.push((
                call.id.clone(),
                refusal_message(&call.name, Refusal::Unavailable),
            )),
        }
    }

    (permitted, refused)
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAFE: [ToolKind; 4] = [
        ToolKind::Read,
        ToolKind::Search,
        ToolKind::Fetch,
        ToolKind::Think,
    ];
    const EDITING: [ToolKind; 3] = [ToolKind::Edit, ToolKind::Delete, ToolKind::Move];
    const RISKY: [ToolKind; 2] = [ToolKind::Execute, ToolKind::Other];

    const EVERY_ORIGIN: [Origin; 5] = [
        Origin::Acp(SessionMode::Default),
        Origin::Acp(SessionMode::AcceptEdits),
        Origin::Acp(SessionMode::Bypass),
        Origin::Channel,
        Origin::Trusted,
    ];

    fn call(id: &str, name: &str) -> crate::provider::ToolCall {
        crate::provider::ToolCall {
            id: id.to_string(),
            name: name.to_string(),
            input: serde_json::json!({}),
        }
    }

    #[test]
    fn safe_kinds_never_ask_on_any_origin() {
        for kind in SAFE {
            for origin in EVERY_ORIGIN {
                assert_eq!(decide(origin, kind), Decision::Allow, "{origin:?} {kind:?}");
            }
        }
    }

    #[test]
    fn editing_asks_only_in_acp_default() {
        for kind in EDITING {
            assert_eq!(
                decide(Origin::Acp(SessionMode::Default), kind),
                Decision::Ask
            );
            assert_eq!(
                decide(Origin::Acp(SessionMode::AcceptEdits), kind),
                Decision::Allow
            );
            assert_eq!(
                decide(Origin::Acp(SessionMode::Bypass), kind),
                Decision::Allow
            );
            assert_eq!(decide(Origin::Channel, kind), Decision::Allow);
            assert_eq!(decide(Origin::Trusted, kind), Decision::Allow);
        }
    }

    /// `shell` and every MCP tool live here. A channel must never reach
    /// them — that is the one behavioural change this feature makes to
    /// an existing transport.
    #[test]
    fn risky_kinds_are_refused_on_channels_and_asked_in_acp() {
        for kind in RISKY {
            assert_eq!(
                decide(Origin::Acp(SessionMode::Default), kind),
                Decision::Ask
            );
            assert_eq!(
                decide(Origin::Acp(SessionMode::AcceptEdits), kind),
                Decision::Ask
            );
            assert_eq!(
                decide(Origin::Acp(SessionMode::Bypass), kind),
                Decision::Allow
            );
            assert_eq!(decide(Origin::Channel, kind), Decision::Deny);
            assert_eq!(decide(Origin::Trusted, kind), Decision::Allow);
        }
    }

    /// `Trusted` is `/rpc`, voice, the heartbeat and `/a2a` — paths that
    /// were already authenticated and have no UI to ask through. Nothing
    /// they can call may change behaviour.
    #[test]
    fn trusted_allows_everything() {
        for kind in SAFE.iter().chain(&EDITING).chain(&RISKY) {
            assert_eq!(decide(Origin::Trusted, *kind), Decision::Allow);
        }
    }

    /// `SwitchMode` is declared upstream and by no tool here, so it is
    /// in neither the safe nor the risky set. `decide`'s doc comment
    /// claims it therefore lands in the middle row; this is the only
    /// thing stopping a later edit from quietly moving it. Without this,
    /// pulling `SwitchMode` into `risky` would compile and pass.
    #[test]
    fn an_unclassified_kind_lands_in_the_middle_row() {
        let kind = ToolKind::SwitchMode;
        assert_eq!(
            decide(Origin::Acp(SessionMode::Default), kind),
            Decision::Ask
        );
        assert_eq!(
            decide(Origin::Acp(SessionMode::AcceptEdits), kind),
            Decision::Allow
        );
        assert_eq!(
            decide(Origin::Acp(SessionMode::Bypass), kind),
            Decision::Allow
        );
        assert_eq!(decide(Origin::Channel, kind), Decision::Allow);
        assert_eq!(decide(Origin::Trusted, kind), Decision::Allow);
    }

    #[test]
    fn mode_ids_round_trip() {
        for mode in SessionMode::ALL {
            assert_eq!(SessionMode::from_id(mode.id()), Some(mode));
        }
        // `plan` is deliberately not implemented; it must not silently
        // resolve to some other mode.
        assert_eq!(SessionMode::from_id("plan"), None);
        assert_eq!(SessionMode::ALL.len(), 3);
    }

    #[test]
    fn only_always_variants_are_sticky() {
        assert!(!Approval::AllowOnce.is_sticky());
        assert!(Approval::AllowAlways.is_sticky());
        assert!(!Approval::RejectOnce.is_sticky());
        assert!(Approval::RejectAlways.is_sticky());

        assert!(Approval::AllowOnce.allows());
        assert!(Approval::AllowAlways.allows());
        assert!(!Approval::RejectOnce.allows());
        assert!(!Approval::RejectAlways.allows());
    }

    /// An unregistered name must not be treated as safe on its way to
    /// `ToolSet::execute`'s "Unknown tool" reply.
    #[test]
    fn an_unknown_tool_name_is_other() {
        let kinds = vec![("file_read".to_string(), ToolKind::Read)];
        assert_eq!(kind_of("file_read", &kinds), ToolKind::Read);
        assert_eq!(kind_of("no_such_tool", &kinds), ToolKind::Other);
    }

    /// Both refusal reasons name the tool, so the model can tell which
    /// of several calls was refused, and say why.
    #[test]
    fn refusal_messages_name_the_tool_and_the_reason() {
        let declined = refusal_message("shell", Refusal::UserDeclined);
        assert!(declined.contains("shell"), "got {declined}");
        assert!(declined.contains("declined"), "got {declined}");

        let unavailable = refusal_message("shell", Refusal::Unavailable);
        assert!(unavailable.contains("shell"), "got {unavailable}");
        assert_ne!(
            declined, unavailable,
            "the model should be able to tell a refusal from an unavailability"
        );
    }

    /// The channel path's whole gate, in one call: safe calls survive,
    /// risky ones are dropped and come back as refusals that still name
    /// their call id.
    #[test]
    fn partition_drops_risky_calls_and_reports_them() {
        let kinds = vec![
            ("file_read".to_string(), ToolKind::Read),
            ("shell".to_string(), ToolKind::Execute),
            ("mcp__x__y".to_string(), ToolKind::Other),
        ];
        let calls = vec![
            call("c1", "file_read"),
            call("c2", "shell"),
            call("c3", "mcp__x__y"),
        ];

        let (permitted, refused) = partition_without_asking(Origin::Channel, &calls, &kinds);

        let kept: Vec<&str> = permitted.iter().map(|c| c.id.as_str()).collect();
        assert_eq!(kept, vec!["c1"]);

        let refused_ids: Vec<&str> = refused.iter().map(|(id, _)| id.as_str()).collect();
        assert_eq!(refused_ids, vec!["c2", "c3"]);
        assert!(refused[0].1.contains("shell"), "got {}", refused[0].1);
    }

    /// The seven tools that touch the agent's own machine. Off unless
    /// the operator turned them on — including for `Origin::Trusted`,
    /// which is the voice pipeline and the heartbeat.
    #[test]
    fn host_tools_are_denied_when_host_access_is_off() {
        for name in HOST_TOOLS {
            assert!(
                host_tool_denied(name, false),
                "{name} must be denied with host access off"
            );
        }
    }

    #[test]
    fn host_tools_are_allowed_through_when_host_access_is_on() {
        for name in HOST_TOOLS {
            assert!(
                !host_tool_denied(name, true),
                "{name} must fall through to the policy table when enabled"
            );
        }
    }

    /// The gate is about *which machine*, not about the tool's risk, so
    /// a workspace-scoped tool is never caught by it.
    #[test]
    fn workspace_tools_are_not_host_tools() {
        for name in ["memory_add", "workspace_search", "timer_set", "web_search"] {
            assert!(!host_tool_denied(name, false), "{name} is not a host tool");
        }
    }

    /// The hole this closes: `file_delete` is `Delete`, which
    /// `Origin::Channel` allows unasked, so a Discord message can
    /// delete a file on the agent's host today.
    #[test]
    fn a_channel_turn_cannot_reach_file_delete_with_host_access_off() {
        assert_eq!(
            decide(Origin::Channel, ToolKind::Delete),
            Decision::Allow,
            "the policy table alone still allows it — which is the point"
        );
        assert!(
            host_tool_denied("file_delete", false),
            "the host gate is what stops it"
        );
    }

    /// A trusted origin refuses nothing, so the helper is a no-op there.
    #[test]
    fn partition_keeps_everything_for_a_trusted_origin() {
        let kinds = vec![("shell".to_string(), ToolKind::Execute)];
        let calls = vec![call("c1", "shell")];

        let (permitted, refused) = partition_without_asking(Origin::Trusted, &calls, &kinds);

        assert_eq!(permitted.len(), 1);
        assert!(refused.is_empty());
    }

    /// `Ask` cannot be honoured without a human, so an origin that
    /// somehow produces one is refused rather than waved through. This
    /// is what keeps a future policy change from silently opening the
    /// channel path.
    #[test]
    fn partition_refuses_rather_than_allows_an_ask() {
        let kinds = vec![("shell".to_string(), ToolKind::Execute)];
        let calls = vec![call("c1", "shell")];

        let (permitted, refused) =
            partition_without_asking(Origin::Acp(SessionMode::Default), &calls, &kinds);

        assert!(permitted.is_empty(), "an Ask must not be treated as Allow");
        assert_eq!(refused.len(), 1);
    }
}
