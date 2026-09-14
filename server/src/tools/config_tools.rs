//! The agent's own definition files, as tools (#265).
//!
//! `<workspace>/heartbeat/*.md`, `<workspace>/autonomous/*.md` and
//! `<workspace>/agents/*.md` are the files an operator hand-edits to
//! describe work the agent does on its own. One `ConfigTool`, registered
//! three times with only the directory differing, is what lets a chat
//! room read and write them.
//!
//! Two properties hold the design together:
//!
//! - **A room must be named.** `[tools.admin].rooms` is a host-layer
//!   grant, and it is checked twice: the tools are not registered at all
//!   with an empty list (`register_admin_tools`, in the task that wires
//!   `main`), and every action refuses at run time in a room that is not
//!   on it. The room comes from the `TimerOrigin::Chat` the channel path
//!   already scopes around every tool call — no new plumbing, and the
//!   transports that have no room of their own (`/rpc`, `/acp`, voice)
//!   get `None`, which is a refusal.
//! - **The write side refuses what the loader would drop.** `parse_definition`
//!   is the loader's own parse, failing instead of skipping, so a
//!   definition that would never fire cannot be written and then puzzled
//!   over.

use crate::autonomous::{CONTINUE_PROMPT, marker};
use crate::config::{Config, DEFAULT_NAMESPACE_NAME};
use crate::provider::ChatMessage;
use crate::provider::ToolSpec;
use crate::serve::{AutonomousHost, LlmTurnOutcome, ServeState, TurnStop};
use crate::tools::subagent::SubagentTool;
use crate::tools::{Tool, ToolKind, ToolSet};
use anyhow::{Context, Result, anyhow};
use async_trait::async_trait;
use sapphire_framework::workspace::WorkspaceState;
use serde_json::json;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, Weak};
use tracing::warn;

/// Which of the three definition directories a [`ConfigTool`] speaks for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConfigDir {
    Heartbeat,
    Autonomous,
    Agents,
}

impl ConfigDir {
    /// Every directory, in the order the tool names sort. Task 7 walks
    /// this to register one tool per entry.
    pub const ALL: [ConfigDir; 3] = [
        ConfigDir::Agents,
        ConfigDir::Autonomous,
        ConfigDir::Heartbeat,
    ];

    /// The directory under the workspace root that holds the definitions.
    pub fn dir_name(&self) -> &'static str {
        match self {
            ConfigDir::Heartbeat => "heartbeat",
            ConfigDir::Autonomous => "autonomous",
            ConfigDir::Agents => "agents",
        }
    }

    /// The tool name this directory is registered under.
    pub fn tool_name(&self) -> &'static str {
        match self {
            ConfigDir::Heartbeat => "heartbeat_config",
            ConfigDir::Autonomous => "autonomous_config",
            ConfigDir::Agents => "agent_config",
        }
    }

    /// Whether a definition here has an `enabled:` flag to flip.
    ///
    /// A subagent definition does not: it runs when it is called, which
    /// is a decision the parent model makes per delegation rather than an
    /// on/off switch on the file. Offering `set_enabled` there would mean
    /// inventing a flag nothing reads.
    pub fn supports_enabled(&self) -> bool {
        !matches!(self, Self::Agents)
    }
}

/// Resolve a definition name to `<workspace_root>/<dir>/<stem>.md`.
///
/// The name has to be a bare file stem. `file_write` accepts absolute
/// paths and `~`, but this does not, deliberately: the value of a tool
/// that writes the agent's own definitions is that its intent is legible
/// from the call, and `name` being one word of a file in one directory is
/// what makes it so. Anything that could name a *different* file — a
/// separator, a leading dot, a `..` — is refused rather than normalised.
pub(crate) fn definition_path(workspace_root: &Path, dir: &str, name: &str) -> Result<PathBuf> {
    let stem = name.strip_suffix(".md").unwrap_or(name);
    if stem.is_empty()
        || stem.starts_with('.')
        || stem.contains('/')
        || stem.contains('\\')
        || stem.contains("..")
    {
        anyhow::bail!(
            "{name:?} is not a definition name: pass the bare file stem, e.g. \
             \"morning_call\", with no directory and no path"
        );
    }
    Ok(workspace_root.join(dir).join(format!("{stem}.md")))
}

/// The room this tool call came from, when it came from one.
///
/// The channel path (`Agent::handle_message`) scopes `TimerOrigin::Chat`
/// around every tool call it makes, and the heartbeat's chat leg goes
/// through that same path, so a heartbeat-fired turn carries the room it
/// targets. `/rpc` and `/acp` synthesise a `room_id` out of the session
/// id and voice is `TimerOrigin::Voice`, so those get `None` — which is a
/// refusal. That is the point rather than an oversight: the allow-list
/// names places the operator knows who can write in, and those are not
/// such places.
pub(crate) fn current_call_room() -> Option<String> {
    match crate::timer::current_origin() {
        Some(crate::timer::TimerOrigin::Chat { room_id }) => Some(room_id),
        _ => None,
    }
}

/// What every action answers with when the calling room is not on
/// `[tools.admin].rooms`.
///
/// The wording names the config key on purpose: a model that is refused
/// has no other way to find out, and an operator being told by the agent
/// *which* setting to change is the whole reason the refusal is not just
/// "denied".
pub(crate) const ROOM_REFUSAL: &str = "Permission denied: the config tools are not available in this room. An operator can allow them with `[tools.admin].rooms` in the host config.";

/// The *effective* `enabled` value of a definition, as the loaders read
/// it: `enabled:` when present, `true` when absent (both loaders declare
/// `#[serde(default = "default_true")]`, so an unmentioned definition is
/// an enabled one), and `None` when there is no frontmatter at all.
///
/// Only a top-level `enabled:` counts, matching `frontmatter::set_enabled`
/// — an indented one belongs to a nested mapping such as `voice:`.
pub(crate) fn declared_enabled(raw: &str) -> Option<bool> {
    let (fm, _) = crate::frontmatter::split(raw)?;
    for line in fm.split_inclusive('\n') {
        let trimmed = line.trim_end_matches(['\n', '\r']);
        if let Some(value) = trimmed.strip_prefix("enabled:") {
            let value = value.trim();
            // `unwrap_or(true)` mirrors the loaders' default: a value
            // serde would reject leaves the file skipped by the loader,
            // where this is only a display hint.
            return Some(value.parse::<bool>().unwrap_or(true));
        }
    }
    Some(true)
}

fn lock(state: &Mutex<WorkspaceState>) -> std::sync::MutexGuard<'_, WorkspaceState> {
    state.lock().expect("WorkspaceState mutex poisoned")
}

/// `list` / `read` / `write` / `set_enabled` over one definition
/// directory, gated on the calling room and on content the loaders can
/// actually use.
pub struct ConfigTool {
    dir: ConfigDir,
    workspace_root: PathBuf,
    config: Config,
    /// The workspace's own writer, not `std::fs`, so a definition written
    /// here lands in `workspace_search`'s index like any other file —
    /// same path `memory_add` takes.
    ws: Arc<Mutex<WorkspaceState>>,
    /// The live `subagent` tool, for `Agents` only: a definition written
    /// while the agent runs is swapped into it so the next delegation can
    /// use it. `Weak` because `ToolSet` owns the tool and the tool must
    /// not own the `ToolSet` back (see `after_write`).
    subagent: Option<Weak<SubagentTool>>,
    /// The `ToolSet` this tool is registered in, for `Agents` only: the
    /// spec the model is *offered* for `subagent` lives there, so
    /// refreshing the tool without refreshing that spec would leave a
    /// definition callable but unadvertised.
    tool_set: Weak<ToolSet>,
    spec: ToolSpec,
}

impl ConfigTool {
    pub fn new(
        dir: ConfigDir,
        workspace_root: PathBuf,
        config: Config,
        ws: Arc<Mutex<WorkspaceState>>,
        subagent: Option<Weak<SubagentTool>>,
        tool_set: Weak<ToolSet>,
    ) -> Self {
        let actions: Vec<&str> = if dir.supports_enabled() {
            vec!["list", "read", "write", "set_enabled"]
        } else {
            vec!["list", "read", "write"]
        };

        let mut description = format!(
            "Read and write the agent's own {name} definitions: the Markdown files with \
             YAML frontmatter under `<workspace>/{dir}/*.md`. `list` reports each \
             definition and, for each, whether it is effectively enabled; `read` returns \
             one exactly as it is on disk; `write` creates or replaces one, refusing \
             content the agent's own loader would drop. `set_enabled` changes only the \
             `enabled:` line of an existing definition — comments, other keys and the \
             body are left byte-for-byte as they were — so prefer it over `write` when \
             all you want is to turn something on or off.",
            name = dir.tool_name(),
            dir = dir.dir_name(),
        );
        if dir == ConfigDir::Agents {
            description.push_str(
                " A subagent definition has no `enabled:` flag — it runs when it is \
                 called — so `set_enabled` is not one of this tool's actions. A \
                 definition written here is offered for delegation immediately, without \
                 restarting the agent.",
            );
        }
        description.push_str(
            " These tools are available only in the rooms an operator listed in \
             `[tools.admin].rooms`; in every other room, every action including `list` \
             is refused.",
        );

        let spec = ToolSpec {
            name: dir.tool_name().into(),
            description: description.into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": actions,
                        "description": "Which operation to perform.",
                    },
                    "name": {
                        "type": "string",
                        "description": "The definition's file stem — one word, no \
                            directory and no extension (e.g. \"morning_call\").",
                    },
                    "content": {
                        "type": "string",
                        "description": "For `write`: the whole file, frontmatter and \
                            body, which replaces any existing definition of that name.",
                    },
                    "enabled": {
                        "type": "boolean",
                        "description": "For `set_enabled`: whether the definition \
                            should be enabled.",
                    },
                },
                "required": ["action"],
            }),
        };

        Self {
            dir,
            workspace_root,
            config,
            ws,
            subagent,
            tool_set,
            spec,
        }
    }

    /// The directory this tool reads and writes.
    fn dir_path(&self) -> PathBuf {
        self.workspace_root.join(self.dir.dir_name())
    }

    /// A workspace-relative path for `WorkspaceState::write_file`.
    ///
    /// The path is built from a fixed directory name and a validated stem
    /// and so is always inside the workspace; this is the check that says
    /// so rather than an expectation that it could fail.
    fn rel(&self, abs: &Path) -> Result<PathBuf> {
        abs.strip_prefix(&self.workspace_root)
            .map(Path::to_path_buf)
            .map_err(|_| anyhow!("refusing to touch {}: outside the workspace", abs.display()))
    }

    /// The room gate. A refusal here is the only permission check these
    /// tools have — they are `ToolKind::Edit` and deliberately outside
    /// `ToolPolicy`, because the grant that matters is the host's room
    /// list, not a per-tool policy the workspace can set.
    fn gate(&self) -> Result<()> {
        if self
            .config
            .config_tools_allowed_in(current_call_room().as_deref())
        {
            Ok(())
        } else {
            Err(anyhow!(ROOM_REFUSAL))
        }
    }

    /// What this tool's actions are, for an error message. Names the
    /// missing `set_enabled` where there is not one, since a model that
    /// asked for it is exactly the reader who needs to be told why.
    fn actions_help(&self) -> String {
        if self.dir.supports_enabled() {
            format!(
                "{} accepts `list`, `read`, `write`, `set_enabled`.",
                self.dir.tool_name()
            )
        } else {
            format!(
                "{} accepts `list`, `read`, `write`. A subagent definition runs when it \
                 is called, so it has no on/off state for `set_enabled` to flip — \
                 rewrite it with `write`.",
                self.dir.tool_name()
            )
        }
    }

    fn list(&self) -> Result<String> {
        let dir = self.dir_path();
        let mut lines: Vec<String> = Vec::new();
        // Plain `std::fs`, the loader's own path, so the listing matches
        // what the agent would actually load rather than what the
        // workspace index happens to know.
        if let Ok(entries) = std::fs::read_dir(&dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.extension().and_then(|e| e.to_str()) != Some("md") {
                    continue;
                }
                let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
                    continue;
                };
                if !self.dir.supports_enabled() {
                    lines.push(stem.to_string());
                    continue;
                }
                match std::fs::read_to_string(&path)
                    .ok()
                    .and_then(|raw| declared_enabled(&raw))
                {
                    Some(true) => lines.push(format!("{stem} (enabled)")),
                    Some(false) => lines.push(format!("{stem} (disabled)")),
                    // No frontmatter: not a definition the loader loads
                    // at all, so it has no effective state to report.
                    // Still listed, since hiding a file the operator can
                    // see would be the more confusing of the two.
                    None => lines.push(stem.to_string()),
                }
            }
        }
        if lines.is_empty() {
            return Ok(format!("No {} definitions.", self.dir.dir_name()));
        }
        lines.sort();
        Ok(lines.join("\n"))
    }

    fn read(&self, input: &serde_json::Value) -> Result<String> {
        let name = input["name"].as_str().context("missing 'name'")?;
        let abs = definition_path(&self.workspace_root, self.dir.dir_name(), name)?;
        // Verbatim, not through `serde`: a definition is a file people
        // hand-edit, and the model has to see the comments that say why.
        std::fs::read_to_string(&abs)
            .with_context(|| format!("failed to read {}", self.rel(&abs).unwrap_or(abs).display()))
    }

    async fn write(&self, input: &serde_json::Value) -> Result<String> {
        let name = input["name"].as_str().context("missing 'name'")?;
        let content = input["content"].as_str().context("missing 'content'")?;
        let abs = definition_path(&self.workspace_root, self.dir.dir_name(), name)?;
        // Refuse before anything touches the disk: a definition the
        // loader would drop must not be left behind for the model to
        // find on the next `list` and believe.
        self.validate(name, content)?;
        let rel = self.rel(&abs)?;
        {
            let state = lock(&self.ws);
            state
                .write_file(&rel, content)
                .with_context(|| format!("failed to write {}", rel.display()))?;
        }
        self.after_write().await;
        Ok(format!("Wrote {}", rel.display()))
    }

    async fn set_enabled(&self, input: &serde_json::Value) -> Result<String> {
        if !self.dir.supports_enabled() {
            anyhow::bail!(
                "{} does not support the action `set_enabled`. {}",
                self.dir.tool_name(),
                self.actions_help()
            );
        }
        let name = input["name"].as_str().context("missing 'name'")?;
        let enabled = input["enabled"]
            .as_bool()
            .context("missing 'enabled' (true or false)")?;
        let abs = definition_path(&self.workspace_root, self.dir.dir_name(), name)?;
        let rel = self.rel(&abs)?;
        let raw = std::fs::read_to_string(&abs)
            .with_context(|| format!("failed to read {}", rel.display()))?;
        let Some(updated) = crate::frontmatter::set_enabled(&raw, enabled) else {
            // Not repaired by inventing a frontmatter block: the file is
            // markdown the agent does not load, and pretending otherwise
            // would be this tool writing a definition nobody asked for.
            anyhow::bail!(
                "{} has no YAML frontmatter, so there is no `enabled:` key to set",
                rel.display()
            );
        };
        {
            let state = lock(&self.ws);
            state
                .write_file(&rel, &updated)
                .with_context(|| format!("failed to write {}", rel.display()))?;
        }
        Ok(format!("Set enabled: {enabled} in {}", rel.display()))
    }

    /// Refuse content the corresponding loader would not be able to use.
    ///
    /// Each arm is the loader's own parse (`parse_definition`), plus the
    /// checks that loader applies elsewhere — a heartbeat `schedule:` the
    /// cron parser cannot read, and the subagent profile references
    /// `main` validates at startup.
    fn validate(&self, name: &str, content: &str) -> Result<()> {
        match self.dir {
            ConfigDir::Heartbeat => {
                let task = crate::heartbeat_config::parse_definition(name, content)
                    .map_err(|e| anyhow!("refusing to write {name}.md: {e}"))?;
                // A task whose cron does not parse is *kept* by the
                // loader and skipped by `next_due`, which is a different
                // failure from a file that is skipped — and the one that
                // matters on the write side, since the file would sit
                // there looking enabled and never fire.
                if task.parsed_schedule().is_none() {
                    anyhow::bail!(
                        "refusing to write {name}.md: the `schedule:` value is not a cron \
                         expression the agent can read, so the task would never fire"
                    );
                }
                Ok(())
            }
            ConfigDir::Autonomous => {
                crate::autonomous_config::parse_definition(name, content)
                    .map_err(|e| anyhow!("refusing to write {name}.md: {e}"))?;
                Ok(())
            }
            ConfigDir::Agents => {
                let def = crate::agents::parse_definition(name, content)
                    .map_err(|e| anyhow!("refusing to write {name}.md: {e}"))?;
                // The same check `main` runs at startup. Now that a
                // definition can be written while the agent is running,
                // leaving it out here would be a way around the startup
                // check — and the failure would be an agent quietly
                // running on a provider nobody chose.
                let errors = self
                    .config
                    .validate_subagent_profiles(std::slice::from_ref(&def));
                if !errors.is_empty() {
                    anyhow::bail!("refusing to write {name}.md: {}", errors.join("; "));
                }
                // A misspelled name in `tools:` is *not* refused: a
                // subagent warns about it and drops that one name, which
                // is a mistake in a line rather than a broken definition.
                Ok(())
            }
        }
    }

    /// Make a freshly written agent definition usable without a restart.
    ///
    /// Only `Agents` has state to refresh. Both back-references are
    /// `Weak` and either may be gone (the tool built for a test, or a
    /// deployment that did not wire them up); when that happens the file
    /// is still written and the change simply takes effect at the next
    /// start.
    ///
    /// The two halves go together: `set_agents` is what `execute`
    /// dispatches from, and `replace_spec` is what the model is offered.
    /// Refreshing only the first would make a definition callable but
    /// invisible.
    async fn after_write(&self) {
        if self.dir != ConfigDir::Agents {
            return;
        }
        let Some(subagent) = self.subagent.as_ref().and_then(Weak::upgrade) else {
            return;
        };
        let Some(tool_set) = self.tool_set.upgrade() else {
            return;
        };
        subagent.set_agents(crate::agents::load_agents_dir(&self.dir_path()));
        tool_set
            .replace_spec(
                crate::tools::subagent::SUBAGENT_TOOL_NAME,
                subagent.live_spec(),
            )
            .await;
    }
}

#[async_trait]
impl Tool for ConfigTool {
    fn kind(&self) -> ToolKind {
        ToolKind::Edit
    }

    fn spec(&self) -> &ToolSpec {
        &self.spec
    }

    async fn execute(&self, input: &serde_json::Value) -> Result<String> {
        // Before the action is even looked at: `list` is refused as
        // firmly as `write`, since knowing which definitions exist is
        // itself something the operator granted to a room.
        self.gate()?;
        match input["action"].as_str() {
            Some("list") => self.list(),
            Some("read") => self.read(input),
            Some("write") => self.write(input).await,
            Some("set_enabled") => self.set_enabled(input).await,
            other => anyhow::bail!(
                "{}: unknown action {:?}. {}",
                self.dir.tool_name(),
                other.unwrap_or(""),
                self.actions_help()
            ),
        }
    }
}

/// The most turns a `task_test` run may spend.
///
/// A tool whose job is to check that a task does not eat tokens must not
/// eat tokens itself, so the cap is a constant rather than a knob: a
/// definition's own `max_turns` can only ever lower it (see
/// [`capped_test_turns`]).
const MAX_TEST_TURNS: usize = 3;

/// The `room_id` a test session is created under.
///
/// **Deliberately not the task's name.** `autonomous::is_due` anchors a
/// task's cooldown on the latest session whose `room_id == task.name`, so
/// a test that claimed that name would reset the real task's cooldown —
/// merely testing a task would postpone it. The `test:` prefix keeps every
/// test session out of that query.
fn test_room_id(kind: &str, name: &str) -> String {
    format!("test:{kind}:{name}")
}

/// How many turns a requested `max_turns` means for a test run.
///
/// `Some(0)` is one turn — a zero-turn run would test nothing — and every
/// other request is capped at [`MAX_TEST_TURNS`]. `None` means "use the
/// definition's own `max_turns`"; that is signalled as `0` rather than an
/// `Option` so the caller has one number to resolve.
fn capped_test_turns(requested: Option<usize>) -> usize {
    match requested {
        Some(0) => 1,
        Some(n) => n.min(MAX_TEST_TURNS),
        None => 0,
    }
}

/// One line for why a test run ended.
///
/// `TurnStop`'s `Debug` prints `BudgetExhausted { partial_text: "..." }`,
/// which is not a sentence an operator can read in a report.
fn describe_stop(stop: &TurnStop) -> &'static str {
    match stop {
        TurnStop::Replied => "replied",
        TurnStop::ProviderError => "provider error",
        TurnStop::BudgetExhausted { .. } => "tool-round budget exhausted",
    }
}

/// Run one of the agent's own task definitions once, before it is enabled.
///
/// The run is production's in every respect but two: the session it lands
/// in is named `test:<kind>:<name>` rather than after the task, and it
/// stops after [`MAX_TEST_TURNS`]. Same `serve::run_llm_turn`, same prompt
/// assembly, same permission row (`[autonomous] origin`) — a test that ran
/// with looser permissions than production would prove nothing.
///
/// `enabled: false` is not an obstacle: it is the case this exists for.
pub struct TaskTestTool {
    state: Arc<ServeState>,
    workspace_root: PathBuf,
    spec: ToolSpec,
}

impl TaskTestTool {
    pub fn new(state: Arc<ServeState>) -> Self {
        // Derived rather than passed: the workspace root is already in the
        // state, and a second copy could disagree with it.
        let workspace_root = state.workspace.dir().to_path_buf();
        let spec = ToolSpec {
            name: "task_test".into(),
            description: "Run one of the agent's own task definitions once, in a throwaway \
                 session, to see what it does before enabling it. The task runs even while \
                 `enabled: false` — that is the point: try it first, then turn it on with \
                 `heartbeat_config` / `autonomous_config` and `set_enabled`. The run uses the \
                 same model, the same prompt assembly and the same permission row as the real \
                 thing, but its session is named `test:<kind>:<name>` instead of after the \
                 task, so it touches neither the `enabled:` flag nor the task's cooldown. \
                 `max_turns` caps an autonomous test at 3 turns; a heartbeat task is one \
                 prompt and is always exactly one turn. Delivery is not verified: a heartbeat \
                 task's `room_id:` / `voice:` targets are ignored, because what this tool \
                 checks is the prompt and how the task ends, not where a result would go. \
                 These tools are available only in the rooms an operator listed in \
                 `[tools.admin].rooms`."
                .into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "kind": {
                        "type": "string",
                        "enum": ["heartbeat", "autonomous"],
                        "description": "Which definition directory the task lives in.",
                    },
                    "name": {
                        "type": "string",
                        "description": "The definition's file stem — one word, no \
                            directory and no extension (e.g. \"journal\").",
                    },
                    "max_turns": {
                        "type": "integer",
                        "description": "For `autonomous`: how many turns the test may \
                            spend, at most 3. Omit to use the definition's own \
                            `max_turns`. Ignored for `heartbeat`, which is one prompt.",
                    },
                },
                "required": ["kind", "name"],
            }),
        };
        Self {
            state,
            workspace_root,
            spec,
        }
    }

    /// The permission row this run's turns are judged by — production's
    /// own, from `[autonomous] origin`.
    fn origin(&self) -> crate::tools::policy::Origin {
        match self.state.config.autonomous.origin {
            crate::config::AutonomousOrigin::Channel => crate::tools::policy::Origin::Channel,
            crate::config::AutonomousOrigin::Trusted => crate::tools::policy::Origin::Trusted,
        }
    }

    /// The namespace the test session lands in: the calling room's, so the
    /// report is readable where the operator already is. `None` only
    /// happens on a path the gate has already refused.
    fn namespace(&self) -> &str {
        match current_call_room() {
            Some(room) => self.state.config.namespace_for_room(&room),
            None => DEFAULT_NAMESPACE_NAME,
        }
    }

    /// A workspace-relative path, for an error message or a report.
    fn rel_display(&self, abs: &Path) -> String {
        abs.strip_prefix(&self.workspace_root)
            .unwrap_or(abs)
            .display()
            .to_string()
    }

    /// The session's own path, workspace-relative — the report's `Session:`
    /// line. Derived from the store rather than assembled from a template,
    /// so it cannot drift from the real file name.
    fn session_rel(&self, session_id: &str) -> String {
        let ns = self.namespace();
        self.state
            .autonomous_session_store
            .absolute_path_for(session_id)
            .map(|p| self.rel_display(&p))
            .unwrap_or_else(|| format!("sessions/{ns}/autonomous/{session_id}.jsonl"))
    }

    fn load_heartbeat(&self, name: &str) -> Result<crate::heartbeat_config::HeartbeatTask> {
        let abs = definition_path(&self.workspace_root, "heartbeat", name)?;
        let raw = std::fs::read_to_string(&abs)
            .with_context(|| format!("failed to read {}", self.rel_display(&abs)))?;
        crate::heartbeat_config::parse_definition(name, &raw)
            .map_err(|e| anyhow!("cannot test {name}: {e}"))
    }

    fn load_autonomous(&self, name: &str) -> Result<crate::autonomous_config::AutonomousTask> {
        let abs = definition_path(&self.workspace_root, "autonomous", name)?;
        let raw = std::fs::read_to_string(&abs)
            .with_context(|| format!("failed to read {}", self.rel_display(&abs)))?;
        crate::autonomous_config::parse_definition(name, &raw)
            .map_err(|e| anyhow!("cannot test {name}: {e}"))
    }

    /// One turn, run exactly as production runs it.
    async fn turn(&self, session_id: &str, text: String) -> LlmTurnOutcome {
        crate::serve::run_llm_turn(
            Arc::clone(&self.state),
            session_id.to_string(),
            ChatMessage::user(text),
            Arc::new(AutonomousHost {
                origin: self.origin(),
            }),
            None,
        )
        .await
    }

    /// Create the test session, under the `test:` name and the caller's
    /// namespace.
    fn open_session(&self, kind: &str, name: &str) -> Result<String> {
        self.state
            .autonomous_session_store
            .create_autonomous_session(&test_room_id(kind, name), self.namespace())
            .with_context(|| format!("failed to create a test session for {name}"))
    }

    /// Close the session, then answer with what happened.
    ///
    /// Closed *before* the report is built, and closed even when the run
    /// failed: a session left open is the one state a later `session_list`
    /// would read wrongly, as a test still in progress.
    fn finish(
        &self,
        kind: &str,
        name: &str,
        session_id: &str,
        turns: usize,
        stop: &TurnStop,
        answer: Option<String>,
    ) -> Result<String> {
        if let Err(e) = self
            .state
            .autonomous_session_store
            .close_session(session_id)
        {
            warn!("task_test: cannot close the test session for {name}: {e}");
        }
        let answer = answer.unwrap_or_else(|| "(no answer)".to_string());
        Ok(format!(
            "{kind} task '{name}' ran {turns} turn(s) ({}).\nSession: {}\nResult: {answer}",
            describe_stop(stop),
            self.session_rel(session_id),
        ))
    }

    /// A heartbeat task is one prompt, so its test is exactly one turn —
    /// a heartbeat definition has no `max_turns` to honour anyway.
    async fn run_heartbeat(&self, name: &str) -> Result<String> {
        let task = self.load_heartbeat(name)?;
        let session_id = self.open_session("heartbeat", name)?;
        // The same prefix `Heartbeat::fire_task` puts in front of the body.
        let text = format!("[Heartbeat: {name}]\n\n{}", task.body);
        let outcome = self.turn(&session_id, text).await;
        self.finish(
            "heartbeat",
            name,
            &session_id,
            1,
            &outcome.stop,
            outcome.text,
        )
    }

    /// The autonomous test: production's own turn-1 assembly, then
    /// `CONTINUE_PROMPT` — so a run that works here works there.
    async fn run_autonomous(&self, name: &str, requested: Option<usize>) -> Result<String> {
        let task = self.load_autonomous(name)?;
        // `0` is the "not requested" sentinel; the definition's own cap is
        // still bounded, so a task declaring 50 turns cannot turn a test
        // into a long run.
        let cap = match capped_test_turns(requested) {
            0 => task.max_turns.min(MAX_TEST_TURNS),
            n => n,
        };
        let session_id = self.open_session("autonomous", name)?;
        let session_rel = self.session_rel(&session_id);

        let mut turns = 0;
        let mut last: Option<String> = None;
        // Overwritten by the first turn, which always runs: a parsed task's
        // `max_turns` is at least one and `cap` is therefore at least one.
        let mut stop = TurnStop::Replied;
        while turns < cap {
            let text = if turns == 0 {
                format!("{}\nSession: {session_rel}\n\n{}", marker(name), task.body)
            } else {
                format!("{}\n{CONTINUE_PROMPT}", marker(name))
            };
            let outcome = self.turn(&session_id, text).await;
            turns += 1;
            stop = outcome.stop;
            // Kept even when this turn produced no answer, so a run whose
            // final turn broke the provider still reports what the model
            // last said.
            match outcome.text {
                Some(text) => last = Some(text),
                None => break,
            }
            if last.as_deref().is_some_and(|t| t.trim() == "DONE") {
                break;
            }
        }

        self.finish("autonomous", name, &session_id, turns, &stop, last)
    }
}

#[async_trait]
impl Tool for TaskTestTool {
    fn kind(&self) -> ToolKind {
        ToolKind::Edit
    }

    fn spec(&self) -> &ToolSpec {
        &self.spec
    }

    async fn execute(&self, input: &serde_json::Value) -> Result<String> {
        // Before the arguments are even read: the room gate every config
        // tool shares. A run is the most powerful thing on this surface —
        // it drives a whole turn — so it is refused as firmly as a write.
        if !self
            .state
            .config
            .config_tools_allowed_in(current_call_room().as_deref())
        {
            anyhow::bail!(ROOM_REFUSAL);
        }
        let kind = input["kind"]
            .as_str()
            .context("missing 'kind' (\"heartbeat\" or \"autonomous\")")?;
        let name = input["name"]
            .as_str()
            .context("missing 'name' (the definition's file stem)")?;
        match kind {
            "heartbeat" => self.run_heartbeat(name).await,
            "autonomous" => {
                // Presence and type are different questions: a `max_turns`
                // that is there but not a non-negative integer is refused
                // rather than read as "omitted", because the caller asked
                // for a cap and would otherwise silently get the
                // definition's own.
                let requested = match input.get("max_turns") {
                    None => None,
                    Some(v) if v.is_null() => None,
                    Some(v) => Some(v.as_u64().with_context(|| {
                        format!("'max_turns' must be a non-negative integer, got {v}")
                    })? as usize),
                };
                self.run_autonomous(name, requested).await
            }
            other => anyhow::bail!(
                "task_test: unknown kind {other:?}. `kind` is \"heartbeat\" or \"autonomous\"."
            ),
        }
    }
}

/// Register the four admin tools when the deployment has named a room.
///
/// Extracted from `main.rs` so the condition is testable, because the
/// condition *is* the grant: with `[tools.admin].rooms` empty, `main`
/// must register nothing at all — four tools that always refuse would
/// still tell the model they exist, and "exists but never works" is a
/// worse answer than "does not exist". The run-time gate in
/// [`ConfigTool::gate`] and [`TaskTestTool::execute`] is the second half
/// of the same grant, for the transports that have no room of their own.
///
/// Called once `serve_state` exists, since `task_test` needs it.
pub async fn register_admin_tools(
    tool_set: &Arc<ToolSet>,
    workspace_root: &Path,
    config: Config,
    ws: Arc<Mutex<WorkspaceState>>,
    subagent: Option<&Arc<SubagentTool>>,
    serve_state: Arc<ServeState>,
) {
    if !config.config_tools_enabled() {
        return;
    }
    // `agent_config` refreshes the spec `ToolSet::specs` advertises, and
    // `ToolSet` owns those specs — so it holds a `Weak` to the very set
    // it is registered into. Same shape `RefreshSystemPromptTool` uses
    // for `Agent`, and the same reason: a strong ref would be a cycle.
    let tool_set_weak = Arc::downgrade(tool_set);
    for dir in ConfigDir::ALL {
        tool_set
            .register_tool(Box::new(ConfigTool::new(
                dir,
                workspace_root.to_path_buf(),
                config.clone(),
                Arc::clone(&ws),
                if dir == ConfigDir::Agents {
                    subagent.map(Arc::downgrade)
                } else {
                    None
                },
                tool_set_weak.clone(),
            )))
            .await;
    }
    tool_set
        .register_tool(Box::new(TaskTestTool::new(serve_state)))
        .await;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::ChatResponse;
    use crate::timer::{TimerOrigin, scope_timer_origin};
    use sapphire_framework::workspace::{AppContext, Workspace, WorkspaceState};
    use serde_json::json;
    use std::future::Future;
    use std::sync::Mutex;

    /// A workspace with the three definition directories created, plus a
    /// `TempDir` the caller must hold on to for the files to survive.
    fn test_workspace() -> (tempfile::TempDir, PathBuf, Arc<Mutex<WorkspaceState>>) {
        static TEST_CTX: AppContext = AppContext::new("sapphire-agent").allow_external_paths();
        TEST_CTX.set_cache_dir(std::env::temp_dir().join("sapphire-agent-config-tools-test-cache"));
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().to_path_buf();
        std::fs::create_dir_all(root.join(".sapphire-agent")).unwrap();
        for sub in ["heartbeat", "autonomous", "agents"] {
            std::fs::create_dir_all(root.join(sub)).unwrap();
        }
        let ws = Workspace::from_root(&TEST_CTX, &root).unwrap();
        let state = Arc::new(Mutex::new(WorkspaceState::open(ws).unwrap()));
        (dir, root, state)
    }

    struct Fixture {
        _dir: tempfile::TempDir,
        root: PathBuf,
        tool: ConfigTool,
    }

    /// The tool as the deployment wires it: `[tools.admin].rooms =
    /// ["!ops:x"]`, the three directories present, no subagent / tool set
    /// back-reference (that is the agents-directory wiring, tested where
    /// it exists).
    fn fixture(dir: ConfigDir) -> Fixture {
        let (d, root, ws) = test_workspace();
        let mut config = Config::for_test();
        config.tools.admin.rooms = vec!["!ops:x".to_string()];
        let tool = ConfigTool::new(dir, root.clone(), config, ws, None, Weak::new());
        Fixture {
            _dir: d,
            root,
            tool,
        }
    }

    /// Every call goes through the same `TimerOrigin::Chat` scope the
    /// channel path uses, rather than a test-only hook: the gate is only
    /// worth testing on the path it actually reads.
    async fn in_room<F: Future>(room: &str, fut: F) -> F::Output {
        scope_timer_origin(
            TimerOrigin::Chat {
                room_id: room.to_string(),
            },
            fut,
        )
        .await
    }

    async fn call(tool: &ConfigTool, input: serde_json::Value) -> anyhow::Result<String> {
        in_room("!ops:x", tool.execute(&input)).await
    }

    #[test]
    fn definition_path_refuses_anything_but_a_bare_stem() {
        let root = Path::new("/ws");
        for bad in ["", ".", "..", "../etc/passwd", "a/b", "a\\b", ".hidden"] {
            assert!(
                definition_path(root, "heartbeat", bad).is_err(),
                "accepted {bad:?}"
            );
        }
        assert_eq!(
            definition_path(root, "heartbeat", "morning_call.md").unwrap(),
            Path::new("/ws/heartbeat/morning_call.md"),
            "a bare stem with the extension normalises to the same file"
        );
    }

    #[test]
    fn declared_enabled_defaults_to_true() {
        let body = "---\nschedule: \"0 8 * * *\"\n---\nBody\n";
        assert_eq!(declared_enabled(body), Some(true), "the loaders' default");
        assert_eq!(
            declared_enabled("---\nenabled: true\n---\nBody\n"),
            Some(true)
        );
        assert_eq!(
            declared_enabled("---\nenabled: false\n---\nBody\n"),
            Some(false)
        );
        assert_eq!(declared_enabled("# no frontmatter\n"), None);
    }

    #[tokio::test]
    async fn agent_config_has_no_set_enabled_action() {
        let f = fixture(ConfigDir::Agents);
        let err = call(
            &f.tool,
            json!({"action": "set_enabled", "name": "reviewer", "enabled": false}),
        )
        .await
        .unwrap_err()
        .to_string();
        assert!(err.contains("set_enabled"), "{err}");
        assert!(err.contains("agent_config"), "{err}");
    }

    #[tokio::test]
    async fn heartbeat_write_refuses_a_schedule_the_agent_cannot_read() {
        let f = fixture(ConfigDir::Heartbeat);
        let err = call(
            &f.tool,
            json!({
                "action": "write",
                "name": "morning",
                "content": "---\nschedule: \"every day at 8\"\n---\nWake up.\n"
            }),
        )
        .await
        .unwrap_err()
        .to_string();
        assert!(err.contains("refusing to write"), "{err}");
        assert!(
            !f.root.join("heartbeat/morning.md").exists(),
            "a refused write must not leave a file behind"
        );
    }

    #[tokio::test]
    async fn autonomous_write_refuses_an_empty_body() {
        let f = fixture(ConfigDir::Autonomous);
        let err = call(
            &f.tool,
            json!({"action": "write", "name": "journal", "content": "---\n---\n\n"}),
        )
        .await
        .unwrap_err()
        .to_string();
        assert!(err.contains("refusing to write"), "{err}");
        assert!(!f.root.join("autonomous/journal.md").exists());
    }

    #[tokio::test]
    async fn list_reports_names_and_effective_enabled() {
        let f = fixture(ConfigDir::Heartbeat);
        std::fs::write(
            f.root.join("heartbeat/off.md"),
            "---\nschedule: \"0 8 * * *\"\nenabled: false\n---\nOff.\n",
        )
        .unwrap();
        std::fs::write(
            f.root.join("heartbeat/on.md"),
            "---\nschedule: \"0 9 * * *\"\n---\nOn.\n",
        )
        .unwrap();

        let listing = call(&f.tool, json!({"action": "list"})).await.unwrap();
        assert!(listing.contains("off (disabled)"), "{listing}");
        assert!(listing.contains("on (enabled)"), "{listing}");

        // `read` hands back the file as it is, not a re-serialised form.
        let raw = call(&f.tool, json!({"action": "read", "name": "off"}))
            .await
            .unwrap();
        assert!(raw.starts_with("---\nschedule:"), "{raw}");
        assert!(raw.ends_with("Off.\n"), "{raw}");

        let empty = fixture(ConfigDir::Autonomous);
        assert_eq!(
            call(&empty.tool, json!({"action": "list"})).await.unwrap(),
            "No autonomous definitions."
        );
    }

    #[tokio::test]
    async fn set_enabled_keeps_the_rest_of_the_file() {
        let f = fixture(ConfigDir::Autonomous);
        let raw = "---\n# keep me\npriority: 10\nmax_turns: 2\n---\n\nJournal the day.\n";
        let created = call(
            &f.tool,
            json!({"action": "write", "name": "journal.md", "content": raw}),
        )
        .await
        .unwrap();
        assert!(created.contains("autonomous/journal.md"), "{created}");

        call(
            &f.tool,
            json!({"action": "set_enabled", "name": "journal", "enabled": false}),
        )
        .await
        .unwrap();

        let after = std::fs::read_to_string(f.root.join("autonomous/journal.md")).unwrap();
        assert!(after.contains("# keep me"), "{after}");
        assert!(after.contains("enabled: false"), "{after}");
        assert!(after.contains("priority: 10"), "{after}");
        assert!(after.ends_with("Journal the day.\n"), "{after}");
    }

    #[tokio::test]
    async fn set_enabled_refuses_a_file_without_frontmatter() {
        let f = fixture(ConfigDir::Heartbeat);
        let path = f.root.join("heartbeat/bare.md");
        std::fs::write(&path, "# Just markdown\n").unwrap();

        let err = call(
            &f.tool,
            json!({"action": "set_enabled", "name": "bare", "enabled": false}),
        )
        .await
        .unwrap_err()
        .to_string();
        assert!(err.contains("frontmatter"), "{err}");
        assert_eq!(
            std::fs::read_to_string(&path).unwrap(),
            "# Just markdown\n",
            "a file with nothing to flip is refused, not repaired"
        );
    }

    #[tokio::test]
    async fn every_action_is_refused_outside_an_allowed_room() {
        let f = fixture(ConfigDir::Heartbeat);
        for input in [
            json!({"action": "list"}),
            json!({"action": "read", "name": "morning"}),
            json!({
                "action": "write",
                "name": "morning",
                "content": "---\nschedule: \"0 8 * * *\"\n---\nHi\n"
            }),
            json!({"action": "set_enabled", "name": "morning", "enabled": false}),
        ] {
            let err = in_room("!random:z", f.tool.execute(&input))
                .await
                .unwrap_err()
                .to_string();
            assert!(err.contains("Permission denied"), "{input}: {err}");
        }

        // No scope at all is `/rpc`, `/acp` and voice — never a named room.
        let err = f
            .tool
            .execute(&json!({"action": "list"}))
            .await
            .unwrap_err()
            .to_string();
        assert!(err.contains("Permission denied"), "{err}");
    }

    #[tokio::test]
    async fn the_admin_tools_are_edits() {
        for dir in ConfigDir::ALL {
            let f = fixture(dir);
            assert_eq!(f.tool.kind(), ToolKind::Edit, "{:?}", dir.tool_name());
            assert_eq!(f.tool.spec().name, dir.tool_name());
        }
        assert_eq!(
            ConfigDir::ALL.map(|d| d.tool_name()),
            ["agent_config", "autonomous_config", "heartbeat_config"]
        );
    }

    /// The same shape `autonomous.rs`'s `mod tests` builds: one final
    /// answer, no tool calls.
    fn test_response(text: &str) -> ChatResponse {
        ChatResponse {
            prompt_usage: None,
            text: Some(text.to_string()),
            tool_calls: Vec::new(),
            stop_reason: None,
        }
    }

    /// A `ServeState` whose scripted provider answers with `responses`,
    /// with `!ops:x` allow-listed so the tool's room gate is passable.
    fn test_state(responses: Vec<ChatResponse>) -> Arc<ServeState> {
        let mut state = ServeState::for_test_scripted(false, responses);
        Arc::get_mut(&mut state)
            .expect("uniquely owned immediately after construction")
            .config
            .tools
            .admin
            .rooms = vec!["!ops:x".to_string()];
        state
    }

    fn test_tool(responses: Vec<ChatResponse>) -> (Arc<ServeState>, TaskTestTool) {
        let state = test_state(responses);
        let tool = TaskTestTool::new(Arc::clone(&state));
        (state, tool)
    }

    /// The invariant the whole design hangs on: `autonomous::is_due`
    /// anchors a task's cooldown on the latest session whose `room_id` is
    /// the task's name, so a test session must never claim that name.
    #[test]
    fn a_test_session_does_not_claim_the_task_name() {
        assert_eq!(
            test_room_id("autonomous", "journal"),
            "test:autonomous:journal"
        );
        assert_ne!(test_room_id("autonomous", "journal"), "journal");
        assert_eq!(
            test_room_id("heartbeat", "morning"),
            "test:heartbeat:morning"
        );
    }

    /// `enabled: false` is the case this tool exists for: the task runs
    /// anyway, on the production prompt assembly, and the session it used
    /// is closed so a later `session_list` does not show a running test.
    #[tokio::test]
    async fn a_disabled_task_can_be_tested_and_the_session_is_closed() {
        let (state, tool) = test_tool(vec![test_response("working"), test_response("DONE")]);
        let ws = state.workspace.dir().to_path_buf();
        std::fs::create_dir_all(ws.join("autonomous")).unwrap();
        std::fs::write(
            ws.join("autonomous/journal.md"),
            "---\nenabled: false\n---\nWrite the journal.\n",
        )
        .unwrap();

        let out = in_room(
            "!ops:x",
            tool.execute(&json!({"kind": "autonomous", "name": "journal"})),
        )
        .await
        .unwrap();

        assert!(
            out.contains("autonomous task 'journal' ran 2 turn(s) (replied)"),
            "{out}"
        );
        // The model's own answer is what is reported — proving the turn
        // really ran rather than merely created a session.
        assert!(out.contains("Result: DONE"), "{out}");
        assert!(
            out.contains("Session: sessions/default/autonomous/"),
            "the report should name the session, workspace-relative: {out}"
        );

        let rows = state.autonomous_session_store.session_rows();
        assert_eq!(rows.len(), 1, "one test, one session");
        assert_eq!(rows[0].meta.room_id, "test:autonomous:journal");
        assert!(rows[0].is_closed, "a finished test must not look running");

        // The definition's body went into the first user message exactly
        // as production assembles it.
        let history = state
            .autonomous_session_store
            .load_session(&rows[0].meta.session_id)
            .unwrap();
        let first = match &history[0].parts[0] {
            crate::provider::ContentPart::Text(t) => t.clone(),
            other => panic!("expected text, got {other:?}"),
        };
        assert!(first.starts_with("[Autonomous: journal]\n"), "{first}");
        assert!(first.ends_with("Write the journal.\n"), "{first}");
    }

    /// A test must never be a long run: the cap is what makes it a check
    /// on token appetite rather than a source of one.
    #[test]
    fn max_turns_is_capped_for_a_test() {
        assert_eq!(capped_test_turns(Some(50)), MAX_TEST_TURNS);
        assert_eq!(capped_test_turns(Some(MAX_TEST_TURNS)), MAX_TEST_TURNS);
        assert_eq!(capped_test_turns(Some(2)), 2);
        assert_eq!(capped_test_turns(Some(0)), 1, "zero would test nothing");
        assert_eq!(
            capped_test_turns(None),
            0,
            "`0` is the sentinel for \"use the definition's own max_turns\""
        );
    }

    /// Two answers are scripted; a one-turn run consumes only the first,
    /// which is what `Result: first` pins.
    #[tokio::test]
    async fn a_heartbeat_test_runs_exactly_one_turn() {
        let (state, tool) = test_tool(vec![test_response("first"), test_response("second")]);
        let ws = state.workspace.dir().to_path_buf();
        std::fs::create_dir_all(ws.join("heartbeat")).unwrap();
        std::fs::write(
            ws.join("heartbeat/morning.md"),
            "---\nschedule: \"0 8 * * *\"\n---\nWake up.\n",
        )
        .unwrap();

        let out = in_room(
            "!ops:x",
            tool.execute(&json!({"kind": "heartbeat", "name": "morning"})),
        )
        .await
        .unwrap();

        assert!(out.contains("1 turn(s) (replied)"), "{out}");
        assert!(out.contains("Result: first"), "{out}");
        assert!(!out.contains("second"), "a second turn was run: {out}");

        let rows = state.autonomous_session_store.session_rows();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].meta.room_id, "test:heartbeat:morning");
        assert!(rows[0].is_closed);
    }

    /// The room gate comes first — before `kind` or `name` are read — and
    /// an unknown `kind` names one of two directories or is refused.
    #[tokio::test]
    async fn task_test_is_refused_without_a_chat_room() {
        let (_state, tool) = test_tool(vec![test_response("ok")]);

        // No `TimerOrigin::Chat` scope at all is `/rpc`, `/acp` and voice.
        let err = tool
            .execute(&json!({"kind": "autonomous", "name": "journal"}))
            .await
            .unwrap_err()
            .to_string();
        assert!(err.contains("Permission denied"), "{err}");

        // A room is not the only thing that can be wrong: `kind` names one
        // of two definition directories, and anything else is refused
        // before a file is read.
        let err = in_room(
            "!ops:x",
            tool.execute(&json!({"kind": "agents", "name": "reviewer"})),
        )
        .await
        .unwrap_err()
        .to_string();
        assert!(err.contains("heartbeat"), "{err}");

        // A `max_turns` that is present but not a non-negative integer is
        // refused, not read as "omitted" — a caller that asked for a cap
        // must not silently get the definition's own.
        for bad in [json!(-1), json!("5"), json!(3.5)] {
            let err = in_room(
                "!ops:x",
                tool.execute(&json!({"kind": "autonomous", "name": "journal", "max_turns": bad})),
            )
            .await
            .unwrap_err()
            .to_string();
            assert!(err.contains("max_turns"), "{bad}: {err}");
            assert!(err.contains("non-negative integer"), "{bad}: {err}");
        }
    }
    /// A `rooms` that names nobody registers nothing — not four tools that
    /// always refuse. The difference matters: the model cannot ask for a
    /// tool it was never offered, and a refusal it can discover still tells
    /// it that this surface exists.
    #[tokio::test]
    async fn no_room_means_no_registration() {
        let (_dir, root, ws) = test_workspace();
        let serve_state = test_state(vec![test_response("ok")]);
        let set = Arc::new(ToolSet::new(Vec::new(), Vec::new()));
        let mut config = Config::for_test();
        config.tools.admin.rooms = Vec::new();
        assert!(!config.config_tools_enabled());

        register_admin_tools(&set, &root, config, ws, None, serve_state).await;

        assert!(
            set.specs_filtered(|_| true).await.is_empty(),
            "no room names one: none of the admin tools may be registered"
        );
        assert!(set.kinds().await.is_empty());
    }

    /// One room is all it takes, and it registers exactly four tools: the
    /// three `ConfigTool`s (`ConfigDir::ALL`) plus `task_test`. `kinds()`
    /// is checked for length as well as the specs are for their names, so a
    /// directory registered twice — or a spec pushed without its tool —
    /// fails here rather than at the first call.
    #[tokio::test]
    async fn one_room_registers_all_four() {
        let (_dir, root, ws) = test_workspace();
        let serve_state = test_state(vec![test_response("ok")]);
        let set = Arc::new(ToolSet::new(Vec::new(), Vec::new()));
        let mut config = Config::for_test();
        config.tools.admin.rooms = vec!["!ops:x".to_string()];

        register_admin_tools(&set, &root, config, ws, None, serve_state).await;

        let mut names: Vec<String> = set
            .specs_filtered(|_| true)
            .await
            .into_iter()
            .map(|s| s.name.to_string())
            .collect();
        names.sort();
        assert_eq!(
            names,
            [
                "agent_config",
                "autonomous_config",
                "heartbeat_config",
                "task_test",
            ]
        );

        let kinds = set.kinds().await;
        assert_eq!(kinds.len(), 4, "each name registered once: {kinds:?}");
        assert!(
            kinds.iter().all(|(_, kind)| *kind == ToolKind::Edit),
            "every admin tool is an edit: {kinds:?}"
        );
    }
}
