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

use crate::config::Config;
use crate::provider::ToolSpec;
use crate::tools::subagent::SubagentTool;
use crate::tools::{Tool, ToolKind, ToolSet};
use anyhow::{Context, Result, anyhow};
use async_trait::async_trait;
use sapphire_framework::workspace::WorkspaceState;
use serde_json::json;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, Weak};

/// Which of the three definition directories a [`ConfigTool`] speaks for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
// Consumed by `register_admin_tools` once `main` wires it up (#265, Task 7).
pub enum ConfigDir {
    Heartbeat,
    Autonomous,
    Agents,
}

#[allow(dead_code)]
// Consumed by `register_admin_tools` once `main` wires it up (#265, Task 7).
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
#[allow(dead_code)]
// Consumed by `register_admin_tools` once `main` wires it up (#265, Task 7).
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
#[allow(dead_code)]
// Consumed by `register_admin_tools` once `main` wires it up (#265, Task 7).
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
#[allow(dead_code)]
// Consumed by `register_admin_tools` once `main` wires it up (#265, Task 7).
pub(crate) const ROOM_REFUSAL: &str = "Permission denied: the config tools are not available in this room. An operator can allow them with `[tools.admin].rooms` in the host config.";

/// The *effective* `enabled` value of a definition, as the loaders read
/// it: `enabled:` when present, `true` when absent (both loaders declare
/// `#[serde(default = "default_true")]`, so an unmentioned definition is
/// an enabled one), and `None` when there is no frontmatter at all.
///
/// Only a top-level `enabled:` counts, matching `frontmatter::set_enabled`
/// — an indented one belongs to a nested mapping such as `voice:`.
#[allow(dead_code)]
// Consumed by `register_admin_tools` once `main` wires it up (#265, Task 7).
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

#[allow(dead_code)]
// Consumed by `register_admin_tools` once `main` wires it up (#265, Task 7).
fn lock(state: &Mutex<WorkspaceState>) -> std::sync::MutexGuard<'_, WorkspaceState> {
    state.lock().expect("WorkspaceState mutex poisoned")
}

/// `list` / `read` / `write` / `set_enabled` over one definition
/// directory, gated on the calling room and on content the loaders can
/// actually use.
#[allow(dead_code)]
// Consumed by `register_admin_tools` once `main` wires it up (#265, Task 7).
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

#[allow(dead_code)]
// Consumed by `register_admin_tools` once `main` wires it up (#265, Task 7).
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

#[cfg(test)]
mod tests {
    use super::*;
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
}
