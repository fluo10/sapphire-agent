use anyhow::{Context, Result};
use sapphire_workspace::SyncConfig;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Config {
    /// Matrix channel configuration. Required if `discord` is not set.
    #[serde(default)]
    pub matrix: Option<MatrixConfig>,
    /// Discord channel configuration. Required if `matrix` is not set.
    #[serde(default)]
    pub discord: Option<DiscordConfig>,
    pub anthropic: AnthropicConfig,
    /// Context compression configuration.
    #[serde(default)]
    pub compression: CompressionConfig,
    /// Tool configuration (search APIs, etc.).
    #[serde(default)]
    pub tools: ToolsConfig,
    /// HTTP API server configuration.
    #[serde(default)]
    pub serve: Option<ServeConfig>,
    /// Directory containing AGENT.md and MEMORY.md.
    /// Defaults to the config file's parent directory.
    pub workspace_dir: Option<String>,
    /// Directory for persisted JSONL sessions.
    /// Defaults to `<workspace_dir>/sessions`.
    pub sessions_dir: Option<String>,
    /// Hour (0–23, local time) at which a new "day" begins.
    /// Used for session resets and daily log generation. Default: 0 (midnight).
    #[serde(default)]
    pub day_boundary_hour: u8,
    /// Default session policy applied at the day boundary. Can be overridden
    /// per room via `[rooms."<id>"]`. Default: `reset` (back-compat).
    #[serde(default)]
    pub session_policy: SessionPolicy,
    /// Per-room overrides keyed by `room_id`.
    #[serde(default)]
    pub rooms: HashMap<String, RoomConfig>,
    /// Additional LLM providers beyond the built-in `anthropic` one.
    /// Keyed by user-chosen name (e.g. `"local"`, `"openai"`).
    #[serde(default)]
    pub providers: HashMap<String, ProviderConfig>,
    /// Named profiles that bind a use-case (e.g. `"default"`, `"nsfw"`)
    /// to a provider name. Rooms select a profile via `RoomConfig.profile`;
    /// the API can also pass a profile name on a per-request basis.
    #[serde(default)]
    pub profiles: HashMap<String, ProfileConfig>,
    /// Memory namespaces. Each namespace owns its own subtree under
    /// `memory/<namespace>/` (daily/weekly/monthly/yearly logs and
    /// MEMORY.md). Profiles pin their writes to one namespace, and
    /// rooms reading the system prompt also pull in the parent
    /// namespaces declared via `include`.
    ///
    /// The `"default"` namespace is implicitly present (with `include = []`)
    /// even when no `[memory_namespace.*]` block is configured, so that
    /// every config has a valid root.
    #[serde(default, rename = "memory_namespace")]
    pub memory_namespaces: HashMap<String, MemoryNamespaceConfig>,
    /// Whether to generate a daily log at the day boundary. Default: true.
    #[serde(default = "default_true")]
    pub daily_log_enabled: bool,
    /// Whether to compact MEMORY.md at the day boundary. Default: true.
    #[serde(default = "default_true")]
    pub memory_compaction_enabled: bool,
    /// Whether to enable heartbeat (day-boundary + cron) tasks. Default: true.
    /// Set to false in test environments to avoid duplicate heartbeat tasks
    /// when both test and production instances share the same config.
    #[serde(default = "default_true")]
    pub heartbeat_enabled: bool,
    /// Cold-standby mode: only perform git sync, skip channel listening and
    /// heartbeat tasks. Useful for maintaining a backup node that stays in
    /// sync without actively processing messages. Default: false.
    #[serde(default)]
    pub standby_mode: bool,
    /// Workspace sync configuration.
    ///
    /// The workspace-level config (`{workspace_dir}/.sapphire-agent/config.toml`)
    /// provides shared defaults. This per-user `[sync]` section, when present,
    /// takes precedence — allowing each user to override the workspace defaults.
    #[serde(default)]
    pub sync: Option<SyncConfig>,
    /// How often the agent runs the periodic workspace sync cycle, in
    /// minutes. Unset or `0` disables periodic sync entirely. Each tick
    /// runs `WorkspaceState::periodic_sync`, which does a git sync **and**
    /// an mtime-based refresh of the retrieve cache — one cadence drives
    /// both.
    ///
    /// Lives at the config root (not inside `[sync]`) because the cadence
    /// spans both `sapphire-sync` and `sapphire-retrieve`; nesting it
    /// under `[sync]` would have implied a sync-only knob and forced a
    /// duplicate for the retrieve side. Upstream relocated it out of
    /// `SyncConfig` for the same reason in sapphire-workspace 0.10.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sync_interval_minutes: Option<u32>,
    /// Periodic log digest configuration (weekly / monthly / yearly).
    #[serde(default)]
    pub digest: DigestConfig,
}

fn default_true() -> bool {
    true
}

/// Action taken at the day boundary for a given conversation.
///
/// - `Reset`: close the session and clear in-memory caches (legacy behavior).
///   The next message starts a fresh session; prior-run summary is injected via
///   `restart_summaries`.
/// - `Compact`: keep the same session alive, but force-summarize the current
///   in-memory history and replace it with a summary stub. The SummaryLine is
///   appended to the session JSONL. Session continuity is preserved.
/// - `None`: no day-boundary action.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum SessionPolicy {
    #[default]
    Reset,
    Compact,
    None,
}

/// Per-room configuration overrides.
#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct RoomConfig {
    /// Override the day-boundary session policy for this room.
    pub session_policy: Option<SessionPolicy>,
    /// Profile name to use for this room. If unset, falls back to the
    /// `"default"` profile when defined, otherwise the built-in
    /// `anthropic` provider.
    pub profile: Option<String>,
}

/// Definition of an additional LLM provider.
///
/// Tagged by `type` to allow future provider kinds. Currently only
/// `openai_compatible` is supported.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "type")]
pub enum ProviderConfig {
    /// llama.cpp `llama-server`, OpenAI proper, Ollama, vLLM, etc.
    #[serde(rename = "openai_compatible")]
    OpenAiCompatible(crate::provider::openai_compatible::OpenAICompatibleConfig),
}

/// Definition of a named profile.
///
/// A profile picks a provider (and optionally a fallback) for a given
/// use-case. The `"default"` profile, if defined, is used by rooms that
/// don't specify their own profile.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ProfileConfig {
    /// Name of the provider to use. Either `"anthropic"` (built-in) or a
    /// key from the top-level `[providers]` table.
    pub provider: String,
    /// Optional fallback provider used when the primary refuses a request
    /// (e.g. NSFW content). Wired up by the routing layer.
    #[serde(default)]
    pub fallback_provider: Option<String>,
    /// Memory namespace this profile reads and writes. Defaults to
    /// `"default"`. Must reference a defined `[memory_namespace.<name>]`
    /// (or the implicit `"default"`).
    #[serde(default)]
    pub memory_namespace: Option<String>,
}

/// Definition of a memory namespace — a subtree under `memory/<name>/`
/// that owns its own MEMORY.md and periodic logs. The `include` list
/// names parent namespaces whose memory should also be visible to
/// rooms using this namespace; reads chain through the include DAG,
/// writes go only to the leaf namespace.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct MemoryNamespaceConfig {
    /// Names of parent namespaces whose memory should be merged in
    /// when assembling the system prompt for this namespace. Forms a
    /// DAG; cycles are rejected at startup.
    #[serde(default)]
    pub include: Vec<String>,
}

/// Built-in name of the Anthropic provider — referenced by profiles.
pub const ANTHROPIC_PROVIDER_NAME: &str = "anthropic";

/// Conventional name of the default profile.
pub const DEFAULT_PROFILE_NAME: &str = "default";

/// Conventional name of the profile used by background tasks (daily-log,
/// memory compaction, periodic digests). When this profile is defined the
/// background tasks honour its `provider` and `fallback_provider`; when
/// it isn't, those tasks run on the built-in Anthropic provider with no
/// fallback.
pub const BACKGROUND_PROFILE_NAME: &str = "background";

/// Implicit name of the root memory namespace. Always present, even when
/// no `[memory_namespace.*]` block is configured — backstop so every
/// profile / room resolves to a valid namespace.
pub const DEFAULT_NAMESPACE_NAME: &str = "default";

/// Configuration for the HTTP API server (serve command).
#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct ServeConfig {
    #[serde(default = "default_serve_host")]
    pub host: String,
    #[serde(default = "default_serve_port")]
    pub port: u16,
}

fn default_serve_host() -> String {
    "127.0.0.1".to_string()
}

fn default_serve_port() -> u16 {
    9000
}

/// Configuration for built-in tools.
#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct ToolsConfig {
    /// Tavily API key for `web_search`. If absent the tool is not registered.
    pub tavily_api_key: Option<String>,
    /// External MCP servers to connect to. Each server's tools are registered
    /// with the naming convention `mcp__<name>__<tool_name>`.
    #[serde(default)]
    pub mcp_servers: Vec<McpServerConfig>,
}

/// Configuration for a single external MCP server.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct McpServerConfig {
    /// Human-readable name (used in tool prefix: `mcp__<name>__<tool>`).
    pub name: String,
    /// Transport configuration.
    #[serde(flatten)]
    pub transport: McpTransportConfig,
}

/// Transport configuration for connecting to an MCP server.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "type")]
pub enum McpTransportConfig {
    /// Streamable HTTP transport.
    #[serde(rename = "http")]
    Http {
        /// Server URL (e.g. `http://localhost:3000/mcp`).
        url: String,
        /// Optional API key / bearer token.
        #[serde(default)]
        api_key: Option<String>,
    },
    /// stdio transport — spawn a child process and communicate via stdin/stdout.
    #[serde(rename = "stdio")]
    Stdio {
        /// Command to execute (e.g. `"npx"`, `"uvx"`, `"/path/to/server"`).
        command: String,
        /// Command arguments.
        #[serde(default)]
        args: Vec<String>,
        /// Additional environment variables passed to the child process.
        #[serde(default)]
        env: std::collections::HashMap<String, String>,
    },
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct MatrixConfig {
    pub homeserver: String,
    pub access_token: String,
    pub user_id: String,
    pub device_id: String,
    /// Rooms the bot listens to. Accepts either a TOML array
    /// (`room_ids = ["!a:srv", "!b:srv"]`) or — for backward compatibility —
    /// a single string key named `room_id`.
    #[serde(default, alias = "room_id", deserialize_with = "deserialize_room_ids")]
    pub room_ids: Vec<String>,
    #[serde(default)]
    pub allowed_users: Vec<String>,
    /// E2EE recovery key (optional)
    pub recovery_key: Option<String>,
    /// Directory for matrix-sdk state/crypto store. Defaults to
    /// `~/.local/share/sapphire-agent/matrix`.
    pub state_dir: Option<String>,
}

/// Accept either `"!a:srv"` (legacy single string) or `["!a:srv", "!b:srv"]`
/// for the `room_ids` / legacy `room_id` field.
fn deserialize_room_ids<'de, D>(deserializer: D) -> std::result::Result<Vec<String>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum OneOrMany {
        One(String),
        Many(Vec<String>),
    }
    match OneOrMany::deserialize(deserializer)? {
        OneOrMany::One(s) => Ok(vec![s]),
        OneOrMany::Many(v) => Ok(v),
    }
}

impl MatrixConfig {
    /// Primary room — first configured room. Used as the default target for
    /// heartbeat tasks that don't name a specific room.
    pub fn primary_room_id(&self) -> Option<&str> {
        self.room_ids.first().map(|s| s.as_str())
    }

    pub fn resolved_state_dir(&self) -> PathBuf {
        if let Some(dir) = &self.state_dir {
            PathBuf::from(shellexpand::tilde(dir).as_ref())
        } else if let Some(dirs) = directories::ProjectDirs::from("", "", "sapphire-agent") {
            dirs.data_local_dir().join("matrix")
        } else {
            PathBuf::from(".sapphire-agent/matrix")
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct DiscordConfig {
    pub bot_token: String,
    /// Text channel IDs the bot listens to. Empty = all channels the bot can see.
    #[serde(default)]
    pub channel_ids: Vec<String>,
    /// Discord user IDs allowed to interact. Empty = all users.
    #[serde(default)]
    pub allowed_users: Vec<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct AnthropicConfig {
    pub api_key: String,
    #[serde(default = "default_model")]
    pub model: String,
    /// Cheaper model for casual (non-coding) conversations.
    /// If set, the agent uses this model by default and switches to `model`
    /// when the message appears to be coding-related.
    pub light_model: Option<String>,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,
    pub system_prompt: Option<String>,
}

/// Context compression configuration (provider-agnostic).
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct CompressionConfig {
    /// Whether context compression is enabled. Default: true.
    #[serde(default = "default_true")]
    pub enabled: bool,
    /// Context window size in tokens. Defaults to 200,000.
    #[serde(default = "default_context_window")]
    pub context_window: usize,
    /// Fraction of context window at which compression triggers (0.0–1.0).
    /// Defaults to 0.80.
    #[serde(default = "default_compression_threshold")]
    pub threshold: f64,
    /// Number of recent messages to preserve verbatim during compression.
    /// Defaults to 20.
    #[serde(default = "default_preserve_recent")]
    pub preserve_recent: usize,
}

impl Default for CompressionConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            context_window: default_context_window(),
            threshold: default_compression_threshold(),
            preserve_recent: default_preserve_recent(),
        }
    }
}

fn default_model() -> String {
    "claude-opus-4-6".to_string()
}

fn default_max_tokens() -> u32 {
    8192
}

fn default_context_window() -> usize {
    200_000
}

fn default_compression_threshold() -> f64 {
    0.80
}

fn default_preserve_recent() -> usize {
    20
}

// ---------------------------------------------------------------------------
// Digest config
// ---------------------------------------------------------------------------

/// Frontmatter-digest injection & generation config.
///
/// At each day boundary the agent generates weekly, monthly, and yearly log
/// files under `memory/{weekly,monthly,yearly}/`. Each file carries a YAML
/// `digest:` array of importance-ordered bullets. The top-N items per file
/// are injected into the system prompt so the agent retains long-horizon
/// context without paying full-body token cost.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct DigestConfig {
    /// Top-N digest items injected per daily log (used for "This Week's
    /// Digests" — days before yesterday within the current ISO week).
    #[serde(default = "default_digest_daily_items")]
    pub daily_items: usize,
    /// Top-N items injected per weekly log (used for "This Month's Digests").
    #[serde(default = "default_digest_weekly_items")]
    pub weekly_items: usize,
    /// Top-N items injected per monthly log (used for "This Year's Digests").
    #[serde(default = "default_digest_monthly_items")]
    pub monthly_items: usize,
    /// Top-N items injected per yearly log (used for "Past Years' Digests").
    #[serde(default = "default_digest_yearly_items")]
    pub yearly_items: usize,
    /// Generate a weekly log at each Monday day-boundary. Default: true.
    #[serde(default = "default_true")]
    pub weekly_enabled: bool,
    /// Generate a monthly log on the 1st of each month. Default: true.
    #[serde(default = "default_true")]
    pub monthly_enabled: bool,
    /// Generate a yearly log on Jan 1. Default: true.
    #[serde(default = "default_true")]
    pub yearly_enabled: bool,
}

impl Default for DigestConfig {
    fn default() -> Self {
        Self {
            daily_items: default_digest_daily_items(),
            weekly_items: default_digest_weekly_items(),
            monthly_items: default_digest_monthly_items(),
            yearly_items: default_digest_yearly_items(),
            weekly_enabled: true,
            monthly_enabled: true,
            yearly_enabled: true,
        }
    }
}

fn default_digest_daily_items() -> usize {
    3
}

fn default_digest_weekly_items() -> usize {
    3
}

fn default_digest_monthly_items() -> usize {
    5
}

fn default_digest_yearly_items() -> usize {
    5
}

impl Config {
    pub fn load(path: &Path) -> Result<Self> {
        let content = std::fs::read_to_string(path)
            .with_context(|| format!("Failed to read config file: {}", path.display()))?;
        let config: Config =
            toml::from_str(&content).with_context(|| "Failed to parse config file")?;
        Ok(config)
    }

    /// Resolve the workspace directory: explicit config > config file's parent directory.
    pub fn resolved_workspace_dir(&self, config_path: &Path) -> PathBuf {
        if let Some(dir) = &self.workspace_dir {
            PathBuf::from(shellexpand::tilde(dir).as_ref())
        } else {
            config_path
                .parent()
                .unwrap_or_else(|| Path::new("."))
                .to_path_buf()
        }
    }

    /// Resolve the sessions directory for JSONL persistence.
    ///
    /// Explicit config value > `<workspace_dir>/sessions` (default).
    pub fn resolved_sessions_dir(&self, workspace_dir: &Path) -> PathBuf {
        if let Some(dir) = &self.sessions_dir {
            PathBuf::from(shellexpand::tilde(dir).as_ref())
        } else {
            workspace_dir.join("sessions")
        }
    }

    /// Resolve the session policy for a given `room_id`, falling back to the
    /// global default when no room-specific override is set.
    pub fn session_policy_for(&self, room_id: &str) -> SessionPolicy {
        self.rooms
            .get(room_id)
            .and_then(|r| r.session_policy)
            .unwrap_or(self.session_policy)
    }

    /// Resolve the profile name for a given `room_id`.
    ///
    /// Order: explicit room override > `"default"` profile if defined >
    /// `None` (caller should fall back to the built-in anthropic provider).
    pub fn profile_for(&self, room_id: &str) -> Option<&str> {
        if let Some(name) = self
            .rooms
            .get(room_id)
            .and_then(|r| r.profile.as_deref())
        {
            return Some(name);
        }
        if self.profiles.contains_key(DEFAULT_PROFILE_NAME) {
            return Some(DEFAULT_PROFILE_NAME);
        }
        None
    }

    /// Resolve a profile name to its primary provider name.
    ///
    /// Returns `None` if the profile is not defined. Caller is expected to
    /// fall back to the built-in anthropic provider in that case.
    pub fn provider_for_profile(&self, profile_name: &str) -> Option<&str> {
        self.profiles
            .get(profile_name)
            .map(|p| p.provider.as_str())
    }

    /// Validate that every profile points to a known provider, and that
    /// every room's `profile` references a defined profile. Returns
    /// human-readable error messages for each issue found.
    pub fn validate_profiles(&self) -> Vec<String> {
        let mut errors = Vec::new();
        let known_provider = |name: &str| -> bool {
            name == ANTHROPIC_PROVIDER_NAME || self.providers.contains_key(name)
        };
        for (pname, prof) in &self.profiles {
            if !known_provider(&prof.provider) {
                errors.push(format!(
                    "profile '{pname}' references unknown provider '{}'",
                    prof.provider
                ));
            }
            if let Some(fb) = &prof.fallback_provider {
                if !known_provider(fb) {
                    errors.push(format!(
                        "profile '{pname}' references unknown fallback_provider '{fb}'"
                    ));
                }
            }
        }
        for (rid, rcfg) in &self.rooms {
            if let Some(pname) = &rcfg.profile {
                if !self.profiles.contains_key(pname) {
                    errors.push(format!(
                        "room '{rid}' references unknown profile '{pname}'"
                    ));
                }
            }
        }
        // Memory namespace references on profiles.
        for (pname, prof) in &self.profiles {
            if let Some(ns) = &prof.memory_namespace {
                if !self.namespace_is_defined(ns) {
                    errors.push(format!(
                        "profile '{pname}' references unknown memory_namespace '{ns}'"
                    ));
                }
            }
        }
        // Memory namespace include references and cycle detection.
        for (ns_name, ns_cfg) in &self.memory_namespaces {
            for parent in &ns_cfg.include {
                if !self.namespace_is_defined(parent) {
                    errors.push(format!(
                        "memory_namespace '{ns_name}' includes unknown namespace '{parent}'"
                    ));
                }
            }
        }
        for ns_name in self.memory_namespaces.keys() {
            if let Some(cycle) = self.namespace_cycle_starting_at(ns_name) {
                errors.push(format!(
                    "memory_namespace cycle detected: {}",
                    cycle.join(" -> ")
                ));
            }
        }
        errors
    }

    /// True if `name` is either the implicit `"default"` namespace or has a
    /// `[memory_namespace.<name>]` block.
    fn namespace_is_defined(&self, name: &str) -> bool {
        name == DEFAULT_NAMESPACE_NAME || self.memory_namespaces.contains_key(name)
    }

    /// DFS from `start` looking for back-edges. Returns the cyclic path
    /// (start -> ... -> start) on detection, otherwise `None`.
    fn namespace_cycle_starting_at(&self, start: &str) -> Option<Vec<String>> {
        let mut stack: Vec<String> = vec![start.to_string()];
        let mut on_stack = std::collections::HashSet::new();
        on_stack.insert(start.to_string());

        fn dfs(
            cfg: &Config,
            node: &str,
            stack: &mut Vec<String>,
            on_stack: &mut std::collections::HashSet<String>,
        ) -> Option<Vec<String>> {
            let parents: Vec<String> = cfg
                .memory_namespaces
                .get(node)
                .map(|c| c.include.clone())
                .unwrap_or_default();
            for parent in parents {
                if on_stack.contains(&parent) {
                    let mut cycle: Vec<String> = stack.iter().cloned().collect();
                    cycle.push(parent);
                    return Some(cycle);
                }
                stack.push(parent.clone());
                on_stack.insert(parent.clone());
                if let Some(c) = dfs(cfg, &parent, stack, on_stack) {
                    return Some(c);
                }
                stack.pop();
                on_stack.remove(&parent);
            }
            None
        }

        dfs(self, start, &mut stack, &mut on_stack)
    }

    /// Resolve `name` to its include-chain in DFS pre-order: the namespace
    /// itself first, then each parent in include order, with parents'
    /// parents flattened in. Duplicates are removed (first occurrence
    /// wins). The implicit `"default"` namespace, when not configured,
    /// resolves to a single-entry chain `["default"]`.
    pub fn resolve_namespace_chain(&self, name: &str) -> Vec<String> {
        let mut out: Vec<String> = Vec::new();
        let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
        self.namespace_chain_walk(name, &mut out, &mut seen);
        out
    }

    fn namespace_chain_walk(
        &self,
        name: &str,
        out: &mut Vec<String>,
        seen: &mut std::collections::HashSet<String>,
    ) {
        if !seen.insert(name.to_string()) {
            return;
        }
        out.push(name.to_string());
        if let Some(cfg) = self.memory_namespaces.get(name) {
            for parent in &cfg.include {
                self.namespace_chain_walk(parent, out, seen);
            }
        }
    }

    /// Resolve the memory namespace for a given profile.
    ///
    /// Order: explicit `profile.memory_namespace` > `"default"`.
    pub fn namespace_for_profile(&self, profile_name: &str) -> &str {
        self.profiles
            .get(profile_name)
            .and_then(|p| p.memory_namespace.as_deref())
            .unwrap_or(DEFAULT_NAMESPACE_NAME)
    }

    /// Resolve the memory namespace for a given `room_id`. Combines
    /// `profile_for` with `namespace_for_profile`; rooms without a profile
    /// fall through to `"default"`.
    pub fn namespace_for_room(&self, room_id: &str) -> &str {
        match self.profile_for(room_id) {
            Some(p) => self.namespace_for_profile(p),
            None => DEFAULT_NAMESPACE_NAME,
        }
    }

    /// Every memory namespace name relevant to this config: the implicit
    /// `"default"`, every `[memory_namespace.<name>]` key, and every
    /// namespace named by a profile. Used by background catch-up loops to
    /// know what subtrees to enumerate.
    pub fn all_memory_namespaces(&self) -> Vec<String> {
        let mut out: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
        out.insert(DEFAULT_NAMESPACE_NAME.to_string());
        out.extend(self.memory_namespaces.keys().cloned());
        for prof in self.profiles.values() {
            if let Some(ns) = &prof.memory_namespace {
                out.insert(ns.clone());
            }
        }
        out.into_iter().collect()
    }

    /// Resolve the default config path: `~/.config/sapphire-agent/config.toml`
    pub fn default_path() -> PathBuf {
        if let Some(dirs) = directories::ProjectDirs::from("", "", "sapphire-agent") {
            dirs.config_dir().join("config.toml")
        } else {
            PathBuf::from("config.toml")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(s: &str) -> Config {
        toml::from_str(s).expect("config should parse")
    }

    const MINIMAL: &str = r#"
[anthropic]
api_key = "test"
"#;

    #[test]
    fn no_profiles_means_no_resolution() {
        let cfg = parse(MINIMAL);
        assert!(cfg.profile_for("!any:srv").is_none());
        assert!(cfg.validate_profiles().is_empty());
    }

    #[test]
    fn default_profile_is_used_when_room_unspecified() {
        let cfg = parse(
            r#"
[anthropic]
api_key = "test"

[profiles.default]
provider = "anthropic"
"#,
        );
        assert_eq!(cfg.profile_for("!some:srv"), Some("default"));
    }

    #[test]
    fn room_override_wins_over_default_profile() {
        let cfg = parse(
            r#"
[anthropic]
api_key = "test"

[providers.local]
type = "openai_compatible"
base_url = "http://127.0.0.1:8080/v1"
model = "gemma-4-31b-it"

[profiles.default]
provider = "anthropic"

[profiles.nsfw]
provider = "local"

[rooms."!nsfw:srv"]
profile = "nsfw"
"#,
        );
        assert_eq!(cfg.profile_for("!nsfw:srv"), Some("nsfw"));
        assert_eq!(cfg.profile_for("!other:srv"), Some("default"));
        assert_eq!(cfg.provider_for_profile("nsfw"), Some("local"));
        assert_eq!(cfg.provider_for_profile("default"), Some("anthropic"));
        assert!(cfg.validate_profiles().is_empty());
    }

    #[test]
    fn validate_flags_unknown_provider_in_profile() {
        let cfg = parse(
            r#"
[anthropic]
api_key = "test"

[profiles.default]
provider = "ghost"
"#,
        );
        let errors = cfg.validate_profiles();
        assert_eq!(errors.len(), 1, "got: {errors:?}");
        assert!(errors[0].contains("ghost"));
    }

    #[test]
    fn validate_flags_unknown_fallback_provider() {
        let cfg = parse(
            r#"
[anthropic]
api_key = "test"

[profiles.default]
provider = "anthropic"
fallback_provider = "ghost"
"#,
        );
        let errors = cfg.validate_profiles();
        assert_eq!(errors.len(), 1);
        assert!(errors[0].contains("fallback"));
        assert!(errors[0].contains("ghost"));
    }

    #[test]
    fn validate_flags_unknown_profile_in_room() {
        let cfg = parse(
            r#"
[anthropic]
api_key = "test"

[rooms."!x:srv"]
profile = "missing"
"#,
        );
        let errors = cfg.validate_profiles();
        assert_eq!(errors.len(), 1);
        assert!(errors[0].contains("missing"));
    }

    #[test]
    fn shipped_example_parses() {
        // Sanity check: the example file we ship in the repo must parse
        // and validate without errors so first-time users aren't greeted
        // with a confusing TOML error.
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("config.example.toml");
        let cfg = Config::load(&path).expect("config.example.toml should parse");
        assert!(
            cfg.validate_profiles().is_empty(),
            "validation errors: {:?}",
            cfg.validate_profiles()
        );
    }

    #[test]
    fn namespace_default_resolves_when_unconfigured() {
        let cfg = parse(MINIMAL);
        assert_eq!(
            cfg.resolve_namespace_chain(DEFAULT_NAMESPACE_NAME),
            vec!["default".to_string()]
        );
        assert_eq!(cfg.namespace_for_room("!any:srv"), "default");
        assert!(cfg.validate_profiles().is_empty());
    }

    #[test]
    fn namespace_chain_includes_parents_in_dfs_preorder() {
        let cfg = parse(
            r#"
[anthropic]
api_key = "test"

[memory_namespace.user]
include = ["default"]

[memory_namespace.user_nsfw]
include = ["user"]
"#,
        );
        assert_eq!(
            cfg.resolve_namespace_chain("user_nsfw"),
            vec!["user_nsfw".to_string(), "user".to_string(), "default".to_string()]
        );
        assert_eq!(
            cfg.resolve_namespace_chain("user"),
            vec!["user".to_string(), "default".to_string()]
        );
    }

    #[test]
    fn namespace_chain_dedupes_diamond() {
        // a includes b and c; b and c both include d. d should appear once.
        let cfg = parse(
            r#"
[anthropic]
api_key = "test"

[memory_namespace.b]
include = ["d"]

[memory_namespace.c]
include = ["d"]

[memory_namespace.d]

[memory_namespace.a]
include = ["b", "c"]
"#,
        );
        let chain = cfg.resolve_namespace_chain("a");
        assert_eq!(chain.iter().filter(|n| *n == "d").count(), 1);
        assert_eq!(chain[0], "a");
    }

    #[test]
    fn namespace_cycle_is_rejected() {
        let cfg = parse(
            r#"
[anthropic]
api_key = "test"

[memory_namespace.a]
include = ["b"]

[memory_namespace.b]
include = ["a"]
"#,
        );
        let errors = cfg.validate_profiles();
        assert!(
            errors.iter().any(|e| e.contains("cycle")),
            "expected cycle error, got: {errors:?}"
        );
    }

    #[test]
    fn namespace_unknown_include_is_rejected() {
        let cfg = parse(
            r#"
[anthropic]
api_key = "test"

[memory_namespace.user]
include = ["ghost"]
"#,
        );
        let errors = cfg.validate_profiles();
        assert!(
            errors.iter().any(|e| e.contains("ghost")),
            "expected unknown-namespace error, got: {errors:?}"
        );
    }

    #[test]
    fn profile_memory_namespace_resolves() {
        let cfg = parse(
            r#"
[anthropic]
api_key = "test"

[memory_namespace.user_nsfw]
include = ["default"]

[profiles.nsfw]
provider         = "anthropic"
memory_namespace = "user_nsfw"

[rooms."!nsfw:srv"]
profile = "nsfw"
"#,
        );
        assert!(cfg.validate_profiles().is_empty());
        assert_eq!(cfg.namespace_for_room("!nsfw:srv"), "user_nsfw");
        assert_eq!(cfg.namespace_for_room("!other:srv"), "default");
    }

    #[test]
    fn profile_unknown_memory_namespace_is_rejected() {
        let cfg = parse(
            r#"
[anthropic]
api_key = "test"

[profiles.nsfw]
provider         = "anthropic"
memory_namespace = "ghost"
"#,
        );
        let errors = cfg.validate_profiles();
        assert!(
            errors.iter().any(|e| e.contains("ghost")),
            "expected unknown-namespace error, got: {errors:?}"
        );
    }

    #[test]
    fn all_memory_namespaces_unions_sources() {
        let cfg = parse(
            r#"
[anthropic]
api_key = "test"

[memory_namespace.user]
include = ["default"]

[profiles.nsfw]
provider         = "anthropic"
memory_namespace = "user_nsfw"

[memory_namespace.user_nsfw]
include = ["user"]
"#,
        );
        let all = cfg.all_memory_namespaces();
        assert!(all.contains(&"default".to_string()));
        assert!(all.contains(&"user".to_string()));
        assert!(all.contains(&"user_nsfw".to_string()));
    }

    #[test]
    fn provider_config_parses_openai_compatible() {
        let cfg = parse(
            r#"
[anthropic]
api_key = "test"

[providers.local]
type = "openai_compatible"
base_url = "http://127.0.0.1:8080/v1"
model = "gemma-4-31b-it"
"#,
        );
        let local = cfg.providers.get("local").expect("local provider present");
        match local {
            ProviderConfig::OpenAiCompatible(c) => {
                assert_eq!(c.base_url, "http://127.0.0.1:8080/v1");
                assert_eq!(c.model, "gemma-4-31b-it");
                assert!(c.api_key.is_none());
            }
        }
    }
}
