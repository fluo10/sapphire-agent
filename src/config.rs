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
}

/// Built-in name of the Anthropic provider — referenced by profiles.
pub const ANTHROPIC_PROVIDER_NAME: &str = "anthropic";

/// Conventional name of the default profile.
pub const DEFAULT_PROFILE_NAME: &str = "default";

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
        errors
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
