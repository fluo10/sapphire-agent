use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
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
    /// Whether to generate a daily log at the day boundary. Default: true.
    #[serde(default = "default_true")]
    pub daily_log_enabled: bool,
    /// Whether to compact MEMORY.md at the day boundary. Default: true.
    #[serde(default = "default_true")]
    pub memory_compaction_enabled: bool,
}

fn default_true() -> bool {
    true
}

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
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct MatrixConfig {
    pub homeserver: String,
    pub access_token: String,
    pub user_id: String,
    pub device_id: String,
    pub room_id: String,
    #[serde(default)]
    pub allowed_users: Vec<String>,
    /// E2EE recovery key (optional)
    pub recovery_key: Option<String>,
    /// Directory for matrix-sdk state/crypto store. Defaults to
    /// `~/.local/share/sapphire-agent/matrix`.
    pub state_dir: Option<String>,
}

impl MatrixConfig {
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

fn default_model() -> String {
    "claude-opus-4-6".to_string()
}

fn default_max_tokens() -> u32 {
    8192
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

    /// Resolve the default config path: `~/.config/sapphire-agent/config.toml`
    pub fn default_path() -> PathBuf {
        if let Some(dirs) = directories::ProjectDirs::from("", "", "sapphire-agent") {
            dirs.config_dir().join("config.toml")
        } else {
            PathBuf::from("config.toml")
        }
    }
}
