use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Config {
    pub matrix: MatrixConfig,
    pub anthropic: AnthropicConfig,
    /// Directory containing AGENT.md and MEMORY.md.
    /// Defaults to the config file's parent directory.
    pub workspace_dir: Option<String>,
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
pub struct AnthropicConfig {
    pub api_key: String,
    #[serde(default = "default_model")]
    pub model: String,
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

    /// Resolve the default config path: `~/.config/sapphire-agent/config.toml`
    pub fn default_path() -> PathBuf {
        if let Some(dirs) = directories::ProjectDirs::from("", "", "sapphire-agent") {
            dirs.config_dir().join("config.toml")
        } else {
            PathBuf::from("config.toml")
        }
    }
}
