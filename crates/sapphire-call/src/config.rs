//! Persistent satellite configuration.
//!
//! All fields are optional and override-able from the CLI; this file
//! exists so you don't have to type `--server https://… --room-profile …`
//! every time you start `sapphire-call`. Wake-word configuration is
//! intentionally **not** here — the server is the source of truth for
//! the AI's name (see `[room_profile.<n>].wake_word` on the server).
//!
//! Resolution order at startup:
//!   1. explicit CLI flag
//!   2. config file value
//!   3. built-in default

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::Deserialize;

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CallConfig {
    #[serde(default)]
    pub server: ServerConfig,
    #[serde(default)]
    pub audio: AudioConfig,
    #[serde(default)]
    pub language: LanguageConfig,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ServerConfig {
    /// Base URL of the sapphire-agent serve endpoint, e.g.
    /// `https://agent.example.com`. Equivalent to `--server`.
    pub url: Option<String>,
    /// Resume an existing session by 7-char grain id. Equivalent to
    /// `--session`.
    pub session: Option<String>,
    /// Pin the session to a server-side `[room_profile.<n>]`.
    /// Equivalent to `--room-profile`.
    pub room_profile: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AudioConfig {
    /// Exact cpal name of the input device. Discover with
    /// `sapphire-call voice --list-devices`.
    pub input_device: Option<String>,
    /// Exact cpal name of the output device.
    pub output_device: Option<String>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LanguageConfig {
    /// BCP-47 hint sent to STT. Server's voice_pipeline default
    /// applies when both this and the CLI flag are absent.
    pub stt: Option<String>,
}

impl CallConfig {
    /// Load a config file from `path`. The caller is expected to
    /// short-circuit when the file doesn't exist — missing-config is
    /// the default state and shouldn't surface as an error.
    pub fn load(path: &Path) -> Result<Self> {
        let raw = std::fs::read_to_string(path)
            .with_context(|| format!("read {}", path.display()))?;
        let cfg: CallConfig = toml::from_str(&raw)
            .with_context(|| format!("parse {}", path.display()))?;
        Ok(cfg)
    }

    /// Conventional XDG path: `~/.config/sapphire-call/config.toml`.
    /// Returns `None` when the platform has no notion of a config
    /// directory (e.g. WASM, exotic embedded targets).
    pub fn default_path() -> Option<PathBuf> {
        directories::ProjectDirs::from("", "", "sapphire-call")
            .map(|p| p.config_dir().join("config.toml"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_minimal_config() {
        let raw = r#"
[server]
url = "https://agent.example.com"
"#;
        let cfg: CallConfig = toml::from_str(raw).unwrap();
        assert_eq!(cfg.server.url.as_deref(), Some("https://agent.example.com"));
        assert!(cfg.server.session.is_none());
    }

    #[test]
    fn parses_full_config() {
        let raw = r#"
[server]
url = "https://agent.example.com"
session = "abc1234"
room_profile = "home_voice"

[audio]
input_device = "Jabra SPEAK 510 USB"
output_device = "Jabra SPEAK 510 USB"

[language]
stt = "ja"
"#;
        let cfg: CallConfig = toml::from_str(raw).unwrap();
        assert_eq!(cfg.server.room_profile.as_deref(), Some("home_voice"));
        assert_eq!(
            cfg.audio.input_device.as_deref(),
            Some("Jabra SPEAK 510 USB")
        );
        assert_eq!(cfg.language.stt.as_deref(), Some("ja"));
    }

    #[test]
    fn rejects_unknown_top_level_keys() {
        let raw = r#"
[wake_word]
keyword = "ハロー"
"#;
        // wake_word belongs server-side; reject here so users see the
        // mistake instead of silently having it ignored.
        assert!(toml::from_str::<CallConfig>(raw).is_err());
    }

    #[test]
    fn empty_input_yields_all_none() {
        let cfg: CallConfig = toml::from_str("").unwrap();
        assert!(cfg.server.url.is_none());
        assert!(cfg.audio.input_device.is_none());
    }

    #[test]
    fn shipped_example_parses() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("config.example.toml");
        let _ = CallConfig::load(&path).expect("config.example.toml should parse");
    }
}
