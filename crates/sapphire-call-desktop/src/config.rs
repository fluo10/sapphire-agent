//! Desktop client configuration.
//!
//! Persisted to `~/.config/sapphire-call-desktop/config.toml`. The
//! schema is intentionally separate from the CLI's `CallConfig`
//! ([`sapphire_call_core::config::CallConfig`]) — the satellite cares
//! about wake-word / VAD knobs we don't surface in the GUI, and the
//! GUI cares about its own `tts` opt-in and (later) window-state /
//! avatar settings the CLI doesn't.
//!
//! The shared `ServerConfig` block is re-used from `sapphire-call-core`
//! so endpoint + token map 1:1 between the two clients.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use sapphire_call_core::config::ServerConfig;
use serde::{Deserialize, Serialize};

/// Top-level config schema written to disk.
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DesktopConfig {
    #[serde(default)]
    pub server: ServerConfig,
    #[serde(default)]
    pub tts: TtsConfig,
}

/// Whether to request server-side TTS for each chat reply. Default off
/// — first-run users get a silent text chat, then enable TTS in
/// Settings once they confirm the server has a `voice_pipeline`.
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TtsConfig {
    #[serde(default)]
    pub enabled: bool,
}

impl DesktopConfig {
    /// Conventional XDG path. Distinct from the CLI's
    /// `~/.config/sapphire-call/config.toml` so the two clients can
    /// co-exist with separate endpoint / token bindings if needed.
    pub fn default_path() -> Option<PathBuf> {
        directories::ProjectDirs::from("", "", "sapphire-call-desktop")
            .map(|p| p.config_dir().join("config.toml"))
    }

    pub fn load(path: &Path) -> Result<Self> {
        let raw =
            std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
        let cfg: DesktopConfig =
            toml::from_str(&raw).with_context(|| format!("parse {}", path.display()))?;
        Ok(cfg)
    }

    /// Atomic-ish save: write to `<path>.partial` then rename.
    pub fn save(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("create dir {}", parent.display()))?;
        }
        let raw = toml::to_string_pretty(self).context("serialize desktop config")?;
        let tmp = path.with_extension("partial");
        std::fs::write(&tmp, raw).with_context(|| format!("write {}", tmp.display()))?;
        std::fs::rename(&tmp, path)
            .with_context(|| format!("rename {} → {}", tmp.display(), path.display()))?;
        Ok(())
    }

    /// True when the user has supplied the minimum needed to talk to
    /// an agent. Used to decide whether to land on Settings or Chat at
    /// startup.
    pub fn is_complete(&self) -> bool {
        let has_url = self
            .server
            .url
            .as_deref()
            .map(|s| !s.trim().is_empty())
            .unwrap_or(false);
        let has_token = self
            .server
            .token
            .as_deref()
            .map(|s| !s.trim().is_empty())
            .unwrap_or(false);
        has_url && has_token
    }
}
