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
    /// Optional device identity sent to the agent on every session.
    /// The agent uses these strings to tell the model "you are speaking
    /// through the living-room speakerphone; STT may have introduced
    /// typos" — keeps that knowledge out of AGENTS.md and lets each
    /// satellite host describe itself.
    #[serde(default)]
    pub device: DeviceConfig,
    /// Listen-state UX knobs: confirmation beeps and the post-reply
    /// follow-up listening window. All defaults are on / 5 seconds so
    /// fresh installs get the conversational behaviour described in
    /// issue #83 without explicit opt-in.
    #[serde(default)]
    pub behavior: BehaviorConfig,
    /// Microphone gain + wake/VAD sensitivity. All optional with
    /// defaults that match the built-in hard-coded values, so existing
    /// configs keep behaving exactly as before. See issue #87.
    #[serde(default)]
    pub sensitivity: SensitivityConfig,
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

/// Device-identity block. Both fields are optional; when `name` is set
/// the agent renders the session's room name as
/// `"voice channel with <name>"` and uses `description` as the topic.
#[derive(Debug, Default, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DeviceConfig {
    /// Short handle for this device, e.g. `"living-room-speaker"`.
    /// The agent prefixes "voice channel with " server-side.
    pub name: Option<String>,
    /// Free-form description: physical location, who uses it, any
    /// quirks (STT noise, ambient music) the agent should keep in
    /// mind. Becomes the "room description" in the system prompt.
    pub description: Option<String>,
}

impl DeviceConfig {
    /// Convert into the API crate's wire-format struct. Empty when no
    /// field is set so we don't send a meaningless `device: {}` block.
    pub fn to_api(&self) -> Option<sapphire_agent_api::DeviceMetadata> {
        if self.name.is_none() && self.description.is_none() {
            return None;
        }
        Some(sapphire_agent_api::DeviceMetadata {
            name: self.name.clone(),
            description: self.description.clone(),
        })
    }
}

/// Listen-state UX behaviour. See issue #83.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BehaviorConfig {
    /// Play a short rising beep whenever the satellite starts listening
    /// for a command — both the obvious case (wake word fires) and the
    /// silent case (mic re-opens after the AI's TTS reply finishes,
    /// either into the follow-up window in wake-word mode or back to
    /// continuous capture in VAD-only mode). Same role each time: "I'm
    /// now listening to you".
    #[serde(default = "default_true")]
    pub beep_on_wake: bool,
    /// Play a short falling beep when the VAD detects the end of an
    /// utterance and ships it to STT. Confirms "I heard you, processing".
    /// Also fires when the post-reply follow-up window elapses without
    /// the user speaking — same role: "the satellite stopped listening".
    #[serde(default = "default_true")]
    pub beep_on_capture_end: bool,
    /// After the assistant's TTS reply finishes playing, keep the mic
    /// open for this many seconds without requiring the wake word —
    /// captures conversational follow-ups. Set to `0` to disable
    /// (every turn requires re-waking). Only takes effect in
    /// wake-word mode; VAD-only mode is always continuously listening.
    #[serde(default = "default_follow_up_seconds")]
    pub follow_up_listen_seconds: u32,
}

impl Default for BehaviorConfig {
    fn default() -> Self {
        Self {
            beep_on_wake: default_true(),
            beep_on_capture_end: default_true(),
            follow_up_listen_seconds: default_follow_up_seconds(),
        }
    }
}

fn default_true() -> bool {
    true
}

fn default_follow_up_seconds() -> u32 {
    5
}

/// Microphone gain + wake/VAD sensitivity knobs. See issue #87.
///
/// All fields are optional with built-in defaults that mirror the
/// constants previously hard-coded into the satellite — so the
/// historical behaviour is preserved when this block is omitted.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SensitivityConfig {
    /// Linear multiplier applied to mic samples in the cpal input
    /// callback before they are quantised to i16. `1.0` is unity gain
    /// (no change). Anything above unity is clamped to ±full-scale to
    /// avoid wrap-around; this is a software replacement for users
    /// whose OS mixer / hardware can't bring a quiet speakerphone
    /// loud enough for STT.
    #[serde(default = "default_mic_gain")]
    pub mic_gain: f32,
    /// openWakeWord confidence threshold above which a wake fires.
    /// Range `0.0..=1.0`; lower = more sensitive (and more false
    /// positives), higher = stricter. Default mirrors openWakeWord's
    /// own default.
    #[serde(default = "default_wake_threshold")]
    pub wake_threshold: f32,
    /// Cool-down window after a successful wake fire, in milliseconds.
    /// A single sustained utterance otherwise re-triggers the wake
    /// model on consecutive 80 ms chunks. Default ≈ 2 s.
    #[serde(default = "default_wake_cooldown_ms")]
    pub wake_cooldown_ms: u32,
    /// Silero VAD speech probability threshold. Higher = needs more
    /// confident speech to start/extend a segment.
    #[serde(default = "default_vad_threshold")]
    pub vad_threshold: f32,
    /// Silero VAD: how long a silence must last (ms) before the
    /// current speech segment is closed and shipped to STT.
    #[serde(default = "default_vad_min_silence_ms")]
    pub vad_min_silence_ms: u32,
    /// Silero VAD: minimum speech duration (ms) before a segment is
    /// considered valid. Filters out clicks / single-frame noise.
    #[serde(default = "default_vad_min_speech_ms")]
    pub vad_min_speech_ms: u32,
}

impl Default for SensitivityConfig {
    fn default() -> Self {
        Self {
            mic_gain: default_mic_gain(),
            wake_threshold: default_wake_threshold(),
            wake_cooldown_ms: default_wake_cooldown_ms(),
            vad_threshold: default_vad_threshold(),
            vad_min_silence_ms: default_vad_min_silence_ms(),
            vad_min_speech_ms: default_vad_min_speech_ms(),
        }
    }
}

fn default_mic_gain() -> f32 {
    1.0
}
fn default_wake_threshold() -> f32 {
    0.5
}
fn default_wake_cooldown_ms() -> u32 {
    2000
}
fn default_vad_threshold() -> f32 {
    0.5
}
fn default_vad_min_silence_ms() -> u32 {
    250
}
fn default_vad_min_speech_ms() -> u32 {
    250
}

impl CallConfig {
    /// Load a config file from `path`. The caller is expected to
    /// short-circuit when the file doesn't exist — missing-config is
    /// the default state and shouldn't surface as an error.
    pub fn load(path: &Path) -> Result<Self> {
        let raw =
            std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
        let cfg: CallConfig =
            toml::from_str(&raw).with_context(|| format!("parse {}", path.display()))?;
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

[device]
name = "living-room-speaker"
description = "Speakerphone in the living room; STT may produce typos"

[behavior]
beep_on_wake = false
beep_on_capture_end = true
follow_up_listen_seconds = 8

[sensitivity]
mic_gain           = 1.8
wake_threshold     = 0.65
wake_cooldown_ms   = 1500
vad_threshold      = 0.4
vad_min_silence_ms = 350
vad_min_speech_ms  = 200
"#;
        let cfg: CallConfig = toml::from_str(raw).unwrap();
        assert_eq!(cfg.server.room_profile.as_deref(), Some("home_voice"));
        assert_eq!(
            cfg.audio.input_device.as_deref(),
            Some("Jabra SPEAK 510 USB")
        );
        assert_eq!(cfg.language.stt.as_deref(), Some("ja"));
        assert_eq!(cfg.device.name.as_deref(), Some("living-room-speaker"));
        assert!(cfg.device.description.is_some());
        assert!(!cfg.behavior.beep_on_wake);
        assert!(cfg.behavior.beep_on_capture_end);
        assert_eq!(cfg.behavior.follow_up_listen_seconds, 8);
        assert!((cfg.sensitivity.mic_gain - 1.8).abs() < f32::EPSILON);
        assert!((cfg.sensitivity.wake_threshold - 0.65).abs() < f32::EPSILON);
        assert_eq!(cfg.sensitivity.wake_cooldown_ms, 1500);
        assert!((cfg.sensitivity.vad_threshold - 0.4).abs() < f32::EPSILON);
        assert_eq!(cfg.sensitivity.vad_min_silence_ms, 350);
        assert_eq!(cfg.sensitivity.vad_min_speech_ms, 200);
    }

    #[test]
    fn behavior_defaults_when_block_omitted() {
        let cfg: CallConfig = toml::from_str("").unwrap();
        assert!(cfg.behavior.beep_on_wake);
        assert!(cfg.behavior.beep_on_capture_end);
        assert_eq!(cfg.behavior.follow_up_listen_seconds, 5);
    }

    #[test]
    fn sensitivity_defaults_when_block_omitted() {
        let cfg: CallConfig = toml::from_str("").unwrap();
        assert!((cfg.sensitivity.mic_gain - 1.0).abs() < f32::EPSILON);
        assert!((cfg.sensitivity.wake_threshold - 0.5).abs() < f32::EPSILON);
        assert_eq!(cfg.sensitivity.wake_cooldown_ms, 2000);
        assert!((cfg.sensitivity.vad_threshold - 0.5).abs() < f32::EPSILON);
        assert_eq!(cfg.sensitivity.vad_min_silence_ms, 250);
        assert_eq!(cfg.sensitivity.vad_min_speech_ms, 250);
    }

    #[test]
    fn sensitivity_partial_block_uses_per_field_defaults() {
        // Only override mic_gain; every other field should fall back.
        let raw = r#"
[sensitivity]
mic_gain = 2.5
"#;
        let cfg: CallConfig = toml::from_str(raw).unwrap();
        assert!((cfg.sensitivity.mic_gain - 2.5).abs() < f32::EPSILON);
        assert!((cfg.sensitivity.wake_threshold - 0.5).abs() < f32::EPSILON);
        assert_eq!(cfg.sensitivity.wake_cooldown_ms, 2000);
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
