//! Server-side Silero VAD.
//!
//! Used by the Discord voice path to chunk per-speaker 16 kHz f32
//! audio into discrete utterances. Wraps sherpa-onnx's
//! `VoiceActivityDetector`; the ONNX is auto-downloaded from the
//! sherpa-onnx releases page on first use, same way the satellite
//! does it (see `crates/sapphire-call/src/voice/download.rs`).
//!
//! One detector instance per speaker keeps state isolation simple
//! (Silero VAD has internal buffers / smoothing; sharing across
//! speakers would mix their speech state).

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow};
use sherpa_onnx::{SileroVadModelConfig, VadModelConfig, VoiceActivityDetector};
use tracing::info;

/// Public URL of the Silero VAD ONNX on the sherpa-onnx releases
/// page. Pinned so cached files don't get invalidated by upstream
/// version bumps.
const SILERO_VAD_URL: &str =
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx";

/// Silero VAD frame size in samples. Required by the model.
pub const VAD_WINDOW_SAMPLES: usize = 512;
/// Pipeline sample rate (mono 16 kHz). Server-side VAD assumes
/// audio is already resampled to this; callers do downmix +
/// resample upstream.
pub const VAD_SAMPLE_RATE: u32 = 16_000;
/// Max single-utterance duration (s). Matches the satellite
/// configuration so the two ends agree on capture cap.
pub const VAD_MAX_SPEECH_SECONDS: f32 = 30.0;

/// Resolve the path to the Silero VAD ONNX, downloading on first
/// use. Override via `SAPPHIRE_VOICE_CACHE_DIR` (used by tests +
/// containerised deployments).
pub fn ensure_silero_model() -> Result<PathBuf> {
    let dir = cache_dir();
    let dest = dir.join("silero_vad.onnx");
    if dest.exists() {
        return Ok(dest);
    }
    std::fs::create_dir_all(&dir)
        .with_context(|| format!("create dir {}", dir.display()))?;

    info!("Downloading Silero VAD from {SILERO_VAD_URL} to {} (one-time)", dest.display());
    let resp = reqwest::blocking::Client::builder()
        .timeout(None)
        .build()?
        .get(SILERO_VAD_URL)
        .send()?
        .error_for_status()?;
    let tmp = dest.with_extension("partial");
    let mut file = std::fs::File::create(&tmp)?;
    let mut reader = resp;
    std::io::copy(&mut reader, &mut file)?;
    std::fs::rename(&tmp, &dest)?;
    info!("Silero VAD ready at {}", dest.display());
    Ok(dest)
}

fn cache_dir() -> PathBuf {
    if let Ok(custom) = std::env::var("SAPPHIRE_VOICE_CACHE_DIR") {
        return PathBuf::from(shellexpand::tilde(&custom).into_owned());
    }
    if let Some(dirs) = directories::ProjectDirs::from("", "", "sapphire-agent") {
        return dirs.data_local_dir().join("voice-models");
    }
    PathBuf::from(".sapphire-agent/voice-models")
}

/// Build a freshly-initialised Silero detector. One instance per
/// concurrent speaker stream (Discord voice channels with multiple
/// users → multiple detectors).
pub fn build_silero(model_path: &Path) -> Result<VoiceActivityDetector> {
    let silero = SileroVadModelConfig {
        model: Some(model_path.to_string_lossy().into_owned()),
        threshold: 0.5,
        min_silence_duration: 0.25,
        min_speech_duration: 0.25,
        window_size: VAD_WINDOW_SAMPLES as i32,
        max_speech_duration: VAD_MAX_SPEECH_SECONDS,
    };
    let config = VadModelConfig {
        silero_vad: silero,
        ten_vad: Default::default(),
        sample_rate: VAD_SAMPLE_RATE as i32,
        num_threads: 1,
        provider: Some("cpu".to_string()),
        debug: false,
    };
    VoiceActivityDetector::create(&config, VAD_MAX_SPEECH_SECONDS)
        .ok_or_else(|| anyhow!("failed to create sherpa-onnx VoiceActivityDetector"))
}

/// Convenience: build a detector after ensuring the model is on disk.
pub fn build_default() -> Result<VoiceActivityDetector> {
    let path = ensure_silero_model()?;
    build_silero(&path)
}
