//! Voice pipeline subsystem.
//!
//! Exposes STT / TTS provider abstractions and a registry that builds
//! concrete instances from `[stt_provider.*]` / `[tts_provider.*]` config
//! blocks. The MCP `voice/pipeline_run` method (in [`crate::serve`])
//! wires these into a per-request flow:
//!
//! ```text
//! base64 PCM ─► SttProvider::transcribe ─► transcript text
//!            ─► (same LLM turn-processing as the chat method) ─► reply text
//!            ─► TtsProvider::synthesize_stream ─► PCM chunks ─► SSE progress
//! ```
//!
//! Sample rate is fixed at 16 kHz mono s16le throughout the pipeline.
//! Providers expose their native rate via `sample_rate()` so callers can
//! validate at registration time; non-16kHz providers are not supported
//! in v1.

pub mod pipeline;
pub mod stt;
pub mod tts;
#[cfg(feature = "voice-sherpa")]
pub mod vad;

mod providers;

use std::collections::HashMap;
use std::sync::Arc;

use crate::config::{Config, SttProviderConfig, TtsProviderConfig};

pub use stt::SttProvider;
pub use tts::TtsProvider;

/// Fixed pipeline sample rate. Providers must produce/consume at this rate.
pub const PIPELINE_SAMPLE_RATE: u32 = 16_000;

/// Holds every voice provider instantiated from config, keyed by the
/// user-chosen name (e.g. `"whisper_local"`, `"irodori"`).
pub struct VoiceProviders {
    stt: HashMap<String, Arc<dyn SttProvider>>,
    tts: HashMap<String, Arc<dyn TtsProvider>>,
}

impl VoiceProviders {
    /// Build the registry from a fully-validated `Config`. Returns an
    /// error if any provider fails to initialise (e.g. whisper model
    /// not found on disk).
    pub fn from_config(config: &Config) -> anyhow::Result<Self> {
        let mut stt: HashMap<String, Arc<dyn SttProvider>> = HashMap::new();
        for (name, cfg) in &config.stt_providers {
            let provider = build_stt(name, cfg)?;
            stt.insert(name.clone(), provider);
        }
        let mut tts: HashMap<String, Arc<dyn TtsProvider>> = HashMap::new();
        for (name, cfg) in &config.tts_providers {
            let provider = build_tts(name, cfg)?;
            tts.insert(name.clone(), provider);
        }
        Ok(Self { stt, tts })
    }

    pub fn stt(&self, name: &str) -> Option<Arc<dyn SttProvider>> {
        self.stt.get(name).cloned()
    }

    pub fn tts(&self, name: &str) -> Option<Arc<dyn TtsProvider>> {
        self.tts.get(name).cloned()
    }
}

fn build_stt(name: &str, cfg: &SttProviderConfig) -> anyhow::Result<Arc<dyn SttProvider>> {
    match cfg {
        SttProviderConfig::Mock { transcript } => Ok(Arc::new(providers::MockStt::new(
            name.to_string(),
            transcript.clone(),
        ))),
        SttProviderConfig::SherpaOnnx(cfg) => {
            #[cfg(feature = "voice-sherpa")]
            {
                Ok(Arc::new(providers::SherpaOnnxStt::new(
                    name.to_string(),
                    cfg.clone(),
                )?))
            }
            #[cfg(not(feature = "voice-sherpa"))]
            {
                let _ = cfg;
                anyhow::bail!(
                    "stt_provider '{name}': type = \"sherpa_onnx\" requires the \
                     `voice-sherpa` cargo feature to be enabled at build time"
                )
            }
        }
        SttProviderConfig::OpenAiWhisperApi { .. } => {
            anyhow::bail!(
                "stt_provider '{name}': type = \"openai_whisper_api\" is not yet implemented"
            )
        }
    }
}

fn build_tts(name: &str, cfg: &TtsProviderConfig) -> anyhow::Result<Arc<dyn TtsProvider>> {
    match cfg {
        TtsProviderConfig::Mock {
            duration_ms,
            frequency_hz,
        } => Ok(Arc::new(providers::MockTts::new(
            name.to_string(),
            *duration_ms,
            *frequency_hz,
        ))),
        TtsProviderConfig::Gradio {
            base_url,
            fn_name,
            payload,
            audio_field,
        } => Ok(Arc::new(providers::GradioTts::new(
            name.to_string(),
            base_url.clone(),
            fn_name.clone(),
            payload.clone(),
            audio_field.clone(),
        )?)),
        TtsProviderConfig::OpenAiTts { .. } => {
            anyhow::bail!(
                "tts_provider '{name}': type = \"openai_tts\" is not yet implemented"
            )
        }
        TtsProviderConfig::StyleBertVits2(cfg) => Ok(Arc::new(providers::StyleBertVits2Tts::new(
            name.to_string(),
            cfg.clone(),
        )?)),
        TtsProviderConfig::SherpaOnnx(cfg) => {
            #[cfg(feature = "voice-sherpa")]
            {
                Ok(Arc::new(providers::SherpaOnnxTts::new(
                    name.to_string(),
                    cfg.clone(),
                )?))
            }
            #[cfg(not(feature = "voice-sherpa"))]
            {
                let _ = cfg;
                anyhow::bail!(
                    "tts_provider '{name}': type = \"sherpa_onnx\" requires the \
                     `voice-sherpa` cargo feature to be enabled at build time"
                )
            }
        }
    }
}
