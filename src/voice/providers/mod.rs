//! Concrete STT/TTS provider implementations.
//!
//! Local providers (sherpa-onnx STT/TTS) are gated behind the
//! `voice-sherpa` cargo feature so the default build doesn't pay
//! sherpa-onnx-sys's C++ compile cost. Mock providers and the
//! OpenAI TTS client (HTTP-only, works against the public endpoint
//! or any self-hosted server that speaks the same shape) are
//! always available.

mod mock;
mod openai_tts;
#[cfg(feature = "voice-sherpa")]
mod sherpa_stt;
#[cfg(feature = "voice-sherpa")]
mod sherpa_tts;
#[cfg(feature = "voice-sherpa")]
pub(crate) mod sherpa_download;
mod wav_stream;

pub(super) use mock::{MockStt, MockTts};
pub(super) use openai_tts::OpenAiTts;
#[cfg(feature = "voice-sherpa")]
pub(super) use sherpa_stt::SherpaOnnxStt;
#[cfg(feature = "voice-sherpa")]
pub(super) use sherpa_tts::SherpaOnnxTts;
