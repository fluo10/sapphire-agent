//! Concrete STT/TTS provider implementations.
//!
//! Local providers (sherpa-onnx STT/TTS) are gated behind the
//! `voice-sherpa` cargo feature so the default build doesn't pay
//! sherpa-onnx-sys's C++ compile cost. Mock providers and the Gradio
//! TTS client (HTTP-only) are always available.

mod gradio_tts;
mod mock;
#[cfg(feature = "voice-sherpa")]
mod sherpa_stt;
#[cfg(feature = "voice-sherpa")]
mod sherpa_tts;
#[cfg(feature = "voice-sherpa")]
pub(crate) mod sherpa_download;
mod style_bert_vits2_tts;
mod wav_stream;

pub(super) use gradio_tts::GradioTts;
pub(super) use mock::{MockStt, MockTts};
#[cfg(feature = "voice-sherpa")]
pub(super) use sherpa_stt::SherpaOnnxStt;
#[cfg(feature = "voice-sherpa")]
pub(super) use sherpa_tts::SherpaOnnxTts;
pub(super) use style_bert_vits2_tts::StyleBertVits2Tts;
