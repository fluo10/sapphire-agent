//! Concrete STT/TTS provider implementations.
//!
//! Real providers are wired up incrementally and may be gated behind
//! cargo features (currently `voice-whisper` for local whisper.cpp).
//! Mock providers are always available for testing and skeleton work.

mod gradio_tts;
mod mock;
#[cfg(feature = "voice-whisper")]
mod whisper_rs;

pub(super) use gradio_tts::GradioTts;
pub(super) use mock::{MockStt, MockTts};
#[cfg(feature = "voice-whisper")]
pub(super) use whisper_rs::WhisperRsStt;
