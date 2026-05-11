//! Concrete STT/TTS provider implementations.
//!
//! Currently only the `Mock` variants are implemented — real providers
//! (whisper-rs, OpenAI Whisper API, Gradio, OpenAI TTS) are added in
//! follow-up steps.

mod mock;

pub(super) use mock::{MockStt, MockTts};
