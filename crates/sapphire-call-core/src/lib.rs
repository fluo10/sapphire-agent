//! Shared building blocks for sapphire-call clients (CLI + desktop GUI).
//!
//! Both the always-on voice satellite (`sapphire-call-cli`) and the
//! desktop GUI (`sapphire-call-desktop`) need the same config schema
//! and per-installation device id. They live here so future mobile /
//! cross-platform GUI crates can reuse them without depending on the
//! satellite's cpal + sherpa-onnx code.
//!
//! Anything specific to a particular client (the always-listening
//! state machine, push-to-talk capture, egui widgets, …) stays in its
//! own crate.

pub mod config;
pub mod device_id;
