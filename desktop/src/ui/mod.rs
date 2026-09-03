//! egui systems. Bevy-specific glue (resources, system signatures)
//! lives here; the underlying state model is the pure
//! [`crate::state`] module so it can move to a cross-platform
//! `sapphire-call-gui` crate when mobile lands.

pub mod chat;
pub mod settings;
