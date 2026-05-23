//! Pure (bevy-free) application state.
//!
//! Kept independent of bevy types so future mobile / cross-platform GUI
//! crates (`sapphire-call-mobile`, `sapphire-call-gui`) can reuse it.

/// Which screen is currently rendered.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Screen {
    /// First-run setup or "Settings" entered from the chat screen.
    Settings,
    /// Live chat against the configured endpoint.
    Chat,
}

/// Connection lifecycle for the agent session.
#[derive(Debug, Clone)]
pub enum Session {
    /// Either no config yet, or the user backed out of a previous
    /// connection attempt.
    Disconnected,
    /// Background task is running `initialize` against the configured
    /// endpoint.
    Initializing,
    /// Session is ready; chat input is enabled.
    Ready {
        session_id: String,
        display_id: String,
    },
    /// `initialize` failed. The message is surfaced in the chat panel
    /// and the user can retry from Settings.
    Failed { message: String },
}

/// A single message in the on-screen chat history. Tool calls and
/// audio chunks aren't persisted here — they're transient UI affordances
/// (spinner / playback) rendered separately while a turn is in flight.
#[derive(Debug, Clone)]
pub struct ChatEntry {
    pub role: ChatRole,
    pub text: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChatRole {
    User,
    Assistant,
    /// System-level notice (e.g. TTS unavailable, connection error).
    /// Rendered in a muted style.
    System,
}

/// In-flight turn state. Disabled input until the server emits its
/// final `result`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TurnState {
    Idle,
    Sending,
}
