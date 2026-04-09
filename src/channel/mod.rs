pub mod discord;
pub mod matrix;

use async_trait::async_trait;

/// A message received from a channel.
#[derive(Debug, Clone)]
pub struct IncomingMessage {
    /// Platform-specific message ID.
    pub id: String,
    /// Sender's user ID (e.g. `@user:example.com`).
    pub sender: String,
    /// Text content.
    pub content: String,
    /// Room/conversation identifier to reply to.
    pub room_id: String,
    /// Unix timestamp (milliseconds).
    pub timestamp: u64,
    /// Thread identifier for threaded replies, if applicable.
    pub thread_id: Option<String>,
}

/// A message to send through a channel.
#[derive(Debug, Clone)]
pub struct OutgoingMessage {
    pub content: String,
    /// Room/conversation to send to.
    pub room_id: String,
    /// Thread identifier for threaded replies.
    pub thread_id: Option<String>,
}

impl OutgoingMessage {
    pub fn new(content: impl Into<String>, room_id: impl Into<String>) -> Self {
        Self {
            content: content.into(),
            room_id: room_id.into(),
            thread_id: None,
        }
    }
}

/// Core channel trait.
#[async_trait]
pub trait Channel: Send + Sync {
    fn name(&self) -> &str;

    async fn send(&self, message: &OutgoingMessage) -> anyhow::Result<()>;

    /// Long-running: receive messages and forward them through `tx`.
    async fn listen(&self, tx: tokio::sync::mpsc::Sender<IncomingMessage>) -> anyhow::Result<()>;

    async fn start_typing(&self, _room_id: &str) -> anyhow::Result<()> {
        Ok(())
    }

    async fn stop_typing(&self, _room_id: &str) -> anyhow::Result<()> {
        Ok(())
    }
}
