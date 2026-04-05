pub mod anthropic;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

/// A single message in the conversation history.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: Role,
    pub content: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    User,
    Assistant,
}

impl ChatMessage {
    pub fn user(content: impl Into<String>) -> Self {
        Self { role: Role::User, content: content.into() }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self { role: Role::Assistant, content: content.into() }
    }
}

/// A completed response from the provider.
#[derive(Debug, Clone)]
pub struct ChatResponse {
    pub text: String,
}

/// Core provider trait.
#[async_trait]
pub trait Provider: Send + Sync {
    fn name(&self) -> &str;

    /// Send the conversation history and return the assistant's response.
    async fn chat(
        &self,
        system: Option<&str>,
        messages: &[ChatMessage],
    ) -> anyhow::Result<ChatResponse>;
}
