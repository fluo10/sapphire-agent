pub mod anthropic;
pub mod fallback;
pub mod openai_compatible;
pub mod registry;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::borrow::Cow;

// ---------------------------------------------------------------------------
// Tool definitions
// ---------------------------------------------------------------------------

/// Tool specification passed to the LLM.
#[derive(Debug, Clone)]
pub struct ToolSpec {
    pub name: Cow<'static, str>,
    pub description: Cow<'static, str>,
    pub input_schema: Value,
}

/// A tool call returned by the LLM.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub input: Value,
}

// ---------------------------------------------------------------------------
// Message types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    User,
    Assistant,
}

/// A single part of a message's content.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ContentPart {
    Text(String),
    /// Inline image, base64-encoded. `media_type` is the MIME type (e.g. `image/png`).
    Image {
        media_type: String,
        data_base64: String,
    },
    ToolUse {
        id: String,
        name: String,
        input: Value,
    },
    ToolResult {
        tool_use_id: String,
        content: String,
    },
}

/// A message in the conversation history.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: Role,
    pub parts: Vec<ContentPart>,
}

impl ChatMessage {
    pub fn user(text: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            parts: vec![ContentPart::Text(text.into())],
        }
    }

    /// User message with text plus inline images. Images are placed before the
    /// text part — Anthropic recommends this ordering for best comprehension.
    /// An empty `text` is omitted so the API never sees an empty text block.
    pub fn user_with_images(
        text: impl Into<String>,
        images: impl IntoIterator<Item = (String, String)>,
    ) -> Self {
        let mut parts: Vec<ContentPart> = images
            .into_iter()
            .map(|(media_type, data_base64)| ContentPart::Image {
                media_type,
                data_base64,
            })
            .collect();
        let text = text.into();
        if !text.is_empty() {
            parts.push(ContentPart::Text(text));
        }
        Self {
            role: Role::User,
            parts,
        }
    }

    pub fn assistant(text: impl Into<String>) -> Self {
        Self {
            role: Role::Assistant,
            parts: vec![ContentPart::Text(text.into())],
        }
    }

    /// Assistant message that includes both a text response and tool calls.
    pub fn assistant_with_tools(text: Option<String>, tool_calls: Vec<ToolCall>) -> Self {
        let mut parts = Vec::new();
        if let Some(t) = text.filter(|s| !s.is_empty()) {
            parts.push(ContentPart::Text(t));
        }
        for call in tool_calls {
            parts.push(ContentPart::ToolUse {
                id: call.id,
                name: call.name,
                input: call.input,
            });
        }
        Self {
            role: Role::Assistant,
            parts,
        }
    }

    /// User message containing tool execution results.
    pub fn tool_results(results: Vec<(String, String)>) -> Self {
        let parts = results
            .into_iter()
            .map(|(id, content)| ContentPart::ToolResult {
                tool_use_id: id,
                content,
            })
            .collect();
        Self {
            role: Role::User,
            parts,
        }
    }

    /// Returns the concatenated text content (for storing in history summary).
    pub fn text(&self) -> Option<String> {
        let texts: Vec<&str> = self
            .parts
            .iter()
            .filter_map(|p| {
                if let ContentPart::Text(t) = p {
                    Some(t.as_str())
                } else {
                    None
                }
            })
            .collect();
        if texts.is_empty() {
            None
        } else {
            Some(texts.join(""))
        }
    }
}

// ---------------------------------------------------------------------------
// Provider response
// ---------------------------------------------------------------------------

/// A completed response from the provider.
#[derive(Debug, Clone)]
pub struct ChatResponse {
    pub text: Option<String>,
    pub tool_calls: Vec<ToolCall>,
    /// Provider-reported stop / finish reason. Captured verbatim so that
    /// higher-level wrappers (notably the refusal-fallback layer) can
    /// inspect it. Anthropic emits values like `"end_turn"`, `"refusal"`,
    /// `"max_tokens"`; OpenAI-compatible servers emit `"stop"`,
    /// `"content_filter"`, `"length"`, etc.
    pub stop_reason: Option<String>,
}

impl ChatResponse {
    pub fn text_only(text: String) -> Self {
        Self {
            text: Some(text),
            tool_calls: vec![],
            stop_reason: None,
        }
    }

    pub fn has_tool_calls(&self) -> bool {
        !self.tool_calls.is_empty()
    }

    pub fn text_or_empty(&self) -> &str {
        self.text.as_deref().unwrap_or("")
    }
}

// ---------------------------------------------------------------------------
// Provider trait
// ---------------------------------------------------------------------------

#[async_trait]
pub trait Provider: Send + Sync {
    fn name(&self) -> &str;

    async fn chat(
        &self,
        system: Option<&str>,
        messages: &[ChatMessage],
        tools: Option<&[ToolSpec]>,
    ) -> anyhow::Result<ChatResponse>;
}
