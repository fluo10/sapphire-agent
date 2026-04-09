pub mod anthropic;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_json::Value;

// ---------------------------------------------------------------------------
// Tool definitions
// ---------------------------------------------------------------------------

/// Tool specification passed to the LLM.
#[derive(Debug, Clone)]
pub struct ToolSpec {
    pub name: &'static str,
    pub description: &'static str,
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
}

impl ChatResponse {
    pub fn text_only(text: String) -> Self {
        Self {
            text: Some(text),
            tool_calls: vec![],
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
