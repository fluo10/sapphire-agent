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
    /// Carries the actual bytes; appears in in-flight `ChatMessage`s sent to a
    /// provider. Persisted forms swap this for [`ContentPart::ImageRef`] (when
    /// the image cache is enabled) to keep JSONL and long-lived in-memory
    /// history small.
    Image {
        media_type: String,
        data_base64: String,
    },
    /// Reference to an image stored in the workspace-external image cache.
    /// Persisted to JSONL as the canonical compact image representation
    /// and held in long-lived in-memory history. Re-hydrated to
    /// [`ContentPart::Image`] just before each provider call when the
    /// cache still has the bytes; a cache miss degrades to a text marker.
    ImageRef {
        media_type: String,
        sha256: String,
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

/// Input modality of a user-role message. Voice inputs reach the model
/// via server-side STT and can carry transcription errors, so the
/// model is told the modality via a prefix label (see
/// `apply_input_kind_label` in `serve`). `Voice` is a unit variant on
/// purpose: when voice-print speaker identification is later added,
/// new variants (`KnownVoice { speaker_id, .. }` / `UnknownVoice`)
/// will be introduced alongside this one — `Voice` remains the
/// "identification disabled / not yet run" default and existing
/// `{"kind":"voice"}` JSONL stays readable without migration.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum UserInputKind {
    Text,
    Voice,
}

/// A message in the conversation history.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: Role,
    pub parts: Vec<ContentPart>,
    /// Modality this message was authored in. Meaningful only for
    /// `Role::User`; `None` on assistant/tool messages and on
    /// channel-side user messages that don't have a meaningful
    /// modality (e.g. heartbeat-injected user lines).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub input_kind: Option<UserInputKind>,
    /// Identifier of the authenticated user this message is attributed
    /// to. Currently always `None` — populated in a future task once
    /// API-key / channel-ID → user_id mapping lands. The corresponding
    /// profile lives in `<workspace>/users/<namespace>/<user_id>.md`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub user_id: Option<String>,
}

impl ChatMessage {
    pub fn user(text: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            parts: vec![ContentPart::Text(text.into())],
            input_kind: Some(UserInputKind::Text),
            user_id: None,
        }
    }

    /// User message produced from a voice (STT) input. Speaker
    /// identification is not implemented yet, so all such messages
    /// carry `UserInputKind::Voice` — once voice-print clustering
    /// lands, this constructor will accept a richer variant.
    pub fn user_voice(text: impl Into<String>) -> Self {
        Self {
            role: Role::User,
            parts: vec![ContentPart::Text(text.into())],
            input_kind: Some(UserInputKind::Voice),
            user_id: None,
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
            input_kind: Some(UserInputKind::Text),
            user_id: None,
        }
    }

    pub fn assistant(text: impl Into<String>) -> Self {
        Self {
            role: Role::Assistant,
            parts: vec![ContentPart::Text(text.into())],
            input_kind: None,
            user_id: None,
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
            input_kind: None,
            user_id: None,
        }
    }

    /// User message containing tool execution results plus optional
    /// image attachments. Images are appended after all tool_result
    /// parts; they're carried as siblings to the tool_result blocks on
    /// the same user message — the wire format both Anthropic and (in
    /// degraded form) OpenAI-compatible accept.
    ///
    /// Used by tools like `recall_image` to deliver image bytes back to
    /// the model alongside their textual tool_result. Pass an empty
    /// `images` vec for the common text-only case.
    pub fn tool_results_with_images(
        results: Vec<(String, String)>,
        images: Vec<(String, String)>,
    ) -> Self {
        let mut parts: Vec<ContentPart> = results
            .into_iter()
            .map(|(id, content)| ContentPart::ToolResult {
                tool_use_id: id,
                content,
            })
            .collect();
        parts.extend(
            images
                .into_iter()
                .map(|(media_type, data_base64)| ContentPart::Image {
                    media_type,
                    data_base64,
                }),
        );
        Self {
            role: Role::User,
            parts,
            input_kind: None,
            user_id: None,
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
    #[allow(dead_code)]
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

    #[allow(dead_code)]
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
