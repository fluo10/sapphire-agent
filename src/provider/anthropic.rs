use crate::config::AnthropicConfig;
use crate::provider::{ChatMessage, ChatResponse, ContentPart, Provider, Role, ToolCall, ToolSpec};
use anyhow::{Context, Result, bail};
use async_trait::async_trait;
use futures_util::StreamExt;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::BTreeMap;
use tracing::debug;

const ANTHROPIC_API_URL: &str = "https://api.anthropic.com/v1/messages";
const ANTHROPIC_VERSION: &str = "2023-06-01";

pub struct AnthropicProvider {
    api_key: String,
    model: String,
    light_model: Option<String>,
    max_tokens: u32,
    client: Client,
}

impl AnthropicProvider {
    pub fn new(cfg: &AnthropicConfig) -> Self {
        Self {
            api_key: cfg.api_key.clone(),
            model: cfg.model.clone(),
            light_model: cfg.light_model.clone(),
            max_tokens: cfg.max_tokens,
            client: Client::new(),
        }
    }

    /// Choose model based on message content.
    /// Uses `light_model` for casual chat, `model` for coding-related requests.
    fn select_model(&self, messages: &[ChatMessage]) -> &str {
        let Some(light) = &self.light_model else {
            return &self.model;
        };
        let last_user_text = messages
            .iter()
            .rev()
            .find(|m| m.role == Role::User)
            .and_then(|m| m.text())
            .unwrap_or_default();
        if is_coding_related(&last_user_text) {
            &self.model
        } else {
            light
        }
    }
}

/// Heuristic: return true if the text looks like a coding/technical request.
fn is_coding_related(text: &str) -> bool {
    if text.contains("```") {
        return true;
    }
    let lower = text.to_lowercase();
    let keywords = [
        "code",
        "implement",
        "function",
        "method",
        "class",
        "struct",
        "enum",
        "trait",
        "bug",
        "error",
        "debug",
        "fix",
        "compile",
        "refactor",
        "test",
        "algorithm",
        "api",
        "library",
        "crate",
        "cargo",
        "npm",
        "syntax",
        "variable",
        "type",
        "rust",
        "python",
        "javascript",
        "typescript",
        "java",
        "go ",
        " sql",
        "bash",
        "script",
        "コード",
        "実装",
        "関数",
        "バグ",
        "エラー",
        "デバッグ",
    ];
    keywords.iter().any(|kw| lower.contains(kw))
}

// ---------------------------------------------------------------------------
// Request types
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize)]
struct Request<'a> {
    model: &'a str,
    max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<&'a str>,
    messages: Vec<ApiMessage>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<ApiToolSpec<'a>>>,
}

#[derive(Debug, Serialize)]
struct ApiToolSpec<'a> {
    name: &'a str,
    description: &'a str,
    input_schema: &'a Value,
}

#[derive(Debug, Serialize)]
struct ApiMessage {
    role: String,
    content: ApiContent,
}

#[derive(Debug, Serialize)]
#[serde(untagged)]
enum ApiContent {
    /// Simple string (user/assistant text only — wire-format shorthand).
    Text(String),
    /// Array of typed content blocks (required for tool use/results).
    Parts(Vec<ApiPart>),
}

#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ApiPart {
    Text {
        text: String,
    },
    Image {
        source: ApiImageSource,
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

#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ApiImageSource {
    Base64 { media_type: String, data: String },
}

// ---------------------------------------------------------------------------
// SSE response types
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
#[allow(dead_code)]
enum SseEvent {
    #[serde(rename = "message_start")]
    MessageStart { message: MessageStartData },
    #[serde(rename = "content_block_start")]
    ContentBlockStart {
        index: usize,
        content_block: ContentBlockMeta,
    },
    #[serde(rename = "content_block_delta")]
    ContentBlockDelta { index: usize, delta: Delta },
    #[serde(rename = "content_block_stop")]
    ContentBlockStop { index: usize },
    #[serde(rename = "message_delta")]
    MessageDelta { delta: MessageDeltaData },
    #[serde(rename = "message_stop")]
    MessageStop,
    #[serde(rename = "error")]
    Error { error: ApiError },
}

#[derive(Debug, Deserialize)]
#[allow(dead_code)]
struct MessageStartData {
    id: String,
    model: String,
}

#[derive(Debug, Deserialize)]
struct ContentBlockMeta {
    #[serde(rename = "type")]
    kind: String,
    /// Present for `tool_use` blocks.
    id: Option<String>,
    name: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
enum Delta {
    #[serde(rename = "text_delta")]
    Text { text: String },
    #[serde(rename = "input_json_delta")]
    InputJson { partial_json: String },
}

#[derive(Debug, Deserialize)]
struct MessageDeltaData {
    stop_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ApiError {
    message: String,
}

// ---------------------------------------------------------------------------
// Block accumulator (for streaming)
// ---------------------------------------------------------------------------

enum Block {
    Text {
        text: String,
    },
    ToolUse {
        id: String,
        name: String,
        input_json: String,
    },
}

// ---------------------------------------------------------------------------
// Conversion helpers
// ---------------------------------------------------------------------------

fn chat_message_to_api(msg: &ChatMessage) -> ApiMessage {
    let role = match msg.role {
        Role::User => "user",
        Role::Assistant => "assistant",
    };

    // If there's exactly one Text part and no other parts, use the wire shorthand.
    if msg.parts.len() == 1 {
        if let ContentPart::Text(text) = &msg.parts[0] {
            return ApiMessage {
                role: role.to_string(),
                content: ApiContent::Text(text.clone()),
            };
        }
    }

    let parts: Vec<ApiPart> = msg
        .parts
        .iter()
        .map(|p| match p {
            ContentPart::Text(t) => ApiPart::Text { text: t.clone() },
            ContentPart::Image {
                media_type,
                data_base64,
            } => ApiPart::Image {
                source: ApiImageSource::Base64 {
                    media_type: media_type.clone(),
                    data: data_base64.clone(),
                },
            },
            ContentPart::ToolUse { id, name, input } => ApiPart::ToolUse {
                id: id.clone(),
                name: name.clone(),
                input: input.clone(),
            },
            ContentPart::ToolResult {
                tool_use_id,
                content,
            } => ApiPart::ToolResult {
                tool_use_id: tool_use_id.clone(),
                content: content.clone(),
            },
        })
        .collect();

    ApiMessage {
        role: role.to_string(),
        content: ApiContent::Parts(parts),
    }
}

// ---------------------------------------------------------------------------
// Provider implementation
// ---------------------------------------------------------------------------

#[async_trait]
impl Provider for AnthropicProvider {
    fn name(&self) -> &str {
        "anthropic"
    }

    async fn chat(
        &self,
        system: Option<&str>,
        messages: &[ChatMessage],
        tools: Option<&[ToolSpec]>,
    ) -> Result<ChatResponse> {
        let api_messages: Vec<ApiMessage> = messages.iter().map(chat_message_to_api).collect();

        let api_tools: Option<Vec<ApiToolSpec<'_>>> = tools.map(|specs| {
            specs
                .iter()
                .map(|s| ApiToolSpec {
                    name: &s.name,
                    description: &s.description,
                    input_schema: &s.input_schema,
                })
                .collect()
        });

        let model = self.select_model(messages);

        let body = Request {
            model,
            max_tokens: self.max_tokens,
            system,
            messages: api_messages,
            stream: true,
            tools: api_tools,
        };

        debug!("Sending request to Anthropic API (model={model})");

        let response = self
            .client
            .post(ANTHROPIC_API_URL)
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", ANTHROPIC_VERSION)
            .header("content-type", "application/json")
            .json(&body)
            .send()
            .await
            .context("Failed to send request to Anthropic API")?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            bail!("Anthropic API error {status}: {body}");
        }

        // Parse SSE stream, tracking content blocks by index.
        let mut stream = response.bytes_stream();
        let mut buffer = String::new();
        // BTreeMap preserves insertion order by index.
        let mut blocks: BTreeMap<usize, Block> = BTreeMap::new();
        let mut stop_reason: Option<String> = None;

        while let Some(chunk) = stream.next().await {
            let chunk = chunk.context("Error reading SSE stream")?;
            buffer.push_str(&String::from_utf8_lossy(&chunk));

            while let Some(pos) = buffer.find("\n\n") {
                let event_str = buffer[..pos].to_string();
                buffer.drain(..pos + 2);

                for line in event_str.lines() {
                    let Some(data) = line.strip_prefix("data: ") else {
                        continue;
                    };
                    if data == "[DONE]" {
                        break;
                    }
                    match serde_json::from_str::<SseEvent>(data) {
                        Ok(SseEvent::ContentBlockStart {
                            index,
                            content_block,
                        }) => match content_block.kind.as_str() {
                            "text" => {
                                blocks.insert(
                                    index,
                                    Block::Text {
                                        text: String::new(),
                                    },
                                );
                            }
                            "tool_use" => {
                                blocks.insert(
                                    index,
                                    Block::ToolUse {
                                        id: content_block.id.unwrap_or_default(),
                                        name: content_block.name.unwrap_or_default(),
                                        input_json: String::new(),
                                    },
                                );
                            }
                            _ => {}
                        },
                        Ok(SseEvent::ContentBlockDelta { index, delta }) => match delta {
                            Delta::Text { text } => {
                                if let Some(Block::Text { text: acc }) = blocks.get_mut(&index) {
                                    acc.push_str(&text);
                                }
                            }
                            Delta::InputJson { partial_json } => {
                                if let Some(Block::ToolUse { input_json, .. }) =
                                    blocks.get_mut(&index)
                                {
                                    input_json.push_str(&partial_json);
                                }
                            }
                        },
                        Ok(SseEvent::MessageDelta { delta }) => {
                            if let Some(reason) = delta.stop_reason {
                                stop_reason = Some(reason);
                            }
                        }
                        Ok(SseEvent::Error { error }) => {
                            bail!("Anthropic stream error: {}", error.message);
                        }
                        Ok(_) => {}
                        Err(e) => {
                            debug!("Failed to parse SSE event: {e} | data: {data}");
                        }
                    }
                }
            }
        }

        // Assemble final response from accumulated blocks.
        let mut text_parts: Vec<String> = Vec::new();
        let mut tool_calls: Vec<ToolCall> = Vec::new();

        for block in blocks.into_values() {
            match block {
                Block::Text { text } if !text.is_empty() => {
                    text_parts.push(text);
                }
                Block::ToolUse {
                    id,
                    name,
                    input_json,
                } => {
                    let input = if input_json.is_empty() {
                        json!({})
                    } else {
                        serde_json::from_str(&input_json).unwrap_or(json!({}))
                    };
                    tool_calls.push(ToolCall { id, name, input });
                }
                _ => {}
            }
        }

        let text = if text_parts.is_empty() {
            None
        } else {
            Some(text_parts.join(""))
        };
        Ok(ChatResponse {
            text,
            tool_calls,
            stop_reason,
        })
    }
}
