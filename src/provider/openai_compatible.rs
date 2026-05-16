//! OpenAI-compatible chat completions provider.
//!
//! Targets `POST {base_url}/chat/completions` with SSE streaming. Works
//! against llama.cpp's `llama-server`, Ollama (`/v1`), vLLM, and the OpenAI
//! API itself. Tool calls follow the OpenAI `tools` / `tool_calls` shape.

use crate::provider::{ChatMessage, ChatResponse, ContentPart, Provider, Role, ToolCall, ToolSpec};
use anyhow::{Context, Result, bail};
use async_trait::async_trait;
use futures_util::StreamExt;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::BTreeMap;
use tracing::debug;

/// Configuration for an OpenAI-compatible endpoint.
///
/// Not yet wired into the top-level `Config` — that's part of the
/// multi-provider routing work. Construct directly for now.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct OpenAICompatibleConfig {
    /// Base URL up to and including the API version segment, e.g.
    /// `http://127.0.0.1:8080/v1` for llama.cpp or
    /// `https://api.openai.com/v1` for OpenAI proper.
    pub base_url: String,
    /// Bearer token. Optional — llama.cpp's local server does not require one.
    #[serde(default)]
    pub api_key: Option<String>,
    /// Model identifier passed in the request body. For llama.cpp this can
    /// be any string the server accepts (often ignored when only one model
    /// is loaded).
    pub model: String,
    /// Provider name surfaced via `Provider::name()`. Defaults to
    /// `"openai_compatible"`.
    #[serde(default)]
    pub provider_name: Option<String>,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,
}

fn default_max_tokens() -> u32 {
    8192
}

pub struct OpenAICompatibleProvider {
    base_url: String,
    api_key: Option<String>,
    model: String,
    name: String,
    max_tokens: u32,
    client: Client,
}

impl OpenAICompatibleProvider {
    pub fn new(cfg: &OpenAICompatibleConfig) -> Self {
        let base_url = cfg.base_url.trim_end_matches('/').to_string();
        Self {
            base_url,
            api_key: cfg.api_key.clone(),
            model: cfg.model.clone(),
            name: cfg
                .provider_name
                .clone()
                .unwrap_or_else(|| "openai_compatible".to_string()),
            max_tokens: cfg.max_tokens,
            client: Client::new(),
        }
    }

    fn endpoint(&self) -> String {
        format!("{}/chat/completions", self.base_url)
    }
}

// ---------------------------------------------------------------------------
// Wire-format types — request
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize)]
struct Request<'a> {
    model: &'a str,
    messages: Vec<ApiMessage>,
    max_tokens: u32,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<ApiToolSpec<'a>>>,
}

#[derive(Debug, Serialize)]
struct ApiToolSpec<'a> {
    #[serde(rename = "type")]
    kind: &'static str,
    function: ApiToolFunction<'a>,
}

#[derive(Debug, Serialize)]
struct ApiToolFunction<'a> {
    name: &'a str,
    description: &'a str,
    parameters: &'a Value,
}

#[derive(Debug, Serialize)]
struct ApiMessage {
    role: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<ApiContent>,
    /// Present on assistant messages that emit tool calls.
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<ApiAssistantToolCall>>,
    /// Present on `tool` role messages — links the result to a prior call.
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
}

#[derive(Debug, Serialize)]
#[serde(untagged)]
enum ApiContent {
    /// Plain string — used for simple text-only messages and tool results.
    Text(String),
    /// Array of typed parts — required when images are present.
    Parts(Vec<ApiPart>),
}

#[derive(Debug, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
enum ApiPart {
    Text { text: String },
    ImageUrl { image_url: ApiImageUrl },
}

#[derive(Debug, Serialize)]
struct ApiImageUrl {
    /// `data:<media_type>;base64,<data>` for inline images.
    url: String,
}

#[derive(Debug, Serialize)]
struct ApiAssistantToolCall {
    id: String,
    #[serde(rename = "type")]
    kind: &'static str,
    function: ApiAssistantToolFunction,
}

#[derive(Debug, Serialize)]
struct ApiAssistantToolFunction {
    name: String,
    /// JSON-encoded string of the arguments object.
    arguments: String,
}

// ---------------------------------------------------------------------------
// Wire-format types — streaming response
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct StreamChunk {
    #[serde(default)]
    choices: Vec<StreamChoice>,
}

#[derive(Debug, Deserialize)]
struct StreamChoice {
    #[serde(default)]
    delta: StreamDelta,
    /// Populated on the final chunk: `"stop"`, `"length"`,
    /// `"content_filter"`, `"tool_calls"`, etc.
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize, Default)]
struct StreamDelta {
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    tool_calls: Option<Vec<StreamToolCallDelta>>,
}

#[derive(Debug, Deserialize)]
struct StreamToolCallDelta {
    /// Position within the assistant message's tool_calls array. Used as the
    /// accumulator key — the same call streams in across multiple chunks at
    /// the same `index`.
    index: usize,
    #[serde(default)]
    id: Option<String>,
    #[serde(default)]
    function: Option<StreamFunctionDelta>,
}

#[derive(Debug, Deserialize)]
struct StreamFunctionDelta {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    arguments: Option<String>,
}

// ---------------------------------------------------------------------------
// Conversion: ChatMessage -> ApiMessage(s)
// ---------------------------------------------------------------------------

/// Convert one logical `ChatMessage` into one or more wire-level `ApiMessage`s.
///
/// OpenAI's protocol splits tool results into their own `role: "tool"`
/// messages — one per result — whereas the internal `ChatMessage` collapses
/// them into a single User message with multiple `ToolResult` parts.
fn chat_message_to_api(msg: &ChatMessage) -> Vec<ApiMessage> {
    match msg.role {
        Role::User => convert_user_message(msg),
        Role::Assistant => vec![convert_assistant_message(msg)],
    }
}

fn convert_user_message(msg: &ChatMessage) -> Vec<ApiMessage> {
    let mut out = Vec::new();
    let mut text_parts: Vec<ApiPart> = Vec::new();

    for part in &msg.parts {
        match part {
            ContentPart::Text(t) => text_parts.push(ApiPart::Text { text: t.clone() }),
            ContentPart::Image {
                media_type,
                data_base64,
            } => text_parts.push(ApiPart::ImageUrl {
                image_url: ApiImageUrl {
                    url: format!("data:{media_type};base64,{data_base64}"),
                },
            }),
            ContentPart::ToolResult {
                tool_use_id,
                content,
            } => {
                out.push(ApiMessage {
                    role: "tool",
                    content: Some(ApiContent::Text(content.clone())),
                    tool_calls: None,
                    tool_call_id: Some(tool_use_id.clone()),
                });
            }
            ContentPart::ToolUse { .. } => {
                // Should not appear on User messages — silently skip.
            }
        }
    }

    if !text_parts.is_empty() {
        let content = if text_parts.len() == 1 {
            if let ApiPart::Text { text } = &text_parts[0] {
                ApiContent::Text(text.clone())
            } else {
                ApiContent::Parts(text_parts)
            }
        } else {
            ApiContent::Parts(text_parts)
        };
        out.push(ApiMessage {
            role: "user",
            content: Some(content),
            tool_calls: None,
            tool_call_id: None,
        });
    }

    out
}

fn convert_assistant_message(msg: &ChatMessage) -> ApiMessage {
    let mut text: Option<String> = None;
    let mut tool_calls: Vec<ApiAssistantToolCall> = Vec::new();

    for part in &msg.parts {
        match part {
            ContentPart::Text(t) => {
                text.get_or_insert_with(String::new).push_str(t);
            }
            ContentPart::ToolUse { id, name, input } => {
                tool_calls.push(ApiAssistantToolCall {
                    id: id.clone(),
                    kind: "function",
                    function: ApiAssistantToolFunction {
                        name: name.clone(),
                        arguments: serde_json::to_string(input)
                            .unwrap_or_else(|_| "{}".to_string()),
                    },
                });
            }
            // Assistant should not carry images or tool results.
            ContentPart::Image { .. } | ContentPart::ToolResult { .. } => {}
        }
    }

    ApiMessage {
        role: "assistant",
        content: text.map(ApiContent::Text),
        tool_calls: if tool_calls.is_empty() {
            None
        } else {
            Some(tool_calls)
        },
        tool_call_id: None,
    }
}

// ---------------------------------------------------------------------------
// Streaming accumulator
// ---------------------------------------------------------------------------

#[derive(Default)]
struct ToolCallAccum {
    id: String,
    name: String,
    arguments: String,
}

// ---------------------------------------------------------------------------
// Provider impl
// ---------------------------------------------------------------------------

#[async_trait]
impl Provider for OpenAICompatibleProvider {
    fn name(&self) -> &str {
        &self.name
    }

    async fn chat(
        &self,
        system: Option<&str>,
        messages: &[ChatMessage],
        tools: Option<&[ToolSpec]>,
    ) -> Result<ChatResponse> {
        let mut api_messages: Vec<ApiMessage> = Vec::new();
        if let Some(sys) = system {
            api_messages.push(ApiMessage {
                role: "system",
                content: Some(ApiContent::Text(sys.to_string())),
                tool_calls: None,
                tool_call_id: None,
            });
        }
        for m in messages {
            api_messages.extend(chat_message_to_api(m));
        }

        let api_tools: Option<Vec<ApiToolSpec<'_>>> = tools.map(|specs| {
            specs
                .iter()
                .map(|s| ApiToolSpec {
                    kind: "function",
                    function: ApiToolFunction {
                        name: &s.name,
                        description: &s.description,
                        parameters: &s.input_schema,
                    },
                })
                .collect()
        });

        let body = Request {
            model: &self.model,
            messages: api_messages,
            max_tokens: self.max_tokens,
            stream: true,
            tools: api_tools,
        };

        let url = self.endpoint();
        debug!(
            "Sending request to OpenAI-compatible endpoint (url={url}, model={})",
            self.model
        );

        let mut req = self
            .client
            .post(&url)
            .header("content-type", "application/json")
            .json(&body);
        if let Some(key) = &self.api_key {
            req = req.bearer_auth(key);
        }

        let response = req
            .send()
            .await
            .with_context(|| format!("Failed to send request to {url}"))?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await.unwrap_or_default();
            bail!("OpenAI-compatible API error {status}: {body}");
        }

        // Parse SSE stream.
        let mut stream = response.bytes_stream();
        let mut buffer = String::new();
        let mut text_acc = String::new();
        let mut tool_acc: BTreeMap<usize, ToolCallAccum> = BTreeMap::new();
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
                    let data = data.trim();
                    if data == "[DONE]" {
                        break;
                    }
                    let parsed: StreamChunk = match serde_json::from_str(data) {
                        Ok(v) => v,
                        Err(e) => {
                            debug!("Failed to parse SSE chunk: {e} | data: {data}");
                            continue;
                        }
                    };
                    for choice in parsed.choices {
                        if let Some(reason) = choice.finish_reason {
                            stop_reason = Some(reason);
                        }
                        if let Some(t) = choice.delta.content {
                            text_acc.push_str(&t);
                        }
                        if let Some(deltas) = choice.delta.tool_calls {
                            for d in deltas {
                                let entry = tool_acc.entry(d.index).or_default();
                                if let Some(id) = d.id
                                    && !id.is_empty()
                                {
                                    entry.id = id;
                                }
                                if let Some(f) = d.function {
                                    if let Some(n) = f.name
                                        && !n.is_empty()
                                    {
                                        entry.name = n;
                                    }
                                    if let Some(a) = f.arguments {
                                        entry.arguments.push_str(&a);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        let tool_calls: Vec<ToolCall> = tool_acc
            .into_values()
            .filter(|t| !t.name.is_empty())
            .map(|t| {
                let input: Value = if t.arguments.is_empty() {
                    json!({})
                } else {
                    serde_json::from_str(&t.arguments).unwrap_or(json!({}))
                };
                ToolCall {
                    id: t.id,
                    name: t.name,
                    input,
                }
            })
            .collect();

        let text = if text_acc.is_empty() {
            None
        } else {
            Some(text_acc)
        };

        Ok(ChatResponse {
            text,
            tool_calls,
            stop_reason,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::ChatMessage;

    #[test]
    fn user_text_becomes_string_content() {
        let msg = ChatMessage::user("hello");
        let api = chat_message_to_api(&msg);
        assert_eq!(api.len(), 1);
        assert_eq!(api[0].role, "user");
        let json = serde_json::to_value(&api[0]).unwrap();
        assert_eq!(json["content"], "hello");
    }

    #[test]
    fn user_with_image_uses_parts_array() {
        let msg = ChatMessage::user_with_images(
            "describe this",
            vec![("image/png".to_string(), "AAAA".to_string())],
        );
        let api = chat_message_to_api(&msg);
        assert_eq!(api.len(), 1);
        let json = serde_json::to_value(&api[0]).unwrap();
        let parts = json["content"].as_array().expect("content should be array");
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0]["type"], "image_url");
        assert_eq!(parts[0]["image_url"]["url"], "data:image/png;base64,AAAA");
        assert_eq!(parts[1]["type"], "text");
        assert_eq!(parts[1]["text"], "describe this");
    }

    #[test]
    fn tool_results_split_into_separate_tool_messages() {
        let msg = ChatMessage::tool_results(vec![
            ("call_a".to_string(), "result_a".to_string()),
            ("call_b".to_string(), "result_b".to_string()),
        ]);
        let api = chat_message_to_api(&msg);
        assert_eq!(api.len(), 2);
        assert_eq!(api[0].role, "tool");
        assert_eq!(api[0].tool_call_id.as_deref(), Some("call_a"));
        assert_eq!(api[1].role, "tool");
        assert_eq!(api[1].tool_call_id.as_deref(), Some("call_b"));
    }

    #[test]
    fn assistant_with_tool_use_serializes_tool_calls() {
        let msg = ChatMessage::assistant_with_tools(
            Some("I'll check.".to_string()),
            vec![ToolCall {
                id: "call_1".to_string(),
                name: "get_weather".to_string(),
                input: json!({"city": "Tokyo"}),
            }],
        );
        let api = chat_message_to_api(&msg);
        assert_eq!(api.len(), 1);
        let json = serde_json::to_value(&api[0]).unwrap();
        assert_eq!(json["role"], "assistant");
        assert_eq!(json["content"], "I'll check.");
        let calls = json["tool_calls"].as_array().unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(calls[0]["id"], "call_1");
        assert_eq!(calls[0]["function"]["name"], "get_weather");
        // arguments is a JSON-encoded string per OpenAI spec.
        let args: Value =
            serde_json::from_str(calls[0]["function"]["arguments"].as_str().unwrap()).unwrap();
        assert_eq!(args, json!({"city": "Tokyo"}));
    }

    #[test]
    fn endpoint_strips_trailing_slash() {
        let p = OpenAICompatibleProvider::new(&OpenAICompatibleConfig {
            base_url: "http://localhost:8080/v1/".to_string(),
            api_key: None,
            model: "gemma".to_string(),
            provider_name: None,
            max_tokens: 4096,
        });
        assert_eq!(p.endpoint(), "http://localhost:8080/v1/chat/completions");
    }

    #[test]
    fn provider_name_default_and_override() {
        let default = OpenAICompatibleProvider::new(&OpenAICompatibleConfig {
            base_url: "http://x/v1".to_string(),
            api_key: None,
            model: "m".to_string(),
            provider_name: None,
            max_tokens: 1,
        });
        assert_eq!(default.name(), "openai_compatible");

        let custom = OpenAICompatibleProvider::new(&OpenAICompatibleConfig {
            base_url: "http://x/v1".to_string(),
            api_key: None,
            model: "m".to_string(),
            provider_name: Some("llama_cpp".to_string()),
            max_tokens: 1,
        });
        assert_eq!(custom.name(), "llama_cpp");
    }

    #[test]
    fn role_is_assistant_when_no_text_only_tool_use() {
        let msg = ChatMessage::assistant_with_tools(
            None,
            vec![ToolCall {
                id: "c".into(),
                name: "n".into(),
                input: json!({}),
            }],
        );
        let api = chat_message_to_api(&msg);
        let json = serde_json::to_value(&api[0]).unwrap();
        assert_eq!(json["role"], "assistant");
        assert!(json.get("content").is_none() || json["content"].is_null());
        assert!(json["tool_calls"].is_array());
    }
}
