use crate::config::AnthropicConfig;
use crate::provider::{ChatMessage, ChatResponse, Provider, Role};
use anyhow::{Context, Result, bail};
use async_trait::async_trait;
use futures_util::StreamExt;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use tracing::debug;

const ANTHROPIC_API_URL: &str = "https://api.anthropic.com/v1/messages";
const ANTHROPIC_VERSION: &str = "2023-06-01";

pub struct AnthropicProvider {
    api_key: String,
    model: String,
    max_tokens: u32,
    client: Client,
}

impl AnthropicProvider {
    pub fn new(cfg: &AnthropicConfig) -> Self {
        Self {
            api_key: cfg.api_key.clone(),
            model: cfg.model.clone(),
            max_tokens: cfg.max_tokens,
            client: Client::new(),
        }
    }
}

// ---------- Request types ----------

#[derive(Debug, Serialize)]
struct Request<'a> {
    model: &'a str,
    max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<&'a str>,
    messages: Vec<ApiMessage<'a>>,
    stream: bool,
}

#[derive(Debug, Serialize)]
struct ApiMessage<'a> {
    role: &'a str,
    content: &'a str,
}

// ---------- Response types ----------

#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
#[allow(dead_code)]
enum SseEvent {
    #[serde(rename = "message_start")]
    MessageStart { message: MessageStartData },
    #[serde(rename = "content_block_start")]
    ContentBlockStart {
        index: usize,
        content_block: ContentBlock,
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
#[allow(dead_code)]
struct ContentBlock {
    #[serde(rename = "type")]
    kind: String,
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
#[allow(dead_code)]
struct MessageDeltaData {
    stop_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ApiError {
    message: String,
}

// ---------- Provider implementation ----------

#[async_trait]
impl Provider for AnthropicProvider {
    fn name(&self) -> &str {
        "anthropic"
    }

    async fn chat(
        &self,
        system: Option<&str>,
        messages: &[ChatMessage],
    ) -> Result<ChatResponse> {
        let api_messages: Vec<ApiMessage<'_>> = messages
            .iter()
            .map(|m| ApiMessage {
                role: match m.role {
                    Role::User => "user",
                    Role::Assistant => "assistant",
                },
                content: &m.content,
            })
            .collect();

        let body = Request {
            model: &self.model,
            max_tokens: self.max_tokens,
            system,
            messages: api_messages,
            stream: true,
        };

        debug!("Sending request to Anthropic API (model={})", self.model);

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

        // Parse SSE stream
        let mut stream = response.bytes_stream();
        let mut accumulated = String::new();
        let mut buffer = String::new();

        while let Some(chunk) = stream.next().await {
            let chunk = chunk.context("Error reading SSE stream")?;
            buffer.push_str(&String::from_utf8_lossy(&chunk));

            // SSE events are separated by double newlines
            while let Some(pos) = buffer.find("\n\n") {
                let event_str = buffer[..pos].to_string();
                buffer.drain(..pos + 2);

                for line in event_str.lines() {
                    if let Some(data) = line.strip_prefix("data: ") {
                        if data == "[DONE]" {
                            break;
                        }
                        match serde_json::from_str::<SseEvent>(data) {
                            Ok(SseEvent::ContentBlockDelta {
                                delta: Delta::Text { text },
                                ..
                            }) => {
                                accumulated.push_str(&text);
                            }
                            Ok(SseEvent::Error { error }) => {
                                bail!("Anthropic stream error: {}", error.message);
                            }
                            Ok(_) => {} // other events ignored
                            Err(e) => {
                                debug!("Failed to parse SSE event: {e} | data: {data}");
                            }
                        }
                    }
                }
            }
        }

        Ok(ChatResponse { text: accumulated })
    }
}
