use crate::channel::{Channel, OutgoingMessage};
use crate::config::Config;
use crate::provider::{ChatMessage, Provider};
use crate::tools::ToolSet;
use crate::workspace::Workspace;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::mpsc;
use tracing::{error, info, warn};

/// Maximum number of tool-call rounds per message to prevent infinite loops.
const MAX_TOOL_ROUNDS: usize = 10;

type ConversationKey = (String, Option<String>);

pub struct Agent {
    config: Config,
    channel: Arc<dyn Channel>,
    provider: Arc<dyn Provider>,
    workspace: Arc<Workspace>,
    tools: Option<Arc<ToolSet>>,
    history: tokio::sync::Mutex<HashMap<ConversationKey, Vec<ChatMessage>>>,
}

impl Agent {
    pub fn new(
        config: Config,
        channel: Arc<dyn Channel>,
        provider: Arc<dyn Provider>,
        workspace: Arc<Workspace>,
        tools: Option<Arc<ToolSet>>,
    ) -> Self {
        Self {
            config,
            channel,
            provider,
            workspace,
            tools,
            history: tokio::sync::Mutex::new(HashMap::new()),
        }
    }

    pub async fn run(self: Arc<Self>) -> anyhow::Result<()> {
        let (tx, mut rx) = mpsc::channel(64);

        let channel = Arc::clone(&self.channel);
        let listen_handle = tokio::spawn(async move {
            if let Err(e) = channel.listen(tx).await {
                error!("Channel listen error: {e:#}");
            }
        });

        let shutdown = tokio::signal::ctrl_c();
        tokio::pin!(shutdown);

        info!("Agent is running. Press Ctrl-C to stop.");

        loop {
            tokio::select! {
                msg = rx.recv() => {
                    match msg {
                        Some(incoming) => {
                            let agent = Arc::clone(&self);
                            tokio::spawn(async move {
                                if let Err(e) = agent.handle_message(incoming).await {
                                    error!("Error handling message: {e:#}");
                                }
                            });
                        }
                        None => {
                            warn!("Message channel closed");
                            break;
                        }
                    }
                }
                _ = &mut shutdown => {
                    info!("Shutting down...");
                    break;
                }
            }
        }

        listen_handle.abort();
        Ok(())
    }

    async fn handle_message(
        &self,
        incoming: crate::channel::IncomingMessage,
    ) -> anyhow::Result<()> {
        info!("Message from {}: {}", incoming.sender, incoming.content);

        let key: ConversationKey = (incoming.room_id.clone(), incoming.thread_id.clone());

        // Build system prompt (mtime-cached; includes AGENTS.md, SOUL.md, MEMORY.md …)
        let system_prompt = self
            .workspace
            .build_system_prompt(self.config.anthropic.system_prompt.as_deref())
            .await;
        let system = if system_prompt.is_empty() { None } else { Some(system_prompt) };

        // Append user message to history
        {
            let mut history = self.history.lock().await;
            history
                .entry(key.clone())
                .or_default()
                .push(ChatMessage::user(&incoming.content));
        }

        let _ = self.channel.start_typing(&incoming.room_id).await;

        let tool_specs = self.tools.as_ref().map(|t| t.specs().to_vec());

        // Tool-calling loop
        let final_text = loop {
            let messages = {
                let history = self.history.lock().await;
                history.get(&key).cloned().unwrap_or_default()
            };

            let round = messages
                .iter()
                .filter(|m| {
                    m.parts
                        .iter()
                        .any(|p| matches!(p, crate::provider::ContentPart::ToolUse { .. }))
                })
                .count();

            if round >= MAX_TOOL_ROUNDS {
                warn!("Reached max tool rounds ({MAX_TOOL_ROUNDS}), stopping");
                break None;
            }

            let response = self
                .provider
                .chat(
                    system.as_deref(),
                    &messages,
                    tool_specs.as_deref(),
                )
                .await;

            match response {
                Err(e) => {
                    error!("Provider error: {e:#}");
                    let _ = self.channel.stop_typing(&incoming.room_id).await;
                    let out = OutgoingMessage::new(
                        format!("⚠️ Error: {e}"),
                        incoming.room_id.clone(),
                    );
                    let _ = self.channel.send(&out).await;
                    return Ok(());
                }
                Ok(resp) if !resp.has_tool_calls() => {
                    // No more tool calls — done.
                    let text = resp.text.unwrap_or_default();
                    // Store assistant response in history
                    {
                        let mut history = self.history.lock().await;
                        history
                            .entry(key.clone())
                            .or_default()
                            .push(ChatMessage::assistant(&text));
                    }
                    break Some(text);
                }
                Ok(resp) => {
                    // Execute tool calls
                    let tool_calls = resp.tool_calls.clone();

                    // Store assistant message with tool calls in history
                    {
                        let mut history = self.history.lock().await;
                        history
                            .entry(key.clone())
                            .or_default()
                            .push(ChatMessage::assistant_with_tools(
                                resp.text.clone(),
                                tool_calls.clone(),
                            ));
                    }

                    // Execute each tool (blocking, in a spawn_blocking to avoid
                    // blocking the async runtime — workspace ops use std I/O).
                    let tools = Arc::clone(self.tools.as_ref().unwrap());
                    let results: Vec<(String, String)> = {
                        let calls = tool_calls.clone();
                        tokio::task::spawn_blocking(move || {
                            calls
                                .iter()
                                .map(|c| {
                                    info!("Executing tool: {} (id={})", c.name, c.id);
                                    let result = tools.execute(c);
                                    info!("Tool {} result: {}", c.name, result);
                                    (c.id.clone(), result)
                                })
                                .collect()
                        })
                        .await
                        .unwrap_or_default()
                    };

                    // Append tool results to history
                    {
                        let mut history = self.history.lock().await;
                        history
                            .entry(key.clone())
                            .or_default()
                            .push(ChatMessage::tool_results(results));
                    }
                }
            }
        };

        let _ = self.channel.stop_typing(&incoming.room_id).await;

        if let Some(text) = final_text {
            if !text.is_empty() {
                let out = OutgoingMessage {
                    content: text,
                    room_id: incoming.room_id,
                    thread_id: incoming.thread_id,
                };
                self.channel
                    .send(&out)
                    .await
                    .map_err(|e| anyhow::anyhow!("Failed to send response: {e:#}"))?;
            }
        }

        Ok(())
    }
}
