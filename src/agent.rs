use crate::channel::{Channel, OutgoingMessage};
use crate::config::Config;
use crate::provider::{ChatMessage, Provider};
use crate::workspace::Workspace;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::mpsc;
use tracing::{error, info, warn};

/// Conversation history keyed by (room_id, thread_id_or_none).
type ConversationKey = (String, Option<String>);

pub struct Agent {
    config: Config,
    channel: Arc<dyn Channel>,
    provider: Arc<dyn Provider>,
    workspace: Arc<Workspace>,
    /// In-memory conversation history per room/thread.
    history: tokio::sync::Mutex<HashMap<ConversationKey, Vec<ChatMessage>>>,
}

impl Agent {
    pub fn new(
        config: Config,
        channel: Arc<dyn Channel>,
        provider: Arc<dyn Provider>,
        workspace: Arc<Workspace>,
    ) -> Self {
        Self {
            config,
            channel,
            provider,
            workspace,
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

        // Build system prompt fresh on each message so edits to AGENT.md /
        // MEMORY.md take effect immediately (files are mtime-cached).
        let system_prompt = self
            .workspace
            .build_system_prompt(self.config.anthropic.system_prompt.as_deref())
            .await;
        let system = if system_prompt.is_empty() {
            None
        } else {
            Some(system_prompt)
        };

        // Append user message to history
        {
            let mut history = self.history.lock().await;
            let conv = history.entry(key.clone()).or_default();
            conv.push(ChatMessage::user(&incoming.content));
        }

        let _ = self.channel.start_typing(&incoming.room_id).await;

        let messages = {
            let history = self.history.lock().await;
            history.get(&key).cloned().unwrap_or_default()
        };

        let response = self.provider.chat(system.as_deref(), &messages).await;

        let _ = self.channel.stop_typing(&incoming.room_id).await;

        match response {
            Ok(resp) => {
                {
                    let mut history = self.history.lock().await;
                    let conv = history.entry(key).or_default();
                    conv.push(ChatMessage::assistant(&resp.text));
                }

                let outgoing = OutgoingMessage {
                    content: resp.text,
                    room_id: incoming.room_id,
                    thread_id: incoming.thread_id,
                };
                self.channel
                    .send(&outgoing)
                    .await
                    .map_err(|e| anyhow::anyhow!("Failed to send response: {e:#}"))?;
            }
            Err(e) => {
                error!("Provider error: {e:#}");
                let outgoing = OutgoingMessage::new(format!("⚠️ Error: {e}"), incoming.room_id);
                let _ = self.channel.send(&outgoing).await;
            }
        }

        Ok(())
    }
}
