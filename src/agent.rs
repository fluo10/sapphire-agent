use crate::channel::{Channel, OutgoingMessage};
use crate::config::Config;
use crate::provider::{ChatMessage, Provider};
use crate::session::{ConversationKey, SessionStore};
use crate::tools::ToolSet;
use crate::workspace::Workspace;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{Mutex, mpsc};
use tracing::{error, info, warn};

/// Maximum number of tool-call rounds per message to prevent infinite loops.
const MAX_TOOL_ROUNDS: usize = 10;

pub struct Agent {
    config: Config,
    channel: Arc<dyn Channel>,
    provider: Arc<dyn Provider>,
    workspace: Arc<Workspace>,
    tools: Option<Arc<ToolSet>>,
    session_store: Arc<SessionStore>,
    /// In-memory conversation history, keyed by (room_id, thread_id).
    history: Mutex<HashMap<ConversationKey, Vec<ChatMessage>>>,
    /// Maps each ConversationKey to its current active session file (ULID string).
    active_sessions: Mutex<HashMap<ConversationKey, String>>,
}

impl Agent {
    pub fn new(
        config: Config,
        channel: Arc<dyn Channel>,
        provider: Arc<dyn Provider>,
        workspace: Arc<Workspace>,
        tools: Option<Arc<ToolSet>>,
        session_store: SessionStore,
    ) -> Self {
        let (history, active_sessions) = session_store.load_all();
        info!(
            "Loaded {} session(s) from disk",
            active_sessions.len()
        );
        Self {
            config,
            channel,
            provider,
            workspace,
            tools,
            session_store: Arc::new(session_store),
            history: Mutex::new(history),
            active_sessions: Mutex::new(active_sessions),
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

    /// Return the active session_id for `key`, creating a new session file if needed.
    async fn get_or_create_session(&self, key: &ConversationKey) -> String {
        let mut sessions = self.active_sessions.lock().await;
        if let Some(id) = sessions.get(key) {
            return id.clone();
        }
        let channel_name = self.channel.name().to_string();
        match self.session_store.create_session(key, &channel_name) {
            Ok(id) => {
                sessions.insert(key.clone(), id.clone());
                id
            }
            Err(e) => {
                warn!("Failed to create session file: {e}");
                // Return a sentinel so callers can still run without persistence
                String::new()
            }
        }
    }

    /// Persist `msg` to the session store for `key`. No-op if session creation failed.
    fn persist(&self, session_id: &str, msg: &ChatMessage) {
        if session_id.is_empty() {
            return;
        }
        if let Err(e) = self.session_store.append(session_id, msg) {
            warn!("Failed to persist message: {e}");
        }
    }

    async fn handle_message(
        &self,
        incoming: crate::channel::IncomingMessage,
    ) -> anyhow::Result<()> {
        info!("Message from {}: {}", incoming.sender, incoming.content);

        let key: ConversationKey = (incoming.room_id.clone(), incoming.thread_id.clone());
        let session_id = self.get_or_create_session(&key).await;

        // Build system prompt (mtime-cached; includes AGENTS.md, SOUL.md, MEMORY.md …)
        let system_prompt = self
            .workspace
            .build_system_prompt(self.config.anthropic.system_prompt.as_deref())
            .await;
        let system = if system_prompt.is_empty() { None } else { Some(system_prompt) };

        // Append user message to history and session store
        {
            let msg = ChatMessage::user(&incoming.content);
            self.history.lock().await.entry(key.clone()).or_default().push(msg.clone());
            self.persist(&session_id, &msg);
        }

        let _ = self.channel.start_typing(&incoming.room_id).await;

        let tool_specs = self.tools.as_ref().map(|t| t.specs().to_vec());

        // Tool-calling loop
        let final_text = loop {
            let messages = {
                self.history.lock().await.get(&key).cloned().unwrap_or_default()
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
                .chat(system.as_deref(), &messages, tool_specs.as_deref())
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
                    let text = resp.text.unwrap_or_default();
                    let msg = ChatMessage::assistant(&text);
                    self.history.lock().await.entry(key.clone()).or_default().push(msg.clone());
                    self.persist(&session_id, &msg);
                    break Some(text);
                }
                Ok(resp) => {
                    let tool_calls = resp.tool_calls.clone();

                    let msg = ChatMessage::assistant_with_tools(resp.text.clone(), tool_calls.clone());
                    self.history.lock().await.entry(key.clone()).or_default().push(msg.clone());
                    self.persist(&session_id, &msg);

                    // Execute each tool (spawn_blocking — workspace ops use std I/O)
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

                    let msg = ChatMessage::tool_results(results);
                    self.history.lock().await.entry(key.clone()).or_default().push(msg.clone());
                    self.persist(&session_id, &msg);
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
