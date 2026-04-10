use crate::channel::{Channel, OutgoingMessage};
use crate::config::Config;
use crate::provider::{ChatMessage, ContentPart, Provider, ToolCall};
use crate::session::{ConversationKey, SessionStore, local_date_for_timestamp};
use crate::tools::ToolSet;
use crate::workspace::Workspace;
use chrono::{Local, NaiveDate};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{Mutex, mpsc};
use tracing::{error, info, warn};

/// Maximum number of tool-call rounds per message to prevent infinite loops.
const MAX_TOOL_ROUNDS: usize = 10;

// ---------------------------------------------------------------------------
// System prompt snapshot (frozen per ConversationKey, refreshed daily)
// ---------------------------------------------------------------------------

struct SystemSnapshot {
    system_prompt: String,
    date: NaiveDate,
}

// ---------------------------------------------------------------------------
// Agent
// ---------------------------------------------------------------------------

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
    /// Per-ConversationKey system prompt snapshot, refreshed when the local date changes.
    snapshots: Mutex<HashMap<ConversationKey, SystemSnapshot>>,
    /// Background prefetch cache: workspace search results for the next turn.
    prefetch_cache: Mutex<HashMap<ConversationKey, String>>,
}

impl Agent {
    pub fn new(
        config: Config,
        channel: Arc<dyn Channel>,
        provider: Arc<dyn Provider>,
        workspace: Arc<Workspace>,
        tools: Option<Arc<ToolSet>>,
        session_store: Arc<SessionStore>,
    ) -> Self {
        let (history, active_sessions) = session_store.load_all();
        info!("Loaded {} session(s) from disk", active_sessions.len());
        Self {
            config,
            channel,
            provider,
            workspace,
            tools,
            session_store,
            history: Mutex::new(history),
            active_sessions: Mutex::new(active_sessions),
            snapshots: Mutex::new(HashMap::new()),
            prefetch_cache: Mutex::new(HashMap::new()),
        }
    }

    /// Heartbeat trigger: inject a system-style prompt as if it were an
    /// incoming message, so the agent runs through the normal handle_message
    /// pipeline (history, tool loop, channel send) without faking a user
    /// utterance. The content is wrapped with `[Heartbeat: <name>]` so the
    /// agent can recognise (via AGENTS.md / HEARTBEAT.md instructions) that
    /// this is a system trigger, not a real user message.
    pub async fn trigger(
        self: &Arc<Self>,
        task_name: &str,
        prompt: &str,
        room_id: &str,
    ) -> anyhow::Result<()> {
        let content = format!("[Heartbeat: {task_name}]\n\n{prompt}");
        let now_ms = chrono::Utc::now().timestamp_millis().max(0) as u64;
        let incoming = crate::channel::IncomingMessage {
            id: format!("heartbeat-{task_name}-{now_ms}"),
            sender: "heartbeat".to_string(),
            content,
            room_id: room_id.to_string(),
            timestamp: now_ms,
            thread_id: None,
        };
        Arc::clone(self).handle_message(incoming).await
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

    // -----------------------------------------------------------------------
    // Session helpers
    // -----------------------------------------------------------------------

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
                String::new()
            }
        }
    }

    /// Persist `msg` to the session store. No-op if session creation failed.
    fn persist(&self, session_id: &str, msg: &ChatMessage) {
        if session_id.is_empty() {
            return;
        }
        if let Err(e) = self.session_store.append(session_id, msg) {
            warn!("Failed to persist message: {e}");
        }
    }

    /// If the active session for `key` started on a different local day,
    /// close it and clear the in-memory state so a new session is created.
    async fn maybe_reset_session(&self, key: &ConversationKey) {
        let boundary = self.config.day_boundary_hour;
        let today = local_date_for_timestamp(Local::now(), boundary);

        let session_id = {
            let sessions = self.active_sessions.lock().await;
            match sessions.get(key) {
                Some(id) if !id.is_empty() => id.clone(),
                _ => return,
            }
        };

        let session_path = self
            .session_store
            .sessions_dir
            .join(format!("{session_id}.jsonl"));

        if read_session_date(&session_path, boundary) < today {
            info!("Day boundary crossed for {key:?}; resetting session");

            if let Err(e) = self.session_store.close_session(&session_id) {
                warn!("Failed to close session {session_id}: {e}");
            }

            self.history.lock().await.remove(key);
            self.active_sessions.lock().await.remove(key);
            self.snapshots.lock().await.remove(key);
            self.prefetch_cache.lock().await.remove(key);
        }
    }

    // -----------------------------------------------------------------------
    // System prompt snapshot
    // -----------------------------------------------------------------------

    /// Return the system prompt for `key`, rebuilding if the local date has changed.
    async fn get_system_prompt(&self, key: &ConversationKey) -> Option<String> {
        let boundary = self.config.day_boundary_hour;
        let today = local_date_for_timestamp(Local::now(), boundary);

        {
            let snapshots = self.snapshots.lock().await;
            if let Some(snap) = snapshots.get(key) {
                if snap.date == today {
                    return if snap.system_prompt.is_empty() {
                        None
                    } else {
                        Some(snap.system_prompt.clone())
                    };
                }
            }
        }

        let system_prompt = self
            .workspace
            .build_system_prompt(self.config.anthropic.system_prompt.as_deref())
            .await;

        self.snapshots.lock().await.insert(
            key.clone(),
            SystemSnapshot {
                system_prompt: system_prompt.clone(),
                date: today,
            },
        );

        if system_prompt.is_empty() {
            None
        } else {
            Some(system_prompt)
        }
    }

    // -----------------------------------------------------------------------
    // Message handling
    // -----------------------------------------------------------------------

    async fn handle_message(
        self: Arc<Self>,
        incoming: crate::channel::IncomingMessage,
    ) -> anyhow::Result<()> {
        info!("Message from {}: {}", incoming.sender, incoming.content);

        let key: ConversationKey = (incoming.room_id.clone(), incoming.thread_id.clone());

        // Check for day boundary → maybe reset session + snapshot
        self.maybe_reset_session(&key).await;

        let session_id = self.get_or_create_session(&key).await;

        // Get system prompt (snapshot: rebuilt at most once per local day per key)
        let system = self.get_system_prompt(&key).await;

        // Inject prefetch context from previous turn (if any)
        let prefetch_result = self.prefetch_cache.lock().await.remove(&key);
        let system_with_context = match (system, prefetch_result) {
            (Some(sys), Some(ctx)) if !ctx.is_empty() && ctx != "No results found." => Some(
                format!("{sys}\n\n---\n\n<memory-context>\n{ctx}\n</memory-context>"),
            ),
            (sys, _) => sys,
        };

        // Append user message
        {
            let msg = ChatMessage::user(&incoming.content);
            self.history
                .lock()
                .await
                .entry(key.clone())
                .or_default()
                .push(msg.clone());
            self.persist(&session_id, &msg);
        }

        let _ = self.channel.start_typing(&incoming.room_id).await;

        // Refresh MCP tools if any server signalled a change.
        if let Some(tools) = &self.tools {
            tools.refresh_if_needed().await;
        }
        let tool_specs = match &self.tools {
            Some(t) => Some(t.specs().await),
            None => None,
        };

        // Tool-calling loop
        let mut accumulated_text: Vec<String> = Vec::new();
        let final_text = loop {
            let messages = {
                self.history
                    .lock()
                    .await
                    .get(&key)
                    .cloned()
                    .unwrap_or_default()
            };

            let round = messages
                .iter()
                .filter(|m| {
                    m.parts
                        .iter()
                        .any(|p| matches!(p, ContentPart::ToolUse { .. }))
                })
                .count();

            if round >= MAX_TOOL_ROUNDS {
                warn!("Reached max tool rounds ({MAX_TOOL_ROUNDS}), stopping");
                break Some(accumulated_text.join("\n\n"));
            }

            let response = self
                .provider
                .chat(
                    system_with_context.as_deref(),
                    &messages,
                    tool_specs.as_deref(),
                )
                .await;

            match response {
                Err(e) => {
                    error!("Provider error: {e:#}");
                    let _ = self.channel.stop_typing(&incoming.room_id).await;
                    let _ = self
                        .channel
                        .send(&OutgoingMessage::new(
                            format!("⚠️ Error: {e}"),
                            incoming.room_id.clone(),
                        ))
                        .await;
                    return Ok(());
                }
                Ok(resp) if !resp.has_tool_calls() => {
                    let text = resp.text.unwrap_or_default();
                    let msg = ChatMessage::assistant(&text);
                    self.history
                        .lock()
                        .await
                        .entry(key.clone())
                        .or_default()
                        .push(msg.clone());
                    self.persist(&session_id, &msg);
                    if !text.is_empty() {
                        accumulated_text.push(text);
                    }
                    break Some(accumulated_text.join("\n\n"));
                }
                Ok(resp) => {
                    let tool_calls = resp.tool_calls.clone();
                    if let Some(t) = resp.text.as_ref().filter(|s| !s.is_empty()) {
                        accumulated_text.push(t.clone());
                    }
                    let msg =
                        ChatMessage::assistant_with_tools(resp.text.clone(), tool_calls.clone());
                    self.history
                        .lock()
                        .await
                        .entry(key.clone())
                        .or_default()
                        .push(msg.clone());
                    self.persist(&session_id, &msg);

                    // Execute tools concurrently
                    let tools = Arc::clone(self.tools.as_ref().unwrap());
                    let mut handles = Vec::with_capacity(tool_calls.len());
                    for call in tool_calls {
                        let tools = Arc::clone(&tools);
                        handles.push(tokio::spawn(async move {
                            info!("Executing tool: {} (id={})", call.name, call.id);
                            let result = tools.execute(&call).await;
                            info!("Tool {} result: {}", call.name, result);
                            (call.id, result)
                        }));
                    }

                    let mut results = Vec::with_capacity(handles.len());
                    for handle in handles {
                        match handle.await {
                            Ok(r) => results.push(r),
                            Err(e) => warn!("Tool task panicked: {e}"),
                        }
                    }

                    let msg = ChatMessage::tool_results(results);
                    self.history
                        .lock()
                        .await
                        .entry(key.clone())
                        .or_default()
                        .push(msg.clone());
                    self.persist(&session_id, &msg);
                }
            }
        };

        let _ = self.channel.stop_typing(&incoming.room_id).await;

        if let Some(text) = final_text {
            if !text.is_empty() {
                let out = OutgoingMessage {
                    content: text,
                    room_id: incoming.room_id.clone(),
                    thread_id: incoming.thread_id.clone(),
                };
                self.channel
                    .send(&out)
                    .await
                    .map_err(|e| anyhow::anyhow!("Failed to send response: {e:#}"))?;
            }

            // Spawn background prefetch for next turn
            if let Some(tools) = &self.tools {
                let tools = Arc::clone(tools);
                let agent = Arc::clone(&self);
                let key_clone = key.clone();
                let query = incoming.content.clone();
                tokio::spawn(async move {
                    let input = serde_json::json!({ "query": query, "limit": 5 });
                    let result = tools
                        .execute(&ToolCall {
                            id: "prefetch".to_string(),
                            name: "workspace_search".to_string(),
                            input,
                        })
                        .await;
                    agent.prefetch_cache.lock().await.insert(key_clone, result);
                });
            }
        }

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Read the created_at date of a session file and convert to the local day.
fn read_session_date(path: &std::path::Path, boundary_hour: u8) -> NaiveDate {
    use std::io::{BufRead, BufReader};

    let fallback = local_date_for_timestamp(Local::now(), boundary_hour);

    let file = match std::fs::File::open(path) {
        Ok(f) => f,
        Err(_) => return fallback,
    };

    let first_line = match BufReader::new(file).lines().next() {
        Some(Ok(l)) => l,
        _ => return fallback,
    };

    #[derive(serde::Deserialize)]
    struct MetaLine {
        meta: MetaCreatedAt,
    }
    #[derive(serde::Deserialize)]
    struct MetaCreatedAt {
        created_at: chrono::DateTime<chrono::Utc>,
    }

    match serde_json::from_str::<MetaLine>(&first_line) {
        Ok(ml) => {
            let local = ml.meta.created_at.with_timezone(&Local);
            local_date_for_timestamp(local, boundary_hour)
        }
        Err(_) => fallback,
    }
}
