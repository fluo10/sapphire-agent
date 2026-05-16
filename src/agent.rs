use crate::channel::{Attachment, Channels, OutgoingMessage};
use crate::config::{Config, SessionPolicy};
use crate::context_compression::{generate_summary, maybe_compress};
use crate::provider::registry::ProviderRegistry;
use crate::provider::{ChatMessage, ContentPart, Provider, Role, ToolCall};
use crate::session::{ConversationKey, SessionStore, local_date_for_timestamp};
use crate::tools::ToolSet;
use crate::workspace::Workspace;
use chrono::{Local, NaiveDate};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::{Mutex, mpsc};
use tracing::{debug, error, info, warn};

/// Maximum number of tool-call rounds per message to prevent infinite loops.
const MAX_TOOL_ROUNDS: usize = 10;

impl Agent {
    /// Resolve the provider that should serve `room_id`, honouring the
    /// room → profile → provider mapping defined in `Config`. Falls back to
    /// the Anthropic provider when no profile applies.
    fn provider_for(&self, room_id: &str) -> Arc<dyn Provider> {
        self.providers.for_room(&self.config, room_id)
    }
}

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
    channels: Arc<Channels>,
    providers: Arc<ProviderRegistry>,
    workspace: Arc<Workspace>,
    tools: Option<Arc<ToolSet>>,
    session_store: Arc<SessionStore>,
    /// In-memory conversation history, keyed by (room_id, thread_id).
    /// Starts empty on process startup; raw history from disk is never reloaded
    /// (see `restart_summaries` for how prior context is carried across restarts).
    history: Mutex<HashMap<ConversationKey, Vec<ChatMessage>>>,
    /// Maps each ConversationKey to its current active session file (ULID string).
    active_sessions: Mutex<HashMap<ConversationKey, String>>,
    /// Per-ConversationKey system prompt snapshot, refreshed when the local date changes.
    snapshots: Mutex<HashMap<ConversationKey, SystemSnapshot>>,
    /// Background prefetch cache: workspace search results for the next turn.
    prefetch_cache: Mutex<HashMap<ConversationKey, String>>,
    /// Compacted recap of the prior run per ConversationKey. Injected into the
    /// system prompt on the first message after restart, then consumed.
    restart_summaries: Mutex<HashMap<ConversationKey, String>>,
    /// Active sessions whose on-disk history has no `SummaryLine` yet (e.g.
    /// server crashed before graceful shutdown). `bootstrap()` synthesizes a
    /// summary from these raw messages and moves the result into
    /// `restart_summaries`.
    pending_fallback: Mutex<HashMap<ConversationKey, Vec<ChatMessage>>>,
    /// Local date on which the last day-boundary action fired for each key.
    /// Prevents re-firing within the same day for policies that don't rotate
    /// the session file (Compact, None).
    boundary_handled: Mutex<HashMap<ConversationKey, NaiveDate>>,
    /// Wall-clock of the latest activity (user message or our reply) per
    /// conversation. Used by the idle-flush task to decide when to emit a
    /// same-day digest summarising what happened in this session for
    /// cross-session injection.
    last_activity_at: Mutex<HashMap<ConversationKey, chrono::DateTime<chrono::Utc>>>,
    /// Mark of the latest `last_activity_at` value at which we already
    /// flushed an intra-day digest for this key. Prevents repeated flushes
    /// firing on the same idle window — only a fresh activity bump
    /// reopens the window.
    last_flushed_at: Mutex<HashMap<ConversationKey, chrono::DateTime<chrono::Utc>>>,
}

impl Agent {
    pub fn new(
        config: Config,
        channels: Arc<Channels>,
        providers: Arc<ProviderRegistry>,
        workspace: Arc<Workspace>,
        tools: Option<Arc<ToolSet>>,
        session_store: Arc<SessionStore>,
    ) -> Self {
        let (active_sessions, summaries, fallback) = session_store.load_all();
        info!(
            "Loaded {} session(s) from disk ({} with summary, {} awaiting fallback summarization)",
            active_sessions.len(),
            summaries.len(),
            fallback.len(),
        );
        Self {
            config,
            channels,
            providers,
            workspace,
            tools,
            session_store,
            history: Mutex::new(HashMap::new()),
            active_sessions: Mutex::new(active_sessions),
            snapshots: Mutex::new(HashMap::new()),
            prefetch_cache: Mutex::new(HashMap::new()),
            restart_summaries: Mutex::new(summaries),
            pending_fallback: Mutex::new(fallback),
            boundary_handled: Mutex::new(HashMap::new()),
            last_activity_at: Mutex::new(HashMap::new()),
            last_flushed_at: Mutex::new(HashMap::new()),
        }
    }

    /// Drain `pending_fallback` and synthesize summaries for any active session
    /// whose prior run did not produce a `SummaryLine` (e.g. crash). Each
    /// generated summary is appended to the JSONL file and placed into
    /// `restart_summaries` for the next turn.
    ///
    /// Only sessions whose room_id is in the configured room/channel list are
    /// processed. Sessions from unconfigured rooms (e.g. test rooms that share
    /// the same sessions directory) are left as-is so a production restart
    /// never triggers LLM calls for rooms this instance doesn't own.
    /// Exception: if no room IDs are configured (Discord with channel_ids = []),
    /// all sessions are processed (preserving the "listen to all channels"
    /// semantics).
    pub async fn bootstrap(self: &Arc<Self>) {
        let pending: Vec<(ConversationKey, Vec<ChatMessage>)> = {
            let mut map = self.pending_fallback.lock().await;
            map.drain().collect()
        };

        if pending.is_empty() {
            return;
        }

        // Build the set of room/channel IDs this instance is configured
        // to handle. If any configured channel uses the "listen
        // everywhere" wildcard (Discord with channel_ids = []), treat
        // the whole agent as unrestricted — narrowing to only the other
        // channel's explicit set would silently drop everything that
        // channel was supposed to handle.
        let mut wildcard = false;
        let allowed: std::collections::HashSet<String> = {
            let mut set = std::collections::HashSet::new();
            if let Some(m) = &self.config.matrix {
                if m.room_ids.is_empty() {
                    wildcard = true;
                } else {
                    set.extend(m.room_ids.iter().cloned());
                }
            }
            if let Some(d) = &self.config.discord {
                if d.channel_ids.is_empty() {
                    wildcard = true;
                } else {
                    set.extend(d.channel_ids.iter().cloned());
                }
            }
            set
        };

        let (to_process, skipped): (Vec<(ConversationKey, Vec<ChatMessage>)>, usize) =
            if wildcard || allowed.is_empty() {
                // No restriction configured — process all sessions.
                (pending, 0)
            } else {
                let (yes, no): (Vec<_>, Vec<_>) = pending
                    .into_iter()
                    .partition(|(key, _)| allowed.contains(&key.0));
                let skipped = no.len();
                (yes, skipped)
            };

        if skipped > 0 {
            info!(
                "Bootstrap: skipping {} session(s) in unconfigured rooms",
                skipped
            );
        }
        if to_process.is_empty() {
            return;
        }

        info!(
            "Bootstrap: synthesizing summaries for {} session(s) without a SummaryLine",
            to_process.len()
        );

        for (key, messages) in to_process {
            if messages.len() < 2 {
                continue;
            }
            let session_id = {
                let sessions = self.active_sessions.lock().await;
                match sessions.get(&key) {
                    Some(id) if !id.is_empty() => id.clone(),
                    _ => continue,
                }
            };
            let provider = self.provider_for(&key.0);
            match generate_summary(&*provider, &messages).await {
                Ok(summary) if !summary.trim().is_empty() => {
                    if let Err(e) = self.session_store.append_summary(&session_id, &summary) {
                        warn!("Failed to persist fallback summary for {session_id}: {e}");
                    }
                    self.restart_summaries.lock().await.insert(key, summary);
                }
                Ok(_) => warn!("Fallback summary for {session_id} was empty; skipping"),
                Err(e) => warn!("Fallback summary generation failed for {session_id}: {e:#}"),
            }
        }
    }

    /// Summarize each active session's current in-memory history and persist
    /// the result to disk. Called on graceful shutdown so the next process
    /// can pick up where this one left off.
    async fn summarize_on_shutdown(&self) {
        let snapshot: Vec<(ConversationKey, String, Vec<ChatMessage>)> = {
            let history = self.history.lock().await;
            let sessions = self.active_sessions.lock().await;
            history
                .iter()
                .filter_map(|(key, msgs)| {
                    if msgs.len() < 2 {
                        return None;
                    }
                    let sid = sessions.get(key)?.clone();
                    if sid.is_empty() {
                        return None;
                    }
                    Some((key.clone(), sid, msgs.clone()))
                })
                .collect()
        };

        if snapshot.is_empty() {
            return;
        }

        info!(
            "Graceful shutdown: summarizing {} active session(s)",
            snapshot.len()
        );

        for (key, session_id, messages) in snapshot {
            let provider = self.provider_for(&key.0);
            match generate_summary(&*provider, &messages).await {
                Ok(summary) if !summary.trim().is_empty() => {
                    if let Err(e) = self.session_store.append_summary(&session_id, &summary) {
                        warn!("Failed to persist shutdown summary for {session_id}: {e}");
                    }
                    // Also publish an intra-day digest line so the
                    // cross-session today_digest picks up what this
                    // session covered before we went down.
                    if let Err(e) =
                        self.session_store
                            .append_intraday_digest(&session_id, &summary, None)
                    {
                        warn!("Failed to persist shutdown intra-day digest for {session_id}: {e}");
                    }
                }
                Ok(_) => warn!("Shutdown summary for {session_id} was empty; skipping"),
                Err(e) => warn!("Shutdown summary generation failed for {session_id}: {e:#}"),
            }
        }
    }

    /// Drop all cached system-prompt snapshots so the next message rebuilds
    /// from disk. Call this after writing any file that `build_system_prompt`
    /// reads (e.g. a freshly generated daily log) so the change is visible to
    /// the model without waiting for the next day-boundary cache miss.
    pub async fn invalidate_system_prompts(&self) {
        self.snapshots.lock().await.clear();
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
            attachments: Vec::new(),
        };
        Arc::clone(self).handle_message(incoming).await
    }

    pub async fn run(self: Arc<Self>) -> anyhow::Result<()> {
        let (tx, mut rx) = mpsc::channel(64);

        let channels = Arc::clone(&self.channels);
        let listen_handle = tokio::spawn(async move {
            if let Err(e) = channels.listen_all(tx).await {
                error!("Channel listen error: {e:#}");
            }
        });

        // Idle-flush task: every minute, scan last_activity_at and emit a
        // same-day digest line for any conversation that's been quiet long
        // enough. Disabled by `intraday_idle_minutes = 0`.
        let idle_flush_handle = self
            .config
            .intraday_idle_threshold_minutes()
            .map(|threshold| {
                let agent = Arc::clone(&self);
                tokio::spawn(async move { agent.run_idle_flush_loop(threshold).await })
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

        self.summarize_on_shutdown().await;
        listen_handle.abort();
        if let Some(h) = idle_flush_handle {
            h.abort();
        }
        Ok(())
    }

    /// Background loop that wakes once a minute and flushes a same-day
    /// digest line for any conversation idle longer than `threshold_minutes`.
    async fn run_idle_flush_loop(self: Arc<Self>, threshold_minutes: u32) {
        let interval = std::time::Duration::from_secs(60);
        info!(
            "Idle-flush loop active (threshold: {} minute(s))",
            threshold_minutes
        );
        loop {
            tokio::time::sleep(interval).await;
            self.maybe_flush_idle(threshold_minutes).await;
        }
    }

    /// One sweep of the idle-flush check. Public for testing — exposes
    /// the same body the background loop runs.
    async fn maybe_flush_idle(&self, threshold_minutes: u32) {
        let now = chrono::Utc::now();
        let threshold = chrono::Duration::minutes(threshold_minutes as i64);

        // Collect candidates while only briefly holding the lock.
        let candidates: Vec<ConversationKey> = {
            let last = self.last_activity_at.lock().await;
            let flushed = self.last_flushed_at.lock().await;
            last.iter()
                .filter(|(key, ts)| {
                    let idle_long_enough = now - **ts >= threshold;
                    let already_flushed = flushed.get(*key) == Some(*ts);
                    idle_long_enough && !already_flushed
                })
                .map(|(k, _)| k.clone())
                .collect()
        };

        for key in candidates {
            self.flush_intraday_digest(&key).await;
        }
    }

    /// Generate an intra-day digest of the current in-memory history for
    /// `key` and append it to the session JSONL. Recorded with `since` =
    /// the timestamp this Agent first observed activity for the key after
    /// its last flush (or session start), so consumers can sanity-check
    /// the window. Idempotent: marks `last_flushed_at` to the activity
    /// timestamp so the next flush only fires after fresh activity.
    async fn flush_intraday_digest(&self, key: &ConversationKey) {
        let messages = {
            let history = self.history.lock().await;
            history.get(key).cloned().unwrap_or_default()
        };
        let has_real_content = messages.iter().any(|m| {
            m.parts
                .iter()
                .any(|p| matches!(p, ContentPart::Text(t) if !t.trim().is_empty()))
        });
        if messages.len() < 2 || !has_real_content {
            return;
        }
        let session_id = {
            let sessions = self.active_sessions.lock().await;
            match sessions.get(key) {
                Some(id) if !id.is_empty() => id.clone(),
                _ => return,
            }
        };
        let activity_ts = match self.last_activity_at.lock().await.get(key) {
            Some(ts) => *ts,
            None => return,
        };

        info!("Idle flush: summarising session {session_id} for cross-room digest");
        let provider = self.provider_for(&key.0);
        let summary = match generate_summary(&*provider, &messages).await {
            Ok(s) if !s.trim().is_empty() => s,
            Ok(_) => {
                warn!("Idle-flush summary for {session_id} was empty; skipping");
                self.last_flushed_at
                    .lock()
                    .await
                    .insert(key.clone(), activity_ts);
                return;
            }
            Err(e) => {
                warn!("Idle-flush summary generation failed for {session_id}: {e:#}");
                return;
            }
        };

        if let Err(e) = self
            .session_store
            .append_intraday_digest(&session_id, &summary, None)
        {
            warn!("Failed to persist intra-day digest for {session_id}: {e}");
            return;
        }
        self.last_flushed_at
            .lock()
            .await
            .insert(key.clone(), activity_ts);
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
        let channel_name = self
            .channels
            .channel_name_for_room(&key.0)
            .await
            .unwrap_or_else(|| {
                self.channels
                    .names()
                    .first()
                    .map(|s| s.to_string())
                    .unwrap_or_else(|| "unknown".to_string())
            });
        let namespace = self.config.namespace_for_room(&key.0).to_string();
        match self
            .session_store
            .create_session(key, &channel_name, &namespace)
        {
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
    ///
    /// Messages containing `ToolUse` or `ToolResult` parts are intentionally
    /// skipped: tool payloads can be arbitrarily large (file contents, etc.)
    /// and we never reload raw history across restarts, so persisting them
    /// would only bloat the JSONL. Context survives via compaction summaries.
    ///
    /// Messages with `Image` parts are persisted with the raw image data
    /// stripped (replaced with a `[image: <media_type>]` text marker) so the
    /// JSONL stays small while daily logs / summaries can still see that an
    /// image was sent.
    fn persist(&self, session_id: &str, msg: &ChatMessage) {
        if session_id.is_empty() {
            return;
        }
        let has_tool_parts = msg.parts.iter().any(|p| {
            matches!(
                p,
                ContentPart::ToolUse { .. } | ContentPart::ToolResult { .. }
            )
        });
        if has_tool_parts {
            return;
        }
        let scrubbed = strip_image_data(msg);
        let to_write = scrubbed.as_ref().unwrap_or(msg);
        if let Err(e) = self.session_store.append(session_id, to_write) {
            warn!("Failed to persist message: {e}");
        }
    }

    /// Dispatch the configured day-boundary action for `key` when the local
    /// date on the active session file is older than today. Each policy is
    /// idempotent within a day via `boundary_handled`.
    async fn maybe_handle_day_boundary(&self, key: &ConversationKey) {
        let boundary = self.config.day_boundary_hour;
        let today = local_date_for_timestamp(Local::now(), boundary);
        let policy = self.config.session_policy_for(&key.0);

        if policy == SessionPolicy::None {
            return;
        }

        let session_id = {
            let sessions = self.active_sessions.lock().await;
            match sessions.get(key) {
                Some(id) if !id.is_empty() => id.clone(),
                _ => return,
            }
        };

        let Some(session_path) = self.session_store.absolute_path_for(&session_id) else {
            return;
        };
        if read_session_date(&session_path, boundary) >= today {
            return;
        }

        // Idempotence: don't re-fire within the same local day if we already
        // handled it (only relevant for policies that don't rotate the file).
        {
            let handled = self.boundary_handled.lock().await;
            if handled.get(key).copied() == Some(today) {
                return;
            }
        }

        match policy {
            SessionPolicy::Reset => {
                info!("Day boundary crossed for {key:?}; resetting session");

                if let Err(e) = self.session_store.close_session(&session_id) {
                    warn!("Failed to close session {session_id}: {e}");
                }

                self.history.lock().await.remove(key);
                self.active_sessions.lock().await.remove(key);
                self.snapshots.lock().await.remove(key);
                self.prefetch_cache.lock().await.remove(key);
                self.last_activity_at.lock().await.remove(key);
                self.last_flushed_at.lock().await.remove(key);
                // Reset rotates the session file, so no need to mark handled;
                // the new session will have today's created_at.
            }
            SessionPolicy::Compact => {
                self.compact_at_boundary(key, &session_id).await;
                self.boundary_handled
                    .lock()
                    .await
                    .insert(key.clone(), today);
            }
            SessionPolicy::None => unreachable!(),
        }
    }

    /// Force-summarize the current in-memory history for `key` and replace
    /// it with a summary stub so the session can keep growing into the new
    /// day without re-sending stale context to the model.
    async fn compact_at_boundary(&self, key: &ConversationKey, session_id: &str) {
        let messages = {
            let history = self.history.lock().await;
            history.get(key).cloned().unwrap_or_default()
        };

        // Nothing to compact if history is empty or already just a stub.
        let has_real_content = messages.iter().any(|m| {
            m.parts
                .iter()
                .any(|p| matches!(p, ContentPart::Text(t) if !t.is_empty()))
        });
        if messages.len() < 2 || !has_real_content {
            return;
        }

        info!("Day boundary crossed for {key:?}; compacting session {session_id}");

        let provider = self.provider_for(&key.0);
        let summary = match generate_summary(&*provider, &messages).await {
            Ok(s) if !s.trim().is_empty() => s,
            Ok(_) => {
                warn!("Boundary summary for {session_id} was empty; skipping");
                return;
            }
            Err(e) => {
                warn!("Boundary summary generation failed for {session_id}: {e:#}");
                return;
            }
        };

        if let Err(e) = self.session_store.append_summary(session_id, &summary) {
            warn!("Failed to persist boundary summary for {session_id}: {e}");
        }

        let stub = vec![
            ChatMessage {
                role: Role::User,
                parts: vec![ContentPart::Text(format!(
                    "[Context Summary — prior-day messages were compacted]\n\n{summary}"
                ))],
            },
            ChatMessage::assistant(
                "Understood. I have the context from the prior day's conversation.",
            ),
        ];

        self.history.lock().await.insert(key.clone(), stub);
        // Refresh the system prompt snapshot so it rebuilds with today's date.
        self.snapshots.lock().await.remove(key);
        self.prefetch_cache.lock().await.remove(key);
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
            if let Some(snap) = snapshots.get(key)
                && snap.date == today
            {
                return if snap.system_prompt.is_empty() {
                    None
                } else {
                    Some(snap.system_prompt.clone())
                };
            }
        }

        let namespace = self.config.namespace_for_room(&key.0);
        let chain = self.config.resolve_namespace_chain(namespace);
        // Best-effort: a missing or unreachable RoomInfo just leaves the
        // "# Current Room" block out — never blocks the turn.
        let room_info = self.channels.room_info(&key.0).await;
        let system_prompt = self
            .workspace
            .build_system_prompt(
                self.config.anthropic.system_prompt.as_deref(),
                self.config.day_boundary_hour,
                &chain,
                room_info.as_ref(),
            )
            .await;

        info!(
            "Rebuilt system-prompt snapshot for {key:?} (date={today}, {} chars)",
            system_prompt.len()
        );
        debug!("System-prompt snapshot for {key:?}:\n{system_prompt}");

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

        // Check for day boundary → dispatch configured session policy
        self.maybe_handle_day_boundary(&key).await;

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

        // First-turn-after-restart injection: if a compacted recap of the
        // prior run exists for this conversation, paste it into the system
        // prompt and consume the entry. Raw history is never reloaded across
        // restarts, so this summary is the sole bridge.
        let system_with_context = {
            let restart_summary = self.restart_summaries.lock().await.remove(&key);
            match (system_with_context, restart_summary) {
                (sys, Some(summary)) if !summary.trim().is_empty() => {
                    let base = sys.unwrap_or_default();
                    Some(format!(
                        "{base}\n\n---\n\n<prior-session-recap>\nサーバー再起動のため直前のやり取り自体は失われています。\
                         以下は前回セッションの要約です。これと今回の発言のみを頼りに応答してください。\
                         必要なら「再起動直後で記憶が曖昧」と率直に述べて構いません。\n\n{summary}\n</prior-session-recap>"
                    ))
                }
                (sys, _) => sys,
            }
        };

        // Append user message
        {
            let msg = build_user_message(&incoming.content, &incoming.attachments);
            self.history
                .lock()
                .await
                .entry(key.clone())
                .or_default()
                .push(msg.clone());
            self.persist(&session_id, &msg);
        }
        // Mark activity for the idle-flush loop. Done after the message
        // is queued so a freshly-active key won't be culled by a flush
        // that's already running this turn.
        self.last_activity_at
            .lock()
            .await
            .insert(key.clone(), chrono::Utc::now());

        let _ = self.channels.start_typing(&incoming.room_id).await;

        // Refresh MCP tools if any server signalled a change.
        if let Some(tools) = &self.tools {
            tools.refresh_if_needed().await;
        }
        let tool_specs = match &self.tools {
            Some(t) => Some(t.specs().await),
            None => None,
        };

        // Context compression config
        let compression_config = &self.config.compression;

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

            // Check if context compression is needed
            let provider = self.provider_for(&incoming.room_id);
            let messages = match maybe_compress(
                &*provider,
                system_with_context.as_deref(),
                &messages,
                compression_config,
            )
            .await
            {
                Ok(Some(result)) => {
                    // Replace in-memory history with compressed version
                    *self.history.lock().await.entry(key.clone()).or_default() =
                        result.compressed.clone();
                    if let Err(e) = self
                        .session_store
                        .append_summary(&session_id, &result.summary)
                    {
                        warn!("Failed to persist compaction summary: {e}");
                    }
                    result.compressed
                }
                Ok(None) => messages,
                Err(e) => {
                    warn!("Context compression failed, continuing with full history: {e}");
                    messages
                }
            };

            let response = provider
                .chat(
                    system_with_context.as_deref(),
                    &messages,
                    tool_specs.as_deref(),
                )
                .await;

            match response {
                Err(e) => {
                    error!("Provider error: {e:#}");
                    let _ = self.channels.stop_typing(&incoming.room_id).await;
                    let _ = self
                        .channels
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

                    // Execute tools concurrently. Each task gets the
                    // room's memory namespace via task_local so the
                    // memory tool writes under `memory/<namespace>/...`.
                    // tokio::spawn does not propagate task_local, so we
                    // re-bind it inside each spawned future.
                    let tools = Arc::clone(self.tools.as_ref().unwrap());
                    let ns = self
                        .config
                        .namespace_for_room(&incoming.room_id)
                        .to_string();
                    let mut handles = Vec::with_capacity(tool_calls.len());
                    for call in tool_calls {
                        let tools = Arc::clone(&tools);
                        let ns = ns.clone();
                        handles.push(tokio::spawn(
                            crate::tools::workspace_tools::scope_memory_namespace(ns, async move {
                                info!("Executing tool: {} (id={})", call.name, call.id);
                                let result = tools.execute(&call).await;
                                info!("Tool {} result: {}", call.name, result);
                                (call.id, result)
                            }),
                        ));
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

        let _ = self.channels.stop_typing(&incoming.room_id).await;

        if let Some(text) = final_text {
            if !text.is_empty() {
                let out = OutgoingMessage {
                    content: text,
                    room_id: incoming.room_id.clone(),
                    thread_id: incoming.thread_id.clone(),
                };
                self.channels
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

/// Build a user `ChatMessage` from incoming text + binary attachments.
/// Attachments are base64-encoded and emitted as `ContentPart::Image` parts.
fn build_user_message(text: &str, attachments: &[Attachment]) -> ChatMessage {
    if attachments.is_empty() {
        return ChatMessage::user(text);
    }
    use base64::{Engine, engine::general_purpose::STANDARD};
    let images = attachments
        .iter()
        .map(|a| (a.media_type.clone(), STANDARD.encode(&a.data)));
    ChatMessage::user_with_images(text, images)
}

/// Return a copy of `msg` with every `Image` part replaced by a small text
/// marker, or `None` if the message has no `Image` parts (avoids cloning).
fn strip_image_data(msg: &ChatMessage) -> Option<ChatMessage> {
    if !msg
        .parts
        .iter()
        .any(|p| matches!(p, ContentPart::Image { .. }))
    {
        return None;
    }
    let parts = msg
        .parts
        .iter()
        .map(|p| match p {
            ContentPart::Image { media_type, .. } => {
                ContentPart::Text(format!("[image: {media_type}]"))
            }
            other => other.clone(),
        })
        .collect();
    Some(ChatMessage {
        role: msg.role.clone(),
        parts,
    })
}

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
