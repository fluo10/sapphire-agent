//! JSONL-based session persistence.
//!
//! Each session is stored as a single `.jsonl` file in `<workspace>/sessions/`,
//! named by a ULID (time-ordered, lexicographically sortable).
//!
//! File layout:
//! ```text
//! {"meta": {"session_id":"01JX...","room_id":"!abc:m.org","thread_id":null,"channel":"matrix","created_at":"2026-04-06T10:00:00Z"}}
//! {"timestamp":"2026-04-06T10:00:01Z","role":"user","parts":[{"Text":"hello"}]}
//! {"timestamp":"2026-04-06T10:00:05Z","role":"assistant","parts":[{"Text":"hi"}]}
//! {"digest_at":"2026-04-06T10:30:00Z","since":"2026-04-06T04:00:00Z","digest":"..."}  ← intra-day flush
//! {"closed_at":"2026-04-06T11:00:00Z"}   ← optional, appended on reset/close
//! ```
//!
//! Timestamps are ISO 8601 / RFC 3339 (chrono) for human readability and
//! AI retrieval. `closed_at` acts as an append-only archive marker; presence
//! of this line means the session is no longer active.

use crate::provider::{ChatMessage, ContentPart, Role};
use base64::{Engine, engine::general_purpose::STANDARD as BASE64_STANDARD};
use chrono::{DateTime, Duration, Local, NaiveDate, TimeZone, Timelike, Utc};
use sapphire_workspace::WorkspaceState;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fs::{self, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use tracing::warn;
use uuid::Uuid;

pub type ConversationKey = (String, Option<String>);

// ---------------------------------------------------------------------------
// Stored types
// ---------------------------------------------------------------------------

/// Metadata written as the first line of each session file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionMeta {
    pub session_id: String,
    pub room_id: String,
    pub thread_id: Option<String>,
    pub channel: String,
    pub created_at: DateTime<Utc>,
    /// Human-readable alias (grain-id, 7 chars). Only set for API sessions.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub public_id: Option<String>,
    /// Memory namespace this session writes/reads under, captured at
    /// session creation so cross-session digest builders can route digests
    /// to the correct namespace even for sessions where the namespace is
    /// not derivable from `room_id` (e.g. API/voice sessions pinning a
    /// non-default room_profile). `None` for legacy files predating this
    /// field; consumers fall back to room-id derivation.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub namespace: Option<String>,
    /// External-AI logical project key for MCP-driven sessions
    /// (`write_report` / `recall_memory`). Stable across hosts and
    /// sources for the same project — the MCP layer reverse-looks-up
    /// `(namespace, project) -> session_id` from this field. Absent on
    /// chat/API/voice sessions.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub project: Option<String>,
    /// Short auto-generated title, populated from a later `session_title` line.
    #[serde(skip)]
    pub title: Option<String>,
}

/// A single stored message: `ChatMessage` + wall-clock timestamp.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredMessage {
    pub timestamp: DateTime<Utc>,
    pub role: Role,
    pub parts: Vec<ContentPart>,
    /// Provenance for messages written through MCP `write_report`.
    /// Absent on normal chat messages and on assistant-side replies.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub report_meta: Option<ReportMeta>,
}

/// Per-report provenance and structured fields. `source` distinguishes
/// external AI clients (e.g. "claude-code"); `hostname` records the
/// machine that originated the report, since a single project may
/// legitimately be touched from multiple hosts. `summary`, `body`,
/// and `files` mirror the `write_report` arguments so `recall_memory`
/// can return structured data without re-parsing the rendered text
/// that lives in the message's `parts`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportMeta {
    pub source: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hostname: Option<String>,
    pub summary: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub body: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub files: Option<Vec<String>>,
}

impl StoredMessage {
    pub fn from_chat(msg: &ChatMessage) -> Self {
        Self {
            timestamp: Utc::now(),
            role: msg.role.clone(),
            parts: msg.parts.clone(),
            report_meta: None,
        }
    }

    pub fn into_chat_message(self) -> ChatMessage {
        ChatMessage {
            role: self.role,
            parts: self.parts,
        }
    }
}

// Line wrappers used for JSON discrimination --------------------------------

#[derive(Serialize, Deserialize)]
struct MetaLine {
    meta: SessionMeta,
}

#[derive(Serialize, Deserialize)]
struct ClosedLine {
    closed_at: DateTime<Utc>,
}

#[derive(Serialize, Deserialize)]
struct TitleLine {
    session_title: String,
}

/// Compacted recap of a session, appended whenever an in-memory compression
/// fires and on graceful shutdown. Restart uses the latest `SummaryLine` to
/// inject context into the system prompt without replaying the raw (and
/// potentially tool-unpaired) message history.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SummaryLine {
    pub summary_at: DateTime<Utc>,
    pub summary: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub up_to_timestamp: Option<DateTime<Utc>>,
}

/// A short summary describing what happened in a single session during the
/// current local day. Emitted on idle-flush and graceful shutdown. Distinct
/// from `SummaryLine` because its scope is "today only" — `SessionPolicy::
/// Compact` sessions can carry context across the day boundary, so their
/// cumulative `SummaryLine` is not safe to splice into another room's
/// system prompt as "what happened today."
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IntradayDigestLine {
    pub digest_at: DateTime<Utc>,
    pub digest: String,
    /// Informational lower bound on the timestamps covered by this digest;
    /// when set, consumers may reject digests whose `since` predates the
    /// current local day. Not currently used for filtering — `digest_at`
    /// is the canonical "today?" predicate.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub since: Option<DateTime<Utc>>,
}

// ---------------------------------------------------------------------------
// SessionStore
// ---------------------------------------------------------------------------

pub struct SessionStore {
    /// Base sessions directory (e.g. `<workspace>/sessions`). Per-session
    /// files live under `<base_dir>/<namespace>/<kind>/<session_id>.jsonl`
    /// — the namespace split is mechanical (matches `memory/<namespace>/`)
    /// so retrieve indexing can scope itself by directory and never
    /// accidentally mix NSFW sessions with default-namespace ones.
    pub base_dir: PathBuf,
    /// Second-level subdirectory: `"channel"` for Matrix/Discord, `"api"`
    /// for HTTP. Lets the Agent and ServeState keep separate
    /// `SessionStore` instances while sharing one base dir.
    pub kind: &'static str,
    /// Optional sapphire-workspace state. When set, file modifications notify
    /// the workspace so the index/cache and git staging stay in sync.
    ws_state: Option<Arc<Mutex<WorkspaceState>>>,
    /// `session_id → absolute path` cache. Populated lazily by
    /// `resolve_path` (filesystem scan) and eagerly by `create_session` /
    /// `ensure_session`. Avoids re-scanning per `append` call.
    path_cache: Mutex<HashMap<String, PathBuf>>,
}

impl SessionStore {
    #[allow(dead_code)]
    pub fn new(base_dir: PathBuf, kind: &'static str) -> Self {
        Self {
            base_dir,
            kind,
            ws_state: None,
            path_cache: Mutex::new(HashMap::new()),
        }
    }

    pub fn with_workspace(
        base_dir: PathBuf,
        kind: &'static str,
        ws_state: Arc<Mutex<WorkspaceState>>,
    ) -> Self {
        Self {
            base_dir,
            kind,
            ws_state: Some(ws_state),
            path_cache: Mutex::new(HashMap::new()),
        }
    }

    /// Compute (without filesystem checks) the path a new session file
    /// should live at. Used by `create_session` / `ensure_session`. Also
    /// seeds the path cache so subsequent `append` calls hit it directly.
    fn path_for_new(&self, session_id: &str, namespace: &str) -> PathBuf {
        let p = self
            .base_dir
            .join(namespace)
            .join(self.kind)
            .join(format!("{session_id}.jsonl"));
        if let Ok(mut cache) = self.path_cache.lock() {
            cache.insert(session_id.to_string(), p.clone());
        }
        p
    }

    /// Public accessor exposing the cached path of an existing session.
    /// Used by callers that need to read raw bytes (e.g. parsing the meta
    /// line for `read_session_date`) rather than going through the
    /// `SessionStore` write methods.
    pub fn absolute_path_for(&self, session_id: &str) -> Option<PathBuf> {
        self.resolve_path(session_id)
    }

    /// Locate an existing session file by id, scanning every namespace
    /// subdirectory under `<base_dir>/<*>/<kind>/`. Returns `None` if the
    /// file isn't found. Hot path for `append`-style methods, so cached.
    fn resolve_path(&self, session_id: &str) -> Option<PathBuf> {
        if let Ok(cache) = self.path_cache.lock()
            && let Some(p) = cache.get(session_id)
        {
            return Some(p.clone());
        }
        let target = format!("{session_id}.jsonl");
        for path in collect_session_files(&self.base_dir, self.kind) {
            if path.file_name().and_then(|s| s.to_str()) == Some(target.as_str()) {
                if let Ok(mut cache) = self.path_cache.lock() {
                    cache.insert(session_id.to_string(), path.clone());
                }
                return Some(path);
            }
        }
        None
    }

    /// Notify sapphire-workspace that a session file was created or modified.
    /// No-op if no WorkspaceState is attached or the path is outside the workspace.
    fn notify_updated(&self, abs_path: &Path) {
        let Some(state) = &self.ws_state else { return };
        let guard = match state.lock() {
            Ok(g) => g,
            Err(e) => {
                warn!("WorkspaceState mutex poisoned: {e}");
                return;
            }
        };
        if !abs_path.starts_with(&guard.workspace.root) {
            return;
        }
        if let Err(e) = guard.on_file_updated(abs_path) {
            warn!(
                "Failed to notify workspace of update {}: {e}",
                abs_path.display()
            );
        }
    }

    /// Notify sapphire-workspace that a session file was deleted.
    #[allow(dead_code)]
    fn notify_deleted(&self, abs_path: &Path) {
        let Some(state) = &self.ws_state else { return };
        let guard = match state.lock() {
            Ok(g) => g,
            Err(e) => {
                warn!("WorkspaceState mutex poisoned: {e}");
                return;
            }
        };
        if !abs_path.starts_with(&guard.workspace.root) {
            return;
        }
        if let Err(e) = guard.on_file_deleted(abs_path) {
            warn!(
                "Failed to notify workspace of delete {}: {e}",
                abs_path.display()
            );
        }
    }

    /// Delete a session file (used when an empty session is discarded).
    #[allow(dead_code)]
    pub fn delete_session(&self, session_id: &str) -> anyhow::Result<()> {
        if let Some(path) = self.resolve_path(session_id) {
            if path.exists() {
                fs::remove_file(&path)?;
                self.notify_deleted(&path);
            }
            if let Ok(mut cache) = self.path_cache.lock() {
                cache.remove(session_id);
            }
        }
        Ok(())
    }

    /// Create a new session file for `key`. Returns the new session_id (ULID string).
    pub fn create_session(
        &self,
        key: &ConversationKey,
        channel: &str,
        namespace: &str,
    ) -> anyhow::Result<String> {
        let session_id = Uuid::now_v7().to_string();
        let path = self.path_for_new(&session_id, namespace);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let meta = SessionMeta {
            session_id: session_id.clone(),
            room_id: key.0.clone(),
            thread_id: key.1.clone(),
            channel: channel.to_string(),
            created_at: Utc::now(),
            public_id: None,
            namespace: Some(namespace.to_string()),
            project: None,
            title: None,
        };
        let line = serde_json::to_string(&MetaLine { meta })?;
        let mut file = OpenOptions::new().create(true).append(true).open(&path)?;
        writeln!(file, "{line}")?;
        drop(file);
        self.notify_updated(&path);
        Ok(session_id)
    }

    /// Append a `ChatMessage` (with current timestamp) to an existing session.
    pub fn append(&self, session_id: &str, msg: &ChatMessage) -> anyhow::Result<()> {
        let scrubbed = scrub_images_for_storage(msg);
        let to_store = scrubbed.as_ref().unwrap_or(msg);
        let stored = StoredMessage::from_chat(to_store);
        let line = serde_json::to_string(&stored)?;
        let path = self
            .resolve_path(session_id)
            .ok_or_else(|| anyhow::anyhow!("Session file not found for {session_id}"))?;
        let mut file = OpenOptions::new().create(true).append(true).open(&path)?;
        writeln!(file, "{line}")?;
        drop(file);
        self.notify_updated(&path);
        Ok(())
    }

    /// Append a compaction summary to the session file.
    pub fn append_summary(&self, session_id: &str, summary: &str) -> anyhow::Result<()> {
        let line = serde_json::to_string(&SummaryLine {
            summary_at: Utc::now(),
            summary: summary.to_string(),
            up_to_timestamp: None,
        })?;
        let path = self
            .resolve_path(session_id)
            .ok_or_else(|| anyhow::anyhow!("Session file not found for {session_id}"))?;
        let mut file = OpenOptions::new().create(true).append(true).open(&path)?;
        writeln!(file, "{line}")?;
        drop(file);
        self.notify_updated(&path);
        Ok(())
    }

    /// Append a same-day digest line. Used by the idle-flush task and the
    /// graceful-shutdown path to publish "what this session has covered
    /// today" for cross-session injection into other rooms.
    pub fn append_intraday_digest(
        &self,
        session_id: &str,
        digest: &str,
        since: Option<DateTime<Utc>>,
    ) -> anyhow::Result<()> {
        let line = serde_json::to_string(&IntradayDigestLine {
            digest_at: Utc::now(),
            digest: digest.to_string(),
            since,
        })?;
        let path = self
            .resolve_path(session_id)
            .ok_or_else(|| anyhow::anyhow!("Session file not found for {session_id}"))?;
        let mut file = OpenOptions::new().create(true).append(true).open(&path)?;
        writeln!(file, "{line}")?;
        drop(file);
        self.notify_updated(&path);
        Ok(())
    }

    /// Walk every session file under `sessions_dir` and return the latest
    /// `IntradayDigestLine` per session whose `digest_at` falls inside the
    /// local-time `date` window (under `boundary_hour`), paired with the
    /// session's metadata. Used to assemble the cross-session "today
    /// digest" injected into the system prompt of newly opened rooms.
    pub fn intraday_digests_for_day(
        &self,
        date: NaiveDate,
        boundary_hour: u8,
    ) -> Vec<(SessionMeta, IntradayDigestLine)> {
        let (day_start, day_end) = day_window(date, boundary_hour);
        let mut out = Vec::new();
        for path in collect_session_files(&self.base_dir, self.kind) {
            // mtime pre-filter: a file last touched before day_start can't
            // possibly carry a digest in this window.
            if let Ok(meta_fs) = path.metadata()
                && let Ok(mtime) = meta_fs.modified()
            {
                let mtime_utc: DateTime<Utc> = mtime.into();
                if mtime_utc < day_start {
                    continue;
                }
            }
            let Some((meta, digest)) = load_meta_and_latest_intraday_digest(&path) else {
                continue;
            };
            let Some(d) = digest else { continue };
            if d.digest_at >= day_start && d.digest_at < day_end {
                out.push((meta, d));
            }
        }
        out.sort_by_key(|(meta, _)| meta.created_at);
        out
    }

    /// Close a session by appending a `closed_at` marker.
    /// The session becomes inactive; future messages create a new session.
    pub fn close_session(&self, session_id: &str) -> anyhow::Result<()> {
        let line = serde_json::to_string(&ClosedLine {
            closed_at: Utc::now(),
        })?;
        let path = self
            .resolve_path(session_id)
            .ok_or_else(|| anyhow::anyhow!("Session file not found for {session_id}"))?;
        let mut file = OpenOptions::new().create(true).append(true).open(&path)?;
        writeln!(file, "{line}")?;
        drop(file);
        self.notify_updated(&path);
        Ok(())
    }

    /// Load all sessions from disk on startup.
    ///
    /// For each `ConversationKey`, picks the latest ULID-ordered session that
    /// does **not** have a `closed_at` marker (i.e. is still active).
    ///
    /// Raw message history is intentionally NOT reconstructed into in-memory
    /// history: Anthropic's API requires paired tool_use/tool_result, and we
    /// skip persisting tool messages to disk, so reloading would break that
    /// invariant. Instead, callers get:
    ///
    /// - `active`: which session file is current per conversation
    /// - `summaries`: the latest `SummaryLine` per conversation (when present)
    /// - `fallback_messages`: raw `ChatMessage` list for active sessions that
    ///   have NO summary yet — Agent bootstrap uses these to synthesize a
    ///   summary on startup (e.g. after a crash that skipped graceful shutdown)
    #[allow(clippy::type_complexity)]
    pub fn load_all(
        &self,
    ) -> (
        HashMap<ConversationKey, String>,
        HashMap<ConversationKey, String>,
        HashMap<ConversationKey, Vec<ChatMessage>>,
    ) {
        type SessionEntry = (
            String,
            ConversationKey,
            Vec<StoredMessage>,
            bool,
            Option<String>,
        );
        let mut entries: Vec<SessionEntry> = Vec::new();

        for path in collect_session_files(&self.base_dir, self.kind) {
            let stem = match path.file_stem().and_then(|s| s.to_str()) {
                Some(s) => s.to_string(),
                None => continue,
            };
            if let Some((meta, messages, is_closed, summary)) = load_session_file(&path) {
                let key: ConversationKey = (meta.room_id.clone(), meta.thread_id.clone());
                if !is_closed {
                    // Seed the path cache for active sessions so the first
                    // `append` after bootstrap doesn't pay a scan.
                    if let Ok(mut cache) = self.path_cache.lock() {
                        cache.insert(stem.clone(), path.clone());
                    }
                }
                entries.push((stem, key, messages, is_closed, summary.map(|s| s.summary)));
            }
        }

        entries.sort_by(|a, b| a.0.cmp(&b.0));

        let mut active: HashMap<ConversationKey, String> = HashMap::new();
        let mut summaries: HashMap<ConversationKey, String> = HashMap::new();
        let mut fallback: HashMap<ConversationKey, Vec<ChatMessage>> = HashMap::new();

        for (session_id, key, messages, is_closed, summary) in entries {
            if !is_closed {
                active.insert(key.clone(), session_id);
                match summary {
                    Some(s) => {
                        summaries.insert(key.clone(), s);
                        fallback.remove(&key);
                    }
                    None => {
                        summaries.remove(&key);
                        if !messages.is_empty() {
                            let chat_messages: Vec<ChatMessage> = messages
                                .into_iter()
                                .map(|m| m.into_chat_message())
                                .collect();
                            fallback.insert(key, chat_messages);
                        } else {
                            fallback.remove(&key);
                        }
                    }
                }
            }
        }

        (active, summaries, fallback)
    }

    /// List metadata for all sessions in this store (used by API for session listing).
    pub fn list_sessions(&self) -> Vec<SessionMeta> {
        let mut metas: Vec<SessionMeta> = collect_session_files(&self.base_dir, self.kind)
            .into_iter()
            .filter_map(|p| load_session_file(&p).map(|(meta, _, _, _)| meta))
            .collect();
        metas.sort_by_key(|m| m.created_at);
        metas
    }

    /// Load a single session's conversation history by ID.
    /// Returns None if the file doesn't exist or is malformed.
    pub fn load_session(&self, session_id: &str) -> Option<Vec<ChatMessage>> {
        let path = self.resolve_path(session_id)?;
        let (_, messages, _, _) = load_session_file(&path)?;
        Some(
            messages
                .into_iter()
                .map(|m| m.into_chat_message())
                .collect(),
        )
    }

    /// Load a session preserving wall-clock timestamps and
    /// `report_meta` provenance, alongside the latest `SummaryLine`
    /// if one has been written. Used by `recall_memory`: the summary
    /// becomes `project_summary` (older content compacted) and the
    /// messages provide the recent verbatim reports. Plain
    /// `load_session` is unsuitable because the `ChatMessage`
    /// conversion drops both fields.
    pub fn load_session_full(
        &self,
        session_id: &str,
    ) -> Option<(Vec<StoredMessage>, Option<SummaryLine>)> {
        let path = self.resolve_path(session_id)?;
        let (_, messages, _, summary) = load_session_file(&path)?;
        Some((messages, summary))
    }

    /// Create a new MCP-driven session for a logical `project`. Unlike
    /// `create_session` there's no `ConversationKey` — MCP sessions
    /// don't map to a chat room, so `room_id` is left empty and the
    /// `project` field on `SessionMeta` serves as the reverse lookup
    /// key. Files land under `<base_dir>/<namespace>/mcp/<ULID>.jsonl`
    /// when this store is constructed with `kind = "mcp"`.
    pub fn create_mcp_session(&self, namespace: &str, project: &str) -> anyhow::Result<String> {
        let session_id = Uuid::now_v7().to_string();
        let path = self.path_for_new(&session_id, namespace);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let meta = SessionMeta {
            session_id: session_id.clone(),
            room_id: String::new(),
            thread_id: None,
            channel: "mcp".to_string(),
            created_at: Utc::now(),
            public_id: None,
            namespace: Some(namespace.to_string()),
            project: Some(project.to_string()),
            title: None,
        };
        let line = serde_json::to_string(&MetaLine { meta })?;
        let mut file = OpenOptions::new().create(true).append(true).open(&path)?;
        writeln!(file, "{line}")?;
        drop(file);
        self.notify_updated(&path);
        Ok(session_id)
    }

    /// Append a user-role report message tagged with MCP provenance.
    /// `rendered_text` is what lives in the message's `parts` (used
    /// as LLM context for the ねぎらい reply and any future feature
    /// that reads sessions as conversation); `meta` carries the
    /// structured form `recall_memory` returns to clients. The
    /// assistant's reply is written through the regular `append`
    /// path so the session reads back as a normal conversation.
    pub fn append_report(
        &self,
        session_id: &str,
        rendered_text: &str,
        meta: ReportMeta,
    ) -> anyhow::Result<()> {
        let stored = StoredMessage {
            timestamp: Utc::now(),
            role: Role::User,
            parts: vec![ContentPart::Text(rendered_text.to_string())],
            report_meta: Some(meta),
        };
        let line = serde_json::to_string(&stored)?;
        let path = self
            .resolve_path(session_id)
            .ok_or_else(|| anyhow::anyhow!("Session file not found for {session_id}"))?;
        let mut file = OpenOptions::new().create(true).append(true).open(&path)?;
        writeln!(file, "{line}")?;
        drop(file);
        self.notify_updated(&path);
        Ok(())
    }

    /// Ensure a session file exists for the given caller-supplied ID.
    /// Unlike `create_session`, this uses the provided ID rather than generating a new UUID.
    ///
    /// For API sessions (`channel == "api"`), a grain-id `public_id` is generated on creation
    /// unless `public_id_override` is supplied (used to commit a deferred public_id).
    /// Returns the `public_id` if present (new or existing).
    pub fn ensure_session(
        &self,
        session_id: &str,
        key: &ConversationKey,
        channel: &str,
        public_id_override: Option<String>,
        namespace: &str,
    ) -> anyhow::Result<Option<String>> {
        if let Some(existing) = self.resolve_path(session_id) {
            // Return existing public_id if the file already existed
            let pub_id = load_session_file(&existing).and_then(|(meta, _, _, _)| meta.public_id);
            return Ok(pub_id);
        }
        let path = self.path_for_new(session_id, namespace);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        let public_id = if channel == "api" {
            Some(public_id_override.unwrap_or_else(|| grain_id::GrainId::random().to_string()))
        } else {
            None
        };
        let meta = SessionMeta {
            session_id: session_id.to_string(),
            room_id: key.0.clone(),
            thread_id: key.1.clone(),
            channel: channel.to_string(),
            created_at: Utc::now(),
            public_id: public_id.clone(),
            namespace: Some(namespace.to_string()),
            project: None,
            title: None,
        };
        let line = serde_json::to_string(&MetaLine { meta })?;
        let mut file = OpenOptions::new().create(true).append(true).open(&path)?;
        writeln!(file, "{line}")?;
        drop(file);
        self.notify_updated(&path);
        Ok(public_id)
    }

    /// Append a title for a session (append-only; last line wins on read).
    pub fn set_title(&self, session_id: &str, title: &str) -> anyhow::Result<()> {
        let line = serde_json::to_string(&TitleLine {
            session_title: title.to_string(),
        })?;
        let path = self
            .resolve_path(session_id)
            .ok_or_else(|| anyhow::anyhow!("Session file not found for {session_id}"))?;
        let mut file = OpenOptions::new().create(true).append(true).open(&path)?;
        writeln!(file, "{line}")?;
        drop(file);
        self.notify_updated(&path);
        Ok(())
    }

    /// Find a session by its human-readable `public_id` (grain-id).
    /// Returns the internal UUID `session_id` if found.
    pub fn find_by_public_id(&self, public_id: &str) -> Option<String> {
        for path in collect_session_files(&self.base_dir, self.kind) {
            if let Some((meta, _, _, _)) = load_session_file(&path)
                && meta.public_id.as_deref() == Some(public_id)
            {
                return Some(meta.session_id);
            }
        }
        None
    }

    /// Return all sessions that contain at least one message falling within
    /// the given local-time day window.
    ///
    /// The "day" is `[date @ boundary_hour:00:00 local, (date+1) @ boundary_hour:00:00 local)`.
    pub fn sessions_for_day(
        &self,
        date: NaiveDate,
        boundary_hour: u8,
    ) -> Vec<(SessionMeta, Vec<StoredMessage>)> {
        let (day_start, day_end) = day_window(date, boundary_hour);
        let mut results = Vec::new();

        for path in collect_session_files(&self.base_dir, self.kind) {
            // mtime pre-filter: skip files last modified before day_start - 1 day
            if let Ok(meta_fs) = path.metadata()
                && let Ok(mtime) = meta_fs.modified()
            {
                let mtime_utc: DateTime<Utc> = mtime.into();
                if mtime_utc < day_start - Duration::days(1) {
                    continue;
                }
            }

            if let Some((meta, messages, _, _)) = load_session_file(&path) {
                let day_messages: Vec<StoredMessage> = messages
                    .into_iter()
                    .filter(|m| m.timestamp >= day_start && m.timestamp < day_end)
                    .collect();

                if !day_messages.is_empty() {
                    results.push((meta, day_messages));
                }
            }
        }

        results.sort_by_key(|(meta, _)| meta.created_at);
        results
    }

    /// Like `sessions_for_day`, but only returns sessions for which
    /// `predicate(&meta)` is true. Used by daily-log generation when it
    /// runs per memory namespace: the caller supplies a predicate that
    /// keeps only rooms mapped to the namespace being generated.
    pub fn sessions_for_day_filtered<F>(
        &self,
        date: NaiveDate,
        boundary_hour: u8,
        predicate: F,
    ) -> Vec<(SessionMeta, Vec<StoredMessage>)>
    where
        F: Fn(&SessionMeta) -> bool,
    {
        self.sessions_for_day(date, boundary_hour)
            .into_iter()
            .filter(|(meta, _)| predicate(meta))
            .collect()
    }

    /// Like `all_session_dates`, but only counts sessions whose `meta`
    /// satisfies `predicate`. Used so per-namespace daily-log catch-up
    /// only enumerates dates that have at least one in-namespace session.
    pub fn all_session_dates_filtered<F>(&self, boundary_hour: u8, predicate: F) -> Vec<NaiveDate>
    where
        F: Fn(&SessionMeta) -> bool,
    {
        let mut dates = std::collections::HashSet::new();

        for path in collect_session_files(&self.base_dir, self.kind) {
            if let Some((meta, messages, _, _)) = load_session_file(&path) {
                if !predicate(&meta) {
                    continue;
                }
                for msg in messages {
                    let local_ts = msg.timestamp.with_timezone(&Local);
                    let date = local_date_for_timestamp(local_ts, boundary_hour);
                    dates.insert(date);
                }
            }
        }

        let mut sorted: Vec<NaiveDate> = dates.into_iter().collect();
        sorted.sort();
        sorted
    }

    /// Return all local dates for which at least one session message exists.
    /// Used by daily_log to find dates that need a log generated.
    #[allow(dead_code)]
    pub fn all_session_dates(&self, boundary_hour: u8) -> Vec<NaiveDate> {
        let mut dates = std::collections::HashSet::new();

        for path in collect_session_files(&self.base_dir, self.kind) {
            if let Some((_, messages, _, _)) = load_session_file(&path) {
                for msg in messages {
                    let local_ts = msg.timestamp.with_timezone(&Local);
                    let date = local_date_for_timestamp(local_ts, boundary_hour);
                    dates.insert(date);
                }
            }
        }

        let mut sorted: Vec<NaiveDate> = dates.into_iter().collect();
        sorted.sort();
        sorted
    }
}

// ---------------------------------------------------------------------------
// Image scrubbing for persistence
// ---------------------------------------------------------------------------

/// Replace every `ContentPart::Image` in `msg` with a text marker that
/// preserves the MIME type and a SHA-256 of the raw bytes, returning the
/// rewritten message. Returns `None` when `msg` has no image parts (so
/// callers can skip the allocation).
///
/// Format: `[image: <media_type> sha256=<hex>]` — the hash gives future
/// out-of-band caches a stable key (planned follow-up: ImageRef +
/// workspace-external cache) without dragging multi-MB base64 blobs into
/// JSONL session files or the in-memory history on reload.
///
/// An undecodable `data_base64` is recorded as `sha256=invalid-base64`
/// rather than failing the append — a corrupt image shouldn't lose the
/// surrounding turn from persistence.
pub(crate) fn scrub_images_for_storage(msg: &ChatMessage) -> Option<ChatMessage> {
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
            ContentPart::Image {
                media_type,
                data_base64,
            } => {
                let hash = match BASE64_STANDARD.decode(data_base64) {
                    Ok(bytes) => sha256_hex(&bytes),
                    Err(_) => "invalid-base64".to_string(),
                };
                ContentPart::Text(format!("[image: {media_type} sha256={hash}]"))
            }
            other => other.clone(),
        })
        .collect();
    Some(ChatMessage {
        role: msg.role.clone(),
        parts,
    })
}

fn sha256_hex(bytes: &[u8]) -> String {
    use std::fmt::Write;
    let mut h = Sha256::new();
    h.update(bytes);
    let digest = h.finalize();
    let mut s = String::with_capacity(64);
    for b in digest.iter() {
        let _ = write!(&mut s, "{b:02x}");
    }
    s
}

// ---------------------------------------------------------------------------
// Namespace-scoped filesystem walking
// ---------------------------------------------------------------------------

/// Enumerate `<base_dir>/<namespace>/<kind>/*.jsonl` across every namespace
/// directory. Returns an empty Vec when `base_dir` doesn't exist yet (fresh
/// install) or has no namespace subdirs. Each returned path is absolute.
fn collect_session_files(base_dir: &Path, kind: &str) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let Ok(entries) = fs::read_dir(base_dir) else {
        return out;
    };
    for entry in entries.flatten() {
        let ns_dir = entry.path();
        if !ns_dir.is_dir() {
            continue;
        }
        let kind_dir = ns_dir.join(kind);
        let Ok(kind_entries) = fs::read_dir(&kind_dir) else {
            continue;
        };
        for k_entry in kind_entries.flatten() {
            let path = k_entry.path();
            if path.extension().and_then(|e| e.to_str()) == Some("jsonl") {
                out.push(path);
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Day window helpers
// ---------------------------------------------------------------------------

/// Compute the UTC start and end of the local-time day window for `date`
/// with a given `boundary_hour`.
///
/// Window: `[date @ boundary_hour:00:00 local, (date+1day) @ boundary_hour:00:00 local)`
fn day_window(date: NaiveDate, boundary_hour: u8) -> (DateTime<Utc>, DateTime<Utc>) {
    let start_local = date
        .and_hms_opt(boundary_hour as u32, 0, 0)
        .expect("valid time");
    let end_local = (date + Duration::days(1))
        .and_hms_opt(boundary_hour as u32, 0, 0)
        .expect("valid time");

    let start_utc = Local
        .from_local_datetime(&start_local)
        .single()
        .unwrap_or_else(|| Local.from_local_datetime(&start_local).earliest().unwrap())
        .with_timezone(&Utc);

    let end_utc = Local
        .from_local_datetime(&end_local)
        .single()
        .unwrap_or_else(|| Local.from_local_datetime(&end_local).earliest().unwrap())
        .with_timezone(&Utc);

    (start_utc, end_utc)
}

/// Given a local timestamp, return the local date it belongs to for a given
/// `boundary_hour`. Timestamps before `boundary_hour` belong to the previous day.
pub fn local_date_for_timestamp(local_ts: DateTime<Local>, boundary_hour: u8) -> NaiveDate {
    let date = local_ts.date_naive();
    if local_ts.hour() < boundary_hour as u32 {
        date - Duration::days(1)
    } else {
        date
    }
}

// ---------------------------------------------------------------------------
// File parsing helpers
// ---------------------------------------------------------------------------

/// Parse a single session `.jsonl` file.
///
/// Returns `(meta, messages, is_closed, latest_summary)` or `None` if the
/// file is unreadable or has a malformed first line.
fn load_session_file(
    path: &Path,
) -> Option<(SessionMeta, Vec<StoredMessage>, bool, Option<SummaryLine>)> {
    let file = fs::File::open(path).ok()?;
    let mut lines = BufReader::new(file).lines();

    let first = lines.next()?.ok()?;
    let meta_line: MetaLine = serde_json::from_str(first.trim()).ok()?;
    let mut meta = meta_line.meta;

    let mut messages = Vec::new();
    let mut is_closed = false;
    let mut latest_summary: Option<SummaryLine> = None;

    for raw in lines.map_while(Result::ok) {
        let raw = raw.trim().to_string();
        if raw.is_empty() {
            continue;
        }

        let value: serde_json::Value = match serde_json::from_str(&raw) {
            Ok(v) => v,
            Err(e) => {
                warn!("Skipping unparseable line in {}: {e}", path.display());
                continue;
            }
        };

        if value.get("closed_at").is_some() {
            is_closed = true;
        } else if let Some(title) = value.get("session_title").and_then(|v| v.as_str()) {
            meta.title = Some(title.to_string());
        } else if value.get("summary_at").is_some() {
            match serde_json::from_value::<SummaryLine>(value) {
                Ok(s) => latest_summary = Some(s),
                Err(e) => {
                    warn!("Skipping malformed summary in {}: {e}", path.display());
                }
            }
        } else if value.get("digest_at").is_some() {
            // Intra-day digest lines are not returned by this loader;
            // `intraday_digests_for_day` reads them through its own helper.
            continue;
        } else if value.get("timestamp").is_some() {
            match serde_json::from_value::<StoredMessage>(value) {
                Ok(stored) => messages.push(stored),
                Err(e) => {
                    warn!("Skipping malformed message in {}: {e}", path.display());
                }
            }
        }
    }

    Some((meta, messages, is_closed, latest_summary))
}

/// Minimal-cost variant of `load_session_file`: returns just the metadata
/// and the latest `IntradayDigestLine`, skipping message accumulation.
fn load_meta_and_latest_intraday_digest(
    path: &Path,
) -> Option<(SessionMeta, Option<IntradayDigestLine>)> {
    let file = fs::File::open(path).ok()?;
    let mut lines = BufReader::new(file).lines();

    let first = lines.next()?.ok()?;
    let meta_line: MetaLine = serde_json::from_str(first.trim()).ok()?;
    let meta = meta_line.meta;

    let mut latest: Option<IntradayDigestLine> = None;
    for raw in lines.map_while(Result::ok) {
        let raw = raw.trim();
        if raw.is_empty() {
            continue;
        }
        let value: serde_json::Value = match serde_json::from_str(raw) {
            Ok(v) => v,
            Err(_) => continue,
        };
        if value.get("digest_at").is_some()
            && let Ok(d) = serde_json::from_value::<IntradayDigestLine>(value)
        {
            latest = Some(d);
        }
    }
    Some((meta, latest))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scrub_returns_none_when_no_images() {
        let msg = ChatMessage::user("plain text");
        assert!(scrub_images_for_storage(&msg).is_none());
    }

    #[test]
    fn scrub_replaces_image_with_hash_marker() {
        let bytes = b"\xff\xd8\xff\xe0fake-jpeg".to_vec();
        let b64 = BASE64_STANDARD.encode(&bytes);
        let msg =
            ChatMessage::user_with_images("look", std::iter::once(("image/jpeg".to_string(), b64)));
        let scrubbed = scrub_images_for_storage(&msg).expect("scrub should rewrite");

        // No Image parts remain on the persisted shape.
        assert!(
            !scrubbed
                .parts
                .iter()
                .any(|p| matches!(p, ContentPart::Image { .. })),
            "scrubbed message still contains Image part"
        );

        // The marker is text and carries the expected hash.
        let expected = sha256_hex(&bytes);
        let has_marker = scrubbed
            .parts
            .iter()
            .any(|p| matches!(p, ContentPart::Text(s) if s.contains(&expected) && s.contains("image/jpeg")));
        assert!(
            has_marker,
            "missing hash marker; parts={:?}",
            scrubbed.parts
        );
    }

    #[test]
    fn scrub_invalid_base64_records_marker_without_panic() {
        let msg = ChatMessage {
            role: Role::User,
            parts: vec![ContentPart::Image {
                media_type: "image/png".to_string(),
                data_base64: "@@@not-base64@@@".to_string(),
            }],
        };
        let scrubbed = scrub_images_for_storage(&msg).expect("scrub should rewrite");
        let has_marker = scrubbed
            .parts
            .iter()
            .any(|p| matches!(p, ContentPart::Text(s) if s.contains("invalid-base64")));
        assert!(has_marker, "expected invalid-base64 marker");
    }
}
