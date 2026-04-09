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
//! {"closed_at":"2026-04-06T11:00:00Z"}   ← optional, appended on reset/close
//! ```
//!
//! Timestamps are ISO 8601 / RFC 3339 (chrono) for human readability and
//! AI retrieval. `closed_at` acts as an append-only archive marker; presence
//! of this line means the session is no longer active.

use crate::provider::{ChatMessage, ContentPart, Role};
use chrono::{DateTime, Duration, Local, NaiveDate, TimeZone, Timelike, Utc};
use sapphire_workspace::WorkspaceState;
use serde::{Deserialize, Serialize};
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
}

impl StoredMessage {
    pub fn from_chat(msg: &ChatMessage) -> Self {
        Self {
            timestamp: Utc::now(),
            role: msg.role.clone(),
            parts: msg.parts.clone(),
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

// ---------------------------------------------------------------------------
// SessionStore
// ---------------------------------------------------------------------------

pub struct SessionStore {
    pub sessions_dir: PathBuf,
    /// Optional sapphire-workspace state. When set, file modifications notify
    /// the workspace so the index/cache and git staging stay in sync.
    ws_state: Option<Arc<Mutex<WorkspaceState>>>,
}

impl SessionStore {
    pub fn new(sessions_dir: PathBuf) -> Self {
        Self {
            sessions_dir,
            ws_state: None,
        }
    }

    pub fn with_workspace(sessions_dir: PathBuf, ws_state: Arc<Mutex<WorkspaceState>>) -> Self {
        Self {
            sessions_dir,
            ws_state: Some(ws_state),
        }
    }

    fn session_path(&self, session_id: &str) -> PathBuf {
        self.sessions_dir.join(format!("{session_id}.jsonl"))
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
    pub fn delete_session(&self, session_id: &str) -> anyhow::Result<()> {
        let path = self.session_path(session_id);
        if path.exists() {
            fs::remove_file(&path)?;
            self.notify_deleted(&path);
        }
        Ok(())
    }

    /// Create a new session file for `key`. Returns the new session_id (ULID string).
    pub fn create_session(&self, key: &ConversationKey, channel: &str) -> anyhow::Result<String> {
        fs::create_dir_all(&self.sessions_dir)?;
        let session_id = Uuid::now_v7().to_string();
        let meta = SessionMeta {
            session_id: session_id.clone(),
            room_id: key.0.clone(),
            thread_id: key.1.clone(),
            channel: channel.to_string(),
            created_at: Utc::now(),
            public_id: None,
            title: None,
        };
        let line = serde_json::to_string(&MetaLine { meta })?;
        let path = self.session_path(&session_id);
        let mut file = OpenOptions::new().create(true).append(true).open(&path)?;
        writeln!(file, "{line}")?;
        drop(file);
        self.notify_updated(&path);
        Ok(session_id)
    }

    /// Append a `ChatMessage` (with current timestamp) to an existing session.
    pub fn append(&self, session_id: &str, msg: &ChatMessage) -> anyhow::Result<()> {
        let stored = StoredMessage::from_chat(msg);
        let line = serde_json::to_string(&stored)?;
        let path = self.session_path(session_id);
        let mut file = OpenOptions::new().create(true).append(true).open(&path)?;
        writeln!(file, "{line}")?;
        drop(file);
        self.notify_updated(&path);
        Ok(())
    }

    /// Close a session by appending a `closed_at` marker.
    /// The session becomes inactive; future messages create a new session.
    pub fn close_session(&self, session_id: &str) -> anyhow::Result<()> {
        let line = serde_json::to_string(&ClosedLine {
            closed_at: Utc::now(),
        })?;
        let path = self.session_path(session_id);
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
    /// Returns `(history, active_session_ids)`.
    pub fn load_all(
        &self,
    ) -> (
        HashMap<ConversationKey, Vec<ChatMessage>>,
        HashMap<ConversationKey, String>,
    ) {
        type SessionEntry = (String, ConversationKey, Vec<StoredMessage>, bool);
        let mut entries: Vec<SessionEntry> = Vec::new();

        let dir = match fs::read_dir(&self.sessions_dir) {
            Ok(d) => d,
            Err(_) => return (HashMap::new(), HashMap::new()),
        };

        for entry in dir.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("jsonl") {
                continue;
            }
            let stem = match path.file_stem().and_then(|s| s.to_str()) {
                Some(s) => s.to_string(),
                None => continue,
            };

            if let Some((meta, messages, is_closed)) = load_session_file(&path) {
                let key: ConversationKey = (meta.room_id.clone(), meta.thread_id.clone());
                entries.push((stem, key, messages, is_closed));
            }
        }

        // Sort by session_id (ULID ⟹ time-ordered, lexicographic)
        entries.sort_by(|a, b| a.0.cmp(&b.0));

        let mut history: HashMap<ConversationKey, Vec<ChatMessage>> = HashMap::new();
        let mut active: HashMap<ConversationKey, String> = HashMap::new();

        for (session_id, key, messages, is_closed) in entries {
            if !is_closed {
                let chat_messages = messages
                    .into_iter()
                    .map(|m| m.into_chat_message())
                    .collect();
                history.insert(key.clone(), chat_messages);
                active.insert(key, session_id);
            }
        }

        (history, active)
    }

    /// List metadata for all sessions in this store (used by API for session listing).
    pub fn list_sessions(&self) -> Vec<SessionMeta> {
        let dir = match fs::read_dir(&self.sessions_dir) {
            Ok(d) => d,
            Err(_) => return vec![],
        };
        let mut metas: Vec<SessionMeta> = dir
            .flatten()
            .filter(|e| e.path().extension().and_then(|x| x.to_str()) == Some("jsonl"))
            .filter_map(|e| load_session_file(&e.path()).map(|(meta, _, _)| meta))
            .collect();
        metas.sort_by_key(|m| m.created_at);
        metas
    }

    /// Load a single session's conversation history by ID.
    /// Returns None if the file doesn't exist or is malformed.
    pub fn load_session(&self, session_id: &str) -> Option<Vec<ChatMessage>> {
        let path = self.session_path(session_id);
        let (_, messages, _) = load_session_file(&path)?;
        Some(
            messages
                .into_iter()
                .map(|m| m.into_chat_message())
                .collect(),
        )
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
    ) -> anyhow::Result<Option<String>> {
        fs::create_dir_all(&self.sessions_dir)?;
        let path = self.session_path(session_id);
        if path.exists() {
            // Return existing public_id if the file already existed
            let pub_id = load_session_file(&path).and_then(|(meta, _, _)| meta.public_id);
            return Ok(pub_id);
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
        let path = self.session_path(session_id);
        let mut file = OpenOptions::new().create(true).append(true).open(&path)?;
        writeln!(file, "{line}")?;
        drop(file);
        self.notify_updated(&path);
        Ok(())
    }

    /// Find a session by its human-readable `public_id` (grain-id).
    /// Returns the internal UUID `session_id` if found.
    pub fn find_by_public_id(&self, public_id: &str) -> Option<String> {
        let dir = fs::read_dir(&self.sessions_dir).ok()?;
        for entry in dir.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("jsonl") {
                continue;
            }
            if let Some((meta, _, _)) = load_session_file(&path) {
                if meta.public_id.as_deref() == Some(public_id) {
                    return Some(meta.session_id);
                }
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

        let dir = match fs::read_dir(&self.sessions_dir) {
            Ok(d) => d,
            Err(_) => return vec![],
        };

        let mut results = Vec::new();

        for entry in dir.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("jsonl") {
                continue;
            }

            // mtime pre-filter: skip files last modified before day_start - 1 day
            if let Ok(meta_fs) = path.metadata() {
                if let Ok(mtime) = meta_fs.modified() {
                    let mtime_utc: DateTime<Utc> = mtime.into();
                    if mtime_utc < day_start - Duration::days(1) {
                        continue;
                    }
                }
            }

            if let Some((meta, messages, _)) = load_session_file(&path) {
                let day_messages: Vec<StoredMessage> = messages
                    .into_iter()
                    .filter(|m| m.timestamp >= day_start && m.timestamp < day_end)
                    .collect();

                if !day_messages.is_empty() {
                    results.push((meta, day_messages));
                }
            }
        }

        // Sort by session created_at for chronological ordering
        results.sort_by_key(|(meta, _)| meta.created_at);
        results
    }

    /// Return all local dates for which at least one session message exists.
    /// Used by daily_log to find dates that need a log generated.
    pub fn all_session_dates(&self, boundary_hour: u8) -> Vec<NaiveDate> {
        let dir = match fs::read_dir(&self.sessions_dir) {
            Ok(d) => d,
            Err(_) => return vec![],
        };

        let mut dates = std::collections::HashSet::new();

        for entry in dir.flatten() {
            let path = entry.path();
            if path.extension().and_then(|e| e.to_str()) != Some("jsonl") {
                continue;
            }

            if let Some((_, messages, _)) = load_session_file(&path) {
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
/// Returns `(meta, messages, is_closed)` or `None` if the file is unreadable
/// or has a malformed first line.
///
/// Messages are returned as `StoredMessage` (with timestamps preserved).
fn load_session_file(path: &Path) -> Option<(SessionMeta, Vec<StoredMessage>, bool)> {
    let file = fs::File::open(path).ok()?;
    let mut lines = BufReader::new(file).lines();

    // First line must be the meta object
    let first = lines.next()?.ok()?;
    let meta_line: MetaLine = serde_json::from_str(first.trim()).ok()?;
    let mut meta = meta_line.meta;

    let mut messages = Vec::new();
    let mut is_closed = false;

    for raw in lines.flatten() {
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
        } else if value.get("timestamp").is_some() {
            match serde_json::from_value::<StoredMessage>(value) {
                Ok(stored) => messages.push(stored),
                Err(e) => {
                    warn!("Skipping malformed message in {}: {e}", path.display());
                }
            }
        }
    }

    Some((meta, messages, is_closed))
}
