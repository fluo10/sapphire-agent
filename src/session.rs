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
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::{self, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
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
        ChatMessage { role: self.role, parts: self.parts }
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

// ---------------------------------------------------------------------------
// SessionStore
// ---------------------------------------------------------------------------

pub struct SessionStore {
    sessions_dir: PathBuf,
}

impl SessionStore {
    pub fn new(sessions_dir: PathBuf) -> Self {
        Self { sessions_dir }
    }

    fn session_path(&self, session_id: &str) -> PathBuf {
        self.sessions_dir.join(format!("{session_id}.jsonl"))
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
        };
        let line = serde_json::to_string(&MetaLine { meta })?;
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(self.session_path(&session_id))?;
        writeln!(file, "{line}")?;
        Ok(session_id)
    }

    /// Append a `ChatMessage` (with current timestamp) to an existing session.
    pub fn append(&self, session_id: &str, msg: &ChatMessage) -> anyhow::Result<()> {
        let stored = StoredMessage::from_chat(msg);
        let line = serde_json::to_string(&stored)?;
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(self.session_path(session_id))?;
        writeln!(file, "{line}")?;
        Ok(())
    }

    /// Close a session by appending a `closed_at` marker.
    /// The session becomes inactive; future messages create a new session.
    pub fn close_session(&self, session_id: &str) -> anyhow::Result<()> {
        let line = serde_json::to_string(&ClosedLine { closed_at: Utc::now() })?;
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(self.session_path(session_id))?;
        writeln!(file, "{line}")?;
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
        // Collect (session_id, meta, messages, is_closed) per file
        type SessionEntry = (String, ConversationKey, Vec<ChatMessage>, bool);
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
                // Later entries (larger ULID) overwrite earlier ones
                history.insert(key.clone(), messages);
                active.insert(key, session_id);
            }
        }

        (history, active)
    }
}

// ---------------------------------------------------------------------------
// File parsing helpers
// ---------------------------------------------------------------------------

/// Parse a single session `.jsonl` file.
///
/// Returns `(meta, messages, is_closed)` or `None` if the file is unreadable
/// or has a malformed first line.
fn load_session_file(path: &Path) -> Option<(SessionMeta, Vec<ChatMessage>, bool)> {
    let file = fs::File::open(path).ok()?;
    let mut lines = BufReader::new(file).lines();

    // First line must be the meta object
    let first = lines.next()?.ok()?;
    let meta_line: MetaLine = serde_json::from_str(first.trim()).ok()?;
    let meta = meta_line.meta;

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
        } else if value.get("timestamp").is_some() {
            match serde_json::from_value::<StoredMessage>(value) {
                Ok(stored) => messages.push(stored.into_chat_message()),
                Err(e) => {
                    warn!("Skipping malformed message in {}: {e}", path.display());
                }
            }
        }
    }

    Some((meta, messages, is_closed))
}
