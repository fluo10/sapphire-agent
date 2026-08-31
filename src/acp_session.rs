//! The ACP session store.
//!
//! Separate from `SessionStore` because ACP is an externally-defined
//! standard that will drift from the format we chose for ourselves, and
//! because an editor's thread list should not be showing Matrix
//! conversations. The directory-walking machinery is shared; the line
//! format is not.
//!
//! Every event carries a UUIDv7 `id` and the `parent` it was appended
//! after. That pair cannot be reconstructed later, which is why it is
//! here now even though nothing reads it yet: it is what will make an
//! offline divergence detectable if remote-workspace sync is ever
//! implemented, and what makes splitting one file per event a
//! mechanical move rather than a redesign.
//!
//! **The chain is the authority on order, not the id's timestamp.**
//! UUIDv7 embeds the writer's clock, so two devices with skewed clocks
//! produce ids that sort wrongly against each other.

use crate::provider::{ChatMessage, ContentPart, Role};
use crate::tool_result_cache::ToolResultCache;
use anyhow::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use tracing::warn;
use uuid::Uuid;

/// The directory name under `<sessions>/<namespace>/`.
const KIND: &str = "acp";

/// What the model is told when a tool result is no longer in the cache.
///
/// The pairing between `tool_use` and `tool_result` is what the API
/// validates, not the content — so a placeholder keeps the history
/// valid and the conversation's shape intact. This is why the store
/// needs no resume summary: a summary would cost a model call and throw
/// the turn structure away to solve a problem a sentence solves.
pub const MISSING_RESULT: &str =
    "[this tool result is no longer stored; call the tool again if you need it]";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SessionHeader {
    pub session_id: String,
    pub namespace: String,
    /// The client's workspace root. Required — an ACP session always
    /// reports one, which is why this is not an `Option` the way the
    /// old store's `cwd` had to be.
    pub cwd: String,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Event {
    pub id: Uuid,
    /// `None` means this is the session's first event. The header is
    /// not a root event — making it one would need a special case in
    /// every traversal.
    pub parent: Option<Uuid>,
    pub at: DateTime<Utc>,
    pub body: EventBody,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum EventBody {
    Message { role: Role, parts: Vec<StoredPart> },
    Title { title: String },
    Closed,
}

/// A message part as it appears on disk.
///
/// Distinct from `provider::ContentPart` on purpose: a tool result is a
/// hash here and the real content in memory, and that difference exists
/// only at the storage boundary. `image_cache` put its `ImageRef` into
/// `ContentPart` because those references stay in the in-memory history
/// to avoid re-billing; a tool result has to be whole when the model
/// sees it, so the reference never leaves the disk.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum StoredPart {
    Text(String),
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    ToolResultRef {
        tool_use_id: String,
        sha256: String,
    },
    /// Anything the storage layer does not model — images, for now.
    /// Kept as a marker so the message is not silently emptied.
    Other,
}

/// One line of the log. Everything is `kind`-tagged, including the
/// header: a single internally-tagged enum has no ambiguity, whereas
/// mixing a tagged event with an untagged `{"header": …}` wrapper does.
#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum Line {
    Header(SessionHeader),
    Message {
        id: Uuid,
        parent: Option<Uuid>,
        at: DateTime<Utc>,
        role: Role,
        parts: Vec<StoredPart>,
    },
    Title {
        id: Uuid,
        parent: Option<Uuid>,
        at: DateTime<Utc>,
        title: String,
    },
    Closed {
        id: Uuid,
        parent: Option<Uuid>,
        at: DateTime<Utc>,
    },
}

pub struct AcpSessionStore {
    base_dir: PathBuf,
    cache: Arc<ToolResultCache>,
    /// `session_id` → the id of the last event written.
    ///
    /// An append needs its parent, and re-reading the whole file to
    /// find the tip would make every write cost the length of the
    /// conversation. Populated lazily on the first append to a session.
    tips: Mutex<HashMap<String, Uuid>>,
}

impl AcpSessionStore {
    pub fn new(base_dir: PathBuf, cache: Arc<ToolResultCache>) -> Self {
        Self {
            base_dir,
            cache,
            tips: Mutex::new(HashMap::new()),
        }
    }

    fn path(&self, session_id: &str, namespace: &str) -> PathBuf {
        self.base_dir
            .join(namespace)
            .join(KIND)
            .join(format!("{session_id}.jsonl"))
    }

    /// Find an existing session's file by scanning namespaces.
    fn find(&self, session_id: &str) -> Option<PathBuf> {
        let target = format!("{session_id}.jsonl");
        crate::session::collect_session_files(&self.base_dir, KIND)
            .into_iter()
            .find(|p| p.file_name().and_then(|s| s.to_str()) == Some(target.as_str()))
    }

    #[cfg(test)]
    pub fn path_for_test(&self, session_id: &str) -> PathBuf {
        self.find(session_id).expect("the session exists")
    }

    pub fn create(&self, session_id: &str, namespace: &str, cwd: &str) -> Result<()> {
        let path = self.path(session_id, namespace);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let header = Line::Header(SessionHeader {
            session_id: session_id.to_string(),
            namespace: namespace.to_string(),
            cwd: cwd.to_string(),
            created_at: Utc::now(),
        });
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)?;
        writeln!(file, "{}", serde_json::to_string(&header)?)?;
        Ok(())
    }

    /// The id of the last event, reading the file once if this process
    /// has not appended to the session yet.
    fn tip(&self, session_id: &str) -> Option<Uuid> {
        if let Some(id) = self.tips.lock().unwrap().get(session_id) {
            return Some(*id);
        }
        let last = self.events(session_id)?.last().map(|e| e.id)?;
        self.tips
            .lock()
            .unwrap()
            .insert(session_id.to_string(), last);
        Some(last)
    }

    fn append_line(&self, session_id: &str, id: Uuid, line: Line) -> Result<()> {
        let path = self
            .find(session_id)
            .ok_or_else(|| anyhow::anyhow!("no ACP session '{session_id}'"))?;
        let mut file = std::fs::OpenOptions::new().append(true).open(&path)?;
        writeln!(file, "{}", serde_json::to_string(&line)?)?;
        self.tips.lock().unwrap().insert(session_id.to_string(), id);
        Ok(())
    }

    pub fn append_message(&self, session_id: &str, msg: &ChatMessage) -> Result<()> {
        let parts = msg
            .parts
            .iter()
            .map(|part| self.store_part(part))
            .collect::<Result<Vec<_>>>()?;
        let id = Uuid::now_v7();
        self.append_line(
            session_id,
            id,
            Line::Message {
                id,
                parent: self.tip(session_id),
                at: Utc::now(),
                role: msg.role.clone(),
                parts,
            },
        )
    }

    pub fn append_title(&self, session_id: &str, title: &str) -> Result<()> {
        let id = Uuid::now_v7();
        self.append_line(
            session_id,
            id,
            Line::Title {
                id,
                parent: self.tip(session_id),
                at: Utc::now(),
                title: title.to_string(),
            },
        )
    }

    pub fn close(&self, session_id: &str) -> Result<()> {
        let id = Uuid::now_v7();
        self.append_line(
            session_id,
            id,
            Line::Closed {
                id,
                parent: self.tip(session_id),
                at: Utc::now(),
            },
        )
    }

    /// A tool result's content goes to the cache; the log keeps a hash.
    fn store_part(&self, part: &ContentPart) -> Result<StoredPart> {
        Ok(match part {
            ContentPart::Text(t) => StoredPart::Text(t.clone()),
            ContentPart::ToolUse { id, name, input } => StoredPart::ToolUse {
                id: id.clone(),
                name: name.clone(),
                input: input.clone(),
            },
            ContentPart::ToolResult {
                tool_use_id,
                content,
            } => StoredPart::ToolResultRef {
                tool_use_id: tool_use_id.clone(),
                sha256: self.cache.put(content)?,
            },
            // Images are not carried by this version. Recorded as a
            // marker rather than dropped, so a message that was only an
            // image does not read back as an empty one.
            _ => StoredPart::Other,
        })
    }
}

impl AcpSessionStore {
    fn lines(&self, session_id: &str) -> Option<Vec<Line>> {
        let path = self.find(session_id)?;
        let text = std::fs::read_to_string(path).ok()?;
        Some(
            text.lines()
                .filter(|l| !l.trim().is_empty())
                .filter_map(|l| match serde_json::from_str::<Line>(l) {
                    Ok(line) => Some(line),
                    Err(e) => {
                        warn!("ACP session {session_id}: skipping unreadable line: {e}");
                        None
                    }
                })
                .collect(),
        )
    }

    pub fn header(&self, session_id: &str) -> Option<SessionHeader> {
        self.lines(session_id)?.into_iter().find_map(|l| match l {
            Line::Header(h) => Some(h),
            _ => None,
        })
    }

    /// Every event in the order the file holds them.
    ///
    /// File order, not chain order — Task 3's reader is what walks the
    /// chain. They agree for a session written by one process, which is
    /// every session today.
    pub fn events(&self, session_id: &str) -> Option<Vec<Event>> {
        Some(
            self.lines(session_id)?
                .into_iter()
                .filter_map(|l| match l {
                    Line::Header(_) => None,
                    Line::Message {
                        id,
                        parent,
                        at,
                        role,
                        parts,
                    } => Some(Event {
                        id,
                        parent,
                        at,
                        body: EventBody::Message { role, parts },
                    }),
                    Line::Title {
                        id,
                        parent,
                        at,
                        title,
                    } => Some(Event {
                        id,
                        parent,
                        at,
                        body: EventBody::Title { title },
                    }),
                    Line::Closed { id, parent, at } => Some(Event {
                        id,
                        parent,
                        at,
                        body: EventBody::Closed,
                    }),
                })
                .collect(),
        )
    }

    pub fn is_closed(&self, session_id: &str) -> bool {
        self.events(session_id)
            .map(|evs| evs.iter().any(|e| matches!(e.body, EventBody::Closed)))
            .unwrap_or(false)
    }

    /// What the listing needs about one session, oldest first.
    ///
    /// Scoped to a namespace rather than global: another namespace's
    /// sessions are not this caller's to see. The ACP handler checks
    /// the boundary too — this is the store declining to hand them over
    /// in the first place.
    ///
    /// Everything here comes out of the one read the header already
    /// requires, which is why `has_messages` and `title` are computed
    /// eagerly rather than left to per-session follow-up calls.
    pub fn list_summaries(&self, namespace: &str) -> Vec<SessionSummary> {
        let dir = self.base_dir.join(namespace).join(KIND);
        let Ok(entries) = std::fs::read_dir(&dir) else {
            return Vec::new();
        };
        let mut out: Vec<SessionSummary> = entries
            .flatten()
            .filter_map(|e| {
                let stem = e.path().file_stem()?.to_str()?.to_string();
                self.summary(&stem)
            })
            .collect();
        out.sort_by_key(|s| s.header.created_at);
        out
    }

    /// One session's listing row. Public because `session/load` and
    /// `session/resume` need `cwd`, `title` and `is_closed` for a single
    /// session without paying for the whole namespace's listing.
    pub fn summary(&self, session_id: &str) -> Option<SessionSummary> {
        let lines = self.lines(session_id)?;
        let mut header = None;
        let mut title = None;
        let mut has_messages = false;
        let mut is_closed = false;
        for line in lines {
            match line {
                Line::Header(h) => header = Some(h),
                Line::Message { .. } => has_messages = true,
                // Last title wins: a session can be retitled.
                Line::Title { title: t, .. } => title = Some(t),
                Line::Closed { .. } => is_closed = true,
            }
        }
        Some(SessionSummary {
            header: header?,
            title,
            has_messages,
            is_closed,
        })
    }
}

/// One row of the session list.
#[derive(Debug, Clone, PartialEq)]
pub struct SessionSummary {
    pub header: SessionHeader,
    pub title: Option<String>,
    /// False for a session the editor opened and never typed into.
    pub has_messages: bool,
    pub is_closed: bool,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::{ChatMessage, ContentPart, Role};

    fn store() -> (tempfile::TempDir, AcpSessionStore) {
        let dir = tempfile::tempdir().unwrap();
        let cache_dir = dir.path().join("cache");
        let cache = crate::tool_result_cache::ToolResultCache::open(cache_dir).unwrap();
        let store = AcpSessionStore::new(dir.path().join("sessions"), cache);
        (dir, store)
    }

    #[test]
    fn a_header_round_trips() {
        let (_d, store) = store();
        store.create("s1", "default", "/home/u/project").unwrap();

        let h = store.header("s1").expect("the session exists");
        assert_eq!(h.session_id, "s1");
        assert_eq!(h.namespace, "default");
        assert_eq!(h.cwd, "/home/u/project");
        assert!(!store.is_closed("s1"));
    }

    /// The first event has no parent; every later one points at the
    /// event that was the tip when it was written. This chain is what
    /// makes an offline divergence detectable, so it is the property
    /// most worth pinning.
    #[test]
    fn the_parent_chain_records_append_order() {
        let (_d, store) = store();
        store.create("s1", "default", "/p").unwrap();
        store.append_message("s1", &ChatMessage::user("one")).unwrap();
        store.append_message("s1", &ChatMessage::assistant("two")).unwrap();
        store.append_title("s1", "a title").unwrap();

        let events = store.events("s1").expect("the session exists");
        assert_eq!(events.len(), 3);
        assert_eq!(events[0].parent, None, "the first event has no parent");
        assert_eq!(events[1].parent, Some(events[0].id));
        assert_eq!(events[2].parent, Some(events[1].id));
    }

    /// Ids are UUIDv7 so a directory listing sorts chronologically once
    /// events become files. The chain, not the timestamp, is the
    /// authority on order — but the ids must still be v7.
    #[test]
    fn event_ids_are_uuid_v7() {
        let (_d, store) = store();
        store.create("s1", "default", "/p").unwrap();
        store.append_message("s1", &ChatMessage::user("x")).unwrap();

        let events = store.events("s1").unwrap();
        assert_eq!(events[0].id.get_version_num(), 7);
    }

    /// Closing appends an event rather than rewriting anything, so the
    /// log stays append-only.
    #[test]
    fn closing_appends_an_event() {
        let (_d, store) = store();
        store.create("s1", "default", "/p").unwrap();
        store.append_message("s1", &ChatMessage::user("x")).unwrap();
        store.close("s1").unwrap();

        assert!(store.is_closed("s1"));
        let events = store.events("s1").unwrap();
        assert_eq!(events.len(), 2);
        assert!(matches!(events[1].body, EventBody::Closed));
    }

    /// A tool result is written to the cache and referenced by hash;
    /// the log never carries the content.
    #[test]
    fn a_tool_result_is_stored_by_reference() {
        let (dir, store) = store();
        store.create("s1", "default", "/p").unwrap();
        store
            .append_message(
                "s1",
                &ChatMessage {
                    role: Role::User,
                    parts: vec![ContentPart::ToolResult {
                        tool_use_id: "c1".to_string(),
                        content: "a very long file listing".to_string(),
                    }],
                    input_kind: None,
                    user_id: None,
                },
            )
            .unwrap();

        let raw = std::fs::read_to_string(store.path_for_test("s1")).unwrap();
        assert!(
            !raw.contains("a very long file listing"),
            "the content must live in the cache, not the log: {raw}"
        );
        assert!(raw.contains("tool_result_ref"), "got {raw}");
        drop(dir);
    }

    /// The listing is per namespace. Another namespace's sessions are
    /// not this connection's to see — the boundary is enforced in the
    /// ACP handler too, but the store should not hand them over either.
    #[test]
    fn listing_is_scoped_to_one_namespace() {
        let (_d, store) = store();
        store.create("mine", "default", "/p").unwrap();
        store.create("theirs", "someone-else", "/p").unwrap();

        let ids: Vec<String> = store
            .list_summaries("default")
            .into_iter()
            .map(|s| s.header.session_id)
            .collect();
        assert_eq!(ids, vec!["mine".to_string()]);
    }

    /// The editor opens a thread on every panel open, so most sessions
    /// are created and never typed into. The listing says which ones
    /// carry a message, and the handler drops the rest — computed from
    /// the read the header already needs, so it costs nothing extra.
    #[test]
    fn a_summary_reports_whether_anything_was_said() {
        let (_d, store) = store();
        store.create("empty", "default", "/p").unwrap();
        store.create("used", "default", "/p").unwrap();
        store.append_message("used", &ChatMessage::user("hi")).unwrap();
        store.append_title("used", "greetings").unwrap();

        let by_id: std::collections::HashMap<String, SessionSummary> = store
            .list_summaries("default")
            .into_iter()
            .map(|s| (s.header.session_id.clone(), s))
            .collect();

        assert!(!by_id["empty"].has_messages);
        assert_eq!(by_id["empty"].title, None);
        assert!(by_id["used"].has_messages);
        assert_eq!(by_id["used"].title.as_deref(), Some("greetings"));
    }
}
