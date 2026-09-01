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
use crate::session::{IntradayDigestLine, SessionMeta, StoredMessage};
use crate::tool_result_cache::ToolResultCache;
use anyhow::Result;
use chrono::{DateTime, NaiveDate, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::Write;
use std::path::PathBuf;
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
        /// `None` means the result had nowhere to be stored — the cache
        /// was unavailable when this line was written. Distinct from a
        /// hash whose entry has since been evicted, but only in how it
        /// arose: a reader treats both as "the pairing is here, the
        /// content is not", and `load_part` produces the same
        /// `MISSING_RESULT` for each.
        ///
        /// `Option<String>` also reads a bare string as `Some`, so a
        /// line written before this field became optional still loads.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        sha256: Option<String>,
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
    /// `None` when the tool-result cache directory could not be opened
    /// at startup (read-only or missing `~/.cache` / `%LOCALAPPDATA%`).
    /// Degrades rather than making the whole store unusable: a session
    /// must still load. See `store_part`'s `None` arm for what this
    /// costs — currently nothing reachable in production, since
    /// `run_llm_turn` does not persist `tool_use`/`tool_result` at all
    /// yet (issue #191).
    cache: Option<Arc<ToolResultCache>>,
    /// `session_id` → the id of the last event written.
    ///
    /// An append needs its parent, and re-reading the whole file to
    /// find the tip would make every write cost the length of the
    /// conversation. Populated lazily on the first append to a session.
    tips: Mutex<HashMap<String, Uuid>>,
}

impl AcpSessionStore {
    pub fn new(base_dir: PathBuf, cache: Option<Arc<ToolResultCache>>) -> Self {
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

    /// Absolute path of an existing session's file, for callers that
    /// need to read raw bytes (e.g. `mtime` for `updated_at`) rather than
    /// going through the store's own accessors. Mirrors the old
    /// `SessionStore::absolute_path_for`, which `session/list` used the
    /// same way.
    pub fn absolute_path_for(&self, session_id: &str) -> Option<PathBuf> {
        self.find(session_id)
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

    /// Read the tip, write `body(parent)` as the next line, and record
    /// the new tip — all under one held lock.
    ///
    /// `tip()` and the write used to be separate lock acquisitions, so
    /// two threads appending to the *same session* on the *same store*
    /// could both observe the same tip and both write with that parent,
    /// fabricating a fork that means nothing (unlike the cross-process
    /// case this format is meant to tolerate, where a fork reflects a
    /// writer that genuinely didn't know about the other's latest
    /// event). Holding `tips` across the read-modify-write closes that
    /// window: the whole "who is the tip, append after them, become the
    /// tip" sequence is now one atomic step. Reading the file to find a
    /// lazy tip happens while the lock is held too — `events()` never
    /// touches `tips`, so this cannot deadlock.
    fn append_line(
        &self,
        session_id: &str,
        id: Uuid,
        body: impl FnOnce(Option<Uuid>) -> Line,
    ) -> Result<()> {
        let mut tips = self.tips.lock().unwrap();
        let parent = match tips.get(session_id) {
            Some(tip) => Some(*tip),
            None => self
                .events(session_id)
                .and_then(|evs| evs.last().map(|e| e.id)),
        };
        let path = self
            .find(session_id)
            .ok_or_else(|| anyhow::anyhow!("no ACP session '{session_id}'"))?;
        let mut file = std::fs::OpenOptions::new().append(true).open(&path)?;
        writeln!(file, "{}", serde_json::to_string(&body(parent))?)?;
        tips.insert(session_id.to_string(), id);
        Ok(())
    }

    pub fn append_message(&self, session_id: &str, msg: &ChatMessage) -> Result<()> {
        let parts = msg
            .parts
            .iter()
            .map(|part| self.store_part(part))
            .collect::<Result<Vec<_>>>()?;
        let id = Uuid::now_v7();
        let role = msg.role.clone();
        self.append_line(session_id, id, move |parent| Line::Message {
            id,
            parent,
            at: Utc::now(),
            role,
            parts,
        })
    }

    pub fn append_title(&self, session_id: &str, title: &str) -> Result<()> {
        let id = Uuid::now_v7();
        let title = title.to_string();
        self.append_line(session_id, id, move |parent| Line::Title {
            id,
            parent,
            at: Utc::now(),
            title,
        })
    }

    /// ACP exposes no close-or-delete operation yet, so nothing in
    /// production ever writes a `Closed` event — the read path's refusal
    /// of a closed session (`adopt_session` in `src/serve/acp.rs`) is
    /// defensive, ahead of there being any way to produce one. This is
    /// compiled only for the tests that exercise that refusal path,
    /// which is why it is `#[cfg(test)]` rather than a real store method
    /// right now: once a real close/delete request lands, its handler
    /// becomes this method's production caller and the attribute comes
    /// off.
    #[cfg(test)]
    pub fn close(&self, session_id: &str) -> Result<()> {
        let id = Uuid::now_v7();
        self.append_line(session_id, id, move |parent| Line::Closed {
            id,
            parent,
            at: Utc::now(),
        })
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
            } => {
                // A cache miss on read degrades gracefully; a missing
                // cache on write must not be allowed to degrade any
                // further. Writing `Other` here would drop the
                // `tool_use_id`, leaving a `tool_use` with no matching
                // `tool_result` — which the API rejects outright, so the
                // session would fail to load rather than load thinner.
                let sha256 = match &self.cache {
                    Some(cache) => Some(cache.put(content)?),
                    None => {
                        warn!(
                            "Tool-result cache unavailable; recording '{tool_use_id}' \
                             with no content"
                        );
                        None
                    }
                };
                StoredPart::ToolResultRef {
                    tool_use_id: tool_use_id.clone(),
                    sha256,
                }
            }
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

    /// The conversation as the model should see it.
    ///
    /// Walks the parent chain rather than trusting file order, because
    /// file order is only accidentally right: it agrees with the chain
    /// for a session one process wrote, and says nothing useful for one
    /// that was synced or merged.
    pub fn history(&self, session_id: &str) -> Option<Vec<ChatMessage>> {
        let events = self.events(session_id)?;

        // parent -> children, so the walk is a lookup rather than a scan.
        let mut children: HashMap<Option<Uuid>, Vec<&Event>> = HashMap::new();
        for event in &events {
            children.entry(event.parent).or_default().push(event);
        }

        let mut out = Vec::new();
        let mut cursor = None;
        let mut seen = 0usize;
        loop {
            let Some(next) = children.get(&cursor) else {
                break;
            };
            if next.len() > 1 {
                warn!(
                    "ACP session {session_id}: {} events share one parent; taking the first branch. Two writers appended to this session and there is no merge story yet.",
                    next.len()
                );
            }
            let event = next[0];
            if let EventBody::Message { role, parts } = &event.body {
                out.push(ChatMessage {
                    role: role.clone(),
                    parts: parts.iter().map(|p| self.load_part(p)).collect(),
                    input_kind: None,
                    user_id: None,
                });
            }
            cursor = Some(event.id);

            // A hand-edited or partially-synced file could contain a
            // cycle. The chain cannot be longer than the file.
            seen += 1;
            if seen > events.len() {
                warn!("ACP session {session_id}: the parent chain cycles; stopping");
                break;
            }
        }

        // A `tool_use` can end up on disk without its `tool_result` —
        // the second `append_message` failed (the cache write erred),
        // the process died between the two appends, a sync landed only
        // half the pair. The write side already warns and moves on
        // rather than losing the turn; left uncorrected here, the gap
        // it leaves is worse than a lost message, because the Anthropic
        // API rejects the *whole* conversation on reload, not just the
        // orphaned call — the session becomes permanently unloadable.
        // Synthesise the missing half instead of dropping the `tool_use`:
        // dropping would erase the fact that the agent attempted the
        // call, and `MISSING_RESULT` is exactly the shape a cache miss
        // already produces, so the model sees nothing it doesn't already
        // know how to handle.
        let tool_use_ids: Vec<String> = out
            .iter()
            .flat_map(|m| &m.parts)
            .filter_map(|p| match p {
                ContentPart::ToolUse { id, .. } => Some(id.clone()),
                _ => None,
            })
            .collect();
        let answered: std::collections::HashSet<String> = out
            .iter()
            .flat_map(|m| &m.parts)
            .filter_map(|p| match p {
                ContentPart::ToolResult { tool_use_id, .. } => Some(tool_use_id.clone()),
                _ => None,
            })
            .collect();
        let orphan_results: Vec<ContentPart> = tool_use_ids
            .into_iter()
            .filter(|id| !answered.contains(id))
            .map(|tool_use_id| ContentPart::ToolResult {
                tool_use_id,
                content: MISSING_RESULT.to_string(),
            })
            .collect();
        if !orphan_results.is_empty() {
            out.push(ChatMessage {
                role: Role::User,
                parts: orphan_results,
                input_kind: None,
                user_id: None,
            });
        }

        Some(out)
    }

    fn load_part(&self, part: &StoredPart) -> ContentPart {
        match part {
            StoredPart::Text(t) => ContentPart::Text(t.clone()),
            StoredPart::ToolUse { id, name, input } => ContentPart::ToolUse {
                id: id.clone(),
                name: name.clone(),
                input: input.clone(),
            },
            StoredPart::ToolResultRef {
                tool_use_id,
                sha256,
            } => ContentPart::ToolResult {
                tool_use_id: tool_use_id.clone(),
                // Absent either way — no hash was ever written, or the
                // hash's entry is gone. The model can call the tool
                // again if it needs to; what it cannot recover from is
                // an unpaired `tool_use`.
                content: sha256
                    .as_ref()
                    .and_then(|sha| self.cache.as_ref()?.get(sha))
                    .unwrap_or_else(|| MISSING_RESULT.to_string()),
            },
            StoredPart::Other => {
                ContentPart::Text("[a message part that this version does not store]".to_string())
            }
        }
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

impl AcpSessionStore {
    /// Today's digest per session, in the vocabulary the cross-session
    /// digest builder already speaks.
    ///
    /// The text comes from `cache`; everything around it comes from the
    /// session's header. The projection is read-only and deliberate:
    /// `build_today_digest_for_namespace` is shared with four other
    /// stores, and teaching it a second shape would spread this store's
    /// format into code with no business knowing it.
    pub fn intraday_digests_for_day(
        &self,
        date: NaiveDate,
        boundary_hour: u8,
        cache: &crate::digest_cache::DigestCache,
    ) -> Vec<(SessionMeta, IntradayDigestLine)> {
        let (day_start, day_end) = crate::session::day_window(date, boundary_hour);
        let mut out = Vec::new();
        for session_id in self.all_session_ids() {
            let Some(digest) = cache.get(&session_id) else {
                continue;
            };
            if digest.digest_at < day_start || digest.digest_at >= day_end {
                continue;
            }
            let Some(summary) = self.summary(&session_id) else {
                continue;
            };
            out.push((self.project_meta(&summary), digest));
        }
        out.sort_by_key(|(meta, _)| meta.created_at);
        out
    }

    /// Sessions whose newest message post-dates their cached digest, and
    /// which said something inside `date`'s window.
    ///
    /// Deliberately not an idle threshold. The sweep runs on a fixed
    /// cadence, so the only question worth asking is whether anything
    /// was added since last time — which makes the next update time
    /// predictable instead of a function of when the user stopped
    /// typing. Both sides of the comparison are durable (a cache file
    /// and a session event), so a restart does not reset the schedule.
    pub fn sessions_needing_digest(
        &self,
        cache: &crate::digest_cache::DigestCache,
        date: NaiveDate,
        boundary_hour: u8,
    ) -> Vec<String> {
        let (day_start, day_end) = crate::session::day_window(date, boundary_hour);
        let mut due = Vec::new();
        for session_id in self.all_session_ids() {
            let Some(events) = self.events(&session_id) else {
                continue;
            };
            let last_message = events
                .iter()
                .filter(|e| matches!(e.body, EventBody::Message { .. }))
                .map(|e| e.at)
                .next_back();
            let Some(last_message) = last_message else {
                continue;
            };
            // Said nothing today: its day is already written up.
            if last_message < day_start || last_message >= day_end {
                continue;
            }
            if cache
                .get(&session_id)
                .is_some_and(|d| d.digest_at >= last_message)
            {
                continue;
            }
            due.push(session_id);
        }
        due.sort();
        due
    }

    /// This store's sessions in the vocabulary the log and digest
    /// builders share. `channel` is `"acp"` so those builders can route
    /// on it the way they already route on `"rpc"` and
    /// `"device-default"`.
    fn project_meta(&self, summary: &SessionSummary) -> SessionMeta {
        let header = &summary.header;
        SessionMeta {
            session_id: header.session_id.clone(),
            // ACP has no rooms. Empty rather than synthetic, so a
            // room-derived namespace lookup can never accidentally
            // match one.
            room_id: String::new(),
            thread_id: None,
            channel: "acp".to_string(),
            created_at: header.created_at,
            public_id: None,
            namespace: Some(header.namespace.clone()),
            project: None,
            device_id: None,
            room_profile: None,
            title: summary.title.clone(),
        }
    }

    /// Every session id this store holds, across namespaces.
    fn all_session_ids(&self) -> Vec<String> {
        crate::session::collect_session_files(&self.base_dir, KIND)
            .into_iter()
            .filter_map(|p| Some(p.file_stem()?.to_str()?.to_string()))
            .collect()
    }
}

impl AcpSessionStore {
    /// This store's sessions for one local day, in the shape
    /// `format_sessions` reads.
    ///
    /// Only text is projected. A tool result whose content has fallen
    /// out of the cache would otherwise write its placeholder sentence
    /// into a permanent, searchable record — and the daily log is a
    /// narrative of what was said, not a transcript of what was read.
    pub fn sessions_for_day(
        &self,
        date: NaiveDate,
        boundary_hour: u8,
    ) -> Vec<(SessionMeta, Vec<StoredMessage>)> {
        let (day_start, day_end) = crate::session::day_window(date, boundary_hour);
        let mut out = Vec::new();
        for session_id in self.all_session_ids() {
            let Some(summary) = self.summary(&session_id) else {
                continue;
            };
            let Some(events) = self.events(&session_id) else {
                continue;
            };
            let messages: Vec<StoredMessage> = events
                .iter()
                .filter(|e| e.at >= day_start && e.at < day_end)
                .filter_map(|e| match &e.body {
                    EventBody::Message { role, parts } => {
                        let text: Vec<ContentPart> = parts
                            .iter()
                            .filter_map(|p| match p {
                                StoredPart::Text(t) if !t.trim().is_empty() => {
                                    Some(ContentPart::Text(t.clone()))
                                }
                                _ => None,
                            })
                            .collect();
                        if text.is_empty() {
                            return None;
                        }
                        Some(StoredMessage {
                            timestamp: e.at,
                            role: role.clone(),
                            parts: text,
                            input_kind: None,
                            user_id: None,
                            report_meta: None,
                        })
                    }
                    _ => None,
                })
                .collect();
            if messages.is_empty() {
                continue;
            }
            out.push((self.project_meta(&summary), messages));
        }
        out.sort_by_key(|(meta, _)| meta.created_at);
        out
    }

    /// Local dates on which this store has at least one message from a
    /// session `predicate` accepts, so daily-log catch-up knows which
    /// days are pending *for its own namespace*. Without this filter
    /// every namespace's catch-up would see every other namespace's
    /// ACP dates as pending too: `generate_daily_log` would then find
    /// no in-namespace sessions for that date, write nothing, and the
    /// date would stay pending forever — re-walked on every startup
    /// and catch-up tick.
    pub fn session_dates<F>(&self, boundary_hour: u8, predicate: F) -> Vec<NaiveDate>
    where
        F: Fn(&SessionMeta) -> bool,
    {
        let mut dates = std::collections::HashSet::new();
        for session_id in self.all_session_ids() {
            let Some(summary) = self.summary(&session_id) else {
                continue;
            };
            if !predicate(&self.project_meta(&summary)) {
                continue;
            }
            let Some(events) = self.events(&session_id) else {
                continue;
            };
            for event in events {
                if !matches!(event.body, EventBody::Message { .. }) {
                    continue;
                }
                let local = event.at.with_timezone(&chrono::Local);
                dates.insert(crate::session::local_date_for_timestamp(
                    local,
                    boundary_hour,
                ));
            }
        }
        let mut sorted: Vec<NaiveDate> = dates.into_iter().collect();
        sorted.sort();
        sorted
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::{ChatMessage, ContentPart, Role};

    /// The local date "now" belongs to under `boundary_hour`.
    ///
    /// Deliberately not `Local::now().date_naive()`. A timestamp before
    /// the boundary hour belongs to the *previous* local day, so the
    /// naive date and the store's day window disagree for those hours —
    /// which made these tests pass all day and fail only when CI
    /// happened to run between midnight and 04:00.
    fn today(boundary_hour: u8) -> chrono::NaiveDate {
        crate::session::local_date_for_timestamp(chrono::Local::now(), boundary_hour)
    }

    /// The rule the helper above encodes, pinned on its own so a future
    /// reader does not have to infer it from a failing CI run.
    #[test]
    fn a_timestamp_before_the_boundary_belongs_to_the_previous_day() {
        use chrono::TimeZone as _;
        let at_two_am = chrono::Local
            .with_ymd_and_hms(2026, 9, 1, 2, 0, 0)
            .single()
            .expect("an unambiguous local time");
        assert_eq!(
            crate::session::local_date_for_timestamp(at_two_am, 4),
            chrono::NaiveDate::from_ymd_opt(2026, 8, 31).unwrap(),
            "02:00 under a 04:00 boundary is still the previous day"
        );

        let at_ten_am = chrono::Local
            .with_ymd_and_hms(2026, 9, 1, 10, 0, 0)
            .single()
            .expect("an unambiguous local time");
        assert_eq!(
            crate::session::local_date_for_timestamp(at_ten_am, 4),
            chrono::NaiveDate::from_ymd_opt(2026, 9, 1).unwrap(),
        );
    }

    fn store() -> (tempfile::TempDir, AcpSessionStore) {
        let dir = tempfile::tempdir().unwrap();
        let cache_dir = dir.path().join("cache");
        let cache = crate::tool_result_cache::ToolResultCache::open(cache_dir).unwrap();
        let store = AcpSessionStore::new(dir.path().join("sessions"), Some(cache));
        (dir, store)
    }

    /// A store whose tool-result cache failed to open — the degraded
    /// shape `AcpSessionStore::new` must still function under.
    fn store_without_cache() -> (tempfile::TempDir, AcpSessionStore) {
        let dir = tempfile::tempdir().unwrap();
        let store = AcpSessionStore::new(dir.path().join("sessions"), None);
        (dir, store)
    }

    #[test]
    fn a_header_round_trips() {
        let (_d, store) = store();
        store.create("s1", "default", "/home/u/project").unwrap();

        let summary = store.summary("s1").expect("the session exists");
        assert_eq!(summary.header.session_id, "s1");
        assert_eq!(summary.header.namespace, "default");
        assert_eq!(summary.header.cwd, "/home/u/project");
        assert!(!summary.is_closed);
    }

    /// The first event has no parent; every later one points at the
    /// event that was the tip when it was written. This chain is what
    /// makes an offline divergence detectable, so it is the property
    /// most worth pinning.
    #[test]
    fn the_parent_chain_records_append_order() {
        let (_d, store) = store();
        store.create("s1", "default", "/p").unwrap();
        store
            .append_message("s1", &ChatMessage::user("one"))
            .unwrap();
        store
            .append_message("s1", &ChatMessage::assistant("two"))
            .unwrap();
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

        assert!(store.summary("s1").unwrap().is_closed);
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

    /// A cache that failed to open at startup must degrade the store,
    /// not break it: a session must still be creatable and a tool
    /// result still appendable (as an unrecoverable marker) rather than
    /// erroring the whole turn.
    #[test]
    fn a_tool_result_appends_cleanly_when_the_cache_is_unavailable() {
        let (dir, store) = store_without_cache();
        store.create("s1", "default", "/p").unwrap();
        let result = store.append_message(
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
        );
        assert!(
            result.is_ok(),
            "appending must not fail just because the cache is unavailable: {result:?}"
        );
        drop(dir);
    }

    /// The regression this task exists for. A tool result stored with no
    /// cache must still read back as a `ToolResult` carrying its
    /// `tool_use_id` — an unpaired `tool_use` is rejected by the API,
    /// which is a broken session rather than a degraded one.
    #[test]
    fn a_result_stored_without_a_cache_keeps_its_pairing() {
        let (_d, store) = store_without_cache();
        store.create("s1", "default", "/p").unwrap();
        store
            .append_message(
                "s1",
                &tool_result_message("c1", "content that cannot be cached"),
            )
            .unwrap();

        let history = store.history("s1").expect("the session loads");
        assert_eq!(
            history[0].parts[0],
            ContentPart::ToolResult {
                tool_use_id: "c1".to_string(),
                content: MISSING_RESULT.to_string(),
            },
            "the id survives even though the content never had anywhere to go"
        );
    }

    /// The two ways a result can be absent — never cached, or cached and
    /// later evicted — must read back identically. A reader has no
    /// reason to tell them apart: both mean "the pairing is here, the
    /// content is not".
    #[test]
    fn never_cached_and_evicted_read_back_the_same() {
        let (dir_a, cached) = store();
        cached.create("s1", "default", "/p").unwrap();
        cached
            .append_message("s1", &tool_result_message("c1", "gone later"))
            .unwrap();
        std::fs::remove_dir_all(dir_a.path().join("cache")).unwrap();
        let evicted = cached.history("s1").unwrap();

        let (_dir_b, uncached) = store_without_cache();
        uncached.create("s1", "default", "/p").unwrap();
        uncached
            .append_message("s1", &tool_result_message("c1", "never stored"))
            .unwrap();
        let never = uncached.history("s1").unwrap();

        assert_eq!(evicted[0].parts, never[0].parts);
    }

    /// A cache that is present still stores the hash, not the content.
    #[test]
    fn a_cached_result_still_records_its_hash() {
        let (_d, store) = store();
        store.create("s1", "default", "/p").unwrap();
        store
            .append_message("s1", &tool_result_message("c1", "the real output"))
            .unwrap();

        let raw = std::fs::read_to_string(store.path_for_test("s1")).unwrap();
        assert!(!raw.contains("the real output"), "got {raw}");
        assert!(raw.contains("tool_result_ref"), "got {raw}");
        assert_eq!(
            store.history("s1").unwrap()[0].parts[0],
            ContentPart::ToolResult {
                tool_use_id: "c1".to_string(),
                content: "the real output".to_string(),
            }
        );
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
        store
            .append_message("used", &ChatMessage::user("hi"))
            .unwrap();
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

    /// Two threads appending to the *same session* on the *same store*
    /// must be ordered by the `tips` lock, not raced: the second writer
    /// must always see the first writer's freshly-written tip. Before
    /// the fix, `tip()` (read) and the write that followed it were two
    /// separate lock acquisitions, so both threads could observe the
    /// same tip and both write with that parent — a fabricated fork
    /// that means nothing, unlike the cross-process case this format
    /// tolerates on purpose.
    ///
    /// This is a thread-based test rather than a structural one: since
    /// the fix makes the whole "read tip, write, become tip" sequence
    /// one held lock, the two threads are fully serialized by
    /// construction and this test is deterministic post-fix (not
    /// flaky) — there is no longer a window where they could interleave
    /// mid-operation. It is the direct expression of the invariant:
    /// after both appends land, no two events may share a parent.
    #[test]
    fn concurrent_appends_to_one_session_do_not_fork() {
        let (_d, store) = store();
        store.create("s1", "default", "/p").unwrap();
        let store = std::sync::Arc::new(store);

        let barrier = std::sync::Arc::new(std::sync::Barrier::new(2));
        let handles: Vec<_> = (0..2)
            .map(|i| {
                let store = std::sync::Arc::clone(&store);
                let barrier = std::sync::Arc::clone(&barrier);
                std::thread::spawn(move || {
                    barrier.wait();
                    store
                        .append_message("s1", &ChatMessage::user(format!("msg{i}")))
                        .unwrap();
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }

        let events = store.events("s1").expect("the session exists");
        assert_eq!(events.len(), 2, "both appends must land: {events:?}");
        let parents: std::collections::HashSet<Option<uuid::Uuid>> =
            events.iter().map(|e| e.parent).collect();
        assert_eq!(
            parents.len(),
            2,
            "each event must have a distinct parent — a shared parent means \
             a fabricated fork: {events:?}"
        );
        assert!(
            events.iter().any(|e| e.parent.is_none()),
            "exactly one event is the session's first: {events:?}"
        );
    }

    #[test]
    fn history_comes_back_in_chain_order() {
        let (_d, store) = store();
        store.create("s1", "default", "/p").unwrap();
        store
            .append_message("s1", &ChatMessage::user("one"))
            .unwrap();
        store
            .append_message("s1", &ChatMessage::assistant("two"))
            .unwrap();
        store.append_title("s1", "ignored by history").unwrap();
        store
            .append_message("s1", &ChatMessage::user("three"))
            .unwrap();

        let history = store.history("s1").expect("the session exists");
        let texts: Vec<&str> = history
            .iter()
            .filter_map(|m| match m.parts.first() {
                Some(ContentPart::Text(t)) => Some(t.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(texts, vec!["one", "two", "three"]);
    }

    /// A tool result survives a round trip through the cache.
    #[test]
    fn a_cached_tool_result_is_restored_whole() {
        let (_d, store) = store();
        store.create("s1", "default", "/p").unwrap();
        store
            .append_message("s1", &tool_result_message("c1", "the original output"))
            .unwrap();

        let history = store.history("s1").unwrap();
        assert_eq!(
            history[0].parts[0],
            ContentPart::ToolResult {
                tool_use_id: "c1".to_string(),
                content: "the original output".to_string(),
            }
        );
    }

    /// The whole reason a resume summary is unnecessary: a lost result
    /// degrades to a sentence, and the `tool_use`/`tool_result` pairing
    /// the API validates stays intact. Losing the cache must never make
    /// a session unloadable.
    #[test]
    fn a_lost_tool_result_becomes_a_placeholder_rather_than_an_error() {
        let (dir, store) = store();
        store.create("s1", "default", "/p").unwrap();
        store
            .append_message("s1", &tool_result_message("c1", "gone tomorrow"))
            .unwrap();

        // Simulate the cache being cleared between runs.
        std::fs::remove_dir_all(dir.path().join("cache")).unwrap();

        let history = store.history("s1").expect("the session still loads");
        assert_eq!(
            history[0].parts[0],
            ContentPart::ToolResult {
                tool_use_id: "c1".to_string(),
                content: MISSING_RESULT.to_string(),
            },
            "the pairing must survive even though the content did not"
        );
    }

    /// Two events claiming the same parent means two writers appended to
    /// one session — the divergence the parent chain exists to catch.
    /// Until there is a merge story, take the first branch and say so
    /// rather than interleaving two conversations by timestamp.
    #[test]
    fn a_branch_is_reported_and_the_first_child_is_taken() {
        let (_d, store) = store();
        store.create("s1", "default", "/p").unwrap();
        store
            .append_message("s1", &ChatMessage::user("root"))
            .unwrap();

        // Hand-write a second child of the root to forge a divergence.
        let root = store.events("s1").unwrap()[0].id;
        let path = store.path_for_test("s1");
        let forged = serde_json::json!({
            "kind": "message",
            "id": Uuid::now_v7(),
            "parent": root,
            "at": Utc::now(),
            "role": "user",
            "parts": [{"text": "the other branch"}],
        });
        let mut f = std::fs::OpenOptions::new()
            .append(true)
            .open(&path)
            .unwrap();
        use std::io::Write as _;
        writeln!(f, "{forged}").unwrap();
        drop(f);

        // The legitimate continuation, appended after the forgery, is
        // the file's *last* line but the root's *second* child.
        store
            .append_message("s1", &ChatMessage::user("mine"))
            .unwrap();

        let history = store.history("s1").expect("a branched session still loads");
        let texts: Vec<&str> = history
            .iter()
            .filter_map(|m| match m.parts.first() {
                Some(ContentPart::Text(t)) => Some(t.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(
            texts,
            vec!["root", "the other branch"],
            "one branch, not both interleaved"
        );
    }

    /// An event whose parent is not in the file — a truncated sync, a
    /// hand-edit — must not silently drop the rest of the conversation
    /// or spin. Everything reachable from the root is returned.
    #[test]
    fn an_orphan_event_is_skipped_rather_than_ending_the_walk() {
        let (_d, store) = store();
        store.create("s1", "default", "/p").unwrap();
        store
            .append_message("s1", &ChatMessage::user("root"))
            .unwrap();

        let path = store.path_for_test("s1");
        let orphan = serde_json::json!({
            "kind": "message",
            "id": Uuid::now_v7(),
            "parent": Uuid::now_v7(),
            "at": Utc::now(),
            "role": "user",
            "parts": [{"text": "unreachable"}],
        });
        let mut f = std::fs::OpenOptions::new()
            .append(true)
            .open(&path)
            .unwrap();
        use std::io::Write as _;
        writeln!(f, "{orphan}").unwrap();
        drop(f);

        let history = store.history("s1").expect("the session still loads");
        assert_eq!(
            history.len(),
            1,
            "the orphan is not reachable from the root"
        );
    }

    /// The read-side fix for the write side's warn-and-continue: an
    /// `append_message` for a `tool_result` can fail after its
    /// `tool_use` sibling already landed, or the process can die between
    /// the two. Either way, a `tool_use` with nothing answering it must
    /// not brick the session — `history()` closes the gap itself rather
    /// than handing the API an unpaired call on reload.
    #[test]
    fn an_unanswered_tool_use_gets_a_synthesised_result() {
        let (_d, store) = store();
        store.create("s1", "default", "/p").unwrap();
        store
            .append_message("s1", &ChatMessage::user("run it"))
            .unwrap();

        // Hand-write a `tool_use` with no matching `tool_result` after
        // it — standing in for a crash or a failed cache write between
        // the two appends `run_llm_turn` makes.
        let parent = store.events("s1").unwrap()[0].id;
        let path = store.path_for_test("s1");
        let orphan_use = serde_json::json!({
            "kind": "message",
            "id": Uuid::now_v7(),
            "parent": parent,
            "at": Utc::now(),
            "role": "assistant",
            "parts": [{"tool_use": {"id": "call-1", "name": "risky", "input": {}}}],
        });
        let mut f = std::fs::OpenOptions::new()
            .append(true)
            .open(&path)
            .unwrap();
        use std::io::Write as _;
        writeln!(f, "{orphan_use}").unwrap();
        drop(f);

        let history = store.history("s1").expect("the session still loads");
        let last = history.last().expect("a synthesised message was appended");
        assert_eq!(
            last.parts,
            vec![ContentPart::ToolResult {
                tool_use_id: "call-1".to_string(),
                content: MISSING_RESULT.to_string(),
            }],
            "the unanswered tool_use gets a placeholder result so the \
             pairing holds on reload"
        );
    }

    /// The fix above must only fire on a genuine gap — a session whose
    /// `tool_use`/`tool_result` pair is already complete gains nothing.
    #[test]
    fn a_complete_pair_gains_no_synthesised_result() {
        let (_d, store) = store();
        store.create("s1", "default", "/p").unwrap();
        store
            .append_message(
                "s1",
                &ChatMessage::assistant_with_tools(
                    None,
                    vec![crate::provider::ToolCall {
                        id: "call-1".to_string(),
                        name: "risky".to_string(),
                        input: serde_json::json!({}),
                    }],
                ),
            )
            .unwrap();
        store
            .append_message("s1", &tool_result_message("call-1", "the real output"))
            .unwrap();

        let history = store.history("s1").expect("the session loads");
        let result_count = history
            .iter()
            .flat_map(|m| &m.parts)
            .filter(|p| matches!(p, ContentPart::ToolResult { .. }))
            .count();
        assert_eq!(
            result_count, 1,
            "no extra result should be synthesised when the pairing is already complete"
        );
    }

    fn tool_result_message(tool_use_id: &str, content: &str) -> ChatMessage {
        ChatMessage {
            role: Role::User,
            parts: vec![ContentPart::ToolResult {
                tool_use_id: tool_use_id.to_string(),
                content: content.to_string(),
            }],
            input_kind: None,
            user_id: None,
        }
    }

    fn digest_cache(dir: &tempfile::TempDir) -> Arc<crate::digest_cache::DigestCache> {
        crate::digest_cache::DigestCache::open(dir.path().join("digests")).unwrap()
    }

    /// The digest's text comes from the cache; its namespace, title and
    /// creation time come from the store's header. Neither side
    /// duplicates the other, which is what makes an external cache
    /// affordable here.
    #[test]
    fn a_digest_is_joined_against_the_stores_header() {
        let (dir, store) = store();
        let cache = digest_cache(&dir);
        store.create("s1", "work", "/p").unwrap();
        store
            .append_message("s1", &ChatMessage::user("did a thing"))
            .unwrap();
        store.append_title("s1", "parser hunt").unwrap();
        cache.put("s1", "we fixed the parser", None).unwrap();

        let today = today(4);
        let found = store.intraday_digests_for_day(today, 4, &cache);
        assert_eq!(found.len(), 1);
        let (meta, digest) = &found[0];
        assert_eq!(digest.digest, "we fixed the parser");
        assert_eq!(meta.session_id, "s1");
        assert_eq!(
            meta.channel, "acp",
            "the projection says which store this came from — \
             build_today_digest_for_namespace routes on it"
        );
        assert_eq!(meta.namespace.as_deref(), Some("work"));
        assert_eq!(meta.title.as_deref(), Some("parser hunt"));
    }

    /// Losing the cache must not lose the session.
    #[test]
    fn a_session_with_no_cached_digest_is_simply_absent() {
        let (dir, store) = store();
        let cache = digest_cache(&dir);
        store.create("s1", "work", "/p").unwrap();
        store.append_message("s1", &ChatMessage::user("x")).unwrap();

        let today = today(4);
        assert!(store.intraday_digests_for_day(today, 4, &cache).is_empty());
        assert!(store.history("s1").is_some(), "the session still loads");
    }

    /// The scheduling rule, stated as a test: a session whose newest
    /// message post-dates its cached digest is due for a new one. Not
    /// "has been idle for N minutes" — the sweep runs on a fixed
    /// cadence, so the only question is whether anything was added.
    #[test]
    fn a_session_with_events_newer_than_its_digest_is_due() {
        let (dir, store) = store();
        let cache = digest_cache(&dir);
        store.create("s1", "default", "/p").unwrap();
        cache
            .put_at(
                "s1",
                "covered up to here",
                None,
                Utc::now() - chrono::Duration::hours(1),
            )
            .unwrap();
        store
            .append_message("s1", &ChatMessage::user("something new"))
            .unwrap();

        let today = today(4);
        assert_eq!(
            store.sessions_needing_digest(&cache, today, 4),
            vec!["s1".to_string()]
        );
    }

    #[test]
    fn a_session_with_nothing_new_since_its_digest_is_not_due() {
        let (dir, store) = store();
        let cache = digest_cache(&dir);
        store.create("s1", "default", "/p").unwrap();
        store
            .append_message("s1", &ChatMessage::user("said something"))
            .unwrap();
        cache.put("s1", "covered", None).unwrap();

        let today = today(4);
        assert!(store.sessions_needing_digest(&cache, today, 4).is_empty());
    }

    #[test]
    fn a_never_digested_session_with_messages_is_due() {
        let (dir, store) = store();
        let cache = digest_cache(&dir);
        store.create("s1", "default", "/p").unwrap();
        store
            .append_message("s1", &ChatMessage::user("first words"))
            .unwrap();

        let today = today(4);
        assert_eq!(
            store.sessions_needing_digest(&cache, today, 4),
            vec!["s1".to_string()]
        );
    }

    /// A session that has said nothing today is not resurrected. Its
    /// day has been written up already; digesting it now would file old
    /// work under today.
    #[test]
    fn a_session_with_no_messages_today_is_not_due() {
        let (dir, store) = store();
        let cache = digest_cache(&dir);
        store.create("s1", "default", "/p").unwrap();
        store
            .append_message("s1", &ChatMessage::user("old news"))
            .unwrap();

        let tomorrow = today(4) + chrono::Duration::days(1);
        assert!(
            store
                .sessions_needing_digest(&cache, tomorrow, 4)
                .is_empty()
        );
    }

    /// The daily log is the durable record — the one that stays in the
    /// workspace and stays searchable. An ACP session has to reach it.
    #[test]
    fn a_days_conversation_projects_into_the_daily_log_shape() {
        let (_d, store) = store();
        store.create("s1", "work", "/p").unwrap();
        store
            .append_message("s1", &ChatMessage::user("what broke?"))
            .unwrap();
        store
            .append_message("s1", &ChatMessage::assistant("the parser"))
            .unwrap();
        store.append_title("s1", "parser hunt").unwrap();

        let today = today(4);
        let sessions = store.sessions_for_day(today, 4);
        assert_eq!(sessions.len(), 1);
        let (meta, messages) = &sessions[0];
        assert_eq!(meta.session_id, "s1");
        assert_eq!(meta.channel, "acp");
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].role, Role::User);
        assert_eq!(messages[1].role, Role::Assistant);
        assert_eq!(
            messages[0].parts[0],
            ContentPart::Text("what broke?".to_string())
        );
    }

    /// A tool result that has fallen out of the cache must not put a
    /// placeholder sentence into the permanent record. The daily log
    /// wants what was said, and `format_sessions` keeps only text parts —
    /// so tool traffic should not be projected at all.
    #[test]
    fn tool_traffic_is_left_out_of_the_daily_log_projection() {
        let (_d, store) = store();
        store.create("s1", "default", "/p").unwrap();
        store
            .append_message("s1", &tool_result_message("c1", "a big file listing"))
            .unwrap();
        store
            .append_message("s1", &ChatMessage::user("thanks"))
            .unwrap();

        let today = today(4);
        let (_, messages) = store.sessions_for_day(today, 4).remove(0);
        let texts: Vec<&str> = messages
            .iter()
            .flat_map(|m| m.parts.iter())
            .filter_map(|p| match p {
                ContentPart::Text(t) => Some(t.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(texts, vec!["thanks"]);
    }

    #[test]
    fn session_dates_lists_the_days_that_have_messages() {
        let (_d, store) = store();
        store.create("s1", "default", "/p").unwrap();
        store.append_message("s1", &ChatMessage::user("x")).unwrap();

        let today = today(4);
        assert_eq!(store.session_dates(4, |_meta| true), vec![today]);
    }

    /// A date that only has messages in another namespace's ACP session
    /// must not be reported pending for this namespace — otherwise every
    /// namespace's catch-up sees every other namespace's dates as
    /// pending forever (`generate_daily_log` finds nothing in-namespace,
    /// writes nothing, and the phantom date is re-walked every tick).
    #[test]
    fn session_dates_excludes_a_date_that_belongs_only_to_another_namespace() {
        let (_d, store) = store();
        store.create("s-default", "default", "/p").unwrap();
        store
            .append_message("s-default", &ChatMessage::user("x"))
            .unwrap();
        store.create("s-work", "work", "/p").unwrap();
        store
            .append_message("s-work", &ChatMessage::user("y"))
            .unwrap();

        let today = today(4);
        let default_dates =
            store.session_dates(4, |meta| meta.namespace.as_deref() == Some("default"));
        assert_eq!(default_dates, vec![today]);

        let work_dates = store.session_dates(4, |meta| meta.namespace.as_deref() == Some("work"));
        assert_eq!(work_dates, vec![today]);

        let ghost_dates = store.session_dates(4, |meta| meta.namespace.as_deref() == Some("ghost"));
        assert!(ghost_dates.is_empty());
    }
}
