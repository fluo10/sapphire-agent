//! Periodic log (daily / weekly / monthly / yearly) generation, digest
//! catch-up, and system-prompt injection.
//!
//! Each log file lives under `memory/{daily,weekly,monthly,yearly}/{stem}.md`
//! and carries a YAML frontmatter `digest:` array of importance-ordered
//! bullets. The top-N items per kind are injected into the system prompt
//! (see `workspace.rs`) so the agent retains long-horizon context without
//! paying full-body token cost.
//!
//! Input strategy (calendar-aligned, not chained):
//! - Daily ← sessions for that local date.
//! - Weekly ← the 7 daily bodies of that ISO week (Mon–Sun).
//! - Monthly ← all daily bodies whose local date falls in that calendar
//!   month. Monthlies intentionally do **not** chain through weeklies:
//!   ISO weeks straddle month boundaries, so chaining would leak days
//!   across the boundary.
//! - Yearly ← the 12 monthly bodies of that calendar year.

use crate::provider::{ChatMessage, ContentPart, Provider, Role};
use crate::session::{SessionMeta, SessionStore, StoredMessage};
use chrono::{Datelike, Local, NaiveDate};
use sapphire_workspace::WorkspaceState;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use tracing::{info, warn};

// ---------------------------------------------------------------------------
// LogKind + path layout
// ---------------------------------------------------------------------------

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum LogKind {
    Daily,
    Weekly,
    Monthly,
    Yearly,
}

impl LogKind {
    /// Directory name under `memory/` for this kind.
    pub fn dir(self) -> &'static str {
        match self {
            LogKind::Daily => "daily",
            LogKind::Weekly => "weekly",
            LogKind::Monthly => "monthly",
            LogKind::Yearly => "yearly",
        }
    }

    /// Label used in generated file headings and LLM prompts.
    pub fn heading_label(self) -> &'static str {
        match self {
            LogKind::Daily => "Daily Log",
            LogKind::Weekly => "Weekly Log",
            LogKind::Monthly => "Monthly Log",
            LogKind::Yearly => "Yearly Log",
        }
    }
}

/// `memory/{dir}/{stem}.md` — workspace-relative path.
pub fn log_rel_path(kind: LogKind, stem: &str) -> PathBuf {
    Path::new("memory")
        .join(kind.dir())
        .join(format!("{stem}.md"))
}

/// Absolute path to the log file under `workspace_dir`.
pub fn log_abs_path(workspace_dir: &Path, kind: LogKind, stem: &str) -> PathBuf {
    workspace_dir.join(log_rel_path(kind, stem))
}

// ---------------------------------------------------------------------------
// Stem helpers
// ---------------------------------------------------------------------------

/// `"YYYY-MM-DD"`.
pub fn daily_stem(date: NaiveDate) -> String {
    date.format("%Y-%m-%d").to_string()
}

/// `"YYYY-Www"` using the **ISO week-year** (which may differ from the
/// calendar year at year boundaries — e.g. 2023-01-01 → `"2022-W52"`).
pub fn weekly_stem(iso_year: i32, iso_week: u32) -> String {
    format!("{iso_year:04}-W{iso_week:02}")
}

/// `"YYYY-MM"`.
pub fn monthly_stem(year: i32, month: u32) -> String {
    format!("{year:04}-{month:02}")
}

/// `"YYYY"`.
pub fn yearly_stem(year: i32) -> String {
    format!("{year:04}")
}

/// Convenience: derive the weekly stem from a date that falls within the
/// target ISO week. Uses `NaiveDate::iso_week()` to get the correct ISO
/// year (not the calendar year).
pub fn weekly_stem_from_date(date: NaiveDate) -> String {
    let iso = date.iso_week();
    weekly_stem(iso.year(), iso.week())
}

// ---------------------------------------------------------------------------
// Daily log generation (moved from daily_log.rs)
// ---------------------------------------------------------------------------

/// Returns dates that have session messages but no corresponding daily log
/// file, up to and including yesterday (local time).
pub fn pending_daily_dates(
    session_store: &SessionStore,
    workspace_dir: &Path,
    boundary_hour: u8,
) -> Vec<NaiveDate> {
    let today = crate::session::local_date_for_timestamp(Local::now(), boundary_hour);
    let mut dates = session_store.all_session_dates(boundary_hour);
    dates.retain(|&date| date < today && !log_abs_path(workspace_dir, LogKind::Daily, &daily_stem(date)).exists());
    dates
}

/// Generate a daily log for `date` and write it to `memory/daily/YYYY-MM-DD.md`.
/// Returns `Ok(true)` if a log was written, `Ok(false)` if there were no
/// sessions for that day (no-op).
///
/// Writes a YAML frontmatter `digest:` array (top-10 importance-ordered
/// bullets) alongside the Markdown summary body.
pub async fn generate_daily_log(
    session_store: &SessionStore,
    provider: &dyn Provider,
    ws_state: &Arc<Mutex<WorkspaceState>>,
    date: NaiveDate,
    boundary_hour: u8,
) -> anyhow::Result<bool> {
    let sessions = session_store.sessions_for_day(date, boundary_hour);
    if sessions.is_empty() {
        info!("No sessions found for {date}, skipping daily log");
        return Ok(false);
    }

    let transcript = format_sessions(&sessions, date);
    let stem = daily_stem(date);
    write_log_with_digest(
        provider,
        ws_state,
        LogKind::Daily,
        &stem,
        "conversation transcripts",
        &transcript,
    )
    .await?;

    Ok(true)
}

// ---------------------------------------------------------------------------
// Shared LLM-backed writer
// ---------------------------------------------------------------------------

/// Drive one LLM call that produces both a body summary and a YAML
/// frontmatter `digest:` array, then write `memory/{kind}/{stem}.md`.
///
/// The LLM is instructed to return a Markdown file shaped as:
/// ```text
/// ---
/// digest:
///   - …
/// ---
///
/// …body…
/// ```
///
/// `input_description` is a short phrase like `"conversation transcripts"`
/// or `"daily logs for ISO week 2026-W16"` used in the system prompt.
///
/// `input_body` is the concatenated source material (transcript, daily
/// bodies, etc.) passed as the user message.
async fn write_log_with_digest(
    provider: &dyn Provider,
    ws_state: &Arc<Mutex<WorkspaceState>>,
    kind: LogKind,
    stem: &str,
    input_description: &str,
    input_body: &str,
) -> anyhow::Result<()> {
    let label = kind.heading_label();
    let system = format!(
        "You are generating a concise {label} entry from {input_description}.

Output a Markdown file with this exact structure:

1. A YAML frontmatter block delimited by `---` lines, containing exactly one key:
   `digest`: an array of 5–10 importance-ordered bullet points (strings). Each
   bullet is a single short sentence capturing a key decision, topic, task
   completed, or unresolved item. Order: most important first.
2. After the closing `---`, a blank line, then the Markdown summary body.
   The body is free-form but should cover: key topics discussed, decisions
   made, tasks completed, and unresolved items.

Do NOT include a top-level `#` heading — one will be added automatically.
Write in the same language as the source material (Japanese if it is in
Japanese). Emit raw Markdown, not a fenced code block."
    );

    let user_msg = ChatMessage::user(input_body);
    let response = provider.chat(Some(&system), &[user_msg], None).await?;
    let raw = response
        .text
        .unwrap_or_else(|| String::from("(no summary generated)"));

    let (digest, body) = parse_digest_response(&raw);
    if digest.is_empty() {
        warn!(
            "{} for {stem}: LLM response had no digest array; writing with empty digest",
            label
        );
    }
    let heading = format!("# {label}: {stem}\n\n");
    let file = serialize_log_file(&digest, &format!("{heading}{}\n", body.trim_end()))?;

    let rel = log_rel_path(kind, stem);
    ws_state
        .lock()
        .expect("WorkspaceState mutex poisoned")
        .write_file(&rel, &file)?;
    info!("{} written: {}", label, rel.display());
    Ok(())
}

/// Parse an LLM response shaped as `--- digest:[…] --- body`. On malformed
/// input returns `(vec![], raw_body_with_wrappers_stripped)` so a best-effort
/// body still gets written.
fn parse_digest_response(raw: &str) -> (Vec<String>, String) {
    let cleaned = strip_code_fence(raw);
    match crate::frontmatter::split(&cleaned) {
        Some((fm, body)) => {
            let items = crate::frontmatter::parse_mapping(fm)
                .get("digest")
                .and_then(yaml_value_to_string_vec)
                .unwrap_or_default();
            (items, body.trim_start_matches('\n').to_string())
        }
        None => (Vec::new(), cleaned.trim_start_matches('\n').to_string()),
    }
}

/// Some models wrap everything in ` ```markdown … ``` `. Strip that.
fn strip_code_fence(raw: &str) -> String {
    let trimmed = raw.trim();
    let open_tags: &[&str] = &["```markdown\n", "```md\n", "```\n"];
    for tag in open_tags {
        if let Some(rest) = trimmed.strip_prefix(tag)
            && let Some(inner) = rest.strip_suffix("```").or_else(|| rest.strip_suffix("```\n"))
        {
            return inner.trim_end().to_string();
        }
    }
    trimmed.to_string()
}

/// Coerce a `serde_yaml::Value` into `Vec<String>`, dropping non-string items
/// with a warn log (non-strings indicate the LLM drifted from the format).
fn yaml_value_to_string_vec(v: &serde_yaml::Value) -> Option<Vec<String>> {
    let seq = v.as_sequence()?;
    let mut out = Vec::with_capacity(seq.len());
    for item in seq {
        match item.as_str() {
            Some(s) => out.push(s.trim().to_string()),
            None => warn!("digest item is not a string: {item:?} — dropping"),
        }
    }
    Some(out)
}

/// Build a log file string from a digest list and body. Uses the shared
/// `frontmatter::serialize` helper so the formatting is consistent.
fn serialize_log_file(digest: &[String], body: &str) -> anyhow::Result<String> {
    let mut mapping = serde_yaml::Mapping::new();
    let seq = digest
        .iter()
        .map(|s| serde_yaml::Value::String(s.clone()))
        .collect::<Vec<_>>();
    mapping.insert(
        serde_yaml::Value::String("digest".to_string()),
        serde_yaml::Value::Sequence(seq),
    );
    crate::frontmatter::serialize(&mapping, body)
}

// ---------------------------------------------------------------------------
// Session → transcript formatting
// ---------------------------------------------------------------------------

/// Format sessions into a transcript suitable for LLM summarisation. Tool
/// use / tool result parts are skipped to keep context concise.
fn format_sessions(sessions: &[(SessionMeta, Vec<StoredMessage>)], date: NaiveDate) -> String {
    let mut parts = vec![format!("Conversations for {date}:\n")];

    for (meta, messages) in sessions {
        let thread = meta.thread_id.as_deref().unwrap_or("main");
        parts.push(format!(
            "## Session {} (thread: {})\n",
            meta.session_id, thread
        ));

        for msg in messages {
            let text: Vec<&str> = msg
                .parts
                .iter()
                .filter_map(|p| {
                    if let ContentPart::Text(t) = p {
                        Some(t.as_str())
                    } else {
                        None
                    }
                })
                .collect();

            if text.is_empty() {
                continue;
            }

            let role_label = match msg.role {
                Role::User => "User",
                Role::Assistant => "Assistant",
            };
            let local_ts = msg.timestamp.with_timezone(&Local);
            parts.push(format!(
                "[{}] {}: {}",
                local_ts.format("%H:%M"),
                role_label,
                text.join(" ")
            ));
        }

        parts.push(String::new());
    }

    parts.join("\n")
}

// ---------------------------------------------------------------------------
// Digest catch-up (back-fill existing dailies that lack a `digest:` array)
// ---------------------------------------------------------------------------

/// True if the file either has no frontmatter or is missing a non-empty
/// `digest:` sequence. Used by `catchup_missing_daily_digests` to decide
/// which existing dailies need an LLM-driven back-fill pass.
fn needs_digest_catchup(raw: &str) -> bool {
    match crate::frontmatter::split(raw) {
        None => true,
        Some((fm, _)) => {
            let mapping = crate::frontmatter::parse_mapping(fm);
            match mapping.get("digest").and_then(|v| v.as_sequence()) {
                Some(seq) => seq.is_empty(),
                None => true,
            }
        }
    }
}

/// Insert/replace the `digest:` key on a file's frontmatter while preserving
/// every other key and the body. If the file has no frontmatter at all,
/// synthesise one wrapping the existing content.
fn upsert_digest(raw: &str, digest: &[String]) -> anyhow::Result<String> {
    let (mut mapping, body) = match crate::frontmatter::split(raw) {
        Some((fm, body)) => (
            crate::frontmatter::parse_mapping(fm),
            body.trim_start_matches('\n').to_string(),
        ),
        None => (serde_yaml::Mapping::new(), raw.to_string()),
    };
    let seq = digest
        .iter()
        .map(|s| serde_yaml::Value::String(s.clone()))
        .collect::<Vec<_>>();
    mapping.insert(
        serde_yaml::Value::String("digest".to_string()),
        serde_yaml::Value::Sequence(seq),
    );
    crate::frontmatter::serialize(&mapping, &body)
}

/// Ask the LLM to emit a `digest:` frontmatter block for an existing log
/// body. Returns the parsed bullet list (empty `Vec` if parsing failed).
async fn extract_digest_from_body(
    provider: &dyn Provider,
    kind: LogKind,
    body: &str,
) -> anyhow::Result<Vec<String>> {
    let label = kind.heading_label();
    let system = format!(
        "You are extracting a concise digest from an existing {label} entry.

Output ONLY a YAML frontmatter block, exactly:

---
digest:
  - …
  - …
---

The `digest` array contains 5–10 importance-ordered bullet points (strings).
Each bullet is a single short sentence capturing a key decision, topic,
task completed, or unresolved item. Most important first. Write in the
same language as the source. Do NOT emit any text outside the frontmatter
block."
    );
    let user = ChatMessage::user(body);
    let resp = provider.chat(Some(&system), &[user], None).await?;
    let raw = resp.text.unwrap_or_default();
    let (items, _) = parse_digest_response(&raw);
    Ok(items)
}

/// Strip any leading frontmatter from `raw` and return the body. Used as
/// LLM input during digest back-fill so the model sees only the summary.
fn body_without_frontmatter(raw: &str) -> String {
    match crate::frontmatter::split(raw) {
        Some((_, body)) => body.trim_start_matches('\n').to_string(),
        None => raw.to_string(),
    }
}

/// Scan `memory/daily/*.md` and, for every file that lacks a non-empty
/// `digest:` array, call the LLM to produce one and merge it into the
/// file's frontmatter (preserving every other key — crucial for the
/// memory tool's `last_read_at` / `read_count` keys).
///
/// Errors are logged per-file, not propagated, so a single bad file does
/// not stall the pass. Returns the number of files successfully updated.
pub async fn catchup_missing_daily_digests(
    provider: &dyn Provider,
    ws_state: &Arc<Mutex<WorkspaceState>>,
    workspace_dir: &Path,
) -> usize {
    let dir = workspace_dir.join("memory").join("daily");
    let entries = match std::fs::read_dir(&dir) {
        Ok(e) => e,
        Err(_) => return 0,
    };

    let mut pending = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|s| s.to_str()) != Some("md") {
            continue;
        }
        let Ok(raw) = std::fs::read_to_string(&path) else {
            continue;
        };
        if needs_digest_catchup(&raw) {
            pending.push((path, raw));
        }
    }

    if pending.is_empty() {
        return 0;
    }
    info!("Back-filling digest for {} daily log(s)…", pending.len());

    let mut filled = 0;
    for (path, raw) in pending {
        let display_path = path.display().to_string();
        let body = body_without_frontmatter(&raw);
        let items = match extract_digest_from_body(provider, LogKind::Daily, &body).await {
            Ok(v) if !v.is_empty() => v,
            Ok(_) => {
                warn!("Empty digest extracted from {display_path} — skipping");
                continue;
            }
            Err(e) => {
                warn!("Failed to extract digest from {display_path}: {e:#}");
                continue;
            }
        };
        let new_raw = match upsert_digest(&raw, &items) {
            Ok(s) => s,
            Err(e) => {
                warn!("Failed to merge digest for {display_path}: {e:#}");
                continue;
            }
        };
        let rel = path
            .strip_prefix(workspace_dir)
            .map(|p| p.to_path_buf())
            .unwrap_or_else(|_| path.clone());
        match ws_state
            .lock()
            .expect("WorkspaceState mutex poisoned")
            .write_file(&rel, &new_raw)
        {
            Ok(()) => {
                filled += 1;
                info!("Digest added: {}", rel.display());
            }
            Err(e) => warn!("Failed to write digest for {display_path}: {e:#}"),
        }
    }
    filled
}

// ---------------------------------------------------------------------------
// Startup catch-up
// ---------------------------------------------------------------------------

/// Generate all pending daily logs (e.g. from days the agent was offline).
/// Errors are logged but not propagated so startup is not blocked.
/// Returns the number of logs successfully generated, so callers can decide
/// whether to invalidate downstream caches.
pub async fn catchup_pending_daily_logs(
    session_store: &SessionStore,
    provider: &dyn Provider,
    ws_state: &Arc<Mutex<WorkspaceState>>,
    workspace_dir: &Path,
    boundary_hour: u8,
) -> usize {
    let pending = pending_daily_dates(session_store, workspace_dir, boundary_hour);
    if pending.is_empty() {
        return 0;
    }
    info!("Generating {} pending daily log(s)…", pending.len());
    let mut generated = 0;
    for date in pending {
        match generate_daily_log(session_store, provider, ws_state, date, boundary_hour).await {
            Ok(true) => generated += 1,
            Ok(false) => {}
            Err(e) => warn!("Failed to generate daily log for {date}: {e:#}"),
        }
    }
    generated
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stems_basic() {
        let d = NaiveDate::from_ymd_opt(2026, 4, 15).unwrap();
        assert_eq!(daily_stem(d), "2026-04-15");
        assert_eq!(monthly_stem(2026, 4), "2026-04");
        assert_eq!(yearly_stem(2026), "2026");
    }

    #[test]
    fn weekly_stem_iso_year_may_differ_from_calendar_year() {
        // 2023-01-01 is Sunday → ISO week 52 of 2022.
        let d = NaiveDate::from_ymd_opt(2023, 1, 1).unwrap();
        assert_eq!(weekly_stem_from_date(d), "2022-W52");
        // 2024-12-30 is Monday → ISO week 1 of 2025.
        let d = NaiveDate::from_ymd_opt(2024, 12, 30).unwrap();
        assert_eq!(weekly_stem_from_date(d), "2025-W01");
    }

    #[test]
    fn weekly_stem_zero_padded() {
        assert_eq!(weekly_stem(2026, 3), "2026-W03");
        assert_eq!(weekly_stem(2026, 16), "2026-W16");
    }

    #[test]
    fn paths() {
        let abs = log_abs_path(Path::new("/ws"), LogKind::Weekly, "2026-W16");
        assert_eq!(abs, PathBuf::from("/ws/memory/weekly/2026-W16.md"));
        let rel = log_rel_path(LogKind::Monthly, "2026-04");
        assert_eq!(rel, PathBuf::from("memory/monthly/2026-04.md"));
    }

    #[test]
    fn parse_digest_response_happy_path() {
        let raw = "---\ndigest:\n  - item A\n  - item B\n  - item C\n---\n\n# body line\n";
        let (items, body) = parse_digest_response(raw);
        assert_eq!(items, vec!["item A", "item B", "item C"]);
        assert_eq!(body.trim(), "# body line");
    }

    #[test]
    fn parse_digest_response_missing_frontmatter() {
        let raw = "# Just a body\n\nno digest here\n";
        let (items, body) = parse_digest_response(raw);
        assert!(items.is_empty());
        assert!(body.contains("no digest here"));
    }

    #[test]
    fn parse_digest_response_strips_code_fence() {
        let raw = "```markdown\n---\ndigest:\n  - x\n---\n\nbody\n```";
        let (items, body) = parse_digest_response(raw);
        assert_eq!(items, vec!["x"]);
        assert_eq!(body.trim(), "body");
    }

    #[test]
    fn parse_digest_response_drops_non_string_items() {
        let raw = "---\ndigest:\n  - ok\n  - 42\n  - also ok\n---\n\nbody\n";
        let (items, _) = parse_digest_response(raw);
        assert_eq!(items, vec!["ok", "also ok"]);
    }

    #[test]
    fn serialize_log_file_includes_digest() {
        let out = serialize_log_file(&["a".into(), "b".into()], "# H\n\nbody\n").unwrap();
        assert!(out.starts_with("---\n"));
        assert!(out.contains("digest:"));
        assert!(out.contains("- a"));
        assert!(out.contains("- b"));
        assert!(out.contains("# H"));
        assert!(out.contains("body"));
    }

    #[test]
    fn needs_digest_catchup_cases() {
        assert!(needs_digest_catchup("# body only\n"));
        assert!(needs_digest_catchup(
            "---\nother: 1\n---\n# body\n"
        ));
        assert!(needs_digest_catchup(
            "---\ndigest: []\n---\n# body\n"
        ));
        assert!(!needs_digest_catchup(
            "---\ndigest:\n  - one\n---\n# body\n"
        ));
    }

    #[test]
    fn upsert_digest_preserves_unrelated_keys() {
        let raw = "---\nlast_read_at: 2026-04-15T21:57:41.312Z\nread_count: 3\n---\n\n# Daily Log: 2026-04-15\n\nbody\n";
        let new = upsert_digest(raw, &["one".into(), "two".into()]).unwrap();
        let (fm, body) = crate::frontmatter::split(&new).unwrap();
        let mapping = crate::frontmatter::parse_mapping(fm);
        assert!(mapping.get("last_read_at").is_some());
        assert_eq!(mapping.get("read_count").and_then(|v| v.as_u64()), Some(3));
        let digest: Vec<String> = mapping
            .get("digest")
            .and_then(|v| v.as_sequence())
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap().to_string())
            .collect();
        assert_eq!(digest, vec!["one", "two"]);
        assert!(body.contains("# Daily Log: 2026-04-15"));
        assert!(body.contains("body"));
    }

    #[test]
    fn upsert_digest_synthesises_frontmatter_when_missing() {
        let raw = "# Daily Log: 2026-04-15\n\nbody\n";
        let new = upsert_digest(raw, &["only".into()]).unwrap();
        assert!(new.starts_with("---\n"));
        let (fm, body) = crate::frontmatter::split(&new).unwrap();
        let mapping = crate::frontmatter::parse_mapping(fm);
        let digest: Vec<String> = mapping
            .get("digest")
            .and_then(|v| v.as_sequence())
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap().to_string())
            .collect();
        assert_eq!(digest, vec!["only"]);
        assert!(body.contains("# Daily Log: 2026-04-15"));
        assert!(body.contains("body"));
    }

    #[test]
    fn upsert_digest_replaces_existing_digest() {
        let raw = "---\ndigest:\n  - stale\nother: keep\n---\n\nbody\n";
        let new = upsert_digest(raw, &["fresh".into()]).unwrap();
        let (fm, _) = crate::frontmatter::split(&new).unwrap();
        let mapping = crate::frontmatter::parse_mapping(fm);
        assert_eq!(mapping.get("other").and_then(|v| v.as_str()), Some("keep"));
        let digest: Vec<String> = mapping
            .get("digest")
            .and_then(|v| v.as_sequence())
            .unwrap()
            .iter()
            .map(|v| v.as_str().unwrap().to_string())
            .collect();
        assert_eq!(digest, vec!["fresh"]);
    }
}
