//! Daily log generation.
//!
//! Reads JSONL sessions for a given day and uses the LLM provider to produce
//! a Markdown summary written to `memory/daily/YYYY-MM-DD.md`.

use crate::provider::{ChatMessage, ContentPart, Provider, Role};
use crate::session::{SessionMeta, SessionStore, StoredMessage};
use chrono::{Local, NaiveDate};
use std::path::{Path, PathBuf};
use tracing::{info, warn};

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Returns dates that have session messages but no corresponding daily log file,
/// up to and including yesterday (local time).
pub fn pending_log_dates(
    session_store: &SessionStore,
    workspace_dir: &Path,
    boundary_hour: u8,
) -> Vec<NaiveDate> {
    let today = crate::session::local_date_for_timestamp(Local::now(), boundary_hour);

    let mut dates = session_store.all_session_dates(boundary_hour);
    dates.retain(|&date| date < today && !daily_log_path(workspace_dir, date).exists());
    dates
}

/// Generate a daily log for `date` and write it to `memory/daily/YYYY-MM-DD.md`.
/// No-op if there are no sessions for that day.
pub async fn generate_daily_log(
    session_store: &SessionStore,
    provider: &dyn Provider,
    workspace_dir: &Path,
    date: NaiveDate,
    boundary_hour: u8,
) -> anyhow::Result<()> {
    let sessions = session_store.sessions_for_day(date, boundary_hour);

    if sessions.is_empty() {
        info!("No sessions found for {date}, skipping daily log");
        return Ok(());
    }

    let transcript = format_sessions(&sessions, date);

    let system = "You are generating a concise daily log entry from conversation transcripts. \
        Summarize: key topics discussed, decisions made, tasks completed, and unresolved items. \
        Write in the same language as the conversations (Japanese if conversations are in Japanese). \
        Output plain Markdown suitable for a daily log file. \
        Do not include a top-level heading — that will be added automatically.";

    let user_msg = ChatMessage::user(&transcript);
    let response = provider.chat(Some(system), &[user_msg], None).await?;
    let summary = response
        .text
        .unwrap_or_else(|| "(no summary generated)".to_string());

    let log_path = daily_log_path(workspace_dir, date);
    if let Some(parent) = log_path.parent() {
        std::fs::create_dir_all(parent)?;
    }

    let content = format!("# Daily Log: {date}\n\n{summary}\n");
    std::fs::write(&log_path, &content)?;

    info!("Daily log written: {}", log_path.display());
    Ok(())
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

pub fn daily_log_path(workspace_dir: &Path, date: NaiveDate) -> PathBuf {
    workspace_dir
        .join("memory")
        .join("daily")
        .join(format!("{date}.md"))
}

/// Format sessions into a transcript suitable for LLM summarization.
/// Tool use / tool result parts are skipped to keep context concise.
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
// Startup catch-up
// ---------------------------------------------------------------------------

/// Generate all pending daily logs (e.g. from days the agent was offline).
/// Errors are logged but not propagated so startup is not blocked.
pub async fn catchup_pending_logs(
    session_store: &SessionStore,
    provider: &dyn Provider,
    workspace_dir: &Path,
    boundary_hour: u8,
) {
    let pending = pending_log_dates(session_store, workspace_dir, boundary_hour);
    if pending.is_empty() {
        return;
    }
    info!("Generating {} pending daily log(s)…", pending.len());
    for date in pending {
        if let Err(e) =
            generate_daily_log(session_store, provider, workspace_dir, date, boundary_hour).await
        {
            warn!("Failed to generate daily log for {date}: {e:#}");
        }
    }
}
