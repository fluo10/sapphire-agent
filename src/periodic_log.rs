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

use chrono::{Datelike, NaiveDate};
use std::path::{Path, PathBuf};

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
}
