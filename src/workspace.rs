use crate::config::DigestConfig;
use crate::periodic_log::{self, LogKind};
use std::path::{Path, PathBuf};
use std::time::SystemTime;
use tokio::sync::Mutex;
use tracing::{debug, info};

/// Maximum characters loaded from a single workspace file (matches openclaw default).
const MAX_FILE_CHARS: usize = 20_000;

// ---------------------------------------------------------------------------
// Workspace file definitions
// ---------------------------------------------------------------------------

/// A single workspace file entry: one or more candidate filenames to try (in
/// order), plus the Markdown heading to use when injecting into the system
/// prompt.
struct WorkspaceFileDef {
    /// Candidate filenames tried in order; the first one found is used.
    candidates: &'static [&'static str],
    /// Heading inserted above the file content (e.g. "# Agent Instructions").
    heading: &'static str,
}

/// Ordered list of workspace files, following openclaw's convention.
/// Files that don't exist are silently skipped. MEMORY.md is **not**
/// included here — it lives under `memory/<namespace>/MEMORY.md` and is
/// assembled per turn from the room's namespace chain.
/// See: https://github.com/openclaw/openclaw (src/agents/workspace.ts)
static WORKSPACE_FILES: &[WorkspaceFileDef] = &[
    // openclaw uses "AGENTS.md" (plural); we also accept "AGENT.md" for
    // users who create the file without the trailing 's'.
    WorkspaceFileDef {
        candidates: &["AGENTS.md", "AGENT.md"],
        heading: "# Agent Instructions",
    },
    WorkspaceFileDef {
        candidates: &["SOUL.md"],
        heading: "# Soul",
    },
    WorkspaceFileDef {
        candidates: &["IDENTITY.md"],
        heading: "# Identity",
    },
    WorkspaceFileDef {
        candidates: &["USER.md"],
        heading: "# User",
    },
    WorkspaceFileDef {
        candidates: &["TOOLS.md"],
        heading: "# Tools",
    },
    WorkspaceFileDef {
        candidates: &["BOOTSTRAP.md"],
        heading: "# Bootstrap",
    },
];

// ---------------------------------------------------------------------------
// File cache
// ---------------------------------------------------------------------------

struct CachedFile {
    content: String,
    mtime: SystemTime,
}

// ---------------------------------------------------------------------------
// Workspace
// ---------------------------------------------------------------------------

/// Reads workspace files (AGENTS.md, SOUL.md, IDENTITY.md, USER.md, TOOLS.md,
/// HEARTBEAT.md, BOOTSTRAP.md, MEMORY.md) and assembles them into the system
/// prompt on every turn.
///
/// Files are cached by mtime so edits take effect immediately on the next
/// message without restarting the agent.
pub struct Workspace {
    dir: PathBuf,
    digest_cfg: DigestConfig,
    cache: Mutex<std::collections::HashMap<PathBuf, CachedFile>>,
}

impl Workspace {
    pub fn new(dir: PathBuf, digest_cfg: DigestConfig) -> Self {
        info!("Workspace dir: {}", dir.display());
        Self {
            dir,
            digest_cfg,
            cache: Mutex::new(std::collections::HashMap::new()),
        }
    }

    /// Build the full system prompt:
    /// 1. Base system_prompt from config (if any)
    /// 2. Each workspace file that exists, in openclaw order
    /// 3. Chained MEMORY.md from the room's namespace and its includes
    /// 4. Previous day's daily log (if it exists)
    ///
    /// `namespace_chain` is the DFS-pre-order list of namespaces this
    /// room reads from — typically computed via
    /// `Config::resolve_namespace_chain(Config::namespace_for_room(room_id))`.
    /// The first entry is the room's own namespace; later entries are
    /// included parents.
    pub async fn build_system_prompt(
        &self,
        base: Option<&str>,
        boundary_hour: u8,
        namespace_chain: &[String],
    ) -> String {
        let mut parts: Vec<String> = Vec::new();

        if let Some(b) = base.filter(|s| !s.is_empty()) {
            parts.push(b.to_string());
        }

        // Inject current date/time so the agent has temporal awareness
        // (needed for tools like `memory` which write date-stamped files).
        let now_local = chrono::Local::now();
        parts.push(format!(
            "# Current Date and Time\n\n{} ({})",
            now_local.format("%Y-%m-%d %H:%M:%S %z"),
            now_local.format("%A")
        ));

        for def in WORKSPACE_FILES {
            if let Some((filename, content)) = self.read_first_existing(def.candidates).await {
                debug!("Injecting workspace file: {filename}");
                parts.push(format!("{}\n\n{content}", def.heading));
            }
        }

        // Per-namespace MEMORY.md, chained.
        if let Some(block) = self.build_memory_block(namespace_chain).await {
            parts.push(block);
        }

        // Inject periodic logs (yesterday's full body + digest blocks for
        // this week / month / year / past years).
        let today = crate::session::local_date_for_timestamp(now_local, boundary_hour);
        self.inject_periodic_logs(&mut parts, today, namespace_chain);

        parts.join("\n\n---\n\n")
    }

    /// Read MEMORY.md from each namespace in the chain (closest first) and
    /// concatenate as `## <namespace>` subsections under one combined
    /// `# Memory` heading. Namespaces with no MEMORY.md are skipped.
    /// Returns `None` if the entire chain has no MEMORY.md to inject.
    async fn build_memory_block(&self, namespace_chain: &[String]) -> Option<String> {
        let mut subsections = Vec::new();
        for ns in namespace_chain {
            let rel = format!("memory/{ns}/MEMORY.md");
            if let Some(content) = self.read_file(&rel).await
                && !content.trim().is_empty()
            {
                subsections.push(format!("## {ns}\n\n{content}"));
            }
        }
        if subsections.is_empty() {
            None
        } else {
            Some(format!("# Memory\n\n{}", subsections.join("\n\n")))
        }
    }

    /// Append log injection blocks to `parts`: yesterday's full body
    /// (room's own namespace only) plus top-N digest blocks aggregated
    /// across the namespace chain.
    fn inject_periodic_logs(
        &self,
        parts: &mut Vec<String>,
        today: chrono::NaiveDate,
        namespace_chain: &[String],
    ) {
        // Yesterday's full body — read only from the room's direct
        // namespace (the first chain entry). Reading from the chain would
        // balloon the body verbatim; parents' yesterday context is
        // conveyed through digest items below.
        if let Some(direct_ns) = namespace_chain.first()
            && let Some(yesterday) = today.pred_opt()
            && let Some(body) = periodic_log::read_body(
                &self.dir,
                direct_ns,
                LogKind::Daily,
                &periodic_log::daily_stem(yesterday),
            )
            && !body.trim().is_empty()
        {
            let truncated = truncate_chars(&body, MAX_FILE_CHARS);
            debug!("Injecting yesterday's daily log from '{direct_ns}': {yesterday}");
            parts.push(format!("# Yesterday's Log\n\n{truncated}"));
        }

        // "This Week's Digests" — daily files in `[iso_week_start, yesterday)`.
        if self.digest_cfg.daily_items > 0 {
            let stems = periodic_log::daily_stems_in_current_iso_week_before(today);
            if let Some(b) = build_chained_digest_block(
                "# This Week's Digests",
                &self.dir,
                namespace_chain,
                LogKind::Daily,
                &stems,
                self.digest_cfg.daily_items,
            ) {
                parts.push(b);
            }
        }

        // "This Month's Digests" — weekly files whose Monday is in this
        // calendar month, excluding the current ISO week.
        if self.digest_cfg.weekly_items > 0 {
            let stems = periodic_log::week_stems_in_month_before(today);
            if let Some(b) = build_chained_digest_block(
                "# This Month's Digests",
                &self.dir,
                namespace_chain,
                LogKind::Weekly,
                &stems,
                self.digest_cfg.weekly_items,
            ) {
                parts.push(b);
            }
        }

        // "This Year's Digests" — monthly files Jan..(current month - 1).
        if self.digest_cfg.monthly_items > 0 {
            let stems = periodic_log::month_stems_in_year_before(today);
            if let Some(b) = build_chained_digest_block(
                "# This Year's Digests",
                &self.dir,
                namespace_chain,
                LogKind::Monthly,
                &stems,
                self.digest_cfg.monthly_items,
            ) {
                parts.push(b);
            }
        }

        // "Past Years' Digests" — every yearly file on disk for any
        // namespace in the chain. We compute a per-namespace stem list
        // since each namespace can have its own yearly files.
        if self.digest_cfg.yearly_items > 0 {
            let mut subsections: Vec<String> = Vec::new();
            for ns in namespace_chain {
                let stems = periodic_log::existing_yearly_stems(&self.dir, ns);
                for stem in &stems {
                    if let Some(items) = periodic_log::read_digest_top_n(
                        &self.dir,
                        ns,
                        LogKind::Yearly,
                        stem,
                        self.digest_cfg.yearly_items,
                    ) {
                        if items.is_empty() {
                            continue;
                        }
                        let bullets: Vec<String> =
                            items.into_iter().map(|i| format!("- {i}")).collect();
                        subsections.push(format!("## {ns}/{stem}\n\n{}", bullets.join("\n")));
                    }
                }
            }
            if !subsections.is_empty() {
                parts.push(format!("# Past Years' Digests\n\n{}", subsections.join("\n\n")));
            }
        }
    }

    /// Try each candidate filename in order; return the first one found.
    async fn read_first_existing(&self, candidates: &[&str]) -> Option<(String, String)> {
        for &filename in candidates {
            if let Some(content) = self.read_file(filename).await {
                return Some((filename.to_string(), content));
            }
        }
        None
    }

    /// Read a workspace file with mtime caching. Returns `None` if not found.
    async fn read_file(&self, filename: &str) -> Option<String> {
        let path = self.dir.join(filename);

        match Self::file_mtime(&path) {
            None => {
                self.cache.lock().await.remove(&path);
                None
            }
            Some(mtime) => {
                {
                    let cache = self.cache.lock().await;
                    if let Some(entry) = cache.get(&path) {
                        if entry.mtime == mtime {
                            debug!("Workspace cache hit: {filename}");
                            return Some(entry.content.clone());
                        }
                    }
                }

                match std::fs::read_to_string(&path) {
                    Ok(raw) => {
                        let content = truncate_chars(&raw, MAX_FILE_CHARS);
                        info!(
                            "Loaded workspace file: {filename} ({} chars)",
                            content.len()
                        );
                        self.cache.lock().await.insert(
                            path,
                            CachedFile {
                                content: content.clone(),
                                mtime,
                            },
                        );
                        Some(content)
                    }
                    Err(e) => {
                        tracing::warn!("Failed to read {filename}: {e}");
                        None
                    }
                }
            }
        }
    }

    fn file_mtime(path: &Path) -> Option<SystemTime> {
        std::fs::metadata(path).ok()?.modified().ok()
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Assemble a heading + per-`(namespace, stem)` bulleted subsections of
/// top-N digest items, walking each stem against every namespace in the
/// chain. Each subsection is introduced by `## {namespace}/{stem}`.
/// `(namespace, stem)` pairs whose file is missing or has an empty digest
/// are skipped. Returns `None` if no pair produced a subsection.
fn build_chained_digest_block(
    heading: &str,
    workspace_dir: &Path,
    namespace_chain: &[String],
    kind: LogKind,
    stems: &[String],
    n: usize,
) -> Option<String> {
    let mut subsections = Vec::new();
    for ns in namespace_chain {
        for stem in stems {
            let items = periodic_log::read_digest_top_n(workspace_dir, ns, kind, stem, n)
                .unwrap_or_default();
            if items.is_empty() {
                continue;
            }
            let bullets: Vec<String> = items.into_iter().map(|i| format!("- {i}")).collect();
            subsections.push(format!("## {ns}/{stem}\n\n{}", bullets.join("\n")));
        }
    }
    if subsections.is_empty() {
        None
    } else {
        debug!("Injecting {heading} ({} subsection(s))", subsections.len());
        Some(format!("{heading}\n\n{}", subsections.join("\n\n")))
    }
}

/// Truncate `s` to at most `max_chars` Unicode scalar values.
fn truncate_chars(s: &str, max_chars: usize) -> String {
    let mut chars = s.chars();
    let truncated: String = (&mut chars).take(max_chars).collect();
    if chars.next().is_some() {
        format!("{truncated}\n\n[... truncated to {max_chars} characters ...]")
    } else {
        truncated
    }
}
