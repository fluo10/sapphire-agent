mod agent;
mod call;
mod channel;
mod config;
mod context_compression;
mod frontmatter;
mod heartbeat;
mod heartbeat_config;
mod mcp_client;
mod memory_compaction;
mod periodic_log;
mod provider;
mod serve;
mod session;
mod tools;
mod workspace;

use agent::Agent;
use anyhow::{Context, Result};
use channel::discord::DiscordChannel;
use channel::matrix::MatrixChannel;
use clap::{Parser, Subcommand};
use config::Config;
use heartbeat::Heartbeat;
use periodic_log::{
    catchup_missing_daily_digests, catchup_pending_daily_logs, catchup_pending_monthly_logs,
    catchup_pending_weekly_logs, catchup_pending_yearly_logs,
};
use provider::registry::ProviderRegistry;
use sapphire_workspace::{AppContext, DeviceDefaults, Workspace as SwWorkspace, WorkspaceState};

static APP_CTX: AppContext = AppContext::new("sapphire-agent").allow_external_paths();

/// Inject host-platform paths and device facts into [`APP_CTX`] before any
/// code touches a [`SwWorkspace`]. The sapphire-workspace library deliberately
/// does not depend on `dirs` / `hostname`, so each host app has to wire these
/// up itself at startup. Missing this made `APP_CTX.device()` panic the first
/// time the git sync backend tried to record device info.
fn init_app_ctx() {
    let base = directories::BaseDirs::new();
    let cache_dir = base
        .as_ref()
        .map(|b| b.cache_dir().to_path_buf())
        .unwrap_or_else(std::env::temp_dir)
        .join(env!("CARGO_PKG_NAME"));
    let data_dir = base
        .as_ref()
        .map(|b| b.data_dir().to_path_buf())
        .unwrap_or_else(std::env::temp_dir)
        .join(env!("CARGO_PKG_NAME"));
    APP_CTX.set_cache_dir(cache_dir);
    APP_CTX.set_data_dir(data_dir);

    let hostname = hostname::get()
        .ok()
        .and_then(|s| s.into_string().ok())
        .unwrap_or_default();
    APP_CTX.set_device_defaults(DeviceDefaults {
        hostname,
        app_id: env!("CARGO_PKG_NAME").to_owned(),
        app_version: env!("CARGO_PKG_VERSION").to_owned(),
        platform: std::env::consts::OS.to_owned(),
        arch: std::env::consts::ARCH.to_owned(),
    });
}
use session::SessionStore;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use tracing_subscriber::{EnvFilter, fmt};
use workspace::Workspace;

#[derive(Parser)]
#[command(
    name = "sapphire-agent",
    about = "Personal AI assistant — Anthropic + Matrix/Discord"
)]
struct Cli {
    /// Path to config file (default: ~/.config/sapphire-agent/config.toml)
    #[arg(short, long, value_name = "FILE")]
    config: Option<PathBuf>,

    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand)]
enum Command {
    /// Start the agent — Matrix/Discord channels + HTTP API server (default)
    Serve {
        /// Override bind address (e.g. 127.0.0.1:9000)
        #[arg(long, value_name = "ADDR")]
        bind: Option<String>,
    },
    /// Validate the config file and exit
    Verify,
    /// Interactive session with a running serve server
    Call {
        /// Server base URL
        #[arg(long, default_value = "http://localhost:9000")]
        server: String,
        /// Grain-id of an existing session to resume (e.g. a3b7k9p)
        #[arg(long)]
        session: Option<String>,
        /// List available API sessions and exit
        #[arg(long)]
        list: bool,
        /// Send a single message and exit instead of entering the REPL.
        /// Useful as a CJK-safe fallback or for IDE/editor integration.
        #[arg(short, long, value_name = "TEXT")]
        message: Option<String>,
        /// Dump the session history and exit (no message sent, no REPL).
        /// Intended for IDE integrations restoring a session.
        #[arg(long)]
        history: bool,
        /// Emit machine-readable JSON output. Applies to --list, --history,
        /// and --message; ignored in REPL mode.
        #[arg(long)]
        json: bool,
        /// Profile name to bind to a newly created session. Must match a
        /// `[profiles.<name>]` entry on the server side. Ignored when
        /// resuming an existing session via --session.
        #[arg(long)]
        profile: Option<String>,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    init_app_ctx();

    let cli = Cli::parse();

    // `call` needs no config file — handle before loading config
    if let Some(Command::Call {
        server,
        session,
        list,
        message,
        history,
        json,
        profile,
    }) = cli.command
    {
        return call::run(server, session, list, message, history, json, profile).await;
    }

    let config_path = cli.config.unwrap_or_else(Config::default_path);
    let config = Config::load(&config_path)
        .with_context(|| format!("Failed to load config from {}", config_path.display()))?;

    match cli.command.unwrap_or(Command::Serve { bind: None }) {
        Command::Verify => {
            let workspace_dir = config.resolved_workspace_dir(&config_path);
            println!("Config OK");
            if let Some(m) = &config.matrix {
                println!("  Channel           : matrix");
                println!("  Matrix homeserver : {}", m.homeserver);
                println!("  Matrix user_id    : {}", m.user_id);
                println!("  Matrix rooms      : {:?}", m.room_ids);
            } else if let Some(d) = &config.discord {
                println!("  Channel           : discord");
                println!("  Discord channels  : {:?}", d.channel_ids);
                println!("  Allowed users     : {:?}", d.allowed_users);
            } else {
                println!("  Channel           : NONE (add [discord] or [matrix] to config)");
            }
            println!("  Anthropic model   : {}", config.anthropic.model);
            println!("  Anthropic max_tok : {}", config.anthropic.max_tokens);
            println!("  Workspace dir     : {}", workspace_dir.display());
            println!(
                "  Day boundary hour : {}:00 local",
                config.day_boundary_hour
            );
            println!("  Heartbeat enabled : {}", config.heartbeat_enabled);
            println!("  Standby mode      : {}", config.standby_mode);
            println!();
            let workspace_files = [
                ("AGENTS.md / AGENT.md", vec!["AGENTS.md", "AGENT.md"]),
                ("SOUL.md", vec!["SOUL.md"]),
                ("IDENTITY.md", vec!["IDENTITY.md"]),
                ("USER.md", vec!["USER.md"]),
                ("TOOLS.md", vec!["TOOLS.md"]),
                ("BOOTSTRAP.md", vec!["BOOTSTRAP.md"]),
                ("MEMORY.md", vec!["MEMORY.md", "memory.md"]),
            ];
            for (label, candidates) in &workspace_files {
                let found = candidates.iter().find_map(|f| {
                    let p = workspace_dir.join(f);
                    if p.exists() { Some(*f) } else { None }
                });
                match found {
                    Some(f) => println!("  {label:<28} found ({f})"),
                    None => println!("  {label:<28} -"),
                }
            }
        }
        Command::Serve { bind } => {
            let workspace_dir = config.resolved_workspace_dir(&config_path);

            // ── Migrate pre-namespace memory layout (one-time) ─────────────
            if let Err(e) = migrate_pre_namespace_layout(&workspace_dir) {
                anyhow::bail!("Memory layout migration failed: {e:#}");
            }

            // ── Bootstrap file loader (AGENTS.md, SOUL.md, MEMORY.md …) ────
            let workspace = Arc::new(Workspace::new(workspace_dir.clone(), config.digest.clone()));

            // ── sapphire-workspace (search, file ops, git sync) ─────────────
            let sw_workspace = SwWorkspace::resolve(&APP_CTX, Some(&workspace_dir))
                .context("Failed to resolve sapphire-workspace")?;
            // Use the [sync] section from the agent config directly.
            // WorkspaceConfig was removed in sapphire-workspace 0.8.0;
            // open_configured now takes &SyncConfig. In 0.10 the periodic
            // cadence moved out of SyncConfig because it drives both
            // sapphire-sync and sapphire-retrieve — keeping one knob
            // avoids a duplicate `[retrieve]` cadence. It now lives at
            // the agent config root as `sync_interval_minutes`, and each
            // `periodic_sync()` call refreshes the retrieve cache too.
            let sync_config = config.sync.clone().unwrap_or_default();
            let ws_sync_interval = config
                .sync_interval_minutes
                .filter(|&m| m > 0)
                .map(|m| std::time::Duration::from_secs(m as u64 * 60));
            let ws_state = WorkspaceState::open_configured(sw_workspace, &sync_config)
                .context("Failed to open WorkspaceState")?;
            if let Err(e) = ws_state.periodic_sync() {
                tracing::warn!("Initial workspace sync failed: {e}");
            }
            let ws_state = Arc::new(Mutex::new(ws_state));

            // ── Periodic workspace sync (if enabled in workspace config) ────
            if let Some(dur) = ws_sync_interval {
                tracing::info!("Periodic workspace sync enabled: every {}s", dur.as_secs());
                let ws = Arc::clone(&ws_state);
                tokio::spawn(async move {
                    let mut tick = tokio::time::interval(dur);
                    tick.tick().await; // skip immediate fire
                    loop {
                        tick.tick().await;
                        let state = ws.lock().expect("ws_state mutex poisoned");
                        match state.periodic_sync() {
                            Ok((u, r)) => {
                                tracing::info!("Periodic ws sync: {u} upserted, {r} removed");
                            }
                            Err(e) => tracing::warn!("Periodic ws sync failed: {e:#}"),
                        }
                    }
                });
            }

            // ── Tools ───────────────────────────────────────────────────────
            let tool_set = tools::default_tool_set(
                Arc::clone(&ws_state),
                config.tools.tavily_api_key.clone(),
                &config.tools.mcp_servers,
            )
            .await;

            // ── Session store base directory ────────────────────────────────
            let sessions_base = config.resolved_sessions_dir(&workspace_dir);

            // ── Provider registry ────────────────────────────────────────────
            // Builds the Anthropic provider plus any `[providers.<name>]`
            // entries, then validates every profile/room reference. Failure
            // here is fatal — better to refuse to start than to surprise the
            // user mid-session with a misrouted request.
            let registry = Arc::new(
                ProviderRegistry::from_config(&config)
                    .context("Failed to build provider registry")?,
            );

            // ── API session store (sessions/api/) ───────────────────────────
            let api_session_store = Arc::new(SessionStore::with_workspace(
                sessions_base.join("api"),
                Arc::clone(&ws_state),
            ));

            if config.standby_mode {
                tracing::info!(
                    "Standby mode enabled: git sync only, skipping channel and heartbeat"
                );
            }

            // Captured below so main can await the agent's graceful shutdown
            // (summarize_on_shutdown) before returning. Without this, the
            // tokio runtime drops the spawned task the moment serve::run
            // returns, cancelling any in-flight LLM call (#48).
            let mut agent_handle: Option<tokio::task::JoinHandle<()>> = None;

            // ── Channel + Agent (Matrix or Discord, if configured) ──────────
            if !config.standby_mode && (config.matrix.is_some() || config.discord.is_some()) {
                let channel_name = if config.discord.is_some() {
                    "discord"
                } else {
                    "matrix"
                };
                let channel_session_store = Arc::new(SessionStore::with_workspace(
                    sessions_base.join(channel_name),
                    Arc::clone(&ws_state),
                ));

                let channel: Arc<dyn channel::Channel> = if let Some(d) = &config.discord {
                    Arc::new(
                        DiscordChannel::new(d).context("Failed to initialise Discord channel")?,
                    )
                } else if let Some(m) = &config.matrix {
                    Arc::new(MatrixChannel::new(m))
                } else {
                    unreachable!()
                };

                // ── Catch-up: iterate every configured memory namespace ────
                // Each namespace's background tasks run on its own provider
                // (resolved via `background_provider_for_namespace`), so an
                // NSFW namespace can route to its local model up front
                // instead of bouncing through Anthropic refusal-fallback.
                let today_local = session::local_date_for_timestamp(
                    chrono::Local::now(),
                    config.day_boundary_hour,
                );
                for ns in config.all_memory_namespaces() {
                    let provider = registry.background_provider_for_namespace(&config, &ns);
                    let cfg_for_predicate = config.clone();
                    let ns_for_predicate = ns.clone();
                    let predicate = move |meta: &session::SessionMeta| -> bool {
                        if meta.channel == "api" {
                            return ns_for_predicate == config::DEFAULT_NAMESPACE_NAME;
                        }
                        cfg_for_predicate.namespace_for_room(&meta.room_id) == ns_for_predicate
                    };
                    catchup_pending_daily_logs(
                        &channel_session_store,
                        provider.as_ref(),
                        &ws_state,
                        &workspace_dir,
                        &ns,
                        config.day_boundary_hour,
                        &predicate,
                    )
                    .await;
                    catchup_missing_daily_digests(
                        provider.as_ref(),
                        &ws_state,
                        &workspace_dir,
                        &ns,
                    )
                    .await;
                    if config.digest.weekly_enabled {
                        catchup_pending_weekly_logs(
                            provider.as_ref(),
                            &ws_state,
                            &workspace_dir,
                            &ns,
                            today_local,
                        )
                        .await;
                    }
                    if config.digest.monthly_enabled {
                        catchup_pending_monthly_logs(
                            provider.as_ref(),
                            &ws_state,
                            &workspace_dir,
                            &ns,
                            today_local,
                        )
                        .await;
                    }
                    if config.digest.yearly_enabled {
                        catchup_pending_yearly_logs(
                            provider.as_ref(),
                            &ws_state,
                            &workspace_dir,
                            &ns,
                            today_local,
                        )
                        .await;
                    }
                }

                // ── Agent ───────────────────────────────────────────────────
                let agent = Arc::new(Agent::new(
                    config.clone(),
                    channel,
                    Arc::clone(&registry),
                    Arc::clone(&workspace),
                    Some(Arc::clone(&tool_set)),
                    Arc::clone(&channel_session_store),
                ));
                agent.bootstrap().await;

                // ── Heartbeat (day-boundary + cron loops) ───────────────────
                let default_room_id = config
                    .matrix
                    .as_ref()
                    .and_then(|m| m.primary_room_id().map(str::to_string))
                    .or_else(|| {
                        config
                            .discord
                            .as_ref()
                            .and_then(|d| d.channel_ids.first().cloned())
                    });
                let heartbeat = Heartbeat {
                    workspace_dir: workspace_dir.clone(),
                    ws_state: Arc::clone(&ws_state),
                    day_boundary_hour: config.day_boundary_hour,
                    daily_log_enabled: config.daily_log_enabled,
                    memory_compaction_enabled: config.memory_compaction_enabled,
                    digest_cfg: config.digest.clone(),
                    session_store: Arc::clone(&channel_session_store),
                    registry: Arc::clone(&registry),
                    agent: Arc::clone(&agent),
                    default_room_id,
                    config: config.clone(),
                };
                if config.heartbeat_enabled {
                    heartbeat.spawn();
                } else {
                    tracing::info!("Heartbeat disabled by config");
                }

                let agent_run = Arc::clone(&agent);
                agent_handle = Some(tokio::spawn(async move {
                    if let Err(e) = agent_run.run().await {
                        tracing::error!("Agent error: {e:#}");
                    }
                }));
            }

            if config.standby_mode {
                // In standby mode, keep the process alive for periodic git
                // sync only — no HTTP server, no channel, no heartbeat.
                tracing::info!("Standby mode: waiting for shutdown signal (Ctrl-C)");
                tokio::signal::ctrl_c()
                    .await
                    .expect("Failed to listen for Ctrl-C");
                tracing::info!("Shutting down standby process");
            } else {
                // ── HTTP API server ─────────────────────────────────────────
                let addr = bind
                    .or_else(|| {
                        config
                            .serve
                            .as_ref()
                            .map(|s| format!("{}:{}", s.host, s.port))
                    })
                    .unwrap_or_else(|| "127.0.0.1:9000".to_string());

                serve::run(
                    addr,
                    config,
                    Arc::clone(&registry),
                    workspace,
                    tool_set,
                    api_session_store,
                )
                .await?;
            }

            // Wait for the agent task's graceful shutdown to finish so its
            // summarize_on_shutdown LLM call isn't aborted by runtime drop.
            if let Some(handle) = agent_handle {
                if let Err(e) = handle.await {
                    tracing::warn!("Agent task did not finish cleanly: {e}");
                }
            }
        }
        Command::Call { .. } => unreachable!(),
    }

    Ok(())
}

/// One-shot migration from the pre-namespace `memory/{daily,weekly,monthly,
/// yearly}/` layout to the namespaced `memory/default/{...}` layout. Also
/// moves a top-level MEMORY.md (or lowercase memory.md) into
/// `memory/default/MEMORY.md`.
///
/// Idempotent: skips silently when `memory/default/` already exists with
/// no top-level periodic dirs to move. Refuses to run when both old and
/// new layouts coexist (ambiguous merge — the user must reconcile).
fn migrate_pre_namespace_layout(workspace_dir: &std::path::Path) -> anyhow::Result<()> {
    let memory_root = workspace_dir.join("memory");
    let has_top_memory_md = ["MEMORY.md", "memory.md"]
        .iter()
        .any(|f| workspace_dir.join(f).is_file());
    if !memory_root.is_dir() && !has_top_memory_md {
        return Ok(());
    }

    let default_root = memory_root.join(config::DEFAULT_NAMESPACE_NAME);
    let kinds = ["daily", "weekly", "monthly", "yearly"];
    let pre_dirs: Vec<_> = kinds
        .iter()
        .filter(|k| memory_root.join(k).is_dir())
        .copied()
        .collect();
    let pre_memory_md = ["MEMORY.md", "memory.md"]
        .iter()
        .map(|f| workspace_dir.join(f))
        .find(|p| p.is_file());

    if pre_dirs.is_empty() && pre_memory_md.is_none() {
        return Ok(());
    }

    if default_root.is_dir() && !pre_dirs.is_empty() {
        anyhow::bail!(
            "Both pre-namespace memory dirs ({}) and memory/{} exist — refuse to migrate. \
             Reconcile manually before starting.",
            pre_dirs.join(", "),
            config::DEFAULT_NAMESPACE_NAME,
        );
    }

    std::fs::create_dir_all(&default_root)
        .with_context(|| format!("create {}", default_root.display()))?;

    for kind in &pre_dirs {
        let src = memory_root.join(kind);
        let dst = default_root.join(kind);
        std::fs::rename(&src, &dst).with_context(|| {
            format!("rename {} -> {}", src.display(), dst.display())
        })?;
        tracing::info!(
            "Migrated memory subdir: {} -> {}",
            src.display(),
            dst.display()
        );
    }

    if let Some(src) = pre_memory_md {
        let dst = default_root.join("MEMORY.md");
        std::fs::rename(&src, &dst)
            .with_context(|| format!("rename {} -> {}", src.display(), dst.display()))?;
        tracing::info!(
            "Migrated MEMORY.md: {} -> {}",
            src.display(),
            dst.display()
        );
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_stub(path: &std::path::Path, body: &str) {
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, body).unwrap();
    }

    #[test]
    fn migration_no_op_on_fresh_workspace() {
        let td = tempfile::tempdir().unwrap();
        migrate_pre_namespace_layout(td.path()).expect("clean migration");
        // Nothing created.
        assert!(!td.path().join("memory").exists());
    }

    #[test]
    fn migration_moves_pre_namespace_layout() {
        let td = tempfile::tempdir().unwrap();
        let root = td.path();
        write_stub(&root.join("memory/daily/2026-04-15.md"), "daily body");
        write_stub(&root.join("memory/weekly/2026-W16.md"), "weekly body");
        write_stub(&root.join("memory/monthly/2026-04.md"), "monthly body");
        write_stub(&root.join("memory/yearly/2026.md"), "yearly body");
        write_stub(&root.join("MEMORY.md"), "core memory");

        migrate_pre_namespace_layout(root).expect("migration");

        assert_eq!(
            std::fs::read_to_string(root.join("memory/default/daily/2026-04-15.md")).unwrap(),
            "daily body"
        );
        assert_eq!(
            std::fs::read_to_string(root.join("memory/default/weekly/2026-W16.md")).unwrap(),
            "weekly body"
        );
        assert_eq!(
            std::fs::read_to_string(root.join("memory/default/monthly/2026-04.md")).unwrap(),
            "monthly body"
        );
        assert_eq!(
            std::fs::read_to_string(root.join("memory/default/yearly/2026.md")).unwrap(),
            "yearly body"
        );
        assert_eq!(
            std::fs::read_to_string(root.join("memory/default/MEMORY.md")).unwrap(),
            "core memory"
        );
        // Old paths gone.
        assert!(!root.join("memory/daily").exists());
        assert!(!root.join("MEMORY.md").exists());
    }

    #[test]
    fn migration_accepts_lowercase_memory_md() {
        let td = tempfile::tempdir().unwrap();
        let root = td.path();
        write_stub(&root.join("memory.md"), "lowercase memory");

        migrate_pre_namespace_layout(root).expect("migration");

        assert_eq!(
            std::fs::read_to_string(root.join("memory/default/MEMORY.md")).unwrap(),
            "lowercase memory"
        );
    }

    #[test]
    fn migration_is_idempotent() {
        let td = tempfile::tempdir().unwrap();
        let root = td.path();
        write_stub(&root.join("memory/daily/2026-04-15.md"), "daily body");
        migrate_pre_namespace_layout(root).expect("first migration");
        // Run again — pre-namespace dirs no longer exist; default tree
        // already in place.
        migrate_pre_namespace_layout(root).expect("idempotent migration");
        assert_eq!(
            std::fs::read_to_string(root.join("memory/default/daily/2026-04-15.md")).unwrap(),
            "daily body"
        );
    }

    #[test]
    fn migration_refuses_when_layouts_coexist() {
        let td = tempfile::tempdir().unwrap();
        let root = td.path();
        // Both old and new layouts present — ambiguous.
        write_stub(&root.join("memory/daily/2026-04-15.md"), "old");
        write_stub(&root.join("memory/default/daily/2026-04-16.md"), "new");

        let err = migrate_pre_namespace_layout(root).err().expect("error");
        assert!(format!("{err:#}").contains("Reconcile manually"));
        // Files untouched.
        assert!(root.join("memory/daily/2026-04-15.md").exists());
        assert!(root.join("memory/default/daily/2026-04-16.md").exists());
    }
}
