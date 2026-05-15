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
mod voice;
mod workspace;

use agent::Agent;
use anyhow::{Context, Result};
use channel::discord::DiscordChannel;
use channel::matrix::MatrixChannel;
use clap::{Parser, Subcommand};
use config::Config;
use heartbeat::Heartbeat;
use periodic_log::{
    build_all_today_digests, catchup_missing_daily_digests, catchup_pending_daily_logs,
    catchup_pending_monthly_logs, catchup_pending_weekly_logs, catchup_pending_yearly_logs,
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
        /// Room profile name to bind to a newly created session. Must match a
        /// `[room_profile.<name>]` entry on the server side. Ignored when
        /// resuming an existing session via --session.
        #[arg(long)]
        room_profile: Option<String>,
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
        room_profile,
    }) = cli.command
    {
        // The in-tree `sapphire-agent call` CLI has no per-device config
        // file, so no DeviceMetadata is forwarded; standalone callers
        // (sapphire-call) plumb their own `[device]` block through.
        return call::run(server, session, list, message, history, json, room_profile, None)
            .await;
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
            // ── Consolidate sessions/{matrix,discord} into sessions/channel ──
            // Both chat channels can run concurrently now and share one
            // session store keyed by ULID; the originating channel is
            // still recorded inside each session's metadata.
            let sessions_base_for_migration = config.resolved_sessions_dir(&workspace_dir);
            if let Err(e) = migrate_per_channel_sessions(&sessions_base_for_migration) {
                anyhow::bail!("Session layout migration failed: {e:#}");
            }
            // ── Move sessions/<kind>/* into sessions/<namespace>/<kind>/* ──
            // Reads each session's meta.namespace (falling back to "default")
            // and slots the file under the matching namespace directory so
            // sapphire-retrieve can scope indexing by directory rather than
            // by reading each file's frontmatter.
            if let Err(e) = migrate_sessions_to_namespaced_layout(&sessions_base_for_migration) {
                anyhow::bail!("Session namespace migration failed: {e:#}");
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

            // Standby mode runs a minimal periodic-sync loop. The
            // with-channels code path replaces this with a richer loop
            // below that also rebuilds today_digests on the same tick
            // (so we don't pay periodic_sync twice per interval).
            if config.standby_mode && let Some(dur) = ws_sync_interval {
                tracing::info!("Periodic workspace sync enabled: every {}s", dur.as_secs());
                let ws = Arc::clone(&ws_state);
                tokio::spawn(async move {
                    let mut tick = tokio::time::interval(dur);
                    tick.tick().await;
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

            // ── API session store (sessions/<namespace>/api/) ──────────────
            let api_session_store = Arc::new(SessionStore::with_workspace(
                sessions_base.clone(),
                "api",
                Arc::clone(&ws_state),
            ));

            // ── Voice providers + ServeState (built early) ──────────────────
            // ServeState owns the voice_subscribers registry that heartbeat
            // pushes through, so it has to exist before heartbeat is spawned.
            // The STT/TTS bundle download still happens once, on the
            // blocking pool, before any channel/heartbeat task starts —
            // a small startup-latency cost in exchange for a heartbeat
            // path that can target voice satellites.
            let voice_providers = if config.stt_providers.is_empty()
                && config.tts_providers.is_empty()
                || config.standby_mode
            {
                None
            } else {
                let cfg = config.clone();
                let providers =
                    tokio::task::spawn_blocking(move || voice::VoiceProviders::from_config(&cfg))
                        .await
                        .map_err(|e| anyhow::anyhow!("voice provider init panicked: {e}"))??;
                Some(Arc::new(providers))
            };
            let serve_state = Arc::new(serve::ServeState::new(
                config.clone(),
                Arc::clone(&registry),
                Arc::clone(&workspace),
                Arc::clone(&tool_set),
                Arc::clone(&api_session_store),
                voice_providers,
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

            // ── Channel + Agent (Matrix and/or Discord, if configured) ──────
            if !config.standby_mode && (config.matrix.is_some() || config.discord.is_some()) {
                // Sessions from every chat channel land under
                // `sessions/<namespace>/channel/<uuid>.jsonl`. Each session
                // still records its originating channel name in metadata.
                let channel_session_store = Arc::new(SessionStore::with_workspace(
                    sessions_base.clone(),
                    "channel",
                    Arc::clone(&ws_state),
                ));

                let mut channel_list: Vec<(String, Arc<dyn channel::Channel>)> = Vec::new();
                if let Some(m) = &config.matrix {
                    channel_list.push(("matrix".to_string(), Arc::new(MatrixChannel::new(m))));
                }
                if let Some(d) = &config.discord {
                    channel_list.push((
                        "discord".to_string(),
                        Arc::new(
                            DiscordChannel::new(d)
                                .context("Failed to initialise Discord channel")?,
                        ),
                    ));
                }
                let channels = Arc::new(channel::Channels::new(
                    channel_list,
                    channel::seed_routing_from_config(&config),
                ));
                tracing::info!(
                    "Channels active: {}",
                    channels.names().join(", ")
                );

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
                    Arc::clone(&channels),
                    Arc::clone(&registry),
                    Arc::clone(&workspace),
                    Some(Arc::clone(&tool_set)),
                    Arc::clone(&channel_session_store),
                ));
                agent.bootstrap().await;

                // ── Periodic workspace sync + today-digest rebuild ──────
                // Same cadence drives both: when `periodic_sync` pulls
                // session JSONLs from another device via git, the digest
                // builder picks them up on the same tick so cross-device
                // "today's notes" become visible without waiting for the
                // next day-boundary daily-log generation.
                if let Some(dur) = ws_sync_interval {
                    tracing::info!(
                        "Periodic workspace sync enabled: every {}s",
                        dur.as_secs()
                    );
                    let ws = Arc::clone(&ws_state);
                    let cfg_for_loop = config.clone();
                    let workspace_for_loop = Arc::clone(&workspace);
                    let workspace_dir_for_loop = workspace_dir.clone();
                    let channel_store_for_loop = Arc::clone(&channel_session_store);
                    let api_store_for_loop = Arc::clone(&api_session_store);
                    let agent_for_loop = Arc::clone(&agent);
                    tokio::spawn(async move {
                        let mut tick = tokio::time::interval(dur);
                        tick.tick().await; // skip immediate fire
                        loop {
                            tick.tick().await;
                            {
                                let state = ws.lock().expect("ws_state mutex poisoned");
                                match state.periodic_sync() {
                                    Ok((u, r)) => tracing::info!(
                                        "Periodic ws sync: {u} upserted, {r} removed"
                                    ),
                                    Err(e) => tracing::warn!("Periodic ws sync failed: {e:#}"),
                                }
                            }
                            rebuild_today_digests(
                                &cfg_for_loop,
                                &workspace_for_loop,
                                &workspace_dir_for_loop,
                                &channel_store_for_loop,
                                &api_store_for_loop,
                                &agent_for_loop,
                            )
                            .await;
                        }
                    });
                } else {
                    // Even without periodic sync, populate today's
                    // cache once at startup so the first turn after
                    // restart sees prior intra-day flushes.
                    rebuild_today_digests(
                        &config,
                        &workspace,
                        &workspace_dir,
                        &channel_session_store,
                        &api_session_store,
                        &agent,
                    )
                    .await;
                }

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
                    serve_state: Some(Arc::clone(&serve_state)),
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

                serve::run(addr, Arc::clone(&serve_state)).await?;
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

/// Rebuild Workspace's per-namespace "today's digest" cache from the
/// session JSONLs on disk and notify the agent that its cached system
/// prompts are now stale. Invoked from the periodic-sync loop so a git
/// pull on one machine becomes visible on the other within one tick.
///
/// Cheap when there are no fresh digests: each store walks `sessions/*`
/// once with an mtime pre-filter that rejects files untouched before
/// today's local-day window.
async fn rebuild_today_digests(
    config: &Config,
    workspace: &Arc<Workspace>,
    workspace_dir: &std::path::Path,
    channel_store: &Arc<SessionStore>,
    api_store: &Arc<SessionStore>,
    agent: &Arc<Agent>,
) {
    let today = session::local_date_for_timestamp(
        chrono::Local::now(),
        config.day_boundary_hour,
    );
    let namespaces = config.all_memory_namespaces();
    let cfg = config.clone();
    let map = build_all_today_digests(
        &namespaces,
        today,
        config.day_boundary_hour,
        channel_store,
        Some(api_store.as_ref()),
        |room_id: &str| cfg.namespace_for_room(room_id).to_string(),
    );
    let had_content = !map.is_empty();
    workspace.replace_today_digests(map).await;
    let _ = workspace_dir; // currently unused but kept for symmetry
    if had_content {
        agent.invalidate_system_prompts().await;
    }
}

/// Migrate flat `sessions/<kind>/<uuid>.jsonl` layouts to the namespaced
/// `sessions/<namespace>/<kind>/<uuid>.jsonl` form. Each session's
/// namespace is read from `meta.namespace` (newer files), falling back
/// to `"default"` for legacy files that predate that field.
///
/// Idempotent: skips silently when no flat `sessions/<kind>/*.jsonl`
/// files exist. Refuses to run when a target path is already occupied
/// (extremely unlikely with ULID/UUID file names but handled for
/// safety).
fn migrate_sessions_to_namespaced_layout(
    sessions_base: &std::path::Path,
) -> anyhow::Result<()> {
    use std::io::{BufRead, BufReader};

    for kind in ["channel", "api"] {
        let kind_dir = sessions_base.join(kind);
        if !kind_dir.is_dir() {
            continue;
        }
        let entries = match std::fs::read_dir(&kind_dir) {
            Ok(e) => e,
            Err(e) => {
                tracing::warn!("Skipping {}: {e}", kind_dir.display());
                continue;
            }
        };
        let mut moved = 0;
        for entry in entries.flatten() {
            let path = entry.path();
            if !path.is_file() {
                continue;
            }
            if path.extension().and_then(|e| e.to_str()) != Some("jsonl") {
                continue;
            }
            // Read just the first line to extract meta.namespace.
            let namespace = match std::fs::File::open(&path) {
                Ok(f) => {
                    let mut first = String::new();
                    let _ = BufReader::new(f).read_line(&mut first);
                    serde_json::from_str::<serde_json::Value>(first.trim())
                        .ok()
                        .and_then(|v| {
                            v.get("meta")
                                .and_then(|m| m.get("namespace"))
                                .and_then(|n| n.as_str())
                                .map(str::to_string)
                        })
                        .unwrap_or_else(|| config::DEFAULT_NAMESPACE_NAME.to_string())
                }
                Err(_) => config::DEFAULT_NAMESPACE_NAME.to_string(),
            };

            let Some(file_name) = path.file_name() else {
                continue;
            };
            let dst_dir = sessions_base.join(&namespace).join(kind);
            std::fs::create_dir_all(&dst_dir)
                .with_context(|| format!("create {}", dst_dir.display()))?;
            let dst = dst_dir.join(file_name);
            if dst.exists() {
                anyhow::bail!(
                    "Refusing to migrate {}: destination {} already exists. \
                     Reconcile manually before starting.",
                    path.display(),
                    dst.display(),
                );
            }
            std::fs::rename(&path, &dst).with_context(|| {
                format!("rename {} -> {}", path.display(), dst.display())
            })?;
            moved += 1;
        }
        // Remove the now-empty flat directory; non-fatal if it still has
        // unexpected leftovers (e.g. user dropped files).
        let _ = std::fs::remove_dir(&kind_dir);
        if moved > 0 {
            tracing::info!(
                "Migrated {} session file(s) from {} into namespaced layout",
                moved,
                kind_dir.display()
            );
        }
    }
    Ok(())
}

/// One-shot migration from the per-channel session layout
/// (`sessions/matrix/` + `sessions/discord/`) to the consolidated
/// `sessions/channel/` directory. The originating channel is still
/// recorded in each session's `meta.channel` field, so the move is
/// pure reshuffling.
///
/// Idempotent: skips silently when neither legacy directory exists.
/// Refuses to run when a name collision would clobber an existing
/// `sessions/channel/<id>.jsonl` (extremely unlikely with ULIDs but
/// handled for safety).
fn migrate_per_channel_sessions(sessions_base: &std::path::Path) -> anyhow::Result<()> {
    let target = sessions_base.join("channel");
    let mut sources: Vec<std::path::PathBuf> = Vec::new();
    for legacy in ["matrix", "discord"] {
        let dir = sessions_base.join(legacy);
        if dir.is_dir() {
            sources.push(dir);
        }
    }
    if sources.is_empty() {
        return Ok(());
    }

    std::fs::create_dir_all(&target)
        .with_context(|| format!("create {}", target.display()))?;

    for src in sources {
        let entries = match std::fs::read_dir(&src) {
            Ok(e) => e,
            Err(e) => {
                tracing::warn!("Skipping {}: {e}", src.display());
                continue;
            }
        };
        let mut moved = 0;
        for entry in entries.flatten() {
            let path = entry.path();
            if !path.is_file() {
                continue;
            }
            let Some(file_name) = path.file_name() else {
                continue;
            };
            let dst = target.join(file_name);
            if dst.exists() {
                anyhow::bail!(
                    "Refusing to migrate {}: destination {} already exists. \
                     Reconcile manually before starting.",
                    path.display(),
                    dst.display(),
                );
            }
            std::fs::rename(&path, &dst).with_context(|| {
                format!("rename {} -> {}", path.display(), dst.display())
            })?;
            moved += 1;
        }
        // Remove the now-empty legacy directory; non-fatal if it has
        // unexpected leftover entries (e.g. user-dropped files).
        let _ = std::fs::remove_dir(&src);
        tracing::info!(
            "Migrated {} session file(s) from {} to {}",
            moved,
            src.display(),
            target.display()
        );
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

    #[test]
    fn session_migration_no_op_on_fresh_workspace() {
        let td = tempfile::tempdir().unwrap();
        migrate_per_channel_sessions(td.path()).expect("clean migration");
        assert!(!td.path().join("channel").exists());
    }

    #[test]
    fn session_migration_consolidates_matrix_and_discord() {
        let td = tempfile::tempdir().unwrap();
        let base = td.path();
        write_stub(&base.join("matrix/01H1.jsonl"), "matrix");
        write_stub(&base.join("matrix/01H2.jsonl"), "matrix2");
        write_stub(&base.join("discord/01H3.jsonl"), "discord");

        migrate_per_channel_sessions(base).expect("migration");

        for stem in ["01H1", "01H2", "01H3"] {
            let path = base.join("channel").join(format!("{stem}.jsonl"));
            assert!(path.exists(), "missing: {}", path.display());
        }
        assert!(!base.join("matrix").exists(), "matrix dir should be removed");
        assert!(!base.join("discord").exists(), "discord dir should be removed");
    }

    #[test]
    fn session_migration_is_idempotent() {
        let td = tempfile::tempdir().unwrap();
        let base = td.path();
        write_stub(&base.join("matrix/01H1.jsonl"), "matrix");
        migrate_per_channel_sessions(base).expect("first");
        migrate_per_channel_sessions(base).expect("second is a no-op");
        assert!(base.join("channel/01H1.jsonl").exists());
    }

    #[test]
    fn session_migration_refuses_on_collision() {
        let td = tempfile::tempdir().unwrap();
        let base = td.path();
        write_stub(&base.join("matrix/01H1.jsonl"), "matrix");
        write_stub(&base.join("channel/01H1.jsonl"), "pre-existing");
        let err = migrate_per_channel_sessions(base).err().expect("error");
        assert!(
            format!("{err:#}").contains("already exists"),
            "got: {err:#}"
        );
        // Original files untouched.
        assert_eq!(
            std::fs::read_to_string(base.join("matrix/01H1.jsonl")).unwrap(),
            "matrix"
        );
        assert_eq!(
            std::fs::read_to_string(base.join("channel/01H1.jsonl")).unwrap(),
            "pre-existing"
        );
    }
}
