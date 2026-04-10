mod agent;
mod call;
mod channel;
mod config;
mod daily_log;
mod heartbeat;
mod heartbeat_config;
mod mcp_client;
mod memory_compaction;
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
use daily_log::catchup_pending_logs;
use heartbeat::Heartbeat;
use provider::anthropic::AnthropicProvider;
use sapphire_workspace::{AppContext, Workspace as SwWorkspace, WorkspaceConfig, WorkspaceState};

static APP_CTX: AppContext = AppContext::new("sapphire-agent");
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
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    let cli = Cli::parse();

    // `call` needs no config file — handle before loading config
    if let Some(Command::Call {
        server,
        session,
        list,
        message,
        history,
        json,
    }) = cli.command
    {
        return call::run(server, session, list, message, history, json).await;
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
                println!("  Matrix room_id    : {}", m.room_id);
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

            // ── Bootstrap file loader (AGENTS.md, SOUL.md, MEMORY.md …) ────
            let workspace = Arc::new(Workspace::new(workspace_dir.clone()));

            // ── sapphire-workspace (search, file ops, git sync) ─────────────
            let sw_workspace = SwWorkspace::resolve(&APP_CTX, Some(&workspace_dir))
                .context("Failed to resolve sapphire-workspace")?;
            // Load the workspace config so we can read sync_interval_minutes.
            // Workspace config provides shared defaults; the per-user agent
            // config [sync] section takes precedence when present.
            let mut ws_config =
                WorkspaceConfig::load_from(&sw_workspace.config_path()).unwrap_or_default();
            if let Some(agent_sync) = &config.sync {
                ws_config.sync = agent_sync.clone();
            }
            let ws_sync_interval = ws_config.sync.sync_interval();
            let ws_state =
                WorkspaceState::open(sw_workspace).context("Failed to open WorkspaceState")?;
            if let Err(e) = ws_state.sync() {
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
                        match state.sync() {
                            Ok((u, r)) => {
                                tracing::info!("Periodic ws sync: {u} upserted, {r} removed");
                                if let Some(backend) = state.sync_backend() {
                                    if let Err(e) = backend.sync() {
                                        tracing::warn!("Periodic ws git sync failed: {e:#}");
                                    }
                                }
                            }
                            Err(e) => tracing::warn!("Periodic ws index sync failed: {e:#}"),
                        }
                    }
                });
            }

            // ── Tools ───────────────────────────────────────────────────────
            let tool_set = Arc::new(
                tools::default_tool_set(
                    Arc::clone(&ws_state),
                    config.tools.tavily_api_key.clone(),
                    &config.tools.mcp_servers,
                )
                .await,
            );

            // ── Session store base directory ────────────────────────────────
            let sessions_base = config.resolved_sessions_dir(&workspace_dir);

            // ── Provider ────────────────────────────────────────────────────
            let provider: Arc<dyn provider::Provider> =
                Arc::new(AnthropicProvider::new(&config.anthropic));

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

            // ── Channel + Agent (Matrix or Discord, if configured) ──────────
            if !config.standby_mode
                && (config.matrix.is_some() || config.discord.is_some())
            {
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

                // ── Catch up on any pending daily logs ──────────────────────
                catchup_pending_logs(
                    &channel_session_store,
                    provider.as_ref(),
                    &workspace_dir,
                    config.day_boundary_hour,
                )
                .await;

                // ── Agent ───────────────────────────────────────────────────
                let agent = Arc::new(Agent::new(
                    config.clone(),
                    channel,
                    Arc::clone(&provider),
                    Arc::clone(&workspace),
                    Some(Arc::clone(&tool_set)),
                    Arc::clone(&channel_session_store),
                ));

                // ── Heartbeat (day-boundary + cron loops) ───────────────────
                let default_room_id =
                    config
                        .matrix
                        .as_ref()
                        .map(|m| m.room_id.clone())
                        .or_else(|| {
                            config
                                .discord
                                .as_ref()
                                .and_then(|d| d.channel_ids.first().cloned())
                        });
                let heartbeat = Heartbeat {
                    workspace_dir: workspace_dir.clone(),
                    day_boundary_hour: config.day_boundary_hour,
                    daily_log_enabled: config.daily_log_enabled,
                    memory_compaction_enabled: config.memory_compaction_enabled,
                    session_store: Arc::clone(&channel_session_store),
                    provider: Arc::clone(&provider),
                    agent: Arc::clone(&agent),
                    default_room_id,
                };
                if config.heartbeat_enabled {
                    heartbeat.spawn();
                } else {
                    tracing::info!("Heartbeat disabled by config");
                }

                let agent_run = Arc::clone(&agent);
                tokio::spawn(async move {
                    if let Err(e) = agent_run.run().await {
                        tracing::error!("Agent error: {e:#}");
                    }
                });
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
                    provider,
                    workspace,
                    tool_set,
                    api_session_store,
                )
                .await?;
            }
        }
        Command::Call { .. } => unreachable!(),
    }

    Ok(())
}
