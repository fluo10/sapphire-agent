mod agent;
mod channel;
mod config;
mod daily_log;
mod heartbeat;
mod provider;
mod session;
mod tools;
mod workspace;

use agent::Agent;
use anyhow::{Context, Result};
use channel::matrix::MatrixChannel;
use clap::{Parser, Subcommand};
use config::Config;
use daily_log::catchup_pending_logs;
use heartbeat::Heartbeat;
use provider::anthropic::AnthropicProvider;
use sapphire_workspace::{Workspace as SwWorkspace, WorkspaceState};
use session::SessionStore;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use tracing_subscriber::{EnvFilter, fmt};
use workspace::Workspace;

#[derive(Parser)]
#[command(name = "sapphire-agent", about = "Personal AI assistant — Matrix + Anthropic")]
struct Cli {
    /// Path to config file (default: ~/.config/sapphire-agent/config.toml)
    #[arg(short, long, value_name = "FILE")]
    config: Option<PathBuf>,

    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand)]
enum Command {
    /// Start the agent (default)
    Run,
    /// Validate the config file and exit
    Verify,
}

#[tokio::main]
async fn main() -> Result<()> {
    fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    let cli = Cli::parse();
    let config_path = cli.config.unwrap_or_else(Config::default_path);
    let config = Config::load(&config_path)
        .with_context(|| format!("Failed to load config from {}", config_path.display()))?;

    match cli.command.unwrap_or(Command::Run) {
        Command::Verify => {
            let workspace_dir = config.resolved_workspace_dir(&config_path);
            println!("Config OK");
            println!("  Matrix homeserver : {}", config.matrix.homeserver);
            println!("  Matrix user_id    : {}", config.matrix.user_id);
            println!("  Matrix room_id    : {}", config.matrix.room_id);
            println!("  Anthropic model   : {}", config.anthropic.model);
            println!("  Anthropic max_tok : {}", config.anthropic.max_tokens);
            println!("  Workspace dir     : {}", workspace_dir.display());
            println!("  Day boundary hour : {}:00 local", config.day_boundary_hour);
            println!();
            let workspace_files = [
                ("AGENTS.md / AGENT.md", vec!["AGENTS.md", "AGENT.md"]),
                ("SOUL.md",              vec!["SOUL.md"]),
                ("IDENTITY.md",          vec!["IDENTITY.md"]),
                ("USER.md",              vec!["USER.md"]),
                ("TOOLS.md",             vec!["TOOLS.md"]),
                ("HEARTBEAT.md",         vec!["HEARTBEAT.md"]),
                ("BOOTSTRAP.md",         vec!["BOOTSTRAP.md"]),
                ("MEMORY.md",            vec!["MEMORY.md", "memory.md"]),
            ];
            for (label, candidates) in &workspace_files {
                let found = candidates.iter().find_map(|f| {
                    let p = workspace_dir.join(f);
                    if p.exists() { Some(*f) } else { None }
                });
                match found {
                    Some(f) => println!("  {label:<28} found ({f})"),
                    None    => println!("  {label:<28} -"),
                }
            }
        }
        Command::Run => {
            let workspace_dir = config.resolved_workspace_dir(&config_path);

            // ── Bootstrap file loader (AGENTS.md, SOUL.md, MEMORY.md …) ────
            let workspace = Arc::new(Workspace::new(workspace_dir.clone()));

            // ── sapphire-workspace (search, file ops, git sync) ─────────────
            let sw_workspace = SwWorkspace::resolve(Some(&workspace_dir))
                .context("Failed to resolve sapphire-workspace")?;
            let ws_state = WorkspaceState::open(sw_workspace)
                .context("Failed to open WorkspaceState")?;
            if let Err(e) = ws_state.sync() {
                tracing::warn!("Initial workspace sync failed: {e}");
            }
            let ws_state = Arc::new(Mutex::new(ws_state));

            // ── Tools ───────────────────────────────────────────────────────
            let tool_set = Arc::new(tools::default_tool_set(
                Arc::clone(&ws_state),
                config.tools.tavily_api_key.clone(),
            ));

            // ── Session store ───────────────────────────────────────────────
            let sessions_dir = config.resolved_sessions_dir(&workspace_dir);
            let session_store = Arc::new(SessionStore::new(sessions_dir));

            // ── Channel + Provider ──────────────────────────────────────────
            let channel = Arc::new(MatrixChannel::new(&config.matrix));
            let provider: Arc<dyn provider::Provider> =
                Arc::new(AnthropicProvider::new(&config.anthropic));

            // ── Catch up on any pending daily logs (agent was offline) ──────
            catchup_pending_logs(
                &session_store,
                provider.as_ref(),
                &workspace_dir,
                config.day_boundary_hour,
            )
            .await;

            // ── Heartbeat (daily log at boundary hour) ───────────────────────
            let heartbeat = Heartbeat {
                day_boundary_hour: config.day_boundary_hour,
                session_store: Arc::clone(&session_store),
                provider: Arc::clone(&provider),
                workspace_dir: workspace_dir.clone(),
            };
            tokio::spawn(heartbeat.run());

            // ── Agent ───────────────────────────────────────────────────────
            let agent = Arc::new(Agent::new(
                config,
                channel,
                provider,
                workspace,
                Some(tool_set),
                session_store,
            ));
            agent.run().await?;
        }
    }

    Ok(())
}
