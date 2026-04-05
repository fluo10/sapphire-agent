mod agent;
mod channel;
mod config;
mod provider;
mod workspace;

use agent::Agent;
use anyhow::{Context, Result};
use channel::matrix::MatrixChannel;
use clap::{Parser, Subcommand};
use config::Config;
use provider::anthropic::AnthropicProvider;
use std::path::PathBuf;
use std::sync::Arc;
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
            println!();
            // Show status of each workspace file (openclaw-compatible set)
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
                let found = candidates
                    .iter()
                    .find_map(|f| {
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
            let workspace = Arc::new(Workspace::new(workspace_dir));
            let channel = Arc::new(MatrixChannel::new(&config.matrix));
            let provider = Arc::new(AnthropicProvider::new(&config.anthropic));
            let agent = Arc::new(Agent::new(config, channel, provider, workspace));
            agent.run().await?;
        }
    }

    Ok(())
}
