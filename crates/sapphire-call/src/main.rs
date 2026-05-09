use anyhow::Result;
use clap::Parser;
use tracing_subscriber::{EnvFilter, fmt};

#[derive(Parser)]
#[command(
    name = "sapphire-call",
    about = "Interactive client for sapphire-agent"
)]
struct Cli {
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
}

#[tokio::main]
async fn main() -> Result<()> {
    fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("warn")),
        )
        .init();

    let cli = Cli::parse();
    sapphire_agent_api::run(
        cli.server,
        cli.session,
        cli.list,
        cli.message,
        cli.history,
        cli.json,
        cli.room_profile,
    )
    .await
}
