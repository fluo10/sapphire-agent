mod voice;

use anyhow::Result;
use clap::{Parser, Subcommand};
use tracing_subscriber::{EnvFilter, fmt};

#[derive(Parser)]
#[command(
    name = "sapphire-call",
    about = "Interactive client for sapphire-agent (text or voice)"
)]
struct Cli {
    /// Server base URL
    #[arg(long, default_value = "http://localhost:9000", global = true)]
    server: String,

    /// Grain-id of an existing session to resume (e.g. a3b7k9p)
    #[arg(long, global = true)]
    session: Option<String>,

    /// Room profile name to bind to a newly created session. Must match a
    /// `[room_profile.<name>]` entry on the server side. Ignored when
    /// resuming an existing session via --session.
    #[arg(long, global = true)]
    room_profile: Option<String>,

    /// List available API sessions and exit (chat mode only)
    #[arg(long)]
    list: bool,

    /// Send a single message and exit instead of entering the REPL.
    /// Useful as a CJK-safe fallback or for IDE/editor integration.
    /// (chat mode only)
    #[arg(short, long, value_name = "TEXT")]
    message: Option<String>,

    /// Dump the session history and exit (no message sent, no REPL).
    /// Intended for IDE integrations restoring a session. (chat mode only)
    #[arg(long)]
    history: bool,

    /// Emit machine-readable JSON output. Applies to --list, --history,
    /// and --message; ignored in REPL mode.
    #[arg(long)]
    json: bool,

    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Subcommand)]
enum Command {
    /// Run as a voice satellite — hold Space to talk, release to send.
    Voice {
        /// BCP-47 language hint passed to STT (e.g. "ja", "en"). When
        /// omitted, the server's voice_pipeline default applies.
        #[arg(long)]
        language: Option<String>,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("warn")),
        )
        .init();

    let cli = Cli::parse();

    if let Some(Command::Voice { language }) = cli.command {
        return voice::run(
            cli.server,
            cli.session,
            cli.room_profile,
            language,
        )
        .await;
    }

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
