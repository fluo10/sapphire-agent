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
    /// Run as a voice satellite — always-on listening with Silero VAD.
    /// When `--wake-word-model` is set, audio only flows to VAD after
    /// a wake phrase fires (KWS gate).
    Voice {
        /// BCP-47 language hint passed to STT (e.g. "ja", "en"). When
        /// omitted, the server's voice_pipeline default applies.
        #[arg(long)]
        language: Option<String>,

        /// Sherpa-onnx KWS bundle name to enable wake-word mode.
        /// Example: `sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01`.
        /// When omitted, the satellite is always-on (VAD only).
        #[arg(long)]
        wake_word_model: Option<String>,

        /// Path to a keywords file (sherpa-onnx tokenised format).
        /// Defaults to `<bundle>/keywords.txt`. Ignored without
        /// `--wake-word-model`.
        #[arg(long)]
        keywords_file: Option<String>,
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

    if let Some(Command::Voice {
        language,
        wake_word_model,
        keywords_file,
    }) = cli.command
    {
        return voice::run(
            cli.server,
            cli.session,
            cli.room_profile,
            voice::VoiceOptions {
                language,
                wake_word_model,
                keywords_file,
            },
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
