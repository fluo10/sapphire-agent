mod config;
mod device_id;
mod voice;

use std::path::PathBuf;

use anyhow::Result;
use clap::{Parser, Subcommand};
use tracing_subscriber::{EnvFilter, fmt};

use config::CallConfig;

const DEFAULT_SERVER_URL: &str = "http://localhost:9000";

#[derive(Parser)]
#[command(
    name = "sapphire-call",
    about = "Interactive client for sapphire-agent (text or voice)"
)]
struct Cli {
    /// Server base URL. Overrides `[server].url` in the config file.
    #[arg(long, global = true)]
    server: Option<String>,

    /// Grain-id of an existing session to resume (e.g. a3b7k9p).
    /// Overrides `[server].session` in the config file.
    #[arg(long, global = true)]
    session: Option<String>,

    /// Room profile name to bind to a newly created session. Must match a
    /// `[room_profile.<name>]` entry on the server side. Ignored when
    /// resuming an existing session via --session. Overrides
    /// `[server].room_profile` in the config file.
    #[arg(long, global = true)]
    room_profile: Option<String>,

    /// Path to a TOML config file. Defaults to
    /// `~/.config/sapphire-call/config.toml` when present; missing is
    /// fine (all values fall back to CLI flags / built-ins).
    #[arg(long, global = true)]
    config: Option<PathBuf>,

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

        /// Enumerate every cpal input + output device visible on this
        /// host and exit. Useful when running headless against a USB
        /// speakerphone — the system "default" rarely points at it.
        #[arg(long)]
        list_devices: bool,

        /// Capture audio from the device whose cpal name matches this
        /// string exactly. Discover names with `--list-devices`.
        /// Defaults to the host's default input device.
        #[arg(long)]
        input_device: Option<String>,

        /// Play audio through the device whose cpal name matches this
        /// string exactly. Defaults to the host's default output device.
        #[arg(long)]
        output_device: Option<String>,
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

    // Load the config file if one exists. Explicit --config requires
    // it to be readable; the implicit XDG location is best-effort.
    let file_cfg = match cli.config.as_deref() {
        Some(path) => CallConfig::load(path)?,
        None => match CallConfig::default_path() {
            Some(p) if p.exists() => CallConfig::load(&p)?,
            _ => CallConfig::default(),
        },
    };

    // Resolution order: CLI flag > config file > built-in default.
    let server = cli
        .server
        .clone()
        .or(file_cfg.server.url)
        .unwrap_or_else(|| DEFAULT_SERVER_URL.to_string());
    let session = cli.session.clone().or(file_cfg.server.session);
    let room_profile = cli.room_profile.clone().or(file_cfg.server.room_profile);
    let device = file_cfg.device.to_api();

    if let Some(Command::Voice {
        language,
        list_devices,
        input_device,
        output_device,
    }) = cli.command
    {
        return voice::run(
            server,
            session,
            room_profile,
            voice::VoiceOptions {
                language: language.or(file_cfg.language.stt),
                list_devices,
                input_device: input_device.or(file_cfg.audio.input_device),
                output_device: output_device.or(file_cfg.audio.output_device),
                device,
                behavior: file_cfg.behavior,
                sensitivity: file_cfg.sensitivity,
            },
        )
        .await;
    }

    sapphire_agent_api::run(
        server,
        session,
        cli.list,
        cli.message,
        cli.history,
        cli.json,
        room_profile,
        device,
    )
    .await
}
