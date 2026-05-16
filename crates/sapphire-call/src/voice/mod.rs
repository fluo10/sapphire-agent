//! Voice satellite subcommand.
//!
//! Two modes, selected by whether `--wake-word-model` is set:
//!
//! - **VAD-only (default):** hot mic for the process lifetime, Silero
//!   VAD chunks the stream into utterances, each segment ships to the
//!   agent. Good for quiet rooms / single-user desks.
//! - **Wake-then-VAD:** sherpa-onnx KWS gates the VAD. While Idle the
//!   satellite only feeds audio to the keyword spotter; on a wake
//!   match it switches to VAD until an utterance completes, ships it,
//!   plays the reply, and returns to wake-listening.
//!
//! In both modes the mic is gated off during reply playback so the
//! satellite doesn't transcribe its own TTS.

mod download;
mod oww;

use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow};
use cpal::SampleFormat;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use sapphire_agent_api::{
    VoiceEvent, VoicePushEvent, voice::PIPELINE_SAMPLE_RATE, voice_pipeline_run, voice_subscribe,
};
use sherpa_onnx::{SileroVadModelConfig, VadModelConfig, VoiceActivityDetector};
use tokio::sync::mpsc;

/// Command sent from the `voice/subscribe` consumer task into the
/// listen loop so server-initiated pushes (heartbeat fires) can route
/// audio through the same playback gate the regular reply path uses.
enum ListenCommand {
    /// One PCM chunk from the server push, at [`PIPELINE_SAMPLE_RATE`].
    /// The listen loop mutes the mic on the first chunk of a push and
    /// appends to the playback queue.
    PushAudio(Vec<i16>),
    /// Server has finished pushing; listen loop should drain playback
    /// and arm the follow-up listening window.
    PushDone,
}

/// Silero VAD frame size in samples. Required by the model.
const VAD_WINDOW_SAMPLES: usize = 512;
/// Hard cap on a single utterance (seconds). Matches the server's
/// default capture_max_ms / 1000.
const VAD_MAX_SPEECH_SECONDS: f32 = 30.0;

/// Public URL of the Silero VAD ONNX model on the sherpa-onnx
/// releases page. Auto-downloaded on first run.
const SILERO_VAD_URL: &str =
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx";

/// Run-time options collected from CLI flags.
///
/// Wake-word configuration lives server-side under `[voice]`. The
/// satellite fetches the ONNX classifier inline via `voice/config`
/// right after device id resolution.
pub struct VoiceOptions {
    pub language: Option<String>,
    /// When true, enumerate cpal devices and exit before opening any
    /// stream or talking to the agent.
    pub list_devices: bool,
    /// Pin capture to the cpal device with this exact name; None ⇒
    /// `default_input_device()`.
    pub input_device: Option<String>,
    /// Pin playback to the cpal device with this exact name; None ⇒
    /// `default_output_device()`.
    pub output_device: Option<String>,
    /// Optional device identity sent in every `voice/pipeline_run` so the
    /// agent can render "voice channel with <name>" in the system prompt.
    pub device: Option<sapphire_agent_api::DeviceMetadata>,
    /// Listen-state UX: confirmation beeps + post-reply follow-up
    /// listening window. See [`crate::config::BehaviorConfig`].
    pub behavior: crate::config::BehaviorConfig,
    /// Mic gain + wake/VAD sensitivity knobs (issue #87). Defaults
    /// preserve the historical hard-coded values.
    pub sensitivity: crate::config::SensitivityConfig,
}

/// Frequency / duration of the confirmation beeps. Picked to be
/// distinct (rising vs falling) and short enough not to clip the start
/// of a quick command.
const BEEP_DURATION_MS: u32 = 150;
const BEEP_WAKE_HZ: f32 = 880.0;
const BEEP_CAPTURE_END_HZ: f32 = 660.0;

// ── cpal stream supervisor tuning ──────────────────────────────────────
//
// cpal hands per-stream errors to `err_fn`; a misbehaving ALSA backend
// can flood that hook (the user-observed case: `alsa::poll()` returned
// POLLERR repeatedly, with no recovery API exposed by cpal). Two
// defences, applied symmetrically to *both* the input and output
// streams (we don't know the root cause of the original input flood,
// so it's prudent to assume the output side is equally exposed):
//   1. Rate-limit stderr writes so the terminal stays responsive — a
//      flooded stderr was preventing Ctrl-C from being delivered.
//   2. Track errors in a sliding window: once the window crosses a
//      threshold, ask the supervisor thread to drop the Stream and
//      build a fresh one.
//
// The numbers are chosen to tolerate occasional xruns (which can
// surface as a single POLLERR) while reacting quickly to a sustained
// failure.

/// Errors observed in `STREAM_ERROR_WINDOW` before the supervisor is
/// told to rebuild the stream.
const STREAM_ERROR_THRESHOLD: u32 = 10;
/// How fresh an error must be to count toward `STREAM_ERROR_THRESHOLD`.
/// Older errors are forgiven — a long-running session shouldn't restart
/// a stream just because three xruns piled up over an hour.
const STREAM_ERROR_WINDOW: Duration = Duration::from_secs(5);
/// Throttle for the per-error stderr line. cpal can call `err_fn`
/// hundreds of times per second when ALSA is stuck.
const STREAM_ERROR_PRINT_INTERVAL: Duration = Duration::from_secs(1);
/// Initial back-off between a `Stream` drop and the next build
/// attempt. Doubles per failure, capped.
const STREAM_REBUILD_BACKOFF_INITIAL: Duration = Duration::from_millis(500);
const STREAM_REBUILD_BACKOFF_MAX: Duration = Duration::from_secs(30);
/// Give up the satellite after this many *consecutive* rebuild
/// failures (the underlying audio device is probably physically gone
/// — e.g. USB unplugged — and no amount of retrying will help). The
/// counter resets every time we successfully (re)acquire the stream,
/// so a long uptime with one error-recovery event doesn't accumulate.
const STREAM_REBUILD_MAX_FAILURES: u32 = 5;

/// Shared state between cpal's `err_fn` and the supervisor std::thread
/// that owns the [`cpal::Stream`]. Lives behind `Arc` because
/// (a) cpal's error callback requires `Send + 'static`, and
/// (b) `cpal::Stream` is `!Send` on most backends, forcing the
/// supervisor to live on its own dedicated OS thread.
///
/// One instance per stream — input and output each get their own so
/// errors on one side don't accidentally rebuild the other.
struct StreamSupervisor {
    /// Sliding-window error tracker + last-print timestamp for the
    /// stderr rate-limit. err_fn updates this on every callback.
    error_window: Mutex<ErrorWindow>,
    /// Set to true by err_fn when the threshold is crossed, or by the
    /// Ctrl-C handler on shutdown. The supervisor `wait`s on `cvar`
    /// until either this flips or shutdown is observed.
    needs_restart: Mutex<bool>,
    cvar: Condvar,
}

#[derive(Default)]
struct ErrorWindow {
    /// When the first error in the current window was observed.
    /// Cleared by the supervisor after a successful (re)build, and
    /// rolled forward by err_fn when the window has elapsed.
    first_at: Option<Instant>,
    /// Errors counted toward `STREAM_ERROR_THRESHOLD` since `first_at`.
    count: u32,
    /// Last time we actually wrote an error line to stderr.
    last_print_at: Option<Instant>,
}

impl StreamSupervisor {
    fn new() -> Self {
        Self {
            error_window: Mutex::new(ErrorWindow::default()),
            needs_restart: Mutex::new(false),
            cvar: Condvar::new(),
        }
    }

    /// Wake the supervisor from its `wait_timeout` — used by err_fn
    /// when crossing the error threshold, and by the Ctrl-C handler
    /// to short-circuit the shutdown path.
    fn notify(&self) {
        self.cvar.notify_one();
    }
}

/// Result of feeding one error into the error window. Returned to
/// err_fn (which lives on cpal's worker thread) so the I/O side-effects
/// — printing to stderr and flipping the restart flag — happen *outside*
/// the lock.
struct ErrorDecision {
    /// Print this error line to stderr.
    print: bool,
    /// Total errors in the current window after this one.
    count: u32,
    /// We just crossed `STREAM_ERROR_THRESHOLD` — ask the supervisor
    /// to rebuild the stream.
    request_restart: bool,
}

/// Pure error-window update — exercised by unit tests so the
/// rate-limit + restart-threshold logic can be verified without
/// spinning up cpal. Returns what err_fn should do next; the caller
/// owns the side-effects.
fn record_stream_error(window: &mut ErrorWindow, now: Instant) -> ErrorDecision {
    // Roll the window forward if the previous window has fully
    // elapsed — that way three errors per minute don't add up to a
    // false-positive restart over the course of an hour.
    if let Some(first) = window.first_at
        && now.duration_since(first) > STREAM_ERROR_WINDOW
    {
        window.first_at = None;
        window.count = 0;
    }
    if window.first_at.is_none() {
        window.first_at = Some(now);
    }
    window.count += 1;

    let print = match window.last_print_at {
        Some(last) if now.duration_since(last) < STREAM_ERROR_PRINT_INTERVAL => false,
        _ => {
            window.last_print_at = Some(now);
            true
        }
    };

    // Fire the restart request exactly once per window: when we cross
    // the threshold. The supervisor will reset first_at/count after
    // the rebuild, so subsequent errors start a fresh window.
    let request_restart = window.count == STREAM_ERROR_THRESHOLD;

    ErrorDecision {
        print,
        count: window.count,
        request_restart,
    }
}

/// Common err_fn body: rate-limit stderr, count errors, and flip the
/// supervisor's restart flag when the window crosses the threshold.
/// Used by both `open_input_stream` and `open_output_stream` so the
/// behaviour is identical on either side; only the `label` differs.
fn handle_stream_error(
    supervisor: &Arc<StreamSupervisor>,
    label: &'static str,
    e: cpal::StreamError,
) {
    let now = Instant::now();
    let decision = {
        let mut w = supervisor
            .error_window
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        record_stream_error(&mut w, now)
    };
    if decision.print {
        eprintln!("[{label}: {e}] (#{} in window)", decision.count);
    }
    if decision.request_restart {
        let mut needs = supervisor
            .needs_restart
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        if !*needs {
            *needs = true;
            supervisor.notify();
        }
    }
}

/// Entry point for `sapphire-call voice`.
pub async fn run(
    server: String,
    _session: Option<String>,
    room_profile: Option<String>,
    options: VoiceOptions,
) -> Result<()> {
    // `--list-devices` short-circuits before we touch the network /
    // download any models / open any audio streams.
    if options.list_devices {
        return list_devices();
    }

    let base = server.trim_end_matches('/').to_string();
    let client = reqwest::Client::new();

    // Voice conversations are routed by (device_id, room_profile) so
    // the satellite picks up where it left off across restarts.
    // `session` (legacy chat session id) is ignored in voice mode.
    let device_id = crate::device_id::ensure_device_id()
        .context("failed to load or generate sapphire-call device id")?;
    let room_profile = room_profile.ok_or_else(|| {
        anyhow!(
            "voice mode requires a room_profile — set [server].room_profile in the config \
             or pass --room-profile <name>"
        )
    })?;
    eprintln!("sapphire-call voice (device: {device_id}, room_profile: {room_profile})",);

    // ── Wake-word config: fetch the inline ONNX from the server ─────────
    let server_wake = sapphire_agent_api::voice_config(&client, &base)
        .await
        .unwrap_or_else(|e| {
            eprintln!(
                "warning: voice/config fetch failed ({e:#}); proceeding without wake-word gating"
            );
            sapphire_agent_api::WakeWordConfig::default()
        });

    // ── VAD model ────────────────────────────────────────────────────────
    // The download + sherpa-onnx construction are synchronous and use
    // `reqwest::blocking` internally, which spawns its own tokio
    // runtime. Run them on the blocking thread pool so the inner
    // runtime can drop without conflicting with our outer #[tokio::main].
    let vad_model_path = tokio::task::spawn_blocking(ensure_silero_model)
        .await
        .map_err(|e| anyhow!("VAD download task panicked: {e}"))?
        .context("failed to fetch Silero VAD model")?;
    eprintln!("VAD model: {}", vad_model_path.display());
    let vad = tokio::task::spawn_blocking({
        let path = vad_model_path.clone();
        let sens = options.sensitivity.clone();
        move || build_vad(&path, &sens)
    })
    .await
    .map_err(|e| anyhow!("VAD build task panicked: {e}"))??;

    // ── Optional wake-word detector ─────────────────────────────────────
    let wake_detector: Option<oww::OpenWakeWordDetector> = match server_wake.model {
        Some(model) => {
            let sapphire_agent_api::WakeWordModel {
                filename,
                sha256,
                bytes,
            } = model;
            // Derive a display label from the filename stem (without
            // version suffix or extension) so wake events print
            // something recognisable like "[wake: saphina]".
            let label = std::path::Path::new(&filename)
                .file_stem()
                .and_then(|s| s.to_str())
                .map(|s| s.split('-').next().unwrap_or(s).to_string())
                .unwrap_or_else(|| "wake".to_string());
            let wake_threshold = options.sensitivity.wake_threshold;
            let wake_cooldown_chunks =
                oww::cooldown_chunks_from_ms(options.sensitivity.wake_cooldown_ms);
            let detector = tokio::task::spawn_blocking({
                let label = label.clone();
                move || {
                    let (mel, embed) = download::ensure_oww_frontend()
                        .context("failed to fetch openWakeWord frontend models")?;
                    let wake_path = download::cache_inline_oww(&bytes, &sha256)
                        .context("failed to cache openWakeWord classifier")?;
                    oww::OpenWakeWordDetector::create(
                        &mel,
                        &embed,
                        &wake_path,
                        label,
                        wake_threshold,
                        wake_cooldown_chunks,
                    )
                }
            })
            .await
            .map_err(|e| anyhow!("openWakeWord init task panicked: {e}"))??;
            eprintln!("wake-word: {filename} (label: {label})");
            Some(detector)
        }
        None => None,
    };

    // ── Audio I/O ────────────────────────────────────────────────────────
    //
    // Both the input and output sides are wrapped in supervisor
    // std::threads that own their `cpal::Stream` for its lifetime and
    // rebuild it on err_fn-triggered restart requests. `cpal::Stream`
    // is `!Send` on most backends so we can't park it across an
    // `.await`; the dedicated OS thread sidesteps that.
    //
    // Each supervisor is independent (its own `StreamSupervisor` state
    // + cvar) — errors on the mic don't cause a speaker rebuild and
    // vice versa. If either supervisor exhausts its rebuild budget it
    // sets `shutdown` and exits, which propagates to the rest of the
    // satellite. See [`supervise_stream`] for the rebuild / give-up
    // policy.
    let (audio_tx, audio_rx) = mpsc::unbounded_channel::<Vec<i16>>();
    let mic_enabled = Arc::new(AtomicBool::new(true));
    let shutdown = Arc::new(AtomicBool::new(false));
    let input_supervisor = Arc::new(StreamSupervisor::new());
    let output_supervisor = Arc::new(StreamSupervisor::new());
    let playback_queue: Arc<std::sync::Mutex<VecDeque<i16>>> =
        Arc::new(std::sync::Mutex::new(VecDeque::new()));

    let (input_ready_tx, input_ready_rx) = tokio::sync::oneshot::channel::<Result<(u32, u16)>>();
    let input_handle = std::thread::Builder::new()
        .name("sapphire-call input-supervisor".into())
        .spawn({
            let name = options.input_device.clone();
            let enabled = Arc::clone(&mic_enabled);
            let shutdown = Arc::clone(&shutdown);
            let supervisor = Arc::clone(&input_supervisor);
            let mic_gain = options.sensitivity.mic_gain;
            move || {
                if let Some(ref n) = name {
                    eprintln!("input device: {n}");
                }
                let supervisor_for_build = Arc::clone(&supervisor);
                let build = move || -> Result<(cpal::Stream, u32, u16)> {
                    let (stream, rate, channels) = open_input_stream(
                        name.as_deref(),
                        audio_tx.clone(),
                        Arc::clone(&enabled),
                        mic_gain,
                        Arc::clone(&supervisor_for_build),
                    )?;
                    stream.play().context("input stream play")?;
                    Ok((stream, rate, channels))
                };
                supervise_stream("input stream", build, shutdown, supervisor, input_ready_tx);
            }
        })
        .context("failed to spawn input-stream supervisor thread")?;

    let (output_ready_tx, output_ready_rx) = tokio::sync::oneshot::channel::<Result<(u32, u16)>>();
    let output_handle = std::thread::Builder::new()
        .name("sapphire-call output-supervisor".into())
        .spawn({
            let name = options.output_device.clone();
            let queue = Arc::clone(&playback_queue);
            let shutdown = Arc::clone(&shutdown);
            let supervisor = Arc::clone(&output_supervisor);
            move || {
                if let Some(ref n) = name {
                    eprintln!("output device: {n}");
                }
                let supervisor_for_build = Arc::clone(&supervisor);
                let build = move || -> Result<(cpal::Stream, u32, u16)> {
                    let (stream, rate, channels) = open_output_stream(
                        name.as_deref(),
                        Arc::clone(&queue),
                        Arc::clone(&supervisor_for_build),
                    )?;
                    stream.play().context("output stream play")?;
                    Ok((stream, rate, channels))
                };
                supervise_stream(
                    "output stream",
                    build,
                    shutdown,
                    supervisor,
                    output_ready_tx,
                );
            }
        })
        .context("failed to spawn output-stream supervisor thread")?;

    let (input_rate, input_channels) = input_ready_rx
        .await
        .map_err(|e| anyhow!("input supervisor dropped before initial build: {e}"))??;
    let (output_rate, _output_channels) = output_ready_rx
        .await
        .map_err(|e| anyhow!("output supervisor dropped before initial build: {e}"))??;

    eprintln!("input: {input_rate} Hz × {input_channels}ch  output: {output_rate} Hz",);
    eprintln!("Listening. Ctrl-C to quit.");

    // ── Ctrl-C handler ──────────────────────────────────────────────────
    {
        let shutdown = Arc::clone(&shutdown);
        let in_sup = Arc::clone(&input_supervisor);
        let out_sup = Arc::clone(&output_supervisor);
        tokio::spawn(async move {
            if tokio::signal::ctrl_c().await.is_ok() {
                eprintln!("\nShutting down...");
                shutdown.store(true, Ordering::SeqCst);
                // Wake both supervisors so they drop their cpal
                // Streams promptly. The input supervisor dropping its
                // audio_tx clone is also what closes the channel and
                // pops listen_loop out of its audio_rx.recv() await.
                in_sup.notify();
                out_sup.notify();
            }
        });
    }

    // ── Background voice/subscribe consumer ──────────────────────────────
    // Heartbeat pushes (morning calls etc.) land here. The consumer
    // translates push events into ListenCommands so the listen loop
    // owns the playback queue / mic gate as it does for regular replies.
    let (cmd_tx, cmd_rx) = mpsc::unbounded_channel::<ListenCommand>();
    let subscribe_handle = tokio::spawn(subscribe_loop(
        client.clone(),
        base.clone(),
        device_id.clone(),
        room_profile.clone(),
        cmd_tx,
        Arc::clone(&shutdown),
    ));

    let awaiting_wake = wake_detector.is_some();
    let listen_result = listen_loop(ListenCtx {
        audio_rx,
        input_rate,
        input_channels,
        mic_enabled,
        playback_queue,
        output_rate,
        client,
        base,
        device_id,
        room_profile,
        language: options.language,
        device: options.device,
        shutdown: Arc::clone(&shutdown),
        vad,
        wake: wake_detector,
        awaiting_wake,
        behavior: options.behavior,
        follow_up_until: None,
        cmd_rx,
        push_active: false,
    })
    .await;
    subscribe_handle.abort();

    // Make sure both supervisors see shutdown before we try to join
    // them. listen_loop may have exited cleanly (audio_rx closed
    // because the input supervisor gave up) or returned an error —
    // in either case we want both supervisor threads to terminate so
    // their OS threads don't outlive the process.
    shutdown.store(true, Ordering::SeqCst);
    input_supervisor.notify();
    output_supervisor.notify();
    if let Err(e) = input_handle.join() {
        eprintln!("[input supervisor thread panicked: {e:?}]");
    }
    if let Err(e) = output_handle.join() {
        eprintln!("[output supervisor thread panicked: {e:?}]");
    }

    listen_result?;

    Ok(())
}

struct ListenCtx {
    audio_rx: mpsc::UnboundedReceiver<Vec<i16>>,
    input_rate: u32,
    input_channels: u16,
    mic_enabled: Arc<AtomicBool>,
    playback_queue: Arc<std::sync::Mutex<VecDeque<i16>>>,
    output_rate: u32,
    client: reqwest::Client,
    base: String,
    device_id: String,
    room_profile: String,
    language: Option<String>,
    device: Option<sapphire_agent_api::DeviceMetadata>,
    shutdown: Arc<AtomicBool>,
    vad: VoiceActivityDetector,
    /// `Some` enables wake-word mode; the satellite gates VAD behind
    /// successful keyword spotting on this detector.
    wake: Option<oww::OpenWakeWordDetector>,
    /// In wake-word mode: true while we're waiting for the next wake
    /// trigger, false once it fires and we're capturing the utterance.
    /// Always false in VAD-only mode.
    awaiting_wake: bool,
    /// Listen-state UX (beeps + follow-up window).
    behavior: crate::config::BehaviorConfig,
    /// In wake-word mode, when set, the satellite is in the post-reply
    /// follow-up window: VAD is active without requiring another wake
    /// word until this deadline. Cleared when a follow-up utterance
    /// starts being processed (and re-armed once that turn finishes)
    /// or when the deadline elapses without speech.
    follow_up_until: Option<Instant>,
    /// Server-initiated push commands (heartbeat fires) from the
    /// `voice/subscribe` consumer task. Multiplexed alongside
    /// `audio_rx` via `tokio::select!` so the listen loop stays the
    /// single owner of the mic gate and playback queue.
    cmd_rx: mpsc::UnboundedReceiver<ListenCommand>,
    /// True between the first `PushAudio` of a server push and the
    /// matching `PushDone`. While set, the listen loop has muted the
    /// mic and is buffering the pushed PCM in the playback queue.
    push_active: bool,
}

async fn listen_loop(mut ctx: ListenCtx) -> Result<()> {
    let mut window_buf: Vec<f32> = Vec::with_capacity(VAD_WINDOW_SAMPLES);
    while !ctx.shutdown.load(Ordering::SeqCst) {
        // Race the mic stream against any pending server push command
        // (heartbeat fires from voice/subscribe). In the post-reply
        // follow-up window, the recv branch is wrapped in a deadline
        // so silence eventually drops us back into wake-listening mode.
        let event = match ctx.follow_up_until {
            Some(deadline) => {
                let now = Instant::now();
                if now >= deadline {
                    expire_follow_up(&mut ctx, &mut window_buf).await;
                    continue;
                }
                let remaining = deadline - now;
                tokio::select! {
                    biased;
                    cmd = ctx.cmd_rx.recv() => match cmd {
                        Some(c) => ListenEvent::Command(c),
                        None => ListenEvent::CommandClosed,
                    },
                    audio = tokio::time::timeout(remaining, ctx.audio_rx.recv()) => match audio {
                        Ok(Some(buf)) => ListenEvent::Audio(buf),
                        Ok(None) => break,
                        Err(_) => {
                            expire_follow_up(&mut ctx, &mut window_buf).await;
                            continue;
                        }
                    },
                }
            }
            None => tokio::select! {
                biased;
                cmd = ctx.cmd_rx.recv() => match cmd {
                    Some(c) => ListenEvent::Command(c),
                    None => ListenEvent::CommandClosed,
                },
                audio = ctx.audio_rx.recv() => match audio {
                    Some(buf) => ListenEvent::Audio(buf),
                    None => break,
                },
            },
        };

        let raw = match event {
            ListenEvent::Audio(buf) => buf,
            ListenEvent::Command(cmd) => {
                handle_push_command(&mut ctx, cmd, &mut window_buf).await;
                continue;
            }
            // Subscribe loop ended (it errored / was aborted). Carry
            // on listening normally — the satellite still works as a
            // pure command station.
            ListenEvent::CommandClosed => continue,
        };
        if !ctx.mic_enabled.load(Ordering::SeqCst) {
            continue;
        }
        // Convert to mono + resample to the pipeline rate, then to f32.
        let mono = to_mono(&raw, ctx.input_channels);
        let pcm16k = resample_to(&mono, ctx.input_rate, PIPELINE_SAMPLE_RATE);
        let pcm_f32: Vec<f32> = pcm16k.iter().map(|s| *s as f32 / 32768.0).collect();

        // Wake-word gate: while we're awaiting a wake trigger, feed
        // audio to the openWakeWord detector only — VAD doesn't see
        // the stream until the wake fires.
        if ctx.awaiting_wake {
            if let Some(ref mut wake) = ctx.wake
                && let Some(keyword) = wake.feed(&pcm16k)?
            {
                eprintln!("[wake: {keyword}] now listening for command...");
                ctx.awaiting_wake = false;
                // Clear any audio left over from before the wake so
                // the VAD starts on the post-wake utterance.
                ctx.vad.reset();
                window_buf.clear();
                if ctx.behavior.beep_on_wake {
                    // Mute, play the rising tone, drain anything
                    // already in flight, then re-open. A residual
                    // input frame queued before the mute would
                    // otherwise be re-processed as part of the
                    // command itself.
                    ctx.mic_enabled.store(false, Ordering::SeqCst);
                    while ctx.audio_rx.try_recv().is_ok() {}
                    enqueue_beep(&ctx.playback_queue, BEEP_WAKE_HZ, ctx.output_rate);
                    wait_for_playback_drain(&ctx.playback_queue).await;
                    while ctx.audio_rx.try_recv().is_ok() {}
                    ctx.mic_enabled.store(true, Ordering::SeqCst);
                }
            }
            continue;
        }

        for f in &pcm_f32 {
            window_buf.push(*f);
            if window_buf.len() == VAD_WINDOW_SAMPLES {
                ctx.vad.accept_waveform(&window_buf);
                window_buf.clear();
            }
        }
        // Drain every completed speech segment Silero has emitted.
        while !ctx.vad.is_empty() {
            let Some(segment) = ctx.vad.front() else {
                break;
            };
            // f32 mono → i16 mono (pipeline format).
            let utterance: Vec<i16> = segment
                .samples()
                .iter()
                .map(|f| (f.clamp(-1.0, 1.0) * i16::MAX as f32) as i16)
                .collect();
            ctx.vad.pop();
            // Gate the mic so the agent's TTS reply doesn't feed back
            // into the VAD as a new utterance.
            ctx.mic_enabled.store(false, Ordering::SeqCst);
            while ctx.audio_rx.try_recv().is_ok() {}
            ctx.vad.reset();
            window_buf.clear();
            // Capturing this utterance means any pending follow-up window
            // is being consumed; clear it so it can't fire mid-process.
            // It will be re-armed after playback if behavior allows.
            ctx.follow_up_until = None;

            // Capture-end beep: queued onto the playback path so it plays
            // before the TTS reply that arrives a moment later. No drain
            // wait here — the inevitable post-reply drain covers both.
            if ctx.behavior.beep_on_capture_end {
                enqueue_beep(&ctx.playback_queue, BEEP_CAPTURE_END_HZ, ctx.output_rate);
            }

            eprintln!(
                "→ uploading {} samples ({:.2}s)",
                utterance.len(),
                utterance.len() as f32 / PIPELINE_SAMPLE_RATE as f32
            );
            if let Err(e) = process_utterance(
                &ctx.client,
                &ctx.base,
                &ctx.device_id,
                &ctx.room_profile,
                &utterance,
                ctx.language.as_deref(),
                ctx.device.as_ref(),
                Arc::clone(&ctx.playback_queue),
                ctx.output_rate,
            )
            .await
            {
                eprintln!("[error: {e:#}]");
            }
            wait_for_playback_drain(&ctx.playback_queue).await;
            while ctx.audio_rx.try_recv().is_ok() {}
            ctx.vad.reset();
            window_buf.clear();
            // In wake-word mode, optionally stay in a short follow-up
            // window so the user can reply without re-waking; otherwise
            // (or in VAD-only mode) keep the existing behaviour.
            //
            // `into_listen` = true means the mic is about to re-open
            // for command capture (VAD-only mode, or wake-mode with a
            // follow-up window). In that case we play `beep_on_wake`
            // before unmuting so the user knows the turn handed back
            // to them. When we're going back to wake-wait
            // (secs == 0), the next listen-start beep will fire on
            // the actual wake event itself — don't double up.
            let into_listen = match (ctx.wake.as_mut(), ctx.behavior.follow_up_listen_seconds) {
                (Some(wake), 0) => {
                    wake.reset();
                    ctx.awaiting_wake = true;
                    eprintln!("Waiting for wake word.");
                    false
                }
                (Some(_), secs) => {
                    ctx.awaiting_wake = false;
                    ctx.follow_up_until = Some(Instant::now() + Duration::from_secs(secs as u64));
                    eprintln!("Listening for follow-up... ({secs}s)");
                    true
                }
                (None, _) => {
                    eprintln!("Listening.");
                    true
                }
            };
            if into_listen && ctx.behavior.beep_on_wake {
                enqueue_beep(&ctx.playback_queue, BEEP_WAKE_HZ, ctx.output_rate);
                wait_for_playback_drain(&ctx.playback_queue).await;
                while ctx.audio_rx.try_recv().is_ok() {}
            }
            ctx.mic_enabled.store(true, Ordering::SeqCst);
        }
    }
    Ok(())
}

/// What the listen loop's main `select!` produced this iteration.
enum ListenEvent {
    Audio(Vec<i16>),
    Command(ListenCommand),
    CommandClosed,
}

/// Apply a server push command. The first `PushAudio` of a push mutes
/// the mic and starts buffering pushed PCM in the playback queue;
/// `PushDone` waits for the queue to drain and arms the follow-up
/// listening window so the user can reply without re-waking.
async fn handle_push_command(ctx: &mut ListenCtx, cmd: ListenCommand, window_buf: &mut Vec<f32>) {
    match cmd {
        ListenCommand::PushAudio(pcm) => {
            if !ctx.push_active {
                ctx.push_active = true;
                // Mute mic for the duration of the push so the AI's
                // own audio doesn't feed back into VAD as a new
                // utterance. Drain anything already queued in
                // audio_rx — that's pre-mute audio that would
                // otherwise pollute the next VAD window.
                ctx.mic_enabled.store(false, Ordering::SeqCst);
                while ctx.audio_rx.try_recv().is_ok() {}
                ctx.vad.reset();
                window_buf.clear();
            }
            let upsampled = resample_to(&pcm, PIPELINE_SAMPLE_RATE, ctx.output_rate);
            if let Ok(mut q) = ctx.playback_queue.lock() {
                q.extend(upsampled);
            }
        }
        ListenCommand::PushDone => {
            wait_for_playback_drain(&ctx.playback_queue).await;
            while ctx.audio_rx.try_recv().is_ok() {}
            ctx.vad.reset();
            window_buf.clear();
            ctx.push_active = false;
            // Same follow-up arming as the regular reply path: in
            // wake-word mode, give the user a window to reply
            // without re-waking; otherwise just keep listening.
            let into_listen = match (ctx.wake.as_mut(), ctx.behavior.follow_up_listen_seconds) {
                (Some(wake), 0) => {
                    wake.reset();
                    ctx.awaiting_wake = true;
                    ctx.follow_up_until = None;
                    eprintln!("Push complete. Waiting for wake word.");
                    false
                }
                (Some(_), secs) => {
                    ctx.awaiting_wake = false;
                    ctx.follow_up_until = Some(Instant::now() + Duration::from_secs(secs as u64));
                    eprintln!("Push complete. Listening for follow-up... ({secs}s)");
                    true
                }
                (None, _) => {
                    eprintln!("Push complete. Listening.");
                    true
                }
            };
            if into_listen && ctx.behavior.beep_on_wake {
                enqueue_beep(&ctx.playback_queue, BEEP_WAKE_HZ, ctx.output_rate);
                wait_for_playback_drain(&ctx.playback_queue).await;
                while ctx.audio_rx.try_recv().is_ok() {}
            }
            ctx.mic_enabled.store(true, Ordering::SeqCst);
        }
    }
}

/// Background task: keep a `voice/subscribe` SSE stream open and
/// translate incoming push events into [`ListenCommand`]s for the
/// listen loop. Reconnects with a small backoff on disconnection so a
/// flaky network doesn't permanently silence heartbeat notifications.
async fn subscribe_loop(
    client: reqwest::Client,
    base: String,
    device_id: String,
    room_profile: String,
    cmd_tx: mpsc::UnboundedSender<ListenCommand>,
    shutdown: Arc<AtomicBool>,
) {
    let mut backoff_secs: u64 = 1;
    while !shutdown.load(Ordering::SeqCst) {
        let (push_tx, mut push_rx) = mpsc::channel::<VoicePushEvent>(32);
        let conn = voice_subscribe(&client, &base, &device_id, &room_profile, push_tx);
        // Forward push events as ListenCommands while the subscribe
        // call runs to completion in parallel.
        let forwarder = {
            let cmd_tx = cmd_tx.clone();
            tokio::spawn(async move {
                while let Some(evt) = push_rx.recv().await {
                    match evt {
                        VoicePushEvent::PushStart { task } => {
                            eprintln!(
                                "[push start{}]",
                                task.as_deref()
                                    .map(|t| format!(": {t}"))
                                    .unwrap_or_default()
                            );
                        }
                        VoicePushEvent::AssistantText { text } => {
                            eprintln!("(push) {text}");
                        }
                        VoicePushEvent::AudioChunk { pcm } => {
                            if cmd_tx.send(ListenCommand::PushAudio(pcm)).is_err() {
                                break;
                            }
                        }
                        VoicePushEvent::PushDone => {
                            let _ = cmd_tx.send(ListenCommand::PushDone);
                        }
                        VoicePushEvent::Error { message } => {
                            eprintln!("[push error: {message}]");
                            // Still emit PushDone so the listen loop
                            // can flush any partial playback and reset.
                            let _ = cmd_tx.send(ListenCommand::PushDone);
                        }
                    }
                }
            })
        };

        match conn.await {
            Ok(()) => {
                // Server closed cleanly. Reset backoff and reconnect.
                backoff_secs = 1;
            }
            Err(e) => {
                eprintln!("voice/subscribe disconnected: {e:#}");
            }
        }
        forwarder.abort();
        if shutdown.load(Ordering::SeqCst) {
            break;
        }
        tokio::time::sleep(Duration::from_secs(backoff_secs)).await;
        backoff_secs = (backoff_secs * 2).min(30);
    }
}

/// Time-out hook for the follow-up window: silence elapsed without a
/// new utterance, so reset transcription state and return to
/// wake-listening. Caller `continue`s after this.
///
/// Fires `beep_on_capture_end` (falling tone) before we leave the
/// listening state — listen-end is conceptually the same event as the
/// "I shipped your utterance to STT" beep: the satellite has stopped
/// taking audio for this turn.
async fn expire_follow_up(ctx: &mut ListenCtx, window_buf: &mut Vec<f32>) {
    ctx.follow_up_until = None;
    ctx.vad.reset();
    window_buf.clear();
    if let Some(ref mut wake) = ctx.wake {
        wake.reset();
        ctx.awaiting_wake = true;
        eprintln!("Follow-up window elapsed. Waiting for wake word.");
        if ctx.behavior.beep_on_capture_end {
            // Mute mic for the duration of the beep so we don't
            // re-capture the falling tone as a new utterance.
            ctx.mic_enabled.store(false, Ordering::SeqCst);
            while ctx.audio_rx.try_recv().is_ok() {}
            enqueue_beep(&ctx.playback_queue, BEEP_CAPTURE_END_HZ, ctx.output_rate);
            wait_for_playback_drain(&ctx.playback_queue).await;
            while ctx.audio_rx.try_recv().is_ok() {}
            ctx.mic_enabled.store(true, Ordering::SeqCst);
        }
    }
}

async fn wait_for_playback_drain(queue: &Arc<std::sync::Mutex<VecDeque<i16>>>) {
    loop {
        let empty = queue.lock().map(|q| q.is_empty()).unwrap_or(true);
        if empty {
            tokio::time::sleep(std::time::Duration::from_millis(150)).await;
            return;
        }
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }
}

// ── Silero VAD setup ────────────────────────────────────────────────────

fn ensure_silero_model() -> Result<PathBuf> {
    let dest = download::cache_dir().join("silero_vad.onnx");
    download::ensure_single_file(SILERO_VAD_URL, &dest)
}

fn build_vad(
    model_path: &std::path::Path,
    sensitivity: &crate::config::SensitivityConfig,
) -> Result<VoiceActivityDetector> {
    let silero = SileroVadModelConfig {
        model: Some(model_path.to_string_lossy().into_owned()),
        threshold: sensitivity.vad_threshold,
        min_silence_duration: sensitivity.vad_min_silence_ms as f32 / 1000.0,
        min_speech_duration: sensitivity.vad_min_speech_ms as f32 / 1000.0,
        window_size: VAD_WINDOW_SAMPLES as i32,
        max_speech_duration: VAD_MAX_SPEECH_SECONDS,
    };
    let config = VadModelConfig {
        silero_vad: silero,
        ten_vad: Default::default(),
        sample_rate: PIPELINE_SAMPLE_RATE as i32,
        num_threads: 1,
        provider: Some("cpu".to_string()),
        debug: false,
    };
    VoiceActivityDetector::create(&config, VAD_MAX_SPEECH_SECONDS)
        .ok_or_else(|| anyhow!("failed to create sherpa-onnx VoiceActivityDetector"))
}

// ── Device enumeration / selection ─────────────────────────────────────

#[derive(Clone, Copy)]
enum DeviceKind {
    Input,
    Output,
}

/// Resolve a cpal device by exact name match, or fall back to the
/// host's default for the requested kind.
///
/// Enumerates the host's device list **once** and uses that single
/// snapshot for both the match attempt and the error report. ALSA's
/// device probing isn't idempotent (`plughw:*` and friends open the
/// underlying PCM during enumeration, which can fail transiently), so
/// enumerating twice — once to match, once to report — can produce
/// a list in the error that the matcher never actually saw.
fn pick_device(host: &cpal::Host, name: Option<&str>, kind: DeviceKind) -> Result<cpal::Device> {
    if let Some(want) = name {
        let candidates: Vec<cpal::Device> = match kind {
            DeviceKind::Input => host.input_devices()?.collect(),
            DeviceKind::Output => host.output_devices()?.collect(),
        };
        let mut seen: Vec<String> = Vec::with_capacity(candidates.len());
        for d in candidates {
            match d.description().map(|desc| desc.name().to_string()) {
                Ok(n) if n == want => return Ok(d),
                Ok(n) => seen.push(n),
                Err(_) => {}
            }
        }
        seen.sort();
        anyhow::bail!(
            "no {} device named '{}'. Available: {}",
            match kind {
                DeviceKind::Input => "input",
                DeviceKind::Output => "output",
            },
            want,
            seen.join(", ")
        );
    }
    match kind {
        DeviceKind::Input => host
            .default_input_device()
            .ok_or_else(|| anyhow!("no default input device")),
        DeviceKind::Output => host
            .default_output_device()
            .ok_or_else(|| anyhow!("no default output device")),
    }
}

/// `--list-devices` implementation. Dumps every input + output device
/// visible on the default cpal host, marking the system defaults so
/// the operator can decide which name to feed `--input-device` /
/// `--output-device`.
fn list_devices() -> Result<()> {
    let host = cpal::default_host();
    let default_in = host
        .default_input_device()
        .and_then(|d| d.description().ok().map(|desc| desc.name().to_string()));
    let default_out = host
        .default_output_device()
        .and_then(|d| d.description().ok().map(|desc| desc.name().to_string()));

    println!("cpal host: {}", host.id().name());
    println!();
    println!("input devices:");
    print_devices(host.input_devices()?, default_in.as_deref(), true);
    println!();
    println!("output devices:");
    print_devices(host.output_devices()?, default_out.as_deref(), false);
    Ok(())
}

fn print_devices<I: Iterator<Item = cpal::Device>>(
    devices: I,
    default_name: Option<&str>,
    is_input: bool,
) {
    let mut any = false;
    for d in devices {
        any = true;
        let name = d
            .description()
            .map(|desc| desc.name().to_string())
            .unwrap_or_else(|_| "<unnamed>".to_string());
        let marker = if Some(name.as_str()) == default_name {
            "*"
        } else {
            " "
        };
        let cfg = if is_input {
            d.default_input_config().ok()
        } else {
            d.default_output_config().ok()
        };
        match cfg {
            Some(c) => println!(
                "  {marker} {name}\n      {} Hz × {}ch ({:?})",
                c.sample_rate(),
                c.channels(),
                c.sample_format(),
            ),
            None => println!("  {marker} {name}\n      (config unavailable)"),
        }
    }
    if !any {
        println!("  (none)");
    }
    if default_name.is_some() {
        println!("\n  * = default");
    }
}

// ── Audio capture / playback (cpal) — unchanged ─────────────────────────

fn open_input_stream(
    name: Option<&str>,
    tx: mpsc::UnboundedSender<Vec<i16>>,
    enabled: Arc<AtomicBool>,
    mic_gain: f32,
    supervisor: Arc<StreamSupervisor>,
) -> Result<(cpal::Stream, u32, u16)> {
    let host = cpal::default_host();
    let device = pick_device(&host, name, DeviceKind::Input)?;
    let supported = device
        .default_input_config()
        .context("failed to query input config")?;
    let rate = supported.sample_rate();
    let channels = supported.channels();
    let format = supported.sample_format();
    let config: cpal::StreamConfig = supported.clone().into();

    // err_fn lives on cpal's worker thread. Funnel its events through
    // [`handle_stream_error`] so stderr only prints once per
    // STREAM_ERROR_PRINT_INTERVAL and the supervisor learns when to
    // rebuild. The closure must be Send + 'static — capture by Arc.
    let err_fn = {
        let supervisor = Arc::clone(&supervisor);
        move |e| handle_stream_error(&supervisor, "input stream error", e)
    };

    // Whether we need to multiply at all. Comparing against 1.0 exactly
    // is fine — the config layer hands us f32::from(1.0) when the user
    // omits the field, and any explicit override will differ.
    let gain = mic_gain;
    let stream = match format {
        SampleFormat::F32 => {
            let tx = tx.clone();
            let enabled = Arc::clone(&enabled);
            device.build_input_stream(
                &config,
                move |data: &[f32], _| {
                    if !enabled.load(Ordering::SeqCst) {
                        return;
                    }
                    let pcm: Vec<i16> = data
                        .iter()
                        .map(|f| {
                            let v = if gain == 1.0 { *f } else { *f * gain };
                            (v.clamp(-1.0, 1.0) * i16::MAX as f32) as i16
                        })
                        .collect();
                    let _ = tx.send(pcm);
                },
                err_fn,
                None,
            )?
        }
        SampleFormat::I16 => {
            let tx = tx.clone();
            let enabled = Arc::clone(&enabled);
            device.build_input_stream(
                &config,
                move |data: &[i16], _| {
                    if !enabled.load(Ordering::SeqCst) {
                        return;
                    }
                    let pcm: Vec<i16> = if gain == 1.0 {
                        data.to_vec()
                    } else {
                        data.iter().map(|s| apply_gain_i16(*s, gain)).collect()
                    };
                    let _ = tx.send(pcm);
                },
                err_fn,
                None,
            )?
        }
        SampleFormat::U16 => {
            let tx = tx.clone();
            let enabled = Arc::clone(&enabled);
            device.build_input_stream(
                &config,
                move |data: &[u16], _| {
                    if !enabled.load(Ordering::SeqCst) {
                        return;
                    }
                    let pcm: Vec<i16> = data
                        .iter()
                        .map(|s| {
                            let centered = (*s as i32 - 32768) as i16;
                            if gain == 1.0 {
                                centered
                            } else {
                                apply_gain_i16(centered, gain)
                            }
                        })
                        .collect();
                    let _ = tx.send(pcm);
                },
                err_fn,
                None,
            )?
        }
        other => anyhow::bail!("unsupported input sample format: {other:?}"),
    };
    Ok((stream, rate, channels))
}

/// Supervisor thread entry point: owns the [`cpal::Stream`] for its
/// lifetime, rebuilds it on err_fn-triggered restart requests, and
/// signals shutdown if rebuilding repeatedly fails (the device has
/// most likely been physically removed).
///
/// Runs on a dedicated `std::thread` rather than a tokio task because
/// `cpal::Stream` is `!Send` on most backends — we can't park it
/// across an `.await`.
///
/// Generic over the build closure so the same loop handles both
/// directions: the input flavour captures the audio mpsc sender, the
/// output flavour captures the playback queue Arc. When the supervisor
/// returns, the closure drops with it — that's how the input side
/// closes its mpsc channel (signalling `listen_loop` to exit).
///
/// `label` is the prefix used for stderr lines ("input stream", "output
/// stream"). `initial_ready` carries the first build's `(rate,
/// channels)` (or its error) back to `run()` so async startup can
/// finish.
fn supervise_stream<F>(
    label: &'static str,
    build: F,
    shutdown: Arc<AtomicBool>,
    supervisor: Arc<StreamSupervisor>,
    initial_ready: tokio::sync::oneshot::Sender<Result<(u32, u16)>>,
) where
    F: Fn() -> Result<(cpal::Stream, u32, u16)>,
{
    let mut first_build = true;
    let mut initial_ready = Some(initial_ready);
    let mut rebuild_backoff = STREAM_REBUILD_BACKOFF_INITIAL;
    let mut consecutive_failures: u32 = 0;

    'supervise: loop {
        if shutdown.load(Ordering::SeqCst) {
            break;
        }
        let (stream, rate, channels) = match build() {
            Ok(triple) => triple,
            Err(e) => {
                if let Some(tx) = initial_ready.take() {
                    let _ = tx.send(Err(e));
                    return;
                }
                consecutive_failures += 1;
                eprintln!(
                    "[{label} rebuild failed: {e:#}; attempt {consecutive_failures}/{STREAM_REBUILD_MAX_FAILURES}, retrying in {rebuild_backoff:?}]"
                );
                if consecutive_failures >= STREAM_REBUILD_MAX_FAILURES {
                    eprintln!(
                        "[{label} gave up after {STREAM_REBUILD_MAX_FAILURES} consecutive rebuild failures; shutting down satellite]"
                    );
                    shutdown.store(true, Ordering::SeqCst);
                    break 'supervise;
                }
                if shutdown_sleep(&shutdown, &supervisor, rebuild_backoff) {
                    break 'supervise;
                }
                rebuild_backoff = (rebuild_backoff * 2).min(STREAM_REBUILD_BACKOFF_MAX);
                continue;
            }
        };

        // A fresh stream is live — reset error/restart bookkeeping so
        // the next window starts clean.
        {
            let mut w = supervisor
                .error_window
                .lock()
                .unwrap_or_else(|p| p.into_inner());
            *w = ErrorWindow::default();
        }
        {
            let mut needs = supervisor
                .needs_restart
                .lock()
                .unwrap_or_else(|p| p.into_inner());
            *needs = false;
        }
        consecutive_failures = 0;
        rebuild_backoff = STREAM_REBUILD_BACKOFF_INITIAL;

        if first_build {
            first_build = false;
            if let Some(tx) = initial_ready.take() {
                let _ = tx.send(Ok((rate, channels)));
            }
        } else {
            eprintln!("[{label} rebuilt: {rate} Hz × {channels}ch]");
        }

        // Park until the stream needs to come down. Either err_fn
        // crossed the threshold (`needs_restart` flips true) or the
        // satellite is shutting down. `wait_timeout` is belt-and-
        // braces: notifications can be lost across thread boundaries
        // in theory, so wake periodically to re-check shutdown.
        let mut needs = supervisor
            .needs_restart
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        let exit_reason = loop {
            if shutdown.load(Ordering::SeqCst) {
                break SupervisorWake::Shutdown;
            }
            if *needs {
                break SupervisorWake::Restart;
            }
            let (g, _) = supervisor
                .cvar
                .wait_timeout(needs, Duration::from_secs(5))
                .unwrap_or_else(|p| p.into_inner());
            needs = g;
        };
        drop(needs);

        // Drop the stream before sleeping — the ALSA / CoreAudio layer
        // needs the handle released before a fresh open will succeed.
        drop(stream);

        match exit_reason {
            SupervisorWake::Shutdown => break,
            SupervisorWake::Restart => {
                eprintln!("[{label} restart requested; dropping and rebuilding...]");
                if shutdown_sleep(&shutdown, &supervisor, STREAM_REBUILD_BACKOFF_INITIAL) {
                    break;
                }
            }
        }
    }
}

enum SupervisorWake {
    Restart,
    Shutdown,
}

/// Sleep up to `dur`, returning `true` if shutdown was observed
/// (caller should exit) and `false` otherwise. Polls every 200 ms so a
/// late Ctrl-C doesn't have to wait out the full back-off.
fn shutdown_sleep(
    shutdown: &Arc<AtomicBool>,
    supervisor: &Arc<StreamSupervisor>,
    dur: Duration,
) -> bool {
    let deadline = Instant::now() + dur;
    while Instant::now() < deadline {
        if shutdown.load(Ordering::SeqCst) {
            return true;
        }
        // Also wake on the cvar in case Ctrl-C nudges us during the
        // back-off — saves up to 200 ms on the shutdown path.
        let lock = supervisor
            .needs_restart
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        let remaining = deadline.saturating_duration_since(Instant::now());
        let wait_for = remaining.min(Duration::from_millis(200));
        let _ = supervisor.cvar.wait_timeout(lock, wait_for);
    }
    shutdown.load(Ordering::SeqCst)
}

/// Multiply an i16 sample by `gain` in f32 space and clamp back into
/// the i16 range. Used by the mic_gain path for I16 / U16 capture so a
/// boost above unity can't wrap around.
fn apply_gain_i16(sample: i16, gain: f32) -> i16 {
    let scaled = sample as f32 * gain;
    scaled.clamp(i16::MIN as f32, i16::MAX as f32) as i16
}

fn open_output_stream(
    name: Option<&str>,
    queue: Arc<std::sync::Mutex<VecDeque<i16>>>,
    supervisor: Arc<StreamSupervisor>,
) -> Result<(cpal::Stream, u32, u16)> {
    let host = cpal::default_host();
    let device = pick_device(&host, name, DeviceKind::Output)?;
    let supported = device
        .default_output_config()
        .context("failed to query output config")?;
    let rate = supported.sample_rate();
    let channels = supported.channels();
    let format = supported.sample_format();
    let config: cpal::StreamConfig = supported.clone().into();

    // Same rate-limit / restart-request plumbing as the input side —
    // see [`handle_stream_error`].
    let err_fn = {
        let supervisor = Arc::clone(&supervisor);
        move |e| handle_stream_error(&supervisor, "output stream error", e)
    };

    let stream = match format {
        SampleFormat::F32 => {
            let queue = Arc::clone(&queue);
            device.build_output_stream(
                &config,
                move |data: &mut [f32], _| {
                    let mut q = match queue.lock() {
                        Ok(g) => g,
                        Err(_) => {
                            for s in data.iter_mut() {
                                *s = 0.0;
                            }
                            return;
                        }
                    };
                    for frame in data.chunks_mut(channels as usize) {
                        let sample = q.pop_front().unwrap_or(0);
                        let f = sample as f32 / i16::MAX as f32;
                        for slot in frame.iter_mut() {
                            *slot = f;
                        }
                    }
                },
                err_fn,
                None,
            )?
        }
        SampleFormat::I16 => {
            let queue = Arc::clone(&queue);
            device.build_output_stream(
                &config,
                move |data: &mut [i16], _| {
                    let mut q = match queue.lock() {
                        Ok(g) => g,
                        Err(_) => {
                            for s in data.iter_mut() {
                                *s = 0;
                            }
                            return;
                        }
                    };
                    for frame in data.chunks_mut(channels as usize) {
                        let sample = q.pop_front().unwrap_or(0);
                        for slot in frame.iter_mut() {
                            *slot = sample;
                        }
                    }
                },
                err_fn,
                None,
            )?
        }
        SampleFormat::U16 => {
            let queue = Arc::clone(&queue);
            device.build_output_stream(
                &config,
                move |data: &mut [u16], _| {
                    let mut q = match queue.lock() {
                        Ok(g) => g,
                        Err(_) => {
                            for s in data.iter_mut() {
                                *s = 32768;
                            }
                            return;
                        }
                    };
                    for frame in data.chunks_mut(channels as usize) {
                        let sample = q.pop_front().unwrap_or(0);
                        let u = (sample as i32 + 32768) as u16;
                        for slot in frame.iter_mut() {
                            *slot = u;
                        }
                    }
                },
                err_fn,
                None,
            )?
        }
        other => anyhow::bail!("unsupported output sample format: {other:?}"),
    };
    Ok((stream, rate, channels))
}

// ── Per-utterance flow ──────────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
async fn process_utterance(
    client: &reqwest::Client,
    base: &str,
    device_id: &str,
    room_profile: &str,
    pcm_16khz: &[i16],
    language: Option<&str>,
    device_meta: Option<&sapphire_agent_api::DeviceMetadata>,
    playback_queue: Arc<std::sync::Mutex<VecDeque<i16>>>,
    output_rate: u32,
) -> Result<()> {
    // Note: the playback queue is intentionally NOT cleared here. The
    // listen loop wait_for_playback_drain'd the previous reply before
    // calling us, and a capture-end beep may have been enqueued in the
    // meantime — it must survive until the cpal callback consumes it.

    let (event_tx, mut event_rx) = mpsc::channel::<VoiceEvent>(64);
    let server_call = tokio::spawn({
        let client = client.clone();
        let base = base.to_string();
        let device = device_id.to_string();
        let room = room_profile.to_string();
        let pcm = pcm_16khz.to_vec();
        let lang = language.map(String::from);
        let device_meta = device_meta.cloned();
        async move {
            voice_pipeline_run(
                &client,
                &base,
                &device,
                &room,
                &pcm,
                lang.as_deref(),
                device_meta.as_ref(),
                event_tx,
            )
            .await
        }
    });

    while let Some(evt) = event_rx.recv().await {
        match evt {
            VoiceEvent::StageStart { stage } => {
                eprint!("[{stage}…] ");
            }
            VoiceEvent::StageEnd { stage } => {
                let _ = stage;
            }
            VoiceEvent::SttFinal { text } => {
                eprintln!("> {text}");
            }
            VoiceEvent::AssistantText { text } => {
                eprintln!("{text}");
            }
            VoiceEvent::AudioChunk { pcm } => {
                let upsampled = resample_to(&pcm, PIPELINE_SAMPLE_RATE, output_rate);
                if let Ok(mut q) = playback_queue.lock() {
                    q.extend(upsampled);
                }
            }
            VoiceEvent::ToolStart { name } => {
                eprint!("[tool: {name}] ");
            }
            VoiceEvent::ToolEnd { name } => {
                let _ = name;
            }
            VoiceEvent::Done { .. } => break,
            VoiceEvent::Error { message } => {
                eprintln!("[error: {message}]");
                break;
            }
        }
    }
    let _ = server_call.await;
    Ok(())
}

// ── Beep helpers ────────────────────────────────────────────────────────

/// Synthesise a short sine-wave beep at the pipeline rate. Amplitude
/// is intentionally modest (0.4 of full-scale) — confirmation tones
/// shouldn't blow ears even on a turned-up speakerphone — and the
/// envelope tapers in/out over 8 ms to avoid the click an abrupt
/// square-edge waveform would produce.
fn generate_beep(freq_hz: f32, duration_ms: u32) -> Vec<i16> {
    let sr = PIPELINE_SAMPLE_RATE as f32;
    let total = (sr * duration_ms as f32 / 1000.0) as usize;
    let fade = ((sr * 0.008) as usize).min(total / 2);
    let amp = 0.4;
    let mut out = Vec::with_capacity(total);
    for i in 0..total {
        let t = i as f32 / sr;
        let env = if i < fade {
            i as f32 / fade as f32
        } else if i >= total - fade {
            (total - i) as f32 / fade as f32
        } else {
            1.0
        };
        let v = (2.0 * std::f32::consts::PI * freq_hz * t).sin() * amp * env;
        out.push((v * i16::MAX as f32) as i16);
    }
    out
}

/// Generate a beep at the pipeline rate and append it to the playback
/// queue, resampling to the output rate so the cpal callback can
/// consume it without a per-sample rate convert.
fn enqueue_beep(queue: &Arc<std::sync::Mutex<VecDeque<i16>>>, freq_hz: f32, output_rate: u32) {
    let pcm = generate_beep(freq_hz, BEEP_DURATION_MS);
    let upsampled = resample_to(&pcm, PIPELINE_SAMPLE_RATE, output_rate);
    if let Ok(mut q) = queue.lock() {
        q.extend(upsampled);
    }
}

// ── Format helpers ──────────────────────────────────────────────────────

fn to_mono(samples: &[i16], channels: u16) -> Vec<i16> {
    if channels <= 1 {
        return samples.to_vec();
    }
    samples
        .chunks(channels as usize)
        .map(|frame| {
            let sum: i32 = frame.iter().map(|s| *s as i32).sum();
            (sum / channels as i32) as i16
        })
        .collect()
}

/// Linear interpolation resample. Quality is fine for speech.
fn resample_to(input: &[i16], src_rate: u32, dst_rate: u32) -> Vec<i16> {
    if input.is_empty() || src_rate == dst_rate {
        return input.to_vec();
    }
    let ratio = src_rate as f64 / dst_rate as f64;
    let out_len = ((input.len() as f64) / ratio).round() as usize;
    let mut out = Vec::with_capacity(out_len);
    for i in 0..out_len {
        let src_pos = i as f64 * ratio;
        let idx = src_pos.floor() as usize;
        let frac = src_pos - idx as f64;
        let s0 = input[idx.min(input.len() - 1)] as f64;
        let s1 = input[(idx + 1).min(input.len() - 1)] as f64;
        let v = s0 * (1.0 - frac) + s1 * frac;
        out.push(v.round().clamp(i16::MIN as f64, i16::MAX as f64) as i16);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn to_mono_averages_stereo() {
        let frames: Vec<i16> = vec![100, 200, 300, 400];
        let mono = to_mono(&frames, 2);
        assert_eq!(mono, vec![150, 350]);
    }

    #[test]
    fn to_mono_passes_through_mono() {
        let frames: Vec<i16> = vec![1, 2, 3];
        assert_eq!(to_mono(&frames, 1), frames);
    }

    #[test]
    fn resample_no_op_on_equal_rates() {
        let frames: Vec<i16> = vec![1, 2, 3];
        assert_eq!(resample_to(&frames, 16000, 16000), frames);
    }

    #[test]
    fn beep_length_matches_duration() {
        let pcm = generate_beep(880.0, 150);
        // 150 ms @ 16 kHz = 2400 samples.
        assert_eq!(pcm.len(), (PIPELINE_SAMPLE_RATE as usize) * 150 / 1000);
    }

    #[test]
    fn beep_envelope_starts_and_ends_silent() {
        let pcm = generate_beep(880.0, 150);
        // First and last samples should be near zero — the linear
        // fade-in/out reaches the edge of the envelope. Tolerance
        // accounts for the final sample landing 1/fade short of 0.
        let edge_limit = i16::MAX / 200;
        assert!(
            pcm.first().copied().unwrap_or(0).abs() < edge_limit,
            "first sample too loud: {:?}",
            pcm.first()
        );
        assert!(
            pcm.last().copied().unwrap_or(0).abs() < edge_limit,
            "last sample too loud: {:?}",
            pcm.last()
        );
        // Plateau region should swing through real amplitude. Look at
        // peak magnitude over the central third instead of one sample
        // (zero-crossings of the sine land on individual samples at
        // certain frequency/rate combos).
        let lo = pcm.len() / 3;
        let hi = pcm.len() * 2 / 3;
        let peak = pcm[lo..hi].iter().map(|s| s.abs()).max().unwrap_or(0);
        assert!(peak > i16::MAX / 4, "plateau peak too quiet: {peak}");
    }

    #[test]
    fn enqueue_beep_appends_to_queue() {
        let q = Arc::new(std::sync::Mutex::new(VecDeque::new()));
        enqueue_beep(&q, 660.0, PIPELINE_SAMPLE_RATE);
        let len = q.lock().unwrap().len();
        // Equal sample rates → no resample → exact pipeline-rate length.
        assert_eq!(len, generate_beep(660.0, BEEP_DURATION_MS).len());
    }

    #[test]
    fn apply_gain_i16_unity_is_identity() {
        // Spot-check the helper directly so the input-stream callback
        // logic is exercised without spinning up cpal.
        for s in [-32768i16, -1, 0, 1, 32767] {
            assert_eq!(apply_gain_i16(s, 1.0), s);
        }
    }

    #[test]
    fn apply_gain_i16_boost_clamps_to_full_scale() {
        // 2× boost on a near-peak sample must clamp at i16::MAX,
        // not wrap. This is what protects against wrap-around when
        // the operator dials gain above what the source can support.
        assert_eq!(apply_gain_i16(20_000, 2.0), i16::MAX);
        assert_eq!(apply_gain_i16(-20_000, 2.0), i16::MIN);
    }

    #[test]
    fn apply_gain_i16_attenuation() {
        // 0.5× on full-scale should land at half-scale (rounded toward
        // zero by the as-cast).
        assert_eq!(apply_gain_i16(20_000, 0.5), 10_000);
    }

    // ── StreamSupervisor error-window logic ─────────────────────────────
    //
    // `record_stream_error` is pure: it only touches the window passed
    // in. Test it directly so we can exercise the threshold / sliding-
    // window / rate-limit branches without instantiating cpal or
    // spawning threads. Same code path is used by both the input and
    // output supervisors — covered once.

    #[test]
    fn record_stream_error_counts_up_to_threshold() {
        let mut w = ErrorWindow::default();
        let t0 = Instant::now();
        for i in 1..STREAM_ERROR_THRESHOLD {
            let d = record_stream_error(&mut w, t0 + Duration::from_millis(i as u64 * 10));
            assert_eq!(d.count, i);
            assert!(!d.request_restart, "shouldn't fire before threshold");
        }
        // The N-th error fires the restart request exactly once.
        let d = record_stream_error(
            &mut w,
            t0 + Duration::from_millis(STREAM_ERROR_THRESHOLD as u64 * 10),
        );
        assert_eq!(d.count, STREAM_ERROR_THRESHOLD);
        assert!(d.request_restart);

        // Errors *past* the threshold keep counting but don't re-fire
        // the restart request — the supervisor handles each window
        // once.
        let d = record_stream_error(
            &mut w,
            t0 + Duration::from_millis((STREAM_ERROR_THRESHOLD as u64 + 1) * 10),
        );
        assert_eq!(d.count, STREAM_ERROR_THRESHOLD + 1);
        assert!(!d.request_restart);
    }

    #[test]
    fn record_stream_error_rolls_window_after_quiet_period() {
        // A long-running session that's seen 3 xruns over an hour
        // shouldn't restart the stream on the next xrun. After the
        // window elapses, the counter resets.
        let mut w = ErrorWindow::default();
        let t0 = Instant::now();
        record_stream_error(&mut w, t0);
        record_stream_error(&mut w, t0 + Duration::from_millis(100));
        // Long quiet period: well past STREAM_ERROR_WINDOW.
        let later = t0 + STREAM_ERROR_WINDOW + Duration::from_secs(60);
        let d = record_stream_error(&mut w, later);
        assert_eq!(d.count, 1, "window should have rolled and reset to 1");
        assert!(!d.request_restart);
    }

    #[test]
    fn record_stream_error_rate_limits_prints() {
        // First call prints; immediately following calls are
        // suppressed until STREAM_ERROR_PRINT_INTERVAL elapses.
        let mut w = ErrorWindow::default();
        let t0 = Instant::now();
        assert!(record_stream_error(&mut w, t0).print);
        assert!(
            !record_stream_error(&mut w, t0 + Duration::from_millis(50)).print,
            "50 ms later should be suppressed"
        );
        assert!(
            !record_stream_error(&mut w, t0 + Duration::from_millis(500)).print,
            "500 ms later should still be suppressed"
        );
        assert!(
            record_stream_error(
                &mut w,
                t0 + STREAM_ERROR_PRINT_INTERVAL + Duration::from_millis(1)
            )
            .print,
            "past the interval should print again"
        );
    }
}
