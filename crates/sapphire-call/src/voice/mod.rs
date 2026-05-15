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
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow};
use cpal::SampleFormat;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use sapphire_agent_api::{
    VoiceEvent, VoicePushEvent, voice::PIPELINE_SAMPLE_RATE, voice_pipeline_run, voice_subscribe,
};
use sherpa_onnx::{
    SileroVadModelConfig, VadModelConfig, VoiceActivityDetector,
};
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
}

/// Frequency / duration of the confirmation beeps. Picked to be
/// distinct (rising vs falling) and short enough not to clip the start
/// of a quick command.
const BEEP_DURATION_MS: u32 = 150;
const BEEP_WAKE_HZ: f32 = 880.0;
const BEEP_CAPTURE_END_HZ: f32 = 660.0;

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
    eprintln!(
        "sapphire-call voice (device: {device_id}, room_profile: {room_profile})",
    );

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
        move || build_vad(&path)
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
            let detector = tokio::task::spawn_blocking({
                let label = label.clone();
                move || {
                    let (mel, embed) = download::ensure_oww_frontend()
                        .context("failed to fetch openWakeWord frontend models")?;
                    let wake_path = download::cache_inline_oww(&bytes, &sha256)
                        .context("failed to cache openWakeWord classifier")?;
                    oww::OpenWakeWordDetector::create(&mel, &embed, &wake_path, label)
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
    let (audio_tx, audio_rx) = mpsc::unbounded_channel::<Vec<i16>>();
    let mic_enabled = Arc::new(AtomicBool::new(true));
    let (input_stream, input_rate, input_channels) = open_input_stream(
        options.input_device.as_deref(),
        audio_tx,
        Arc::clone(&mic_enabled),
    )?;
    input_stream.play()?;

    let playback_queue: Arc<std::sync::Mutex<VecDeque<i16>>> =
        Arc::new(std::sync::Mutex::new(VecDeque::new()));
    let (output_stream, output_rate, _output_channels) = open_output_stream(
        options.output_device.as_deref(),
        Arc::clone(&playback_queue),
    )?;
    output_stream.play()?;

    eprintln!(
        "input: {input_rate} Hz × {input_channels}ch  output: {output_rate} Hz",
    );
    eprintln!("Listening. Ctrl-C to quit.");

    // ── Ctrl-C handler ──────────────────────────────────────────────────
    let shutdown = Arc::new(AtomicBool::new(false));
    {
        let shutdown = Arc::clone(&shutdown);
        tokio::spawn(async move {
            if tokio::signal::ctrl_c().await.is_ok() {
                eprintln!("\nShutting down...");
                shutdown.store(true, Ordering::SeqCst);
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
        shutdown,
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
                    expire_follow_up(&mut ctx, &mut window_buf);
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
                            expire_follow_up(&mut ctx, &mut window_buf);
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
            if let Some(ref mut wake) = ctx.wake {
                if let Some(keyword) = wake.feed(&pcm16k)? {
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
            match (ctx.wake.as_mut(), ctx.behavior.follow_up_listen_seconds) {
                (Some(wake), 0) => {
                    wake.reset();
                    ctx.awaiting_wake = true;
                    eprintln!("Waiting for wake word.");
                }
                (Some(_), secs) => {
                    ctx.awaiting_wake = false;
                    ctx.follow_up_until =
                        Some(Instant::now() + Duration::from_secs(secs as u64));
                    eprintln!("Listening for follow-up... ({secs}s)");
                }
                (None, _) => {
                    eprintln!("Listening.");
                }
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
async fn handle_push_command(
    ctx: &mut ListenCtx,
    cmd: ListenCommand,
    window_buf: &mut Vec<f32>,
) {
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
            match (ctx.wake.as_mut(), ctx.behavior.follow_up_listen_seconds) {
                (Some(wake), 0) => {
                    wake.reset();
                    ctx.awaiting_wake = true;
                    ctx.follow_up_until = None;
                    eprintln!("Push complete. Waiting for wake word.");
                }
                (Some(_), secs) => {
                    ctx.awaiting_wake = false;
                    ctx.follow_up_until =
                        Some(Instant::now() + Duration::from_secs(secs as u64));
                    eprintln!("Push complete. Listening for follow-up... ({secs}s)");
                }
                (None, _) => {
                    eprintln!("Push complete. Listening.");
                }
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
        let conn = voice_subscribe(
            &client,
            &base,
            &device_id,
            &room_profile,
            push_tx,
        );
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
                                task.as_deref().map(|t| format!(": {t}")).unwrap_or_default()
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
fn expire_follow_up(ctx: &mut ListenCtx, window_buf: &mut Vec<f32>) {
    ctx.follow_up_until = None;
    ctx.vad.reset();
    window_buf.clear();
    if let Some(ref mut wake) = ctx.wake {
        wake.reset();
        ctx.awaiting_wake = true;
        eprintln!("Follow-up window elapsed. Waiting for wake word.");
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

fn build_vad(model_path: &std::path::Path) -> Result<VoiceActivityDetector> {
    let silero = SileroVadModelConfig {
        model: Some(model_path.to_string_lossy().into_owned()),
        threshold: 0.5,
        min_silence_duration: 0.25,
        min_speech_duration: 0.25,
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
fn pick_device(
    host: &cpal::Host,
    name: Option<&str>,
    kind: DeviceKind,
) -> Result<cpal::Device> {
    if let Some(want) = name {
        let candidates: Vec<cpal::Device> = match kind {
            DeviceKind::Input => host.input_devices()?.collect(),
            DeviceKind::Output => host.output_devices()?.collect(),
        };
        let mut seen: Vec<String> = Vec::with_capacity(candidates.len());
        for d in candidates {
            match d.name() {
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
        .and_then(|d| d.name().ok());
    let default_out = host
        .default_output_device()
        .and_then(|d| d.name().ok());

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
        let name = d.name().unwrap_or_else(|_| "<unnamed>".to_string());
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
                c.sample_rate().0,
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
) -> Result<(cpal::Stream, u32, u16)> {
    let host = cpal::default_host();
    let device = pick_device(&host, name, DeviceKind::Input)?;
    if let Some(n) = name {
        eprintln!("input device: {n}");
    }
    let supported = device
        .default_input_config()
        .context("failed to query input config")?;
    let rate = supported.sample_rate().0;
    let channels = supported.channels();
    let format = supported.sample_format();
    let config: cpal::StreamConfig = supported.clone().into();

    let err_fn = |e| eprintln!("[input stream error: {e}]");

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
                        .map(|f| (f.clamp(-1.0, 1.0) * i16::MAX as f32) as i16)
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
                    let _ = tx.send(data.to_vec());
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
                    let pcm: Vec<i16> =
                        data.iter().map(|s| (*s as i32 - 32768) as i16).collect();
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

fn open_output_stream(
    name: Option<&str>,
    queue: Arc<std::sync::Mutex<VecDeque<i16>>>,
) -> Result<(cpal::Stream, u32, u16)> {
    let host = cpal::default_host();
    let device = pick_device(&host, name, DeviceKind::Output)?;
    if let Some(n) = name {
        eprintln!("output device: {n}");
    }
    let supported = device
        .default_output_config()
        .context("failed to query output config")?;
    let rate = supported.sample_rate().0;
    let channels = supported.channels();
    let format = supported.sample_format();
    let config: cpal::StreamConfig = supported.clone().into();

    let err_fn = |e| eprintln!("[output stream error: {e}]");

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
fn enqueue_beep(
    queue: &Arc<std::sync::Mutex<VecDeque<i16>>>,
    freq_hz: f32,
    output_rate: u32,
) {
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
}
