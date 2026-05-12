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
mod wake;

use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{Context, Result, anyhow};
use cpal::SampleFormat;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use sapphire_agent_api::{VoiceEvent, voice::PIPELINE_SAMPLE_RATE, voice_pipeline_run};
use sherpa_onnx::{
    SileroVadModelConfig, VadModelConfig, VoiceActivityDetector,
};
use tokio::sync::mpsc;

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
/// Wake-word configuration intentionally lives server-side under the
/// session's `[room_profile.*]`. The satellite fetches it via the
/// `voice/config` MCP method right after `initialize`. CLI overrides
/// stay available for one-off testing.
pub struct VoiceOptions {
    pub language: Option<String>,
    /// Override the server-supplied KWS bundle name. When neither
    /// this nor the server's `wake_word_model` is set, the satellite
    /// runs in VAD-only mode.
    pub wake_word_model: Option<String>,
    /// Override the server-supplied wake phrase. Same fallback as
    /// `wake_word_model` — empty here AND empty server-side means no
    /// wake-word gating.
    pub wake_word: Option<String>,
    /// Advanced escape hatch: skip phrase tokenisation and feed a
    /// pre-tokenised sherpa-onnx keywords file straight to the KWS.
    /// When set, `wake_word` (CLI or server-supplied) is ignored.
    pub keywords_file: Option<String>,
    /// When true, enumerate cpal devices and exit before opening any
    /// stream or talking to the agent.
    pub list_devices: bool,
    /// Pin capture to the cpal device with this exact name; None ⇒
    /// `default_input_device()`.
    pub input_device: Option<String>,
    /// Pin playback to the cpal device with this exact name; None ⇒
    /// `default_output_device()`.
    pub output_device: Option<String>,
}

/// Entry point for `sapphire-call voice`.
pub async fn run(
    server: String,
    session: Option<String>,
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

    let (mcp_session_id, display_id, is_new) =
        sapphire_agent_api::initialize(&client, &base, session, room_profile.as_deref())
            .await
            .context("failed to initialize MCP session")?;
    eprintln!(
        "sapphire-call voice (session: {display_id}{})",
        if is_new { ", new" } else { ", resumed" }
    );

    // ── Wake-word config: fetch from server, let CLI flags override ─────
    let server_wake = sapphire_agent_api::voice_config(&client, &base, &mcp_session_id)
        .await
        .unwrap_or_else(|e| {
            eprintln!(
                "warning: voice/config fetch failed ({e:#}); proceeding without server-side wake config"
            );
            sapphire_agent_api::WakeWordConfig::default()
        });
    let wake_model = options
        .wake_word_model
        .clone()
        .or(server_wake.model);
    let wake_phrase = options.wake_word.clone().or(server_wake.phrase);

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
    let wake_detector = if let Some(bundle) = wake_model.as_deref() {
        let bundle_owned = bundle.to_string();
        let keywords_file = options.keywords_file.clone();
        let phrase = wake_phrase.clone();
        let detector = tokio::task::spawn_blocking(move || {
            let source = if let Some(path) = keywords_file.as_deref() {
                wake::WakeKeywordSource::File(std::path::Path::new(path))
            } else if let Some(p) = phrase.as_deref() {
                // Single phrase from CLI/server; pass as 1-element slice.
                let phrases = vec![p.to_string()];
                return wake::WakeDetector::create(
                    &bundle_owned,
                    wake::WakeKeywordSource::Phrases(&phrases),
                );
            } else {
                wake::WakeKeywordSource::BundleDefault
            };
            wake::WakeDetector::create(&bundle_owned, source)
        })
        .await
        .map_err(|e| anyhow!("wake-word init task panicked: {e}"))?
        .context("failed to initialise wake-word detector")?;
        let trigger_hint = wake_phrase
            .as_deref()
            .map(|p| format!(" — say \"{p}\" to talk"))
            .unwrap_or_default();
        eprintln!("wake-word: {}{trigger_hint}", bundle);
        Some(detector)
    } else {
        None
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

    let awaiting_wake = wake_detector.is_some();
    listen_loop(ListenCtx {
        audio_rx,
        input_rate,
        input_channels,
        mic_enabled,
        playback_queue,
        output_rate,
        client,
        base,
        mcp_session_id,
        language: options.language,
        shutdown,
        vad,
        wake: wake_detector,
        awaiting_wake,
    })
    .await?;

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
    mcp_session_id: String,
    language: Option<String>,
    shutdown: Arc<AtomicBool>,
    vad: VoiceActivityDetector,
    /// `Some` enables wake-word mode; the satellite gates VAD behind
    /// successful keyword spotting on this detector.
    wake: Option<wake::WakeDetector>,
    /// In wake-word mode: true while we're waiting for the next wake
    /// trigger, false once it fires and we're capturing the utterance.
    /// Always false in VAD-only mode.
    awaiting_wake: bool,
}

async fn listen_loop(mut ctx: ListenCtx) -> Result<()> {
    let mut window_buf: Vec<f32> = Vec::with_capacity(VAD_WINDOW_SAMPLES);
    while !ctx.shutdown.load(Ordering::SeqCst) {
        let Some(raw) = ctx.audio_rx.recv().await else {
            break;
        };
        if !ctx.mic_enabled.load(Ordering::SeqCst) {
            continue;
        }
        // Convert to mono + resample to the pipeline rate, then to f32.
        let mono = to_mono(&raw, ctx.input_channels);
        let pcm16k = resample_to(&mono, ctx.input_rate, PIPELINE_SAMPLE_RATE);
        let pcm_f32: Vec<f32> = pcm16k.iter().map(|s| *s as f32 / 32768.0).collect();

        // Wake-word gate: while we're awaiting a wake trigger, feed
        // audio to KWS only — VAD doesn't see the stream until the
        // wake phrase has fired.
        if ctx.awaiting_wake {
            if let Some(ref mut wake) = ctx.wake {
                if let Some(keyword) = wake.feed(&pcm_f32)? {
                    eprintln!("[wake: {keyword}] now listening for command...");
                    ctx.awaiting_wake = false;
                    // Clear any audio left over from before the wake so
                    // the VAD starts on the post-wake utterance.
                    ctx.vad.reset();
                    window_buf.clear();
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

            eprintln!(
                "→ uploading {} samples ({:.2}s)",
                utterance.len(),
                utterance.len() as f32 / PIPELINE_SAMPLE_RATE as f32
            );
            if let Err(e) = process_utterance(
                &ctx.client,
                &ctx.base,
                &ctx.mcp_session_id,
                &utterance,
                ctx.language.as_deref(),
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
            // In wake-word mode, return to wake-listening for the next
            // command. In VAD-only mode, stay in capture mode.
            if let Some(ref mut wake) = ctx.wake {
                wake.reset();
                ctx.awaiting_wake = true;
                eprintln!("Waiting for wake word.");
            } else {
                eprintln!("Listening.");
            }
            ctx.mic_enabled.store(true, Ordering::SeqCst);
        }
    }
    Ok(())
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
        for d in candidates {
            if d.name().ok().as_deref() == Some(want) {
                return Ok(d);
            }
        }
        // Not found — list what we did see so the user can re-issue.
        let mut seen: Vec<String> = match kind {
            DeviceKind::Input => host.input_devices()?,
            DeviceKind::Output => host.output_devices()?,
        }
        .filter_map(|d| d.name().ok())
        .collect();
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

async fn process_utterance(
    client: &reqwest::Client,
    base: &str,
    mcp_session_id: &str,
    pcm_16khz: &[i16],
    language: Option<&str>,
    playback_queue: Arc<std::sync::Mutex<VecDeque<i16>>>,
    output_rate: u32,
) -> Result<()> {
    if let Ok(mut q) = playback_queue.lock() {
        q.clear();
    }

    let (event_tx, mut event_rx) = mpsc::channel::<VoiceEvent>(64);
    let server_call = tokio::spawn({
        let client = client.clone();
        let base = base.to_string();
        let sid = mcp_session_id.to_string();
        let pcm = pcm_16khz.to_vec();
        let lang = language.map(String::from);
        async move {
            voice_pipeline_run(&client, &base, &sid, &pcm, lang.as_deref(), event_tx).await
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
}
