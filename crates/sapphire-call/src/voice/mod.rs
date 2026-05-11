//! Voice satellite subcommand: always-on listening with Silero VAD.
//!
//! v1 has no wake word and no push-to-talk — the mic is hot for the
//! lifetime of the process. Sherpa-onnx's Silero VAD chunks the
//! incoming stream into utterances and each completed segment is
//! shipped to the agent via `voice/pipeline_run`. The mic is gated
//! off while the reply plays back so the satellite doesn't transcribe
//! its own TTS. Wake-word support arrives in a follow-up.

mod download;

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

/// Entry point for `sapphire-call voice`.
pub async fn run(
    server: String,
    session: Option<String>,
    room_profile: Option<String>,
    language: Option<String>,
) -> Result<()> {
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

    // ── VAD model ────────────────────────────────────────────────────────
    let vad_model_path = ensure_silero_model()
        .context("failed to fetch Silero VAD model")?;
    eprintln!("VAD model: {}", vad_model_path.display());
    let vad = build_vad(&vad_model_path)?;

    // ── Audio I/O ────────────────────────────────────────────────────────
    let (audio_tx, audio_rx) = mpsc::unbounded_channel::<Vec<i16>>();
    let mic_enabled = Arc::new(AtomicBool::new(true));
    let (input_stream, input_rate, input_channels) =
        open_input_stream(audio_tx, Arc::clone(&mic_enabled))?;
    input_stream.play()?;

    let playback_queue: Arc<std::sync::Mutex<VecDeque<i16>>> =
        Arc::new(std::sync::Mutex::new(VecDeque::new()));
    let (output_stream, output_rate, _output_channels) =
        open_output_stream(Arc::clone(&playback_queue))?;
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
        language,
        shutdown,
        vad,
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
        // Convert to mono + resample to the pipeline rate, then to f32
        // for sherpa-onnx.
        let mono = to_mono(&raw, ctx.input_channels);
        let pcm16k = resample_to(&mono, ctx.input_rate, PIPELINE_SAMPLE_RATE);
        for s in &pcm16k {
            window_buf.push(*s as f32 / 32768.0);
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
            ctx.mic_enabled.store(true, Ordering::SeqCst);
            eprintln!("Listening.");
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

// ── Audio capture / playback (cpal) — unchanged ─────────────────────────

fn open_input_stream(
    tx: mpsc::UnboundedSender<Vec<i16>>,
    enabled: Arc<AtomicBool>,
) -> Result<(cpal::Stream, u32, u16)> {
    let host = cpal::default_host();
    let device = host
        .default_input_device()
        .ok_or_else(|| anyhow!("no default input device"))?;
    let supported = device
        .default_input_config()
        .context("failed to query default input config")?;
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
    queue: Arc<std::sync::Mutex<VecDeque<i16>>>,
) -> Result<(cpal::Stream, u32, u16)> {
    let host = cpal::default_host();
    let device = host
        .default_output_device()
        .ok_or_else(|| anyhow!("no default output device"))?;
    let supported = device
        .default_output_config()
        .context("failed to query default output config")?;
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
