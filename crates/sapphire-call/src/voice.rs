//! Voice satellite subcommand: always-on listening with energy VAD.
//!
//! v1 has no wake word and no push-to-talk — the mic is hot for the
//! lifetime of the process. An energy-based VAD chunks the incoming
//! stream into utterances (200 ms of speech to confirm onset; 600 ms
//! of silence to confirm end; 30 s hard cap), and each completed
//! utterance is shipped to the agent via `voice/pipeline_run`. The
//! mic is gated off while the reply plays back so the agent doesn't
//! transcribe its own TTS.
//!
//! Wake word support arrives in a follow-up; until then the energy
//! threshold is a best-effort filter — quiet rooms work well, noisy
//! environments may chatter spuriously.

use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{Context, Result, anyhow};
use cpal::SampleFormat;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use sapphire_agent_api::{VoiceEvent, voice::PIPELINE_SAMPLE_RATE, voice_pipeline_run};
use tokio::sync::mpsc;

/// 20 ms frame at the pipeline rate (16 kHz mono).
const VAD_FRAME_SAMPLES: usize = (PIPELINE_SAMPLE_RATE as usize) / 50;
/// Minimum sustained speech (ms) before VAD declares an utterance open.
const VAD_MIN_SPEECH_MS: u32 = 200;
/// Trailing silence (ms) that closes an open utterance.
const VAD_END_SILENCE_MS: u32 = 600;
/// Hard cap on a single utterance, matches the server's default
/// `capture_max_ms`.
const VAD_MAX_UTTERANCE_MS: u32 = 30_000;
/// RMS amplitude threshold (i16 scale, ~ -36 dBFS) above which a frame
/// is considered speech. Quiet rooms register ~50–200, normal speech
/// 1000+. Tune later if needed.
const VAD_ENERGY_THRESHOLD: f64 = 500.0;

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

    // ── Audio I/O ────────────────────────────────────────────────────────
    let (audio_tx, audio_rx) = mpsc::unbounded_channel::<Vec<i16>>();
    let mic_enabled = Arc::new(AtomicBool::new(true));
    let (input_stream, input_rate, input_channels) =
        open_input_stream(audio_tx, Arc::clone(&mic_enabled))?;
    input_stream.play()?;

    let playback_queue: Arc<std::sync::Mutex<VecDeque<i16>>> =
        Arc::new(std::sync::Mutex::new(VecDeque::new()));
    let (output_stream, output_rate, output_channels) =
        open_output_stream(Arc::clone(&playback_queue))?;
    output_stream.play()?;

    eprintln!(
        "input: {input_rate} Hz × {input_channels}ch  output: {output_rate} Hz × {output_channels}ch",
    );
    eprintln!("Listening. Ctrl-C to quit.");

    // ── Spawn ctrl-c handler (no terminal raw mode this time) ───────────
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

    listen_loop(
        ListenCtx {
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
        },
    )
    .await?;

    // cpal streams drop here, closing the audio devices.
    let _ = output_channels;
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
}

async fn listen_loop(mut ctx: ListenCtx) -> Result<()> {
    let mut vad = EnergyVad::new(
        VAD_ENERGY_THRESHOLD,
        VAD_MIN_SPEECH_MS,
        VAD_END_SILENCE_MS,
        VAD_MAX_UTTERANCE_MS,
    );

    while !ctx.shutdown.load(Ordering::SeqCst) {
        let Some(raw) = ctx.audio_rx.recv().await else {
            break;
        };
        if !ctx.mic_enabled.load(Ordering::SeqCst) {
            continue;
        }
        // Convert to mono + resample to the pipeline rate before VAD.
        let mono = to_mono(&raw, ctx.input_channels);
        let pcm16k = resample_to(&mono, ctx.input_rate, PIPELINE_SAMPLE_RATE);
        for utterance in vad.feed(&pcm16k) {
            // Gate the mic so the agent's TTS reply doesn't feed back
            // into the VAD as a new utterance.
            ctx.mic_enabled.store(false, Ordering::SeqCst);
            // Drop any buffered audio from the suppressed window.
            while ctx.audio_rx.try_recv().is_ok() {}

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
            // Wait for playback to fully drain before re-enabling the mic.
            wait_for_playback_drain(&ctx.playback_queue).await;
            // Flush any audio captured during the suppressed window
            // (including TTS bleed through the mic).
            while ctx.audio_rx.try_recv().is_ok() {}
            vad.reset();
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
            // Brief settle so the trailing tail of the output buffer
            // (handled by cpal internally) has time to play before the
            // mic re-opens.
            tokio::time::sleep(std::time::Duration::from_millis(150)).await;
            return;
        }
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }
}

// ── Energy VAD ──────────────────────────────────────────────────────────

#[derive(PartialEq, Eq)]
enum VadState {
    Idle,
    Speaking,
}

struct EnergyVad {
    threshold: f64,
    min_speech_frames: u32,
    end_silence_frames: u32,
    max_utterance_frames: u32,
    state: VadState,
    consecutive_speech: u32,
    consecutive_silence: u32,
    buffer: Vec<i16>,
    pending: Vec<i16>,
    utterance_frames: u32,
}

impl EnergyVad {
    fn new(
        threshold: f64,
        min_speech_ms: u32,
        end_silence_ms: u32,
        max_utterance_ms: u32,
    ) -> Self {
        let frame_ms = 1000 / 50; // 20 ms per frame
        Self {
            threshold,
            min_speech_frames: min_speech_ms / frame_ms,
            end_silence_frames: end_silence_ms / frame_ms,
            max_utterance_frames: max_utterance_ms / frame_ms,
            state: VadState::Idle,
            consecutive_speech: 0,
            consecutive_silence: 0,
            buffer: Vec::new(),
            pending: Vec::new(),
            utterance_frames: 0,
        }
    }

    fn reset(&mut self) {
        self.state = VadState::Idle;
        self.consecutive_speech = 0;
        self.consecutive_silence = 0;
        self.buffer.clear();
        self.pending.clear();
        self.utterance_frames = 0;
    }

    /// Push 16 kHz mono PCM into the VAD. Returns zero or more
    /// completed utterances. Caller should `reset()` after gating the
    /// mic and re-enabling so onset thresholds are computed fresh.
    fn feed(&mut self, samples: &[i16]) -> Vec<Vec<i16>> {
        self.pending.extend_from_slice(samples);
        let mut completed = Vec::new();
        while self.pending.len() >= VAD_FRAME_SAMPLES {
            let frame: Vec<i16> = self.pending.drain(..VAD_FRAME_SAMPLES).collect();
            if let Some(utterance) = self.process_frame(&frame) {
                completed.push(utterance);
            }
        }
        completed
    }

    fn process_frame(&mut self, frame: &[i16]) -> Option<Vec<i16>> {
        let rms = rms_of(frame);
        let is_speech = rms > self.threshold;
        match self.state {
            VadState::Idle => {
                if is_speech {
                    self.consecutive_speech += 1;
                    self.buffer.extend_from_slice(frame);
                    self.utterance_frames += 1;
                    if self.consecutive_speech >= self.min_speech_frames {
                        self.state = VadState::Speaking;
                        self.consecutive_silence = 0;
                    }
                } else {
                    self.consecutive_speech = 0;
                    self.buffer.clear();
                    self.utterance_frames = 0;
                }
                None
            }
            VadState::Speaking => {
                self.buffer.extend_from_slice(frame);
                self.utterance_frames += 1;
                if is_speech {
                    self.consecutive_silence = 0;
                } else {
                    self.consecutive_silence += 1;
                }
                let end_of_speech = self.consecutive_silence >= self.end_silence_frames;
                let hit_cap = self.utterance_frames >= self.max_utterance_frames;
                if end_of_speech || hit_cap {
                    let utterance = std::mem::take(&mut self.buffer);
                    self.state = VadState::Idle;
                    self.consecutive_speech = 0;
                    self.consecutive_silence = 0;
                    self.utterance_frames = 0;
                    return Some(utterance);
                }
                None
            }
        }
    }
}

fn rms_of(frame: &[i16]) -> f64 {
    if frame.is_empty() {
        return 0.0;
    }
    let sum_sq: f64 = frame.iter().map(|s| (*s as f64).powi(2)).sum();
    (sum_sq / frame.len() as f64).sqrt()
}

// ── Audio capture / playback (cpal) ─────────────────────────────────────

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
    // Clear any residual playback so a re-entry doesn't blend audio.
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

/// Linear interpolation resample. Same approach as the server-side
/// Gradio TTS helper; good enough for speech, swap to rubato later
/// if needed.
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
    fn vad_detects_speech_then_silence() {
        let mut vad = EnergyVad::new(500.0, 200, 600, 30_000);
        // 200 ms of speech (10 frames of 20 ms) at amplitude 5000.
        let speech_frame = vec![5000i16; VAD_FRAME_SAMPLES];
        // 700 ms of silence (35 frames) — exceeds 600 ms close threshold.
        let silent_frame = vec![0i16; VAD_FRAME_SAMPLES];

        let mut utterances: Vec<Vec<i16>> = Vec::new();
        for _ in 0..10 {
            utterances.extend(vad.feed(&speech_frame));
        }
        for _ in 0..35 {
            utterances.extend(vad.feed(&silent_frame));
        }
        assert_eq!(utterances.len(), 1, "exactly one utterance expected");
        // 10 speech frames + at most 30 silence frames before close
        // (end_silence_frames = 30 at 20ms/frame).
        let u = &utterances[0];
        assert!(
            u.len() >= 10 * VAD_FRAME_SAMPLES,
            "utterance shorter than 200 ms: {} samples",
            u.len()
        );
    }

    #[test]
    fn vad_ignores_brief_noise_below_min_speech() {
        let mut vad = EnergyVad::new(500.0, 200, 600, 30_000);
        // Only 60 ms of speech (3 frames) — below 200 ms threshold.
        let speech_frame = vec![5000i16; VAD_FRAME_SAMPLES];
        let silent_frame = vec![0i16; VAD_FRAME_SAMPLES];
        let mut utterances: Vec<Vec<i16>> = Vec::new();
        for _ in 0..3 {
            utterances.extend(vad.feed(&speech_frame));
        }
        for _ in 0..35 {
            utterances.extend(vad.feed(&silent_frame));
        }
        assert!(
            utterances.is_empty(),
            "brief noise should not trigger utterance"
        );
    }
}
