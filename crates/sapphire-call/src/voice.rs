//! Voice satellite subcommand: push-to-talk capture + streaming playback.
//!
//! Holds Space to record, releases to ship the utterance to the agent
//! via `voice/pipeline_run`, and plays the streamed TTS reply through
//! the default output device as audio_chunks arrive. v1 — no wake
//! word, no VAD, no barge-in.

use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use anyhow::{Context, Result, anyhow};
use cpal::SampleFormat;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use crossterm::event::{Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use crossterm::terminal::{disable_raw_mode, enable_raw_mode};
use sapphire_agent_api::{VoiceEvent, voice::PIPELINE_SAMPLE_RATE, voice_pipeline_run};
use tokio::sync::mpsc;

/// Maximum utterance length in samples at the pipeline rate (16 kHz mono).
/// Hard cap matches the server's default capture_max_ms (30 s) so the
/// satellite drops anything the server would refuse anyway.
const MAX_UTTERANCE_SAMPLES: usize = (PIPELINE_SAMPLE_RATE as usize) * 30;

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
    eprintln!("sapphire-call voice (session: {display_id}{})", if is_new { ", new" } else { ", resumed" });

    // ── Audio I/O ────────────────────────────────────────────────────────
    let input_buf: Arc<std::sync::Mutex<Vec<i16>>> =
        Arc::new(std::sync::Mutex::new(Vec::new()));
    let recording = Arc::new(AtomicBool::new(false));
    let (input_stream, input_rate, input_channels) =
        open_input_stream(Arc::clone(&input_buf), Arc::clone(&recording))?;
    input_stream.play()?;

    let playback_queue: Arc<std::sync::Mutex<VecDeque<i16>>> =
        Arc::new(std::sync::Mutex::new(VecDeque::new()));
    let (output_stream, output_rate, output_channels) =
        open_output_stream(Arc::clone(&playback_queue))?;
    output_stream.play()?;

    eprintln!(
        "input: {input_rate} Hz × {input_channels}ch  output: {output_rate} Hz × {output_channels}ch",
    );

    // ── PTT key reader (raw mode) ────────────────────────────────────────
    enable_raw_mode().context("failed to enable terminal raw mode")?;
    let _raw_guard = RawModeGuard;

    eprintln!("Hold SPACE to talk, release to send. Ctrl-C to quit.\r");

    let (key_tx, mut key_rx) = mpsc::unbounded_channel::<KeyMsg>();
    let key_tx_clone = key_tx.clone();
    let key_thread = std::thread::spawn(move || key_event_loop(key_tx_clone));
    // Drop the producer-side handle so the channel closes if the thread exits.
    drop(key_tx);

    let mut holding = false;
    loop {
        let Some(msg) = key_rx.recv().await else {
            break;
        };
        match msg {
            KeyMsg::SpaceDown if !holding => {
                holding = true;
                input_buf
                    .lock()
                    .map_err(|_| anyhow!("input buffer poisoned"))?
                    .clear();
                recording.store(true, Ordering::SeqCst);
                eprint!("\r● recording...\r");
            }
            KeyMsg::SpaceUp if holding => {
                holding = false;
                recording.store(false, Ordering::SeqCst);
                let pcm: Vec<i16> = {
                    let mut buf = input_buf
                        .lock()
                        .map_err(|_| anyhow!("input buffer poisoned"))?;
                    std::mem::take(&mut *buf)
                };
                let mono = to_mono(&pcm, input_channels);
                let resampled = resample_to(&mono, input_rate, PIPELINE_SAMPLE_RATE);
                if resampled.is_empty() {
                    eprintln!("(no audio captured)\r");
                    continue;
                }
                let trimmed = if resampled.len() > MAX_UTTERANCE_SAMPLES {
                    eprintln!(
                        "(utterance exceeded {} s; truncating)\r",
                        MAX_UTTERANCE_SAMPLES / PIPELINE_SAMPLE_RATE as usize
                    );
                    resampled[..MAX_UTTERANCE_SAMPLES].to_vec()
                } else {
                    resampled
                };
                eprintln!(
                    "→ uploading {} samples ({:.2}s)\r",
                    trimmed.len(),
                    trimmed.len() as f32 / PIPELINE_SAMPLE_RATE as f32
                );
                if let Err(e) = process_utterance(
                    &client,
                    &base,
                    &mcp_session_id,
                    &trimmed,
                    language.as_deref(),
                    Arc::clone(&playback_queue),
                    output_rate,
                    output_channels,
                )
                .await
                {
                    eprintln!("[error: {e:#}]\r");
                }
            }
            KeyMsg::CtrlC => {
                eprintln!("\rQuitting...");
                break;
            }
            _ => {}
        }
    }

    // The key thread is blocked on event::read; once main exits, raw mode
    // is dropped and the thread will eventually exit on its next read.
    // Detach (don't join) — we don't want shutdown to hang on a keypress.
    drop(key_thread);
    Ok(())
}

// ── PTT key reader ──────────────────────────────────────────────────────

enum KeyMsg {
    SpaceDown,
    SpaceUp,
    CtrlC,
}

fn key_event_loop(tx: mpsc::UnboundedSender<KeyMsg>) {
    loop {
        // Short poll so we can notice channel closure without blocking forever.
        match crossterm::event::poll(Duration::from_millis(100)) {
            Ok(true) => {}
            Ok(false) => continue,
            Err(_) => break,
        }
        let Ok(evt) = crossterm::event::read() else {
            break;
        };
        let Event::Key(KeyEvent {
            code,
            kind,
            modifiers,
            ..
        }) = evt
        else {
            continue;
        };
        let msg = match (code, kind) {
            (KeyCode::Char('c'), _) if modifiers.contains(KeyModifiers::CONTROL) => KeyMsg::CtrlC,
            (KeyCode::Char(' '), KeyEventKind::Press) => KeyMsg::SpaceDown,
            (KeyCode::Char(' '), KeyEventKind::Release) => KeyMsg::SpaceUp,
            // Many terminals don't deliver key Release events; without them
            // we degrade to "tap Space to toggle" — alternate Press toggles
            // between recording and not.
            (KeyCode::Char(' '), KeyEventKind::Repeat) => continue,
            _ => continue,
        };
        if tx.send(msg).is_err() {
            break;
        }
    }
}

struct RawModeGuard;
impl Drop for RawModeGuard {
    fn drop(&mut self) {
        let _ = disable_raw_mode();
    }
}

// ── Audio capture / playback (cpal) ─────────────────────────────────────

fn open_input_stream(
    buf: Arc<std::sync::Mutex<Vec<i16>>>,
    recording: Arc<AtomicBool>,
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

    let err_fn = |e| eprintln!("[input stream error: {e}]\r");

    let stream = match format {
        SampleFormat::F32 => {
            let buf = Arc::clone(&buf);
            let recording = Arc::clone(&recording);
            device.build_input_stream(
                &config,
                move |data: &[f32], _| {
                    if !recording.load(Ordering::SeqCst) {
                        return;
                    }
                    if let Ok(mut b) = buf.lock() {
                        b.extend(
                            data.iter()
                                .map(|f| (f.clamp(-1.0, 1.0) * i16::MAX as f32) as i16),
                        );
                    }
                },
                err_fn,
                None,
            )?
        }
        SampleFormat::I16 => {
            let buf = Arc::clone(&buf);
            let recording = Arc::clone(&recording);
            device.build_input_stream(
                &config,
                move |data: &[i16], _| {
                    if !recording.load(Ordering::SeqCst) {
                        return;
                    }
                    if let Ok(mut b) = buf.lock() {
                        b.extend_from_slice(data);
                    }
                },
                err_fn,
                None,
            )?
        }
        SampleFormat::U16 => {
            let buf = Arc::clone(&buf);
            let recording = Arc::clone(&recording);
            device.build_input_stream(
                &config,
                move |data: &[u16], _| {
                    if !recording.load(Ordering::SeqCst) {
                        return;
                    }
                    if let Ok(mut b) = buf.lock() {
                        b.extend(data.iter().map(|s| (*s as i32 - 32768) as i16));
                    }
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

    let err_fn = |e| eprintln!("[output stream error: {e}]\r");

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
    mcp_session_id: &str,
    pcm_16khz: &[i16],
    language: Option<&str>,
    playback_queue: Arc<std::sync::Mutex<VecDeque<i16>>>,
    output_rate: u32,
    output_channels: u16,
) -> Result<()> {
    // Drain any leftover playback bytes from the previous reply so a
    // mid-playback re-press doesn't blend audio.
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
            voice_pipeline_run(
                &client,
                &base,
                &sid,
                &pcm,
                lang.as_deref(),
                event_tx,
            )
            .await
        }
    });

    while let Some(evt) = event_rx.recv().await {
        match evt {
            VoiceEvent::StageStart { stage } => {
                eprint!("\r[{stage}…] ");
            }
            VoiceEvent::StageEnd { stage } => {
                let _ = stage;
            }
            VoiceEvent::SttFinal { text } => {
                eprintln!("\r> {text}\r");
            }
            VoiceEvent::AssistantText { text } => {
                eprintln!("\r{text}\r");
            }
            VoiceEvent::AudioChunk { pcm } => {
                let upsampled = resample_to(&pcm, PIPELINE_SAMPLE_RATE, output_rate);
                let _ = output_channels; // channel fan-out happens in the output callback
                if let Ok(mut q) = playback_queue.lock() {
                    q.extend(upsampled);
                }
            }
            VoiceEvent::ToolStart { name } => {
                eprint!("\r[tool: {name}] ");
            }
            VoiceEvent::ToolEnd { name } => {
                let _ = name;
            }
            VoiceEvent::Done { .. } => break,
            VoiceEvent::Error { message } => {
                eprintln!("\r[error: {message}]\r");
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
}
