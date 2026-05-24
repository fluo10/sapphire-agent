//! Desktop-side audio I/O.
//!
//! Two responsibilities, both backed by cpal:
//!
//! * [`AudioPlayer`] — drains a shared `VecDeque<i16>` of server-sourced
//!   PCM (s16le / mono / 16 kHz) into the default output device,
//!   resampling on the fly to whatever rate the device prefers.
//! * [`MicRecorder`] — opens the default input device on click, hands
//!   the captured PCM to Silero VAD running on a worker thread, and
//!   reports back either the auto-stopped utterance (silence detected)
//!   or whatever has accumulated when the user clicks Stop.
//!
//! cpal's `Stream` is `!Send` on most platforms, so each stream lives
//! on its own dedicated `std::thread`. The bevy side only ever holds
//! the `Arc`-shared queues + shutdown flags exposed by the structs in
//! this module.
//!
//! The whole module is intentionally bevy-free so it can move to a
//! shared `sapphire-call-gui` crate when mobile lands.

use std::collections::VecDeque;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::Duration;

use anyhow::{Context, Result, anyhow};
use cpal::SampleFormat;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use sherpa_onnx::{SileroVadModelConfig, VadModelConfig, VoiceActivityDetector};

/// Server-side pipeline sample rate. The agent ships TTS PCM at this
/// rate and expects mic uploads in the same shape; keep the constant
/// here so this crate doesn't pull `sapphire-agent-rpc`'s.
pub const PIPELINE_SAMPLE_RATE: u32 = 16_000;

/// Silero VAD window. The model is trained on 32 ms frames at 16 kHz,
/// which is exactly 512 samples — matches the satellite's constant.
const VAD_WINDOW_SAMPLES: usize = 512;
const VAD_MAX_SPEECH_SECONDS: f32 = 30.0;

/// Auto-download URL for the Silero VAD ONNX. Pinned to the
/// asr-models release — same source the CLI satellite uses, so the
/// cached file is interchangeable between clients (we don't share the
/// cache dir, but operators can copy the file across if desired).
const SILERO_VAD_URL: &str =
    "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx";

// =======================================================================
// TTS playback
// =======================================================================

/// Shared handle to the output stream. The cpal `Stream` is owned by a
/// dedicated supervisor thread (because it's `!Send`); the bevy side
/// only ever touches the queue + shutdown flag.
pub struct AudioPlayer {
    queue: Arc<Mutex<VecDeque<i16>>>,
    shutdown: Arc<AtomicBool>,
    /// Sample rate the output device was opened at. Server PCM (always
    /// 16 kHz) gets resampled to this on the way into the queue.
    output_rate: u32,
    join: Option<JoinHandle<()>>,
}

impl AudioPlayer {
    /// Open the default output device. Returns `None` if no output
    /// device is present (e.g. a headless CI box) so the chat path
    /// still works — TTS audio just gets dropped silently in that case.
    pub fn start() -> Result<Self> {
        let queue: Arc<Mutex<VecDeque<i16>>> = Arc::new(Mutex::new(VecDeque::new()));
        let shutdown = Arc::new(AtomicBool::new(false));

        // Open the stream on its owning thread, then send back the
        // negotiated sample rate so the bevy side knows what to
        // resample to.
        let (ready_tx, ready_rx) = std::sync::mpsc::channel::<Result<u32>>();
        let join = {
            let queue = Arc::clone(&queue);
            let shutdown = Arc::clone(&shutdown);
            std::thread::Builder::new()
                .name("sapphire-call-desktop audio-out".into())
                .spawn(move || run_output_thread(queue, shutdown, ready_tx))
                .context("spawn audio-out thread")?
        };

        let output_rate = match ready_rx.recv() {
            Ok(Ok(rate)) => rate,
            Ok(Err(e)) => return Err(e),
            Err(_) => return Err(anyhow!("audio-out thread dropped before reporting ready")),
        };

        Ok(Self {
            queue,
            shutdown,
            output_rate,
            join: Some(join),
        })
    }

    /// Append one TTS chunk to the playback queue. `pcm` is mono
    /// 16-kHz PCM (the format the agent's stream_chat_tts emits);
    /// resampled to the device rate before it lands on the queue.
    pub fn push_pcm_16khz(&self, pcm: &[i16]) {
        if pcm.is_empty() {
            return;
        }
        let resampled: Vec<i16> = if self.output_rate == PIPELINE_SAMPLE_RATE {
            pcm.to_vec()
        } else {
            resample_to(pcm, PIPELINE_SAMPLE_RATE, self.output_rate)
        };
        if let Ok(mut q) = self.queue.lock() {
            q.extend(resampled);
        }
    }

    /// Drop any queued audio. Called when the session resets or the
    /// server reports `tts_error` so the user doesn't hear stale
    /// fragments after the failure.
    pub fn drain(&self) {
        if let Ok(mut q) = self.queue.lock() {
            q.clear();
        }
    }
}

impl Drop for AudioPlayer {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::SeqCst);
        if let Some(handle) = self.join.take() {
            let _ = handle.join();
        }
    }
}

fn run_output_thread(
    queue: Arc<Mutex<VecDeque<i16>>>,
    shutdown: Arc<AtomicBool>,
    ready_tx: std::sync::mpsc::Sender<Result<u32>>,
) {
    let stream_and_rate = open_output_stream(&queue);
    let stream = match stream_and_rate {
        Ok((stream, rate)) => {
            let _ = ready_tx.send(Ok(rate));
            stream
        }
        Err(e) => {
            let _ = ready_tx.send(Err(e));
            return;
        }
    };
    if let Err(e) = stream.play() {
        tracing::warn!("audio output play() failed: {e:#}");
        return;
    }
    // Park until shutdown — `Stream` keeps the cpal worker alive for
    // its lifetime; dropping it (when this thread exits) stops audio.
    while !shutdown.load(Ordering::SeqCst) {
        std::thread::sleep(Duration::from_millis(100));
    }
    drop(stream);
}

fn open_output_stream(queue: &Arc<Mutex<VecDeque<i16>>>) -> Result<(cpal::Stream, u32)> {
    let host = cpal::default_host();
    let device = host
        .default_output_device()
        .ok_or_else(|| anyhow!("no default output device"))?;
    let supported = device
        .default_output_config()
        .context("query default output config")?;
    let rate = supported.sample_rate();
    let channels = supported.channels();
    let format = supported.sample_format();
    let config: cpal::StreamConfig = supported.clone().into();

    let err_fn = |e| tracing::warn!("audio output stream error: {e}");

    let stream = match format {
        SampleFormat::F32 => {
            let queue = Arc::clone(queue);
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
            let queue = Arc::clone(queue);
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
            let queue = Arc::clone(queue);
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
    Ok((stream, rate))
}

// =======================================================================
// Mic capture + Silero VAD auto-stop
// =======================================================================

/// Reported state of an in-flight mic capture. The UI polls this once
/// per frame and reacts: `Recording { speech_detected }` keeps the
/// stop button visible, `Done { pcm }` triggers a `voice/pipeline_run`
/// submission, `Failed` surfaces a message in chat history.
pub enum MicState {
    /// Still capturing; `speech_detected` flips true once Silero VAD
    /// has seen the start of an utterance — useful for "listening…"
    /// vs. "speak now…" affordance later.
    Recording { speech_detected: bool },
    /// VAD reported end-of-speech (silence threshold crossed) OR the
    /// user pressed Stop. PCM is mono s16le @ 16 kHz, ready to upload.
    Done { pcm: Vec<i16> },
    /// Capture aborted before producing PCM (no mic, VAD download
    /// failed, etc.). Message is human-readable.
    Failed { message: String },
}

/// Handle to an in-flight mic capture. Drop to cancel.
pub struct MicRecorder {
    poll_rx: std::sync::mpsc::Receiver<MicState>,
    stop_flag: Arc<AtomicBool>,
    finished: bool,
    /// Latched recording flag updated by `poll`. Used by the UI to
    /// keep "listening" affordance lit while we wait for the next
    /// state event.
    speech_detected: bool,
    join: Option<JoinHandle<()>>,
}

impl MicRecorder {
    /// Spawn the capture supervisor. The VAD model is downloaded
    /// lazily — first call may block on the network for a few seconds
    /// (subsequent calls hit the cache).
    pub fn start() -> Self {
        let (poll_tx, poll_rx) = std::sync::mpsc::channel::<MicState>();
        let stop_flag = Arc::new(AtomicBool::new(false));
        let join = {
            let stop_flag = Arc::clone(&stop_flag);
            std::thread::Builder::new()
                .name("sapphire-call-desktop mic".into())
                .spawn(move || run_mic_thread(stop_flag, poll_tx))
                .ok()
        };
        // If spawn failed, poll() will receive nothing — fall through
        // to a Failed event so the UI doesn't get stuck.
        if join.is_none() {
            let _ = poll_rx.try_recv();
        }
        Self {
            poll_rx,
            stop_flag,
            finished: false,
            speech_detected: false,
            join,
        }
    }

    /// Drain any state changes the worker thread has produced since the
    /// last call. Returns the most recent state, or `None` if nothing
    /// has changed.
    pub fn poll(&mut self) -> Option<MicState> {
        if self.finished {
            return None;
        }
        let mut latest = None;
        while let Ok(evt) = self.poll_rx.try_recv() {
            if let MicState::Recording { speech_detected } = &evt {
                self.speech_detected = *speech_detected;
            }
            let is_terminal = matches!(evt, MicState::Done { .. } | MicState::Failed { .. });
            latest = Some(evt);
            if is_terminal {
                self.finished = true;
                break;
            }
        }
        latest
    }

    /// Whether the worker has reported a finished/failed state. Once
    /// true, calling [`Self::poll`] will return `None` from here on.
    pub fn is_finished(&self) -> bool {
        self.finished
    }

    /// True once VAD has seen the start of speech. Used by the UI for
    /// the "listening…" indicator.
    pub fn speech_detected(&self) -> bool {
        self.speech_detected
    }

    /// Ask the worker to stop and emit a `Done` event with whatever
    /// PCM has accumulated. Idempotent.
    pub fn request_stop(&self) {
        self.stop_flag.store(true, Ordering::SeqCst);
    }
}

impl Drop for MicRecorder {
    fn drop(&mut self) {
        self.stop_flag.store(true, Ordering::SeqCst);
        if let Some(h) = self.join.take() {
            let _ = h.join();
        }
    }
}

fn run_mic_thread(stop_flag: Arc<AtomicBool>, tx: std::sync::mpsc::Sender<MicState>) {
    let vad_model = match ensure_silero_model() {
        Ok(p) => p,
        Err(e) => {
            let _ = tx.send(MicState::Failed {
                message: format!("VAD model unavailable: {e:#}"),
            });
            return;
        }
    };
    let mut vad = match build_vad(&vad_model) {
        Ok(v) => v,
        Err(e) => {
            let _ = tx.send(MicState::Failed {
                message: format!("VAD init failed: {e:#}"),
            });
            return;
        }
    };

    // cpal owns its callback thread — we receive samples here via a
    // std mpsc the callback feeds.
    let (sample_tx, sample_rx) = std::sync::mpsc::channel::<Vec<i16>>();
    let (stream, input_rate, input_channels) = match open_input_stream(sample_tx) {
        Ok(triple) => triple,
        Err(e) => {
            let _ = tx.send(MicState::Failed {
                message: format!("mic unavailable: {e:#}"),
            });
            return;
        }
    };
    if let Err(e) = stream.play() {
        let _ = tx.send(MicState::Failed {
            message: format!("mic play() failed: {e:#}"),
        });
        return;
    }

    let _ = tx.send(MicState::Recording {
        speech_detected: false,
    });

    let mut window_buf: Vec<f32> = Vec::with_capacity(VAD_WINDOW_SAMPLES);
    // Raw 16 kHz mono PCM accumulated since capture began. If the
    // user presses Stop before VAD fires, we ship this directly.
    let mut accum_pcm: Vec<i16> = Vec::new();
    let mut last_speech_state = false;

    loop {
        if stop_flag.load(Ordering::SeqCst) {
            // Manual stop: flush whatever VAD has assembled into a
            // segment, falling back to the raw accumulator if it
            // hasn't.
            let segment = drain_vad_segment(&mut vad);
            let pcm = segment.unwrap_or(accum_pcm);
            let _ = tx.send(MicState::Done { pcm });
            return;
        }
        let raw = match sample_rx.recv_timeout(Duration::from_millis(50)) {
            Ok(buf) => buf,
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => continue,
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                let _ = tx.send(MicState::Failed {
                    message: "mic stream closed unexpectedly".into(),
                });
                return;
            }
        };
        let mono = to_mono(&raw, input_channels);
        let resampled = if input_rate == PIPELINE_SAMPLE_RATE {
            mono
        } else {
            resample_to(&mono, input_rate, PIPELINE_SAMPLE_RATE)
        };
        accum_pcm.extend_from_slice(&resampled);
        for s in &resampled {
            window_buf.push(*s as f32 / i16::MAX as f32);
            if window_buf.len() == VAD_WINDOW_SAMPLES {
                vad.accept_waveform(&window_buf);
                window_buf.clear();
            }
        }

        // Detect the start of speech ↔ tell the UI to stop showing
        // "speak now" hint. Silero exposes a `is_detected()` we
        // mirror to a single bool.
        let now_speech = vad.detected();
        if now_speech != last_speech_state {
            last_speech_state = now_speech;
            let _ = tx.send(MicState::Recording {
                speech_detected: now_speech,
            });
        }

        // Auto-stop on silence: a popped front segment means
        // Silero saw speech start, then a min_silence_duration of
        // silence after it. Ship the segment and we're done.
        if let Some(segment) = drain_vad_segment(&mut vad) {
            let _ = tx.send(MicState::Done { pcm: segment });
            return;
        }
    }
}

fn drain_vad_segment(vad: &mut VoiceActivityDetector) -> Option<Vec<i16>> {
    if vad.is_empty() {
        return None;
    }
    let segment = vad.front()?;
    let pcm: Vec<i16> = segment
        .samples()
        .iter()
        .map(|f| (f.clamp(-1.0, 1.0) * i16::MAX as f32) as i16)
        .collect();
    vad.pop();
    Some(pcm)
}

fn build_vad(model_path: &Path) -> Result<VoiceActivityDetector> {
    let silero = SileroVadModelConfig {
        model: Some(model_path.to_string_lossy().into_owned()),
        // Defaults mirror the call CLI's "balanced" sensitivity. The
        // desktop client doesn't expose sensitivity knobs yet — if
        // operators report false-stops we'll surface them in Settings.
        threshold: 0.5,
        min_silence_duration: 0.8,
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

fn ensure_silero_model() -> Result<PathBuf> {
    let dest = cache_dir()?.join("silero_vad.onnx");
    if dest.exists() {
        return Ok(dest);
    }
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("create {}", parent.display()))?;
    }
    tracing::info!(
        "downloading Silero VAD model from {SILERO_VAD_URL} → {} (one-time)",
        dest.display()
    );
    let resp = reqwest::blocking::Client::builder()
        .timeout(None)
        .build()
        .context("build reqwest blocking client")?
        .get(SILERO_VAD_URL)
        .send()
        .context("Silero VAD download")?
        .error_for_status()
        .context("Silero VAD HTTP status")?;
    let bytes = resp.bytes().context("Silero VAD body")?;
    let tmp = dest.with_extension("partial");
    std::fs::write(&tmp, &bytes).with_context(|| format!("write {}", tmp.display()))?;
    std::fs::rename(&tmp, &dest)
        .with_context(|| format!("rename {} → {}", tmp.display(), dest.display()))?;
    Ok(dest)
}

fn cache_dir() -> Result<PathBuf> {
    directories::ProjectDirs::from("", "", "sapphire-call-desktop")
        .map(|p| p.data_local_dir().join("voice-models"))
        .ok_or_else(|| anyhow!("no XDG data dir available"))
}

fn open_input_stream(
    tx: std::sync::mpsc::Sender<Vec<i16>>,
) -> Result<(cpal::Stream, u32, u16)> {
    let host = cpal::default_host();
    let device = host
        .default_input_device()
        .ok_or_else(|| anyhow!("no default input device"))?;
    let supported = device
        .default_input_config()
        .context("query default input config")?;
    let rate = supported.sample_rate();
    let channels = supported.channels();
    let format = supported.sample_format();
    let config: cpal::StreamConfig = supported.clone().into();

    let err_fn = |e| tracing::warn!("audio input stream error: {e}");

    let stream = match format {
        SampleFormat::F32 => {
            let tx = tx.clone();
            device.build_input_stream(
                &config,
                move |data: &[f32], _| {
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
            device.build_input_stream(
                &config,
                move |data: &[i16], _| {
                    let _ = tx.send(data.to_vec());
                },
                err_fn,
                None,
            )?
        }
        SampleFormat::U16 => {
            let tx = tx.clone();
            device.build_input_stream(
                &config,
                move |data: &[u16], _| {
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

// =======================================================================
// Helpers
// =======================================================================

fn to_mono(samples: &[i16], channels: u16) -> Vec<i16> {
    if channels <= 1 {
        return samples.to_vec();
    }
    let ch = channels as usize;
    samples
        .chunks(ch)
        .map(|frame| {
            let sum: i32 = frame.iter().map(|s| *s as i32).sum();
            (sum / ch as i32) as i16
        })
        .collect()
}

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
        assert_eq!(to_mono(&frames, 2), vec![150, 350]);
    }

    #[test]
    fn resample_no_op_on_equal_rates() {
        let frames: Vec<i16> = vec![1, 2, 3];
        assert_eq!(resample_to(&frames, 16000, 16000), frames);
    }
}
