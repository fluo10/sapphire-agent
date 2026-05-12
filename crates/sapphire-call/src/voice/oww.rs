//! openWakeWord runtime — three-stage ONNX pipeline that lets us
//! spot custom AI-name wake words trained externally with
//! <https://github.com/dscripka/openWakeWord>.
//!
//! Three models in sequence per 80 ms audio chunk:
//!
//! 1. **melspectrogram.onnx** (shared) — raw i16 audio (cast to f32,
//!    *un-normalised*) → mel features. Output gets the standard
//!    `x/10 + 2` post-transform applied so the downstream embedding
//!    model sees the same input range as Google's original
//!    speech_embedding TF Hub model.
//! 2. **embedding_model.onnx** (shared) — last 76 mel frames →
//!    96-dim embedding.
//! 3. **`<custom>.onnx`** (per wake word, trained by the user) — last
//!    16 embeddings → confidence score.
//!
//! Faithfully ports the streaming flow from openWakeWord's
//! `openwakeword/utils.py::AudioFeatures._streaming_features` —
//! comments mark the corresponding Python steps. The wrappers around
//! `Session::run` use manual error conversion because ort's `Error`
//! type carries non-Send pointers and won't auto-flow through `?`
//! into `anyhow::Error`.

use std::collections::VecDeque;
use std::path::Path;

use anyhow::{Result, anyhow};
use ort::session::{Session, builder::GraphOptimizationLevel};
use ort::value::TensorRef;

#[cfg(test)]
use sapphire_agent_api::voice::PIPELINE_SAMPLE_RATE;

/// 80 ms at 16 kHz. The OWW pipeline only advances state on this
/// boundary; sub-chunk audio is buffered until the next multiple.
const CHUNK_SAMPLES: usize = 1280;
/// Lookback (in samples) added to the mel input to give the streaming
/// melspec model enough left-context. Matches `160*3` from the Python.
const MEL_LOOKBACK_SAMPLES: usize = 480;
/// Mel buffer window the embedding model expects: 76 frames × 32 mel
/// bins.
const MEL_WINDOW_FRAMES: usize = 76;
const MEL_BINS: usize = 32;
/// Empirically OWW's mel produces 8 new frames per 80 ms chunk.
const MEL_FRAMES_PER_CHUNK: usize = 8;
/// Wake word classifier window: last 16 embeddings.
const EMBED_WINDOW: usize = 16;
const EMBED_DIM: usize = 96;
/// Default confidence threshold above which wake fires.
const DEFAULT_THRESHOLD: f32 = 0.5;
/// Cool-down after a successful wake so a single sustained utterance
/// doesn't fire repeatedly. 25 chunks ≈ 2 s.
const COOLDOWN_CHUNKS: usize = 25;

pub struct OpenWakeWordDetector {
    mel_session: Session,
    embed_session: Session,
    wake_session: Session,
    audio_pending: Vec<i16>,
    audio_history: VecDeque<i16>,
    mel_buf: VecDeque<[f32; MEL_BINS]>,
    embed_buf: VecDeque<[f32; EMBED_DIM]>,
    label: String,
    threshold: f32,
    cooldown_left: usize,
    /// Resolved tensor names per session — Session::inputs()[0].name.
    mel_input_name: String,
    embed_input_name: String,
    wake_input_name: String,
}

// SAFETY: ort::Session is not Send by default (NonNull pointers), but
// each detector is only ever touched from the satellite's single
// listen loop after construction completes. Send is needed so the
// init can run on a `tokio::task::spawn_blocking` thread and be
// moved back to the await point.
unsafe impl Send for OpenWakeWordDetector {}

impl OpenWakeWordDetector {
    /// Build the detector. `wake_model_path` should point at a
    /// `.onnx` produced by openWakeWord's training pipeline; the mel
    /// and embedding ONNX frontends are shared across every custom
    /// wake word and auto-cached by [`super::download`].
    pub fn create(
        mel_path: &Path,
        embed_path: &Path,
        wake_model_path: &Path,
        label: String,
    ) -> Result<Self> {
        let mel_session = build_session(mel_path)?;
        let embed_session = build_session(embed_path)?;
        let wake_session = build_session(wake_model_path)?;

        let mel_input_name = first_input_name(&mel_session)?;
        let embed_input_name = first_input_name(&embed_session)?;
        let wake_input_name = first_input_name(&wake_session)?;

        Ok(Self {
            mel_session,
            embed_session,
            wake_session,
            audio_pending: Vec::with_capacity(CHUNK_SAMPLES * 2),
            audio_history: VecDeque::with_capacity(MEL_LOOKBACK_SAMPLES + CHUNK_SAMPLES),
            mel_buf: VecDeque::with_capacity(MEL_WINDOW_FRAMES + MEL_FRAMES_PER_CHUNK),
            embed_buf: VecDeque::with_capacity(EMBED_WINDOW + 1),
            label,
            threshold: DEFAULT_THRESHOLD,
            cooldown_left: 0,
            mel_input_name,
            embed_input_name,
            wake_input_name,
        })
    }

    /// Feed 16 kHz mono i16 audio. Returns `Some(label)` on the chunk
    /// where the wake fires; subsequent calls within the cool-down
    /// window are suppressed.
    pub fn feed(&mut self, samples: &[i16]) -> Result<Option<String>> {
        if samples.is_empty() {
            return Ok(None);
        }
        self.audio_pending.extend_from_slice(samples);
        let mut detected: Option<String> = None;
        while self.audio_pending.len() >= CHUNK_SAMPLES {
            let chunk: Vec<i16> = self.audio_pending.drain(..CHUNK_SAMPLES).collect();
            if let Some(label) = self.process_chunk(&chunk)? {
                detected = Some(label);
            }
        }
        Ok(detected)
    }

    pub fn reset(&mut self) {
        self.audio_pending.clear();
        self.audio_history.clear();
        self.mel_buf.clear();
        self.embed_buf.clear();
        self.cooldown_left = 0;
    }

    fn process_chunk(&mut self, chunk: &[i16]) -> Result<Option<String>> {
        // 1. Update lookback history for mel left-context.
        for s in chunk {
            self.audio_history.push_back(*s);
        }
        while self.audio_history.len() > MEL_LOOKBACK_SAMPLES + CHUNK_SAMPLES {
            self.audio_history.pop_front();
        }

        // 2. Mel input: (1, N) f32 with int16 range (NOT normalised).
        let n = self.audio_history.len();
        let mel_audio: Vec<f32> = self.audio_history.iter().map(|s| *s as f32).collect();
        let mel_input = TensorRef::from_array_view((vec![1i64, n as i64], mel_audio.as_slice()))
            .map_err(ort_err("OWW mel TensorRef"))?;

        // Extract the mel slice into an owned Vec inside this block so
        // the borrow on `mel_session` ends before we later mutate
        // `self.mel_buf` / call other &mut self methods.
        let mel_flat_owned: Vec<f32> = {
            let mel_outputs = self
                .mel_session
                .run(ort::inputs![self.mel_input_name.as_str() => mel_input])
                .map_err(ort_err("OWW mel run"))?;
            let (_, mel_flat) = mel_outputs[0]
                .try_extract_tensor::<f32>()
                .map_err(ort_err("OWW mel extract"))?;
            mel_flat.to_vec()
        };
        let _ = push_mel(&mut self.mel_buf, &mel_flat_owned)?;

        if self.mel_buf.len() < MEL_WINDOW_FRAMES {
            self.tick_cooldown();
            return Ok(None);
        }

        // 3. Embedding input: shape (1, 76, 32, 1), last 76 mel frames.
        let mut window_flat: Vec<f32> = Vec::with_capacity(MEL_WINDOW_FRAMES * MEL_BINS);
        let start = self.mel_buf.len() - MEL_WINDOW_FRAMES;
        for row in self.mel_buf.iter().skip(start) {
            window_flat.extend_from_slice(row);
        }
        let embed_input = TensorRef::from_array_view((
            vec![1i64, MEL_WINDOW_FRAMES as i64, MEL_BINS as i64, 1i64],
            window_flat.as_slice(),
        ))
        .map_err(ort_err("OWW embed TensorRef"))?;
        let embed_flat_owned: Vec<f32> = {
            let embed_outputs = self
                .embed_session
                .run(ort::inputs![self.embed_input_name.as_str() => embed_input])
                .map_err(ort_err("OWW embed run"))?;
            let (_, embed_flat) = embed_outputs[0]
                .try_extract_tensor::<f32>()
                .map_err(ort_err("OWW embed extract"))?;
            embed_flat.to_vec()
        };
        push_embedding(&mut self.embed_buf, &embed_flat_owned)?;

        // Trim buffers to keep memory bounded.
        while self.mel_buf.len() > MEL_WINDOW_FRAMES + MEL_FRAMES_PER_CHUNK {
            self.mel_buf.pop_front();
        }
        while self.embed_buf.len() > EMBED_WINDOW {
            self.embed_buf.pop_front();
        }

        if self.embed_buf.len() < EMBED_WINDOW {
            self.tick_cooldown();
            return Ok(None);
        }

        // 4. Wake input: shape (1, 16, 96), last 16 embeddings.
        let mut feat_flat: Vec<f32> = Vec::with_capacity(EMBED_WINDOW * EMBED_DIM);
        for row in &self.embed_buf {
            feat_flat.extend_from_slice(row);
        }
        let wake_input = TensorRef::from_array_view((
            vec![1i64, EMBED_WINDOW as i64, EMBED_DIM as i64],
            feat_flat.as_slice(),
        ))
        .map_err(ort_err("OWW wake TensorRef"))?;
        let confidence: f32 = {
            let wake_outputs = self
                .wake_session
                .run(ort::inputs![self.wake_input_name.as_str() => wake_input])
                .map_err(ort_err("OWW wake run"))?;
            let (_, wake_flat) = wake_outputs[0]
                .try_extract_tensor::<f32>()
                .map_err(ort_err("OWW wake extract"))?;
            wake_flat.first().copied().unwrap_or(0.0)
        };

        // 5. Threshold + cool-down.
        if self.cooldown_left > 0 {
            self.cooldown_left -= 1;
            return Ok(None);
        }
        if confidence >= self.threshold {
            self.cooldown_left = COOLDOWN_CHUNKS;
            return Ok(Some(self.label.clone()));
        }
        Ok(None)
    }

    fn tick_cooldown(&mut self) {
        if self.cooldown_left > 0 {
            self.cooldown_left -= 1;
        }
    }
}

fn build_session(path: &Path) -> Result<Session> {
    Session::builder()
        .map_err(ort_err("session builder"))?
        .with_optimization_level(GraphOptimizationLevel::Level1)
        .map_err(ort_err("session opt level"))?
        .with_intra_threads(1)
        .map_err(ort_err("session threads"))?
        .commit_from_file(path)
        .map_err(|e| anyhow!("ort: load '{}': {e}", path.display()))
}

fn first_input_name(session: &Session) -> Result<String> {
    session
        .inputs()
        .first()
        .map(|i| i.name().to_string())
        .ok_or_else(|| anyhow!("ort session has no inputs"))
}

/// Convert an ort error into an anyhow error, prepending context.
/// Needed because ort::Error contains non-Send pointers and won't
/// flow through `?` into anyhow::Error automatically.
fn ort_err<E: std::fmt::Display>(ctx: &'static str) -> impl Fn(E) -> anyhow::Error {
    move |e| anyhow!("{ctx}: {e}")
}

/// Append new mel frames to the rolling buffer, applying OWW's
/// `x / 10 + 2` post-transform. Returns the number of frames added.
fn push_mel(buf: &mut VecDeque<[f32; MEL_BINS]>, flat: &[f32]) -> Result<usize> {
    if !flat.len().is_multiple_of(MEL_BINS) {
        anyhow::bail!(
            "OWW mel output not divisible by MEL_BINS ({MEL_BINS}); got {}",
            flat.len()
        );
    }
    let n_frames = flat.len() / MEL_BINS;
    for frame in flat.chunks_exact(MEL_BINS) {
        let mut row = [0f32; MEL_BINS];
        for (i, v) in frame.iter().enumerate() {
            row[i] = v / 10.0 + 2.0;
        }
        buf.push_back(row);
    }
    Ok(n_frames)
}

fn push_embedding(buf: &mut VecDeque<[f32; EMBED_DIM]>, flat: &[f32]) -> Result<()> {
    if flat.len() != EMBED_DIM {
        anyhow::bail!(
            "OWW embedding output expected {EMBED_DIM} elements, got {}",
            flat.len()
        );
    }
    let mut row = [0f32; EMBED_DIM];
    row.copy_from_slice(flat);
    buf.push_back(row);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constants_sanity() {
        // 80 ms at 16 kHz = 1280 samples.
        assert_eq!(CHUNK_SAMPLES, (PIPELINE_SAMPLE_RATE as usize / 1000) * 80);
        assert_eq!(MEL_WINDOW_FRAMES * MEL_BINS, 76 * 32);
        assert_eq!(EMBED_WINDOW * EMBED_DIM, 16 * 96);
    }
}
