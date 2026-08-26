//! Registered speakers, loaded from workspace reference audio.
//!
//! `voices/<id>/*.wav` — the directory name is **both** the speaker id and
//! the display name. Embeddings are cached outside the workspace, keyed by
//! (reference file sha256 x model id), so renaming a directory triggers no
//! recomputation and swapping the embedding model recomputes automatically.
//! No model-dependent data ever lands in the workspace, which was the
//! requirement that shaped this split.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use tracing::warn;

use crate::ambient::audio::SpeakerEmbedder;
use crate::image_cache::sha256_hex;

#[derive(Debug, Clone, PartialEq)]
pub struct SpeakerMatch {
    pub id: String,
    pub score: f32,
}

pub struct SpeakerRegistry {
    voices_dir: PathBuf,
    emb_cache_dir: PathBuf,
    model_id: String,
    threshold: f32,
    /// Speaker id -> its centroid embedding.
    speakers: HashMap<String, Vec<f32>>,
}

impl SpeakerRegistry {
    pub fn open(
        voices_dir: PathBuf,
        emb_cache_dir: PathBuf,
        model_id: String,
        threshold: f32,
    ) -> Result<Self> {
        std::fs::create_dir_all(&emb_cache_dir)
            .with_context(|| format!("creating embedding cache dir {emb_cache_dir:?}"))?;
        Ok(Self {
            voices_dir,
            emb_cache_dir,
            model_id,
            threshold,
            speakers: HashMap::new(),
        })
    }

    /// Scan `voices/`, embedding each reference file (or reading its cached
    /// vector) and averaging per speaker.
    ///
    /// A speaker whose files cannot be read is warned about and skipped.
    /// One unreadable WAV must not take ambient ingest down with it.
    pub fn load_reference_audio(&mut self, embedder: &dyn SpeakerEmbedder) -> Result<()> {
        let entries = match std::fs::read_dir(&self.voices_dir) {
            Ok(e) => e,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
            Err(e) => return Err(e).context("reading voices dir"),
        };
        for entry in entries.flatten() {
            if !entry.file_type().map(|t| t.is_dir()).unwrap_or(false) {
                continue;
            }
            let id = entry.file_name().to_string_lossy().into_owned();
            let mut vectors = Vec::new();
            for file in std::fs::read_dir(entry.path())
                .into_iter()
                .flatten()
                .flatten()
            {
                let path = file.path();
                if path.extension().and_then(|e| e.to_str()) != Some("wav") {
                    continue;
                }
                match self.embedding_for(&path, embedder) {
                    Ok(v) => vectors.push(v),
                    Err(e) => warn!("speaker {id}: skipping {path:?}: {e}"),
                }
            }
            if vectors.is_empty() {
                warn!("speaker {id}: no usable reference audio; speaker disabled");
                continue;
            }
            self.speakers.insert(id, centroid(&vectors));
        }
        Ok(())
    }

    /// Cached embedding for one reference file, computing and storing it on
    /// a miss. The cache key is the file's content hash and the model id.
    fn embedding_for(&self, path: &Path, embedder: &dyn SpeakerEmbedder) -> Result<Vec<f32>> {
        let bytes = std::fs::read(path).with_context(|| format!("reading {path:?}"))?;
        let key = format!("{}.{}.emb", sha256_hex(&bytes), self.model_id);
        let cached = self.emb_cache_dir.join(&key);
        if let Ok(raw) = std::fs::read(&cached) {
            return Ok(decode_embedding(&raw));
        }
        let pcm = read_wav_mono_16k(&bytes)?;
        let vector = embedder.embed(&pcm)?;
        if let Err(e) = std::fs::write(&cached, encode_embedding(&vector)) {
            warn!("could not cache embedding at {cached:?}: {e}");
        }
        Ok(vector)
    }

    /// Register an embedding at runtime — used for auto-enrolled candidates,
    /// so a newly seen voice matches on its next segment.
    pub fn add_runtime(&mut self, id: String, embedding: Vec<f32>) {
        self.speakers.insert(id, embedding);
    }

    /// Best speaker above the threshold, or `None`.
    pub fn match_speaker(&self, embedding: &[f32]) -> Option<SpeakerMatch> {
        let mut best: Option<SpeakerMatch> = None;
        for (id, speaker_vector) in &self.speakers {
            let score = cosine_similarity(embedding, speaker_vector);
            if score < self.threshold {
                continue;
            }
            if best.as_ref().is_none_or(|b| score > b.score) {
                best = Some(SpeakerMatch {
                    id: id.clone(),
                    score,
                });
            }
        }
        best
    }

    /// Display name for a speaker id. Names live in the workspace as the
    /// directory name itself, so a renamed directory is picked up here
    /// without touching stored transcripts.
    // Consumed by the later transcript-rendering task, which resolves a
    // stored speaker id to a display name at read time. Delete this
    // attribute once that task adds a caller.
    #[allow(dead_code)]
    pub fn display_name(&self, id: &str) -> String {
        id.to_string()
    }
}

pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    dot / (na * nb)
}

fn centroid(vectors: &[Vec<f32>]) -> Vec<f32> {
    let dim = vectors[0].len();
    let mut out = vec![0.0; dim];
    for v in vectors {
        for (i, x) in v.iter().take(dim).enumerate() {
            out[i] += x;
        }
    }
    for x in &mut out {
        *x /= vectors.len() as f32;
    }
    out
}

/// Serialize an embedding to the little-endian f32 bytes stored in the
/// cache. `pub` because Task 9 (candidate store) reuses the same on-disk
/// vector encoding.
pub fn encode_embedding(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|f| f.to_le_bytes()).collect()
}

/// Inverse of [`encode_embedding`]. `pub` for the same reason.
pub fn decode_embedding(raw: &[u8]) -> Vec<f32> {
    raw.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn read_wav_mono_16k(bytes: &[u8]) -> Result<Vec<i16>> {
    let mut reader = hound::WavReader::new(std::io::Cursor::new(bytes))?;
    let spec = reader.spec();
    anyhow::ensure!(spec.channels == 1, "reference audio must be mono");
    anyhow::ensure!(
        spec.sample_rate == crate::voice::PIPELINE_SAMPLE_RATE,
        "reference audio must be {} Hz, got {}",
        crate::voice::PIPELINE_SAMPLE_RATE,
        spec.sample_rate
    );
    Ok(reader.samples::<i16>().collect::<Result<Vec<_>, _>>()?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ambient::audio::FixedEmbedder;

    /// Write a 1-second 16 kHz mono WAV of silence.
    fn write_wav(path: &std::path::Path) {
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: 16_000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut w = hound::WavWriter::create(path, spec).unwrap();
        for _ in 0..16_000 {
            w.write_sample(0i16).unwrap();
        }
        w.finalize().unwrap();
    }

    fn voices_with(names: &[&str]) -> (tempfile::TempDir, PathBuf) {
        let tmp = tempfile::tempdir().unwrap();
        let voices = tmp.path().join("voices");
        for n in names {
            let dir = voices.join(n);
            std::fs::create_dir_all(&dir).unwrap();
            write_wav(&dir.join("sample.wav"));
        }
        (tmp, voices)
    }

    #[test]
    fn cosine_similarity_is_one_for_identical_vectors() {
        assert!((cosine_similarity(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0]) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_similarity_is_zero_for_orthogonal_vectors() {
        assert!(cosine_similarity(&[1.0, 0.0], &[0.0, 1.0]).abs() < 1e-6);
    }

    #[test]
    fn matches_a_registered_speaker_above_the_threshold() {
        let (tmp, voices) = voices_with(&["me"]);
        let mut reg =
            SpeakerRegistry::open(voices, tmp.path().join("emb"), "test-model".into(), 0.55)
                .unwrap();
        reg.load_reference_audio(&FixedEmbedder::new(vec![1.0, 0.0, 0.0]))
            .unwrap();

        let m = reg.match_speaker(&[1.0, 0.0, 0.0]).expect("matched");
        assert_eq!(m.id, "me");
        assert!(m.score > 0.99);
    }

    #[test]
    fn returns_none_below_the_threshold() {
        let (tmp, voices) = voices_with(&["me"]);
        let mut reg =
            SpeakerRegistry::open(voices, tmp.path().join("emb"), "test-model".into(), 0.55)
                .unwrap();
        reg.load_reference_audio(&FixedEmbedder::new(vec![1.0, 0.0, 0.0]))
            .unwrap();
        assert!(reg.match_speaker(&[0.0, 1.0, 0.0]).is_none());
    }

    #[test]
    fn the_directory_name_is_the_speaker_id() {
        let (tmp, voices) = voices_with(&["blithe-otter-42"]);
        let mut reg =
            SpeakerRegistry::open(voices, tmp.path().join("emb"), "test-model".into(), 0.55)
                .unwrap();
        reg.load_reference_audio(&FixedEmbedder::new(vec![1.0, 0.0, 0.0]))
            .unwrap();
        assert_eq!(
            reg.match_speaker(&[1.0, 0.0, 0.0]).unwrap().id,
            "blithe-otter-42"
        );
    }

    #[test]
    fn embeddings_are_cached_by_reference_sha_and_model_id() {
        let (tmp, voices) = voices_with(&["me"]);
        let emb_dir = tmp.path().join("emb");
        let mut reg =
            SpeakerRegistry::open(voices.clone(), emb_dir.clone(), "model-a".into(), 0.55).unwrap();
        reg.load_reference_audio(&FixedEmbedder::new(vec![1.0, 0.0, 0.0]))
            .unwrap();
        let after_first = std::fs::read_dir(&emb_dir).unwrap().count();
        assert_eq!(after_first, 1);

        // Same files, different model id: a second cache entry, because the
        // vector is model-dependent and the workspace stores no vectors.
        let mut reg_b =
            SpeakerRegistry::open(voices, emb_dir.clone(), "model-b".into(), 0.55).unwrap();
        reg_b
            .load_reference_audio(&FixedEmbedder::new(vec![0.0, 1.0, 0.0]))
            .unwrap();
        assert_eq!(std::fs::read_dir(&emb_dir).unwrap().count(), 2);
        assert_eq!(reg_b.match_speaker(&[0.0, 1.0, 0.0]).unwrap().id, "me");
    }

    #[test]
    fn an_unreadable_reference_file_disables_that_speaker_without_failing_the_load() {
        let (tmp, voices) = voices_with(&["me"]);
        let broken = voices.join("broken");
        std::fs::create_dir_all(&broken).unwrap();
        std::fs::write(broken.join("sample.wav"), b"not a wav at all").unwrap();

        let mut reg =
            SpeakerRegistry::open(voices, tmp.path().join("emb"), "test-model".into(), 0.55)
                .unwrap();
        reg.load_reference_audio(&FixedEmbedder::new(vec![1.0, 0.0, 0.0]))
            .expect("load succeeds despite the broken speaker");
        assert_eq!(reg.match_speaker(&[1.0, 0.0, 0.0]).unwrap().id, "me");
    }
}
