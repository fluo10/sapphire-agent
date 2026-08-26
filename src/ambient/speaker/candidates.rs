//! Auto-enrolled speakers awaiting a name.
//!
//! A day of ambient audio contains television, shop staff, train
//! announcements and passers-by. Writing every first-seen voice straight
//! into the workspace would bury the handful of people worth naming under
//! hundreds of one-off entries, so first sight enrols a **candidate** in
//! the cache, and only sustained presence promotes it into `voices/`.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use chrono::{DateTime, NaiveDate, Utc};
use serde::{Deserialize, Serialize};
use tracing::warn;

use super::registry::{decode_embedding, encode_embedding};
use crate::voice::PIPELINE_SAMPLE_RATE;

/// How many sample utterances to keep per candidate, so a human deciding
/// whether to name it has something to recognise it by.
const MAX_SAMPLES: usize = 5;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CandidateStats {
    /// Cumulative gated speech, in whole seconds.
    pub speech_seconds: u32,
    /// Distinct logical days this voice was heard on.
    pub days_seen: Vec<NaiveDate>,
    pub first_seen: DateTime<Utc>,
    /// A few transcribed utterances, for recognition.
    #[serde(default)]
    pub samples: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct Candidate {
    pub id: String,
    pub centroid: Vec<f32>,
    pub stats: CandidateStats,
    /// Number of observations folded into `centroid`, for the running mean.
    observations: u32,
}

pub struct CandidateStore {
    dir: PathBuf,
    candidates: HashMap<String, Candidate>,
}

impl CandidateStore {
    /// Open the store, loading every candidate already on disk.
    pub fn open(dir: PathBuf) -> Result<Self> {
        std::fs::create_dir_all(&dir)
            .with_context(|| format!("creating candidate dir {dir:?}"))?;
        let mut candidates = HashMap::new();
        for entry in std::fs::read_dir(&dir)?.flatten() {
            if !entry.file_type().map(|t| t.is_dir()).unwrap_or(false) {
                continue;
            }
            let id = entry.file_name().to_string_lossy().into_owned();
            match load_candidate(&entry.path(), &id) {
                Ok(c) => {
                    candidates.insert(id, c);
                }
                Err(e) => warn!("skipping unreadable candidate {id}: {e}"),
            }
        }
        Ok(Self { dir, candidates })
    }

    /// Register a newly seen voice. Returns the minted id.
    pub fn enrol(
        &mut self,
        embedding: Vec<f32>,
        clip: &[i16],
        day: NaiveDate,
        speech_ms: u32,
        sample_text: &str,
    ) -> Result<String> {
        let id = grain_id::GrainId::random().to_string();
        let dir = self.dir.join(&id);
        std::fs::create_dir_all(&dir)?;
        write_wav(&dir.join("clip.wav"), clip)?;
        std::fs::write(dir.join("centroid.emb"), encode_embedding(&embedding))?;

        let stats = CandidateStats {
            speech_seconds: speech_ms / 1000,
            days_seen: vec![day],
            first_seen: Utc::now(),
            samples: sample_vec(sample_text),
        };
        std::fs::write(dir.join("stats.json"), serde_json::to_vec_pretty(&stats)?)?;

        self.candidates.insert(
            id.clone(),
            Candidate {
                id: id.clone(),
                centroid: embedding,
                stats,
                observations: 1,
            },
        );
        Ok(id)
    }

    /// Fold another segment into an existing candidate.
    pub fn observe(
        &mut self,
        id: &str,
        embedding: &[f32],
        day: NaiveDate,
        speech_ms: u32,
        sample_text: &str,
    ) -> Result<()> {
        let Some(c) = self.candidates.get_mut(id) else {
            bail!("no such candidate: {id}");
        };
        // Running mean, so a candidate's centroid tracks the voice rather
        // than being pinned to whatever the first segment sounded like.
        let n = c.observations as f32;
        for (i, x) in embedding.iter().enumerate().take(c.centroid.len()) {
            c.centroid[i] = (c.centroid[i] * n + x) / (n + 1.0);
        }
        c.observations += 1;
        c.stats.speech_seconds += speech_ms / 1000;
        if !c.stats.days_seen.contains(&day) {
            c.stats.days_seen.push(day);
        }
        if !sample_text.trim().is_empty() && c.stats.samples.len() < MAX_SAMPLES {
            c.stats.samples.push(sample_text.to_string());
        }

        let dir = self.dir.join(id);
        std::fs::write(dir.join("centroid.emb"), encode_embedding(&c.centroid))?;
        std::fs::write(dir.join("stats.json"), serde_json::to_vec_pretty(&c.stats)?)?;
        Ok(())
    }

    pub fn list(&self) -> Vec<&Candidate> {
        let mut out: Vec<&Candidate> = self.candidates.values().collect();
        out.sort_by(|a, b| b.stats.speech_seconds.cmp(&a.stats.speech_seconds));
        out
    }

    /// Look up a single candidate by id.
    // Consumed by the later worker/tool wiring task, which matches
    // segments against candidates as well as registered speakers. Delete
    // this attribute once that task adds a caller.
    #[allow(dead_code)]
    pub fn get(&self, id: &str) -> Option<&Candidate> {
        self.candidates.get(id)
    }

    /// Both thresholds must be cleared. Cumulative time alone would promote
    /// a television left on all afternoon; distinct days alone would promote
    /// a passing greeting heard twice.
    pub fn is_promotable(&self, id: &str, after_seconds: u32, after_days: u32) -> bool {
        self.candidates.get(id).is_some_and(|c| {
            c.stats.speech_seconds >= after_seconds
                && c.stats.days_seen.len() as u32 >= after_days
        })
    }

    /// Export a candidate into `voices/<name>/` and stop tracking it.
    /// Returns the directory name used.
    pub fn promote(&mut self, id: &str, name: Option<&str>, voices_dir: &Path) -> Result<String> {
        let Some(c) = self.candidates.get(id) else {
            bail!("no such candidate: {id}");
        };
        let dir_name = match name {
            Some(n) => {
                validate_speaker_name(n)?;
                n.to_string()
            }
            None => c.id.clone(),
        };
        let target = voices_dir.join(&dir_name);
        std::fs::create_dir_all(&target)
            .with_context(|| format!("creating {target:?}"))?;
        std::fs::copy(self.dir.join(id).join("clip.wav"), target.join("clip.wav"))
            .with_context(|| format!("copying candidate clip into {target:?}"))?;

        std::fs::remove_dir_all(self.dir.join(id)).ok();
        self.candidates.remove(id);
        Ok(dir_name)
    }
}

/// A speaker name becomes a directory name under the workspace, so it must
/// be a single harmless path segment.
fn validate_speaker_name(name: &str) -> Result<()> {
    if name.trim().is_empty() {
        bail!("speaker name must not be empty");
    }
    if name.contains('/') || name.contains('\\') || name.contains("..") {
        bail!("speaker name must be a single path segment: {name:?}");
    }
    Ok(())
}

fn sample_vec(text: &str) -> Vec<String> {
    if text.trim().is_empty() {
        Vec::new()
    } else {
        vec![text.to_string()]
    }
}

fn load_candidate(dir: &Path, id: &str) -> Result<Candidate> {
    let centroid = decode_embedding(&std::fs::read(dir.join("centroid.emb"))?);
    let stats: CandidateStats = serde_json::from_slice(&std::fs::read(dir.join("stats.json"))?)?;
    Ok(Candidate {
        id: id.to_string(),
        centroid,
        observations: stats.days_seen.len().max(1) as u32,
        stats,
    })
}

fn write_wav(path: &Path, pcm: &[i16]) -> Result<()> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: PIPELINE_SAMPLE_RATE,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut w = hound::WavWriter::create(path, spec)?;
    for s in pcm {
        w.write_sample(*s)?;
    }
    w.finalize()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::NaiveDate;

    fn day(d: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(2026, 8, d).unwrap()
    }

    fn store() -> (tempfile::TempDir, CandidateStore) {
        let tmp = tempfile::tempdir().unwrap();
        let store = CandidateStore::open(tmp.path().join("candidates")).unwrap();
        (tmp, store)
    }

    #[test]
    fn enrolling_mints_an_id_and_persists_the_candidate() {
        let (tmp, mut store) = store();
        let id = store
            .enrol(vec![1.0, 0.0], &vec![0i16; 16_000], day(26), 1_000, "hello")
            .unwrap();
        assert!(!id.is_empty());
        assert!(tmp.path().join("candidates").join(&id).join("clip.wav").exists());
        assert!(tmp.path().join("candidates").join(&id).join("stats.json").exists());

        // Reload from disk: candidates survive a restart.
        let reloaded = CandidateStore::open(tmp.path().join("candidates")).unwrap();
        assert_eq!(reloaded.list().len(), 1);
        assert_eq!(reloaded.list()[0].id, id);
    }

    #[test]
    fn is_not_promotable_at_fifty_nine_seconds() {
        let (_tmp, mut store) = store();
        let id = store
            .enrol(vec![1.0, 0.0], &vec![0i16; 16_000], day(26), 59_000, "a")
            .unwrap();
        store.observe(&id, &[1.0, 0.0], day(27), 0, "b").unwrap();
        assert!(!store.is_promotable(&id, 60, 2), "59s < 60s threshold");
    }

    #[test]
    fn is_promotable_at_exactly_sixty_seconds_across_two_days() {
        let (_tmp, mut store) = store();
        let id = store
            .enrol(vec![1.0, 0.0], &vec![0i16; 16_000], day(26), 30_000, "a")
            .unwrap();
        store.observe(&id, &[1.0, 0.0], day(27), 30_000, "b").unwrap();
        assert!(store.is_promotable(&id, 60, 2));
    }

    #[test]
    fn is_not_promotable_on_a_single_day_however_long() {
        let (_tmp, mut store) = store();
        let id = store
            .enrol(vec![1.0, 0.0], &vec![0i16; 16_000], day(26), 600_000, "a")
            .unwrap();
        assert!(
            !store.is_promotable(&id, 60, 2),
            "one day of television must not promote"
        );
    }

    #[test]
    fn observing_the_same_day_twice_counts_it_once() {
        let (_tmp, mut store) = store();
        let id = store
            .enrol(vec![1.0, 0.0], &vec![0i16; 16_000], day(26), 40_000, "a")
            .unwrap();
        store.observe(&id, &[1.0, 0.0], day(26), 40_000, "b").unwrap();
        assert!(!store.is_promotable(&id, 60, 2), "still only one day seen");
    }

    #[test]
    fn promotion_writes_the_clip_into_the_workspace_under_the_id() {
        let (tmp, mut store) = store();
        let voices = tmp.path().join("voices");
        let id = store
            .enrol(vec![1.0, 0.0], &vec![0i16; 16_000], day(26), 60_000, "a")
            .unwrap();
        let name = store.promote(&id, None, &voices).unwrap();
        assert_eq!(name, id, "no name given, so the grain-id is the directory");
        assert!(voices.join(&id).join("clip.wav").exists());
        assert!(store.list().iter().all(|c| c.id != id), "promoted, so no longer a candidate");
    }

    #[test]
    fn promotion_with_a_name_uses_it_as_the_directory() {
        let (tmp, mut store) = store();
        let voices = tmp.path().join("voices");
        let id = store
            .enrol(vec![1.0, 0.0], &vec![0i16; 16_000], day(26), 60_000, "a")
            .unwrap();
        let name = store.promote(&id, Some("tanaka-san"), &voices).unwrap();
        assert_eq!(name, "tanaka-san");
        assert!(voices.join("tanaka-san").join("clip.wav").exists());
    }

    #[test]
    fn promotion_refuses_a_name_that_would_escape_the_voices_directory() {
        let (tmp, mut store) = store();
        let voices = tmp.path().join("voices");
        let id = store
            .enrol(vec![1.0, 0.0], &vec![0i16; 16_000], day(26), 60_000, "a")
            .unwrap();
        assert!(store.promote(&id, Some("../../etc/passwd"), &voices).is_err());
        assert!(store.promote(&id, Some("has/slash"), &voices).is_err());
    }
}
