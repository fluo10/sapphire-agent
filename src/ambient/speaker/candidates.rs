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
    /// Number of observations folded into the centroid, for the running
    /// mean. Persisted so a restart does not corrupt the weighting: this
    /// is deliberately distinct from `days_seen.len()`, which counts only
    /// distinct days and undercounts a voice heard several times in one
    /// afternoon. `#[serde(default)]` keeps this tolerant of any file
    /// that predates the field, though nothing on disk does yet.
    #[serde(default)]
    pub observations: u32,
}

#[derive(Debug, Clone)]
pub struct Candidate {
    pub id: String,
    pub centroid: Vec<f32>,
    pub stats: CandidateStats,
}

pub struct CandidateStore {
    dir: PathBuf,
    candidates: HashMap<String, Candidate>,
}

impl CandidateStore {
    /// Open the store, loading every candidate already on disk.
    pub fn open(dir: PathBuf) -> Result<Self> {
        std::fs::create_dir_all(&dir).with_context(|| format!("creating candidate dir {dir:?}"))?;
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
            observations: 1,
        };
        std::fs::write(dir.join("stats.json"), serde_json::to_vec_pretty(&stats)?)?;

        self.candidates.insert(
            id.clone(),
            Candidate {
                id: id.clone(),
                centroid: embedding,
                stats,
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
        // `observations` is persisted (see `CandidateStats`) precisely so
        // this weight survives a restart intact.
        let n = c.stats.observations as f32;
        for (i, x) in embedding.iter().enumerate().take(c.centroid.len()) {
            c.centroid[i] = (c.centroid[i] * n + x) / (n + 1.0);
        }
        c.stats.observations += 1;
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
    pub fn get(&self, id: &str) -> Option<&Candidate> {
        self.candidates.get(id)
    }

    /// Both thresholds must be cleared. Cumulative time alone would promote
    /// a television left on all afternoon; distinct days alone would promote
    /// a passing greeting heard twice.
    pub fn is_promotable(&self, id: &str, after_seconds: u32, after_days: u32) -> bool {
        self.candidates.get(id).is_some_and(|c| {
            c.stats.speech_seconds >= after_seconds && c.stats.days_seen.len() as u32 >= after_days
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

        // `voices_dir` is ours to create; canonicalise it once so the
        // eventual write always lands on a fully resolved, symlink-free
        // root rather than whatever relative form the caller passed in.
        std::fs::create_dir_all(voices_dir).with_context(|| format!("creating {voices_dir:?}"))?;
        let voices_root = voices_dir
            .canonicalize()
            .with_context(|| format!("resolving {voices_dir:?}"))?;

        // The character blocklist in `validate_speaker_name` is a guess
        // about this OS's path grammar, and it has already been wrong
        // twice on this same function: first backslash, then a bare
        // Windows drive prefix. Assert the actual invariant instead of
        // guessing again: join `dir_name` onto the un-canonicalised
        // `voices_dir`, exactly as this function would, and require the
        // join's parent to still be `voices_dir`. A component that
        // hijacks `Path::join` (a drive prefix, a UNC prefix, and so on)
        // replaces the whole path rather than appending to it, so the
        // parent comes out completely different; this catches that
        // regardless of which character caused it. It is also
        // side-effect free, since comparing `Path` values touches no
        // disk, so a rejected name never causes anything to be created
        // that would need cleaning up afterwards. Canonicalising the
        // target here instead would both require it to exist first,
        // which it does not yet, and risk creating -- and then having to
        // destroy -- whatever real location a hijacked join resolved to,
        // which for a bare drive prefix could be nowhere near
        // `voices_dir` at all.
        let target = voices_dir.join(&dir_name);
        if target.parent() != Some(voices_dir) {
            bail!("speaker name would escape the voices directory: {dir_name:?}");
        }
        let target = voices_root.join(&dir_name);

        let target_existed = target.exists();
        std::fs::create_dir_all(&target).with_context(|| format!("creating {target:?}"))?;
        if let Err(e) = std::fs::copy(self.dir.join(id).join("clip.wav"), target.join("clip.wav")) {
            // Don't leave a stray, empty directory ahead of a promotion
            // that did not actually happen -- but only if this call is
            // the one that just created it; a pre-existing target (an
            // overwrite) is not ours to delete.
            if !target_existed {
                std::fs::remove_dir_all(&target).ok();
            }
            return Err(e).with_context(|| format!("copying candidate clip into {target:?}"));
        }

        std::fs::remove_dir_all(self.dir.join(id)).ok();
        self.candidates.remove(id);
        Ok(dir_name)
    }
}

/// A speaker name becomes a directory name under the workspace, so it must
/// be a single harmless path segment.
fn validate_speaker_name(name: &str) -> Result<()> {
    let trimmed = name.trim();
    if trimmed.is_empty() || trimmed == "." || trimmed == ".." {
        bail!("speaker name must not be empty or a directory reference: {name:?}");
    }
    if name.contains('/') || name.contains('\\') || name.contains(':') || name.contains("..") {
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
        assert!(
            tmp.path()
                .join("candidates")
                .join(&id)
                .join("clip.wav")
                .exists()
        );
        assert!(
            tmp.path()
                .join("candidates")
                .join(&id)
                .join("stats.json")
                .exists()
        );

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
        store
            .observe(&id, &[1.0, 0.0], day(27), 30_000, "b")
            .unwrap();
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
        store
            .observe(&id, &[1.0, 0.0], day(26), 40_000, "b")
            .unwrap();
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
        assert!(
            store.list().iter().all(|c| c.id != id),
            "promoted, so no longer a candidate"
        );
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
        assert!(
            store
                .promote(&id, Some("../../etc/passwd"), &voices)
                .is_err()
        );
        assert!(store.promote(&id, Some("has/slash"), &voices).is_err());
    }

    /// A voice heard four times in one afternoon should weight its fourth
    /// observation as a fifth of the running mean, not half of it. That
    /// requires the observation count to survive a restart, not just the
    /// count of distinct days.
    #[test]
    fn reopening_preserves_the_observation_count_not_just_distinct_days() {
        let (tmp, mut store) = store();
        let id = store
            .enrol(vec![1.0, 0.0], &vec![0i16; 16_000], day(26), 10_000, "a")
            .unwrap();
        store
            .observe(&id, &[1.0, 0.0], day(26), 10_000, "b")
            .unwrap();
        store
            .observe(&id, &[1.0, 0.0], day(26), 10_000, "c")
            .unwrap();

        let reloaded = CandidateStore::open(tmp.path().join("candidates")).unwrap();
        let reloaded_candidate = reloaded.list().into_iter().find(|c| c.id == id).unwrap();
        assert_eq!(
            reloaded_candidate.stats.observations, 3,
            "three observations were folded in, even though they all land on one day"
        );
    }

    /// Windows accepts backslash as a path separator too, so a check that
    /// only rejects `/` would still let `..\..\` escape the voices
    /// directory on this OS.
    #[test]
    fn promotion_refuses_a_name_with_backslash_traversal() {
        let (tmp, mut store) = store();
        let voices = tmp.path().join("voices");
        let id = store
            .enrol(vec![1.0, 0.0], &vec![0i16; 16_000], day(26), 60_000, "a")
            .unwrap();
        assert!(
            store
                .promote(&id, Some("..\\..\\etc\\passwd"), &voices)
                .is_err()
        );
        assert!(store.promote(&id, Some("has\\backslash"), &voices).is_err());
    }

    /// On Windows, `PathBuf::join` treats a component with a drive prefix
    /// as absolute and *replaces* the base path instead of appending to
    /// it: `voices_dir.join("C:foo")` is `"C:foo"`, entirely outside
    /// `voices_dir`. Neither `/` nor `\` nor `..` catches this; only a
    /// dedicated rejection (and the join-parent containment check) does.
    #[test]
    fn promotion_refuses_a_drive_qualified_name() {
        let (tmp, mut store) = store();
        let voices = tmp.path().join("voices");
        let id = store
            .enrol(vec![1.0, 0.0], &vec![0i16; 16_000], day(26), 60_000, "a")
            .unwrap();
        assert!(store.promote(&id, Some("C:foo"), &voices).is_err());
        assert!(store.promote(&id, Some("C:"), &voices).is_err());
    }

    /// `voices_dir.join(".")` is `voices_dir` itself, so a bare "." would
    /// collapse every candidate promoted under that name into the voices
    /// root rather than a subdirectory of it. The join-parent containment
    /// check happens to reject this too -- Rust's `Path` normalises away
    /// a non-leading "." component, so `target.parent()` strips
    /// `voices_dir`'s own last component instead of the phantom ".", and
    /// comes out one level too high -- but that is an accident of how
    /// `Path::parent` is implemented, not a stated guarantee. A
    /// `starts_with`-style containment check would *not* catch it, since
    /// `voices_dir` trivially starts with itself. Keep this explicit
    /// rejection regardless, so the behaviour does not depend on which
    /// comparison the containment check happens to use.
    #[test]
    fn promotion_refuses_a_bare_dot_name() {
        let (tmp, mut store) = store();
        let voices = tmp.path().join("voices");
        let id = store
            .enrol(vec![1.0, 0.0], &vec![0i16; 16_000], day(26), 60_000, "a")
            .unwrap();
        assert!(store.promote(&id, Some("."), &voices).is_err());
        assert!(store.promote(&id, Some(".."), &voices).is_err());
    }

    /// `promote` creates the target directory before copying the clip into
    /// it. If the copy fails, that directory must not survive — a stray,
    /// empty `voices/<name>/` would be a partial workspace write ahead of
    /// a promotion that never actually happened.
    #[test]
    fn promotion_removes_a_freshly_created_directory_when_the_copy_fails() {
        let (tmp, mut store) = store();
        let voices = tmp.path().join("voices");
        let id = store
            .enrol(vec![1.0, 0.0], &vec![0i16; 16_000], day(26), 60_000, "a")
            .unwrap();
        // Knock out the candidate's own clip so the copy inside `promote`
        // fails partway through.
        std::fs::remove_file(store.dir.join(&id).join("clip.wav")).unwrap();

        assert!(store.promote(&id, Some("ghost"), &voices).is_err());
        assert!(
            !voices.join("ghost").exists(),
            "a failed promotion must not leave a stray directory behind"
        );
    }
}
