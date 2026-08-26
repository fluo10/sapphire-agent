//! Resolves the ambient subsystem's two ONNX models (Silero VAD re-gate,
//! speaker embedding) from config, and derives the cache-key model id from
//! the embedding model that was actually loaded.
//!
//! The model id matters more than it looks: reference-audio embeddings are
//! cached outside the workspace keyed by (file sha256 x model id), so that
//! swapping the embedding model recomputes every cached vector rather than
//! silently reusing vectors computed by a different model. See
//! `docs/superpowers/specs/2026-08-26-ambient-audio-ingest-design.md`.

use std::fs;
use std::path::{Path, PathBuf};

use crate::config::AmbientConfig;
use crate::image_cache::sha256_hex;
use crate::voice::providers::sherpa_download::{BundleCategory, ensure_bundle, pick_file};

/// Everything the ambient worker needs to re-gate and attribute speech.
// Constructed only by the later config-wiring task that reads `[ambient]`
// and starts the worker. Delete this attribute once that task adds a caller.
#[allow(dead_code)]
pub struct ResolvedModels {
    pub gate: Box<dyn crate::ambient::audio::SpeechGate>,
    pub embedder: Box<dyn crate::ambient::audio::SpeakerEmbedder>,
    /// Identifies the embedding model for the embedding cache key.
    pub model_id: String,
}

/// Resolve either an explicit `model_dir` or a bundle `model` name to one
/// model file inside the resolved directory. Pure aside from filesystem
/// reads — no sherpa-onnx involved, so this is testable without the
/// `voice-sherpa` feature.
// Outside tests, called only from `resolve` below, which the later
// config-wiring task is what actually calls. Delete this attribute once
// that task adds a caller.
#[allow(dead_code)]
pub fn resolve_model_file(
    model: Option<&str>,
    model_dir: Option<&str>,
    category: BundleCategory,
    candidates: &[&str],
) -> anyhow::Result<PathBuf> {
    let dir = ensure_bundle(model_dir, model, category)?;
    pick_file(&dir, candidates)
}

/// A stable short id for a model file, for use as (part of) an embedding
/// cache key. Depends only on the file's contents, not its name or path,
/// so relocating a model directory never invalidates cached vectors.
// Outside tests, called only from `resolve` below, which the later
// config-wiring task is what actually calls. Delete this attribute once
// that task adds a caller.
#[allow(dead_code)]
pub fn model_id_for(path: &Path) -> anyhow::Result<String> {
    let bytes = fs::read(path)
        .map_err(|e| anyhow::anyhow!("failed to read model file '{}': {e}", path.display()))?;
    let full = sha256_hex(&bytes);
    Ok(full[..16].to_string())
}

// Constructed only by the later config-wiring task that reads `[ambient]`
// and starts the worker. Delete this attribute once that task adds a caller.
#[allow(dead_code)]
#[cfg(feature = "voice-sherpa")]
pub fn resolve(cfg: &AmbientConfig) -> anyhow::Result<ResolvedModels> {
    use crate::ambient::audio::{SherpaEmbedder, SileroGate};

    let vad_path = resolve_model_file(
        cfg.vad_model.as_deref(),
        cfg.vad_model_dir.as_deref(),
        BundleCategory::Vad,
        &["silero_vad.onnx", "model.onnx"],
    )?;
    let embedding_path = resolve_model_file(
        cfg.embedding_model.as_deref(),
        cfg.embedding_model_dir.as_deref(),
        BundleCategory::SpeakerEmbedding,
        &["model.onnx", "model.int8.onnx"],
    )?;

    // Derived from the embedding model only: the VAD model never touches a
    // cached vector, so it must never perturb the cache key.
    let model_id = model_id_for(&embedding_path)?;

    let gate = SileroGate::new(
        vad_path.to_string_lossy().into_owned(),
        cfg.vad_threshold,
    )?;
    let embedder = SherpaEmbedder::new(
        embedding_path.to_string_lossy().into_owned(),
        cfg.embedding_num_threads,
    )?;

    Ok(ResolvedModels {
        gate: Box::new(gate),
        embedder: Box::new(embedder),
        model_id,
    })
}

// Constructed only by the later config-wiring task that reads `[ambient]`
// and starts the worker. Delete this attribute once that task adds a caller.
#[allow(dead_code)]
#[cfg(not(feature = "voice-sherpa"))]
pub fn resolve(_cfg: &AmbientConfig) -> anyhow::Result<ResolvedModels> {
    anyhow::bail!(
        "ambient capture needs the `voice-sherpa` cargo feature for speech gating and \
         speaker attribution; rebuild with it or set [ambient].enabled = false"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write(dir: &std::path::Path, name: &str, bytes: &[u8]) -> PathBuf {
        std::fs::create_dir_all(dir).unwrap();
        let p = dir.join(name);
        std::fs::write(&p, bytes).unwrap();
        p
    }

    #[test]
    fn resolve_model_file_prefers_an_explicit_dir_over_a_bundle_name() {
        let tmp = tempfile::tempdir().unwrap();
        write(tmp.path(), "model.onnx", b"weights");
        let got = resolve_model_file(
            Some("some-bundle-that-would-be-downloaded"),
            Some(tmp.path().to_str().unwrap()),
            BundleCategory::SpeakerEmbedding,
            &["model.onnx"],
        )
        .unwrap();
        assert_eq!(got, tmp.path().join("model.onnx"));
    }

    #[test]
    fn resolve_model_file_takes_the_first_candidate_that_exists() {
        let tmp = tempfile::tempdir().unwrap();
        write(tmp.path(), "model.int8.onnx", b"quantised");
        let got = resolve_model_file(
            None,
            Some(tmp.path().to_str().unwrap()),
            BundleCategory::SpeakerEmbedding,
            &["model.onnx", "model.int8.onnx"],
        )
        .unwrap();
        assert_eq!(got, tmp.path().join("model.int8.onnx"));
    }

    #[test]
    fn resolve_model_file_errors_naming_the_directory_when_nothing_matches() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(tmp.path()).unwrap();
        let err = resolve_model_file(
            None,
            Some(tmp.path().to_str().unwrap()),
            BundleCategory::SpeakerEmbedding,
            &["model.onnx"],
        )
        .unwrap_err()
        .to_string();
        assert!(
            err.contains(&tmp.path().display().to_string()),
            "error should name the directory searched: {err}"
        );
    }

    #[test]
    fn resolve_model_file_errors_when_neither_model_nor_dir_is_set() {
        assert!(
            resolve_model_file(None, None, BundleCategory::SpeakerEmbedding, &["model.onnx"])
                .is_err()
        );
    }

    #[test]
    fn model_id_is_stable_for_identical_bytes_and_differs_for_different_bytes() {
        let tmp = tempfile::tempdir().unwrap();
        let a = write(tmp.path(), "a.onnx", b"weights-v1");
        let b = write(tmp.path(), "b.onnx", b"weights-v1");
        let c = write(tmp.path(), "c.onnx", b"weights-v2");

        let ida = model_id_for(&a).unwrap();
        assert_eq!(ida, model_id_for(&b).unwrap(), "same bytes, same id");
        assert_ne!(ida, model_id_for(&c).unwrap(), "different bytes, different id");
        assert_eq!(ida.len(), 16, "short enough to read in a filename");
        assert!(ida.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn model_id_does_not_depend_on_the_file_name_or_path() {
        // The id keys an embedding cache. If it varied with the path, moving the
        // model directory would silently invalidate every cached vector.
        let tmp = tempfile::tempdir().unwrap();
        let a = write(&tmp.path().join("one"), "model.onnx", b"same");
        let b = write(&tmp.path().join("two"), "other-name.onnx", b"same");
        assert_eq!(model_id_for(&a).unwrap(), model_id_for(&b).unwrap());
    }
}
