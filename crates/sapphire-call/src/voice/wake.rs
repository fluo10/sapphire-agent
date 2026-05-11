//! Wake-word detection via sherpa-onnx's keyword spotter.
//!
//! Wraps `KeywordSpotter` + `OnlineStream` into a single-utterance API
//! suitable for the satellite's `listen_loop`: feed audio chunks via
//! [`WakeDetector::feed`], receive `Some(keyword)` when a configured
//! phrase is detected. Reset between phrases with [`WakeDetector::reset`].

use std::path::Path;

use anyhow::{Result, anyhow};
use sherpa_onnx::{
    KeywordSpotter, KeywordSpotterConfig, OnlineModelConfig, OnlineStream,
    OnlineTransducerModelConfig,
};

use super::download::{BundleCategory, ensure_bundle};
use sapphire_agent_api::voice::PIPELINE_SAMPLE_RATE;

pub struct WakeDetector {
    spotter: KeywordSpotter,
    stream: OnlineStream,
}

// SAFETY: same reasoning as the server-side sherpa wrappers — single
// owner, never expose the raw pointer, sherpa-onnx C++ is thread-safe.
unsafe impl Send for WakeDetector {}
unsafe impl Sync for WakeDetector {}

impl WakeDetector {
    /// Build a keyword spotter from a sherpa-onnx KWS bundle. The
    /// bundle is auto-downloaded from the `kws-models` release tag if
    /// not already in the cache. `keywords_file_override` lets the
    /// caller substitute a different keywords file (otherwise the
    /// bundle's own `keywords.txt` is used).
    pub fn create(bundle: &str, keywords_file_override: Option<&str>) -> Result<Self> {
        let dir = ensure_bundle(bundle, BundleCategory::Kws)?;

        let encoder = find_by_substring(&dir, "encoder")?;
        let decoder = find_by_substring(&dir, "decoder")?;
        let joiner = find_by_substring(&dir, "joiner")?;
        let tokens = pick_file(&dir, "tokens.txt")?;
        let keywords = match keywords_file_override {
            Some(path) => std::path::PathBuf::from(shellexpand::tilde(path).into_owned()),
            None => pick_file(&dir, "keywords.txt")?,
        };

        let mut config = KeywordSpotterConfig::default();
        config.model_config = OnlineModelConfig {
            transducer: OnlineTransducerModelConfig {
                encoder: Some(encoder.to_string_lossy().into_owned()),
                decoder: Some(decoder.to_string_lossy().into_owned()),
                joiner: Some(joiner.to_string_lossy().into_owned()),
            },
            tokens: Some(tokens.to_string_lossy().into_owned()),
            provider: Some("cpu".into()),
            num_threads: 1,
            ..Default::default()
        };
        config.keywords_file = Some(keywords.to_string_lossy().into_owned());

        let spotter = KeywordSpotter::create(&config)
            .ok_or_else(|| anyhow!("KeywordSpotter::create returned None"))?;
        let stream = spotter.create_stream();
        Ok(Self { spotter, stream })
    }

    /// Feed PCM audio (f32 normalised, 16 kHz mono). Returns the
    /// detected keyword string when a match fires this call.
    ///
    /// After a hit the detector is automatically reset, so callers
    /// don't need to call [`Self::reset`] inside the same audio chunk —
    /// only between distinct conversation cycles (after the agent
    /// finishes replying).
    pub fn feed(&mut self, samples_f32: &[f32]) -> Result<Option<String>> {
        if samples_f32.is_empty() {
            return Ok(None);
        }
        self.stream
            .accept_waveform(PIPELINE_SAMPLE_RATE as i32, samples_f32);
        let mut detected: Option<String> = None;
        while self.spotter.is_ready(&self.stream) {
            self.spotter.decode(&self.stream);
            if let Some(result) = self.spotter.get_result(&self.stream) {
                if !result.keyword.is_empty() {
                    detected = Some(result.keyword.clone());
                    // sherpa-onnx requires an explicit reset between
                    // hits to begin scanning for the next one.
                    self.spotter.reset(&self.stream);
                }
            }
        }
        Ok(detected)
    }

    /// Drop the in-flight audio buffer and return to scratch state.
    /// Called after each user→agent cycle to prevent the next wake
    /// listen from triggering on residue.
    pub fn reset(&mut self) {
        // Recreate the stream — sherpa-onnx has no full-buffer flush.
        self.stream = self.spotter.create_stream();
    }
}

fn pick_file(dir: &Path, name: &str) -> Result<std::path::PathBuf> {
    let p = dir.join(name);
    if p.exists() {
        Ok(p)
    } else {
        Err(anyhow!("expected file '{name}' not found in {}", dir.display()))
    }
}

fn find_by_substring(dir: &Path, needle: &str) -> Result<std::path::PathBuf> {
    // Prefer int8 variants when both ship.
    let int8_candidate = find_by_predicate(dir, |s| {
        s.contains(needle) && s.ends_with(".int8.onnx")
    });
    if let Ok(p) = int8_candidate {
        return Ok(p);
    }
    find_by_predicate(dir, |s| s.contains(needle) && s.ends_with(".onnx"))
}

fn find_by_predicate(
    dir: &Path,
    pred: impl Fn(&str) -> bool,
) -> Result<std::path::PathBuf> {
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let s = entry.file_name().to_string_lossy().into_owned();
        if pred(&s) {
            return Ok(entry.path());
        }
    }
    Err(anyhow!(
        "no matching file in {} (predicate)",
        dir.display()
    ))
}
