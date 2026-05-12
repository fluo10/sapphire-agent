//! Wake-word detection via sherpa-onnx's keyword spotter.
//!
//! Wraps `KeywordSpotter` + `OnlineStream` into a single-utterance API
//! suitable for the satellite's `listen_loop`: feed audio chunks via
//! [`WakeDetector::feed`], receive `Some(keyword)` when a configured
//! phrase is detected. Reset between phrases with [`WakeDetector::reset`].

use std::collections::HashSet;
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

/// Where the configured keywords come from. The satellite preferes
/// a server-supplied natural-language phrase (tokenised against the
/// bundle's `tokens.txt`); operators can override with a raw
/// sherpa-onnx keywords file for unusual setups.
pub enum WakeKeywordSource<'a> {
    /// One or more natural-language phrases. Each is split into
    /// characters and looked up in the bundle's `tokens.txt`. Failure
    /// surfaces with the missing characters listed so the user can
    /// pick a different phrase or a model whose vocab covers them.
    Phrases(&'a [String]),
    /// Path to a pre-tokenised sherpa-onnx keywords file. Bypasses
    /// the character-tokenisation step entirely; provided as an
    /// escape hatch for users running BPE-based models where
    /// character splitting isn't appropriate.
    File(&'a Path),
    /// Use the bundle's bundled `keywords.txt` (legacy default).
    BundleDefault,
}

impl WakeDetector {
    /// Build a keyword spotter from a sherpa-onnx KWS bundle.
    ///
    /// The bundle is auto-downloaded from the `kws-models` release
    /// tag on first use. `source` selects how the spotter learns
    /// what to listen for:
    ///   * `Phrases` — natural language, tokenised here
    ///   * `File` — pre-tokenised file
    ///   * `BundleDefault` — fall back to `<bundle>/keywords.txt`
    pub fn create(bundle: &str, source: WakeKeywordSource<'_>) -> Result<Self> {
        let dir = ensure_bundle(bundle, BundleCategory::Kws)?;

        let encoder = find_by_substring(&dir, "encoder")?;
        let decoder = find_by_substring(&dir, "decoder")?;
        let joiner = find_by_substring(&dir, "joiner")?;
        let tokens_path = pick_file(&dir, "tokens.txt")?;

        let mut config = KeywordSpotterConfig::default();
        config.model_config = OnlineModelConfig {
            transducer: OnlineTransducerModelConfig {
                encoder: Some(encoder.to_string_lossy().into_owned()),
                decoder: Some(decoder.to_string_lossy().into_owned()),
                joiner: Some(joiner.to_string_lossy().into_owned()),
            },
            tokens: Some(tokens_path.to_string_lossy().into_owned()),
            provider: Some("cpu".into()),
            num_threads: 1,
            ..Default::default()
        };

        match source {
            WakeKeywordSource::Phrases(phrases) => {
                let vocab = load_tokens_vocab(&tokens_path)?;
                let buf = tokenise_phrases(phrases, &vocab)?;
                config.keywords_buf = Some(buf);
            }
            WakeKeywordSource::File(path) => {
                let expanded = shellexpand::tilde(&path.to_string_lossy()).into_owned();
                config.keywords_file = Some(expanded);
            }
            WakeKeywordSource::BundleDefault => {
                let p = pick_file(&dir, "keywords.txt")?;
                config.keywords_file = Some(p.to_string_lossy().into_owned());
            }
        }

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

/// Read the model's `tokens.txt` into a set of valid token strings.
/// Format: `<token> <id>` per line, whitespace-delimited.
fn load_tokens_vocab(path: &Path) -> Result<HashSet<String>> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| anyhow!("failed to read tokens.txt at {}: {e}", path.display()))?;
    let mut vocab = HashSet::new();
    for line in content.lines() {
        if let Some(token) = line.split_whitespace().next() {
            vocab.insert(token.to_string());
        }
    }
    if vocab.is_empty() {
        anyhow::bail!("tokens.txt at {} is empty", path.display());
    }
    Ok(vocab)
}

/// Tokenise one or more natural-language phrases into sherpa-onnx's
/// `keywords` buffer format: one line per phrase, each line
/// `<tok1> <tok2> ... @<label>` where tokens are characters drawn from
/// `vocab` and the label is the original phrase (with punctuation
/// kept, for display in the matched event).
///
/// Characters that are whitespace or common Japanese / ASCII
/// punctuation are silently dropped from the token list — they
/// shouldn't influence wake detection. Anything else missing from
/// `vocab` fails the whole call with the offenders listed.
fn tokenise_phrases(phrases: &[String], vocab: &HashSet<String>) -> Result<String> {
    if phrases.is_empty() {
        anyhow::bail!("wake_word: at least one phrase required");
    }
    let mut lines: Vec<String> = Vec::with_capacity(phrases.len());
    for phrase in phrases {
        let line = tokenise_one(phrase, vocab)?;
        lines.push(line);
    }
    // sherpa-onnx parses keywords_buf line by line; trailing newline
    // is required so the last line is treated as complete.
    Ok(format!("{}\n", lines.join("\n")))
}

fn tokenise_one(phrase: &str, vocab: &HashSet<String>) -> Result<String> {
    let trimmed = phrase.trim();
    if trimmed.is_empty() {
        anyhow::bail!("wake_word phrase is empty");
    }
    let mut tokens: Vec<String> = Vec::new();
    let mut missing: Vec<char> = Vec::new();
    for ch in trimmed.chars() {
        if is_skippable(ch) {
            continue;
        }
        let s = ch.to_string();
        if vocab.contains(&s) {
            tokens.push(s);
        } else {
            missing.push(ch);
        }
    }
    if !missing.is_empty() {
        let unique: Vec<String> = {
            let mut seen = HashSet::new();
            missing
                .into_iter()
                .filter(|c| seen.insert(*c))
                .map(|c| format!("'{c}'"))
                .collect()
        };
        anyhow::bail!(
            "wake_word phrase '{phrase}' contains {} character(s) not in the KWS model's vocab: {}.\n\
             Either pick a phrase the model can spell, or choose a wake_word_model whose tokens.txt \
             covers your language.",
            unique.len(),
            unique.join(", ")
        );
    }
    if tokens.is_empty() {
        anyhow::bail!("wake_word phrase '{phrase}' became empty after stripping punctuation");
    }
    // Label keeps the punctuation so the matched event prints
    // recognisably (e.g. "ハロー、クロード" not "ハロークロード").
    Ok(format!("{} @{}", tokens.join(" "), trimmed))
}

/// Skip whitespace + common Japanese / ASCII punctuation when
/// building the token list. The label keeps the original phrase, so
/// these never appear in the user-facing match text.
fn is_skippable(ch: char) -> bool {
    ch.is_whitespace()
        || matches!(
            ch,
            '、' | '。' | '・' | '！' | '？' | '!' | '?' | ',' | '.' | '「' | '」'
        )
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

#[cfg(test)]
mod tests {
    use super::*;

    fn vocab(chars: &[char]) -> HashSet<String> {
        chars.iter().map(|c| c.to_string()).collect()
    }

    #[test]
    fn tokenise_simple_phrase() {
        let v = vocab(&['ク', 'ロ', 'ー', 'ド']);
        let out = tokenise_phrases(&["クロード".to_string()], &v).unwrap();
        assert_eq!(out, "ク ロ ー ド @クロード\n");
    }

    #[test]
    fn tokenise_keeps_punctuation_in_label_but_drops_from_tokens() {
        let v = vocab(&['ハ', 'ロ', 'ー', 'ク', 'ド']);
        let out = tokenise_phrases(&["ハロー、クロード".to_string()], &v).unwrap();
        // ロ ー appears twice; comma dropped from token list but kept
        // in the @label suffix.
        assert!(
            out.starts_with("ハ ロ ー ク ロ ー ド @"),
            "got: {out:?}"
        );
        assert!(out.contains("@ハロー、クロード"));
    }

    #[test]
    fn tokenise_reports_missing_characters() {
        let v = vocab(&['ク', 'ロ', 'ー', 'ド']);
        // ハ is missing.
        let err = tokenise_phrases(&["ハロー、クロード".to_string()], &v).unwrap_err();
        let s = format!("{err}");
        assert!(s.contains("'ハ'"), "expected missing char in error, got: {s}");
    }

    #[test]
    fn tokenise_multiple_phrases() {
        let v = vocab(&['ハ', 'ロ', 'ー', 'ク', 'ド']);
        let out = tokenise_phrases(
            &[
                "クロード".to_string(),
                "ハロー、クロード".to_string(),
            ],
            &v,
        )
        .unwrap();
        let lines: Vec<&str> = out.trim_end_matches('\n').lines().collect();
        assert_eq!(lines.len(), 2);
    }

    #[test]
    fn tokenise_rejects_empty_input() {
        let v = vocab(&['a']);
        assert!(tokenise_phrases(&[], &v).is_err());
        assert!(tokenise_phrases(&["".to_string()], &v).is_err());
        assert!(tokenise_phrases(&["、。!".to_string()], &v).is_err());
    }
}
