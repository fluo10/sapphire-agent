//! Auto-download sherpa-onnx model bundles for the voice satellite.
//!
//! Mirrors the server-side helper at
//! `sapphire-agent/src/voice/providers/sherpa_download.rs` but uses a
//! client-specific cache directory so the two binaries don't fight
//! each other. Bundles come from sherpa-onnx GitHub releases as
//! tar.bz2 archives.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use anyhow::Result;
use bzip2::read::BzDecoder;
use tracing::info;

#[allow(dead_code)]
#[derive(Debug, Clone, Copy)]
pub enum BundleCategory {
    Asr,
    Tts,
    Kws,
    Vad,
}

impl BundleCategory {
    fn tag(&self) -> &'static str {
        match self {
            BundleCategory::Asr => "asr-models",
            BundleCategory::Tts => "tts-models",
            BundleCategory::Kws => "kws-models",
            BundleCategory::Vad => "vad-models",
        }
    }
}

pub fn cache_dir() -> PathBuf {
    if let Ok(custom) = std::env::var("SAPPHIRE_CALL_CACHE_DIR") {
        return PathBuf::from(shellexpand::tilde(&custom).into_owned());
    }
    if let Some(dirs) = directories::ProjectDirs::from("", "", "sapphire-call") {
        return dirs.data_local_dir().join("voice-models");
    }
    PathBuf::from(".sapphire-call/voice-models")
}

/// Subdirectory under [`cache_dir`] that holds openWakeWord shared
/// frontend models (melspectrogram + embedding) and per-wake-word
/// custom classifier ONNXes hashed by SHA-256.
pub fn oww_cache_dir() -> PathBuf {
    cache_dir().join("oww")
}

/// openWakeWord frontend model release. Pinned so cached files don't
/// get invalidated by upstream version bumps.
const OWW_RELEASE_TAG: &str = "v0.5.1";
const OWW_MELSPEC_FILENAME: &str = "melspectrogram.onnx";
const OWW_EMBEDDING_FILENAME: &str = "embedding_model.onnx";

/// Make sure the openWakeWord melspectrogram and embedding ONNX
/// frontend models are present in the local cache, downloading from
/// the upstream GitHub release on first use. Returns the two
/// resolved paths.
pub fn ensure_oww_frontend() -> Result<(PathBuf, PathBuf)> {
    let dir = oww_cache_dir();
    fs::create_dir_all(&dir)?;
    let mel = ensure_single_file(
        &format!(
            "https://github.com/dscripka/openWakeWord/releases/download/{}/{}",
            OWW_RELEASE_TAG, OWW_MELSPEC_FILENAME
        ),
        &dir.join(OWW_MELSPEC_FILENAME),
    )?;
    let embed = ensure_single_file(
        &format!(
            "https://github.com/dscripka/openWakeWord/releases/download/{}/{}",
            OWW_RELEASE_TAG, OWW_EMBEDDING_FILENAME
        ),
        &dir.join(OWW_EMBEDDING_FILENAME),
    )?;
    Ok((mel, embed))
}

/// Write a server-supplied wake-word ONNX blob into the cache under
/// `<sha256>.onnx` and return the path. Re-uses the cached file when
/// the SHA already matches — the inline distribution path stays
/// idempotent across restarts.
pub fn cache_inline_oww(bytes: &[u8], sha256: &str) -> Result<PathBuf> {
    let dir = oww_cache_dir();
    fs::create_dir_all(&dir)?;
    let path = dir.join(format!("{sha256}.onnx"));
    if path.exists() {
        return Ok(path);
    }
    let tmp = path.with_extension("partial");
    fs::write(&tmp, bytes)?;
    fs::rename(&tmp, &path)?;
    Ok(path)
}

/// Download a single file (not a bundle) and place it at `dest`. Used
/// for the Silero VAD model, which ships as a bare .onnx on the VAD
/// releases tag.
pub fn ensure_single_file(url: &str, dest: &Path) -> Result<PathBuf> {
    if dest.exists() {
        return Ok(dest.to_path_buf());
    }
    if let Some(parent) = dest.parent() {
        fs::create_dir_all(parent)?;
    }
    info!("Downloading {url} → {} (one-time)", dest.display());
    let resp = reqwest::blocking::Client::builder()
        .timeout(None)
        .build()?
        .get(url)
        .send()?
        .error_for_status()?;
    let tmp = dest.with_extension("partial");
    let mut file = fs::File::create(&tmp)?;
    let mut reader = resp;
    io::copy(&mut reader, &mut file)?;
    fs::rename(&tmp, dest)?;
    info!("Saved to {}", dest.display());
    Ok(dest.to_path_buf())
}

/// Resolve `bundle` to an extracted directory under [`cache_dir`],
/// downloading + unpacking from the sherpa-onnx releases tag for
/// `category` on first use.
#[allow(dead_code)]
pub fn ensure_bundle(bundle: &str, category: BundleCategory) -> Result<PathBuf> {
    let dest = cache_dir().join(bundle);
    if dest.exists() {
        return Ok(dest);
    }
    let url = format!(
        "https://github.com/k2-fsa/sherpa-onnx/releases/download/{}/{}.tar.bz2",
        category.tag(),
        bundle
    );
    let cache = cache_dir();
    fs::create_dir_all(&cache)?;

    info!("Downloading sherpa-onnx bundle '{bundle}' from {url} (one-time)");
    let resp = reqwest::blocking::Client::builder()
        .timeout(None)
        .build()?
        .get(&url)
        .send()?
        .error_for_status()?;
    let bytes = resp.bytes()?;
    let decompressor = BzDecoder::new(io::Cursor::new(bytes));
    let mut archive = tar::Archive::new(decompressor);

    let partial = dest.with_extension("partial");
    if partial.exists() {
        fs::remove_dir_all(&partial)?;
    }
    fs::create_dir_all(&partial)?;

    for entry in archive.entries()? {
        let mut entry = entry?;
        let path = entry.path()?;
        let stripped: PathBuf = path.components().skip(1).collect();
        if stripped.as_os_str().is_empty() {
            continue;
        }
        let out_path = partial.join(stripped);
        if let Some(parent) = out_path.parent() {
            fs::create_dir_all(parent)?;
        }
        entry.unpack(&out_path)?;
    }

    fs::rename(&partial, &dest)?;
    info!("Bundle ready at {}", dest.display());
    Ok(dest)
}
