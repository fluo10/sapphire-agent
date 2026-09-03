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
