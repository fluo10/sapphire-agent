//! Per-installation device identifier.
//!
//! The voice satellite needs a stable handle the server can map to a
//! conversation thread (since voice doesn't carry an explicit session
//! id). Generated once on first run as a UUID v7, persisted under the
//! XDG data dir (`~/.local/share/sapphire-call/device-id` on Linux).
//!
//! Plain-text storage is intentional — this isn't a secret, just a
//! routing key. Anyone with shell access to the box already has full
//! control over the satellite.

use std::path::PathBuf;

use anyhow::{Context, Result};
use uuid::Uuid;

/// Read the device id from disk, generating + persisting a fresh UUID v7
/// on first invocation. The returned string is the canonical UUID
/// form (`xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx`).
pub fn ensure_device_id() -> Result<String> {
    let path = path()?;
    if path.exists() {
        let raw = std::fs::read_to_string(&path)
            .with_context(|| format!("read {}", path.display()))?;
        let id = raw.trim().to_string();
        if !id.is_empty() {
            return Ok(id);
        }
        // Empty file — fall through to regenerate.
    }
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("create dir {}", parent.display()))?;
    }
    let id = Uuid::now_v7().to_string();
    let tmp = path.with_extension("partial");
    std::fs::write(&tmp, &id)
        .with_context(|| format!("write {}", tmp.display()))?;
    std::fs::rename(&tmp, &path)
        .with_context(|| format!("rename {} → {}", tmp.display(), path.display()))?;
    Ok(id)
}

/// XDG path where the device id is stored. Override via the
/// `SAPPHIRE_CALL_DEVICE_ID_PATH` env var (used by tests + container
/// deployments).
fn path() -> Result<PathBuf> {
    if let Ok(custom) = std::env::var("SAPPHIRE_CALL_DEVICE_ID_PATH") {
        return Ok(PathBuf::from(shellexpand::tilde(&custom).into_owned()));
    }
    if let Some(dirs) = directories::ProjectDirs::from("", "", "sapphire-call") {
        return Ok(dirs.data_local_dir().join("device-id"));
    }
    anyhow::bail!("no XDG data dir available — set SAPPHIRE_CALL_DEVICE_ID_PATH")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn idempotent_across_calls() {
        // Use a fresh temp file so we don't trample the developer's
        // real device-id.
        let tmp = tempfile::TempDir::new().unwrap();
        let path = tmp.path().join("device-id");
        // SAFETY: tests run single-threaded by default with cargo
        // test; if multi-threaded test introduces races, switch to
        // a per-test env-isolation helper.
        unsafe {
            std::env::set_var("SAPPHIRE_CALL_DEVICE_ID_PATH", &path);
        }
        let first = ensure_device_id().unwrap();
        let second = ensure_device_id().unwrap();
        assert_eq!(first, second);
        // UUID format sanity.
        assert_eq!(first.len(), 36);
        unsafe {
            std::env::remove_var("SAPPHIRE_CALL_DEVICE_ID_PATH");
        }
    }
}
