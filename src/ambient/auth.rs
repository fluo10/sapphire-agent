//! Device authentication: bearer token -> key file entry -> device name.
//!
//! The token lives only in `sapphire-framework`'s key file. `config.toml`
//! names a `key_id`, so a config that leaks reveals no credential, and
//! revoking a device is a key-file edit rather than an agent restart.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use sapphire_framework::remote_server::KeyStore;

use crate::config::DeviceConfig;

/// Resolves a presented bearer token to the name of the device it belongs
/// to. Cheap to share behind an `Arc`; holds no mutable state.
#[derive(Debug)]
pub struct DeviceRegistry {
    keys: KeyStore,
    /// `KeyEntry::id` -> the `[device.<name>]` key.
    by_key_id: HashMap<uuid::Uuid, String>,
}

impl DeviceRegistry {
    /// Load the key file and index the configured devices by key id.
    ///
    /// A missing key file is an error rather than an empty store.
    /// `KeyStore::load` treats absence as "no keys", which would leave
    /// ambient ingest running and rejecting every device forever — a
    /// misconfiguration that looks exactly like a broken device.
    pub fn open(keys_file: &Path, devices: &HashMap<String, DeviceConfig>) -> Result<Self> {
        let keys = KeyStore::load(keys_file)
            .with_context(|| format!("loading key file {}", keys_file.display()))?;
        if !keys.has_usable_key() {
            bail!(
                "key file {} has no usable key; ambient ingest would reject every device",
                keys_file.display()
            );
        }
        let by_key_id = devices
            .iter()
            .map(|(name, cfg)| (cfg.key_id, name.clone()))
            .collect();
        Ok(Self { keys, by_key_id })
    }

    /// Device name for `token`, or `None`.
    ///
    /// Three distinct failures collapse to `None` on purpose — the caller
    /// answers 401 for all of them and logs the distinction rather than
    /// returning it. `KeyStore::authenticate` already does the
    /// constant-time comparison and the `expires_at` check.
    pub fn resolve(&self, token: &str) -> Option<&str> {
        let entry = self.keys.authenticate(token)?;
        self.by_key_id.get(&entry.id).map(String::as_str)
    }

    /// Default key file location, used when `[keys].file` is unset.
    pub fn default_key_file() -> Option<PathBuf> {
        dirs::config_dir().map(|d| d.join("sapphire-agent").join("keys.toml"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    /// Write a key file and return (path, the KeyEntry ids by label).
    fn key_file(dir: &std::path::Path, keys: &[(&str, &str, Option<&str>)]) -> PathBuf {
        // keys: (token, label, expires_at RFC3339)
        let mut body = String::new();
        for (token, label, expires) in keys {
            body.push_str("[[key]]\n");
            body.push_str(&format!("token = \"{token}\"\n"));
            body.push_str(&format!("label = \"{label}\"\n"));
            if let Some(e) = expires {
                body.push_str(&format!("expires_at = \"{e}\"\n"));
            }
            body.push('\n');
        }
        let path = dir.join("keys.toml");
        std::fs::write(&path, body).unwrap();
        path
    }

    fn id_of(path: &std::path::Path, label: &str) -> uuid::Uuid {
        let store = sapphire_framework::remote_server::KeyStore::load(path).unwrap();
        store
            .entries()
            .iter()
            .find(|e| e.label.as_deref() == Some(label))
            .expect("label present")
            .id
    }

    fn devices(pairs: &[(&str, uuid::Uuid)]) -> HashMap<String, crate::config::DeviceConfig> {
        pairs
            .iter()
            .map(|(name, id)| {
                (
                    name.to_string(),
                    crate::config::DeviceConfig {
                        key_id: *id,
                        label: None,
                        room_profile: None,
                    },
                )
            })
            .collect()
    }

    #[test]
    fn resolves_a_bound_token_to_its_device_name() {
        let tmp = tempfile::tempdir().unwrap();
        let path = key_file(tmp.path(), &[("sa-dev-good", "pendant-key", None)]);
        let id = id_of(&path, "pendant-key");
        let reg = DeviceRegistry::open(&path, &devices(&[("pendant", id)])).unwrap();
        assert_eq!(reg.resolve("sa-dev-good"), Some("pendant"));
    }

    #[test]
    fn rejects_a_token_absent_from_the_key_file() {
        let tmp = tempfile::tempdir().unwrap();
        let path = key_file(tmp.path(), &[("sa-dev-good", "pendant-key", None)]);
        let id = id_of(&path, "pendant-key");
        let reg = DeviceRegistry::open(&path, &devices(&[("pendant", id)])).unwrap();
        assert_eq!(reg.resolve("sa-dev-nope"), None);
    }

    #[test]
    fn rejects_an_expired_token_even_though_it_matches() {
        let tmp = tempfile::tempdir().unwrap();
        let path = key_file(
            tmp.path(),
            &[("sa-dev-old", "pendant-key", Some("2020-01-01T00:00:00Z"))],
        );
        let id = id_of(&path, "pendant-key");
        let reg = DeviceRegistry::open(&path, &devices(&[("pendant", id)])).unwrap();
        assert_eq!(reg.resolve("sa-dev-old"), None);
    }

    #[test]
    fn rejects_a_valid_token_bound_to_no_device() {
        let tmp = tempfile::tempdir().unwrap();
        let path = key_file(tmp.path(), &[("sa-dev-loose", "unbound", None)]);
        // No [device.*] references this key at all.
        let reg = DeviceRegistry::open(&path, &HashMap::new()).unwrap();
        assert_eq!(reg.resolve("sa-dev-loose"), None);
    }

    #[test]
    fn open_fails_when_the_key_file_is_missing() {
        let tmp = tempfile::tempdir().unwrap();
        let missing = tmp.path().join("absent.toml");
        let err = DeviceRegistry::open(&missing, &HashMap::new()).unwrap_err();
        assert!(
            err.to_string().contains("no usable key"),
            "unexpected error: {err}"
        );
    }
}
