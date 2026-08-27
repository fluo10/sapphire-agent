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
pub struct DeviceRegistry {
    keys: KeyStore,
    /// `KeyEntry::id` -> the `[device.<name>]` key.
    by_key_id: HashMap<uuid::Uuid, String>,
}

// Hand-written rather than `#[derive(Debug)]`: `sapphire-framework` stores
// API tokens in `KeyStore` as plaintext by documented design, and currently
// has no `Debug` impl of its own. A derive here would compile today but
// silently start printing live bearer tokens the moment the framework ever
// grows one. Naming only the key-file-derived index forecloses that for
// good — this impl structurally cannot reach `keys`.
impl std::fmt::Debug for DeviceRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeviceRegistry")
            .field("by_key_id", &self.by_key_id)
            .finish()
    }
}

impl DeviceRegistry {
    /// Load the key file and index the configured devices by key id.
    ///
    /// A missing key file is an error rather than an empty store.
    /// `KeyStore::load` treats absence as "no keys", which would leave
    /// ambient ingest running and rejecting every device forever — a
    /// misconfiguration that looks exactly like a broken device.
    ///
    /// Two `[device.*]` blocks naming the same `key_id` is fatal for the
    /// same reason. One key per device is a stated requirement of the
    /// design — a shared key collides in the `segment` idempotency
    /// namespace — and indexing by key id silently resolves such a pair to
    /// whichever name a randomly-seeded `HashMap` iteration reached last,
    /// so the device name recorded in transcripts would change at every
    /// restart.
    pub fn open(keys_file: &Path, devices: &HashMap<String, DeviceConfig>) -> Result<Self> {
        let keys = KeyStore::load(keys_file)
            .with_context(|| format!("loading key file {}", keys_file.display()))?;
        if !keys.has_usable_key() {
            bail!(
                "key file {} has no usable key; ambient ingest would reject every device",
                keys_file.display()
            );
        }
        // Sorted, so the pair named in the error is the same one on every
        // run rather than whichever collision `HashMap` order surfaced.
        let mut ordered: Vec<(&String, &DeviceConfig)> = devices.iter().collect();
        ordered.sort_by(|a, b| a.0.cmp(b.0));
        let mut by_key_id: HashMap<uuid::Uuid, String> = HashMap::new();
        for (name, cfg) in ordered {
            if let Some(existing) = by_key_id.get(&cfg.key_id) {
                bail!(
                    "devices [device.{existing}] and [device.{name}] share key_id {}; \
                     one API key per device is required — a shared key collides in the \
                     segment idempotency namespace and makes the recorded device name \
                     depend on process start-up order",
                    cfg.key_id
                );
            }
            by_key_id.insert(cfg.key_id, name.clone());
        }
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
        // Carries a second, live key alongside the expired one: `open` bails
        // when the *whole* key file has no usable key (by design, so a
        // misconfiguration is loud), so a fixture with only an expired key
        // would fail at `open` rather than exercising the expiry check in
        // `resolve`. The live key keeps the store usable and lets this test
        // actually reach `KeyStore::authenticate`'s expiry rejection.
        let path = key_file(
            tmp.path(),
            &[
                ("sa-dev-live", "other-key", None),
                ("sa-dev-old", "pendant-key", Some("2020-01-01T00:00:00Z")),
            ],
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

    /// One key per device is a requirement, not a suggestion: a shared key
    /// collides in the `segment` idempotency namespace. Two `[device.*]`
    /// blocks naming the same `key_id` used to collapse into a `HashMap`
    /// whose iteration order is randomised per process, so *which* device
    /// name a segment was recorded under changed at every restart.
    #[test]
    fn open_rejects_two_devices_sharing_one_key_id() {
        let tmp = tempfile::tempdir().unwrap();
        let path = key_file(tmp.path(), &[("sa-dev-good", "shared-key", None)]);
        let id = id_of(&path, "shared-key");
        let err = DeviceRegistry::open(&path, &devices(&[("pendant", id), ("desk-mic", id)]))
            .unwrap_err();
        let msg = format!("{err:#}");
        assert!(
            msg.contains("pendant") && msg.contains("desk-mic"),
            "the error must name both devices so the config can be fixed, got: {msg}"
        );
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
