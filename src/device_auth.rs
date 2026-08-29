//! Device authentication: bearer token -> key file entry -> workspace device
//! -> room profile.
//!
//! One mechanism for every authenticated entry point: ambient ingest, `/a2a`,
//! `/acp` and `/mcp`. The token lives only in the framework key file, which is
//! host-local; the device table lives in the workspace and is synced. The link
//! runs key -> device (`KeyEntry.device_id`) rather than device -> key, because
//! one physical device talking to two hosts has two keys in two files.
//!
//! Which room profile a device runs under is host config, not table data:
//! `[room_profile.<n>].devices`. The table is rewritten in full by
//! `sapphire-agent device`, so it is no place for a decision a human makes.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use grain_id::GrainId;
use sapphire_framework::registry::{Device, Devices};
use sapphire_framework::remote_server::KeyStore;

use crate::config::RoomProfileConfig;

/// What a presented token resolved to.
pub struct Resolved<'a> {
    pub device: &'a Device,
    /// Never empty: `open` refuses to build a `DeviceAuth` in which a live
    /// device is bound to no room profile.
    pub room_profile: &'a str,
}

pub struct DeviceAuth {
    keys: KeyStore,
    devices: Devices,
    /// `Device::id` -> the `[room_profile.<name>]` key.
    room_profile_by_device: HashMap<GrainId, String>,
}

// Hand-written rather than `#[derive(Debug)]`: the framework stores API tokens
// in `KeyStore` as plaintext by documented design and has no `Debug` impl of
// its own. A derive would compile today but start printing live bearer tokens
// the moment the framework ever grows one. Naming only the derived index
// forecloses that — this impl structurally cannot reach `keys`.
impl std::fmt::Debug for DeviceAuth {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeviceAuth")
            .field("room_profile_by_device", &self.room_profile_by_device)
            .finish()
    }
}

impl DeviceAuth {
    /// Load both tables and check that the routing in `room_profiles` agrees
    /// with them.
    ///
    /// A missing key file is an error rather than an empty store.
    /// `KeyStore::load` treats absence as "no keys", which would leave every
    /// endpoint running and rejecting every device forever — a
    /// misconfiguration that looks exactly like a broken device.
    pub fn open(
        keys_file: &Path,
        devices_file: &Path,
        room_profiles: &HashMap<String, RoomProfileConfig>,
    ) -> Result<Self> {
        let keys = KeyStore::load(keys_file)
            .with_context(|| format!("loading key file {}", keys_file.display()))?;
        if !keys.has_usable_key() {
            bail!(
                "key file {} has no usable key; every authenticated endpoint would \
                 reject every device",
                keys_file.display()
            );
        }
        let devices = Devices::load(devices_file)
            .with_context(|| format!("loading device table {}", devices_file.display()))?;

        // Sorted, so a name in an error is the same one on every run rather
        // than whichever `HashMap` order surfaced.
        let mut ordered: Vec<(&String, &RoomProfileConfig)> = room_profiles.iter().collect();
        ordered.sort_by(|a, b| a.0.cmp(b.0));

        let mut room_profile_by_device: HashMap<GrainId, String> = HashMap::new();
        for (rp_name, rp) in ordered {
            for selector in &rp.devices {
                let id = devices
                    .resolve(selector)
                    .with_context(|| {
                        format!(
                            "room_profile '{rp_name}' names device {selector:?}, which is not in {}",
                            devices_file.display()
                        )
                    })?
                    .id;
                if let Some(prev) = room_profile_by_device.get(&id) {
                    bail!(
                        "device {selector:?} appears in room_profiles '{prev}' and '{rp_name}'; \
                         a device runs under exactly one room profile, which is what decides \
                         its LLM profile and memory namespace"
                    );
                }
                room_profile_by_device.insert(id, rp_name.clone());
            }
        }

        // Every live device must be routed. Retired ones are exempt: they stay
        // in the table so historical references resolve, and requiring dead
        // routing entries for them would make config.toml grow forever.
        let unbound: Vec<&str> = devices
            .entries()
            .iter()
            .filter(|d| !d.is_retired() && !room_profile_by_device.contains_key(&d.id))
            .map(|d| d.name.as_str())
            .collect();
        if !unbound.is_empty() {
            bail!(
                "device(s) {} are in {} but no room_profile lists them. Add each id to a \
                 `[room_profile.<name>].devices` array — that binding is what gives a device \
                 its LLM profile and memory namespace.",
                unbound.join(", "),
                devices_file.display()
            );
        }

        Ok(Self {
            keys,
            devices,
            room_profile_by_device,
        })
    }

    /// Device and room profile for `token`, or `None`.
    ///
    /// Six distinct failures collapse to `None` on purpose — the caller answers
    /// 401 for all of them and logs the distinction rather than returning it:
    /// no such token, expired, the key names no device, the device is not in
    /// the table, the device is retired, the device is unrouted (which `open`
    /// already refuses, so it cannot happen here).
    /// `KeyStore::authenticate` does the constant-time comparison and the
    /// `expires_at` check.
    pub fn resolve(&self, token: &str) -> Option<Resolved<'_>> {
        let entry = self.keys.authenticate(token)?;
        let device = self.devices.get(entry.device_id?)?;
        if device.is_retired() {
            return None;
        }
        let room_profile = self.room_profile_by_device.get(&device.id)?;
        Some(Resolved {
            device,
            room_profile,
        })
    }

    /// `(device name, room_profile)` pairs, sorted by device name. For `verify`.
    pub fn bindings(&self) -> Vec<(&str, &str)> {
        let mut rows: Vec<(&str, &str)> = self
            .devices
            .entries()
            .iter()
            .filter(|d| !d.is_retired())
            .filter_map(|d| {
                self.room_profile_by_device
                    .get(&d.id)
                    .map(|rp| (d.name.as_str(), rp.as_str()))
            })
            .collect();
        rows.sort();
        rows
    }

    /// Default key file location, used when `[keys].file` is unset.
    pub fn default_key_file() -> Option<PathBuf> {
        dirs::config_dir().map(|d| d.join("sapphire-agent").join("keys.toml"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::RoomProfileConfig;

    /// Write a key file the way `device add` would, and return the device table
    /// alongside it. `bindings` is `(device name, room_profile name)`.
    fn fixture(
        dir: &Path,
        devices: &[(&str, Option<&str>)], // (name, expires_at RFC3339)
        bindings: &[(&str, &str)],
    ) -> (PathBuf, PathBuf, HashMap<String, RoomProfileConfig>) {
        let devices_file = dir.join("devices.toml");
        let keys_file = dir.join("keys.toml");

        let mut table = Devices::load(&devices_file).unwrap();
        let mut keys = KeyStore::load(&keys_file).unwrap();
        for (name, expires) in devices {
            let d = table.add(name, None, None).unwrap();
            let expires_at = expires.map(|e| e.parse::<chrono::DateTime<chrono::Utc>>().unwrap());
            keys.generate("sat", None, Some(d.id), Some((*name).into()), expires_at)
                .unwrap();
        }

        let mut room_profiles: HashMap<String, RoomProfileConfig> = HashMap::new();
        for (device, rp) in bindings {
            let id = table.resolve(device).unwrap().id.to_string();
            room_profiles
                .entry((*rp).to_string())
                .or_insert_with(|| RoomProfileConfig {
                    profile: "sonnet".into(),
                    ..Default::default()
                })
                .devices
                .push(id);
        }
        (keys_file, devices_file, room_profiles)
    }

    /// The token `device add` minted for `name`.
    fn token_for(keys_file: &Path, name: &str) -> String {
        KeyStore::load(keys_file)
            .unwrap()
            .entries()
            .iter()
            .find(|e| e.label.as_deref() == Some(name))
            .expect("key present")
            .token
            .clone()
    }

    #[test]
    fn resolves_a_bound_token_to_its_device_and_room_profile() {
        let tmp = tempfile::tempdir().unwrap();
        let (keys, devices, rps) =
            fixture(tmp.path(), &[("pendant", None)], &[("pendant", "home")]);
        let token = token_for(&keys, "pendant");

        let auth = DeviceAuth::open(&keys, &devices, &rps).unwrap();
        let resolved = auth.resolve(&token).expect("should resolve");

        assert_eq!(resolved.device.name, "pendant");
        assert_eq!(resolved.room_profile, "home");
    }

    #[test]
    fn rejects_a_token_absent_from_the_key_file() {
        let tmp = tempfile::tempdir().unwrap();
        let (keys, devices, rps) =
            fixture(tmp.path(), &[("pendant", None)], &[("pendant", "home")]);

        let auth = DeviceAuth::open(&keys, &devices, &rps).unwrap();

        assert!(auth.resolve("sat_nope").is_none());
    }

    #[test]
    fn rejects_an_expired_token_even_though_it_matches() {
        let tmp = tempfile::tempdir().unwrap();
        // A live key rides alongside: `open` bails when the whole key file has
        // no usable key, so a fixture with only the expired one would fail at
        // `open` instead of exercising the expiry check in `resolve`.
        let (keys, devices, rps) = fixture(
            tmp.path(),
            &[("pendant", Some("2020-01-01T00:00:00Z")), ("desk", None)],
            &[("pendant", "home"), ("desk", "work")],
        );
        let token = token_for(&keys, "pendant");

        let auth = DeviceAuth::open(&keys, &devices, &rps).unwrap();

        assert!(auth.resolve(&token).is_none());
    }

    #[test]
    fn rejects_a_valid_key_that_names_no_device() {
        let tmp = tempfile::tempdir().unwrap();
        let keys_file = tmp.path().join("keys.toml");
        let devices_file = tmp.path().join("devices.toml");
        let mut keys = KeyStore::load(&keys_file).unwrap();
        // A hand-written key with no device_id at all.
        let entry = keys.generate("sat", None, None, Some("loose".into()), None).unwrap();

        let auth = DeviceAuth::open(&keys_file, &devices_file, &HashMap::new()).unwrap();

        assert!(auth.resolve(&entry.token).is_none());
    }

    #[test]
    fn rejects_a_key_whose_device_is_not_in_the_table() {
        let tmp = tempfile::tempdir().unwrap();
        let keys_file = tmp.path().join("keys.toml");
        let devices_file = tmp.path().join("devices.toml");
        let mut keys = KeyStore::load(&keys_file).unwrap();
        let entry = keys
            .generate("sat", None, Some(GrainId::random()), Some("ghost".into()), None)
            .unwrap();

        let auth = DeviceAuth::open(&keys_file, &devices_file, &HashMap::new()).unwrap();

        assert!(auth.resolve(&entry.token).is_none());
    }

    #[test]
    fn rejects_a_retired_device() {
        let tmp = tempfile::tempdir().unwrap();
        let (keys, devices_file, rps) =
            fixture(tmp.path(), &[("pendant", None), ("desk", None)],
                    &[("pendant", "home"), ("desk", "work")]);
        let token = token_for(&keys, "pendant");
        Devices::load(&devices_file).unwrap().retire("pendant").unwrap();

        let auth = DeviceAuth::open(&keys, &devices_file, &rps).unwrap();

        // Retiring the device is enough to stop it; revoking the key is a
        // separate, additional step.
        assert!(auth.resolve(&token).is_none());
    }

    #[test]
    fn open_rejects_a_device_bound_to_no_room_profile() {
        let tmp = tempfile::tempdir().unwrap();
        let (keys, devices, _) = fixture(tmp.path(), &[("pendant", None)], &[]);

        let err = DeviceAuth::open(&keys, &devices, &HashMap::new()).unwrap_err();

        let msg = format!("{err:#}");
        assert!(msg.contains("pendant"), "the error must name the device: {msg}");
    }

    #[test]
    fn open_rejects_a_device_bound_to_two_room_profiles() {
        let tmp = tempfile::tempdir().unwrap();
        let (keys, devices, mut rps) =
            fixture(tmp.path(), &[("pendant", None)], &[("pendant", "home")]);
        let id = Devices::load(&devices).unwrap().resolve("pendant").unwrap().id.to_string();
        rps.insert(
            "work".into(),
            RoomProfileConfig { profile: "sonnet".into(), devices: vec![id], ..Default::default() },
        );

        let err = DeviceAuth::open(&keys, &devices, &rps).unwrap_err();

        let msg = format!("{err:#}");
        assert!(msg.contains("home") && msg.contains("work"), "name both: {msg}");
    }

    #[test]
    fn open_rejects_a_room_profile_naming_an_unknown_device() {
        let tmp = tempfile::tempdir().unwrap();
        let (keys, devices, mut rps) =
            fixture(tmp.path(), &[("pendant", None)], &[("pendant", "home")]);
        rps.get_mut("home").unwrap().devices.push("zzzzzzz".into());

        let err = DeviceAuth::open(&keys, &devices, &rps).unwrap_err();

        assert!(format!("{err:#}").contains("zzzzzzz"));
    }

    /// A retired device keeps resolving for display purposes but must not be
    /// required to hold a room_profile — otherwise dead routing entries pile up
    /// in config.toml forever.
    #[test]
    fn open_does_not_require_a_retired_device_to_be_bound() {
        let tmp = tempfile::tempdir().unwrap();
        let (keys, devices_file, rps) =
            fixture(tmp.path(), &[("pendant", None), ("old", None)],
                    &[("pendant", "home"), ("old", "home")]);
        Devices::load(&devices_file).unwrap().retire("old").unwrap();
        let mut rps = rps;
        let old_id = Devices::load(&devices_file).unwrap().resolve("old").unwrap().id.to_string();
        rps.get_mut("home").unwrap().devices.retain(|d| *d != old_id);

        assert!(DeviceAuth::open(&keys, &devices_file, &rps).is_ok());
    }

    #[test]
    fn open_fails_when_the_key_file_has_no_usable_key() {
        let tmp = tempfile::tempdir().unwrap();
        let missing = tmp.path().join("absent.toml");
        let devices = tmp.path().join("devices.toml");

        let err = DeviceAuth::open(&missing, &devices, &HashMap::new()).unwrap_err();

        assert!(err.to_string().contains("no usable key"), "unexpected: {err}");
    }

    #[test]
    fn debug_never_prints_a_token() {
        let tmp = tempfile::tempdir().unwrap();
        let (keys, devices, rps) =
            fixture(tmp.path(), &[("pendant", None)], &[("pendant", "home")]);
        let token = token_for(&keys, "pendant");

        let auth = DeviceAuth::open(&keys, &devices, &rps).unwrap();

        assert!(!format!("{auth:?}").contains(&token));
    }
}
