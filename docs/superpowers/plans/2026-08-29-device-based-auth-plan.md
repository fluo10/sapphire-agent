# デバイス台帳による認証 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `[device.*]` をワークスペースのデバイス台帳へ出し、`device` / `user` サブコマンドで鍵ごと生成できるようにし、ambient と `/a2a` `/acp` `/mcp` の認証を 1 本に畳む。

**Architecture:** 認証は `DeviceAuth` 1 個に集約する — トークン → `KeyStore`（ホストローカル）→ `KeyEntry.device_id` → `Devices`（ワークスペース）→ デバイス、そこからホスト設定の `[room_profile.*].devices` を反転した索引で room_profile。起動時に 1 回組んで `Arc` で共有し、`ServeState` と ambient の `IngestState` が同じものを見る。ルーティングの決定（どのデバイスがどの room_profile か）は人間が所有する `config.toml` に残し、台帳はコマンドが全上書きする「名前・説明・user_id」だけの帳簿にする。

**Tech Stack:** Rust 2024 edition, `clap` 4（derive）, `axum` 0.8, `sapphire-framework`（`workspace` + `remote-server` + `registry`）, `grain-id` 0.16, `chrono`, `anyhow`

**Spec:** `docs/superpowers/specs/2026-08-29-device-based-auth-design.md`

## Global Constraints

- **`grain-id` を 0.14 から 0.16 へ上げる。** framework が 0.16 を使うので、上げないと `KeyEntry.device_id` の型（framework の `GrainId`）と agent の `GrainId` が**別の型**になり、コンパイルが通らない。
- `sapphire-framework` のフィーチャに `"registry"` を足す。
- トークンの接頭辞は **`"sat"`**。framework の `mint_token` は `<prefix>_<random>` を作る。
- `KeyStore::generate` は **5 引数**（`prefix`, `id: Option<Uuid>`, `device_id: Option<GrainId>`, `label`, `expires_at`）。
- **既存の設定を黙って無視しない。** `[device.*]` と `[room_profile.*].api_keys` が残っていたら起動時にエラーで落とす。先例は `main.rs` の `standby_mode`。
- 認証の失敗はすべて 401 に潰し、区別はログに出す。既存の `src/ambient/auth.rs` の `resolve` の方針を引き継ぐ。
- **`allow_unknown_device` のようなトグルは作らない。** 台帳に無い鍵は通さない。
- **spec からの一点の詰め直し。** spec は「デバイスの検査を `validate_profiles` に置く」と書いているが、`validate_profiles` は `Config` だけを見る同期メソッドで `devices.toml` を読めない。台帳を要する検査（存在するか・ちょうど 1 つの room_profile に属するか）は `DeviceAuth::open` が行い、起動時に `bail!` する。`Config` 側に残るのは台帳を要さない検査（廃止設定の検知＝`migration_errors`）だけ。効果は spec の意図どおり（不正なら起動しない）で、置き場所だけが違う。
- ドキュメントコメントは既存ファイルに合わせて**英語**で書く（`src/` は英語、`config.example.toml` も英語）。

---

### Task 1: 依存の更新

**Files:**
- Modify: `Cargo.toml`（`grain-id`, `sapphire-framework` の features）
- Modify: `Cargo.lock`

**Interfaces:**
- Consumes: なし
- Produces: `sapphire_framework::registry::{Device, Devices, User, Users}` が使えるようになる

- [ ] **Step 1: 現状のテストが通ることを確認する（ベースライン）**

Run: `cargo test`
Expected: PASS。落ちるものがあれば本数と名前を控えておく（本計画と無関係の既存の失敗）。

- [ ] **Step 2: `grain-id` を 0.16 へ上げる**

`Cargo.toml`:

```toml
# Human-readable ids: session ids, and the device/user ids in the workspace
# registry. Must match the version sapphire-framework uses — `KeyEntry.device_id`
# is a `GrainId`, and two semver-incompatible grain-id versions in one graph are
# two different types.
grain-id = { version = "0.16", features = ["serde"] }
```

- [ ] **Step 3: framework に `registry` フィーチャを足す**

`Cargo.toml` の `sapphire-framework` の行を差し替える:

```toml
sapphire-framework = { version = "0.1", git = "https://github.com/fluo10/sapphire-framework", branch = "main", default-features = false, features = ["workspace", "remote-server", "registry"] }
```

- [ ] **Step 4: ピンを上げる**

Run: `cargo update -p sapphire-framework -p grain-id`

- [ ] **Step 5: ビルドとテストを確認する**

Run: `cargo build`
Expected: 成功

grain-id 0.16 で API が変わっていて壊れる箇所があれば直す。候補は
`src/ambient/speaker/candidates.rs`（`grain_id::GrainId::random()`）と
`crates/sapphire-agent-rpc/src/lib.rs`。

Run: `cargo test`
Expected: PASS（Step 1 と同じ本数）

- [ ] **Step 6: Commit**

```bash
git add Cargo.toml Cargo.lock
git commit -m "chore: grain-id 0.16 and the framework's registry feature"
```

---

### Task 2: `DeviceAuth`

**Files:**
- Create: `src/device_auth.rs`
- Modify: `src/main.rs`（`mod device_auth;` の宣言）
- Modify: `src/config.rs`（`workspace_devices_path` / `workspace_users_path` ヘルパ）
- Test: `src/device_auth.rs`（インライン `#[cfg(test)] mod tests`）

**Interfaces:**
- Consumes: `sapphire_framework::registry::{Device, Devices}`, `sapphire_framework::remote_server::KeyStore`, `grain_id::GrainId`
- Produces:
  - `config::workspace_devices_path(workspace_dir: &Path) -> PathBuf` — `{dir}/.sapphire-agent/devices.toml`
  - `config::workspace_users_path(workspace_dir: &Path) -> PathBuf` — `{dir}/.sapphire-agent/users.toml`
  - `device_auth::DeviceAuth`
  - `DeviceAuth::open(keys_file: &Path, devices_file: &Path, room_profiles: &HashMap<String, RoomProfileConfig>) -> anyhow::Result<DeviceAuth>`
  - `DeviceAuth::resolve(&self, token: &str) -> Option<Resolved<'_>>`
  - `Resolved<'a> { pub device: &'a Device, pub room_profile: &'a str }`
  - `DeviceAuth::bindings(&self) -> Vec<(&str, &str)>` — `(device name, room_profile)`、`verify` の表示用にソート済み
  - `DeviceAuth::default_key_file() -> Option<PathBuf>`（`src/ambient/auth.rs` から移設）

- [ ] **Step 1: パスのヘルパを足す**

`src/config.rs` の `workspace_config_path` の直後に:

```rust
/// Path to the workspace device table.
///
/// Mirrors `workspace_config_path`. The framework has `Workspace::devices_path`
/// for the same convention, but the agent resolves its workspace as a plain
/// `PathBuf` and never builds a framework `Workspace` for config purposes —
/// that constructor canonicalizes and requires the marker directory to already
/// exist, neither of which is true when `device add` runs on a fresh checkout.
pub fn workspace_devices_path(workspace_dir: &Path) -> PathBuf {
    workspace_dir.join(".sapphire-agent").join("devices.toml")
}

/// Path to the workspace user table. See `workspace_devices_path`.
pub fn workspace_users_path(workspace_dir: &Path) -> PathBuf {
    workspace_dir.join(".sapphire-agent").join("users.toml")
}
```

- [ ] **Step 2: 失敗するテストを書く**

`src/device_auth.rs` を作り、末尾に:

```rust
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
```

- [ ] **Step 3: テストが失敗することを確認する**

`src/main.rs` の `mod` 宣言の並びに `mod device_auth;` を足してから:

Run: `cargo test device_auth`
Expected: コンパイルエラー（`DeviceAuth` が存在しない）

- [ ] **Step 4: `DeviceAuth` を実装する**

`src/device_auth.rs` のテストの上に:

```rust
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
```

- [ ] **Step 5: `RoomProfileConfig` に `devices` を足す（テストを通すため）**

`src/config.rs` の `RoomProfileConfig` の `api_keys` の下に:

```rust
    /// Device ids (from the workspace `devices.toml`) that run under this room
    /// profile. A device id appears in exactly one room profile; the binding is
    /// what gives an authenticated device its LLM profile and memory namespace.
    /// Replaces `api_keys`, which held raw tokens in this file.
    #[serde(default)]
    pub devices: Vec<String>,
```

- [ ] **Step 6: テストが通ることを確認する**

Run: `cargo test device_auth`
Expected: PASS（12 テスト）

- [ ] **Step 7: Commit**

```bash
git add src/device_auth.rs src/config.rs src/main.rs
git commit -m "feat(auth): one DeviceAuth for ambient, a2a, acp and mcp"
```

---

### Task 3: 設定の移行検査と allowlist

**Files:**
- Modify: `src/config.rs`（`DeviceConfig` の廃止、`Config.devices` の残置、移行検査）
- Modify: `src/main.rs`（起動時の移行検査）
- Modify: `src/config_layer.rs`（`WORKSPACE_ALLOWLIST`）
- Test: `src/config.rs`, `src/config_layer.rs`（既存の `#[cfg(test)] mod tests`）

**Interfaces:**
- Consumes: `Config.room_profiles`, `Config.devices`（Task 2 で足した `RoomProfileConfig.devices`）
- Produces:
  - `Config::migration_errors(&self) -> Vec<String>` — 廃止された設定が残っているときの説明。空なら問題なし。
  - `config_layer::WORKSPACE_ALLOWLIST` に `["room_profile", "*", "devices"]`

- [ ] **Step 1: 失敗するテストを書く（移行検査）**

`src/config.rs` の `mod tests` に:

```rust
    #[test]
    fn migration_errors_name_a_leftover_device_block() {
        let cfg: Config = toml::from_str(
            r#"
[anthropic]
api_key = "test"

[device.pendant]
key_id = "550e8400-e29b-41d4-a716-446655440000"
"#,
        )
        .unwrap();

        let errors = cfg.migration_errors();

        assert_eq!(errors.len(), 1, "{errors:?}");
        assert!(errors[0].contains("pendant"), "{errors:?}");
        // The message has to say what to run, not just what is wrong: the
        // token cannot be carried over, so the operator must re-issue it.
        assert!(errors[0].contains("device add"), "{errors:?}");
    }

    #[test]
    fn migration_errors_name_a_leftover_api_keys_array() {
        let cfg: Config = toml::from_str(
            r#"
[anthropic]
api_key = "test"

[profiles.sonnet]
provider = "anthropic"

[room_profile.work]
profile = "sonnet"
api_keys = ["sa-acp-token"]
"#,
        )
        .unwrap();

        let errors = cfg.migration_errors();

        assert_eq!(errors.len(), 1, "{errors:?}");
        assert!(errors[0].contains("work"), "{errors:?}");
        assert!(errors[0].contains("devices"), "{errors:?}");
    }

    #[test]
    fn a_migrated_config_has_no_migration_errors() {
        let cfg: Config = toml::from_str(
            r#"
[anthropic]
api_key = "test"

[profiles.sonnet]
provider = "anthropic"

[room_profile.work]
profile = "sonnet"
devices = ["a3f9k2p"]
"#,
        )
        .unwrap();

        assert!(cfg.migration_errors().is_empty());
    }
```

- [ ] **Step 2: 失敗するテストを書く（allowlist）**

`src/config_layer.rs` の `mod tests` に:

```rust
    #[test]
    fn the_workspace_layer_may_route_devices_but_not_hold_tokens() {
        // `devices` is routing, exactly like `rooms`, which is already allowed.
        // `api_keys` held raw credentials and stays refused.
        assert!(path_allowed(&["room_profile", "work", "devices"]));
        assert!(!path_allowed(&["room_profile", "work", "api_keys"]));
    }
```

- [ ] **Step 3: テストが失敗することを確認する**

Run: `cargo test migration_errors the_workspace_layer_may_route`
Expected: FAIL（`migration_errors` が無い / `devices` が allowlist に無い）

- [ ] **Step 4: allowlist に足す**

`src/config_layer.rs` の `WORKSPACE_ALLOWLIST` の `&["room_profile", "*", "rooms"],` の次に:

```rust
    // Routing, like `rooms` above — which device runs under which profile. Not
    // a credential: the tokens live in the host-local key file and a key names
    // its own device, so a poisoned workspace layer can re-route a device but
    // cannot admit one.
    &["room_profile", "*", "devices"],
```

- [ ] **Step 5: `migration_errors` を実装する**

`src/config.rs` の `validate_profiles` の直後に:

```rust
    /// Settings that were removed when device-based auth landed.
    ///
    /// Reported as a hard error at start-up rather than ignored, following the
    /// `standby_mode` precedent in `main.rs`. Ignoring them would turn a broken
    /// *config* into what looks like a broken *device*: dropping `api_keys`
    /// makes `/acp` 401 every client, and dropping `[device.*]` makes ambient
    /// refuse every segment. Neither symptom sends anyone to the config file.
    pub fn migration_errors(&self) -> Vec<String> {
        let mut errors = Vec::new();

        let mut names: Vec<&String> = self.devices.keys().collect();
        names.sort();
        for name in names {
            errors.push(format!(
                "[device.{name}] no longer lives in this file. Devices moved to the workspace \
                 table at <workspace>/.sapphire-agent/devices.toml. Run `sapphire-agent device \
                 add --name {name}`, put the printed token on the device, and add the printed id \
                 to a `[room_profile.<name>].devices` array. The old token cannot be carried \
                 over: it was hand-written plaintext with no entry in the key file."
            ));
        }

        let mut rp_names: Vec<&String> = self
            .room_profiles
            .iter()
            .filter(|(_, rp)| !rp.api_keys.is_empty())
            .map(|(name, _)| name)
            .collect();
        rp_names.sort();
        for name in rp_names {
            errors.push(format!(
                "[room_profile.{name}].api_keys was replaced by `devices`. Raw tokens no longer \
                 live in this file; run `sapphire-agent device add --name <device>` for each \
                 client and list the printed ids in `[room_profile.{name}].devices`."
            ));
        }

        errors
    }
```

- [ ] **Step 6: `DeviceConfig` を廃止し、`Config.devices` を検知用に残す**

`src/config.rs` の `DeviceConfig` を差し替える。`key_id` を必須のまま残すと、
それを書いていない古い設定がパースエラーになって `migration_errors` に届かない。
全フィールドを任意にして、**存在すること自体を検知する**。

```rust
/// Removed in the device-registry migration. Retained only so an existing
/// config that still has `[device.*]` blocks fails loudly with instructions
/// instead of silently coming up with ambient ingest rejecting every segment.
/// Every field is optional on purpose: this type only has to parse, never to
/// carry a value. See `Config::migration_errors`.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct DeviceConfig {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub key_id: Option<uuid::Uuid>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub room_profile: Option<String>,
}
```

`RoomProfileConfig.api_keys` の doc コメントも差し替える:

```rust
    /// Removed in the device-registry migration; replaced by `devices`.
    /// Retained so a config that still sets it fails loudly — see
    /// `Config::migration_errors`.
    #[serde(default)]
    pub api_keys: Vec<String>,
```

- [ ] **Step 7: 起動時に検査する**

`src/main.rs` の `standby_mode` の検査の直後に:

```rust
    let migration_errors = config.migration_errors();
    if !migration_errors.is_empty() {
        anyhow::bail!(
            "config at {} uses settings that were removed in the device-registry \
             migration:\n\n  - {}\n",
            config_path.display(),
            migration_errors.join("\n\n  - ")
        );
    }
```

- [ ] **Step 8: `validate_profiles` から `api_keys` の重複検査を外す**

`api_keys` は空でなければ `migration_errors` が落とすので、重複検査は到達しない。
`seen_api_keys` の宣言とその `for key in &rp.api_keys { ... }` ブロックを削除し、
古くなったテスト（`validate_rejects_duplicate_api_keys_across_profiles` と
`validate_rejects_empty_api_key`）も削除する。それらが見ていた性質は
`migration_errors` のテストが引き継いでいる。

- [ ] **Step 9: テストが通ることを確認する**

Run: `cargo test`
Expected: PASS

`resolve_a2a_token` を使う既存テストはこの時点ではまだ残っている（Task 4 で消す）。
壊れていたら、`api_keys` を使うフィクスチャを `devices` に書き換えるのではなく、
**Task 4 まで手をつけない** — この 2 つを 1 タスクにすると差分が読めなくなる。

- [ ] **Step 10: Commit**

```bash
git add src/config.rs src/config_layer.rs src/main.rs
git commit -m "feat(config)!: room_profile.devices, and a loud death for the settings it replaces"
```

---

### Task 4: 配線 — ambient と `/a2a` `/acp` `/mcp`

**Files:**
- Modify: `src/serve/mod.rs`（`ServeState` に `device_auth`、組み立て）
- Modify: `src/serve/a2a.rs:186`, `src/serve/acp.rs:267`, `src/serve/mcp.rs:122`
- Modify: `src/ambient/startup.rs:85-97`, `src/ambient/ingest.rs:62,69,149`
- Delete: `src/ambient/auth.rs`
- Modify: `src/ambient/mod.rs`（`mod auth;` を外す）
- Modify: `src/config.rs`（`resolve_a2a_token` の削除）
- Test: `src/serve/mod.rs`, `src/ambient/ingest.rs`（既存テストの更新）

**Interfaces:**
- Consumes: `DeviceAuth::{open, resolve, bindings, default_key_file}`, `Resolved`（Task 2）
- Produces:
  - `ServeState.device_auth: Arc<DeviceAuth>`
  - `AmbientState.devices: Arc<DeviceAuth>`（型が `DeviceRegistry` から変わる）
  - `Config::resolve_a2a_token` は**削除される**

- [ ] **Step 1: `ServeState` に持たせる**

`src/serve/mod.rs` の `ServeState` に:

```rust
    /// Bearer token -> device -> room profile. Shared with ambient ingest so
    /// there is exactly one answer to "who is this token" in the process.
    pub(crate) device_auth: Arc<crate::device_auth::DeviceAuth>,
```

組み立て箇所を探す:

Run: `grep -rn "ServeState {" src/`

見つかったリテラル（本体と、テストのフィクスチャがあればそれも）に `device_auth`
を足す。値は次のように作る:

```rust
    let keys_file = config
        .keys
        .file
        .clone()
        .or_else(crate::device_auth::DeviceAuth::default_key_file)
        .ok_or_else(|| {
            anyhow::anyhow!(
                "[keys].file is unset and no platform config directory is resolvable; \
                 set [keys].file explicitly"
            )
        })?;
    let devices_file = crate::config::workspace_devices_path(&workspace_dir);
    let device_auth = Arc::new(crate::device_auth::DeviceAuth::open(
        &keys_file,
        &devices_file,
        &config.room_profiles,
    )?);
```

- [ ] **Step 2: 3 つの呼び出しを差し替える**

`src/serve/a2a.rs:186`:

```rust
    let profile_name = match state.device_auth.resolve(&bearer) {
        Some(r) => r.room_profile.to_string(),
        None => {
            return jsonrpc_error_response(
                req_id,
                codes::AUTH_REQUIRED,
                "unknown or revoked bearer token",
            );
        }
    };
```

`src/serve/acp.rs:267`:

```rust
    let Some(profile_name) = state
        .device_auth
        .resolve(&bearer)
        .map(|r| r.room_profile.to_string())
    else {
        warn!("ACP: rejected an unknown or revoked bearer token");
        return (StatusCode::UNAUTHORIZED, "unknown or revoked bearer token").into_response();
    };
```

`src/serve/mcp.rs:122`:

```rust
    let profile_name = match state.device_auth.resolve(&bearer) {
        Some(r) => r.room_profile.to_string(),
        None => {
            return jsonrpc_error(
                req_id,
                codes::AUTH_REQUIRED,
                "unknown or revoked bearer token",
            );
        }
    };
```

- [ ] **Step 3: `resolve_a2a_token` を削除する**

`src/config.rs` の `pub fn resolve_a2a_token` と、それを使うテスト
（`resolve_a2a_token_finds_owning_profile` ほか、`src/serve/mod.rs:2603` 付近の
アサーションを含む）を削除する。

`src/serve/mod.rs` のテストフィクスチャ（`api_keys = ["sa-acp-token"]` を書いている
TOML）は、そのままだと `migration_errors` に引っかかる設定になる。`devices` を
使う形に書き換え、鍵ファイルと台帳を用意して `DeviceAuth` を組む
`Task 2` の `fixture` と同じ手を使うこと。

- [ ] **Step 4: ambient を載せ替える**

`src/ambient/ingest.rs`:

```rust
// 62 行目付近
    pub devices: Arc<crate::device_auth::DeviceAuth>,

// 69 行目付近
    pub fn new(
        config: AmbientConfig,
        devices: Arc<crate::device_auth::DeviceAuth>,
        tx: mpsc::Sender<Segment>,
    ) -> Self {

// 149 行目付近
    match state.devices.resolve(&token) {
        Some(r) => Ok(r.device.name.clone()),
        None => {
            debug!(
                "ambient: rejected bearer (unknown, expired, retired, or bound to no device)"
            );
            Err(StatusCode::UNAUTHORIZED)
        }
    }
```

`src/ambient/startup.rs` の 2 番のブロックを差し替える。鍵ファイルの解決は
`ServeState` 側と同じものを使うので、`DeviceAuth` を引数で受け取る形にして
ここでは組まない。`startup` のシグネチャに
`device_auth: Arc<crate::device_auth::DeviceAuth>` を足し、`DeviceRegistry::open`
の呼び出しを削除して、そのまま `IngestState` に渡す。呼び出し元は
`ServeState` が持っている `Arc` を clone して渡す。

- [ ] **Step 5: `src/ambient/auth.rs` を削除する**

```bash
git rm src/ambient/auth.rs
```

`src/ambient/mod.rs` から `mod auth;`（および `pub use auth::...` があれば）を外す。
`auth.rs` のテストは Task 2 で `src/device_auth.rs` に移植済み。

- [ ] **Step 6: テストが通ることを確認する**

Run: `cargo test`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add -A src/serve src/ambient src/config.rs
git commit -m "refactor(auth)!: one token-resolution path, no plaintext keys in config"
```

---

### Task 5: `device` / `user` サブコマンド

**Files:**
- Modify: `src/main.rs`（`enum Command`、ディスパッチ）
- Create: `src/cli_device.rs`
- Test: `src/cli_device.rs`（インライン）、`src/main.rs`（パースのテスト）

**Interfaces:**
- Consumes: `sapphire_framework::registry::{Devices, Users}`, `KeyStore`, `config::{workspace_devices_path, workspace_users_path}`, `DeviceAuth::default_key_file`
- Produces:
  - `Command::Device { command: DeviceCommand }`, `Command::User { command: UserCommand }`
  - `cli_device::run_device(cmd: DeviceCommand, devices_file: &Path, users_file: &Path, keys_file: &Path) -> anyhow::Result<()>`
  - `cli_device::run_user(cmd: UserCommand, users_file: &Path) -> anyhow::Result<()>`
  - `cli_device::parse_duration(s: &str) -> anyhow::Result<chrono::Duration>`
  - `TOKEN_PREFIX: &str = "sat"`

- [ ] **Step 1: 失敗するテストを書く**

`src/cli_device.rs` の末尾に:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    struct Files {
        _dir: tempfile::TempDir,
        devices: PathBuf,
        users: PathBuf,
        keys: PathBuf,
    }

    fn files() -> Files {
        let dir = tempfile::tempdir().unwrap();
        Files {
            devices: dir.path().join("devices.toml"),
            users: dir.path().join("users.toml"),
            keys: dir.path().join("keys.toml"),
            _dir: dir,
        }
    }

    fn add(f: &Files, name: &str) -> anyhow::Result<()> {
        run_device(
            DeviceCommand::Add {
                name: name.into(),
                description: None,
                user: None,
                expires_in: None,
            },
            &f.devices,
            &f.users,
            &f.keys,
        )
    }

    #[test]
    fn add_writes_both_the_device_row_and_a_key_bound_to_it() {
        let f = files();

        add(&f, "pendant").unwrap();

        let devices = Devices::load(&f.devices).unwrap();
        let device = devices.resolve("pendant").unwrap();
        let keys = KeyStore::load(&f.keys).unwrap();
        assert_eq!(keys.entries().len(), 1);
        let key = &keys.entries()[0];
        assert_eq!(key.device_id, Some(device.id), "the key must name the device");
        assert!(key.token.starts_with("sat_"));
        assert_eq!(key.label.as_deref(), Some("pendant"));
    }

    /// `add` writes the device row first, so an interrupted run leaves an inert
    /// row rather than an orphan key nobody sweeps up. Re-running must finish
    /// the job instead of dead-ending on the duplicate name — otherwise there
    /// is no way out of the partial state (`rotate` needs an existing key).
    #[test]
    fn add_finishes_a_device_row_that_has_no_key_yet() {
        let f = files();
        let id = Devices::load(&f.devices)
            .unwrap()
            .add("pendant", None, None)
            .unwrap()
            .id;

        add(&f, "pendant").unwrap();

        let keys = KeyStore::load(&f.keys).unwrap();
        assert_eq!(keys.entries().len(), 1);
        assert_eq!(keys.entries()[0].device_id, Some(id), "reuses the existing row");
        assert_eq!(Devices::load(&f.devices).unwrap().entries().len(), 1);
    }

    #[test]
    fn add_refuses_a_device_that_already_has_a_key() {
        let f = files();
        add(&f, "pendant").unwrap();

        let err = add(&f, "pendant").unwrap_err();

        let msg = format!("{err:#}");
        assert!(msg.contains("rotate"), "must point at the way forward: {msg}");
    }

    #[test]
    fn add_binds_a_user_when_asked() {
        let f = files();
        run_user(UserCommand::Add { name: "fluo10".into(), description: None }, &f.users).unwrap();

        run_device(
            DeviceCommand::Add {
                name: "pendant".into(),
                description: Some("首から下げるやつ".into()),
                user: Some("fluo10".into()),
                expires_in: None,
            },
            &f.devices,
            &f.users,
            &f.keys,
        )
        .unwrap();

        let users = Users::load(&f.users).unwrap();
        let devices = Devices::load(&f.devices).unwrap();
        let device = devices.resolve("pendant").unwrap();
        assert_eq!(device.user_id, Some(users.resolve("fluo10").unwrap().id));
        assert_eq!(device.description.as_deref(), Some("首から下げるやつ"));
    }

    #[test]
    fn add_errors_on_an_unknown_user_without_writing_anything() {
        let f = files();

        let err = run_device(
            DeviceCommand::Add {
                name: "pendant".into(),
                description: None,
                user: Some("nobody".into()),
                expires_in: None,
            },
            &f.devices,
            &f.users,
            &f.keys,
        )
        .unwrap_err();

        assert!(format!("{err:#}").contains("nobody"));
        assert!(
            Devices::load(&f.devices).unwrap().entries().is_empty(),
            "the user is resolved before anything is written"
        );
    }

    #[test]
    fn add_turns_expires_in_into_an_absolute_time() {
        let f = files();

        run_device(
            DeviceCommand::Add {
                name: "pendant".into(),
                description: None,
                user: None,
                expires_in: Some("90d".into()),
            },
            &f.devices,
            &f.users,
            &f.keys,
        )
        .unwrap();

        let keys = KeyStore::load(&f.keys).unwrap();
        let expires = keys.entries()[0].expires_at.expect("an expiry was asked for");
        let expected = chrono::Utc::now() + chrono::Duration::days(90);
        assert!((expires - expected).num_seconds().abs() < 5);
    }

    #[test]
    fn add_errors_instead_of_panicking_on_an_absurd_expiry() {
        let f = files();

        let result = run_device(
            DeviceCommand::Add {
                name: "pendant".into(),
                description: None,
                user: None,
                expires_in: Some("99999999999d".into()),
            },
            &f.devices,
            &f.users,
            &f.keys,
        );

        assert!(result.is_err());
    }

    #[test]
    fn rotate_replaces_the_token_and_keeps_the_device() {
        let f = files();
        add(&f, "pendant").unwrap();
        let before = KeyStore::load(&f.keys).unwrap().entries()[0].clone();

        run_device(
            DeviceCommand::Rotate { selector: "pendant".into(), expires_in: None },
            &f.devices,
            &f.users,
            &f.keys,
        )
        .unwrap();

        let after = &KeyStore::load(&f.keys).unwrap().entries()[0].clone();
        assert_ne!(after.token, before.token);
        assert_eq!(after.device_id, before.device_id);
    }

    #[test]
    fn retire_marks_the_device_and_revokes_its_key() {
        let f = files();
        add(&f, "pendant").unwrap();

        run_device(
            DeviceCommand::Retire { selector: "pendant".into(), purge: false },
            &f.devices,
            &f.users,
            &f.keys,
        )
        .unwrap();

        let devices = Devices::load(&f.devices).unwrap();
        // The row stays: device ids get written into content elsewhere.
        assert!(devices.resolve("pendant").unwrap().is_retired());
        // The key does not: retiring a device must actually stop it.
        assert!(KeyStore::load(&f.keys).unwrap().entries().is_empty());
    }

    #[test]
    fn retire_with_purge_removes_the_row_too() {
        let f = files();
        add(&f, "pendant").unwrap();

        run_device(
            DeviceCommand::Retire { selector: "pendant".into(), purge: true },
            &f.devices,
            &f.users,
            &f.keys,
        )
        .unwrap();

        assert!(Devices::load(&f.devices).unwrap().entries().is_empty());
    }

    #[test]
    fn user_add_rejects_a_duplicate_name() {
        let f = files();
        run_user(UserCommand::Add { name: "fluo10".into(), description: None }, &f.users).unwrap();

        let err = run_user(
            UserCommand::Add { name: "fluo10".into(), description: None },
            &f.users,
        )
        .unwrap_err();

        assert!(format!("{err:#}").contains("fluo10"));
    }

    #[test]
    fn parse_duration_accepts_days_hours_and_minutes() {
        assert_eq!(parse_duration("90d").unwrap(), chrono::Duration::days(90));
        assert_eq!(parse_duration("12h").unwrap(), chrono::Duration::hours(12));
        assert_eq!(parse_duration("30m").unwrap(), chrono::Duration::minutes(30));
    }

    #[test]
    fn parse_duration_rejects_junk() {
        // A unit is mandatory: `90` could be seconds or days, and neither
        // reading is safe to guess for a credential's lifetime.
        for s in ["", "90", "d90", "-1d", "90y", "1000000000000d"] {
            assert!(parse_duration(s).is_err(), "{s} passed");
        }
    }
}
```

- [ ] **Step 2: テストが失敗することを確認する**

`src/main.rs` に `mod cli_device;` を足してから:

Run: `cargo test cli_device`
Expected: コンパイルエラー

- [ ] **Step 3: `src/cli_device.rs` を実装する**

```rust
//! The `device` and `user` subcommands.
//!
//! `device add` is the only place in the agent that mints a token. It writes
//! two files that are not in the same place — the device row goes into the
//! workspace table, the key into the host-local key file — so the order
//! matters: the row goes first, because a row with no key is inert while an
//! orphan key is litter nobody sweeps. `add` is therefore resumable: it
//! finishes a row that has no key rather than dead-ending on the duplicate
//! name.

use std::path::{Path, PathBuf};

use anyhow::{Context as _, Result, anyhow, bail};
use chrono::{DateTime, Duration, Utc};
use clap::Subcommand;
use sapphire_framework::registry::{Devices, Users};
use sapphire_framework::remote_server::KeyStore;

/// Prefix on tokens this agent mints (sapphire-agent token).
pub const TOKEN_PREFIX: &str = "sat";

#[derive(Subcommand, Debug)]
pub enum DeviceCommand {
    /// Register a device and mint the key it authenticates with.
    Add {
        #[arg(long, value_name = "DEVICE_NAME")]
        name: String,
        /// A note for you — what this device is.
        #[arg(long, value_name = "TEXT")]
        description: Option<String>,
        /// Whose device this is: a user id or name from users.toml.
        #[arg(long, value_name = "SELECTOR")]
        user: Option<String>,
        /// Expire the key after this long, e.g. `90d`, `12h`.
        #[arg(long, value_name = "DURATION")]
        expires_in: Option<String>,
    },
    /// List devices, their users, and whether they hold a key on this host.
    List,
    /// Re-issue a device's token, keeping its id and its row.
    ///
    /// `--expires-in` REPLACES the expiry rather than keeping it: omitting the
    /// flag makes the key non-expiring.
    Rotate {
        /// The device's id, or its name.
        selector: String,
        #[arg(long, value_name = "DURATION")]
        expires_in: Option<String>,
    },
    /// Stop a device: revoke its key, and mark the row retired.
    Retire {
        /// The device's id, or its name.
        selector: String,
        /// Delete the row outright instead of retiring it. Device ids get
        /// written into content (a journal entry's `updated_by`, say), so this
        /// makes those references unresolvable. Retiring is the default for
        /// that reason.
        #[arg(long)]
        purge: bool,
    },
}

#[derive(Subcommand, Debug)]
pub enum UserCommand {
    /// Register a user.
    Add {
        #[arg(long, value_name = "USER_NAME")]
        name: String,
        #[arg(long, value_name = "TEXT")]
        description: Option<String>,
    },
    /// List users.
    List,
}

/// Turn `90d` / `12h` / `30m` into a [`Duration`].
///
/// The unit is mandatory. A bare `90` is refused — nothing should have to guess
/// whether a credential lives for ninety seconds or ninety days.
///
/// `try_days` rather than `days`: the latter panics on out-of-range input, so a
/// typo in `--expires-in` would abort the process instead of erroring.
pub fn parse_duration(s: &str) -> Result<Duration> {
    let split = s
        .find(|c: char| !c.is_ascii_digit())
        .with_context(|| format!("duration needs a unit (d/h/m): {s:?}"))?;
    let (value, unit) = s.split_at(split);
    if value.is_empty() {
        bail!("duration must start with digits before the unit: {s:?}");
    }
    let n: i64 = value.parse().with_context(|| format!("bad duration: {s:?}"))?;
    let d = match unit {
        "d" => Duration::try_days(n),
        "h" => Duration::try_hours(n),
        "m" => Duration::try_minutes(n),
        other => bail!("unknown duration unit {other:?} in {s:?} (use d, h or m)"),
    };
    d.ok_or_else(|| anyhow!("duration is out of range: {s:?}"))
}

/// Relative expiry to an absolute instant.
///
/// `checked_add_signed`, because `Utc::now() + d` panics on a time it cannot
/// represent — chrono's `Duration` range is much wider than `DateTime`'s, so a
/// value that survives `parse_duration` can still blow up here.
fn absolute_expiry(expires_in: Option<&str>) -> Result<Option<DateTime<Utc>>> {
    expires_in
        .map(parse_duration)
        .transpose()?
        .map(|d| {
            Utc::now()
                .checked_add_signed(d)
                .ok_or_else(|| anyhow!("expiry is too far in the future: {d}"))
        })
        .transpose()
}

pub fn run_device(
    command: DeviceCommand,
    devices_file: &Path,
    users_file: &Path,
    keys_file: &Path,
) -> Result<()> {
    let mut devices = Devices::load(devices_file)
        .with_context(|| format!("loading device table {}", devices_file.display()))?;
    let mut keys = KeyStore::load(keys_file)
        .with_context(|| format!("loading key file {}", keys_file.display()))?;

    match command {
        DeviceCommand::Add {
            name,
            description,
            user,
            expires_in,
        } => {
            // Resolve everything that can fail before writing anything.
            let expires_at = absolute_expiry(expires_in.as_deref())?;
            let user_id = match user {
                Some(selector) => {
                    let users = Users::load(users_file).with_context(|| {
                        format!("loading user table {}", users_file.display())
                    })?;
                    Some(users.resolve(&selector)?.id)
                }
                None => None,
            };

            // Take an owned copy before the match: `resolve` borrows `devices`
            // immutably and the other arm needs it mutably, so holding the
            // reference across the match does not borrow-check.
            let existing = devices.resolve(&name).ok().cloned();
            let device = match existing {
                // The row already exists. Either this is a resumed `add` whose
                // key write did not happen, or the name is genuinely taken.
                Some(existing) => {
                    if keys.entries().iter().any(|k| k.device_id == Some(existing.id)) {
                        bail!(
                            "device {name:?} already exists and already holds a key on this \
                             host; use `sapphire-agent device rotate {name}` to re-issue its \
                             token"
                        );
                    }
                    existing
                }
                None => devices.add(&name, description, user_id)?,
            };

            let entry = keys.generate(
                TOKEN_PREFIX,
                None,
                Some(device.id),
                Some(device.name.clone()),
                expires_at,
            )?;

            println!("{}", entry.token);
            eprintln!(
                "id {}  created {}{}",
                device.id,
                device.created_at.to_rfc3339(),
                entry
                    .expires_at
                    .map(|e| format!("  expires {}", e.to_rfc3339()))
                    .unwrap_or_default()
            );
            // Routing lives in config.toml, which this command does not touch,
            // so the config is invalid until the operator adds this line. Say
            // exactly what to paste rather than letting the next start-up
            // explain it.
            eprintln!(
                "\nnext: bind it to a room profile in your config.toml\n\n    \
                 [room_profile.<name>]\n    devices = [\"{}\"]\n",
                device.id
            );
        }
        DeviceCommand::List => {
            for d in devices.entries() {
                let has_key = keys.entries().iter().any(|k| k.device_id == Some(d.id));
                println!(
                    "{}  {}  {}  {}  {}",
                    d.id,
                    d.name,
                    d.user_id
                        .map(|u| u.to_string())
                        .unwrap_or_else(|| "-".to_owned()),
                    if has_key { "key" } else { "no-key" },
                    if d.is_retired() { "retired" } else { "active" },
                );
            }
        }
        DeviceCommand::Rotate {
            selector,
            expires_in,
        } => {
            let expires_at = absolute_expiry(expires_in.as_deref())?;
            let device = devices.resolve(&selector)?.clone();
            let entry = keys.rotate(TOKEN_PREFIX, &device.name, expires_at)?;
            println!("{}", entry.token);
            eprintln!("rotated {} ({})", device.id, device.name);
            eprintln!("a running agent keeps accepting the old token until it restarts");
        }
        DeviceCommand::Retire { selector, purge } => {
            let device = devices.resolve(&selector)?.clone();
            // Revoke first: the point of retiring is to stop the device, and a
            // crash between the two writes must not leave a live key behind.
            if keys.entries().iter().any(|k| k.device_id == Some(device.id)) {
                keys.revoke(&device.name)?;
            }
            if purge {
                devices.purge(&selector)?;
                eprintln!("purged {} ({})", device.id, device.name);
            } else {
                devices.retire(&selector)?;
                eprintln!("retired {} ({})", device.id, device.name);
            }
        }
    }
    Ok(())
}

pub fn run_user(command: UserCommand, users_file: &Path) -> Result<()> {
    let mut users = Users::load(users_file)
        .with_context(|| format!("loading user table {}", users_file.display()))?;
    match command {
        UserCommand::Add { name, description } => {
            let user = users.add(&name, description)?;
            println!("{}", user.id);
            eprintln!("added {} ({})", user.id, user.name);
        }
        UserCommand::List => {
            for u in users.entries() {
                println!(
                    "{}  {}  {}",
                    u.id,
                    u.name,
                    if u.is_retired() { "retired" } else { "active" }
                );
            }
        }
    }
    Ok(())
}
```

`PathBuf` を使っていない場合は import から外すこと。

- [ ] **Step 4: `main.rs` に配線する**

`enum Command` に:

```rust
    /// Manage the devices that authenticate to this agent.
    Device {
        #[command(subcommand)]
        command: cli_device::DeviceCommand,
    },
    /// Manage the users devices belong to.
    User {
        #[command(subcommand)]
        command: cli_device::UserCommand,
    },
```

`match cli.command` に:

```rust
        Some(Command::Device { command }) => {
            let workspace_dir = config.resolved_workspace_dir(&config_path);
            let keys_file = config
                .keys
                .file
                .clone()
                .or_else(device_auth::DeviceAuth::default_key_file)
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "[keys].file is unset and no platform config directory is \
                         resolvable; set [keys].file explicitly"
                    )
                })?;
            cli_device::run_device(
                command,
                &config::workspace_devices_path(&workspace_dir),
                &config::workspace_users_path(&workspace_dir),
                &keys_file,
            )
        }
        Some(Command::User { command }) => {
            let workspace_dir = config.resolved_workspace_dir(&config_path);
            cli_device::run_user(command, &config::workspace_users_path(&workspace_dir))
        }
```

- [ ] **Step 5: CLI パースのテストを書く**

`src/main.rs` に `#[cfg(test)] mod tests`（無ければ作る）:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser as _;

    #[test]
    fn device_add_requires_a_name() {
        assert!(Cli::try_parse_from(["sapphire-agent", "device", "add"]).is_err());
    }

    #[test]
    fn device_add_takes_a_name_and_a_description() {
        let cli = Cli::try_parse_from([
            "sapphire-agent",
            "device",
            "add",
            "--name",
            "pendant",
            "--description",
            "the one on the lanyard",
        ])
        .unwrap();
        assert!(matches!(
            cli.command,
            Some(Command::Device {
                command: cli_device::DeviceCommand::Add { ref name, description: Some(_), .. }
            }) if name == "pendant"
        ));
    }

    #[test]
    fn device_retire_defaults_to_keeping_the_row() {
        let cli =
            Cli::try_parse_from(["sapphire-agent", "device", "retire", "pendant"]).unwrap();
        assert!(matches!(
            cli.command,
            Some(Command::Device {
                command: cli_device::DeviceCommand::Retire { purge: false, .. }
            })
        ));
    }
}
```

- [ ] **Step 6: テストが通ることを確認する**

Run: `cargo test`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add src/cli_device.rs src/main.rs
git commit -m "feat(cli): device and user subcommands, one command per token"
```

---

### Task 6: `verify`、ドキュメント、イシュー

**Files:**
- Modify: `src/main.rs`（`Command::Verify` の出力）
- Modify: `config.example.toml`（`[device.*]` と `api_keys` の節）
- Modify: `README.md`
- Test: なし（表示のみ。`cargo test` の回帰で足りる）

**Interfaces:**
- Consumes: `DeviceAuth::bindings`
- Produces: なし

- [ ] **Step 1: `verify` にデバイスの対応を出す**

`Command::Verify` の中、`Config OK` を出した後に:

```rust
            // The device -> room_profile binding is written by hand, so it
            // needs somewhere to be checked.
            let keys_file = config
                .keys
                .file
                .clone()
                .or_else(device_auth::DeviceAuth::default_key_file);
            match keys_file {
                Some(keys_file) => {
                    let devices_file = config::workspace_devices_path(&workspace_dir);
                    match device_auth::DeviceAuth::open(
                        &keys_file,
                        &devices_file,
                        &config.room_profiles,
                    ) {
                        Ok(auth) => {
                            println!("  Device table      : {}", devices_file.display());
                            for (device, rp) in auth.bindings() {
                                println!("    {device:<20} -> room_profile '{rp}'");
                            }
                        }
                        Err(e) => println!("  Devices           : INVALID: {e:#}"),
                    }
                }
                None => println!("  Devices           : [keys].file unset"),
            }
```

- [ ] **Step 2: `config.example.toml` を書き換える**

`grep -n "device\|api_keys" config.example.toml` で該当箇所を洗い出す。

- `[keys]` の節（484 行付近）は残す。鍵ファイルの場所は変わらない。
- `[device.pendant]` の例（498–510 行付近）を**削除**し、代わりに
  `sapphire-agent device add --name pendant --description "..."` を案内する。
  台帳は `<workspace>/.sapphire-agent/devices.toml` にあり、**この設定ファイルでは
  なくコマンドが所有する**こと、`.sapphire-agent/config.toml` の隣にあることを書く。
- `[room_profile.*].api_keys` の例をすべて `devices = ["<device id>"]` に置き換える。
  「id は `device add` が出力する 7 文字の grain-id」と書く。
- すべてのデバイスがちょうど 1 つの room_profile に現れなければならないこと、
  ambient 専用のデバイスも例外ではないこと（会話を始めたときの LLM プロファイルと
  メモリ名前空間がそこから決まるため）を書く。

Run: `cargo test --test config_example` — もし example を読むテストがあれば通す
（`crates/sapphire-call-cli/tests/config_example.rs` は別物なので注意）。

Run: `cargo test`
Expected: PASS

- [ ] **Step 3: README を更新する**

Run: `grep -n "api_keys\|\[device\." README.md docs/*.md`

見つかった箇所を `devices` / `device add` に合わせて書き換える。

- [ ] **Step 4: ambient トランスクリプトのイシューを立てる**

```bash
gh issue create \
  --title "Ambient transcripts are one pool, but room profiles carry memory namespaces" \
  --body 'S4 で ambient が会話を始めると顕在化する不整合。

`TranscriptStore` は `<cache_root>/transcripts/<日付>.jsonl` の単一プール
（`src/ambient/startup.rs`）で、memory namespace とは無関係。一方 room_profile は
メモリ名前空間を運ぶ。デバイスが room_profile に紐づいた今、メモリ名前空間の
異なる 2 つの room_profile が同じトランスクリプトプールを共有する — 仕事用の
会話とプライベートの会話が同じ場所に落ちる。

`Disposition::RecordAndConverse` に到達する経路ができた時点で、トランスクリプトも
room_profile（あるいはそのメモリ名前空間）で分ける必要がある。

設計: `docs/superpowers/specs/2026-08-29-device-based-auth-design.md` の
「別イシューに切り出すもの」節'
```

- [ ] **Step 5: 全体の lint**

Run: `cargo clippy --all-targets -- -D warnings`
Expected: 警告なし

Run: `cargo fmt --all -- --check`
Expected: 差分なし

- [ ] **Step 6: Commit**

```bash
git add src/main.rs config.example.toml README.md
git commit -m "docs: device add replaces hand-written keys and api_keys"
```

---

## 完了条件

- `cargo test` が通る
- `cargo clippy --all-targets -- -D warnings` が通る
- `sapphire-agent device add --name X --description Y` が台帳の行と鍵を作り、トークンを stdout に、貼り付け用の `[room_profile.<name>] devices = [...]` を stderr に出す
- `/a2a` `/acp` `/mcp` と ambient ingest が同じ `DeviceAuth` を通る
- `Config::resolve_a2a_token` と `src/ambient/auth.rs` が存在しない
- `[device.*]` または `[room_profile.*].api_keys` が残っている設定は、対処法を名指しするエラーで起動に失敗する
- ちょうど 1 つの room_profile に属さない現役デバイスがあると起動に失敗する
- `sapphire-agent verify` がデバイスと room_profile の対応を出す
- ambient トランスクリプトの名前空間のイシューが立っている

## 上流への依存

このリポジトリの作業は、`sapphire-framework` の
`docs/superpowers/plans/2026-08-29-device-user-registry-plan.md` が `main` に
入ってから始めること。Task 1 の `cargo update` がそれを取ってくる。
