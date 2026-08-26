# Ambient Audio Ingest Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Accept audio segments from an always-on wearable microphone over a plain HTTP POST, transcribe them, attribute each to a speaker, and cache both audio and transcripts outside the workspace — without ever starting an LLM turn.

**Architecture:** A new `src/ambient/` subsystem, separate from `src/voice/` (which is the interactive audio-in/audio-out pipeline). `POST /audio/ingest` authenticates a device by bearer token via `sapphire-framework`'s `KeyStore`, admits the segment to a bounded queue, and returns immediately. A background worker re-gates the audio with a VAD, transcribes it through the existing `SttProvider`, computes a speaker embedding, matches it against registered speakers and auto-enrolled candidates, and appends one JSONL line per segment.

**Tech Stack:** Rust 2024, axum 0.8, tokio, `sapphire-framework` (`remote-server` feature, for `KeyStore`), `sherpa-onnx` 1.13 (VAD + speaker embedding, behind the existing `voice-sherpa` feature), `hound` (WAV), `chrono`, `uuid`, `grain-id`, `sha2`.

**Spec:** `docs/superpowers/specs/2026-08-26-ambient-audio-ingest-design.md`

## Global Constraints

Every task's requirements implicitly include these. Values are copied verbatim from the spec.

- **Sample rate is 16000 Hz, mono, s16le.** Anything else is rejected, never resampled. The constant already exists: `crate::voice::PIPELINE_SAMPLE_RATE`.
- **Ingest never starts an LLM turn.** No code path in `src/ambient/` may call `run_llm_turn` or any provider.
- **No audio and no transcripts inside the workspace.** They live under `dirs::cache_dir()/sapphire-agent/ambient/`. The single exception is reference audio under `voices/`, which is user-curated input.
- **No secret in `config.toml`.** Device blocks carry `key_id` (a UUID), never a token.
- **Transcripts store a speaker *id*, never a display name.** Names resolve from the workspace at read time.
- **Day boundaries use `config.day_boundary_hour`** via the existing `crate::session::local_date_for_timestamp`, not UTC midnight and not local midnight.
- **The wire protocol must be implementable by a microcontroller.** Query parameters and a raw body; no JSON body, no base64, no framing.
- **Commit scope is `(ambient)`.** Per `CLAUDE.md`, agent-internal scopes need no `cliff.toml` change.

### Running the tests

Default features compile `sherpa-onnx`, which is a 5–10 minute C++ build on a cold cache. Every task except Task 7 is testable without it:

```sh
cargo test --no-default-features --features "redb-store,lancedb-store,fastembed-embed" ambient
```

Task 7 and the final verification need the full default build:

```sh
cargo test ambient
```

---

### Task 1: Configuration and the `KeyStore` dependency

**Files:**
- Modify: `Cargo.toml`
- Modify: `src/config.rs`
- Test: `src/config.rs` (inline `#[cfg(test)] mod tests`, matching the file's existing convention)

**Interfaces:**
- Consumes: nothing.
- Produces: `AmbientConfig`, `KeysConfig`, `DeviceConfig`, and three new `Config` fields: `ambient: AmbientConfig`, `keys: KeysConfig`, `devices: HashMap<String, DeviceConfig>` (TOML name `device`).

- [ ] **Step 1: Add the dependency features**

In `Cargo.toml`, the `sapphire-framework` dependency gains `remote-server`, and `uuid` gains `serde` (needed to deserialize `key_id` from TOML):

```toml
sapphire-framework = { version = "0.1", git = "https://github.com/fluo10/sapphire-framework", branch = "main", default-features = false, features = ["workspace", "remote-server"] }

uuid = { version = "1", features = ["v7", "serde"] }
```

Leave a comment above the framework line recording why the fat feature is acceptable:

```toml
# `remote-server` is enabled for one type: `remote_server::KeyStore`, which
# authenticates ambient capture devices. It drags in the whole JSON-RPC sync
# server (axum, redb, blob, retrieve, track) for that one struct; splitting it
# out is fluo10/sapphire-framework#103, after which this becomes `keys`.
```

- [ ] **Step 2: Run the build to confirm the dependency resolves**

Run: `cargo check --no-default-features --features "redb-store,lancedb-store,fastembed-embed"`
Expected: PASS. If the framework's `remote-server` feature fails to resolve, stop and report — nothing later works without it.

- [ ] **Step 3: Write the failing config tests**

Append to the `#[cfg(test)] mod tests` block in `src/config.rs`:

```rust
#[test]
fn ambient_config_defaults_to_disabled_with_documented_values() {
    let cfg: crate::config::AmbientConfig = toml::from_str("").unwrap();
    assert!(!cfg.enabled, "ambient must be opt-in");
    assert_eq!(cfg.audio_retention_days, 7);
    assert_eq!(cfg.min_embed_ms, 1500);
    assert_eq!(cfg.match_threshold, 0.55);
    assert_eq!(cfg.promote_after_seconds, 60);
    assert_eq!(cfg.promote_after_days, 2);
    assert_eq!(cfg.max_queue, 1000);
    assert!(cfg.cache_dir.is_none());
}

#[test]
fn device_blocks_parse_a_key_id_and_never_a_token() {
    let toml_src = r#"
[anthropic]
api_key = "sk-test"

[keys]
file = "/etc/sapphire/keys.toml"

[device.pendant]
key_id = "6c8f4a2e-1d33-4b90-9a71-0e5b2f8c4d17"
label = "the one on the lanyard"
room_profile = "default"
"#;
    let cfg: crate::config::Config = toml::from_str(toml_src).unwrap();
    let dev = cfg.devices.get("pendant").expect("device.pendant parsed");
    assert_eq!(
        dev.key_id.to_string(),
        "6c8f4a2e-1d33-4b90-9a71-0e5b2f8c4d17"
    );
    assert_eq!(dev.label.as_deref(), Some("the one on the lanyard"));
    assert_eq!(dev.room_profile.as_deref(), Some("default"));
    assert_eq!(
        cfg.keys.file.as_deref(),
        Some(std::path::Path::new("/etc/sapphire/keys.toml"))
    );
}

#[test]
fn a_device_block_carrying_a_token_is_a_parse_error() {
    // The whole point of key_id is that a secret cannot live here. An
    // `api_key` field must not silently parse and be ignored.
    let toml_src = r#"
[anthropic]
api_key = "sk-test"

[device.pendant]
key_id = "6c8f4a2e-1d33-4b90-9a71-0e5b2f8c4d17"
api_key = "sa-dev-oops"
"#;
    let err = toml::from_str::<crate::config::Config>(toml_src).unwrap_err();
    assert!(
        err.to_string().contains("api_key"),
        "unexpected error: {err}"
    );
}
```

- [ ] **Step 4: Run the tests to verify they fail**

Run: `cargo test --no-default-features --features "redb-store,lancedb-store,fastembed-embed" config::tests::ambient_config_defaults`
Expected: FAIL — `AmbientConfig` does not exist.

- [ ] **Step 5: Add the config types**

In `src/config.rs`, near `ImageCacheConfig`:

```rust
/// Ambient (always-on) audio capture ingest. Opt-in: the endpoint is
/// mounted unconditionally but refuses requests when `enabled` is false,
/// matching how `[a2a]` and `[acp]` behave.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AmbientConfig {
    #[serde(default)]
    pub enabled: bool,
    /// Override the cache root. `None` resolves to
    /// `dirs::cache_dir() / "sapphire-agent" / "ambient"` at startup.
    #[serde(default)]
    pub cache_dir: Option<PathBuf>,
    /// Days of raw audio to keep. Transcripts are never swept.
    #[serde(default = "default_audio_retention_days")]
    pub audio_retention_days: u32,
    /// Name of the `[stt_provider.*]` block used for ambient transcription.
    #[serde(default)]
    pub stt_provider: String,
    /// Segments with less gated speech than this get no speaker attribution.
    /// Embeddings from very short utterances are unreliable and are the main
    /// driver of speaker-id inflation.
    #[serde(default = "default_min_embed_ms")]
    pub min_embed_ms: u32,
    /// `SpeakerEmbeddingManager::search` threshold.
    #[serde(default = "default_match_threshold")]
    pub match_threshold: f32,
    #[serde(default = "default_promote_after_seconds")]
    pub promote_after_seconds: u32,
    #[serde(default = "default_promote_after_days")]
    pub promote_after_days: u32,
    /// Admission queue depth. A full queue answers 429, which a device
    /// handles exactly like being offline.
    #[serde(default = "default_max_queue")]
    pub max_queue: usize,
}

fn default_audio_retention_days() -> u32 { 7 }
fn default_min_embed_ms() -> u32 { 1500 }
fn default_match_threshold() -> f32 { 0.55 }
fn default_promote_after_seconds() -> u32 { 60 }
fn default_promote_after_days() -> u32 { 2 }
fn default_max_queue() -> usize { 1000 }

impl Default for AmbientConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            cache_dir: None,
            audio_retention_days: default_audio_retention_days(),
            stt_provider: String::new(),
            min_embed_ms: default_min_embed_ms(),
            match_threshold: default_match_threshold(),
            promote_after_seconds: default_promote_after_seconds(),
            promote_after_days: default_promote_after_days(),
            max_queue: default_max_queue(),
        }
    }
}

/// Location of the `sapphire-framework` key file. Host-local: it names
/// the only place tokens are stored, so it must never be settable from
/// the workspace config layer.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct KeysConfig {
    #[serde(default)]
    pub file: Option<PathBuf>,
}

/// One capture device. Carries no secret: `key_id` names a `KeyEntry` in
/// the key file, and the token itself never appears here. That is what
/// lets this block be shared through the workspace config layer.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DeviceConfig {
    /// `KeyEntry::id` from the framework key file.
    pub key_id: uuid::Uuid,
    /// Display metadata; reaches the system prompt. Distinct from the key
    /// file's own `label`, which the framework documents as a note nothing
    /// in the system reads.
    #[serde(default)]
    pub label: Option<String>,
    /// Which room profile a conversation from this device runs under (S4).
    #[serde(default)]
    pub room_profile: Option<String>,
}
```

Then add the three fields to `Config`:

```rust
    #[serde(default)]
    pub ambient: AmbientConfig,

    #[serde(default)]
    pub keys: KeysConfig,

    /// Capture devices, keyed by a stable human-readable name. The key is
    /// the device id recorded in transcripts.
    #[serde(default, rename = "device")]
    pub devices: HashMap<String, DeviceConfig>,
```

`deny_unknown_fields` on `DeviceConfig` is what makes the third test pass: a stray `api_key` is a hard error rather than a silently ignored field.

- [ ] **Step 6: Run the tests to verify they pass**

Run: `cargo test --no-default-features --features "redb-store,lancedb-store,fastembed-embed" config::tests`
Expected: PASS, including the pre-existing config tests.

- [ ] **Step 7: Commit**

```bash
git add Cargo.toml Cargo.lock src/config.rs
git commit -m "feat(ambient): add [ambient], [keys] and [device.*] config"
```

---

### Task 2: Device authentication

**Files:**
- Create: `src/ambient/mod.rs`
- Create: `src/ambient/auth.rs`
- Modify: `src/main.rs` (add `mod ambient;`)
- Test: `src/ambient/auth.rs` (inline test module)

**Interfaces:**
- Consumes: `DeviceConfig`, `KeysConfig` from Task 1.
- Produces: `DeviceRegistry::open(keys_file: &Path, devices: &HashMap<String, DeviceConfig>) -> anyhow::Result<DeviceRegistry>` and `DeviceRegistry::resolve(&self, token: &str) -> Option<&str>` returning the device name.

- [ ] **Step 1: Create the module skeleton**

`src/ambient/mod.rs`:

```rust
//! Always-on ambient audio capture ingest.
//!
//! Deliberately separate from [`crate::voice`]. `voice` is the
//! **interactive** pipeline: audio in, LLM turn, audio out. `ambient`
//! **records without answering** — nothing in this module may start an
//! LLM turn. See
//! `docs/superpowers/specs/2026-08-26-ambient-audio-ingest-design.md`.

pub mod auth;
```

Add `mod ambient;` to `src/main.rs` alongside the other module declarations.

- [ ] **Step 2: Write the failing auth tests**

`src/ambient/auth.rs`, test module:

```rust
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
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `cargo test --no-default-features --features "redb-store,lancedb-store,fastembed-embed" ambient::auth`
Expected: FAIL — `DeviceRegistry` does not exist.

- [ ] **Step 4: Implement `DeviceRegistry`**

At the top of `src/ambient/auth.rs`:

```rust
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
```

Add `pub mod auth;` is already in place from Step 1.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `cargo test --no-default-features --features "redb-store,lancedb-store,fastembed-embed" ambient::auth`
Expected: PASS, 5 tests.

- [ ] **Step 6: Commit**

```bash
git add src/ambient/ src/main.rs
git commit -m "feat(ambient): resolve a device from its bearer token via KeyStore"
```

---

### Task 3: Audio cache

**Files:**
- Create: `src/ambient/cache.rs`
- Modify: `src/ambient/mod.rs`
- Test: `src/ambient/cache.rs` (inline test module)

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `AudioCache::open(dir: PathBuf) -> Result<Arc<AudioCache>>`, `AudioCache::default_dir() -> Option<PathBuf>`, `AudioCache::put(&self, bytes: &[u8]) -> Result<String>` (returns sha256 hex), `AudioCache::get(&self, sha256: &str) -> Option<Vec<u8>>`, `AudioCache::sweep(&self, max_age: std::time::Duration) -> Result<usize>`.

- [ ] **Step 1: Write the failing cache tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn put_is_content_addressed_and_round_trips() {
        let tmp = tempfile::tempdir().unwrap();
        let cache = AudioCache::open(tmp.path().to_path_buf()).unwrap();
        let sha = cache.put(b"pcm-bytes").unwrap();
        assert_eq!(sha.len(), 64, "sha256 hex");
        assert_eq!(cache.get(&sha).as_deref(), Some(&b"pcm-bytes"[..]));
        // Same bytes, same address, no duplicate file.
        assert_eq!(cache.put(b"pcm-bytes").unwrap(), sha);
        assert_eq!(std::fs::read_dir(tmp.path()).unwrap().count(), 1);
    }

    #[test]
    fn get_misses_return_none_rather_than_erroring() {
        let tmp = tempfile::tempdir().unwrap();
        let cache = AudioCache::open(tmp.path().to_path_buf()).unwrap();
        assert!(cache.get(&"0".repeat(64)).is_none());
    }

    #[test]
    fn sweep_removes_blobs_older_than_the_limit_and_keeps_the_rest() {
        let tmp = tempfile::tempdir().unwrap();
        let cache = AudioCache::open(tmp.path().to_path_buf()).unwrap();
        let fresh = cache.put(b"fresh").unwrap();
        let stale = cache.put(b"stale").unwrap();

        // Backdate one blob by ten days.
        let ten_days_ago = std::time::SystemTime::now() - std::time::Duration::from_secs(10 * 86400);
        filetime::set_file_mtime(
            tmp.path().join(&stale),
            filetime::FileTime::from_system_time(ten_days_ago),
        )
        .unwrap();

        let removed = cache
            .sweep(std::time::Duration::from_secs(7 * 86400))
            .unwrap();
        assert_eq!(removed, 1);
        assert!(cache.get(&stale).is_none(), "stale blob swept");
        assert!(cache.get(&fresh).is_some(), "fresh blob kept");
    }
}
```

- [ ] **Step 2: Add the `filetime` dev-dependency**

The sweep test needs to backdate a file. In `Cargo.toml` under `[dev-dependencies]`:

```toml
filetime = "0.2"
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `cargo test --no-default-features --features "redb-store,lancedb-store,fastembed-embed" ambient::cache`
Expected: FAIL — `AudioCache` does not exist.

- [ ] **Step 4: Implement `AudioCache`**

```rust
//! Workspace-external cache for raw ambient audio segments.
//!
//! Same shape and same reasoning as [`crate::image_cache`]: the filename
//! is the sha256 hex with no extension, identical bytes from different
//! devices share one file, and write failures degrade rather than panic.
//!
//! Unlike the image cache this one **does** evict: a day of ambient
//! recording is 200–450 MB, so unbounded growth is not a theoretical
//! concern. Transcripts are kept; only the audio is swept.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, SystemTime};

use anyhow::{Context, Result};
use sha2::{Digest, Sha256};
use std::fmt::Write as _;
use tracing::warn;

pub struct AudioCache {
    dir: PathBuf,
}

impl AudioCache {
    pub fn open(dir: PathBuf) -> Result<Arc<Self>> {
        std::fs::create_dir_all(&dir)
            .with_context(|| format!("creating ambient audio cache dir {dir:?}"))?;
        Ok(Arc::new(Self { dir }))
    }

    /// Platform-standard default, suitable as a config default.
    pub fn default_dir() -> Option<PathBuf> {
        dirs::cache_dir().map(|d| d.join("sapphire-agent").join("ambient").join("audio"))
    }

    /// Store `bytes` and return their sha256 hex address. Idempotent: an
    /// existing file with the same hash is left untouched.
    pub fn put(&self, bytes: &[u8]) -> Result<String> {
        let sha = sha256_hex(bytes);
        let path = self.dir.join(&sha);
        if path.exists() {
            return Ok(sha);
        }
        std::fs::write(&path, bytes)
            .with_context(|| format!("writing ambient audio blob {path:?}"))?;
        Ok(sha)
    }

    /// Raw bytes for `sha256`, or `None` on any miss.
    pub fn get(&self, sha256: &str) -> Option<Vec<u8>> {
        std::fs::read(self.dir.join(sha256)).ok()
    }

    /// Delete blobs whose mtime is older than `max_age`. Returns how many
    /// were removed. A blob that cannot be read or removed is warned about
    /// and skipped — a sweep must never abort the process.
    pub fn sweep(&self, max_age: Duration) -> Result<usize> {
        let cutoff = SystemTime::now()
            .checked_sub(max_age)
            .unwrap_or(SystemTime::UNIX_EPOCH);
        let mut removed = 0;
        for entry in std::fs::read_dir(&self.dir)
            .with_context(|| format!("reading ambient audio cache dir {:?}", self.dir))?
        {
            let entry = match entry {
                Ok(e) => e,
                Err(e) => {
                    warn!("ambient sweep: unreadable dir entry: {e}");
                    continue;
                }
            };
            let modified = match entry.metadata().and_then(|m| m.modified()) {
                Ok(m) => m,
                Err(e) => {
                    warn!("ambient sweep: no mtime for {:?}: {e}", entry.path());
                    continue;
                }
            };
            if modified < cutoff {
                match std::fs::remove_file(entry.path()) {
                    Ok(()) => removed += 1,
                    Err(e) => warn!("ambient sweep: cannot remove {:?}: {e}", entry.path()),
                }
            }
        }
        Ok(removed)
    }
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut out = String::with_capacity(64);
    for byte in digest {
        let _ = write!(out, "{byte:02x}");
    }
    out
}
```

Add `pub mod cache;` to `src/ambient/mod.rs`.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `cargo test --no-default-features --features "redb-store,lancedb-store,fastembed-embed" ambient::cache`
Expected: PASS, 3 tests.

- [ ] **Step 6: Commit**

```bash
git add Cargo.toml Cargo.lock src/ambient/
git commit -m "feat(ambient): content-addressed audio cache with age-based sweep"
```

---

### Task 4: Transcript store

**Files:**
- Create: `src/ambient/transcript.rs`
- Modify: `src/ambient/mod.rs`
- Test: `src/ambient/transcript.rs` (inline test module)

**Interfaces:**
- Consumes: `crate::session::local_date_for_timestamp` (existing).
- Produces: `TranscriptRecord` (public fields listed below), `TranscriptStore::open(dir: PathBuf, day_boundary_hour: u8) -> Result<TranscriptStore>`, `TranscriptStore::append(&self, rec: &TranscriptRecord) -> Result<()>`, `TranscriptStore::read(&self, from: DateTime<Utc>, to: DateTime<Utc>, speaker: Option<&str>) -> Result<Vec<TranscriptRecord>>`.

- [ ] **Step 1: Write the failing transcript tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    fn rec(started_at: DateTime<Utc>, speaker: Option<&str>, text: &str) -> TranscriptRecord {
        TranscriptRecord {
            segment: format!("seg-{}", started_at.timestamp()),
            device: "pendant".into(),
            started_at,
            speech_ms: 4200,
            speaker: speaker.map(str::to_string),
            speaker_score: speaker.map(|_| 0.87),
            text: text.into(),
            audio: "a".repeat(64),
        }
    }

    #[test]
    fn append_then_read_round_trips_a_record() {
        let tmp = tempfile::tempdir().unwrap();
        let store = TranscriptStore::open(tmp.path().to_path_buf(), 4).unwrap();
        let at = Utc.with_ymd_and_hms(2026, 8, 26, 5, 3, 11).unwrap();
        let r = rec(at, Some("me"), "hello");
        store.append(&r).unwrap();

        let got = store
            .read(at - chrono::Duration::hours(1), at + chrono::Duration::hours(1), None)
            .unwrap();
        assert_eq!(got, vec![r]);
    }

    #[test]
    fn a_record_before_the_boundary_hour_lands_in_the_previous_day_file() {
        let tmp = tempfile::tempdir().unwrap();
        let store = TranscriptStore::open(tmp.path().to_path_buf(), 4).unwrap();
        // 02:00 local on the 27th is still the 26th when the day starts at 04:00.
        let local = chrono::Local
            .with_ymd_and_hms(2026, 8, 27, 2, 0, 0)
            .unwrap();
        store.append(&rec(local.with_timezone(&Utc), None, "late night")).unwrap();

        let files: Vec<String> = std::fs::read_dir(tmp.path())
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        assert_eq!(files, vec!["2026-08-26.jsonl".to_string()]);
    }

    #[test]
    fn read_filters_by_speaker_and_by_range() {
        let tmp = tempfile::tempdir().unwrap();
        let store = TranscriptStore::open(tmp.path().to_path_buf(), 4).unwrap();
        let base = Utc.with_ymd_and_hms(2026, 8, 26, 12, 0, 0).unwrap();
        store.append(&rec(base, Some("me"), "mine")).unwrap();
        store
            .append(&rec(base + chrono::Duration::minutes(1), Some("tanaka-san"), "theirs"))
            .unwrap();
        store
            .append(&rec(base + chrono::Duration::hours(5), Some("me"), "later"))
            .unwrap();

        let mine = store
            .read(base - chrono::Duration::hours(1), base + chrono::Duration::hours(1), Some("me"))
            .unwrap();
        assert_eq!(mine.len(), 1, "speaker filter and range both applied");
        assert_eq!(mine[0].text, "mine");
    }

    #[test]
    fn read_spans_multiple_day_files_in_time_order() {
        let tmp = tempfile::tempdir().unwrap();
        let store = TranscriptStore::open(tmp.path().to_path_buf(), 4).unwrap();
        let day2 = Utc.with_ymd_and_hms(2026, 8, 27, 12, 0, 0).unwrap();
        let day1 = Utc.with_ymd_and_hms(2026, 8, 26, 12, 0, 0).unwrap();
        store.append(&rec(day2, Some("me"), "second")).unwrap();
        store.append(&rec(day1, Some("me"), "first")).unwrap();

        let all = store
            .read(day1 - chrono::Duration::days(1), day2 + chrono::Duration::days(1), None)
            .unwrap();
        let texts: Vec<&str> = all.iter().map(|r| r.text.as_str()).collect();
        assert_eq!(texts, vec!["first", "second"], "sorted by started_at");
    }

    #[test]
    fn a_corrupt_line_is_skipped_rather_than_failing_the_read() {
        let tmp = tempfile::tempdir().unwrap();
        let store = TranscriptStore::open(tmp.path().to_path_buf(), 4).unwrap();
        let at = Utc.with_ymd_and_hms(2026, 8, 26, 12, 0, 0).unwrap();
        store.append(&rec(at, Some("me"), "good")).unwrap();
        let day_file = tmp.path().join("2026-08-26.jsonl");
        let mut body = std::fs::read_to_string(&day_file).unwrap();
        body.push_str("{not json at all\n");
        std::fs::write(&day_file, body).unwrap();

        let got = store
            .read(at - chrono::Duration::hours(1), at + chrono::Duration::hours(1), None)
            .unwrap();
        assert_eq!(got.len(), 1);
    }
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test --no-default-features --features "redb-store,lancedb-store,fastembed-embed" ambient::transcript`
Expected: FAIL — `TranscriptStore` does not exist.

- [ ] **Step 3: Implement the transcript store**

```rust
//! Append-only transcript store: one JSONL file per logical day.
//!
//! `speaker` holds an **id**, never a display name. Names resolve from the
//! workspace `voices/` directory at read time, so renaming
//! `voices/blithe-otter-42/` to `voices/tanaka-san/` makes every past
//! transcript read back under the new name with no rewrite pass.
//!
//! Day files use the agent's `day_boundary_hour`, not UTC midnight, so a
//! 02:00 conversation lands in the same file as the evening before it —
//! the same rule the daily logs already follow.

use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;

use anyhow::{Context, Result};
use chrono::{DateTime, Local, NaiveDate, Utc};
use serde::{Deserialize, Serialize};
use tracing::warn;

use crate::session::local_date_for_timestamp;

/// One ingested segment as recorded on disk.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TranscriptRecord {
    /// The device-supplied idempotency key.
    pub segment: String,
    /// `[device.<name>]` key the bearer token resolved to.
    pub device: String,
    /// When the audio was **recorded**, not when it arrived.
    pub started_at: DateTime<Utc>,
    /// Speech duration after the VAD re-gate — not the length of the
    /// submitted segment. The same measure feeds `min_embed_ms` and a
    /// candidate's promotion total, so all three agree.
    pub speech_ms: u32,
    /// Speaker id, or `None` when the segment was too short to attribute.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub speaker: Option<String>,
    /// `SpeakerEmbeddingManager` match score. Absent when `speaker` is None.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub speaker_score: Option<f32>,
    pub text: String,
    /// sha256 address of the raw audio in the [`super::cache::AudioCache`].
    pub audio: String,
}

pub struct TranscriptStore {
    dir: PathBuf,
    day_boundary_hour: u8,
}

impl TranscriptStore {
    pub fn open(dir: PathBuf, day_boundary_hour: u8) -> Result<Self> {
        std::fs::create_dir_all(&dir)
            .with_context(|| format!("creating transcript dir {dir:?}"))?;
        Ok(Self {
            dir,
            day_boundary_hour,
        })
    }

    /// The logical day `at` belongs to, honouring `day_boundary_hour`.
    pub fn day_of(&self, at: DateTime<Utc>) -> NaiveDate {
        local_date_for_timestamp(at.with_timezone(&Local), self.day_boundary_hour)
    }

    fn path_for_day(&self, day: NaiveDate) -> PathBuf {
        self.dir.join(format!("{day}.jsonl"))
    }

    /// Append one record to its day file, creating the file if needed.
    pub fn append(&self, rec: &TranscriptRecord) -> Result<()> {
        let path = self.path_for_day(self.day_of(rec.started_at));
        let line = serde_json::to_string(rec).context("serializing transcript record")?;
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .with_context(|| format!("opening transcript file {path:?}"))?;
        writeln!(file, "{line}").with_context(|| format!("appending to {path:?}"))?;
        Ok(())
    }

    /// Every record with `from <= started_at <= to`, optionally restricted
    /// to one speaker id, sorted by `started_at`.
    ///
    /// Unparseable lines are skipped with a warning. A half-written line
    /// from a crash must not make the whole day unreadable.
    pub fn read(
        &self,
        from: DateTime<Utc>,
        to: DateTime<Utc>,
        speaker: Option<&str>,
    ) -> Result<Vec<TranscriptRecord>> {
        let mut out = Vec::new();
        // Widen by a day on each side: the boundary hour means a record's
        // logical day can differ from its calendar day.
        let mut day = self.day_of(from) - chrono::Duration::days(1);
        let last = self.day_of(to) + chrono::Duration::days(1);
        while day <= last {
            let path = self.path_for_day(day);
            if let Ok(file) = std::fs::File::open(&path) {
                for line in BufReader::new(file).lines() {
                    let Ok(line) = line else { continue };
                    if line.trim().is_empty() {
                        continue;
                    }
                    match serde_json::from_str::<TranscriptRecord>(&line) {
                        Ok(rec) => out.push(rec),
                        Err(e) => warn!("skipping corrupt transcript line in {path:?}: {e}"),
                    }
                }
            }
            day += chrono::Duration::days(1);
        }
        out.retain(|r| {
            r.started_at >= from
                && r.started_at <= to
                && speaker.is_none_or(|s| r.speaker.as_deref() == Some(s))
        });
        out.sort_by_key(|r| r.started_at);
        Ok(out)
    }
}
```

Add `pub mod transcript;` to `src/ambient/mod.rs`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cargo test --no-default-features --features "redb-store,lancedb-store,fastembed-embed" ambient::transcript`
Expected: PASS, 5 tests.

- [ ] **Step 5: Commit**

```bash
git add src/ambient/
git commit -m "feat(ambient): per-day JSONL transcript store honouring day_boundary_hour"
```

---

### Task 5: Ingest endpoint and admission queue

**Files:**
- Create: `src/ambient/ingest.rs`
- Modify: `src/ambient/mod.rs`
- Modify: `src/serve/mod.rs` (router, `ServeState`)
- Test: `src/ambient/ingest.rs` (inline test module, driving the router through `tower::ServiceExt`)

**Interfaces:**
- Consumes: `DeviceRegistry` (Task 2), `AmbientConfig` (Task 1).
- Produces:
  - `pub struct Segment { pub segment: String, pub device: String, pub started_at: DateTime<Utc>, pub live: bool, pub pcm: Vec<i16> }`
  - `pub struct AmbientState { pub config: AmbientConfig, pub devices: DeviceRegistry, pub tx: tokio::sync::mpsc::Sender<Segment>, seen: Mutex<HashSet<String>> }`
  - `pub fn routes() -> axum::Router<Arc<crate::serve::ServeState>>` mounting `/audio/ingest` and `/audio/hello`.

- [ ] **Step 1: Write the failing endpoint tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use tower::ServiceExt;

    /// Build a router plus the receiving end of the admission queue.
    fn harness() -> (axum::Router, tokio::sync::mpsc::Receiver<Segment>) {
        // Full construction is in Task 12's wiring; here we build the
        // minimal state the two handlers actually touch.
        unimplemented!("replaced in Step 3")
    }

    fn pcm_body(samples: usize) -> Body {
        Body::from(vec![0u8; samples * 2])
    }

    fn ingest_uri(segment: &str, live: bool) -> String {
        format!(
            "/audio/ingest?segment={segment}&started_at=1787000000000&rate=16000&live={}",
            if live { 1 } else { 0 }
        )
    }

    #[tokio::test]
    async fn accepts_a_well_formed_segment_and_enqueues_it() {
        let (app, mut rx) = harness();
        let res = app
            .oneshot(
                Request::post(ingest_uri("seg-1", true))
                    .header("authorization", "Bearer sa-dev-good")
                    .header("content-type", "audio/L16")
                    .body(pcm_body(16_000))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);
        let seg = rx.try_recv().expect("segment enqueued");
        assert_eq!(seg.segment, "seg-1");
        assert_eq!(seg.device, "pendant", "identity comes from the key, not the URL");
        assert!(seg.live);
        assert_eq!(seg.pcm.len(), 16_000);
    }

    #[tokio::test]
    async fn a_repeated_segment_id_is_accepted_and_discarded() {
        let (app, mut rx) = harness();
        for _ in 0..2 {
            let res = app
                .clone()
                .oneshot(
                    Request::post(ingest_uri("seg-dup", false))
                        .header("authorization", "Bearer sa-dev-good")
                        .header("content-type", "audio/L16")
                        .body(pcm_body(1_600))
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(res.status(), StatusCode::OK, "replay is normal, not an error");
        }
        assert!(rx.try_recv().is_ok(), "first delivery enqueued");
        assert!(rx.try_recv().is_err(), "duplicate not enqueued twice");
    }

    #[tokio::test]
    async fn rejects_an_unknown_bearer_with_401() {
        let (app, _rx) = harness();
        let res = app
            .oneshot(
                Request::post(ingest_uri("seg-2", false))
                    .header("authorization", "Bearer sa-dev-nope")
                    .header("content-type", "audio/L16")
                    .body(pcm_body(1_600))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn rejects_an_unsupported_content_type_with_415() {
        let (app, _rx) = harness();
        let res = app
            .oneshot(
                Request::post(ingest_uri("seg-3", false))
                    .header("authorization", "Bearer sa-dev-good")
                    .header("content-type", "audio/opus")
                    .body(pcm_body(1_600))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(
            res.status(),
            StatusCode::UNSUPPORTED_MEDIA_TYPE,
            "opus is the documented forward-compat seam, not a v1 format"
        );
    }

    #[tokio::test]
    async fn rejects_a_non_16k_rate_rather_than_resampling() {
        let (app, _rx) = harness();
        let res = app
            .oneshot(
                Request::post(
                    "/audio/ingest?segment=seg-4&started_at=1787000000000&rate=48000&live=0",
                )
                .header("authorization", "Bearer sa-dev-good")
                .header("content-type", "audio/L16")
                .body(pcm_body(1_600))
                .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn answers_429_when_the_queue_is_full() {
        let (app, _rx) = harness_with_queue_depth(1);
        // Fill the queue, then overflow it.
        for i in 0..3 {
            let res = app
                .clone()
                .oneshot(
                    Request::post(ingest_uri(&format!("seg-q{i}"), false))
                        .header("authorization", "Bearer sa-dev-good")
                        .header("content-type", "audio/L16")
                        .body(pcm_body(1_600))
                        .unwrap(),
                )
                .await
                .unwrap();
            if i == 2 {
                assert_eq!(res.status(), StatusCode::TOO_MANY_REQUESTS);
            }
        }
    }

    #[tokio::test]
    async fn hello_reports_ingest_parameters_without_being_required() {
        let (app, _rx) = harness();
        let res = app
            .oneshot(
                Request::post("/audio/hello")
                    .header("authorization", "Bearer sa-dev-good")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);
        let bytes = axum::body::to_bytes(res.into_body(), 64 * 1024).await.unwrap();
        let v: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["device"], "pendant");
        assert_eq!(v["sample_rate"], 16000);
        assert_eq!(v["accepts"][0], "audio/L16");
        assert_eq!(v["downlink"], false, "reserved for S4, not implemented");
    }
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test --no-default-features --features "redb-store,lancedb-store,fastembed-embed" ambient::ingest`
Expected: FAIL — `Segment`, `harness` and the handlers do not exist.

- [ ] **Step 3: Implement the endpoint**

```rust
//! `POST /audio/ingest` — admission for ambient capture segments.
//!
//! The body is raw audio and the metadata is in query parameters. This is
//! deliberately not JSON-RPC on `/rpc`: base64 would inflate the payload by
//! a third and force a microcontroller to assemble JSON around a
//! multi-kilobyte blob, and radio-on time is the device's dominant battery
//! cost. `curl` must be able to speak this.
//!
//! Admission is separate from processing. Reconnecting after a day offline
//! dumps a day of spooled segments at once; the handler enqueues and
//! returns so the device can put its radio back to sleep.

use std::collections::HashSet;
use std::sync::{Arc, Mutex};

use axum::extract::{Query, State};
use axum::http::{HeaderMap, StatusCode};
use axum::response::IntoResponse;
use axum::routing::post;
use axum::{Json, Router};
use chrono::{DateTime, TimeZone, Utc};
use serde::Deserialize;
use serde_json::json;
use tokio::sync::mpsc;
use tracing::{debug, warn};

use super::auth::DeviceRegistry;
use crate::config::AmbientConfig;
use crate::serve::ServeState;
use crate::voice::PIPELINE_SAMPLE_RATE;

/// Content types accepted in v1. `audio/opus` is the documented
/// forward-compatibility seam and is deliberately absent until the
/// device's power draw has been measured.
const ACCEPTED_CONTENT_TYPES: &[&str] = &["audio/l16", "audio/wav"];

/// One admitted segment on its way to the processing worker.
#[derive(Debug, Clone)]
pub struct Segment {
    pub segment: String,
    /// Resolved from the bearer token, never from the request.
    pub device: String,
    /// When the audio was recorded.
    pub started_at: DateTime<Utc>,
    /// True only for realtime audio. Replayed spool is false, which is what
    /// keeps a future wake-word path from answering hours-old speech.
    pub live: bool,
    pub pcm: Vec<i16>,
}

pub struct AmbientState {
    pub config: AmbientConfig,
    pub devices: DeviceRegistry,
    pub tx: mpsc::Sender<Segment>,
    /// Segment ids already admitted, for idempotency.
    seen: Mutex<HashSet<String>>,
}

impl AmbientState {
    pub fn new(config: AmbientConfig, devices: DeviceRegistry, tx: mpsc::Sender<Segment>) -> Self {
        Self {
            config,
            devices,
            tx,
            seen: Mutex::new(HashSet::new()),
        }
    }

    /// Record `id` as admitted. Returns false when it was already present.
    fn admit_once(&self, id: &str) -> bool {
        self.seen
            .lock()
            .expect("ambient seen-set poisoned")
            .insert(id.to_string())
    }
}

#[derive(Debug, Deserialize)]
pub struct IngestParams {
    segment: String,
    started_at: i64,
    #[serde(default = "default_rate")]
    rate: u32,
    #[serde(default)]
    live: u8,
}

fn default_rate() -> u32 {
    PIPELINE_SAMPLE_RATE
}

pub fn routes() -> Router<Arc<ServeState>> {
    Router::new()
        .route("/audio/ingest", post(handle_ingest))
        .route("/audio/hello", post(handle_hello))
}

/// Resolve the bearer to a device name, or the 401 to return.
///
/// All three failures — unknown token, expired token, token bound to no
/// device — collapse to the same response. The distinction is logged, not
/// returned.
fn authenticate(state: &ServeState, headers: &HeaderMap) -> Result<String, StatusCode> {
    let ambient = state.ambient.as_ref().ok_or(StatusCode::NOT_FOUND)?;
    if !ambient.config.enabled {
        return Err(StatusCode::NOT_FOUND);
    }
    let token = crate::serve::extract_bearer(headers).ok_or(StatusCode::UNAUTHORIZED)?;
    match ambient.devices.resolve(&token) {
        Some(name) => Ok(name.to_string()),
        None => {
            debug!("ambient: rejected bearer (unknown, expired, or bound to no device)");
            Err(StatusCode::UNAUTHORIZED)
        }
    }
}

async fn handle_ingest(
    State(state): State<Arc<ServeState>>,
    Query(params): Query<IngestParams>,
    headers: HeaderMap,
    body: axum::body::Bytes,
) -> impl IntoResponse {
    let device = match authenticate(&state, &headers) {
        Ok(d) => d,
        Err(status) => return status.into_response(),
    };
    let ambient = state.ambient.as_ref().expect("checked in authenticate");

    if params.rate != PIPELINE_SAMPLE_RATE {
        return (
            StatusCode::BAD_REQUEST,
            format!("rate must be {PIPELINE_SAMPLE_RATE}; resampling is not performed"),
        )
            .into_response();
    }

    let content_type = headers
        .get(axum::http::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .split(';')
        .next()
        .unwrap_or("")
        .trim()
        .to_ascii_lowercase();
    if !ACCEPTED_CONTENT_TYPES.contains(&content_type.as_str()) {
        return (
            StatusCode::UNSUPPORTED_MEDIA_TYPE,
            format!("accepted: {}", ACCEPTED_CONTENT_TYPES.join(", ")),
        )
            .into_response();
    }

    // A replay of an already-admitted segment is a normal condition, not an
    // error: spool replay and live delivery share one path.
    if !ambient.admit_once(&params.segment) {
        debug!("ambient: duplicate segment {} discarded", params.segment);
        return StatusCode::OK.into_response();
    }

    let pcm = match content_type.as_str() {
        "audio/wav" => match decode_wav(&body) {
            Ok(pcm) => pcm,
            Err(e) => return (StatusCode::BAD_REQUEST, e.to_string()).into_response(),
        },
        _ => decode_l16(&body),
    };

    let Some(started_at) = Utc.timestamp_millis_opt(params.started_at).single() else {
        return (StatusCode::BAD_REQUEST, "started_at out of range").into_response();
    };

    let seg = Segment {
        segment: params.segment,
        device,
        started_at,
        live: params.live != 0,
        pcm,
    };

    match ambient.tx.try_send(seg) {
        Ok(()) => StatusCode::OK.into_response(),
        Err(mpsc::error::TrySendError::Full(_)) => {
            warn!("ambient: admission queue full; asking the device to retry");
            StatusCode::TOO_MANY_REQUESTS.into_response()
        }
        Err(mpsc::error::TrySendError::Closed(_)) => {
            warn!("ambient: worker gone; refusing ingest");
            StatusCode::SERVICE_UNAVAILABLE.into_response()
        }
    }
}

async fn handle_hello(
    State(state): State<Arc<ServeState>>,
    headers: HeaderMap,
) -> impl IntoResponse {
    let device = match authenticate(&state, &headers) {
        Ok(d) => d,
        Err(status) => return status.into_response(),
    };
    Json(json!({
        "device": device,
        "sample_rate": PIPELINE_SAMPLE_RATE,
        "accepts": ["audio/L16", "audio/wav"],
        // Reserved as GET /audio/events for S4; not implemented.
        "downlink": false,
    }))
    .into_response()
}

fn decode_l16(bytes: &[u8]) -> Vec<i16> {
    bytes
        .chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]))
        .collect()
}

fn decode_wav(bytes: &[u8]) -> anyhow::Result<Vec<i16>> {
    let mut reader = hound::WavReader::new(std::io::Cursor::new(bytes))?;
    let spec = reader.spec();
    if spec.channels != 1 {
        anyhow::bail!("wav must be mono, got {} channels", spec.channels);
    }
    if spec.sample_rate != PIPELINE_SAMPLE_RATE {
        anyhow::bail!(
            "wav must be {PIPELINE_SAMPLE_RATE} Hz, got {}",
            spec.sample_rate
        );
    }
    Ok(reader.samples::<i16>().collect::<Result<Vec<_>, _>>()?)
}
```

Make `extract_bearer` reachable: in `src/serve/mod.rs` change `fn extract_bearer` to `pub(crate) fn extract_bearer`.

Add to `ServeState` in `src/serve/mod.rs`:

```rust
    /// Ambient capture ingest. `None` when `[ambient].enabled = false` or
    /// no key file is configured — the routes then answer 404, matching
    /// how a disabled `[a2a]` behaves.
    pub(crate) ambient: Option<Arc<crate::ambient::ingest::AmbientState>>,
```

Mount the routes in `run()`:

```rust
    let app = Router::new()
        .route("/rpc", post(rpc_post).get(rpc_get))
        .route("/a2a", post(a2a::handle_a2a_post))
        .route("/mcp", post(mcp::handle_mcp_post))
        .merge(crate::ambient::ingest::routes())
        .route(
            "/.well-known/agent-card.json",
            axum::routing::get(a2a::handle_agent_card),
        )
```

Now replace the `harness()` stub from Step 1 with the real one:

```rust
    fn harness() -> (axum::Router, mpsc::Receiver<Segment>) {
        harness_with_queue_depth(16)
    }

    fn harness_with_queue_depth(depth: usize) -> (axum::Router, mpsc::Receiver<Segment>) {
        let tmp = Box::leak(Box::new(tempfile::tempdir().unwrap()));
        let key_path = tmp.path().join("keys.toml");
        std::fs::write(
            &key_path,
            "[[key]]\ntoken = \"sa-dev-good\"\nlabel = \"pendant-key\"\n",
        )
        .unwrap();
        let id = {
            let store = sapphire_framework::remote_server::KeyStore::load(&key_path).unwrap();
            store.entries()[0].id
        };
        let mut devices = std::collections::HashMap::new();
        devices.insert(
            "pendant".to_string(),
            crate::config::DeviceConfig {
                key_id: id,
                label: None,
                room_profile: None,
            },
        );
        let registry = DeviceRegistry::open(&key_path, &devices).unwrap();
        let (tx, rx) = mpsc::channel(depth);
        let mut cfg = AmbientConfig::default();
        cfg.enabled = true;
        let ambient = Arc::new(AmbientState::new(cfg, registry, tx));
        let state = Arc::new(crate::serve::ServeState::for_ambient_test(ambient));
        (routes().with_state(state), rx)
    }
```

Add the test-only constructor to `src/serve/mod.rs`:

```rust
#[cfg(test)]
impl ServeState {
    /// Minimal state for the ambient endpoint tests: every field the
    /// ambient handlers do not touch is left at its cheapest value.
    /// Kept next to the struct so a new required field breaks here
    /// loudly rather than in a distant test module.
    pub(crate) fn for_ambient_test(
        ambient: Arc<crate::ambient::ingest::AmbientState>,
    ) -> Self {
        // Fill in every existing field with Default::default() /
        // empty collections, then set `ambient`. Follow whatever the
        // struct requires at the time of implementation.
        todo!("construct with the struct's fields as they stand")
    }
}
```

**Implementer note:** `ServeState` is large. If constructing it in a test proves impractical, the acceptable alternative is to make `authenticate`, the params validation and `decode_wav`/`decode_l16` free functions taking `&AmbientState` (they nearly are already), test those directly, and cover the routing with a single smoke test. Do not weaken the assertions — move where they are made.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cargo test --no-default-features --features "redb-store,lancedb-store,fastembed-embed" ambient::ingest`
Expected: PASS, 7 tests.

- [ ] **Step 5: Verify the curl property by hand**

The spec promises a microcontroller can speak this. Confirm the shape is what you'd hand to a firmware author:

```sh
head -c 32000 /dev/zero > /tmp/seg.raw
curl -i -X POST "http://127.0.0.1:8080/audio/ingest?segment=manual-1&started_at=$(date +%s000)&rate=16000&live=0" \
  -H "Authorization: Bearer sa-dev-good" \
  -H "Content-Type: audio/L16" \
  --data-binary @/tmp/seg.raw
```

Expected: `HTTP/1.1 200 OK`. Record the command in the commit message if it needed adjusting.

- [ ] **Step 6: Commit**

```bash
git add src/ambient/ src/serve/mod.rs
git commit -m "feat(ambient): admit capture segments over POST /audio/ingest"
```

---

### Task 6: Segment router

**Files:**
- Create: `src/ambient/router.rs`
- Modify: `src/ambient/mod.rs`
- Test: `src/ambient/router.rs` (inline test module)

**Interfaces:**
- Consumes: `Segment` (Task 5).
- Produces: `pub enum DeviceState { Idle, Conversing }`, `pub enum Disposition { RecordOnly, RecordAndConverse }`, `pub fn route(seg: &Segment, state: DeviceState) -> Disposition`.

**Reviewer note:** this task exists to build a seam nothing uses yet — a deliberate YAGNI exception. The argument is in the spec: S4 changes what a segment *means*, and introducing the fork later means threading it through the whole pipeline, whereas adding a branch to an existing fork is cheap. If that argument is rejected, drop this task and call the worker directly; nothing else in the plan depends on `Disposition` beyond one match arm in Task 10.

- [ ] **Step 1: Write the failing router tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    fn seg(live: bool) -> crate::ambient::ingest::Segment {
        crate::ambient::ingest::Segment {
            segment: "seg".into(),
            device: "pendant".into(),
            started_at: Utc::now(),
            live,
            pcm: vec![0; 16_000],
        }
    }

    #[test]
    fn replayed_audio_never_reaches_the_conversation_branch() {
        // Even if the device were somehow marked Conversing, audio recorded
        // hours ago must not be answered.
        assert_eq!(
            route(&seg(false), DeviceState::Conversing),
            Disposition::RecordOnly
        );
    }

    #[test]
    fn live_audio_from_an_idle_device_is_recorded_only() {
        assert_eq!(
            route(&seg(true), DeviceState::Idle),
            Disposition::RecordOnly
        );
    }

    #[test]
    fn live_audio_from_a_conversing_device_also_converses() {
        assert_eq!(
            route(&seg(true), DeviceState::Conversing),
            Disposition::RecordAndConverse
        );
    }
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test --no-default-features --features "redb-store,lancedb-store,fastembed-embed" ambient::router`
Expected: FAIL — `route` does not exist.

- [ ] **Step 3: Implement the router**

```rust
//! Per-device segment routing.
//!
//! Only [`Disposition::RecordOnly`] is reachable today: the worker pins
//! every device to [`DeviceState::Idle`]. The fork exists now because S4
//! (server-side wake word, live conversation) changes what a segment
//! *means*, and introducing that decision point later would mean threading
//! it through the whole pipeline. Adding a branch to an existing fork is
//! cheap; creating the fork is not.

use super::ingest::Segment;

/// What a device is currently doing. Pinned to `Idle` until S4.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceState {
    Idle,
    #[allow(dead_code)]
    Conversing,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Disposition {
    /// Transcribe and store. No LLM turn.
    RecordOnly,
    /// Transcribe, store, and feed the conversation (S4).
    #[allow(dead_code)]
    RecordAndConverse,
}

/// Decide what to do with `seg`.
///
/// `live` gates the conversation branch unconditionally. This is the
/// safety property the explicit flag exists for: the agent must never
/// answer something said six hours ago because the recording of it just
/// arrived.
pub fn route(seg: &Segment, state: DeviceState) -> Disposition {
    if !seg.live {
        return Disposition::RecordOnly;
    }
    match state {
        DeviceState::Idle => Disposition::RecordOnly,
        DeviceState::Conversing => Disposition::RecordAndConverse,
    }
}
```

Add `pub mod router;` to `src/ambient/mod.rs`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cargo test --no-default-features --features "redb-store,lancedb-store,fastembed-embed" ambient::router`
Expected: PASS, 3 tests.

- [ ] **Step 5: Commit**

```bash
git add src/ambient/
git commit -m "feat(ambient): route segments by liveness and device state"
```

---

### Task 7: Speech gate and speaker embedder abstractions

**Files:**
- Create: `src/ambient/audio.rs`
- Modify: `src/ambient/mod.rs`
- Modify: `Cargo.toml` (no new deps; sherpa code is gated on the existing `voice-sherpa` feature)
- Test: `src/ambient/audio.rs` (inline test module)

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `pub struct GatedSpeech { pub pcm: Vec<i16>, pub speech_ms: u32 }`
  - `pub trait SpeechGate: Send + Sync { fn gate(&self, pcm: &[i16]) -> Option<GatedSpeech>; }`
  - `pub trait SpeakerEmbedder: Send + Sync { fn dim(&self) -> usize; fn embed(&self, pcm: &[i16]) -> anyhow::Result<Vec<f32>>; }`
  - `pub struct PassthroughGate` and `pub struct FixedEmbedder` (test doubles, `#[cfg(test)]`-free so later tasks can use them).
  - `pub struct SileroGate` and `pub struct SherpaEmbedder` behind `#[cfg(feature = "voice-sherpa")]`.

- [ ] **Step 1: Write the failing tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn passthrough_gate_reports_duration_from_sample_count() {
        let gate = PassthroughGate::new();
        // 24000 samples at 16 kHz is 1500 ms.
        let g = gate.gate(&vec![0; 24_000]).expect("speech present");
        assert_eq!(g.speech_ms, 1500);
        assert_eq!(g.pcm.len(), 24_000);
    }

    #[test]
    fn passthrough_gate_reports_nothing_for_an_empty_segment() {
        assert!(PassthroughGate::new().gate(&[]).is_none());
    }

    #[test]
    fn silent_gate_drops_everything() {
        // Models a re-gate that found no speech at all.
        assert!(SilentGate.gate(&vec![0; 24_000]).is_none());
    }

    #[test]
    fn fixed_embedder_returns_the_configured_vector() {
        let e = FixedEmbedder::new(vec![1.0, 0.0, 0.0]);
        assert_eq!(e.dim(), 3);
        assert_eq!(e.embed(&[0; 100]).unwrap(), vec![1.0, 0.0, 0.0]);
    }
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test --no-default-features --features "redb-store,lancedb-store,fastembed-embed" ambient::audio`
Expected: FAIL — the traits do not exist.

- [ ] **Step 3: Implement the traits and test doubles**

```rust
//! VAD re-gating and speaker embedding, behind traits.
//!
//! Both real implementations need sherpa-onnx models, which makes them
//! useless in unit tests and slow to build. The traits let every later
//! task — the worker, the registry, promotion — be tested with cheap
//! doubles, exactly as `MockStt` already does for transcription.

use anyhow::Result;

use crate::voice::PIPELINE_SAMPLE_RATE;

/// Speech surviving the re-gate.
#[derive(Debug, Clone, PartialEq)]
pub struct GatedSpeech {
    pub pcm: Vec<i16>,
    /// Duration of the **speech**, not of the submitted segment. This is
    /// the value compared against `min_embed_ms` and accumulated into a
    /// candidate's promotion total.
    pub speech_ms: u32,
}

/// Second-pass VAD. The capture device runs a cheap classical VAD tuned to
/// over-capture, because on-device the point of gating is to let the radio
/// sleep, not to be accurate. This trims what it sent.
pub trait SpeechGate: Send + Sync {
    /// Speech in `pcm`, or `None` when the segment holds none.
    fn gate(&self, pcm: &[i16]) -> Option<GatedSpeech>;
}

pub trait SpeakerEmbedder: Send + Sync {
    fn dim(&self) -> usize;
    fn embed(&self, pcm: &[i16]) -> Result<Vec<f32>>;
}

pub fn samples_to_ms(samples: usize) -> u32 {
    ((samples as u64 * 1000) / PIPELINE_SAMPLE_RATE as u64) as u32
}

/// Keeps everything it is given. Used in tests and as the fallback when no
/// VAD model is configured — over-keeping costs STT time, never data.
pub struct PassthroughGate;

impl PassthroughGate {
    pub fn new() -> Self {
        Self
    }
}

impl SpeechGate for PassthroughGate {
    fn gate(&self, pcm: &[i16]) -> Option<GatedSpeech> {
        if pcm.is_empty() {
            return None;
        }
        Some(GatedSpeech {
            speech_ms: samples_to_ms(pcm.len()),
            pcm: pcm.to_vec(),
        })
    }
}

/// Drops everything. Test double for "the re-gate found no speech".
pub struct SilentGate;

impl SpeechGate for SilentGate {
    fn gate(&self, _pcm: &[i16]) -> Option<GatedSpeech> {
        None
    }
}

/// Returns one fixed vector regardless of input. Test double.
pub struct FixedEmbedder {
    vector: Vec<f32>,
}

impl FixedEmbedder {
    pub fn new(vector: Vec<f32>) -> Self {
        Self { vector }
    }
}

impl SpeakerEmbedder for FixedEmbedder {
    fn dim(&self) -> usize {
        self.vector.len()
    }
    fn embed(&self, _pcm: &[i16]) -> Result<Vec<f32>> {
        Ok(self.vector.clone())
    }
}
```

- [ ] **Step 4: Add the sherpa-backed implementations**

Append, gated on the feature that already guards every other sherpa use:

```rust
#[cfg(feature = "voice-sherpa")]
mod sherpa_impl {
    use super::*;
    use sherpa_onnx::{
        SileroVadModelConfig, SpeakerEmbeddingExtractor, SpeakerEmbeddingExtractorConfig,
        VadModelConfig, VoiceActivityDetector,
    };

    /// Silero VAD re-gate. Concatenates every detected speech run, so a
    /// segment with a pause in the middle yields one `GatedSpeech` with the
    /// silence removed.
    pub struct SileroGate {
        model_path: String,
        threshold: f32,
    }

    impl SileroGate {
        pub fn new(model_path: String, threshold: f32) -> Self {
            Self {
                model_path,
                threshold,
            }
        }
    }

    impl SpeechGate for SileroGate {
        fn gate(&self, pcm: &[i16]) -> Option<GatedSpeech> {
            let mut config = VadModelConfig::default();
            config.silero_vad = SileroVadModelConfig {
                model: self.model_path.clone(),
                threshold: self.threshold,
                ..Default::default()
            };
            config.sample_rate = PIPELINE_SAMPLE_RATE as i32;
            let vad = VoiceActivityDetector::new(&config, 30.0)?;

            let samples: Vec<f32> = pcm.iter().map(|s| *s as f32 / 32768.0).collect();
            let mut kept: Vec<i16> = Vec::new();
            for window in samples.chunks(512) {
                vad.accept_waveform(window);
                while !vad.is_empty() {
                    let seg = vad.front();
                    kept.extend(seg.samples.iter().map(|s| (s * 32768.0) as i16));
                    vad.pop();
                }
            }
            vad.flush();
            while !vad.is_empty() {
                let seg = vad.front();
                kept.extend(seg.samples.iter().map(|s| (s * 32768.0) as i16));
                vad.pop();
            }
            if kept.is_empty() {
                return None;
            }
            Some(GatedSpeech {
                speech_ms: samples_to_ms(kept.len()),
                pcm: kept,
            })
        }
    }

    pub struct SherpaEmbedder {
        extractor: SpeakerEmbeddingExtractor,
    }

    impl SherpaEmbedder {
        pub fn new(model_path: String, num_threads: i32) -> anyhow::Result<Self> {
            let config = SpeakerEmbeddingExtractorConfig {
                model: model_path,
                num_threads,
                ..Default::default()
            };
            let extractor = SpeakerEmbeddingExtractor::create(&config)
                .ok_or_else(|| anyhow::anyhow!("failed to load speaker embedding model"))?;
            Ok(Self { extractor })
        }
    }

    impl SpeakerEmbedder for SherpaEmbedder {
        fn dim(&self) -> usize {
            self.extractor.dim() as usize
        }

        fn embed(&self, pcm: &[i16]) -> anyhow::Result<Vec<f32>> {
            let stream = self
                .extractor
                .create_stream()
                .ok_or_else(|| anyhow::anyhow!("cannot create embedding stream"))?;
            let samples: Vec<f32> = pcm.iter().map(|s| *s as f32 / 32768.0).collect();
            stream.accept_waveform(PIPELINE_SAMPLE_RATE as i32, &samples);
            stream.input_finished();
            if !self.extractor.is_ready(&stream) {
                anyhow::bail!("not enough audio for a speaker embedding");
            }
            self.extractor
                .compute(&stream)
                .ok_or_else(|| anyhow::anyhow!("embedding computation failed"))
        }
    }
}

#[cfg(feature = "voice-sherpa")]
pub use sherpa_impl::{SherpaEmbedder, SileroGate};
```

**Implementer note:** the exact `sherpa-onnx` 1.13 method names for `VoiceActivityDetector` and `OnlineStream` may differ from the sketch above (`accept_waveform`, `front`, `pop`, `flush`, `is_empty`, `input_finished`). Check the crate source at `~/.cargo/registry/src/*/sherpa-onnx-1.13.3/src/{vad,speaker_embedding}.rs` and adapt. The trait signatures above are the contract and must not change; only the bodies adapt.

Add `pub mod audio;` to `src/ambient/mod.rs`.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `cargo test --no-default-features --features "redb-store,lancedb-store,fastembed-embed" ambient::audio`
Expected: PASS, 4 tests.

- [ ] **Step 6: Verify the sherpa implementations compile**

Run: `cargo check --features voice-sherpa`
Expected: PASS. This is the slow build; expect 5–10 minutes on a cold cache.

- [ ] **Step 7: Commit**

```bash
git add src/ambient/
git commit -m "feat(ambient): speech-gate and speaker-embedder traits with sherpa backends"
```

---

### Task 8: Speaker registry

**Files:**
- Create: `src/ambient/speaker/mod.rs`
- Create: `src/ambient/speaker/registry.rs`
- Modify: `src/ambient/mod.rs`
- Test: `src/ambient/speaker/registry.rs` (inline test module)

**Interfaces:**
- Consumes: `SpeakerEmbedder`, `FixedEmbedder` (Task 7).
- Produces:
  - `pub struct SpeakerMatch { pub id: String, pub score: f32 }`
  - `pub struct SpeakerRegistry`
  - `SpeakerRegistry::open(voices_dir: PathBuf, emb_cache_dir: PathBuf, model_id: String, threshold: f32) -> Result<SpeakerRegistry>`
  - `SpeakerRegistry::load_reference_audio(&mut self, embedder: &dyn SpeakerEmbedder) -> Result<()>`
  - `SpeakerRegistry::match_speaker(&self, embedding: &[f32]) -> Option<SpeakerMatch>`
  - `SpeakerRegistry::add_runtime(&mut self, id: String, embedding: Vec<f32>)`
  - `pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32`

- [ ] **Step 1: Write the failing registry tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::ambient::audio::FixedEmbedder;

    /// Write a 1-second 16 kHz mono WAV of silence.
    fn write_wav(path: &std::path::Path) {
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: 16_000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut w = hound::WavWriter::create(path, spec).unwrap();
        for _ in 0..16_000 {
            w.write_sample(0i16).unwrap();
        }
        w.finalize().unwrap();
    }

    fn voices_with(names: &[&str]) -> (tempfile::TempDir, PathBuf) {
        let tmp = tempfile::tempdir().unwrap();
        let voices = tmp.path().join("voices");
        for n in names {
            let dir = voices.join(n);
            std::fs::create_dir_all(&dir).unwrap();
            write_wav(&dir.join("sample.wav"));
        }
        (tmp, voices)
    }

    #[test]
    fn cosine_similarity_is_one_for_identical_vectors() {
        assert!((cosine_similarity(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0]) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_similarity_is_zero_for_orthogonal_vectors() {
        assert!(cosine_similarity(&[1.0, 0.0], &[0.0, 1.0]).abs() < 1e-6);
    }

    #[test]
    fn matches_a_registered_speaker_above_the_threshold() {
        let (tmp, voices) = voices_with(&["me"]);
        let mut reg = SpeakerRegistry::open(
            voices,
            tmp.path().join("emb"),
            "test-model".into(),
            0.55,
        )
        .unwrap();
        reg.load_reference_audio(&FixedEmbedder::new(vec![1.0, 0.0, 0.0]))
            .unwrap();

        let m = reg.match_speaker(&[1.0, 0.0, 0.0]).expect("matched");
        assert_eq!(m.id, "me");
        assert!(m.score > 0.99);
    }

    #[test]
    fn returns_none_below_the_threshold() {
        let (tmp, voices) = voices_with(&["me"]);
        let mut reg = SpeakerRegistry::open(
            voices,
            tmp.path().join("emb"),
            "test-model".into(),
            0.55,
        )
        .unwrap();
        reg.load_reference_audio(&FixedEmbedder::new(vec![1.0, 0.0, 0.0]))
            .unwrap();
        assert!(reg.match_speaker(&[0.0, 1.0, 0.0]).is_none());
    }

    #[test]
    fn the_directory_name_is_the_speaker_id() {
        let (tmp, voices) = voices_with(&["blithe-otter-42"]);
        let mut reg = SpeakerRegistry::open(
            voices,
            tmp.path().join("emb"),
            "test-model".into(),
            0.55,
        )
        .unwrap();
        reg.load_reference_audio(&FixedEmbedder::new(vec![1.0, 0.0, 0.0]))
            .unwrap();
        assert_eq!(
            reg.match_speaker(&[1.0, 0.0, 0.0]).unwrap().id,
            "blithe-otter-42"
        );
    }

    #[test]
    fn embeddings_are_cached_by_reference_sha_and_model_id() {
        let (tmp, voices) = voices_with(&["me"]);
        let emb_dir = tmp.path().join("emb");
        let mut reg =
            SpeakerRegistry::open(voices.clone(), emb_dir.clone(), "model-a".into(), 0.55).unwrap();
        reg.load_reference_audio(&FixedEmbedder::new(vec![1.0, 0.0, 0.0]))
            .unwrap();
        let after_first = std::fs::read_dir(&emb_dir).unwrap().count();
        assert_eq!(after_first, 1);

        // Same files, different model id: a second cache entry, because the
        // vector is model-dependent and the workspace stores no vectors.
        let mut reg_b =
            SpeakerRegistry::open(voices, emb_dir.clone(), "model-b".into(), 0.55).unwrap();
        reg_b
            .load_reference_audio(&FixedEmbedder::new(vec![0.0, 1.0, 0.0]))
            .unwrap();
        assert_eq!(std::fs::read_dir(&emb_dir).unwrap().count(), 2);
        assert_eq!(reg_b.match_speaker(&[0.0, 1.0, 0.0]).unwrap().id, "me");
    }

    #[test]
    fn an_unreadable_reference_file_disables_that_speaker_without_failing_the_load() {
        let (tmp, voices) = voices_with(&["me"]);
        let broken = voices.join("broken");
        std::fs::create_dir_all(&broken).unwrap();
        std::fs::write(broken.join("sample.wav"), b"not a wav at all").unwrap();

        let mut reg = SpeakerRegistry::open(
            voices,
            tmp.path().join("emb"),
            "test-model".into(),
            0.55,
        )
        .unwrap();
        reg.load_reference_audio(&FixedEmbedder::new(vec![1.0, 0.0, 0.0]))
            .expect("load succeeds despite the broken speaker");
        assert_eq!(reg.match_speaker(&[1.0, 0.0, 0.0]).unwrap().id, "me");
    }
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test --no-default-features --features "redb-store,lancedb-store,fastembed-embed" ambient::speaker::registry`
Expected: FAIL — `SpeakerRegistry` does not exist.

- [ ] **Step 3: Implement the registry**

`src/ambient/speaker/mod.rs`:

```rust
//! Speaker identity: the workspace registry and the candidate store.

pub mod registry;
```

`src/ambient/speaker/registry.rs`:

```rust
//! Registered speakers, loaded from workspace reference audio.
//!
//! `voices/<id>/*.wav` — the directory name is **both** the speaker id and
//! the display name. Embeddings are cached outside the workspace, keyed by
//! (reference file sha256 x model id), so renaming a directory triggers no
//! recomputation and swapping the embedding model recomputes automatically.
//! No model-dependent data ever lands in the workspace, which was the
//! requirement that shaped this split.

use std::collections::HashMap;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use sha2::{Digest, Sha256};
use tracing::warn;

use crate::ambient::audio::SpeakerEmbedder;

#[derive(Debug, Clone, PartialEq)]
pub struct SpeakerMatch {
    pub id: String,
    pub score: f32,
}

pub struct SpeakerRegistry {
    voices_dir: PathBuf,
    emb_cache_dir: PathBuf,
    model_id: String,
    threshold: f32,
    /// Speaker id -> its centroid embedding.
    speakers: HashMap<String, Vec<f32>>,
}

impl SpeakerRegistry {
    pub fn open(
        voices_dir: PathBuf,
        emb_cache_dir: PathBuf,
        model_id: String,
        threshold: f32,
    ) -> Result<Self> {
        std::fs::create_dir_all(&emb_cache_dir)
            .with_context(|| format!("creating embedding cache dir {emb_cache_dir:?}"))?;
        Ok(Self {
            voices_dir,
            emb_cache_dir,
            model_id,
            threshold,
            speakers: HashMap::new(),
        })
    }

    /// Scan `voices/`, embedding each reference file (or reading its cached
    /// vector) and averaging per speaker.
    ///
    /// A speaker whose files cannot be read is warned about and skipped.
    /// One unreadable WAV must not take ambient ingest down with it.
    pub fn load_reference_audio(&mut self, embedder: &dyn SpeakerEmbedder) -> Result<()> {
        let entries = match std::fs::read_dir(&self.voices_dir) {
            Ok(e) => e,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
            Err(e) => return Err(e).context("reading voices dir"),
        };
        for entry in entries.flatten() {
            if !entry.file_type().map(|t| t.is_dir()).unwrap_or(false) {
                continue;
            }
            let id = entry.file_name().to_string_lossy().into_owned();
            let mut vectors = Vec::new();
            for file in std::fs::read_dir(entry.path()).into_iter().flatten().flatten() {
                let path = file.path();
                if path.extension().and_then(|e| e.to_str()) != Some("wav") {
                    continue;
                }
                match self.embedding_for(&path, embedder) {
                    Ok(v) => vectors.push(v),
                    Err(e) => warn!("speaker {id}: skipping {path:?}: {e}"),
                }
            }
            if vectors.is_empty() {
                warn!("speaker {id}: no usable reference audio; speaker disabled");
                continue;
            }
            self.speakers.insert(id, centroid(&vectors));
        }
        Ok(())
    }

    /// Cached embedding for one reference file, computing and storing it on
    /// a miss. The cache key is the file's content hash and the model id.
    fn embedding_for(&self, path: &Path, embedder: &dyn SpeakerEmbedder) -> Result<Vec<f32>> {
        let bytes = std::fs::read(path).with_context(|| format!("reading {path:?}"))?;
        let key = format!("{}.{}.emb", sha256_hex(&bytes), self.model_id);
        let cached = self.emb_cache_dir.join(&key);
        if let Ok(raw) = std::fs::read(&cached) {
            return Ok(decode_embedding(&raw));
        }
        let pcm = read_wav_mono_16k(&bytes)?;
        let vector = embedder.embed(&pcm)?;
        if let Err(e) = std::fs::write(&cached, encode_embedding(&vector)) {
            warn!("could not cache embedding at {cached:?}: {e}");
        }
        Ok(vector)
    }

    /// Register an embedding at runtime — used for auto-enrolled candidates,
    /// so a newly seen voice matches on its next segment.
    pub fn add_runtime(&mut self, id: String, embedding: Vec<f32>) {
        self.speakers.insert(id, embedding);
    }

    /// Best speaker above the threshold, or `None`.
    pub fn match_speaker(&self, embedding: &[f32]) -> Option<SpeakerMatch> {
        let mut best: Option<SpeakerMatch> = None;
        for (id, centroid) in &self.speakers {
            let score = cosine_similarity(embedding, centroid);
            if score < self.threshold {
                continue;
            }
            if best.as_ref().is_none_or(|b| score > b.score) {
                best = Some(SpeakerMatch {
                    id: id.clone(),
                    score,
                });
            }
        }
        best
    }

    /// Display name for a speaker id. Names live in the workspace, so a
    /// renamed directory is picked up here without touching transcripts.
    pub fn display_name(&self, id: &str) -> String {
        id.to_string()
    }
}

pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    dot / (na * nb)
}

fn centroid(vectors: &[Vec<f32>]) -> Vec<f32> {
    let dim = vectors[0].len();
    let mut out = vec![0.0; dim];
    for v in vectors {
        for (i, x) in v.iter().take(dim).enumerate() {
            out[i] += x;
        }
    }
    for x in &mut out {
        *x /= vectors.len() as f32;
    }
    out
}

pub fn encode_embedding(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|f| f.to_le_bytes()).collect()
}

pub fn decode_embedding(raw: &[u8]) -> Vec<f32> {
    raw.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn read_wav_mono_16k(bytes: &[u8]) -> Result<Vec<i16>> {
    let mut reader = hound::WavReader::new(std::io::Cursor::new(bytes))?;
    let spec = reader.spec();
    anyhow::ensure!(spec.channels == 1, "reference audio must be mono");
    anyhow::ensure!(
        spec.sample_rate == crate::voice::PIPELINE_SAMPLE_RATE,
        "reference audio must be {} Hz, got {}",
        crate::voice::PIPELINE_SAMPLE_RATE,
        spec.sample_rate
    );
    Ok(reader.samples::<i16>().collect::<Result<Vec<_>, _>>()?)
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut out = String::with_capacity(64);
    for byte in digest {
        let _ = write!(out, "{byte:02x}");
    }
    out
}
```

Add `pub mod speaker;` to `src/ambient/mod.rs`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cargo test --no-default-features --features "redb-store,lancedb-store,fastembed-embed" ambient::speaker::registry`
Expected: PASS, 7 tests.

- [ ] **Step 5: Commit**

```bash
git add src/ambient/
git commit -m "feat(ambient): match speakers against workspace reference audio"
```

---

### Task 9: Candidates and promotion

**Files:**
- Create: `src/ambient/speaker/candidates.rs`
- Modify: `src/ambient/speaker/mod.rs`
- Test: `src/ambient/speaker/candidates.rs` (inline test module)

**Interfaces:**
- Consumes: `cosine_similarity`, `encode_embedding`, `decode_embedding` (Task 8).
- Produces:
  - `pub struct CandidateStats { pub speech_seconds: u32, pub days_seen: Vec<NaiveDate>, pub first_seen: DateTime<Utc> }`
  - `pub struct Candidate { pub id: String, pub centroid: Vec<f32>, pub stats: CandidateStats, pub samples: Vec<String> }`
  - `pub struct CandidateStore`
  - `CandidateStore::open(dir: PathBuf) -> Result<CandidateStore>`
  - `CandidateStore::enrol(&mut self, embedding: Vec<f32>, clip: &[i16], day: NaiveDate, speech_ms: u32, sample_text: &str) -> Result<String>`
  - `CandidateStore::observe(&mut self, id: &str, embedding: &[f32], day: NaiveDate, speech_ms: u32, sample_text: &str) -> Result<()>`
  - `CandidateStore::list(&self) -> Vec<&Candidate>`
  - `CandidateStore::is_promotable(&self, id: &str, after_seconds: u32, after_days: u32) -> bool`
  - `CandidateStore::promote(&mut self, id: &str, name: Option<&str>, voices_dir: &Path) -> Result<String>`

- [ ] **Step 1: Write the failing candidate tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use chrono::NaiveDate;

    fn day(d: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(2026, 8, d).unwrap()
    }

    fn store() -> (tempfile::TempDir, CandidateStore) {
        let tmp = tempfile::tempdir().unwrap();
        let store = CandidateStore::open(tmp.path().join("candidates")).unwrap();
        (tmp, store)
    }

    #[test]
    fn enrolling_mints_an_id_and_persists_the_candidate() {
        let (tmp, mut store) = store();
        let id = store
            .enrol(vec![1.0, 0.0], &vec![0i16; 16_000], day(26), 1_000, "hello")
            .unwrap();
        assert!(!id.is_empty());
        assert!(tmp.path().join("candidates").join(&id).join("clip.wav").exists());
        assert!(tmp.path().join("candidates").join(&id).join("stats.json").exists());

        // Reload from disk: candidates survive a restart.
        let reloaded = CandidateStore::open(tmp.path().join("candidates")).unwrap();
        assert_eq!(reloaded.list().len(), 1);
        assert_eq!(reloaded.list()[0].id, id);
    }

    #[test]
    fn is_not_promotable_at_fifty_nine_seconds() {
        let (_tmp, mut store) = store();
        let id = store
            .enrol(vec![1.0, 0.0], &vec![0i16; 16_000], day(26), 59_000, "a")
            .unwrap();
        store.observe(&id, &[1.0, 0.0], day(27), 0, "b").unwrap();
        assert!(!store.is_promotable(&id, 60, 2), "59s < 60s threshold");
    }

    #[test]
    fn is_promotable_at_exactly_sixty_seconds_across_two_days() {
        let (_tmp, mut store) = store();
        let id = store
            .enrol(vec![1.0, 0.0], &vec![0i16; 16_000], day(26), 30_000, "a")
            .unwrap();
        store.observe(&id, &[1.0, 0.0], day(27), 30_000, "b").unwrap();
        assert!(store.is_promotable(&id, 60, 2));
    }

    #[test]
    fn is_not_promotable_on_a_single_day_however_long() {
        let (_tmp, mut store) = store();
        let id = store
            .enrol(vec![1.0, 0.0], &vec![0i16; 16_000], day(26), 600_000, "a")
            .unwrap();
        assert!(
            !store.is_promotable(&id, 60, 2),
            "one day of television must not promote"
        );
    }

    #[test]
    fn observing_the_same_day_twice_counts_it_once() {
        let (_tmp, mut store) = store();
        let id = store
            .enrol(vec![1.0, 0.0], &vec![0i16; 16_000], day(26), 40_000, "a")
            .unwrap();
        store.observe(&id, &[1.0, 0.0], day(26), 40_000, "b").unwrap();
        assert!(!store.is_promotable(&id, 60, 2), "still only one day seen");
    }

    #[test]
    fn promotion_writes_the_clip_into_the_workspace_under_the_id() {
        let (tmp, mut store) = store();
        let voices = tmp.path().join("voices");
        let id = store
            .enrol(vec![1.0, 0.0], &vec![0i16; 16_000], day(26), 60_000, "a")
            .unwrap();
        let name = store.promote(&id, None, &voices).unwrap();
        assert_eq!(name, id, "no name given, so the grain-id is the directory");
        assert!(voices.join(&id).join("clip.wav").exists());
        assert!(store.list().iter().all(|c| c.id != id), "promoted, so no longer a candidate");
    }

    #[test]
    fn promotion_with_a_name_uses_it_as_the_directory() {
        let (tmp, mut store) = store();
        let voices = tmp.path().join("voices");
        let id = store
            .enrol(vec![1.0, 0.0], &vec![0i16; 16_000], day(26), 60_000, "a")
            .unwrap();
        let name = store.promote(&id, Some("tanaka-san"), &voices).unwrap();
        assert_eq!(name, "tanaka-san");
        assert!(voices.join("tanaka-san").join("clip.wav").exists());
    }

    #[test]
    fn promotion_refuses_a_name_that_would_escape_the_voices_directory() {
        let (tmp, mut store) = store();
        let voices = tmp.path().join("voices");
        let id = store
            .enrol(vec![1.0, 0.0], &vec![0i16; 16_000], day(26), 60_000, "a")
            .unwrap();
        assert!(store.promote(&id, Some("../../etc/passwd"), &voices).is_err());
        assert!(store.promote(&id, Some("has/slash"), &voices).is_err());
    }
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test --no-default-features --features "redb-store,lancedb-store,fastembed-embed" ambient::speaker::candidates`
Expected: FAIL — `CandidateStore` does not exist.

- [ ] **Step 3: Implement the candidate store**

```rust
//! Auto-enrolled speakers awaiting a name.
//!
//! A day of ambient audio contains television, shop staff, train
//! announcements and passers-by. Writing every first-seen voice straight
//! into the workspace would bury the handful of people worth naming under
//! hundreds of one-off entries, so first sight enrols a **candidate** in
//! the cache, and only sustained presence promotes it into `voices/`.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use chrono::{DateTime, NaiveDate, Utc};
use serde::{Deserialize, Serialize};
use tracing::warn;

use super::registry::{decode_embedding, encode_embedding};
use crate::voice::PIPELINE_SAMPLE_RATE;

/// How many sample utterances to keep per candidate, so a human deciding
/// whether to name it has something to recognise it by.
const MAX_SAMPLES: usize = 5;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CandidateStats {
    /// Cumulative gated speech, in whole seconds.
    pub speech_seconds: u32,
    /// Distinct logical days this voice was heard on.
    pub days_seen: Vec<NaiveDate>,
    pub first_seen: DateTime<Utc>,
    /// A few transcribed utterances, for recognition.
    #[serde(default)]
    pub samples: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct Candidate {
    pub id: String,
    pub centroid: Vec<f32>,
    pub stats: CandidateStats,
    /// Number of observations folded into `centroid`, for the running mean.
    observations: u32,
}

pub struct CandidateStore {
    dir: PathBuf,
    candidates: HashMap<String, Candidate>,
}

impl CandidateStore {
    /// Open the store, loading every candidate already on disk.
    pub fn open(dir: PathBuf) -> Result<Self> {
        std::fs::create_dir_all(&dir)
            .with_context(|| format!("creating candidate dir {dir:?}"))?;
        let mut candidates = HashMap::new();
        for entry in std::fs::read_dir(&dir)?.flatten() {
            if !entry.file_type().map(|t| t.is_dir()).unwrap_or(false) {
                continue;
            }
            let id = entry.file_name().to_string_lossy().into_owned();
            match load_candidate(&entry.path(), &id) {
                Ok(c) => {
                    candidates.insert(id, c);
                }
                Err(e) => warn!("skipping unreadable candidate {id}: {e}"),
            }
        }
        Ok(Self { dir, candidates })
    }

    /// Register a newly seen voice. Returns the minted id.
    pub fn enrol(
        &mut self,
        embedding: Vec<f32>,
        clip: &[i16],
        day: NaiveDate,
        speech_ms: u32,
        sample_text: &str,
    ) -> Result<String> {
        let id = grain_id::GrainId::new().to_string();
        let dir = self.dir.join(&id);
        std::fs::create_dir_all(&dir)?;
        write_wav(&dir.join("clip.wav"), clip)?;
        std::fs::write(dir.join("centroid.emb"), encode_embedding(&embedding))?;

        let stats = CandidateStats {
            speech_seconds: speech_ms / 1000,
            days_seen: vec![day],
            first_seen: Utc::now(),
            samples: sample_vec(sample_text),
        };
        std::fs::write(dir.join("stats.json"), serde_json::to_vec_pretty(&stats)?)?;

        self.candidates.insert(
            id.clone(),
            Candidate {
                id: id.clone(),
                centroid: embedding,
                stats,
                observations: 1,
            },
        );
        Ok(id)
    }

    /// Fold another segment into an existing candidate.
    pub fn observe(
        &mut self,
        id: &str,
        embedding: &[f32],
        day: NaiveDate,
        speech_ms: u32,
        sample_text: &str,
    ) -> Result<()> {
        let Some(c) = self.candidates.get_mut(id) else {
            bail!("no such candidate: {id}");
        };
        // Running mean, so a candidate's centroid tracks the voice rather
        // than being pinned to whatever the first segment sounded like.
        let n = c.observations as f32;
        for (i, x) in embedding.iter().enumerate().take(c.centroid.len()) {
            c.centroid[i] = (c.centroid[i] * n + x) / (n + 1.0);
        }
        c.observations += 1;
        c.stats.speech_seconds += speech_ms / 1000;
        if !c.stats.days_seen.contains(&day) {
            c.stats.days_seen.push(day);
        }
        if !sample_text.trim().is_empty() && c.stats.samples.len() < MAX_SAMPLES {
            c.stats.samples.push(sample_text.to_string());
        }

        let dir = self.dir.join(id);
        std::fs::write(dir.join("centroid.emb"), encode_embedding(&c.centroid))?;
        std::fs::write(dir.join("stats.json"), serde_json::to_vec_pretty(&c.stats)?)?;
        Ok(())
    }

    pub fn list(&self) -> Vec<&Candidate> {
        let mut out: Vec<&Candidate> = self.candidates.values().collect();
        out.sort_by(|a, b| b.stats.speech_seconds.cmp(&a.stats.speech_seconds));
        out
    }

    pub fn get(&self, id: &str) -> Option<&Candidate> {
        self.candidates.get(id)
    }

    /// Both thresholds must be cleared. Cumulative time alone would promote
    /// a television left on all afternoon; distinct days alone would promote
    /// a passing greeting heard twice.
    pub fn is_promotable(&self, id: &str, after_seconds: u32, after_days: u32) -> bool {
        self.candidates.get(id).is_some_and(|c| {
            c.stats.speech_seconds >= after_seconds
                && c.stats.days_seen.len() as u32 >= after_days
        })
    }

    /// Export a candidate into `voices/<name>/` and stop tracking it.
    /// Returns the directory name used.
    pub fn promote(&mut self, id: &str, name: Option<&str>, voices_dir: &Path) -> Result<String> {
        let Some(c) = self.candidates.get(id) else {
            bail!("no such candidate: {id}");
        };
        let dir_name = match name {
            Some(n) => {
                validate_speaker_name(n)?;
                n.to_string()
            }
            None => c.id.clone(),
        };
        let target = voices_dir.join(&dir_name);
        std::fs::create_dir_all(&target)
            .with_context(|| format!("creating {target:?}"))?;
        std::fs::copy(self.dir.join(id).join("clip.wav"), target.join("clip.wav"))
            .with_context(|| format!("copying candidate clip into {target:?}"))?;

        std::fs::remove_dir_all(self.dir.join(id)).ok();
        self.candidates.remove(id);
        Ok(dir_name)
    }
}

/// A speaker name becomes a directory name under the workspace, so it must
/// be a single harmless path segment.
fn validate_speaker_name(name: &str) -> Result<()> {
    if name.trim().is_empty() {
        bail!("speaker name must not be empty");
    }
    if name.contains('/') || name.contains('\\') || name.contains("..") {
        bail!("speaker name must be a single path segment: {name:?}");
    }
    Ok(())
}

fn sample_vec(text: &str) -> Vec<String> {
    if text.trim().is_empty() {
        Vec::new()
    } else {
        vec![text.to_string()]
    }
}

fn load_candidate(dir: &Path, id: &str) -> Result<Candidate> {
    let centroid = decode_embedding(&std::fs::read(dir.join("centroid.emb"))?);
    let stats: CandidateStats = serde_json::from_slice(&std::fs::read(dir.join("stats.json"))?)?;
    Ok(Candidate {
        id: id.to_string(),
        centroid,
        observations: stats.days_seen.len().max(1) as u32,
        stats,
    })
}

fn write_wav(path: &Path, pcm: &[i16]) -> Result<()> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: PIPELINE_SAMPLE_RATE,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut w = hound::WavWriter::create(path, spec)?;
    for s in pcm {
        w.write_sample(*s)?;
    }
    w.finalize()?;
    Ok(())
}
```

Make `encode_embedding` / `decode_embedding` `pub` in `registry.rs` (they already are in Task 8's listing) and add `pub mod candidates;` to `src/ambient/speaker/mod.rs`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cargo test --no-default-features --features "redb-store,lancedb-store,fastembed-embed" ambient::speaker::candidates`
Expected: PASS, 8 tests.

- [ ] **Step 5: Commit**

```bash
git add src/ambient/
git commit -m "feat(ambient): enrol unknown voices as candidates, promote on sustained presence"
```

---

### Task 10: Processing worker

**Files:**
- Create: `src/ambient/worker.rs`
- Modify: `src/ambient/mod.rs`
- Test: `src/ambient/worker.rs` (inline test module)

**Interfaces:**
- Consumes: `Segment` (5), `route`/`DeviceState`/`Disposition` (6), `SpeechGate`/`SpeakerEmbedder` (7), `SpeakerRegistry` (8), `CandidateStore` (9), `TranscriptStore` (4), `AudioCache` (3), `SttProvider` (existing).
- Produces: `pub struct Worker` and `Worker::process(&mut self, seg: Segment) -> Result<Option<TranscriptRecord>>`, plus `pub async fn run(worker: Worker, rx: mpsc::Receiver<Segment>)`.

- [ ] **Step 1: Write the failing worker tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::ambient::audio::{FixedEmbedder, PassthroughGate, SilentGate};
    use crate::ambient::speaker::registry::SpeakerRegistry;
    use chrono::Utc;

    fn seg(pcm_len: usize, live: bool) -> Segment {
        Segment {
            segment: format!("seg-{pcm_len}-{live}"),
            device: "pendant".into(),
            started_at: Utc::now(),
            live,
            pcm: vec![0; pcm_len],
        }
    }

    struct Harness {
        _tmp: tempfile::TempDir,
        worker: Worker,
    }

    fn harness(
        gate: Box<dyn crate::ambient::audio::SpeechGate>,
        embedder: Box<dyn crate::ambient::audio::SpeakerEmbedder>,
        registered: &[(&str, Vec<f32>)],
    ) -> Harness {
        let tmp = tempfile::tempdir().unwrap();
        let root = tmp.path();
        let cache = crate::ambient::cache::AudioCache::open(root.join("audio")).unwrap();
        let transcripts =
            crate::ambient::transcript::TranscriptStore::open(root.join("transcripts"), 4).unwrap();
        let mut registry = SpeakerRegistry::open(
            root.join("voices"),
            root.join("emb"),
            "test-model".into(),
            0.55,
        )
        .unwrap();
        for (id, v) in registered {
            registry.add_runtime(id.to_string(), v.clone());
        }
        let candidates =
            crate::ambient::speaker::candidates::CandidateStore::open(root.join("candidates"))
                .unwrap();
        let stt = std::sync::Arc::new(crate::voice::providers::MockStt::new(
            "mock".into(),
            "transcribed text".into(),
        ));
        let worker = Worker {
            gate,
            embedder,
            stt,
            registry,
            candidates,
            cache,
            transcripts,
            voices_dir: root.join("voices"),
            min_embed_ms: 1500,
            promote_after_seconds: 60,
            promote_after_days: 2,
            day_boundary_hour: 4,
            language: None,
        };
        Harness { _tmp: tmp, worker }
    }

    #[tokio::test]
    async fn a_silence_only_segment_produces_no_transcript() {
        let mut h = harness(Box::new(SilentGate), Box::new(FixedEmbedder::new(vec![1.0, 0.0])), &[]);
        let out = h.worker.process(seg(16_000, false)).await.unwrap();
        assert!(out.is_none(), "re-gate found no speech");
    }

    #[tokio::test]
    async fn a_registered_speaker_is_named_on_the_transcript() {
        let mut h = harness(
            Box::new(PassthroughGate::new()),
            Box::new(FixedEmbedder::new(vec![1.0, 0.0])),
            &[("me", vec![1.0, 0.0])],
        );
        let rec = h.worker.process(seg(32_000, true)).await.unwrap().unwrap();
        assert_eq!(rec.speaker.as_deref(), Some("me"));
        assert!(rec.speaker_score.unwrap() > 0.99);
        assert_eq!(rec.text, "transcribed text");
        assert_eq!(rec.speech_ms, 2000);
        assert_eq!(rec.device, "pendant");
    }

    #[tokio::test]
    async fn a_short_segment_gets_a_transcript_but_no_speaker() {
        let mut h = harness(
            Box::new(PassthroughGate::new()),
            Box::new(FixedEmbedder::new(vec![1.0, 0.0])),
            &[("me", vec![1.0, 0.0])],
        );
        // 16000 samples = 1000 ms, below min_embed_ms of 1500.
        let rec = h.worker.process(seg(16_000, true)).await.unwrap().unwrap();
        assert_eq!(rec.speaker, None, "too short to attribute reliably");
        assert_eq!(rec.speaker_score, None);
        assert_eq!(rec.text, "transcribed text");
    }

    #[tokio::test]
    async fn an_unmatched_voice_is_enrolled_and_matches_on_its_next_segment() {
        let mut h = harness(
            Box::new(PassthroughGate::new()),
            Box::new(FixedEmbedder::new(vec![0.0, 1.0])),
            &[("me", vec![1.0, 0.0])],
        );
        let first = h.worker.process(seg(32_000, true)).await.unwrap().unwrap();
        let enrolled = first.speaker.expect("enrolled on first sight");
        assert_ne!(enrolled, "me");
        assert_eq!(h.worker.candidates.list().len(), 1);

        let mut next = seg(32_000, true);
        next.segment = "seg-second".into();
        let second = h.worker.process(next).await.unwrap().unwrap();
        assert_eq!(
            second.speaker.as_deref(),
            Some(enrolled.as_str()),
            "same voice, same id — this is what gives cross-day stability"
        );
        assert_eq!(h.worker.candidates.list().len(), 1, "no second candidate");
    }

    #[tokio::test]
    async fn the_audio_blob_is_cached_and_referenced_by_the_transcript() {
        let mut h = harness(
            Box::new(PassthroughGate::new()),
            Box::new(FixedEmbedder::new(vec![1.0, 0.0])),
            &[("me", vec![1.0, 0.0])],
        );
        let rec = h.worker.process(seg(32_000, true)).await.unwrap().unwrap();
        assert_eq!(rec.audio.len(), 64);
        assert!(h.worker.cache.get(&rec.audio).is_some());
    }

    #[tokio::test]
    async fn a_candidate_clearing_both_thresholds_is_promoted_into_the_workspace() {
        let mut h = harness(
            Box::new(PassthroughGate::new()),
            Box::new(FixedEmbedder::new(vec![0.0, 1.0])),
            &[],
        );
        // One long segment on day one, one on day two.
        let mut a = seg(16_000 * 40, true); // 40 s
        a.started_at = chrono::Utc::now() - chrono::Duration::days(1);
        a.segment = "day-one".into();
        h.worker.process(a).await.unwrap();

        let mut b = seg(16_000 * 40, true); // another 40 s
        b.segment = "day-two".into();
        h.worker.process(b).await.unwrap();

        let voices = h.worker.voices_dir.clone();
        let promoted: Vec<_> = std::fs::read_dir(&voices)
            .unwrap()
            .filter_map(|e| e.ok())
            .collect();
        assert_eq!(promoted.len(), 1, "80s over two days clears both thresholds");
        assert!(promoted[0].path().join("clip.wav").exists());
    }
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test --no-default-features --features "redb-store,lancedb-store,fastembed-embed" ambient::worker`
Expected: FAIL — `Worker` does not exist.

- [ ] **Step 3: Implement the worker**

```rust
//! The ambient processing pipeline.
//!
//! Re-gate, transcribe, attribute, store. **Nothing here starts an LLM
//! turn** — that is the whole distinction between `ambient` and `voice`,
//! and the [`Disposition::RecordAndConverse`] arm is deliberately left
//! unreachable until S4.

use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Result;
use chrono::Local;
use tokio::sync::mpsc;
use tracing::{info, warn};

use super::audio::{SpeakerEmbedder, SpeechGate};
use super::cache::AudioCache;
use super::ingest::Segment;
use super::router::{DeviceState, Disposition, route};
use super::speaker::candidates::CandidateStore;
use super::speaker::registry::SpeakerRegistry;
use super::transcript::{TranscriptRecord, TranscriptStore};
use crate::session::local_date_for_timestamp;
use crate::voice::SttProvider;

pub struct Worker {
    pub gate: Box<dyn SpeechGate>,
    pub embedder: Box<dyn SpeakerEmbedder>,
    pub stt: Arc<dyn SttProvider>,
    pub registry: SpeakerRegistry,
    pub candidates: CandidateStore,
    pub cache: Arc<AudioCache>,
    pub transcripts: TranscriptStore,
    pub voices_dir: PathBuf,
    pub min_embed_ms: u32,
    pub promote_after_seconds: u32,
    pub promote_after_days: u32,
    pub day_boundary_hour: u8,
    pub language: Option<String>,
}

impl Worker {
    /// Process one segment. Returns the stored record, or `None` when the
    /// segment held no speech.
    pub async fn process(&mut self, seg: Segment) -> Result<Option<TranscriptRecord>> {
        // Every device is pinned to Idle until S4 gives the state machine
        // something to change it with.
        match route(&seg, DeviceState::Idle) {
            Disposition::RecordOnly => {}
            Disposition::RecordAndConverse => {
                warn!("ambient: conversation disposition is not implemented; recording only");
            }
        }

        let Some(gated) = self.gate.gate(&seg.pcm) else {
            info!("ambient: segment {} held no speech; dropped", seg.segment);
            return Ok(None);
        };

        let audio_sha = self.cache.put(&pcm_to_bytes(&gated.pcm))?;
        let text = self
            .stt
            .transcribe(&gated.pcm, self.language.as_deref())
            .await?;

        let day = local_date_for_timestamp(seg.started_at.with_timezone(&Local), self.day_boundary_hour);

        let (speaker, speaker_score) = if gated.speech_ms < self.min_embed_ms {
            // Embeddings from very short utterances are unreliable, and
            // trusting them is the main driver of speaker-id inflation.
            (None, None)
        } else {
            match self.embedder.embed(&gated.pcm) {
                Ok(embedding) => self.attribute(&embedding, &gated.pcm, day, gated.speech_ms, &text)?,
                Err(e) => {
                    warn!("ambient: embedding failed for {}: {e}", seg.segment);
                    (None, None)
                }
            }
        };

        let record = TranscriptRecord {
            segment: seg.segment,
            device: seg.device,
            started_at: seg.started_at,
            speech_ms: gated.speech_ms,
            speaker,
            speaker_score,
            text,
            audio: audio_sha,
        };
        self.transcripts.append(&record)?;
        Ok(Some(record))
    }

    /// Match, or enrol on a miss so the same voice matches next time.
    fn attribute(
        &mut self,
        embedding: &[f32],
        pcm: &[i16],
        day: chrono::NaiveDate,
        speech_ms: u32,
        text: &str,
    ) -> Result<(Option<String>, Option<f32>)> {
        if let Some(m) = self.registry.match_speaker(embedding) {
            // A match may be a candidate rather than a registered speaker;
            // observing keeps its statistics and centroid current.
            if self.candidates.get(&m.id).is_some() {
                self.candidates.observe(&m.id, embedding, day, speech_ms, text)?;
                self.maybe_promote(&m.id)?;
            }
            return Ok((Some(m.id), Some(m.score)));
        }

        let id = self
            .candidates
            .enrol(embedding.to_vec(), pcm, day, speech_ms, text)?;
        self.registry.add_runtime(id.clone(), embedding.to_vec());
        info!("ambient: enrolled a new voice as {id}");
        self.maybe_promote(&id)?;
        Ok((Some(id), None))
    }

    fn maybe_promote(&mut self, id: &str) -> Result<()> {
        if !self
            .candidates
            .is_promotable(id, self.promote_after_seconds, self.promote_after_days)
        {
            return Ok(());
        }
        match self.candidates.promote(id, None, &self.voices_dir) {
            Ok(name) => info!("ambient: promoted candidate {id} to voices/{name}; rename it to finish registering"),
            Err(e) => warn!("ambient: could not promote {id}: {e}"),
        }
        Ok(())
    }
}

/// Drain the admission queue forever. One segment at a time on purpose:
/// enrolment mutates shared speaker state, and a day of spooled audio
/// arriving at once is a throughput problem, not a latency one.
pub async fn run(mut worker: Worker, mut rx: mpsc::Receiver<Segment>) {
    while let Some(seg) = rx.recv().await {
        let id = seg.segment.clone();
        if let Err(e) = worker.process(seg).await {
            // One bad segment must never stop the pipeline; the audio stays
            // in the cache either way.
            warn!("ambient: segment {id} failed: {e}");
        }
    }
    info!("ambient: admission queue closed; worker stopping");
}

fn pcm_to_bytes(pcm: &[i16]) -> Vec<u8> {
    pcm.iter().flat_map(|s| s.to_le_bytes()).collect()
}
```

Add `pub mod worker;` to `src/ambient/mod.rs`. Make `crate::voice::providers::MockStt` reachable from tests — if `providers` is private, add `pub(crate) use providers::MockStt;` to `src/voice/mod.rs`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cargo test --no-default-features --features "redb-store,lancedb-store,fastembed-embed" ambient::worker`
Expected: PASS, 6 tests.

- [ ] **Step 5: Commit**

```bash
git add src/ambient/ src/voice/mod.rs
git commit -m "feat(ambient): transcribe, attribute and store admitted segments"
```

---

### Task 11: Agent-facing tools

**Files:**
- Create: `src/tools/ambient_tools.rs`
- Modify: `src/tools/mod.rs`
- Test: `src/tools/ambient_tools.rs` (inline test module)

**Interfaces:**
- Consumes: `TranscriptStore` (4), `CandidateStore` (9), `SpeakerRegistry` (8).
- Produces: `TranscriptReadTool`, `SpeakerCandidatesTool`, `SpeakerPromoteTool`, each implementing `crate::tools::Tool`, plus `pub fn ambient_tools(state: Arc<Mutex<AmbientToolState>>) -> Vec<Box<dyn Tool>>` and `pub struct AmbientToolState { pub transcripts: TranscriptStore, pub candidates: CandidateStore, pub voices_dir: PathBuf }`.

- [ ] **Step 1: Write the failing tool tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use chrono::{TimeZone, Utc};
    use serde_json::json;

    fn state() -> (tempfile::TempDir, Arc<Mutex<AmbientToolState>>) {
        let tmp = tempfile::tempdir().unwrap();
        let transcripts =
            crate::ambient::transcript::TranscriptStore::open(tmp.path().join("t"), 4).unwrap();
        let candidates =
            crate::ambient::speaker::candidates::CandidateStore::open(tmp.path().join("c")).unwrap();
        let st = AmbientToolState {
            transcripts,
            candidates,
            voices_dir: tmp.path().join("voices"),
        };
        (tmp, Arc::new(Mutex::new(st)))
    }

    #[tokio::test]
    async fn transcript_read_returns_records_in_the_requested_window() {
        let (_tmp, st) = state();
        {
            let s = st.lock().unwrap();
            for (h, speaker, text) in [(9, "me", "morning"), (14, "tanaka-san", "afternoon")] {
                s.transcripts
                    .append(&crate::ambient::transcript::TranscriptRecord {
                        segment: format!("seg-{h}"),
                        device: "pendant".into(),
                        started_at: Utc.with_ymd_and_hms(2026, 8, 26, h, 0, 0).unwrap(),
                        speech_ms: 3000,
                        speaker: Some(speaker.into()),
                        speaker_score: Some(0.9),
                        text: text.into(),
                        audio: "a".repeat(64),
                    })
                    .unwrap();
            }
        }
        let tool = TranscriptReadTool::new(Arc::clone(&st));
        let out = tool
            .execute(&json!({
                "from": "2026-08-26T08:00:00Z",
                "to": "2026-08-26T12:00:00Z"
            }))
            .await
            .unwrap();
        assert!(out.contains("morning"));
        assert!(!out.contains("afternoon"), "outside the window");
    }

    #[tokio::test]
    async fn transcript_read_can_filter_by_speaker() {
        let (_tmp, st) = state();
        {
            let s = st.lock().unwrap();
            for (h, speaker, text) in [(9, "me", "mine"), (10, "tanaka-san", "theirs")] {
                s.transcripts
                    .append(&crate::ambient::transcript::TranscriptRecord {
                        segment: format!("seg-{h}"),
                        device: "pendant".into(),
                        started_at: Utc.with_ymd_and_hms(2026, 8, 26, h, 0, 0).unwrap(),
                        speech_ms: 3000,
                        speaker: Some(speaker.into()),
                        speaker_score: Some(0.9),
                        text: text.into(),
                        audio: "a".repeat(64),
                    })
                    .unwrap();
            }
        }
        let tool = TranscriptReadTool::new(Arc::clone(&st));
        let out = tool
            .execute(&json!({
                "from": "2026-08-26T00:00:00Z",
                "to": "2026-08-26T23:00:00Z",
                "speaker": "tanaka-san"
            }))
            .await
            .unwrap();
        assert!(out.contains("theirs"));
        assert!(!out.contains("mine"));
    }

    #[tokio::test]
    async fn speaker_candidates_lists_what_is_awaiting_a_name() {
        let (_tmp, st) = state();
        let id = {
            let mut s = st.lock().unwrap();
            s.candidates
                .enrol(
                    vec![1.0, 0.0],
                    &vec![0i16; 16_000],
                    chrono::NaiveDate::from_ymd_opt(2026, 8, 26).unwrap(),
                    30_000,
                    "something they said",
                )
                .unwrap()
        };
        let tool = SpeakerCandidatesTool::new(Arc::clone(&st));
        let out = tool.execute(&json!({})).await.unwrap();
        assert!(out.contains(&id));
        assert!(out.contains("something they said"), "samples aid recognition");
        assert!(out.contains("30"), "cumulative seconds shown");
    }

    #[tokio::test]
    async fn speaker_promote_names_a_candidate_in_one_call() {
        let (_tmp, st) = state();
        let id = {
            let mut s = st.lock().unwrap();
            s.candidates
                .enrol(
                    vec![1.0, 0.0],
                    &vec![0i16; 16_000],
                    chrono::NaiveDate::from_ymd_opt(2026, 8, 26).unwrap(),
                    30_000,
                    "hi",
                )
                .unwrap()
        };
        let tool = SpeakerPromoteTool::new(Arc::clone(&st));
        let out = tool
            .execute(&json!({ "id": id, "name": "tanaka-san" }))
            .await
            .unwrap();
        assert!(out.contains("tanaka-san"));

        let s = st.lock().unwrap();
        assert!(s.voices_dir.join("tanaka-san").join("clip.wav").exists());
        assert!(s.candidates.list().is_empty());
    }

    #[tokio::test]
    async fn speaker_promote_rejects_an_unknown_id() {
        let (_tmp, st) = state();
        let tool = SpeakerPromoteTool::new(Arc::clone(&st));
        assert!(tool.execute(&json!({ "id": "nope" })).await.is_err());
    }
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test --no-default-features --features "redb-store,lancedb-store,fastembed-embed" tools::ambient_tools`
Expected: FAIL — the tools do not exist.

- [ ] **Step 3: Implement the tools**

```rust
//! Tools that let the agent read ambient transcripts and name speakers.
//!
//! This is the S1/S2 boundary: the daily summarisation in S2 is an LLM turn
//! that calls `transcript_read` and writes to the journal over MCP.
//!
//! `speaker_promote` takes an optional name so registering a speaker is a
//! sentence in chat — "that was Tanaka-san" — rather than a file operation.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde_json::{Value, json};

use crate::ambient::speaker::candidates::CandidateStore;
use crate::ambient::transcript::TranscriptStore;
use crate::provider::ToolSpec;
use crate::tools::Tool;

pub struct AmbientToolState {
    pub transcripts: TranscriptStore,
    pub candidates: CandidateStore,
    pub voices_dir: PathBuf,
}

fn parse_time(v: &Value, field: &str) -> Result<DateTime<Utc>> {
    let raw = v[field]
        .as_str()
        .with_context(|| format!("missing '{field}' (RFC 3339)"))?;
    Ok(DateTime::parse_from_rfc3339(raw)
        .with_context(|| format!("'{field}' must be RFC 3339, got {raw:?}"))?
        .with_timezone(&Utc))
}

// -- transcript_read ---------------------------------------------------------

pub struct TranscriptReadTool {
    state: Arc<Mutex<AmbientToolState>>,
    spec: ToolSpec,
}

impl TranscriptReadTool {
    pub fn new(state: Arc<Mutex<AmbientToolState>>) -> Self {
        Self {
            state,
            spec: ToolSpec {
                name: "transcript_read".into(),
                description: "Read ambient (always-on microphone) transcripts for a time \
                              window, optionally filtered to one speaker. Speakers are \
                              identified by id; ids that look like random words are \
                              auto-enrolled voices nobody has named yet."
                    .into(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "from": { "type": "string", "description": "Start of the window, RFC 3339." },
                        "to": { "type": "string", "description": "End of the window, RFC 3339." },
                        "speaker": { "type": "string", "description": "Optional speaker id to filter to." }
                    },
                    "required": ["from", "to"]
                }),
            },
        }
    }
}

#[async_trait]
impl Tool for TranscriptReadTool {
    fn spec(&self) -> &ToolSpec {
        &self.spec
    }

    async fn execute(&self, input: &Value) -> Result<String> {
        let from = parse_time(input, "from")?;
        let to = parse_time(input, "to")?;
        let speaker = input["speaker"].as_str();
        let records = {
            let st = self.state.lock().expect("ambient tool state poisoned");
            st.transcripts.read(from, to, speaker)?
        };
        if records.is_empty() {
            return Ok("No ambient transcript in that window.".into());
        }
        let mut out = String::new();
        for r in &records {
            out.push_str(&format!(
                "[{}] {}: {}\n",
                r.started_at.to_rfc3339(),
                r.speaker.as_deref().unwrap_or("unattributed"),
                r.text
            ));
        }
        Ok(out)
    }
}

// -- speaker_candidates ------------------------------------------------------

pub struct SpeakerCandidatesTool {
    state: Arc<Mutex<AmbientToolState>>,
    spec: ToolSpec,
}

impl SpeakerCandidatesTool {
    pub fn new(state: Arc<Mutex<AmbientToolState>>) -> Self {
        Self {
            state,
            spec: ToolSpec {
                name: "speaker_candidates".into(),
                description: "List auto-enrolled voices that have not been named yet, with \
                              how much they have spoken and what they said. Use with \
                              speaker_promote to give one a name."
                    .into(),
                input_schema: json!({ "type": "object", "properties": {} }),
            },
        }
    }
}

#[async_trait]
impl Tool for SpeakerCandidatesTool {
    fn spec(&self) -> &ToolSpec {
        &self.spec
    }

    async fn execute(&self, _input: &Value) -> Result<String> {
        let st = self.state.lock().expect("ambient tool state poisoned");
        let candidates = st.candidates.list();
        if candidates.is_empty() {
            return Ok("No unnamed speakers.".into());
        }
        let mut out = String::new();
        for c in candidates {
            out.push_str(&format!(
                "{} — {}s across {} day(s), first heard {}\n",
                c.id,
                c.stats.speech_seconds,
                c.stats.days_seen.len(),
                c.stats.first_seen.to_rfc3339()
            ));
            for s in &c.stats.samples {
                out.push_str(&format!("    \"{s}\"\n"));
            }
        }
        Ok(out)
    }
}

// -- speaker_promote ---------------------------------------------------------

pub struct SpeakerPromoteTool {
    state: Arc<Mutex<AmbientToolState>>,
    spec: ToolSpec,
}

impl SpeakerPromoteTool {
    pub fn new(state: Arc<Mutex<AmbientToolState>>) -> Self {
        Self {
            state,
            spec: ToolSpec {
                name: "speaker_promote".into(),
                description: "Register an auto-enrolled voice in the workspace, optionally \
                              naming it at the same time. Past transcripts pick the name up \
                              retroactively, because they store the speaker id."
                    .into(),
                input_schema: json!({
                    "type": "object",
                    "properties": {
                        "id": { "type": "string", "description": "Candidate id from speaker_candidates." },
                        "name": { "type": "string", "description": "Optional display name, e.g. a person's name." }
                    },
                    "required": ["id"]
                }),
            },
        }
    }
}

#[async_trait]
impl Tool for SpeakerPromoteTool {
    fn spec(&self) -> &ToolSpec {
        &self.spec
    }

    async fn execute(&self, input: &Value) -> Result<String> {
        let id = input["id"].as_str().context("missing 'id'")?;
        let name = input["name"].as_str();
        let mut st = self.state.lock().expect("ambient tool state poisoned");
        let voices_dir = st.voices_dir.clone();
        let dir = st.candidates.promote(id, name, &voices_dir)?;
        Ok(format!(
            "Registered {id} as voices/{dir}. Rename that directory any time; \
             transcripts follow automatically."
        ))
    }
}

pub fn ambient_tools(state: Arc<Mutex<AmbientToolState>>) -> Vec<Box<dyn Tool>> {
    vec![
        Box::new(TranscriptReadTool::new(Arc::clone(&state))),
        Box::new(SpeakerCandidatesTool::new(Arc::clone(&state))),
        Box::new(SpeakerPromoteTool::new(state)),
    ]
}
```

Add `pub mod ambient_tools;` to `src/tools/mod.rs`.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cargo test --no-default-features --features "redb-store,lancedb-store,fastembed-embed" tools::ambient_tools`
Expected: PASS, 5 tests.

- [ ] **Step 5: Commit**

```bash
git add src/tools/
git commit -m "feat(ambient): transcript_read, speaker_candidates and speaker_promote tools"
```

---

### Task 12: Startup wiring, retention sweeper, and documentation

**Files:**
- Create: `src/ambient/startup.rs`
- Modify: `src/ambient/mod.rs`, `src/main.rs`, `src/serve/mod.rs`
- Modify: `config.example.toml`
- Modify: `README.md`
- Test: `crates/../tests` not needed; inline test in `src/ambient/startup.rs`

**Interfaces:**
- Consumes: everything from Tasks 1–11.
- Produces: `pub async fn build(config: &Config, workspace_root: &Path, stt: Arc<dyn SttProvider>) -> Result<Option<(Arc<AmbientState>, Worker, Arc<AudioCache>)>>` and `pub fn spawn_sweeper(cache: Arc<AudioCache>, retention_days: u32)`.

- [ ] **Step 1: Write the failing startup tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn build_returns_none_when_ambient_is_disabled() {
        let mut config = crate::config::Config::minimal_for_test();
        config.ambient.enabled = false;
        let out = build(&config, std::path::Path::new("/tmp"), mock_stt()).await.unwrap();
        assert!(out.is_none(), "disabled means no endpoint, no worker");
    }

    #[tokio::test]
    async fn build_fails_when_enabled_without_a_usable_key_file() {
        let tmp = tempfile::tempdir().unwrap();
        let mut config = crate::config::Config::minimal_for_test();
        config.ambient.enabled = true;
        config.keys.file = Some(tmp.path().join("absent.toml"));
        let err = build(&config, tmp.path(), mock_stt()).await.unwrap_err();
        assert!(
            err.to_string().contains("key"),
            "a misconfiguration must fail loudly at startup, not reject every device silently: {err}"
        );
    }

    #[tokio::test]
    async fn build_fails_when_the_named_stt_provider_is_absent() {
        // Guards against a config typo silently producing empty transcripts.
        let tmp = tempfile::tempdir().unwrap();
        let mut config = crate::config::Config::minimal_for_test();
        config.ambient.enabled = true;
        config.ambient.stt_provider = "does-not-exist".into();
        config.keys.file = Some(write_usable_key_file(tmp.path()));
        let err = build(&config, tmp.path(), mock_stt()).await.unwrap_err();
        assert!(err.to_string().contains("does-not-exist"), "{err}");
    }
}
```

**Implementer note:** `Config::minimal_for_test()` and `write_usable_key_file` are test helpers you add here. `minimal_for_test` builds a `Config` with the required `anthropic` block filled in and everything else defaulted; if `Config` already has such a helper in `src/config.rs`'s test module, promote it to `#[cfg(test)] pub(crate)` rather than writing a second one.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test --no-default-features --features "redb-store,lancedb-store,fastembed-embed" ambient::startup`
Expected: FAIL — `build` does not exist.

- [ ] **Step 3: Implement startup**

```rust
//! Assembling the ambient subsystem at process start.
//!
//! Every failure here is fatal by design. Ambient ingest that starts but
//! cannot authenticate, transcribe or store looks exactly like a broken
//! device from the outside, and the device has no way to tell you.

use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result, bail};
use tokio::sync::mpsc;
use tracing::info;

use super::audio::{PassthroughGate, SpeakerEmbedder, SpeechGate};
use super::auth::DeviceRegistry;
use super::cache::AudioCache;
use super::ingest::{AmbientState, Segment};
use super::speaker::candidates::CandidateStore;
use super::speaker::registry::SpeakerRegistry;
use super::transcript::TranscriptStore;
use super::worker::Worker;
use crate::config::Config;
use crate::voice::SttProvider;

/// Build the ambient subsystem, or `None` when it is switched off.
pub async fn build(
    config: &Config,
    workspace_root: &Path,
    stt: Arc<dyn SttProvider>,
) -> Result<Option<(Arc<AmbientState>, Worker, Arc<AudioCache>, mpsc::Receiver<Segment>)>> {
    if !config.ambient.enabled {
        return Ok(None);
    }

    let root = match &config.ambient.cache_dir {
        Some(dir) => dir.clone(),
        None => dirs::cache_dir()
            .context("no platform cache dir; set [ambient].cache_dir")?
            .join("sapphire-agent")
            .join("ambient"),
    };

    let key_file = config
        .keys
        .file
        .clone()
        .or_else(DeviceRegistry::default_key_file)
        .context("no key file; set [keys].file")?;
    let devices = DeviceRegistry::open(&key_file, &config.devices)?;
    info!(
        "ambient: {} device(s) bound against {}",
        config.devices.len(),
        key_file.display()
    );

    if config.ambient.stt_provider.is_empty() {
        bail!("[ambient].stt_provider must name an [stt_provider.*] block");
    }
    if stt.name() != config.ambient.stt_provider {
        bail!(
            "[ambient].stt_provider = {:?} does not match any configured provider",
            config.ambient.stt_provider
        );
    }

    let cache = AudioCache::open(root.join("audio"))?;
    let transcripts =
        TranscriptStore::open(root.join("transcripts"), config.day_boundary_hour)?;
    let voices_dir = workspace_root.join("voices");

    // The gate and embedder are the two pieces that need models. Without
    // the sherpa feature the gate keeps everything (over-keeping costs STT
    // time, never data) and speaker attribution is switched off rather
    // than guessed at.
    let gate: Box<dyn SpeechGate> = Box::new(PassthroughGate::new());
    let embedder: Box<dyn SpeakerEmbedder> = build_embedder()?;
    let model_id = "default".to_string();

    let mut registry = SpeakerRegistry::open(
        voices_dir.clone(),
        root.join("speakers").join("registered"),
        model_id,
        config.ambient.match_threshold,
    )?;
    registry.load_reference_audio(embedder.as_ref())?;
    let candidates = CandidateStore::open(root.join("speakers").join("candidates"))?;
    for c in candidates.list() {
        registry.add_runtime(c.id.clone(), c.centroid.clone());
    }

    let (tx, rx) = mpsc::channel(config.ambient.max_queue);
    let state = Arc::new(AmbientState::new(config.ambient.clone(), devices, tx));
    let worker = Worker {
        gate,
        embedder,
        stt,
        registry,
        candidates,
        cache: Arc::clone(&cache),
        transcripts,
        voices_dir,
        min_embed_ms: config.ambient.min_embed_ms,
        promote_after_seconds: config.ambient.promote_after_seconds,
        promote_after_days: config.ambient.promote_after_days,
        day_boundary_hour: config.day_boundary_hour,
        language: None,
    };
    Ok(Some((state, worker, cache, rx)))
}

#[cfg(feature = "voice-sherpa")]
fn build_embedder() -> Result<Box<dyn SpeakerEmbedder>> {
    // Model path resolution follows the same download-and-cache path the
    // sherpa STT provider already uses; see voice/providers/sherpa_download.rs.
    bail!(
        "speaker embedding model wiring: resolve the model path the way \
         SherpaOnnxStt does, then construct SherpaEmbedder"
    )
}

#[cfg(not(feature = "voice-sherpa"))]
fn build_embedder() -> Result<Box<dyn SpeakerEmbedder>> {
    bail!(
        "speaker attribution needs the `voice-sherpa` feature; rebuild with it \
         or set [ambient].enabled = false"
    )
}

/// Sweep expired audio once at start and then daily.
pub fn spawn_sweeper(cache: Arc<AudioCache>, retention_days: u32) {
    let max_age = std::time::Duration::from_secs(retention_days as u64 * 86_400);
    tokio::spawn(async move {
        loop {
            match cache.sweep(max_age) {
                Ok(n) if n > 0 => info!("ambient: swept {n} expired audio blob(s)"),
                Ok(_) => {}
                Err(e) => tracing::warn!("ambient: sweep failed: {e}"),
            }
            tokio::time::sleep(std::time::Duration::from_secs(86_400)).await;
        }
    });
}
```

**Implementer note on `build_embedder`:** the `voice-sherpa` arm is left as an explicit `bail!` with instructions rather than a guess, because the model download path is the one piece this plan cannot specify without reading `sherpa_download.rs` in detail. Resolving it is part of this step: mirror how `SherpaOnnxStt::new` obtains its model, add an `embedding_model` field to `AmbientConfig` if a path is needed, and construct `SherpaEmbedder`. Do not ship the `bail!`.

- [ ] **Step 4: Wire it into `main.rs` and the server**

Where `ServeState` is constructed, call `ambient::startup::build`, put the `Arc<AmbientState>` on `ServeState.ambient`, and spawn both background tasks:

```rust
    let ambient = ambient::startup::build(&config, workspace_root, ambient_stt).await?;
    let ambient_state = match ambient {
        Some((state, worker, cache, rx)) => {
            ambient::startup::spawn_sweeper(cache, config.ambient.audio_retention_days);
            tokio::spawn(ambient::worker::run(worker, rx));
            Some(state)
        }
        None => None,
    };
```

Register the three tools alongside the other built-ins, guarded on `ambient_state.is_some()`.

- [ ] **Step 5: Document the configuration**

Append to `config.example.toml`:

```toml
# ---------------------------------------------------------------------------
# Ambient audio ingest — always-on microphone capture.
#
# Accepts audio segments from a wearable capture device over
# `POST /audio/ingest`, transcribes them, attributes each to a speaker, and
# stores transcripts outside the workspace. It never starts a conversation:
# ingest records, it does not answer.
# ---------------------------------------------------------------------------
# [ambient]
# enabled = true
# stt_provider = "sherpa_ja"        # name of an [stt_provider.*] block
# audio_retention_days = 7          # transcripts are kept indefinitely
# min_embed_ms = 1500               # shorter segments get no speaker
# match_threshold = 0.55
# promote_after_seconds = 60        # both thresholds must be cleared before
# promote_after_days = 2            # an auto-enrolled voice reaches the workspace
# max_queue = 1000

# Where sapphire-framework's API keys live. Host-local: this names the only
# place tokens are stored, so it must never come from the workspace layer.
# [keys]
# file = "~/.config/sapphire-agent/keys.toml"

# One capture device. Note there is no token here — `key_id` is the `id` of a
# [[key]] in the key file above. Mint the key first, then copy its id here.
# [device.pendant]
# key_id = "6c8f4a2e-1d33-4b90-9a71-0e5b2f8c4d17"
# label = "the one on the lanyard"
# room_profile = "default"
```

Add a README section under the existing feature list:

```markdown
### Ambient capture

`POST /audio/ingest` accepts audio from an always-on wearable microphone,
transcribes it, and attributes each segment to a speaker — without ever
answering. Reference voices live in the workspace as `voices/<name>/*.wav`;
voices it has not heard before are auto-enrolled and surface through the
`speaker_candidates` tool, so naming one is a sentence in chat rather than a
file operation. Audio is cached outside the workspace and swept after a week;
transcripts are kept.

Devices authenticate with a bearer token held in `sapphire-framework`'s key
file — the agent's config names only the key's `id`, never the token. See
`docs/superpowers/specs/2026-08-26-ambient-audio-ingest-design.md`.
```

- [ ] **Step 6: Run the whole suite**

Run: `cargo test --no-default-features --features "redb-store,lancedb-store,fastembed-embed"`
Expected: PASS, with no regressions in the existing tests.

Then the full build, which is the one that compiles the sherpa paths:

Run: `cargo test`
Expected: PASS.

- [ ] **Step 7: Run clippy and fmt**

Run: `cargo fmt --check && cargo clippy --all-targets -- -D warnings`
Expected: clean. Fix anything reported before committing.

- [ ] **Step 8: Commit**

```bash
git add src/ config.example.toml README.md
git commit -m "feat(ambient): wire ingest into startup, sweep expired audio, document the config"
```

---

## Plan self-review

**Spec coverage.** Walked each spec section against the tasks:

| Spec section | Task |
|---|---|
| Ingest endpoint, wire format, `Content-Type`, `rate`, `segment`, `started_at`, `live` | 5 |
| Device identity from the API key, `key_id`, resolution order, 401 cases | 1, 2 |
| Reaching `KeyStore` (`remote-server` decision) | 1 |
| `POST /audio/hello`, optional | 5 |
| Segment router | 6 |
| Processing pipeline steps 1–5 | 7, 8, 9, 10 |
| Cache layout, transcript record shape, `speech_ms` / `speaker_score` / day boundary | 3, 4 |
| Retention (audio 7 days, transcripts kept) | 3, 12 |
| Speaker registry, embedding cache keyed by sha × model, rename transparency | 8 |
| Candidates, two-tier promotion, both thresholds | 9 |
| Agent-facing tools | 11 |
| Configuration | 1, 12 |
| Error handling: 429, per-segment failure, unreadable reference audio, 401, startup failure | 5, 8, 10, 12 |
| Testing list | distributed; every listed case appears |

Two spec test cases needed a home and got one: "a `live=0` segment never reaches the conversation branch" is Task 6, and "rename transparency" is covered structurally — Task 8 proves the id is the directory name and Task 4 proves transcripts store ids, which together are the property. **A dedicated end-to-end rename test is not in any task**; it would need the full worker plus a workspace, so the honest statement is that the property is covered by construction rather than by one test. If a reviewer wants it explicit, add it to Task 10.

**Placeholder scan.** Two `bail!`-with-instructions remain, both flagged in bold implementer notes rather than hidden: `build_embedder`'s sherpa arm (Task 12 Step 3) and `ServeState::for_ambient_test` (Task 5 Step 3). Both are places where the plan cannot honestly specify code without reading a file it has not read — the sherpa model download path, and the current shape of a large struct. Each says exactly what to do and forbids shipping the stub. Every other step carries real code.

**Type consistency.** Checked the names that cross task boundaries: `Segment` (5 → 6, 10), `GatedSpeech.speech_ms` (7 → 10 → 4), `SpeakerMatch { id, score }` (8 → 10), `encode_embedding`/`decode_embedding` (8 → 9), `CandidateStore::{enrol, observe, is_promotable, promote, list, get}` (9 → 10, 11), `TranscriptRecord` field names (4 → 10, 11), `AmbientConfig` field names (1 → 5, 12). `speech_ms` is `u32` milliseconds everywhere and `CandidateStats::speech_seconds` is `u32` seconds — the conversion happens in exactly one place (`CandidateStore::enrol`/`observe`), which is deliberate and noted.

---

## Execution handoff

Plan complete and saved to `docs/superpowers/plans/2026-08-26-ambient-audio-ingest-plan.md`. Two execution options:

**1. Subagent-Driven (recommended)** — a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — execute tasks in this session using executing-plans, batch execution with checkpoints.

Which approach?
