//! Wires the ambient subsystem into the running process: opens every
//! store, resolves the STT provider and the VAD/embedding models, and
//! assembles the worker, the ingest routes, and the agent tools.
//!
//! Split into [`build`] (pure assembly, no tokio tasks) and [`spawn`]
//! (starts the worker loop and the daily sweep) so `build` stays testable
//! without a runtime spawning tasks that would outlive the test. Every
//! failure in `build` is fatal by design: an ambient subsystem that starts
//! but cannot authenticate, transcribe, or store looks exactly like a
//! broken device from the outside, and the device has no way to tell you.

use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::{Context, Result};
use tokio::sync::mpsc;
use tracing::{info, warn};

use crate::config::Config;
use crate::tools::Tool;
use crate::tools::ambient_tools::{AmbientToolState, ambient_tools};
use crate::voice::VoiceProviders;

use super::auth::DeviceRegistry;
use super::cache::AudioCache;
use super::ingest::{self, AmbientState, Segment};
use super::models;
use super::speaker::candidates::CandidateStore;
use super::speaker::registry::SpeakerRegistry;
use super::transcript::TranscriptStore;
use super::worker::Worker;

/// Everything the ambient subsystem needs to run, before it is spawned.
pub struct AmbientRuntime {
    pub routes: axum::Router,
    pub tools: Vec<Box<dyn Tool>>,
    worker: Worker,
    rx: mpsc::Receiver<Segment>,
    cache: Arc<AudioCache>,
    retention_days: u32,
}

/// Build everything, or `None` when `[ambient].enabled` is false.
///
/// Assembles, in order, failing loudly at each step:
/// 1. the cache root (`[ambient].cache_dir`, else a platform default);
/// 2. the device key store (`[keys].file`, else a platform default);
/// 3. the configured STT provider, by name;
/// 4. the VAD/embedding models;
/// 5. the audio cache and the transcript store;
/// 6. the speaker registry, loaded from workspace reference audio;
/// 7. the candidate store, seeding the registry with every candidate
///    already on disk so a known voice matches on its next segment;
/// 8. the admission channel, the ingest state, and the worker;
/// 9. the agent tools, sharing the *same* `Arc<Mutex<CandidateStore>>` as
///    the worker — two stores over one directory would diverge in memory,
///    and promoting through a tool would leave the worker still matching
///    and re-promoting that voice.
pub fn build(
    config: &Config,
    workspace_dir: &Path,
    voice: Option<&VoiceProviders>,
) -> Result<Option<AmbientRuntime>> {
    if !config.ambient.enabled {
        return Ok(None);
    }

    // 1. Cache root. `AudioCache::default_dir()` names the audio leaf
    // specifically (it is also usable standalone); strip it back off to
    // get the root that the other stores hang subdirectories off of.
    let cache_root = config
        .ambient
        .cache_dir
        .clone()
        .or_else(|| AudioCache::default_dir().and_then(|p| p.parent().map(Path::to_path_buf)))
        .ok_or_else(|| {
            anyhow::anyhow!(
                "[ambient].cache_dir is unset and no platform cache directory is resolvable; \
                 set [ambient].cache_dir explicitly"
            )
        })?;

    // 2. Device key store. A missing or unusable key file is fatal here
    // (not just an empty store) — see `DeviceRegistry::open`'s own doc.
    let key_file = config
        .keys
        .file
        .clone()
        .or_else(DeviceRegistry::default_key_file)
        .ok_or_else(|| {
            anyhow::anyhow!(
                "[keys].file is unset and no platform config directory is resolvable; \
                 set [keys].file explicitly"
            )
        })?;
    let devices = DeviceRegistry::open(&key_file, &config.devices)?;

    // 3. STT provider, resolved by name. A typo here must not degrade
    // into silent empty transcripts, so a missing provider is an error
    // that names what was configured.
    let stt = voice
        .and_then(|v| v.stt(&config.ambient.stt_provider))
        .ok_or_else(|| {
            anyhow::anyhow!(
                "[ambient].stt_provider {:?} does not name a configured [stt_provider.*] block",
                config.ambient.stt_provider
            )
        })?;

    // 4. VAD re-gate + speaker embedding models.
    let resolved = models::resolve(&config.ambient)?;

    // 5. Audio cache + transcript store. The transcript store is opened
    // twice (once here for the worker, once below for the tools) rather
    // than shared: unlike `CandidateStore`, it holds no in-memory state
    // that two instances could diverge on, it is pure file I/O.
    let cache =
        AudioCache::open(cache_root.join("audio")).context("opening ambient audio cache")?;
    let transcripts_dir = cache_root.join("transcripts");
    let worker_transcripts =
        TranscriptStore::open(transcripts_dir.clone(), config.day_boundary_hour)
            .context("opening ambient transcript store")?;
    let tool_transcripts = TranscriptStore::open(transcripts_dir, config.day_boundary_hour)
        .context("opening ambient transcript store (tools)")?;

    // 6. Speaker registry, loaded from workspace reference audio.
    let voices_dir = workspace_dir.join("voices");
    let mut registry = SpeakerRegistry::open(
        voices_dir.clone(),
        cache_root.join("speakers").join("registered"),
        resolved.model_id.clone(),
        config.ambient.match_threshold,
    )
    .context("opening speaker registry")?;
    registry
        .load_reference_audio(resolved.embedder.as_ref())
        .context("loading workspace reference audio")?;

    // 7. Candidate store, seeding the registry so a candidate that was
    // already enrolled before a restart still matches on its next
    // segment instead of being enrolled a second time under a new id.
    let candidate_store = CandidateStore::open(cache_root.join("speakers").join("candidates"))
        .context("opening candidate store")?;
    for c in candidate_store.list() {
        registry.add_runtime(c.id.clone(), c.centroid.clone());
    }
    let candidates = Arc::new(Mutex::new(candidate_store));

    // 8. Admission channel, ingest state, worker.
    let (tx, rx) = mpsc::channel(config.ambient.max_queue);
    let ambient_state = Arc::new(AmbientState::new(config.ambient.clone(), devices, tx));
    let routes = ingest::routes(ambient_state);

    let worker = Worker {
        gate: resolved.gate,
        embedder: resolved.embedder,
        stt,
        registry,
        candidates: Arc::clone(&candidates),
        cache: Arc::clone(&cache),
        transcripts: worker_transcripts,
        voices_dir: voices_dir.clone(),
        min_embed_ms: config.ambient.min_embed_ms,
        promote_after_seconds: config.ambient.promote_after_seconds,
        promote_after_days: config.ambient.promote_after_days,
        day_boundary_hour: config.day_boundary_hour,
        language: None,
    };

    // 9. Tools, sharing the worker's own `Arc<Mutex<CandidateStore>>`.
    let tool_state = Arc::new(Mutex::new(AmbientToolState {
        transcripts: tool_transcripts,
        candidates: Arc::clone(&candidates),
        voices_dir,
    }));
    let tools = ambient_tools(tool_state);

    Ok(Some(AmbientRuntime {
        routes,
        tools,
        worker,
        rx,
        cache,
        retention_days: config.ambient.audio_retention_days,
    }))
}

/// Consume the runtime: spawn the worker and the daily sweep, return the
/// routes and tools for the caller to mount / register.
pub fn spawn(runtime: AmbientRuntime) -> (axum::Router, Vec<Box<dyn Tool>>) {
    let AmbientRuntime {
        routes,
        tools,
        worker,
        rx,
        cache,
        retention_days,
    } = runtime;

    tokio::spawn(super::worker::run(worker, rx));

    let max_age = Duration::from_secs(retention_days as u64 * 86_400);
    tokio::spawn(async move {
        loop {
            match cache.sweep(max_age) {
                Ok(n) if n > 0 => info!("ambient: swept {n} expired audio blob(s)"),
                Ok(_) => {}
                Err(e) => warn!("ambient: sweep failed: {e}"),
            }
            tokio::time::sleep(Duration::from_secs(86_400)).await;
        }
    });

    (routes, tools)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_returns_none_when_disabled() {
        let mut cfg = Config::for_test();
        cfg.ambient.enabled = false;
        let tmp = tempfile::tempdir().unwrap();
        let result = build(&cfg, tmp.path(), None).unwrap();
        assert!(result.is_none(), "ambient is opt-in");
    }

    #[test]
    fn build_errors_on_a_missing_key_file_and_names_it() {
        let mut cfg = Config::for_test();
        cfg.ambient.enabled = true;
        let tmp = tempfile::tempdir().unwrap();
        let missing = tmp.path().join("keys.toml");
        cfg.keys.file = Some(missing.clone());

        let err = build(&cfg, tmp.path(), None)
            .err()
            .expect("expected an error");
        let msg = format!("{err:#}");
        assert!(
            msg.contains(&missing.display().to_string()),
            "error should name the key file, got: {msg}"
        );
    }

    #[test]
    fn build_errors_when_the_stt_provider_name_is_unconfigured() {
        let mut cfg = Config::for_test();
        cfg.ambient.enabled = true;
        cfg.ambient.stt_provider = "nonexistent".into();
        let tmp = tempfile::tempdir().unwrap();
        let key_path = tmp.path().join("keys.toml");
        std::fs::write(
            &key_path,
            "[[key]]\ntoken = \"sa-dev-good\"\nlabel = \"pendant-key\"\n",
        )
        .unwrap();
        cfg.keys.file = Some(key_path);

        // No `voice` registry at all — indistinguishable, from `build`'s
        // point of view, from one that exists but never configured a
        // `[stt_provider.nonexistent]` block.
        let err = build(&cfg, tmp.path(), None)
            .err()
            .expect("expected an error");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("nonexistent"),
            "error should name the missing provider, got: {msg}"
        );
    }

    #[test]
    fn build_reaches_model_resolution_once_stt_resolves() {
        // Confirms step ordering: with a valid key file and a real STT
        // entry, `build` gets past auth and STT and fails at model
        // resolution instead — this crate is built without
        // `voice-sherpa` in this test run, so that failure names the
        // feature rather than a model path.
        let mut cfg = Config::for_test();
        cfg.ambient.enabled = true;
        cfg.ambient.stt_provider = "mock".into();
        cfg.stt_providers.insert(
            "mock".into(),
            crate::config::SttProviderConfig::Mock {
                transcript: "hi".into(),
            },
        );
        let tmp = tempfile::tempdir().unwrap();
        let key_path = tmp.path().join("keys.toml");
        std::fs::write(
            &key_path,
            "[[key]]\ntoken = \"sa-dev-good\"\nlabel = \"pendant-key\"\n",
        )
        .unwrap();
        cfg.keys.file = Some(key_path);

        let voice = crate::voice::VoiceProviders::from_config(&cfg).unwrap();
        let err = build(&cfg, tmp.path(), Some(&voice))
            .err()
            .expect("expected an error");
        let msg = format!("{err:#}");
        assert!(
            msg.contains("voice-sherpa"),
            "should fail at model resolution, naming the feature, got: {msg}"
        );
    }
}
