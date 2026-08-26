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
//!
//! This module owns its own axum state ([`AmbientState`]) rather than
//! sharing `crate::serve::ServeState`. `ServeState` exists to serve LLM
//! turns; nothing here may start one, so it needs none of the machinery
//! (provider registry, workspace, tool set, session stores) that
//! `ServeState` carries. [`routes`] therefore applies its own
//! `.with_state()` and hands back a plain `Router` for a later task to
//! merge into the main one.

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

/// Everything the ambient ingest routes need. Deliberately separate from
/// `crate::serve::ServeState` — see the module doc.
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
struct IngestParams {
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

/// Build the ambient ingest routes, bound to their own state.
pub fn routes(state: Arc<AmbientState>) -> Router {
    Router::new()
        .route("/audio/ingest", post(handle_ingest))
        .route("/audio/hello", post(handle_hello))
        .with_state(state)
}

/// Resolve the bearer to a device name, or the error response to return.
///
/// `[ambient].enabled = false` and a missing/invalid bearer both collapse
/// to a response here rather than the route being conditionally mounted,
/// because a later task decides whether to mount at all. All three auth
/// failure modes (unknown token, expired token, token bound to no device)
/// collapse to the same 401 — the distinction is logged, not returned.
fn authenticate(state: &AmbientState, headers: &HeaderMap) -> Result<String, StatusCode> {
    if !state.config.enabled {
        return Err(StatusCode::NOT_FOUND);
    }
    let token = crate::serve::extract_bearer(headers).ok_or(StatusCode::UNAUTHORIZED)?;
    match state.devices.resolve(&token) {
        Some(name) => Ok(name.to_string()),
        None => {
            debug!("ambient: rejected bearer (unknown, expired, or bound to no device)");
            Err(StatusCode::UNAUTHORIZED)
        }
    }
}

async fn handle_ingest(
    State(state): State<Arc<AmbientState>>,
    Query(params): Query<IngestParams>,
    headers: HeaderMap,
    body: axum::body::Bytes,
) -> impl IntoResponse {
    let device = match authenticate(&state, &headers) {
        Ok(d) => d,
        Err(status) => return status.into_response(),
    };

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
    if !state.admit_once(&params.segment) {
        debug!("ambient: duplicate segment {} discarded", params.segment);
        return StatusCode::OK.into_response();
    }

    let pcm = match content_type.as_str() {
        "audio/wav" => match decode_wav(&body) {
            Ok(pcm) => pcm,
            Err(e) => return (StatusCode::BAD_REQUEST, e.to_string()).into_response(),
        },
        _ => match decode_l16(&body) {
            Ok(pcm) => pcm,
            Err(e) => return (StatusCode::BAD_REQUEST, e.to_string()).into_response(),
        },
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

    match state.tx.try_send(seg) {
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
    State(state): State<Arc<AmbientState>>,
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

/// Decode raw s16le. An odd-length body means a truncated transmission —
/// rejected rather than silently dropping the trailing byte, the same
/// failure class as accepting the wrong WAV bit depth in [`decode_wav`].
fn decode_l16(bytes: &[u8]) -> anyhow::Result<Vec<i16>> {
    if bytes.len() % 2 != 0 {
        anyhow::bail!(
            "audio/L16 body has odd length {} bytes; s16le must be a whole number of samples",
            bytes.len()
        );
    }
    Ok(bytes
        .chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]))
        .collect())
}

/// Decode a WAV blob, rejecting anything that isn't already exactly
/// 16 kHz mono 16-bit PCM. Unlike
/// [`crate::voice::providers::wav_stream::decode_wav`] this deliberately
/// does not scale other bit depths into range: an 8-bit capture silently
/// reinterpreted as 16-bit is corrupted audio, not merely low-quality
/// audio, and the "16 kHz mono s16le only" constraint exists precisely to
/// keep that kind of reinterpretation from happening unnoticed.
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
    if spec.sample_format != hound::SampleFormat::Int || spec.bits_per_sample != 16 {
        anyhow::bail!(
            "wav must be 16-bit PCM, got {:?} at {} bits",
            spec.sample_format,
            spec.bits_per_sample
        );
    }
    Ok(reader.samples::<i16>().collect::<Result<Vec<_>, _>>()?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::Request;
    use tower::ServiceExt;

    /// Build a router plus the receiving end of the admission queue.
    fn harness() -> (Router, mpsc::Receiver<Segment>) {
        harness_with_queue_depth(16)
    }

    fn harness_with_queue_depth(depth: usize) -> (Router, mpsc::Receiver<Segment>) {
        harness_with(depth, true)
    }

    fn harness_with(depth: usize, enabled: bool) -> (Router, mpsc::Receiver<Segment>) {
        let tmp = tempfile::tempdir().unwrap();
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
        cfg.enabled = enabled;
        let state = Arc::new(AmbientState::new(cfg, registry, tx));
        (routes(state), rx)
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
        assert_eq!(
            seg.started_at.timestamp_millis(),
            1_787_000_000_000,
            "started_at round-trips from the query param, in milliseconds"
        );
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
        // Capacity 2: verified empirically (a throwaway probe test, removed
        // after use) that `mpsc::channel(N)` accepts exactly N `try_send`s
        // before `Full` — so two requests must land before the third
        // overflows. Asserting only the last request (as the original test
        // did) would also pass an implementation that answered 429
        // unconditionally; asserting the whole sequence pins down exactly
        // when the queue starts refusing.
        let (app, _rx) = harness_with_queue_depth(2);
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
            if i < 2 {
                assert_eq!(res.status(), StatusCode::OK, "request {i} should still fit");
            } else {
                assert_eq!(
                    res.status(),
                    StatusCode::TOO_MANY_REQUESTS,
                    "request {i} should overflow the queue"
                );
            }
        }
    }

    #[tokio::test]
    async fn disabled_ambient_answers_404_rather_than_401() {
        // `[ambient].enabled = false` must be distinguishable from a bad
        // bearer: 404 says "this feature is off", 401 says "your
        // credentials are wrong" — a device probing for the feature
        // should not be told its key is bad.
        let (app, _rx) = harness_with(16, false);
        let res = app
            .oneshot(
                Request::post(ingest_uri("seg-disabled", false))
                    .header("authorization", "Bearer sa-dev-good")
                    .header("content-type", "audio/L16")
                    .body(pcm_body(1_600))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn rejects_an_odd_length_l16_body_rather_than_dropping_the_last_byte() {
        let (app, _rx) = harness();
        let res = app
            .oneshot(
                Request::post(ingest_uri("seg-odd", false))
                    .header("authorization", "Bearer sa-dev-good")
                    .header("content-type", "audio/L16")
                    .body(Body::from(vec![0u8; 1_601])) // odd: not a whole number of i16 samples
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(
            res.status(),
            StatusCode::BAD_REQUEST,
            "a truncated s16le body must be rejected, not silently short by one sample"
        );
    }

    /// Build an in-memory WAV file at the given bit depth. 16-bit samples
    /// are widened/narrowed as `hound`'s own `Sample` impls do.
    fn wav_bytes(bits_per_sample: u16, samples: &[i16]) -> Vec<u8> {
        let spec = hound::WavSpec {
            channels: 1,
            sample_rate: PIPELINE_SAMPLE_RATE,
            bits_per_sample,
            sample_format: hound::SampleFormat::Int,
        };
        let mut cursor = std::io::Cursor::new(Vec::new());
        {
            let mut writer = hound::WavWriter::new(&mut cursor, spec).unwrap();
            match bits_per_sample {
                8 => {
                    for &s in samples {
                        writer.write_sample((s >> 8) as i8).unwrap();
                    }
                }
                16 => {
                    for &s in samples {
                        writer.write_sample(s).unwrap();
                    }
                }
                other => panic!("test helper does not support {other}-bit WAVs"),
            }
            writer.finalize().unwrap();
        }
        cursor.into_inner()
    }

    #[tokio::test]
    async fn accepts_a_16_bit_mono_16k_wav_segment() {
        let (app, mut rx) = harness();
        let body = wav_bytes(16, &[0i16; 1_600]);
        let res = app
            .oneshot(
                Request::post("/audio/ingest?segment=seg-wav&started_at=1787000000000&rate=16000&live=0")
                    .header("authorization", "Bearer sa-dev-good")
                    .header("content-type", "audio/wav")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::OK);
        let seg = rx.try_recv().expect("segment enqueued");
        assert_eq!(seg.pcm.len(), 1_600);
    }

    #[tokio::test]
    async fn rejects_an_8_bit_wav_rather_than_silently_reinterpreting_it() {
        // hound reads 8-bit PCM successfully (it errors only on 24- and
        // 32-bit input), widening each unsigned byte into an i16 without
        // rescaling to the 16-bit range. Accepting that silently would
        // enqueue corrupted audio under a 16-bit label — precisely what
        // "16 kHz mono s16le only" is meant to prevent.
        let (app, _rx) = harness();
        let body = wav_bytes(8, &[0i16; 1_600]);
        let res = app
            .oneshot(
                Request::post("/audio/ingest?segment=seg-wav8&started_at=1787000000000&rate=16000&live=0")
                    .header("authorization", "Bearer sa-dev-good")
                    .header("content-type", "audio/wav")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(
            res.status(),
            StatusCode::BAD_REQUEST,
            "8-bit PCM must not be silently reinterpreted as 16-bit"
        );
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
