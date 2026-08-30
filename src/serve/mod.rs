//! HTTP API server for sapphire-agent control API (JSON-RPC 2.0 over HTTP).
//!
//! Endpoint: POST /rpc  (chat, initialize, list_sessions, get_session, voice/*)
//!           GET  /rpc  (Phase 2: server→client SSE push, currently 405)
//!           POST /a2a  (Agent2Agent Protocol; gated by [a2a].enabled)
//!           GET  /.well-known/agent-card.json
//!           POST /mcp  (MCP server; write_report / recall_memory tools)
//!           GET  /acp  (Agent Client Protocol over WebSocket; gated by [acp].enabled)
//!
//! Session management uses a `Session-Id` request/response header.

pub mod a2a;
pub mod acp;
pub mod acp_permissions;
pub mod mcp;

use crate::channel::RoomInfo;
use crate::config::Config;
use crate::context_compression::{generate_summary, maybe_compress};
use crate::provider::registry::ProviderRegistry;
use crate::provider::{ChatMessage, ContentPart, Provider, UserInputKind};
use crate::session::{ConversationKey, SessionStore};
use crate::tools::ToolSet;
use crate::voice::VoiceProviders;
use crate::workspace::Workspace;
use axum::extract::State;
use axum::http::{HeaderMap, HeaderValue, StatusCode};
use axum::response::IntoResponse;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::routing::post;
use axum::{Json, Router};
use serde::Deserialize;
use serde_json::{Value, json};
use std::collections::HashMap;
use std::convert::Infallible;
use std::sync::Arc;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tracing::{error, info, warn};

const MAX_TOOL_ROUNDS: usize = 10;

// ---------------------------------------------------------------------------
// Shared server state
// ---------------------------------------------------------------------------

/// Map of `device_id` → (`room_profile`, push channel), populated by
/// `voice/subscribe`. Heartbeat (and any future server-initiated voice
/// notifier) looks up subscribers here to deliver TTS audio.
///
/// Keyed by `device_id` alone because a single satellite only ever
/// holds one active voice session at a time — the satellite tells us
/// which room_profile it's bound to when it subscribes, and we keep
/// that around as the reverse index so heartbeat tasks don't have to
/// duplicate the value.
pub type VoiceSubscribers =
    tokio::sync::Mutex<HashMap<String, (String, mpsc::Sender<crate::voice::VoicePushItem>)>>;

pub struct ServeState {
    pub(crate) config: Config,
    pub(crate) registry: Arc<ProviderRegistry>,
    pub(crate) workspace: Arc<Workspace>,
    pub(crate) tools: Arc<ToolSet>,
    /// Cross-device session store (kind = `"rpc"` for now; #122 PR 3
    /// renames the on-disk dir to `cross-device/`). Holds the
    /// user-selectable, multi-device sessions resumed via
    /// `--resume <grain-id>`.
    pub(crate) cross_device_session_store: Arc<SessionStore>,
    /// Device-default session store (kind = `"device-default"`). Holds the
    /// per-`(device_id, room_profile)` always-on session that heartbeat
    /// pushes target and that a satellite falls into when no other session
    /// is selected. Lazy-created, daily-rotated. See #122.
    pub(crate) device_default_session_store: Arc<SessionStore>,
    /// MCP session store (kind = `"mcp"`). Holds long-lived
    /// per-project sessions written through `/mcp`'s `write_report`
    /// tool — kept physically separate from `cross_device_session_store` so the
    /// project index scan and any future MCP-specific retention only
    /// see MCP traffic.
    pub(crate) mcp_session_store: Arc<SessionStore>,
    /// Reverse index `(namespace, project) → session_id` for the MCP
    /// session store. Seeded at startup from `SessionMeta.project` and
    /// maintained on `create_mcp_session`. The mapping isn't
    /// persisted to its own file: each session file's first-line meta
    /// IS the source of truth, so a restart rebuilds the index by
    /// scanning `sessions/<ns>/mcp/*.jsonl` meta lines.
    pub(crate) mcp_project_index: tokio::sync::Mutex<HashMap<(String, String), String>>,
    /// In-memory conversation history, keyed by session_id.
    /// Lazy-loaded from JSONL on first access.
    pub(crate) sessions: tokio::sync::Mutex<HashMap<String, Vec<ChatMessage>>>,
    /// Sessions that have been issued an ID via `initialize` but have not yet
    /// received a message — file creation is deferred until the first chat so
    /// that quitting without sending anything leaves no empty file behind.
    /// Maps internal UUID → reserved public_id (grain-id).
    pub(crate) pending_sessions: tokio::sync::Mutex<HashMap<String, String>>,
    /// Per-session room_profile pin from `initialize`. Sessions absent
    /// from this map fall through to the background provider. Not
    /// persisted across restarts — clients must re-pass `room_profile`
    /// on resume.
    pub(crate) session_room_profiles: tokio::sync::Mutex<HashMap<String, String>>,
    /// Per-session room metadata supplied by the client at `initialize`
    /// (sapphire-call's `[device]` block, principally). Mirrors the
    /// channel-side `Channel::room_info()` lookup so the agent can tell
    /// the model "you are speaking through the living-room speaker; STT
    /// may have introduced typos" without baking that into AGENTS.md.
    /// Not persisted across restarts — clients must re-pass `device` on
    /// resume.
    pub(crate) session_room_metadata: tokio::sync::Mutex<HashMap<String, RoomInfo>>,
    /// Voice provider registry. `None` when no `[stt_provider.*]` /
    /// `[tts_provider.*]` blocks are configured — in that case the
    /// `voice/pipeline_run` method returns a method-not-available error.
    pub(crate) voice: Option<Arc<VoiceProviders>>,
    /// Workspace-external image cache. `None` when the operator set
    /// `[image_cache] enabled = false`, when cache directory resolution
    /// failed at startup, or when `dirs::cache_dir()` returned `None`
    /// (rare). Absent → no in-memory scrub; on-disk persistence still
    /// gets the hash-marker fallback from `SessionStore::append`.
    pub(crate) image_cache: Option<Arc<crate::image_cache::ImageCache>>,
    /// Active satellites, keyed by `(device_id, room_profile)`. Inserted
    /// by `voice/subscribe`, removed by the per-subscription writer task
    /// when its SSE channel closes (i.e. satellite disconnects).
    pub(crate) voice_subscribers: Arc<VoiceSubscribers>,
    /// Bearer token -> device -> room profile. Shared with ambient ingest so
    /// there is exactly one answer to "who is this token" in the process.
    pub(crate) device_auth: Arc<crate::device_auth::DeviceAuth>,
}

impl ServeState {
    /// Construct a runtime ready for both the HTTP RPC server and
    /// the in-process channel handlers (Discord voice in particular).
    /// Shared across both so they read from the same session store /
    /// in-memory conversation map.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        config: Config,
        registry: Arc<ProviderRegistry>,
        workspace: Arc<Workspace>,
        tools: Arc<ToolSet>,
        cross_device_session_store: Arc<SessionStore>,
        device_default_session_store: Arc<SessionStore>,
        mcp_session_store: Arc<SessionStore>,
        voice: Option<Arc<VoiceProviders>>,
        image_cache: Option<Arc<crate::image_cache::ImageCache>>,
        device_auth: Arc<crate::device_auth::DeviceAuth>,
    ) -> Self {
        // Scan once on startup: each MCP session's first-line meta
        // carries `namespace` + `project`, so this reproduces the
        // logical mapping without a side-channel index file. The same
        // map is updated in-place when `write_report` creates a new
        // project session.
        let mut mcp_index: HashMap<(String, String), String> = HashMap::new();
        for meta in mcp_session_store.list_sessions() {
            let (Some(ns), Some(proj)) = (meta.namespace.clone(), meta.project.clone()) else {
                continue;
            };
            // `list_sessions` is sorted by `created_at`, so overwriting
            // here keeps the most recent session per (ns, project) —
            // matters only if a project ever ended up with multiple
            // session files (manual surgery, future reset semantics).
            mcp_index.insert((ns, proj), meta.session_id);
        }

        Self {
            config,
            registry,
            workspace,
            tools,
            cross_device_session_store,
            device_default_session_store,
            mcp_session_store,
            mcp_project_index: tokio::sync::Mutex::new(mcp_index),
            sessions: tokio::sync::Mutex::new(HashMap::new()),
            pending_sessions: tokio::sync::Mutex::new(HashMap::new()),
            session_room_profiles: tokio::sync::Mutex::new(HashMap::new()),
            session_room_metadata: tokio::sync::Mutex::new(HashMap::new()),
            voice,
            voice_subscribers: Arc::new(tokio::sync::Mutex::new(HashMap::new())),
            image_cache,
            device_auth,
        }
    }

    /// Pick the [`SessionStore`] that owns `session_id`. Device-default
    /// sessions land in `device-default/`; everything else (cross-device
    /// text sessions, deferred sessions awaiting their first message)
    /// lives in `cross_device_session_store`'s `rpc/` tree. Falls back
    /// to the cross-device store so newly-`ensure_session`'d files
    /// (which haven't hit disk yet) commit to the right place. See #122.
    pub(crate) fn store_for_session(&self, session_id: &str) -> &Arc<SessionStore> {
        if self
            .device_default_session_store
            .absolute_path_for(session_id)
            .is_some()
        {
            &self.device_default_session_store
        } else {
            &self.cross_device_session_store
        }
    }

    /// Look up the MCP session id for `(namespace, project)`. Returns
    /// `None` if the project has never received a report.
    pub(crate) async fn mcp_session_for_project(
        &self,
        namespace: &str,
        project: &str,
    ) -> Option<String> {
        self.mcp_project_index
            .lock()
            .await
            .get(&(namespace.to_string(), project.to_string()))
            .cloned()
    }

    /// Look up (or create) the MCP session for `(namespace, project)`.
    /// First call for a project creates the underlying session file
    /// and registers it in the index; subsequent calls hit the index.
    /// Concurrent calls for the same new project are serialized
    /// through the index mutex so only one session file is created.
    pub(crate) async fn mcp_session_for_project_or_create(
        &self,
        namespace: &str,
        project: &str,
    ) -> anyhow::Result<String> {
        let key = (namespace.to_string(), project.to_string());
        {
            let idx = self.mcp_project_index.lock().await;
            if let Some(id) = idx.get(&key) {
                return Ok(id.clone());
            }
        }
        // Hold the lock across creation so two simultaneous
        // first-time writers for the same project don't each spawn
        // a session file. Double-check inside the lock in case
        // another task won the race between our two acquisitions.
        let mut idx = self.mcp_project_index.lock().await;
        if let Some(id) = idx.get(&key) {
            return Ok(id.clone());
        }
        let session_id = self
            .mcp_session_store
            .create_mcp_session(namespace, project)?;
        idx.insert(key, session_id.clone());
        Ok(session_id)
    }

    /// Provider that should serve the given session. Resolves the
    /// session's pinned room_profile to its `profile`, then to a
    /// concrete provider (with optional refusal fallback). Falls back
    /// to the background provider when no room_profile is pinned.
    pub(crate) async fn provider_for_session(&self, session_id: &str) -> Arc<dyn Provider> {
        let rp_name = self
            .session_room_profiles
            .lock()
            .await
            .get(session_id)
            .cloned();
        match rp_name.and_then(|n| self.config.room_profile(&n).map(|rp| rp.profile.clone())) {
            Some(profile_name) => self.registry.for_profile(&self.config, &profile_name),
            None => self.registry.background_provider(&self.config),
        }
    }
}

// ---------------------------------------------------------------------------
// JSON-RPC 2.0 types
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct JsonRpcRequest {
    #[allow(dead_code)]
    jsonrpc: Option<String>,
    id: Option<Value>,
    method: String,
    params: Option<Value>,
}

fn error_response(id: Value, code: i32, message: &str) -> (StatusCode, axum::Json<Value>) {
    let body = json!({
        "jsonrpc": "2.0",
        "id": id,
        "error": { "code": code, "message": message },
    });
    (StatusCode::OK, axum::Json(body))
}

fn notification_event(method: &'static str, params: Value) -> Event {
    let data = json!({
        "jsonrpc": "2.0",
        "method": method,
        "params": params,
    });
    Event::default().data(data.to_string())
}

fn result_event(id: &Value, result: Value) -> Event {
    let data = json!({
        "jsonrpc": "2.0",
        "id": id,
        "result": result,
    });
    Event::default().data(data.to_string())
}

fn error_event(id: &Value, code: i32, message: &str) -> Event {
    let data = json!({
        "jsonrpc": "2.0",
        "id": id,
        "error": { "code": code, "message": message },
    });
    Event::default().data(data.to_string())
}

// ---------------------------------------------------------------------------
// Router entry point
// ---------------------------------------------------------------------------

/// `extra` mounts the ambient audio ingest router (`POST /audio/ingest`,
/// `POST /audio/hello`) alongside `/rpc`, when the subsystem started.
/// It is a plain `Router` — not a field on `ServeState` — because the
/// ingest endpoint owns its own state deliberately: it never starts an
/// LLM turn, so it has no use for the runtime `ServeState` exists to
/// serve. Both routers are already resolved to `Router<()>` via their
/// own `.with_state()` call, which is what makes `.merge` valid here
/// despite the two states having nothing in common.
pub async fn run(
    addr: String,
    state: Arc<ServeState>,
    extra: Option<axum::Router>,
) -> anyhow::Result<()> {
    // Routes are intentionally separated so future protocol endpoints
    // (`/mcp` for the MCP server in #79/#80) can be mounted alongside
    // `/rpc` without colliding with the methods (`chat`,
    // `initialize`, `voice/*`, …) that live here. The A2A protocol
    // endpoints below are mounted unconditionally — the handler refuses
    // requests when `[a2a].enabled = false` so we don't pay a route
    // table conditional but still preserve the opt-in semantic.
    let mut app = Router::new()
        .route("/rpc", post(rpc_post).get(rpc_get))
        .route("/a2a", post(a2a::handle_a2a_post))
        .route("/mcp", post(mcp::handle_mcp_post))
        .route("/acp", axum::routing::get(acp::handle_acp_ws))
        .route(
            "/.well-known/agent-card.json",
            axum::routing::get(a2a::handle_agent_card),
        )
        .layer(tower_http::cors::CorsLayer::permissive())
        .with_state(Arc::clone(&state));
    if let Some(extra) = extra {
        app = app.merge(extra);
    }

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    info!("sapphire-agent: API server listening on http://{addr}");
    let shutdown_state = Arc::clone(&state);
    axum::serve(listener, app)
        .with_graceful_shutdown(async move {
            if let Err(e) = tokio::signal::ctrl_c().await {
                error!("Failed to install Ctrl-C handler: {e}");
            }
            info!("HTTP server shutting down...");
        })
        .await?;
    summarize_all_sessions(&shutdown_state).await;
    Ok(())
}

/// Summarize every in-memory API session and append a `SummaryLine` so the
/// next process can recover context without replaying raw history.
async fn summarize_all_sessions(state: &Arc<ServeState>) {
    let snapshot: Vec<(String, Vec<ChatMessage>)> = {
        let sessions = state.sessions.lock().await;
        sessions
            .iter()
            .filter(|(_, msgs)| msgs.len() >= 2)
            .map(|(sid, msgs)| (sid.clone(), msgs.clone()))
            .collect()
    };
    if snapshot.is_empty() {
        return;
    }
    info!(
        "Graceful shutdown: summarizing {} RPC session(s)",
        snapshot.len()
    );
    for (session_id, messages) in snapshot {
        let provider = state.provider_for_session(&session_id).await;
        let store = state.store_for_session(&session_id);
        match generate_summary(&*provider, &messages).await {
            Ok(summary) if !summary.trim().is_empty() => {
                if let Err(e) = store.append_summary(&session_id, &summary) {
                    warn!("Failed to persist shutdown summary for {session_id}: {e}");
                }
                if let Err(e) = store.append_intraday_digest(&session_id, &summary, None) {
                    warn!("Failed to persist shutdown intra-day digest for {session_id}: {e}");
                }
            }
            Ok(_) => warn!("Shutdown summary for {session_id} was empty; skipping"),
            Err(e) => warn!("Shutdown summary generation failed for {session_id}: {e:#}"),
        }
    }
}

// ---------------------------------------------------------------------------
// POST /rpc  — dispatch JSON-RPC methods
// ---------------------------------------------------------------------------

/// JSON-RPC error code returned when the Authorization header is present
/// but the bearer token does not resolve through `DeviceAuth`. Mirrors
/// `codes::AUTH_REQUIRED` used by `/a2a` and `/mcp` so the three protocol
/// surfaces stay symmetrical.
const RPC_AUTH_REQUIRED: i32 = -32001;

async fn rpc_post(
    State(state): State<Arc<ServeState>>,
    headers: HeaderMap,
    Json(req): Json<JsonRpcRequest>,
) -> impl IntoResponse {
    let session_id = headers
        .get("session-id")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string());

    let req_id = req.id.clone().unwrap_or(Value::Null);

    // Bearer auth → room_profile reverse lookup. The token IS the
    // profile selector; clients no longer pass `room_profile` in
    // params. Missing/empty bearer → 401 at the HTTP layer (matches
    // /a2a); unknown token → JSON-RPC AUTH_REQUIRED.
    let bearer = match extract_bearer(&headers) {
        Some(b) => b,
        None => {
            return (StatusCode::UNAUTHORIZED, "missing bearer token").into_response();
        }
    };
    let (profile_name, authenticated_device_id) = match state.device_auth.resolve(&bearer) {
        Some(r) => (r.room_profile.to_string(), r.device.id.to_string()),
        None => {
            let body = error_response(req_id, RPC_AUTH_REQUIRED, "unknown or revoked bearer token");
            return body.into_response();
        }
    };

    match req.method.as_str() {
        "initialize" => {
            handle_initialize(state, req_id, req.params, session_id, profile_name).await
        }
        "chat" => handle_chat(state, req_id, req.params, session_id).await,
        "list_sessions" => handle_list_sessions(state, req_id).await,
        "get_session" => handle_get_session(state, req_id, session_id).await,
        "voice/config" => handle_voice_config(state, req_id, req.params).await,
        "voice/pipeline_run" => {
            handle_voice_pipeline_run(
                state,
                req_id,
                req.params,
                profile_name,
                authenticated_device_id,
            )
            .await
        }
        "voice/subscribe" => {
            handle_voice_subscribe(
                state,
                req_id,
                req.params,
                profile_name,
                authenticated_device_id,
            )
            .await
        }
        _ => {
            let body = error_response(req_id, -32601, "Method not found");
            body.into_response()
        }
    }
}

/// Extract a `Bearer <token>` from an `Authorization` header, trimming
/// whitespace.
///
/// Returns `None` when the header is absent, uses another scheme, or
/// carries an empty token — every one of which the endpoints treat as
/// "unauthenticated" rather than "malformed".
///
/// `pub(crate)` so `/rpc`, `/a2a`, `/mcp`, `/acp` and
/// `crate::ambient::ingest` share one set of parsing rules rather than
/// each duplicating them.
pub(crate) fn extract_bearer(headers: &HeaderMap) -> Option<String> {
    let value = headers.get(axum::http::header::AUTHORIZATION)?;
    let s = value.to_str().ok()?;
    let token = s
        .strip_prefix("Bearer ")
        .or_else(|| s.strip_prefix("bearer "))?;
    let trimmed = token.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

// ---------------------------------------------------------------------------
// GET /rpc  — Phase 2 placeholder (server→client push)
// ---------------------------------------------------------------------------

async fn rpc_get() -> impl IntoResponse {
    (
        StatusCode::METHOD_NOT_ALLOWED,
        "GET /rpc is reserved for Phase 2 server-initiated tool requests",
    )
}

// ---------------------------------------------------------------------------
// initialize
// ---------------------------------------------------------------------------

async fn handle_initialize(
    state: Arc<ServeState>,
    req_id: Value,
    params: Option<Value>,
    existing_header_session: Option<String>,
    profile_name: String,
) -> axum::response::Response {
    // `room_profile` is no longer accepted as a JSON-RPC param — the
    // bearer token resolved in `rpc_post` is the sole profile selector
    // (mirrors A2A / MCP). `DeviceAuth::resolve` only returns profile
    // names that exist in the config, so no extra validation is needed
    // here.

    // Resolve to an internal UUID session_id.
    // - Session-Id header: already a UUID (internal), use directly.
    // - params.session_id: must be a 7-char grain-id (public) or "new"/absent.
    let resolved: Option<String> = if let Some(uuid) = existing_header_session {
        // Header carries the internal UUID we issued — trust it directly.
        Some(uuid)
    } else {
        let param_id = params
            .as_ref()
            .and_then(|p| p["session_id"].as_str())
            .filter(|s| *s != "new")
            .map(|s| s.to_string());

        match param_id {
            None => None,
            Some(ref id) if id.len() == 7 => {
                match state.cross_device_session_store.find_by_public_id(id) {
                    Some(uuid) => Some(uuid),
                    None => {
                        let body = error_response(req_id, -32602, "Session not found");
                        return body.into_response();
                    }
                }
            }
            Some(_) => {
                let body = error_response(
                    req_id,
                    -32602,
                    "Invalid session id (expected 7-char grain-id)",
                );
                return body.into_response();
            }
        }
    };

    let (session_id, is_new) = match resolved {
        Some(id) => {
            let exists = state.cross_device_session_store.load_session(&id).is_some();
            (id, !exists)
        }
        None => (uuid::Uuid::now_v7().to_string(), true),
    };

    // For brand-new sessions, defer file creation until the first chat arrives.
    // Reserve the public_id now so the client can display it immediately.
    let public_id = if is_new {
        let pid = grain_id::GrainId::random().to_string();
        state
            .pending_sessions
            .lock()
            .await
            .insert(session_id.clone(), pid.clone());
        Some(pid)
    } else {
        // Existing session: load metadata to retrieve the stored public_id
        // and pre-load history into memory.
        let mut sessions = state.sessions.lock().await;
        sessions.entry(session_id.clone()).or_insert_with(|| {
            state
                .cross_device_session_store
                .load_session(&session_id)
                .unwrap_or_default()
        });
        // Look up the public_id from the existing file metadata.
        state
            .cross_device_session_store
            .list_sessions()
            .into_iter()
            .find(|m| m.session_id == session_id)
            .and_then(|m| m.public_id)
    };

    state
        .session_room_profiles
        .lock()
        .await
        .insert(session_id.clone(), profile_name.clone());

    // Optional `params.device = { name, description }` from sapphire-call /
    // other voice clients. We treat `name` as the device handle (e.g.
    // "living-room-speaker") and render the full room name server-side
    // — that way every voice client doesn't have to agree on a template
    // and the agent stays in control of how the metadata is presented.
    if let Some(device) = params.as_ref().and_then(|p| p.get("device")) {
        let device_name = device
            .get("name")
            .and_then(|v| v.as_str())
            .map(str::to_string);
        let device_description = device
            .get("description")
            .and_then(|v| v.as_str())
            .map(str::to_string);
        if let Some(name) = device_name {
            let room_info = RoomInfo {
                name: format!("voice channel with {name}"),
                description: device_description,
                kind: "voice".to_string(),
            };
            state
                .session_room_metadata
                .lock()
                .await
                .insert(session_id.clone(), room_info);
        }
    }

    let mut result = json!({
        "session_id": session_id,
        "is_new": is_new,
    });
    if let Some(ref pub_id) = public_id {
        result["public_id"] = json!(pub_id);
    }
    if let Some(name) = state.session_room_profiles.lock().await.get(&session_id) {
        result["room_profile"] = json!(name);
    }

    let body = json!({
        "jsonrpc": "2.0",
        "id": req_id,
        "result": result,
    });

    let mut response = axum::response::Response::builder()
        .status(StatusCode::OK)
        .header("content-type", "application/json")
        .header(
            "session-id",
            HeaderValue::from_str(&session_id).unwrap_or_else(|_| HeaderValue::from_static("")),
        )
        .body(axum::body::Body::from(body.to_string()))
        .unwrap();

    // Also set Session-Id in the response headers (accessible via response)
    response.headers_mut().insert(
        "session-id",
        HeaderValue::from_str(&session_id).unwrap_or_else(|_| HeaderValue::from_static("")),
    );

    response
}

// ---------------------------------------------------------------------------
// chat  — returns SSE stream
// ---------------------------------------------------------------------------

async fn handle_chat(
    state: Arc<ServeState>,
    req_id: Value,
    params: Option<Value>,
    session_id: Option<String>,
) -> axum::response::Response {
    let session_id = match session_id {
        Some(id) => id,
        None => {
            // No session: return JSON error
            let body = error_response(req_id, -32602, "Missing Session-Id header");
            return body.into_response();
        }
    };

    let content = match params.as_ref().and_then(|p| p["content"].as_str()) {
        Some(c) => c.to_string(),
        None => {
            let body = error_response(req_id, -32602, "Missing params.content");
            return body.into_response();
        }
    };

    // Clients may opt into out-of-band TTS by setting
    // `modalities: ["text", "audio"]`. Default = text-only (existing
    // behaviour preserved for the CLI REPL / one-shot / channel paths).
    // Unknown modality strings are silently dropped — additive future
    // modalities won't break older clients that don't recognise them.
    let want_audio = params
        .as_ref()
        .and_then(|p| p.get("modalities"))
        .and_then(|v| v.as_array())
        .map(|arr| arr.iter().any(|m| m.as_str() == Some("audio")))
        .unwrap_or(false);

    let (tx, rx) = mpsc::channel::<Result<Event, Infallible>>(32);

    // Spawn the turn processor
    tokio::spawn(async move {
        run_turn(state, session_id, content, want_audio, req_id, tx).await;
    });

    let stream = ReceiverStream::new(rx);
    Sse::new(stream)
        .keep_alive(KeepAlive::new().interval(std::time::Duration::from_secs(15)))
        .into_response()
}

// ---------------------------------------------------------------------------
// list_sessions
// ---------------------------------------------------------------------------

async fn handle_list_sessions(state: Arc<ServeState>, req_id: Value) -> axum::response::Response {
    let metas = state.cross_device_session_store.list_sessions();
    let items: Vec<Value> = metas
        .into_iter()
        .map(|m| {
            let mut v = json!({
                "session_id": m.session_id,
                "created_at": m.created_at,
            });
            if let Some(pub_id) = m.public_id {
                v["public_id"] = json!(pub_id);
            }
            if let Some(title) = m.title {
                v["title"] = json!(title);
            }
            v
        })
        .collect();

    let body = json!({
        "jsonrpc": "2.0",
        "id": req_id,
        "result": { "sessions": items },
    });
    (StatusCode::OK, axum::Json(body)).into_response()
}

// ---------------------------------------------------------------------------
// get_session  — returns stored messages for the current session
// ---------------------------------------------------------------------------

async fn handle_get_session(
    state: Arc<ServeState>,
    req_id: Value,
    session_id: Option<String>,
) -> axum::response::Response {
    let session_id = match session_id {
        Some(id) => id,
        None => {
            let body = error_response(req_id, -32602, "Missing Session-Id header");
            return body.into_response();
        }
    };

    let messages = state
        .store_for_session(&session_id)
        .load_session(&session_id)
        .unwrap_or_default();

    let items: Vec<Value> = messages
        .iter()
        .map(|m| {
            let role = match m.role {
                crate::provider::Role::User => "user",
                crate::provider::Role::Assistant => "assistant",
            };
            let parts: Vec<Value> = m
                .parts
                .iter()
                .map(|p| match p {
                    ContentPart::Text(t) => json!({ "type": "text", "text": t }),
                    ContentPart::Image { media_type, .. } => {
                        // Image bytes are not exposed via the RPC listing; surface a marker only.
                        json!({ "type": "image", "media_type": media_type })
                    }
                    ContentPart::ImageRef { media_type, sha256 } => {
                        // Same shape as Image, with the cache key surfaced so
                        // a caller can later fetch the bytes out of band.
                        json!({ "type": "image", "media_type": media_type, "sha256": sha256 })
                    }
                    ContentPart::ToolUse { id, name, input } => {
                        json!({ "type": "tool_use", "id": id, "name": name, "input": input })
                    }
                    ContentPart::ToolResult { tool_use_id, content } => {
                        json!({ "type": "tool_result", "tool_use_id": tool_use_id, "content": content })
                    }
                })
                .collect();
            json!({ "role": role, "parts": parts })
        })
        .collect();

    let body = json!({
        "jsonrpc": "2.0",
        "id": req_id,
        "result": { "messages": items },
    });
    (StatusCode::OK, axum::Json(body)).into_response()
}

// ---------------------------------------------------------------------------
// voice/pipeline_run  — STT → LLM turn → TTS, streamed via SSE
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// voice/config — return the room_profile's wake-word + (future)
// per-session voice settings so satellites can self-configure
// ---------------------------------------------------------------------------

async fn handle_voice_config(
    state: Arc<ServeState>,
    req_id: Value,
    _params: Option<Value>,
) -> axum::response::Response {
    use base64::Engine as _;
    use sha2::{Digest, Sha256};

    // Wake-word config is global — the same Saphina (or whatever)
    // greets the user across every room_profile. `params` is reserved
    // for a future per-profile override but currently unused.
    let mut result = json!({});
    if let Some(path_str) = &state.config.voice.wake_word_model {
        let expanded = shellexpand::tilde(path_str).into_owned();
        match std::fs::read(&expanded) {
            Ok(bytes) => {
                let mut hasher = Sha256::new();
                hasher.update(&bytes);
                let hash = hasher.finalize();
                use std::fmt::Write;
                let mut sha = String::with_capacity(64);
                for b in hash.iter() {
                    let _ = write!(&mut sha, "{b:02x}");
                }
                let b64 = base64::engine::general_purpose::STANDARD.encode(&bytes);
                let filename = std::path::Path::new(&expanded)
                    .file_name()
                    .and_then(|s| s.to_str())
                    .unwrap_or("wake.onnx")
                    .to_string();
                result["wake_word_model"] = json!({
                    "format": "onnx_inline",
                    "filename": filename,
                    "sha256": sha,
                    "data_b64": b64,
                });
            }
            Err(e) => {
                error!("voice/config: failed to read openWakeWord model '{expanded}': {e}");
                let body = error_response(
                    req_id,
                    -32603,
                    &format!("voice.wake_word_model '{expanded}' could not be read: {e}"),
                );
                return body.into_response();
            }
        }
    }

    let body = json!({
        "jsonrpc": "2.0",
        "id": req_id,
        "result": result,
    });
    (StatusCode::OK, axum::Json(body)).into_response()
}

async fn handle_voice_pipeline_run(
    state: Arc<ServeState>,
    req_id: Value,
    params: Option<Value>,
    room_profile: String,
    authenticated_device_id: String,
) -> axum::response::Response {
    if state.voice.is_none() {
        let body = error_response(
            req_id,
            -32601,
            "voice/pipeline_run unavailable: no STT/TTS providers configured",
        );
        return body.into_response();
    }

    let params = params.unwrap_or(Value::Null);
    let audio_b64 = match params["audio"].as_str() {
        Some(s) => s.to_string(),
        None => {
            let body = error_response(req_id, -32602, "Missing params.audio (base64 PCM)");
            return body.into_response();
        }
    };
    let device_id = match params["device_id"].as_str() {
        Some(s) => s.to_string(),
        None => {
            let body = error_response(req_id, -32602, "Missing params.device_id");
            return body.into_response();
        }
    };
    // The client-supplied `device_id` is only ever used to key sessions and
    // the voice-push registry — it is never itself an auth check. Without
    // this comparison, an authenticated client could claim another device's
    // id and read/write that device's session or displace its push channel
    // (`voice_subscribers` is keyed on `device_id` alone). `DeviceAuth`
    // already established which device this bearer token belongs to; hold
    // it to that.
    if device_id != authenticated_device_id {
        let body = error_response(
            req_id,
            -32602,
            "params.device_id does not match the authenticated device",
        );
        return body.into_response();
    }
    // `room_profile` comes from the bearer token resolved in
    // `rpc_post`; clients no longer pass it as a param.
    let language = params["language"].as_str().map(|s| s.to_string());

    // Resolve / lazily-create the device-default session for this
    // `(device_id, room_profile)` pair under that profile's memory
    // namespace. Daily rotation falls out naturally: a satellite
    // reconnecting after the day boundary finds yesterday's file as
    // "not in today's window" and a fresh UUID file is opened. See #122.
    let namespace = state
        .config
        .namespace_for_room_profile(&room_profile)
        .to_string();
    let session_id = match state
        .device_default_session_store
        .find_or_create_for_device(
            &device_id,
            &room_profile,
            &namespace,
            state.config.day_boundary_hour,
        ) {
        Ok(id) => id,
        Err(e) => {
            let body = error_response(
                req_id,
                -32603,
                &format!("failed to resolve device-default session: {e}"),
            );
            return body.into_response();
        }
    };
    state
        .session_room_profiles
        .lock()
        .await
        .insert(session_id.clone(), room_profile.clone());

    // Same `device` block accepted by `initialize` — refreshed on every
    // pipeline_run so satellites can update their description without a
    // separate handshake. Treated as room metadata for the session.
    if let Some(device) = params.get("device") {
        let device_name = device
            .get("name")
            .and_then(|v| v.as_str())
            .map(str::to_string);
        let device_description = device
            .get("description")
            .and_then(|v| v.as_str())
            .map(str::to_string);
        if let Some(name) = device_name {
            let room_info = RoomInfo {
                name: format!("voice channel with {name}"),
                description: device_description,
                kind: "voice".to_string(),
            };
            state
                .session_room_metadata
                .lock()
                .await
                .insert(session_id.clone(), room_info);
        }
    }

    let (tx, rx) = mpsc::channel::<Result<Event, Infallible>>(64);

    let device_id_for_timer = device_id.clone();
    tokio::spawn(async move {
        run_voice_turn(
            state,
            session_id,
            audio_b64,
            language,
            req_id,
            tx,
            Some(device_id_for_timer),
        )
        .await;
    });

    let stream = ReceiverStream::new(rx);
    Sse::new(stream)
        .keep_alive(KeepAlive::new().interval(std::time::Duration::from_secs(15)))
        .into_response()
}

// ---------------------------------------------------------------------------
// voice/subscribe — long-lived SSE for server→satellite voice pushes
// ---------------------------------------------------------------------------

async fn handle_voice_subscribe(
    state: Arc<ServeState>,
    req_id: Value,
    params: Option<Value>,
    room_profile: String,
    authenticated_device_id: String,
) -> axum::response::Response {
    let params = params.unwrap_or(Value::Null);
    let device_id = match params["device_id"].as_str() {
        Some(s) => s.to_string(),
        None => {
            let body = error_response(req_id, -32602, "Missing params.device_id");
            return body.into_response();
        }
    };
    // See the identical check in `handle_voice_pipeline_run`: without it, an
    // authenticated client could claim another device's id and displace its
    // push channel, since `voice_subscribers` is keyed on `device_id` alone.
    if device_id != authenticated_device_id {
        let body = error_response(
            req_id,
            -32602,
            "params.device_id does not match the authenticated device",
        );
        return body.into_response();
    }
    // `room_profile` comes from the bearer token resolved in
    // `rpc_post`; clients no longer pass it as a param.

    // Replace any prior subscription for this device (typical case:
    // the same satellite reconnects after a brief network blip). The
    // old sender is dropped; its writer task exits on the first
    // failed send. The room_profile may also have changed across
    // reconnect, so the freshest value wins — that's the satellite's
    // current binding.
    let (push_tx, push_rx) = mpsc::channel::<crate::voice::VoicePushItem>(32);
    {
        let mut subs = state.voice_subscribers.lock().await;
        subs.insert(device_id.clone(), (room_profile.clone(), push_tx));
    }
    info!("voice/subscribe: registered (device={device_id}, room_profile={room_profile})");

    let (sse_tx, sse_rx) = mpsc::channel::<Result<Event, Infallible>>(32);
    let cleanup_state = Arc::clone(&state);
    let cleanup_device = device_id.clone();
    tokio::spawn(async move {
        translate_voice_pushes(push_rx, sse_tx).await;
        // SSE writer exited (satellite disconnected or push channel
        // closed). Remove the subscriber entry — but only if it still
        // points at our (now-dropped) sender, since a subsequent
        // reconnect may have already replaced it.
        let mut subs = cleanup_state.voice_subscribers.lock().await;
        if subs
            .get(&cleanup_device)
            .map(|(_, tx)| tx.is_closed())
            .unwrap_or(false)
            && let Some((rp, _)) = subs.remove(&cleanup_device)
        {
            info!("voice/subscribe: unregistered (device={cleanup_device}, room_profile={rp})");
        }
    });

    let stream = ReceiverStream::new(sse_rx);
    Sse::new(stream)
        .keep_alive(KeepAlive::new().interval(std::time::Duration::from_secs(15)))
        .into_response()
}

/// Forward [`VoicePushItem`]s from the per-subscriber mpsc channel into
/// SSE notification events. Exits when either the push channel closes
/// (server cleanup) or the SSE channel closes (client disconnect).
async fn translate_voice_pushes(
    mut push_rx: mpsc::Receiver<crate::voice::VoicePushItem>,
    sse_tx: mpsc::Sender<Result<Event, Infallible>>,
) {
    use base64::Engine;
    while let Some(item) = push_rx.recv().await {
        let evt = match item {
            crate::voice::VoicePushItem::Start { task } => {
                let mut params = json!({"kind": "push_start"});
                if let Some(t) = task {
                    params["task"] = json!(t);
                }
                notification_event("notifications/voice_push", params)
            }
            crate::voice::VoicePushItem::AssistantText(text) => notification_event(
                "notifications/voice_push",
                json!({"kind": "assistant_text", "text": text}),
            ),
            crate::voice::VoicePushItem::AudioChunk(pcm) => {
                let bytes: Vec<u8> = pcm.iter().flat_map(|s| s.to_le_bytes()).collect();
                let b64 = base64::engine::general_purpose::STANDARD.encode(&bytes);
                notification_event(
                    "notifications/voice_push",
                    json!({"kind": "audio_chunk", "data": b64}),
                )
            }
            crate::voice::VoicePushItem::Done => {
                notification_event("notifications/voice_push", json!({"kind": "push_done"}))
            }
            crate::voice::VoicePushItem::Error(message) => notification_event(
                "notifications/voice_push",
                json!({"kind": "error", "message": message}),
            ),
        };
        if sse_tx.send(Ok(evt)).await.is_err() {
            break;
        }
    }
}

async fn run_voice_turn(
    state: Arc<ServeState>,
    session_id: String,
    audio_b64: String,
    language: Option<String>,
    req_id: Value,
    tx: mpsc::Sender<Result<Event, Infallible>>,
    device_id: Option<String>,
) {
    use base64::Engine;

    let send = |evt: Event| {
        let tx = tx.clone();
        async move {
            let _ = tx.send(Ok(evt)).await;
        }
    };

    // Resolve voice pipeline (need STT here; from_text resolves TTS again).
    let pipeline = match resolve_voice_pipeline(&state, &session_id).await {
        Ok(p) => p,
        Err(VoicePipelineLookup::NoVoice) => {
            send(error_event(
                &req_id,
                -32601,
                "voice/pipeline_run unavailable: no STT/TTS providers configured",
            ))
            .await;
            return;
        }
        Err(VoicePipelineLookup::NotConfigured) => {
            send(error_event(
                &req_id,
                -32602,
                "Session's room_profile has no voice_pipeline configured",
            ))
            .await;
            return;
        }
    };
    let voice_registry = state.voice.as_ref().expect("checked above").clone();
    let stt = match voice_registry.stt(&pipeline.stt_provider) {
        Some(p) => p,
        None => {
            send(error_event(
                &req_id,
                -32603,
                &format!("stt_provider '{}' not instantiated", pipeline.stt_provider),
            ))
            .await;
            return;
        }
    };

    // Decode audio.
    let audio_bytes = match base64::engine::general_purpose::STANDARD.decode(audio_b64.as_bytes()) {
        Ok(b) => b,
        Err(e) => {
            send(error_event(
                &req_id,
                -32602,
                &format!("Invalid base64 audio: {e}"),
            ))
            .await;
            return;
        }
    };
    if audio_bytes.len() % 2 != 0 {
        send(error_event(
            &req_id,
            -32602,
            "Audio byte length is not a multiple of 2 (expected s16le)",
        ))
        .await;
        return;
    }
    let pcm: Vec<i16> = audio_bytes
        .chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]))
        .collect();

    // Stage: STT
    info!(
        "voice/pipeline_run: STT via '{}' ({} samples, lang={:?})",
        stt.name(),
        pcm.len(),
        language.as_deref().or(pipeline.language.as_deref()),
    );
    send(notification_event(
        "notifications/progress",
        json!({"kind": "stage", "stage": "stt", "status": "start"}),
    ))
    .await;
    let lang = language.as_deref().or(pipeline.language.as_deref());
    let transcript = match stt.transcribe(&pcm, lang).await {
        Ok(t) => t,
        Err(e) => {
            error!("STT failed: {e:#}");
            send(error_event(&req_id, -32603, &format!("STT failed: {e}"))).await;
            return;
        }
    };
    send(notification_event(
        "notifications/progress",
        json!({"kind": "stt_final", "text": transcript}),
    ))
    .await;
    send(notification_event(
        "notifications/progress",
        json!({"kind": "stage", "stage": "stt", "status": "end"}),
    ))
    .await;

    // Hand off to the from-text path for everything past STT.
    run_voice_turn_from_text_sse(state, session_id, transcript, req_id, tx, device_id).await;
}

/// Voice pipeline failure when looking up the per-session config.
enum VoicePipelineLookup {
    NoVoice,
    NotConfigured,
}

async fn resolve_voice_pipeline(
    state: &Arc<ServeState>,
    session_id: &str,
) -> Result<crate::config::VoicePipelineConfig, VoicePipelineLookup> {
    if state.voice.is_none() {
        return Err(VoicePipelineLookup::NoVoice);
    }
    let rp_name = state
        .session_room_profiles
        .lock()
        .await
        .get(session_id)
        .cloned();
    rp_name
        .as_deref()
        .and_then(|n| state.config.voice_pipeline_for_room_profile(n))
        .cloned()
        .ok_or(VoicePipelineLookup::NotConfigured)
}

/// LLM turn + TTS streaming, with progress emitted as SSE notifications
/// for the original `voice/pipeline_run` caller. The final JSON-RPC
/// result event ends the stream.
async fn run_voice_turn_from_text_sse(
    state: Arc<ServeState>,
    session_id: String,
    user_text: String,
    req_id: Value,
    tx: mpsc::Sender<Result<Event, Infallible>>,
    device_id: Option<String>,
) {
    use base64::Engine;

    let send = |evt: Event| {
        let tx = tx.clone();
        async move {
            let _ = tx.send(Ok(evt)).await;
        }
    };

    let pipeline = match resolve_voice_pipeline(&state, &session_id).await {
        Ok(p) => p,
        Err(VoicePipelineLookup::NoVoice) => {
            send(error_event(
                &req_id,
                -32601,
                "voice unavailable: no STT/TTS providers configured",
            ))
            .await;
            return;
        }
        Err(VoicePipelineLookup::NotConfigured) => {
            send(error_event(
                &req_id,
                -32602,
                "Session's room_profile has no voice_pipeline configured",
            ))
            .await;
            return;
        }
    };
    let voice_registry = state.voice.as_ref().expect("checked above").clone();
    let tts = match voice_registry.tts(&pipeline.tts_provider) {
        Some(p) => p,
        None => {
            send(error_event(
                &req_id,
                -32603,
                &format!("tts_provider '{}' not instantiated", pipeline.tts_provider),
            ))
            .await;
            return;
        }
    };

    // Stage: LLM (intent)
    send(notification_event(
        "notifications/progress",
        json!({"kind": "stage", "stage": "intent", "status": "start"}),
    ))
    .await;
    let outcome = run_llm_turn(
        Arc::clone(&state),
        session_id.clone(),
        ChatMessage::user_voice(&user_text),
        Arc::new(SseProgress::new(tx.clone(), req_id.clone())),
        device_id
            .clone()
            .map(|d| crate::timer::TimerOrigin::Voice { device_id: d }),
    )
    .await;
    send(notification_event(
        "notifications/progress",
        json!({"kind": "stage", "stage": "intent", "status": "end"}),
    ))
    .await;
    let reply_text = match outcome.text {
        Some(t) => t,
        None => {
            // run_llm_turn already emitted a provider error_event.
            return;
        }
    };
    send(notification_event(
        "notifications/progress",
        json!({"kind": "assistant_text", "text": reply_text}),
    ))
    .await;

    // Stage: TTS
    info!(
        "voice/pipeline_run: TTS via '{}' ({} chars)",
        tts.name(),
        reply_text.len(),
    );
    send(notification_event(
        "notifications/progress",
        json!({"kind": "stage", "stage": "tts", "status": "start"}),
    ))
    .await;
    let (pcm_tx, mut pcm_rx) = mpsc::channel::<Vec<i16>>(32);
    let reply_for_tts = reply_text.clone();
    let synth_handle =
        tokio::spawn(async move { tts.synthesize_stream(&reply_for_tts, pcm_tx).await });
    let mut chunks_emitted = 0usize;
    while let Some(chunk) = pcm_rx.recv().await {
        let bytes: Vec<u8> = chunk.iter().flat_map(|s| s.to_le_bytes()).collect();
        let b64 = base64::engine::general_purpose::STANDARD.encode(&bytes);
        send(notification_event(
            "notifications/progress",
            json!({"kind": "audio_chunk", "data": b64}),
        ))
        .await;
        chunks_emitted += 1;
    }
    // Surface TTS failures to the client — without this the satellite
    // saw a silent "no audio_chunks" stream and assumed playback was
    // empty, which looked like a text-only reply.
    match synth_handle.await {
        Ok(Ok(())) => {
            if chunks_emitted == 0 {
                warn!(
                    "TTS returned no audio chunks (provider: {})",
                    pipeline.tts_provider
                );
                send(error_event(
                    &req_id,
                    -32603,
                    &format!(
                        "TTS provider '{}' produced no audio (check fn_name / payload / audio_field)",
                        pipeline.tts_provider
                    ),
                ))
                .await;
                return;
            }
        }
        Ok(Err(e)) => {
            error!("TTS synthesis error: {e:#}");
            send(error_event(
                &req_id,
                -32603,
                &format!("TTS synthesis failed: {e:#}"),
            ))
            .await;
            return;
        }
        Err(join_err) => {
            error!("TTS task panicked: {join_err}");
            send(error_event(
                &req_id,
                -32603,
                &format!("TTS task panicked: {join_err}"),
            ))
            .await;
            return;
        }
    }
    send(notification_event(
        "notifications/progress",
        json!({"kind": "stage", "stage": "tts", "status": "end"}),
    ))
    .await;

    // Final result: transcript + reply text. Audio was streamed via
    // progress events; no need to duplicate it here.
    send(result_event(
        &req_id,
        json!({
            "transcript": user_text,
            "assistant_text": reply_text,
        }),
    ))
    .await;

    // Title generation on first turn — same as run_turn.
    if outcome.was_first_turn {
        let state2 = Arc::clone(&state);
        let sid = session_id.clone();
        let reply = reply_text.clone();
        tokio::spawn(async move {
            let p = state2.provider_for_session(&sid).await;
            if let Some(title) = generate_session_title(&*p, &user_text, &reply).await
                && let Err(e) = state2.store_for_session(&sid).set_title(&sid, &title)
            {
                warn!("Failed to store session title: {e}");
            }
        });
    }
}

/// Failure modes for [`push_voice_text_to_subscriber`]. `Offline` lets
/// the heartbeat caller decide whether to fall back to a chat room.
pub enum VoicePushError {
    /// The server has no `[stt_provider.*]` / `[tts_provider.*]` blocks
    /// configured at all — voice push is fundamentally unavailable.
    NoVoice,
    /// The room_profile is unknown or has no `voice_pipeline` set.
    NotConfigured,
    /// No satellite is currently subscribed for this `(device_id,
    /// room_profile)` pair. Caller should fall back to chat if the
    /// heartbeat task has a `room_id`, or log and skip otherwise.
    Offline,
    /// Any other failure (TTS, LLM, etc.) surfaced for logging.
    Other(String),
}

/// Server-initiated voice push: run the LLM turn against the voice
/// session bound to `device_id` and stream the TTS audio to the
/// satellite subscribed via `voice/subscribe`.
///
/// The satellite supplied its current `room_profile` when it
/// subscribed — that value is the authoritative reverse index, so the
/// caller never has to duplicate it (a satellite's room_profile can
/// only change via a fresh subscription, which atomically replaces
/// the binding).
///
/// `task_name` becomes the heartbeat task identifier echoed in the
/// `PushStart` event so the satellite can label notifications.
pub(crate) async fn push_voice_text_to_subscriber(
    state: Arc<ServeState>,
    device_id: String,
    task_name: Option<String>,
    user_text: String,
) -> Result<(), VoicePushError> {
    // Look up the active subscription up front — if the satellite is
    // offline, surface that without burning an LLM call. The map also
    // tells us which room_profile the satellite is bound to.
    let (room_profile, push_tx) = {
        let subs = state.voice_subscribers.lock().await;
        match subs.get(&device_id) {
            Some((rp, tx)) => (rp.clone(), tx.clone()),
            None => return Err(VoicePushError::Offline),
        }
    };
    if state.config.room_profile(&room_profile).is_none() {
        return Err(VoicePushError::NotConfigured);
    }

    // Resolve / lazily-create the device-default session for this
    // `(device_id, room_profile)` pair under that profile's memory
    // namespace, then pin the room_profile so `resolve_voice_pipeline`
    // and `run_llm_turn` find the right config. See #122.
    let namespace = state
        .config
        .namespace_for_room_profile(&room_profile)
        .to_string();
    let session_id = state
        .device_default_session_store
        .find_or_create_for_device(
            &device_id,
            &room_profile,
            &namespace,
            state.config.day_boundary_hour,
        )
        .map_err(|e| VoicePushError::Other(format!("device-default lookup: {e}")))?;
    state
        .session_room_profiles
        .lock()
        .await
        .insert(session_id.clone(), room_profile.clone());

    let pipeline = match resolve_voice_pipeline(&state, &session_id).await {
        Ok(p) => p,
        Err(VoicePipelineLookup::NoVoice) => return Err(VoicePushError::NoVoice),
        Err(VoicePipelineLookup::NotConfigured) => return Err(VoicePushError::NotConfigured),
    };
    let voice_registry = state.voice.as_ref().ok_or(VoicePushError::NoVoice)?.clone();
    let tts = voice_registry.tts(&pipeline.tts_provider).ok_or_else(|| {
        VoicePushError::Other(format!(
            "tts_provider '{}' not instantiated",
            pipeline.tts_provider
        ))
    })?;

    // Notify the satellite that a push is starting so it can mute the
    // mic before the first audio chunk lands.
    let _ = push_tx
        .send(crate::voice::VoicePushItem::Start {
            task: task_name.clone(),
        })
        .await;

    // LLM turn (no SSE response channel — nobody watches this
    // heartbeat-injected turn's tool_start/tool_end progress).
    // Heartbeat-injected user line — synthesised by the timer pipeline,
    // not authored by a human, so no input modality applies.
    let injected_msg = ChatMessage {
        role: crate::provider::Role::User,
        parts: vec![crate::provider::ContentPart::Text(user_text.clone())],
        input_kind: None,
        user_id: None,
    };
    let outcome = run_llm_turn(
        Arc::clone(&state),
        session_id.clone(),
        injected_msg,
        Arc::new(NullProgress),
        Some(crate::timer::TimerOrigin::Voice {
            device_id: device_id.clone(),
        }),
    )
    .await;
    let reply_text = match outcome.text {
        Some(t) => t,
        None => {
            let msg = "LLM turn produced no text".to_string();
            let _ = push_tx
                .send(crate::voice::VoicePushItem::Error(msg.clone()))
                .await;
            let _ = push_tx.send(crate::voice::VoicePushItem::Done).await;
            return Err(VoicePushError::Other(msg));
        }
    };
    let _ = push_tx
        .send(crate::voice::VoicePushItem::AssistantText(
            reply_text.clone(),
        ))
        .await;

    // TTS: stream chunks to the subscriber as soon as they're synthesised.
    let (pcm_tx, mut pcm_rx) = mpsc::channel::<Vec<i16>>(32);
    let reply_for_tts = reply_text.clone();
    let synth_handle =
        tokio::spawn(async move { tts.synthesize_stream(&reply_for_tts, pcm_tx).await });
    let mut chunks_emitted = 0usize;
    while let Some(chunk) = pcm_rx.recv().await {
        if push_tx
            .send(crate::voice::VoicePushItem::AudioChunk(chunk))
            .await
            .is_err()
        {
            // Satellite disconnected mid-stream; abort the synth task.
            synth_handle.abort();
            return Err(VoicePushError::Offline);
        }
        chunks_emitted += 1;
    }
    match synth_handle.await {
        Ok(Ok(())) if chunks_emitted == 0 => {
            let msg = format!("TTS provider '{}' produced no audio", pipeline.tts_provider);
            warn!("{msg}");
            let _ = push_tx
                .send(crate::voice::VoicePushItem::Error(msg.clone()))
                .await;
            let _ = push_tx.send(crate::voice::VoicePushItem::Done).await;
            return Err(VoicePushError::Other(msg));
        }
        Ok(Ok(())) => {}
        Ok(Err(e)) => {
            let msg = format!("TTS synthesis failed: {e:#}");
            error!("{msg}");
            let _ = push_tx
                .send(crate::voice::VoicePushItem::Error(msg.clone()))
                .await;
            let _ = push_tx.send(crate::voice::VoicePushItem::Done).await;
            return Err(VoicePushError::Other(msg));
        }
        Err(join_err) => {
            let msg = format!("TTS task panicked: {join_err}");
            error!("{msg}");
            let _ = push_tx
                .send(crate::voice::VoicePushItem::Error(msg.clone()))
                .await;
            let _ = push_tx.send(crate::voice::VoicePushItem::Done).await;
            return Err(VoicePushError::Other(msg));
        }
    }

    let _ = push_tx.send(crate::voice::VoicePushItem::Done).await;
    Ok(())
}

// ---------------------------------------------------------------------------
// Turn processing (tool-calling loop)
// ---------------------------------------------------------------------------

/// Rewrite a user-role message so the model sees the input modality.
/// Voice transcripts are prefixed with `[voice input]` (English, since
/// the model is more reliable with English meta-tags) so the assistant
/// can treat the body as STT output rather than typed text. Text and
/// modality-less messages pass through unchanged.
fn apply_input_kind_label(mut msg: ChatMessage) -> ChatMessage {
    let prefix = match &msg.input_kind {
        Some(UserInputKind::Voice) => "[voice input]\n",
        _ => return msg,
    };
    for part in msg.parts.iter_mut() {
        if let ContentPart::Text(s) = part {
            *s = format!("{prefix}{s}");
            return msg;
        }
    }
    msg.parts
        .insert(0, ContentPart::Text(prefix.trim_end().to_string()));
    msg
}

/// Where a turn reports its per-tool progress.
///
/// `run_llm_turn` is the shared executor behind `/rpc`, the voice pipeline,
/// A2A and the ACP endpoint — but each caller wants its
/// `tool_start`/`tool_end`/error notifications shaped differently: `/rpc`
/// and voice relay them as JSON-RPC notifications over SSE, ACP sends them
/// as `session/update` notifications instead, and some callers (voice
/// heartbeats, A2A) don't surface intermediate progress at all. Putting
/// reporting behind this trait keeps the turn executor itself agnostic to
/// which of those shapes (if any) is listening.
///
/// The per-transport hook a turn reports through — and, now, asks
/// through.
///
/// Renamed from `TurnProgress`: it is no longer only about reporting.
/// Both new methods carry defaults, so a transport that has no way to
/// ask a human implements neither and keeps behaving exactly as it did.
#[async_trait::async_trait]
pub(crate) trait TurnHost: Send + Sync {
    async fn tool_start(&self, id: &str, name: &str);
    async fn tool_end(&self, id: &str, name: &str);
    async fn turn_error(&self, message: &str);

    /// Which row of the permission table this turn is judged by.
    ///
    /// `Trusted` by default: `/rpc`, voice and the heartbeat were
    /// authenticated before the turn started and have no UI to ask
    /// through, so they must keep running everything.
    fn origin(&self) -> crate::tools::policy::Origin {
        crate::tools::policy::Origin::Trusted
    }

    /// Called only when `decide` returned `Ask`.
    ///
    /// The default answers `AllowOnce` — a host that never returns an
    /// asking `origin()` can never reach this, so the default exists
    /// for safety rather than for use.
    async fn approve(
        &self,
        call: &crate::provider::ToolCall,
        kind: crate::tools::ToolKind,
    ) -> crate::tools::policy::Approval {
        let _ = (call, kind);
        crate::tools::policy::Approval::AllowOnce
    }
}

/// Builds the `{id, name}` params shared by the `tool_start`/`tool_end`
/// wire notifications. Pulled out so tests can pin the field names
/// directly without inspecting an opaque SSE `Event`.
fn tool_event_params(id: &str, name: &str) -> Value {
    json!({ "id": id, "name": name })
}

/// The `/rpc` and voice shape: JSON-RPC notifications (and, on provider
/// failure, a JSON-RPC error) delivered over the SSE channel — exactly
/// what `run_llm_turn` used to emit inline before progress reporting moved
/// behind [`TurnHost`].
pub(crate) struct SseProgress {
    tx: mpsc::Sender<Result<Event, Infallible>>,
    req_id: Value,
}

impl SseProgress {
    pub(crate) fn new(tx: mpsc::Sender<Result<Event, Infallible>>, req_id: Value) -> Self {
        Self { tx, req_id }
    }
}

#[async_trait::async_trait]
impl TurnHost for SseProgress {
    async fn tool_start(&self, id: &str, name: &str) {
        let _ = self
            .tx
            .send(Ok(notification_event(
                "tool_start",
                tool_event_params(id, name),
            )))
            .await;
    }

    async fn tool_end(&self, id: &str, name: &str) {
        let _ = self
            .tx
            .send(Ok(notification_event(
                "tool_end",
                tool_event_params(id, name),
            )))
            .await;
    }

    async fn turn_error(&self, message: &str) {
        let _ = self
            .tx
            .send(Ok(error_event(&self.req_id, -32603, message)))
            .await;
    }
}

/// Discard progress. Used by callers that drive a turn to completion with
/// nobody watching intermediate events (voice heartbeats, A2A v1).
pub(crate) struct NullProgress;

#[async_trait::async_trait]
impl TurnHost for NullProgress {
    async fn tool_start(&self, _id: &str, _name: &str) {}
    async fn tool_end(&self, _id: &str, _name: &str) {}
    async fn turn_error(&self, _message: &str) {}
}

/// Why a turn stopped.
///
/// Exists because `text: None` conflates two materially different endings:
/// the provider broke, and the model was still working when its tool-round
/// budget ran out. A transport that has a way to say "budget" — ACP's
/// `StopReason::MaxTurnRequests` — cannot tell them apart from the text
/// alone, and answering "internal error" for a turn that was merely long is
/// wrong in a way the user sees.
///
/// Callers that don't care may keep reading `text` alone: this is an extra
/// field on the outcome, not a replacement, and every existing caller
/// ignores it.
pub(crate) enum TurnStop {
    /// The model produced its final message; `text` is `Some`.
    Replied,
    /// A `Provider::chat` call failed. `TurnHost::turn_error` has
    /// already been handed the message, so the cause is available to
    /// whoever is reporting; `text` is `None`.
    ProviderError,
    /// [`MAX_TOOL_ROUNDS`] was reached with the model still calling tools.
    /// `text` is `None` — deliberately, because every caller that predates
    /// ACP treats this as a failed turn and must keep doing so — but the
    /// prose the model emitted alongside its tool calls is real work, and
    /// is carried here rather than discarded. It may be empty.
    BudgetExhausted { partial_text: String },
}

/// Outcome of [`run_llm_turn`].
pub(crate) struct LlmTurnOutcome {
    /// Final assistant text, when the turn completed successfully. `None`
    /// on provider error or when MAX_TOOL_ROUNDS was hit without resolving.
    text: Option<String>,
    /// True iff the session had no prior turns before this one. Used by
    /// callers to decide whether to spawn a title-generation task.
    was_first_turn: bool,
    /// Which of those endings this was. See [`TurnStop`].
    stop: TurnStop,
}

/// Execute one full LLM turn for an established session: hydrate history,
/// run the tool-calling loop, persist user + assistant messages to JSONL,
/// and report per-tool `tool_start` / `tool_end` progress through
/// `progress`. Does NOT send the final JSON-RPC result event — the caller
/// is responsible for shaping the final payload (text reply, voice audio,
/// etc.) and emitting the appropriate result event.
///
/// # This future may be dropped mid-turn
///
/// `/rpc` and the voice/heartbeat paths run this inside a detached
/// `tokio::spawn`, so it finishes whether or not the client is still there.
/// Two callers can drop it instead: the ACP endpoint does so deliberately,
/// on `session/cancel` and on a vanished client (`src/serve/acp.rs`, the
/// `session/prompt` handler's `tokio::select!`), and `/a2a` awaits it
/// directly in an axum handler, whose future hyper may drop when an HTTP
/// client disconnects mid-request. Either way it is dropped at whatever
/// await point it happens to be sitting on.
///
/// Nothing here unwinds, so a dropped turn leaves a *split*:
///
/// - The user message and any compaction summary are already on disk — they
///   are appended to JSONL as they happen (steps 4 and 5). The `state.sessions`
///   write-back at the end never runs. So a cancelled prompt is invisible to
///   the next turn's in-memory model context, yet present in `list_sessions`
///   and after a restart; a compaction summary can be persisted and then
///   thrown away, and will be produced again next turn.
/// - Tool futures in flight are dropped too. `ShellTool` therefore sets
///   `kill_on_drop(true)` (`src/tools/builtin_tools.rs`) — without it a
///   cancelled turn left a shell command running against the workspace.
///   Any tool added later that owns an external process, a lock or a
///   partially-written file must be drop-safe for the same reason.
///
/// Anything added to this function that must happen exactly once per turn
/// needs to be written with that in mind: reaching the end of the body is
/// not guaranteed.
pub(crate) async fn run_llm_turn(
    state: Arc<ServeState>,
    session_id: String,
    user_msg: ChatMessage,
    progress: Arc<dyn TurnHost>,
    timer_origin: Option<crate::timer::TimerOrigin>,
) -> LlmTurnOutcome {
    // Pick the right store up front so every persistence call in this
    // turn lands in the same place (device-default vs cross-device).
    let store = Arc::clone(state.store_for_session(&session_id));

    // 1. Load or lazy-hydrate in-memory history
    let mut history: Vec<ChatMessage> = {
        let mut sessions = state.sessions.lock().await;
        sessions
            .entry(session_id.clone())
            .or_insert_with(|| store.load_session(&session_id).unwrap_or_default())
            .clone()
    };
    let was_first_turn = history.is_empty();

    // 2. Resolve provider once per turn — sessions can pin a profile at
    //    initialize-time; absent that, the background provider is used.
    let provider = state.provider_for_session(&session_id).await;

    // 2a. Namespace chain follows the session's pinned room_profile when
    //     set; otherwise the implicit default namespace. Resolved here
    //     so it can be recorded in the session metadata on first chat
    //     (used by the today-digest builder to route NSFW digests away
    //     from default-namespace rooms).
    let namespace = match state.session_room_profiles.lock().await.get(&session_id) {
        Some(rp_name) => state.config.namespace_for_room_profile(rp_name).to_string(),
        None => crate::config::DEFAULT_NAMESPACE_NAME.to_string(),
    };
    let namespace_chain = state.config.resolve_namespace_chain(&namespace);

    // 2b. Ensure JSONL file exists. If this session was deferred at initialize
    //     time, commit it now using the reserved public_id.
    //
    // Device-default sessions are always already-on-disk by the time
    // run_llm_turn runs (find_or_create_for_device writes the meta
    // line at create time) so `ensure_session` against them is a
    // no-op. Skip it to avoid synthesising a spurious grain-id
    // public_id that device-default sessions don't need.
    let key: ConversationKey = (session_id.clone(), None);
    if Arc::ptr_eq(&store, &state.cross_device_session_store) {
        let pending_pub_id = state.pending_sessions.lock().await.remove(&session_id);
        if let Err(e) = store
            .ensure_session(&session_id, &key, "rpc", pending_pub_id, &namespace)
            .map(|_| ())
        {
            warn!("Failed to ensure session file: {e}");
        }
    }

    // 3a. System prompt (rebuilt fresh per request).
    let room_info = state
        .session_room_metadata
        .lock()
        .await
        .get(&session_id)
        .cloned();
    let system = {
        let sp = state
            .workspace
            .build_system_prompt(
                state.config.anthropic.system_prompt.as_deref(),
                state.config.day_boundary_hour,
                &namespace_chain,
                room_info.as_ref(),
            )
            .await;
        if sp.is_empty() { None } else { Some(sp) }
    };

    // 4. Append user message. Image scrubbing for storage is handled inside
    //    `SessionStore::append` so the in-memory history keeps full image
    //    bytes for the provider call while JSONL gets a hash marker.
    history.push(user_msg.clone());
    if let Err(e) = store.append(&session_id, &user_msg) {
        warn!("Failed to persist user message: {e}");
    }

    // 5. Tool-calling loop — refresh MCP tools if any server signalled a change.
    state.tools.refresh_if_needed().await;
    let tool_specs = state.tools.specs().await;
    let compression_config = &state.config.compression;
    let mut accumulated_text: Vec<String> = Vec::new();
    let (final_text, stop) = loop {
        let round = history
            .iter()
            .filter(|m| {
                m.parts
                    .iter()
                    .any(|p| matches!(p, ContentPart::ToolUse { .. }))
            })
            .count();

        if round >= MAX_TOOL_ROUNDS {
            warn!("Reached max tool rounds ({MAX_TOOL_ROUNDS})");
            // `None` for the text, as before — callers that predate ACP
            // report this as a failed turn and must keep doing so — but the
            // prose accumulated so far rides along on the stop reason for
            // transports that can show partial work.
            break (
                None,
                TurnStop::BudgetExhausted {
                    partial_text: accumulated_text.join("\n\n"),
                },
            );
        }

        // Check if context compression is needed
        match maybe_compress(&*provider, system.as_deref(), &history, compression_config).await {
            Ok(Some(result)) => {
                history = result.compressed;
                if let Err(e) = store.append_summary(&session_id, &result.summary) {
                    warn!("Failed to persist compaction summary: {e}");
                }
            }
            Ok(None) => {}
            Err(e) => {
                warn!("Context compression failed, continuing with full history: {e}");
            }
        }

        // Hydrate `ImageRef` parts from the image cache into full
        // `Image` parts for the provider call. `Image` parts (just
        // arrived this turn) and Text/Tool parts pass through;
        // `ImageRef` parts are intentionally degraded to text markers
        // so historical images aren't re-billed every turn (the cache
        // still retains the bytes for an on-demand recall tool).
        // After hydration, fold each user message's input modality
        // into a textual prefix so the model knows when a body is a
        // voice transcript (likely to contain STT errors).
        let history_for_provider: Vec<ChatMessage> = crate::image_cache::hydrate_history(&history)
            .into_iter()
            .map(apply_input_kind_label)
            .collect();
        let response = provider
            .chat(system.as_deref(), &history_for_provider, Some(&tool_specs))
            .await;

        match response {
            Err(e) => {
                error!("Provider error: {e:#}");
                progress.turn_error(&e.to_string()).await;
                break (None, TurnStop::ProviderError);
            }
            Ok(resp) if !resp.has_tool_calls() => {
                let text = resp.text.unwrap_or_default();
                let msg = ChatMessage::assistant(&text);
                history.push(msg.clone());
                if let Err(e) = store.append(&session_id, &msg) {
                    warn!("Failed to persist assistant message: {e}");
                }
                if !text.is_empty() {
                    accumulated_text.push(text);
                }
                break (Some(accumulated_text.join("\n\n")), TurnStop::Replied);
            }
            Ok(resp) => {
                let tool_calls = resp.tool_calls.clone();
                if let Some(t) = resp.text.as_ref().filter(|s| !s.is_empty()) {
                    accumulated_text.push(t.clone());
                }
                let msg = ChatMessage::assistant_with_tools(resp.text.clone(), tool_calls.clone());
                history.push(msg.clone());
                // Tool_use messages are intentionally not persisted: they
                // can be arbitrarily large and we never reload raw tool
                // history across restarts anyway (compaction summaries cover
                // the semantic context).

                // Notify client of each tool starting
                for call in &tool_calls {
                    progress.tool_start(&call.id, &call.name).await;
                }

                // Permission gate.
                //
                // Serial on purpose. `decide` is a cheap pure call, but
                // `approve` puts a dialog in front of a human, and
                // firing several at once would stack them on the poor
                // soul in the editor. Execution below stays concurrent.
                let kinds = state.tools.kinds().await;
                let origin = progress.origin();
                let mut permitted: Vec<crate::provider::ToolCall> = Vec::new();
                let mut refused: Vec<(String, String)> = Vec::new();
                for call in &tool_calls {
                    use crate::tools::policy::{Decision, Refusal, kind_of, refusal_message};

                    let kind = kind_of(&call.name, &kinds);
                    let verdict = crate::tools::policy::decide(origin, kind);
                    let refusal = match verdict {
                        Decision::Allow => None,
                        Decision::Deny => Some(refusal_message(&call.name, Refusal::Unavailable)),
                        Decision::Ask => {
                            if progress.approve(call, kind).await.allows() {
                                None
                            } else {
                                Some(refusal_message(&call.name, Refusal::UserDeclined))
                            }
                        }
                    };

                    match refusal {
                        None => permitted.push(call.clone()),
                        Some(reason) => {
                            info!("Refused tool {} (id={}): {verdict:?}", call.name, call.id);
                            refused.push((call.id.clone(), reason));
                        }
                    }
                }

                // Execute all tools concurrently — each call wrapped in
                // the session's memory namespace (task_local) so the
                // memory tool writes under `memory/<namespace>/...`.
                let tools = Arc::clone(&state.tools);
                let ns = namespace.clone();
                let timer_origin = timer_origin.clone();
                let mut results: Vec<(String, crate::tools::ToolOutput)> =
                    futures_util::future::join_all(permitted.iter().map(|c| {
                        let tools = Arc::clone(&tools);
                        let c = c.clone();
                        let ns = ns.clone();
                        let origin = timer_origin.clone();
                        async move {
                            let fut = crate::tools::workspace_tools::scope_memory_namespace(
                                ns,
                                async move {
                                    info!("Executing tool: {} (id={})", c.name, c.id);
                                    let output = tools.execute(&c).await;
                                    info!("Tool {} done", c.name);
                                    (c.id, output)
                                },
                            );
                            match origin {
                                Some(o) => crate::timer::scope_timer_origin(o, fut).await,
                                None => fut.await,
                            }
                        }
                    }))
                    .await;

                // A refused call still owes the model a tool_result:
                // every tool_use in the assistant message must be
                // answered, and the reason is more useful to the model
                // than silence. The turn is NOT ended — the model may
                // have another route, and ACP's `Refusal` stop reason
                // means "the agent declined", which is a different
                // thing from "the user declined".
                for (id, reason) in refused {
                    results.push((id, crate::tools::ToolOutput::from(reason)));
                }

                // Notify client of each tool completing
                for call in &tool_calls {
                    progress.tool_end(&call.id, &call.name).await;
                }

                let mut text_results = Vec::with_capacity(results.len());
                let mut images = Vec::new();
                for (id, output) in results {
                    text_results.push((id, output.text));
                    images.extend(output.images);
                }
                let result_msg = ChatMessage::tool_results_with_images(text_results, images);
                history.push(result_msg.clone());
                // Tool_result payloads are not persisted — see the matching
                // tool_use branch above for rationale.
            }
        }
    };

    // Scrub `Image` parts in the just-completed history into compact
    // `ImageRef` references backed by the workspace-external image
    // cache. After this, long-lived in-memory storage is hash-only;
    // the next turn re-hydrates from cache for the provider call.
    crate::image_cache::scrub_history_inplace(&mut history, state.image_cache.as_deref());

    // Update in-memory sessions map
    state
        .sessions
        .lock()
        .await
        .insert(session_id.clone(), history);

    LlmTurnOutcome {
        text: final_text,
        was_first_turn,
        stop,
    }
}

async fn run_turn(
    state: Arc<ServeState>,
    session_id: String,
    user_message: String,
    want_audio: bool,
    req_id: Value,
    tx: mpsc::Sender<Result<Event, Infallible>>,
) {
    let send = |evt: Event| {
        let tx = tx.clone();
        async move {
            let _ = tx.send(Ok(evt)).await;
        }
    };

    let outcome = run_llm_turn(
        Arc::clone(&state),
        session_id.clone(),
        ChatMessage::user(&user_message),
        Arc::new(SseProgress::new(tx.clone(), req_id.clone())),
        None,
    )
    .await;

    // GUI clients that asked for `audio` get the text-then-audio
    // sequence over progress notifications before the final result.
    // TTS failures here are non-fatal: text is still useful, so we
    // surface a `tts_error` notification and continue to the result.
    if want_audio && let Some(text) = outcome.text.as_deref() {
        send(notification_event(
            "notifications/progress",
            json!({"kind": "assistant_text", "text": text}),
        ))
        .await;
        stream_chat_tts(&state, &session_id, text, &tx).await;
    }

    // Send final result
    match &outcome.text {
        Some(text) => {
            send(result_event(&req_id, json!({ "content": text }))).await;
        }
        None => {
            send(error_event(&req_id, -32603, "No response generated")).await;
        }
    }

    // Generate and store session title after the first successful turn.
    if outcome.was_first_turn
        && let Some(text) = outcome.text
    {
        let state2 = Arc::clone(&state);
        let sid = session_id.clone();
        let user_msg = user_message.clone();
        tokio::spawn(async move {
            let p = state2.provider_for_session(&sid).await;
            if let Some(title) = generate_session_title(&*p, &user_msg, &text).await
                && let Err(e) = state2.store_for_session(&sid).set_title(&sid, &title)
            {
                warn!("Failed to store session title: {e}");
            }
        });
    }
}

// ---------------------------------------------------------------------------
// Chat → TTS bridge (modalities=["text","audio"])
// ---------------------------------------------------------------------------

/// Synthesize `reply_text` via the session's voice_pipeline TTS provider
/// and stream the resulting PCM as `audio_chunk` progress notifications.
///
/// Best-effort: any failure (no voice configured, provider missing,
/// synth error, zero chunks) surfaces as a `tts_error` progress
/// notification and returns — the caller still emits the text result,
/// since GUI clients prefer "text without audio" over "no answer at
/// all" when TTS is misconfigured.
///
/// Intentionally separate from `run_voice_turn_from_text_sse`'s TTS
/// block: the voice path emits `stage` markers and treats TTS failure
/// as fatal (it aborts the JSON-RPC result). Sharing would risk
/// regressing that behaviour.
async fn stream_chat_tts(
    state: &Arc<ServeState>,
    session_id: &str,
    reply_text: &str,
    tx: &mpsc::Sender<Result<Event, Infallible>>,
) {
    use base64::Engine;

    let send = |evt: Event| {
        let tx = tx.clone();
        async move {
            let _ = tx.send(Ok(evt)).await;
        }
    };
    let emit_tts_error = |message: String| {
        let tx = tx.clone();
        async move {
            let _ = tx
                .send(Ok(notification_event(
                    "notifications/progress",
                    json!({"kind": "tts_error", "message": message}),
                )))
                .await;
        }
    };

    let pipeline = match resolve_voice_pipeline(state, session_id).await {
        Ok(p) => p,
        Err(VoicePipelineLookup::NoVoice) => {
            emit_tts_error(
                "voice unavailable: no STT/TTS providers configured server-side".to_string(),
            )
            .await;
            return;
        }
        Err(VoicePipelineLookup::NotConfigured) => {
            emit_tts_error("session's room_profile has no voice_pipeline configured".to_string())
                .await;
            return;
        }
    };
    let voice_registry = match state.voice.as_ref() {
        Some(v) => v.clone(),
        None => {
            emit_tts_error("voice registry unavailable".to_string()).await;
            return;
        }
    };
    let tts = match voice_registry.tts(&pipeline.tts_provider) {
        Some(p) => p,
        None => {
            emit_tts_error(format!(
                "tts_provider '{}' not instantiated",
                pipeline.tts_provider
            ))
            .await;
            return;
        }
    };

    let (pcm_tx, mut pcm_rx) = mpsc::channel::<Vec<i16>>(32);
    let reply_for_tts = reply_text.to_string();
    let synth_handle =
        tokio::spawn(async move { tts.synthesize_stream(&reply_for_tts, pcm_tx).await });
    let mut chunks_emitted = 0usize;
    while let Some(chunk) = pcm_rx.recv().await {
        let bytes: Vec<u8> = chunk.iter().flat_map(|s| s.to_le_bytes()).collect();
        let b64 = base64::engine::general_purpose::STANDARD.encode(&bytes);
        send(notification_event(
            "notifications/progress",
            json!({"kind": "audio_chunk", "data": b64}),
        ))
        .await;
        chunks_emitted += 1;
    }
    match synth_handle.await {
        Ok(Ok(())) => {
            if chunks_emitted == 0 {
                emit_tts_error(format!(
                    "TTS provider '{}' produced no audio",
                    pipeline.tts_provider
                ))
                .await;
            }
        }
        Ok(Err(e)) => {
            error!("chat TTS synthesis error: {e:#}");
            emit_tts_error(format!("TTS synthesis failed: {e:#}")).await;
        }
        Err(join_err) => {
            error!("chat TTS task panicked: {join_err}");
            emit_tts_error(format!("TTS task panicked: {join_err}")).await;
        }
    }
}

// ---------------------------------------------------------------------------
// Title generation
// ---------------------------------------------------------------------------

async fn generate_session_title(
    provider: &dyn Provider,
    user_message: &str,
    assistant_response: &str,
) -> Option<String> {
    let user_snippet: String = user_message.chars().take(300).collect();
    let asst_snippet: String = assistant_response.chars().take(300).collect();
    let prompt = format!(
        "Generate a concise title (max 60 characters) for this conversation. \
        Respond with only the title text — no quotes, no punctuation at the end.\n\n\
        User: {user_snippet}\nAssistant: {asst_snippet}"
    );
    let messages = vec![ChatMessage::user(&prompt)];
    match provider.chat(None, &messages, None).await {
        Ok(resp) => resp.text.map(|t| {
            let t = t.trim().to_string();
            if t.chars().count() > 60 {
                t.chars().take(60).collect()
            } else {
                t
            }
        }),
        Err(e) => {
            warn!("Title generation failed: {e:#}");
            None
        }
    }
}

// ---------------------------------------------------------------------------
// Test fixtures — ServeState backed by a scripted / hanging stub provider.
// Every endpoint/turn test elsewhere needs a ServeState; this is the one
// place that knows how to build one without a real Anthropic API key.
// ---------------------------------------------------------------------------

/// The one tool the fixture advertises, so a scripted turn has something
/// real to call: a turn whose tool call names a tool nobody registered
/// would exercise `ToolSet`'s "Unknown tool" path instead of an execution.
#[cfg(test)]
pub(crate) struct EchoTool {
    spec: crate::provider::ToolSpec,
}

#[cfg(test)]
impl EchoTool {
    pub(crate) fn new() -> Self {
        Self {
            spec: crate::provider::ToolSpec {
                name: "echo".into(),
                description: "Echo the given text back.".into(),
                input_schema: json!({
                    "type": "object",
                    "properties": { "text": { "type": "string" } },
                    "required": ["text"]
                }),
            },
        }
    }
}

#[cfg(test)]
#[async_trait::async_trait]
impl crate::tools::Tool for EchoTool {
    fn spec(&self) -> &crate::provider::ToolSpec {
        &self.spec
    }

    /// A test fixture standing in for an ordinary read-only tool.
    /// Deliberately NOT `Other`: that is the ask-me bucket, and the
    /// pre-existing ACP tests drive turns with a helper that answers no
    /// permission requests, so leaving this unclassified would make them
    /// hang rather than fail once the permission gate lands.
    fn kind(&self) -> crate::tools::ToolKind {
        crate::tools::ToolKind::Read
    }

    async fn execute(&self, input: &Value) -> anyhow::Result<String> {
        Ok(input["text"].as_str().unwrap_or_default().to_string())
    }
}

/// A stand-in for `shell`: `ToolKind::Execute`, so the policy asks or
/// refuses. Carries a per-instance "did I run" flag, which is how the
/// gate tests tell "refused" from "ran and returned an error".
///
/// Per instance, not a `static`: `cargo test` runs tests in parallel and
/// several tests construct one of these, so a process-global flag would
/// make the assertions depend on scheduling.
#[cfg(test)]
pub(crate) struct RiskyTool {
    spec: crate::provider::ToolSpec,
    ran: Arc<std::sync::atomic::AtomicBool>,
}

#[cfg(test)]
impl RiskyTool {
    pub(crate) fn new() -> Self {
        Self {
            spec: crate::provider::ToolSpec {
                name: "risky".into(),
                description: "Pretend to run a command.".into(),
                input_schema: json!({ "type": "object", "properties": {} }),
            },
            ran: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        }
    }

    /// A handle the test keeps after the tool is boxed into the ToolSet.
    pub(crate) fn ran_flag(&self) -> Arc<std::sync::atomic::AtomicBool> {
        Arc::clone(&self.ran)
    }
}

#[cfg(test)]
#[async_trait::async_trait]
impl crate::tools::Tool for RiskyTool {
    fn spec(&self) -> &crate::provider::ToolSpec {
        &self.spec
    }

    fn kind(&self) -> crate::tools::ToolKind {
        crate::tools::ToolKind::Execute
    }

    async fn execute(&self, _input: &Value) -> anyhow::Result<String> {
        self.ran.store(true, std::sync::atomic::Ordering::SeqCst);
        Ok("ran".to_string())
    }
}

/// Provider double for tests. In "scripted" mode it pops one
/// [`crate::provider::ChatResponse`] off a queue per `chat()` call. In
/// "hanging" mode `chat()` never resolves — used to keep a turn in
/// flight while a cancellation test races it.
#[cfg(test)]
pub(crate) struct StubProvider {
    script: Option<std::sync::Mutex<std::collections::VecDeque<crate::provider::ChatResponse>>>,
    /// Only set in hanging mode. Counts entries into the parked `chat()` —
    /// see [`HangingChat::entered`].
    hang_entered: Option<Arc<std::sync::atomic::AtomicUsize>>,
    /// Only set in hanging mode. Counts in-flight `chat()` futures that were
    /// dropped (e.g. their task was aborted), so a cancellation test can
    /// assert the turn was actually torn down rather than merely observing
    /// that it never completed on its own.
    hang_dropped: Option<Arc<std::sync::atomic::AtomicUsize>>,
}

/// The two observations a test can make about a hanging provider's
/// `chat()` calls.
///
/// `entered` exists so a test never has to *guess* that a turn has reached
/// the provider. Cancelling a turn that has not got that far yet drops a
/// future that never built the guard, so `dropped` would stay behind and the
/// assertion would be a coin toss; waiting for `entered` first makes it a
/// fact.
///
/// Both are counters rather than flags because prompts on one ACP connection
/// run concurrently: a test that opens two turns needs to wait for *both* to
/// reach the provider, and to see both of them torn down.
#[cfg(test)]
pub(crate) struct HangingChat {
    /// Incremented just before `chat()` parks forever, so a test can wait
    /// until N turns are genuinely inside the provider before taking their
    /// connection away.
    pub(crate) entered: Arc<std::sync::atomic::AtomicUsize>,
    /// Incremented when one of those parked futures is dropped: the proof
    /// that an abandoned turn actually stopped calling the provider.
    pub(crate) dropped: Arc<std::sync::atomic::AtomicUsize>,
}

#[cfg(test)]
impl HangingChat {
    /// Wait until one of the counters reaches `at_least`, or fail with
    /// `what` naming the thing that never happened.
    pub(crate) async fn wait_for(
        counter: &std::sync::atomic::AtomicUsize,
        at_least: usize,
        what: &str,
    ) {
        let waited = tokio::time::timeout(std::time::Duration::from_secs(10), async {
            while counter.load(std::sync::atomic::Ordering::SeqCst) < at_least {
                tokio::time::sleep(std::time::Duration::from_millis(5)).await;
            }
        })
        .await;
        assert!(
            waited.is_ok(),
            "timed out waiting for {what} (reached {}, wanted {at_least})",
            counter.load(std::sync::atomic::Ordering::SeqCst)
        );
    }
}

#[cfg(test)]
impl StubProvider {
    pub(crate) fn new(responses: Vec<crate::provider::ChatResponse>) -> Self {
        Self {
            script: Some(std::sync::Mutex::new(responses.into())),
            hang_entered: None,
            hang_dropped: None,
        }
    }

    /// A provider whose `chat()` never resolves. Returns the provider plus
    /// the flags described on [`HangingChat`].
    pub(crate) fn new_hanging() -> (Self, HangingChat) {
        let entered = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let dropped = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        (
            Self {
                script: None,
                hang_entered: Some(Arc::clone(&entered)),
                hang_dropped: Some(Arc::clone(&dropped)),
            },
            HangingChat { entered, dropped },
        )
    }
}

#[cfg(test)]
#[async_trait::async_trait]
impl Provider for StubProvider {
    fn name(&self) -> &str {
        "stub"
    }

    async fn chat(
        &self,
        _system: Option<&str>,
        _messages: &[ChatMessage],
        _tools: Option<&[crate::provider::ToolSpec]>,
    ) -> anyhow::Result<crate::provider::ChatResponse> {
        let Some(script) = &self.script else {
            // Hold a guard whose Drop flips `hang_dropped` before awaiting
            // forever, so a caller that aborts this future (rather than
            // waiting for it) leaves proof behind.
            struct DroppedFlag(Arc<std::sync::atomic::AtomicUsize>);
            impl Drop for DroppedFlag {
                fn drop(&mut self) {
                    self.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                }
            }
            let flag = self
                .hang_dropped
                .clone()
                .expect("hanging StubProvider always carries a drop flag");
            let _guard = DroppedFlag(flag);
            // Announce the guard only once it exists: a test that waits for
            // this before cancelling knows the drop flag has something to
            // fire.
            self.hang_entered
                .as_ref()
                .expect("hanging StubProvider always carries an entry counter")
                .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            std::future::pending::<()>().await;
            unreachable!()
        };
        let next = script.lock().unwrap().pop_front();
        next.ok_or_else(|| anyhow::anyhow!("StubProvider script exhausted"))
    }
}

#[cfg(test)]
impl ServeState {
    /// State backed by temp directories and a stub provider that answers "ok".
    pub(crate) fn for_test(acp_enabled: bool) -> Arc<Self> {
        Self::for_test_scripted(
            acp_enabled,
            vec![crate::provider::ChatResponse {
                text: Some("ok".to_string()),
                tool_calls: Vec::new(),
                stop_reason: None,
            }],
        )
    }

    pub(crate) fn for_test_scripted(
        acp_enabled: bool,
        responses: Vec<crate::provider::ChatResponse>,
    ) -> Arc<Self> {
        Self::build_for_test(acp_enabled, StubProvider::new(responses))
    }

    /// State whose provider never returns, so a turn stays in flight.
    /// The returned [`HangingChat`] reports when that `chat()` call was
    /// entered and when it was dropped.
    pub(crate) fn for_test_hanging(acp_enabled: bool) -> (Arc<Self>, HangingChat) {
        let (provider, hanging) = StubProvider::new_hanging();
        (Self::build_for_test(acp_enabled, provider), hanging)
    }

    fn build_for_test(acp_enabled: bool, provider: StubProvider) -> Arc<Self> {
        // Leak the TempDir guard on purpose: this is a test binary and the
        // OS reclaims the directory when it exits.
        let dir = Box::leak(Box::new(tempfile::tempdir().unwrap()));
        let base = dir.path().to_path_buf();
        let workspace_dir = base.join("workspace");

        // Device auth fixture: one device ("developer-device") routed to
        // room_profile "developer", with its bearer token fixed at
        // "sa-acp-token" so the many tests presenting that literal keep
        // working. Written by hand rather than through
        // `KeyStore::generate` (which always mints a random token) — the
        // key file format documents `token` as its only required field,
        // exactly like this. Mirrors the fixture shape in
        // `device_auth::tests`, minus the random token.
        let keys_file = base.join("keys.toml");
        let devices_file = crate::config::workspace_devices_path(&workspace_dir);
        std::fs::create_dir_all(devices_file.parent().unwrap()).unwrap();
        let mut devices = sapphire_framework::registry::Devices::load(&devices_file).unwrap();
        let device = devices.add("developer-device", None, None).unwrap();
        std::fs::write(
            &keys_file,
            format!(
                "[[key]]\ntoken = \"sa-acp-token\"\nlabel = \"developer-device\"\ndevice_id = \"{}\"\n",
                device.id
            ),
        )
        .unwrap();

        let mut config = Config::parse_for_test(
            r#"
[anthropic]
api_key = "test"

[profiles.dev]
provider = "stub"

[room_profile.developer]
profile  = "dev"
rooms    = []
"#,
        );
        config.acp = Some(crate::config::AcpConfig {
            enabled: acp_enabled,
        });
        config.keys.file = Some(keys_file.clone());
        config.room_profiles.get_mut("developer").unwrap().devices = vec![device.id.to_string()];

        let device_auth = Arc::new(
            crate::device_auth::DeviceAuth::open(&keys_file, &devices_file, &config.room_profiles)
                .expect("test device_auth fixture should build"),
        );

        // Registered under both names: `provider_for_session` falls
        // through to the background provider when no room_profile is
        // pinned (as in the fixture_state_serves_the_scripted_provider
        // test above, which looks up an unpinned session), and the
        // background provider resolves the built-in "anthropic" key.
        let registry = ProviderRegistry::for_test(&["anthropic", "stub"], Arc::new(provider));

        Arc::new(Self {
            config,
            registry: Arc::new(registry),
            workspace: Arc::new(Workspace::new(
                workspace_dir,
                crate::config::DigestConfig::default(),
            )),
            tools: Arc::new(ToolSet::new(vec![Box::new(EchoTool::new())], Vec::new())),
            cross_device_session_store: Arc::new(SessionStore::new(base.join("sessions"), "rpc")),
            device_default_session_store: Arc::new(SessionStore::new(
                base.join("device-default"),
                "device-default",
            )),
            mcp_session_store: Arc::new(SessionStore::new(base.join("mcp"), "mcp")),
            mcp_project_index: Default::default(),
            sessions: Default::default(),
            pending_sessions: Default::default(),
            session_room_profiles: Default::default(),
            session_room_metadata: Default::default(),
            voice: None,
            image_cache: None,
            voice_subscribers: Default::default(),
            device_auth,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::{Role, UserInputKind};

    /// A refused tool still gets a `tool_result`, and the turn carries
    /// on. Refusing must not look to the model like the tool vanished,
    /// and must not end the turn — the model may have another route.
    #[tokio::test]
    async fn a_refused_tool_returns_a_result_and_the_turn_continues() {
        use crate::tools::policy::Origin;

        struct ChannelHost;
        #[async_trait::async_trait]
        impl TurnHost for ChannelHost {
            async fn tool_start(&self, _id: &str, _name: &str) {}
            async fn tool_end(&self, _id: &str, _name: &str) {}
            async fn turn_error(&self, _message: &str) {}
            fn origin(&self) -> Origin {
                Origin::Channel
            }
        }

        // `risky` is Execute, so `Origin::Channel` refuses it. The
        // second scripted response is the model carrying on afterwards.
        let state = ServeState::for_test_scripted(
            true,
            vec![
                crate::provider::ChatResponse {
                    text: None,
                    tool_calls: vec![crate::provider::ToolCall {
                        id: "call-1".to_string(),
                        name: "risky".to_string(),
                        input: json!({}),
                    }],
                    stop_reason: None,
                },
                crate::provider::ChatResponse {
                    text: Some("could not run that".to_string()),
                    tool_calls: Vec::new(),
                    stop_reason: None,
                },
            ],
        );
        let risky = RiskyTool::new();
        let ran = risky.ran_flag();
        state.tools.register_tool(Box::new(risky)).await;

        let outcome = run_llm_turn(
            Arc::clone(&state),
            "s-refused".to_string(),
            ChatMessage::user("run it"),
            Arc::new(ChannelHost),
            None,
        )
        .await;

        assert_eq!(outcome.text.as_deref(), Some("could not run that"));
        assert!(
            !ran.load(std::sync::atomic::Ordering::SeqCst),
            "a refused tool must not have executed"
        );
    }

    /// The same call is allowed once the origin is a trusted one, so
    /// the refusal above is the policy talking and not a broken tool.
    #[tokio::test]
    async fn a_trusted_origin_runs_the_same_tool() {
        let state = ServeState::for_test_scripted(
            true,
            vec![
                crate::provider::ChatResponse {
                    text: None,
                    tool_calls: vec![crate::provider::ToolCall {
                        id: "call-1".to_string(),
                        name: "risky".to_string(),
                        input: json!({}),
                    }],
                    stop_reason: None,
                },
                crate::provider::ChatResponse {
                    text: Some("ran it".to_string()),
                    tool_calls: Vec::new(),
                    stop_reason: None,
                },
            ],
        );
        let risky = RiskyTool::new();
        let ran = risky.ran_flag();
        state.tools.register_tool(Box::new(risky)).await;

        let outcome = run_llm_turn(
            Arc::clone(&state),
            "s-allowed".to_string(),
            ChatMessage::user("run it"),
            Arc::new(NullProgress),
            None,
        )
        .await;

        assert_eq!(outcome.text.as_deref(), Some("ran it"));
        assert!(
            ran.load(std::sync::atomic::Ordering::SeqCst),
            "a trusted origin must have executed it"
        );
    }

    /// The pre-existing transports keep today's behaviour. `Trusted` is
    /// what makes that true: `decide` allows everything for it, so no
    /// `/rpc`, voice or heartbeat turn can start asking for permission.
    #[test]
    fn existing_transports_are_trusted() {
        use crate::tools::policy::Origin;

        let (tx, _rx) = mpsc::channel(4);
        let sse = SseProgress::new(tx, json!(1));
        assert_eq!(sse.origin(), Origin::Trusted);
        assert_eq!(NullProgress.origin(), Origin::Trusted);
    }

    /// A host that cannot ask must not block the call. The default lets
    /// it through, which is what keeps the existing transports behaving
    /// exactly as before.
    #[tokio::test]
    async fn the_default_approval_allows_once() {
        use crate::tools::{ToolKind, policy::Approval};

        let call = crate::provider::ToolCall {
            id: "c1".to_string(),
            name: "shell".to_string(),
            input: json!({}),
        };
        assert_eq!(
            NullProgress.approve(&call, ToolKind::Execute).await,
            Approval::AllowOnce
        );
    }

    #[test]
    fn apply_label_passes_text_through_unchanged() {
        let msg = ChatMessage::user("hello");
        let labeled = apply_input_kind_label(msg.clone());
        assert_eq!(labeled.parts.len(), 1);
        match &labeled.parts[0] {
            ContentPart::Text(s) => assert_eq!(s, "hello"),
            _ => panic!("expected Text part"),
        }
    }

    #[test]
    fn apply_label_passes_none_input_kind_through_unchanged() {
        let msg = ChatMessage {
            role: Role::User,
            parts: vec![ContentPart::Text("hi".to_string())],
            input_kind: None,
            user_id: None,
        };
        let labeled = apply_input_kind_label(msg);
        match &labeled.parts[0] {
            ContentPart::Text(s) => assert_eq!(s, "hi"),
            _ => panic!("expected Text part"),
        }
    }

    #[test]
    fn apply_label_voice_prefixes_first_text_part() {
        let msg = ChatMessage::user_voice("what's the weather");
        let labeled = apply_input_kind_label(msg);
        match &labeled.parts[0] {
            ContentPart::Text(s) => {
                assert!(s.starts_with("[voice input]\n"), "got: {s}");
                assert!(s.ends_with("what's the weather"));
            }
            _ => panic!("expected Text part"),
        }
    }

    #[test]
    fn apply_label_voice_inserts_label_when_no_text_part_present() {
        let msg = ChatMessage {
            role: Role::User,
            parts: vec![ContentPart::Image {
                media_type: "image/png".to_string(),
                data_base64: "AAAA".to_string(),
            }],
            input_kind: Some(UserInputKind::Voice),
            user_id: None,
        };
        let labeled = apply_input_kind_label(msg);
        assert_eq!(labeled.parts.len(), 2);
        match &labeled.parts[0] {
            ContentPart::Text(s) => assert_eq!(s, "[voice input]"),
            _ => panic!("expected inserted Text part"),
        }
        assert!(matches!(labeled.parts[1], ContentPart::Image { .. }));
    }

    #[test]
    fn extract_bearer_accepts_both_cases_and_trims() {
        let mut h = HeaderMap::new();
        h.insert("authorization", HeaderValue::from_static("Bearer  tok-1 "));
        assert_eq!(extract_bearer(&h), Some("tok-1".to_string()));

        let mut h = HeaderMap::new();
        h.insert("authorization", HeaderValue::from_static("bearer tok-2"));
        assert_eq!(extract_bearer(&h), Some("tok-2".to_string()));
    }

    #[test]
    fn extract_bearer_rejects_missing_wrong_scheme_and_empty() {
        assert_eq!(extract_bearer(&HeaderMap::new()), None);

        let mut h = HeaderMap::new();
        h.insert("authorization", HeaderValue::from_static("Basic tok"));
        assert_eq!(extract_bearer(&h), None);

        let mut h = HeaderMap::new();
        h.insert("authorization", HeaderValue::from_static("Bearer   "));
        assert_eq!(extract_bearer(&h), None);
    }

    #[tokio::test]
    async fn fixture_state_serves_the_scripted_provider() {
        let state = ServeState::for_test_scripted(
            true,
            vec![crate::provider::ChatResponse {
                text: Some("scripted reply".to_string()),
                tool_calls: Vec::new(),
                stop_reason: None,
            }],
        );
        let provider = state.provider_for_session("no-such-session").await;
        let resp = provider
            .chat(None, &[ChatMessage::user("hi")], None)
            .await
            .unwrap();
        assert_eq!(resp.text.as_deref(), Some("scripted reply"));
    }

    #[tokio::test]
    async fn fixture_state_resolves_the_test_token() {
        let state = ServeState::for_test(true);
        let resolved = state
            .device_auth
            .resolve("sa-acp-token")
            .expect("fixture token should resolve");
        assert_eq!(resolved.room_profile, "developer");
        assert!(state.device_auth.resolve("sa-wrong").is_none());
    }

    /// Without this check, an authenticated client could pass someone else's
    /// `device_id` in `voice/subscribe` params and displace that device's
    /// push channel — `voice_subscribers` is keyed on `device_id` alone, and
    /// the value came from client-supplied JSON, not from the token.
    #[tokio::test]
    async fn voice_subscribe_rejects_a_device_id_that_disagrees_with_the_token() {
        let state = ServeState::for_test(true);
        let authenticated_device_id = state
            .device_auth
            .resolve("sa-acp-token")
            .expect("fixture token should resolve")
            .device
            .id
            .to_string();

        let response = handle_voice_subscribe(
            Arc::clone(&state),
            Value::Null,
            Some(json!({ "device_id": "not-the-authenticated-device" })),
            "developer".to_string(),
            authenticated_device_id,
        )
        .await;

        let bytes = axum::body::to_bytes(response.into_body(), 64 * 1024)
            .await
            .unwrap();
        let v: Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(v["error"]["code"], -32602);
        assert!(
            v["error"]["message"]
                .as_str()
                .unwrap()
                .contains("does not match"),
            "{v}"
        );
        // Nothing should have been registered under the impersonated id.
        assert!(
            !state
                .voice_subscribers
                .lock()
                .await
                .contains_key("not-the-authenticated-device")
        );
    }

    // `handle_voice_pipeline_run` carries the identical guard (added right
    // after its own `device_id` extraction), but exercising it directly
    // needs a configured `VoiceProviders`, which `ServeState::for_test`
    // deliberately leaves `None` (see its `voice/pipeline_run unavailable`
    // early return) — outside what's worth building a real STT/TTS fixture
    // for here.

    #[tokio::test]
    async fn hanging_provider_sets_the_drop_flag_when_its_chat_future_is_aborted() {
        let (state, hanging) = ServeState::for_test_hanging(true);
        let handle = tokio::spawn(async move {
            let provider = state.provider_for_session("no-such-session").await;
            let _ = provider.chat(None, &[ChatMessage::user("hi")], None).await;
        });
        // Give the spawned task a chance to actually enter `chat()` and
        // start awaiting `pending::<()>()` before we abort it — otherwise
        // we might cancel it before the guard is ever constructed.
        tokio::task::yield_now().await;
        handle.abort();
        let _ = handle.await;
        assert_eq!(
            hanging.entered.load(std::sync::atomic::Ordering::SeqCst),
            1,
            "expected the chat() call to have parked before it was aborted"
        );
        assert_eq!(
            hanging.dropped.load(std::sync::atomic::Ordering::SeqCst),
            1,
            "expected the in-flight chat() future's drop guard to fire on abort"
        );
    }

    #[test]
    fn tool_event_params_pins_the_wire_field_names() {
        assert_eq!(
            tool_event_params("call-1", "recall"),
            json!({ "id": "call-1", "name": "recall" })
        );
    }

    #[tokio::test]
    async fn sse_progress_emits_one_event_per_call() {
        let (tx, mut rx) = mpsc::channel(8);
        let progress = SseProgress::new(tx, json!(7));

        progress.tool_start("call-1", "recall").await;
        progress.tool_end("call-1", "recall").await;
        drop(progress);

        let mut seen = Vec::new();
        while let Some(item) = rx.recv().await {
            seen.push(item);
        }
        assert_eq!(seen.len(), 2, "one event per call");
        assert!(seen[0].is_ok());
        assert!(seen[1].is_ok());
    }

    #[tokio::test]
    async fn sse_progress_turn_error_emits_one_event() {
        let (tx, mut rx) = mpsc::channel(8);
        let progress = SseProgress::new(tx, json!(7));

        progress.turn_error("provider exploded").await;
        drop(progress);

        let mut seen = Vec::new();
        while let Some(item) = rx.recv().await {
            seen.push(item);
        }
        assert_eq!(seen.len(), 1, "one event for the error");
        assert!(seen[0].is_ok());
    }

    #[tokio::test]
    async fn null_progress_discards_without_panicking_and_sends_nothing() {
        // NullProgress holds no channel to observe directly, so the
        // absence of a panic across all three methods is the assertion.
        let progress = NullProgress;
        progress.tool_start("recall", "call-1").await;
        progress.tool_end("recall", "call-1").await;
        progress.turn_error("ignored").await;
    }
}
