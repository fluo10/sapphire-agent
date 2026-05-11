//! HTTP API server implementing MCP Streamable HTTP transport + JSON-RPC 2.0.
//!
//! Endpoint: POST /mcp  (chat, initialize, list_sessions)
//!           GET  /mcp  (Phase 2: server→client SSE push, currently 405)
//!
//! Session management follows the MCP standard: `Mcp-Session-Id` request header.

use crate::config::Config;
use crate::context_compression::{generate_summary, maybe_compress};
use crate::provider::registry::ProviderRegistry;
use crate::provider::{ChatMessage, ContentPart, Provider};
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

pub struct ServeState {
    config: Config,
    registry: Arc<ProviderRegistry>,
    workspace: Arc<Workspace>,
    tools: Arc<ToolSet>,
    api_session_store: Arc<SessionStore>,
    /// In-memory conversation history, keyed by session_id.
    /// Lazy-loaded from JSONL on first access.
    sessions: tokio::sync::Mutex<HashMap<String, Vec<ChatMessage>>>,
    /// Sessions that have been issued an ID via `initialize` but have not yet
    /// received a message — file creation is deferred until the first chat so
    /// that quitting without sending anything leaves no empty file behind.
    /// Maps internal UUID → reserved public_id (grain-id).
    pending_sessions: tokio::sync::Mutex<HashMap<String, String>>,
    /// Per-session room_profile pin from `initialize`. Sessions absent
    /// from this map fall through to the background provider. Not
    /// persisted across restarts — clients must re-pass `room_profile`
    /// on resume.
    session_room_profiles: tokio::sync::Mutex<HashMap<String, String>>,
    /// Voice provider registry. `None` when no `[stt_provider.*]` /
    /// `[tts_provider.*]` blocks are configured — in that case the
    /// `voice/pipeline_run` method returns a method-not-available error.
    voice: Option<Arc<VoiceProviders>>,
}

impl ServeState {
    /// Provider that should serve the given session. Resolves the
    /// session's pinned room_profile to its `profile`, then to a
    /// concrete provider (with optional refusal fallback). Falls back
    /// to the background provider when no room_profile is pinned.
    async fn provider_for_session(&self, session_id: &str) -> Arc<dyn Provider> {
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

pub async fn run(
    addr: String,
    config: Config,
    registry: Arc<ProviderRegistry>,
    workspace: Arc<Workspace>,
    tools: Arc<ToolSet>,
    api_session_store: Arc<SessionStore>,
    voice: Option<Arc<VoiceProviders>>,
) -> anyhow::Result<()> {
    let state = Arc::new(ServeState {
        config,
        registry,
        workspace,
        tools,
        api_session_store,
        sessions: tokio::sync::Mutex::new(HashMap::new()),
        pending_sessions: tokio::sync::Mutex::new(HashMap::new()),
        session_room_profiles: tokio::sync::Mutex::new(HashMap::new()),
        voice,
    });

    let app = Router::new()
        .route("/mcp", post(mcp_post).get(mcp_get))
        .layer(tower_http::cors::CorsLayer::permissive())
        .with_state(Arc::clone(&state));

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
        "Graceful shutdown: summarizing {} API session(s)",
        snapshot.len()
    );
    for (session_id, messages) in snapshot {
        let provider = state.provider_for_session(&session_id).await;
        match generate_summary(&*provider, &messages).await {
            Ok(summary) if !summary.trim().is_empty() => {
                if let Err(e) = state
                    .api_session_store
                    .append_summary(&session_id, &summary)
                {
                    warn!("Failed to persist shutdown summary for {session_id}: {e}");
                }
            }
            Ok(_) => warn!("Shutdown summary for {session_id} was empty; skipping"),
            Err(e) => warn!("Shutdown summary generation failed for {session_id}: {e:#}"),
        }
    }
}

// ---------------------------------------------------------------------------
// POST /mcp  — dispatch JSON-RPC methods
// ---------------------------------------------------------------------------

async fn mcp_post(
    State(state): State<Arc<ServeState>>,
    headers: HeaderMap,
    Json(req): Json<JsonRpcRequest>,
) -> impl IntoResponse {
    let session_id = headers
        .get("mcp-session-id")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string());

    let req_id = req.id.clone().unwrap_or(Value::Null);

    match req.method.as_str() {
        "initialize" => handle_initialize(state, req_id, req.params, session_id).await,
        "chat" => handle_chat(state, req_id, req.params, session_id).await,
        "list_sessions" => handle_list_sessions(state, req_id).await,
        "get_session" => handle_get_session(state, req_id, session_id).await,
        "voice/pipeline_run" => {
            handle_voice_pipeline_run(state, req_id, req.params, session_id).await
        }
        _ => {
            let body = error_response(req_id, -32601, "Method not found");
            body.into_response()
        }
    }
}

// ---------------------------------------------------------------------------
// GET /mcp  — Phase 2 placeholder (server→client push)
// ---------------------------------------------------------------------------

async fn mcp_get() -> impl IntoResponse {
    (
        StatusCode::METHOD_NOT_ALLOWED,
        "GET /mcp is reserved for Phase 2 server-initiated tool requests",
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
) -> axum::response::Response {
    // Optional `params.room_profile` — must reference a defined
    // `[room_profile.<n>]` entry. Rejected eagerly so misspellings
    // surface at session start rather than as a silent fallback to
    // the background provider.
    let requested_room_profile: Option<String> = params
        .as_ref()
        .and_then(|p| p["room_profile"].as_str())
        .map(|s| s.to_string());
    if let Some(name) = &requested_room_profile {
        if state.config.room_profile(name).is_none() {
            let body =
                error_response(req_id, -32602, &format!("Unknown room_profile '{name}'"));
            return body.into_response();
        }
    }

    // Resolve to an internal UUID session_id.
    // - Mcp-Session-Id header: already a UUID (internal), use directly.
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
            Some(ref id) if id.len() == 7 => match state.api_session_store.find_by_public_id(id) {
                Some(uuid) => Some(uuid),
                None => {
                    let body = error_response(req_id, -32602, "Session not found");
                    return body.into_response();
                }
            },
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
            let exists = state.api_session_store.load_session(&id).is_some();
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
                .api_session_store
                .load_session(&session_id)
                .unwrap_or_default()
        });
        // Look up the public_id from the existing file metadata.
        state
            .api_session_store
            .list_sessions()
            .into_iter()
            .find(|m| m.session_id == session_id)
            .and_then(|m| m.public_id)
    };

    if let Some(name) = requested_room_profile {
        state
            .session_room_profiles
            .lock()
            .await
            .insert(session_id.clone(), name);
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
            "mcp-session-id",
            HeaderValue::from_str(&session_id).unwrap_or_else(|_| HeaderValue::from_static("")),
        )
        .body(axum::body::Body::from(body.to_string()))
        .unwrap();

    // Also set Mcp-Session-Id in the response headers (accessible via response)
    response.headers_mut().insert(
        "mcp-session-id",
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
            let body = error_response(req_id, -32602, "Missing Mcp-Session-Id header");
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

    let (tx, rx) = mpsc::channel::<Result<Event, Infallible>>(32);

    // Spawn the turn processor
    tokio::spawn(async move {
        run_turn(state, session_id, content, req_id, tx).await;
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
    let metas = state.api_session_store.list_sessions();
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
            let body = error_response(req_id, -32602, "Missing Mcp-Session-Id header");
            return body.into_response();
        }
    };

    let messages = state
        .api_session_store
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
                        // Image bytes are not exposed via the API listing; surface a marker only.
                        json!({ "type": "image", "media_type": media_type })
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

async fn handle_voice_pipeline_run(
    state: Arc<ServeState>,
    req_id: Value,
    params: Option<Value>,
    session_id: Option<String>,
) -> axum::response::Response {
    let session_id = match session_id {
        Some(id) => id,
        None => {
            let body = error_response(req_id, -32602, "Missing Mcp-Session-Id header");
            return body.into_response();
        }
    };

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
    let language = params["language"].as_str().map(|s| s.to_string());

    let (tx, rx) = mpsc::channel::<Result<Event, Infallible>>(64);

    tokio::spawn(async move {
        run_voice_turn(state, session_id, audio_b64, language, req_id, tx).await;
    });

    let stream = ReceiverStream::new(rx);
    Sse::new(stream)
        .keep_alive(KeepAlive::new().interval(std::time::Duration::from_secs(15)))
        .into_response()
}

async fn run_voice_turn(
    state: Arc<ServeState>,
    session_id: String,
    audio_b64: String,
    language: Option<String>,
    req_id: Value,
    tx: mpsc::Sender<Result<Event, Infallible>>,
) {
    use base64::Engine;

    let send = |evt: Event| {
        let tx = tx.clone();
        async move {
            let _ = tx.send(Ok(evt)).await;
        }
    };

    // Resolve voice providers from the session's pinned room_profile.
    let voice_registry = match state.voice.as_ref() {
        Some(v) => Arc::clone(v),
        None => {
            send(error_event(
                &req_id,
                -32601,
                "voice/pipeline_run unavailable: no STT/TTS providers configured",
            ))
            .await;
            return;
        }
    };

    let rp_name = state
        .session_room_profiles
        .lock()
        .await
        .get(&session_id)
        .cloned();
    let pipeline = rp_name
        .as_deref()
        .and_then(|n| state.config.voice_pipeline_for_room_profile(n));
    let pipeline = match pipeline {
        Some(p) => p.clone(),
        None => {
            send(error_event(
                &req_id,
                -32602,
                "Session's room_profile has no voice_pipeline configured",
            ))
            .await;
            return;
        }
    };
    let stt = match voice_registry.stt(&pipeline.stt_provider) {
        Some(p) => p,
        None => {
            send(error_event(
                &req_id,
                -32603,
                &format!(
                    "stt_provider '{}' not instantiated",
                    pipeline.stt_provider
                ),
            ))
            .await;
            return;
        }
    };
    let tts = match voice_registry.tts(&pipeline.tts_provider) {
        Some(p) => p,
        None => {
            send(error_event(
                &req_id,
                -32603,
                &format!(
                    "tts_provider '{}' not instantiated",
                    pipeline.tts_provider
                ),
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

    // Stage: LLM (intent)
    send(notification_event(
        "notifications/progress",
        json!({"kind": "stage", "stage": "intent", "status": "start"}),
    ))
    .await;
    let outcome = run_llm_turn(
        Arc::clone(&state),
        session_id.clone(),
        transcript.clone(),
        req_id.clone(),
        tx.clone(),
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
    let synth_handle = tokio::spawn(async move {
        let res = tts.synthesize_stream(&reply_for_tts, pcm_tx).await;
        if let Err(e) = res {
            warn!("TTS synthesis error: {e:#}");
        }
    });
    while let Some(chunk) = pcm_rx.recv().await {
        let bytes: Vec<u8> = chunk
            .iter()
            .flat_map(|s| s.to_le_bytes())
            .collect();
        let b64 = base64::engine::general_purpose::STANDARD.encode(&bytes);
        send(notification_event(
            "notifications/progress",
            json!({"kind": "audio_chunk", "data": b64}),
        ))
        .await;
    }
    let _ = synth_handle.await;
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
            "transcript": transcript,
            "assistant_text": reply_text,
        }),
    ))
    .await;

    // Title generation on first turn — same as run_turn.
    if outcome.was_first_turn {
        let state2 = Arc::clone(&state);
        let sid = session_id.clone();
        let user_msg = transcript.clone();
        let reply = reply_text.clone();
        tokio::spawn(async move {
            let p = state2.provider_for_session(&sid).await;
            if let Some(title) = generate_session_title(&*p, &user_msg, &reply).await {
                if let Err(e) = state2.api_session_store.set_title(&sid, &title) {
                    warn!("Failed to store session title: {e}");
                }
            }
        });
    }
}

// ---------------------------------------------------------------------------
// Turn processing (tool-calling loop)
// ---------------------------------------------------------------------------

/// Outcome of [`run_llm_turn`].
struct LlmTurnOutcome {
    /// Final assistant text, when the turn completed successfully. `None`
    /// on provider error or when MAX_TOOL_ROUNDS was hit without resolving.
    text: Option<String>,
    /// True iff the session had no prior turns before this one. Used by
    /// callers to decide whether to spawn a title-generation task.
    was_first_turn: bool,
}

/// Execute one full LLM turn for an established session: hydrate history,
/// run the tool-calling loop, persist user + assistant messages to JSONL,
/// and emit per-tool `tool_start` / `tool_end` SSE notifications. Does NOT
/// send the final JSON-RPC result event — the caller is responsible for
/// shaping the final payload (text reply, voice audio, etc.) and emitting
/// the appropriate result event.
async fn run_llm_turn(
    state: Arc<ServeState>,
    session_id: String,
    user_message: String,
    req_id: Value,
    tx: mpsc::Sender<Result<Event, Infallible>>,
) -> LlmTurnOutcome {
    let send = |evt: Event| {
        let tx = tx.clone();
        async move {
            let _ = tx.send(Ok(evt)).await;
        }
    };

    // 1. Load or lazy-hydrate in-memory history
    let mut history: Vec<ChatMessage> = {
        let mut sessions = state.sessions.lock().await;
        sessions
            .entry(session_id.clone())
            .or_insert_with(|| {
                state
                    .api_session_store
                    .load_session(&session_id)
                    .unwrap_or_default()
            })
            .clone()
    };
    let was_first_turn = history.is_empty();

    // 2. Ensure JSONL file exists. If this session was deferred at initialize
    //    time, commit it now using the reserved public_id.
    let key: ConversationKey = (session_id.clone(), None);
    let pending_pub_id = state.pending_sessions.lock().await.remove(&session_id);
    if let Err(e) = state
        .api_session_store
        .ensure_session(&session_id, &key, "api", pending_pub_id)
        .map(|_| ())
    {
        warn!("Failed to ensure session file: {e}");
    }

    // 3. Resolve provider once per turn — sessions can pin a profile at
    //    initialize-time; absent that, the background provider is used.
    let provider = state.provider_for_session(&session_id).await;

    // 3a. System prompt (rebuilt fresh per request). Namespace chain
    //     follows the session's pinned room_profile when set; otherwise
    //     the implicit default namespace.
    let namespace = match state.session_room_profiles.lock().await.get(&session_id) {
        Some(rp_name) => state.config.namespace_for_room_profile(rp_name).to_string(),
        None => crate::config::DEFAULT_NAMESPACE_NAME.to_string(),
    };
    let namespace_chain = state.config.resolve_namespace_chain(&namespace);
    let system = {
        let sp = state
            .workspace
            .build_system_prompt(
                state.config.anthropic.system_prompt.as_deref(),
                state.config.day_boundary_hour,
                &namespace_chain,
            )
            .await;
        if sp.is_empty() { None } else { Some(sp) }
    };

    // 4. Append user message
    let user_msg = ChatMessage::user(&user_message);
    history.push(user_msg.clone());
    if let Err(e) = state.api_session_store.append(&session_id, &user_msg) {
        warn!("Failed to persist user message: {e}");
    }

    // 5. Tool-calling loop — refresh MCP tools if any server signalled a change.
    state.tools.refresh_if_needed().await;
    let tool_specs = state.tools.specs().await;
    let compression_config = &state.config.compression;
    let mut accumulated_text: Vec<String> = Vec::new();
    let final_text = loop {
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
            break None;
        }

        // Check if context compression is needed
        match maybe_compress(
            &*provider,
            system.as_deref(),
            &history,
            &compression_config,
        )
        .await
        {
            Ok(Some(result)) => {
                history = result.compressed;
                if let Err(e) = state
                    .api_session_store
                    .append_summary(&session_id, &result.summary)
                {
                    warn!("Failed to persist compaction summary: {e}");
                }
            }
            Ok(None) => {}
            Err(e) => {
                warn!("Context compression failed, continuing with full history: {e}");
            }
        }

        let response = provider
            .chat(system.as_deref(), &history, Some(&tool_specs))
            .await;

        match response {
            Err(e) => {
                error!("Provider error: {e:#}");
                send(error_event(&req_id, -32603, &e.to_string())).await;
                break None;
            }
            Ok(resp) if !resp.has_tool_calls() => {
                let text = resp.text.unwrap_or_default();
                let msg = ChatMessage::assistant(&text);
                history.push(msg.clone());
                if let Err(e) = state.api_session_store.append(&session_id, &msg) {
                    warn!("Failed to persist assistant message: {e}");
                }
                if !text.is_empty() {
                    accumulated_text.push(text);
                }
                break Some(accumulated_text.join("\n\n"));
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
                    send(notification_event(
                        "tool_start",
                        json!({ "id": call.id, "name": call.name }),
                    ))
                    .await;
                }

                // Execute all tools concurrently — each call wrapped in
                // the session's memory namespace (task_local) so the
                // memory tool writes under `memory/<namespace>/...`.
                let tools = Arc::clone(&state.tools);
                let ns = namespace.clone();
                let results: Vec<(String, String)> =
                    futures_util::future::join_all(tool_calls.iter().map(|c| {
                        let tools = Arc::clone(&tools);
                        let c = c.clone();
                        let ns = ns.clone();
                        crate::tools::workspace_tools::scope_memory_namespace(ns, async move {
                            info!("Executing tool: {} (id={})", c.name, c.id);
                            let result = tools.execute(&c).await;
                            info!("Tool {} done", c.name);
                            (c.id, result)
                        })
                    }))
                    .await;

                // Notify client of each tool completing
                for call in &tool_calls {
                    send(notification_event(
                        "tool_end",
                        json!({ "id": call.id, "name": call.name }),
                    ))
                    .await;
                }

                let result_msg = ChatMessage::tool_results(results);
                history.push(result_msg.clone());
                // Tool_result payloads are not persisted — see the matching
                // tool_use branch above for rationale.
            }
        }
    };

    // Update in-memory sessions map
    state
        .sessions
        .lock()
        .await
        .insert(session_id.clone(), history);

    LlmTurnOutcome {
        text: final_text,
        was_first_turn,
    }
}

async fn run_turn(
    state: Arc<ServeState>,
    session_id: String,
    user_message: String,
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
        user_message.clone(),
        req_id.clone(),
        tx.clone(),
    )
    .await;

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
    if outcome.was_first_turn {
        if let Some(text) = outcome.text {
            let state2 = Arc::clone(&state);
            let sid = session_id.clone();
            let user_msg = user_message.clone();
            tokio::spawn(async move {
                let p = state2.provider_for_session(&sid).await;
                if let Some(title) = generate_session_title(&*p, &user_msg, &text).await {
                    if let Err(e) = state2.api_session_store.set_title(&sid, &title) {
                        warn!("Failed to store session title: {e}");
                    }
                }
            });
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
    let user_snippet = &user_message[..user_message.len().min(300)];
    let asst_snippet = &assistant_response[..assistant_response.len().min(300)];
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
