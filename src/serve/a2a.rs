//! A2A (Agent2Agent Protocol) v1 server endpoints.
//!
//! Exposes:
//!   - `POST /a2a` — JSON-RPC 2.0 dispatch. Currently only `SendMessage`
//!     is implemented (synchronous: receive a `Message`, drive the
//!     internal chat turn to completion, return a terminal `Task`).
//!   - `GET /.well-known/agent-card.json` — A2A agent card discovery.
//!
//! Auth: `Authorization: Bearer <token>` matched against
//! `[room_profile.<n>].api_keys`. The match resolves the request's
//! `room_profile` implicitly — clients don't need to (and can't) name
//! it. A token leak therefore exposes one profile, not all.
//!
//! Out of scope (v1): `SendStreamingMessage` (SSE), `GetTask`,
//! `CancelTask`, `SubscribeToTask`, push notifications, `FilePart`
//! (vision). Wire-format types come from `a2a-lf`; the JSON-RPC
//! dispatch is hand-rolled here to share `ServeState` with the
//! existing `/rpc` endpoint.

use std::convert::Infallible;
use std::sync::Arc;

use a2a::{
    Message, Part, PartContent, Role, SendMessageRequest, SendMessageResponse, Task, TaskState,
    TaskStatus, new_task_id,
};
use axum::extract::State;
use axum::http::{HeaderMap, StatusCode};
use axum::response::IntoResponse;
use axum::{Json, response::sse::Event};
use chrono::Utc;
use serde_json::{Value, json};
use tokio::sync::mpsc;
use tracing::warn;

use super::ServeState;

/// Method name for the only A2A method we implement. The A2A v1 spec
/// uses PascalCase JSON-RPC method names (`SendMessage`, `GetTask`,
/// …), unlike the v0.3 slash form (`message/send`).
const METHOD_SEND_MESSAGE: &str = "SendMessage";

/// JSON-RPC error codes used by this handler. Standard 2.0 codes plus
/// the A2A-spec-aligned application range (-32000 … -32099 reserved).
mod codes {
    pub const PARSE_ERROR: i32 = -32700;
    pub const INVALID_REQUEST: i32 = -32600;
    pub const METHOD_NOT_FOUND: i32 = -32601;
    pub const INVALID_PARAMS: i32 = -32602;
    pub const INTERNAL_ERROR: i32 = -32603;
    /// A2A "authenticated request required" application-level code.
    pub const AUTH_REQUIRED: i32 = -32001;
}

// ---------------------------------------------------------------------------
// Agent Card
// ---------------------------------------------------------------------------

/// `GET /.well-known/agent-card.json` — A2A discovery endpoint.
///
/// Returns 404 when A2A is disabled so probes don't reveal that the
/// server exists at all when the operator hasn't opted in.
pub async fn handle_agent_card(State(state): State<Arc<ServeState>>) -> impl IntoResponse {
    let cfg = match state.config.a2a.as_ref() {
        Some(cfg) if cfg.enabled => cfg,
        _ => return (StatusCode::NOT_FOUND, "A2A disabled").into_response(),
    };

    let name = cfg
        .agent_name
        .clone()
        .unwrap_or_else(|| "sapphire-agent".to_string());
    let description = cfg.agent_description.clone().unwrap_or_else(|| {
        "Personal partner AI agent with persistent character, memory, and tools."
            .to_string()
    });
    let url = cfg.public_url.clone().unwrap_or_default();

    let card = json!({
        "name": name,
        "description": description,
        "version": env!("CARGO_PKG_VERSION"),
        "supportedInterfaces": [
            {
                "url": url,
                "protocolBinding": "jsonrpc",
                "protocolVersion": a2a::VERSION,
            }
        ],
        "capabilities": {
            "streaming": false,
            "pushNotifications": false,
            "extendedAgentCard": false,
        },
        "defaultInputModes": ["text/plain"],
        "defaultOutputModes": ["text/plain"],
        "skills": [
            {
                "id": "chat",
                "name": "Chat with the agent",
                "description":
                    "Hold a multi-turn conversation; the agent remembers context across calls \
                     within the same contextId and applies its server-side persona / memory.",
                "tags": ["chat", "conversation"],
                "inputModes": ["text/plain"],
                "outputModes": ["text/plain"]
            }
        ],
        "securitySchemes": {
            "bearer": {
                "httpAuthSecurityScheme": {
                    "scheme": "Bearer"
                }
            }
        },
        "securityRequirements": [
            { "bearer": [] }
        ]
    });

    (StatusCode::OK, Json(card)).into_response()
}

// ---------------------------------------------------------------------------
// POST /a2a — JSON-RPC 2.0 dispatch
// ---------------------------------------------------------------------------

/// JSON-RPC 2.0 envelope. Kept local so we don't depend on `a2a::JsonRpcId`
/// (which would force us to convert IDs in/out of the existing
/// serde_json `Value` flow).
#[derive(Debug, serde::Deserialize)]
struct JsonRpcEnvelope {
    #[serde(default)]
    jsonrpc: Option<String>,
    #[serde(default)]
    id: Option<Value>,
    method: String,
    #[serde(default)]
    params: Option<Value>,
}

pub async fn handle_a2a_post(
    State(state): State<Arc<ServeState>>,
    headers: HeaderMap,
    body: axum::body::Bytes,
) -> axum::response::Response {
    // 0. Feature gate
    let enabled = state
        .config
        .a2a
        .as_ref()
        .is_some_and(|c| c.enabled);
    if !enabled {
        return (StatusCode::NOT_FOUND, "A2A disabled").into_response();
    }

    // 1. Parse envelope. We parse manually so a malformed body returns
    //    a JSON-RPC parse-error response rather than a 400 — A2A
    //    clients expect a JSON-RPC envelope on every reply.
    let envelope: JsonRpcEnvelope = match serde_json::from_slice(&body) {
        Ok(env) => env,
        Err(e) => {
            return jsonrpc_error_response(
                Value::Null,
                codes::PARSE_ERROR,
                &format!("invalid JSON: {e}"),
            );
        }
    };
    let req_id = envelope.id.clone().unwrap_or(Value::Null);
    if envelope.jsonrpc.as_deref() != Some("2.0") {
        return jsonrpc_error_response(
            req_id,
            codes::INVALID_REQUEST,
            "jsonrpc field must be \"2.0\"",
        );
    }

    // 2. Bearer auth → room_profile reverse lookup. Empty/missing bearer
    //    returns 401 at the HTTP layer (no JSON-RPC envelope) per A2A
    //    spec: auth failures are an HTTP-level concern, not an
    //    application error.
    let bearer = match extract_bearer(&headers) {
        Some(b) => b,
        None => {
            return (StatusCode::UNAUTHORIZED, "missing bearer token").into_response();
        }
    };
    let profile_name = match state.config.resolve_a2a_token(&bearer) {
        Some(name) => name.to_string(),
        None => {
            return jsonrpc_error_response(
                req_id,
                codes::AUTH_REQUIRED,
                "unknown or revoked bearer token",
            );
        }
    };

    // 3. Method dispatch
    match envelope.method.as_str() {
        METHOD_SEND_MESSAGE => {
            handle_send_message(state, req_id, envelope.params, profile_name).await
        }
        other => jsonrpc_error_response(
            req_id,
            codes::METHOD_NOT_FOUND,
            &format!("A2A method '{other}' is not implemented"),
        ),
    }
}

// ---------------------------------------------------------------------------
// SendMessage
// ---------------------------------------------------------------------------

async fn handle_send_message(
    state: Arc<ServeState>,
    req_id: Value,
    params: Option<Value>,
    profile_name: String,
) -> axum::response::Response {
    let raw = params.unwrap_or(Value::Null);
    let request: SendMessageRequest = match serde_json::from_value(raw) {
        Ok(r) => r,
        Err(e) => {
            return jsonrpc_error_response(
                req_id,
                codes::INVALID_PARAMS,
                &format!("malformed SendMessageRequest: {e}"),
            );
        }
    };

    // Extract plain-text content from message parts. v1 of this handler
    // ignores non-text parts (FilePart vision, DataPart structured input);
    // a future change will route those into the multimodal provider path.
    let user_text = collect_text(&request.message.parts);
    if user_text.trim().is_empty() {
        return jsonrpc_error_response(
            req_id,
            codes::INVALID_PARAMS,
            "message must contain at least one non-empty text part",
        );
    }

    // contextId: client-supplied (resume) or server-generated (new). The
    // params-level field wins over message-level if both are present —
    // A2A spec lets either carry it.
    let context_id = request
        .message
        .context_id
        .clone()
        .filter(|s| !s.is_empty())
        .unwrap_or_else(a2a::new_context_id);

    // Internal session id: deterministic from (profile, contextId) so a
    // crash-restart with the same contextId resumes the same JSONL
    // session. Prefix `a2a-` makes the kind self-documenting in logs;
    // hyphens (not colons) keep the resulting `<session_id>.jsonl`
    // filename portable to Windows.
    let session_id = format!("a2a-{profile_name}-{context_id}");

    // Pin the profile so `run_llm_turn`'s provider/namespace resolution
    // picks up the right room_profile.
    state
        .session_room_profiles
        .lock()
        .await
        .insert(session_id.clone(), profile_name.clone());

    // run_llm_turn streams notifications (tool_start/tool_end, errors)
    // through this mpsc; we drain into oblivion since A2A v1 doesn't
    // surface intermediate progress to the caller. A bounded buffer
    // keeps the producer from blocking if we drain slowly.
    let (tx, mut rx) = mpsc::channel::<Result<Event, Infallible>>(32);
    let drain = tokio::spawn(async move {
        while rx.recv().await.is_some() {
            // discard — A2A v1 is synchronous, no SSE relay
        }
    });

    let outcome = super::run_llm_turn(
        Arc::clone(&state),
        session_id.clone(),
        user_text,
        req_id.clone(),
        tx,
    )
    .await;
    drop(drain); // sender dropped at end of run_llm_turn, drain returns

    // Build the terminal Task. Failed turns surface as TASK_STATE_FAILED
    // with a diagnostic message part, matching what A2A clients expect.
    let (state_enum, reply_text) = match outcome.text {
        Some(t) if !t.is_empty() => (TaskState::Completed, t),
        Some(_) => (TaskState::Failed, "(empty response)".to_string()),
        None => (TaskState::Failed, "agent failed to generate a reply".to_string()),
    };

    let reply_message = Message {
        message_id: a2a::new_message_id(),
        context_id: Some(context_id.clone()),
        task_id: None,
        role: Role::Agent,
        parts: vec![Part::text(reply_text)],
        metadata: None,
        extensions: None,
        reference_task_ids: None,
    };

    let task = Task {
        id: new_task_id(),
        context_id,
        status: TaskStatus {
            state: state_enum,
            message: Some(reply_message),
            timestamp: Some(Utc::now()),
        },
        artifacts: None,
        history: None,
        metadata: None,
    };

    let response = SendMessageResponse::Task(task);
    let result = match serde_json::to_value(&response) {
        Ok(v) => v,
        Err(e) => {
            return jsonrpc_error_response(
                req_id,
                codes::INTERNAL_ERROR,
                &format!("failed to serialize Task: {e}"),
            );
        }
    };

    let body = json!({
        "jsonrpc": "2.0",
        "id": req_id,
        "result": result,
    });
    (StatusCode::OK, Json(body)).into_response()
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn extract_bearer(headers: &HeaderMap) -> Option<String> {
    let value = headers.get(axum::http::header::AUTHORIZATION)?;
    let s = value.to_str().ok()?;
    let token = s.strip_prefix("Bearer ").or_else(|| s.strip_prefix("bearer "))?;
    let trimmed = token.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

fn collect_text(parts: &[Part]) -> String {
    let mut out = String::new();
    for p in parts {
        if let PartContent::Text(s) = &p.content {
            if !out.is_empty() {
                out.push('\n');
            }
            out.push_str(s);
        } else {
            // Non-text part observed but unsupported; flag in logs so
            // operators noticing missing context in replies have a
            // pointer rather than having to deserialize the request.
            warn!("a2a: ignoring non-text Part (vision/data unsupported in v1)");
        }
    }
    out
}

fn jsonrpc_error_response(id: Value, code: i32, message: &str) -> axum::response::Response {
    let body = json!({
        "jsonrpc": "2.0",
        "id": id,
        "error": { "code": code, "message": message },
    });
    (StatusCode::OK, Json(body)).into_response()
}
