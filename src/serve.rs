//! HTTP API server implementing MCP Streamable HTTP transport + JSON-RPC 2.0.
//!
//! Endpoint: POST /mcp  (chat, initialize, list_sessions)
//!           GET  /mcp  (Phase 2: server→client SSE push, currently 405)
//!
//! Session management follows the MCP standard: `Mcp-Session-Id` request header.

use crate::config::Config;
use crate::provider::{ChatMessage, ContentPart, Provider};
use crate::session::{ConversationKey, SessionStore};
use crate::tools::ToolSet;
use crate::workspace::Workspace;
use axum::extract::State;
use axum::http::{HeaderMap, HeaderValue, StatusCode};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::IntoResponse;
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
    provider: Arc<dyn Provider>,
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
    provider: Arc<dyn Provider>,
    workspace: Arc<Workspace>,
    tools: Arc<ToolSet>,
    api_session_store: Arc<SessionStore>,
) -> anyhow::Result<()> {
    let state = Arc::new(ServeState {
        config,
        provider,
        workspace,
        tools,
        api_session_store,
        sessions: tokio::sync::Mutex::new(HashMap::new()),
        pending_sessions: tokio::sync::Mutex::new(HashMap::new()),
    });

    let app = Router::new()
        .route("/mcp", post(mcp_post).get(mcp_get))
        .layer(tower_http::cors::CorsLayer::permissive())
        .with_state(state);

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    info!("sapphire-agent: API server listening on http://{addr}");
    axum::serve(listener, app)
        .with_graceful_shutdown(async {
            if let Err(e) = tokio::signal::ctrl_c().await {
                error!("Failed to install Ctrl-C handler: {e}");
            }
            info!("HTTP server shutting down...");
        })
        .await?;
    Ok(())
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
            Some(ref id) if id.len() == 7 => {
                match state.api_session_store.find_by_public_id(id) {
                    Some(uuid) => Some(uuid),
                    None => {
                        let body = error_response(req_id, -32602, "Session not found");
                        return body.into_response();
                    }
                }
            }
            Some(_) => {
                let body = error_response(req_id, -32602, "Invalid session id (expected 7-char grain-id)");
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

    let mut result = json!({
        "session_id": session_id,
        "is_new": is_new,
    });
    if let Some(ref pub_id) = public_id {
        result["public_id"] = json!(pub_id);
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
// Turn processing (tool-calling loop)
// ---------------------------------------------------------------------------

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
    let is_first_turn = history.is_empty();

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

    // 3. System prompt (rebuilt fresh per request)
    let system = {
        let sp = state
            .workspace
            .build_system_prompt(state.config.anthropic.system_prompt.as_deref())
            .await;
        if sp.is_empty() { None } else { Some(sp) }
    };

    // 4. Append user message
    let user_msg = ChatMessage::user(&user_message);
    history.push(user_msg.clone());
    if let Err(e) = state.api_session_store.append(&session_id, &user_msg) {
        warn!("Failed to persist user message: {e}");
    }

    // 5. Tool-calling loop
    let tool_specs = state.tools.specs().to_vec();
    let mut accumulated_text: Vec<String> = Vec::new();
    let final_text = loop {
        let round = history
            .iter()
            .filter(|m| m.parts.iter().any(|p| matches!(p, ContentPart::ToolUse { .. })))
            .count();

        if round >= MAX_TOOL_ROUNDS {
            warn!("Reached max tool rounds ({MAX_TOOL_ROUNDS})");
            break None;
        }

        let response = state
            .provider
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
                if let Err(e) = state.api_session_store.append(&session_id, &msg) {
                    warn!("Failed to persist tool-call message: {e}");
                }

                // Notify client of each tool starting
                for call in &tool_calls {
                    send(notification_event(
                        "tool_start",
                        json!({ "id": call.id, "name": call.name }),
                    ))
                    .await;
                }

                // Execute all tools concurrently
                let tools = Arc::clone(&state.tools);
                let results: Vec<(String, String)> = futures_util::future::join_all(
                    tool_calls.iter().map(|c| {
                        let tools = Arc::clone(&tools);
                        let c = c.clone();
                        async move {
                            info!("Executing tool: {} (id={})", c.name, c.id);
                            let result = tools.execute(&c).await;
                            info!("Tool {} done", c.name);
                            (c.id, result)
                        }
                    }),
                )
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
                if let Err(e) = state.api_session_store.append(&session_id, &result_msg) {
                    warn!("Failed to persist tool results: {e}");
                }
            }
        }
    };

    // 6. Send final result
    match &final_text {
        Some(text) => {
            send(result_event(&req_id, json!({ "content": text }))).await;
        }
        None => {
            send(error_event(&req_id, -32603, "No response generated")).await;
        }
    }

    // 7. Update in-memory sessions map
    state.sessions.lock().await.insert(session_id.clone(), history);

    // 8. Generate and store session title after the first successful turn
    if is_first_turn {
        if let Some(text) = final_text {
            let state2 = Arc::clone(&state);
            let sid = session_id.clone();
            let user_msg = user_message.clone();
            tokio::spawn(async move {
                if let Some(title) = generate_session_title(&*state2.provider, &user_msg, &text).await {
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
