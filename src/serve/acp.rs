//! Agent Client Protocol over WebSocket, at `GET /acp`.
//!
//! Auth: `Authorization: Bearer <token>` on the upgrade request, resolved
//! through `DeviceAuth` — the same mechanism as `/a2a` and `/mcp`. The
//! match resolves the room profile the ACP session runs under, so a
//! dedicated Zed token can pin the editor to its own profile, provider
//! and memory namespace.
//!
//! Rejection happens *before* the 101: an error delivered after a successful
//! upgrade reaches the operator as an unexplained disconnect, whereas a
//! status code reaches them through `websocat`.
//!
//! Framing follows the ACP transport RFD: one JSON-RPC message per WebSocket
//! text frame. That is exactly the newline framing the SDK's `Lines`
//! transport expects, so [`lines_transport`] adapts the socket without
//! reframing anything.
//!
//! `initialize`, `session/new`, `session/prompt` and `session/cancel` are
//! answered here.
//!
//! A prompt is not implemented here beyond its ACP shape: it extracts the
//! text, hands the turn to [`super::run_llm_turn`] — the same executor
//! behind `/rpc`, voice and A2A — and translates that turn's progress back
//! into `session/update` notifications. There is deliberately no second
//! tool loop, history handling or persistence on this path, so an editor's
//! conversation lands in the same session store, under the same memory
//! namespace and system prompt, as every other transport's.
//!
//! A turn does *not* run in the handler that received it. The SDK's request
//! callbacks hold its dispatch loop, which parses no further frame on the
//! connection until they return, so awaiting a turn there would make the
//! `session/cancel` meant to stop it unreadable until the turn had already
//! finished. Instead the handler spawns the turn with `ConnectionTo::spawn`
//! and sends the `Responder` along with it — see the `session/prompt`
//! handler. The visible consequence is that prompts on one connection run
//! concurrently rather than one after another.
//!
//! Two caveats come with that, both about history, and both worth knowing
//! before building on this:
//!
//! - **Concurrent prompts on *one session* are not history-safe.**
//!   [`super::run_llm_turn`] clones the session's history at the top and
//!   writes the whole vector back at the end, so two overlapping turns on
//!   the same session are last-writer-wins in memory — while both have
//!   already appended their user messages to JSONL. The durable transcript
//!   and the in-memory history diverge. The race is not ACP's: `/rpc` and
//!   the voice heartbeat path can already reach it. What is new here is
//!   that concurrency became *documented* rather than accidental, so this
//!   says plainly that concurrent prompts on separate sessions are fine and
//!   concurrent prompts on one session are not.
//! - **A cancelled turn's future is dropped**, and `run_llm_turn` was not
//!   written to be. Its user message is already on disk; its write-back to
//!   `state.sessions` never runs. So a cancelled prompt is missing from the
//!   next turn's model context but present in `list_sessions` and after a
//!   restart. See the drop note on [`super::run_llm_turn`] itself.

use super::{ServeState, extract_bearer};
use agent_client_protocol::schema::ProtocolVersion;
use agent_client_protocol::schema::v1::{
    AgentCapabilities, CancelNotification, ContentBlock, ContentChunk, CurrentModeUpdate, Error,
    InitializeRequest, InitializeResponse, ListSessionsRequest, ListSessionsResponse,
    LoadSessionRequest, LoadSessionResponse, NewSessionRequest, NewSessionResponse,
    PermissionOption, PermissionOptionKind, PromptRequest, PromptResponse,
    RequestPermissionOutcome, RequestPermissionRequest, ResumeSessionRequest,
    ResumeSessionResponse, SessionCapabilities, SessionId, SessionInfo, SessionListCapabilities,
    SessionMode as AcpSessionMode, SessionModeState, SessionNotification,
    SessionResumeCapabilities, SessionUpdate, SetSessionModeRequest, SetSessionModeResponse,
    StopReason, TextContent, ToolCall as AcpToolCall, ToolCallId, ToolCallStatus, ToolCallUpdate,
    ToolCallUpdateFields,
};
use agent_client_protocol::{
    Agent, Client, ConnectionTo, Lines, on_receive_notification, on_receive_request,
};
use axum::extract::State;
use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use futures_util::{SinkExt, StreamExt};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;
use tokio_util::sync::CancellationToken;
use tracing::{info, warn};

/// One ACP session, mapped onto an agent session.
struct AcpSession {
    /// The agent-side session id, minted the same way `handle_initialize`
    /// (`src/serve/mod.rs`) mints one for a brand-new `/rpc` session:
    /// `uuid::Uuid::now_v7().to_string()`. This is also the key under which
    /// `state.session_room_profiles` pins the session to a room profile, so
    /// it is what routes this ACP session through the same namespace chain
    /// and provider selection every other transport uses. It duplicates the
    /// map key (a `SessionId` wrapping this same string) because the
    /// `session/prompt` handler needs the plain `String` to call
    /// `run_llm_turn`, which does not know about ACP's `SessionId` type.
    agent_session_id: String,
    /// The client's workspace root, as reported in `session/new`'s `cwd`
    /// field. This is an absolute path on the *client's* machine, not this
    /// server's — nothing in this phase may canonicalise it, check it
    /// exists, or otherwise treat it as a local path. It is recorded now
    /// because a later phase needs it as the default working directory for
    /// client-side terminals (`terminal/create`); until then it is inert.
    ///
    /// Written straight through to `pending_cwd` on `session/new`, and
    /// recorded the same way by `session/load` — which does NOT feed it
    /// back into the store: `SessionStore::ensure_session` only writes a
    /// session's cwd on first creation, so a loaded session's recorded
    /// cwd cannot be updated through this field, and no code should try.
    ///
    /// Still nothing reads this field back out of the struct (a future
    /// `terminal/create` handler is the intended reader, per the module
    /// doc), so the attribute stays until that lands.
    #[allow(dead_code)]
    cwd: PathBuf,
    /// Cancellation tokens for the turns *currently in flight* on this
    /// session, keyed by their connection-wide turn number.
    ///
    /// Every live turn, not just the newest. Prompts on one connection run
    /// concurrently (see the `session/prompt` handler), so a client can have
    /// two turns open on one session, and `session/cancel` is scoped to the
    /// session rather than to a request — the schema calls it "ongoing
    /// operations for a session". Keeping only the latest token would leave
    /// an older turn calling the provider until the connection died and then
    /// answering `EndTurn`, and `Cancelled` is a MUST.
    ///
    /// A fresh token per turn rather than one per session, because a
    /// `CancellationToken` stays cancelled for good: reusing one would make
    /// every prompt after a cancel come straight back as `cancelled` without
    /// the provider ever being called. Each is a child of the connection's
    /// token, which is how one vanished client stops every turn at once.
    ///
    /// Entries are removed by the turn that owns them, so this holds only
    /// turns that are still running.
    turns: HashMap<u64, CancellationToken>,
    /// The permission mode this session is in. Per session, not per
    /// connection: two sessions on one socket are judged separately.
    mode: crate::tools::policy::SessionMode,
}

/// Sessions live for the lifetime of one ACP connection: a `HashMap` behind
/// a `tokio::sync::Mutex`, shared across this connection's request handlers
/// via `Arc`.
#[derive(Default)]
struct AcpSessions {
    inner: tokio::sync::Mutex<HashMap<SessionId, AcpSession>>,
    /// Hands every turn on this connection a number, so a turn that finishes
    /// can remove exactly its own token — `CancellationToken` has no identity
    /// to match on, and "the newest" is not the answer once turns overlap.
    next_turn: std::sync::atomic::AtomicU64,
}

/// Reports one prompt turn's progress to an ACP client as `session/update`
/// notifications, and remembers why a turn failed.
///
/// The remembering is not incidental: [`super::LlmTurnOutcome`] carries only
/// the final text, so a failed turn reaches the prompt handler as `None` with
/// no cause attached, and `turn_error` is the only place the provider's
/// message is ever offered. Stashing it here is what lets the JSON-RPC error
/// say *what* went wrong rather than merely *that* something did.
struct AcpProgress {
    session_id: SessionId,
    /// The connection this turn is running on. `ConnectionTo<Client>` is the
    /// handle the SDK passes to every request handler; it is `Clone`, so the
    /// turn keeps its own to push notifications through while it runs.
    connection: ConnectionTo<Client>,
    /// The message from the most recent `turn_error`. A blocking mutex is
    /// enough: it is never held across an await.
    error: std::sync::Mutex<Option<String>>,
    /// The room profile this connection's bearer token resolved to.
    /// Standing permission answers are recorded under it.
    profile: String,
    /// The session's mode as of the moment this turn started. Copied
    /// rather than shared: a `session/set_mode` arriving mid-turn must
    /// not change the rules under a call already being judged.
    mode: crate::tools::policy::SessionMode,
    permissions: Arc<super::acp_permissions::PermissionStore>,
}

impl AcpProgress {
    fn new(
        session_id: SessionId,
        connection: ConnectionTo<Client>,
        profile: String,
        mode: crate::tools::policy::SessionMode,
        permissions: Arc<super::acp_permissions::PermissionStore>,
    ) -> Self {
        Self {
            session_id,
            connection,
            error: std::sync::Mutex::new(None),
            profile,
            mode,
            permissions,
        }
    }

    /// Push one `session/update` for this turn's session.
    ///
    /// A send failure means the client is already gone, which the connection
    /// task notices on its own; there is nothing useful for a turn in flight
    /// to do about it beyond saying so.
    fn notify(&self, update: SessionUpdate) {
        if let Err(e) = self
            .connection
            .send_notification(SessionNotification::new(self.session_id.clone(), update))
        {
            warn!(
                "ACP: dropped a session/update for session {}: {e}",
                self.session_id
            );
        }
    }

    /// The reason the turn failed, if it reported one.
    fn failure(&self) -> Option<String> {
        self.error.lock().unwrap().clone()
    }
}

#[async_trait::async_trait]
impl super::TurnHost for AcpProgress {
    /// The provider's own tool-call id becomes ACP's `toolCallId`, so the
    /// completion below can name the call it completes. There is no input to
    /// report — `TurnHost` does not carry one — so the tool's name serves
    /// as the title.
    ///
    /// `Pending`, not `InProgress`. This fires *before* the permission
    /// gate, so at this moment the call may be waiting on the user's
    /// answer, or about to be refused outright — which is exactly what
    /// `Pending` means. It said `InProgress` when nothing could stand
    /// between the executor and the call; that stopped being true when
    /// the gate landed.
    async fn tool_start(&self, id: &str, name: &str) {
        self.notify(SessionUpdate::ToolCall(
            AcpToolCall::new(ToolCallId::new(id), name).status(ToolCallStatus::Pending),
        ));
    }

    /// Only the status changes: everything else the client already has from
    /// `tool_start`, and `ToolCallUpdate` carries just what moved.
    async fn tool_end(&self, id: &str, _name: &str) {
        self.notify(SessionUpdate::ToolCallUpdate(ToolCallUpdate::new(
            ToolCallId::new(id),
            ToolCallUpdateFields::new().status(ToolCallStatus::Completed),
        )));
    }

    /// Recorded rather than sent: ACP reports a failed turn by answering the
    /// `session/prompt` request with a JSON-RPC error, and this is the only
    /// place the cause is offered to put in it.
    async fn turn_error(&self, message: &str) {
        *self.error.lock().unwrap() = Some(message.to_string());
    }

    fn origin(&self) -> crate::tools::policy::Origin {
        crate::tools::policy::Origin::Acp(self.mode)
    }

    /// Move a call from `Pending` to `InProgress`.
    ///
    /// `tool_start` fires before the permission gate, so every call
    /// begins as `Pending` — which is accurate then, and wrong the
    /// moment the call is actually running. Without this edge a
    /// permitted call would sit at `Pending` for its whole runtime and
    /// then jump to `Completed`, telling the user it was still waiting
    /// on them while it ran.
    async fn tool_allowed(&self, id: &str) {
        self.notify(SessionUpdate::ToolCallUpdate(ToolCallUpdate::new(
            ToolCallId::new(id),
            ToolCallUpdateFields::new().status(ToolCallStatus::InProgress),
        )));
    }

    /// Put the call to the user, unless a standing answer settles it.
    ///
    /// The standing answer is consulted here rather than inside
    /// `decide` because `decide` is a pure function over the policy
    /// table and knows nothing about what this host has been told
    /// before.
    async fn approve(
        &self,
        call: &crate::provider::ToolCall,
        kind: crate::tools::ToolKind,
    ) -> crate::tools::policy::Approval {
        use crate::tools::policy::Approval;

        // A standing answer settles it without a round trip. Moving the
        // call off `Pending` is not done here: the gate calls
        // `tool_allowed` for every permitted call, asked or not.
        match self.permissions.standing(&self.profile, &call.name) {
            Some(true) => return Approval::AllowAlways,
            Some(false) => return Approval::RejectAlways,
            None => {}
        }

        let request = RequestPermissionRequest::new(
            self.session_id.clone(),
            ToolCallUpdate::new(
                ToolCallId::new(call.id.as_str()),
                ToolCallUpdateFields::new()
                    .title(call.name.clone())
                    .kind(kind)
                    .raw_input(call.input.clone()),
            ),
            vec![
                PermissionOption::new("allow_once", "Allow once", PermissionOptionKind::AllowOnce),
                PermissionOption::new(
                    "allow_always",
                    "Always allow this tool",
                    PermissionOptionKind::AllowAlways,
                ),
                PermissionOption::new("reject_once", "Reject", PermissionOptionKind::RejectOnce),
                PermissionOption::new(
                    "reject_always",
                    "Never allow this tool",
                    PermissionOptionKind::RejectAlways,
                ),
            ],
        );

        // `block_task`, and it is only sound because of where this
        // runs. The SDK's own documentation says never to await a sent
        // request inside a handler — the dispatch loop would stop
        // parsing frames and the client's answer could never arrive, so
        // it deadlocks. `approve` is called from `run_llm_turn`, which
        // the `session/prompt` handler hands to `ConnectionTo::spawn`
        // precisely so the turn runs outside that loop. That is the
        // case the SDK marks as safe.
        let answer = match self.connection.send_request(request).block_task().await {
            Ok(a) => a,
            Err(e) => {
                // The client went away, or refused the method. Either
                // way nobody said yes, and running unguarded because
                // the question failed to arrive is the wrong direction
                // to fail in.
                warn!(
                    "ACP: could not ask about '{}' on session {}: {e}. Treating as declined.",
                    call.name, self.session_id
                );
                return Approval::RejectOnce;
            }
        };

        let approval = match answer.outcome {
            // The turn is being cancelled; the cancel path answers the
            // prompt with `Cancelled`, so this call must simply not run.
            RequestPermissionOutcome::Cancelled => Approval::RejectOnce,
            RequestPermissionOutcome::Selected(selected) => match selected.option_id.0.as_ref() {
                "allow_once" => Approval::AllowOnce,
                "allow_always" => Approval::AllowAlways,
                "reject_always" => Approval::RejectAlways,
                "reject_once" => Approval::RejectOnce,
                other => {
                    warn!(
                        "ACP: unknown permission option '{other}' for '{}'; treating as declined.",
                        call.name
                    );
                    Approval::RejectOnce
                }
            },
            // `RequestPermissionOutcome` is `#[non_exhaustive]`, so a
            // future ACP version can add an outcome this build has
            // never heard of. Nobody said yes, so nothing runs.
            _ => {
                warn!(
                    "ACP: unrecognised permission outcome for '{}'; treating as declined.",
                    call.name
                );
                Approval::RejectOnce
            }
        };

        if approval.is_sticky() {
            self.permissions.record(&self.profile, &call.name, approval);
        }
        approval
    }
}

/// How one prompt turn stopped.
enum TurnEnd {
    /// The client asked for it to stop, or its connection went away.
    Cancelled,
    /// The shared executor finished — successfully or not; the outcome says
    /// which.
    Ran(super::LlmTurnOutcome),
}

/// Log, rather than propagate, a failure to answer a `session/prompt`.
///
/// The turn that produced this answer runs in a task spawned on the
/// connection, and the SDK shuts the whole connection down when one of those
/// returns an error (`ConnectionTo::spawn`'s own documentation says so). A
/// send that fails here only ever means the client already left — which the
/// connection notices on its own — so it must not be reported that way.
fn answered(session_id: &SessionId, sent: Result<(), Error>) -> Result<(), Error> {
    if let Err(e) = sent {
        warn!("ACP: could not answer session/prompt for session {session_id}: {e}");
    }
    Ok(())
}

/// The wire name of a prompt content block, for the log line that says one
/// was dropped. `ContentBlock` is `#[non_exhaustive]`, so the catch-all arm
/// is load-bearing rather than defensive.
fn prompt_block_kind(block: &ContentBlock) -> &'static str {
    match block {
        ContentBlock::Text(_) => "text",
        ContentBlock::Image(_) => "image",
        ContentBlock::Audio(_) => "audio",
        ContentBlock::ResourceLink(_) => "resource_link",
        ContentBlock::Resource(_) => "resource",
        _ => "unknown",
    }
}

pub async fn handle_acp_ws(
    State(state): State<Arc<ServeState>>,
    headers: HeaderMap,
    ws: WebSocketUpgrade,
) -> Response {
    // 0. Feature gate, mirroring /a2a.
    if !state.config.acp.as_ref().is_some_and(|c| c.enabled) {
        return (StatusCode::NOT_FOUND, "ACP disabled").into_response();
    }

    // 1. Bearer auth → room profile. Both failure modes are 401 at the HTTP
    //    layer; ACP never sees an unauthenticated peer.
    let Some(bearer) = extract_bearer(&headers) else {
        return (StatusCode::UNAUTHORIZED, "missing bearer token").into_response();
    };
    let Some(profile_name) = state
        .device_auth
        .resolve(&bearer)
        .map(|r| r.room_profile.to_string())
    else {
        warn!("ACP: rejected an unknown or revoked bearer token");
        return (StatusCode::UNAUTHORIZED, "unknown or revoked bearer token").into_response();
    };

    info!("ACP: connection accepted for room profile '{profile_name}'");
    ws.on_upgrade(move |socket| serve_connection(socket, state, profile_name))
}

/// Wrap the socket as the SDK's line transport.
///
/// Per the ACP transport RFD one JSON-RPC message rides in one text frame,
/// and `Lines` hands the sink one JSON-RPC message per `String` with no
/// trailing newline — so a text frame *is* a line and neither direction
/// needs reframing, buffering or splitting.
///
/// Everything that is not a text frame is dropped on the floor: binary
/// frames carry no ACP meaning, axum answers ping/pong itself, and a close
/// frame is followed by the end of the stream, which is what actually ends
/// the connection.
///
/// `connection_cancel` is cancelled when this socket stops delivering
/// frames — see [`cancel_when_exhausted`].
fn lines_transport(
    socket: WebSocket,
    connection_cancel: CancellationToken,
) -> Lines<
    impl futures_util::Sink<String, Error = std::io::Error> + Send + 'static,
    impl futures_util::Stream<Item = std::io::Result<String>> + Send + 'static,
> {
    let (tx, rx) = socket.split();

    let outgoing = tx
        .sink_map_err(std::io::Error::other)
        .with(|line: String| async move { Ok::<_, std::io::Error>(Message::Text(line.into())) });

    let frames = rx.filter_map(|frame| async move {
        match frame {
            Ok(Message::Text(text)) => Some(Ok(text.to_string())),
            // Not ACP: no reply, and the connection carries on.
            Ok(Message::Binary(_) | Message::Ping(_) | Message::Pong(_)) => None,
            // The stream ends right after this, which closes the connection.
            Ok(Message::Close(_)) => None,
            Err(e) => Some(Err(std::io::Error::other(e))),
        }
    });

    Lines::new(outgoing, cancel_when_exhausted(frames, connection_cancel))
}

/// Cancel `token` as soon as `frames` stops producing items.
///
/// This is *not* what keeps a turn from outliving its connection — the SDK
/// already does that. `Builder::connect_to` runs `incoming_closed()` then
/// `drain_outgoing()` as the foreground of the connection future
/// (`jsonrpc.rs:1602-1611`), with the task actor as the background; when the
/// foreground finishes the background is dropped, taking every spawned turn
/// with it. Cancelling after `connect_to` returns would be redundant, not too
/// late.
///
/// What this adds is *timing*. The guard fires at EOF, which is before the
/// drain — and a client that has gone away without reading can hold the drain
/// up indefinitely by backpressuring the outgoing sink. Until the drain
/// finishes, a turn would otherwise still be calling the provider.
///
/// The token rides in the stream's own state as a `DropGuard`: `Unfold` drops
/// the completed step future, and with it the guard, at the poll that observes
/// `None`. On the error path the item is `Some(Err(_))` and the state is
/// retained, so there the guard fires only when the SDK drops the stream while
/// failing its actors — same outcome, different mechanism.
fn cancel_when_exhausted<S>(
    frames: S,
    token: CancellationToken,
) -> impl futures_util::Stream<Item = S::Item> + Send + 'static
where
    S: futures_util::Stream + Send + 'static,
{
    futures_util::stream::unfold(
        (Box::pin(frames), token.drop_guard()),
        |(mut frames, guard)| async move { frames.next().await.map(|item| (item, (frames, guard))) },
    )
}

/// The mode list every session starts with.
///
/// A loaded session starts in `default` like a new one: the mode is a
/// statement about how the editor wants the agent to behave right now,
/// not a property of the conversation, so it is not persisted.
fn mode_state() -> SessionModeState {
    SessionModeState::new(
        crate::tools::policy::SessionMode::Default.id(),
        crate::tools::policy::SessionMode::ALL
            .into_iter()
            .map(|m| AcpSessionMode::new(m.id(), m.name()).description(m.description()))
            .collect(),
    )
}

/// Count this connection as holding `id`, warning when another
/// connection already does.
///
/// Shared by `adopt_session` (`session/load`, `session/resume`) and
/// `session/new` — every path that hands a connection a session must
/// count it, since the decrement in `serve_connection` walks every
/// session a connection's map holds and expects each to have been
/// counted on the way in.
async fn count_session_open(state: &Arc<ServeState>, id: &str) {
    let mut open = state.open_acp_sessions.lock().await;
    let count = open.entry(id.to_string()).or_insert(0);
    *count += 1;
    if *count > 1 {
        warn!(
            "ACP: session {id} is now open on {count} connections. Concurrent prompts on \
             one session are not history-safe: run_llm_turn clones the history at the top \
             and writes it back whole, so the last turn to finish wins in memory while both \
             have already appended to the transcript."
        );
    }
}

/// Validate an existing session id and adopt it onto this connection.
///
/// Shared by `session/load` and `session/resume`, which differ only in
/// whether they replay afterwards. `Err` carries the refusal to answer
/// with — one wording for both "no such session" and "not yours", so
/// the pair cannot be used to enumerate ids.
async fn adopt_session(
    state: &Arc<ServeState>,
    sessions: &Arc<AcpSessions>,
    profile_name: &str,
    session_id: &SessionId,
    cwd: PathBuf,
) -> Result<String, Error> {
    let id = session_id.to_string();
    let namespace = state
        .config
        .namespace_for_room_profile(profile_name)
        .to_string();
    let refuse = || Error::invalid_params().data("no such session is available on this connection");

    let Some((meta, _closed)) = state.cross_device_session_store.session_header(&id) else {
        return Err(refuse());
    };
    if meta.namespace.as_deref() != Some(namespace.as_str()) {
        warn!(
            "ACP: refused adopting {id}: it belongs to namespace {:?}, not {namespace}",
            meta.namespace
        );
        return Err(refuse());
    }

    state
        .session_room_profiles
        .lock()
        .await
        .insert(id.clone(), profile_name.to_string());
    sessions.inner.lock().await.insert(
        session_id.clone(),
        AcpSession {
            agent_session_id: id.clone(),
            cwd,
            turns: HashMap::new(),
            mode: crate::tools::policy::SessionMode::Default,
        },
    );
    count_session_open(state, &id).await;
    Ok(id)
}

/// Drive one ACP connection until the peer goes away.
async fn serve_connection(socket: WebSocket, state: Arc<ServeState>, profile_name: String) {
    let sessions = Arc::new(AcpSessions::default());
    // Cancelled when the socket goes away (see `lines_transport`). Every
    // turn's token is a child of this one, so one client leaving stops every
    // turn it left running — including turns on sessions created later.
    let connection_cancel = CancellationToken::new();

    let result = Agent
        .builder()
        .name("sapphire-agent")
        .on_receive_request(
            async move |req: InitializeRequest, responder, _connection| {
                // Answer with the version we will actually speak, which the
                // ACP specification defines as the client's version if we
                // support it and otherwise the latest version we do support
                // (the prose spec's Initialization / "Protocol Version
                // Negotiation" section; the SDK only carries the version
                // constants themselves, in `schema/src/version.rs`). Handing
                // the request back unchanged lies in both directions: a v2
                // client told "2" sends v2-shaped traffic nothing here
                // understands, and a v0 client told "0" is the same lie at the
                // other end — v0 being a pre-release the schema documents as
                // one to treat as unsupported.
                //
                // An explicit supported set rather than a clamp: with one
                // version implemented the correct answer is `V1` for every
                // input, and this keeps saying what we actually support as
                // versions accumulate, instead of silently claiming each new
                // gap in the range.
                //
                // `V1`, not `LATEST`: this is a claim about what *this* code
                // implements. `LATEST` means "newest stable the SDK knows of"
                // and would resume telling exactly this lie the day the SDK
                // promotes v2 (it is cfg-gated off when `unstable_protocol_v2`
                // is enabled, precisely so that choice stays explicit).
                const SUPPORTED: [ProtocolVersion; 1] = [ProtocolVersion::V1];
                let version = if SUPPORTED.contains(&req.protocol_version) {
                    req.protocol_version
                } else {
                    ProtocolVersion::V1
                };

                // `authMethods` is empty because the bearer token checked
                // above already authenticated the peer, so ACP never sees
                // an unauthenticated client.
                responder.respond(
                    InitializeResponse::new(version).agent_capabilities(
                        AgentCapabilities::new()
                            .load_session(true)
                            .session_capabilities(
                                SessionCapabilities::new()
                                    .list(SessionListCapabilities::new())
                                    .resume(SessionResumeCapabilities::new()),
                            ),
                    ),
                )
            },
            on_receive_request!(),
        )
        .on_receive_request(
            {
                let sessions = Arc::clone(&sessions);
                let state = Arc::clone(&state);
                let profile_name = profile_name.clone();
                async move |req: NewSessionRequest, responder, _connection| {
                    // Mint the agent-side session id the same way a
                    // brand-new `/rpc` session gets one in `handle_initialize`
                    // (`src/serve/mod.rs`), rather than inventing a second
                    // convention.
                    let agent_session_id = uuid::Uuid::now_v7().to_string();

                    // `mcp_servers` is not honoured, and silence about that
                    // would be indistinguishable from a bug: the editor's
                    // configured servers would simply be absent, with no
                    // tools from them and nothing said. They are the
                    // *client's* servers, mostly stdio commands to spawn on
                    // the client's machine, which an agent reached over a
                    // socket cannot launch and must not try to — this
                    // process is not on that machine. MCP servers for this
                    // agent are configured server-side, in its own config.
                    if !req.mcp_servers.is_empty() {
                        warn!(
                            "ACP: ignoring {} MCP server(s) offered by the client for session {}: \
                             they are the client's to launch, and this agent runs elsewhere. \
                             Configure MCP servers in the agent's own config instead.",
                            req.mcp_servers.len(),
                            agent_session_id,
                        );
                    }

                    // Pin the session to the room profile the bearer token
                    // resolved to at connection time. That pin is what gives
                    // the ACP session its namespace chain and provider
                    // through the paths that already exist for `/rpc`.
                    state
                        .session_room_profiles
                        .lock()
                        .await
                        .insert(agent_session_id.clone(), profile_name.clone());

                    // The file does not exist yet — `ensure_session`
                    // creates it on the first turn — so the cwd waits in
                    // `pending_cwd` until then.
                    state.pending_cwd.lock().await.insert(
                        agent_session_id.clone(),
                        req.cwd.to_string_lossy().to_string(),
                    );

                    let session_id = SessionId::new(agent_session_id.clone());
                    count_session_open(&state, &agent_session_id).await;
                    sessions.inner.lock().await.insert(
                        session_id.clone(),
                        AcpSession {
                            agent_session_id,
                            cwd: req.cwd.clone(),
                            // No turn is running yet, so there is nothing for
                            // a `session/cancel` to reach.
                            turns: HashMap::new(),
                            mode: crate::tools::policy::SessionMode::Default,
                        },
                    );

                    // The client learns the modes here, and starts in
                    // the one that asks. `NewSessionResponse.modes` is a
                    // plain `Option` in the schema, with no capability
                    // gating it, so `initialize` needs no change.
                    responder.respond(NewSessionResponse::new(session_id).modes(mode_state()))
                }
            },
            on_receive_request!(),
        )
        .on_receive_request(
            {
                let state = Arc::clone(&state);
                let profile_name = profile_name.clone();
                async move |req: ListSessionsRequest, responder, _connection| {
                    let namespace = state
                        .config
                        .namespace_for_room_profile(&profile_name)
                        .to_string();
                    let store = Arc::clone(&state.cross_device_session_store);
                    let wanted_cwd = req.cwd.as_ref().map(|c| c.to_string_lossy().to_string());

                    let sessions: Vec<SessionInfo> = store
                        .list_session_headers()
                        .into_iter()
                        // A closed session is archived, not current.
                        .filter(|(_, is_closed)| !is_closed)
                        .map(|(meta, _)| meta)
                        .filter(|meta| {
                            // Three filters, and the namespace one is a
                            // boundary rather than a convenience. A file
                            // too old to name its namespace is not shown:
                            // an unknown owner is not the same as "mine".
                            meta.namespace.as_deref() == Some(namespace.as_str())
                        })
                        .filter(|meta| match &wanted_cwd {
                            Some(wanted) => meta.cwd.as_deref() == Some(wanted.as_str()),
                            None => true,
                        })
                        .filter_map(|meta| {
                            let path = store.absolute_path_for(&meta.session_id)?;
                            let updated_at = std::fs::metadata(&path)
                                .and_then(|m| m.modified())
                                .ok()
                                .map(|t| chrono::DateTime::<chrono::Utc>::from(t).to_rfc3339());
                            let cwd = meta
                                .cwd
                                .as_deref()
                                .map(PathBuf::from)
                                // No client ever reported a cwd for this
                                // session — it predates the field, or it
                                // came in over /rpc, voice or chat.
                                // `SessionInfo.cwd` is required and must
                                // be absolute, so an empty path would be
                                // a contract violation dressed up as a
                                // null. The agent's own workspace root is
                                // both absolute and true: the
                                // conversation belongs to the agent, not
                                // to an editor project. An editor
                                // filtering by its project directory
                                // therefore will not match it, which is
                                // the intended behaviour for these
                                // sessions.
                                .unwrap_or_else(|| state.workspace.root());
                            let mut info =
                                SessionInfo::new(SessionId::new(meta.session_id.clone()), cwd);
                            info = info.title(meta.title.clone());
                            info = info.updated_at(updated_at);
                            Some(info)
                        })
                        .collect();

                    // No pagination: the whole list, every time. A
                    // cursor earns its keep when a namespace holds
                    // thousands of sessions, and none does yet.
                    responder.respond(ListSessionsResponse::new(sessions))
                }
            },
            on_receive_request!(),
        )
        .on_receive_request(
            {
                let sessions = Arc::clone(&sessions);
                let state = Arc::clone(&state);
                let profile_name = profile_name.clone();
                async move |req: LoadSessionRequest, responder, connection: ConnectionTo<Client>| {
                    // Validation and adoption are shared with
                    // `session/resume` — see `adopt_session`. `load`'s
                    // own job starts after that: replaying the history
                    // the resumed session already has.
                    let id = match adopt_session(
                        &state,
                        &sessions,
                        &profile_name,
                        &req.session_id,
                        req.cwd.clone(),
                    )
                    .await
                    {
                        Ok(id) => id,
                        Err(e) => return responder.respond_with_error(e),
                    };
                    let store = Arc::clone(&state.cross_device_session_store);

                    // Replay BEFORE answering: the ACP specification
                    // orders it that way, and a client that got the
                    // reply first would render an empty thread and then
                    // watch messages appear underneath it.
                    for message in store.load_session(&id).unwrap_or_default() {
                        let text: String = message
                            .parts
                            .iter()
                            .filter_map(|part| match part {
                                crate::provider::ContentPart::Text(t) => Some(t.as_str()),
                                _ => None,
                            })
                            .collect::<Vec<_>>()
                            .join("\n");
                        if text.is_empty() {
                            continue;
                        }
                        let chunk = ContentChunk::new(ContentBlock::Text(TextContent::new(text)));
                        let update = match message.role {
                            crate::provider::Role::User => SessionUpdate::UserMessageChunk(chunk),
                            _ => SessionUpdate::AgentMessageChunk(chunk),
                        };
                        if let Err(e) = connection.send_notification(SessionNotification::new(
                            req.session_id.clone(),
                            update,
                        )) {
                            warn!("ACP: dropped a replay update for {id}: {e}");
                        }
                    }

                    responder.respond(LoadSessionResponse::new().modes(mode_state()))
                }
            },
            on_receive_request!(),
        )
        .on_receive_request(
            {
                let sessions = Arc::clone(&sessions);
                let state = Arc::clone(&state);
                let profile_name = profile_name.clone();
                async move |req: ResumeSessionRequest, responder, _connection| {
                    // Same adoption as `load`, no replay. The ACP
                    // specification frames `resume` as the fallback for
                    // agents that cannot load at all; offered here so a
                    // client can skip redrawing a long conversation.
                    match adopt_session(
                        &state,
                        &sessions,
                        &profile_name,
                        &req.session_id,
                        req.cwd.clone(),
                    )
                    .await
                    {
                        Ok(_) => {
                            responder.respond(ResumeSessionResponse::new().modes(mode_state()))
                        }
                        Err(e) => responder.respond_with_error(e),
                    }
                }
            },
            on_receive_request!(),
        )
        .on_receive_notification(
            {
                let sessions = Arc::clone(&sessions);
                async move |notif: CancelNotification, _connection: ConnectionTo<Client>| {
                    // *Every* turn on the session, not the newest one: the
                    // notification names a session, and the schema scopes it
                    // to that session's "ongoing operations" — plural, which
                    // concurrent prompts make reachable.
                    //
                    // A session with nothing running is not an error; the
                    // client may simply have raced the reply, and the empty
                    // map makes that a no-op. An unknown session is a
                    // different thing, and a notification has no way to
                    // report it but the log.
                    match sessions.inner.lock().await.get(&notif.session_id) {
                        Some(session) => {
                            for turn in session.turns.values() {
                                turn.cancel();
                            }
                        }
                        None => warn!(
                            "ACP: session/cancel named a session this connection never \
                             minted: {}",
                            notif.session_id
                        ),
                    }
                    Ok(())
                }
            },
            on_receive_notification!(),
        )
        .on_receive_request(
            {
                let sessions = Arc::clone(&sessions);
                async move |req: SetSessionModeRequest,
                            responder,
                            connection: ConnectionTo<Client>| {
                    let Some(mode) =
                        crate::tools::policy::SessionMode::from_id(req.mode_id.0.as_ref())
                    else {
                        // `plan` lands here. An error is the honest
                        // reply: silently picking another mode would
                        // leave the user believing the agent is
                        // planning when it is about to act.
                        return responder.respond_with_error(
                            Error::invalid_params().data(format!("unknown mode '{}'", req.mode_id)),
                        );
                    };

                    {
                        let mut guard = sessions.inner.lock().await;
                        let Some(session) = guard.get_mut(&req.session_id) else {
                            return responder.respond_with_error(
                                Error::invalid_params()
                                    .data(format!("unknown session '{}'", req.session_id)),
                            );
                        };
                        session.mode = mode;
                    }

                    // Announce it: a client that changed the mode from
                    // one surface should see it reflected on the others.
                    if let Err(e) = connection.send_notification(SessionNotification::new(
                        req.session_id.clone(),
                        SessionUpdate::CurrentModeUpdate(CurrentModeUpdate::new(mode.id())),
                    )) {
                        warn!("ACP: dropped a current_mode_update: {e}");
                    }

                    responder.respond(SetSessionModeResponse::new())
                }
            },
            on_receive_request!(),
        )
        .on_receive_request(
            {
                let sessions = Arc::clone(&sessions);
                let state = Arc::clone(&state);
                let connection_cancel = connection_cancel.clone();
                let profile_name = profile_name.clone();
                async move |req: PromptRequest, responder, connection: ConnectionTo<Client>| {
                    // Register this turn's cancellation token in the same
                    // lock that resolves the session, so a `session/cancel`
                    // arriving after this point cannot miss it.
                    let turn = sessions
                        .next_turn
                        .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    let looked_up = {
                        let mut guard = sessions.inner.lock().await;
                        guard.get_mut(&req.session_id).map(|session| {
                            let turn_cancel = connection_cancel.child_token();
                            session.turns.insert(turn, turn_cancel.clone());
                            (session.agent_session_id.clone(), turn_cancel, session.mode)
                        })
                    };
                    let Some((agent_session_id, turn_cancel, mode)) = looked_up else {
                        // Not created on the fly: a prompt naming a session
                        // this connection never minted is a client bug, and
                        // starting one here would quietly open a second
                        // conversation the client believes it is continuing.
                        // `invalid_params`, because the offending thing is a
                        // parameter of this request.
                        return responder.respond_with_error(Error::invalid_params().data(
                            format!(
                                "unknown session '{}'; call session/new first",
                                req.session_id
                            ),
                        ));
                    };

                    // Flatten the prompt's blocks into one user message.
                    //
                    // Text and ResourceLink are both mandatory for every
                    // agent — the schema's words are "All agents MUST support
                    // resource links in prompts" — and a resource link is how
                    // Zed sends an `@file` mention. Dropping one would leave
                    // the model reading "explain" with no idea what it refers
                    // to. The link is folded in by name and uri: this build
                    // cannot open the file (client-side filesystem access is
                    // a later phase), but naming the reference keeps the
                    // mention in the conversation until it can.
                    //
                    // Image, Audio and Resource are capability-gated and
                    // `initialize` advertises no prompt capabilities, so they
                    // should never arrive — if one does, it is dropped
                    // loudly rather than silently.
                    let text = req
                        .prompt
                        .iter()
                        .filter_map(|block| match block {
                            ContentBlock::Text(t) => Some(t.text.clone()),
                            ContentBlock::ResourceLink(link) => Some(format!(
                                "[referenced resource: {} ({})]",
                                link.name, link.uri
                            )),
                            other => {
                                warn!(
                                    "ACP: dropped an unsupported prompt block ({}) from a \
                                     session/prompt for session {}",
                                    prompt_block_kind(other),
                                    req.session_id
                                );
                                None
                            }
                        })
                        .collect::<Vec<_>>()
                        .join("\n");

                    // Everything past this point belongs to `run_llm_turn`:
                    // history, the tool loop, persistence and the memory
                    // namespace all come from the shared executor, so an
                    // editor's conversation lands in the same session store,
                    // with the same system prompt, as `/rpc` and A2A.
                    let session_id = req.session_id.clone();
                    let progress = Arc::new(AcpProgress::new(
                        session_id.clone(),
                        connection.clone(),
                        profile_name.clone(),
                        mode,
                        Arc::clone(&state.permissions),
                    ));

                    // The turn runs OUTSIDE the dispatch loop, and the
                    // `Responder` travels with it.
                    //
                    // This is not a stylistic choice and must not be
                    // "simplified" back into an `await` here. Handlers
                    // registered with `on_receive_request` run *inside* the
                    // SDK's dispatch loop, which parses no further frame on
                    // this connection until the handler returns. Awaiting the
                    // turn here would mean the `session/cancel` sent to stop
                    // it could not be read until the turn had already
                    // finished — the feature would be unimplementable rather
                    // than merely slow. `ConnectionTo::spawn` is the SDK's own
                    // escape hatch for exactly this, and `Responder` is
                    // movable, so the spawned task answers the request
                    // whenever the turn really ends.
                    //
                    // The consequence, accepted deliberately: prompts on one
                    // connection now run concurrently instead of one after
                    // another.
                    connection.spawn({
                        let state = Arc::clone(&state);
                        let sessions = Arc::clone(&sessions);
                        async move {
                            // Bound to a `let` on purpose: that drops the
                            // losing branch's future here, before the answer
                            // goes out, so a client told its turn was
                            // cancelled knows the provider call is already
                            // abandoned rather than merely disowned.
                            let end = tokio::select! {
                                // `biased`, so a cancellation that lands in
                                // the same poll as a finished turn still wins:
                                // the schema makes `Cancelled` a MUST "even if
                                // the cancellation causes exceptions in
                                // underlying operations", which means the
                                // error path below must be unreachable once
                                // the token has fired.
                                biased;
                                () = turn_cancel.cancelled() => TurnEnd::Cancelled,
                                outcome = super::run_llm_turn(
                                    state,
                                    agent_session_id,
                                    crate::provider::ChatMessage::user(&text),
                                    Arc::clone(&progress) as Arc<dyn super::TurnHost>,
                                    None,
                                ) => TurnEnd::Ran(outcome),
                            };

                            // This turn is no longer an "ongoing operation",
                            // so drop its token: a later `session/cancel`
                            // should not have to fire at turns that have
                            // already answered, and a long-lived session must
                            // not accumulate one token per prompt it ever
                            // received. The local clone above still works, so
                            // the reply paths below are unaffected.
                            if let Some(session) = sessions.inner.lock().await.get_mut(&session_id)
                            {
                                session.turns.remove(&turn);
                            }

                            let TurnEnd::Ran(outcome) = end else {
                                return answered(
                                    &session_id,
                                    responder.respond(PromptResponse::new(StopReason::Cancelled)),
                                );
                            };

                            let Some(reply) = outcome.text else {
                                if turn_cancel.is_cancelled() {
                                    // The turn failed *because* it was being
                                    // cancelled, or was cancelled in the
                                    // moment it failed. Either way the client
                                    // asked for this, and the schema says it
                                    // hears `Cancelled`.
                                    return answered(
                                        &session_id,
                                        responder
                                            .respond(PromptResponse::new(StopReason::Cancelled)),
                                    );
                                }

                                // Running out of tool rounds is not a
                                // failure, and ACP has the exact word for
                                // it: `MaxTurnRequests`, "the agent reached
                                // the maximum number of allowed agent
                                // requests between user turns". The budget
                                // is `MAX_TOOL_ROUNDS` — ten — which an
                                // editor reaches on an ordinary "search,
                                // read a few files, edit two" prompt, so
                                // this is a routine ending, not an
                                // exceptional one, and showing the user an
                                // error dialog for it is wrong twice over:
                                // the agent is fine, and the work it did do
                                // would go on the floor. The prose it
                                // emitted alongside its tool calls is
                                // delivered as the reply.
                                if let super::TurnStop::BudgetExhausted { partial_text } =
                                    &outcome.stop
                                {
                                    if !partial_text.is_empty() {
                                        progress.notify(SessionUpdate::AgentMessageChunk(
                                            ContentChunk::new(ContentBlock::Text(
                                                TextContent::new(partial_text.clone()),
                                            )),
                                        ));
                                    }
                                    return answered(
                                        &session_id,
                                        responder.respond(PromptResponse::new(
                                            StopReason::MaxTurnRequests,
                                        )),
                                    );
                                }

                                // A failed turn is a JSON-RPC error, not a
                                // stop reason: none of ACP v1's stop reasons
                                // means "the agent broke", and `Refusal` would
                                // tell the user the agent *declined*, which is
                                // a materially different thing to show them.
                                return answered(
                                    &session_id,
                                    responder.respond_with_internal_error(
                                        progress.failure().unwrap_or_else(|| {
                                            "the turn produced no reply".to_string()
                                        }),
                                    ),
                                );
                            };

                            // One chunk, not a stream: `Provider::chat`
                            // returns the whole response at once, so there is
                            // nothing to stream, and splitting it here would
                            // invent chunk boundaries the model never
                            // produced. An empty reply is no chunk at all
                            // rather than an empty one.
                            if !reply.is_empty() {
                                progress.notify(SessionUpdate::AgentMessageChunk(
                                    ContentChunk::new(ContentBlock::Text(TextContent::new(reply))),
                                ));
                            }
                            answered(
                                &session_id,
                                responder.respond(PromptResponse::new(StopReason::EndTurn)),
                            )
                        }
                    })?;

                    Ok(())
                }
            },
            on_receive_request!(),
        )
        .connect_to(lines_transport(socket, connection_cancel.clone()))
        .await;

    // Release every session this connection held. Missing this would turn
    // `open_acp_sessions` into a leak that warns forever: the next
    // connection to load the same session would find a stale count already
    // above one and warn about a collision that no longer exists.
    {
        let held: Vec<String> = sessions
            .inner
            .lock()
            .await
            .values()
            .map(|s| s.agent_session_id.clone())
            .collect();
        let mut open = state.open_acp_sessions.lock().await;
        for id in held {
            if let Some(count) = open.get_mut(&id) {
                *count = count.saturating_sub(1);
                if *count == 0 {
                    open.remove(&id);
                }
            }
        }
    }

    // Belt and braces, and known to be so. By the time this runs the socket
    // has gone, so `cancel_when_exhausted` has already fired the token, and
    // the SDK has dropped the task actor holding any turn along with it. It
    // is kept because it states the connection's contract where a reader
    // looks for it, and costs one atomic on a path that runs once per
    // connection.
    connection_cancel.cancel();

    // Name the profile on the way out as well as the way in: with several
    // editors connected under different profiles, a bare "connection closed"
    // cannot be matched to the connection it belongs to.
    if let Err(e) = result {
        warn!("ACP: connection for room profile '{profile_name}' ended with an error: {e}");
    } else {
        info!("ACP: connection for room profile '{profile_name}' closed");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::serve::HangingChat;
    use futures_util::{SinkExt, StreamExt};
    use tokio::net::TcpListener;
    use tokio_tungstenite::tungstenite::Message;
    use tokio_tungstenite::tungstenite::client::IntoClientRequest;
    use tokio_tungstenite::{MaybeTlsStream, WebSocketStream};

    /// One end of an ACP socket, as the test client holds it.
    type TestSocket = WebSocketStream<MaybeTlsStream<tokio::net::TcpStream>>;

    /// Bind the router on an ephemeral port and return its `host:port`.
    async fn spawn(state: Arc<ServeState>) -> String {
        let app = axum::Router::new()
            .route("/acp", axum::routing::get(handle_acp_ws))
            .with_state(state);
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        format!("127.0.0.1:{}", addr.port())
    }

    /// Attempt the upgrade. Returns the HTTP status the server refused with,
    /// or `None` when the upgrade succeeded (101).
    async fn upgrade_status(addr: &str, token: Option<&str>) -> Option<u16> {
        let mut req = format!("ws://{addr}/acp").into_client_request().unwrap();
        if let Some(t) = token {
            req.headers_mut()
                .insert("authorization", format!("Bearer {t}").parse().unwrap());
        }
        match tokio_tungstenite::connect_async(req).await {
            Ok(_) => None,
            Err(tokio_tungstenite::tungstenite::Error::Http(resp)) => Some(resp.status().as_u16()),
            Err(e) => panic!("unexpected transport error: {e}"),
        }
    }

    #[tokio::test]
    async fn disabled_endpoint_is_not_found() {
        let addr = spawn(ServeState::for_test(false)).await;
        assert_eq!(upgrade_status(&addr, Some("sa-acp-token")).await, Some(404));
    }

    #[tokio::test]
    async fn missing_and_unknown_tokens_are_unauthorized() {
        let addr = spawn(ServeState::for_test(true)).await;
        assert_eq!(upgrade_status(&addr, None).await, Some(401));
        assert_eq!(upgrade_status(&addr, Some("sa-wrong")).await, Some(401));
    }

    #[tokio::test]
    async fn valid_token_upgrades() {
        let addr = spawn(ServeState::for_test(true)).await;
        assert_eq!(upgrade_status(&addr, Some("sa-acp-token")).await, None);
    }

    /// Open an authenticated ACP socket.
    pub(super) async fn connect(addr: &str) -> TestSocket {
        let mut req = format!("ws://{addr}/acp").into_client_request().unwrap();
        req.headers_mut()
            .insert("authorization", "Bearer sa-acp-token".parse().unwrap());
        tokio_tungstenite::connect_async(req).await.unwrap().0
    }

    pub(super) fn initialize_request(id: i64) -> serde_json::Value {
        initialize_request_asking_for(id, 1)
    }

    /// An `initialize` that asks for a specific protocol version.
    fn initialize_request_asking_for(id: i64, protocol_version: u16) -> serde_json::Value {
        serde_json::json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": "initialize",
            "params": {
                "protocolVersion": protocol_version,
                "clientCapabilities": {
                    "fs": { "readTextFile": true, "writeTextFile": true },
                    "terminal": true
                },
                "clientInfo": { "name": "test-client", "version": "0.0.0" }
            }
        })
    }

    fn test_cwd() -> &'static str {
        if cfg!(windows) {
            "C:\\work\\proj"
        } else {
            "/work/proj"
        }
    }

    fn new_session_request(id: i64) -> serde_json::Value {
        serde_json::json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": "session/new",
            "params": { "cwd": test_cwd(), "mcpServers": [] }
        })
    }

    /// Read the next frame. A missing reply fails loudly instead of hanging
    /// the test run forever.
    async fn next_frame(ws: &mut TestSocket) -> Message {
        tokio::time::timeout(std::time::Duration::from_secs(10), ws.next())
            .await
            .expect("timed out waiting for an ACP frame")
            .expect("stream ended")
            .unwrap()
    }

    /// Run several requests over ONE connection, in order.
    pub(super) async fn conversation(
        addr: &str,
        requests: Vec<serde_json::Value>,
    ) -> Vec<serde_json::Value> {
        let mut ws = connect(addr).await;
        let mut responses = Vec::new();
        for request in requests {
            let want_id = request["id"].clone();
            ws.send(Message::Text(request.to_string().into()))
                .await
                .unwrap();
            loop {
                match next_frame(&mut ws).await {
                    Message::Text(t) => {
                        let v: serde_json::Value = serde_json::from_str(&t).unwrap();
                        if v["id"] == want_id {
                            responses.push(v);
                            break;
                        }
                    }
                    Message::Ping(_) | Message::Pong(_) => continue,
                    other => panic!("unexpected frame: {other:?}"),
                }
            }
        }
        responses
    }

    /// Send one request on a fresh connection and read the reply to it.
    async fn roundtrip(addr: &str, request: serde_json::Value) -> serde_json::Value {
        conversation(addr, vec![request])
            .await
            .pop()
            .expect("one reply per request")
    }

    #[tokio::test]
    async fn initialize_answers_with_v1_capabilities() {
        let addr = spawn(ServeState::for_test(true)).await;
        let resp = roundtrip(&addr, initialize_request(0)).await;

        assert_eq!(resp["id"], 0);
        let result = &resp["result"];
        assert_eq!(result["protocolVersion"], 1);
        assert_eq!(result["agentCapabilities"]["loadSession"], true);
        assert_eq!(
            result["authMethods"],
            serde_json::json!([]),
            "auth already happened at the HTTP layer"
        );
    }

    /// A client offering a version this build does not implement must be told
    /// v1 — the version we will actually speak — not handed its own request
    /// back. That holds in both directions: answering "2" to a v2 client makes
    /// it send v2-shaped traffic nothing here understands, and answering "0"
    /// to a v0 client is the same lie at the other end of the range. v0 is a
    /// pre-release the schema documents as unsupported, and this build does
    /// not implement it either.
    #[tokio::test]
    async fn unsupported_versions_are_negotiated_to_v1() {
        let addr = spawn(ServeState::for_test(true)).await;

        for asked in [0u16, 2, 99] {
            let resp = roundtrip(&addr, initialize_request_asking_for(0, asked)).await;
            assert_eq!(
                resp["result"]["protocolVersion"], 1,
                "asked for {asked}, must be negotiated to v1, got {resp}"
            );
        }
    }

    #[tokio::test]
    async fn malformed_frame_errors_without_closing_the_connection() {
        let addr = spawn(ServeState::for_test(true)).await;
        let mut ws = connect(&addr).await;

        ws.send(Message::Text("{ this is not json".into()))
            .await
            .unwrap();
        // A parse error carries a null id per JSON-RPC.
        let first = loop {
            match next_frame(&mut ws).await {
                Message::Text(t) => break serde_json::from_str::<serde_json::Value>(&t).unwrap(),
                Message::Ping(_) | Message::Pong(_) => continue,
                other => panic!("unexpected frame: {other:?}"),
            }
        };
        assert!(first["error"].is_object(), "got {first}");
        assert_eq!(
            first["id"],
            serde_json::Value::Null,
            "an error that cannot be correlated to a request carries a null id"
        );
        assert_eq!(first["error"]["code"], -32700, "JSON-RPC parse error");

        // The connection is still usable.
        ws.send(Message::Text(initialize_request(1).to_string().into()))
            .await
            .unwrap();
        loop {
            match next_frame(&mut ws).await {
                Message::Text(t) => {
                    let v: serde_json::Value = serde_json::from_str(&t).unwrap();
                    if v["id"] == 1 {
                        assert_eq!(v["result"]["protocolVersion"], 1);
                        return;
                    }
                }
                Message::Ping(_) | Message::Pong(_) => continue,
                other => panic!("unexpected frame: {other:?}"),
            }
        }
    }

    /// A method nothing here registers a handler for, and whose params carry
    /// no `sessionId` field (so the SDK's session-scoped retry —
    /// `role/acp.rs:293-311`, keyed on `Dispatch::has_session_id` — never
    /// applies), falls straight through to JSON-RPC `method not found`.
    ///
    /// DEVIATION from the plan's expectation. The plan predicted the RFD's
    /// rule — `initialize` must come first — would be enforced, so that an
    /// unhandled method on a fresh connection is rejected *because* it is out
    /// of order. The SDK enforces no ordering at all: it has no notion of an
    /// initialized connection. What actually rejects it is that no handler is
    /// registered, so the dispatch loop falls through to `method not found` —
    /// and, as the second half of this test shows, the answer is identical
    /// once `initialize` has been done.
    ///
    /// This test originally used `session/new` as its unhandled example.
    /// Task 7 registered a `session/new` handler, so it was replaced with a
    /// fabricated method name: any real ACP method scoped to a session (e.g.
    /// `session/load`) carries `sessionId` in its own params and would hang
    /// on the retry instead of demonstrating this behaviour.
    #[tokio::test]
    async fn an_unimplemented_method_is_answered_with_method_not_found() {
        fn unknown_method_request(id: i64) -> serde_json::Value {
            serde_json::json!({
                "jsonrpc": "2.0",
                "id": id,
                "method": "totally/unknown",
                "params": {}
            })
        }

        let addr = spawn(ServeState::for_test(true)).await;

        let before = roundtrip(&addr, unknown_method_request(0)).await;
        assert_eq!(before["id"], 0);
        assert_eq!(
            before["error"]["code"], -32601,
            "no handler is registered for it, got {before}"
        );

        // Initializing first changes nothing: the rejection is about the
        // missing handler, not about the connection's state.
        let after = conversation(
            &addr,
            vec![initialize_request(0), unknown_method_request(1)],
        )
        .await;
        assert!(after[0]["result"].is_object(), "got {}", after[0]);
        assert_eq!(
            after[1]["error"]["code"], -32601,
            "ordering is not what rejected it, got {}",
            after[1]
        );
    }

    #[tokio::test]
    async fn binary_frames_are_ignored() {
        let addr = spawn(ServeState::for_test(true)).await;
        let mut ws = connect(&addr).await;

        ws.send(Message::Binary(vec![0xde, 0xad].into()))
            .await
            .unwrap();
        ws.send(Message::Text(initialize_request(0).to_string().into()))
            .await
            .unwrap();
        loop {
            match next_frame(&mut ws).await {
                Message::Text(t) => {
                    let v: serde_json::Value = serde_json::from_str(&t).unwrap();
                    assert_eq!(v["id"], 0, "the binary frame produced no reply of its own");
                    return;
                }
                Message::Ping(_) | Message::Pong(_) => continue,
                other => panic!("unexpected frame: {other:?}"),
            }
        }
    }

    /// The close arm of the frame filter, which every later task's connection
    /// lifetime rests on. A close frame must let the incoming stream end so
    /// the ACP connection finishes and the server drops its half — rather than
    /// the filter swallowing it and leaving the connection task parked forever.
    /// The 10s timeout is the real assertion here: a leak shows up as a hang.
    #[tokio::test]
    async fn a_close_frame_ends_the_connection() {
        let addr = spawn(ServeState::for_test(true)).await;
        let mut ws = connect(&addr).await;

        ws.send(Message::Close(None)).await.unwrap();

        loop {
            let next = tokio::time::timeout(std::time::Duration::from_secs(10), ws.next())
                .await
                .expect("server still holding the connection open after a close frame");
            match next {
                // The server closed its side: what this test is for.
                None => return,
                // axum echoes the close frame back before closing.
                Some(Ok(Message::Close(_))) => continue,
                // A transport error also means the connection is gone. What
                // this rules out is the server hanging on to it, not the
                // niceties of the closing handshake.
                Some(Err(_)) => return,
                Some(Ok(other)) => panic!("unexpected frame after close: {other:?}"),
            }
        }
    }

    #[tokio::test]
    async fn session_new_returns_a_session_id() {
        let state = ServeState::for_test(true);
        let addr = spawn(Arc::clone(&state)).await;
        let responses =
            conversation(&addr, vec![initialize_request(0), new_session_request(1)]).await;

        let session_id = responses[1]["result"]["sessionId"]
            .as_str()
            .expect("sessionId present");
        assert!(!session_id.is_empty());

        // The central new behaviour of this task: `session/new` must pin the
        // agent-side session id to the room profile the bearer token
        // resolved to (the fixture's `sa-acp-token` resolves to
        // `"developer"`). Asserting the exact mapping, not just that the map
        // is non-empty, so a swapped key/value or a write into the wrong map
        // fails this test.
        let profiles = state.session_room_profiles.lock().await;
        assert_eq!(
            profiles.get(session_id).map(String::as_str),
            Some("developer"),
            "session/new must pin the session to the token's room profile, got {profiles:?}"
        );
    }

    /// The `prompt` array of a text-only message.
    fn text_prompt(text: &str) -> serde_json::Value {
        serde_json::json!([{ "type": "text", "text": text }])
    }

    fn prompt_request(id: i64, session_id: &str, prompt: serde_json::Value) -> serde_json::Value {
        serde_json::json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": "session/prompt",
            "params": { "sessionId": session_id, "prompt": prompt }
        })
    }

    /// initialize → session/new → session/prompt on ONE connection.
    ///
    /// Returns the session id the agent minted, every `session/update`
    /// notification the turn emitted in arrival order, and the whole
    /// JSON-RPC reply to the prompt (so a caller can inspect either
    /// `result.stopReason` or `error`).
    ///
    /// The session id is returned rather than discarded so callers can
    /// assert against *that* session in the shared store, instead of
    /// against whatever session happens to be the only one there.
    ///
    /// `conversation` cannot be used here: it filters frames down to one
    /// request id and would drop exactly the notifications under test,
    /// turning a wrong ordering into a slow hang instead of a failure.
    async fn drive(
        addr: &str,
        prompt: serde_json::Value,
    ) -> (String, Vec<serde_json::Value>, serde_json::Value) {
        let mut ws = connect(addr).await;
        for request in [initialize_request(0), new_session_request(1)] {
            ws.send(Message::Text(request.to_string().into()))
                .await
                .unwrap();
        }

        let mut session_id: Option<String> = None;
        let mut updates = Vec::new();

        loop {
            let frame = next_frame(&mut ws).await;
            let Message::Text(t) = frame else { continue };
            let v: serde_json::Value = serde_json::from_str(&t).unwrap();

            if v["id"] == 1 {
                let id = v["result"]["sessionId"]
                    .as_str()
                    .expect("sessionId present")
                    .to_string();
                session_id = Some(id.clone());
                ws.send(Message::Text(
                    prompt_request(2, &id, prompt.clone()).to_string().into(),
                ))
                .await
                .unwrap();
            } else if v["method"] == "session/update" {
                assert_eq!(
                    v["params"]["sessionId"],
                    *session_id.as_ref().expect("a session exists by now"),
                    "an update must name the session it belongs to, got {v}"
                );
                updates.push(v["params"]["update"].clone());
            } else if v["id"] == 2 {
                return (
                    session_id.expect("the prompt was sent, so a session exists"),
                    updates,
                    v,
                );
            }
        }
    }

    /// Like `drive`, but answers any `session/request_permission` the
    /// agent sends with `option_id`. Returns the session updates, the
    /// final reply, and how many permission requests arrived.
    ///
    /// The count is the point: a test that only checks the outcome
    /// cannot tell "allowed without asking" from "asked and allowed",
    /// and those are the two things this feature is about.
    async fn drive_answering(
        addr: &str,
        prompt: serde_json::Value,
        option_id: &str,
    ) -> (Vec<serde_json::Value>, serde_json::Value, usize) {
        let mut ws = connect(addr).await;
        for request in [initialize_request(0), new_session_request(1)] {
            ws.send(Message::Text(request.to_string().into()))
                .await
                .unwrap();
        }

        let mut updates = Vec::new();
        let mut asked = 0usize;

        loop {
            let Message::Text(t) = next_frame(&mut ws).await else {
                continue;
            };
            let v: serde_json::Value = serde_json::from_str(&t).unwrap();

            if v["id"] == 1 {
                let id = v["result"]["sessionId"].as_str().unwrap().to_string();
                ws.send(Message::Text(
                    prompt_request(2, &id, prompt.clone()).to_string().into(),
                ))
                .await
                .unwrap();
            } else if v["method"] == "session/request_permission" {
                asked += 1;
                let answer = serde_json::json!({
                    "jsonrpc": "2.0",
                    "id": v["id"],
                    "result": {
                        "outcome": { "outcome": "selected", "optionId": option_id }
                    }
                });
                ws.send(Message::Text(answer.to_string().into()))
                    .await
                    .unwrap();
            } else if v["method"] == "session/update" {
                updates.push(v["params"]["update"].clone());
            } else if v["id"] == 2 {
                return (updates, v, asked);
            }
        }
    }

    /// `Provider::chat` returns a whole response, so the reply arrives as
    /// exactly one `agent_message_chunk` — one, not zero, and not a faked
    /// token stream.
    #[tokio::test]
    async fn prompt_answers_with_one_chunk_and_ends_the_turn() {
        let state = ServeState::for_test_scripted(
            true,
            vec![crate::provider::ChatResponse {
                text: Some("hello from the agent".to_string()),
                tool_calls: Vec::new(),
                stop_reason: None,
            }],
        );
        let addr = spawn(state).await;
        let (_session_id, updates, reply) = drive(&addr, text_prompt("hi")).await;

        let chunks: Vec<&str> = updates
            .iter()
            .filter(|u| u["sessionUpdate"] == "agent_message_chunk")
            .map(|u| u["content"]["text"].as_str().unwrap())
            .collect();
        assert_eq!(chunks, vec!["hello from the agent"], "got {updates:?}");
        assert_eq!(reply["result"]["stopReason"], "end_turn", "got {reply}");
    }

    /// A prompt turn must land in the same session store as every other
    /// transport: `session/prompt` delegates to `run_llm_turn`, so the
    /// user message and the reply are in the session's history afterwards.
    #[tokio::test]
    async fn prompt_history_lands_in_the_shared_session_store() {
        let state = ServeState::for_test_scripted(
            true,
            vec![crate::provider::ChatResponse {
                text: Some("hello from the agent".to_string()),
                tool_calls: Vec::new(),
                stop_reason: None,
            }],
        );
        let addr = spawn(Arc::clone(&state)).await;
        let (session_id, _updates, _reply) = drive(&addr, text_prompt("hi")).await;

        // Keyed on the session id the agent handed the client, not on
        // "whatever single entry exists": a handler that minted a session
        // id of its own — quietly starting a second conversation — would
        // still leave exactly one entry here and pass a looser assertion.
        let sessions = state.sessions.lock().await;
        let history = sessions
            .get(&session_id)
            .unwrap_or_else(|| panic!("no history under {session_id}, got {:?}", sessions.keys()));
        let texts: Vec<&str> = history
            .iter()
            .filter_map(|m| match m.parts.first() {
                Some(crate::provider::ContentPart::Text(t)) => Some(t.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(texts, vec!["hi", "hello from the agent"], "got {texts:?}");
    }

    /// `session/new`'s `cwd` is carried until the session is first
    /// persisted, then lands in the meta line. `session/list` will have
    /// nothing else to filter a project by, so if this regresses the
    /// listing is always empty.
    #[tokio::test]
    async fn a_new_sessions_cwd_reaches_the_store() {
        let state = ServeState::for_test_scripted(
            true,
            vec![crate::provider::ChatResponse {
                text: Some("ok".to_string()),
                tool_calls: Vec::new(),
                stop_reason: None,
            }],
        );
        let store = Arc::clone(&state.cross_device_session_store);
        let addr = spawn(state).await;

        let (session_id, _updates, _reply) = drive(&addr, text_prompt("hi")).await;

        let meta = store
            .session_header(&session_id)
            .map(|(m, _)| m)
            .expect("the turn persisted the session");
        assert_eq!(meta.cwd.as_deref(), Some(test_cwd()));
    }

    /// Zed sends every `@file` mention as a `resource_link` block, which
    /// the schema makes mandatory for all agents ("All agents MUST support
    /// resource links in prompts"). It is not capability-gated, so it will
    /// arrive, and it must reach the model rather than being dropped on the
    /// way — otherwise "explain @main.rs" reaches the provider as "explain".
    #[tokio::test]
    async fn a_resource_link_reaches_the_model() {
        let state = ServeState::for_test_scripted(
            true,
            vec![crate::provider::ChatResponse {
                text: Some("ok".to_string()),
                tool_calls: Vec::new(),
                stop_reason: None,
            }],
        );
        let addr = spawn(Arc::clone(&state)).await;
        let (session_id, _updates, reply) = drive(
            &addr,
            serde_json::json!([
                { "type": "text", "text": "explain" },
                {
                    "type": "resource_link",
                    "name": "main.rs",
                    "uri": "file:///work/proj/src/main.rs"
                }
            ]),
        )
        .await;
        assert_eq!(reply["result"]["stopReason"], "end_turn", "got {reply}");

        let sessions = state.sessions.lock().await;
        let history = sessions.get(&session_id).expect("history for the session");
        let Some(crate::provider::ContentPart::Text(user)) = history[0].parts.first() else {
            panic!("expected a text user message, got {:?}", history[0].parts);
        };
        assert!(user.contains("explain"), "got {user:?}");
        assert!(
            user.contains("main.rs"),
            "the reference is lost, got {user:?}"
        );
        assert!(
            user.contains("file:///work/proj/src/main.rs"),
            "the reference is lost, got {user:?}"
        );
    }

    /// Tool progress is reported as it happens, so a tool call must reach
    /// the client BEFORE the reply that depended on it — presence alone
    /// would also pass if everything were flushed at the end of the turn.
    #[tokio::test]
    async fn tool_calls_are_reported_before_the_reply() {
        // The fixture registers one tool named "echo"; script a turn that
        // calls it and then answers.
        let state = ServeState::for_test_scripted(
            true,
            vec![
                crate::provider::ChatResponse {
                    text: None,
                    tool_calls: vec![crate::provider::ToolCall {
                        id: "call-1".to_string(),
                        name: "echo".to_string(),
                        input: serde_json::json!({ "text": "ping" }),
                    }],
                    stop_reason: None,
                },
                crate::provider::ChatResponse {
                    text: Some("done".to_string()),
                    tool_calls: Vec::new(),
                    stop_reason: None,
                },
            ],
        );
        let addr = spawn(state).await;
        let (_session_id, updates, reply) = drive(&addr, text_prompt("use the tool")).await;

        let kinds: Vec<&str> = updates
            .iter()
            .map(|u| u["sessionUpdate"].as_str().unwrap())
            .collect();
        let started = kinds
            .iter()
            .position(|k| *k == "tool_call")
            .unwrap_or_else(|| panic!("no tool_call update, got {kinds:?}"));
        // Two `tool_call_update`s now, in order: the gate clearing the
        // call, then the executor finishing it. `tool_call` itself
        // fires before the gate, so it can only say `pending`.
        let mut updated = kinds
            .iter()
            .enumerate()
            .filter(|(_, k)| **k == "tool_call_update")
            .map(|(i, _)| i);
        let in_progress = updated
            .next()
            .unwrap_or_else(|| panic!("no tool_call_update, got {kinds:?}"));
        let completed = updated
            .next()
            .unwrap_or_else(|| panic!("only one tool_call_update, got {kinds:?}"));
        let first_chunk = kinds
            .iter()
            .position(|k| *k == "agent_message_chunk")
            .unwrap_or_else(|| panic!("no agent_message_chunk, got {kinds:?}"));
        assert!(
            started < in_progress && in_progress < completed && completed < first_chunk,
            "start, allowed, end, then reply, got {kinds:?}"
        );

        // The provider's own tool-call id is what ACP's toolCallId carries,
        // so a client can correlate every later update with the start.
        assert_eq!(updates[started]["toolCallId"], "call-1");
        assert_eq!(updates[started]["title"], "echo");
        // `Pending` is the schema's default, so it is omitted from the
        // wire rather than sent — a client seeing no status reads it as
        // pending. Absent is therefore the correct assertion here, and
        // anything else would mean the call claimed to be underway
        // before the gate had cleared it.
        assert!(
            updates[started]["status"].is_null(),
            "tool_call fires before the gate, so it must not claim a status: {}",
            updates[started]
        );
        assert_eq!(updates[in_progress]["toolCallId"], "call-1");
        assert_eq!(updates[in_progress]["status"], "in_progress");
        assert_eq!(updates[completed]["toolCallId"], "call-1");
        assert_eq!(updates[completed]["status"], "completed");
        assert_eq!(reply["result"]["stopReason"], "end_turn", "got {reply}");
    }

    /// Running out of tool rounds is an ordinary ending, not a broken agent.
    /// The budget is ten, which "search, read four files, edit two" reaches
    /// on its own, so this is the routine case — and ACP has a stop reason
    /// for exactly it. Answering a JSON-RPC error instead would put an
    /// internal-error dialog in front of the user *and* throw away the prose
    /// the model produced on the way, so both halves are pinned here.
    #[tokio::test]
    async fn exhausting_the_tool_budget_ends_the_turn_with_max_turn_requests() {
        // One scripted response per permitted round, each calling the
        // fixture's `echo` tool and saying something first. The count is
        // exact: an eleventh provider call would find the script empty and
        // fail the turn as a provider error, which this test would see.
        let script: Vec<crate::provider::ChatResponse> = (0..super::super::MAX_TOOL_ROUNDS)
            .map(|i| crate::provider::ChatResponse {
                text: Some(format!("step {i}")),
                tool_calls: vec![crate::provider::ToolCall {
                    id: format!("call-{i}"),
                    name: "echo".to_string(),
                    input: serde_json::json!({ "text": "ping" }),
                }],
                stop_reason: None,
            })
            .collect();
        let addr = spawn(ServeState::for_test_scripted(true, script)).await;
        let (_session_id, updates, reply) = drive(&addr, text_prompt("do a big refactor")).await;

        assert!(
            reply.get("error").is_none(),
            "a spent budget is not an error, got {reply}"
        );
        assert_eq!(
            reply["result"]["stopReason"], "max_turn_requests",
            "got {reply}"
        );

        // The work done before the budget ran out reaches the editor.
        let expected: String = (0..super::super::MAX_TOOL_ROUNDS)
            .map(|i| format!("step {i}"))
            .collect::<Vec<_>>()
            .join("\n\n");
        let chunks: Vec<&str> = updates
            .iter()
            .filter(|u| u["sessionUpdate"] == "agent_message_chunk")
            .map(|u| u["content"]["text"].as_str().unwrap())
            .collect();
        assert_eq!(chunks, vec![expected.as_str()], "got {chunks:?}");
    }

    /// A session id this connection never minted is answered — not queued
    /// on the SDK's session-scoped retry, and not treated as a fresh
    /// session.
    #[tokio::test]
    async fn prompt_for_an_unknown_session_is_an_error() {
        let addr = spawn(ServeState::for_test(true)).await;
        let responses = conversation(
            &addr,
            vec![
                initialize_request(0),
                prompt_request(1, "no-such-session", text_prompt("hi")),
            ],
        )
        .await;

        assert_eq!(
            responses[1]["error"]["code"], -32602,
            "an unknown sessionId is a bad parameter, got {}",
            responses[1]
        );
    }

    /// A provider failure is a JSON-RPC error carrying the cause, not a
    /// stop reason: none of ACP's stop reasons means "the agent broke",
    /// and `refusal` would tell the user the agent declined.
    #[tokio::test]
    async fn a_provider_failure_answers_with_an_error_carrying_the_cause() {
        // An empty script makes the stub provider fail the chat call.
        let state = ServeState::for_test_scripted(true, Vec::new());
        let addr = spawn(state).await;
        let (_session_id, updates, reply) = drive(&addr, text_prompt("hi")).await;

        assert!(
            !updates
                .iter()
                .any(|u| u["sessionUpdate"] == "agent_message_chunk"),
            "a failed turn has no reply to send, got {updates:?}"
        );
        assert!(reply["result"].is_null(), "got {reply}");
        assert_eq!(reply["error"]["code"], -32603, "got {reply}");
        let data = reply["error"]["data"].as_str().unwrap_or_default();
        assert!(
            data.contains("script exhausted"),
            "the error must carry the provider's cause, got {reply}"
        );
    }

    /// `session/cancel` carries no id: it is a notification, and gets no
    /// reply of its own. The reply it produces is the one to the prompt.
    fn cancel_notification(session_id: &str) -> serde_json::Value {
        serde_json::json!({
            "jsonrpc": "2.0",
            "method": "session/cancel",
            "params": { "sessionId": session_id }
        })
    }

    /// initialize → session/new → session/prompt on one connection, returning
    /// as soon as the prompt (request id 2) is on the wire.
    ///
    /// Unlike [`drive`], the socket comes back open and undrained, so the
    /// caller decides how the turn ends — by cancelling it or by vanishing.
    /// Only usable with a provider that does not answer, which is what makes
    /// "still in flight" true rather than merely likely.
    async fn prompt_in_flight(addr: &str) -> (TestSocket, String) {
        let mut ws = connect(addr).await;
        for request in [initialize_request(0), new_session_request(1)] {
            ws.send(Message::Text(request.to_string().into()))
                .await
                .unwrap();
        }

        loop {
            let Message::Text(t) = next_frame(&mut ws).await else {
                continue;
            };
            let v: serde_json::Value = serde_json::from_str(&t).unwrap();
            if v["id"] == 1 {
                let session_id = v["result"]["sessionId"]
                    .as_str()
                    .expect("sessionId present")
                    .to_string();
                ws.send(Message::Text(
                    prompt_request(2, &session_id, text_prompt("hang"))
                        .to_string()
                        .into(),
                ))
                .await
                .unwrap();
                return (ws, session_id);
            }
        }
    }

    /// The schema makes `cancelled` a MUST for a turn stopped by
    /// `session/cancel`, "even if the cancellation causes exceptions in
    /// underlying operations".
    ///
    /// The provider never returns, so the turn is unambiguously still running
    /// when the cancel arrives — which also makes this a test of *where* the
    /// turn runs. A turn awaited inside the dispatch loop would stop that loop
    /// from ever parsing this notification, and the reply would never come.
    #[tokio::test]
    async fn session_cancel_ends_the_turn_with_cancelled() {
        let (state, hanging) = ServeState::for_test_hanging(true);
        let addr = spawn(state).await;
        let (mut ws, session_id) = prompt_in_flight(&addr).await;

        // Cancelling before the turn reaches the provider would prove
        // nothing about tearing a live call down, so wait for it to get
        // there first.
        HangingChat::wait_for(&hanging.entered, 1, "the turn to reach the provider").await;

        ws.send(Message::Text(
            cancel_notification(&session_id).to_string().into(),
        ))
        .await
        .unwrap();

        loop {
            let Message::Text(t) = next_frame(&mut ws).await else {
                continue;
            };
            let v: serde_json::Value = serde_json::from_str(&t).unwrap();
            if v["id"] == 2 {
                assert_eq!(v["result"]["stopReason"], "cancelled", "got {v}");
                break;
            }
        }

        // The stop reason alone would also be satisfied by a handler that
        // answers `cancelled` and leaves the turn running.
        assert_eq!(
            hanging.dropped.load(std::sync::atomic::Ordering::SeqCst),
            1,
            "the provider call outlived the cancellation it reported"
        );
    }

    /// A client that leaves mid-turn must stop the turn: a tool loop still
    /// calling a provider with nobody listening spends money and can still
    /// write to the workspace.
    ///
    /// The assertion is the hanging provider's drop counter, not the server's
    /// liveness: a test that merely reconnects and finds the server healthy
    /// passes whether or not the turn is still burning tokens.
    async fn assert_a_disconnect_stops_the_turn(say_goodbye: bool) {
        let (state, hanging) = ServeState::for_test_hanging(true);
        let addr = spawn(state).await;
        let (mut ws, _session_id) = prompt_in_flight(&addr).await;

        HangingChat::wait_for(&hanging.entered, 1, "the turn to reach the provider").await;
        if say_goodbye {
            ws.send(Message::Close(None)).await.unwrap();
        }
        drop(ws);

        HangingChat::wait_for(
            &hanging.dropped,
            1,
            "the turn to stop after the client left",
        )
        .await;
    }

    /// The polite exit — the one an editor performs on quit.
    ///
    /// Neither this nor its rude sibling isolates *our* cancellation: both
    /// were measured passing with `cancel_when_exhausted`'s guard removed,
    /// because the SDK tears the turn down on its own either way (a clean EOF
    /// finishes `connect_to`'s foreground and drops the task actor; an
    /// abrupt one fails the actors outright). They pin the observable
    /// contract — a client that leaves stops paying for a turn — and
    /// `the_transport_cancels_when_the_frame_stream_ends` pins the mechanism
    /// this module adds on top.
    #[tokio::test]
    async fn a_closing_client_stops_an_in_flight_turn() {
        assert_a_disconnect_stops_the_turn(true).await;
    }

    /// The rude exit: a client whose process died.
    #[tokio::test]
    async fn a_vanishing_client_stops_an_in_flight_turn() {
        assert_a_disconnect_stops_the_turn(false).await;
    }

    /// The connection's cancellation must fire the moment the socket stops
    /// producing frames.
    ///
    /// Tested directly on [`cancel_when_exhausted`] rather than through a
    /// live connection, because an end-to-end test cannot tell this guard
    /// apart from the SDK's own teardown: at EOF `connect_to`'s foreground
    /// (`incoming_closed` then `drain_outgoing`) finishes and the task actor
    /// holding the turn is dropped with it, so a turn stops either way. What
    /// only the guard gives is the *timing* — before the drain, which a
    /// client that has stopped reading can hold up indefinitely.
    #[tokio::test]
    async fn the_transport_cancels_when_the_frame_stream_ends() {
        let cancel = CancellationToken::new();
        let mut frames = Box::pin(cancel_when_exhausted(
            futures_util::stream::iter(vec!["{}".to_string()]),
            cancel.clone(),
        ));

        assert_eq!(frames.next().await.as_deref(), Some("{}"));
        assert!(
            !cancel.is_cancelled(),
            "a connection still delivering frames must not be cancelled"
        );

        assert!(frames.next().await.is_none(), "the stream ends here");
        assert!(
            cancel.is_cancelled(),
            "the end of the frame stream must cancel the connection's turns"
        );
    }

    /// `session/cancel` names a *session*, not a request — the schema scopes
    /// it to "ongoing operations for a session" — and prompts on one
    /// connection now run concurrently, so a session can have two turns open
    /// at once. Both must answer `cancelled`.
    ///
    /// Keeping only the newest turn's token would leave the older one calling
    /// the provider until the connection died, and then answering `end_turn`:
    /// the one path in this design where a cancelled turn does not report
    /// `Cancelled`, which the schema makes a MUST.
    #[tokio::test]
    async fn session_cancel_ends_every_turn_on_the_session() {
        let (state, hanging) = ServeState::for_test_hanging(true);
        let addr = spawn(state).await;
        // Leaves prompt id 2 in flight.
        let (mut ws, session_id) = prompt_in_flight(&addr).await;
        HangingChat::wait_for(&hanging.entered, 1, "the first turn to reach the provider").await;

        // A second prompt on the SAME session while the first still runs.
        ws.send(Message::Text(
            prompt_request(3, &session_id, text_prompt("hang as well"))
                .to_string()
                .into(),
        ))
        .await
        .unwrap();
        HangingChat::wait_for(&hanging.entered, 2, "the second turn to reach the provider").await;

        // One cancel, naming the session both turns belong to.
        ws.send(Message::Text(
            cancel_notification(&session_id).to_string().into(),
        ))
        .await
        .unwrap();

        let mut cancelled = Vec::new();
        while cancelled.len() < 2 {
            let Message::Text(t) = next_frame(&mut ws).await else {
                continue;
            };
            let v: serde_json::Value = serde_json::from_str(&t).unwrap();
            if v["id"] == 2 || v["id"] == 3 {
                assert_eq!(
                    v["result"]["stopReason"], "cancelled",
                    "every turn on a cancelled session answers cancelled, got {v}"
                );
                cancelled.push(v["id"].clone());
            }
        }
        cancelled.sort_by_key(|id| id.as_i64().unwrap_or_default());
        assert_eq!(cancelled, vec![serde_json::json!(2), serde_json::json!(3)]);

        assert_eq!(
            hanging.dropped.load(std::sync::atomic::Ordering::SeqCst),
            2,
            "both provider calls must be abandoned, not just the newest"
        );
    }

    /// Scripts a turn that calls `risky` once, then replies.
    fn risky_then_reply(reply: &str) -> Vec<crate::provider::ChatResponse> {
        vec![
            crate::provider::ChatResponse {
                text: None,
                tool_calls: vec![crate::provider::ToolCall {
                    id: "call-1".to_string(),
                    name: "risky".to_string(),
                    input: serde_json::json!({}),
                }],
                stop_reason: None,
            },
            crate::provider::ChatResponse {
                text: Some(reply.to_string()),
                tool_calls: Vec::new(),
                stop_reason: None,
            },
        ]
    }

    /// An `Execute` tool in the default mode puts the question to the
    /// user, and an allow lets it run.
    #[tokio::test]
    async fn an_execute_tool_asks_and_runs_when_allowed() {
        let state = ServeState::for_test_scripted(true, risky_then_reply("done"));
        let risky = super::super::RiskyTool::new();
        let ran = risky.ran_flag();
        state.tools.register_tool(Box::new(risky)).await;
        let addr = spawn(state).await;

        let (_updates, reply, asked) =
            drive_answering(&addr, text_prompt("run it"), "allow_once").await;

        assert_eq!(asked, 1, "exactly one permission request, got {asked}");
        assert!(
            ran.load(std::sync::atomic::Ordering::SeqCst),
            "an allowed tool must actually run"
        );
        assert_eq!(reply["result"]["stopReason"], "end_turn", "got {reply}");
    }

    /// A refusal does not end the turn: the model gets a tool_result
    /// saying so and answers normally. Showing the user an error dialog
    /// because they declined would be wrong twice over — the agent is
    /// fine, and they already know what they chose.
    #[tokio::test]
    async fn a_refusal_does_not_end_the_turn() {
        let state = ServeState::for_test_scripted(true, risky_then_reply("understood"));
        let risky = super::super::RiskyTool::new();
        let ran = risky.ran_flag();
        state.tools.register_tool(Box::new(risky)).await;
        let addr = spawn(state).await;

        let (updates, reply, asked) =
            drive_answering(&addr, text_prompt("run it"), "reject_once").await;

        assert_eq!(asked, 1);
        assert!(
            !ran.load(std::sync::atomic::Ordering::SeqCst),
            "a declined tool must not run"
        );
        assert_eq!(
            reply["result"]["stopReason"], "end_turn",
            "a declined tool is not a failed turn, got {reply}"
        );
        let chunks: Vec<&str> = updates
            .iter()
            .filter(|u| u["sessionUpdate"] == "agent_message_chunk")
            .filter_map(|u| u["content"]["text"].as_str())
            .collect();
        assert_eq!(chunks, vec!["understood"]);
    }

    /// A `Read` tool is never put to the user. This is what keeps the
    /// feature usable: a dialog per `file_read` would be intolerable.
    #[tokio::test]
    async fn a_safe_tool_is_not_put_to_the_user() {
        let state = ServeState::for_test_scripted(
            true,
            vec![
                crate::provider::ChatResponse {
                    text: None,
                    tool_calls: vec![crate::provider::ToolCall {
                        id: "call-1".to_string(),
                        name: "echo".to_string(),
                        input: serde_json::json!({ "text": "ping" }),
                    }],
                    stop_reason: None,
                },
                crate::provider::ChatResponse {
                    text: Some("done".to_string()),
                    tool_calls: Vec::new(),
                    stop_reason: None,
                },
            ],
        );
        let addr = spawn(state).await;

        let (_updates, reply, asked) =
            drive_answering(&addr, text_prompt("echo"), "allow_once").await;

        assert_eq!(asked, 0, "a Read tool must not ask");
        assert_eq!(reply["result"]["stopReason"], "end_turn");
    }

    /// `allow_always` is recorded, so a second call in the same turn
    /// uses the standing answer instead of asking again.
    #[tokio::test]
    async fn allow_always_is_not_asked_twice() {
        let state = ServeState::for_test_scripted(
            true,
            vec![
                crate::provider::ChatResponse {
                    text: None,
                    tool_calls: vec![crate::provider::ToolCall {
                        id: "call-1".to_string(),
                        name: "risky".to_string(),
                        input: serde_json::json!({}),
                    }],
                    stop_reason: None,
                },
                crate::provider::ChatResponse {
                    text: None,
                    tool_calls: vec![crate::provider::ToolCall {
                        id: "call-2".to_string(),
                        name: "risky".to_string(),
                        input: serde_json::json!({}),
                    }],
                    stop_reason: None,
                },
                crate::provider::ChatResponse {
                    text: Some("done".to_string()),
                    tool_calls: Vec::new(),
                    stop_reason: None,
                },
            ],
        );
        state
            .tools
            .register_tool(Box::new(super::super::RiskyTool::new()))
            .await;
        let addr = spawn(state).await;

        let (_updates, reply, asked) =
            drive_answering(&addr, text_prompt("run it twice"), "allow_always").await;

        assert_eq!(asked, 1, "the second call must use the recorded answer");
        assert_eq!(reply["result"]["stopReason"], "end_turn", "got {reply}");
    }

    fn set_mode_request(id: i64, session_id: &str, mode_id: &str) -> serde_json::Value {
        serde_json::json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": "session/set_mode",
            "params": { "sessionId": session_id, "modeId": mode_id }
        })
    }

    /// The client learns the modes when the session is created, and
    /// starts in the one that asks.
    #[tokio::test]
    async fn session_new_advertises_the_three_modes() {
        let state = ServeState::for_test_scripted(true, Vec::new());
        let addr = spawn(state).await;
        let replies =
            conversation(&addr, vec![initialize_request(0), new_session_request(1)]).await;

        let modes = &replies[1]["result"]["modes"];
        assert_eq!(modes["currentModeId"], "default", "got {modes}");
        let ids: Vec<&str> = modes["availableModes"]
            .as_array()
            .expect("availableModes is an array")
            .iter()
            .map(|m| m["id"].as_str().unwrap())
            .collect();
        assert_eq!(ids, vec!["default", "accept_edits", "bypass"]);
        // A picker with no labels is not a picker.
        assert!(
            modes["availableModes"][0]["name"]
                .as_str()
                .is_some_and(|n| !n.is_empty()),
            "each mode needs a human-readable name, got {modes}"
        );
    }

    /// Switching modes is acknowledged and announced.
    #[tokio::test]
    async fn set_mode_switches_and_notifies() {
        let state = ServeState::for_test_scripted(true, Vec::new());
        let addr = spawn(state).await;

        let mut ws = connect(&addr).await;
        for request in [initialize_request(0), new_session_request(1)] {
            ws.send(Message::Text(request.to_string().into()))
                .await
                .unwrap();
        }

        let mut saw_mode_update = false;
        loop {
            let Message::Text(t) = next_frame(&mut ws).await else {
                continue;
            };
            let v: serde_json::Value = serde_json::from_str(&t).unwrap();
            if v["id"] == 1 {
                let id = v["result"]["sessionId"].as_str().unwrap().to_string();
                ws.send(Message::Text(
                    set_mode_request(2, &id, "bypass").to_string().into(),
                ))
                .await
                .unwrap();
            } else if v["method"] == "session/update"
                && v["params"]["update"]["sessionUpdate"] == "current_mode_update"
            {
                assert_eq!(v["params"]["update"]["currentModeId"], "bypass");
                saw_mode_update = true;
            } else if v["id"] == 2 {
                assert!(v["error"].is_null(), "set_mode failed: {v}");
                break;
            }
        }
        assert!(saw_mode_update, "a mode change must be announced");
    }

    /// `plan` is not implemented, and must not silently resolve to
    /// something else — a client told "fine" would believe the agent is
    /// planning when it is about to act.
    ///
    /// Hand-rolled rather than via `roundtrip`, because a session lives
    /// only as long as the connection that minted it: a second
    /// connection would fail on the session id, not on the mode id, and
    /// the test would pass for the wrong reason.
    #[tokio::test]
    async fn an_unknown_mode_is_invalid_params() {
        let state = ServeState::for_test_scripted(true, Vec::new());
        let addr = spawn(state).await;

        let mut ws = connect(&addr).await;
        for request in [initialize_request(0), new_session_request(1)] {
            ws.send(Message::Text(request.to_string().into()))
                .await
                .unwrap();
        }

        loop {
            let Message::Text(t) = next_frame(&mut ws).await else {
                continue;
            };
            let v: serde_json::Value = serde_json::from_str(&t).unwrap();
            if v["id"] == 1 {
                let id = v["result"]["sessionId"].as_str().unwrap().to_string();
                ws.send(Message::Text(
                    set_mode_request(2, &id, "plan").to_string().into(),
                ))
                .await
                .unwrap();
            } else if v["id"] == 2 {
                assert_eq!(v["error"]["code"], -32602, "got {v}");
                assert!(
                    v["error"]["data"]
                        .as_str()
                        .is_some_and(|d| d.contains("plan")),
                    "the error should name the mode it rejected, got {v}"
                );
                break;
            }
        }
    }

    /// The mode is not decoration: `bypass` stops the asking.
    #[tokio::test]
    async fn bypass_mode_does_not_ask() {
        let state = ServeState::for_test_scripted(true, risky_then_reply("done"));
        let risky = super::super::RiskyTool::new();
        let ran = risky.ran_flag();
        state.tools.register_tool(Box::new(risky)).await;
        let addr = spawn(state).await;

        let mut ws = connect(&addr).await;
        for request in [initialize_request(0), new_session_request(1)] {
            ws.send(Message::Text(request.to_string().into()))
                .await
                .unwrap();
        }

        loop {
            let Message::Text(t) = next_frame(&mut ws).await else {
                continue;
            };
            let v: serde_json::Value = serde_json::from_str(&t).unwrap();
            if v["id"] == 1 {
                let id = v["result"]["sessionId"].as_str().unwrap().to_string();
                ws.send(Message::Text(
                    set_mode_request(2, &id, "bypass").to_string().into(),
                ))
                .await
                .unwrap();
                ws.send(Message::Text(
                    prompt_request(3, &id, text_prompt("run it"))
                        .to_string()
                        .into(),
                ))
                .await
                .unwrap();
            } else if v["method"] == "session/request_permission" {
                panic!("bypass mode must not ask, got {v}");
            } else if v["id"] == 3 {
                assert_eq!(v["result"]["stopReason"], "end_turn", "got {v}");
                break;
            }
        }
        assert!(
            ran.load(std::sync::atomic::Ordering::SeqCst),
            "bypass should have run the tool, not merely skipped asking"
        );
    }

    /// `accept_edits` is the middle mode, and the only one whose whole
    /// point is that it changes *some* answers and not others. An
    /// `Execute` tool must still be asked about there.
    #[tokio::test]
    async fn accept_edits_still_asks_about_commands() {
        let state = ServeState::for_test_scripted(true, risky_then_reply("done"));
        state
            .tools
            .register_tool(Box::new(super::super::RiskyTool::new()))
            .await;
        let addr = spawn(state).await;

        let mut ws = connect(&addr).await;
        for request in [initialize_request(0), new_session_request(1)] {
            ws.send(Message::Text(request.to_string().into()))
                .await
                .unwrap();
        }

        let mut asked = 0usize;
        loop {
            let Message::Text(t) = next_frame(&mut ws).await else {
                continue;
            };
            let v: serde_json::Value = serde_json::from_str(&t).unwrap();
            if v["id"] == 1 {
                let id = v["result"]["sessionId"].as_str().unwrap().to_string();
                ws.send(Message::Text(
                    set_mode_request(2, &id, "accept_edits").to_string().into(),
                ))
                .await
                .unwrap();
                ws.send(Message::Text(
                    prompt_request(3, &id, text_prompt("run it"))
                        .to_string()
                        .into(),
                ))
                .await
                .unwrap();
            } else if v["method"] == "session/request_permission" {
                asked += 1;
                let answer = serde_json::json!({
                    "jsonrpc": "2.0",
                    "id": v["id"],
                    "result": { "outcome": { "outcome": "selected", "optionId": "allow_once" } }
                });
                ws.send(Message::Text(answer.to_string().into()))
                    .await
                    .unwrap();
            } else if v["id"] == 3 {
                assert_eq!(v["result"]["stopReason"], "end_turn", "got {v}");
                break;
            }
        }
        assert_eq!(
            asked, 1,
            "accept_edits must still ask about an Execute tool"
        );
    }

    fn list_request(id: i64, cwd: Option<&str>) -> serde_json::Value {
        let params = match cwd {
            Some(cwd) => serde_json::json!({ "cwd": cwd }),
            None => serde_json::json!({}),
        };
        serde_json::json!({
            "jsonrpc": "2.0", "id": id, "method": "session/list", "params": params
        })
    }

    /// The boundary. A token pinned to one room profile must not see
    /// another profile's conversations, and a file too old to say which
    /// namespace it belongs to is not shown either — an unknown owner is
    /// not the same as "mine".
    #[tokio::test]
    async fn list_only_returns_this_namespaces_sessions() {
        let state = ServeState::for_test_scripted(true, Vec::new());
        let store = Arc::clone(&state.cross_device_session_store);
        let ours = state
            .config
            .namespace_for_room_profile("developer")
            .to_string();

        let mine = store
            .create_session(&("r-mine".to_string(), None), "rpc", &ours)
            .unwrap();
        let theirs = store
            .create_session(&("r-theirs".to_string(), None), "rpc", "someone-else")
            .unwrap();

        let addr = spawn(state).await;
        let replies = conversation(&addr, vec![initialize_request(0), list_request(1, None)]).await;

        let ids: Vec<&str> = replies[1]["result"]["sessions"]
            .as_array()
            .expect("sessions is an array")
            .iter()
            .map(|s| s["sessionId"].as_str().unwrap())
            .collect();
        assert!(ids.contains(&mine.as_str()), "got {ids:?}");
        assert!(
            !ids.contains(&theirs.as_str()),
            "another namespace leaked into the list: {ids:?}"
        );
        assert_eq!(replies[1]["result"]["nextCursor"], serde_json::Value::Null);
    }

    /// A file written before `namespace` existed cannot say whose it is.
    /// An unknown owner is not the same as "mine", so it is not listed.
    /// Written by hand because `create_session` always records one.
    #[tokio::test]
    async fn list_omits_sessions_with_no_namespace() {
        let state = ServeState::for_test_scripted(true, Vec::new());
        let store = Arc::clone(&state.cross_device_session_store);
        let ours = state
            .config
            .namespace_for_room_profile("developer")
            .to_string();
        let mine = store
            .create_session(&("r-mine".to_string(), None), "rpc", &ours)
            .unwrap();

        // A legacy meta line: no `namespace` key at all.
        let legacy_dir = store
            .absolute_path_for(&mine)
            .unwrap()
            .parent()
            .unwrap()
            .to_path_buf();
        let legacy = legacy_dir.join("00000000-0000-7000-8000-00000000dead.jsonl");
        std::fs::write(
            &legacy,
            format!(
                "{}\n",
                serde_json::json!({"meta": {
                    "session_id": "00000000-0000-7000-8000-00000000dead",
                    "room_id": "r-legacy",
                    "thread_id": null,
                    "channel": "rpc",
                    "created_at": "2020-01-01T00:00:00Z"
                }})
            ),
        )
        .unwrap();

        let addr = spawn(state).await;
        let replies = conversation(&addr, vec![initialize_request(0), list_request(1, None)]).await;

        let ids: Vec<&str> = replies[1]["result"]["sessions"]
            .as_array()
            .unwrap()
            .iter()
            .map(|s| s["sessionId"].as_str().unwrap())
            .collect();
        assert_eq!(
            ids,
            vec![mine.as_str()],
            "a namespace-less file leaked: {ids:?}"
        );
    }

    /// A closed session is archived, not current. Listing it would offer
    /// the user a thread the agent has already summarised and moved on
    /// from.
    #[tokio::test]
    async fn list_omits_closed_sessions() {
        let state = ServeState::for_test_scripted(true, Vec::new());
        let store = Arc::clone(&state.cross_device_session_store);
        let ours = state
            .config
            .namespace_for_room_profile("developer")
            .to_string();

        let open = store
            .create_session(&("r-open".to_string(), None), "rpc", &ours)
            .unwrap();
        let closed = store
            .create_session(&("r-closed".to_string(), None), "rpc", &ours)
            .unwrap();
        store.close_session(&closed).unwrap();

        let addr = spawn(state).await;
        let replies = conversation(&addr, vec![initialize_request(0), list_request(1, None)]).await;

        let ids: Vec<&str> = replies[1]["result"]["sessions"]
            .as_array()
            .unwrap()
            .iter()
            .map(|s| s["sessionId"].as_str().unwrap())
            .collect();
        assert_eq!(ids, vec![open.as_str()], "got {ids:?}");
    }

    /// The editor filters by project. A session with no recorded cwd
    /// belongs to no project, so it is absent when a cwd is asked for
    /// and present when one is not — which is how conversations from
    /// before cwd was recorded stay reachable.
    #[tokio::test]
    async fn list_honours_the_cwd_filter() {
        let state = ServeState::for_test_scripted(true, Vec::new());
        let store = Arc::clone(&state.cross_device_session_store);
        let ours = state
            .config
            .namespace_for_room_profile("developer")
            .to_string();

        let no_cwd = store
            .create_session(&("r-nocwd".to_string(), None), "rpc", &ours)
            .unwrap();
        let here = store
            .ensure_session(
                "s-here",
                &("r-here".to_string(), None),
                "rpc",
                None,
                &ours,
                Some("/projects/here".to_string()),
            )
            .map(|_| "s-here".to_string())
            .unwrap();
        store
            .ensure_session(
                "s-elsewhere",
                &("r-elsewhere".to_string(), None),
                "rpc",
                None,
                &ours,
                Some("/projects/elsewhere".to_string()),
            )
            .unwrap();

        let addr = spawn(state).await;
        let replies = conversation(
            &addr,
            vec![
                initialize_request(0),
                list_request(1, Some("/projects/here")),
                list_request(2, None),
            ],
        )
        .await;

        let ids = |r: &serde_json::Value| -> Vec<String> {
            r["result"]["sessions"]
                .as_array()
                .unwrap()
                .iter()
                .map(|s| s["sessionId"].as_str().unwrap().to_string())
                .collect()
        };

        assert_eq!(ids(&replies[1]), vec![here.clone()], "filtered by cwd");
        let all = ids(&replies[2]);
        assert!(
            all.contains(&no_cwd),
            "unfiltered must include the cwd-less session: {all:?}"
        );
        assert_eq!(all.len(), 3, "got {all:?}");
    }

    /// A session with no client-reported cwd (it predates the field, or
    /// came in over `/rpc`, voice or chat) still has to report an
    /// absolute path: `SessionInfo.cwd` is required and the schema
    /// documents it as absolute, so an empty string would be a contract
    /// violation dressed up as a null. The agent's own workspace root
    /// stands in for it — true, because the conversation belongs to the
    /// agent rather than to any editor project — which is also why it
    /// must never match a project-scoped `cwd` filter.
    #[tokio::test]
    async fn list_reports_the_workspace_root_for_a_session_with_no_cwd() {
        let state = ServeState::for_test_scripted(true, Vec::new());
        let store = Arc::clone(&state.cross_device_session_store);
        let ours = state
            .config
            .namespace_for_room_profile("developer")
            .to_string();
        let workspace_root = state.workspace.root();

        let no_cwd = store
            .create_session(&("r-nocwd".to_string(), None), "rpc", &ours)
            .unwrap();

        let addr = spawn(state).await;
        let replies = conversation(
            &addr,
            vec![
                initialize_request(0),
                list_request(1, None),
                list_request(2, Some("/projects/elsewhere")),
            ],
        )
        .await;

        let session = |r: &serde_json::Value| -> Option<serde_json::Value> {
            r["result"]["sessions"]
                .as_array()
                .unwrap()
                .iter()
                .find(|s| s["sessionId"].as_str() == Some(no_cwd.as_str()))
                .cloned()
        };

        let unfiltered = session(&replies[1]).expect("present when no cwd is requested");
        let reported_cwd = unfiltered["cwd"].as_str().expect("cwd is a string");
        assert!(
            std::path::Path::new(reported_cwd).is_absolute(),
            "got {reported_cwd:?}"
        );
        assert_eq!(
            PathBuf::from(reported_cwd),
            workspace_root,
            "must be the agent's own workspace root"
        );

        assert!(
            session(&replies[2]).is_none(),
            "absent when the request filters by an unrelated project directory"
        );
    }

    #[tokio::test]
    async fn initialize_advertises_listing() {
        let state = ServeState::for_test_scripted(true, Vec::new());
        let addr = spawn(state).await;
        let reply = roundtrip(&addr, initialize_request(0)).await;
        assert!(
            !reply["result"]["agentCapabilities"]["sessionCapabilities"]["list"].is_null(),
            "got {reply}"
        );
    }

    fn load_request(id: i64, session_id: &str) -> serde_json::Value {
        serde_json::json!({
            "jsonrpc": "2.0", "id": id, "method": "session/load",
            "params": { "sessionId": session_id, "cwd": test_cwd(), "mcpServers": [] }
        })
    }

    /// Loading replays the conversation as session/update notifications,
    /// and does so BEFORE answering — a client that got the reply first
    /// would render an empty thread and then have messages appear under
    /// it.
    #[tokio::test]
    async fn load_replays_the_conversation_before_replying() {
        let state = ServeState::for_test_scripted(true, Vec::new());
        let store = Arc::clone(&state.cross_device_session_store);
        let ours = state
            .config
            .namespace_for_room_profile("developer")
            .to_string();
        let sid = store
            .create_session(&("r".to_string(), None), "rpc", &ours)
            .unwrap();
        store
            .append(&sid, &crate::provider::ChatMessage::user("first"))
            .unwrap();
        store
            .append(&sid, &crate::provider::ChatMessage::assistant("second"))
            .unwrap();

        let addr = spawn(state).await;
        let mut ws = connect(&addr).await;
        for request in [initialize_request(0), load_request(1, &sid)] {
            ws.send(Message::Text(request.to_string().into()))
                .await
                .unwrap();
        }

        let mut replayed: Vec<(String, String)> = Vec::new();
        loop {
            let Message::Text(t) = next_frame(&mut ws).await else {
                continue;
            };
            let v: serde_json::Value = serde_json::from_str(&t).unwrap();
            if v["method"] == "session/update" {
                let u = &v["params"]["update"];
                replayed.push((
                    u["sessionUpdate"].as_str().unwrap().to_string(),
                    u["content"]["text"]
                        .as_str()
                        .unwrap_or_default()
                        .to_string(),
                ));
            } else if v["id"] == 1 {
                assert!(v["error"].is_null(), "load failed: {v}");
                break;
            }
        }

        assert_eq!(
            replayed,
            vec![
                ("user_message_chunk".to_string(), "first".to_string()),
                ("agent_message_chunk".to_string(), "second".to_string()),
            ],
            "the replay must arrive in order, before the reply"
        );
    }

    /// The boundary that filtering the list cannot provide: `load` takes
    /// an id directly, so a session that never appears in any list is
    /// still reachable by anyone who learns its id.
    #[tokio::test]
    async fn load_refuses_another_namespaces_session() {
        let state = ServeState::for_test_scripted(true, Vec::new());
        let store = Arc::clone(&state.cross_device_session_store);
        let theirs = store
            .create_session(&("r".to_string(), None), "rpc", "someone-else")
            .unwrap();

        let addr = spawn(state).await;
        let replies =
            conversation(&addr, vec![initialize_request(0), load_request(1, &theirs)]).await;
        assert_eq!(replies[1]["error"]["code"], -32602, "got {}", replies[1]);
    }

    /// The two refusals must be indistinguishable. If "not yours" reads
    /// differently from "no such session", the pair enumerates ids.
    #[tokio::test]
    async fn an_unknown_and_a_forbidden_session_look_the_same() {
        let state = ServeState::for_test_scripted(true, Vec::new());
        let store = Arc::clone(&state.cross_device_session_store);
        let theirs = store
            .create_session(&("r".to_string(), None), "rpc", "someone-else")
            .unwrap();

        let addr = spawn(state).await;
        let replies = conversation(
            &addr,
            vec![
                initialize_request(0),
                load_request(1, &theirs),
                load_request(2, "01900000-0000-7000-8000-000000000000"),
            ],
        )
        .await;

        assert_eq!(
            replies[1]["error"], replies[2]["error"],
            "the two refusals differ"
        );
    }

    /// A loaded session is a real session: prompting it continues the
    /// conversation rather than starting a new one. This is what proves
    /// the adapter registered the *existing* id, not a fresh one.
    #[tokio::test]
    async fn a_loaded_session_continues_its_history() {
        let state = ServeState::for_test_scripted(
            true,
            vec![crate::provider::ChatResponse {
                text: Some("third".to_string()),
                tool_calls: Vec::new(),
                stop_reason: None,
            }],
        );
        let store = Arc::clone(&state.cross_device_session_store);
        let ours = state
            .config
            .namespace_for_room_profile("developer")
            .to_string();
        let sid = store
            .create_session(&("r".to_string(), None), "rpc", &ours)
            .unwrap();
        store
            .append(&sid, &crate::provider::ChatMessage::user("first"))
            .unwrap();

        let addr = spawn(state).await;
        let mut ws = connect(&addr).await;
        for request in [
            initialize_request(0),
            load_request(1, &sid),
            prompt_request(2, &sid, text_prompt("second")),
        ] {
            ws.send(Message::Text(request.to_string().into()))
                .await
                .unwrap();
        }
        loop {
            let Message::Text(t) = next_frame(&mut ws).await else {
                continue;
            };
            let v: serde_json::Value = serde_json::from_str(&t).unwrap();
            if v["id"] == 2 {
                assert_eq!(v["result"]["stopReason"], "end_turn", "got {v}");
                break;
            }
        }

        let history = store.load_session(&sid).expect("the session still exists");
        let texts: Vec<String> = history
            .iter()
            .flat_map(|m| &m.parts)
            .filter_map(|p| match p {
                crate::provider::ContentPart::Text(t) => Some(t.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(texts, vec!["first", "second", "third"], "got {texts:?}");
    }

    #[tokio::test]
    async fn initialize_advertises_loading() {
        let state = ServeState::for_test_scripted(true, Vec::new());
        let addr = spawn(state).await;
        let reply = roundtrip(&addr, initialize_request(0)).await;
        assert_eq!(reply["result"]["agentCapabilities"]["loadSession"], true);
    }

    fn resume_request(id: i64, session_id: &str) -> serde_json::Value {
        serde_json::json!({
            "jsonrpc": "2.0", "id": id, "method": "session/resume",
            "params": { "sessionId": session_id, "cwd": test_cwd(), "mcpServers": [] }
        })
    }

    /// `resume` is `load` without the replay — for a client that wants
    /// the session back without paying to redraw a long conversation.
    /// It must still continue the history.
    #[tokio::test]
    async fn resume_continues_without_replaying() {
        let state = ServeState::for_test_scripted(
            true,
            vec![crate::provider::ChatResponse {
                text: Some("second".to_string()),
                tool_calls: Vec::new(),
                stop_reason: None,
            }],
        );
        let store = Arc::clone(&state.cross_device_session_store);
        let ours = state
            .config
            .namespace_for_room_profile("developer")
            .to_string();
        let sid = store
            .create_session(&("r".to_string(), None), "rpc", &ours)
            .unwrap();
        store
            .append(&sid, &crate::provider::ChatMessage::user("first"))
            .unwrap();

        let addr = spawn(state).await;
        let mut ws = connect(&addr).await;
        for request in [initialize_request(0), resume_request(1, &sid)] {
            ws.send(Message::Text(request.to_string().into()))
                .await
                .unwrap();
        }

        let mut updates_before_reply = 0usize;
        loop {
            let Message::Text(t) = next_frame(&mut ws).await else {
                continue;
            };
            let v: serde_json::Value = serde_json::from_str(&t).unwrap();
            if v["method"] == "session/update" {
                updates_before_reply += 1;
            } else if v["id"] == 1 {
                assert!(v["error"].is_null(), "resume failed: {v}");
                break;
            }
        }
        assert_eq!(updates_before_reply, 0, "resume must not replay");

        // ...and the history is still there for the next turn.
        ws.send(Message::Text(
            prompt_request(2, &sid, text_prompt("x")).to_string().into(),
        ))
        .await
        .unwrap();
        loop {
            let Message::Text(t) = next_frame(&mut ws).await else {
                continue;
            };
            let v: serde_json::Value = serde_json::from_str(&t).unwrap();
            if v["id"] == 2 {
                assert_eq!(v["result"]["stopReason"], "end_turn", "got {v}");
                break;
            }
        }
        let history = store.load_session(&sid).unwrap();
        assert!(history.len() >= 3, "the resumed session kept its history");
    }

    #[tokio::test]
    async fn initialize_advertises_resume() {
        let state = ServeState::for_test_scripted(true, Vec::new());
        let addr = spawn(state).await;
        let reply = roundtrip(&addr, initialize_request(0)).await;
        assert!(
            !reply["result"]["agentCapabilities"]["sessionCapabilities"]["resume"].is_null(),
            "got {reply}"
        );
    }

    /// A file too old to say whose it is refuses the same way an unknown
    /// or another namespace's session does. `session/list` already omits
    /// these (`list_omits_sessions_with_no_namespace`); `load` must not
    /// let a client reach one directly by id either — an unknown owner is
    /// not the same as "mine".
    #[tokio::test]
    async fn load_refuses_a_session_with_no_namespace() {
        let state = ServeState::for_test_scripted(true, Vec::new());
        let store = Arc::clone(&state.cross_device_session_store);
        let ours = state
            .config
            .namespace_for_room_profile("developer")
            .to_string();
        let mine = store
            .create_session(&("r-mine".to_string(), None), "rpc", &ours)
            .unwrap();

        // A legacy meta line: no `namespace` key at all. Same technique as
        // `list_omits_sessions_with_no_namespace`, since `create_session`
        // always records one.
        let legacy_dir = store
            .absolute_path_for(&mine)
            .unwrap()
            .parent()
            .unwrap()
            .to_path_buf();
        let legacy_id = "00000000-0000-7000-8000-00000000dead";
        let legacy = legacy_dir.join(format!("{legacy_id}.jsonl"));
        std::fs::write(
            &legacy,
            format!(
                "{}\n",
                serde_json::json!({"meta": {
                    "session_id": legacy_id,
                    "room_id": "r-legacy",
                    "thread_id": null,
                    "channel": "rpc",
                    "created_at": "2020-01-01T00:00:00Z"
                }})
            ),
        )
        .unwrap();

        let addr = spawn(state).await;
        let replies = conversation(
            &addr,
            vec![initialize_request(0), load_request(1, legacy_id)],
        )
        .await;
        assert_eq!(replies[1]["error"]["code"], -32602, "got {}", replies[1]);
    }

    /// Two connections can now hold the same session, which makes the
    /// history race in `run_llm_turn` reachable across connections
    /// rather than only within one. The race is not fixed here — this
    /// only makes hitting it observable, because a corrupted transcript
    /// with no log line is undebuggable.
    #[tokio::test]
    async fn opening_a_session_twice_is_counted() {
        let state = ServeState::for_test_scripted(true, Vec::new());
        let store = Arc::clone(&state.cross_device_session_store);
        let counts = Arc::clone(&state.open_acp_sessions);
        let ours = state
            .config
            .namespace_for_room_profile("developer")
            .to_string();
        let sid = store
            .create_session(&("r".to_string(), None), "rpc", &ours)
            .unwrap();

        let addr = spawn(state).await;
        let mut a = connect(&addr).await;
        let mut b = connect(&addr).await;
        for ws in [&mut a, &mut b] {
            for request in [initialize_request(0), load_request(1, &sid)] {
                ws.send(Message::Text(request.to_string().into()))
                    .await
                    .unwrap();
            }
            loop {
                let Message::Text(t) = next_frame(ws).await else {
                    continue;
                };
                let v: serde_json::Value = serde_json::from_str(&t).unwrap();
                if v["id"] == 1 {
                    assert!(v["error"].is_null(), "load failed: {v}");
                    break;
                }
            }
        }

        assert_eq!(
            counts.lock().await.get(&sid).copied(),
            Some(2),
            "both connections should be counted as holding the session"
        );
    }

    /// The other half of the counter: a connection that goes away must
    /// release every session it held. Skipping this would turn
    /// `open_acp_sessions` into a leak — the next `session/load` for a
    /// session whose real owner disconnected long ago would find a
    /// stale count still above one and warn about a collision that no
    /// longer exists.
    #[tokio::test]
    async fn closing_a_connection_releases_its_sessions() {
        let state = ServeState::for_test_scripted(true, Vec::new());
        let store = Arc::clone(&state.cross_device_session_store);
        let counts = Arc::clone(&state.open_acp_sessions);
        let ours = state
            .config
            .namespace_for_room_profile("developer")
            .to_string();
        let sid = store
            .create_session(&("r".to_string(), None), "rpc", &ours)
            .unwrap();

        let addr = spawn(state).await;
        let mut ws = connect(&addr).await;
        for request in [initialize_request(0), load_request(1, &sid)] {
            ws.send(Message::Text(request.to_string().into()))
                .await
                .unwrap();
        }
        loop {
            let Message::Text(t) = next_frame(&mut ws).await else {
                continue;
            };
            let v: serde_json::Value = serde_json::from_str(&t).unwrap();
            if v["id"] == 1 {
                assert!(v["error"].is_null(), "load failed: {v}");
                break;
            }
        }
        assert_eq!(
            counts.lock().await.get(&sid).copied(),
            Some(1),
            "load should have registered the session as open"
        );

        ws.send(Message::Close(None)).await.unwrap();
        drop(ws);

        let released = tokio::time::timeout(std::time::Duration::from_secs(10), async {
            loop {
                if counts.lock().await.get(&sid).is_none() {
                    break;
                }
                tokio::time::sleep(std::time::Duration::from_millis(5)).await;
            }
        })
        .await;
        assert!(
            released.is_ok(),
            "timed out waiting for the closed connection to release its session"
        );
    }
}
