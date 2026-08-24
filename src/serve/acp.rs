//! Agent Client Protocol over WebSocket, at `GET /acp`.
//!
//! Auth: `Authorization: Bearer <token>` on the upgrade request, matched
//! against `[room_profile.<n>].api_keys` — the same mechanism as `/a2a` and
//! `/mcp`. The match resolves the room profile the ACP session runs under,
//! so a dedicated Zed token can pin the editor to its own profile, provider
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

use super::{ServeState, extract_bearer};
use agent_client_protocol::schema::ProtocolVersion;
use agent_client_protocol::schema::v1::{
    AgentCapabilities, CancelNotification, ContentBlock, ContentChunk, Error, InitializeRequest,
    InitializeResponse, NewSessionRequest, NewSessionResponse, PromptRequest, PromptResponse,
    SessionId, SessionNotification, SessionUpdate, StopReason, TextContent,
    ToolCall as AcpToolCall, ToolCallId, ToolCallStatus, ToolCallUpdate, ToolCallUpdateFields,
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
}

impl AcpProgress {
    fn new(session_id: SessionId, connection: ConnectionTo<Client>) -> Self {
        Self {
            session_id,
            connection,
            error: std::sync::Mutex::new(None),
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
impl super::TurnProgress for AcpProgress {
    /// The provider's own tool-call id becomes ACP's `toolCallId`, so the
    /// completion below can name the call it completes. There is no input to
    /// report — `TurnProgress` does not carry one — so the tool's name serves
    /// as the title. `InProgress` rather than the default `Pending`: the
    /// executor has already started the call, whereas `Pending` tells a
    /// client the call is still waiting on input or approval.
    async fn tool_start(&self, id: &str, name: &str) {
        self.notify(SessionUpdate::ToolCall(
            AcpToolCall::new(ToolCallId::new(id), name).status(ToolCallStatus::InProgress),
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
    let Some(profile_name) = state.config.resolve_a2a_token(&bearer).map(str::to_string) else {
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
                // schema defines as the client's version if we support it and
                // otherwise the highest we do (`version.rs:26-32`). Handing
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

                // `loadSession` is false because `session/load` is not
                // implemented. `authMethods` is empty because the bearer token
                // checked above already authenticated the peer, so ACP never
                // sees an unauthenticated client.
                responder.respond(
                    InitializeResponse::new(version)
                        .agent_capabilities(AgentCapabilities::new().load_session(false)),
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

                    // Pin the session to the room profile the bearer token
                    // resolved to at connection time. That pin is what gives
                    // the ACP session its namespace chain and provider
                    // through the paths that already exist for `/rpc`.
                    state
                        .session_room_profiles
                        .lock()
                        .await
                        .insert(agent_session_id.clone(), profile_name.clone());

                    let session_id = SessionId::new(agent_session_id.clone());
                    sessions.inner.lock().await.insert(
                        session_id.clone(),
                        AcpSession {
                            agent_session_id,
                            cwd: req.cwd.clone(),
                            // No turn is running yet, so there is nothing for
                            // a `session/cancel` to reach.
                            turns: HashMap::new(),
                        },
                    );

                    responder.respond(NewSessionResponse::new(session_id))
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
                let state = Arc::clone(&state);
                let connection_cancel = connection_cancel.clone();
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
                            (session.agent_session_id.clone(), turn_cancel)
                        })
                    };
                    let Some((agent_session_id, turn_cancel)) = looked_up else {
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
                    let progress =
                        Arc::new(AcpProgress::new(session_id.clone(), connection.clone()));

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
                                    Arc::clone(&progress) as Arc<dyn super::TurnProgress>,
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
        assert_eq!(result["agentCapabilities"]["loadSession"], false);
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
        let completed = kinds
            .iter()
            .position(|k| *k == "tool_call_update")
            .unwrap_or_else(|| panic!("no tool_call_update, got {kinds:?}"));
        let first_chunk = kinds
            .iter()
            .position(|k| *k == "agent_message_chunk")
            .unwrap_or_else(|| panic!("no agent_message_chunk, got {kinds:?}"));
        assert!(
            started < completed && completed < first_chunk,
            "start then end then reply, got {kinds:?}"
        );

        // The provider's own tool-call id is what ACP's toolCallId carries,
        // so a client can correlate the completion with the start.
        assert_eq!(updates[started]["toolCallId"], "call-1");
        assert_eq!(updates[started]["title"], "echo");
        assert_eq!(updates[completed]["toolCallId"], "call-1");
        assert_eq!(updates[completed]["status"], "completed");
        assert_eq!(reply["result"]["stopReason"], "end_turn", "got {reply}");
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
}
