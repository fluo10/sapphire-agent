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
//! Only `initialize` is answered here; `session/new` and `session/prompt`
//! land in later tasks and currently come back as JSON-RPC
//! `method not found`.

use super::{ServeState, extract_bearer};
use agent_client_protocol::schema::ProtocolVersion;
use agent_client_protocol::schema::v1::{AgentCapabilities, InitializeRequest, InitializeResponse};
use agent_client_protocol::{Agent, Lines, on_receive_request};
use axum::extract::State;
use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use futures_util::{SinkExt, StreamExt};
use std::sync::Arc;
use tracing::{info, warn};

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
fn lines_transport(
    socket: WebSocket,
) -> Lines<
    impl futures_util::Sink<String, Error = std::io::Error> + Send + 'static,
    impl futures_util::Stream<Item = std::io::Result<String>> + Send + 'static,
> {
    let (tx, rx) = socket.split();

    let outgoing = tx
        .sink_map_err(std::io::Error::other)
        .with(|line: String| async move { Ok::<_, std::io::Error>(Message::Text(line.into())) });

    let incoming = rx.filter_map(|frame| async move {
        match frame {
            Ok(Message::Text(text)) => Some(Ok(text.to_string())),
            // Not ACP: no reply, and the connection carries on.
            Ok(Message::Binary(_) | Message::Ping(_) | Message::Pong(_)) => None,
            // The stream ends right after this, which closes the connection.
            Ok(Message::Close(_)) => None,
            Err(e) => Some(Err(std::io::Error::other(e))),
        }
    });

    Lines::new(outgoing, incoming)
}

/// Drive one ACP connection until the peer goes away.
async fn serve_connection(socket: WebSocket, state: Arc<ServeState>, profile_name: String) {
    // `initialize` needs neither, but the session methods added by the next
    // tasks need both, so they stay plumbed through.
    let _ = (&state, &profile_name);

    let result = Agent
        .builder()
        .name("sapphire-agent")
        .on_receive_request(
            async move |req: InitializeRequest, responder, _connection| {
                // Answer with the version we will actually speak: the lower of
                // what the client asked for and the highest this build
                // implements. Echoing the request back unclamped would tell a
                // v2 client it got v2, after which it sends v2-shaped traffic
                // nothing here understands. `ProtocolVersion` derives `Ord`
                // over its `u16`, so this is the schema's own ordering.
                //
                // `V1`, not `LATEST`: this is a claim about what *this* code
                // implements, and `LATEST` would silently start claiming v2
                // the day the SDK promotes it.
                let version = req.protocol_version.min(ProtocolVersion::V1);

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
        .connect_to(lines_transport(socket))
        .await;

    if let Err(e) = result {
        warn!("ACP: connection ended with an error: {e}");
    } else {
        info!("ACP: connection closed");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures_util::{SinkExt, StreamExt};
    use tokio::net::TcpListener;
    use tokio_tungstenite::tungstenite::Message;
    use tokio_tungstenite::tungstenite::client::IntoClientRequest;
    use tokio_tungstenite::{MaybeTlsStream, WebSocketStream};

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
    pub(super) async fn connect(
        addr: &str,
    ) -> WebSocketStream<MaybeTlsStream<tokio::net::TcpStream>> {
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

    fn new_session_request(id: i64) -> serde_json::Value {
        serde_json::json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": "session/new",
            "params": { "cwd": "/tmp", "mcpServers": [] }
        })
    }

    /// Read the next frame. A missing reply fails loudly instead of hanging
    /// the test run forever.
    async fn next_frame(
        ws: &mut WebSocketStream<MaybeTlsStream<tokio::net::TcpStream>>,
    ) -> Message {
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
    /// back. Answering "2" to a v2 client makes it send v2-shaped traffic that
    /// nothing here understands, and the failure surfaces far from its cause.
    #[tokio::test]
    async fn a_newer_client_is_negotiated_down_to_v1() {
        let addr = spawn(ServeState::for_test(true)).await;

        for asked in [2u16, 99] {
            let resp = roundtrip(&addr, initialize_request_asking_for(0, asked)).await;
            assert_eq!(
                resp["result"]["protocolVersion"], 1,
                "asked for {asked}, must be negotiated down to v1, got {resp}"
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

    /// DEVIATION from the plan's expectation. The plan predicted the RFD's
    /// rule — `initialize` must come first — would be enforced, so that
    /// `session/new` on a fresh connection is rejected *because* it is out of
    /// order. The SDK enforces no ordering at all: it has no notion of an
    /// initialized connection. What actually rejects `session/new` is that no
    /// handler is registered for it, so the dispatch loop falls through to
    /// `method not found` — and, as the second half of this test shows, the
    /// answer is identical once `initialize` has been done.
    ///
    /// Task 7 registers a `session/new` handler, at which point this test
    /// must be replaced: either by dropping the ordering claim, or by
    /// implementing the ordering check inside the handler.
    #[tokio::test]
    async fn an_unimplemented_method_is_answered_with_method_not_found() {
        let addr = spawn(ServeState::for_test(true)).await;

        let before = roundtrip(&addr, new_session_request(0)).await;
        assert_eq!(before["id"], 0);
        assert_eq!(
            before["error"]["code"], -32601,
            "no handler is registered yet, got {before}"
        );

        // Initializing first changes nothing: the rejection is about the
        // missing handler, not about the connection's state.
        let after = conversation(&addr, vec![initialize_request(0), new_session_request(1)]).await;
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
}
