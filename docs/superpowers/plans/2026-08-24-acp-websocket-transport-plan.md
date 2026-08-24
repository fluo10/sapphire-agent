# ACP over WebSocket (`/acp`) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Serve ACP from the already-running sapphire-agent at `GET /acp` over WebSocket, so Zed — via `websocat` — can hold a conversation with the production agent against the production workspace, memory and session store.

**Architecture:** A new `src/serve/acp.rs` sits beside `a2a.rs` and `mcp.rs` on the existing axum listener. It authenticates the WebSocket upgrade with the bearer scheme `/a2a` and `/mcp` already use, wraps the socket as the ACP SDK's `Lines` transport, and drives `agent_client_protocol::Agent`'s builder. `session/prompt` delegates to the existing `run_llm_turn`, whose progress reporting is refactored behind a `TurnProgress` trait so the same turn can emit either SSE events or ACP `session/update` notifications.

**Tech Stack:** Rust 2024, tokio, axum 0.8 (adding the `ws` feature), `agent-client-protocol` 2.0.0, `futures-util`, `tokio-tungstenite` (dev-only).

**Spec:** `docs/superpowers/specs/2026-08-24-acp-websocket-transport-design.md`

## Global Constraints

- **Protocol version is ACP v1.** `agent-client-protocol` 2.0.0 puts v2 behind the `unstable_protocol_v2` feature. Do **not** enable it. All schema types come from `agent_client_protocol::schema::v1`.
- **One WebSocket text frame per JSON-RPC message.** Binary frames are ignored, per the ACP transport RFD.
- **`initialize` is the first message** after the upgrade; any other method before it is a JSON-RPC error.
- **Authentication happens at the HTTP layer**, before the 101. `authMethods` in the `initialize` response is `[]`.
- **`agentCapabilities.loadSession` is `false`.** `session/load` is out of scope.
- **The endpoint is opt-in:** `[acp] enabled` defaults to `false`; disabled means HTTP 404, mirroring `/a2a`.
- **No new allowlist entry.** `src/config_layer.rs` is default-deny, so `[acp]` is host-local automatically. A test asserts this.
- **Every task ends with `cargo fmt` and `cargo clippy --all-targets -- -D warnings` passing** before the commit.
- Work happens on a feature branch **inside the `sapphire-agent` submodule**, branched from `main`.

## File Structure

| File | Responsibility |
|---|---|
| `src/serve/acp.rs` (new) | Upgrade handling, auth gate, WebSocket↔`Lines` adapter, ACP request handlers, `AcpProgress` |
| `src/serve/mod.rs` (modify) | Shared `extract_bearer`, the `TurnProgress` trait + `SseProgress`/`NullProgress`, `run_llm_turn` signature, `/acp` route, test fixture |
| `src/serve/a2a.rs`, `src/serve/mcp.rs` (modify) | Drop their private `extract_bearer` copies |
| `src/config.rs` (modify) | `AcpConfig`, `Config::acp` |
| `src/config_layer.rs` (test only) | Guard test that `acp.*` is host-only |
| `src/provider/registry.rs` (modify) | `#[cfg(test)] ProviderRegistry::for_test` |
| `src/main.rs` (modify) | `verify` prints the ACP endpoint state |
| `Cargo.toml` (modify) | axum `ws` feature, `agent-client-protocol`, dev-dep `tokio-tungstenite` |
| `config.example.toml`, `README.md` (modify) | `[acp]` block and the Zed + `websocat` setup |

**Task order note:** the test fixture (Task 3) comes before the endpoint (Task 4) because every endpoint test needs a `ServeState`, and `ServeState` has no test constructor today.

---

### Task 1: Lift `extract_bearer` into a shared helper

`extract_bearer` is copied verbatim into `src/serve/a2a.rs:369` and `src/serve/mcp.rs:590`. `/acp` would be the third copy, so it moves to `src/serve/mod.rs` first.

**Files:**
- Modify: `src/serve/mod.rs` (add the function and its tests)
- Modify: `src/serve/a2a.rs:369-378` (delete the copy), `src/serve/mcp.rs:590-599` (delete the copy)

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `pub(crate) fn extract_bearer(headers: &HeaderMap) -> Option<String>` in `crate::serve`.

- [ ] **Step 1: Write the failing tests**

Add to the existing `mod tests` at the bottom of `src/serve/mod.rs`:

```rust
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
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test --lib serve::tests::extract_bearer`
Expected: FAIL — `cannot find function 'extract_bearer' in this scope`.

- [ ] **Step 3: Add the shared function**

In `src/serve/mod.rs`, above the `// Router entry point` banner comment:

```rust
/// Extract a bearer token from an `Authorization` header.
///
/// Returns `None` when the header is absent, uses another scheme, or
/// carries an empty token — every one of which the endpoints treat as
/// "unauthenticated" rather than "malformed".
pub(crate) fn extract_bearer(headers: &HeaderMap) -> Option<String> {
    let s = headers.get("authorization")?.to_str().ok()?;
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
```

- [ ] **Step 4: Delete both copies and point the callers at the shared one**

In `src/serve/a2a.rs`, delete the `fn extract_bearer` definition and add `use super::extract_bearer;` to the imports at the top of the file. Do the same in `src/serve/mcp.rs`. If either file has tests referencing its local copy, they now exercise the shared one unchanged.

- [ ] **Step 5: Run the whole suite**

Run: `cargo test --lib`
Expected: PASS, with no `unused import` or `dead_code` warnings for the deleted copies.

- [ ] **Step 6: Format, lint and commit**

```bash
cargo fmt
cargo clippy --all-targets -- -D warnings
git add src/serve/mod.rs src/serve/a2a.rs src/serve/mcp.rs
git commit -m "refactor(serve): share one extract_bearer across the protocol endpoints"
```

---

### Task 2: `[acp]` config block

**Files:**
- Modify: `src/config.rs` (add `AcpConfig`, the `acp` field, and tests)
- Modify: `src/config_layer.rs` (guard test only)
- Modify: `src/main.rs` (one `verify` line)
- Modify: `config.example.toml`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `Config::acp: Option<AcpConfig>` where `pub struct AcpConfig { pub enabled: bool }`. Read as `config.acp.as_ref().is_some_and(|c| c.enabled)`.

- [ ] **Step 1: Write the failing tests**

Add to `mod tests` in `src/config.rs` (use the existing `parse` helper the other config tests use):

```rust
    #[test]
    fn acp_absent_means_disabled() {
        let cfg = parse(
            r#"
[anthropic]
api_key = "test"
"#,
        );
        assert!(cfg.acp.is_none());
        assert!(!cfg.acp.as_ref().is_some_and(|c| c.enabled));
    }

    #[test]
    fn acp_block_parses_and_defaults_to_disabled() {
        let cfg = parse(
            r#"
[anthropic]
api_key = "test"

[acp]
"#,
        );
        assert!(!cfg.acp.as_ref().expect("[acp] parsed").enabled);

        let cfg = parse(
            r#"
[anthropic]
api_key = "test"

[acp]
enabled = true
"#,
        );
        assert!(cfg.acp.as_ref().expect("[acp] parsed").enabled);
    }
```

Add to `mod tests` in `src/config_layer.rs`:

```rust
    #[test]
    fn acp_is_host_only() {
        // The endpoint a host exposes is a property of the host, not of the
        // shared workspace. The allowlist is default-deny, so this passes
        // without an entry — the test exists to catch someone adding one.
        assert!(!path_allowed(&["acp"]));
        assert!(!path_allowed(&["acp", "enabled"]));
    }
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test --lib config`
Expected: FAIL — `no field 'acp' on type 'Config'`. The `config_layer` test passes already; that is expected and is the point of it.

- [ ] **Step 3: Add the config type**

In `src/config.rs`, beside the other optional protocol blocks:

```rust
/// `[acp]` — the Agent Client Protocol endpoint at `GET /acp`.
///
/// Host-local by construction: `src/config_layer.rs` is default-deny, so the
/// workspace layer cannot turn an endpoint on for every host that syncs it.
#[derive(Debug, Clone, Deserialize, Default)]
pub struct AcpConfig {
    /// Serve `/acp`. Off by default; the endpoint 404s while disabled.
    #[serde(default)]
    pub enabled: bool,
}
```

and add the field to `struct Config`, next to `pub a2a: Option<A2aConfig>`:

```rust
    #[serde(default)]
    pub acp: Option<AcpConfig>,
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cargo test --lib config`
Expected: PASS.

- [ ] **Step 5: Report the endpoint from `verify`**

In `src/main.rs`, in the `Command::Verify` arm, after the line that prints `Channel`:

```rust
            println!(
                "  ACP endpoint      : {}",
                if config.acp.as_ref().is_some_and(|c| c.enabled) {
                    "enabled (GET /acp)"
                } else {
                    "disabled"
                }
            );
```

- [ ] **Step 6: Document the block**

Append to `config.example.toml`:

```toml
# Agent Client Protocol endpoint (GET /acp, WebSocket).
# Lets an ACP client such as Zed drive this agent. Authentication reuses the
# bearer tokens in [room_profile.<name>].api_keys, and the token that connects
# selects the room profile the ACP session runs under.
# Host-local: the workspace config layer cannot set this.
[acp]
enabled = false
```

- [ ] **Step 7: Verify the binary reports it**

Run: `cargo run -- verify`
Expected: the output contains `ACP endpoint      : disabled`.

- [ ] **Step 8: Format, lint and commit**

```bash
cargo fmt
cargo clippy --all-targets -- -D warnings
git add src/config.rs src/config_layer.rs src/main.rs config.example.toml
git commit -m "feat(config): add the host-local [acp] endpoint block"
```

---

### Task 3: A `ServeState` test fixture with a stub provider

Every endpoint and turn test below needs a `ServeState`, and there is no test constructor for one today: `ProviderRegistry`'s `providers` field is private and `from_config` always builds a real Anthropic provider. This task exists first so no later task has to invent a throwaway fixture.

**Files:**
- Modify: `src/provider/registry.rs` (test-only constructor)
- Modify: `src/serve/mod.rs` (`ServeState::for_test*`, `StubProvider`)

**Interfaces:**
- Consumes: `Config::acp` (Task 2).
- Produces:
  - `#[cfg(test)] ProviderRegistry::for_test(name: &str, provider: Arc<dyn Provider>) -> Self`
  - `#[cfg(test)] StubProvider::new(responses: Vec<ChatResponse>) -> Self` and `StubProvider::new_hanging() -> Self`
  - `#[cfg(test)] ServeState::for_test(acp_enabled: bool) -> Arc<Self>`
  - `#[cfg(test)] ServeState::for_test_scripted(acp_enabled: bool, responses: Vec<ChatResponse>) -> Arc<Self>`
  - `#[cfg(test)] ServeState::for_test_hanging(acp_enabled: bool) -> Arc<Self>`
  - The fixture's config always defines room profile `developer` with the single api_key `sa-acp-token`.

- [ ] **Step 1: Write the failing test**

Add to `mod tests` in `src/serve/mod.rs`:

```rust
    #[tokio::test]
    async fn fixture_state_serves_the_scripted_provider() {
        let state = ServeState::for_test_scripted(
            true,
            vec![ChatResponse {
                text: Some("scripted reply".to_string()),
                tool_calls: Vec::new(),
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
        assert_eq!(
            state.config.resolve_a2a_token("sa-acp-token"),
            Some("developer")
        );
        assert_eq!(state.config.resolve_a2a_token("sa-wrong"), None);
    }
```

Adjust the `ChatResponse` literal to that type's actual fields — read `src/provider/mod.rs` first and construct it exactly, including any fields not listed here.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test --lib serve::tests::fixture_state`
Expected: FAIL — `no function 'for_test_scripted'`.

- [ ] **Step 3: Write the registry constructor**

In `src/provider/registry.rs`:

```rust
#[cfg(test)]
impl ProviderRegistry {
    /// Registry holding exactly one provider, for tests that script replies
    /// instead of reaching a real API.
    pub(crate) fn for_test(name: &str, provider: Arc<dyn Provider>) -> Self {
        let mut providers: HashMap<String, Arc<dyn Provider>> = HashMap::new();
        providers.insert(name.to_string(), provider);
        Self { providers }
    }
}
```

- [ ] **Step 4: Write the stub provider**

In `src/serve/mod.rs`:

```rust
#[cfg(test)]
pub(crate) struct StubProvider {
    /// `None` means "never return" — used to keep a turn in flight while a
    /// cancellation test races it.
    script: Option<std::sync::Mutex<std::collections::VecDeque<crate::provider::ChatResponse>>>,
}

#[cfg(test)]
impl StubProvider {
    pub(crate) fn new(responses: Vec<crate::provider::ChatResponse>) -> Self {
        Self {
            script: Some(std::sync::Mutex::new(responses.into())),
        }
    }

    /// A provider whose `chat` never resolves.
    pub(crate) fn new_hanging() -> Self {
        Self { script: None }
    }
}

#[cfg(test)]
#[async_trait]
impl Provider for StubProvider {
    fn name(&self) -> &str {
        "stub"
    }

    async fn chat(
        &self,
        _system: Option<&str>,
        _messages: &[ChatMessage],
        _tools: Option<&[ToolSpec]>,
    ) -> anyhow::Result<crate::provider::ChatResponse> {
        let Some(script) = &self.script else {
            std::future::pending::<()>().await;
            unreachable!()
        };
        let next = script.lock().unwrap().pop_front();
        next.ok_or_else(|| anyhow::anyhow!("StubProvider script exhausted"))
    }
}
```

- [ ] **Step 5: Write the state fixture**

In `src/serve/mod.rs`:

```rust
#[cfg(test)]
impl ServeState {
    /// State backed by temp directories and a stub provider that answers "ok".
    pub(crate) fn for_test(acp_enabled: bool) -> Arc<Self> {
        Self::for_test_scripted(
            acp_enabled,
            vec![crate::provider::ChatResponse {
                text: Some("ok".to_string()),
                tool_calls: Vec::new(),
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
    pub(crate) fn for_test_hanging(acp_enabled: bool) -> Arc<Self> {
        Self::build_for_test(acp_enabled, StubProvider::new_hanging())
    }

    fn build_for_test(acp_enabled: bool, provider: StubProvider) -> Arc<Self> {
        // Leak the TempDir guard on purpose: this is a test binary and the
        // OS reclaims the directory when it exits.
        let dir = Box::leak(Box::new(tempfile::tempdir().unwrap()));
        let base = dir.path().to_path_buf();

        let mut config = Config::parse_str_for_test(
            r#"
[anthropic]
api_key = "test"

[profiles.dev]
provider = "stub"

[room_profile.developer]
profile  = "dev"
rooms    = []
api_keys = ["sa-acp-token"]
"#,
        );
        config.acp = Some(crate::config::AcpConfig {
            enabled: acp_enabled,
        });

        let registry = ProviderRegistry::for_test("stub", Arc::new(provider));

        Arc::new(Self {
            config,
            registry: Arc::new(registry),
            workspace: Arc::new(Workspace::open_for_test(base.join("workspace"))),
            tools: Arc::new(ToolSet::new(Vec::new(), Vec::new())),
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
        })
    }
}
```

Two helpers referenced above may not exist under those names — resolve each against the real code before writing:

- `Config::parse_str_for_test` — `mod tests` in `src/config.rs` already parses a TOML string through some helper. Reuse it, promoting it to `pub(crate)` behind `#[cfg(test)]` if it is private to that module.
- `Workspace::open_for_test` — check `src/workspace.rs` for how a `Workspace` is opened. If only an async or fallible constructor exists, make `build_for_test` and its three callers `async`, and add `.await` at every call site in later tasks.

Keep every `ServeState` field in the literal; the compiler will name any that changed.

- [ ] **Step 6: Run the tests to verify they pass**

Run: `cargo test --lib serve::tests::fixture_state`
Expected: PASS — both tests.

- [ ] **Step 7: Format, lint and commit**

```bash
cargo fmt
cargo clippy --all-targets -- -D warnings
cargo test --lib
git add src/provider/registry.rs src/serve/mod.rs
git commit -m "test(serve): add a ServeState fixture backed by a scripted provider"
```

---

### Task 4: `/acp` upgrade with the feature gate and auth gate

The endpoint accepts or refuses the WebSocket upgrade. It speaks no ACP yet — the socket closes immediately after a successful upgrade. This isolates every rejection path from the protocol work that follows.

**Files:**
- Create: `src/serve/acp.rs`
- Modify: `src/serve/mod.rs` (declare `pub mod acp;`, add the route)
- Modify: `Cargo.toml` (axum `ws` feature, dev-dep `tokio-tungstenite`)

**Interfaces:**
- Consumes: `extract_bearer` (Task 1), `Config::acp` (Task 2), `ServeState::for_test` (Task 3).
- Produces: `pub async fn handle_acp_ws(State<Arc<ServeState>>, HeaderMap, WebSocketUpgrade) -> Response` in `crate::serve::acp`, mounted at `GET /acp`, plus the test helpers `spawn` and `upgrade_status` that later tasks reuse.

- [ ] **Step 1: Add the dependencies**

In `Cargo.toml`, extend the axum line's features with `"ws"`:

```toml
axum = { version = "0.8", default-features = false, features = ["http1", "json", "tokio", "ws"] }
```

and add to `[dev-dependencies]`:

```toml
tokio-tungstenite = "0.28"
```

- [ ] **Step 2: Write the failing tests**

Create `src/serve/acp.rs` containing only this test module for now:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use tokio::net::TcpListener;

    /// Bind the router on an ephemeral port and return its `host:port`.
    pub(super) async fn spawn(state: Arc<ServeState>) -> String {
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
        use tokio_tungstenite::tungstenite::client::IntoClientRequest;
        let mut req = format!("ws://{addr}/acp").into_client_request().unwrap();
        if let Some(t) = token {
            req.headers_mut()
                .insert("authorization", format!("Bearer {t}").parse().unwrap());
        }
        match tokio_tungstenite::connect_async(req).await {
            Ok(_) => None,
            Err(tokio_tungstenite::tungstenite::Error::Http(resp)) => {
                Some(resp.status().as_u16())
            }
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
}
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `cargo test --lib serve::acp`
Expected: FAIL to compile — `handle_acp_ws` is undefined.

- [ ] **Step 4: Write the handler**

At the top of `src/serve/acp.rs`, above the test module:

```rust
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
//! The ACP transport RFD fixes the framing — one JSON-RPC message per
//! WebSocket text frame, binary frames ignored — which is exactly the
//! newline framing the SDK's `Lines` transport expects.

use super::{ServeState, extract_bearer};
use axum::extract::State;
use axum::extract::ws::{WebSocket, WebSocketUpgrade};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
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

/// Drive one ACP connection. Task 6 replaces the body with the SDK.
async fn serve_connection(socket: WebSocket, _state: Arc<ServeState>, _profile_name: String) {
    drop(socket);
}
```

Add to `src/serve/mod.rs`, beside `pub mod a2a;`:

```rust
pub mod acp;
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `cargo test --lib serve::acp`
Expected: PASS — three tests covering 404, 401 (missing), 401 (unknown) and 101.

- [ ] **Step 6: Mount the route**

In `src/serve/mod.rs`, in `run()`, add to the router chain after the `/mcp` line:

```rust
        .route("/acp", axum::routing::get(acp::handle_acp_ws))
```

- [ ] **Step 7: Format, lint and commit**

```bash
cargo fmt
cargo clippy --all-targets -- -D warnings
cargo test --lib
git add Cargo.toml Cargo.lock src/serve/acp.rs src/serve/mod.rs
git commit -m "feat(serve): gate and authenticate the /acp WebSocket upgrade"
```

---

### Task 5: Decouple turn progress from SSE

`run_llm_turn` takes `tx: mpsc::Sender<Result<Event, Infallible>>` plus a `req_id` and emits `tool_start` / `tool_end` as axum SSE events (`src/serve/mod.rs:1793` and `:1832`). ACP needs the same two events as `session/update` notifications, so reporting moves behind a trait.

`src/serve/mod.rs:1483` already calls `run_llm_turn` with a throwaway channel purely to discard these events; that call site becomes `NullProgress`.

**Files:**
- Modify: `src/serve/mod.rs` (trait, two impls, `run_llm_turn` signature, three call sites, tests)

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `pub(crate) trait TurnProgress: Send + Sync` with `async fn tool_start(&self, name: &str, input: &Value)` and `async fn tool_end(&self, name: &str, output: &str)`
  - `pub(crate) struct SseProgress` with `SseProgress::new(tx: mpsc::Sender<Result<Event, Infallible>>, req_id: Value) -> Self`
  - `pub(crate) struct NullProgress;`
  - `pub(crate) async fn run_llm_turn(state: Arc<ServeState>, session_id: String, user_msg: ChatMessage, progress: Arc<dyn TurnProgress>, timer_origin: Option<TimerOrigin>) -> LlmTurnOutcome` — `req_id` and `tx` are gone; both the function and `LlmTurnOutcome` become `pub(crate)` so `serve::acp` can call it in Task 8.

- [ ] **Step 1: Write the failing tests**

Add to `mod tests` in `src/serve/mod.rs`:

```rust
    #[tokio::test]
    async fn sse_progress_emits_the_same_events_as_before() {
        let (tx, mut rx) = mpsc::channel(8);
        let progress = SseProgress::new(tx, json!(7));

        progress.tool_start("recall", &json!({"q": "cats"})).await;
        progress.tool_end("recall", "two cats").await;
        drop(progress);

        let mut seen = Vec::new();
        while let Some(Ok(event)) = rx.recv().await {
            seen.push(format!("{event:?}"));
        }
        assert_eq!(seen.len(), 2, "one event per call");
        assert!(seen[0].contains("tool_start"), "got {}", seen[0]);
        assert!(seen[0].contains("recall"), "got {}", seen[0]);
        assert!(seen[1].contains("tool_end"), "got {}", seen[1]);
        assert!(seen[1].contains("two cats"), "got {}", seen[1]);
    }

    #[tokio::test]
    async fn null_progress_discards_without_panicking() {
        let progress = NullProgress;
        progress.tool_start("recall", &json!({})).await;
        progress.tool_end("recall", "ignored").await;
    }
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test --lib serve::tests::sse_progress`
Expected: FAIL — `cannot find type 'SseProgress'`.

- [ ] **Step 3: Add the trait and implementations**

In `src/serve/mod.rs`, above `run_llm_turn`:

```rust
/// Where a turn reports its per-tool progress.
///
/// `run_llm_turn` used to hold an SSE sender and a JSON-RPC request id
/// directly, which tied the turn executor to one transport. The ACP endpoint
/// needs the same two events shaped as `session/update` notifications, and
/// one existing caller wants them discarded entirely.
#[async_trait]
pub(crate) trait TurnProgress: Send + Sync {
    async fn tool_start(&self, name: &str, input: &Value);
    async fn tool_end(&self, name: &str, output: &str);
}

/// The `/rpc` and voice shape: JSON-RPC notifications over SSE.
pub(crate) struct SseProgress {
    tx: mpsc::Sender<Result<Event, Infallible>>,
    req_id: Value,
}

impl SseProgress {
    pub(crate) fn new(tx: mpsc::Sender<Result<Event, Infallible>>, req_id: Value) -> Self {
        Self { tx, req_id }
    }
}

#[async_trait]
impl TurnProgress for SseProgress {
    async fn tool_start(&self, name: &str, input: &Value) {
        let evt = notification_event(
            &self.req_id,
            "tool_start",
            json!({ "tool": name, "input": input }),
        );
        let _ = self.tx.send(Ok(evt)).await;
    }

    async fn tool_end(&self, name: &str, output: &str) {
        let evt = notification_event(
            &self.req_id,
            "tool_end",
            json!({ "tool": name, "output": output }),
        );
        let _ = self.tx.send(Ok(evt)).await;
    }
}

/// Discard progress. Used by the caller that runs a turn with nobody watching.
pub(crate) struct NullProgress;

#[async_trait]
impl TurnProgress for NullProgress {
    async fn tool_start(&self, _name: &str, _input: &Value) {}
    async fn tool_end(&self, _name: &str, _output: &str) {}
}
```

Read the two existing emission sites at `src/serve/mod.rs:1793` and `:1832` first and reproduce their exact JSON shape and helper (`notification_event` above is a placeholder for whatever they call). The point of this refactor is that the wire output does not change.

- [ ] **Step 4: Change `run_llm_turn` and its call sites**

Replace the `req_id: Value` and `tx: mpsc::Sender<Result<Event, Infallible>>` parameters with `progress: Arc<dyn TurnProgress>`, and mark both `run_llm_turn` and `LlmTurnOutcome` `pub(crate)`. Inside the body, replace the two emission blocks with `progress.tool_start(&name, &input).await;` and `progress.tool_end(&name, &output).await;`. Delete the `send` closure if nothing else uses it.

Update all three call sites: the two passing a real SSE channel become `Arc::new(SseProgress::new(tx.clone(), req_id.clone()))`, and the one at `src/serve/mod.rs:1483` becomes `Arc::new(NullProgress)` — deleting the throwaway channel it created.

- [ ] **Step 5: Run the full suite**

Run: `cargo test --lib`
Expected: PASS, including the two new tests.

- [ ] **Step 6: Format, lint and commit**

```bash
cargo fmt
cargo clippy --all-targets -- -D warnings
git add src/serve/mod.rs
git commit -m "refactor(serve): report turn progress through a TurnProgress sink"
```

---

### Task 6: `initialize` over the WebSocket transport

The first real ACP exchange. The socket is wrapped as the SDK's `Lines` transport — a `futures::Sink<String>` plus a `futures::Stream<io::Result<String>>`, which is exactly what one-JSON-RPC-message-per-text-frame gives us.

**Files:**
- Modify: `Cargo.toml` (`agent-client-protocol`)
- Modify: `src/serve/acp.rs`

**Interfaces:**
- Consumes: `serve_connection`, `spawn` (Task 4).
- Produces: `fn lines_transport(socket: WebSocket) -> Lines<...>`, a `serve_connection` that answers `initialize`, and the test helpers `roundtrip`, `conversation` and `initialize_request` that Tasks 7–9 reuse.

- [ ] **Step 1: Add the dependency**

```bash
cargo add agent-client-protocol@2.0.0
```

Confirm the manifest line has **no** `features = [...]`: `unstable_protocol_v2` must stay off, which is what pins this to ACP v1.

- [ ] **Step 2: Write the failing tests**

Add to `mod tests` in `src/serve/acp.rs`:

```rust
    use futures_util::{SinkExt, StreamExt};
    use tokio_tungstenite::tungstenite::Message;
    use tokio_tungstenite::tungstenite::client::IntoClientRequest;
    use tokio_tungstenite::{MaybeTlsStream, WebSocketStream};

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
        serde_json::json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": "initialize",
            "params": {
                "protocolVersion": 1,
                "clientCapabilities": {
                    "fs": { "readTextFile": true, "writeTextFile": true },
                    "terminal": true
                },
                "clientInfo": { "name": "test-client", "version": "0.0.0" }
            }
        })
    }

    /// Send one request on a fresh connection and read the reply to it.
    async fn roundtrip(addr: &str, request: serde_json::Value) -> serde_json::Value {
        let mut ws = connect(addr).await;
        let want_id = request["id"].clone();
        ws.send(Message::Text(request.to_string().into())).await.unwrap();
        loop {
            match ws.next().await.expect("stream ended").unwrap() {
                Message::Text(t) => {
                    let v: serde_json::Value = serde_json::from_str(&t).unwrap();
                    if v["id"] == want_id {
                        return v;
                    }
                }
                Message::Ping(_) | Message::Pong(_) => continue,
                other => panic!("unexpected frame: {other:?}"),
            }
        }
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
            ws.send(Message::Text(request.to_string().into())).await.unwrap();
            loop {
                match ws.next().await.expect("stream ended").unwrap() {
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

    #[tokio::test]
    async fn malformed_frame_errors_without_closing_the_connection() {
        let addr = spawn(ServeState::for_test(true)).await;
        let mut ws = connect(&addr).await;

        ws.send(Message::Text("{ this is not json".into())).await.unwrap();
        // A parse error carries a null id per JSON-RPC.
        let first = loop {
            match ws.next().await.expect("stream ended").unwrap() {
                Message::Text(t) => break serde_json::from_str::<serde_json::Value>(&t).unwrap(),
                Message::Ping(_) | Message::Pong(_) => continue,
                other => panic!("unexpected frame: {other:?}"),
            }
        };
        assert!(first["error"].is_object(), "got {first}");

        // The connection is still usable.
        ws.send(Message::Text(initialize_request(1).to_string().into()))
            .await
            .unwrap();
        loop {
            match ws.next().await.expect("stream ended").unwrap() {
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

    #[tokio::test]
    async fn a_method_before_initialize_is_rejected() {
        let addr = spawn(ServeState::for_test(true)).await;
        let resp = roundtrip(
            &addr,
            serde_json::json!({
                "jsonrpc": "2.0",
                "id": 0,
                "method": "session/new",
                "params": { "cwd": "/tmp", "mcpServers": [] }
            }),
        )
        .await;
        assert!(
            resp["error"].is_object(),
            "the RFD makes initialize the required first message, got {resp}"
        );
    }

    #[tokio::test]
    async fn binary_frames_are_ignored() {
        let addr = spawn(ServeState::for_test(true)).await;
        let mut ws = connect(&addr).await;

        ws.send(Message::Binary(vec![0xde, 0xad].into())).await.unwrap();
        ws.send(Message::Text(initialize_request(0).to_string().into()))
            .await
            .unwrap();
        loop {
            match ws.next().await.expect("stream ended").unwrap() {
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
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `cargo test --lib serve::acp`
Expected: FAIL — the connection closes without a reply, so `stream ended` panics.

- [ ] **Step 4: Implement the transport and the handler**

Replace `serve_connection` in `src/serve/acp.rs`:

```rust
use agent_client_protocol::schema::v1::{AgentCapabilities, InitializeRequest, InitializeResponse};
use agent_client_protocol::{Agent, Lines, on_receive_request};
use axum::extract::ws::Message;
use futures_util::{SinkExt, StreamExt};

/// Wrap the socket as the SDK's line transport.
///
/// Per the ACP transport RFD one JSON-RPC message rides in one text frame,
/// so a text frame maps to a line with no reframing. Binary frames are
/// ignored, as are the control frames axum surfaces.
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
            Ok(Message::Text(t)) => Some(Ok(t.to_string())),
            Ok(Message::Binary(_) | Message::Ping(_) | Message::Pong(_)) => None,
            Ok(Message::Close(_)) => None,
            Err(e) => Some(Err(std::io::Error::other(e))),
        }
    });

    Lines::new(Box::pin(outgoing), Box::pin(incoming))
}

async fn serve_connection(socket: WebSocket, state: Arc<ServeState>, profile_name: String) {
    let _ = (&state, &profile_name);

    let result = Agent
        .builder()
        .name("sapphire-agent")
        .on_receive_request(
            async move |req: InitializeRequest, responder, _connection| {
                responder.respond(
                    InitializeResponse::new(req.protocol_version)
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
```

`Lines::new` requires `Sink<String, Error = std::io::Error> + Send + 'static` and `Stream<Item = std::io::Result<String>> + Send + 'static`. If the compiler rejects the `impl Trait` return position, box both sides as `Pin<Box<dyn ...>>` exactly as the SDK's own `src/stdio.rs` does in its `with_debug` branch — that file is the worked example for this whole function.

If the SDK answers a malformed frame by closing rather than replying, adjust `malformed_frame_errors_without_closing_the_connection` to assert the actual behaviour and note the deviation from the spec's error table in the commit message. Do not paper over it.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `cargo test --lib serve::acp`
Expected: PASS — six tests.

- [ ] **Step 6: Smoke test against the real Zed**

This is the step that replaces guesswork about what Zed negotiates. Run a dev instance with `[acp] enabled = true` and a known token, install `websocat`, and add to Zed's `settings.json`:

```jsonc
"agent_servers": {
  "sapphire": {
    "type": "custom",
    "command": "websocat",
    "args": ["--text", "-H", "Authorization: Bearer sa-acp-token", "ws://127.0.0.1:9000/acp"]
  }
}
```

Open the agent panel in Zed and select `sapphire`. Record in the commit message: the `protocolVersion` Zed sent, and the exact `clientCapabilities` object it advertised — phase 5b needs the latter. Zed will fail after `initialize` because `session/new` is unimplemented; that is the expected outcome.

If Zed sends a `protocolVersion` this build does not accept, **stop and report before starting Task 7** — every later task depends on the answer.

- [ ] **Step 7: Format, lint and commit**

```bash
cargo fmt
cargo clippy --all-targets -- -D warnings
git add Cargo.toml Cargo.lock src/serve/acp.rs
git commit -m "feat(acp): answer initialize over the /acp WebSocket transport"
```

---

### Task 7: `session/new`

**Files:**
- Modify: `src/serve/acp.rs`

**Interfaces:**
- Consumes: `serve_connection`, `conversation`, `initialize_request` (Task 6).
- Produces: `struct AcpSession { agent_session_id: String, cwd: PathBuf }` and `struct AcpSessions { inner: tokio::sync::Mutex<HashMap<SessionId, AcpSession>> }`, shared across handlers as `Arc<AcpSessions>`.

- [ ] **Step 1: Write the failing tests**

Add to `mod tests` in `src/serve/acp.rs`:

```rust
    fn test_cwd() -> &'static str {
        if cfg!(windows) { "C:\\work\\proj" } else { "/work/proj" }
    }

    fn new_session_request(id: i64) -> serde_json::Value {
        serde_json::json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": "session/new",
            "params": { "cwd": test_cwd(), "mcpServers": [] }
        })
    }

    #[tokio::test]
    async fn session_new_returns_a_session_id() {
        let addr = spawn(ServeState::for_test(true)).await;
        let responses =
            conversation(&addr, vec![initialize_request(0), new_session_request(1)]).await;

        let session_id = responses[1]["result"]["sessionId"]
            .as_str()
            .expect("sessionId present");
        assert!(!session_id.is_empty());
    }

    #[tokio::test]
    async fn prompt_for_an_unknown_session_is_an_error() {
        let addr = spawn(ServeState::for_test(true)).await;
        let responses = conversation(
            &addr,
            vec![
                initialize_request(0),
                serde_json::json!({
                    "jsonrpc": "2.0", "id": 1, "method": "session/prompt",
                    "params": {
                        "sessionId": "no-such-session",
                        "prompt": [{ "type": "text", "text": "hi" }]
                    }
                }),
            ],
        )
        .await;

        assert!(
            responses[1]["error"].is_object(),
            "expected a JSON-RPC error, got {}",
            responses[1]
        );
    }
```

The second test passes trivially until Task 8 registers a prompt handler; it is written here so Task 8 cannot regress it.

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test --lib serve::acp::tests::session_new`
Expected: FAIL — the response carries a JSON-RPC error for an unhandled method rather than a `sessionId`.

- [ ] **Step 3: Implement `session/new`**

In `src/serve/acp.rs`:

```rust
use agent_client_protocol::schema::v1::{NewSessionRequest, NewSessionResponse, SessionId};
use std::collections::HashMap;
use std::path::PathBuf;

/// One ACP session, mapped onto an agent session.
struct AcpSession {
    /// The agent-side session id. `run_llm_turn` routes it to
    /// `cross_device_session_store`, the same store `/rpc` sessions use.
    agent_session_id: String,
    /// The client's workspace root — absolute on the *client's* machine, so
    /// nothing in this phase touches it. Phase 5b uses it as the default
    /// `cwd` for `terminal/create` and the base for relative paths.
    #[allow(dead_code)]
    cwd: PathBuf,
}

#[derive(Default)]
struct AcpSessions {
    inner: tokio::sync::Mutex<HashMap<SessionId, AcpSession>>,
}
```

Build `let sessions = Arc::new(AcpSessions::default());` in `serve_connection` before the builder, clone it into each handler, and register:

```rust
        .on_receive_request(
            {
                let sessions = Arc::clone(&sessions);
                let state = Arc::clone(&state);
                let profile_name = profile_name.clone();
                async move |req: NewSessionRequest, responder, _connection| {
                    // Pin the session to the room profile the bearer token
                    // resolved to. That pin is what gives the ACP session its
                    // namespace chain and provider through the existing paths.
                    let agent_session_id = uuid::Uuid::new_v4().to_string();
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
                        },
                    );

                    responder.respond(NewSessionResponse::new(session_id))
                }
            },
            on_receive_request!(),
        )
```

Check `NewSessionResponse`'s constructor against the crate source before writing it, and mint the id the same way `handle_initialize` in `src/serve/mod.rs` does rather than inventing a second convention.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cargo test --lib serve::acp`
Expected: PASS — eight tests.

- [ ] **Step 5: Format, lint and commit**

```bash
cargo fmt
cargo clippy --all-targets -- -D warnings
git add src/serve/acp.rs
git commit -m "feat(acp): create sessions pinned to the token's room profile"
```

---

### Task 8: `session/prompt` runs a real turn

**Files:**
- Modify: `src/serve/acp.rs`
- Modify: `src/serve/mod.rs` (register one tool in the fixture)

**Interfaces:**
- Consumes: `AcpSessions` (Task 7), `TurnProgress` / `run_llm_turn` (Task 5), `ServeState::for_test_scripted` (Task 3).
- Produces: `struct AcpProgress` implementing `TurnProgress`, and a test helper `drive(addr, prompt) -> (Vec<serde_json::Value>, String)` returning the `session/update` notifications and the stop reason.

- [ ] **Step 1: Write the failing test**

Add to `mod tests` in `src/serve/acp.rs`:

```rust
    /// initialize → session/new → session/prompt on one connection.
    /// Returns every `session/update` notification and the stop reason.
    async fn drive(addr: &str, prompt: &str) -> (Vec<serde_json::Value>, String) {
        let mut ws = connect(addr).await;
        for request in [initialize_request(0), new_session_request(1)] {
            ws.send(Message::Text(request.to_string().into())).await.unwrap();
        }

        let mut session_id: Option<String> = None;
        let mut updates = Vec::new();

        loop {
            let frame = ws.next().await.expect("stream ended").unwrap();
            let Message::Text(t) = frame else { continue };
            let v: serde_json::Value = serde_json::from_str(&t).unwrap();

            if v["id"] == 1 {
                let id = v["result"]["sessionId"].as_str().unwrap().to_string();
                session_id = Some(id.clone());
                ws.send(Message::Text(
                    serde_json::json!({
                        "jsonrpc": "2.0", "id": 2, "method": "session/prompt",
                        "params": {
                            "sessionId": id,
                            "prompt": [{ "type": "text", "text": prompt }]
                        }
                    })
                    .to_string()
                    .into(),
                ))
                .await
                .unwrap();
            } else if v["method"] == "session/update" {
                assert_eq!(v["params"]["sessionId"], *session_id.as_ref().unwrap());
                updates.push(v["params"]["update"].clone());
            } else if v["id"] == 2 {
                let stop = v["result"]["stopReason"].as_str().unwrap().to_string();
                return (updates, stop);
            }
        }
    }

    #[tokio::test]
    async fn prompt_streams_the_reply_and_ends_the_turn() {
        let state = ServeState::for_test_scripted(
            true,
            vec![crate::provider::ChatResponse {
                text: Some("hello from the agent".to_string()),
                tool_calls: Vec::new(),
            }],
        );
        let addr = spawn(state).await;
        let (updates, stop) = drive(&addr, "hi").await;

        let chunks: Vec<&str> = updates
            .iter()
            .filter(|u| u["sessionUpdate"] == "agent_message_chunk")
            .map(|u| u["content"]["text"].as_str().unwrap())
            .collect();
        assert_eq!(chunks, vec!["hello from the agent"]);
        assert_eq!(stop, "end_turn");
    }

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
                },
                crate::provider::ChatResponse {
                    text: Some("done".to_string()),
                    tool_calls: Vec::new(),
                },
            ],
        );
        let addr = spawn(state).await;
        let (updates, stop) = drive(&addr, "use the tool").await;

        let kinds: Vec<&str> = updates
            .iter()
            .map(|u| u["sessionUpdate"].as_str().unwrap())
            .collect();
        let first_tool = kinds.iter().position(|k| *k == "tool_call").unwrap();
        let first_chunk = kinds
            .iter()
            .position(|k| *k == "agent_message_chunk")
            .unwrap();
        assert!(first_tool < first_chunk, "got {kinds:?}");
        assert_eq!(stop, "end_turn");
    }
```

Construct `crate::provider::ToolCall` from its real definition in `src/provider/mod.rs`; the field names above are the plan's best guess and the compiler is the authority.

- [ ] **Step 2: Register a tool in the fixture**

In `src/serve/mod.rs`, replace `ToolSet::new(Vec::new(), Vec::new())` in `build_for_test` with a set holding one trivial tool:

```rust
#[cfg(test)]
pub(crate) struct EchoTool {
    spec: ToolSpec,
}

#[cfg(test)]
impl EchoTool {
    pub(crate) fn new() -> Self {
        Self {
            spec: ToolSpec {
                name: "echo".to_string(),
                description: "Echo the given text back.".to_string(),
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
#[async_trait]
impl crate::tools::Tool for EchoTool {
    fn spec(&self) -> &ToolSpec {
        &self.spec
    }

    async fn execute(&self, input: &Value) -> anyhow::Result<String> {
        Ok(input["text"].as_str().unwrap_or_default().to_string())
    }
}
```

and use `ToolSet::new(vec![Box::new(EchoTool::new())], Vec::new())`. Match `ToolSpec`'s real fields.

- [ ] **Step 3: Run the tests to verify they fail**

Run: `cargo test --lib serve::acp::tests::prompt_streams`
Expected: FAIL — `session/prompt` is unhandled, so the drive loop sees an error response for id 2.

- [ ] **Step 4: Implement `AcpProgress` and the prompt handler**

```rust
use agent_client_protocol::schema::v1::{
    ContentBlock, ContentChunk, PromptRequest, PromptResponse, SessionNotification, SessionUpdate,
    StopReason, ToolCall as AcpToolCall, ToolCallStatus, ToolCallUpdate, ToolCallUpdateFields,
};

/// Reports turn progress to an ACP client as `session/update` notifications.
struct AcpProgress<C> {
    session_id: SessionId,
    connection: C,
}

#[async_trait]
impl<C> super::TurnProgress for AcpProgress<C>
where
    C: Send + Sync,
{
    async fn tool_start(&self, name: &str, input: &Value) {
        let call = AcpToolCall::new(name.to_string(), name.to_string()).raw_input(input.clone());
        let _ = self.connection.send_notification(SessionNotification::new(
            self.session_id.clone(),
            SessionUpdate::ToolCall(call),
        ));
    }

    async fn tool_end(&self, name: &str, _output: &str) {
        let fields = ToolCallUpdateFields::default().status(ToolCallStatus::Completed);
        let _ = self.connection.send_notification(SessionNotification::new(
            self.session_id.clone(),
            SessionUpdate::ToolCallUpdate(ToolCallUpdate::new(name.to_string(), fields)),
        ));
    }
}
```

The `C` bound must name the concrete connection handle the SDK passes as the handler's third argument; take it from the compiler error rather than guessing, and drop the generic if a concrete type is simpler. `send_notification` is a synchronous `fn` returning `Result` (`jsonrpc.rs:3480`), so no `.await`.

Register the handler:

```rust
        .on_receive_request(
            {
                let sessions = Arc::clone(&sessions);
                let state = Arc::clone(&state);
                async move |req: PromptRequest, responder, connection| {
                    let Some(agent_session_id) = sessions
                        .inner
                        .lock()
                        .await
                        .get(&req.session_id)
                        .map(|s| s.agent_session_id.clone())
                    else {
                        return responder.respond_error(/* unknown session */);
                    };

                    // Flatten the prompt's blocks into one user message. Only
                    // Text is handled: PromptCapabilities advertises nothing
                    // else, so nothing else should arrive.
                    let text = req
                        .prompt
                        .iter()
                        .filter_map(|block| match block {
                            ContentBlock::Text(t) => Some(t.text.clone()),
                            _ => None,
                        })
                        .collect::<Vec<_>>()
                        .join("\n");

                    let progress = Arc::new(AcpProgress {
                        session_id: req.session_id.clone(),
                        connection: connection.clone(),
                    });

                    let outcome = crate::serve::run_llm_turn(
                        Arc::clone(&state),
                        agent_session_id,
                        ChatMessage::user(&text),
                        progress,
                        None,
                    )
                    .await;

                    // No streaming provider, so the whole reply is one chunk.
                    if let Some(reply) = outcome.text {
                        let _ = connection.send_notification(SessionNotification::new(
                            req.session_id.clone(),
                            SessionUpdate::AgentMessageChunk(ContentChunk::new(
                                ContentBlock::from(reply),
                            )),
                        ));
                        responder.respond(PromptResponse::new(StopReason::EndTurn))
                    } else {
                        responder.respond(PromptResponse::new(StopReason::Refusal))
                    }
                }
            },
            on_receive_request!(),
        )
```

Build the reply's `ContentBlock` with whatever constructor `schema/v1/content.rs` provides for text rather than assuming `From<String>`, and use the SDK's real error-response method in place of `respond_error`.

- [ ] **Step 5: Run the tests to verify they pass**

Run: `cargo test --lib serve::acp`
Expected: PASS — ten tests, including `prompt_for_an_unknown_session_is_an_error` from Task 7, which now exercises the real handler.

- [ ] **Step 6: Format, lint and commit**

```bash
cargo fmt
cargo clippy --all-targets -- -D warnings
git add src/serve/acp.rs src/serve/mod.rs
git commit -m "feat(acp): run prompt turns through the shared turn executor"
```

---

### Task 9: Cancellation

Two paths end a turn early: an explicit `session/cancel` notification, and the client vanishing. The second matters more than it looks — a tool loop that keeps calling a provider with nobody listening spends money and can still write to the workspace.

**Files:**
- Modify: `Cargo.toml` (`tokio-util`, unless a `tokio::sync::watch` flag is preferred — pick one and use it consistently)
- Modify: `src/serve/acp.rs`

**Interfaces:**
- Consumes: `AcpSession`, `AcpSessions` (Task 7), the prompt handler (Task 8), `ServeState::for_test_hanging` (Task 3).
- Produces: `AcpSession.cancel: tokio_util::sync::CancellationToken`.

- [ ] **Step 1: Write the failing test**

Add to `mod tests` in `src/serve/acp.rs`:

```rust
    #[tokio::test]
    async fn cancel_ends_the_turn_with_cancelled() {
        // The provider never returns, so the turn is still running when the
        // cancel arrives.
        let addr = spawn(ServeState::for_test_hanging(true)).await;
        let mut ws = connect(&addr).await;

        for request in [initialize_request(0), new_session_request(1)] {
            ws.send(Message::Text(request.to_string().into())).await.unwrap();
        }

        let mut sent_prompt = false;
        loop {
            let frame = ws.next().await.expect("stream ended").unwrap();
            let Message::Text(t) = frame else { continue };
            let v: serde_json::Value = serde_json::from_str(&t).unwrap();

            if v["id"] == 1 {
                let id = v["result"]["sessionId"].as_str().unwrap().to_string();
                ws.send(Message::Text(
                    serde_json::json!({
                        "jsonrpc": "2.0", "id": 2, "method": "session/prompt",
                        "params": {
                            "sessionId": id,
                            "prompt": [{ "type": "text", "text": "hang" }]
                        }
                    })
                    .to_string()
                    .into(),
                ))
                .await
                .unwrap();

                // The prompt is in flight now; cancel it. session/cancel is a
                // notification, so it carries no id and gets no reply.
                ws.send(Message::Text(
                    serde_json::json!({
                        "jsonrpc": "2.0", "method": "session/cancel",
                        "params": { "sessionId": id }
                    })
                    .to_string()
                    .into(),
                ))
                .await
                .unwrap();
                sent_prompt = true;
            } else if v["id"] == 2 {
                assert!(sent_prompt);
                assert_eq!(
                    v["result"]["stopReason"], "cancelled",
                    "spec requires Cancelled even when cancellation errors underneath"
                );
                return;
            }
        }
    }

    #[tokio::test]
    async fn disconnect_stops_an_in_flight_turn() {
        let addr = spawn(ServeState::for_test_hanging(true)).await;
        let mut ws = connect(&addr).await;

        for request in [initialize_request(0), new_session_request(1)] {
            ws.send(Message::Text(request.to_string().into())).await.unwrap();
        }

        loop {
            let frame = ws.next().await.expect("stream ended").unwrap();
            let Message::Text(t) = frame else { continue };
            let v: serde_json::Value = serde_json::from_str(&t).unwrap();
            if v["id"] == 1 {
                let id = v["result"]["sessionId"].as_str().unwrap().to_string();
                ws.send(Message::Text(
                    serde_json::json!({
                        "jsonrpc": "2.0", "id": 2, "method": "session/prompt",
                        "params": {
                            "sessionId": id,
                            "prompt": [{ "type": "text", "text": "hang" }]
                        }
                    })
                    .to_string()
                    .into(),
                ))
                .await
                .unwrap();
                break;
            }
        }

        // Drop the socket mid-turn. The assertion is that the server does not
        // hang or panic: the connection task must finish. Without the cancel
        // wiring this test hangs, which is the failure we are fixing.
        drop(ws);
        tokio::time::timeout(std::time::Duration::from_secs(5), async {
            // Reconnecting proves the server is still healthy and accepting.
            let _ = connect(&addr).await;
        })
        .await
        .expect("server still responsive after a mid-turn disconnect");
    }
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test --lib serve::acp::tests::cancel_ends`
Expected: FAIL — `session/cancel` is unhandled, so the test blocks on the hanging provider until the harness times it out.

- [ ] **Step 3: Implement cancellation**

Add `tokio-util = { version = "0.7", features = ["rt"] }` to `[dependencies]`, then:

1. Give `AcpSession` a `cancel: tokio_util::sync::CancellationToken`, created in the `session/new` handler with `CancellationToken::new()`.
2. Register a notification handler with `on_receive_notification` / `on_receive_notification!()` for `CancelNotification`:

```rust
        .on_receive_notification(
            {
                let sessions = Arc::clone(&sessions);
                async move |notif: CancelNotification, _connection| {
                    if let Some(session) = sessions.inner.lock().await.get(&notif.session_id) {
                        session.cancel.cancel();
                    }
                }
            },
            on_receive_notification!(),
        )
```

3. In the prompt handler, widen the existing session lookup to take the token in the same lock, then race the turn against it:

```rust
                    let looked_up = sessions
                        .inner
                        .lock()
                        .await
                        .get(&req.session_id)
                        .map(|s| (s.agent_session_id.clone(), s.cancel.clone()));
                    let Some((agent_session_id, cancel)) = looked_up else {
                        return responder.respond_error(/* unknown session */);
                    };

                    let outcome = tokio::select! {
                        biased;
                        () = cancel.cancelled() => {
                            return responder.respond(PromptResponse::new(StopReason::Cancelled));
                        }
                        outcome = crate::serve::run_llm_turn(
                            Arc::clone(&state),
                            agent_session_id,
                            ChatMessage::user(&text),
                            progress,
                            None,
                        ) => outcome,
                    };
```

`biased;` makes the cancel branch win a tie, which is what the spec means by requiring `Cancelled` "even if the cancellation causes exceptions in underlying operations": the error path must not be reachable once the token fires. Move the `text` and `progress` bindings above this block if they are not already.

4. In `serve_connection`, after `connect_to(...).await` returns — which is when the socket is gone — cancel every token in `sessions`, so a disconnect stops in-flight turns.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cargo test --lib serve::acp`
Expected: PASS — twelve tests.

- [ ] **Step 5: Format, lint and commit**

```bash
cargo fmt
cargo clippy --all-targets -- -D warnings
git add Cargo.toml Cargo.lock src/serve/acp.rs
git commit -m "feat(acp): cancel in-flight turns on session/cancel and disconnect"
```

---

### Task 10: Ship it — documentation and the end-to-end smoke test

**Files:**
- Modify: `README.md`
- Check first: `cliff.toml` and `release-plz.toml`. If the changelog is generated from commit messages, do not hand-edit `CHANGELOG.md`; make sure the commits above carry the change instead.

**Interfaces:**
- Consumes: everything above.
- Produces: no code.

- [ ] **Step 1: Document the Zed setup in `README.md`**

Add a section covering: enabling `[acp]`, minting a token under a `[room_profile.<name>].api_keys` entry, installing `websocat`, and the `agent_servers` block.

```jsonc
"agent_servers": {
  "sapphire": {
    "type": "custom",
    "command": "websocat",
    "args": ["--text", "-H", "Authorization: Bearer <token>", "wss://your-host/acp"]
  }
}
```

State plainly why `websocat` is in the loop: Zed's `agent_servers` takes a command, not a URL, and ACP itself still specifies only stdio — so a local bridge is unavoidable, and `websocat`'s line mode already matches ACP's framing. Note that TLS is expected in front of the endpoint for anything but a loopback connection, and that the bearer token selects the room profile.

Record the two known limitations so nobody files them as bugs: the reply is not streamed token-by-token, and a dropped connection ends the session because ACP v1 has no resumption.

- [ ] **Step 2: Run the end-to-end smoke test**

Against a real Zed and a real dev instance:

1. Hold a multi-turn conversation; confirm replies stay coherent across turns — that is the session store working.
2. Ask for something only the production workspace knows; confirm it answers from memory.
3. Trigger a tool call; confirm Zed renders it.
4. Cancel a turn mid-flight from Zed's UI; confirm the agent stops.
5. Kill `websocat` mid-turn; confirm the agent logs the disconnect and stops the turn.
6. Point the client at a wrong token; confirm the 401 surfaces.

- [ ] **Step 3: Record the results**

Write what actually happened — including anything that failed — into the commit message. If any step fails, stop and report rather than documenting an untested flow.

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "docs: connect Zed to the /acp endpoint through websocat"
```

---

## Notes for the implementer

- **The ACP SDK is `futures`-based, not tokio-native.** It uses `futures::Sink`/`Stream` and its own actor tasks. `Lines` needs no blocking adapter (only the SDK's `Stdio` transport does), so it runs inside a tokio task fine — but expect the generic bounds to be the fiddly part, and lean on the SDK's own `src/stdio.rs` as the worked example.
- **Do not enable `unstable_protocol_v2`.** ACP v2 is draft; v1 is what this plan targets and what the `schema::v1::…` paths assume.
- **The SDK type names in Tasks 8 and 9 are the least certain part of this plan.** The connection handle and responder types were not read in full while writing it. Take the names from the compiler, not from the snippets, and keep the behaviour the tests assert.
- **The tests are the contract.** Where a snippet and a test disagree, the test is right.
- **Do not paper over a spec deviation.** If the SDK behaves differently from the spec's error table (for example by closing on a malformed frame rather than replying), change the test to assert what actually happens and say so in the commit message.
