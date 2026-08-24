# Design: ACP over WebSocket — `/acp` transport and adapter

Date: 2026-08-24

## Context

Phase 5a of `docs/superpowers/specs/2026-08-20-zed-acp-remote-workspace-roadmap.md`.

The goal is to drive the **already-running** production agent from Zed. The roadmap's
revision settles the shape: a WebSocket ACP endpoint at `/acp` on the axum listener that
already serves `/rpc`, `/a2a` and `/mcp`, with Zed reaching it through `websocat` because
Zed can only spawn local subprocesses.

What already exists and is reused rather than rebuilt:

- `ServeState` (`src/serve/mod.rs`) is entirely `Arc`/`Mutex`-shared and already serves four
  concurrent consumers (`/rpc`, `/a2a`, `/mcp`, and the Matrix/Discord/voice paths). An ACP
  connection is a fifth consumer, not a second agent.
- `run_llm_turn` is the shared turn executor: history hydration, the tool-calling loop,
  JSONL persistence, and per-tool progress events.
- `Config::resolve_a2a_token` maps a bearer token to a **room profile**, which is how
  `/a2a` and `/mcp` authenticate today.

This phase adds transport and protocol only. The model gets the tools it already has. The
client-side `fs/*` and `terminal/*` tools are phase 5b.

## Decisions

### The endpoint is `/acp`, WebSocket only

The ACP RFD *Streamable HTTP & WebSocket Transport* fixes the path (`/acp`), the upgrade
(standard HTTP), the framing (**one WebSocket text frame per JSON-RPC message**, binary
frames ignored) and the ordering (`initialize` first after the upgrade). Following it costs
nothing over inventing something, and it is what a future native client will expect.

Streamable HTTP is deliberately not implemented. WebSocket alone covers Zed via `websocat`,
and it carries agent→client calls on the same socket without needing SSE alongside POST.
The RFD does require a client supporting remote ACP over HTTP to support *both* halves, so a
native client may eventually need the HTTP half — it mounts on the same handler and the same
adapter, so deferring costs nothing but the deferral.

### Authentication reuses the existing bearer scheme, and the token still selects a profile

`Authorization: Bearer <token>` on the upgrade request, resolved through
`Config::resolve_a2a_token`.

This is conformant, not a shortcut: the RFD puts authentication explicitly out of scope and
layers it "via HTTP headers, query parameters, or WebSocket subprotocols". ACP's own
`authenticate` method is for agents that hold their own provider credentials, which is not
this situation.

The valuable part is the second-order effect. The token already resolves to a room profile,
so an ACP session gets a room profile for free, which means it also gets a namespace chain,
a provider and a session policy through the paths that already exist. A dedicated Zed token
under its own `[room_profile.dev]` is then how the operator gives the editor a different
model or a different memory namespace from the Matrix rooms — no new mechanism.

`authMethods` in the `initialize` response is therefore `[]`: by the time ACP speaks, the
connection is already authenticated.

**Rejection happens before the upgrade.** A missing or unknown token is answered with HTTP
401 rather than a 101 followed by an ACP error. It matches what `/a2a` and `/mcp` already do,
it costs nothing, and `websocat` surfaces the status to the operator, whereas an error
arriving after a successful upgrade would surface as a mysterious immediate disconnect.

`extract_bearer` is currently copied into both `src/serve/a2a.rs` and `src/serve/mcp.rs`.
This would be the third copy, which is where it gets lifted into a shared helper in
`src/serve/mod.rs` instead.

### The adapter is a front-end over the existing turn executor

`session/prompt` maps onto `run_llm_turn`. It does not get its own tool loop, its own history
handling or its own persistence. A Zed conversation therefore lands in the same session store
as everything else, with the same memory and the same system prompt.

This is the whole reason the endpoint is worth building: getting a *different* agent in Zed
would have been easy and useless.

### Turn progress is decoupled from SSE

`run_llm_turn` currently takes `tx: mpsc::Sender<Result<Event, Infallible>>` and a JSON-RPC
`req_id`, and emits `tool_start` / `tool_end` as axum SSE events. ACP needs the same events
shaped as `session/update` notifications.

So progress reporting moves behind a small trait:

```rust
#[async_trait]
pub(crate) trait TurnProgress: Send + Sync {
    async fn tool_start(&self, name: &str, input: &Value);
    async fn tool_end(&self, name: &str, output: &str);
}
```

with three implementations: the existing SSE shape (holding `tx` and `req_id`), the ACP shape
(emitting `session/update`), and a no-op. The no-op is not speculative — `run_llm_turn` is
already called at `src/serve/mod.rs:1483` with a throwaway channel purely to discard these
events.

This is the only change to shared code in this phase, and it removes a coupling that was
already awkward rather than adding one.

### One ACP session per connection, stored as a cross-device session

`session/new` creates a session in `cross_device_session_store` (kind `rpc`) — the same store
`/rpc` sessions use, and the one `store_for_session` falls through to.

The alternative is a dedicated `acp` store, as `/mcp` has. It was rejected for this phase:
the MCP store exists because MCP traffic is machine-written per-project reports that would
drown the user-facing session list, whereas an ACP session is a person having a conversation
— exactly what the cross-device store holds. Sharing it also means a Zed conversation is
visible to `list_sessions` and resumable elsewhere, which is the continuity the whole roadmap
is about.

If Zed sessions turn out to be noisy in practice, splitting them out later is a store swap in
one place.

### `cwd` is recorded and not yet used

`session/new`'s `cwd` is an absolute path **on the client's machine** and is normative — the
spec requires it to be used for the session regardless of where the agent was spawned. On a
remote agent that path does not exist locally, and nothing in phase 5a should touch it.

It is stored on the session anyway, because phase 5b needs it as the default `cwd` for
`terminal/create` and as the base for resolving relative paths the model produces. Recording
it now costs one field and avoids a protocol change later.

### The assistant reply is not streamed, and that is a provider limitation

`Provider::chat` returns a complete `ChatResponse`; there is no streaming method on the trait.
So `session/update` carries the reply as a **single `agent_message_chunk`** at the end of the
turn. Zed will show tool calls appearing progressively — those are per-round and genuinely
incremental — and then the reply arriving at once.

This is worth stating plainly because it is the most visible difference from other ACP agents
in Zed. Fixing it means adding a streaming method to `Provider` and threading it through
`run_llm_turn`, which affects every channel and belongs in its own piece of work, not smuggled
into a transport phase.

### No session resumption

`agentCapabilities.loadSession` is advertised as `false` and `session/load` is not
implemented. ACP v1 has no message replay or stream resumption — the RFD defers durability to
v2 — so a dropped WebSocket ends the session exactly as a closed stdio pipe does.

The session's history is still persisted and still visible to `list_sessions`; what is absent
is Zed's ability to re-enter it. Adding `session/load` later is additive and does not change
anything here.

### The endpoint is opt-in and host-local

A new `[acp]` block with `enabled` defaulting to `false`, mirroring `[a2a]`. Because the
config allowlist introduced in phase 2 is default-deny, `[acp]` is host-local automatically
and no allowlist entry is added — which is correct: whether *this* host exposes an ACP
endpoint is a property of the host, not of the shared workspace.

## Architecture

```
Zed
 └─ spawns: websocat --text -H 'Authorization: Bearer …' wss://prod/acp
      │   1 line of stdin  → 1 text frame
      │   1 text frame     → 1 line of stdout
      ▼
axum listener (existing, one process)
 ├─ POST /rpc  /a2a  /mcp          ← unchanged
 └─ GET  /acp   → 401 unless the bearer resolves
                → 101, then src/serve/acp.rs
                     │
                     ├─ initialize      → capabilities, authMethods: []
                     ├─ session/new     → session in cross_device_session_store,
                     │                     room profile from the token, cwd recorded
                     ├─ session/prompt  → run_llm_turn(…, AcpProgress)
                     │                     └─ session/update: tool_call,
                     │                        tool_call_update, agent_message_chunk
                     └─ session/cancel  → cancel the in-flight turn
                                          │
                                          ▼
                       the same ServeState: workspace, memory,
                       session stores, providers, ToolSet
```

`src/serve/acp.rs` sits beside `a2a.rs` and `mcp.rs` and is mounted from `serve::run` as
`.route("/acp", axum::routing::get(acp::handle_acp_ws))`.

### Dependencies

`axum` is currently declared with `default-features = false` and the features `http1`, `json`
and `tokio`. WebSocket support is behind axum's **`ws`** feature, which this phase adds — the
only production dependency change beyond the ACP crate itself. `tokio-tungstenite` is added
as a dev-dependency for the endpoint tests; nothing in production needs a WebSocket *client*,
because `websocat` fills that role outside the process.

The `agent-client-protocol` crate supplies the protocol types and the JSON-RPC plumbing; its
transport abstractions (`ByteStreams`, `Lines`) are not stdio-bound, so the WebSocket frame
stream feeds it directly. Which protocol version is negotiated depends on the Zed build in
use, and `initialize` negotiates it — the implementation plan's first task is an end-to-end
smoke test against the installed Zed to pin that down before anything else is built on it.

## Error handling

| Case | Behaviour |
|---|---|
| Missing / unparseable / unknown bearer | HTTP 401 at the upgrade, no 101 |
| `[acp] enabled = false` | HTTP 404 with a short body, mirroring `/a2a`'s existing feature gate |
| Binary frame | Ignored, per the RFD |
| Malformed JSON in a text frame | JSON-RPC parse error response; connection stays open |
| Method before `initialize` | JSON-RPC error; the RFD makes `initialize` the required first message |
| Unknown `sessionId` | JSON-RPC error, connection stays open |
| Client disconnects mid-turn | Cancel the in-flight turn. A tool loop that keeps calling a provider with nobody listening spends money and can still write to the workspace |
| `MAX_TOOL_ROUNDS` exhausted | End the turn with the corresponding ACP stop reason rather than a protocol error |
| Provider error | Same — a failed turn is a turn outcome, not a transport failure |

## Testing

- **Bearer resolution** — unit tests on the lifted shared helper, covering the cases
  `a2a.rs` and `mcp.rs` already cover so the lift does not lose coverage.
- **Framing** — the adapter driven over an in-memory duplex stream, no network: `initialize`
  → `session/new` → `session/prompt` → notifications → result, plus the malformed-frame and
  wrong-order cases from the table above.
- **Endpoint** — an axum test server with a `tokio-tungstenite` client: 401 without a token,
  401 with an unknown token, 404 when disabled, 101 and a working `initialize` with a valid
  token.
- **Turn integration** — a stub `Provider` that returns one tool call then a final message,
  asserting that the emitted `session/update` sequence matches the SSE path's `tool_start` /
  `tool_end` sequence for the same script. This is what protects the `TurnProgress` refactor.
- **Manual smoke test** — real Zed, real `websocat`, against a dev instance. This is the only
  way to learn what Zed actually advertises and negotiates, and it gates the phase.

## Out of scope

- `fs/*` and `terminal/*` client-side tools, and the session-scoped tool overlay — phase 5b.
- Token-level streaming of the assistant reply; needs a streaming `Provider` method.
- `session/load`, `session/list`, session modes, slash commands, elicitation.
- Streamable HTTP.
- A bridge subcommand. `websocat` is the bridge.

## Open questions for the implementation plan

- **Protocol version.** Which version the installed Zed negotiates, and whether the crate
  version pinned here speaks it. Settled by the smoke test, before other work depends on it.
- **Concurrent connections.** Whether to cap them. One editor is the expected load, and
  `ServeState` is already concurrent, so the default is no cap — but a cheap limit may be
  worth it given the endpoint is network-exposed.
- **Where `cwd` is stored.** On `SessionMeta` (durable, and phase 3 fixes every `SessionMeta`
  field at creation, which suits a `cwd` that cannot change) versus an in-memory per-session
  map like `session_room_profiles` (no persistence format change). The tradeoff is only worth
  resolving once 5b confirms what it needs.
- **Permission prompts.** `session/request_permission` is not needed in 5a because the
  existing tools do not act on the client host, but if any current tool warrants a prompt in
  the Zed UI, wiring it here is cheaper than retrofitting in 5b.
