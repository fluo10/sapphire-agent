# sapphire-agent

A personal AI assistant agent that lives in a [`sapphire-framework`](https://github.com/fluo10/sapphire-framework) workspace and talks to me through Matrix and Discord.

> **Status: personal project.** This is something I built for my own use. It only has to work in my environment, and that is the only environment I test it in. You are welcome to use it, fork it, or send pull requests, but I am not going to maintain providers, channels, or features I do not personally use. If your use case overlaps with mine, great; if not, fork freely.
>
> The reason this exists at all is that other agents I tried (openclaw, zeroclaw, …) either did not support what I needed, were not actually tested for the parts I cared about, or did not accept fixes. So I wrote my own. Please calibrate expectations accordingly.

## What it does

- **Channels**: Matrix (E2EE via `matrix-sdk`) and Discord (`serenity`), running concurrently.
- **Providers**: Anthropic Messages API with SSE streaming and a multi-round tool-use loop, plus OpenAI-compatible backends (local LLMs, OpenRouter, …) selectable per room/session via the `[providers]` / `[profiles]` / `[room_profile]` schema.
- **Workspace**: backed by [`sapphire-framework`](https://github.com/fluo10/sapphire-framework) — file index, full-text + vector search (redb + tantivy, plus LanceDB vectors by default).
- **Built-in tools**: `file_read`, `file_write`, `file_append`, `file_delete`, `dir_list`, `dir_walk`, `web_search`, `weather`, `shell`, `timer_set` / `timer_preset` / `timer_cancel` / `timer_status` (incl. Pomodoro presets), plus workspace memory / search / sync tools.
- **Sessions**: human-readable [`grain-id`](https://crates.io/crates/grain-id) aliases, auto-generated titles, history dump on resume.
- **Background**: heartbeat cron tasks, periodic memory compaction, periodic workspace re-index, daily / weekly / monthly / yearly logs with catch-up.
- **Voice**: optional `sapphire-call voice` satellite with local STT/TTS (via `sherpa-onnx`), Silero VAD, and an openWakeWord wake detector. See [crates/sapphire-call](crates/sapphire-call/).
- **Agent-to-agent**: `/a2a` endpoint speaks the v1 A2A protocol (JSON-RPC `SendMessage`, AgentCard) with per-profile bearer-token auth — enable via `[a2a].enabled = true`.
- **External AI integration**: `/mcp` endpoint publishes `write_report` and `recall_memory` tools so Claude Code (and other MCP clients) can share project context with the agent — see [docs/mcp-integration.md](docs/mcp-integration.md).
- **Editor integration**: `/acp` endpoint speaks the Agent Client Protocol over WebSocket, so Zed can drive the running agent — enable via `[acp].enabled = true`; see [Zed / ACP](#zed--acp) below.
- **Commands**:
  - `sapphire-agent` — start the channel listeners + JSON-RPC HTTP control API (`/rpc`, `/mcp`, `/a2a`, `/acp`)
  - `sapphire-agent verify` — validate config and report loaded workspace files
  - `sapphire-call` — interactive REPL / voice satellite client (separate crate; see [crates/sapphire-call](crates/sapphire-call/))

## Install

```sh
cargo install sapphire-agent
```

Or from source:

```sh
git clone https://github.com/fluo10/sapphire-agent
cd sapphire-agent
cargo build --release
```

## Configure

Copy `config.example.toml` to your config directory (`~/.config/sapphire-agent/config.toml` on Linux) and fill in the Anthropic API key, workspace path, and whichever channels you actually want.

Then:

```sh
sapphire-agent verify   # sanity-check config and workspace
sapphire-agent          # start the channel listeners + HTTP control API
sapphire-call           # one-off interactive session (separate crate)
```

## Zed / ACP

`sapphire-agent` can be driven directly from [Zed](https://zed.dev) as an ACP
(Agent Client Protocol) agent, running against the same production workspace,
memory and sessions as every other transport — a Zed conversation is not a
separate agent, it lands in the same session store as `/rpc`, Matrix and
Discord.

**1. Enable the endpoint** in your config (host-local; the workspace config
layer cannot turn this on):

```toml
[acp]
enabled = true
```

**2. Mint a token.** Authentication reuses the same bearer scheme as `/a2a`
and `/mcp`: add a token under a `[room_profile.<name>]`'s `api_keys`, and
that token both authenticates the connection and selects the room profile
(and therefore the provider and memory namespace) the ACP session runs
under. A dedicated profile keeps the editor on its own model or namespace:

```toml
[room_profile.zed]
profile  = "default"
rooms    = []                      # ACP-only; no chat rooms map here
api_keys = ["sa-acp-<long random>"]
```

Note that `[acp].enabled = true` opens `/acp` to **every** token in every
`[room_profile.*].api_keys`, not only to one minted for the editor — each
simply connects under its own profile. No privilege is gained by doing so
(`/a2a` already runs the full tool set through the same executor for the same
tokens), but if you want the editor confined to its own model, namespace or
audit trail, that is what the dedicated profile above is for; giving it a
token does not take `/acp` away from the others.

**3. Install [`websocat`](https://github.com/vi/websocat).** Zed's
`agent_servers` setting only takes a `command` to spawn, not a URL — and ACP
itself still specifies only stdio transport, so *some* local process has to
bridge stdio to the network. `websocat` is that bridge, and no bridge code
was written for it: its `--text` stdio mode maps one line of stdin to one
WebSocket text frame and one received text frame to one line of stdout,
which is exactly the one-JSON-RPC-message-per-frame framing ACP's WebSocket
transport uses.

**4. Point Zed at it**, in Zed's `settings.json`:

```jsonc
"agent_servers": {
  "sapphire": {
    "type": "custom",
    "command": "websocat",
    "args": ["--text", "-H", "Authorization: Bearer <token>", "wss://your-host/acp"]
  }
}
```

Use `wss://`, not `ws://`, for anything beyond a loopback connection — the
bearer token travels in a plain header, so TLS is expected to already be
terminating in front of the endpoint (a reverse proxy, typically) rather than
being provided by the agent itself.

### Known limitations

- **Replies are not streamed token by token.** `Provider::chat` returns a
  complete response, not a stream, so the whole reply arrives as a single
  chunk at the end of the turn. Tool calls *do* still appear progressively as
  they run.
- **No session resumption.** A dropped connection ends the session. ACP v1
  has no resumption mechanism, and `session/load` is not implemented.
- **Tool calls reach the editor as a bare name.** No arguments and no results
  are sent — the shared turn executor reports only a tool's id and name — so
  Zed shows "shell" rather than the command it ran or what came back.
- **A failed tool is reported as completed.** The executor's `ToolOutput`
  carries no success bit, so every tool that finishes is sent to the editor
  as `completed`, whether it succeeded or errored. The model still sees the
  error text and reacts to it; only the editor's status display is wrong.
- **Not yet exercised against a real Zed.** The endpoint is covered by
  automated tests (framing, auth, `initialize`/`session/new`/`session/prompt`/
  `session/cancel`), but has not yet been driven end-to-end from an actual
  Zed install. Treat it as untested against real client behaviour until
  someone does that smoke test.

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT License ([LICENSE-MIT](LICENSE-MIT))

at your option.
