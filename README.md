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
- **Ambient audio ingest**: optional always-on capture from a wearable/pendant device — `POST /audio/ingest` takes raw audio (metadata in query params, no JSON/base64 framing) from a bearer-authenticated device, re-gates it, transcribes it, attributes it to a speaker against reference audio curated in the workspace, and stores the transcript outside the workspace. Records without answering: nothing in this path starts an LLM turn. `transcript_read`, `speaker_candidates` and `speaker_promote` expose the result as agent tools. Disabled by default — enable via `[ambient].enabled = true`; see `config.example.toml`.
- **Agent-to-agent**: `/a2a` endpoint speaks the v1 A2A protocol (JSON-RPC `SendMessage`, AgentCard) with per-device bearer-token auth — enable via `[a2a].enabled = true`.
- **External AI integration**: `/mcp` endpoint publishes `write_report` and `recall_memory` tools so Claude Code (and other MCP clients) can share project context with the agent — see [docs/mcp-integration.md](docs/mcp-integration.md).
- **Editor integration**: `/acp` endpoint speaks the Agent Client Protocol over WebSocket, so Zed can drive the running agent — enable via `[acp].enabled = true`; see [Zed / ACP](#zed--acp) below.
- **Commands**:
  - `sapphire-agent` — start the channel listeners + JSON-RPC HTTP control API (`/rpc`, `/mcp`, `/a2a`, `/acp`)
  - `sapphire-agent verify` — validate config and report loaded workspace files, including the device -> room_profile bindings
  - `sapphire-agent device add|list|rotate|retire` — register a device, mint or replace its bearer token, or stop it; `device add` prints the token to stdout and the `[room_profile.<n>].devices` line to paste to stderr
  - `sapphire-agent user add|list` — register the person or agent a device belongs to
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

**2. Mint a token.** Authentication reuses the same device table as `/a2a`
and `/mcp`: register a device, bind its id under a `[room_profile.<name>]`'s
`devices`, and that device's token both authenticates the connection and
selects the room profile (and therefore the provider and memory namespace)
the ACP session runs under. A dedicated profile keeps the editor on its own
model or namespace:

```sh
sapphire-agent device add --name zed-editor
# token on stdout, device id + a devices=[...] line to paste on stderr
```

```toml
[room_profile.zed]
profile = "default"
rooms   = []              # ACP-only; no chat rooms map here
devices = ["a3f9k2p"]     # id printed by `device add --name zed-editor`
```

Note that `[acp].enabled = true` opens `/acp` to **every** device bound to
**any** room profile, not only to one minted for the editor — each simply
connects under its own profile. No privilege is gained by doing so (`/a2a`
already runs the full tool set through the same executor for the same
devices), but if you want the editor confined to its own model, namespace or
audit trail, that is what the dedicated profile above is for; giving it a
device does not take `/acp` away from the others.

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

### Permission and modes

Tool calls from the editor are gated. Each tool declares what it does (read,
search, edit, delete, execute), and what happens next depends on the session's
mode:

| Mode | Reads, searches, fetches | Edits and deletes | Commands, MCP tools |
|---|---|---|---|
| `default` | run | **ask** | **ask** |
| `accept_edits` | run | run | **ask** |
| `bypass` | run | run | run |

Zed's mode picker switches between them; the session starts in `default`.

When the agent asks, "Always allow this tool" and "Never allow this tool" are
recorded in `~/.config/sapphire-agent/acp-permissions.json`, keyed by room
profile and tool name. Delete the file to be asked everything again. It sits
beside the host-local config rather than in the workspace on purpose: trusting
a particular editor is a statement about *this machine*, not something to sync
to other hosts. The agent's own tools cannot write to that directory — see the
refusal in `file_write` — because a tool that could edit the permission record
could grant itself anything.

Both standing answers only apply where the agent would otherwise ask. "Never
allow this tool" is not a kill switch: `bypass` runs everything regardless, and
`accept_edits` runs edits regardless.

Declining a tool does not end the turn. The model is told the call was
refused and can try another route.

**Chat channels are not asked — they are restricted.** Matrix and Discord
cannot call `shell` or any MCP tool at all. A chat turn is asynchronous, so
holding one open waiting for a human could hang it for hours; and routing the
question through the model would let it broker its own permission request.
`/rpc`, the voice pipeline and `/a2a` are unchanged and still run everything.

**The heartbeat's chat leg counts as a channel.** A scheduled task under
`<workspace>/heartbeat/` runs through the same path as a chat message when it
replies to a room, so it cannot call `shell` or MCP tools either. This is
deliberate rather than an oversight: heartbeat task bodies are workspace files,
and `file_write` is an edit, which a channel may perform without being asked —
so trusting that path would let a chat message write itself a task that runs a
command on the next tick.

### Loading past sessions

Zed's session picker lists this token's own conversations: `session/list`
returns every open session under the connecting device's room profile's
memory namespace, and Zed can further narrow it to the project directory it
has open (`cwd`).
Picking one hands it to `session/load`, which replays the stored
conversation into the editor before answering the request, or to
`session/resume`, which does the same without the replay. Either way the
session is a real one afterwards — prompting it continues the existing
history rather than starting a fresh conversation.

**Sessions from before this feature, and any created over `/rpc`, have no
recorded `cwd`.** The project filter matches on `cwd`, so these sessions
never show up in a project-filtered list — only in an unfiltered one.
`cwd` is recorded once, from `session/new`'s own field, at the moment a
session is first created; there is no way to backfill it onto a session
that already exists.

**Tool calls are not restored on replay.** See "A loaded session's tool
calls are not replayed" below — the JSONL transcript a replay reads from
never recorded a tool call's name, input or result in the first place, for
the same reason it never recorded usage.

**Opening the same session on two connections at once is not safe to
prompt from both.** Nothing stops a second Zed window — or any other ACP
client — from loading a session that is already open elsewhere, but
`run_llm_turn` clones a session's history at the start of a turn and
writes the whole vector back at the end. Two turns racing on one session
are last-writer-wins in memory, while both have already appended their own
messages to the on-disk transcript, so the two views diverge. Loading does
not create this race — `/rpc` and the voice heartbeat could already reach
it with two prompts on one connection — it makes it reachable *across*
connections too. It is not fixed here: a `session/load` or
`session/resume` that lands on an already-open session logs a `session
{id} is now open on N connections` warning, so a divergence at least
leaves a trace, but the fix itself needs a per-session lock held across a
whole turn.

### Known limitations

- **Replies are not streamed token by token.** `Provider::chat` returns a
  complete response, not a stream, so the whole reply arrives as a single
  chunk at the end of the turn. Tool calls *do* still appear progressively as
  they run.
- **No automatic reconnection.** A dropped connection does not resume on
  its own — ACP v1 has no mechanism for the editor to reconnect
  transparently. The conversation is not lost, though: reopen it through
  `session/load` or `session/resume` from Zed's session picker, see
  "Loading past sessions" above.
- **Tool calls reach the editor as a bare name.** No arguments and no results
  are sent — the shared turn executor reports only a tool's id and name — so
  Zed shows "shell" rather than the command it ran or what came back. The one
  exception is a permission request, which does carry the tool's kind and its
  raw input so you can see what you are approving.
- **A failed or refused tool is reported as completed.** The executor's
  `ToolOutput` carries no success bit, so every tool that finishes is sent to
  the editor as `completed`, whether it succeeded, errored, or was declined.
  The model still sees the reason and reacts to it; only the editor's status
  display is wrong.
- **Partly exercised against a real Zed** (2026-08-31). Confirmed working
  from an actual Zed install: the connection, `initialize`, `session/new`
  with its mode list, and the permission prompt for an `Execute` tool —
  both allowing and declining. Still unconfirmed against a real client:
  the mode picker actually switching modes, "always allow" surviving a
  reconnect, multi-turn conversations, and cancellation.
- **No token usage is reported.** The provider layer discards the `usage`
  the API returns, so neither `session/update: usage_update` nor
  `PromptResponse.usage` is sent, and the editor cannot show what a turn
  cost or how full the context is.
- **A loaded session's tool calls are not replayed.** `session/load`
  reconstructs only user and assistant text from the JSONL transcript — it
  never recorded a tool call's name, input or result, so there is nothing
  to replay it from. A restored conversation shows what was said, not what
  the agent did.

### Configuring `websocat` in Zed

One trap, since Zed spawns the command **without a shell**: do not quote
the header argument. In a terminal you would write
`-H 'Authorization: Bearer …'` and the shell would strip the quotes; in
`settings.json` the quotes become part of the header value and the server
answers `401` before the WebSocket upgrade. Write it bare:

```jsonc
"args": [
  "--text",
  "wss://your-host/acp",
  "-H",
  "Authorization: Bearer <token>"   // no surrounding quotes
]
```

`-H` takes multiple values, so it has to come *after* the URL or it
swallows it.

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT License ([LICENSE-MIT](LICENSE-MIT))

at your option.
