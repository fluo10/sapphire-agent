# sapphire-agent

A personal AI assistant agent that lives in a [`sapphire-framework`](https://github.com/fluo10/sapphire-framework) workspace and talks to me through Matrix and Discord.

> **Status: personal project.** This is something I built for my own use. It only has to work in my environment, and that is the only environment I test it in. You are welcome to use it, fork it, or send pull requests, but I am not going to maintain providers, channels, or features I do not personally use. If your use case overlaps with mine, great; if not, fork freely.
>
> The reason this exists at all is that other agents I tried (openclaw, zeroclaw, …) either did not support what I needed, were not actually tested for the parts I cared about, or did not accept fixes. So I wrote my own. Please calibrate expectations accordingly.

## What it does

- **Channels**: Matrix (E2EE via `matrix-sdk`) and Discord (`serenity`), running concurrently.
- **Providers**: Anthropic Messages API with SSE streaming and a multi-round tool-use loop, plus OpenAI-compatible backends (local LLMs, OpenRouter, …) selectable per room/session via the `[providers]` / `[profiles]` / `[room_profile]` schema.
- **Workspace**: backed by [`sapphire-framework`](https://github.com/fluo10/sapphire-framework) — file index, full-text + vector search (redb + tantivy, plus LanceDB vectors by default).
- **Built-in tools**: `file_read`, `file_write`, `file_append`, `file_delete`, `dir_list`, `dir_walk`, `web_search`, `weather`, `shell`, `timer_set` / `timer_preset` / `timer_cancel` / `timer_status` (incl. Pomodoro presets), plus workspace memory / search / sync tools. The seven that touch this agent's own filesystem and shell (`file_read`, `file_write`, `file_append`, `file_delete`, `dir_list`, `dir_walk`, `shell`) are opt-in: `[tools.host_access] enabled = false` by default, for every origin including `/rpc` and `/a2a`. Turning it on is a deliberate act; running the agent in a container is the recommended way to do it once it is.
- **Client-side tools**: `client_file_read`, `client_file_write`, `client_shell`, `client_shell_start`, `client_shell_output`, `client_shell_kill` — touch the *editor's* machine instead, over `/acp`'s `fs/*` and `terminal/*` requests. Only offered inside an ACP session, and only for the capabilities the connected editor actually declared at `initialize`. See [Client-side tools: whose machine](#client-side-tools-whose-machine) below.
- **Sessions**: human-readable [`grain-id`](https://crates.io/crates/grain-id) aliases, auto-generated titles, history dump on resume.
- **Background**: heartbeat cron tasks, periodic memory compaction, periodic workspace re-index, daily / weekly / monthly / yearly logs with catch-up.
- **Voice**: optional `sapphire-call voice` satellite with local STT/TTS (via `sherpa-onnx`), Silero VAD, and an openWakeWord wake detector. See [crates/sapphire-call](crates/sapphire-call/).
- **Ambient audio ingest**: optional always-on capture from a wearable/pendant device — `POST /audio/ingest` takes raw audio (metadata in query params, no JSON/base64 framing) from a bearer-authenticated device, re-gates it, transcribes it, attributes it to a speaker against reference audio curated in the workspace, and stores the transcript outside the workspace. Records without answering: nothing in this path starts an LLM turn. `transcript_read`, `speaker_candidates` and `speaker_promote` expose the result as agent tools. Disabled by default — enable via `[ambient].enabled = true`; see `config.example.toml`.
- **Agent-to-agent**: `/a2a` endpoint speaks the v1 A2A protocol (JSON-RPC `SendMessage`, AgentCard) with per-device bearer-token auth — enable via `[a2a].enabled = true`.
- **External AI integration**: `/mcp` endpoint publishes `write_report` and `recall_memory` tools so Claude Code (and other MCP clients) can share project context with the agent — see [docs/mcp-integration.md](docs/mcp-integration.md).
- **Subagents**: `<workspace>/agents/<name>.md` definitions the main agent can delegate a task to via the `subagent` tool — its own system prompt, its own tool loop, only the final answer comes back. See [Subagents](#subagents) below.
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

## Subagents

The main agent can delegate a task to a specialised subagent via the
`subagent` tool: a nested conversation with its own system prompt and its
own tool-calling loop, which hands back only its final answer. Use it to
keep a large investigation (read 30 files, conclude in 3 lines) out of the
conversation that would otherwise carry all 30 files forward on every later
turn.

**Defining one.** Drop a Markdown file under `<workspace>/agents/<name>.md`
— the same shape as `<workspace>/heartbeat/*.md`, reusing that convention
rather than inventing a second one. YAML frontmatter for the metadata, the
body for the prompt:

```markdown
---
description: Reviews a diff. Reads and reports; does not edit.
tools: [client_file_read, workspace_search, memory_read]
---
You are a reviewer. Read the diff, report problems, and stop there.
```

- **`description`** is the *only* thing the parent model sees before
  deciding whether to delegate — a weak one means the agent never gets
  called.
- **`tools`** is optional. Omit it and the subagent inherits the parent's
  whole visible tool set (see "What a subagent does not inherit" below for
  the one exception); give it a list — including `[]` — and the subagent is
  restricted to exactly that. Both are legitimate: an empty list is a valid
  definition, for an agent that only needs the prompt to summarise or judge.
- **The body** is the subagent's entire system prompt. That is genuinely
  all of it — see below.

Definitions are read once at process startup, the same as heartbeat tasks;
adding or editing one needs a restart. A file with no or invalid
frontmatter, or no `description`, is skipped with a warning at load time —
one broken definition does not take the others down.

**`tools:` is enforced, not a hint.** `TurnLoop::run` refuses, before any
other check, any tool call whose name was not in the round's own advertised
list — this is what makes a reviewer definition that says "reviews, does
not edit" actually unable to edit, rather than merely undocumented for
editing. The same check is what caps delegation depth (see below) and
applies identically on the parent's own turn, closing a hallucinated- or
out-of-list tool name there too.

**What a subagent does not inherit.** Its system prompt is the
definition's body plus the current date and time — and nothing else. In
particular, none of the following reach it: `SOUL.md`, `IDENTITY.md`,
`USER.md`, `AGENTS.md`, `TOOLS.md` or any other workspace file the main
agent's prompt is built from, the memory digest, today's cross-session
digest, room metadata, or the configured `anthropic.system_prompt`. **This
is the point of the feature, not an oversight.** The main agent carries all
of that deliberately — it is meant to be someone to work *with* — but a
code review does not need yesterday's conversation, and dropping the parts
a task doesn't need is what a subagent is *for*. The date is the one
exception: an agent that doesn't know today's date can't correctly use a
tool that writes one, and that's a fact rather than a personality trait.

That statement is about the *prompt text*, though, not about *tool reach*:
a definition with no `tools:` key still inherits whichever `memory_*` tools
the parent can see, and a subagent's memory calls land in the same
namespace the delegating conversation is already in. So "a subagent has no
memory" is true of what it's told up front, not of what it's able to go
look up — that distinction is entirely in the definition's own `tools:`
list.

**Isolation is about history and the store, not about visibility.** A
subagent's tool calls still fire the same `tool_start`/`tool_end`
notifications on the parent's host that the parent's own calls do, so they
show up live in an ACP session's stream — that's necessary for a
permission prompt triggered by a subagent's call to make sense as coming
from *this* conversation. What doesn't happen is persistence: nothing from
a subagent's internal turns reaches the parent's stored history or the ACP
session store. Only the returned final answer does, as the `subagent`
tool's own result — reload the session later and you see "delegated to
reviewer, got back Y", not the 30 files it read to get there.

That "only the returned final answer" is text-only, too: `SubagentTool::execute`
returns a plain `String`, so a nested tool's image output has nowhere to
go. If a subagent's own tool list includes `recall_image`, calling it
produces nothing the parent (or the model) can see — the image bytes are
simply dropped rather than attached to anything.

**Delegation is depth 1.** A subagent's own tool list never contains
`subagent`, regardless of what its definition's `tools:` says — and, per
the enforcement above, this isn't just an omission from the list a
subagent is offered: a call literally naming `subagent` is refused the
same way any other out-of-list name is, so a subagent cannot recurse by
guessing.

**Permission requests work normally.** A subagent's tool calls are judged
by the parent's own `Origin`, through the parent's own `TurnHost` — the
same gate, the same person asked, the same standing allow/deny answers.
Delegating is not a way to get done by proxy what the model would be
refused directly.

**`subagent` runs unconditionally on every trusted transport, and is
excluded only on Matrix and Discord.** Its `ToolKind` is `Other` — what a
subagent actually does depends entirely on which agent it delegates to, so
it can't honestly be classified as `Read`, `Edit`, `Execute`, etc., and
`Other` is the policy table's most conservative bucket. `Origin::Trusted`
(`/rpc`, `/a2a`, voice) allows every kind unconditionally, so those three
run `subagent` the same as any other tool; on ACP it is asked for in
`default` mode and allowed once edits are accepted or bypassed, same as
any other `Other` call. `Origin::Channel` is the one origin that denies
every `Other` call unconditionally, so Matrix and Discord — and only
Matrix and Discord — cannot call `subagent` at all. See the
[permission table](#permission-and-modes) below.

**Project conventions are not shared yet.** A client-side project's
`CLAUDE.md` isn't threaded into a subagent's prompt — but the main agent
doesn't read it either, currently. Both are tracked together in
[#199](https://github.com/fluo10/sapphire-agent/issues/199); when that
lands, subagents get it too.

## Zed / ACP

`sapphire-agent` can be driven directly from [Zed](https://zed.dev) as an ACP
(Agent Client Protocol) agent, running against the same production workspace
and memory as every other transport — a Zed conversation is not a separate
agent. Its sessions do get their own store, though: a separate directory
tree and line format from `/rpc`, Matrix and Discord, not a `cwd`-tagged row
folded into one of theirs. See
[Loading past sessions](#loading-past-sessions) below for the layout.

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
`/rpc`, the voice pipeline and `/a2a` are unchanged in how *this* gate treats
them — they are still never asked. But none of the three can reach the seven
host-machine tools either, once `[tools.host_access] enabled = false` (the
default this crate ships): that gate applies to every origin, not just chat.
See "The agent's own filesystem and shell are opt-in" below.

**The heartbeat's chat leg counts as a channel.** A scheduled task under
`<workspace>/heartbeat/` runs through the same path as a chat message when it
replies to a room, so it cannot call `shell` or MCP tools either. This is
deliberate rather than an oversight: heartbeat task bodies are workspace files,
and `file_write` is an edit, which a channel may perform without being asked
*when host access is on* — so trusting that path would let a chat message
write itself a task that runs a command on the next tick. With host access
off (the default), `file_write` is one of the seven host-machine tools and is
refused for every origin before this reasoning is even reached — but it still
governs the moment host access is turned on.

### Client-side tools: whose machine

Six tools reach the connected editor's machine instead of this agent's own,
by making ACP `agent → client` requests instead of touching a filesystem or
spawning a process locally:

| Tool | ACP call(s) | `ToolKind` |
|---|---|---|
| `client_file_read` | `fs/read_text_file` | `Read` |
| `client_file_write` | `fs/write_text_file` | `Edit` |
| `client_shell` | `terminal/create` → `wait_for_exit` → `output` → `release` | `Execute` |
| `client_shell_start` | `terminal/create` | `Execute` |
| `client_shell_output` | `terminal/output` | `Read` |
| `client_shell_kill` | `terminal/kill` + `terminal/release` | `Execute` |

Each is offered only inside an ACP session, and only for the capability the
connected editor actually declared in `initialize`. `fs.read_text_file` and
`fs.write_text_file` are read **independently** — an editor that can read but
not write files gets `client_file_read` and nothing else; `terminal` gates
all four shell tools together. There is no round trip spent finding this
out: an unsupported tool is simply absent from the list the model sees.

`client_file_read` passes `line`/`limit` straight through to
`fs/read_text_file`; that pair exists in ACP for exactly this reason, so a
large file can be read in pieces instead of shipped over the wire whole.

**`client_shell`'s timeout does not kill the command.** It waits up to
`timeout_secs` (default 120, overridable per call, capped at 600) and, if the
command is still running when that runs out, hands back the terminal handle
instead of an error — the process is left running. ACP defines
`terminal/release` as also killing the command, so releasing on timeout would
throw away however long a build had already run, and for a non-idempotent
command (`git push`, a migration, a script that writes files) a subsequent
retry would run it a second time. The result text tells the model explicitly
not to re-run, and points it at `client_shell_output` / `client_shell_kill`
instead.

**Terminals are tracked per session, not per connection, and a dropped
connection releases nothing.** `terminal/release` kills the command, so a
network blip must not trigger one — tracking by session id (rather than by
connection) is what lets a client that reconnects and reloads the session
reach the same handles with `client_shell_output` / `client_shell_kill`. A
handle is released in exactly two cases — the one-shot `client_shell`
finishing inside its timeout, or the model calling `client_shell_kill` — plus
one case that is an *untrack* rather than a release: the client reporting the
handle unknown, where there is nothing left to free. An output check that
fails does **not** untrack the handle: the failure could be transient, and
dropping tracking of a terminal that might still be running would be
unrecoverable. `client_shell_kill`, by contrast, untracks unconditionally
even when the underlying kill or release call itself errors — over-counting
is the recoverable direction (the model can check or kill again); a
permanently stuck cap slot is not.

Each session may hold at most 8 terminals at once, counting both
`client_shell_start`'s handles and any `client_shell` call that timed out
without finishing. A ninth is refused, and the refusal names every handle
currently held so the model knows what to `client_shell_kill` first.

**No client-side directory listing, delete, or append.** ACP's
`agent → client` surface is exactly `session/request_permission`, `fs/*` and
`terminal/*` — there is no request for listing, deleting, or appending.
Layering a structured wrapper over `terminal/create` to fake them would mean
inventing a second, protocol-unbacked convention — a made-up output format,
absorbing every client's differences in `ls` — and maintaining it
indefinitely. So that work goes through `client_shell` instead. This is a
decision, not a gap: ACP does not have this surface.

**The agent's own filesystem and shell are opt-in, and that closed a real
hole.** The seven host-machine tools (`file_read`, `file_write`,
`file_append`, `file_delete`, `dir_list`, `dir_walk`, `shell`) require
`[tools.host_access] enabled = true` in config; the default is `false`, for
every origin. Before this switch existed, `Origin::Channel` (Matrix/Discord)
refused only `Execute` and `Other` calls, so `file_write` (`Edit`) and
`file_delete` (`Delete`) went through unconditionally — a Discord message
asking to delete a file reached the agent's own filesystem. Enabling host
access is now a deliberate, config-level act; running the agent in a
container is the recommended way to do it once it is on.

### Loading past sessions

An ACP session lives at `<sessions_dir>/<namespace>/acp/<id>.jsonl`, its own
tree with its own line format — separate from `/rpc`'s. It is routed by the
`namespace` recorded in the session's own header, not by a room id (an ACP
session has none), and it is recorded with `channel: "acp"`. Every line is
`kind`-tagged; the first is the header (`session_id`, `namespace`, `cwd`,
`created_at`), and every event after it carries a UUIDv7 `id` and the
`parent` it was appended after. **The `parent` chain is the authority on
order, not the `id`'s embedded clock** — two writers with skewed clocks
would otherwise sort against each other wrongly.

Zed's session picker lists this token's own conversations: `session/list`
returns every open session under the connecting device's room profile's
memory namespace — that is the isolation boundary, not the room profile
itself, so two profiles that leave `memory_namespace` unset both land on
`default` and see each other's sessions. Zed can further narrow the list to
the project directory it has open (`cwd`), which every ACP session records
(the header field is required, not optional, since `session/new` always
reports one).
Picking one hands it to `session/load`, which replays the stored
conversation into the editor before answering the request, or to
`session/resume`, which does the same without the replay. Either way the
session is a real one afterwards — prompting it continues the existing
history rather than starting a fresh conversation.

**A per-session digest is cached outside the workspace, addressed by
session id.** `<cache_dir>/sapphire-agent/digests/<session_id>.json`
holds one intra-day digest per session — "what this session has covered
today" — overwritten in place rather than appended to. Losing it costs
only today's cross-session block — the block other rooms' system prompts
use to see what this session has been doing today. It is not the durable
record: the daily log is, built from the session's own events rather than
from the digest.

A second, similarly workspace-external location,
`<cache_dir>/sapphire-agent/tool-results/<sha256>`, is a content-addressed
cache holding a tool call's result by hash — the JSONL itself stores only
the tool name, the result's hash, not the result body, and the input as
given. The input has nowhere to be hashed out to the way a result does, so
it goes into the JSONL directly; one that serialises past the same 50,000
byte cap is elided to a small marker object on the way to disk rather than
stored, so a single oversized call cannot put its raw content into the
(indexed) workspace. An ACP session's `tool_use` and `tool_result` messages
are persisted (see "Tool calls are persisted for ACP sessions" below);
every other transport still is not.

**Losing the cache degrades a session rather than breaking it.** Delete
the directory, or lose an entry to a write failure, and the session still
loads: the missing result is rendered as a placeholder, but the `tool_use`
/ `tool_result` pair itself is kept intact. That pairing is not optional —
the Anthropic API rejects a history where a `tool_use` has no matching
`tool_result` in the very next message — so `history()` also repairs an
orphan it finds on read: a `tool_use` whose `tool_result` append never
made it to disk (a cache write that failed, or the process dying between
the two appends) gets a synthesised placeholder result spliced in
immediately after the message that carries it. This repair is read-side
only — it is never written back to the store, so a later read repeats it
if the underlying orphan is still there.

Both caches sit outside the workspace for the same reason: `<workspace>/sessions`
is in the retrieve search index, and a digest that is rewritten as the
conversation grows would otherwise pile up near-duplicate summaries inside
an indexed file and skew search — sync cost is secondary. A digest also
cannot be pruned out of the session log itself: events are chained by
`parent`, and removing one orphans its children.

**ACP sessions appear in both the cross-session "today" digest and the
daily log.** The digest is kept current by a sweep on a rolling 30-minute
cadence from process start, which regenerates only the sessions whose
newest message postdates their cached digest. `/rpc`, device-default and
MCP sessions still do not reach the daily log —
[#189](https://github.com/fluo10/sapphire-agent/issues/189).

**Tool calls are persisted for ACP sessions.** `run_llm_turn` writes
`tool_use` and `tool_result` messages to the ACP session's JSONL (see
"A second, similarly workspace-external location" above for what the
JSONL stores versus the result cache). `/rpc`, device-default and MCP
sessions do not — their line format has no reference form for a result,
so writing one raw would put the content in the workspace and the
retrieve index, which is exactly what the ACP store's external cache
exists to avoid for results ([#194](https://github.com/fluo10/sapphire-agent/issues/194)).
That protection is only for results, though — a tool *input* has no
hash-and-cache indirection, so it lands in the JSONL directly, bounded
rather than eliminated: capped at the same 50,000-byte limit, and
elided to a marker object above it. Reloading an ACP session therefore
restores what the agent did as well as what was said — but the editor
does not currently render it: see "A loaded session's tool calls are
not shown in the editor" below.

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
- **`/rpc`, device-default and MCP sessions still do not persist tool
  calls.** Only ACP sessions write `tool_use` / `tool_result` to their
  JSONL; the other transports' line format has no reference form for a
  result, so a reload of one of those restores what was said but not what
  was done. Tracked as [#194](https://github.com/fluo10/sapphire-agent/issues/194).
- **A loaded session's tool calls are not shown in the editor.** Even for
  an ACP session, whose `tool_use` / `tool_result` pairs are on disk and
  do reach the model on the next turn, `session/load`'s replay only
  projects text parts to `session/update` notifications — tool calls in
  the stored history are silently skipped rather than rendered. The
  model sees them; the editor's thread view does not. Tracked as
  [#192](https://github.com/fluo10/sapphire-agent/issues/192).
- **Tool output is capped at 50,000 bytes** (roughly 20,000 from the
  head, 30,000 from the tail, with a marker spliced in between) — this
  applies to every tool, on every transport, not only the shell tool over
  ACP; `ToolSet::execute` is the one path both `/acp` and the
  Matrix/Discord turn executor call through. The head, tail and marker
  budgets are sized to sum to the cap exactly, so applying the truncation
  to an already-truncated result is a no-op rather than cutting again and
  nesting markers.

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
