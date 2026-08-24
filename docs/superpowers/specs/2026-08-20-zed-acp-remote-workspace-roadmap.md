# Roadmap: sapphire-agent as a Zed agent

Date: 2026-08-20
Revised: 2026-08-24 — the agent stays on the production host and Zed reaches it
over a WebSocket ACP endpoint. See "Revision" below.

This is a **roadmap**, not an implementation spec. It records the decomposition
and the decisions that constrain it. Each phase below gets its own design spec
and implementation plan.

## Goal

Use `sapphire-agent` as an agent inside Zed, without the agent losing the
memory, sessions and workspace it accumulates on the production host.

## The framing: ACP and remote-workspace solve different problems

These were initially considered as alternatives. They are not.

- **ACP** (Agent Client Protocol) is how an editor drives an agent — prompt
  turns, tool-call permission, streaming updates. It is about *interaction*.
- **remote-workspace** (the framework's `remote-server` / `remote-client` /
  `RemoteBackend`) replicates workspace *files* between hosts. It is about
  *state*.

The original revision of this roadmap concluded from this that Zed could not
reach a remote agent, and therefore that the agent had to run locally with its
state replicated. The first half of that was true of stdio and not of ACP in
general; the second half does not follow. See the revision.

## Revision (2026-08-24): the agent stays on the production host

Zed reaches the already-running systemd instance over a **WebSocket ACP
endpoint at `/acp`**, alongside the existing `/rpc`, `/a2a` and `/mcp`. Bearer
authentication is the mechanism `/a2a` and `/mcp` already use.

This reaches the goal directly: the ACP session runs *in* the production agent,
on the production workspace, so there is no state to replicate and nothing to
merge.

**Rejected: run a second agent locally under Zed over stdio.**

This was the original decision. It fails on its own terms:

- Zed would start a second `sapphire-agent` process. On the production host that
  is a bind-address conflict on the serve port and two processes racing on the
  same workspace — the failure `main.rs` already refuses to allow when it rejects
  `standby_mode = true`.
- On a separate dev host it avoids the conflict only by paying for phases 3 and
  4 first, plus API keys and a workspace replica on the laptop.

**Rejected: ssh to a locally-listening socket.**

An intermediate design had the resident agent listen on a unix socket and Zed
spawn `ssh -T prod …` as a byte pipe, delegating authentication to ssh. It
works, and it needs no new authentication code. It was dropped because it
reaches the agent only from hosts with ssh access, whereas `/acp` is symmetric
with the endpoints the agent already serves and reuses the bearer machinery
already written. The exposure is not new: `/a2a` and `/mcp` are already
bearer-authenticated endpoints on the same listener.

**Still rejected: an ACP proxy through `sapphire-call`.**

Unchanged from the original roadmap, and for the original reason — it would put
a bespoke protocol surface in two crates and take on tracking an evolving spec
in both. `sapphire-call` stays a voice satellite. Note that this rejection never
applied to a *transport*: a component that carries ACP bytes without parsing
them takes on no spec-tracking cost at all.

## Constraint: Zed cannot connect to a WebSocket ACP endpoint

Not today, and this is not something the agent side can fix.

- Zed's `agent_servers` settings accept `type`, `command`, `args` and `env`.
  There is no `url`. Every ACP agent is a local subprocess.
- ACP itself specifies only stdio. `protocol/v1/transports` marks Streamable
  HTTP as *"In discussion, draft proposal in progress."*, and the introduction
  says "Full support for remote agents is a work in progress."

So Zed needs a local command that bridges stdio to the WebSocket. **`websocat`
is that command**, and no code needs to be written for it: its stdio mode maps
one line of stdin to one text frame and one received text frame to one line of
stdout, which is exactly ACP's newline-delimited framing.

```jsonc
"agent_servers": {
  "sapphire": {
    "type": "custom",
    "command": "websocat",
    "args": ["--text", "-H", "Authorization: Bearer <token>", "wss://prod/acp"]
  }
}
```

A bridge subcommand inside `sapphire-agent` was considered and rejected: it
would duplicate `websocat` and add a binary that has to be installed on every
client host anyway.

## What the WebSocket transport RFD already fixes

The transport is a draft, but not an open field. The ACP RFD *Streamable HTTP &
WebSocket Transport* (Active since July 2026) already settles the parts this
design depends on:

- a single **`/acp`** endpoint, upgraded from HTTP by the standard mechanism
- **one WebSocket text frame per JSON-RPC message**; binary frames ignored
- `initialize` is the first message after the upgrade
- authentication is explicitly **out of scope** and layered on top "via HTTP
  headers, query parameters, or WebSocket subprotocols" — which is what makes
  reusing the existing bearer scheme conformant rather than a deviation

What it does *not* yet fix, and what this project therefore does without:
resumability and message replay (deferred to ACP v2 — a dropped connection ends
the session, exactly as a closed stdio pipe does), and the `Acp-Protocol-Version`
header (unimplemented upstream).

Only the framing layer is exposed to this churn. The ACP method surface is
identical over stdio and WebSocket.

## Constraint: the framework merges per path, last-writer-wins

`ws_store::push` in `sapphire-framework-remote-server` resolves conflicts **per
path, last-writer-wins on `updated_at`**, with no line-level merge, and
`RemoteBackend::append_file` pushes the **whole file** on every append.

This still holds, but it now constrains only phase 4, which is no longer on the
path to the goal.

## Phases

### 1. sapphire-framework `main` migration — done

Merged in #170. Moved off the stale `feat/framework-migration` pin, adopted the
`sapphire-framework` facade crate, dropped the APIs deleted in framework #90 and
removed `standby_mode`.

Spec: `docs/superpowers/specs/2026-08-20-framework-main-migration-design.md`.

### 2. Config layering — in flight

Merges a shared **workspace-level** config with a **host-local** config.

Spec: `docs/superpowers/specs/2026-08-21-config-layering-design.md`.
In flight on `feat/config-layering`.

Its motivation weakens under this revision — with one agent on one host there
is no second host to share settings with — but the work is nearly complete, its
allowlist is the trust boundary phase 4 would need, and phase 5a reuses the
same host-local-by-default reasoning for the ACP endpoint's own settings.

### 5a. ACP WebSocket transport and adapter — next

A `/acp` WebSocket endpoint on the existing axum listener, bearer-authenticated
through `Config::resolve_a2a_token` as `/a2a` and `/mcp` already are, driving an
ACP adapter over the agent's **existing** tools.

Done, Zed can hold a conversation with the production agent against the
production memory and session store. Independently valuable and independently
testable.

Spec: `docs/superpowers/specs/2026-08-24-acp-websocket-transport-design.md`.

### 5b. Client-side filesystem and terminal tools

ACP's `fs/*` and `terminal/*` are **client-side** methods: the agent calls them
and the *client* executes them on the *client's* machine. They are what makes a
remote agent usable for development in Zed — and `fs/read_text_file` reads
unsaved editor buffers, so they are the right path even for a local agent.

This phase adds an ACP-session-scoped tool overlay exposing them to the model.
The overlay is the point: `ToolSet` is process-shared with the Matrix, Discord
and voice paths, and those must never gain access to the editor host.

Sequenced after 5a so that transport bugs and tool bugs cannot surface at the
same time.

### 3. Session persistence redesign — one file per event

`src/session.rs` stores each session as one append-only `.jsonl`. It is already
a conflict-free event log semantically; the only problem is that every event
shares one path. Splitting one file per event makes the framework's existing
path-level sync perform the union merge for free.

**No longer a prerequisite for anything.** With one agent process, concurrent
appends to one session file do not occur, which was the entire reason this
preceded phase 4. It is now motivated only by phase 4 itself.

One item does not wait comfortably: recording each event's **`parent` event id**
turns the log into a DAG, so two devices continuing a session offline produce
detectable branches rather than an incoherent interleaved transcript. It costs
almost nothing to write and is **impossible to backfill**. If the event format
is ever going to change, changing it before more history accumulates is cheaper.

Other open items for that spec: read-path cost of `load_session` going from one
file to N (keeping `meta.json` separate is what stops metadata-only scans from
opening event files); file count against the workspace indexer; whether closed
sessions get compacted and by which host; migration from today's `.jsonl`, for
which `migrate_to_device_default_layout` is the precedent.

### 4. remote-workspace adoption

Open the agent's workspace through the framework's `WorkspaceBackend` /
`RemoteBackend` so a second instance can mirror the production workspace.

**No longer on the path to the goal.** It becomes relevant again only if a
genuinely offline or second-host agent is wanted. Still blocked on phase 3.

## Non-goals

- Reintroducing git sync. Deferred to a future framework-provided option
  (framework #91).
- Line-level or file-content merging in the framework's sync layer.
- An ACP proxy through `sapphire-call`.
- A bridge subcommand duplicating `websocat`.
- Streamable HTTP. WebSocket alone covers Zed via `websocat`. Note that the RFD
  requires clients supporting remote ACP over HTTP to support *both*, so a
  future native client may need the HTTP half; it lands on the same handler.
