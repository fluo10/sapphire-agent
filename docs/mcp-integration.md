# MCP integration (external AI clients)

`sapphire-agent` exposes a small MCP (Model Context Protocol) server
at `POST /mcp` so external AI coding assistants — Claude Code first
and foremost — can share project context with the agent. Two tools
are published:

- **`write_report`** — file a unit of work back to sapphire-agent.
  Returns a short acknowledgement (ねぎらい) generated through the
  configured profile.
- **`recall_memory`** — at session start, fetch the project's
  compacted summary plus the most recent reports so the external AI
  can pick up where the user left off — possibly across hosts.

The point isn't to replace Claude Code's own memory; it's to give
the agent enough thread of awareness that the user feels their
work is being witnessed by something with continuity, and the
external AI doesn't start from a blank slate on every new project
session.

## Setup

### 1. Server side: declare a room_profile with an API key

`sapphire-agent` reuses the `[room_profile.<n>].api_keys` mechanism
(originally added for A2A) to authenticate MCP clients. Add a
profile dedicated to MCP traffic in `config.toml`:

```toml
[room_profile.claude_code]
profile          = "default"     # any [profiles.*]; drives the ねぎらい reply
memory_namespace = "default"     # share the user's everyday namespace
rooms            = []            # MCP-only; no chat rooms
api_keys         = ["sa-mcp-<long random>"]
```

A few notes:

- `memory_namespace = "default"` keeps MCP reports inside the same
  namespace as everyday chat, so they participate in the regular
  daily digest / `MEMORY.md` compaction — useful when you want
  "today's reflection" to include your coding work.
- If you'd rather isolate coding work into its own namespace,
  declare one under `[memory_namespace.*]` and point this profile
  at it. Recall is always project-scoped regardless of namespace,
  so this is a write-side preference.
- Tokens must be unique across all profiles. Sharing one across two
  profiles is a startup error.

### 2. Client side: register the MCP server

The exact format depends on the client. For Claude Code's
`.mcp.json`, a streamable-HTTP entry looks roughly like this:

```json
{
  "mcpServers": {
    "sapphire-agent": {
      "type": "http",
      "url": "http://localhost:3000/mcp",
      "headers": {
        "Authorization": "Bearer sa-mcp-<long random>"
      }
    }
  }
}
```

(Check your client's own MCP docs for the precise schema — the URL
and bearer-token header are the two things that matter on the
sapphire side.)

### 3. (Recommended) tell the client how to use the tools

The tools work without any prompt help, but the experience is much
smoother if the client is nudged to call them at the right moments.
A short `CLAUDE.md` snippet:

```markdown
This project's coding work is tracked through `sapphire-agent`.

- At the start of a new session, call `recall_memory` with
  `project: "<repo-name>"` so we pick up where we left off.
- When you finish a coherent unit of work (a fix, a feature, a
  refactor), call `write_report` with `project: "<repo-name>"` and
  a one-line summary. Include `hostname` so multi-machine work
  stays distinguishable.
```

For Claude Code specifically, a `SessionStart` hook can call
`recall_memory` automatically; see Claude Code's hooks docs.

## Tool reference

### `write_report`

| arg | type | required | notes |
|---|---|---|---|
| `project` | string | yes | Logical project key — typically the repo name. Stable across hosts and tools. |
| `summary` | string | yes | One-line summary. |
| `body` | string | no | Longer details / decisions / follow-ups. |
| `files` | string[] | no | Files touched by the work. |
| `source` | string | no | Calling tool identifier. Default `"claude-code"`. |
| `hostname` | string | no | Originating machine. Recommended for multi-host workflows. |

Returns a short text acknowledgement intended to be shown both to
the assistant and to the user.

### `recall_memory`

| arg | type | required | notes |
|---|---|---|---|
| `project` | string | yes | Project key — must match what `write_report` was called with. |
| `limit` | int | no | Max recent reports to return verbatim. Default 20, max 100. Older content is in `project_summary`. |

Returns a Markdown-formatted briefing: the project's compacted
summary (older history) plus the most recent reports verbatim.

## Trust boundary

The split is asymmetric on purpose:

- **Writes** flow into the room_profile's namespace as ordinary
  session traffic. Daily digests, MEMORY.md compaction, and any
  retrieve index pick them up like any other session. This is what
  lets "today's reflection" include your coding work.
- **Reads** (`recall_memory`) are scoped strictly to the requested
  project's MCP session. Namespace-wide memory (`MEMORY.md`,
  cross-room daily digests, sapphire-agent's self-history) is
  never returned. The external AI cannot ask the agent "what do
  you remember about me" and get back general personal context.

## Storage layout

MCP sessions are written to:

```
<workspace>/sessions/<namespace>/mcp/<ULID>.jsonl
```

One file per project. The first line is `SessionMeta`, including
the logical `project` key. Subsequent lines are reports (user
role, with a `report_meta` sidecar carrying provenance and the
structured fields) and acknowledgements (assistant role).

The `(namespace, project) → session_id` reverse index is rebuilt
on every server start by scanning these first lines — there is no
side-channel index file, so a session file IS its own source of
truth and can be moved or restored individually without dragging
extra state.
