# Roadmap: sapphire-agent as a Zed agent — local ACP + remote-workspace

Date: 2026-08-20

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
  Zed today starts the agent as a **subprocess over stdio**; remote transports
  (http/tcp) are upstream work-in-progress.
- **remote-workspace** (the framework's `remote-server` / `remote-client` /
  `RemoteBackend`) replicates workspace *files* between hosts. It is about
  *state*.

So remote-workspace does not let Zed reach a remote agent. What it enables is a
different shape: **run the agent locally under Zed over stdio ACP, and replicate
its state**.

## Decision: the agent runs locally on the dev machine

Zed starts `sapphire-agent` over stdio ACP; memory and sessions are shared with
the production instance through remote-workspace.

**Rejected alternative:** extend sapphire-agent's JSON-RPC so that
`sapphire-call` acts as an **ACP proxy** to a server-side agent. Rejected
because it requires a bespoke protocol surface in two crates and carries the
cost of tracking an evolving ACP spec, whereas replicating state needs only the
stdio transport upstream already ships. `sapphire-call` stays a voice satellite.

**Consequences**

- API keys and a workspace replica live on the dev host. This is what makes the
  config split (phase 2) a requirement rather than a convenience.
- Concurrent session writes become the normal case, not an edge case — agent
  instances on the laptop and on the server both append. This is what makes the
  session redesign (phase 3) a prerequisite rather than an optimisation.

## Constraint: the framework merges per path, last-writer-wins

`ws_store::push` in `sapphire-framework-remote-server` resolves conflicts **per
path, last-writer-wins on `updated_at`**, with no line-level merge, and
`RemoteBackend::append_file` pushes the **whole file** on every append. Two
devices appending to one session file therefore either have the push rejected or
silently overwrite the other side's messages.

Teaching the sync layer to merge files was considered and rejected: it would
complicate the framework's sync substantially, and it is unnecessary — see phase
3.

## Phases

Each is an independent sub-project with its own spec → plan → implementation.

### 1. sapphire-framework `main` migration

Move off the stale `feat/framework-migration` pin onto `main`, adopt the
`sapphire-framework` facade crate, drop the APIs deleted in framework #90, and
remove `standby_mode`.

Spec: `docs/superpowers/specs/2026-08-20-framework-main-migration-design.md`.
Independent of every later phase; unblocked.

### 2. Config layering

Merge a shared **workspace-level** config with a **host-local** config so that
prompts, room profiles and digest settings can be shared while API keys and
remote-workspace publication settings stay per host.

Notes carried forward:

- `Config::load` currently reads exactly one file. The workspace-level config
  that `config.rs` documents today belongs to the *framework's* old `[sync]`
  handling, which phase 1 removes — so after phase 1 there is no workspace-level
  config at all, and this is new agent-side machinery.
- Deep-merging `toml::Value` before a single `Deserialize` keeps the change far
  smaller than making every `Config` field an `Option`.
- **Trust boundary, to be decided in this phase's spec:** once the
  workspace-level config arrives over remote-workspace sync it is remote-
  controlled input executed on every host. `[[tools.mcp_servers]]` carries
  commands to spawn. Anything that starts a process or holds a credential should
  stay host-local, with an allowlist governing what the workspace layer may set.

### 3. Session persistence redesign — one file per event

`src/session.rs` stores each session as one append-only `.jsonl` (meta line,
message lines, digest lines, title lines — `set_title` appends rather than
rewrites — and `closed_at`). It is already a conflict-free event log
semantically; the only problem is that every event shares one path.

Splitting one file per event makes the framework's existing path-level sync
perform the union merge for free, with no change to the sync layer, and replaces
a full-file push per append with one small change.

Proposed layout, to be settled in this phase's spec:

```
sessions/<namespace>/<kind>/<session_id>/
  meta.json               ← written once; every SessionMeta field is fixed at creation
  events/<uuidv7>.json    ← one file per message / digest / title / close event
```

Open items for the spec:

- **`parent` event id — decide before writing any events.** Recording each
  event's predecessor turns the log into a DAG, so two devices continuing the
  same session offline produce detectable *branches* rather than one interleaved,
  incoherent transcript. Near-zero write cost, and **impossible to backfill**.
- Read-path cost: `load_session` goes from one file to N. Keeping `meta.json`
  separate is what stops metadata-only scans (`sessions_for_day`,
  `all_session_dates`, `list_sessions`) from opening event files.
- File count against the workspace indexer, and whether closed (hence immutable)
  sessions get compacted — and if so, which host does it.
- Migration from the existing `.jsonl` files. `migrate_to_device_default_layout`
  is the precedent.

Independent of phase 4 and can land while everything is still local, which is
why it goes first.

### 4. remote-workspace adoption in sapphire-agent

Open the agent's workspace through the framework's `WorkspaceBackend` /
`RemoteBackend` so a local instance mirrors the production workspace.

**Blocked on phase 3**: syncing today's one-file-per-session format would put
conflict-prone files on the wire.

### 5. ACP

A thin stdio ACP adapter so Zed can start the agent. No proxy, no bespoke
transport.

## Non-goals

- Reintroducing git sync. Deferred to a future framework-provided option
  (framework #91).
- Line-level or file-content merging in the framework's sync layer.
- An ACP proxy through `sapphire-call`.
