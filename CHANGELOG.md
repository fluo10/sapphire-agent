# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.4.1] - 2026-04-20

### Added

- **Unsupported image format notification** — when a Matrix message contains an
  image whose MIME type is not on Anthropic's allowlist, the agent now sends an
  explicit reply informing the user that the format is not supported, instead of
  silently dropping the attachment.

### Fixed

- **Image MIME type validation** — image attachments in Matrix messages are now
  validated against Anthropic's supported media-type allowlist
  (`image/jpeg`, `image/png`, `image/gif`, `image/webp`) before being forwarded
  to the API, preventing request errors caused by unsupported formats.

## [0.4.0] - 2026-04-19

### Added

- **`weather` tool** — fetches a short-term weather forecast via the Open-Meteo
  API (no API key required). Accepts either a `location` name (geocoded
  internally) or explicit `latitude` / `longitude`, plus an optional `days`
  (1–7, default 3). Returns current conditions and a daily min/max /
  precipitation / weather-code summary.
- **`file_append`, `dir_list`, `dir_walk` tools** — round out the filesystem
  toolset: append text to a file (creating it if missing), list direct children
  of a directory, and recursively walk a tree bounded by `max_depth` /
  `max_entries`. All three share the same path semantics as `file_read` /
  `file_write` (absolute, `~/...`, or workspace-relative) and update the
  retrieve index + git sync for internal paths.
- **Weekly / monthly / yearly log auto-generation** — heartbeat now writes
  summarised logs under `memory/{weekly,monthly,yearly}/` at the appropriate
  day boundaries (Monday / 1st / Jan 1). Each log carries a YAML frontmatter
  `digest:` array of importance-ordered bullets produced by the same LLM call
  that writes the body, so the agent no longer has to call the `memory` tool
  to recall long-horizon context.
- **Periodic digest injection** — the system prompt gains four new blocks
  after "Yesterday's Log": "This Week's Digests", "This Month's Digests",
  "This Year's Digests", and "Past Years' Digests". Each block pulls the
  top-N items of the relevant logs' digests (N per kind is configurable via
  the new `[digest]` config section: `daily_items`, `weekly_items`,
  `monthly_items`, `yearly_items`; defaults 3/3/5/5).
- **Daily digest back-fill** — pre-existing daily logs that lack a `digest:`
  frontmatter are upgraded in-place at startup and in the hourly catchup
  loop. Unrelated frontmatter keys the memory tool writes (`last_read_at`,
  `read_count`) are preserved.

### Changed

- **`terminal` → `shell`** — the shell-execution tool is renamed for clarity
  and now honours `$SHELL` by default (falling back to `/bin/sh`) instead of
  hard-coding `sh`. An optional `shell` parameter lets the agent pick a
  specific interpreter (e.g. `bash`, `zsh`, `fish`, or an absolute path) per
  call. Callers that hard-coded the old `terminal` name must update.
- **Tool naming — `<namespace>_<operation>`** — the three file tools are
  renamed from `read_file` / `write_file` / `delete_file` to `file_read` /
  `file_write` / `file_delete`, matching the convention used by every other
  tool in the agent (`memory_*`, `workspace_*`, `web_search`, `mcp_reconnect`).
  Callers that hard-coded the old names must update.
- **Unified file-operation tools** — `file_read`, `file_write`, and
  `file_delete` now route all paths through `WorkspaceState`, so a single tool
  handles both workspace-internal and external paths. Paths may be absolute,
  `~/...`, or relative to the workspace root; internal paths still update the
  search index and git sync automatically, external paths fall through to
  plain `std::fs`. The redundant `workspace_read` and `workspace_write` tools
  are removed — the generic `file_read` / `file_write` replace them. Enabled
  by `AppContext::allow_external_paths()` (new in sapphire-workspace 0.9).
- **sapphire-workspace 0.9** — upgraded the workspace dependency from 0.8.1
  to 0.9.0. The retrieve store's search API now takes typed `FtsQuery` /
  `VectorQuery` builders and `search_similar` embeds the query internally
  (no pre-computed vector needed), so the `workspace_search` tool no longer
  needs to manually call the embedder or dedup chunk results. Semantic search
  output drops the per-result title (the new `FileSearchResult` is already
  file-level with `id`, `path`, `score`, `chunks`); results now render as
  `- {path} [score]`.
- **Daily log format** — newly generated dailies now carry YAML frontmatter
  with a `digest:` array. The existing body format (`# Daily Log: YYYY-MM-DD`
  + summary) is unchanged. Files can be hand-edited freely.

## [0.3.3] - 2026-04-16

### Added

- **Daily log catchup loop** — heartbeat now runs an hourly catchup task that
  regenerates any missing daily log without waiting for the next process
  restart, so a transient failure at the midnight boundary self-heals within
  the hour.

### Fixed

- **Shutdown summary cancellation** — `main` now awaits the agent task after
  `serve::run` returns, so the shutdown summary's LLM call completes instead
  of being cancelled along with the runtime (previously surfaced as a
  misleading `dns error: task NN was cancelled`).
- **Stale system prompt after late daily log** — freshly written daily logs
  now invalidate the per-conversation system-prompt cache, so the model sees
  the new log immediately instead of staying on the cached snapshot until the
  next day boundary.

## [0.3.2] - 2026-04-15

### Fixed

- **Daily log sync** — daily log writes now go through `WorkspaceState` so
  generated logs are picked up by the workspace git sync instead of being
  written directly to disk and skipped.
- **Discord reconnect** — the Discord gateway now reconnects with exponential
  backoff on disconnect instead of silently stopping, matching the Matrix
  sync reconnect behavior.

## [0.3.1] - 2026-04-15

### Fixed

- **Windows release build** — bundle SQLite into the binary via matrix-sdk's
  `bundled-sqlite` feature so the Windows release no longer fails on the
  missing system SQLite library.

### Changed

- **Release workflow** — macOS and Windows matrix entries are now
  `continue-on-error`, so a failure on those best-effort platforms does not
  block the Linux x86_64/aarch64 release.

## [0.3.0] - 2026-04-15

### Added

- **Daily log injection** — previous day's daily log is now injected into the
  system prompt at session start, giving the agent continuity from yesterday's
  activity without loading raw history.
- **MCP manual reconnect** — new `mcp_reconnect` built-in tool lets the agent
  manually reconnect to an MCP server that has dropped without restarting the
  whole process.
- **Configurable day-boundary policy** — `session.day_boundary` option controls
  whether and how a new session is opened at midnight; defaults to the previous
  rolling behavior.
- **Matrix multi-room** — multiple Matrix rooms can now be configured with
  independent session state.
- **Install scripts** — `install.sh` (Unix) and `install.ps1` (Windows) added
  for installing pre-built binaries without Cargo.

### Changed

- **Session context on restart** — sessions now carry a generated summary into
  the next session rather than replaying raw message history, keeping the
  context window usage predictable after long-running sessions.
- **Workspace config consolidation** — workspace-level settings are now merged
  from the agent `config.toml` so a separate workspace config file is no longer
  required.
- **Release workflow** — GitHub Actions release workflow improved for more
  reliable artifact publishing.

### Fixed

- **Matrix sync reconnect** — the Matrix sync loop now auto-reconnects on
  network disconnect instead of silently stopping.
- **Bootstrap room filter** — fallback summarization is skipped for rooms that
  have no agent config, preventing spurious errors on startup.

## [0.2.1] - 2026-04-13

### Fixed

- **`sapphire-workspace` 0.8.0 → 0.8.1** — fixes git sync push not working
  due to a bug in `sapphire-sync` where the push step was silently skipped.

## [0.2.0] - 2026-04-12

### Added

- **MCP client** — built-in MCP client for integrating external tool servers,
  with support for `notifications/tools/list_changed` to dynamically refresh
  the tool list at runtime.
- **Context compression** — automatically compresses the conversation when the
  session approaches the context window limit, keeping sessions alive longer.
- **Heartbeat config** — `heartbeat_enabled` and `standby_mode` options in the
  agent config to pause or suspend background heartbeat tasks without
  restarting.
- **Crate split** — `sapphire-agent-api` (shared types + SSE client) and
  `sapphire-call` (standalone CLI client) extracted into their own workspace
  crates so they can be published and used independently.
- **Dependabot** — automated dependency update PRs with grouped Cargo patch
  updates on Fridays and auto-merge for patch-level bumps.

### Changed

- **`sapphire-workspace` 0.5.0 → 0.8.0** — picks up upstream improvements to
  file indexing, vector search, and git sync.
- **`reedline` upgraded to 0.47** — handles the new non-exhaustive `Signal`
  enum without warnings.
- **Memory tools** — `MemoryTool` redesigned into per-file entry tools with
  frontmatter tracking for finer-grained read/write control.

### Fixed

- `[sync]` config is now read from the agent config file rather than only from
  the workspace config, so sync settings specified in `config.toml` are
  actually honoured.

## [0.1.0] - 2026-04-09

Initial release of `sapphire-agent` — a personal AI assistant built around the
Anthropic API, with Matrix and Discord chat frontends, an interactive REPL, and
an HTTP/MCP server mode.

### Added

- **Core agent** — Anthropic API client with tool-use support and multi-round
  tool execution that preserves assistant text across rounds.
- **Matrix frontend** — `matrix-sdk` based client with end-to-end encryption
  and Markdown rendering.
- **Discord frontend** — `serenity` based bot with channel support and
  per-channel light/heavy model switching.
- **`call` command** — interactive REPL powered by `reedline` with CJK-safe
  input, history, cursor movement, history dump on resume, and date in prompt.
- **`serve` command** — HTTP server exposing an MCP Streamable HTTP API,
  with graceful shutdown on Ctrl-C.
- **Built-in tools** — `read_file`, `write_file`, `delete_file`, `web_search`,
  `terminal` (defaulting to the workspace root), and `workspace_search`.
- **Workspace integration** — `sapphire-workspace` 0.5 with file indexing,
  full-text search, vector search via LanceDB (default feature
  `lancedb-store`), and git sync.
- **Sessions** — UUIDv7-backed session files, human-readable `grain-id`
  aliases for API sessions, auto-generated session titles, and deferral of
  empty sessions so they are not persisted.
- **Memory system** — long-term memory with periodic compaction.
- **Heartbeat** — cron-scheduled background tasks defined via YAML
  frontmatter, daily logs, async tool execution, and periodic workspace sync.
- **Configuration** — TOML config with XDG directories, `~/` path expansion,
  and workspace-aware writes.
- **Logging** — `tracing` with env-filter and ANSI output.

[0.4.1]: https://github.com/fluo10/sapphire-agent/releases/tag/v0.4.1
[0.4.0]: https://github.com/fluo10/sapphire-agent/releases/tag/v0.4.0
[0.3.3]: https://github.com/fluo10/sapphire-agent/releases/tag/v0.3.3
[0.3.2]: https://github.com/fluo10/sapphire-agent/releases/tag/v0.3.2
[0.3.1]: https://github.com/fluo10/sapphire-agent/releases/tag/v0.3.1
[0.3.0]: https://github.com/fluo10/sapphire-agent/releases/tag/v0.3.0
[0.2.1]: https://github.com/fluo10/sapphire-agent/releases/tag/v0.2.1
[0.2.0]: https://github.com/fluo10/sapphire-agent/releases/tag/v0.2.0
[0.1.0]: https://github.com/fluo10/sapphire-agent/releases/tag/v0.1.0
