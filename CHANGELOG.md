# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

[0.3.0]: https://github.com/fluo10/sapphire-agent/releases/tag/v0.3.0
[0.2.1]: https://github.com/fluo10/sapphire-agent/releases/tag/v0.2.1
[0.2.0]: https://github.com/fluo10/sapphire-agent/releases/tag/v0.2.0
[0.1.0]: https://github.com/fluo10/sapphire-agent/releases/tag/v0.1.0
