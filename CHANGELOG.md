# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

[0.1.0]: https://github.com/fluo10/sapphire-agent/releases/tag/v0.1.0
