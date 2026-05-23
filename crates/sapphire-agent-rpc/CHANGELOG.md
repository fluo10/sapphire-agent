# Changelog

All notable changes to `sapphire-agent-rpc` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Historical changes prior to `0.7.0` (when this crate shared the workspace
version with `sapphire-agent`) are recorded in the root `CHANGELOG.md`.

## [Unreleased]

## [0.7.0] - 2026-05-23

Aligns the crate version with `sapphire-agent` 0.7.0. `0.6.1` was an
unintended interim publication produced by the first release-plz run
before this crate had its own deliberate release; the source between
`0.6.1` and `0.7.0` is identical apart from the version bumps and
path-dep updates.

### Added

- **Chat audio output modality** — the `chat` JSON-RPC method now
  accepts a `modalities` parameter with `["text", "audio"]`. When
  `audio` is requested, the SSE stream emits server-side TTS frames
  (base64-encoded PCM) alongside the text response, so a single chat
  request can drive both the chat UI and the speakers. Text-only
  clients are unchanged. (#127)

### Changed

- **Crate renamed from `sapphire-agent-api` to `sapphire-agent-rpc`** to
  match the `/rpc` endpoint it talks to. Clients depending on the old
  crate name should update their `Cargo.toml`; the type surface is the
  same. The session-kind directory on the server side also moves from
  `sessions/<ns>/api/` to `sessions/<ns>/rpc/`, with a dual-read shim
  keeping legacy files visible until the bundled migration runs. The
  `api_keys` field on the server's `[room_profile.<n>]` config is
  deliberately unchanged — it gates `/rpc`, `/mcp`, and `/a2a`
  together. (#112)
