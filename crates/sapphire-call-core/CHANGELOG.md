# Changelog

All notable changes to `sapphire-call-core` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.7.0] - 2026-05-23

First release of `sapphire-call-core` as a standalone crate. Extracted
from the former `sapphire-call` crate so the CLI (`sapphire-call-cli`)
and the desktop GUI (`sapphire-call-desktop`) can share the same source
of truth for endpoint configuration and per-installation identity.

The version is aligned with `sapphire-agent` / `sapphire-agent-rpc`
0.7.0 to make the workspace easy to read at a glance; future patch and
minor bumps will track this crate's own change cadence.

### Added

- **`ServerConfig`** — TOML schema covering the agent endpoint URL and
  per-`room_profile` bearer-token map. Loaded from
  `~/.config/sapphire-call/config.toml` by the CLI and from
  `~/.config/sapphire-call-desktop/config.toml` by the desktop client;
  both reuse the same struct.
- **`device_id` resolution** — UUID v7 generated on first run and
  persisted alongside the config, so the agent's voice / device-default
  session routing has a stable identifier for each installation.
- **Path expansion + XDG dir helpers** — `~/` and platform-appropriate
  config / data / cache dirs are resolved via `shellexpand` +
  `directories`, surfaced as small helper functions the CLI and desktop
  binaries both call.
