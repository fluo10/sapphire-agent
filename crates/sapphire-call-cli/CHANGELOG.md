# Changelog

All notable changes to `sapphire-call-cli` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

This crate is the continuation of the former `sapphire-call` crate
(published as `sapphire-call` on crates.io up through `0.6.1`). The
binary it ships is still named `sapphire-call`; only the crate name
changed. Historical entries describing the voice-satellite pipeline,
wake / VAD / TTS plumbing, and earlier feature work live in the root
`CHANGELOG.md`.

## [Unreleased]

## [0.7.0] - 2026-05-23

First release under the new crate name. The version is aligned with
`sapphire-agent` 0.7.0 so the workspace versions are easy to read at a
glance; future bumps will track this crate's own change cadence.

### Changed

- **Renamed from `sapphire-call` to `sapphire-call-cli`** as part of the
  workspace split that introduced `sapphire-call-desktop`. The binary
  produced by `cargo install sapphire-call-cli` is still
  `sapphire-call`, so existing scripts and `sapphire-call voice ...`
  invocations are unaffected. Users installing from crates.io should
  switch from `cargo install sapphire-call` to
  `cargo install sapphire-call-cli`.
- **Shared config + device-id helpers extracted to `sapphire-call-core`.**
  The CLI now depends on `sapphire-call-core` for `ServerConfig` and
  per-installation `device_id` resolution, so adding new client targets
  (e.g. the desktop GUI) doesn't fork those types. The on-disk config
  format and path (`~/.config/sapphire-call/config.toml`) are
  unchanged.
