# Repository conventions for AI assistants

Loaded by Claude Code at session start. Keep it short and operational; deep design notes belong in `docs/` or per-crate READMEs.

## Conventional commit scopes

This workspace ships five crates and uses **conventional-commit scopes** to route changelog entries (and only changelog entries — see "Misattribution" below).

| Scope | Targets |
|---|---|
| `(desktop)` | `crates/sapphire-call-desktop/` (bevy GUI client) |
| `(cli)` or `(call)` | `crates/sapphire-call-cli/` (voice satellite binary) |
| `(rpc)` | `crates/sapphire-agent-rpc/` (RPC client library) |
| `(core)` | `crates/sapphire-call-core/` (shared config + device_id) |
| unscoped, or `(agent)` / `(messages)` / `(voice)` / `(serve)` / `(channel)` / `(matrix)` / `(discord)` / `(sessions)` / `(chat)` / `(timer)` / `(heartbeat)` / `(memory)` / `(image-cache)` / `(api)` / `(tools)` / `(search)` / `(fts)` / `(mcp)` / `(features)` / `(workspace)` / `(deps)` | the agent binary itself (top-level crate) |
| `(release)`, `(release-plz)`, `(fmt)`, `(ci)`, `(test)` | infrastructure — workspace-wide, no semver impact intended |

Use the scope that names the crate or sub-area you're changing. The release-plz changelog filter (`cliff.toml`) keys off this.

## Misattribution: agent gets false-positive release PRs

`sapphire-agent`'s package manifest lives at the workspace root (`./Cargo.toml`), so **release-plz attributes any workspace-root file change to it** — including `Cargo.lock`, `release-plz.toml`, `README.md`. Any sibling-crate commit that touches `Cargo.lock` (which is almost all of them, since deps update) therefore proposes a `sapphire-agent` bump too.

This was investigated and confirmed empirically (see closed [#143](https://github.com/fluo10/sapphire-agent/pull/143), [#144](https://github.com/fluo10/sapphire-agent/pull/144)). release-plz's release-trigger logic is purely path-based; no config (regex filter, git-cliff `skip = true`, etc.) suppresses it. The only structural fix is moving `sapphire-agent` to `crates/sapphire-agent/`, which we explicitly chose not to do — the main agent crate stays at the workspace root.

**Operational rule**: when release-plz proposes a `sapphire-agent` bump whose changelog body is empty or only contains entries unrelated to agent code, treat it as misattribution and close the PR. The `cliff.toml` filter makes this obvious: legitimate agent releases produce a non-empty, agent-focused changelog; misattribution releases produce a blank or noise-only one.

## Extending the changelog filter

`cliff.toml` skips commits whose scope is in this list:

```
desktop | cli | call | rpc | core | release | release-plz | fmt | ci | test | deps
```

- **Sibling crate scope, new addition**: extend the alternation in `cliff.toml` → `[[git.commit_parsers]] message = '^[a-z]+\\((…)\\):'` at the same time as the first commit using it. Otherwise the first such commit will pollute the agent changelog.
- **New agent-internal sub-area** (e.g. you introduce `feat(notifications):`): **no `cliff.toml` change needed**. Agent-internal scopes fall through to the conventional-type parsers (`feat` → "added", `fix` → "fixed", etc.) and show up in the agent changelog as expected.

Rust's `regex` crate has no negative lookahead, so the filter is an allowlist of "scopes to drop" rather than "scopes to keep". This is why new agent-internal scopes don't need re-listing — only new sibling/infra scopes do.

## Release flow recap

- `release-plz` creates per-package tags (`sapphire-agent-v*`, `sapphire-call-desktop-v*`, ...) and GitHub releases on push to main.
- `.github/workflows/release-plz.yml` parses the `releases` output with `jq` and chains into reusable build workflows (`release.yml`, `release-cli.yml`, `release-desktop.yml`) which attach platform binaries to each release.
- Tags pushed by `GITHUB_TOKEN` don't fire downstream workflows on their own — the `workflow_call` chain inside `release-plz.yml` is the path; `workflow_dispatch` with `-f tag=…` is the retroactive escape hatch.
- `sapphire-call-desktop` carries `publish = false`; release-plz still tags + releases it because `release-plz.toml` sets `release = true` explicitly (the default for `publish = false` crates is to skip them).
