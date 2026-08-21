# sapphire-agent → sapphire-framework `main` migration — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move sapphire-agent onto sapphire-framework `main` through the `sapphire-framework` facade crate, removing its use of the sync/git/device APIs deleted in framework #90 and the `standby_mode` that existed only to run git sync.

**Architecture:** This is a removal/refactor, not a feature — there is no new behaviour, so there are no new tests to write. Each task's verification is that the crate still compiles and the existing tests still pass. Tasks 1 and 2 are **forward-compatible**: `WorkspaceState::open` and `WorkspaceState::sync` exist on both the currently-pinned framework branch and on `main` (verified in `crates/sapphire-framework-workspace/src/workspace_state.rs` on `origin/feat/framework-migration`, lines 141 and 748), so they land while the dependency still points at the old branch. Task 3 flips the dependency to the facade crate on `main` **and** rewrites the imports in one commit — they cannot be separated, because the `sapphire-framework` facade crate does not exist on the old branch. Task 4 is the verification sweep and manual smoke test.

**Tech Stack:** Rust 2024 edition, single root crate `sapphire-agent` plus four sibling crates under `crates/`; git dependency on `sapphire-framework`; tokio, axum, matrix-sdk, serenity, redb/tantivy via the framework.

**Spec:** `docs/superpowers/specs/2026-08-20-framework-main-migration-design.md`

## Global Constraints

- Work happens **only** in the `sapphire-agent` submodule, on the existing branch `feat/framework-migration`. Commit there; make no superproject commits.
- **Git deps only — no `cargo publish`.** The framework dependency stays `git = "https://github.com/fluo10/sapphire-framework", branch = "main"`.
- Do **not** adopt the facade's `backend`, `remote-client`, `remote-server` or `gui` features, the shared `WorkspaceRegistry`, or reintroduce git sync. Those are later phases in `docs/superpowers/specs/2026-08-20-zed-acp-remote-workspace-roadmap.md`.
- Keep `sync_interval_minutes` (now the re-index interval) and the tool name `workspace_sync` — the tool name is model-facing and appears in users' `AGENTS.md`.
- Conventional-commit scopes matter here: `cliff.toml` filters the changelog. Use unscoped or `(workspace)` / `(config)` / `(tools)` / `(deps)` scopes, which all route to the agent changelog. Do **not** use `(core)`, `(cli)`, `(desktop)`, `(rpc)` — those are sibling crates and get filtered out.
- A cold `cargo build` compiles sherpa-onnx via CMake and takes roughly 5–10 minutes. Use `cargo check` for per-task verification; the full `cargo build` runs once, in Task 4.
- After each of Tasks 1–2, the crate must still compile against the **currently pinned** framework. The dependency flip is Task 3.

---

### Task 1: Remove `standby_mode`

Cold-standby mode existed only to run git sync on a backup node. With git sync gone from the framework, the mode has no role, so it is removed rather than repurposed.

**Files:**
- Modify: `src/config.rs:8-16`, `src/config.rs:93-97`
- Modify: `src/main.rs:143`, `src/main.rs:226-249`, `src/main.rs:318-321`, `src/main.rs:391-395`, `src/main.rs:404`, `src/main.rs:613-634`
- Modify: `config.example.toml:21-25`

**Interfaces:**
- Produces: `Config` **without** a `standby_mode` field. Task 2 modifies the same struct and must not reintroduce it.

- [ ] **Step 1: Delete the `standby_mode` field from `Config`**

In `src/config.rs`, delete these five lines (the doc comment and the field):

```rust
    /// Cold-standby mode: only perform git sync, skip channel listening and
    /// heartbeat tasks. Useful for maintaining a backup node that stays in
    /// sync without actively processing messages. Default: false.
    #[serde(default)]
    pub standby_mode: bool,
```

- [ ] **Step 2: Fix the `matrix` field doc that references it**

In `src/config.rs`, the `matrix` field's doc comment currently reads:

```rust
    /// Matrix channel configuration. Both `matrix` and `discord` may be
    /// configured at once — when set, both run concurrently in the
    /// same `serve` process. At least one of them is required (unless
    /// `standby_mode = true`).
```

Replace it with:

```rust
    /// Matrix channel configuration. Both `matrix` and `discord` may be
    /// configured at once — when set, both run concurrently in the
    /// same `serve` process. Both may also be omitted, in which case the
    /// agent serves only the HTTP API.
```

- [ ] **Step 3: Delete the status-printout line**

In `src/main.rs`, delete this line:

```rust
            println!("  Standby mode      : {}", config.standby_mode);
```

- [ ] **Step 4: Delete the standby-only periodic-sync loop**

In `src/main.rs`, delete this whole block (comment included). The with-channels path further down has a richer loop that supersedes it:

```rust
            // Standby mode runs a minimal periodic-sync loop. The
            // with-channels code path replaces this with a richer loop
            // below that also rebuilds today_digests on the same tick
            // (so we don't pay periodic_sync twice per interval).
            if config.standby_mode
                && let Some(dur) = ws_sync_interval
            {
                tracing::info!("Periodic workspace sync enabled: every {}s", dur.as_secs());
                let ws = Arc::clone(&ws_state);
                tokio::spawn(async move {
                    let mut tick = tokio::time::interval(dur);
                    tick.tick().await;
                    loop {
                        tick.tick().await;
                        let state = ws.lock().expect("ws_state mutex poisoned");
                        match state.periodic_sync() {
                            Ok((u, r)) => {
                                tracing::info!("Periodic ws sync: {u} upserted, {r} removed");
                            }
                            Err(e) => tracing::warn!("Periodic ws sync failed: {e:#}"),
                        }
                    }
                });
            }
```

- [ ] **Step 5: Drop the clause from the voice-provider guard**

In `src/main.rs`, change:

```rust
            let voice_providers = if config.stt_providers.is_empty()
                && config.tts_providers.is_empty()
                || config.standby_mode
            {
```

to:

```rust
            let voice_providers = if config.stt_providers.is_empty()
                && config.tts_providers.is_empty()
            {
```

- [ ] **Step 6: Delete the standby announcement log**

In `src/main.rs`, delete:

```rust
            if config.standby_mode {
                tracing::info!(
                    "Standby mode enabled: git sync only, skipping channel and heartbeat"
                );
            }
```

- [ ] **Step 7: Simplify the channel guard**

In `src/main.rs`, change:

```rust
            if !config.standby_mode && (config.matrix.is_some() || config.discord.is_some()) {
```

to:

```rust
            if config.matrix.is_some() || config.discord.is_some() {
```

- [ ] **Step 8: Run the HTTP server unconditionally**

In `src/main.rs`, replace this block:

```rust
            if config.standby_mode {
                // In standby mode, keep the process alive for periodic git
                // sync only — no HTTP server, no channel, no heartbeat.
                tracing::info!("Standby mode: waiting for shutdown signal (Ctrl-C)");
                tokio::signal::ctrl_c()
                    .await
                    .expect("Failed to listen for Ctrl-C");
                tracing::info!("Shutting down standby process");
            } else {
                // ── HTTP API server ─────────────────────────────────────────
                let addr = bind
                    .or_else(|| {
                        config
                            .serve
                            .as_ref()
                            .map(|s| format!("{}:{}", s.host, s.port))
                    })
                    .unwrap_or_else(|| "127.0.0.1:9000".to_string());

                serve::run(addr, Arc::clone(&serve_state)).await?;
            }
```

with the `else` body, de-indented one level:

```rust
            // ── HTTP API server ─────────────────────────────────────────────
            let addr = bind
                .or_else(|| {
                    config
                        .serve
                        .as_ref()
                        .map(|s| format!("{}:{}", s.host, s.port))
                })
                .unwrap_or_else(|| "127.0.0.1:9000".to_string());

            serve::run(addr, Arc::clone(&serve_state)).await?;
```

- [ ] **Step 9: Remove the cold-standby block from the config example**

In `config.example.toml`, delete these five lines:

```toml
# Cold-standby mode: only perform git sync — no channel listening,
# no heartbeat, no HTTP server. The process stays alive for periodic
# sync and shuts down on Ctrl-C. Manual switchover to active mode
# by flipping this flag and restarting.
# standby_mode = true
```

- [ ] **Step 10: Verify it compiles and the tests still pass**

Run: `cargo check`
Expected: builds with no errors. If `ws_sync_interval` is reported as unused, that means Step 4 deleted the only consumer — it should not, because the with-channels loop still uses it. Re-check that Step 4 deleted only the standby block.

Run: `cargo test --lib`
Expected: all existing tests pass. No test references `standby_mode`, so none should need editing.

Run: `grep -rn "standby" src/ config.example.toml`
Expected: no output.

- [ ] **Step 11: Commit**

```bash
git add src/config.rs src/main.rs config.example.toml
git commit -m "refactor!: remove standby_mode (framework #90 removed git sync)"
```

---

### Task 2: Drop the sync / git / device APIs deleted in framework #90

**Files:**
- Modify: `src/config.rs:2`, `src/config.rs:93-117` (line numbers shift after Task 1)
- Modify: `src/main.rs:37`, `src/main.rs:41-71`, `src/main.rs:203-222`, `src/main.rs:511-540`
- Modify: `src/tools/workspace_tools.rs:544-560`, `src/tools/workspace_tools.rs:578`
- Modify: `Cargo.toml`

**Interfaces:**
- Consumes: `Config` without `standby_mode` (Task 1).
- Produces: `Config` **without** a `sync` field; `sync_interval_minutes: Option<u32>` stays and is now the re-index interval. All workspace refreshes go through `WorkspaceState::sync() -> Result<(usize, usize)>` (upserted, removed). Task 3 rewrites the import paths of these same files and must not reintroduce any of the removed names.

- [ ] **Step 1: Remove `SyncConfig` from the config**

In `src/config.rs`, delete the import on line 2:

```rust
use sapphire_workspace::SyncConfig;
```

and delete the `sync` field with its doc comment:

```rust
    /// Workspace sync configuration.
    ///
    /// The workspace-level config (`{workspace_dir}/.sapphire-agent/config.toml`)
    /// provides shared defaults. This per-user `[sync]` section, when present,
    /// takes precedence — allowing each user to override the workspace defaults.
    #[serde(default)]
    pub sync: Option<SyncConfig>,
```

- [ ] **Step 2: Rewrite the `sync_interval_minutes` doc**

In `src/config.rs`, replace the doc comment above `pub sync_interval_minutes: Option<u32>`:

```rust
    /// How often the agent runs the periodic workspace sync cycle, in
    /// minutes. Unset or `0` disables periodic sync entirely. Each tick
    /// runs `WorkspaceState::periodic_sync`, which does a git sync **and**
    /// an mtime-based refresh of the retrieve cache — one cadence drives
    /// both.
    ///
    /// Lives at the config root (not inside `[sync]`) because the cadence
    /// spans both `sapphire-sync` and `sapphire-retrieve`; nesting it
    /// under `[sync]` would have implied a sync-only knob and forced a
    /// duplicate for the retrieve side. Upstream relocated it out of
    /// `SyncConfig` for the same reason in sapphire-workspace 0.10.
```

with:

```rust
    /// How often the agent re-indexes the workspace, in minutes. Unset or
    /// `0` disables the periodic re-index entirely. Each tick runs
    /// `WorkspaceState::sync`, an mtime-based refresh of the retrieve
    /// cache — this is what picks up files edited outside the agent.
    ///
    /// There is no git leg any more: the framework removed local-workspace
    /// auto-sync, so nothing is committed or pushed on this cadence.
```

- [ ] **Step 3: Drop `DeviceDefaults` from `main.rs`**

In `src/main.rs`, change the import:

```rust
use sapphire_workspace::{AppContext, DeviceDefaults, Workspace as SwWorkspace, WorkspaceState};
```

to:

```rust
use sapphire_workspace::{AppContext, Workspace as SwWorkspace, WorkspaceState};
```

- [ ] **Step 4: Strip the device wiring from `init_app_ctx`**

In `src/main.rs`, replace the function's doc comment:

```rust
/// Inject host-platform paths and device facts into [`APP_CTX`] before any
/// code touches a [`SwWorkspace`]. The sapphire-workspace library deliberately
/// does not depend on `dirs` / `hostname`, so each host app has to wire these
/// up itself at startup. Missing this made `APP_CTX.device()` panic the first
/// time the git sync backend tried to record device info.
```

with:

```rust
/// Inject host-platform paths into [`APP_CTX`] before any code touches a
/// [`SwWorkspace`]. The framework deliberately does not depend on `dirs`, so
/// each host app resolves its own cache / data directories at startup.
```

and delete the device block from the body — everything from the blank line after `APP_CTX.set_data_dir(data_dir);` to the end of the function:

```rust
    let hostname = hostname::get()
        .ok()
        .and_then(|s| s.into_string().ok())
        .unwrap_or_default();
    APP_CTX.set_device_defaults(DeviceDefaults {
        hostname,
        app_id: env!("CARGO_PKG_NAME").to_owned(),
        app_version: env!("CARGO_PKG_VERSION").to_owned(),
        platform: std::env::consts::OS.to_owned(),
        arch: std::env::consts::ARCH.to_owned(),
    });
```

`set_cache_dir` and `set_data_dir` stay.

- [ ] **Step 5: Open the workspace with `open` instead of `open_configured`**

In `src/main.rs`, replace this block:

```rust
            // ── sapphire-workspace (search, file ops, git sync) ─────────────
            let sw_workspace = SwWorkspace::resolve(&APP_CTX, Some(&workspace_dir))
                .context("Failed to resolve sapphire-workspace")?;
            // Use the [sync] section from the agent config directly.
            // WorkspaceConfig was removed in sapphire-workspace 0.8.0;
            // open_configured now takes &SyncConfig. In 0.10 the periodic
            // cadence moved out of SyncConfig because it drives both
            // sapphire-sync and sapphire-retrieve — keeping one knob
            // avoids a duplicate `[retrieve]` cadence. It now lives at
            // the agent config root as `sync_interval_minutes`, and each
            // `periodic_sync()` call refreshes the retrieve cache too.
            let sync_config = config.sync.clone().unwrap_or_default();
            let ws_sync_interval = config
                .sync_interval_minutes
                .filter(|&m| m > 0)
                .map(|m| std::time::Duration::from_secs(m as u64 * 60));
            let ws_state = WorkspaceState::open_configured(sw_workspace, &sync_config)
                .context("Failed to open WorkspaceState")?;
            if let Err(e) = ws_state.periodic_sync() {
                tracing::warn!("Initial workspace sync failed: {e}");
            }
```

with:

```rust
            // ── framework workspace (search, file ops) ──────────────────────
            let sw_workspace = SwWorkspace::resolve(&APP_CTX, Some(&workspace_dir))
                .context("Failed to resolve the framework workspace")?;
            // `sync_interval_minutes` is the re-index cadence. The framework
            // removed local-workspace auto-sync, so a tick is an mtime-based
            // refresh of the retrieve cache and nothing else.
            let ws_sync_interval = config
                .sync_interval_minutes
                .filter(|&m| m > 0)
                .map(|m| std::time::Duration::from_secs(m as u64 * 60));
            let ws_state =
                WorkspaceState::open(sw_workspace).context("Failed to open WorkspaceState")?;
            if let Err(e) = ws_state.sync() {
                tracing::warn!("Initial workspace re-index failed: {e}");
            }
```

- [ ] **Step 6: Update the with-channels periodic loop**

In `src/main.rs`, replace the comment above the surviving periodic loop:

```rust
                // ── Periodic workspace sync + today-digest rebuild ──────
                // Same cadence drives both: when `periodic_sync` pulls
                // session JSONLs from another device via git, the digest
                // builder picks them up on the same tick so cross-device
                // "today's notes" become visible without waiting for the
                // next day-boundary daily-log generation.
```

with:

```rust
                // ── Periodic workspace re-index + today-digest rebuild ──
                // Same cadence drives both: `sync()` picks up session
                // JSONLs and notes written outside the agent, and the
                // digest builder folds them in on the same tick so
                // "today's notes" stay current without waiting for the
                // next day-boundary daily-log generation.
```

Then, in the `if let Some(dur) = ws_sync_interval` block that follows the comment (the log line sits before the `tokio::spawn`, not inside the loop), change:

```rust
                    tracing::info!("Periodic workspace sync enabled: every {}s", dur.as_secs());
```

to:

```rust
                    tracing::info!(
                        "Periodic workspace re-index enabled: every {}s",
                        dur.as_secs()
                    );
```

and change the call:

```rust
                                match state.periodic_sync() {
```

to:

```rust
                                match state.sync() {
```

- [ ] **Step 7: Update the `workspace_sync` tool**

In `src/tools/workspace_tools.rs`, replace the struct doc:

```rust
/// Sync the workspace via the configured backend (git commit + push).
```

with:

```rust
/// Re-index the workspace so files edited outside the agent become
/// searchable. There is no git leg — the framework removed auto-sync.
```

Replace the tool description:

```rust
                description: "Sync the workspace: index all files and, if a git \
                    remote is configured, commit and push changes."
                    .into(),
```

with:

```rust
                description: "Re-index the workspace so files changed outside \
                    the agent become searchable. Does not touch git."
                    .into(),
```

Replace the call in `execute`:

```rust
        let (upserted, removed) = state.periodic_sync().context("Failed to sync workspace")?;
```

with:

```rust
        let (upserted, removed) = state.sync().context("Failed to re-index workspace")?;
```

The tool **name** `workspace_sync` stays unchanged.

- [ ] **Step 8: Drop the `git-sync` feature and the `hostname` dependency**

In `Cargo.toml`, change the default feature list:

```toml
default = ["redb-store", "lancedb-store", "fastembed-embed", "git-sync", "voice-sherpa"]
```

to:

```toml
default = ["redb-store", "lancedb-store", "fastembed-embed", "voice-sherpa"]
```

Delete the feature definition:

```toml
git-sync = ["sapphire-workspace/git-sync"]
```

Delete the dependency and its comment:

```toml
# System hostname lookup for the sapphire-workspace DeviceDefaults init
hostname = "0.4"
```

- [ ] **Step 9: Verify it compiles and the tests still pass**

Run: `cargo check`
Expected: builds with no errors and no `unused import` warnings.

Run: `cargo test --lib`
Expected: all existing tests pass.

Run: `grep -rn "SyncConfig\|DeviceDefaults\|periodic_sync\|open_configured\|git-sync" src/ Cargo.toml`
Expected: no output.

Run: `grep -rn "hostname::get\|^hostname" src/ Cargo.toml`
Expected: no output. Grep for the bare word `hostname` instead and you will get a dozen false positives from `src/serve/mcp.rs`, where `hostname` is an MCP tool argument name — that is agent's own concept and must not be touched.

- [ ] **Step 10: Commit**

```bash
git add src/config.rs src/main.rs src/tools/workspace_tools.rs Cargo.toml Cargo.lock
git commit -m "refactor!: drop the sync/git/device APIs removed in framework #90"
```

---

### Task 3: Switch to the `sapphire-framework` facade crate on `main`

This task flips the dependency and rewrites the imports together. They cannot be split: the facade crate does not exist on the currently-pinned branch, so an import rewrite alone would not compile.

**Files:**
- Modify: `Cargo.toml`
- Modify: `src/main.rs:37`, `src/heartbeat.rs:23`, `src/periodic_log.rs:22`, `src/session.rs:22`, `src/tools/builtin_tools.rs:7`, `src/tools/mod.rs:187`, `src/tools/workspace_tools.rs:6`
- Modify: `README.md:3`, `README.md:13`
- Modify: `config.example.toml:195`

**Interfaces:**
- Consumes: the code from Tasks 1–2, which no longer names any removed API.
- Produces: every framework type reached through `sapphire_framework::workspace::…`. The extern crate name is `sapphire_framework`; the alias `sapphire_workspace` no longer exists anywhere.

- [ ] **Step 1: Replace the dependency**

In `Cargo.toml`, replace:

```toml
# Workspace: file indexing, FTS, vector search, git sync
# lancedb-store is gated behind the "lancedb-store" feature (default on)
sapphire-workspace = { package = "sapphire-framework-workspace", git = "https://github.com/fluo10/sapphire-framework", branch = "feat/framework-migration", default-features = false }
```

with:

```toml
# Local-first workspace framework: file indexing, FTS, vector search.
# The facade crate (bevy-style) fronts the internal sapphire-framework-*
# crates; each feature re-exports one as a module. `default-features = false`
# plus an explicit `workspace` keeps the facade's own default (workspace +
# redb-store) from bypassing the feature flags below.
# lancedb-store is gated behind the "lancedb-store" feature (default on)
sapphire-framework = { version = "0.1", git = "https://github.com/fluo10/sapphire-framework", branch = "main", default-features = false, features = ["workspace"] }
```

- [ ] **Step 2: Re-route the store / embed features through the facade**

In `Cargo.toml`, replace:

```toml
redb-store = ["sapphire-workspace/redb-store"]
lancedb-store = ["sapphire-workspace/lancedb-store"]
fastembed-embed = ["sapphire-workspace/fastembed-embed"]
```

with:

```toml
redb-store = ["sapphire-framework/redb-store"]
lancedb-store = ["sapphire-framework/lancedb-store"]
fastembed-embed = ["sapphire-framework/fastembed-embed"]
```

- [ ] **Step 3: Update the package description**

In `Cargo.toml`, replace:

```toml
description = "A personal AI assistant agent with Matrix/Discord channels, Anthropic backend, and a sapphire-workspace memory layer"
```

with:

```toml
description = "A personal AI assistant agent with Matrix/Discord channels, Anthropic backend, and a sapphire-framework memory layer"
```

- [ ] **Step 4: Rewrite the six `use` statements**

Apply each of these replacements.

`src/main.rs`:

```rust
use sapphire_workspace::{AppContext, Workspace as SwWorkspace, WorkspaceState};
```
→
```rust
use sapphire_framework::workspace::{AppContext, Workspace as SwWorkspace, WorkspaceState};
```

`src/heartbeat.rs`, `src/periodic_log.rs`, `src/session.rs`, `src/tools/builtin_tools.rs` — each has the same line:

```rust
use sapphire_workspace::WorkspaceState;
```
→
```rust
use sapphire_framework::workspace::WorkspaceState;
```

`src/tools/workspace_tools.rs`:

```rust
use sapphire_workspace::{FtsQuery, VectorQuery, WorkspaceState};
```
→
```rust
use sapphire_framework::workspace::{FtsQuery, VectorQuery, WorkspaceState};
```

Use the explicit `workspace` module path, **not** `sapphire_framework::prelude::*` — the prelude does not export `FtsQuery` / `VectorQuery`, so `workspace_tools.rs` would have to import by a second route.

- [ ] **Step 5: Rewrite the one fully-qualified reference**

In `src/tools/mod.rs`, change:

```rust
    state: Arc<Mutex<sapphire_workspace::WorkspaceState>>,
```

to:

```rust
    state: Arc<Mutex<sapphire_framework::workspace::WorkspaceState>>,
```

- [ ] **Step 6: Verify it compiles and the tests still pass**

Run: `cargo check`
Expected: builds. Cargo re-resolves the git dependency, so the first run fetches `sapphire-framework` at `main` and rebuilds the framework crates. `Cargo.lock` changes.

If the build fails with an unresolved import, the likely cause is a framework API that drifted between the two branches for a reason unrelated to #90. Read the error, find the current name in `sapphire-framework/crates/sapphire-framework-workspace/src/lib.rs`, and fix the call site — do not add framework features to work around it.

Run: `cargo test --lib`
Expected: all existing tests pass.

Run: `grep -rn "sapphire_workspace\|sapphire-workspace" src/ Cargo.toml`
Expected: no output.

- [ ] **Step 7: Update the README**

In `README.md`, replace line 3:

```markdown
A personal AI assistant agent that lives in a [`sapphire-workspace`](https://crates.io/crates/sapphire-workspace) and talks to me through Matrix and Discord.
```

with:

```markdown
A personal AI assistant agent that lives in a [`sapphire-framework`](https://github.com/fluo10/sapphire-framework) workspace and talks to me through Matrix and Discord.
```

and replace the Workspace bullet:

```markdown
- **Workspace**: backed by [`sapphire-workspace`](https://crates.io/crates/sapphire-workspace) — file index, full-text + vector search (LanceDB), git sync.
```

with:

```markdown
- **Workspace**: backed by [`sapphire-framework`](https://github.com/fluo10/sapphire-framework) — file index, full-text + vector search (redb + tantivy, optional LanceDB).
```

The links point at the GitHub repository rather than crates.io because the framework crates are not published yet.

- [ ] **Step 8: Update the config example's stale crate reference**

In `config.example.toml`, replace:

```toml
# Caveat: workspace retrieve (FTS / vector search) currently spans every
# namespace because sapphire-workspace 0.10.x has no path_prefix filter.
```

with:

```toml
# Caveat: workspace retrieve (FTS / vector search) currently spans every
# namespace because sapphire-framework has no path_prefix filter.
```

- [ ] **Step 9: Commit**

```bash
git add Cargo.toml Cargo.lock src/ README.md config.example.toml
git commit -m "feat(workspace)!: depend on the sapphire-framework facade crate on main"
```

---

### Task 4: Verification sweep and smoke test

**Files:** none modified unless a check fails.

**Interfaces:**
- Consumes: the finished migration from Tasks 1–3.

- [ ] **Step 1: Full build with default features**

Run: `cargo build`
Expected: succeeds. This is the slow one — sherpa-onnx builds C++ via CMake, so allow 5–10 minutes on a cold cache.

- [ ] **Step 2: Full test run across the workspace**

Run: `cargo test --workspace`
Expected: all tests pass. The sibling crates (`sapphire-agent-rpc`, `sapphire-call-core`, `sapphire-call-cli`, `sapphire-call-desktop`) do not depend on the framework and should be unaffected; if one fails, it is unrelated to this migration — say so rather than papering over it.

- [ ] **Step 3: Confirm the removed surface is gone**

Run: `grep -rn "sapphire_workspace\|sapphire-workspace\|SyncConfig\|DeviceDefaults\|periodic_sync\|open_configured\|standby\|git-sync" src/ Cargo.toml README.md config.example.toml`
Expected: no output.

Note that `device_id` legitimately remains throughout `src/serve/`, `src/channel/matrix.rs`, `src/heartbeat.rs`, `src/heartbeat_config.rs` and `src/session.rs`. That is agent's own voice-satellite / Matrix device identifier and has nothing to do with the framework's removed device registry — do not remove it.

- [ ] **Step 4: Confirm the dependency graph**

Run: `cargo tree -i libsqlite3-sys`
Expected: a single lineage, from `matrix-sdk-sqlite`. The framework must not appear — that was the point of the redb migration in `808c1bb`.

Run: `cargo tree -i git2`
Expected: `git2` is not in the graph at all (the command reports that nothing matches).

Run: `cargo tree -d`
Expected: `sapphire-framework-workspace` does not appear as a duplicate. A duplicate would mean something still depends on it directly alongside the facade.

- [ ] **Step 5: Smoke-test the binary**

Run: `cargo run -- --help`
Expected: the CLI help prints without panicking.

Then start the agent against a real config and confirm, by watching the log:

1. the workspace resolves and the initial re-index runs (no "Initial workspace re-index failed" warning);
2. the HTTP server binds — with no `[matrix]` or `[discord]` configured it should still bind, which is the behaviour Task 1 Step 8 unlocked;
3. a `workspace_search` call returns hits;
4. a `workspace_sync` call reports "Synced: N files indexed, M removed."

- [ ] **Step 6: Record the breaking changes for the changelog**

The two behaviour removals need to be visible to anyone upgrading. Confirm the commit messages from Tasks 1 and 2 both carry the `!` breaking marker (`refactor!:`), since `cliff.toml` keys the changelog off conventional-commit syntax. If either is missing it, do not rewrite published history — add a note to `CHANGELOG.md` under the unreleased section instead:

```markdown
### Breaking

- `standby_mode` has been removed. Its only purpose was running git sync on a
  backup node, and the framework no longer ships local-workspace auto-sync.
  The setting is ignored if left in an existing config.
- The `[sync]` config section has been removed. `sync_interval_minutes` stays,
  and is now purely the workspace re-index cadence — nothing is committed or
  pushed on that tick.
- Sessions are no longer carried between devices. Keep the workspace on an
  external sync service if you need that, or wait for remote-workspace support.
```

- [ ] **Step 7: Commit any changelog note**

Only if Step 6 required a `CHANGELOG.md` edit:

```bash
git add CHANGELOG.md
git commit -m "docs(changelog): record the framework-main migration breaking changes"
```
