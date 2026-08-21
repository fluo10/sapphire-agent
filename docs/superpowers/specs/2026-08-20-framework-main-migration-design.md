# Design: migrate sapphire-agent to sapphire-framework `main` (facade crate, drop git auto-sync)

Date: 2026-08-20

## Context

`sapphire-agent` already left the standalone `sapphire-workspace` crate: commit
`808c1bb` ("feat(workspace)!: depend on sapphire-framework and enable the redb
store") repointed the dependency at `sapphire-framework-workspace` through a
Cargo `package =` alias, so the extern name `sapphire_workspace` stayed
unchanged. That was half the migration.

The half that remains is that the dependency is pinned to the framework's stale
`feat/framework-migration` branch. Framework `main` has moved on. The decisive
change is `9e03b18` ("refactor!: remove local-workspace auto-sync
(SyncBackend/GitSync/ChangeSource/device)"), which deleted:

- the whole `sapphire-framework-sync` crate — `SyncBackend`, `GitSync`,
  `SyncConfig`/`SyncBackendKind`, `ChangeSource`, `DeviceRegistry`;
- `WorkspaceState::open_configured`, `periodic_sync`, `sync_git`,
  `set/get_sync_backend`, and the auto git-staging in the file hooks;
- `AppContext` device machinery — `device()`, `device_id()`,
  `set_device_defaults()`, `DeviceDefaults`;
- the `git-sync` feature and the `git2` dependency.

That commit states the intent explicitly: "Downstream migrations
(timer/journal/agent) follow in their own PRs." `sapphire-timer` finished its
migration in `289f793` and now pins `branch = "main"`; `sapphire-journal` is
mid-migration on `refactor/migrate-to-framework-main`, and its design doc
(`docs/superpowers/specs/2026-08-10-framework-main-migration-design.md`)
explicitly scopes agent's migration out as a separate spec. This is that spec.

Framework `main` also introduced the **facade crate** `sapphire-framework`
(`93e1596`): a bevy-style single-dependency front for the internal
`sapphire-framework-*` crates, where each feature re-exports one internal crate
as a module (`sapphire_framework::workspace`, `::backend`, …). agent adopts it
here so that a later step needing `backend` or `remote-client` is a feature flag
rather than a new dependency.

### Decision: drop agent's git auto-sync now — a deferral, not a permanent stance

The reasoning journal recorded applies unchanged to agent. Concurrent editing is
the central remote server's job; git becomes a manual, files-as-origin concern;
and a GUI-integrated git story is a separate future rebuild (framework #91).
When the framework offers a git sync again, agent can re-adopt a
*framework-provided* one rather than vendoring its own.

### Decision: remove `standby_mode` rather than repurpose it

agent's cold-standby mode exists solely to run git sync: "only perform git sync,
skip channel listening and heartbeat tasks. Useful for maintaining a backup node
that stays in sync without actively processing messages." With git sync gone a
standby node would only re-index its own local files and would no longer pull
anything from another device — the backup-node role it was built for is gone.
Rather than keep a mode whose documented purpose no longer holds, it is removed.
If a standby role returns, it returns on top of the remote-workspace server,
which is a different design.

This differs from how timer and journal treated their equivalents: they *kept*
`sync_interval_minutes` as a re-index interval. agent keeps that knob too — only
`standby_mode` goes.

### Decision: explicit `sapphire_framework::workspace::…` paths, not the prelude

The facade ships a `prelude` module, but under the `workspace` feature it
exports `AppContext, FileSearchResult, RetrieveConfig, RetrieveParams,
SearchMode, Workspace, WorkspaceState` — **not** `FtsQuery` / `VectorQuery`,
which `src/tools/workspace_tools.rs` needs. Using the prelude would leave that
one file importing by a different route. Every site therefore uses the explicit
`sapphire_framework::workspace::…` path.

## Scope / non-goals

**In scope**

- Replace the `sapphire-framework-workspace` git dependency with the
  `sapphire-framework` facade crate, pinned to `branch = "main"`.
- Remove all use of the #90-deleted sync/git/device APIs.
- Remove `standby_mode` and its branches.
- Keep git deps; build/test/run agent against framework `main`.

**Out of scope (later / separate)**

- Adopting the facade's `backend`, `remote-client`, `remote-server` or `gui`
  features, or the shared `WorkspaceRegistry`.
- Re-introducing git sync, or restoring cross-device session sharing on top of
  the remote-workspace server.
- Fixing the pre-existing gap where periodic re-index only runs on the
  with-channels path (see "Known behaviour retained").
- crates.io release (framework + journal + agent, coordinated, later).

## Changes

### `Cargo.toml`

Replace the workspace dependency:

```toml
# before
sapphire-workspace = { package = "sapphire-framework-workspace", git = "https://github.com/fluo10/sapphire-framework", branch = "feat/framework-migration", default-features = false }
# after
sapphire-framework = { version = "0.1", git = "https://github.com/fluo10/sapphire-framework", branch = "main", default-features = false, features = ["workspace"] }
```

`default-features = false` plus an explicit `workspace` keeps the facade's own
default (`workspace` + `redb-store`) from bypassing agent's feature flags.

Re-route the store/embed features through the facade, which defines a
passthrough for each:

```toml
default = ["redb-store", "lancedb-store", "fastembed-embed", "voice-sherpa"]
redb-store      = ["sapphire-framework/redb-store"]
lancedb-store   = ["sapphire-framework/lancedb-store"]
fastembed-embed = ["sapphire-framework/fastembed-embed"]
```

Also:

- Delete the `git-sync` feature and its `default` entry.
- Delete the `hostname` dependency — only `DeviceDefaults` used it.
- Update `package.description`: "…and a sapphire-workspace memory layer" →
  sapphire-framework.

### `src/config.rs`

- Delete `use sapphire_workspace::SyncConfig;` and the
  `pub sync: Option<SyncConfig>` field with its doc comment.
- Delete the `pub standby_mode: bool` field and its doc comment.
- Keep `sync_interval_minutes`; rewrite its doc comment — it is now the
  **re-index** cadence driving `WorkspaceState::sync()`, with no git leg and no
  reference to `SyncConfig` or `sapphire-workspace 0.10`.

### `src/main.rs`

- Import: `use sapphire_framework::workspace::{AppContext, Workspace as
  SwWorkspace, WorkspaceState};` — `DeviceDefaults` gone.
- `init_app_ctx`: delete the `hostname::get()` lookup and the
  `APP_CTX.set_device_defaults(DeviceDefaults { … })` call; keep `set_cache_dir`
  / `set_data_dir`. Rewrite the function doc, which currently explains that a
  missing `set_device_defaults` made `APP_CTX.device()` panic in the git sync
  backend.
- Workspace open: drop `let sync_config = config.sync.clone().unwrap_or_default();`
  and call `WorkspaceState::open(sw_workspace)` instead of `open_configured`.
  Replace the stale block comment about `WorkspaceConfig` / `open_configured` /
  `sapphire-sync`.
- Initial sync and both periodic loops: `periodic_sync()` → `sync()`. The
  surviving log lines keep the word "sync" but now mean re-index only; rename
  `"Periodic workspace sync enabled: every {}s"` → `"Periodic workspace
  re-index enabled: every {}s"` so operators are not told a git push is
  scheduled. `"Initial workspace sync failed"` and `"Periodic ws sync: {u}
  upserted, {r} removed"` are already accurate for a re-index and stay.
- `standby_mode` removal, six sites:
  1. the `Standby mode : {}` line in the status printout;
  2. the standby-only periodic-sync `tokio::spawn` block, deleted whole (it was
     gated on `config.standby_mode && ws_sync_interval.is_some()`);
  3. `|| config.standby_mode` in the voice-provider guard — drop the clause;
  4. the `if config.standby_mode { tracing::info!("Standby mode enabled: git sync
     only…") }` block, deleted;
  5. `if !config.standby_mode && (config.matrix.is_some() || …)` →
     `if config.matrix.is_some() || config.discord.is_some()`;
  6. the trailing `if config.standby_mode { … ctrl_c … } else { … serve::run … }`
     — delete the standby arm and run the HTTP server unconditionally.
- Update the block comment above the with-channels periodic loop, which explains
  the cadence in terms of `periodic_sync` pulling session JSONLs from another
  device via git.

### `src/tools/workspace_tools.rs`

- Import from `sapphire_framework::workspace::{FtsQuery, VectorQuery,
  WorkspaceState}`.
- `WorkspaceSyncTool::execute`: `periodic_sync()` → `sync()`.
- Rewrite the tool description, which currently promises "index all files and,
  if a git remote is configured, commit and push changes", and the struct doc
  ("Sync the workspace via the configured backend (git commit + push)"). **The
  tool name `workspace_sync` stays** — it is model-facing and appears in users'
  `AGENTS.md` / prompts; renaming it would break those.

### Remaining import sites

`src/heartbeat.rs`, `src/periodic_log.rs`, `src/session.rs`,
`src/tools/builtin_tools.rs` import `WorkspaceState`; `src/tools/mod.rs` names it
fully-qualified. All move to `sapphire_framework::workspace::`.

### Docs

- `config.example.toml`: delete the cold-standby block; update the
  "sapphire-workspace 0.10.x has no path_prefix filter" note.
- `README.md`: the `sapphire-workspace` crates.io links become
  sapphire-framework, and the feature bullet drops "git sync".

## Consequences

- **Cross-device session sharing is lost.** Periodic git sync was what carried
  session JSONLs between devices so the today-digest builder could surface, say,
  a morning voice chat on another host. Users who keep the workspace on an
  external sync service (Syncthing, Dropbox, …) still get those files picked up
  by the periodic re-index; agent simply no longer moves them itself.
- **Operators lose cold standby.** `standby_mode` in an existing config becomes
  an unknown key. agent's `Config` does not `deny_unknown_fields`, so startup is
  unaffected — the setting is silently ignored. This is called out in the commit
  / changelog rather than handled in code.

### Known behaviour retained

Periodic re-indexing only runs on the with-channels path; a channel-less
HTTP-only deployment gets no periodic loop. That was already true before this
change (the other loop was standby-only), so it is preserved rather than fixed.

## Verification

- `cargo build` and `cargo test --workspace` against framework `main`.
- `grep -rn "sapphire_workspace\|SyncConfig\|DeviceDefaults\|periodic_sync\|open_configured\|standby" src/`
  returns nothing. Note that `device_id` legitimately remains throughout
  `src/serve/`, `src/channel/matrix.rs`, `src/heartbeat.rs` and `src/session.rs`:
  that is agent's own voice-satellite / Matrix device identifier, unrelated to
  the framework's removed device registry.
- `cargo tree -i libsqlite3-sys` — a single lineage from `matrix-sdk-sqlite`, as
  `808c1bb` established; `git2` absent.
- `cargo tree -d` — `sapphire-framework-workspace` resolves to one rev, i.e. the
  facade did not pull in a second copy alongside a direct dependency.
- Smoke: start `sapphire-agent`; the workspace indexes; the `workspace_search`
  and `workspace_sync` tools work; the HTTP server binds.

The non-sync framework surface agent depends on was checked against `main`
before writing this spec and is intact: `AppContext::{new,
allow_external_paths, set_cache_dir, set_data_dir}`, `Workspace::resolve`, the
public fields `WorkspaceState::workspace` and `Workspace::root`,
`WorkspaceState::{open, sync, embedder, list_dir}`, and the `FtsQuery` /
`VectorQuery` re-exports. Only the sync/git/device surface is gone.

## Release

Git deps only. No `cargo publish`. A coordinated crates.io release of framework
(facade + internal crates) → journal → agent happens later, once both consumers
are proven stable against the framework API.
