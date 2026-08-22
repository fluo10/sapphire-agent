# Layered Configuration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a workspace-level config supply the settings that describe the agent, while the host-local config keeps credentials, addresses and machine paths, with the workspace layer restricted by an allowlist.

**Architecture:** Both layers are parsed to `toml::Value`, the workspace layer is filtered through an allowlist of TOML key paths, the two are deep-merged with the host on top, and the result is deserialized into `Config` once. All of the merging logic lives in a new `src/config_layer.rs` as pure functions over `toml::Value`, so every rule is unit-testable from TOML string literals; `src/config.rs` only gains a thin `load_layered` that sequences the file reads.

**Tech Stack:** Rust 2024, `toml` 1.0, `serde`, `anyhow`. Single binary crate `sapphire-agent`; modules are declared in `src/main.rs`.

**Spec:** `docs/superpowers/specs/2026-08-21-config-layering-design.md`

## Global Constraints

- Work only in the `sapphire-agent` submodule, on branch `feat/config-layering`. No commits in the parent `project-sapphire` superproject.
- **The allowlist holds TOML key paths, not Rust field names.** serde renames several map fields: `room_profiles` → `room_profile`, `memory_namespaces` → `memory_namespace`, `voice_pipelines` → `voice_pipeline`, `stt_providers`/`tts_providers` → `stt_provider`/`tts_provider`. `providers` and `profiles` are not renamed.
- **The workspace layer is opt-in by existence.** No `{workspace_dir}/.sapphire-agent/config.toml`, no layer, and behaviour must be byte-for-byte what it is today.
- **Host wins** on every conflict. Tables deep-merge key by key; scalars and arrays are replaced wholesale.
- **Never refuse to start over the workspace layer.** A non-allowlisted key is warned about and ignored. The host layer keeps today's behaviour: a malformed host config is still a hard error.
- Do not add a third layer, do not add env-var layering, do not write or create the workspace config, and do not implement hot reload (issue #174) or voice-identity sharing (issue #173).
- Verification is `cargo check --all-targets` and `cargo test --bin sapphire-agent`. Do **not** run `cargo build` or a bare `cargo test`: this crate's default features pull sherpa-onnx, lancedb/datafusion and matrix-sdk, and a full link runs for tens of minutes. `cargo test --bin sapphire-agent` compiles the same binary target the unit tests live in and is the intended command.
- Run every cargo command in the foreground with a 600000 ms timeout. If it times out, run it again — cargo resumes from cache.
- Conventional-commit scope routes the changelog via `cliff.toml`. Never use the scopes `(core)`, `(cli)`, `(desktop)` or `(rpc)` — they belong to sibling crates and get filtered out of this crate's changelog. `(config)` is an agent-internal scope and needs no `cliff.toml` change.

## File Structure

| File | Responsibility |
|---|---|
| `src/config_layer.rs` (new) | Everything about turning two TOML documents into one: the allowlist, the path matcher, the filter, the deep merge, the provenance calculation. Pure functions over `toml::Value`; no filesystem access. |
| `src/config.rs` (modify) | Keeps `Config`. Gains `resolve_workspace_dir` as a free function and `Config::load_layered`, which sequences the two file reads and calls into `config_layer`. `Config::load` stays as the single-file primitive. |
| `src/main.rs` (modify) | Declares the new module, calls `load_layered`, emits the rejected-key warning, and reports provenance from `verify`. |
| `config.example.toml` (modify) | Documents the workspace layer for users. |

---

### Task 1: The allowlist and its path matcher

**Files:**
- Create: `src/config_layer.rs`
- Modify: `src/main.rs` (module declaration list, around line 5-23)

**Interfaces:**
- Produces: `pub const WORKSPACE_ALLOWLIST: &[&[&str]]` and `pub fn path_allowed(path: &[&str]) -> bool`. Task 2 uses `path_allowed`; Task 4 uses both.

- [ ] **Step 1: Declare the module**

In `src/main.rs`, add `mod config_layer;` to the module list, keeping it alphabetical — it goes directly after `mod config;`:

```rust
mod config;
mod config_layer;
mod context_compression;
```

- [ ] **Step 2: Write the failing tests**

Create `src/config_layer.rs` containing only this test module for now:

```rust
//! Layering the workspace-level config under the host-local one.
//!
//! Everything here is a pure function over [`toml::Value`]: the allowlist that
//! bounds what the workspace layer may set, the merge that combines the two
//! layers, and the provenance calculation that lets `verify` say where a value
//! came from. Nothing in this module touches the filesystem — `Config::load_layered`
//! reads the files and calls in.

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_leaf_is_allowed() {
        assert!(path_allowed(&["anthropic", "model"]));
        assert!(path_allowed(&["anthropic", "system_prompt"]));
    }

    #[test]
    fn sibling_of_an_allowed_leaf_is_not_allowed() {
        assert!(!path_allowed(&["anthropic", "api_key"]));
    }

    #[test]
    fn an_entry_authorises_everything_beneath_it() {
        assert!(path_allowed(&["timer"]));
        assert!(path_allowed(&["timer", "preset"]));
        assert!(path_allowed(&["timer", "preset", "steps", "label"]));
    }

    #[test]
    fn wildcard_matches_any_single_map_key() {
        assert!(path_allowed(&["room_profile", "work", "profile"]));
        assert!(path_allowed(&["room_profile", "anything-at-all", "rooms"]));
    }

    #[test]
    fn wildcard_does_not_authorise_a_host_only_sibling() {
        assert!(!path_allowed(&["room_profile", "work", "api_keys"]));
        assert!(!path_allowed(&["providers", "local", "api_key"]));
    }

    #[test]
    fn provider_definition_fields_are_allowed() {
        assert!(path_allowed(&["providers", "local", "type"]));
        assert!(path_allowed(&["providers", "local", "base_url"]));
        assert!(path_allowed(&["providers", "local", "model"]));
    }

    #[test]
    fn host_only_tables_are_not_allowed() {
        assert!(!path_allowed(&["tools", "tavily_api_key"]));
        assert!(!path_allowed(&["tools", "mcp_servers"]));
        assert!(!path_allowed(&["matrix", "access_token"]));
        assert!(!path_allowed(&["serve", "port"]));
        assert!(!path_allowed(&["workspace_dir"]));
        assert!(!path_allowed(&["stt_provider", "whisper", "model_dir"]));
    }

    #[test]
    fn a_table_shorter_than_its_entry_is_not_a_leaf_and_is_not_allowed() {
        // Only leaves are ever tested, so this case cannot arise in practice;
        // pinning it keeps the matcher total.
        assert!(!path_allowed(&["room_profile", "work"]));
    }
}
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `cargo test --bin sapphire-agent config_layer` (timeout 600000)
Expected: FAIL to compile — `cannot find function path_allowed in this scope`.

- [ ] **Step 4: Implement the allowlist and matcher**

Add above the test module in `src/config_layer.rs`:

```rust
/// TOML key paths the workspace-level config layer is permitted to set.
///
/// An entry authorises that path **and everything beneath it**. A key that must
/// stay host-only therefore may not sit under any entry, which is why the
/// `providers` and `room_profile` entries name their permitted leaves one by one
/// instead of using a bare wildcard — `providers.*` would drag `api_key` in with
/// it.
///
/// `*` matches exactly one path segment.
///
/// These are **TOML** names, which are not always the Rust field names: serde
/// renames `room_profiles` to `room_profile`, `memory_namespaces` to
/// `memory_namespace` and `voice_pipelines` to `voice_pipeline`. The fixture test
/// in this module exists to catch a path written in the wrong namespace.
pub const WORKSPACE_ALLOWLIST: &[&[&str]] = &[
    // The agent's identity and model choice.
    &["anthropic", "model"],
    &["anthropic", "light_model"],
    &["anthropic", "max_tokens"],
    &["anthropic", "system_prompt"],
    // Behaviour.
    &["compression"],
    &["day_boundary_hour"],
    &["session_policy"],
    &["daily_log_enabled"],
    &["memory_compaction_enabled"],
    &["heartbeat_enabled"],
    &["intraday_idle_minutes"],
    &["sync_interval_minutes"],
    &["digest"],
    &["timer"],
    // Routing and namespaces.
    &["profiles"],
    &["memory_namespace"],
    &["room_profile", "*", "profile"],
    &["room_profile", "*", "memory_namespace"],
    &["room_profile", "*", "rooms"],
    &["room_profile", "*", "session_policy"],
    &["room_profile", "*", "voice_pipeline"],
    // Provider definitions. `api_key` is deliberately absent.
    &["providers", "*", "type"],
    &["providers", "*", "base_url"],
    &["providers", "*", "model"],
    &["providers", "*", "provider_name"],
    &["providers", "*", "max_tokens"],
    // Which named STT/TTS provider a pipeline uses. The providers themselves are
    // host-only because they name model files on disk (see issue #173).
    &["voice_pipeline"],
];

/// True when the workspace layer may set `path`.
///
/// `path` is a TOML key path split into segments. A path is authorised when some
/// allowlist entry is a segment-wise prefix of it.
pub fn path_allowed(path: &[&str]) -> bool {
    WORKSPACE_ALLOWLIST.iter().any(|entry| {
        entry.len() <= path.len()
            && entry
                .iter()
                .zip(path.iter())
                .all(|(entry_seg, path_seg)| *entry_seg == "*" || entry_seg == path_seg)
    })
}
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `cargo test --bin sapphire-agent config_layer` (timeout 600000)
Expected: PASS, 8 tests.

Run: `cargo check --all-targets` (timeout 600000)
Expected: clean — no errors and no warnings.

- [ ] **Step 6: Commit**

```bash
git add src/config_layer.rs src/main.rs
git commit -m "feat(config): add the workspace-layer allowlist and its path matcher"
```

---

### Task 2: Filter the workspace layer through the allowlist

**Files:**
- Modify: `src/config_layer.rs`

**Interfaces:**
- Consumes: `path_allowed` from Task 1.
- Produces: `pub fn filter_allowed(workspace: toml::Value) -> (toml::Value, Vec<String>)` — the filtered document, and the sorted dot-joined paths that were dropped. Task 4 calls it.

- [ ] **Step 1: Write the failing tests**

Add to the `tests` module in `src/config_layer.rs`:

```rust
    fn parse(s: &str) -> toml::Value {
        s.parse::<toml::Value>().expect("fixture parses")
    }

    #[test]
    fn allowed_leaves_survive_filtering() {
        let (kept, rejected) = filter_allowed(parse(
            r#"
day_boundary_hour = 4

[anthropic]
system_prompt = "you are sapphire"
"#,
        ));
        assert!(rejected.is_empty());
        assert_eq!(kept["day_boundary_hour"].as_integer(), Some(4));
        assert_eq!(
            kept["anthropic"]["system_prompt"].as_str(),
            Some("you are sapphire")
        );
    }

    #[test]
    fn non_allowlisted_leaves_are_dropped_and_reported() {
        let (kept, rejected) = filter_allowed(parse(
            r#"
[anthropic]
model = "claude-opus-5"
api_key = "sk-should-not-travel"

[tools]
tavily_api_key = "also-not"
"#,
        ));
        assert_eq!(
            rejected,
            vec![
                "anthropic.api_key".to_string(),
                "tools.tavily_api_key".to_string()
            ]
        );
        assert_eq!(kept["anthropic"]["model"].as_str(), Some("claude-opus-5"));
        assert!(kept["anthropic"].get("api_key").is_none());
        // `tools` had nothing allowed in it, so the empty table is pruned too.
        assert!(kept.get("tools").is_none());
    }

    #[test]
    fn filtering_is_per_leaf_inside_a_wildcard_table() {
        let (kept, rejected) = filter_allowed(parse(
            r#"
[room_profile.work]
profile = "default"
rooms = ["!a:example.org"]
api_keys = ["sa-secret"]
"#,
        ));
        assert_eq!(rejected, vec!["room_profile.work.api_keys".to_string()]);
        assert_eq!(kept["room_profile"]["work"]["profile"].as_str(), Some("default"));
        assert!(kept["room_profile"]["work"].get("api_keys").is_none());
    }

    #[test]
    fn an_array_is_authorised_as_a_unit() {
        // Arrays are replaced wholesale rather than merged, so they are a leaf.
        let (kept, rejected) = filter_allowed(parse(
            r#"
[[timer.preset]]
name = "pomodoro"

[[tools.mcp_servers]]
name = "evil"
type = "stdio"
command = "rm"
"#,
        ));
        assert_eq!(rejected, vec!["tools.mcp_servers".to_string()]);
        assert!(kept["timer"]["preset"].is_array());
    }

    #[test]
    fn rejected_paths_are_sorted() {
        let (_, rejected) = filter_allowed(parse(
            r#"
workspace_dir = "/tmp/ws"

[serve]
port = 9000

[matrix]
access_token = "t"
"#,
        ));
        assert_eq!(
            rejected,
            vec![
                "matrix.access_token".to_string(),
                "serve.port".to_string(),
                "workspace_dir".to_string()
            ]
        );
    }
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test --bin sapphire-agent config_layer` (timeout 600000)
Expected: FAIL to compile — `cannot find function filter_allowed in this scope`.

- [ ] **Step 3: Implement the filter**

Add to `src/config_layer.rs`, after `path_allowed`:

```rust
/// Drop every leaf of `workspace` the allowlist does not authorise.
///
/// Returns the filtered document and the sorted dot-joined paths that were
/// dropped, so the caller can name them all in one warning.
///
/// A "leaf" is any value that is not a table — arrays included, because arrays
/// are replaced wholesale rather than merged, so they are authorised or rejected
/// as a unit. Tables left empty by filtering are pruned; an empty table would
/// contribute nothing to the merge anyway.
pub fn filter_allowed(workspace: toml::Value) -> (toml::Value, Vec<String>) {
    let mut rejected = Vec::new();
    let mut trail = Vec::new();
    let kept = filter_inner(workspace, &mut trail, &mut rejected)
        .unwrap_or_else(|| toml::Value::Table(toml::map::Map::new()));
    rejected.sort();
    (kept, rejected)
}

/// Returns `None` when the value was dropped entirely.
fn filter_inner(
    value: toml::Value,
    trail: &mut Vec<String>,
    rejected: &mut Vec<String>,
) -> Option<toml::Value> {
    match value {
        toml::Value::Table(table) => {
            let mut kept = toml::map::Map::new();
            for (key, child) in table {
                trail.push(key.clone());
                if let Some(child) = filter_inner(child, trail, rejected) {
                    kept.insert(key, child);
                }
                trail.pop();
            }
            if kept.is_empty() {
                None
            } else {
                Some(toml::Value::Table(kept))
            }
        }
        leaf => {
            let segments: Vec<&str> = trail.iter().map(String::as_str).collect();
            if path_allowed(&segments) {
                Some(leaf)
            } else {
                rejected.push(trail.join("."));
                None
            }
        }
    }
}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cargo test --bin sapphire-agent config_layer` (timeout 600000)
Expected: PASS, 13 tests.

- [ ] **Step 5: Commit**

```bash
git add src/config_layer.rs
git commit -m "feat(config): filter the workspace layer through the allowlist"
```

---

### Task 3: Deep-merge the two layers

**Files:**
- Modify: `src/config_layer.rs`

**Interfaces:**
- Produces: `pub fn deep_merge(base: toml::Value, over: toml::Value) -> toml::Value`. Task 4 calls it.

- [ ] **Step 1: Write the failing tests**

Add to the `tests` module:

```rust
    #[test]
    fn host_wins_on_a_scalar() {
        let merged = deep_merge(
            parse("day_boundary_hour = 4"),
            parse("day_boundary_hour = 6"),
        );
        assert_eq!(merged["day_boundary_hour"].as_integer(), Some(6));
    }

    #[test]
    fn workspace_supplies_what_the_host_omits() {
        let merged = deep_merge(
            parse("day_boundary_hour = 4\nsession_policy = \"compact\""),
            parse("day_boundary_hour = 6"),
        );
        assert_eq!(merged["day_boundary_hour"].as_integer(), Some(6));
        assert_eq!(merged["session_policy"].as_str(), Some("compact"));
    }

    #[test]
    fn tables_merge_key_by_key() {
        // The case that forces deep merge: the workspace supplies the shared
        // fields of a room profile, the host adds its bearer token, and neither
        // has to restate the other.
        let merged = deep_merge(
            parse(
                r#"
[room_profile.work]
profile = "default"
rooms = ["!a:example.org"]
"#,
            ),
            parse(
                r#"
[room_profile.work]
api_keys = ["sa-host-only"]
"#,
            ),
        );
        assert_eq!(merged["room_profile"]["work"]["profile"].as_str(), Some("default"));
        assert_eq!(
            merged["room_profile"]["work"]["api_keys"][0].as_str(),
            Some("sa-host-only")
        );
    }

    #[test]
    fn arrays_are_replaced_not_concatenated() {
        // Concatenation cannot express removal, so the host replaces.
        let merged = deep_merge(
            parse(r#"[room_profile.work]
rooms = ["!a:example.org", "!b:example.org"]"#),
            parse(r#"[room_profile.work]
rooms = ["!c:example.org"]"#),
        );
        let rooms = merged["room_profile"]["work"]["rooms"].as_array().unwrap();
        assert_eq!(rooms.len(), 1);
        assert_eq!(rooms[0].as_str(), Some("!c:example.org"));
    }

    #[test]
    fn a_host_scalar_replaces_a_workspace_table() {
        let merged = deep_merge(parse("[digest]\nkeep = 3"), parse("digest = 7"));
        assert_eq!(merged["digest"].as_integer(), Some(7));
    }
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test --bin sapphire-agent config_layer` (timeout 600000)
Expected: FAIL to compile — `cannot find function deep_merge in this scope`.

- [ ] **Step 3: Implement the merge**

Add to `src/config_layer.rs`:

```rust
/// Merge `over` onto `base`, with `over` winning.
///
/// Tables merge key by key, recursively. Every other value — scalars and arrays
/// alike — is replaced wholesale by `over`. Arrays are deliberately not
/// concatenated: concatenation cannot express removing an entry, and it silently
/// duplicates values both layers list.
pub fn deep_merge(base: toml::Value, over: toml::Value) -> toml::Value {
    match (base, over) {
        (toml::Value::Table(mut base_table), toml::Value::Table(over_table)) => {
            for (key, over_child) in over_table {
                let merged = match base_table.remove(&key) {
                    Some(base_child) => deep_merge(base_child, over_child),
                    None => over_child,
                };
                base_table.insert(key, merged);
            }
            toml::Value::Table(base_table)
        }
        (_, over) => over,
    }
}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cargo test --bin sapphire-agent config_layer` (timeout 600000)
Expected: PASS, 18 tests.

- [ ] **Step 5: Commit**

```bash
git add src/config_layer.rs
git commit -m "feat(config): deep-merge the config layers with the host on top"
```

---

### Task 4: Provenance, the composed entry point, and the fixture guard

**Files:**
- Modify: `src/config_layer.rs`

**Interfaces:**
- Consumes: `filter_allowed`, `deep_merge`, `path_allowed`, `WORKSPACE_ALLOWLIST`.
- Produces: `pub enum Layer { Workspace, Host }` (derives `Debug, Clone, Copy, PartialEq, Eq`); `pub struct MergeOutcome { pub merged: toml::Value, pub rejected: Vec<String>, pub provenance: BTreeMap<String, Layer> }`; `pub fn merge_layers(workspace: toml::Value, host: toml::Value) -> MergeOutcome`; `pub fn provenance_of(workspace: &toml::Value, host: &toml::Value) -> BTreeMap<String, Layer>`. Task 5 calls `merge_layers`; Task 6 reads `rejected` and `provenance` and matches on `Layer`.

- [ ] **Step 1: Write the failing tests**

Add to the `tests` module. The last test is the guard the spec calls for — one hand-written fixture, three assertions that between them prove the allowlist names real fields, covers everything the fixture uses, and has no dead entries.

```rust
    #[test]
    fn provenance_names_the_winning_layer() {
        let outcome = merge_layers(
            parse("day_boundary_hour = 4\nsession_policy = \"compact\""),
            parse("day_boundary_hour = 6"),
        );
        assert_eq!(
            outcome.provenance.get("day_boundary_hour"),
            Some(&Layer::Host)
        );
        assert_eq!(
            outcome.provenance.get("session_policy"),
            Some(&Layer::Workspace)
        );
    }

    #[test]
    fn a_setting_in_neither_layer_has_no_provenance_entry() {
        let outcome = merge_layers(parse(""), parse("day_boundary_hour = 6"));
        assert!(outcome.provenance.get("heartbeat_enabled").is_none());
    }

    #[test]
    fn a_rejected_key_is_neither_merged_nor_attributed() {
        let outcome = merge_layers(
            parse("[anthropic]\napi_key = \"sk-should-not-travel\""),
            parse("[anthropic]\napi_key = \"sk-host\""),
        );
        assert_eq!(outcome.rejected, vec!["anthropic.api_key".to_string()]);
        assert_eq!(outcome.merged["anthropic"]["api_key"].as_str(), Some("sk-host"));
        assert_eq!(
            outcome.provenance.get("anthropic.api_key"),
            Some(&Layer::Host)
        );
    }

    /// A realistic workspace-level config exercising every allowlist entry.
    ///
    /// Written by hand so the values carry the types the real `Config` expects.
    const FIXTURE: &str = r#"
day_boundary_hour = 4
session_policy = "compact"
daily_log_enabled = true
memory_compaction_enabled = true
heartbeat_enabled = true
intraday_idle_minutes = 45
sync_interval_minutes = 15

[anthropic]
model = "claude-opus-5"
light_model = "claude-haiku-4-5-20251001"
max_tokens = 8192
system_prompt = "you are sapphire"

[compression]
enabled = true

[digest]
daily_items = 7

# `cycles` belongs to a preset, not to `[timer]` itself, and `steps` is a
# required field — this block has to deserialize into `Config` for the
# round-trip test below.
[[timer.preset]]
name = "pomodoro"
cycles = 4

[[timer.preset.steps]]
label = "Focus"
minutes = 25.0

[profiles.default]
provider = "anthropic"
fallback_provider = "local"

[memory_namespace.work]
include = ["default"]
background_profile = "default"

[room_profile.work]
profile = "default"
memory_namespace = "work"
rooms = ["!a:example.org"]
session_policy = "compact"
voice_pipeline = "desk"

[providers.local]
type = "openai_compatible"
base_url = "http://llm.lan:8080/v1"
model = "qwen"
provider_name = "local"
max_tokens = 4096

[voice_pipeline.desk]
stt_provider = "whisper"
tts_provider = "piper"
"#;

    #[test]
    fn the_fixture_is_entirely_allowlisted() {
        let (_, rejected) = filter_allowed(parse(FIXTURE));
        assert!(rejected.is_empty(), "fixture has non-allowlisted keys: {rejected:?}");
    }

    #[test]
    fn every_allowlist_entry_is_exercised_by_the_fixture() {
        let mut paths = Vec::new();
        leaf_paths(&parse(FIXTURE), &mut Vec::new(), &mut paths);
        for entry in WORKSPACE_ALLOWLIST {
            let hit = paths.iter().any(|p| {
                let segments: Vec<&str> = p.split('.').collect();
                entry.len() <= segments.len()
                    && entry
                        .iter()
                        .zip(segments.iter())
                        .all(|(e, s)| *e == "*" || e == s)
            });
            assert!(hit, "no fixture key exercises allowlist entry {entry:?}");
        }
    }

    #[test]
    fn every_fixture_key_reaches_a_real_config_field() {
        // The allowlist is string paths, and `Config` does not deny unknown
        // fields, so a path naming a renamed or deleted field would otherwise be
        // ignored in silence. Round-tripping through the type catches it: serde
        // drops an unknown key on the way in, so it is missing on the way out.
        let host = parse(r#"
[anthropic]
api_key = "sk-test"
"#);
        let outcome = merge_layers(parse(FIXTURE), host);
        let config: crate::config::Config =
            outcome.merged.try_into().expect("merged fixture deserializes");
        let round_tripped = toml::Value::try_from(&config).expect("Config re-serializes");

        let mut paths = Vec::new();
        leaf_paths(&parse(FIXTURE), &mut Vec::new(), &mut paths);
        for path in paths {
            let mut cursor = &round_tripped;
            for segment in path.split('.') {
                cursor = cursor
                    .get(segment)
                    .unwrap_or_else(|| panic!("`{path}` did not survive the round trip through Config"));
            }
        }
    }
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test --bin sapphire-agent config_layer` (timeout 600000)
Expected: FAIL to compile — `cannot find function merge_layers` and `cannot find function leaf_paths`.

- [ ] **Step 3: Implement provenance and the composed entry point**

Add to `src/config_layer.rs`. Put `use std::collections::BTreeMap;` at the top of the file:

```rust
/// Which layer supplied the effective value for a setting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Layer {
    /// The workspace-level config, shared by every host using this workspace.
    Workspace,
    /// The host-local config.
    Host,
}

/// The result of layering the workspace config under the host config.
pub struct MergeOutcome {
    /// The merged document, ready to deserialize into `Config`.
    pub merged: toml::Value,
    /// Dot-joined paths dropped from the workspace layer because the allowlist
    /// does not authorise them. Sorted.
    pub rejected: Vec<String>,
    /// Which layer supplied each leaf. See [`provenance_of`].
    pub provenance: BTreeMap<String, Layer>,
}

/// Filter the workspace layer, merge the host over it, and record where each
/// value came from.
pub fn merge_layers(workspace: toml::Value, host: toml::Value) -> MergeOutcome {
    let (workspace, rejected) = filter_allowed(workspace);
    let provenance = provenance_of(&workspace, &host);
    let merged = deep_merge(workspace, host);
    MergeOutcome {
        merged,
        rejected,
        provenance,
    }
}

/// Which layer supplied each leaf, keyed by dot-joined TOML path.
///
/// Only leaves present in one of the two layers appear. A setting absent from
/// both takes its serde default, and callers report that by finding no entry.
///
/// `workspace` is expected to be the **filtered** document, so a rejected key
/// is never attributed to the workspace layer.
pub fn provenance_of(workspace: &toml::Value, host: &toml::Value) -> BTreeMap<String, Layer> {
    let mut workspace_paths = Vec::new();
    leaf_paths(workspace, &mut Vec::new(), &mut workspace_paths);
    let mut host_paths = Vec::new();
    leaf_paths(host, &mut Vec::new(), &mut host_paths);

    let mut out: BTreeMap<String, Layer> = workspace_paths
        .into_iter()
        .map(|path| (path, Layer::Workspace))
        .collect();
    // The host wins in `deep_merge`, so it wins here too.
    for path in host_paths {
        out.insert(path, Layer::Host);
    }
    out
}

/// Collect the dot-joined path of every leaf (non-table) value.
fn leaf_paths(value: &toml::Value, trail: &mut Vec<String>, out: &mut Vec<String>) {
    match value {
        toml::Value::Table(table) => {
            for (key, child) in table {
                trail.push(key.clone());
                leaf_paths(child, trail, out);
                trail.pop();
            }
        }
        _ => out.push(trail.join(".")),
    }
}
```

Note for the implementer: `leaf_paths` is private but the tests call it through `use super::*`, which works because the test module is a child module.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cargo test --bin sapphire-agent config_layer` (timeout 600000)
Expected: PASS, 24 tests.

If `every_fixture_key_reaches_a_real_config_field` fails, the fixture names a field that does not exist under that TOML path — check the `#[serde(rename = ...)]` attributes in `src/config.rs` before assuming the field is gone. If `the_fixture_is_entirely_allowlisted` fails, the fixture and the allowlist disagree; fix whichever is wrong.

- [ ] **Step 5: Commit**

```bash
git add src/config_layer.rs
git commit -m "feat(config): compose the layer merge and guard the allowlist with a fixture"
```

---

### Task 5: Load the two layers

**Files:**
- Modify: `src/config.rs` (the `impl Config` block containing `load` and `resolved_workspace_dir`, around lines 905-935)
- Modify: `src/main.rs:101-103`

**Interfaces:**
- Consumes: `merge_layers` and `MergeOutcome` from Task 4.
- Produces: `pub fn resolve_workspace_dir(explicit: Option<&str>, config_path: &Path) -> PathBuf` and `pub fn workspace_config_path(workspace_dir: &Path) -> PathBuf` in `src/config.rs`; `pub struct LoadedConfig { pub config: Config, pub rejected: Vec<String>, pub provenance: BTreeMap<String, Layer> }` and `pub fn Config::load_layered(host_path: &Path) -> Result<LoadedConfig>`. Task 6 reads `rejected` and `provenance` from `LoadedConfig`.

- [ ] **Step 1: Write the failing tests**

Add to the `tests` module at the bottom of `src/config.rs`:

```rust
    #[test]
    fn load_layered_without_a_workspace_config_matches_plain_load() {
        let dir = tempfile::tempdir().unwrap();
        let host_path = dir.path().join("config.toml");
        std::fs::write(
            &host_path,
            "[anthropic]\napi_key = \"sk-test\"\nday_boundary_hour = 5\n",
        )
        .unwrap();

        let loaded = Config::load_layered(&host_path).unwrap();
        assert_eq!(loaded.config.day_boundary_hour, 5);
        assert!(loaded.rejected.is_empty());
        // With no workspace file every value can only have come from the host.
        assert!(
            loaded
                .provenance
                .values()
                .all(|l| *l == crate::config_layer::Layer::Host)
        );
    }

    #[test]
    fn load_layered_merges_the_workspace_config() {
        let dir = tempfile::tempdir().unwrap();
        let host_path = dir.path().join("config.toml");
        std::fs::write(
            &host_path,
            "[anthropic]\napi_key = \"sk-host\"\nday_boundary_hour = 6\n",
        )
        .unwrap();
        let marker = dir.path().join(".sapphire-agent");
        std::fs::create_dir_all(&marker).unwrap();
        std::fs::write(
            marker.join("config.toml"),
            "day_boundary_hour = 4\n\n[anthropic]\nsystem_prompt = \"shared\"\napi_key = \"sk-should-not-travel\"\n",
        )
        .unwrap();

        let loaded = Config::load_layered(&host_path).unwrap();
        // Host wins where both set a key.
        assert_eq!(loaded.config.day_boundary_hour, 6);
        // Workspace supplies what the host omits.
        assert_eq!(loaded.config.anthropic.system_prompt.as_deref(), Some("shared"));
        // The host's secret is untouched and the workspace's is refused.
        assert_eq!(loaded.config.anthropic.api_key.as_deref(), Some("sk-host"));
        assert_eq!(loaded.rejected, vec!["anthropic.api_key".to_string()]);
        assert_eq!(
            loaded.provenance.get("anthropic.system_prompt"),
            Some(&crate::config_layer::Layer::Workspace)
        );
    }

    #[test]
    fn workspace_dir_from_the_host_layer_locates_the_workspace_config() {
        let dir = tempfile::tempdir().unwrap();
        let ws = dir.path().join("elsewhere");
        std::fs::create_dir_all(ws.join(".sapphire-agent")).unwrap();
        std::fs::write(
            ws.join(".sapphire-agent").join("config.toml"),
            "session_policy = \"none\"\n",
        )
        .unwrap();

        let host_path = dir.path().join("config.toml");
        std::fs::write(
            &host_path,
            format!(
                "workspace_dir = \"{}\"\n\n[anthropic]\napi_key = \"sk-test\"\n",
                ws.display().to_string().replace('\\', "\\\\")
            ),
        )
        .unwrap();

        let loaded = Config::load_layered(&host_path).unwrap();
        assert_eq!(loaded.config.session_policy, SessionPolicy::None);
    }
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cargo test --bin sapphire-agent config::tests` (timeout 600000)
Expected: FAIL to compile — `no function or associated item named load_layered`.

- [ ] **Step 3: Extract the workspace-dir resolution into a free function**

In `src/config.rs`, add these two free functions near `Config::default_path`, and rewrite `resolved_workspace_dir` to delegate. The extraction is what lets `load_layered` resolve the directory from a raw `toml::Value` without first deserializing a possibly-incomplete host layer:

```rust
/// Resolve the workspace directory from an explicit setting, falling back to the
/// config file's own directory.
///
/// Free-standing because the layered loader needs it before a `Config` exists:
/// the workspace directory has to be known to find the workspace config, so it
/// can only ever come from the host layer.
pub fn resolve_workspace_dir(explicit: Option<&str>, config_path: &Path) -> PathBuf {
    match explicit {
        Some(dir) => PathBuf::from(shellexpand::tilde(dir).as_ref()),
        None => config_path
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .to_path_buf(),
    }
}

/// Path of the workspace-level config: `{workspace_dir}/.sapphire-agent/config.toml`.
///
/// This mirrors the framework's `Workspace::config_path()` convention. It is not
/// called through the framework because reaching a `Workspace` value goes via
/// `from_root`, which fails when the marker directory is absent — and the agent
/// resolves its workspace through the marker-free path.
pub fn workspace_config_path(workspace_dir: &Path) -> PathBuf {
    workspace_dir.join(".sapphire-agent").join("config.toml")
}
```

Then replace the body of `Config::resolved_workspace_dir` with:

```rust
    pub fn resolved_workspace_dir(&self, config_path: &Path) -> PathBuf {
        resolve_workspace_dir(self.workspace_dir.as_deref(), config_path)
    }
```

- [ ] **Step 4: Implement `load_layered`**

Add to `src/config.rs`. Put `use crate::config_layer::{self, Layer};` and `use std::collections::BTreeMap;` at the top of the file if they are not already there:

```rust
/// A `Config` plus what the layering did to produce it.
pub struct LoadedConfig {
    pub config: Config,
    /// Workspace-layer keys refused by the allowlist. Reported at startup.
    pub rejected: Vec<String>,
    /// Which layer supplied each setting, for `verify`.
    pub provenance: BTreeMap<String, Layer>,
}

impl Config {
    /// Load the host config, then layer the workspace config under it.
    ///
    /// The workspace layer is opt-in by existence: with no
    /// `{workspace_dir}/.sapphire-agent/config.toml` this behaves exactly like
    /// [`Config::load`].
    ///
    /// A malformed **host** config is an error, as it always was. A malformed
    /// **workspace** config is a warning: from the point the workspace syncs from
    /// a server it is remote input, and one bad file must not stop every host.
    pub fn load_layered(host_path: &Path) -> Result<LoadedConfig> {
        let host_text = std::fs::read_to_string(host_path)
            .with_context(|| format!("Failed to read config file: {}", host_path.display()))?;
        let host: toml::Value = toml::from_str(&host_text)
            .with_context(|| format!("Failed to parse config file: {}", host_path.display()))?;

        let workspace_dir = resolve_workspace_dir(
            host.get("workspace_dir").and_then(toml::Value::as_str),
            host_path,
        );
        let ws_path = workspace_config_path(&workspace_dir);

        let workspace = match std::fs::read_to_string(&ws_path) {
            Ok(text) => match toml::from_str::<toml::Value>(&text) {
                Ok(value) => value,
                Err(e) => {
                    tracing::warn!(
                        "Ignoring malformed workspace config at {}: {e}",
                        ws_path.display()
                    );
                    toml::Value::Table(toml::map::Map::new())
                }
            },
            Err(_) => toml::Value::Table(toml::map::Map::new()),
        };

        let outcome = config_layer::merge_layers(workspace, host);
        let config: Config = outcome
            .merged
            .try_into()
            .with_context(|| "Failed to parse config file")?;

        Ok(LoadedConfig {
            config,
            rejected: outcome.rejected,
            provenance: outcome.provenance,
        })
    }
}
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `cargo test --bin sapphire-agent config` (timeout 600000)
Expected: PASS. All previously passing `config::tests` must still pass — `resolved_workspace_dir` changed shape but not behaviour.

- [ ] **Step 6: Call it from `main`**

In `src/main.rs`, replace lines 101-103:

```rust
    let config_path = cli.config.unwrap_or_else(Config::default_path);
    let config = Config::load(&config_path)
        .with_context(|| format!("Failed to load config from {}", config_path.display()))?;
```

with a destructuring bind, so the two diagnostic fields are available without a second move:

```rust
    let config_path = cli.config.unwrap_or_else(Config::default_path);
    let LoadedConfig {
        config,
        rejected,
        provenance: _provenance,
    } = Config::load_layered(&config_path)
        .with_context(|| format!("Failed to load config from {}", config_path.display()))?;
```

`_provenance` is underscore-prefixed only so this task compiles warning-free; Task 6 renames it
when it starts using it. Add `use config::LoadedConfig;` alongside the existing
`use config::Config;`. Leave the `standby_mode` guard that follows exactly as it is.

- [ ] **Step 7: Warn about refused keys**

Immediately after that binding, still in `src/main.rs`:

```rust
    if !rejected.is_empty() {
        tracing::warn!(
            "Ignoring {} key(s) in the workspace config that the workspace layer may not set: {}. \
             Credentials, MCP servers, bind addresses and machine paths are host-local by design; \
             set them in {} instead.",
            rejected.len(),
            rejected.join(", "),
            config_path.display()
        );
    }
```

- [ ] **Step 8: Verify the whole binary still builds and passes**

Run: `cargo check --all-targets` (timeout 600000)
Expected: clean — no errors and no warnings.

Run: `cargo test --bin sapphire-agent` (timeout 600000)
Expected: all tests pass.

- [ ] **Step 9: Commit**

```bash
git add src/config.rs src/main.rs
git commit -m "feat(config): load the workspace config layer under the host config"
```

---

### Task 6: Report what the layering did

**Files:**
- Modify: `src/main.rs` (after the config load at ~line 101, and inside the `Command::Verify` arm at ~line 119)
- Modify: `config.example.toml`

**Interfaces:**
- Consumes: `LoadedConfig { rejected, provenance }` and `Layer` from Task 5.

- [ ] **Step 1: Start using the provenance binding**

In `src/main.rs`, in the destructuring bind added by Task 5, rename the placeholder:

```rust
    let LoadedConfig {
        config,
        rejected,
        provenance,
    } = Config::load_layered(&config_path)
```

- [ ] **Step 2: Report provenance from `verify`**

Still in `src/main.rs`, inside the `Some(Command::Verify)` arm. After the existing `println!("Config OK");` line, add a line naming the workspace config, then annotate the settings the arm already prints.

Add a small helper above `async fn main`:

```rust
/// One-word tag naming where a setting's effective value came from, for `verify`.
fn layer_tag(provenance: &std::collections::BTreeMap<String, config_layer::Layer>, path: &str) -> &'static str {
    match provenance.get(path) {
        Some(config_layer::Layer::Workspace) => "workspace",
        Some(config_layer::Layer::Host) => "host",
        None => "default",
    }
}
```

Then in the `Verify` arm, after `println!("Config OK");`:

```rust
            let ws_config = config::workspace_config_path(&workspace_dir);
            if ws_config.is_file() {
                println!("  Workspace config  : {}", ws_config.display());
            } else {
                println!("  Workspace config  : none ({} absent)", ws_config.display());
            }
```

and change these three existing lines to carry their provenance tag:

```rust
            println!(
                "  Anthropic model   : {} [{}]",
                config.anthropic.model,
                layer_tag(&provenance, "anthropic.model")
            );
            println!(
                "  Anthropic max_tok : {} [{}]",
                config.anthropic.max_tokens,
                layer_tag(&provenance, "anthropic.max_tokens")
            );
            println!(
                "  Day boundary hour : {}:00 local [{}]",
                config.day_boundary_hour,
                layer_tag(&provenance, "day_boundary_hour")
            );
            println!(
                "  Heartbeat enabled : {} [{}]",
                config.heartbeat_enabled,
                layer_tag(&provenance, "heartbeat_enabled")
            );
```

- [ ] **Step 3: Verify it compiles and runs**

Run: `cargo check --all-targets` (timeout 600000)
Expected: clean, no warnings.

Run: `cargo test --bin sapphire-agent` (timeout 600000)
Expected: all tests pass.

- [ ] **Step 4: Exercise it by hand**

```bash
mkdir -p /tmp/wslayer/.sapphire-agent
printf '[anthropic]\napi_key = "sk-test"\nworkspace_dir = "/tmp/wslayer"\n' > /tmp/wslayer-host.toml
printf '[anthropic]\nsystem_prompt = "shared prompt"\n\n[tools]\ntavily_api_key = "should-be-refused"\n' > /tmp/wslayer/.sapphire-agent/config.toml
cargo run -- --config /tmp/wslayer-host.toml verify
```

Expected: the run warns that `tools.tavily_api_key` was ignored, prints the workspace config path, and tags at least one setting `[workspace]` or `[host]`.

- [ ] **Step 5: Document the layer**

In `config.example.toml`, add a block near the top, after the existing `workspace_dir` comment:

```toml
# ── Layered configuration ────────────────────────────────────────────
# Settings that describe the agent — its system prompt, profiles, memory
# namespaces, providers, digest and heartbeat behaviour — can live in a
# shared file inside the workspace instead of here:
#
#   {workspace_dir}/.sapphire-agent/config.toml
#
# That file is optional. When present its values are used unless this
# host-local file also sets them, in which case this file wins.
#
# It may only set settings that belong to the agent. Credentials
# ([anthropic].api_key, [matrix], [discord], [tools], per-profile
# api_keys), bind addresses, cache and workspace paths, and STT/TTS model
# paths are host-local: if the workspace file sets one, it is ignored and
# named in a warning at startup.
#
# `sapphire-agent verify` prints where each setting's effective value came
# from.
```

- [ ] **Step 6: Commit**

```bash
git add src/main.rs config.example.toml
git commit -m "feat(config): warn about refused workspace keys and report provenance from verify"
```
