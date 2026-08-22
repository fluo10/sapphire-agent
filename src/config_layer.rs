//! Layering the workspace-level config under the host-local one.
//!
//! Everything here is a pure function over [`toml::Value`]: the allowlist that
//! bounds what the workspace layer may set, the merge that combines the two
//! layers, and the provenance calculation that lets `verify` say where a value
//! came from. Nothing in this module touches the filesystem — `Config::load_layered`
//! reads the files and calls in.

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
#[allow(dead_code)]
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
#[allow(dead_code)]
pub fn path_allowed(path: &[&str]) -> bool {
    WORKSPACE_ALLOWLIST.iter().any(|entry| {
        entry.len() <= path.len()
            && entry
                .iter()
                .zip(path.iter())
                .all(|(entry_seg, path_seg)| *entry_seg == "*" || entry_seg == path_seg)
    })
}

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
