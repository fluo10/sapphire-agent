//! The rules every session store follows at the storage boundary.
//!
//! Two stores write conversations to disk — `SessionStore` (four kinds)
//! and `AcpSessionStore` — with different line formats and the same
//! constraints. What is shared here is not the format; it is the set of
//! facts about the workspace and the Anthropic API that neither store is
//! free to decide for itself:
//!
//! - a lost tool result gets one specific sentence, not each store's own
//! - a tool input has nowhere to go but the (indexed) session file, so
//!   an oversized one is elided rather than written
//! - `tool_use` and `tool_result` must be adjacent, whatever gaps a
//!   crash or a partial sync left behind

use serde_json::Value;

/// What the model is told when a tool result is no longer in the cache.
///
/// The pairing between `tool_use` and `tool_result` is what the API
/// validates, not the content — so a placeholder keeps the history
/// valid and the conversation's shape intact. A session that loads
/// thinner is worth having; one that fails to load is not.
pub const MISSING_RESULT: &str =
    "[this tool result is no longer stored; call the tool again if you need it]";

/// Storage-path-only transformation: never touches the in-memory value,
/// only what gets written to the JSONL.
///
/// Unlike a result, an input has nowhere to go but the session file
/// itself — there is no cache/hash indirection for it. That file lives
/// under `<workspace>/sessions`, which the retrieve indexer walks, so an
/// unbounded input (a multi-megabyte `file_write`, say) would put its
/// whole content into the index — exactly what the external tool-result
/// cache exists to keep out.
///
/// Elide rather than truncate: truncated JSON does not parse, and a
/// reload needs `input` to still be valid JSON of the same shape.
pub fn elide_oversized_input(input: &Value) -> Value {
    let size = serde_json::to_string(input).map(|s| s.len()).unwrap_or(0);
    if size <= crate::tools::OUTPUT_CAP_BYTES {
        return input.clone();
    }
    serde_json::json!({
        "_elided": format!("{size} bytes of tool input, too large to store")
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_small_input_passes_through_unchanged() {
        let input = serde_json::json!({ "path": "src/main.rs" });
        assert_eq!(elide_oversized_input(&input), input);
    }

    /// The elided form has to remain valid JSON of the same type, or a
    /// reload produces a `ToolUse` that will not deserialize.
    #[test]
    fn an_oversized_input_becomes_a_small_valid_object() {
        let big = serde_json::json!({ "content": "x".repeat(crate::tools::OUTPUT_CAP_BYTES + 1) });
        let elided = elide_oversized_input(&big);
        assert!(elided.is_object(), "must stay an object: {elided}");
        assert!(elided.get("_elided").is_some(), "missing marker: {elided}");
        assert!(serde_json::to_string(&elided).unwrap().len() < 200);
    }
}
