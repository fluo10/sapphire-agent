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

use crate::provider::{ChatMessage, ContentPart, Role};
use serde_json::Value;
use std::collections::HashSet;

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

/// Make a loaded conversation something the API will accept.
///
/// A `tool_use` can end up on disk without its `tool_result` — the
/// second append failed, the process died between the two, a sync landed
/// only half the pair. The gap is worse than a lost message, because the
/// API requires a `tool_result` to sit in the message *immediately
/// following* its `tool_use`, not merely present somewhere later.
///
/// The check is positional in both directions, and that is the point. A
/// set-based "does this id appear anywhere" check silently accepts
/// pairings the API rejects: two messages carrying the same `tool_use`
/// id where only one is answered, an id answered many messages before
/// the call that (re)issued it, a `tool_result` answering nothing
/// adjacent at all.
///
/// Synthesise rather than drop the `tool_use`: dropping would erase the
/// fact that the agent attempted the call, and `MISSING_RESULT` is
/// exactly the shape a cache miss already produces, so the model sees
/// nothing it does not already handle.
pub fn repair_tool_pairing(messages: Vec<ChatMessage>) -> Vec<ChatMessage> {
    // Pass 1 — drop every `tool_result` whose `tool_use` is not in the
    // message immediately before it. If that empties a message, drop the
    // message too: an empty message is its own API error.
    let mut kept: Vec<ChatMessage> = Vec::with_capacity(messages.len());
    for (idx, message) in messages.iter().enumerate() {
        let prev_uses: HashSet<&str> = idx
            .checked_sub(1)
            .and_then(|p| messages.get(p))
            .map(tool_use_ids)
            .unwrap_or_default();
        let parts: Vec<ContentPart> = message
            .parts
            .iter()
            .filter(|p| match p {
                ContentPart::ToolResult { tool_use_id, .. } => {
                    prev_uses.contains(tool_use_id.as_str())
                }
                _ => true,
            })
            .cloned()
            .collect();
        if parts.is_empty() && !message.parts.is_empty() {
            continue;
        }
        kept.push(ChatMessage {
            parts,
            ..message.clone()
        });
    }

    // Pass 2 — answer every `tool_use` the following message left open.
    let mut out: Vec<ChatMessage> = Vec::with_capacity(kept.len() + 1);
    let mut i = 0;
    while i < kept.len() {
        let message = kept[i].clone();
        let uses = ordered_tool_use_ids(&message);
        if uses.is_empty() {
            out.push(message);
            i += 1;
            continue;
        }
        let next = kept.get(i + 1);
        let answered: HashSet<&str> = next.map(tool_result_ids).unwrap_or_default();
        let missing: Vec<ContentPart> = uses
            .iter()
            .filter(|id| !answered.contains(id.as_str()))
            .map(|id| ContentPart::ToolResult {
                tool_use_id: id.clone(),
                content: MISSING_RESULT.to_string(),
            })
            .collect();
        out.push(message);
        if missing.is_empty() {
            i += 1;
            continue;
        }
        // #195. `answered` being non-empty means the next message is
        // already this one's `tool_result` message — pass 1 dropped any
        // result that answered something else. Merging into it is what
        // keeps the real results adjacent to their `tool_use`; splicing
        // a message in front would displace them by one and be rejected
        // for the very reason this function exists.
        match next {
            Some(next) if !answered.is_empty() => {
                let mut merged = next.clone();
                merged.parts.extend(missing);
                out.push(merged);
                i += 2;
            }
            _ => {
                out.push(ChatMessage {
                    role: Role::User,
                    parts: missing,
                    input_kind: None,
                    user_id: None,
                });
                i += 1;
            }
        }
    }
    out
}

fn tool_use_ids(msg: &ChatMessage) -> HashSet<&str> {
    msg.parts
        .iter()
        .filter_map(|p| match p {
            ContentPart::ToolUse { id, .. } => Some(id.as_str()),
            _ => None,
        })
        .collect()
}

fn tool_result_ids(msg: &ChatMessage) -> HashSet<&str> {
    msg.parts
        .iter()
        .filter_map(|p| match p {
            ContentPart::ToolResult { tool_use_id, .. } => Some(tool_use_id.as_str()),
            _ => None,
        })
        .collect()
}

/// Call order, deduplicated. Two `tool_use` parts that (wrongly) share
/// one id must not produce two placeholders for it.
fn ordered_tool_use_ids(msg: &ChatMessage) -> Vec<String> {
    let mut seen = HashSet::new();
    msg.parts
        .iter()
        .filter_map(|p| match p {
            ContentPart::ToolUse { id, .. } if seen.insert(id.clone()) => Some(id.clone()),
            _ => None,
        })
        .collect()
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

    fn tool_use(id: &str) -> ContentPart {
        ContentPart::ToolUse {
            id: id.to_string(),
            name: "file_read".to_string(),
            input: serde_json::json!({}),
        }
    }

    fn tool_result(id: &str, content: &str) -> ContentPart {
        ContentPart::ToolResult {
            tool_use_id: id.to_string(),
            content: content.to_string(),
        }
    }

    fn assistant(parts: Vec<ContentPart>) -> ChatMessage {
        ChatMessage {
            role: Role::Assistant,
            parts,
            input_kind: None,
            user_id: None,
        }
    }

    fn user(parts: Vec<ContentPart>) -> ChatMessage {
        ChatMessage {
            role: Role::User,
            parts,
            input_kind: None,
            user_id: None,
        }
    }

    fn result_ids(msg: &ChatMessage) -> Vec<&str> {
        msg.parts
            .iter()
            .filter_map(|p| match p {
                ContentPart::ToolResult { tool_use_id, .. } => Some(tool_use_id.as_str()),
                _ => None,
            })
            .collect()
    }

    /// A tool_use whose result never made it to disk gets a synthesised
    /// one immediately after it — the API requires the answer in the
    /// very next message, not merely somewhere later.
    #[test]
    fn an_unanswered_tool_use_gets_a_placeholder_right_after_it() {
        let repaired = repair_tool_pairing(vec![
            user(vec![ContentPart::Text("read it".to_string())]),
            assistant(vec![tool_use("c1")]),
        ]);
        assert_eq!(repaired.len(), 3);
        assert_eq!(result_ids(&repaired[2]), vec!["c1"]);
        assert!(
            matches!(&repaired[2].parts[0], ContentPart::ToolResult { content, .. } if content == MISSING_RESULT)
        );
    }

    /// A tool_result whose tool_use is not in the message right before it
    /// is not a valid pairing wherever else its id appears — drop it, and
    /// drop the message if that empties it.
    #[test]
    fn an_orphaned_tool_result_is_dropped() {
        let repaired = repair_tool_pairing(vec![
            user(vec![ContentPart::Text("hi".to_string())]),
            user(vec![tool_result("c1", "stale")]),
        ]);
        assert_eq!(
            repaired.len(),
            1,
            "the orphan message must go: {repaired:?}"
        );
    }

    /// #195: when the next message already answers *some* of the calls,
    /// the placeholders belong inside it. Splicing a new message in front
    /// would push the real result one further from its tool_use — the
    /// exact rejection the repair exists to prevent.
    #[test]
    fn a_partly_answered_tool_use_merges_rather_than_splices() {
        let repaired = repair_tool_pairing(vec![
            assistant(vec![tool_use("c1"), tool_use("c2")]),
            user(vec![tool_result("c1", "the real result")]),
        ]);

        assert_eq!(
            repaired.len(),
            2,
            "no message may be spliced between the pair: {repaired:?}"
        );
        let mut ids = result_ids(&repaired[1]);
        ids.sort();
        assert_eq!(ids, vec!["c1", "c2"], "both ids answer in one message");
        let real = repaired[1].parts.iter().any(
            |p| matches!(p, ContentPart::ToolResult { tool_use_id, content } if tool_use_id == "c1" && content == "the real result"),
        );
        assert!(real, "the real result must survive: {:?}", repaired[1]);
    }

    /// Two tool_use parts that (wrongly) share one id must not produce
    /// two placeholders for it.
    #[test]
    fn a_duplicated_tool_use_id_gets_one_placeholder() {
        let repaired = repair_tool_pairing(vec![assistant(vec![tool_use("c1"), tool_use("c1")])]);
        assert_eq!(result_ids(&repaired[1]), vec!["c1"]);
    }

    /// A well-formed conversation passes through untouched.
    #[test]
    fn a_paired_conversation_is_left_alone() {
        let input = vec![
            assistant(vec![tool_use("c1")]),
            user(vec![tool_result("c1", "ok")]),
        ];
        assert_eq!(repair_tool_pairing(input.clone()), input);
    }
}
