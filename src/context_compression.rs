//! Context compression: summarize older messages when conversation approaches
//! the model's context window limit.
//!
//! Strategy:
//! 1. Estimate token count of system prompt + messages
//! 2. If above threshold (default 80% of context window), compress
//! 3. Keep the most recent N messages verbatim
//! 4. Summarize everything before that into a single user message
//! 5. Return the compressed history

use crate::config::CompressionConfig;
use crate::provider::{ChatMessage, ContentPart, Provider, Role};
use tracing::{info, warn};

/// Rough token estimate for a string.
///
/// Uses a simple heuristic: ~4 characters per token for ASCII,
/// ~1.5 characters per token for non-ASCII (CJK, etc.).
pub fn estimate_tokens(text: &str) -> usize {
    let mut ascii_chars = 0usize;
    let mut non_ascii_chars = 0usize;
    for ch in text.chars() {
        if ch.is_ascii() {
            ascii_chars += 1;
        } else {
            non_ascii_chars += 1;
        }
    }
    // Rough estimate: ASCII ~4 chars/token, non-ASCII ~1.5 chars/token
    let ascii_tokens = ascii_chars / 4;
    let non_ascii_tokens = (non_ascii_chars * 2 + 2) / 3; // ceil(n * 2/3)
    ascii_tokens + non_ascii_tokens
}

/// Estimate the total token usage for a system prompt + message history.
pub fn estimate_total_tokens(system: Option<&str>, messages: &[ChatMessage]) -> usize {
    let system_tokens = system.map(|s| estimate_tokens(s)).unwrap_or(0);
    let message_tokens: usize = messages.iter().map(|m| estimate_message_tokens(m)).sum();
    // Add a small overhead for message framing (~4 tokens per message)
    system_tokens + message_tokens + messages.len() * 4
}

/// Estimate tokens for a single ChatMessage.
fn estimate_message_tokens(msg: &ChatMessage) -> usize {
    msg.parts
        .iter()
        .map(|p| match p {
            ContentPart::Text(t) => estimate_tokens(t),
            ContentPart::ToolUse { name, input, .. } => {
                estimate_tokens(name) + estimate_tokens(&input.to_string())
            }
            ContentPart::ToolResult { content, .. } => estimate_tokens(content),
        })
        .sum()
}

/// Check whether compression is needed and, if so, compress the history.
///
/// Returns `Ok(None)` if no compression was needed.
/// Returns `Ok(Some(compressed))` with the new message history if compressed.
pub async fn maybe_compress(
    provider: &dyn Provider,
    system: Option<&str>,
    messages: &[ChatMessage],
    config: &CompressionConfig,
) -> anyhow::Result<Option<Vec<ChatMessage>>> {
    if !config.enabled {
        return Ok(None);
    }

    let total_tokens = estimate_total_tokens(system, messages);
    let threshold_tokens = (config.context_window as f64 * config.threshold) as usize;

    if total_tokens < threshold_tokens {
        return Ok(None);
    }

    info!(
        "Context compression triggered: ~{total_tokens} tokens estimated \
         (threshold: {threshold_tokens}, window: {})",
        config.context_window
    );

    // Find the split point: keep the most recent `preserve_recent` messages,
    // but ensure we don't split in the middle of a tool-call/result pair.
    let split = find_safe_split_point(messages, config.preserve_recent);

    if split == 0 {
        // Nothing to compress — all messages are in the "recent" window
        return Ok(None);
    }

    let to_summarize = &messages[..split];
    let to_keep = &messages[split..];

    // Generate a summary of the older messages
    let summary = generate_summary(provider, to_summarize).await?;

    info!(
        "Compressed {} messages into summary ({} → ~{} tokens)",
        split,
        estimate_total_tokens(None, to_summarize),
        estimate_tokens(&summary),
    );

    // Build the compressed history: summary message + recent messages
    let mut compressed = Vec::with_capacity(1 + to_keep.len());
    compressed.push(ChatMessage {
        role: Role::User,
        parts: vec![ContentPart::Text(format!(
            "[Context Summary — earlier messages were compressed]\n\n{summary}"
        ))],
    });
    // Insert a placeholder assistant acknowledgment so that the message
    // sequence alternates correctly (user → assistant → user → …).
    compressed.push(ChatMessage::assistant(
        "Understood. I have the context from our earlier conversation.",
    ));
    compressed.extend_from_slice(to_keep);

    Ok(Some(compressed))
}

/// Find a safe split point that doesn't break tool-call/result pairs.
///
/// We want to keep at least `preserve_recent` messages at the end,
/// but if the boundary lands between a tool-use assistant message and
/// its corresponding tool-result user message, we move the boundary
/// earlier to keep the pair together.
fn find_safe_split_point(messages: &[ChatMessage], preserve_recent: usize) -> usize {
    if messages.len() <= preserve_recent {
        return 0;
    }

    let mut split = messages.len() - preserve_recent;

    // If the message at `split` is a tool-result (user message with ToolResult parts),
    // move split back to include the preceding assistant tool-use message.
    while split > 0 {
        let msg = &messages[split];
        let is_tool_result = msg.role == Role::User
            && msg
                .parts
                .iter()
                .any(|p| matches!(p, ContentPart::ToolResult { .. }));
        if is_tool_result {
            split -= 1;
        } else {
            break;
        }
    }

    // Also check: if the message just before split is an assistant message
    // with tool_use, include it in the "keep" side to maintain the pair.
    if split > 0 {
        let prev = &messages[split - 1];
        let has_tool_use = prev.role == Role::Assistant
            && prev
                .parts
                .iter()
                .any(|p| matches!(p, ContentPart::ToolUse { .. }));
        if has_tool_use {
            // The message at split should be the tool result — keep the pair together
            // by not moving split further.
        }
    }

    split
}

/// Generate a concise summary of a sequence of messages using the LLM.
async fn generate_summary(
    provider: &dyn Provider,
    messages: &[ChatMessage],
) -> anyhow::Result<String> {
    // Build a textual representation of the messages to summarize
    let mut transcript = String::new();
    for msg in messages {
        let role_label = match msg.role {
            Role::User => "User",
            Role::Assistant => "Assistant",
        };
        for part in &msg.parts {
            match part {
                ContentPart::Text(t) => {
                    transcript.push_str(&format!("{role_label}: {t}\n\n"));
                }
                ContentPart::ToolUse { name, .. } => {
                    transcript.push_str(&format!("{role_label}: [Called tool: {name}]\n\n"));
                }
                ContentPart::ToolResult { content, .. } => {
                    // Truncate long tool results to keep the summary prompt manageable
                    let truncated = if content.len() > 500 {
                        format!("{}... (truncated)", &content[..500])
                    } else {
                        content.clone()
                    };
                    transcript.push_str(&format!("{role_label}: [Tool result: {truncated}]\n\n"));
                }
            }
        }
    }

    // Cap the transcript to avoid exceeding context on the summary call itself
    let max_transcript_chars = 50_000;
    if transcript.len() > max_transcript_chars {
        transcript.truncate(max_transcript_chars);
        transcript.push_str("\n... (transcript truncated for summarization)");
    }

    let prompt = format!(
        "Summarize the following conversation concisely. \
         Preserve key information: decisions made, code context, task state, \
         important facts, and any instructions or preferences expressed. \
         Focus on information that would be needed to continue the conversation. \
         Write the summary in the same language(s) used in the conversation.\n\n\
         ---\n\n{transcript}"
    );

    let summary_messages = vec![ChatMessage::user(&prompt)];
    let response = provider.chat(None, &summary_messages, None).await?;

    match response.text {
        Some(text) if !text.is_empty() => Ok(text),
        _ => {
            warn!("Summary generation returned empty response");
            Ok("(Earlier conversation context was compressed but summary generation failed.)".into())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_estimate_tokens_ascii() {
        // "hello world" = 11 chars, ~2-3 tokens
        let tokens = estimate_tokens("hello world");
        assert!(tokens > 0);
        assert!(tokens < 10);
    }

    #[test]
    fn test_estimate_tokens_cjk() {
        // 6 CJK characters, ~4 tokens
        let tokens = estimate_tokens("こんにちは世界");
        assert!(tokens > 0);
    }

    #[test]
    fn test_estimate_tokens_mixed() {
        let tokens = estimate_tokens("Hello こんにちは World");
        assert!(tokens > 0);
    }

    #[test]
    fn test_find_safe_split_point_basic() {
        let messages = vec![
            ChatMessage::user("msg1"),
            ChatMessage::assistant("msg2"),
            ChatMessage::user("msg3"),
            ChatMessage::assistant("msg4"),
            ChatMessage::user("msg5"),
            ChatMessage::assistant("msg6"),
        ];
        let split = find_safe_split_point(&messages, 2);
        assert_eq!(split, 4);
    }

    #[test]
    fn test_find_safe_split_preserves_all_when_few() {
        let messages = vec![
            ChatMessage::user("msg1"),
            ChatMessage::assistant("msg2"),
        ];
        let split = find_safe_split_point(&messages, 5);
        assert_eq!(split, 0);
    }

    #[test]
    fn test_find_safe_split_avoids_breaking_tool_pair() {
        use serde_json::json;

        let messages = vec![
            ChatMessage::user("start"),
            ChatMessage::assistant("thinking"),
            ChatMessage::user("question"),
            ChatMessage::assistant_with_tools(
                None,
                vec![crate::provider::ToolCall {
                    id: "t1".into(),
                    name: "search".into(),
                    input: json!({}),
                }],
            ),
            ChatMessage::tool_results(vec![("t1".into(), "result".into())]),
            ChatMessage::assistant("final answer"),
        ];

        // preserve_recent=2 would normally split at index 4 (tool result),
        // but it should move back to not break the tool pair.
        let split = find_safe_split_point(&messages, 2);
        assert!(split <= 3, "split should be at or before the tool-use message");
    }
}
