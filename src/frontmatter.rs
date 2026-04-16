//! Shared YAML frontmatter split/parse/serialize helpers for Markdown files.
//!
//! Files are assumed to start with `---\n`, followed by YAML, terminated by a
//! line containing only `---`. Both LF and CRLF line endings are accepted.

use anyhow::{Context, Result};

/// Split a Markdown file with YAML frontmatter into `(frontmatter, body)`.
/// Returns `None` when the file has no frontmatter. The body is returned
/// verbatim (with any leading newline intact).
pub fn split(raw: &str) -> Option<(&str, &str)> {
    let rest = raw
        .strip_prefix("---\n")
        .or_else(|| raw.strip_prefix("---\r\n"))?;
    let mut idx = 0;
    for line in rest.split_inclusive('\n') {
        let trimmed = line.trim_end_matches(|c| c == '\n' || c == '\r');
        if trimmed == "---" {
            let fm = &rest[..idx];
            let body_start = idx + line.len();
            return Some((fm, &rest[body_start..]));
        }
        idx += line.len();
    }
    None
}

/// Parse YAML frontmatter into a `serde_yaml::Mapping`. Empty or unparseable
/// input yields an empty mapping — convenient when merging catchup updates.
pub fn parse_mapping(fm: &str) -> serde_yaml::Mapping {
    serde_yaml::from_str(fm).unwrap_or_default()
}

/// Serialize a mapping + body back into a Markdown file with YAML frontmatter.
/// Emits `---\n{yaml}---\n\n{body}`; the body's leading newlines are stripped
/// to guarantee exactly one blank line after the closing `---`.
pub fn serialize(meta: &serde_yaml::Mapping, body: &str) -> Result<String> {
    let fm = serde_yaml::to_string(meta).context("failed to serialize frontmatter")?;
    let body_trimmed = body.trim_start_matches(|c: char| c == '\n' || c == '\r');
    Ok(format!("---\n{fm}---\n\n{body_trimmed}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_lf() {
        let raw = "---\nfoo: 1\nbar: two\n---\n\n# Body\n";
        let (fm, body) = split(raw).unwrap();
        assert_eq!(fm, "foo: 1\nbar: two\n");
        assert_eq!(body, "\n# Body\n");
    }

    #[test]
    fn split_crlf() {
        let raw = "---\r\nfoo: 1\r\n---\r\nbody\r\n";
        let (fm, body) = split(raw).unwrap();
        assert_eq!(fm, "foo: 1\r\n");
        assert_eq!(body, "body\r\n");
    }

    #[test]
    fn split_empty_frontmatter() {
        let raw = "---\n---\nbody\n";
        let (fm, body) = split(raw).unwrap();
        assert_eq!(fm, "");
        assert_eq!(body, "body\n");
    }

    #[test]
    fn split_no_frontmatter() {
        assert!(split("# Just markdown\n").is_none());
        assert!(split("").is_none());
        assert!(split("---\nno closing delimiter\n").is_none());
    }

    #[test]
    fn parse_mapping_accepts_empty() {
        assert!(parse_mapping("").is_empty());
    }

    #[test]
    fn parse_mapping_preserves_keys() {
        let m = parse_mapping("foo: 1\nbar: two\n");
        assert_eq!(m.get("foo").and_then(|v| v.as_i64()), Some(1));
        assert_eq!(m.get("bar").and_then(|v| v.as_str()), Some("two"));
    }

    #[test]
    fn serialize_roundtrip() {
        let original = "---\nfoo: 1\nbar: two\n---\n\nbody text\n";
        let (fm, body) = split(original).unwrap();
        let mapping = parse_mapping(fm);
        let out = serialize(&mapping, body).unwrap();
        // Re-split; keys and body should match.
        let (fm2, body2) = split(&out).unwrap();
        let mapping2 = parse_mapping(fm2);
        assert_eq!(mapping, mapping2);
        assert_eq!(body2.trim(), "body text");
    }
}
