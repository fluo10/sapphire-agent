//! Tool results, kept outside the workspace and addressed by hash.
//!
//! A coding session reads a lot of files. Persisting those results into
//! the workspace would grow it by the size of everything the agent ever
//! looked at — which is why tool calls were not persisted at all until
//! now. But a session cannot be restored without them: the Anthropic API
//! rejects a `tool_use` with no matching `tool_result`.
//!
//! So the results live here and the session log keeps a hash. Same shape
//! as `image_cache`, and for the same reason.
//!
//! Losing this cache is survivable. A missing result is replaced with a
//! placeholder that keeps the history valid; the model can call the tool
//! again if it needs to.
//!
//! **Not yet wired into `run_llm_turn`.** Tool calls are not persisted
//! at all today — `run_llm_turn` skips both `tool_use` and
//! `tool_result` messages when appending to the session store, so
//! nothing in the current request path ever calls `put`/`get` here.
//! This cache, `StoredPart::ToolUse` and `StoredPart::ToolResultRef`
//! exist for the replay design tracked as
//! [#191](https://github.com/fluo10/sapphire-agent/issues/191), not for
//! anything currently reachable.

use anyhow::Result;
use sha2::{Digest, Sha256};
use std::fmt::Write as _;
use std::path::PathBuf;
use std::sync::Arc;
use tracing::warn;

pub struct ToolResultCache {
    dir: PathBuf,
}

impl ToolResultCache {
    pub fn open(dir: PathBuf) -> Result<Arc<Self>> {
        std::fs::create_dir_all(&dir)?;
        Ok(Arc::new(Self { dir }))
    }

    /// `~/.cache/sapphire-agent/tool-results`, beside the image cache.
    pub fn default_dir() -> Option<PathBuf> {
        dirs::cache_dir().map(|d| d.join("sapphire-agent").join("tool-results"))
    }

    fn path_for(&self, sha256: &str) -> PathBuf {
        self.dir.join(sha256)
    }

    /// Store `content` and return its hash. Content-addressed, so
    /// storing the same result twice writes one file.
    pub fn put(&self, content: &str) -> Result<String> {
        let sha = sha256_hex(content.as_bytes());
        let path = self.path_for(&sha);
        if !path.exists() {
            std::fs::write(&path, content)?;
        }
        Ok(sha)
    }

    /// `None` for a hash that is not stored, or whose file cannot be
    /// read as text. Both are misses rather than errors: the caller
    /// substitutes a placeholder, and a lost result must never make a
    /// session unreadable.
    pub fn get(&self, sha256: &str) -> Option<String> {
        match std::fs::read(self.path_for(sha256)) {
            Ok(bytes) => match String::from_utf8(bytes) {
                Ok(text) => Some(text),
                Err(_) => {
                    warn!("tool-result cache: {sha256} is not valid UTF-8; treating as absent");
                    None
                }
            },
            Err(_) => None,
        }
    }
}

pub fn sha256_hex(bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(bytes);
    let digest = hasher.finalize();
    let mut s = String::with_capacity(64);
    for b in digest.iter() {
        let _ = write!(&mut s, "{b:02x}");
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_result_round_trips_by_its_hash() {
        let dir = tempfile::tempdir().unwrap();
        let cache = ToolResultCache::open(dir.path().to_path_buf()).unwrap();

        let sha = cache.put("the file contents").unwrap();
        assert_eq!(cache.get(&sha).as_deref(), Some("the file contents"));
    }

    /// Content-addressed: the same result stored twice is one file, and
    /// the caller gets the same handle back.
    #[test]
    fn identical_results_share_one_entry() {
        let dir = tempfile::tempdir().unwrap();
        let cache = ToolResultCache::open(dir.path().to_path_buf()).unwrap();

        let a = cache.put("same").unwrap();
        let b = cache.put("same").unwrap();
        assert_eq!(a, b);
        assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 1);
    }

    /// A miss is not an error. The caller substitutes a placeholder —
    /// losing a tool result must never make a session unreadable.
    #[test]
    fn an_absent_hash_is_none_rather_than_an_error() {
        let dir = tempfile::tempdir().unwrap();
        let cache = ToolResultCache::open(dir.path().to_path_buf()).unwrap();
        assert_eq!(
            cache.get("0000000000000000000000000000000000000000000000000000000000000000"),
            None
        );
    }

    /// Non-UTF8 on disk is corruption, not a panic.
    #[test]
    fn unreadable_bytes_are_a_miss() {
        let dir = tempfile::tempdir().unwrap();
        let cache = ToolResultCache::open(dir.path().to_path_buf()).unwrap();
        let sha = "1111111111111111111111111111111111111111111111111111111111111111";
        std::fs::write(dir.path().join(sha), [0xff, 0xfe]).unwrap();
        assert_eq!(cache.get(sha), None);
    }
}
