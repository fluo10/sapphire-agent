//! Intra-day session digests, kept outside the workspace.
//!
//! A digest answers "what has this session covered today" for the
//! cross-session block in other rooms' system prompts. It is
//! regenerated as the conversation grows, so successive digests are
//! near-duplicates of each other.
//!
//! That is why they do not live in the session log. `<workspace>/sessions`
//! is in the retrieve index, and a file accumulating a dozen restatements
//! of the same afternoon would skew every search that touches it. Nor can
//! they be pruned from the log: session events are chained by `parent`,
//! and removing one orphans its children.
//!
//! So: one entry per session, overwritten in place, outside the
//! workspace. Nothing accumulates, and losing the whole directory costs
//! today's cross-session block and nothing else — the permanent record
//! is the daily log.

use anyhow::{Result, bail};
use chrono::{DateTime, Utc};
use std::path::PathBuf;
use std::sync::Arc;
use tracing::warn;

use crate::session::IntradayDigestLine;

pub struct DigestCache {
    dir: PathBuf,
}

impl DigestCache {
    pub fn open(dir: PathBuf) -> Result<Arc<Self>> {
        std::fs::create_dir_all(&dir)?;
        Ok(Arc::new(Self { dir }))
    }

    /// `~/.cache/sapphire-agent/digests`, beside the image and
    /// tool-result caches.
    pub fn default_dir() -> Option<PathBuf> {
        dirs::cache_dir().map(|d| d.join("sapphire-agent").join("digests"))
    }

    /// Session ids reach here from a transport, so they are not trusted
    /// to be filenames.
    fn path_for(&self, session_id: &str) -> Result<PathBuf> {
        if session_id.is_empty()
            || session_id.contains(['/', '\\', ':'])
            || session_id.contains("..")
        {
            bail!("session id '{session_id}' is not usable as a cache filename");
        }
        Ok(self.dir.join(format!("{session_id}.json")))
    }

    pub fn put(&self, session_id: &str, digest: &str, since: Option<DateTime<Utc>>) -> Result<()> {
        self.put_at(session_id, digest, since, Utc::now())
    }

    /// `put` with an explicit timestamp. Exists so tests can place an
    /// entry on either side of a prune cutoff.
    pub fn put_at(
        &self,
        session_id: &str,
        digest: &str,
        since: Option<DateTime<Utc>>,
        digest_at: DateTime<Utc>,
    ) -> Result<()> {
        let path = self.path_for(session_id)?;
        let line = IntradayDigestLine {
            digest_at,
            digest: digest.to_string(),
            since,
        };
        std::fs::write(&path, serde_json::to_string(&line)?)?;
        Ok(())
    }

    pub fn get(&self, session_id: &str) -> Option<IntradayDigestLine> {
        let path = self.path_for(session_id).ok()?;
        let text = std::fs::read_to_string(path).ok()?;
        match serde_json::from_str(&text) {
            Ok(line) => Some(line),
            Err(e) => {
                warn!("digest cache: {session_id} is unreadable ({e}); treating as absent");
                None
            }
        }
    }

    /// Drop every entry digested before `cutoff`. Called once the daily
    /// log for that day has been written — the digest has done its job
    /// and the permanent record has taken over. Returns how many went.
    pub fn prune_before(&self, cutoff: DateTime<Utc>) -> usize {
        let Ok(entries) = std::fs::read_dir(&self.dir) else {
            return 0;
        };
        let mut removed = 0;
        for entry in entries.flatten() {
            let path = entry.path();
            let stale = std::fs::read_to_string(&path)
                .ok()
                .and_then(|t| serde_json::from_str::<IntradayDigestLine>(&t).ok())
                .map(|line| line.digest_at < cutoff)
                // Unreadable entries are useless anyway.
                .unwrap_or(true);
            if stale && std::fs::remove_file(&path).is_ok() {
                removed += 1;
            }
        }
        removed
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_digest_round_trips() {
        let dir = tempfile::tempdir().unwrap();
        let cache = DigestCache::open(dir.path().to_path_buf()).unwrap();

        cache.put("s1", "we fixed the parser", None).unwrap();
        let got = cache.get("s1").expect("the digest is cached");
        assert_eq!(got.digest, "we fixed the parser");
    }

    /// One entry per session, overwritten in place. This is what keeps
    /// near-identical digests from piling up: there is nowhere for them
    /// to pile.
    #[test]
    fn a_second_digest_replaces_the_first() {
        let dir = tempfile::tempdir().unwrap();
        let cache = DigestCache::open(dir.path().to_path_buf()).unwrap();

        cache.put("s1", "early", None).unwrap();
        cache.put("s1", "late", None).unwrap();

        assert_eq!(cache.get("s1").unwrap().digest, "late");
        assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 1);
    }

    /// Pruning is what the daily log calls once yesterday has been
    /// written up: the digest has served its purpose and the permanent
    /// record now carries the day.
    #[test]
    fn pruning_drops_entries_older_than_the_cutoff() {
        let dir = tempfile::tempdir().unwrap();
        let cache = DigestCache::open(dir.path().to_path_buf()).unwrap();

        cache.put("old", "yesterday's work", None).unwrap();
        let cutoff = Utc::now() + chrono::Duration::seconds(1);
        cache
            .put_at("fresh", "today's work", None, cutoff + chrono::Duration::hours(1))
            .unwrap();

        assert_eq!(cache.prune_before(cutoff), 1);
        assert!(cache.get("old").is_none());
        assert!(cache.get("fresh").is_some());
    }

    /// A session id is used as a filename, so anything that could climb
    /// out of the cache directory must not.
    #[test]
    fn a_traversing_session_id_is_refused() {
        let dir = tempfile::tempdir().unwrap();
        let cache = DigestCache::open(dir.path().to_path_buf()).unwrap();

        assert!(cache.put("../escape", "nope", None).is_err());
        assert!(cache.get("../escape").is_none());
    }

    /// Corruption is a miss, not a panic — the whole point of a cache.
    #[test]
    fn an_unparseable_entry_is_a_miss() {
        let dir = tempfile::tempdir().unwrap();
        let cache = DigestCache::open(dir.path().to_path_buf()).unwrap();
        std::fs::write(dir.path().join("s1.json"), "{ not json").unwrap();
        assert!(cache.get("s1").is_none());
    }
}
