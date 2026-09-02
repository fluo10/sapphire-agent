# 全セッションストアでツール呼び出しを永続化する実装計画

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `channel` / `cross-device` / `device-default` / `mcp` の 4 ストアでもツール呼び出しを永続化し、5 ストア共通の compaction チェックポイントから LLM 履歴を復元できるようにする。

**Architecture:** ツール結果はワークスペース外の `ToolResultCache` に content-addressed で置き、JSONL には `ContentPart::ToolResultRef` を書く。compaction 時に「この要約がどこまでを吸収したか」を `covers_through`（メッセージの `id`）としてストア自身が算出して記録し、復元はそこから先だけを replay する。表示・恒久記録向けの読み取り経路は全リプレイのまま分離する。

**Tech Stack:** Rust 2024 edition / nightly toolchain、serde + serde_json、chrono、uuid (v7)、tokio、anyhow

**Spec:** [`docs/superpowers/specs/2026-09-02-session-tool-persistence-design.md`](../specs/2026-09-02-session-tool-persistence-design.md)

## Global Constraints

- ツール結果は `ToolSet::execute` が既に 50 000 バイト（`crate::tools::OUTPUT_CAP_BYTES`）に切り詰めている。この計画で上限を変えない。
- 失われたツール結果の代替文は 1 種類のみ: `"[this tool result is no longer stored; call the tool again if you need it]"`（`MISSING_RESULT`）。
- 日次ログ（`sessions_for_day` / `sessions_for_day_filtered`）は **text-only のまま**。ツール結果を hydrate してはならない。
- `tool_use` と `tool_result` の対は「両方書けるか、どちらも書かないか」。片方だけディスクに残すことは許されない。
- 既存の JSONL はすべて読めなければならない。新フィールドは `#[serde(default, skip_serializing_if = "Option::is_none")]`。
- コミット前に必ず: `cargo +nightly fmt --all` → `cargo clippy --workspace -- -D warnings` → `cargo test --workspace`
- コミットスコープは `(sessions)`。`CLAUDE.md` の規約により agent 本体の変更は `cliff.toml` の変更を必要としない。

## 用語

- **チェックポイント**: 最新の compaction 要約と、それが吸収した最後のメッセージを指す `covers_through`。
- **復元スタブ**: 要約を会話の形に戻した 2 メッセージ（user + assistant）。`compaction_stub()` が唯一の生成元。
- **LLM 向け読み取り**: `load_all` / `load_session` / `history_for_model`。チェックポイント切り詰め + hydrate + 対の修復を通す。
- **記録向け読み取り**: `sessions_for_day*` / `load_session_full` / `history`。切り詰めない。

---

## Phase 1 — チャンネル digest を `DigestCache` へ（#190）

### Task 1: `SessionStore::intraday_digests_for_day` がキャッシュを見る

**Files:**
- Modify: `src/session.rs:444-472`（`intraday_digests_for_day`）
- Modify: `src/periodic_log.rs:1236-1242`（呼び出し側）
- Test: `src/session.rs`（末尾の `mod tests`）

**Interfaces:**
- Produces: `SessionStore::intraday_digests_for_day(&self, date: NaiveDate, boundary_hour: u8, cache: Option<&crate::digest_cache::DigestCache>) -> Vec<(SessionMeta, IntradayDigestLine)>`
- Consumes: `DigestCache::get(&self, session_id: &str) -> Option<IntradayDigestLine>`（既存）

- [ ] **Step 1: 失敗するテストを書く**

`src/session.rs` の `mod tests` の末尾に追加:

```rust
// ── intra-day digest がキャッシュから引かれる (#190) ──────────────────

fn store_with_one_session() -> (tempfile::TempDir, SessionStore, String) {
    let tmp = tempfile::TempDir::new().unwrap();
    let store = SessionStore::new(tmp.path().to_path_buf(), "channel");
    let key = ("!room:example.org".to_string(), None);
    let sid = store.create_session(&key, "matrix", "default").unwrap();
    (tmp, store, sid)
}

/// digest はセッションファイルではなくキャッシュに置かれる。ファイル側に
/// 何も書かれていなくても、today ブロックには現れなければならない。
#[test]
fn a_cached_digest_is_returned_without_any_line_in_the_file() {
    let (_tmp, store, sid) = store_with_one_session();
    let cache_dir = tempfile::TempDir::new().unwrap();
    let cache = crate::digest_cache::DigestCache::open(cache_dir.path().to_path_buf()).unwrap();
    cache.put(&sid, "we fixed the parser", None).unwrap();

    let today = local_date_for_timestamp(Local::now(), 4);
    let got = store.intraday_digests_for_day(today, 4, Some(&cache));

    assert_eq!(got.len(), 1, "the cached digest must be found; got {got:?}");
    assert_eq!(got[0].1.digest, "we fixed the parser");
    assert_eq!(got[0].0.session_id, sid);
}

/// アップグレード直後、キャッシュはまだ空でファイルには前バージョンが書いた
/// digest 行が残っている。その日の today ブロックが空になってはいけない。
#[test]
fn a_file_digest_is_the_fallback_when_the_cache_has_nothing() {
    let (_tmp, store, sid) = store_with_one_session();
    let path = store.absolute_path_for(&sid).unwrap();
    let line = serde_json::to_string(&IntradayDigestLine {
        digest_at: Utc::now(),
        digest: "written by the previous version".to_string(),
        since: None,
    })
    .unwrap();
    let mut f = OpenOptions::new().append(true).open(&path).unwrap();
    writeln!(f, "{line}").unwrap();
    drop(f);

    let cache_dir = tempfile::TempDir::new().unwrap();
    let cache = crate::digest_cache::DigestCache::open(cache_dir.path().to_path_buf()).unwrap();

    let today = local_date_for_timestamp(Local::now(), 4);
    let got = store.intraday_digests_for_day(today, 4, Some(&cache));

    assert_eq!(got.len(), 1, "the file's own line must still be read");
    assert_eq!(got[0].1.digest, "written by the previous version");
}

/// キャッシュとファイルの両方にある場合、キャッシュが勝つ — そちらが
/// 現在の書き込み先で、常に新しい。
#[test]
fn the_cache_wins_over_a_stale_file_line() {
    let (_tmp, store, sid) = store_with_one_session();
    let path = store.absolute_path_for(&sid).unwrap();
    let line = serde_json::to_string(&IntradayDigestLine {
        digest_at: Utc::now(),
        digest: "stale".to_string(),
        since: None,
    })
    .unwrap();
    let mut f = OpenOptions::new().append(true).open(&path).unwrap();
    writeln!(f, "{line}").unwrap();
    drop(f);

    let cache_dir = tempfile::TempDir::new().unwrap();
    let cache = crate::digest_cache::DigestCache::open(cache_dir.path().to_path_buf()).unwrap();
    cache.put(&sid, "fresh", None).unwrap();

    let today = local_date_for_timestamp(Local::now(), 4);
    let got = store.intraday_digests_for_day(today, 4, Some(&cache));
    assert_eq!(got[0].1.digest, "fresh");
}
```

- [ ] **Step 2: テストが失敗することを確認**

Run: `cargo test --lib session::tests::a_cached_digest -- --nocapture`
Expected: コンパイルエラー — `intraday_digests_for_day` は引数を 2 つしか取らない

- [ ] **Step 3: `intraday_digests_for_day` にキャッシュ引数を足す**

`src/session.rs:444` の関数を置き換える:

```rust
    /// Walk every session file under `sessions_dir` and return the latest
    /// intra-day digest per session whose `digest_at` falls inside the
    /// local-time `date` window, paired with the session's metadata.
    ///
    /// The digest text comes from `cache` — `<cache_dir>/digests/`, one
    /// entry per session, overwritten in place. It used to be appended
    /// to the session's own JSONL, which put a dozen near-identical
    /// restatements of the same afternoon inside a file the retrieve
    /// indexer walks (#190).
    ///
    /// A digest line still present in the file is read as a fallback, so
    /// the upgrade does not blank out the day it lands on. Nothing
    /// writes those lines any more.
    pub fn intraday_digests_for_day(
        &self,
        date: NaiveDate,
        boundary_hour: u8,
        cache: Option<&crate::digest_cache::DigestCache>,
    ) -> Vec<(SessionMeta, IntradayDigestLine)> {
        let (day_start, day_end) = day_window(date, boundary_hour);
        let mut out = Vec::new();
        for path in collect_session_files(&self.base_dir, self.kind) {
            // mtime pre-filter: a file last touched before day_start
            // belongs to a session that said nothing today, and a
            // session that said nothing today has no digest for today.
            if let Ok(meta_fs) = path.metadata()
                && let Ok(mtime) = meta_fs.modified()
            {
                let mtime_utc: DateTime<Utc> = mtime.into();
                if mtime_utc < day_start {
                    continue;
                }
            }
            let Some((meta, file_digest)) = load_meta_and_latest_intraday_digest(&path) else {
                continue;
            };
            let digest = cache
                .and_then(|c| c.get(&meta.session_id))
                .or(file_digest);
            let Some(d) = digest else { continue };
            if d.digest_at >= day_start && d.digest_at < day_end {
                out.push((meta, d));
            }
        }
        out.sort_by_key(|(meta, _)| meta.created_at);
        out
    }
```

- [ ] **Step 4: 呼び出し側を直す**

`src/periodic_log.rs:1236-1242` の 3 行を置き換える:

```rust
    entries.extend(channel_store.intraday_digests_for_day(today, boundary_hour, digest_cache));
    if let Some(s) = cross_device_store {
        entries.extend(s.intraday_digests_for_day(today, boundary_hour, digest_cache));
    }
    if let Some(s) = device_default_store {
        entries.extend(s.intraday_digests_for_day(today, boundary_hour, digest_cache));
    }
```

- [ ] **Step 5: テストが通ることを確認**

Run: `cargo test --workspace`
Expected: PASS

- [ ] **Step 6: コミット**

```bash
git add src/session.rs src/periodic_log.rs
git commit -m "feat(sessions): read channel intra-day digests from the shared cache

The digest text now comes from <cache_dir>/digests/ rather than the
session's own JSONL. A line still present in the file is read as a
fallback so the upgrade does not blank out the day it lands on.

Nothing writes to the cache for these stores yet — that is the next
commit. Read side first so the two can be reviewed apart."
```

---

### Task 2: `Agent` が digest をキャッシュに書く

**Files:**
- Modify: `src/agent.rs:41-120`（`Agent` の定義と `new`）
- Modify: `src/agent.rs:253-274`（`summarize_on_shutdown`）
- Modify: `src/agent.rs:414-468`（`flush_intraday_digest`）
- Modify: `src/session.rs:415-437`（`append_intraday_digest` を削除）
- Modify: `src/main.rs:674-692, 815-823`（配線）

**Interfaces:**
- Consumes: `DigestCache::put(&self, session_id: &str, digest: &str, since: Option<DateTime<Utc>>) -> Result<()>`（既存）
- Produces: `Agent::new(..., digest_cache: Option<Arc<crate::digest_cache::DigestCache>>)` — 引数は `image_cache` の**後ろ**に足す

- [ ] **Step 1: `Agent` にフィールドを足す**

`src/agent.rs:50` の `image_cache` の直後に:

```rust
    /// Workspace-external intra-day digest cache. `None` disables the
    /// cross-session "today" block for this agent's rooms — the digest
    /// has nowhere to go, and writing it back into the session JSONL is
    /// what #190 exists to stop.
    digest_cache: Option<Arc<crate::digest_cache::DigestCache>>,
```

`Agent::new` の引数リスト（`src/agent.rs:93` の `image_cache` の後）に
`digest_cache: Option<Arc<crate::digest_cache::DigestCache>>,` を足し、
構造体リテラル（`src/agent.rs:109` の `image_cache,` の後）に `digest_cache,` を足す。

- [ ] **Step 2: `flush_intraday_digest` の書き込み先を変える**

`src/agent.rs:427-433` の `session_id` 解決の直後、`activity_ts` 解決の**前**に挿入:

```rust
        // Resolved before the model call, not after: with nowhere to put
        // the result, generating it would burn a provider call for
        // nothing.
        let Some(cache) = self.digest_cache.clone() else {
            debug!("No digest cache; skipping the intra-day digest for {session_id}");
            return;
        };
```

`src/agent.rs:457-463` の `append_intraday_digest` 呼び出しを置き換える:

```rust
        if let Err(e) = cache.put(&session_id, &summary, None) {
            warn!("Failed to cache the intra-day digest for {session_id}: {e}");
            return;
        }
```

- [ ] **Step 3: `summarize_on_shutdown` の digest 発行先を変える**

`src/agent.rs:260-268` を置き換える:

```rust
                    // Also publish an intra-day digest so the
                    // cross-session today block picks up what this
                    // session covered before we went down. It goes to
                    // the workspace-external cache, not the session's
                    // own JSONL (#190).
                    if let Some(cache) = self.digest_cache.as_ref()
                        && let Err(e) = cache.put(&session_id, &summary, None)
                    {
                        warn!("Failed to cache the shutdown intra-day digest for {session_id}: {e}");
                    }
```

- [ ] **Step 4: `append_intraday_digest` を削除**

`src/session.rs:415-437` のメソッドを丸ごと削除する。呼び出し元は上の 2 箇所だけなので、
削除後に `cargo check` が通ればそれが確認になる。

- [ ] **Step 5: `main.rs` の配線**

`src/main.rs:815-823` の `Agent::new` 呼び出しに引数を足す:

```rust
                let agent = Arc::new(Agent::new(
                    config.clone(),
                    Arc::clone(&channels),
                    Arc::clone(&registry),
                    Arc::clone(&workspace),
                    Some(Arc::clone(&tool_set)),
                    Arc::clone(&channel_session_store),
                    image_cache.clone(),
                    digest_cache_for_agent.clone(),
                ));
```

`src/main.rs:692`（`digest_cache` を組み立てた直後、`serve_state` に move される前）に:

```rust
            // Cloned before `ServeState` takes ownership: the channel
            // agent writes its own rooms' digests here too, now that
            // they no longer go into the session JSONL (#190).
            let digest_cache_for_agent = digest_cache.clone();
```

- [ ] **Step 6: 全テストを走らせる**

Run: `cargo +nightly fmt --all && cargo clippy --workspace -- -D warnings && cargo test --workspace`
Expected: PASS

`Agent` には `mod tests` が無いので、この差分の「digest がセッションファイルに書かれない」
側は削除そのものが保証になる — `append_intraday_digest` が存在しないので、書く方法が無い。
読み取り側は Task 1 のテストが押さえている。

- [ ] **Step 7: コミット**

```bash
git add src/agent.rs src/session.rs src/main.rs
git commit -m "feat(sessions): write channel intra-day digests to the cache, not the log

Closes #190.

SessionStore::append_intraday_digest wrote into the session's own JSONL
and then called notify_updated, so every regenerated digest landed in
the retrieve index. A digest is a near-duplicate of the last one by
construction, so a long-running room accumulated a dozen restatements
of the same afternoon inside one indexed document.

They now go to <cache_dir>/digests/ — one entry per session,
overwritten in place, the same place ACP has kept them since #188. The
heartbeat's existing prune_before call, which runs once the daily log
for that day is written, now covers these too.

The cache being unavailable skips the digest rather than falling back
to the log: the fallback is what the issue is about."
```

---

## Phase 2 — 共有化と #195

### Task 3: `src/session_storage.rs` を作り、保存境界の規則を集める

**Files:**
- Create: `src/session_storage.rs`
- Modify: `src/main.rs:16-28` 付近（`mod session_storage;` を追加）
- Modify: `src/acp_session.rs:36-44`（`MISSING_RESULT` を再エクスポートに）
- Modify: `src/acp_session.rs:349-367`（`elide_oversized_input` を移動）

**Interfaces:**
- Produces:
  - `pub const MISSING_RESULT: &str`
  - `pub fn elide_oversized_input(input: &serde_json::Value) -> serde_json::Value`

- [ ] **Step 1: 新しいモジュールを作る**

Create `src/session_storage.rs`:

```rust
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
```

- [ ] **Step 2: モジュールを登録する**

`src/main.rs` の `mod session;` の直後に追加（アルファベット順の位置に合わせる）:

```rust
mod session_storage;
```

- [ ] **Step 3: `acp_session.rs` から移動元を消す**

`src/acp_session.rs:36-44` の `MISSING_RESULT` 定義を削除し、代わりに `use` を足す
（`src/acp_session.rs:22` の `use crate::tool_result_cache::ToolResultCache;` の下）:

```rust
use crate::session_storage::{MISSING_RESULT, elide_oversized_input};
```

`src/acp_session.rs:349-367` の `elide_oversized_input` メソッドを削除し、
`src/acp_session.rs:315` の呼び出しを `input: elide_oversized_input(input),` に変える。

`MISSING_RESULT` は `acp_session` の外からも参照されている可能性があるので確認する:

Run: `grep -rn "MISSING_RESULT" src/`
参照元がすべて `crate::session_storage::MISSING_RESULT` を指すよう直す。

- [ ] **Step 4: ビルドとテスト**

Run: `cargo test --workspace`
Expected: PASS（挙動は変わっていないので、既存の ACP テストがそのまま通る）

- [ ] **Step 5: コミット**

```bash
git add src/session_storage.rs src/main.rs src/acp_session.rs
git commit -m "refactor(sessions): move the storage-boundary rules out of the ACP store

MISSING_RESULT and elide_oversized_input are not facts about ACP. They
are facts about the Anthropic API and about <workspace>/sessions being
inside the retrieve index, and the four SessionStore kinds are about to
need both.

Pure move — no behaviour change."
```

---

### Task 4: 対の修復を共有関数にし、#195 を直す

**Files:**
- Modify: `src/session_storage.rs`（`repair_tool_pairing` を追加）
- Modify: `src/acp_session.rs:534-657`（`history` の修復部分を差し替え）
- Test: `src/session_storage.rs`

**Interfaces:**
- Produces: `pub fn repair_tool_pairing(messages: Vec<ChatMessage>) -> Vec<ChatMessage>`

- [ ] **Step 1: 失敗するテストを書く**

`src/session_storage.rs` の `mod tests` に追加:

```rust
    use crate::provider::{ChatMessage, ContentPart, Role};

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
        ChatMessage { role: Role::Assistant, parts, input_kind: None, user_id: None }
    }

    fn user(parts: Vec<ContentPart>) -> ChatMessage {
        ChatMessage { role: Role::User, parts, input_kind: None, user_id: None }
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
        assert_eq!(repaired.len(), 1, "the orphan message must go: {repaired:?}");
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
```

- [ ] **Step 2: テストが失敗することを確認**

Run: `cargo test --lib session_storage`
Expected: コンパイルエラー — `repair_tool_pairing` が存在しない

- [ ] **Step 3: `repair_tool_pairing` を実装する**

`src/session_storage.rs` の `elide_oversized_input` の下に追加:

```rust
use crate::provider::{ChatMessage, ContentPart, Role};
use std::collections::HashSet;

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
        kept.push(ChatMessage { parts, ..message.clone() });
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
```

- [ ] **Step 4: テストが通ることを確認**

Run: `cargo test --lib session_storage`
Expected: PASS（5 テストすべて）

- [ ] **Step 5: `AcpSessionStore::history` を共有関数に置き換える**

`src/acp_session.rs:534-656` のコメント塊と修復ループを丸ごと削除し、`out` を組み立てた
直後（`src/acp_session.rs:532` の walk ループの閉じ括弧の後）を次にする:

```rust
        // The chain can hold a `tool_use` whose `tool_result` never made
        // it to disk — the second append failed, the process died
        // between the two, a sync landed only half the pair. The repair
        // is shared with the four `SessionStore` kinds, which have the
        // same gap for the same reasons.
        Some(crate::session_storage::repair_tool_pairing(out))
    }
```

- [ ] **Step 6: ACP の既存テストが通ることを確認**

Run: `cargo test --workspace`
Expected: PASS。#195 の対象ケースを直接固定していた ACP 側のテストがあれば、
新しい挙動（マージ）に合わせて期待値を更新する — その差分こそが #195 の修正である。

- [ ] **Step 7: コミット**

```bash
git add src/session_storage.rs src/acp_session.rs
git commit -m "refactor(sessions): share the tool-pairing repair, and fix a partly-answered call

Closes #195.

The repair is about to be needed by four more stores, so it moves out of
AcpSessionStore. The move surfaces the bug #195 describes, and fixing it
here is cheaper than knowingly widening it to Matrix and Discord.

Given an assistant message with tool_use c1 and c2 where only c1 was
answered, the old code spliced a fresh message carrying c2's placeholder
between the pair. That pushed c1's real result one message further from
its tool_use, so the repair produced exactly the rejection it exists to
prevent. The placeholders now merge into the existing tool_result
message; a new message is spliced only when there is no such message to
merge into."
```

---

## Phase 3 — 4 ストアでツール呼び出しを永続化（#194）

### Task 5: `ContentPart::ToolResultRef` を追加する

**Files:**
- Modify: `src/provider/mod.rs:64-73`
- Modify: `src/provider/anthropic.rs:294-323`
- Modify: `src/provider/openai_compatible.rs:252-283, 313-332`
- Modify: `src/context_compression.rs:45-61, 199-232`
- Modify: `src/serve/mod.rs:1062-1083`

**Interfaces:**
- Produces: `ContentPart::ToolResultRef { tool_use_id: String, sha256: Option<String> }`

- [ ] **Step 1: variant を足す**

`src/provider/mod.rs:69-72` の `ToolResult` の直後に追加:

```rust
    /// Reference to a tool result stored in the workspace-external
    /// tool-result cache. **A storage-boundary form only.** Unlike
    /// [`ContentPart::ImageRef`], which stays in long-lived in-memory
    /// history so images are not re-billed, this variant never lives in
    /// memory: a tool result has to be whole when the model sees it, so
    /// every read path hydrates it back to [`ContentPart::ToolResult`].
    ///
    /// The provider arms below are the safety net, not the plan. They
    /// degrade this to a `tool_result` carrying `MISSING_RESULT`, so a
    /// read path that forgot to hydrate produces a thinner history
    /// rather than an API rejection.
    ///
    /// `sha256: None` means the result had nowhere to be stored — the
    /// cache was unavailable when the line was written. A reader treats
    /// that the same as a hash whose entry has since been evicted.
    ToolResultRef {
        tool_use_id: String,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        sha256: Option<String>,
    },
```

- [ ] **Step 2: コンパイルさせて、網羅 match の全箇所を洗い出す**

Run: `cargo check --workspace 2>&1 | grep -A 3 "non-exhaustive\|patterns.*not covered"`
Expected: 6 箇所のエラー。以下の Step で順に埋める。

- [ ] **Step 3: Anthropic provider**

`src/provider/anthropic.rs:317-322` の `ToolResult` アームの直後に:

```rust
            // Only reachable when a read path failed to hydrate. Keep
            // the pairing — that is what the API validates — and let the
            // model call the tool again if it needs the content.
            ContentPart::ToolResultRef { tool_use_id, .. } => ApiPart::ToolResult {
                tool_use_id: tool_use_id.clone(),
                content: crate::session_storage::MISSING_RESULT.to_string(),
            },
```

- [ ] **Step 4: OpenAI-compatible provider（2 箇所）**

`src/provider/openai_compatible.rs` の `convert_user_message`、`ToolResult` アームの直後:

```rust
            // See the Anthropic provider's arm: hydration failed
            // upstream, and the pairing matters more than the content.
            ContentPart::ToolResultRef { tool_use_id, .. } => {
                out.push(ApiMessage {
                    role: "tool",
                    content: Some(ApiContent::Text(
                        crate::session_storage::MISSING_RESULT.to_string(),
                    )),
                    tool_calls: None,
                    tool_call_id: Some(tool_use_id.clone()),
                });
            }
```

`convert_assistant_message` の「Assistant should not carry images or tool results」の
グループにも足す:

```rust
            ContentPart::Image { .. }
            | ContentPart::ImageRef { .. }
            | ContentPart::ToolResult { .. }
            | ContentPart::ToolResultRef { .. } => {}
```

- [ ] **Step 5: `context_compression.rs`（2 箇所）**

`estimate_message_tokens`（`src/context_compression.rs:59`）の `ToolResult` アームの後:

```rust
            // Hydration happens before this runs on any real path, so
            // this is the un-hydrated form only — a placeholder's worth.
            ContentPart::ToolResultRef { .. } => estimate_tokens(MISSING_RESULT),
```

要約用トランスクリプト（`src/context_compression.rs:215` 付近）の `ToolResult` アームの後:

```rust
                ContentPart::ToolResultRef { .. } => {
                    transcript.push_str(&format!("{role_label}: [Tool result: unavailable]\n\n"));
                }
```

ファイル先頭の `use` に `use crate::session_storage::MISSING_RESULT;` を足す。

- [ ] **Step 6: `/rpc` のセッション一覧 JSON**

`src/serve/mod.rs:1079-1081` の `ToolResult` アームの後:

```rust
                    ContentPart::ToolResultRef { tool_use_id, sha256 } => {
                        // Same shape as tool_result, with the cache key
                        // surfaced instead of content a caller cannot be
                        // handed from a listing anyway.
                        json!({ "type": "tool_result", "tool_use_id": tool_use_id, "sha256": sha256 })
                    }
```

- [ ] **Step 7: ビルドとテスト**

Run: `cargo +nightly fmt --all && cargo clippy --workspace -- -D warnings && cargo test --workspace`
Expected: PASS。まだ誰も `ToolResultRef` を書かないので挙動は変わらない。

- [ ] **Step 8: コミット**

```bash
git add src/provider/ src/context_compression.rs src/serve/mod.rs
git commit -m "feat(sessions): add ContentPart::ToolResultRef

The four SessionStore kinds serialize ContentPart directly, so the
reference form they need for a cached tool result has to live on that
enum rather than in a store-private type the way ACP's StoredPart does.

Every match over ContentPart is exhaustive, so adding the variant made
the compiler name all six sites. Both providers degrade it to a
tool_result carrying MISSING_RESULT rather than dropping it: a read path
that forgets to hydrate then produces a thinner history instead of an
unpaired tool_use the API rejects outright.

Nothing writes the variant yet."
```

---

### Task 6: `StoredMessage` に `id` を足す

**Files:**
- Modify: `src/session.rs:85-146`（`StoredMessage` と `from_chat`）
- Modify: `src/session.rs:652-675`（`append_report`）
- Test: `src/session.rs`

**Interfaces:**
- Produces: `StoredMessage.id: Option<Uuid>` — 新規書き込みは常に `Some(Uuid::now_v7())`

- [ ] **Step 1: 失敗するテストを書く**

`src/session.rs` の `mod tests` に追加:

```rust
    /// Every newly written message carries an id. The compaction
    /// checkpoint points at one, so a message without one cannot be a
    /// cursor.
    #[test]
    fn a_new_stored_message_gets_an_id() {
        let stored = StoredMessage::from_chat(&ChatMessage::user("hi"));
        assert!(stored.id.is_some(), "from_chat must stamp an id");
    }

    /// Distinct messages get distinct ids. The checkpoint looks its
    /// cursor up by position, so what it needs is uniqueness, not order —
    /// file order is what orders a session, here and in the future
    /// `parent` migration (decision 3.1).
    #[test]
    fn each_message_gets_its_own_id() {
        let a = StoredMessage::from_chat(&ChatMessage::user("first"));
        let b = StoredMessage::from_chat(&ChatMessage::user("second"));
        assert_ne!(a.id.unwrap(), b.id.unwrap());
    }

    /// Legacy JSONL predates the field and must still load.
    #[test]
    fn a_stored_message_without_an_id_deserializes_as_none() {
        let legacy = r#"{"timestamp":"2026-04-08T11:30:22.372570890Z","role":"user","parts":[{"Text":"hello"}]}"#;
        let msg: StoredMessage = serde_json::from_str(legacy).expect("legacy JSONL parses");
        assert!(msg.id.is_none());
    }
```

- [ ] **Step 2: テストが失敗することを確認**

Run: `cargo test --lib session::tests::a_new_stored_message_gets_an_id`
Expected: コンパイルエラー — `StoredMessage` に `id` フィールドが無い

- [ ] **Step 3: フィールドを足す**

`src/session.rs:86` の `StoredMessage` に、`timestamp` の**前**に:

```rust
    /// Stable identity for this line, generated at write time.
    ///
    /// The compaction checkpoint (`SummaryLine::covers_through`) points
    /// at one of these. A timestamp would be the obvious cursor and is a
    /// worse one: coarse system clocks repeat a value across two rapid
    /// appends, and an NTP step backwards makes timestamps non-monotonic,
    /// either of which silently drops messages from a replay.
    ///
    /// `None` on lines written before this field existed; readers fall
    /// back to file order there. File order is what orders a session —
    /// this is an identity, not a sort key, and no reader compares two
    /// of them.
    ///
    /// Deliberately no `parent`. Unlike the ACP store, this format keeps
    /// one file per session, so file order survives as a reconstruction
    /// hint — a file written before `parent` existed cannot contain a
    /// fork, because a fork needs two writers that both record one. See
    /// decision 3.1 of the design doc.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub id: Option<Uuid>,
```

`StoredMessage::from_chat`（`src/session.rs:127`）に `id: Some(Uuid::now_v7()),` を足す。

`append_report`（`src/session.rs:658`）の構造体リテラルにも `id: Some(Uuid::now_v7()),` を足す。

- [ ] **Step 4: 既存の構造体リテラルを直す**

Run: `cargo check --workspace 2>&1 | grep -B 2 "missing field \`id\`"`
テスト内の `StoredMessage { .. }` リテラルに `id: None,` を足していく
（`src/session.rs` の `mod tests` 内に 3 箇所、`src/acp_session.rs:848` に 1 箇所）。

`src/acp_session.rs:848` の `sessions_for_day` が組み立てる `StoredMessage` は
日次ログ用の投影なので `id: None` でよい — ファイル上の行ではない。

- [ ] **Step 5: テストが通ることを確認**

Run: `cargo test --workspace`
Expected: PASS

- [ ] **Step 6: コミット**

```bash
git add src/session.rs src/acp_session.rs
git commit -m "feat(sessions): give each stored message a UUIDv7 id

The compaction checkpoint needs a stable cursor into a session file and
a timestamp is a poor one — a coarse clock repeats a value across two
rapid appends, and an NTP step backwards makes them non-monotonic. Both
silently drop messages from a replay.

No parent. This format keeps one file per session, so file order remains
a valid reconstruction hint if a chain is ever wanted; a file written
before parent existed cannot hold a fork, since a fork requires two
writers that both record one."
```

---

### Task 7: `SessionStore` がツール結果をキャッシュに逃がす

**Files:**
- Modify: `src/session.rs:199-244`（構造体と 2 つのコンストラクタ）
- Modify: `src/session.rs:383-396`（`append`）
- Modify: `src/session.rs:585-610`（`load_session` / `load_session_full`）
- Modify: `src/session.rs:941-989`（`scrub_images_for_storage` → `scrub_for_storage`）
- Test: `src/session.rs`

**Interfaces:**
- Produces:
  - `SessionStore::new(base_dir: PathBuf, kind: &'static str, tool_results: Option<Arc<ToolResultCache>>) -> Self`
  - `SessionStore::with_workspace(base_dir: PathBuf, kind: &'static str, ws_state: Arc<Mutex<WorkspaceState>>, tool_results: Option<Arc<ToolResultCache>>) -> Self`
  - `SessionStore::hydrate(&self, msgs: Vec<ChatMessage>) -> Vec<ChatMessage>`（`pub(crate)`）

**設計からの微修正**: spec は「`new` は cache 無しのまま残す」としていたが、`new` はテスト専用の
コンストラクタなので、そこから cache を渡せないとキャッシュの往復をテストで固定できない。
両方に引数を足す。本番側（`with_workspace`、main.rs の 4 箇所）のコンパイル強制は変わらない。

- [ ] **Step 1: 失敗するテストを書く**

`src/session.rs` の `mod tests` に追加:

```rust
    // ── ツール結果の永続化 (#194) ────────────────────────────────────

    fn cached_store() -> (
        tempfile::TempDir,
        tempfile::TempDir,
        SessionStore,
        String,
    ) {
        let sessions = tempfile::TempDir::new().unwrap();
        let cache_dir = tempfile::TempDir::new().unwrap();
        let cache =
            crate::tool_result_cache::ToolResultCache::open(cache_dir.path().to_path_buf()).unwrap();
        let store = SessionStore::new(sessions.path().to_path_buf(), "channel", Some(cache));
        let key = ("!room:example.org".to_string(), None);
        let sid = store.create_session(&key, "matrix", "default").unwrap();
        (sessions, cache_dir, store, sid)
    }

    fn tool_use_msg(id: &str) -> ChatMessage {
        ChatMessage {
            role: Role::Assistant,
            parts: vec![ContentPart::ToolUse {
                id: id.to_string(),
                name: "file_read".to_string(),
                input: serde_json::json!({ "path": "a.rs" }),
            }],
            input_kind: None,
            user_id: None,
        }
    }

    fn tool_result_msg(id: &str, content: &str) -> ChatMessage {
        ChatMessage {
            role: Role::User,
            parts: vec![ContentPart::ToolResult {
                tool_use_id: id.to_string(),
                content: content.to_string(),
            }],
            input_kind: None,
            user_id: None,
        }
    }

    /// The whole point: what the agent did survives a reload.
    #[test]
    fn a_tool_result_round_trips_through_the_cache() {
        let (_s, _c, store, sid) = cached_store();
        store.append(&sid, &tool_use_msg("c1")).unwrap();
        store.append(&sid, &tool_result_msg("c1", "fn main() {}")).unwrap();

        let loaded = store.load_session(&sid).expect("the session loads");
        let content = loaded.iter().flat_map(|m| &m.parts).find_map(|p| match p {
            ContentPart::ToolResult { tool_use_id, content } if tool_use_id == "c1" => {
                Some(content.clone())
            }
            _ => None,
        });
        assert_eq!(content.as_deref(), Some("fn main() {}"));
    }

    /// The content must not be in the JSONL — that file is inside the
    /// retrieve index, which is the whole reason for the cache.
    #[test]
    fn the_result_content_never_reaches_the_session_file() {
        let (_s, _c, store, sid) = cached_store();
        store.append(&sid, &tool_use_msg("c1")).unwrap();
        store
            .append(&sid, &tool_result_msg("c1", "SECRET-CANARY-VALUE"))
            .unwrap();

        let raw = fs::read_to_string(store.absolute_path_for(&sid).unwrap()).unwrap();
        assert!(
            !raw.contains("SECRET-CANARY-VALUE"),
            "tool result content leaked into the indexed file:\n{raw}"
        );
        assert!(raw.contains("ToolResultRef"), "expected a reference:\n{raw}");
    }

    /// An evicted result degrades to a placeholder. The pairing is what
    /// the API validates, so this must load rather than fail.
    #[test]
    fn an_evicted_result_becomes_a_placeholder() {
        let (_s, cache_dir, store, sid) = cached_store();
        store.append(&sid, &tool_use_msg("c1")).unwrap();
        store.append(&sid, &tool_result_msg("c1", "gone soon")).unwrap();

        for entry in fs::read_dir(cache_dir.path()).unwrap().flatten() {
            fs::remove_file(entry.path()).unwrap();
        }

        let loaded = store.load_session(&sid).expect("the session still loads");
        let has_placeholder = loaded.iter().flat_map(|m| &m.parts).any(
            |p| matches!(p, ContentPart::ToolResult { tool_use_id, content } if tool_use_id == "c1" && content == crate::session_storage::MISSING_RESULT),
        );
        assert!(has_placeholder, "expected a placeholder: {loaded:?}");
    }

    /// No cache at write time records the pairing with no hash. Writing
    /// nothing would leave a tool_use the API rejects.
    #[test]
    fn no_cache_at_write_time_still_keeps_the_pairing() {
        let sessions = tempfile::TempDir::new().unwrap();
        let store = SessionStore::new(sessions.path().to_path_buf(), "channel", None);
        let key = ("!room:example.org".to_string(), None);
        let sid = store.create_session(&key, "matrix", "default").unwrap();

        store.append(&sid, &tool_use_msg("c1")).unwrap();
        store.append(&sid, &tool_result_msg("c1", "lost")).unwrap();

        let loaded = store.load_session(&sid).expect("the session loads");
        let answered = loaded.iter().flat_map(|m| &m.parts).any(
            |p| matches!(p, ContentPart::ToolResult { tool_use_id, .. } if tool_use_id == "c1"),
        );
        assert!(answered, "the tool_use must still be answered: {loaded:?}");
    }

    /// An oversized tool input is elided rather than written whole — the
    /// session file is indexed and there is no cache indirection for it.
    #[test]
    fn an_oversized_tool_input_is_elided_on_disk() {
        let (_s, _c, store, sid) = cached_store();
        let huge = "x".repeat(crate::tools::OUTPUT_CAP_BYTES + 1);
        store
            .append(
                &sid,
                &ChatMessage {
                    role: Role::Assistant,
                    parts: vec![ContentPart::ToolUse {
                        id: "c1".to_string(),
                        name: "file_write".to_string(),
                        input: serde_json::json!({ "content": huge }),
                    }],
                    input_kind: None,
                    user_id: None,
                },
            )
            .unwrap();

        let raw = fs::read_to_string(store.absolute_path_for(&sid).unwrap()).unwrap();
        assert!(raw.contains("_elided"), "expected elision:\n{}", &raw[..raw.len().min(400)]);
        assert!(raw.len() < 4000, "the line was written whole ({} bytes)", raw.len());
    }

    /// The daily log is a permanent, searchable record. A placeholder
    /// sentence from an evicted result has no business in it.
    #[test]
    fn the_daily_log_projection_carries_no_tool_traffic() {
        let (_s, _c, store, sid) = cached_store();
        store.append(&sid, &ChatMessage::user("what is in a.rs")).unwrap();
        store.append(&sid, &tool_use_msg("c1")).unwrap();
        store.append(&sid, &tool_result_msg("c1", "fn main() {}")).unwrap();

        let today = local_date_for_timestamp(Local::now(), 4);
        let days = store.sessions_for_day(today, 4);
        let parts: Vec<&ContentPart> =
            days.iter().flat_map(|(_, ms)| ms).flat_map(|m| &m.parts).collect();
        assert!(
            parts.iter().all(|p| matches!(p, ContentPart::Text(_) | ContentPart::ToolUse { .. } | ContentPart::ToolResultRef { .. })),
            "no hydrated tool result may appear: {parts:?}"
        );
        assert!(
            !parts.iter().any(|p| matches!(p, ContentPart::ToolResult { .. })),
            "sessions_for_day must not hydrate: {parts:?}"
        );
    }
```

- [ ] **Step 2: テストが失敗することを確認**

Run: `cargo test --lib session::tests::a_tool_result_round_trips`
Expected: コンパイルエラー — `SessionStore::new` は引数を 2 つしか取らない

- [ ] **Step 3: 構造体とコンストラクタ**

`src/session.rs:199` の `SessionStore` に、`ws_state` の後に:

```rust
    /// Workspace-external, content-addressed store for tool results.
    ///
    /// `None` when the platform cache directory could not be opened at
    /// startup. Degrades rather than failing: a result written without a
    /// cache is recorded as a `ToolResultRef` with no hash, which reads
    /// back as `MISSING_RESULT`. Losing the content is survivable;
    /// losing the pairing is not.
    tool_results: Option<Arc<crate::tool_result_cache::ToolResultCache>>,
```

`new` と `with_workspace` の両方に
`tool_results: Option<Arc<crate::tool_result_cache::ToolResultCache>>` を末尾引数として足し、
構造体リテラルに `tool_results,` を足す。`new` から `#[allow(dead_code)]` は外さない。

- [ ] **Step 4: `scrub_for_storage` を書く**

`src/session.rs:958` の `scrub_images_for_storage` を、`SessionStore` のメソッドに置き換える
（キャッシュが要るので自由関数ではいられなくなる）。フリー関数版は削除し、`impl SessionStore` に:

```rust
    /// What a message looks like on disk.
    ///
    /// Three transformations, all storage-path-only — the in-memory
    /// history keeps the full values for the provider call:
    ///
    /// - a tool result's content goes to the cache and the line keeps a
    ///   hash, because `<workspace>/sessions` is inside the retrieve
    ///   index and a day of file reads would both bloat the workspace
    ///   and skew every search that touches it
    /// - an oversized tool *input* is elided, for the same reason and
    ///   with no cache to escape into
    /// - an image becomes a text marker carrying its hash
    ///
    /// Returns `None` when nothing needs rewriting, so the common
    /// text-only append skips the allocation.
    fn scrub_for_storage(&self, msg: &ChatMessage) -> Option<ChatMessage> {
        let needs_work = msg.parts.iter().any(|p| {
            matches!(
                p,
                ContentPart::Image { .. }
                    | ContentPart::ToolResult { .. }
                    | ContentPart::ToolUse { .. }
            )
        });
        if !needs_work {
            return None;
        }
        let parts = msg
            .parts
            .iter()
            .map(|p| match p {
                ContentPart::Image {
                    media_type,
                    data_base64,
                } => {
                    let hash = match BASE64_STANDARD.decode(data_base64) {
                        Ok(bytes) => sha256_hex(&bytes),
                        Err(_) => "invalid-base64".to_string(),
                    };
                    ContentPart::Text(format!("[image: {media_type} sha256={hash}]"))
                }
                ContentPart::ToolUse { id, name, input } => ContentPart::ToolUse {
                    id: id.clone(),
                    name: name.clone(),
                    input: crate::session_storage::elide_oversized_input(input),
                },
                ContentPart::ToolResult {
                    tool_use_id,
                    content,
                } => {
                    // A missing cache must not degrade any further than
                    // a lost hash does. Dropping the part would leave a
                    // `tool_use` with no matching `tool_result`, which
                    // the API rejects outright — the session would fail
                    // to load rather than load thinner.
                    let sha256 = match &self.tool_results {
                        Some(cache) => match cache.put(content) {
                            Ok(sha) => Some(sha),
                            Err(e) => {
                                warn!("Failed to cache tool result '{tool_use_id}': {e}");
                                None
                            }
                        },
                        None => None,
                    };
                    ContentPart::ToolResultRef {
                        tool_use_id: tool_use_id.clone(),
                        sha256,
                    }
                }
                other => other.clone(),
            })
            .collect();
        Some(ChatMessage {
            role: msg.role.clone(),
            parts,
            input_kind: msg.input_kind.clone(),
            user_id: msg.user_id.clone(),
        })
    }

    /// Turn every `ToolResultRef` back into the result the model needs
    /// to see. A miss is a placeholder, not an error.
    pub(crate) fn hydrate(&self, msgs: Vec<ChatMessage>) -> Vec<ChatMessage> {
        msgs.into_iter()
            .map(|m| ChatMessage {
                parts: m
                    .parts
                    .iter()
                    .map(|p| match p {
                        ContentPart::ToolResultRef {
                            tool_use_id,
                            sha256,
                        } => ContentPart::ToolResult {
                            tool_use_id: tool_use_id.clone(),
                            content: sha256
                                .as_ref()
                                .and_then(|s| self.tool_results.as_ref()?.get(s))
                                .unwrap_or_else(|| {
                                    crate::session_storage::MISSING_RESULT.to_string()
                                }),
                        },
                        other => other.clone(),
                    })
                    .collect(),
                ..m
            })
            .collect()
    }
```

`src/session.rs:384` の `append` を直す:

```rust
        let scrubbed = self.scrub_for_storage(msg);
        let to_store = scrubbed.as_ref().unwrap_or(msg);
```

- [ ] **Step 5: 読み取り側で hydrate する**

`load_session`（`src/session.rs:585`）:

```rust
    pub fn load_session(&self, session_id: &str) -> Option<Vec<ChatMessage>> {
        let path = self.resolve_path(session_id)?;
        let (_, messages, _, _) = load_session_file(&path)?;
        let chat: Vec<ChatMessage> = messages
            .into_iter()
            .map(|m| m.into_chat_message())
            .collect();
        Some(crate::session_storage::repair_tool_pairing(
            self.hydrate(chat),
        ))
    }
```

`load_session_full`（`src/session.rs:603`）は `StoredMessage` を返すので、`parts` だけ
hydrate する。`recall_memory` は要約と直近レポートを読む経路で、対の修復は不要
（レポートに tool 呼び出しは無い）:

```rust
    pub fn load_session_full(
        &self,
        session_id: &str,
    ) -> Option<(Vec<StoredMessage>, Option<SummaryLine>)> {
        let path = self.resolve_path(session_id)?;
        let (_, messages, _, summary) = load_session_file(&path)?;
        let hydrated = messages
            .into_iter()
            .map(|mut m| {
                let one = self.hydrate(vec![ChatMessage {
                    role: m.role.clone(),
                    parts: std::mem::take(&mut m.parts),
                    input_kind: None,
                    user_id: None,
                }]);
                m.parts = one.into_iter().next().map(|c| c.parts).unwrap_or_default();
                m
            })
            .collect();
        Some((hydrated, summary))
    }
```

`sessions_for_day` と `sessions_for_day_filtered` は**変更しない** — hydrate しないことが
仕様である。

- [ ] **Step 6: 既存の呼び出し元とテストを直す**

Run: `cargo check --workspace 2>&1 | head -40`

`SessionStore::new` / `with_workspace` の全呼び出し元に引数を足す:
`src/periodic_log.rs:1731, 1794`、`src/serve/mod.rs:3555-3561`、`src/session.rs:1343`、
`src/main.rs:465, 475, 486, 723`（main.rs は Task 8 で本物を渡すので、ここでは一旦 `None`）。

`scrub_images_for_storage` を直接呼んでいた既存テスト 4 本
（`src/session.rs:1176-1247`）は `store.scrub_for_storage(&msg)` を使う形に書き換える。

- [ ] **Step 7: テストが通ることを確認**

Run: `cargo +nightly fmt --all && cargo clippy --workspace -- -D warnings && cargo test --workspace`
Expected: PASS

- [ ] **Step 8: コミット**

```bash
git add src/session.rs src/periodic_log.rs src/serve/mod.rs src/main.rs
git commit -m "feat(sessions): keep tool results out of the workspace, and load them back

SessionStore now takes the same content-addressed cache the ACP store
uses. A tool result's content goes there and the JSONL keeps a
ToolResultRef; an oversized tool input is elided, since there is no
cache indirection for an input and the file is inside the retrieve
index either way.

Hydration is per read path, not inside the parser, because the paths
disagree about what they want. load_session and load_session_full
hydrate. sessions_for_day does not, and must not: the daily log is a
permanent searchable record, and a placeholder sentence from an evicted
result has no business in it. That falls out for free — format_sessions
keeps text parts only, so an un-hydrated ref drops itself.

Nothing calls append with tool traffic yet."
```

---

### Task 8: 全トランスポートがツール呼び出しを永続化する

**Files:**
- Modify: `src/main.rs:437-490, 620-661`（キャッシュ生成の前倒しと 4 ストアへの配線）
- Modify: `src/serve/mod.rs:2203-2265`（`TurnPersistence`）
- Modify: `src/serve/mod.rs:2443-2455, 2658-2672`（2 つの skip 箇所）
- Modify: `src/agent.rs:507-553`（`persist`）
- Modify: `src/agent.rs:929-1035`（対のゲート）

**Interfaces:**
- Produces: `TurnPersistence::append_message_paired(&self, msg: &ChatMessage) -> bool`（`append_acp_only` を置き換え）
- Produces: `Agent::persist(&self, session_id: &str, msg: &ChatMessage) -> bool`（戻り値が増える）

- [ ] **Step 1: `main.rs` でキャッシュ生成を前倒しする**

`src/main.rs:620-657` の `tool_result_cache` ブロック（コメント込み）を切り取り、
`src/main.rs:448`（`let sessions_base = ...`）の**直後**に貼る。doc コメントの
「the request path does not persist tool_use/tool_result messages at all yet (#191)」
という一文はもう嘘なので、次に差し替える:

```rust
            // Every session store persists tool traffic now, so this
            // cache is on the request path of every transport — not just
            // ACP. `None` still degrades rather than aborting startup: a
            // result written without it keeps its pairing and reads back
            // as a placeholder, which is a thinner session rather than an
            // unloadable one.
```

- [ ] **Step 2: 4 ストアにキャッシュを渡す**

`src/main.rs:465, 475, 486` の 3 つの `with_workspace` 呼び出しと、
`src/main.rs:723` の `channel_session_store` に、末尾引数として
`tool_result_cache.clone()` を渡す。

`src/main.rs:658` の `AcpSessionStore::new` は `tool_result_cache.clone()` に変える
（今までは move していた）。

- [ ] **Step 3: `TurnPersistence` から `is_acp` 分岐を外す**

`src/serve/mod.rs:2226-2252` の `append_acp_only` を置き換える:

```rust
    /// Append a `tool_use` or `tool_result` message. Returns whether the
    /// caller may go on to persist a message that must be paired with
    /// this one.
    ///
    /// `false` only when the append was attempted and failed. Every
    /// store persists tool traffic now (#194), so there is no longer a
    /// transport that skips this — what used to be the `is_acp` branch
    /// here always returned `true` for the four `SessionStore` kinds
    /// precisely because they wrote nothing at all.
    fn append_message_paired(&self, msg: &ChatMessage) -> bool {
        let result = if self.is_acp {
            self.acp_store.append_message(&self.session_id, msg)
        } else {
            self.store.append(&self.session_id, msg)
        };
        match result {
            Ok(()) => true,
            Err(e) => {
                warn!("Failed to persist a message: {e}");
                false
            }
        }
    }
```

呼び出し元 2 箇所（`src/serve/mod.rs:2455` と `:2671`）を `append_message_paired` に変え、
`:2443-2448` のコメントを差し替える:

```rust
                    // Whether this append landed is captured and carried
                    // down to the `tool_result` append below: the two
                    // must not be skipped independently of each other
                    // (see the comment there for why).
```

- [ ] **Step 4: `Agent::persist` からストリップを外し、戻り値を足す**

`src/agent.rs:507-553` を置き換える:

```rust
    /// Persist `msg` to the session store. Returns whether it landed, so
    /// a caller holding the other half of a `tool_use` / `tool_result`
    /// pair can decline to write it alone.
    ///
    /// Tool parts used to be stripped here — raw history was never
    /// reloaded, so persisting them would only have bloated the JSONL.
    /// It is reloaded now (#194), and a history that remembers what was
    /// said but not what was done is the gap this exists to close.
    ///
    /// The storage shape is decided inside [`SessionStore::append`]:
    /// tool results go to the workspace-external cache, oversized tool
    /// inputs are elided, images become hash markers. Every persist path
    /// (agent, `/rpc`, `/a2a`, MCP) inherits it.
    ///
    /// A no-op returns `true`: there is no pairing to break when nothing
    /// was ever a candidate for the store.
    fn persist(&self, session_id: &str, msg: &ChatMessage) -> bool {
        if session_id.is_empty() {
            return true;
        }
        let has_content = msg.parts.iter().any(|p| match p {
            ContentPart::Text(t) => !t.is_empty(),
            _ => true,
        });
        if !has_content {
            return true;
        }
        match self.session_store.append(session_id, msg) {
            Ok(()) => true,
            Err(e) => {
                warn!("Failed to persist message: {e}");
                false
            }
        }
    }
```

- [ ] **Step 5: チャンネルの対のゲート**

`src/agent.rs:942` を:

```rust
                    // Carried down to the tool_result append below. A
                    // half-persisted pair is worse than neither: the
                    // in-memory history is correct either way and gets
                    // rebuilt next turn, but a tool_result alone on disk
                    // chains onto the preceding user message and bricks
                    // the session file for good.
                    let tool_use_persisted = self.persist(&session_id, &msg);
```

`src/agent.rs:1034` を:

```rust
                    if tool_use_persisted {
                        self.persist(&session_id, &msg);
                    }
```

`src/agent.rs:792` と `:923` の `self.persist(...)` は戻り値を捨てるので
`let _ = self.persist(...);` にする。

- [ ] **Step 6: 手動確認 — 実際に Discord/Matrix なしで往復を見る**

`src/agent.rs` の `mod tests` が無いため、この経路は `SessionStore` 側のテスト
（Task 7）と、`serve/mod.rs` の既存 ACP テストで担保する。追加で
`src/serve/mod.rs` の `mod tests` に非 ACP の往復を 1 本足す:

```rust
    /// The four SessionStore kinds persist tool traffic now, so a /rpc
    /// session's tool_use and tool_result both reach disk (#194).
    #[test]
    fn a_non_acp_session_persists_both_halves_of_a_tool_call() {
        let base = tempfile::TempDir::new().unwrap();
        let cache_dir = tempfile::TempDir::new().unwrap();
        let cache =
            crate::tool_result_cache::ToolResultCache::open(cache_dir.path().to_path_buf()).unwrap();
        let store = SessionStore::new(base.path().join("sessions"), "rpc", Some(cache));
        let key = ("s1".to_string(), None);
        store.ensure_session("s1", &key, "rpc", None, "default").unwrap();

        store
            .append(
                "s1",
                &ChatMessage::assistant_with_tools(
                    None,
                    vec![crate::provider::ToolCall {
                        id: "c1".to_string(),
                        name: "file_read".to_string(),
                        input: serde_json::json!({}),
                    }],
                ),
            )
            .unwrap();
        store
            .append(
                "s1",
                &ChatMessage::tool_results_with_images(
                    vec![("c1".to_string(), "contents".to_string())],
                    Vec::new(),
                ),
            )
            .unwrap();

        let loaded = store.load_session("s1").expect("the session loads");
        assert!(
            loaded.iter().flat_map(|m| &m.parts).any(|p| matches!(p, ContentPart::ToolUse { id, .. } if id == "c1")),
            "tool_use missing: {loaded:?}"
        );
        assert!(
            loaded.iter().flat_map(|m| &m.parts).any(|p| matches!(p, ContentPart::ToolResult { tool_use_id, content } if tool_use_id == "c1" && content == "contents")),
            "tool_result missing: {loaded:?}"
        );
    }
```

`ToolCall` は `src/provider/mod.rs:25` で `{ id: String, name: String, input: Value }`。

- [ ] **Step 7: 全テスト**

Run: `cargo +nightly fmt --all && cargo clippy --workspace -- -D warnings && cargo test --workspace`
Expected: PASS

- [ ] **Step 8: コミット**

```bash
git add src/main.rs src/serve/mod.rs src/agent.rs
git commit -m "feat(sessions): persist tool calls on every transport

Closes #194.

Being told to run a tool from Discord or Matrix is ordinary use of an
agent, and until now such a session came back remembering what was said
but not what it did. That was the same loss ACP had before #191, on the
transports where the day's conversation actually happens.

The two skip sites in run_llm_turn lose their is_acp branch, and so does
TurnPersistence's own method — the branch existed only because the four
SessionStore kinds wrote nothing at all. One rule for every transport
instead of two.

Agent::persist stops stripping tool parts and starts reporting whether
the write landed, so the channel path gets the same all-or-nothing
pairing gate the ACP path has had. A tool_result alone on disk chains
onto the preceding user message and bricks the file permanently; a
half-persisted pair is worse than neither."
```

---

## Phase 4 — チェックポイントからの復元

### Task 9: 復元スタブと `keep_recent` を用意する

**Files:**
- Modify: `src/context_compression.rs:64-134`
- Modify: `src/agent.rs:657-671`（境界 compaction のスタブ）
- Test: `src/context_compression.rs`

**Interfaces:**
- Produces:
  - `pub fn compaction_stub(summary: &str) -> Vec<ChatMessage>`
  - `CompressionResult { compressed, summary, keep_recent: usize }`

- [ ] **Step 1: 失敗するテストを書く**

`src/context_compression.rs` の `mod tests` に追加:

```rust
    /// The stub has one generator so the compaction path and the restore
    /// path cannot drift into producing different shapes for the same
    /// thing.
    #[test]
    fn the_stub_is_a_user_message_carrying_the_summary_and_an_assistant_ack() {
        let stub = compaction_stub("we fixed the parser");
        assert_eq!(stub.len(), 2);
        assert_eq!(stub[0].role, Role::User);
        assert_eq!(stub[1].role, Role::Assistant);
        assert!(
            matches!(&stub[0].parts[0], ContentPart::Text(t) if t.contains("we fixed the parser")),
            "the summary must be in the user message: {:?}", stub[0]
        );
    }
```

- [ ] **Step 2: テストが失敗することを確認**

Run: `cargo test --lib context_compression::tests::the_stub_is_a_user_message`
Expected: コンパイルエラー — `compaction_stub` が存在しない

- [ ] **Step 3: `compaction_stub` を実装し、`maybe_compress` から使う**

`src/context_compression.rs`、`CompressionResult` の直前に:

```rust
/// A compaction summary rendered back into the conversation.
///
/// One generator, three callers: `maybe_compress`, the day-boundary
/// compaction in `Agent`, and every store's restore path. They used to
/// have two wordings between them, and a restore has no way to know
/// which one produced the summary it is reading — so there is one.
///
/// The wording is not load-bearing; being the same everywhere is.
pub fn compaction_stub(summary: &str) -> Vec<ChatMessage> {
    vec![
        ChatMessage {
            role: Role::User,
            parts: vec![ContentPart::Text(format!(
                "[Context Summary — earlier messages were compressed]\n\n{summary}"
            ))],
            input_kind: None,
            user_id: None,
        },
        ChatMessage::assistant("Understood. I have the context from our earlier conversation."),
    ]
}
```

`CompressionResult` に足す:

```rust
pub struct CompressionResult {
    pub compressed: Vec<ChatMessage>,
    pub summary: String,
    /// How many trailing messages survived verbatim. The store turns
    /// this into a checkpoint cursor by counting back from its file's
    /// tip — the caller has no way to map an in-memory index onto a line
    /// number, and should not have to.
    pub keep_recent: usize,
}
```

`maybe_compress`（`src/context_compression.rs:116-133`）を:

```rust
    let mut compressed = compaction_stub(&summary);
    compressed.extend_from_slice(to_keep);

    Ok(Some(CompressionResult {
        compressed,
        summary,
        keep_recent: to_keep.len(),
    }))
```

- [ ] **Step 4: 境界 compaction を同じスタブにする**

`src/agent.rs:657-669` の `stub` を:

```rust
        let stub = crate::context_compression::compaction_stub(&summary);
```

- [ ] **Step 5: テストが通ることを確認**

Run: `cargo test --workspace`
Expected: PASS

- [ ] **Step 6: コミット**

```bash
git add src/context_compression.rs src/agent.rs
git commit -m "refactor(sessions): one generator for the compaction stub, and report keep_recent

The stub had two wordings — one in maybe_compress, one in the
day-boundary compaction — and the restore path about to read these
summaries back has no way to tell which produced the one it is holding.
The wording is not load-bearing; being the same everywhere is.

CompressionResult now reports how many trailing messages survived, which
is what a store needs to turn a summary into a checkpoint cursor by
counting back from its own file's tip."
```

---

### Task 10: `SessionStore` のチェックポイント

**Files:**
- Modify: `src/session.rs:165-175`（`SummaryLine`）
- Modify: `src/session.rs:398-413`（`append_summary`）
- Modify: `src/session.rs:1080-1138`（`load_session_file`）
- Modify: `src/session.rs:585-594`（`load_session`）
- Modify: `src/agent.rs:874-879`、`src/serve/mod.rs:2258-2264`（呼び出し側）
- Test: `src/session.rs`

**Interfaces:**
- Produces:
  - `SummaryLine.covers_through: Option<Uuid>`
  - `SessionStore::append_summary(&self, session_id: &str, summary: &str, keep_recent: usize) -> anyhow::Result<()>`

- [ ] **Step 1: 失敗するテストを書く**

`src/session.rs` の `mod tests` に追加:

```rust
    // ── compaction チェックポイント ──────────────────────────────────

    /// The checkpoint covers everything but the trailing `keep_recent`
    /// messages, so a restore reproduces the state the process had when
    /// it went down rather than replaying the whole file.
    #[test]
    fn a_restore_replays_the_summary_and_the_kept_tail() {
        let (_s, _c, store, sid) = cached_store();
        for i in 0..5 {
            store.append(&sid, &ChatMessage::user(&format!("m{i}"))).unwrap();
        }
        store.append_summary(&sid, "the first three", 2).unwrap();

        let loaded = store.load_session(&sid).expect("the session loads");
        let texts: Vec<String> = loaded
            .iter()
            .flat_map(|m| &m.parts)
            .filter_map(|p| match p {
                ContentPart::Text(t) => Some(t.clone()),
                _ => None,
            })
            .collect();

        assert!(texts[0].contains("the first three"), "stub first: {texts:?}");
        assert!(texts.iter().any(|t| t == "m3"), "kept tail missing: {texts:?}");
        assert!(texts.iter().any(|t| t == "m4"), "kept tail missing: {texts:?}");
        assert!(!texts.iter().any(|t| t == "m0"), "covered message replayed: {texts:?}");
        assert!(!texts.iter().any(|t| t == "m2"), "covered message replayed: {texts:?}");
    }

    /// A file with no summary replays whole — there is no checkpoint to
    /// start from.
    #[test]
    fn a_file_without_a_summary_replays_everything() {
        let (_s, _c, store, sid) = cached_store();
        store.append(&sid, &ChatMessage::user("only one")).unwrap();
        let loaded = store.load_session(&sid).unwrap();
        assert_eq!(loaded.len(), 1);
    }

    /// A SummaryLine written before covers_through existed means "this
    /// summary stands in for everything before it in the file" — which
    /// is what the shutdown summary it almost certainly is did mean.
    #[test]
    fn a_legacy_summary_line_replays_only_what_follows_it() {
        let (_s, _c, store, sid) = cached_store();
        store.append(&sid, &ChatMessage::user("before")).unwrap();

        let path = store.absolute_path_for(&sid).unwrap();
        let legacy = r#"{"summary_at":"2026-04-06T11:00:00Z","summary":"what happened"}"#;
        let mut f = OpenOptions::new().append(true).open(&path).unwrap();
        writeln!(f, "{legacy}").unwrap();
        drop(f);

        store.append(&sid, &ChatMessage::user("after")).unwrap();

        let loaded = store.load_session(&sid).unwrap();
        let texts: Vec<String> = loaded
            .iter()
            .flat_map(|m| &m.parts)
            .filter_map(|p| match p {
                ContentPart::Text(t) => Some(t.clone()),
                _ => None,
            })
            .collect();
        assert!(texts[0].contains("what happened"), "stub first: {texts:?}");
        assert!(texts.iter().any(|t| t == "after"), "{texts:?}");
        assert!(!texts.iter().any(|t| t == "before"), "{texts:?}");
    }

    /// keep_recent = 0 is the day-boundary compaction: it replaced the
    /// whole in-memory history with a stub, so a restore should too.
    #[test]
    fn a_boundary_compaction_leaves_only_the_stub() {
        let (_s, _c, store, sid) = cached_store();
        for i in 0..3 {
            store.append(&sid, &ChatMessage::user(&format!("m{i}"))).unwrap();
        }
        store.append_summary(&sid, "yesterday", 0).unwrap();

        let loaded = store.load_session(&sid).unwrap();
        assert_eq!(loaded.len(), 2, "stub only: {loaded:?}");
    }

    /// A session spanning the upgrade has messages with no id. The
    /// cursor cannot point at one, so the checkpoint degrades to the
    /// file-order rule rather than pointing at nothing — bounded to a
    /// day under the default Reset policy, which rotates the file.
    #[test]
    fn a_checkpoint_over_id_less_messages_degrades_to_file_order() {
        let (_s, _c, store, sid) = cached_store();
        let path = store.absolute_path_for(&sid).unwrap();
        {
            let mut f = OpenOptions::new().append(true).open(&path).unwrap();
            for i in 0..3 {
                writeln!(
                    f,
                    r#"{{"timestamp":"2026-04-06T10:0{i}:00Z","role":"user","parts":[{{"Text":"old{i}"}}]}}"#
                )
                .unwrap();
            }
        }

        store.append_summary(&sid, "the old ones", 1).unwrap();
        store.append(&sid, &ChatMessage::user("new")).unwrap();

        let loaded = store.load_session(&sid).expect("the session still loads");
        let texts: Vec<String> = loaded
            .iter()
            .flat_map(|m| &m.parts)
            .filter_map(|p| match p {
                ContentPart::Text(t) => Some(t.clone()),
                _ => None,
            })
            .collect();
        assert!(texts[0].contains("the old ones"), "stub first: {texts:?}");
        assert!(texts.iter().any(|t| t == "new"), "{texts:?}");
        assert!(
            !texts.iter().any(|t| t == "old0"),
            "file-order fallback must skip everything before the line: {texts:?}"
        );
    }

    /// The checkpoint must not land between a tool_use and its result.
    /// find_safe_split_point guarantees it on the write side; this pins
    /// that the read side stays loadable if it ever does not.
    #[test]
    fn a_checkpoint_cutting_a_pair_still_loads_paired() {
        let (_s, _c, store, sid) = cached_store();
        store.append(&sid, &tool_use_msg("c1")).unwrap();
        store.append(&sid, &tool_result_msg("c1", "ok")).unwrap();
        // Cover the tool_use, keep only the result — the gap the repair
        // exists for.
        store.append_summary(&sid, "did a thing", 1).unwrap();

        let loaded = store.load_session(&sid).unwrap();
        let stray = loaded.iter().enumerate().any(|(i, m)| {
            m.parts.iter().any(|p| matches!(p, ContentPart::ToolResult { .. }))
                && !loaded
                    .get(i.wrapping_sub(1))
                    .is_some_and(|prev| prev.parts.iter().any(|p| matches!(p, ContentPart::ToolUse { .. })))
        });
        assert!(!stray, "an unpaired tool_result survived: {loaded:?}");
    }
```

- [ ] **Step 2: テストが失敗することを確認**

Run: `cargo test --lib session::tests::a_restore_replays`
Expected: コンパイルエラー — `append_summary` は引数を 2 つしか取らない

- [ ] **Step 3: `SummaryLine` にフィールドを足す**

`src/session.rs:169` の `SummaryLine` に:

```rust
    /// The last message this summary absorbed. A restore replays the
    /// messages *after* it, prefixed with the summary as a stub — which
    /// reproduces the in-memory state at the moment of compaction rather
    /// than replaying a whole session and paying for the compaction all
    /// over again on the first turn back.
    ///
    /// `None` on lines written before this field existed. Those are read
    /// as "this summary stands in for everything before it in file
    /// order", which is what a shutdown summary — almost certainly what
    /// such a line is — actually meant.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub covers_through: Option<Uuid>,
```

- [ ] **Step 4: `append_summary` がカーソルを算出する**

`src/session.rs:398` を置き換える:

```rust
    /// Append a compaction summary, recording how far it reaches.
    ///
    /// `keep_recent` is how many trailing messages the caller kept
    /// verbatim (0 for a day-boundary compaction, which replaces the
    /// whole history). The cursor is computed here rather than passed
    /// in: the caller holds an index into its in-memory history, and
    /// only the store can map that onto its own file, whose message
    /// count is at least as large — earlier compactions trimmed memory,
    /// not the log.
    pub fn append_summary(
        &self,
        session_id: &str,
        summary: &str,
        keep_recent: usize,
    ) -> anyhow::Result<()> {
        let path = self
            .resolve_path(session_id)
            .ok_or_else(|| anyhow::anyhow!("Session file not found for {session_id}"))?;
        let line = serde_json::to_string(&SummaryLine {
            summary_at: Utc::now(),
            summary: summary.to_string(),
            up_to_timestamp: None,
            covers_through: checkpoint_id(&path, keep_recent),
        })?;
        let mut file = OpenOptions::new().create(true).append(true).open(&path)?;
        writeln!(file, "{line}")?;
        drop(file);
        self.notify_updated(&path);
        Ok(())
    }
```

ファイル末尾のヘルパー群（`load_meta_and_latest_intraday_digest` の下）に:

```rust
/// The id of the last message a summary keeping `keep_recent` trailing
/// messages absorbs.
///
/// `None` when the file has no messages, or when the message at that
/// position predates `StoredMessage::id`. The second case costs one
/// checkpoint's worth of tail on a session that spans the upgrade —
/// bounded to a day under the default `Reset` policy, which rotates the
/// file — and the reader's file-order fallback handles it.
fn checkpoint_id(path: &Path, keep_recent: usize) -> Option<Uuid> {
    let ids = message_ids_in_order(path);
    if ids.is_empty() {
        return None;
    }
    // A summary that covers nothing is not reachable from
    // `maybe_compress` (it fires only with at least one message to
    // summarise, and the file holds at least as many as memory did), but
    // clamping costs nothing and keeps the index in range.
    let covered = ids.len().saturating_sub(keep_recent).max(1);
    ids[covered - 1]
}

/// Every message line's id, in file order. Non-message lines are skipped
/// so the positions line up with what a reader replays.
fn message_ids_in_order(path: &Path) -> Vec<Option<Uuid>> {
    let Ok(file) = fs::File::open(path) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for raw in BufReader::new(file).lines().map_while(Result::ok) {
        let raw = raw.trim();
        if raw.is_empty() {
            continue;
        }
        let Ok(value) = serde_json::from_str::<serde_json::Value>(raw) else {
            continue;
        };
        if value.get("timestamp").is_none() {
            continue;
        }
        out.push(
            value
                .get("id")
                .and_then(|v| v.as_str())
                .and_then(|s| Uuid::parse_str(s).ok()),
        );
    }
    out
}
```

- [ ] **Step 5: 読み取り側でチェックポイントを使う**

`load_session_file` の戻り値に「要約行を読んだ時点のメッセージ件数」を足す。5 要素の
タプルは読みにくいので構造体にする。`src/session.rs:1080` の直前に:

```rust
/// One parsed session file.
struct LoadedSession {
    meta: SessionMeta,
    messages: Vec<StoredMessage>,
    is_closed: bool,
    latest_summary: Option<SummaryLine>,
    /// `messages.len()` at the moment the latest summary line was read.
    /// The file-order fallback for a summary with no `covers_through`.
    messages_before_summary: usize,
}
```

`load_session_file` の戻り値を `Option<LoadedSession>` に変え、
`summary_at` のアームで `messages_before_summary = messages.len();` を記録する。
既存の呼び出し元（タプル分解している 8 箇所）はフィールドアクセスに書き換える。

`load_session_file` の下に:

```rust
/// The messages a restore should replay, and the summary to prefix them
/// with.
///
/// Everything before the checkpoint is what the summary already says;
/// replaying it too would re-send a conversation the running process had
/// already compacted away, and pay for compacting it again on the first
/// turn back.
fn model_history(loaded: &LoadedSession) -> (Option<&str>, &[StoredMessage]) {
    let Some(summary) = &loaded.latest_summary else {
        return (None, &loaded.messages);
    };
    let start = match summary.covers_through {
        Some(id) => match loaded.messages.iter().position(|m| m.id == Some(id)) {
            Some(pos) => pos + 1,
            None => {
                warn!(
                    "session checkpoint {id} is not in the file; replaying the whole session"
                );
                0
            }
        },
        None => loaded.messages_before_summary,
    };
    (
        Some(summary.summary.as_str()),
        loaded.messages.get(start..).unwrap_or(&[]),
    )
}
```

`load_session`（`src/session.rs:585`）を:

```rust
    /// One session as the model should see it: the latest compaction
    /// summary rendered as a stub, then the messages it did not absorb,
    /// with tool results hydrated and any broken pairing repaired.
    pub fn load_session(&self, session_id: &str) -> Option<Vec<ChatMessage>> {
        let path = self.resolve_path(session_id)?;
        let loaded = load_session_file(&path)?;
        let (summary, tail) = model_history(&loaded);
        let mut out = summary
            .map(crate::context_compression::compaction_stub)
            .unwrap_or_default();
        out.extend(tail.iter().cloned().map(|m| m.into_chat_message()));
        Some(crate::session_storage::repair_tool_pairing(
            self.hydrate(out),
        ))
    }
```

- [ ] **Step 6: `append_summary` の呼び出し元を直す**

- `src/agent.rs:874-879`: `result.keep_recent` を渡す
- `src/agent.rs:653`（境界 compaction）: `0` を渡す
- `src/agent.rs:211, 257`: Task 12 で消えるが、今は `0` を渡してコンパイルを通す
- `src/serve/mod.rs:2258-2264`（`TurnPersistence::append_summary`）: `keep_recent: usize` 引数を
  足し、呼び出し元（`src/serve/mod.rs:2390`）で `result.keep_recent` を渡す。
  ACP の `is_acp` ガードは Task 11 で外すので、ここではまだ残す。

- [ ] **Step 7: テストが通ることを確認**

Run: `cargo +nightly fmt --all && cargo clippy --workspace -- -D warnings && cargo test --workspace`
Expected: PASS

- [ ] **Step 8: コミット**

```bash
git add src/session.rs src/agent.rs src/serve/mod.rs
git commit -m "feat(sessions): replay a session from its compaction checkpoint

A SummaryLine now records the last message it absorbed, so a restore can
reproduce the state the process had when it went down: the summary as a
stub, then the messages it did not cover. Replaying the whole file
instead would re-send a conversation the running process had already
compacted away, and pay to compact it again on the first turn back.

The cursor is computed by the store, from its own file. The caller holds
an index into its in-memory history and has no way to map that onto a
line number — the file holds at least as many messages as memory did,
because earlier compactions trimmed memory and not the log.

A SummaryLine without the field is read as standing in for everything
before it in file order, which is what a shutdown summary meant."
```

---

### Task 11: ACP のチェックポイント

**Files:**
- Modify: `src/acp_session.rs:57-136`（`Event` / `EventBody` / `Line`）
- Modify: `src/acp_session.rs:249-294`（`append_summary` を追加）
- Modify: `src/acp_session.rs:388-431`（`events` に `Summary` を通す）
- Modify: `src/acp_session.rs:485-533`（連鎖歩行を切り出す）
- Modify: `src/acp_session.rs:462-483`（`summary()` の網羅 match）
- Modify: `src/serve/mod.rs:2745-2757`（`run_llm_turn` の hydration）
- Modify: `src/serve/mod.rs:2254-2264`（`TurnPersistence::append_summary` の `is_acp` を外す）
- Test: `src/acp_session.rs`

**Interfaces:**
- Produces:
  - `AcpSessionStore::append_summary(&self, session_id: &str, summary: &str, keep_recent: usize) -> Result<()>`
  - `AcpSessionStore::history_for_model(&self, session_id: &str) -> Option<Vec<ChatMessage>>`

- [ ] **Step 1: 失敗するテストを書く**

`src/acp_session.rs` の `mod tests` に追加:

```rust
    /// `session/load` replays for the editor, which keeps no transcript
    /// of its own — that one stays whole. The LLM's copy does not.
    #[test]
    fn a_checkpoint_trims_the_model_history_but_not_the_editor_replay() {
        let (_d, store) = store();
        store.create("s1", "default", "/tmp").unwrap();
        for i in 0..5 {
            store
                .append_message("s1", &ChatMessage::user(&format!("m{i}")))
                .unwrap();
        }
        store.append_summary("s1", "the first three", 2).unwrap();

        let full = store.history("s1").expect("the editor replay");
        assert_eq!(full.len(), 5, "session/load must still see everything");

        let model = store.history_for_model("s1").expect("the model history");
        let texts: Vec<String> = model
            .iter()
            .flat_map(|m| &m.parts)
            .filter_map(|p| match p {
                ContentPart::Text(t) => Some(t.clone()),
                _ => None,
            })
            .collect();
        assert!(texts[0].contains("the first three"), "stub first: {texts:?}");
        assert!(texts.iter().any(|t| t == "m4"), "kept tail missing: {texts:?}");
        assert!(!texts.iter().any(|t| t == "m0"), "covered message replayed: {texts:?}");
    }

    /// No summary yet: the two reads agree.
    #[test]
    fn without_a_checkpoint_the_model_sees_the_whole_session() {
        let (_d, store) = store();
        store.create("s1", "default", "/tmp").unwrap();
        store.append_message("s1", &ChatMessage::user("only")).unwrap();
        assert_eq!(
            store.history_for_model("s1").unwrap(),
            store.history("s1").unwrap()
        );
    }

    /// A Summary event is not a message; the daily-log projection and
    /// the digest sweep must not see it as one.
    #[test]
    fn a_summary_event_is_not_a_message() {
        let (_d, store) = store();
        store.create("s1", "default", "/tmp").unwrap();
        store.append_message("s1", &ChatMessage::user("hello")).unwrap();
        store.append_summary("s1", "a recap", 0).unwrap();

        let today = today(4);
        let days = store.sessions_for_day(today, 4);
        let texts: Vec<String> = days
            .iter()
            .flat_map(|(_, ms)| ms)
            .flat_map(|m| &m.parts)
            .filter_map(|p| match p {
                ContentPart::Text(t) => Some(t.clone()),
                _ => None,
            })
            .collect();
        assert!(
            !texts.iter().any(|t| t.contains("a recap")),
            "the summary leaked into the daily log: {texts:?}"
        );
    }
```

`store()` と `today()` は `src/acp_session.rs` の `mod tests` にある既存ヘルパー
（`store()` はツール結果キャッシュ付きの `(TempDir, AcpSessionStore)` を返す）。

- [ ] **Step 2: テストが失敗することを確認**

Run: `cargo test --lib acp_session::tests::a_checkpoint_trims`
Expected: コンパイルエラー — `append_summary` / `history_for_model` が存在しない

- [ ] **Step 3: `Summary` を型に足す**

`src/acp_session.rs:68` の `EventBody` に:

```rust
    /// A compaction summary and the message it absorbed up to.
    ///
    /// Not a message: `history()` and the daily-log projection skip it,
    /// because the editor's transcript and the permanent record are both
    /// about what was said. Only `history_for_model` reads it.
    Summary {
        summary: String,
        covers_through: Uuid,
    },
```

`src/acp_session.rs:116` の `Line` に:

```rust
    Summary {
        id: Uuid,
        parent: Option<Uuid>,
        at: DateTime<Utc>,
        summary: String,
        covers_through: Uuid,
    },
```

`events()`（`src/acp_session.rs:396`）に対応するアームを足す:

```rust
                    Line::Summary {
                        id,
                        parent,
                        at,
                        summary,
                        covers_through,
                    } => Some(Event {
                        id,
                        parent,
                        at,
                        body: EventBody::Summary {
                            summary,
                            covers_through,
                        },
                    }),
```

`summary()`（`src/acp_session.rs:468`）の網羅 match に `Line::Summary { .. } => {}` を足す
（一覧行に出す情報ではない）。

- [ ] **Step 4: `append_summary` を実装する**

`src/acp_session.rs`、`append_title` の後に:

```rust
    /// Record a compaction summary and how far it reaches.
    ///
    /// `keep_recent` is how many trailing messages the caller kept
    /// verbatim. The cursor is resolved here, against this store's own
    /// events — the caller holds an index into its in-memory history and
    /// cannot map it onto the log, which holds at least as many
    /// messages because earlier compactions trimmed memory and not the
    /// file.
    ///
    /// A session with no messages gets no checkpoint; there is nothing
    /// for one to point at.
    pub fn append_summary(&self, session_id: &str, summary: &str, keep_recent: usize) -> Result<()> {
        let message_ids: Vec<Uuid> = self
            .events(session_id)
            .unwrap_or_default()
            .into_iter()
            .filter(|e| matches!(e.body, EventBody::Message { .. }))
            .map(|e| e.id)
            .collect();
        if message_ids.is_empty() {
            return Ok(());
        }
        let covered = message_ids.len().saturating_sub(keep_recent).max(1);
        let covers_through = message_ids[covered - 1];
        let id = Uuid::now_v7();
        let summary = summary.to_string();
        self.append_line(session_id, id, move |parent| Line::Summary {
            id,
            parent,
            at: Utc::now(),
            summary,
            covers_through,
        })
    }
```

`events()` は `tips` を触らないので、`append_line` の外で呼ぶこの順序に
デッドロックは無い（`append_line` の doc がその不変条件を書いている）。

- [ ] **Step 5: 連鎖歩行を切り出し、2 つの読み取りで共有する**

`src/acp_session.rs:491-531` の歩行部分を `history()` から抜き出して:

```rust
    /// The events on the chain, in order, starting from the root.
    ///
    /// File order is only accidentally right — it agrees with the chain
    /// for a session one process wrote, and says nothing useful for one
    /// that was synced or merged.
    fn chain<'a>(&self, session_id: &str, events: &'a [Event]) -> Vec<&'a Event> {
        let mut children: HashMap<Option<Uuid>, Vec<&Event>> = HashMap::new();
        for event in events {
            children.entry(event.parent).or_default().push(event);
        }
        let mut out = Vec::with_capacity(events.len());
        let mut cursor = None;
        loop {
            let Some(next) = children.get(&cursor) else {
                break;
            };
            if next.len() > 1 {
                warn!(
                    "ACP session {session_id}: {} events share one parent; taking the first branch. Two writers appended to this session and there is no merge story yet.",
                    next.len()
                );
            }
            let event = next[0];
            out.push(event);
            cursor = Some(event.id);
            // A hand-edited or partially-synced file could contain a
            // cycle. The chain cannot be longer than the file.
            if out.len() > events.len() {
                warn!("ACP session {session_id}: the parent chain cycles; stopping");
                break;
            }
        }
        out
    }
```

`history()` を:

```rust
    /// The whole conversation, for the editor's `session/load` replay.
    ///
    /// Whole on purpose: the editor keeps no transcript of its own, so a
    /// trimmed replay would render a thread with a hole in it. What the
    /// *model* sees is `history_for_model`, which starts from the latest
    /// compaction checkpoint.
    pub fn history(&self, session_id: &str) -> Option<Vec<ChatMessage>> {
        let events = self.events(session_id)?;
        let out = self.messages_from(&self.chain(session_id, &events));
        Some(crate::session_storage::repair_tool_pairing(out))
    }

    /// The conversation as the model should see it: the latest
    /// compaction summary rendered as a stub, then the messages it did
    /// not absorb.
    ///
    /// Compression runs on ACP turns like every other transport, and
    /// used to throw its summary away on the grounds that the events
    /// answer the same question. They do — but they answer it with the
    /// *whole* session, so every reload replayed everything and paid for
    /// the same compaction again on the first turn back.
    pub fn history_for_model(&self, session_id: &str) -> Option<Vec<ChatMessage>> {
        let events = self.events(session_id)?;
        let chain = self.chain(session_id, &events);

        let checkpoint = chain.iter().rev().find_map(|e| match &e.body {
            EventBody::Summary {
                summary,
                covers_through,
            } => Some((summary.clone(), *covers_through)),
            _ => None,
        });

        let Some((summary, covers_through)) = checkpoint else {
            let out = self.messages_from(&chain);
            return Some(crate::session_storage::repair_tool_pairing(out));
        };

        let messages: Vec<&Event> = chain
            .iter()
            .copied()
            .filter(|e| matches!(e.body, EventBody::Message { .. }))
            .collect();
        let start = match messages.iter().position(|e| e.id == covers_through) {
            Some(pos) => pos + 1,
            None => {
                warn!(
                    "ACP session {session_id}: checkpoint {covers_through} is not on the chain; replaying everything"
                );
                0
            }
        };

        let mut out = crate::context_compression::compaction_stub(&summary);
        out.extend(self.messages_from(&messages[start..]));
        Some(crate::session_storage::repair_tool_pairing(out))
    }

    /// Project the message events among `events` into `ChatMessage`s,
    /// hydrating each part.
    fn messages_from(&self, events: &[&Event]) -> Vec<ChatMessage> {
        events
            .iter()
            .filter_map(|e| match &e.body {
                EventBody::Message { role, parts } => Some(ChatMessage {
                    role: role.clone(),
                    parts: parts.iter().map(|p| self.load_part(p)).collect(),
                    input_kind: None,
                    user_id: None,
                }),
                _ => None,
            })
            .collect()
    }
```

- [ ] **Step 6: `run_llm_turn` と `TurnPersistence` を繋ぐ**

`src/serve/mod.rs:2748-2752` を:

```rust
                if is_acp {
                    state
                        .acp_session_store
                        .history_for_model(&session_id)
                        .unwrap_or_default()
                } else {
                    store.load_session(&session_id).unwrap_or_default()
                }
```

`TurnPersistence::append_summary`（`src/serve/mod.rs:2254`）を:

```rust
    /// Append a compaction summary and the checkpoint it establishes.
    ///
    /// Unconditional now. ACP used to skip this on the grounds that its
    /// events already answer the question — they do, but with the whole
    /// session, so every reload replayed everything and re-paid for the
    /// same compaction on the first turn back.
    fn append_summary(&self, summary: &str, keep_recent: usize) {
        let result = if self.is_acp {
            self.acp_store
                .append_summary(&self.session_id, summary, keep_recent)
        } else {
            self.store
                .append_summary(&self.session_id, summary, keep_recent)
        };
        if let Err(e) = result {
            warn!("Failed to persist compaction summary: {e}");
        }
    }
```

`src/serve/mod.rs:2384-2391` のコメントを差し替え、`p.append_summary(&result.summary, result.keep_recent);` を渡す。

- [ ] **Step 7: テストが通ることを確認**

Run: `cargo +nightly fmt --all && cargo clippy --workspace -- -D warnings && cargo test --workspace`
Expected: PASS

- [ ] **Step 8: コミット**

```bash
git add src/acp_session.rs src/serve/mod.rs
git commit -m "feat(acp): keep the compaction summary, and split the two histories

history() answered two consumers with one transcript. session/load
replays it to the editor, which keeps none of its own and needs all of
it; run_llm_turn fed the same thing to the model, which does not. So
every reload replayed the whole session and paid for the same compaction
again on the first turn back — the summary was generated and discarded
on the grounds that the events already answered the question.

The events do answer it, with the whole session. A Summary event now
records what the summary absorbed, history_for_model starts there, and
history() stays whole for the editor. TurnPersistence::append_summary
loses its is_acp branch, which is the third one this work removes."
```

---

### Task 12: チャンネル履歴を復元し、再起動要約を退役させる

**Files:**
- Modify: `src/session.rs:490-571`（`load_all`）
- Modify: `src/agent.rs:41-120`（フィールドと `new`）
- Modify: `src/agent.rs:122-220`（`bootstrap` を削除）
- Modify: `src/agent.rs:222-274`（`summarize_on_shutdown`）
- Modify: `src/agent.rs:764-781`（`<prior-session-recap>` を削除）
- Modify: `src/main.rs:824`（`agent.bootstrap()` を削除）
- Test: `src/session.rs`

**Interfaces:**
- Produces: `SessionStore::load_all(&self) -> (HashMap<ConversationKey, String>, HashMap<ConversationKey, Vec<ChatMessage>>)`

- [ ] **Step 1: 失敗するテストを書く**

`src/session.rs` の `mod tests` に追加:

```rust
    /// The point of the whole change: a restarted room remembers what it
    /// did, not just what was said.
    #[test]
    fn load_all_restores_tool_traffic_for_an_active_session() {
        let (_s, _c, store, sid) = cached_store();
        store.append(&sid, &ChatMessage::user("read a.rs")).unwrap();
        store.append(&sid, &tool_use_msg("c1")).unwrap();
        store.append(&sid, &tool_result_msg("c1", "fn main() {}")).unwrap();

        let (active, histories) = store.load_all();
        let key = ("!room:example.org".to_string(), None);
        assert_eq!(active.get(&key).map(String::as_str), Some(sid.as_str()));

        let history = histories.get(&key).expect("history restored");
        assert!(
            history.iter().flat_map(|m| &m.parts).any(|p| matches!(p, ContentPart::ToolResult { tool_use_id, content } if tool_use_id == "c1" && content == "fn main() {}")),
            "the tool result did not come back: {history:?}"
        );
    }

    /// A closed session is not resumed — the day boundary rotated it on
    /// purpose.
    #[test]
    fn load_all_skips_closed_sessions() {
        let (_s, _c, store, sid) = cached_store();
        store.append(&sid, &ChatMessage::user("hi")).unwrap();
        store.close_session(&sid).unwrap();

        let (active, histories) = store.load_all();
        assert!(active.is_empty(), "{active:?}");
        assert!(histories.is_empty(), "{histories:?}");
    }
```

- [ ] **Step 2: テストが失敗することを確認**

Run: `cargo test --lib session::tests::load_all_restores`
Expected: コンパイルエラー — `load_all` は 3 要素タプルを返す

- [ ] **Step 3: `load_all` を書き換える**

`src/session.rs:490-571` を置き換える:

```rust
    /// Load every active session from disk on startup.
    ///
    /// For each `ConversationKey`, picks the latest session that has no
    /// `closed_at` marker. Returns:
    ///
    /// - `active`: which session file is current per conversation
    /// - `histories`: that session's conversation as the model should
    ///   see it — the latest compaction summary as a stub, the messages
    ///   it did not absorb, tool results hydrated, pairing repaired
    ///
    /// Raw history used to be withheld on the grounds that Anthropic
    /// requires paired tool traffic and none was persisted. Both halves
    /// of that changed (#194), so the restart summary this replaced is
    /// gone: a summary costs a model call, throws the turn structure
    /// away, and answers a question the log now answers directly.
    pub fn load_all(
        &self,
    ) -> (
        HashMap<ConversationKey, String>,
        HashMap<ConversationKey, Vec<ChatMessage>>,
    ) {
        let mut entries: Vec<(String, ConversationKey)> = Vec::new();

        for path in collect_session_files(&self.base_dir, self.kind) {
            let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
                continue;
            };
            let Some(loaded) = load_session_file(&path) else {
                continue;
            };
            if loaded.is_closed {
                continue;
            }
            // Seed the path cache so the first append after bootstrap
            // doesn't pay a scan.
            if let Ok(mut cache) = self.path_cache.lock() {
                cache.insert(stem.to_string(), path.clone());
            }
            entries.push((
                stem.to_string(),
                (loaded.meta.room_id.clone(), loaded.meta.thread_id.clone()),
            ));
        }

        // Session ids are UUIDv7, so the newest file for a key sorts last.
        entries.sort_by(|a, b| a.0.cmp(&b.0));

        let mut active: HashMap<ConversationKey, String> = HashMap::new();
        let mut histories: HashMap<ConversationKey, Vec<ChatMessage>> = HashMap::new();
        for (session_id, key) in entries {
            match self.load_session(&session_id) {
                Some(history) if !history.is_empty() => {
                    histories.insert(key.clone(), history);
                }
                _ => {
                    histories.remove(&key);
                }
            }
            active.insert(key, session_id);
        }

        (active, histories)
    }
```

- [ ] **Step 4: `Agent` を直す**

`src/agent.rs:63-68` の `restart_summaries` と `pending_fallback` フィールドを削除。
`src/agent.rs:51-54` の `history` の doc を差し替える:

```rust
    /// In-memory conversation history, keyed by (room_id, thread_id).
    /// Seeded at startup from the session log — including tool traffic,
    /// which is why there is no restart summary any more (#194).
    history: Mutex<HashMap<ConversationKey, Vec<ChatMessage>>>,
```

`Agent::new`（`src/agent.rs:95`）を:

```rust
        let (active_sessions, histories) = session_store.load_all();
        info!(
            "Restored {} active session(s) from disk ({} with history)",
            active_sessions.len(),
            histories.len(),
        );
```

構造体リテラルで `history: Mutex::new(histories),` にし、
`restart_summaries` と `pending_fallback` の行を削除する。

- [ ] **Step 5: `bootstrap()` を削除する**

`src/agent.rs:122-220` の `bootstrap` を丸ごと削除。
`src/main.rs:824` の `agent.bootstrap().await;` も削除する。

`generate_summary` の import がこれで未使用にならないか確認する
（`summarize_on_shutdown` と `compact_at_boundary` がまだ使うので残る）。

- [ ] **Step 6: `summarize_on_shutdown` から `SummaryLine` 書き込みを外す**

`src/agent.rs:222-274` を置き換える。要約はまだ生成する — digest には要る:

```rust
    /// Publish an intra-day digest for each active session so the
    /// cross-session today block picks up what they covered before we
    /// went down.
    ///
    /// No `SummaryLine` here any more. That one existed to bridge a
    /// restart, and the log now carries the conversation itself (#194).
    /// A summary written here would be worse than useless: it would
    /// establish a checkpoint covering the whole session, so the next
    /// start would replay a stub instead of the history it just gained.
    /// Compaction still writes one — that is what bounds context, and it
    /// is written at a point where the history really was compacted.
    async fn flush_digests_on_shutdown(&self) {
        let Some(cache) = self.digest_cache.clone() else {
            return;
        };
        let snapshot: Vec<(ConversationKey, String, Vec<ChatMessage>)> = {
            let history = self.history.lock().await;
            let sessions = self.active_sessions.lock().await;
            history
                .iter()
                .filter_map(|(key, msgs)| {
                    if msgs.len() < 2 {
                        return None;
                    }
                    let sid = sessions.get(key)?.clone();
                    if sid.is_empty() {
                        return None;
                    }
                    Some((key.clone(), sid, msgs.clone()))
                })
                .collect()
        };

        if snapshot.is_empty() {
            return;
        }

        info!(
            "Graceful shutdown: digesting {} active session(s)",
            snapshot.len()
        );

        for (key, session_id, messages) in snapshot {
            let provider = self.provider_for(&key.0);
            match generate_summary(&*provider, &messages).await {
                Ok(summary) if !summary.trim().is_empty() => {
                    if let Err(e) = cache.put(&session_id, &summary, None) {
                        warn!("Failed to cache the shutdown digest for {session_id}: {e}");
                    }
                }
                Ok(_) => warn!("Shutdown digest for {session_id} was empty; skipping"),
                Err(e) => warn!("Shutdown digest generation failed for {session_id}: {e:#}"),
            }
        }
    }
```

`src/agent.rs:361` の `self.summarize_on_shutdown().await;` を
`self.flush_digests_on_shutdown().await;` に変える。

- [ ] **Step 7: `<prior-session-recap>` の注入を削除**

`src/agent.rs:764-781` のブロックを丸ごと削除し、`system_with_context` を
その前の `let` の結果のまま使う。

- [ ] **Step 8: 全テストと手動確認**

Run: `cargo +nightly fmt --all && cargo clippy --workspace -- -D warnings && cargo test --workspace`
Expected: PASS

Run: `grep -rn "restart_summaries\|pending_fallback\|prior-session-recap\|summarize_on_shutdown" src/`
Expected: 出力なし

- [ ] **Step 9: コミット**

```bash
git add src/session.rs src/agent.rs src/main.rs
git commit -m "feat(sessions): restore channel history instead of summarising a restart

Matrix and Discord rooms come back with their conversation, tool traffic
included, rather than a paragraph apologising for having lost it. The
restart summary, the crash-fallback summary bootstrap synthesised, and
the <prior-session-recap> block that pasted either into the system
prompt are all gone: they existed because raw history could not be
reloaded, and it can now.

Shutdown still generates a summary, for the intra-day digest only. It no
longer writes a SummaryLine, which would be actively harmful — the line
is a compaction checkpoint now, and one written at shutdown would claim
to cover the whole session, so the next start would replay a stub
instead of the history it just gained.

Compaction still writes one. That is what bounds context, and it is
written where the history really was compacted."
```

---

## 完了時の確認

- [ ] `cargo +nightly fmt --check --all`
- [ ] `cargo clippy --workspace -- -D warnings`
- [ ] `cargo test --workspace`
- [ ] `grep -rn "append_intraday_digest\|append_acp_only\|scrub_images_for_storage" src/` が空
- [ ] `git log --oneline main..HEAD` が 10 コミット（spec 1 + 実装 9）
- [ ] 以下を起票する:
  - 画像が復元されない（`scrub_images_for_storage` の後継を `ImageRef` に倒す）
  - `SummaryLine` の累積
  - `SessionStore` のファイル形式を ACP に収束させるか
