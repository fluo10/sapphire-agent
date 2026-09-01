# ACP セッションを自前のストアに分ける Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** ACP セッションを `acp/` ストアに移し、イベントに `id` と `parent` を持たせ、ツール結果をワークスペース外のキャッシュに逃がす。そのうえで **ACP セッションをデイリーログと横断ダイジェストに載せる**——エージェントと今日何をしたかは残すべき記録なので。

**Architecture:** 新しいストア型が行の形式を所有し、ディレクトリ走査などの機構は既存と共有する。ACP アダプタのハンドラ（namespace 境界、拒否文言、replay、adopt、接続カウンタ）は**ストアに依存しないのでほぼそのまま残る**——変わるのは呼ぶストア型だけ。

デイリーログと横断ダイジェストへの取り込みは、**新ストア側に読み出し専用のアダプタ**を置いて解決する。`format_sessions` と `build_today_digest_for_namespace` が要求するのは `SessionMeta` / `StoredMessage` / `IntradayDigestLine` という既存の語彙なので、新ストアが自分のイベントからその形に射影する。書き込み形式は分離したまま、読み手の下流は既存のままにできる。

**Tech Stack:** Rust 2024, `serde` / `serde_json`, `uuid` (v7), `sha2`, `chrono`, `tempfile`

**Spec:** `docs/superpowers/specs/2026-08-31-acp-store-separation-design.md`

## Global Constraints

- ブランチは既存の `feat/acp-session-load`（PR #188、未マージ）の**上に積む**。先行コミットは個別レビュー済みなので書き換えない。
- テストコマンドは `cargo test -p sapphire-agent`。反復中は `--workspace` を使わない。コミット前に一度だけ `cargo test --workspace`。
- **`cargo check` と `cargo test` を混ぜない。** フィンガープリントが別なので交互に走らせると毎回リンクし直しになる。
- cargo は**フォアグラウンドで、長いタイムアウトで**。`run_in_background` も `Monitor` も使わない。**cargo を2本同時に走らせない**（このホストはディスクが同時書き込みでスロットリングする）。
- **`Cargo.lock` をコミットしない。** `Cargo.toml` が `cron = "0.16"` / `tower-http = "0.6"` を宣言しているのに committed lockfile が 0.17 / 0.7 を持つという `main` 由来の不整合があり、cargo を走らせるたびに書き換わる。各コミット前に `git checkout -- Cargo.lock`。
- テストはソースファイル内の `mod tests` に置く。
- **既存4ストア（`cross-device` / `channel` / `device-default` / `mcp`）の行形式は変えない。** 新設は `acp/` の1つだけ。`src/session.rs` から消すのは **PR #188 がこのブランチで足したもの**（`SessionMeta.cwd` とその周辺）だけで、`main` にある動作は変えない。可視性の緩和（`collect_session_files` を `pub(crate)` に）は許す。
- **`src/agent.rs` は一行も触らない。** 要約3箇所（`:211` / `:257` / `:653`）もダイジェスト2箇所（`:265` / `:459`）も `channel` ストア（Matrix / Discord）向けで、この plan の対象外。
- **`summarize_all_sessions` は関数ごとは消さない。** 消すのは中の `append_summary` 呼び出しだけ。理由は Task 4 に書いてある——同じ関数が `cross_device` ストアの唯一のダイジェスト生成源でもあるため。
- イベント id は **UUIDv7**（`uuid::Uuid::now_v7()`）。`parent` は同じ型の `Option`。
- **`parent: null` は「セッション最初のイベント」。** ヘッダは根イベントにしない。
- **順序の権威は `parent` 鎖であって `id` の時刻ではない。**
- **ACP セッションはデイリーログと横断ダイジェストの両方に載せる**（Task 6・Task 7）。`/rpc` / device-default / MCP の扱いは変えない——[#189](https://github.com/fluo10/sapphire-agent/issues/189) に切り出してある。

## spec からの2つの精緻化

どちらも設計の意図は変えず、符号化だけを決めるもの。レビュアーが逸脱と見なさないよう明記する。

1. **行は単一のタグ付き enum。** spec は ヘッダを `{"header": {...}}`、イベントを `{"kind":"message",...}` と書いていた。実装は**すべてに `kind` を持たせる**（`"header"` / `"message"` / `"title"` / `"closed"`）。untagged との併用は曖昧さを生むので、一つの internally-tagged enum に統一する。
2. **`ContentPart` に新しい variant を足さない。** spec の JSON 例は `tool_result_ref` を parts に置いているが、**参照はディスク上にしか存在しない**。メモリ上の履歴は本物の内容を持つ（モデルが必要とするので）。したがって変換はストアの直列化境界で行い、`provider::ContentPart` は無変更。`image_cache` が `ImageRef` を variant にしたのは、参照がメモリ上の履歴にも残る必要があったからで、事情が違う。

## spec が答えていなかった点

### `maybe_compress` の要約は永続化しない

spec は「ターン中の `maybe_compress` は残す」と書いたが、その要約は今 `append_summary` でストアに書かれており、新ストアに `SummaryLine` は無い。

**解決: 新ストアでは書かない。** 圧縮はメモリ上の最適化で、永続化していたのは再開のため。その再開こそイベントからの復元が置き換えるので、書く理由が消える。プロセスが再起動すればイベントから完全な履歴が読み直され、必要ならまた圧縮される。

### ダイジェストはワークスペース外のキャッシュに置く（spec のとおり）

spec の「サマリーとダイジェストはどちらもワークスペース外へ」は正しい。ストアのイベントにしてはいけない理由が2つある。

**`<workspace>/sessions` は retrieve の索引に入っている。** `SessionStore.base_dir` がそこで、`notify_updated` が `ws_state` に変更を通知している。ダイジェストは会話が伸びるたびに作り直されるので、イベントとして追記すると**似た内容の要約が同じファイルに積み上がり、検索結果を歪める。** 開発のようにターンの長いやりとりでは特に効く。

**刈り取りでは解けない。** イベントは `parent` 鎖で繋がっているので、途中の要素を消すと子が孤児になる。append-only とダイジェストの刈り取りは両立しない。

**メタデータの二重化は起きない。** キャッシュが持つのは本文と時刻だけで、`namespace` / `created_at` / `title` は読み出し時に**ストアのヘッダから結合する**。キャッシュは `session_id` をキーに最新の1件だけを持つので、上書きによってそもそも溜まらない。加えて日跨ぎの取りこぼし用に `prune_before` を用意し、デイリーログを書いた時点で呼ぶ。

`channel` ストアのダイジェスト（`src/agent.rs` が書いているもの）は**今のまま**で、移設しない。

### ACP セッションはデイリーログと横断ダイジェストに載せる

`generate_daily_log` はストアを1つしか読まず、渡っているのは `channel_session_store`。放っておくと ACP セッションはどちらの記録にも現れない。

**Task 6 と Task 7 で載せる。** `/rpc` を同じ扱いにするかは別の判断なので、
[#189](https://github.com/fluo10/sapphire-agent/issues/189) に切り出した。この plan は `/rpc` の扱いを変えない。

---

### Task 1: ツール結果キャッシュ

`image_cache` と同型の、ワークスペース外・ハッシュアドレスのキャッシュ。単体で完結する。

**Files:**
- Create: `src/tool_result_cache.rs`
- Modify: `src/main.rs`（`mod tool_result_cache;`）
- Test: `src/tool_result_cache.rs` の `mod tests`

**Interfaces:**
- Produces: `ToolResultCache::open(PathBuf) -> Result<Arc<Self>>`、`ToolResultCache::default_dir() -> Option<PathBuf>`、`ToolResultCache::put(&self, &str) -> Result<String>`（sha256 を返す）、`ToolResultCache::get(&self, &str) -> Option<String>`

- [ ] **Step 1: 失敗するテストを書く**

```rust
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
        assert_eq!(cache.get("0000000000000000000000000000000000000000000000000000000000000000"), None);
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
```

- [ ] **Step 2: テストが落ちる（コンパイルしない）ことを確認**

Run: `cargo test -p sapphire-agent tool_result_cache`
Expected: FAIL — `ToolResultCache` が未定義。

- [ ] **Step 3: 実装を書く**

`src/tool_result_cache.rs` の先頭に。

```rust
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

use anyhow::Result;
use sha2::{Digest, Sha256};
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
    format!("{:x}", hasher.finalize())
}
```

`src/main.rs` のモジュール宣言に `mod tool_result_cache;` を足す（`mod image_cache;` の隣）。

**確認:** `sha2` と `dirs` が既に依存にあるか。`image_cache` が両方使っているので、あるはず。無ければ `Cargo.toml` に足す（`Cargo.lock` はコミットしない）。

- [ ] **Step 4: テストが通ることを確認**

Run: `cargo test -p sapphire-agent tool_result_cache`
Expected: PASS（4テスト）。

- [ ] **Step 5: コミット**

```bash
git checkout -- Cargo.lock
git add src/tool_result_cache.rs src/main.rs
git commit -m "feat(sessions): a workspace-external cache for tool results"
```

---

### Task 2: ACP セッションストア — 形式と追記

行の形式を所有する新しいストア。`id` と `parent` はここで生まれる。

**Files:**
- Create: `src/acp_session.rs`
- Modify: `src/main.rs`（`mod acp_session;`）
- Modify: `src/session.rs`（`collect_session_files` を `pub(crate)` にする）
- Test: `src/acp_session.rs` の `mod tests`

**Interfaces:**
- Consumes: `ToolResultCache`（Task 1）
- Produces: `AcpSessionStore::new(base_dir: PathBuf, cache: Arc<ToolResultCache>) -> Self`、`create(&self, session_id: &str, namespace: &str, cwd: &str) -> Result<()>`、`append_message(&self, session_id: &str, msg: &ChatMessage) -> Result<()>`、`append_title(&self, session_id: &str, title: &str) -> Result<()>`、`close(&self, session_id: &str) -> Result<()>`、`header(&self, session_id: &str) -> Option<SessionHeader>`、`is_closed(&self, session_id: &str) -> bool`、`summary(&self, session_id: &str) -> Option<SessionSummary>`、`list_summaries(&self, namespace: &str) -> Vec<SessionSummary>`、`SessionHeader { session_id, namespace, cwd, created_at }`、`SessionSummary { header, title, has_messages, is_closed }`

- [ ] **Step 1: 失敗するテストを書く**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::{ChatMessage, ContentPart, Role};

    fn store() -> (tempfile::TempDir, AcpSessionStore) {
        let dir = tempfile::tempdir().unwrap();
        let cache_dir = dir.path().join("cache");
        let cache = crate::tool_result_cache::ToolResultCache::open(cache_dir).unwrap();
        let store = AcpSessionStore::new(dir.path().join("sessions"), cache);
        (dir, store)
    }

    #[test]
    fn a_header_round_trips() {
        let (_d, store) = store();
        store.create("s1", "default", "/home/u/project").unwrap();

        let h = store.header("s1").expect("the session exists");
        assert_eq!(h.session_id, "s1");
        assert_eq!(h.namespace, "default");
        assert_eq!(h.cwd, "/home/u/project");
        assert!(!store.is_closed("s1"));
    }

    /// The first event has no parent; every later one points at the
    /// event that was the tip when it was written. This chain is what
    /// makes an offline divergence detectable, so it is the property
    /// most worth pinning.
    #[test]
    fn the_parent_chain_records_append_order() {
        let (_d, store) = store();
        store.create("s1", "default", "/p").unwrap();
        store.append_message("s1", &ChatMessage::user("one")).unwrap();
        store.append_message("s1", &ChatMessage::assistant("two")).unwrap();
        store.append_title("s1", "a title").unwrap();

        let events = store.events("s1").expect("the session exists");
        assert_eq!(events.len(), 3);
        assert_eq!(events[0].parent, None, "the first event has no parent");
        assert_eq!(events[1].parent, Some(events[0].id));
        assert_eq!(events[2].parent, Some(events[1].id));
    }

    /// Ids are UUIDv7 so a directory listing sorts chronologically once
    /// events become files. The chain, not the timestamp, is the
    /// authority on order — but the ids must still be v7.
    #[test]
    fn event_ids_are_uuid_v7() {
        let (_d, store) = store();
        store.create("s1", "default", "/p").unwrap();
        store.append_message("s1", &ChatMessage::user("x")).unwrap();

        let events = store.events("s1").unwrap();
        assert_eq!(events[0].id.get_version_num(), 7);
    }

    /// Closing appends an event rather than rewriting anything, so the
    /// log stays append-only.
    #[test]
    fn closing_appends_an_event() {
        let (_d, store) = store();
        store.create("s1", "default", "/p").unwrap();
        store.append_message("s1", &ChatMessage::user("x")).unwrap();
        store.close("s1").unwrap();

        assert!(store.is_closed("s1"));
        let events = store.events("s1").unwrap();
        assert_eq!(events.len(), 2);
        assert!(matches!(events[1].body, EventBody::Closed));
    }

    /// A tool result is written to the cache and referenced by hash;
    /// the log never carries the content.
    #[test]
    fn a_tool_result_is_stored_by_reference() {
        let (dir, store) = store();
        store.create("s1", "default", "/p").unwrap();
        store
            .append_message(
                "s1",
                &ChatMessage {
                    role: Role::User,
                    parts: vec![ContentPart::ToolResult {
                        tool_use_id: "c1".to_string(),
                        content: "a very long file listing".to_string(),
                    }],
                    input_kind: None,
                    user_id: None,
                },
            )
            .unwrap();

        let raw = std::fs::read_to_string(store.path_for_test("s1")).unwrap();
        assert!(
            !raw.contains("a very long file listing"),
            "the content must live in the cache, not the log: {raw}"
        );
        assert!(raw.contains("tool_result_ref"), "got {raw}");
        drop(dir);
    }

    /// The listing is per namespace. Another namespace's sessions are
    /// not this connection's to see — the boundary is enforced in the
    /// ACP handler too, but the store should not hand them over either.
    #[test]
    fn listing_is_scoped_to_one_namespace() {
        let (_d, store) = store();
        store.create("mine", "default", "/p").unwrap();
        store.create("theirs", "someone-else", "/p").unwrap();

        let ids: Vec<String> = store
            .list_summaries("default")
            .into_iter()
            .map(|s| s.header.session_id)
            .collect();
        assert_eq!(ids, vec!["mine".to_string()]);
    }

    /// The editor opens a thread on every panel open, so most sessions
    /// are created and never typed into. The listing says which ones
    /// carry a message, and the handler drops the rest — computed from
    /// the read the header already needs, so it costs nothing extra.
    #[test]
    fn a_summary_reports_whether_anything_was_said() {
        let (_d, store) = store();
        store.create("empty", "default", "/p").unwrap();
        store.create("used", "default", "/p").unwrap();
        store.append_message("used", &ChatMessage::user("hi")).unwrap();
        store.append_title("used", "greetings").unwrap();

        let by_id: std::collections::HashMap<String, SessionSummary> = store
            .list_summaries("default")
            .into_iter()
            .map(|s| (s.header.session_id.clone(), s))
            .collect();

        assert!(!by_id["empty"].has_messages);
        assert_eq!(by_id["empty"].title, None);
        assert!(by_id["used"].has_messages);
        assert_eq!(by_id["used"].title.as_deref(), Some("greetings"));
    }
}
```

- [ ] **Step 2: テストが落ちることを確認**

Run: `cargo test -p sapphire-agent acp_session`
Expected: FAIL — `AcpSessionStore` が未定義。

- [ ] **Step 3: 型を書く**

`src/acp_session.rs` の先頭に。

```rust
//! The ACP session store.
//!
//! Separate from `SessionStore` because ACP is an externally-defined
//! standard that will drift from the format we chose for ourselves, and
//! because an editor's thread list should not be showing Matrix
//! conversations. The directory-walking machinery is shared; the line
//! format is not.
//!
//! Every event carries a UUIDv7 `id` and the `parent` it was appended
//! after. That pair cannot be reconstructed later, which is why it is
//! here now even though nothing reads it yet: it is what will make an
//! offline divergence detectable if remote-workspace sync is ever
//! implemented, and what makes splitting one file per event a
//! mechanical move rather than a redesign.
//!
//! **The chain is the authority on order, not the id's timestamp.**
//! UUIDv7 embeds the writer's clock, so two devices with skewed clocks
//! produce ids that sort wrongly against each other.

use crate::provider::{ChatMessage, ContentPart, Role};
use crate::tool_result_cache::ToolResultCache;
use anyhow::Result;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use tracing::warn;
use uuid::Uuid;

/// The directory name under `<sessions>/<namespace>/`.
const KIND: &str = "acp";

/// What the model is told when a tool result is no longer in the cache.
///
/// The pairing between `tool_use` and `tool_result` is what the API
/// validates, not the content — so a placeholder keeps the history
/// valid and the conversation's shape intact. This is why the store
/// needs no resume summary: a summary would cost a model call and throw
/// the turn structure away to solve a problem a sentence solves.
pub const MISSING_RESULT: &str =
    "[this tool result is no longer stored; call the tool again if you need it]";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SessionHeader {
    pub session_id: String,
    pub namespace: String,
    /// The client's workspace root. Required — an ACP session always
    /// reports one, which is why this is not an `Option` the way the
    /// old store's `cwd` had to be.
    pub cwd: String,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Event {
    pub id: Uuid,
    /// `None` means this is the session's first event. The header is
    /// not a root event — making it one would need a special case in
    /// every traversal.
    pub parent: Option<Uuid>,
    pub at: DateTime<Utc>,
    pub body: EventBody,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum EventBody {
    Message { role: Role, parts: Vec<StoredPart> },
    Title { title: String },
    Closed,
}

/// A message part as it appears on disk.
///
/// Distinct from `provider::ContentPart` on purpose: a tool result is a
/// hash here and the real content in memory, and that difference exists
/// only at the storage boundary. `image_cache` put its `ImageRef` into
/// `ContentPart` because those references stay in the in-memory history
/// to avoid re-billing; a tool result has to be whole when the model
/// sees it, so the reference never leaves the disk.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum StoredPart {
    Text(String),
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    ToolResultRef {
        tool_use_id: String,
        sha256: String,
    },
    /// Anything the storage layer does not model — images, for now.
    /// Kept as a marker so the message is not silently emptied.
    Other,
}

/// One line of the log. Everything is `kind`-tagged, including the
/// header: a single internally-tagged enum has no ambiguity, whereas
/// mixing a tagged event with an untagged `{"header": …}` wrapper does.
#[derive(Debug, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum Line {
    Header(SessionHeader),
    Message {
        id: Uuid,
        parent: Option<Uuid>,
        at: DateTime<Utc>,
        role: Role,
        parts: Vec<StoredPart>,
    },
    Title {
        id: Uuid,
        parent: Option<Uuid>,
        at: DateTime<Utc>,
        title: String,
    },
    Closed {
        id: Uuid,
        parent: Option<Uuid>,
        at: DateTime<Utc>,
    },
}
```

- [ ] **Step 4: ストアと追記を書く**

```rust
pub struct AcpSessionStore {
    base_dir: PathBuf,
    cache: Arc<ToolResultCache>,
    /// `session_id` → the id of the last event written.
    ///
    /// An append needs its parent, and re-reading the whole file to
    /// find the tip would make every write cost the length of the
    /// conversation. Populated lazily on the first append to a session.
    tips: Mutex<HashMap<String, Uuid>>,
}

impl AcpSessionStore {
    pub fn new(base_dir: PathBuf, cache: Arc<ToolResultCache>) -> Self {
        Self {
            base_dir,
            cache,
            tips: Mutex::new(HashMap::new()),
        }
    }

    fn path(&self, session_id: &str, namespace: &str) -> PathBuf {
        self.base_dir
            .join(namespace)
            .join(KIND)
            .join(format!("{session_id}.jsonl"))
    }

    /// Find an existing session's file by scanning namespaces.
    fn find(&self, session_id: &str) -> Option<PathBuf> {
        let target = format!("{session_id}.jsonl");
        crate::session::collect_session_files(&self.base_dir, KIND)
            .into_iter()
            .find(|p| p.file_name().and_then(|s| s.to_str()) == Some(target.as_str()))
    }

    #[cfg(test)]
    pub fn path_for_test(&self, session_id: &str) -> PathBuf {
        self.find(session_id).expect("the session exists")
    }

    pub fn create(&self, session_id: &str, namespace: &str, cwd: &str) -> Result<()> {
        let path = self.path(session_id, namespace);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let header = Line::Header(SessionHeader {
            session_id: session_id.to_string(),
            namespace: namespace.to_string(),
            cwd: cwd.to_string(),
            created_at: Utc::now(),
        });
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)?;
        writeln!(file, "{}", serde_json::to_string(&header)?)?;
        Ok(())
    }

    /// The id of the last event, reading the file once if this process
    /// has not appended to the session yet.
    fn tip(&self, session_id: &str) -> Option<Uuid> {
        if let Some(id) = self.tips.lock().unwrap().get(session_id) {
            return Some(*id);
        }
        let last = self.events(session_id)?.last().map(|e| e.id)?;
        self.tips
            .lock()
            .unwrap()
            .insert(session_id.to_string(), last);
        Some(last)
    }

    fn append_line(&self, session_id: &str, id: Uuid, line: Line) -> Result<()> {
        let path = self
            .find(session_id)
            .ok_or_else(|| anyhow::anyhow!("no ACP session '{session_id}'"))?;
        let mut file = std::fs::OpenOptions::new().append(true).open(&path)?;
        writeln!(file, "{}", serde_json::to_string(&line)?)?;
        self.tips.lock().unwrap().insert(session_id.to_string(), id);
        Ok(())
    }

    pub fn append_message(&self, session_id: &str, msg: &ChatMessage) -> Result<()> {
        let parts = msg
            .parts
            .iter()
            .map(|part| self.store_part(part))
            .collect::<Result<Vec<_>>>()?;
        let id = Uuid::now_v7();
        self.append_line(
            session_id,
            id,
            Line::Message {
                id,
                parent: self.tip(session_id),
                at: Utc::now(),
                role: msg.role.clone(),
                parts,
            },
        )
    }

    pub fn append_title(&self, session_id: &str, title: &str) -> Result<()> {
        let id = Uuid::now_v7();
        self.append_line(
            session_id,
            id,
            Line::Title {
                id,
                parent: self.tip(session_id),
                at: Utc::now(),
                title: title.to_string(),
            },
        )
    }

    pub fn close(&self, session_id: &str) -> Result<()> {
        let id = Uuid::now_v7();
        self.append_line(
            session_id,
            id,
            Line::Closed {
                id,
                parent: self.tip(session_id),
                at: Utc::now(),
            },
        )
    }

    /// A tool result's content goes to the cache; the log keeps a hash.
    fn store_part(&self, part: &ContentPart) -> Result<StoredPart> {
        Ok(match part {
            ContentPart::Text(t) => StoredPart::Text(t.clone()),
            ContentPart::ToolUse { id, name, input } => StoredPart::ToolUse {
                id: id.clone(),
                name: name.clone(),
                input: input.clone(),
            },
            ContentPart::ToolResult {
                tool_use_id,
                content,
            } => StoredPart::ToolResultRef {
                tool_use_id: tool_use_id.clone(),
                sha256: self.cache.put(content)?,
            },
            // Images are not carried by this version. Recorded as a
            // marker rather than dropped, so a message that was only an
            // image does not read back as an empty one.
            _ => StoredPart::Other,
        })
    }
}
```

- [ ] **Step 5: 読み出しの最小限を書く**

Task 3 で本格的な読み出し（parent 鎖・分岐検出・hydrate）を書く。ここでは Task 2 のテストが必要とする分だけ。

```rust
impl AcpSessionStore {
    fn lines(&self, session_id: &str) -> Option<Vec<Line>> {
        let path = self.find(session_id)?;
        let text = std::fs::read_to_string(path).ok()?;
        Some(
            text.lines()
                .filter(|l| !l.trim().is_empty())
                .filter_map(|l| match serde_json::from_str::<Line>(l) {
                    Ok(line) => Some(line),
                    Err(e) => {
                        warn!("ACP session {session_id}: skipping unreadable line: {e}");
                        None
                    }
                })
                .collect(),
        )
    }

    pub fn header(&self, session_id: &str) -> Option<SessionHeader> {
        self.lines(session_id)?.into_iter().find_map(|l| match l {
            Line::Header(h) => Some(h),
            _ => None,
        })
    }

    /// Every event in the order the file holds them.
    ///
    /// File order, not chain order — Task 3's reader is what walks the
    /// chain. They agree for a session written by one process, which is
    /// every session today.
    pub fn events(&self, session_id: &str) -> Option<Vec<Event>> {
        Some(
            self.lines(session_id)?
                .into_iter()
                .filter_map(|l| match l {
                    Line::Header(_) => None,
                    Line::Message {
                        id,
                        parent,
                        at,
                        role,
                        parts,
                    } => Some(Event {
                        id,
                        parent,
                        at,
                        body: EventBody::Message { role, parts },
                    }),
                    Line::Title {
                        id,
                        parent,
                        at,
                        title,
                    } => Some(Event {
                        id,
                        parent,
                        at,
                        body: EventBody::Title { title },
                    }),
                    Line::Closed { id, parent, at } => Some(Event {
                        id,
                        parent,
                        at,
                        body: EventBody::Closed,
                    }),
                })
                .collect(),
        )
    }

    pub fn is_closed(&self, session_id: &str) -> bool {
        self.events(session_id)
            .map(|evs| evs.iter().any(|e| matches!(e.body, EventBody::Closed)))
            .unwrap_or(false)
    }

    /// What the listing needs about one session, oldest first.
    ///
    /// Scoped to a namespace rather than global: another namespace's
    /// sessions are not this caller's to see. The ACP handler checks
    /// the boundary too — this is the store declining to hand them over
    /// in the first place.
    ///
    /// Everything here comes out of the one read the header already
    /// requires, which is why `has_messages` and `title` are computed
    /// eagerly rather than left to per-session follow-up calls.
    pub fn list_summaries(&self, namespace: &str) -> Vec<SessionSummary> {
        let dir = self.base_dir.join(namespace).join(KIND);
        let Ok(entries) = std::fs::read_dir(&dir) else {
            return Vec::new();
        };
        let mut out: Vec<SessionSummary> = entries
            .flatten()
            .filter_map(|e| {
                let stem = e.path().file_stem()?.to_str()?.to_string();
                self.summary(&stem)
            })
            .collect();
        out.sort_by_key(|s| s.header.created_at);
        out
    }

    /// One session's listing row. Public because `session/load` and
    /// `session/resume` need `cwd`, `title` and `is_closed` for a single
    /// session without paying for the whole namespace's listing.
    pub fn summary(&self, session_id: &str) -> Option<SessionSummary> {
        let lines = self.lines(session_id)?;
        let mut header = None;
        let mut title = None;
        let mut has_messages = false;
        let mut is_closed = false;
        for line in lines {
            match line {
                Line::Header(h) => header = Some(h),
                Line::Message { .. } => has_messages = true,
                // Last title wins: a session can be retitled.
                Line::Title { title: t, .. } => title = Some(t),
                Line::Closed { .. } => is_closed = true,
            }
        }
        Some(SessionSummary {
            header: header?,
            title,
            has_messages,
            is_closed,
        })
    }
}

/// One row of the session list.
#[derive(Debug, Clone, PartialEq)]
pub struct SessionSummary {
    pub header: SessionHeader,
    pub title: Option<String>,
    /// False for a session the editor opened and never typed into.
    pub has_messages: bool,
    pub is_closed: bool,
}
```

- [ ] **Step 6: `collect_session_files` を公開する**

`src/session.rs` の `fn collect_session_files` を `pub(crate) fn` にする。走査の仕組みは共有する、というのがこの設計の要点。

`src/main.rs` に `mod acp_session;` を足す。

- [ ] **Step 7: テストが通ることを確認**

Run: `cargo test -p sapphire-agent acp_session`
Expected: PASS（7テスト）。

- [ ] **Step 8: コミット**

```bash
git checkout -- Cargo.lock
git add src/acp_session.rs src/session.rs src/main.rs
git commit -m "feat(sessions): an ACP session store whose events carry id and parent"
```

---

### Task 3: ACP セッションストア — 読み出し

parent 鎖を辿って `Vec<ChatMessage>` を組み立てる。ツール結果はキャッシュから戻し、無いものはプレースホルダにする。分岐は警告して直系だけを採る。

**Files:**
- Modify: `src/acp_session.rs`
- Test: `src/acp_session.rs` の `mod tests`

**Interfaces:**
- Consumes: Task 2 の `Event` / `EventBody` / `StoredPart` / `AcpSessionStore::events`
- Produces: `AcpSessionStore::history(&self, session_id: &str) -> Option<Vec<ChatMessage>>`

- [ ] **Step 1: 失敗するテストを書く**

Task 2 の `mod tests` に追記する（`store()` ヘルパを共有する）。

```rust
    #[test]
    fn history_comes_back_in_chain_order() {
        let (_d, store) = store();
        store.create("s1", "default", "/p").unwrap();
        store.append_message("s1", &ChatMessage::user("one")).unwrap();
        store.append_message("s1", &ChatMessage::assistant("two")).unwrap();
        store.append_title("s1", "ignored by history").unwrap();
        store.append_message("s1", &ChatMessage::user("three")).unwrap();

        let history = store.history("s1").expect("the session exists");
        let texts: Vec<&str> = history
            .iter()
            .filter_map(|m| match m.parts.first() {
                Some(ContentPart::Text(t)) => Some(t.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(texts, vec!["one", "two", "three"]);
    }

    /// A tool result survives a round trip through the cache.
    #[test]
    fn a_cached_tool_result_is_restored_whole() {
        let (_d, store) = store();
        store.create("s1", "default", "/p").unwrap();
        store
            .append_message("s1", &tool_result_message("c1", "the original output"))
            .unwrap();

        let history = store.history("s1").unwrap();
        assert_eq!(
            history[0].parts[0],
            ContentPart::ToolResult {
                tool_use_id: "c1".to_string(),
                content: "the original output".to_string(),
            }
        );
    }

    /// The whole reason a resume summary is unnecessary: a lost result
    /// degrades to a sentence, and the `tool_use`/`tool_result` pairing
    /// the API validates stays intact. Losing the cache must never make
    /// a session unloadable.
    #[test]
    fn a_lost_tool_result_becomes_a_placeholder_rather_than_an_error() {
        let (dir, store) = store();
        store.create("s1", "default", "/p").unwrap();
        store
            .append_message("s1", &tool_result_message("c1", "gone tomorrow"))
            .unwrap();

        // Simulate the cache being cleared between runs.
        std::fs::remove_dir_all(dir.path().join("cache")).unwrap();

        let history = store.history("s1").expect("the session still loads");
        assert_eq!(
            history[0].parts[0],
            ContentPart::ToolResult {
                tool_use_id: "c1".to_string(),
                content: MISSING_RESULT.to_string(),
            },
            "the pairing must survive even though the content did not"
        );
    }

    /// Two events claiming the same parent means two writers appended to
    /// one session — the divergence the parent chain exists to catch.
    /// Until there is a merge story, take the first branch and say so
    /// rather than interleaving two conversations by timestamp.
    #[test]
    fn a_branch_is_reported_and_the_first_child_is_taken() {
        let (_d, store) = store();
        store.create("s1", "default", "/p").unwrap();
        store.append_message("s1", &ChatMessage::user("root")).unwrap();

        // Hand-write a second child of the root to forge a divergence.
        let root = store.events("s1").unwrap()[0].id;
        let path = store.path_for_test("s1");
        let forged = serde_json::json!({
            "kind": "message",
            "id": Uuid::now_v7(),
            "parent": root,
            "at": Utc::now(),
            "role": "user",
            "parts": [{"Text": "the other branch"}],
        });
        let mut f = std::fs::OpenOptions::new().append(true).open(&path).unwrap();
        use std::io::Write as _;
        writeln!(f, "{forged}").unwrap();
        drop(f);

        // The legitimate continuation, appended after the forgery, is
        // the file's *last* line but the root's *second* child.
        store.append_message("s1", &ChatMessage::user("mine")).unwrap();

        let history = store.history("s1").expect("a branched session still loads");
        let texts: Vec<&str> = history
            .iter()
            .filter_map(|m| match m.parts.first() {
                Some(ContentPart::Text(t)) => Some(t.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(
            texts,
            vec!["root", "the other branch"],
            "one branch, not both interleaved"
        );
    }

    /// An event whose parent is not in the file — a truncated sync, a
    /// hand-edit — must not silently drop the rest of the conversation
    /// or spin. Everything reachable from the root is returned.
    #[test]
    fn an_orphan_event_is_skipped_rather_than_ending_the_walk() {
        let (_d, store) = store();
        store.create("s1", "default", "/p").unwrap();
        store.append_message("s1", &ChatMessage::user("root")).unwrap();

        let path = store.path_for_test("s1");
        let orphan = serde_json::json!({
            "kind": "message",
            "id": Uuid::now_v7(),
            "parent": Uuid::now_v7(),
            "at": Utc::now(),
            "role": "user",
            "parts": [{"Text": "unreachable"}],
        });
        let mut f = std::fs::OpenOptions::new().append(true).open(&path).unwrap();
        use std::io::Write as _;
        writeln!(f, "{orphan}").unwrap();
        drop(f);

        let history = store.history("s1").expect("the session still loads");
        assert_eq!(history.len(), 1, "the orphan is not reachable from the root");
    }

    fn tool_result_message(tool_use_id: &str, content: &str) -> ChatMessage {
        ChatMessage {
            role: Role::User,
            parts: vec![ContentPart::ToolResult {
                tool_use_id: tool_use_id.to_string(),
                content: content.to_string(),
            }],
            input_kind: None,
            user_id: None,
        }
    }
```

- [ ] **Step 2: テストが落ちることを確認**

Run: `cargo test -p sapphire-agent acp_session`
Expected: FAIL — `history` が未定義。

- [ ] **Step 3: 実装を書く**

```rust
impl AcpSessionStore {
    /// The conversation as the model should see it.
    ///
    /// Walks the parent chain rather than trusting file order, because
    /// file order is only accidentally right: it agrees with the chain
    /// for a session one process wrote, and says nothing useful for one
    /// that was synced or merged.
    pub fn history(&self, session_id: &str) -> Option<Vec<ChatMessage>> {
        let events = self.events(session_id)?;

        // parent -> children, so the walk is a lookup rather than a scan.
        let mut children: HashMap<Option<Uuid>, Vec<&Event>> = HashMap::new();
        for event in &events {
            children.entry(event.parent).or_default().push(event);
        }

        let mut out = Vec::new();
        let mut cursor = None;
        let mut seen = 0usize;
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
            if let EventBody::Message { role, parts } = &event.body {
                out.push(ChatMessage {
                    role: role.clone(),
                    parts: parts.iter().map(|p| self.load_part(p)).collect(),
                    input_kind: None,
                    user_id: None,
                });
            }
            cursor = Some(event.id);

            // A hand-edited or partially-synced file could contain a
            // cycle. The chain cannot be longer than the file.
            seen += 1;
            if seen > events.len() {
                warn!("ACP session {session_id}: the parent chain cycles; stopping");
                break;
            }
        }
        Some(out)
    }

    fn load_part(&self, part: &StoredPart) -> ContentPart {
        match part {
            StoredPart::Text(t) => ContentPart::Text(t.clone()),
            StoredPart::ToolUse { id, name, input } => ContentPart::ToolUse {
                id: id.clone(),
                name: name.clone(),
                input: input.clone(),
            },
            StoredPart::ToolResultRef {
                tool_use_id,
                sha256,
            } => ContentPart::ToolResult {
                tool_use_id: tool_use_id.clone(),
                // A miss is expected, not exceptional: the cache lives
                // outside the workspace and is not synced.
                content: self
                    .cache
                    .get(sha256)
                    .unwrap_or_else(|| MISSING_RESULT.to_string()),
            },
            StoredPart::Other => {
                ContentPart::Text("[a message part that this version does not store]".to_string())
            }
        }
    }
}
```

**注意:** `ContentPart` に `PartialEq` が要る（テストが等値比較する）。現在は
`#[derive(Debug, Clone, Serialize, Deserialize)]` なので `PartialEq` を足す。

- [ ] **Step 4: テストが通ることを確認**

Run: `cargo test -p sapphire-agent acp_session`
Expected: PASS（12テスト）。

- [ ] **Step 5: コミット**

```bash
git checkout -- Cargo.lock
git add src/acp_session.rs src/provider/mod.rs
git commit -m "feat(sessions): read an ACP session by walking its parent chain"
```

---

### Task 4: ターン実行を新ストアに接続する

`run_llm_turn` が ACP セッションでは新ストアに書くようにする。`summarize_all_sessions` の**要約部分だけ**を落とす。

**Files:**
- Modify: `src/serve/mod.rs`
- Modify: `src/main.rs`
- Test: `src/serve/mod.rs` の `mod tests`

**Interfaces:**
- Consumes: Task 2/3 の `AcpSessionStore`
- Produces: `ServeState.acp_session_store: Arc<AcpSessionStore>`、`ServeState.acp_sessions: tokio::sync::Mutex<HashSet<String>>`、`ServeState::is_acp(&self, session_id: &str) -> bool`

#### この Task が消すコードと、消さないコードの厳密な区別

一度間違えている箇所なので、対象を名指しする。

**消す:** `src/serve/mod.rs` の `summarize_all_sessions` の中の

```rust
if let Err(e) = store.append_summary(&session_id, &summary) {
    warn!("Failed to persist shutdown summary for {session_id}: {e}");
}
```

この3行だけ。関数は `digest_all_sessions` に改名して残す。

**関数ごと消さない理由:** 同じ関数の直後で `append_intraday_digest` を呼んでおり、それが `cross_device` ストアの**唯一の**ダイジェスト生成源。関数ごと消すと `build_today_digest_for_namespace`（`src/periodic_log.rs:1166`）が組み立てる「今日の横断メモ」から `/rpc` セッション分が丸ごと落ちる。spec は「サマリーは廃止、**ダイジェストは保持**」と言っているので、関数ごとの削除は spec 自身に反する。

**触らない:** `src/agent.rs` の `:211` `:257` `:653`（要約）と `:265` `:459`（ダイジェスト）。あれは `channel_session_store`（`src/main.rs:691` の `Agent::new` に `Arc::clone(&channel_session_store)` が渡っている）＝ Matrix / Discord。この plan の対象外。

- [ ] **Step 1: 失敗するテストを書く**

```rust
    /// An ACP session's messages go to the ACP store, and the shared
    /// `/rpc` store never sees them. This is the whole point of the
    /// branch: an editor's thread list should not be showing Matrix
    /// conversations, and the two formats must be free to drift.
    #[tokio::test]
    async fn an_acp_session_persists_to_the_acp_store() {
        let (state, _tmp) = test_state().await;
        let sid = "acp-1".to_string();
        state.acp_sessions.lock().await.insert(sid.clone());
        state
            .acp_session_store
            .create(&sid, "default", "/work")
            .unwrap();

        run_llm_turn(
            Arc::clone(&state),
            sid.clone(),
            ChatMessage::user("hello"),
            Arc::new(NoopHost),
            None,
        )
        .await;

        let history = state
            .acp_session_store
            .history(&sid)
            .expect("the ACP store has the session");
        assert!(
            history.iter().any(|m| matches!(
                m.parts.first(),
                Some(ContentPart::Text(t)) if t == "hello"
            )),
            "the user message is in the ACP store"
        );
        assert!(
            state.cross_device_session_store.load_session(&sid).is_none(),
            "the /rpc store must not have been touched"
        );
    }

    /// A session nobody registered as ACP still behaves exactly as it
    /// did — the `/rpc` path is unchanged by this branch.
    #[tokio::test]
    async fn an_rpc_session_still_persists_to_the_rpc_store() {
        let (state, _tmp) = test_state().await;
        let sid = "rpc-1".to_string();

        run_llm_turn(
            Arc::clone(&state),
            sid.clone(),
            ChatMessage::user("hello"),
            Arc::new(NoopHost),
            None,
        )
        .await;

        assert!(
            state.cross_device_session_store.load_session(&sid).is_some(),
            "the /rpc store still receives non-ACP sessions"
        );
    }
```

`test_state()` と `NoopHost` は既存のテストヘルパ。`src/serve/mod.rs` の `mod tests` が `cross_device_session_store: Arc::new(SessionStore::new(base.join("sessions"), "rpc"))` を組み立てている箇所（`:2752` 付近）がそれ。名前が違えば現物に合わせること。**その組み立てに `acp_session_store` と `acp_sessions` の初期化を足す。**

- [ ] **Step 2: テストが落ちることを確認**

Run: `cargo test -p sapphire-agent serve`
Expected: FAIL — `acp_session_store` フィールドが無い。

- [ ] **Step 3: `ServeState` にフィールドを足す**

`src/serve/mod.rs` の `pub struct ServeState` に。

```rust
    /// Sessions an ACP client owns. Separate from the `/rpc` store
    /// because ACP is an externally-defined standard that will drift
    /// from the format we chose for ourselves.
    pub(crate) acp_session_store: Arc<AcpSessionStore>,
    /// Which session ids belong to ACP.
    ///
    /// Registered by the ACP adapter at `session/new` and on adopt.
    /// A set rather than a field on the session, because `run_llm_turn`
    /// is reached from several transports and only one of them knows
    /// the answer.
    pub(crate) acp_sessions: tokio::sync::Mutex<HashSet<String>>,
```

コンストラクタ（`ServeState::new`、`:161` 付近）に引数 `acp_session_store: Arc<AcpSessionStore>` を足し、`acp_sessions: tokio::sync::Mutex::new(HashSet::new())` で初期化する。`src/main.rs` の呼び出し側で組み立てる。

```rust
let tool_result_cache = crate::tool_result_cache::ToolResultCache::open(
    crate::tool_result_cache::ToolResultCache::default_dir()
        .ok_or_else(|| anyhow::anyhow!("no cache directory available"))?,
)?;
let acp_session_store = Arc::new(crate::acp_session::AcpSessionStore::new(
    sessions_dir.clone(),
    tool_result_cache,
));
```

`sessions_dir` は `cross_device_session_store` の構築に使っているのと同じディレクトリ（`SessionStore::new(<dir>, "rpc")` の第1引数）。新ストアは kind が `acp` なので、同じ base の下に別ツリーを作る。

ヘルパを足す。

```rust
impl ServeState {
    pub(crate) async fn is_acp(&self, session_id: &str) -> bool {
        self.acp_sessions.lock().await.contains(session_id)
    }
}
```

- [ ] **Step 4: `run_llm_turn` を分岐させる**

5箇所を変える。**それ以外は触らない。**

(a) 履歴の読み込み（`// 1. Load or lazy-hydrate in-memory history`）:

```rust
    let is_acp = state.is_acp(&session_id).await;
    let mut history: Vec<ChatMessage> = {
        let mut sessions = state.sessions.lock().await;
        sessions
            .entry(session_id.clone())
            .or_insert_with_key(|sid| {
                if is_acp {
                    state.acp_session_store.history(sid).unwrap_or_default()
                } else {
                    store.load_session(sid).unwrap_or_default()
                }
            })
            .clone()
    };
```

(b) `ensure_session` のブロック（`// 2b.`、`:1945` 付近）を ACP では飛ばす。ヘッダは `session/new` の時点で書かれているので、ここで作るものは無い。

```rust
    if !is_acp && Arc::ptr_eq(&store, &state.cross_device_session_store) {
        // ... 中身は既存のまま
    }
```

`pending_cwd` の取得はこのブロックの中にあり、Task 5 でフィールドごと消える。

(c) メッセージの追記。`run_llm_turn` の中の `store.append(&session_id, ...)` を**すべて**次の形にする。`grep -n "store.append(" src/serve/mod.rs` で全箇所を出すこと（ユーザメッセージとアシスタント応答の少なくとも2箇所ある）。

```rust
    if is_acp {
        if let Err(e) = state.acp_session_store.append_message(&session_id, &user_msg) {
            warn!("Failed to persist user message: {e}");
        }
    } else if let Err(e) = store.append(&session_id, &user_msg) {
        warn!("Failed to persist user message: {e}");
    }
```

(d) 圧縮要約の永続化を ACP では飛ばす（`:2021` 付近）。

```rust
            Ok(Some(result)) => {
                history = result.compressed;
                // ACP sessions do not persist a compaction summary: the
                // full event history is re-read from the store on
                // reload, so a stored summary would only be a second,
                // staler answer to a question the events already
                // answer. Compression stays an in-memory optimisation.
                if !is_acp
                    && let Err(e) = store.append_summary(&session_id, &result.summary)
                {
                    warn!("Failed to persist compaction summary: {e}");
                }
            }
```

(e) タイトル設定（`:1488` と `:2265` の `state2.store_for_session(&sid).set_title(&sid, &title)`）。両方を分岐させる。現物は `&&` チェーンの中にあるので、素直に `if let` に開いてよい。

```rust
    let result = if state2.is_acp(&sid).await {
        state2.acp_session_store.append_title(&sid, &title)
    } else {
        state2.store_for_session(&sid).set_title(&sid, &title)
    };
    if let Err(e) = result {
        // 既存の warn! をそのまま
    }
```

- [ ] **Step 5: 再開用要約を落とす**

`summarize_all_sessions` を改名し、`append_summary` の呼び出しを削り、ACP セッションを除外する。

```rust
/// Publish "what this session covered today" for every in-memory API
/// session, so other rooms can pick it up through the cross-session
/// digest block in their system prompt.
///
/// This used to also append a `SummaryLine` for the next process to
/// resume from. It no longer does: a session's history is restored from
/// its own events, and a tool result that has fallen out of the cache
/// degrades to a placeholder — cheaper and more faithful than paying
/// for a model call that throws the turn structure away.
async fn digest_all_sessions(state: &Arc<ServeState>) {
    let snapshot: Vec<(String, Vec<ChatMessage>)> = {
        let sessions = state.sessions.lock().await;
        let acp = state.acp_sessions.lock().await;
        sessions
            .iter()
            .filter(|(sid, msgs)| msgs.len() >= 2 && !acp.contains(*sid))
            .map(|(sid, msgs)| (sid.clone(), msgs.clone()))
            .collect()
    };
    if snapshot.is_empty() {
        return;
    }
    info!(
        "Graceful shutdown: digesting {} RPC session(s)",
        snapshot.len()
    );
    for (session_id, messages) in snapshot {
        let provider = state.provider_for_session(&session_id).await;
        let store = state.store_for_session(&session_id);
        match generate_summary(&*provider, &messages).await {
            Ok(summary) if !summary.trim().is_empty() => {
                if let Err(e) = store.append_intraday_digest(&session_id, &summary, None) {
                    warn!("Failed to persist shutdown intra-day digest for {session_id}: {e}");
                }
            }
            Ok(_) => warn!("Shutdown digest for {session_id} was empty; skipping"),
            Err(e) => warn!("Shutdown digest generation failed for {session_id}: {e:#}"),
        }
    }
}
```

`:395` の呼び出しを `digest_all_sessions(&shutdown_state).await;` に変える。

**ここで ACP セッションを除外する理由:** 除外は「ACP をダイジェストしない」という意味では**ない**。ACP のダイジェストは自分のストアにイベントとして書かれ、それを Task 6 が足す。この関数は `SessionStore::append_intraday_digest` を呼ぶので、新ストアのセッションを渡すと書き込み先が無い。Task 6 でこの関数に ACP 用の分岐が入る。

**このステップのテストで「ACP がダイジェストされないこと」を assert しないこと。** Task 6 がそれを覆す。ここで確かめるのは `/rpc` の経路が壊れていないことだけ。

- [ ] **Step 6: テストが通ることを確認**

Run: `cargo test -p sapphire-agent`
Expected: PASS。

赤が出る場合、それは「ACP セッションが `/rpc` ストアに入ることを前提にしていたテスト」のはずで、Task 5 でアダプタを切り替えるまで残りうる。**その場合はテストが検証している性質が新設計でも成り立つかを判断し、成り立つなら新ストアを見るように書き換える。単に削除しない。**

- [ ] **Step 7: コミット**

```bash
git checkout -- Cargo.lock
git add src/serve/mod.rs src/main.rs
git commit -m "feat(serve): route ACP turns to the ACP store; drop the resume summary"
```

---

### Task 5: ACP アダプタを切り替え、`cwd` の間借りを解く

ハンドラが新ストアを見るようにし、PR #188 が `SessionMeta` に足した `cwd` とその配管を消す。

**Files:**
- Modify: `src/serve/acp.rs`
- Modify: `src/session.rs`
- Modify: `src/serve/mod.rs`
- Test: `src/serve/acp.rs` の `mod tests`

**Interfaces:**
- Consumes: Task 2/3/4 のすべて
- Produces: `AcpSessionStore::absolute_path_for(&self, session_id: &str) -> Option<PathBuf>`

#### この Task が消すコード

すべて **PR #188 がこのブランチで足したもの**で、`main` には無い。既存4ストアの形式は変わらない。

| 消すもの | 場所 | 置き換え |
|---|---|---|
| `SessionMeta.cwd: Option<String>` | `src/session.rs` | `acp_session::SessionHeader.cwd`（必須項目） |
| `SessionStore::session_header` | `src/session.rs` | `AcpSessionStore::summary` |
| `SessionStore::list_session_headers` | `src/session.rs` | `AcpSessionStore::list_summaries` |
| `ServeState.pending_cwd` | `src/serve/mod.rs` | 不要（ヘッダを `session/new` で書く） |
| `SessionStore::ensure_session` の `cwd` 引数 | `src/session.rs` | 同上 |

**`Workspace::root()` は条件付き。** spec は消すと書いたが、`grep -rn "\.root()" src/` で確認し、**ACP の cwd フォールバック以外に呼び出しが無い場合に限り**消す。あれば残す。

- [ ] **Step 1: 失敗するテストを書く**

```rust
    /// The listing shows ACP sessions and nothing else. An editor's
    /// thread list must not contain the user's Matrix conversations —
    /// that mixing is what this branch exists to undo.
    #[tokio::test]
    async fn the_listing_shows_only_acp_sessions() {
        let (state, _tmp) = acp_test_state().await;

        state
            .acp_session_store
            .create("acp-1", "default", "/work")
            .unwrap();
        state
            .acp_session_store
            .append_message("acp-1", &ChatMessage::user("hi"))
            .unwrap();

        // A conversation from the shared store, which must not appear.
        state
            .cross_device_session_store
            .ensure_session(
                "rpc-1",
                &("rpc-1".to_string(), None),
                "rpc",
                None,
                "default",
            )
            .unwrap();

        let ids = list_session_ids(&state, "default").await;
        assert_eq!(ids, vec!["acp-1".to_string()]);
    }

    /// A thread the editor opened and never typed into is not a
    /// conversation. Zed calls `session/new` on every panel open, so
    /// listing those would bury the real ones.
    #[tokio::test]
    async fn an_empty_session_is_not_listed() {
        let (state, _tmp) = acp_test_state().await;
        state
            .acp_session_store
            .create("never-used", "default", "/work")
            .unwrap();

        assert!(list_session_ids(&state, "default").await.is_empty());
    }
```

`acp_test_state` と `list_session_ids` は既存の `session/list` テスト（`:2706` 以降）が使っている組み立てと呼び出しに合わせる。共通部分が無ければヘルパに括り出す。`ensure_session` の引数は Step 3 で `cwd` を落とした後の5引数版で書いてある。

- [ ] **Step 2: テストが落ちることを確認**

Run: `cargo test -p sapphire-agent acp`
Expected: FAIL。

- [ ] **Step 3: ハンドラを切り替える**

**ハンドラの構造・拒否文言・namespace 境界・接続カウンタは変えない。**変わるのは呼ぶストアだけ。

- `session/new`: id を発行したら `state.acp_session_store.create(&id, &namespace, &cwd)` を呼び、`state.acp_sessions.lock().await.insert(id.clone())` で登録する。`state.pending_cwd` への挿入を消す。
- `session/list`（`:761` の `state.cross_device_session_store.list_sessions()`）: `state.acp_session_store.list_summaries(&namespace)` に置き換え、`.filter(|s| s.has_messages)` を掛ける。**ハンドラ側の namespace 比較は残す**（ストアも絞るが、二重の防御は意図的）。
- `session/load` と `session/resume`（`:577` の `session_header`、`:857` の `load_session`）: `state.acp_session_store.summary(&id)` と `state.acp_session_store.history(&id)` に置き換える。adopt したら `state.acp_sessions.lock().await.insert(id.clone())`。

  **`refuse()` クロージャは一字も変えない。** id を差し込まないことが列挙を防いでいる性質で、テストが拒否ペイロードの等値比較で押さえている。
- `:789` の `store.absolute_path_for(&meta.session_id)`: 新ストアに同等を足す。

  ```rust
  impl AcpSessionStore {
      pub fn absolute_path_for(&self, session_id: &str) -> Option<PathBuf> {
          self.find(session_id)
      }
  }
  ```

  `find` は Task 2 で書いた private ヘルパ。
- `SessionInfo.cwd` は `summary.header.cwd` から取る。必須項目になったので `Workspace::root()` フォールバックは要らなくなる。

- [ ] **Step 4: `cwd` の配管を消す**

上の表のとおり。`cargo test -p sapphire-agent` を回しながら、コンパイルエラーが指す箇所を順に落とす。`SessionMeta` の `Option` フィールドは4つに戻る。

- [ ] **Step 5: 既存 ACP テストを新ストアに向ける**

`:2706` 以降の ACP テスト群は `cross_device_session_store` を組み立てている。**各テストが検証している性質——namespace 境界、拒否の区別不能性、replay の順序、接続カウンタ——は新設計でもそのまま成り立つ。ストアの差し替えだけで通るはず。通らないなら性質が変わっているので、消さずに理由を報告すること。**

Run: `cargo test -p sapphire-agent`
Expected: PASS。

- [ ] **Step 6: フォーマットと lint**

```bash
cargo fmt --all
cargo clippy --workspace --all-targets -- -D warnings
```

- [ ] **Step 7: コミット**

```bash
git checkout -- Cargo.lock
git add src/serve/acp.rs src/session.rs src/serve/mod.rs
git commit -m "refactor(acp): serve ACP sessions from their own store"
```

---

### Task 6: ACP セッションを横断ダイジェストに載せる

「今日エージェントと何をしたか」が他の部屋のシステムプロンプトにも出るようにする。ダイジェストはワークスペース外のキャッシュに置き、定期的に作り直す。

**Files:**
- Create: `src/digest_cache.rs`
- Modify: `src/acp_session.rs`
- Modify: `src/serve/mod.rs`
- Modify: `src/periodic_log.rs`
- Modify: `src/main.rs`
- Test: `src/digest_cache.rs`、`src/acp_session.rs`、`src/periodic_log.rs` の `mod tests`

**Interfaces:**
- Consumes: Task 2 の `SessionHeader` / `Event` / `EventBody`、Task 4 の `digest_all_sessions`
- Produces: `DigestCache::open(PathBuf) -> Result<Arc<Self>>`、`DigestCache::default_dir() -> Option<PathBuf>`、`DigestCache::put(&self, session_id: &str, digest: &str, since: Option<DateTime<Utc>>) -> Result<()>`、`DigestCache::get(&self, session_id: &str) -> Option<IntradayDigestLine>`、`DigestCache::prune_before(&self, cutoff: DateTime<Utc>) -> usize`、`AcpSessionStore::intraday_digests_for_day(&self, date, boundary_hour, cache: &DigestCache) -> Vec<(SessionMeta, IntradayDigestLine)>`、`AcpSessionStore::sessions_needing_digest(&self, cache: &DigestCache, date: NaiveDate, boundary_hour: u8) -> Vec<String>`

#### 直前のタスクが header() を削除したので、前提が一つ変わっている

直前のタスクが `AcpSessionStore::header` と `is_closed` を削除した——`summary()` が
`SessionHeader` と `is_closed` の両方を含む上位互換で、実際に呼ぶ側がいなかったため。
**したがってこの Task は `header()` ではなく `summary()` を使う。** `summary()` は
`title` も返すので、`project_meta` がイベントを走査してタイトルを拾う必要も無くなり、
ファイルの読み出しが1回で済む。

#### ダイジェストをストアのイベントにしない理由

**セッション JSONL は retrieve の索引に入っている。** `SessionStore.base_dir` は
`<workspace>/sessions` で、`notify_updated` が `ws_state` に変更を通知している。
ダイジェストをイベントとして追記すると、**似た内容の要約が同じファイルに何度も
積まれ、検索結果を歪める。** 開発のようにターンの長いやりとりでは特に効く。

刈り取りで対処することもできない。イベントは `parent` 鎖で繋がっているので、
途中の要素を消すと子が孤児になる。**append-only とダイジェストの刈り取りは両立しない。**

したがってキャッシュに置く。spec の当初の判断（「ワークスペース外へ」）が正しかった。

**メタデータの二重化は起きない。** キャッシュが持つのは本文と時刻だけで、
`namespace` / `created_at` / `title` は読み出し時に**ストアのヘッダから結合する**。
キャッシュは `session_id` をキーに**最新の1件だけ**を保持するので、上書きによって
そもそも溜まらない。日跨ぎの取りこぼしのために `prune_before` も用意する。

**キャッシュは ACP 専用にしない。** キーはセッション id だけで、ACP を知らない。
[#189](https://github.com/fluo10/sapphire-agent/issues/189) で `/rpc` を同じ扱いに
するなら、そのまま使い回せる。

#### 生成のタイミング

**「30分ごとに定期確認し、前回のダイジェスト以降にイベントが増えたセッションだけ
作り直す」。** アイドル時間で測らないのは、更新のタイミングが予測しづらくなるため。
定期確認なら「次は毎時0分と30分」と分かる。

判定に in-memory の地図は要らない。前回いつダイジェストしたかはキャッシュの
`digest_at` にあり、最後にイベントが増えた時刻はストアの中にある。**再起動を
跨いでも正しく動く**点で、`src/agent.rs` の `last_activity_at` / `last_flushed_at`
方式より素直になる。

- [ ] **Step 1: 失敗するテストを書く（キャッシュ）**

```rust
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
        cache.put_at("fresh", "today's work", None, cutoff + chrono::Duration::hours(1))
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
```

- [ ] **Step 2: テストが落ちることを確認**

Run: `cargo test -p sapphire-agent digest_cache`
Expected: FAIL — `DigestCache` が未定義。

- [ ] **Step 3: キャッシュを実装する**

```rust
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
```

`src/main.rs` に `mod digest_cache;` を足す。

- [ ] **Step 4: テストが通ることを確認**

Run: `cargo test -p sapphire-agent digest_cache`
Expected: PASS（5テスト）。

- [ ] **Step 5: 失敗するテストを書く（ストア側の結合と判定）**

`src/acp_session.rs` の `mod tests` に追記する。

```rust
    fn digest_cache(dir: &tempfile::TempDir) -> Arc<crate::digest_cache::DigestCache> {
        crate::digest_cache::DigestCache::open(dir.path().join("digests")).unwrap()
    }

    /// The digest's text comes from the cache; its namespace, title and
    /// creation time come from the store's header. Neither side
    /// duplicates the other, which is what makes an external cache
    /// affordable here.
    #[test]
    fn a_digest_is_joined_against_the_stores_header() {
        let (dir, store) = store();
        let cache = digest_cache(&dir);
        store.create("s1", "work", "/p").unwrap();
        store.append_message("s1", &ChatMessage::user("did a thing")).unwrap();
        store.append_title("s1", "parser hunt").unwrap();
        cache.put("s1", "we fixed the parser", None).unwrap();

        let today = chrono::Local::now().date_naive();
        let found = store.intraday_digests_for_day(today, 4, &cache);
        assert_eq!(found.len(), 1);
        let (meta, digest) = &found[0];
        assert_eq!(digest.digest, "we fixed the parser");
        assert_eq!(meta.session_id, "s1");
        assert_eq!(
            meta.channel, "acp",
            "the projection says which store this came from — \
             build_today_digest_for_namespace routes on it"
        );
        assert_eq!(meta.namespace.as_deref(), Some("work"));
        assert_eq!(meta.title.as_deref(), Some("parser hunt"));
    }

    /// Losing the cache must not lose the session.
    #[test]
    fn a_session_with_no_cached_digest_is_simply_absent() {
        let (dir, store) = store();
        let cache = digest_cache(&dir);
        store.create("s1", "work", "/p").unwrap();
        store.append_message("s1", &ChatMessage::user("x")).unwrap();

        let today = chrono::Local::now().date_naive();
        assert!(store.intraday_digests_for_day(today, 4, &cache).is_empty());
        assert!(store.history("s1").is_some(), "the session still loads");
    }

    /// The scheduling rule, stated as a test: a session whose newest
    /// message post-dates its cached digest is due for a new one. Not
    /// "has been idle for N minutes" — the sweep runs on a fixed
    /// cadence, so the only question is whether anything was added.
    #[test]
    fn a_session_with_events_newer_than_its_digest_is_due() {
        let (dir, store) = store();
        let cache = digest_cache(&dir);
        store.create("s1", "default", "/p").unwrap();
        cache
            .put_at("s1", "covered up to here", None, Utc::now() - chrono::Duration::hours(1))
            .unwrap();
        store.append_message("s1", &ChatMessage::user("something new")).unwrap();

        let today = chrono::Local::now().date_naive();
        assert_eq!(
            store.sessions_needing_digest(&cache, today, 4),
            vec!["s1".to_string()]
        );
    }

    #[test]
    fn a_session_with_nothing_new_since_its_digest_is_not_due() {
        let (dir, store) = store();
        let cache = digest_cache(&dir);
        store.create("s1", "default", "/p").unwrap();
        store.append_message("s1", &ChatMessage::user("said something")).unwrap();
        cache.put("s1", "covered", None).unwrap();

        let today = chrono::Local::now().date_naive();
        assert!(store.sessions_needing_digest(&cache, today, 4).is_empty());
    }

    #[test]
    fn a_never_digested_session_with_messages_is_due() {
        let (dir, store) = store();
        let cache = digest_cache(&dir);
        store.create("s1", "default", "/p").unwrap();
        store.append_message("s1", &ChatMessage::user("first words")).unwrap();

        let today = chrono::Local::now().date_naive();
        assert_eq!(
            store.sessions_needing_digest(&cache, today, 4),
            vec!["s1".to_string()]
        );
    }

    /// A session that has said nothing today is not resurrected. Its
    /// day has been written up already; digesting it now would file old
    /// work under today.
    #[test]
    fn a_session_with_no_messages_today_is_not_due() {
        let (dir, store) = store();
        let cache = digest_cache(&dir);
        store.create("s1", "default", "/p").unwrap();
        store.append_message("s1", &ChatMessage::user("old news")).unwrap();

        let tomorrow = chrono::Local::now().date_naive() + chrono::Duration::days(1);
        assert!(store.sessions_needing_digest(&cache, tomorrow, 4).is_empty());
    }
```

- [ ] **Step 6: テストが落ちることを確認**

Run: `cargo test -p sapphire-agent acp_session`
Expected: FAIL — `intraday_digests_for_day` が未定義。

- [ ] **Step 7: 結合と判定を実装する**

```rust
impl AcpSessionStore {
    /// Today's digest per session, in the vocabulary the cross-session
    /// digest builder already speaks.
    ///
    /// The text comes from `cache`; everything around it comes from the
    /// session's header. The projection is read-only and deliberate:
    /// `build_today_digest_for_namespace` is shared with four other
    /// stores, and teaching it a second shape would spread this store's
    /// format into code with no business knowing it.
    pub fn intraday_digests_for_day(
        &self,
        date: NaiveDate,
        boundary_hour: u8,
        cache: &crate::digest_cache::DigestCache,
    ) -> Vec<(SessionMeta, IntradayDigestLine)> {
        let (day_start, day_end) = crate::session::day_window(date, boundary_hour);
        let mut out = Vec::new();
        for session_id in self.all_session_ids() {
            let Some(digest) = cache.get(&session_id) else {
                continue;
            };
            if digest.digest_at < day_start || digest.digest_at >= day_end {
                continue;
            }
            let Some(summary) = self.summary(&session_id) else {
                continue;
            };
            out.push((self.project_meta(&summary), digest));
        }
        out.sort_by_key(|(meta, _)| meta.created_at);
        out
    }

    /// Sessions whose newest message post-dates their cached digest, and
    /// which said something inside `date`'s window.
    ///
    /// Deliberately not an idle threshold. The sweep runs on a fixed
    /// cadence, so the only question worth asking is whether anything
    /// was added since last time — which makes the next update time
    /// predictable instead of a function of when the user stopped
    /// typing. Both sides of the comparison are durable (a cache file
    /// and a session event), so a restart does not reset the schedule.
    pub fn sessions_needing_digest(
        &self,
        cache: &crate::digest_cache::DigestCache,
        date: NaiveDate,
        boundary_hour: u8,
    ) -> Vec<String> {
        let (day_start, day_end) = crate::session::day_window(date, boundary_hour);
        let mut due = Vec::new();
        for session_id in self.all_session_ids() {
            let Some(events) = self.events(&session_id) else {
                continue;
            };
            let last_message = events
                .iter()
                .filter(|e| matches!(e.body, EventBody::Message { .. }))
                .map(|e| e.at)
                .last();
            let Some(last_message) = last_message else {
                continue;
            };
            // Said nothing today: its day is already written up.
            if last_message < day_start || last_message >= day_end {
                continue;
            }
            if cache
                .get(&session_id)
                .is_some_and(|d| d.digest_at >= last_message)
            {
                continue;
            }
            due.push(session_id);
        }
        due.sort();
        due
    }

    /// This store's sessions in the vocabulary the log and digest
    /// builders share. `channel` is `"acp"` so those builders can route
    /// on it the way they already route on `"rpc"` and
    /// `"device-default"`.
    fn project_meta(&self, summary: &SessionSummary) -> SessionMeta {
        let header = &summary.header;
        SessionMeta {
            session_id: header.session_id.clone(),
            // ACP has no rooms. Empty rather than synthetic, so a
            // room-derived namespace lookup can never accidentally
            // match one.
            room_id: String::new(),
            thread_id: None,
            channel: "acp".to_string(),
            created_at: header.created_at,
            public_id: None,
            namespace: Some(header.namespace.clone()),
            project: None,
            device_id: None,
            room_profile: None,
            title: summary.title.clone(),
        }
    }

    /// Every session id this store holds, across namespaces.
    fn all_session_ids(&self) -> Vec<String> {
        crate::session::collect_session_files(&self.base_dir, KIND)
            .into_iter()
            .filter_map(|p| Some(p.file_stem()?.to_str()?.to_string()))
            .collect()
    }
}
```

`SessionMeta` のフィールド名と型は現物（`src/session.rs:41`）に合わせること。`day_window` は `src/session.rs` の private 関数なので `pub(crate)` にする。`use` に `chrono::NaiveDate` と `crate::session::{IntradayDigestLine, SessionMeta}` を足す。

- [ ] **Step 8: テストが通ることを確認**

Run: `cargo test -p sapphire-agent acp_session`
Expected: PASS（18テスト）。

- [ ] **Step 9: 30分ごとの掃き出しタスクを足す**

`ServeState` に `pub(crate) digest_cache: Arc<DigestCache>` を足し（`src/main.rs` で `DigestCache::open(DigestCache::default_dir()...)` を組み立てて渡す）、`src/serve/mod.rs` にタスクを足す。

```rust
/// Refresh the digest of every ACP session that has grown since it was
/// last digested.
///
/// A fixed cadence rather than an idle timer: the next refresh is at a
/// time the user can predict, and a long agent turn does not decide
/// when it happens. Sessions with nothing new are skipped, so a quiet
/// half hour costs one directory walk and no model calls.
pub(crate) fn spawn_acp_digest_sweep(state: Arc<ServeState>) {
    tokio::spawn(async move {
        let period = std::time::Duration::from_secs(1800);
        loop {
            tokio::time::sleep(period).await;
            let today = chrono::Local::now().date_naive();
            let boundary = state.config.day_boundary_hour;
            let due = state.acp_session_store.sessions_needing_digest(
                &state.digest_cache,
                today,
                boundary,
            );
            for session_id in due {
                let Some(messages) = state.acp_session_store.history(&session_id) else {
                    continue;
                };
                if messages.len() < 2 {
                    continue;
                }
                let provider = state.provider_for_session(&session_id).await;
                match generate_summary(&*provider, &messages).await {
                    Ok(summary) if !summary.trim().is_empty() => {
                        if let Err(e) = state.digest_cache.put(&session_id, &summary, None) {
                            warn!("Failed to cache ACP digest for {session_id}: {e}");
                        }
                    }
                    Ok(_) => warn!("ACP digest for {session_id} was empty; skipping"),
                    Err(e) => warn!("ACP digest generation failed for {session_id}: {e:#}"),
                }
            }
        }
    });
}
```

`axum::serve` を始める直前で `spawn_acp_digest_sweep(Arc::clone(&state));` を呼ぶ。

**空ダイジェストはキャッシュしない。** キャッシュすると `sessions_needing_digest` が「済み」と見なして、次に本文が書ける機会を潰す。上のコードはそうなっている（`Ok(_)` の腕で何も書かない）ので、**変えないこと。**

- [ ] **Step 10: シャットダウン時にも ACP を拾う**

Task 4 で作った `digest_all_sessions` に ACP の分岐を足す。関数は in-memory の `state.sessions` を見ているので、ACP セッションもそこにいる。

```rust
    for (session_id, messages) in snapshot {
        let provider = state.provider_for_session(&session_id).await;
        match generate_summary(&*provider, &messages).await {
            Ok(summary) if !summary.trim().is_empty() => {
                if is_acp.contains(&session_id) {
                    if let Err(e) = state.digest_cache.put(&session_id, &summary, None) {
                        warn!("Failed to cache shutdown digest for {session_id}: {e}");
                    }
                } else if let Err(e) = state
                    .store_for_session(&session_id)
                    .append_intraday_digest(&session_id, &summary, None)
                {
                    warn!("Failed to persist shutdown intra-day digest for {session_id}: {e}");
                }
            }
            Ok(_) => warn!("Shutdown digest for {session_id} was empty; skipping"),
            Err(e) => warn!("Shutdown digest generation failed for {session_id}: {e:#}"),
        }
    }
```

Task 4 が入れた `!acp.contains(*sid)` フィルタを外し、代わりに ACP の id 集合を `is_acp` としてスナップショットと一緒に取る。ログの文言 `"digesting {} RPC session(s)"` は `"digesting {} session(s)"` に直す。

- [ ] **Step 11: 横断ダイジェストの組み立てに ACP を渡す**

`src/periodic_log.rs` の `build_today_digest_for_namespace` と `build_all_today_digests` に引数を2つ足す（`device_default_store` の隣）。

```rust
    acp_store: Option<&AcpSessionStore>,
    digest_cache: Option<&DigestCache>,
```

```rust
    if let (Some(s), Some(c)) = (acp_store, digest_cache) {
        entries.extend(s.intraday_digests_for_day(today, boundary_hour, c));
    }
```

namespace の解決に `"acp"` の腕を足す。ACP はヘッダに namespace を持つので `device-default` と同じ扱いでよい。

```rust
        let is_rpc = meta.channel == "rpc";
        let is_device_default = meta.channel == "device-default";
        let is_acp = meta.channel == "acp";
        let ns = if is_rpc {
            crate::config::DEFAULT_NAMESPACE_NAME.to_string()
        } else if is_device_default || is_acp {
            // These pin their namespace at creation time — honour it
            // directly rather than deriving one from a room id they do
            // not have.
            meta.namespace
                .clone()
                .unwrap_or_else(|| crate::config::DEFAULT_NAMESPACE_NAME.to_string())
        } else {
            room_to_namespace(&meta.room_id)
        };
```

ラベルの分岐も同じ（`room_id` が空なので room ベースのラベルは使えない）。

```rust
        let room_label = if is_rpc || is_device_default || is_acp {
            meta.title
                .clone()
                .unwrap_or_else(|| format!("{}/{}", meta.channel, short_id(&meta.session_id)))
        } else {
            format!("{}/{}", meta.channel, short_id(&meta.room_id))
        };
```

呼び出し側（`grep -rn "build_all_today_digests\|build_today_digest_for_namespace" src/`）に `Some(&state.acp_session_store), Some(&state.digest_cache)` を渡す。`ServeState` に届かない呼び出し側があれば両方 `None`。

- [ ] **Step 12: `build_today_digest_for_namespace` のテストを1本足す**

```rust
    /// An ACP session's digest reaches the room's system prompt, and it
    /// is routed by the namespace in its header rather than by a room id
    /// it does not have.
    #[test]
    fn an_acp_digest_lands_in_its_own_namespace() {
        // 既存の build_today_digest_for_namespace テストの組み立てにならい、
        // AcpSessionStore と DigestCache を tempdir に作り、
        // create / append_message / cache.put してから
        // acp_store: Some(&store), digest_cache: Some(&cache) で呼ぶ。
        //
        // 期待は2つ:
        //   (1) namespace "work" で呼ぶと本文にダイジェストが含まれる
        //   (2) namespace "default" で呼ぶと None
    }
```

**実装者が既存テストの組み立てに合わせて書くこと。** 上のコメントが要件で、確かめる性質は「自分の namespace のブロックに現れる」「別の namespace には現れない」の2つ。

- [ ] **Step 13: テストが通ることを確認**

Run: `cargo test -p sapphire-agent`
Expected: PASS。

- [ ] **Step 14: コミット**

```bash
git checkout -- Cargo.lock
git add src/digest_cache.rs src/acp_session.rs src/serve/mod.rs src/periodic_log.rs src/main.rs
git commit -m "feat(acp): put ACP sessions in the cross-session today digest"
```

---

### Task 7: ACP セッションをデイリーログに載せる

これが恒久的な記録のほう。ダイジェストは今日限りだが、デイリーログはワークスペースに残り検索対象になる。

**Files:**
- Modify: `src/acp_session.rs`
- Modify: `src/periodic_log.rs`
- Modify: `src/heartbeat.rs`
- Test: `src/acp_session.rs` と `src/periodic_log.rs` の `mod tests`

**Interfaces:**
- Consumes: Task 6 の `project_meta` / `all_session_ids`
- Produces: `AcpSessionStore::sessions_for_day(&self, date: NaiveDate, boundary_hour: u8) -> Vec<(SessionMeta, Vec<StoredMessage>)>`、`AcpSessionStore::session_dates(&self, boundary_hour: u8) -> Vec<NaiveDate>`

- [ ] **Step 1: 失敗するテストを書く**

```rust
    /// The daily log is the durable record — the one that stays in the
    /// workspace and stays searchable. An ACP session has to reach it.
    #[test]
    fn a_days_conversation_projects_into_the_daily_log_shape() {
        let (_d, store) = store();
        store.create("s1", "work", "/p").unwrap();
        store.append_message("s1", &ChatMessage::user("what broke?")).unwrap();
        store.append_message("s1", &ChatMessage::assistant("the parser")).unwrap();
        store.append_title("s1", "parser hunt").unwrap();

        let today = chrono::Local::now().date_naive();
        let sessions = store.sessions_for_day(today, 4);
        assert_eq!(sessions.len(), 1);
        let (meta, messages) = &sessions[0];
        assert_eq!(meta.session_id, "s1");
        assert_eq!(meta.channel, "acp");
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[0].role, Role::User);
        assert_eq!(messages[1].role, Role::Assistant);
        assert_eq!(
            messages[0].parts[0],
            ContentPart::Text("what broke?".to_string())
        );
    }

    /// A tool result that has fallen out of the cache must not put a
    /// placeholder sentence into the permanent record. The daily log
    /// wants what was said, and `format_sessions` keeps only text parts —
    /// so tool traffic should not be projected at all.
    #[test]
    fn tool_traffic_is_left_out_of_the_daily_log_projection() {
        let (_d, store) = store();
        store.create("s1", "default", "/p").unwrap();
        store
            .append_message("s1", &tool_result_message("c1", "a big file listing"))
            .unwrap();
        store.append_message("s1", &ChatMessage::user("thanks")).unwrap();

        let today = chrono::Local::now().date_naive();
        let (_, messages) = store.sessions_for_day(today, 4).remove(0);
        let texts: Vec<&str> = messages
            .iter()
            .flat_map(|m| m.parts.iter())
            .filter_map(|p| match p {
                ContentPart::Text(t) => Some(t.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(texts, vec!["thanks"]);
    }

    #[test]
    fn session_dates_lists_the_days_that_have_messages() {
        let (_d, store) = store();
        store.create("s1", "default", "/p").unwrap();
        store.append_message("s1", &ChatMessage::user("x")).unwrap();

        let today = chrono::Local::now().date_naive();
        assert_eq!(store.session_dates(4), vec![today]);
    }
```

- [ ] **Step 2: テストが落ちることを確認**

Run: `cargo test -p sapphire-agent acp_session`
Expected: FAIL — `sessions_for_day` が未定義。

- [ ] **Step 3: 射影を書く**

```rust
impl AcpSessionStore {
    /// This store's sessions for one local day, in the shape
    /// `format_sessions` reads.
    ///
    /// Only text is projected. A tool result whose content has fallen
    /// out of the cache would otherwise write its placeholder sentence
    /// into a permanent, searchable record — and the daily log is a
    /// narrative of what was said, not a transcript of what was read.
    pub fn sessions_for_day(
        &self,
        date: NaiveDate,
        boundary_hour: u8,
    ) -> Vec<(SessionMeta, Vec<StoredMessage>)> {
        let (day_start, day_end) = crate::session::day_window(date, boundary_hour);
        let mut out = Vec::new();
        for session_id in self.all_session_ids() {
            let Some(summary) = self.summary(&session_id) else {
                continue;
            };
            let Some(events) = self.events(&session_id) else {
                continue;
            };
            let messages: Vec<StoredMessage> = events
                .iter()
                .filter(|e| e.at >= day_start && e.at < day_end)
                .filter_map(|e| match &e.body {
                    EventBody::Message { role, parts } => {
                        let text: Vec<ContentPart> = parts
                            .iter()
                            .filter_map(|p| match p {
                                StoredPart::Text(t) if !t.trim().is_empty() => {
                                    Some(ContentPart::Text(t.clone()))
                                }
                                _ => None,
                            })
                            .collect();
                        if text.is_empty() {
                            return None;
                        }
                        Some(StoredMessage {
                            timestamp: e.at,
                            role: role.clone(),
                            parts: text,
                            input_kind: None,
                            user_id: None,
                            report_meta: None,
                        })
                    }
                    _ => None,
                })
                .collect();
            if messages.is_empty() {
                continue;
            }
            out.push((self.project_meta(&summary), messages));
        }
        out.sort_by_key(|(meta, _)| meta.created_at);
        out
    }

    /// Local dates on which this store has at least one message, so
    /// daily-log catch-up knows which days to generate.
    pub fn session_dates(&self, boundary_hour: u8) -> Vec<NaiveDate> {
        let mut dates = std::collections::HashSet::new();
        for session_id in self.all_session_ids() {
            let Some(events) = self.events(&session_id) else {
                continue;
            };
            for event in events {
                if !matches!(event.body, EventBody::Message { .. }) {
                    continue;
                }
                let local = event.at.with_timezone(&chrono::Local);
                dates.insert(crate::session::local_date_for_timestamp(
                    local,
                    boundary_hour,
                ));
            }
        }
        let mut sorted: Vec<NaiveDate> = dates.into_iter().collect();
        sorted.sort();
        sorted
    }
}
```

`local_date_for_timestamp` は `src/session.rs` の private 関数。`pub(crate)` にする。`StoredMessage` の フィールドは現物（`src/session.rs:97`）に合わせること。

- [ ] **Step 4: `generate_daily_log` に第2の供給源を足す**

`src/periodic_log.rs`:

```rust
pub async fn generate_daily_log<F>(
    session_store: &SessionStore,
    acp_store: Option<&AcpSessionStore>,
    provider: &dyn Provider,
    // ... 以下そのまま
```

```rust
    let mut sessions = session_store.sessions_for_day_filtered(date, boundary_hour, &room_predicate);
    if let Some(acp) = acp_store {
        sessions.extend(
            acp.sessions_for_day(date, boundary_hour)
                .into_iter()
                .filter(|(meta, _)| room_predicate(meta)),
        );
    }
    sessions.sort_by_key(|(meta, _)| meta.created_at);
```

`room_predicate` が `F: Fn(&SessionMeta) -> bool` なので、二度呼ぶために参照で渡す形に変える（`&room_predicate`）。既存の呼び出し側は `&predicate` を渡しているので、境界を `F: Fn(&SessionMeta) -> bool + Copy` にするか、`sessions_for_day_filtered` に `&room_predicate` を渡す。**コンパイラが通る最小の形にすること。**

`catchup_pending_daily_logs` も同様に `acp_store: Option<&AcpSessionStore>` を受け、日付の集合を

```rust
    let mut dates = session_store.all_session_dates_filtered(boundary_hour, &predicate);
    if let Some(acp) = acp_store {
        dates.extend(acp.session_dates(boundary_hour));
    }
    dates.sort();
    dates.dedup();
```

とする。内部で `generate_daily_log` を呼ぶ箇所（`:887`）にも `acp_store` を渡す。

- [ ] **Step 5: `namespace_for_session` に `"acp"` を足す**

`src/heartbeat.rs:75`:

```rust
    fn namespace_for_session(&self, meta: &crate::session::SessionMeta) -> String {
        if meta.channel == "device-default" || meta.channel == "rpc" || meta.channel == "acp" {
            meta.namespace
                .clone()
                .unwrap_or_else(|| crate::config::DEFAULT_NAMESPACE_NAME.to_string())
        } else {
            self.config.namespace_for_room(&meta.room_id).to_string()
        }
    }
```

**これが無いと ACP セッションは `namespace_for_room("")` に落ちて、デフォルト
namespace のログに全部流れ込む。** ヘッダに namespace があるのに使わないのは誤り。

- [ ] **Step 6: デイリーログを書いたらダイジェストを刈る**

`generate_daily_log` が `Ok(true)` を返した——つまりその日の恒久的な記録を書いた——時点で、その日以前のダイジェストは役目を終えている。`heartbeat.rs` の呼び出し側で刈る。

```rust
                        Ok(true) => {
                            any_generated = true;
                            // The permanent record for that day now
                            // exists, so the digests that were standing
                            // in for it can go.
                            if let Some(cache) = self.serve_state.as_ref().map(|s| &s.digest_cache)
                            {
                                let (_, day_end) = crate::session::day_window(
                                    yesterday,
                                    self.day_boundary_hour,
                                );
                                let removed = cache.prune_before(day_end);
                                if removed > 0 {
                                    info!("Pruned {removed} digest(s) covered by the daily log");
                                }
                            }
                        }
```

`catchup_pending_daily_logs` の中の生成箇所（`periodic_log.rs:887`）でも同じことをしたくなるが、**そこでは刈らない。** catch-up は過去日を埋める処理で、そのとき生きているキャッシュは今日のもの。過去日のログを書いたからといって今日のダイジェストを消してはいけない。上のコードが `yesterday` の `day_end` を切り口にしているのは、まさにその境界を守るため。

- [ ] **Step 7: heartbeat から ACP ストアを渡す**

`Heartbeat` は `serve_state: Option<Arc<ServeState>>` を既に持っている。`:128` と `:208` の呼び出しで

```rust
                    self.serve_state.as_ref().map(|s| &*s.acp_session_store),
```

を新しい引数に渡す。`serve_state` が `None`（serve が動いていない構成）なら `None` になり、既存どおりの挙動。

- [ ] **Step 8: テストが通ることを確認**

Run: `cargo test -p sapphire-agent`
Expected: PASS。

`generate_daily_log` の既存テスト（あれば）は引数が1つ増えて落ちる。**`None` を渡して通す。** 引数追加以外に挙動が変わっていないことを確認すること。

- [ ] **Step 9: フォーマットと lint**

```bash
cargo fmt --all
cargo clippy --workspace --all-targets -- -D warnings
```

- [ ] **Step 10: コミット**

```bash
git checkout -- Cargo.lock
git add src/acp_session.rs src/periodic_log.rs src/heartbeat.rs
git commit -m "feat(acp): include ACP sessions in the daily log"
```

---

### Task 8: ドキュメント、そしてワークスペース全体の確認

**Files:**
- Modify: `README.md`
- Modify: `docs/superpowers/specs/2026-08-31-acp-store-separation-design.md`

**Interfaces:** なし。

- [ ] **Step 1: README を更新する**

PR #188 が README に書いた「ACP セッションは `/rpc` セッションと同じストアを共有し、`cwd` の有無で区別される」という説明を差し替える。書くこと:

- ACP セッションは `<sessions_dir>/<namespace>/acp/<id>.jsonl` に入り、`/rpc` とは別のツリー・別の行形式である。
- 各イベントは UUIDv7 の `id` と `parent` を持ち、順序の権威は `parent` 鎖であって id の時刻ではない。
- ツール結果はワークスペース外の `<cache_dir>/sapphire-agent/tool-results/` にハッシュで置かれ、JSONL は参照だけを持つ。**キャッシュを消してもセッションは読める**（結果がプレースホルダになる）。同期対象ではないので、別端末では結果が失われた状態で開く。
- namespace 境界は memory namespace であって room profile ではない、という PR #188 が書いた注意書きは**そのまま残す**。まだ正しい。
- **ACP セッションはデイリーログと横断ダイジェストの両方に載る。** ヘッダの namespace で振り分けられ、`channel` は `"acp"` として記録される。`/rpc` / device-default / MCP は載らない（[#189](https://github.com/fluo10/sapphire-agent/issues/189)）。
- **ダイジェストは30分ごとに掃き出され、前回以降にイベントが増えたセッションだけ作り直される。** 置き場所は `<cache_dir>/sapphire-agent/digests/` で、1セッション1件を上書きする。ワークスペースの外なのは、`<workspace>/sessions` が retrieve の索引に入っており、似た要約が積み上がると検索を歪めるため。デイリーログを書いた時点で、その日以前の分は刈られる。消えても失うのは今日の横断ブロックだけで、恒久的な記録はデイリーログのほう。

- [ ] **Step 2: spec に訂正を追記する**

spec 本文は書き換えず、末尾に `## 実装時の訂正` を新設して足す。

```markdown
## 実装時の訂正

実装中に確認した3点。spec 本文の前提を訂正する。

### `summarize_all_sessions` は関数ごとは消せない

同じ関数が `append_intraday_digest` も呼んでおり、それが `cross_device` ストアの
**唯一の**ダイジェスト生成源だった。関数ごと消すと `build_today_digest_for_namespace`
が組み立てる「今日の横断メモ」から `/rpc` セッション分が落ちる。spec 自身が
「ダイジェストは保持」と言っているので、関数ごとの削除は spec に反していた。

削ったのは `append_summary` の呼び出しだけ。関数は `digest_all_sessions` に改名した。

### `/rpc` セッションはデイリーログに入っていない

`generate_daily_log` は引数のストアを1つだけ読む。渡っているのは
`channel_session_store`（`heartbeat.rs` の `Heartbeat.session_store` に `main.rs` が
`channel_session_store` を入れている）なので、**デイリーログは Matrix / Discord の
会話だけから作られている。**

これは「旧フォーマットの `/rpc` ファイルを移行しなくてよい、デイリーログには
入っているから」という判断の根拠を崩す。判断そのものは変わらない（retrieve の
検索対象ではある）が、根拠は一つ減る。

`/rpc` / device-default / MCP をデイリーログに含めるかは別の判断なので
[#189](https://github.com/fluo10/sapphire-agent/issues/189) に切り出した。
**ACP セッションは含める**——エージェントと今日何をしたかは残すべき記録であり、
それが横断ダイジェストとデイリーログの両方に載ることを、この分離の要件とした。

### ダイジェストをキャッシュに置く理由は、同期コストではなく検索汚染だった

spec は「サマリーとダイジェストはどちらもワークスペース外へ。同期されなくなり、
別端末では再生成になる」と書いた。結論は正しいが、理由がもう一つある。

**`<workspace>/sessions` は retrieve の索引に入っている**（`SessionStore.base_dir`
がそこで、`notify_updated` が `ws_state` に通知している）。ダイジェストは会話が
伸びるたびに作り直されるので、セッションファイルに追記すると似た内容の要約が
積み上がり、検索結果を歪める。ターンの長い開発セッションでは特に効く。

そしてイベントとして持った場合、**刈り取りでは解けない**——イベントは `parent`
鎖で繋がっており、途中を消すと子が孤児になる。append-only と刈り取りは両立しない。

キャッシュは `session_id` をキーに最新1件だけを保持し、上書きで更新する。
横断ダイジェストの読み手が必要とする `namespace` / `created_at` / `title` は
ストアのヘッダから結合するので、メタデータの二重化は起きない。
デイリーログを書いた時点で、その日以前の分を `prune_before` で刈る。

`channel` ストアのダイジェスト（`src/agent.rs` が書いているもの）は移設しない。
同じ検索汚染は起きているはずだが、それは `channel` ストアの話であり、
ストア分離と同じ PR に混ぜると壊れたときに切り分けられない。
```

- [ ] **Step 3: ワークスペース全体のテスト**

```bash
cargo test --workspace
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
```

Expected: all pass.

- [ ] **Step 4: コミット**

```bash
git checkout -- Cargo.lock
git add README.md docs/superpowers/specs/2026-08-31-acp-store-separation-design.md
git commit -m "docs: describe the ACP store and correct the spec's digest premises"
```
