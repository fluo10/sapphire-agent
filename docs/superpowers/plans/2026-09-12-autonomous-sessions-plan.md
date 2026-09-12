# 自律セッション（autonomous session）導入 実装計画

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 他のセッションが静かなとき（既定 30 分）に、`<workspace>/autonomous/*.md` のタスクを優先度順に1つ選び、タスク単位で最大 `max_turns` ターンのセッションを回す。セッションは `sessions/<ns>/autonomous/{date}-{uuid}.jsonl` に普通のセッションとして残り、`session_list` / `session_read` から振り返れる。

**Architecture:** 「自律セッションは普通のセッションである」が軸。新しいターン実行器は作らず、既存の `run_llm_turn` をそのまま呼ぶ。足すのは (1) 4 つ目の `SessionStore`（kind = `"autonomous"`）とそのファイル命名 `{date}-{uuid}.jsonl`、(2) `ServeState` にその store を足して `store_for_session` に1分岐、(3) heartbeat と同型の第3ループ `autonomous.rs`（アイドル判定 → タスク選択 → ターンループ）、(4) `[autonomous]` 設定、(5) `session_list` / `session_read` の探索先に自律ストアを追加。

**Tech Stack:** Rust 2024, `serde` / `serde_yaml` / `toml`, `tokio`, `async-trait`, `chrono`

**Spec:** `docs/superpowers/specs/2026-09-12-autonomous-sessions-design.md`

## Global Constraints

- ブランチは `feat/autonomous-sessions`（`main` = `55363aa` から作成し、**仕様書のコミットを最初に載せる**）。
- テストコマンドは `cargo test -p sapphire-agent`。反復中は `--workspace` を使わない。最後に一度だけ `cargo test --workspace`。
- **`cargo check` と `cargo test` を混ぜない。** フィンガープリントが別なので交互に走らせると毎回リンクし直しになる。
- cargo は**前景で、`timeout: 600000` で**走らせる。**cargo を2本同時に走らせない**。10分のツールタイムアウトに当たったらビルドは温まっているので同じコマンドを走らせ直す。
- **コミット前に `cargo clippy --workspace -- -D warnings`**（CI と同じ形、`--all-targets` を付けない）。
- **`Cargo.lock` をコミットしない。** 各コミット前に `git checkout -- Cargo.lock`。
- テストはソースファイル内の `mod tests` に置く。コード・コメント・コミットメッセージは**英語**（`CONTRIBUTING.md`）、計画書と仕様書は日本語。
- **既定で無効**。`[autonomous] enabled = true` を書かない限り、ループは spawn されず、ディレクトリもファイルも作られない。
- 既存ストア（`channel` / `cross-device` / `device-default` / `mcp` / `acp`）のファイル名・解決順・権限判定を**1バイトも変えない**。
- `Origin` / `decide` / `host_tool_denied` は**一切変更しない**（`AutonomousHost::origin()` が設定値を1つ返すだけ）。

### 仕様書に書かれていない実装上の決定（本計画で確定させる）

1. **`channel` は `"server"`**（仕様 決定 1）。`session_label` は `{channel}/{room_id}` を出すので、`session_list` のラベルは `server/<task>` になる。`room_id` はタスク名（仕様 決定 3）。
2. **`not_before` は `DateTime<Utc>`**（仕様 決定 12 は `Instant` と書いている）。アイドル判定・クールダウン判定と同じ時計1本に揃えるため。
3. **タイトル生成は行わない。** `run_llm_turn` の呼び出し元（`/rpc`・voice）は最初のターンで `generate_session_title` を spawn するが、自律ループは spawn しない — 一覧のラベルは `server/<task>` で十分に読め、モデル呼び出しが1回増えるだけだから。タイトルが無いことは `session_label` の既存分岐がそのまま許容する。
4. **ループ本体は `run_cycle()` に切り出す。** `run()` は「`interval` で `run_cycle()` を呼ぶだけ」にし、テストは `run_cycle()` を直接叩く（`StubProvider` を刺した `ServeState` で、sleep なしに 1 サイクルを検証する）。

---

## File Structure

| ファイル | 役割 |
|---|---|
| `server/src/session.rs`（変更） | `FileNaming` enum と `SessionStore::with_dated_files`。`path_for_new` / `resolve_path` が命名種別を見る。`create_autonomous_session`。`Plain` の挙動は不変 |
| `server/src/autonomous_config.rs`（新規） | `<workspace>/autonomous/*.md` のローダー。`AutonomousTask` / `AutonomousTaskMeta` |
| `server/src/config.rs`（変更） | `[autonomous]` → `AutonomousConfig` / `AutonomousOrigin`。既定 `enabled = false` |
| `server/src/config_layer.rs`（変更） | workspace 層の allowlist に `["autonomous"]` を追加。`FIXTURE` に `[autonomous]` を足す |
| `server/src/serve/mod.rs`（変更） | `ServeState.autonomous_session_store`、`store_for_session` の分岐、`AutonomousHost`（`TurnHost`）、テスト用 fixture に store を足す |
| `server/src/autonomous.rs`（新規） | 状態ファイル `state/autonomous.json`、`is_idle`、`next_task`、`AutonomousLoop::{run, run_cycle}` |
| `server/src/main.rs`（変更） | 自律ストアを構築し `ServeState` に渡す。`[autonomous] enabled` のときだけループを spawn |
| `server/src/tools/session_tools.rs`（変更） | `SessionSources` に自律ストアを足し、`all_rows` / `transcript` が見る |
| `server/src/cli_init.rs`（変更） | `FILE_TEMPLATES` / `EXPECTED_FILES` に `autonomous/example-refactor.md` を追加、ローダーのテストを足す |
| `server/templates/workspace/example-autonomous.md`（新規） | 種として配るタスク定義の例（`enabled: false`） |
| `server/templates/workspace/config.toml`（変更） | workspace 層から設定できる `[autonomous]` の説明（コメントのみ） |
| `server/config.example.toml`（変更） | host 側の `[autonomous]` の説明。`host_access.enabled` が要ることを明記 |
| `README.md`（変更） | `## Autonomous sessions` を短く追加（heartbeat との違い3点と設定） |
| `docs/superpowers/specs/2026-09-12-autonomous-sessions-design.md`（変更・1行） | 受け入れ基準 8 のラベル表記を `server/<task>` に合わせる |

---

### Task 0: ブランチと仕様書のコミット

- [ ] **Step 1: ブランチを切る**

```bash
cd ~/Documents/Dev/project-sapphire/sapphire-agent
git switch -c feat/autonomous-sessions
git status --short
```

期待: `docs/superpowers/specs/2026-09-12-autonomous-sessions-design.md` が `??` で出る。

- [ ] **Step 2: 仕様書と計画書をコミット**

```bash
git add docs/superpowers/specs/2026-09-12-autonomous-sessions-design.md \
        docs/superpowers/plans/2026-09-12-autonomous-sessions-plan.md
git commit -m "docs: specify autonomous sessions (#248)"
```

期待: `2 files changed`。以降のタスクはこのブランチに積む。

---

### Task 1: セッションファイルの命名種別（`{date}-{uuid}.jsonl`）

**Files:**
- Modify: `server/src/session.rs`（`SessionStore` の構造体・`new` / `with_workspace` / `path_for_new` / `resolve_path`、`create_autonomous_session`、`mod tests`）
- Test: 同上 `mod tests`

**Interfaces:**
- Produces:
  - `pub enum FileNaming { Plain, Dated { boundary_hour: u8 } }`（`#[derive(Debug, Clone, Copy)]`）
  - `pub fn SessionStore::with_dated_files(self, boundary_hour: u8) -> Self` — `FileNaming::Dated` に差し替えるビルダー。`Plain`（既定）の store は1バイトも挙動が変わらない
  - `pub fn SessionStore::create_autonomous_session(&self, task: &str, namespace: &str) -> anyhow::Result<String>` — `room_id = task`、`channel = "server"` のセッションを1つ作って id を返す
- Consumes: `crate::session::local_date_for_timestamp(Local::now(), boundary_hour)`（既存）

#### なぜ `resolve_path` を接尾辞一致にするか

`SessionStore::resolve_path` は今 `name == "{session_id}.jsonl"` の完全一致で、`path_cache` も session_id をキーにしている。`Dated` の store では実ファイル名が `2026-09-12-01920f...jsonl` なので、完全一致のままだと `append` も `load_session` も `close_session` もファイルを見つけられない。session_id は v7 UUID（36文字・ハイフン4つ）なので、**`-{session_id}.jsonl` で終わり、その前が日付（10文字）** という条件は他のファイルと衝突しない。

- [ ] **Step 1: 失敗するテストを書く**

`server/src/session.rs` の `mod tests` に追加。

```rust
/// The autonomous store names its files by agent-day, and every write
/// path has to still find them: `append` is the first thing that runs
/// after the session is created.
#[test]
fn a_dated_store_creates_and_resolves_its_own_filenames() {
    let tmp = tempfile::tempdir().unwrap();
    let store = SessionStore::new(tmp.path().to_path_buf(), "autonomous", None)
        .with_dated_files(4);

    let sid = store
        .create_autonomous_session("refactor", "default")
        .unwrap();

    let path = store.absolute_path_for(&sid).expect("the file must resolve");
    let name = path.file_name().unwrap().to_str().unwrap();
    let expected_date = local_date_for_timestamp(Local::now(), 4).to_string();
    assert_eq!(name, format!("{expected_date}-{sid}.jsonl"));
    assert!(path.starts_with(tmp.path().join("default").join("autonomous")));

    store.append(&sid, &ChatMessage::user("hello")).unwrap();
    assert_eq!(store.load_session(&sid).unwrap().len(), 1);

    // The meta line carries what the loop searches by.
    let rows = store.session_rows();
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].meta.room_id, "refactor");
    assert_eq!(rows[0].meta.channel, "server");
    assert_eq!(rows[0].meta.namespace.as_deref(), Some("default"));
}

/// A plain store must not change: its filename is still exactly
/// `{session_id}.jsonl`, and a dated-looking neighbour does not resolve.
#[test]
fn a_plain_store_still_matches_its_filename_exactly() {
    let tmp = tempfile::tempdir().unwrap();
    let store = SessionStore::new(tmp.path().to_path_buf(), "channel", None);
    let sid = store
        .create_session(&("room".to_string(), None), "matrix", "default")
        .unwrap();

    let path = store.absolute_path_for(&sid).unwrap();
    assert_eq!(
        path.file_name().unwrap().to_str().unwrap(),
        format!("{sid}.jsonl")
    );

    // Same id, a dated filename: must not be picked up by the exact match.
    let ns = tmp.path().join("default").join("channel");
    std::fs::write(ns.join(format!("2026-09-12-{sid}.jsonl")), "").unwrap();
    assert_eq!(
        store.resolve_path_for_test(&sid).unwrap().file_name().unwrap(),
        std::ffi::OsStr::new(&format!("{sid}.jsonl"))
    );
}
```

`resolve_path` は private なので、テストから使うため `#[cfg(test)]` のアクセサを足す（Task 1 の実装に含める）。

- [ ] **Step 2: テストが落ちることを確認**

```bash
cargo test -p sapphire-agent a_dated_store_creates_and_resolves 2>&1 | tail -20
```

期待: `error[E0599]: no method named `with_dated_files`` / `error[E0599]: no method named `create_autonomous_session`` でコンパイルエラー。

- [ ] **Step 3: 実装する**

`server/src/session.rs` の `SessionStore` 宣言部（`pub kind: &'static str,` の直後）にフィールドを足す。

```rust
    /// How this store names its session files.
    ///
    /// `Plain` for every store but the autonomous one. A `Dated` store
    /// prefixes the agent-day the file was created in, which is what
    /// makes an overnight run of autonomous sessions readable in a
    /// directory listing — the session's own metadata is still one line
    /// away, this is for the human doing the looking.
    naming: FileNaming,
```

`SessionStore` の宣言の直前に型を置く。

```rust
/// How a [`SessionStore`] names the files it writes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FileNaming {
    /// `{session_id}.jsonl`. Every store but the autonomous one.
    Plain,
    /// `{date}-{session_id}.jsonl`, `date` being the agent-day (per
    /// `boundary_hour`) the file was created in.
    Dated { boundary_hour: u8 },
}
```

`new` と `with_workspace` の `Self { ... }` に `naming: FileNaming::Plain,` を足し、`kind` の直後にビルダーを足す。

```rust
    /// Name this store's files by agent-day. See [`FileNaming`].
    ///
    /// A builder rather than a constructor argument because exactly one
    /// store wants it and every other call site — four in `main.rs`,
    /// eleven in tests — would otherwise have to name the `Plain`
    /// default explicitly.
    pub fn with_dated_files(mut self, boundary_hour: u8) -> Self {
        self.naming = FileNaming::Dated { boundary_hour };
        self
    }
```

`path_for_new` の内部を差し替える。

```rust
    fn path_for_new(&self, session_id: &str, namespace: &str) -> PathBuf {
        let file_name = match self.naming {
            FileNaming::Plain => format!("{session_id}.jsonl"),
            FileNaming::Dated { boundary_hour } => {
                let date = local_date_for_timestamp(Local::now(), boundary_hour);
                format!("{date}-{session_id}.jsonl")
            }
        };
        let p = self.base_dir.join(namespace).join(self.kind).join(file_name);
        if let Ok(mut cache) = self.path_cache.lock() {
            cache.insert(session_id.to_string(), p.clone());
        }
        p
    }
```

`resolve_path` の照合部分を差し替える。

```rust
    fn resolve_path(&self, session_id: &str) -> Option<PathBuf> {
        if let Ok(cache) = self.path_cache.lock()
            && let Some(p) = cache.get(session_id)
        {
            return Some(p.clone());
        }
        let target = format!("{session_id}.jsonl");
        for path in collect_session_files(&self.base_dir, self.kind) {
            let Some(name) = path.file_name().and_then(|s| s.to_str()) else {
                continue;
            };
            let hit = match self.naming {
                FileNaming::Plain => name == target,
                // A v7 UUID is 36 characters with four hyphens, so
                // `-{session_id}.jsonl` with a 10-character date in
                // front of it cannot collide with another session's
                // file.
                FileNaming::Dated { .. } => name
                    .strip_suffix(&format!("-{session_id}.jsonl"))
                    .is_some_and(|date| date.len() == 10),
            };
            if hit {
                if let Ok(mut cache) = self.path_cache.lock() {
                    cache.insert(session_id.to_string(), path.clone());
                }
                return Some(path);
            }
        }
        None
    }
```

`resolve_path` の直後に `#[cfg(test)]` アクセサと、`create_session` の直後に自律セッション作成を足す。

```rust
    /// Test-only: the resolver, so a test can assert which file an id
    /// maps to when two candidates exist.
    #[cfg(test)]
    pub fn resolve_path_for_test(&self, session_id: &str) -> Option<PathBuf> {
        self.resolve_path(session_id)
    }
```

```rust
    /// Create one autonomous session for a task. `room_id` carries the
    /// task name — that is the reverse index the loop searches by, and
    /// it is why there is no separate state file for "which session is
    /// this task in" (spec decision 3).
    ///
    /// `channel` is `"server"`: the same value a `/rpc` session uses,
    /// because an autonomous session is not a chat and has no channel of
    /// its own.
    pub fn create_autonomous_session(
        &self,
        task: &str,
        namespace: &str,
    ) -> anyhow::Result<String> {
        self.create_session(&(task.to_string(), None), "server", namespace)
    }
```

`mod tests` の先頭付近に `use chrono::Local;` と `use crate::provider::ChatMessage;` が無ければ足す（既存テストが `Local` を使っているはずなので、足すのは `resolve_path_for_test` 用の `std::ffi::OsStr` だけかもしれない。コンパイラの指示に従う）。

- [ ] **Step 4: テストが通ることを確認**

```bash
cargo test -p sapphire-agent session:: 2>&1 | tail -20
```

期待: `test result: ok.` で、新規2件を含む。既存の session テストが落ちないこと（`Plain` の挙動が不変であることの証明）。

- [ ] **Step 5: コミット**

```bash
git checkout -- Cargo.lock
cargo clippy --workspace -- -D warnings
git add server/src/session.rs
git commit -m "feat(session): name autonomous session files by agent-day"
```

---

### Task 2: `autonomous/*.md` のローダー

**Files:**
- Create: `server/src/autonomous_config.rs`
- Modify: `server/src/main.rs`（`mod autonomous_config;` を `mod agents;` の直後に追加）
- Test: `server/src/autonomous_config.rs` の `mod tests`

**Interfaces:**
- Produces:
  - `pub struct AutonomousTask { pub name: String, pub enabled: bool, pub priority: i64, pub cooldown_days: u32, pub max_turns: usize, pub body: String }`
  - `pub fn load_autonomous_dir(dir: &Path) -> Vec<AutonomousTask>`
- Consumes: `crate::frontmatter::split`（`agents.rs` / `heartbeat_config.rs` と同じ）

#### 既定値

- `enabled`: `true`（書かなければ動く）
- `priority`: `100` — `Option` にしない。未指定のタスクも順序の中に居る必要があり、`100` は「まだ考えていないタスク」を `priority: 10` の後ろに置く
- `cooldown_days`: `0`（前のセッションが終わったらすぐ次）
- `max_turns`: `3`。`0` は `1` に切り上げ（0 を「1ターンも走らせない」と解釈すると、セッションだけ作って閉じる無意味なタスクになる）

- [ ] **Step 1: 失敗するテストを書く**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    fn write(dir: &Path, name: &str, raw: &str) {
        std::fs::write(dir.join(name), raw).unwrap();
    }

    #[test]
    fn a_missing_directory_is_no_tasks() {
        let d = tempfile::tempdir().unwrap();
        assert!(load_autonomous_dir(&d.path().join("nope")).is_empty());
    }

    /// Every field has a default, so a body-only task is a valid task.
    #[test]
    fn the_defaults_match_the_spec() {
        let d = tempfile::tempdir().unwrap();
        write(d.path(), "journal.md", "---\n---\n\nDo the thing.\n");

        let tasks = load_autonomous_dir(d.path());
        assert_eq!(tasks.len(), 1);
        let t = &tasks[0];
        assert_eq!(t.name, "journal");
        assert!(t.enabled);
        assert_eq!(t.priority, 100);
        assert_eq!(t.cooldown_days, 0);
        assert_eq!(t.max_turns, 3);
        assert_eq!(t.body, "Do the thing.\n");
    }

    #[test]
    fn the_frontmatter_is_parsed_onto_the_task() {
        let d = tempfile::tempdir().unwrap();
        write(
            d.path(),
            "refactor.md",
            "---\nenabled: false\npriority: 10\ncooldown_days: 30\nmax_turns: 5\n---\nRefactor.\n",
        );

        let tasks = load_autonomous_dir(d.path());
        assert_eq!(tasks.len(), 1);
        let t = &tasks[0];
        assert!(!t.enabled);
        assert_eq!(t.priority, 10);
        assert_eq!(t.cooldown_days, 30);
        assert_eq!(t.max_turns, 5);
    }

    /// `max_turns: 0` would mean "make a session and close it without a
    /// turn", which is not a thing anyone wants.
    #[test]
    fn max_turns_zero_is_raised_to_one() {
        let d = tempfile::tempdir().unwrap();
        write(d.path(), "t.md", "---\nmax_turns: 0\n---\nBody.\n");
        assert_eq!(load_autonomous_dir(d.path())[0].max_turns, 1);
    }

    /// A broken file is skipped with a warning, and the rest still load.
    #[test]
    fn a_broken_definition_does_not_take_the_others_with_it() {
        let d = tempfile::tempdir().unwrap();
        write(d.path(), "good.md", "---\n---\nGood.\n");
        write(d.path(), "bad.md", "no frontmatter at all\n");
        write(d.path(), "typo.md", "---\npriorit: 1\n---\nTypo.\n");
        write(d.path(), "empty.md", "---\n---\n");
        write(d.path(), "notes.txt", "ignored");

        let tasks = load_autonomous_dir(d.path());
        assert_eq!(
            tasks.iter().map(|t| t.name.as_str()).collect::<Vec<_>>(),
            vec!["good"]
        );
    }

    /// `priority` alone is not an order — two tasks at the same priority
    /// must not depend on `read_dir`.
    #[test]
    fn tasks_come_back_sorted_by_priority_then_name() {
        let d = tempfile::tempdir().unwrap();
        write(d.path(), "b.md", "---\npriority: 10\n---\nB.\n");
        write(d.path(), "a.md", "---\npriority: 10\n---\nA.\n");
        write(d.path(), "z.md", "---\npriority: 1\n---\nZ.\n");

        let names: Vec<String> = load_autonomous_dir(d.path())
            .into_iter()
            .map(|t| t.name)
            .collect();
        assert_eq!(names, vec!["z", "a", "b"]);
    }
}
```

- [ ] **Step 2: テストが落ちることを確認**

```bash
cargo test -p sapphire-agent autonomous_config 2>&1 | tail -20
```

期待: `error[E0583]: file not found for module `autonomous_config``（`mod` 宣言を先に足していれば `error[E0425]: cannot find function `load_autonomous_dir``）。

- [ ] **Step 3: 実装する**

`server/src/autonomous_config.rs` を新規作成する（`agents.rs` の形をそのまま踏襲し、`deny_unknown_fields` だけ足す — frontmatter の綴り間違いが「黙って既定値で動くタスク」になるより、警告して飛ばすほうがよい）。

```rust
//! Autonomous task definitions, loaded from `<workspace>/autonomous/*.md`.
//!
//! The third user of the shape `<workspace>/heartbeat/*.md` established
//! and `<workspace>/agents/*.md` followed: YAML frontmatter for the
//! control fields, the body for the instruction. The body is the whole
//! first user message of every turn the task runs.
//!
//! Unlike the other two, a task is not a one-shot prompt or a delegate:
//! it is a unit of *work* that may span several turns across several
//! nights. State that outlives a turn — cooldown, how many turns have
//! been spent — is derived from the session store rather than kept here
//! (see the design doc, decisions 3 and 5), so a definition file holds
//! only what a human decides.

use serde::Deserialize;
use std::path::Path;
use tracing::warn;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct AutonomousTaskMeta {
    #[serde(default = "default_true")]
    enabled: bool,
    /// Lower runs first. Default 100, so an un-tuned task queues behind
    /// the ones someone thought about.
    #[serde(default = "default_priority")]
    priority: i64,
    /// Days that must pass after the task's last activity before it is
    /// due again. `0` means "as soon as the previous session ended".
    #[serde(default)]
    cooldown_days: u32,
    /// Turns one session may spend before it is closed. Raised to 1 when
    /// declared as 0.
    #[serde(default = "default_max_turns")]
    max_turns: usize,
}

fn default_true() -> bool {
    true
}

fn default_priority() -> i64 {
    100
}

fn default_max_turns() -> usize {
    3
}

#[derive(Debug, Clone, PartialEq)]
pub struct AutonomousTask {
    /// The file stem, which is also the session's `room_id`.
    pub name: String,
    pub enabled: bool,
    pub priority: i64,
    pub cooldown_days: u32,
    pub max_turns: usize,
    /// The Markdown body, used verbatim as the instruction.
    pub body: String,
}

/// Load every task under `dir`, skipping the ones that cannot be read.
/// A missing directory is no tasks, not an error.
pub fn load_autonomous_dir(dir: &Path) -> Vec<AutonomousTask> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("md") {
            continue;
        }
        let Some(name) = path.file_stem().and_then(|s| s.to_str()) else {
            continue;
        };
        let raw = match std::fs::read_to_string(&path) {
            Ok(r) => r,
            Err(e) => {
                warn!(
                    "failed to read autonomous task {}: {e}",
                    path.display()
                );
                continue;
            }
        };
        match parse_task(name.to_string(), &raw) {
            Some(t) => out.push(t),
            None => warn!(
                "autonomous task {} skipped (no/invalid frontmatter, or an empty body)",
                path.display()
            ),
        }
    }
    out.sort_by(|a, b| a.priority.cmp(&b.priority).then_with(|| a.name.cmp(&b.name)));
    out
}

fn parse_task(name: String, raw: &str) -> Option<AutonomousTask> {
    let (fm, body) = crate::frontmatter::split(raw)?;
    let meta: AutonomousTaskMeta = match serde_yaml::from_str(fm) {
        Ok(m) => m,
        Err(e) => {
            warn!("autonomous task {name}: yaml parse error: {e}");
            return None;
        }
    };
    let body = body.trim_start_matches(['\n', '\r']).to_string();
    // An instruction-less task would burn a session and every turn it
    // allows on nothing.
    if body.trim().is_empty() {
        return None;
    }
    Some(AutonomousTask {
        name,
        enabled: meta.enabled,
        priority: meta.priority,
        cooldown_days: meta.cooldown_days,
        max_turns: meta.max_turns.max(1),
        body,
    })
}
```

`server/src/main.rs` の `mod agents;` の直後に `mod autonomous_config;` を足す（`mod autonomous;` は Task 5 で足す）。

- [ ] **Step 4: テストが通ることを確認**

```bash
cargo test -p sapphire-agent autonomous_config 2>&1 | tail -20
```

期待: `test result: ok. 6 passed`。

- [ ] **Step 5: コミット**

```bash
git checkout -- Cargo.lock
cargo clippy --workspace -- -D warnings
git add server/src/autonomous_config.rs server/src/main.rs
git commit -m "feat(autonomous): load task definitions from autonomous/*.md"
```

---

### Task 3: `[autonomous]` 設定

**Files:**
- Modify: `server/src/config.rs`（`Config` に `autonomous` フィールド、`AutonomousConfig` / `AutonomousOrigin` を `SubagentCacheConfig` の近くに、`mod tests`）
- Modify: `server/src/config_layer.rs`（`WORKSPACE_ALLOWLIST` と `FIXTURE`）
- Modify: `server/templates/workspace/config.toml`
- Modify: `server/config.example.toml`
- Test: `server/src/config.rs` と `server/src/config_layer.rs` の `mod tests`

**Interfaces:**
- Produces:
  - `pub struct AutonomousConfig { pub enabled: bool, pub idle_minutes: u64, pub poll_seconds: u64, pub origin: AutonomousOrigin }`（`Default` 実装あり）
  - `pub enum AutonomousOrigin { Trusted, Channel }`（`#[serde(rename_all = "snake_case")]`、`Default = Trusted`）
  - `Config::autonomous: AutonomousConfig`
- Consumes: なし

- [ ] **Step 1: 失敗するテストを書く**

`server/src/config.rs` の `mod tests` に追加。

```rust
    /// The default is off. An agent that starts working on its own
    /// because a new build shipped is not a behaviour change anyone
    /// asked for.
    #[test]
    fn autonomous_is_disabled_by_default() {
        let cfg = parse("[anthropic]\napi_key = \"test\"\n");
        assert!(!cfg.autonomous.enabled);
        assert_eq!(cfg.autonomous.idle_minutes, 30);
        assert_eq!(cfg.autonomous.poll_seconds, 60);
        assert_eq!(cfg.autonomous.origin, AutonomousOrigin::Trusted);
    }

    #[test]
    fn the_autonomous_table_is_read() {
        let cfg = parse(
            "[anthropic]\napi_key = \"test\"\n\n\
             [autonomous]\nenabled = true\nidle_minutes = 5\npoll_seconds = 1\norigin = \"channel\"\n",
        );
        assert!(cfg.autonomous.enabled);
        assert_eq!(cfg.autonomous.idle_minutes, 5);
        assert_eq!(cfg.autonomous.poll_seconds, 1);
        assert_eq!(cfg.autonomous.origin, AutonomousOrigin::Channel);
    }

    /// A misspelled key in this table is a permission or a cadence the
    /// operator believes is set and is not.
    #[test]
    fn a_typo_in_the_autonomous_table_is_rejected() {
        let raw = "[anthropic]\napi_key = \"test\"\n\n[autonomous]\nenabled = true\nidle_minute = 5\n";
        assert!(toml::from_str::<Config>(raw).is_err());
    }
```

- [ ] **Step 2: テストが落ちることを確認**

```bash
cargo test -p sapphire-agent config:: 2>&1 | tail -20
```

期待: `error[E0609]: no field `autonomous` on type `Config``。

- [ ] **Step 3: 実装する**

`server/src/config.rs` の `Config` に、`subagent_cache` フィールドの直後に足す。

```rust
    /// Autonomous sessions: work the agent does on its own when no other
    /// session is active. Off unless an operator turns it on — it is the
    /// one feature here that spends compute with nobody waiting.
    #[serde(default)]
    pub autonomous: AutonomousConfig,
```

`SubagentCacheConfig` の定義の後ろに足す。

```rust
/// Autonomous-session configuration. See the design doc, decisions 8–12.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AutonomousConfig {
    /// Whether the idle loop runs at all. Default false.
    #[serde(default)]
    pub enabled: bool,
    /// How long every other session must have been quiet for the agent
    /// to consider itself idle, in minutes. See `autonomous::is_idle`.
    #[serde(default = "default_autonomous_idle_minutes")]
    pub idle_minutes: u64,
    /// How often the loop looks for work, including while it is idle.
    ///
    /// This is a poll, not a cadence: a cycle that finds nothing due
    /// costs one directory read and one append-free scan, so a short
    /// interval is cheap, but it is also the granularity at which an
    /// edited task file takes effect.
    #[serde(default = "default_autonomous_poll_seconds")]
    pub poll_seconds: u64,
    /// Which row of the permission table an autonomous turn is judged by.
    ///
    /// `Trusted` by default because the point of the feature is work
    /// that needs `shell` (filing issues, committing). That is a
    /// deliberate hole, not an oversight: the workspace is the
    /// operator's, and the future "file edits are tool-only and
    /// per-profile" policy is what closes it. See decision 10.
    ///
    /// Note this is only half the gate — `[tools] host_access.enabled`
    /// has to be on too, or `host_tool_denied` refuses `shell` and
    /// `file_write` before the table is consulted at all.
    #[serde(default)]
    pub origin: AutonomousOrigin,
}

impl Default for AutonomousConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            idle_minutes: default_autonomous_idle_minutes(),
            poll_seconds: default_autonomous_poll_seconds(),
            origin: AutonomousOrigin::Trusted,
        }
    }
}

fn default_autonomous_idle_minutes() -> u64 {
    30
}

fn default_autonomous_poll_seconds() -> u64 {
    60
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum AutonomousOrigin {
    /// `Origin::Trusted`: `/rpc`, voice and `/a2a`'s row. Everything,
    /// subject to `host_access`.
    Trusted,
    /// `Origin::Channel`: reads and unapproved edits; `Execute` and
    /// `Other` refused. The heartbeat's chat leg's row.
    Channel,
}

impl Default for AutonomousOrigin {
    fn default() -> Self {
        Self::Trusted
    }
}
```

`server/src/config_layer.rs` の `WORKSPACE_ALLOWLIST` の `&["heartbeat_enabled"],` の直後に足す。

```rust
    // The autonomous loop's switch and cadence. Allowed for the same
    // reason `heartbeat_enabled` is: it describes how the agent behaves,
    // not where its credentials live. The cost of allowing it is that
    // whoever can write the workspace can set `enabled = true` and
    // `idle_minutes = 0` — the same power `heartbeat_enabled` already
    // hands them, so not a new hole (design doc, risk 5).
    &["autonomous"],
```

同じファイルの `FIXTURE` の `heartbeat_enabled = true` の近くに足す（`every_allowlist_entry_is_exercised_by_the_fixture` が要求する）。

```toml
[autonomous]
enabled = true
idle_minutes = 15
origin = "channel"
```

`server/templates/workspace/config.toml` の `heartbeat_enabled` の説明の直後に足す。`cli_init` の `every_setting_the_workspace_config_documents_is_one_that_layer_may_set` がこのファイルをコメント解除して `Config` として読むので、**解除しても妥当な TOML でなければならない**。

```toml
# Autonomous sessions: the agent works on tasks from autonomous/*.md
# when no other session has been active for this long. Off in a fresh
# workspace for the same reason heartbeat is — a test rig sharing a
# machine should not start working on its own.
# [autonomous]
# enabled = false
# idle_minutes = 30
# poll_seconds = 60
# origin = "trusted"   # "trusted" (everything, needs [tools] host_access) | "channel" (no shell)
```

`server/config.example.toml` の `heartbeat_enabled` の塊の直後に足す。

```toml
# Autonomous sessions: when no other session has been active for
# `idle_minutes`, the agent picks the highest-priority due task from
# `<workspace>/autonomous/*.md` and works on it for up to that task's
# `max_turns`, as a normal session you can read back with `session_list`
# and `session_read`.
#
# Off by default. Two switches have to agree before an autonomous turn
# can do outward work like filing a GitHub issue:
#
#   - `origin = "trusted"` lets the permission table allow `shell` and
#     `file_write`. With `origin = "channel"` they are refused and the
#     feature can only read and edit workspace files.
#   - `[tools] host_access.enabled = true` — without it the host tools
#     are refused before the permission table is consulted at all, for
#     every origin.
#
# [autonomous]
# enabled = true
# idle_minutes = 30
# poll_seconds = 60
# origin = "trusted"
```

- [ ] **Step 4: テストが通ることを確認**

```bash
cargo test -p sapphire-agent -- config:: config_layer:: cli_init:: 2>&1 | tail -30
```

期待: `test result: ok.`。`config_layer` の `the_fixture_is_entirely_allowlisted` と `every_allowlist_entry_is_exercised_by_the_fixture`、`cli_init` の `every_setting_the_workspace_config_documents_is_one_that_layer_may_set` がいずれも通ること。

- [ ] **Step 5: コミット**

```bash
git checkout -- Cargo.lock
cargo clippy --workspace -- -D warnings
git add server/src/config.rs server/src/config_layer.rs \
        server/templates/workspace/config.toml server/config.example.toml
git commit -m "feat(config): add the [autonomous] table, off by default"
```

---

### Task 4: `ServeState` に自律ストアと `AutonomousHost` を足す

**Files:**
- Modify: `server/src/serve/mod.rs`（`ServeState` のフィールドと `new` の引数、`store_for_session`、`AutonomousHost`、テスト fixture `build_for_test_with` / `for_test_*`）
- Test: `server/src/serve/mod.rs` の `mod tests`

**Interfaces:**
- Consumes: `AutonomousConfig`（Task 3）、`FileNaming` / `with_dated_files`（Task 1）
- Produces:
  - `ServeState::autonomous_session_store: Arc<SessionStore>`
  - `ServeState::new(...)` の引数が `device_default_session_store` の直後に `autonomous_session_store: Arc<SessionStore>` を1本増える
  - `pub(crate) struct AutonomousHost { pub(crate) origin: crate::tools::policy::Origin }` — `TurnHost` 実装。`origin()` だけを上書きし、`round_budget()` は実装しない（既定 `Unattended` が正しい）
  - `ServeState::store_for_session` が自律ストアを **device-default より先に** 判定する

- [ ] **Step 1: 失敗するテストを書く**

`server/src/serve/mod.rs` の `mod tests` に追加。

```rust
    /// The autonomous store owns its sessions, and `store_for_session`
    /// has to say so — otherwise every `append` in an autonomous turn
    /// lands in the cross-device tree and the session is never found
    /// again.
    #[tokio::test]
    async fn an_autonomous_session_resolves_to_the_autonomous_store() {
        let state = ServeState::for_test(false);
        let sid = state
            .autonomous_session_store
            .create_autonomous_session("refactor", "default")
            .unwrap();

        let store = state.store_for_session(&sid);
        assert!(Arc::ptr_eq(store, &state.autonomous_session_store));
        assert!(!Arc::ptr_eq(store, &state.cross_device_session_store));
    }

    /// The default host is `Trusted`, and `round_budget` is left at the
    /// `Unattended` default — nobody can cancel an autonomous turn, so
    /// its budget must stay finite.
    #[test]
    fn the_autonomous_host_is_trusted_and_unattended() {
        let host = AutonomousHost {
            origin: crate::tools::policy::Origin::Trusted,
        };
        assert_eq!(host.origin(), crate::tools::policy::Origin::Trusted);
        assert_eq!(host.round_budget(), RoundBudget::Unattended);

        let host = AutonomousHost {
            origin: crate::tools::policy::Origin::Channel,
        };
        assert_eq!(host.origin(), crate::tools::policy::Origin::Channel);
    }
```

- [ ] **Step 2: テストが落ちることを確認**

```bash
cargo test -p sapphire-agent an_autonomous_session_resolves 2>&1 | tail -20
```

期待: `error[E0609]: no field `autonomous_session_store` on type `ServeState``。

- [ ] **Step 3: 実装する**

`ServeState` に、`device_default_session_store` の直後にフィールドを足す。

```rust
    /// Autonomous session store (kind = `"autonomous"`). Holds the
    /// sessions the idle loop runs, one per task — kept in its own
    /// directory for the same reason the ACP store is: they are ordinary
    /// sessions for the purpose of reading them back, and not chat for
    /// the purpose of anything else. Its files are named by agent-day
    /// (`{date}-{uuid}.jsonl`) — see `SessionStore::with_dated_files`.
    pub(crate) autonomous_session_store: Arc<SessionStore>,
```

`new` のシグネチャを変える。

```rust
    pub fn new(
        config: Config,
        registry: Arc<ProviderRegistry>,
        workspace: Arc<Workspace>,
        tools: Arc<ToolSet>,
        cross_device_session_store: Arc<SessionStore>,
        device_default_session_store: Arc<SessionStore>,
        /// Autonomous sessions, one per task. See the field's doc.
        autonomous_session_store: Arc<SessionStore>,
        mcp_session_store: Arc<SessionStore>,
        voice: Option<Arc<VoiceProviders>>,
        image_cache: Option<Arc<ImageCache>>,
        device_auth: Arc<crate::device_auth::DeviceAuth>,
        acp_session_store: Arc<AcpSessionStore>,
        subagent_cache: Option<Arc<SubagentCache>>,
    ) -> Self {
```

`Self { ... }` の初期化でも同じ位置に `autonomous_session_store,` を足し、テスト fixture `build_for_test_with` の `Self { ... }` にも足す。

```rust
            autonomous_session_store: Arc::new(
                SessionStore::new(base.join("autonomous"), "autonomous", None)
                    .with_dated_files(4),
            ),
```

`store_for_session` を差し替える。

```rust
    pub(crate) fn store_for_session(&self, session_id: &str) -> &Arc<SessionStore> {
        if self
            .autonomous_session_store
            .absolute_path_for(session_id)
            .is_some()
        {
            &self.autonomous_session_store
        } else if self
            .device_default_session_store
            .absolute_path_for(session_id)
            .is_some()
        {
            &self.device_default_session_store
        } else {
            &self.cross_device_session_store
        }
    }
```

`NullProgress` の実装の直後に `AutonomousHost` を足す。

```rust
/// The host for an autonomous turn.
///
/// Deliberately tiny: it carries the permission-table row and nothing
/// else. Everything an autonomous session *is* — the system prompt, the
/// tools, the persistence — is decided by `run_llm_turn` and the
/// configuration, exactly as it is for any other session. This type
/// exists so that the day the workspace-access policy lands (design doc,
/// decision 10) there is one place to teach about a directory scope
/// rather than four.
///
/// `round_budget` is *not* implemented: `Unattended` is the correct
/// answer, because nobody can cancel an autonomous turn in flight, and
/// the default is already that.
pub(crate) struct AutonomousHost {
    pub(crate) origin: crate::tools::policy::Origin,
}

#[async_trait::async_trait]
impl TurnHost for AutonomousHost {
    async fn tool_start(&self, _id: &str, _name: &str) {}
    async fn tool_end(&self, _id: &str, _name: &str) {}
    async fn turn_error(&self, message: &str) {
        warn!("Autonomous turn error: {message}");
    }

    fn origin(&self) -> crate::tools::policy::Origin {
        self.origin
    }
}
```

`for_test` 系の fixture は `ServeState::new` を直接呼んでいないので、`build_for_test_with` の `Self { ... }` に足すだけで済む。`ServeState::new` の本番呼び出しは Task 6 で直す（この Task の時点では `main.rs` が引数不足でコンパイルエラーになるので、**Task 4 と Task 6 の Step 3 は続けて行う**。分けたい場合は Task 6 の Step 3 の `main.rs` 変更だけ先に当ててよい）。

- [ ] **Step 4: テストが通ることを確認**

```bash
cargo test -p sapphire-agent serve:: 2>&1 | tail -20
```

期待: `test result: ok.`。`an_autonomous_session_resolves_to_the_autonomous_store` と `the_autonomous_host_is_trusted_and_unattended` が通ること。

- [ ] **Step 5: コミット**

```bash
git checkout -- Cargo.lock
cargo clippy --workspace -- -D warnings
git add server/src/serve/mod.rs
git commit -m "feat(serve): add the autonomous session store and turn host"
```

---


---

## 残りのタスク（Task 5〜9）

> Task 0〜4 は上で完了済み。ここからは `ServeState` に自律ストアと `AutonomousHost` が
> 居る状態（Task 4 の検証が緑）を前提にする。

---

### Task 5: 自律ループ本体（`server/src/autonomous.rs`）

**Files:**
- Create: `server/src/autonomous.rs`
- Modify: `server/src/main.rs`（`mod autonomous;` を `mod autonomous_config;` の直後に追加）
- Modify: `server/src/serve/mod.rs`（`LlmTurnOutcome` の3フィールドを `pub(crate)` にする。**可視性だけ**。`autonomous.rs` は `serve` の兄弟モジュールなので、`a2a.rs`（`serve` の子）と違って private フィールドは読めない）
- Test: `server/src/autonomous.rs` の `mod tests`

**Interfaces:**
- Produces:
  - `pub fn is_idle(rows: &[SessionRow], now: DateTime<Utc>, idle: Duration) -> bool`
  - `pub fn next_task<'a, F: FnMut(&AutonomousTask) -> bool>(tasks: &'a [AutonomousTask], due: F) -> Option<&'a AutonomousTask>`
  - `pub struct AutonomousState { pub status: String, pub reason: String, pub updated_at: DateTime<Utc> }`
  - `pub const CONTINUE_PROMPT: &str`
  - `pub fn marker(task: &str) -> String`
  - `pub struct AutonomousLoop` — `new(Arc<ServeState>, Option<Arc<SessionStore>>)` / `spawn(self)` / `run(self)` / `run_cycle(&mut self)`
- Consumes: `load_autonomous_dir` / `AutonomousTask`（Task 2）、`AutonomousConfig` / `AutonomousOrigin`（Task 3）、`ServeState::autonomous_session_store` / `AutonomousHost`（Task 4）、`SessionStore::{create_autonomous_session, session_rows, load_session, close_session, absolute_path_for}`（Task 1）、`run_llm_turn`（既存）

#### 設計上の決定（この Task で確定させる）

- **`run_cycle()` と `run()` を分ける。** `run()` は `interval` で `run_cycle()` を呼ぶだけ。テストは `run_cycle()` を直接叩き、sleep なしに1サイクルを検証する（`StubProvider` を刺した `ServeState` で完結する）。
- **タスク選定は「最初の enabled かつ due」。** ローダーが `priority` → `name` で並べているので、ここで並べ直さない（順序を2箇所に持つと、片方だけ直したときに静かに食い違う）。
- **2ターン目以降は本文を送り直さない。** `marker` + `CONTINUE_PROMPT` の2行だけ。本文はセッション履歴に既に入っている。
- **`DONE` は完全一致で見る。** ターンの最終テキストが `trim() == "DONE"` のときだけ早期に閉じる。説明文の途中に `DONE` と書いただけでは閉じない。`max_turns` はあくまで上限のまま（仕様 決定 12）。
- **プロセス内バックオフ（`not_before`）を持つ。** `cooldown_days = 0` + `max_turns = 3` が「セッション作成 → 3ターン → 即次」になると、一晩でセッションが何百本にもなる。store 側のアンカーが**再起動をまたぐ**正しさを、この map が**1プロセス内**の行儀を担う（仕様 決定 12）。
- **アイドル判定は `AutonomousLoop` が行を集める。** channel ストアは `ServeState` に無く `main.rs` が持っているので、コンストラクタで `Option<Arc<SessionStore>>` として受け取る（`None` = chat チャンネルの無い deployment）。**`ServeState` のフィールドは増やさない。**
- **`state/autonomous.json` は書きっぱなし。** 読むのは人間だけ。tmp + rename で書く（仕様 決定 4）。

- [ ] **Step 1: 失敗するテストを書く**

`server/src/autonomous.rs` を新規作成し、`mod tests` から始める（実装は Step 3）。

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::provider::ChatResponse;

    fn response(text: &str) -> ChatResponse {
        ChatResponse {
            prompt_usage: None,
            text: Some(text.to_string()),
            tool_calls: Vec::new(),
            stop_reason: None,
        }
    }

    fn row(room_id: &str, last_at: Option<DateTime<Utc>>, is_closed: bool) -> SessionRow {
        SessionRow {
            meta: crate::session::SessionMeta {
                session_id: "01920f00-0000-7000-8000-000000000000".to_string(),
                room_id: room_id.to_string(),
                thread_id: None,
                channel: "server".to_string(),
                created_at: Utc::now() - Duration::hours(5),
                public_id: None,
                namespace: Some("default".to_string()),
                project: None,
                device_id: None,
                room_profile: None,
                title: None,
            },
            message_count: 2,
            last_at,
            is_closed,
        }
    }

    fn task(name: &str) -> AutonomousTask {
        AutonomousTask {
            name: name.to_string(),
            enabled: true,
            priority: 100,
            cooldown_days: 0,
            max_turns: 1,
            body: "Do the thing.\n".to_string(),
        }
    }

    /// The whole idle rule: the newest activity in any *other* store.
    #[test]
    fn idle_is_true_only_when_nothing_else_moved_recently() {
        let now = Utc::now();
        let idle = Duration::minutes(30);

        assert!(is_idle(&[], now, idle), "no sessions at all is idle");
        assert!(
            is_idle(
                &[row("chat", Some(now - Duration::minutes(31)), false)],
                now,
                idle
            ),
            "older than idle_minutes is idle"
        );
        assert!(
            !is_idle(
                &[row("chat", Some(now - Duration::minutes(29)), false)],
                now,
                idle
            ),
            "inside idle_minutes is busy"
        );
        assert!(
            !is_idle(
                &[
                    row("old", Some(now - Duration::hours(4)), true),
                    row("new", Some(now - Duration::seconds(5)), false),
                ],
                now,
                idle
            ),
            "the max over every row decides, not the first one"
        );
        assert!(
            is_idle(&[row("never-written", None, false)], now, idle),
            "a session with no messages yet does not count as activity"
        );
    }

    #[test]
    fn only_the_first_due_enabled_task_is_picked() {
        let mut off = task("off");
        off.enabled = false;
        let mut later = task("later");
        later.cooldown_days = 7;
        // The loader's order is the priority order; `next_task` must not
        // reorder, only filter.
        let tasks = vec![off, later, task("due")];

        let picked = next_task(&tasks, |t| t.name == "due").expect("the due task");
        assert_eq!(picked.name, "due");

        assert!(next_task(&tasks, |_| false).is_none());
    }

    /// A task that just ran is not due; the same task with
    /// `cooldown_days = 0` is. Read through the real store, so the anchor
    /// rule (closed sessions included) is what is exercised.
    #[test]
    fn the_store_anchor_is_what_makes_a_task_due() {
        let state = crate::serve::ServeState::for_test(false);
        let store = &state.autonomous_session_store;
        let sid = store
            .create_autonomous_session("refactor", "default")
            .unwrap();
        store.append(&sid, &ChatMessage::user("hello")).unwrap();

        let rows = store.session_rows();
        let now = Utc::now();

        let mut fresh = task("refactor");
        fresh.cooldown_days = 0;
        assert!(is_due(&fresh, &rows, &HashMap::new(), now));

        let mut cooled = task("refactor");
        cooled.cooldown_days = 7;
        assert!(!is_due(&cooled, &rows, &HashMap::new(), now));

        // And the in-process back-off wins over the store even at
        // `cooldown_days = 0`.
        let mut backoff = HashMap::new();
        backoff.insert("refactor".to_string(), now + Duration::minutes(10));
        assert!(!is_due(&fresh, &rows, &backoff, now));
    }

    /// The end-to-end shape of one cycle: a session is created in the
    /// autonomous store, the task's body is the first user message with
    /// the marker and the session path in front of it, and `max_turns`
    /// closes it.
    #[tokio::test]
    async fn one_cycle_runs_the_task_and_closes_the_session() {
        let state = crate::serve::ServeState::for_test_scripted(false, vec![response("ok")]);
        let ws = state.workspace.dir().to_path_buf();
        std::fs::create_dir_all(ws.join("autonomous")).unwrap();
        std::fs::write(
            ws.join("autonomous/journal.md"),
            "---\nmax_turns: 1\n---\nSummarise the day.\n",
        )
        .unwrap();

        let mut lp = AutonomousLoop::new(Arc::clone(&state), None);
        lp.run_cycle().await.unwrap();

        let rows = state.autonomous_session_store.session_rows();
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].meta.room_id, "journal");
        assert_eq!(rows[0].meta.channel, "server");
        assert!(rows[0].is_closed, "max_turns = 1 must close the session");

        let sid = rows[0].meta.session_id.clone();
        let history = state.autonomous_session_store.load_session(&sid).unwrap();
        let first = match &history[0].parts[0] {
            crate::provider::ContentPart::Text(t) => t.clone(),
            other => panic!("expected text, got {other:?}"),
        };
        assert!(first.starts_with("[Autonomous: journal]\n"));
        assert!(first.contains("Session: sessions/default/autonomous/"));
        assert!(first.ends_with("Summarise the day.\n"));

        // The status file says what happened, and names the task.
        let raw = std::fs::read_to_string(ws.join("state/autonomous.json")).unwrap();
        let st: AutonomousState = serde_json::from_str(&raw).unwrap();
        assert!(st.reason.contains("journal"), "reason was {:?}", st.reason);
        assert!(st.status == "idle" || st.status == "running");
    }

    /// A cycle for a task inside its cooldown: nothing is created, and the
    /// status file says why nothing happened.
    #[tokio::test]
    async fn a_cooled_down_task_starts_nothing() {
        let state = crate::serve::ServeState::for_test(false);
        let ws = state.workspace.dir().to_path_buf();
        std::fs::create_dir_all(ws.join("autonomous")).unwrap();
        std::fs::write(
            ws.join("autonomous/refactor.md"),
            "---\ncooldown_days: 7\n---\nRefactor something.\n",
        )
        .unwrap();
        let sid = state
            .autonomous_session_store
            .create_autonomous_session("refactor", "default")
            .unwrap();
        state
            .autonomous_session_store
            .append(&sid, &ChatMessage::user("earlier run"))
            .unwrap();

        let mut lp = AutonomousLoop::new(Arc::clone(&state), None);
        lp.run_cycle().await.unwrap();

        assert_eq!(state.autonomous_session_store.session_rows().len(), 1);
        let raw = std::fs::read_to_string(ws.join("state/autonomous.json")).unwrap();
        let st: AutonomousState = serde_json::from_str(&raw).unwrap();
        assert_eq!(st.status, "idle");
        assert!(
            st.reason.contains("nothing due"),
            "reason was {:?}",
            st.reason
        );
    }

    /// A busy store means the cycle does not start a thing — and says so.
    #[tokio::test]
    async fn a_busy_agent_starts_nothing() {
        let state = crate::serve::ServeState::for_test(false);
        let ws = state.workspace.dir().to_path_buf();
        std::fs::create_dir_all(ws.join("autonomous")).unwrap();
        std::fs::write(ws.join("autonomous/journal.md"), "---\n---\nWork.\n").unwrap();

        // A freshly-touched cross-device session, i.e. somebody is talking.
        let chat = state
            .cross_device_session_store
            .create_session(&("room".to_string(), None), "rpc", "default")
            .unwrap();
        state
            .cross_device_session_store
            .append(&chat, &ChatMessage::user("hello"))
            .unwrap();

        let mut lp = AutonomousLoop::new(Arc::clone(&state), None);
        lp.run_cycle().await.unwrap();

        assert!(state.autonomous_session_store.session_rows().is_empty());
        let raw = std::fs::read_to_string(ws.join("state/autonomous.json")).unwrap();
        let st: AutonomousState = serde_json::from_str(&raw).unwrap();
        assert_eq!(st.status, "idle");
        assert!(st.reason.contains("busy"), "reason was {:?}", st.reason);
    }
}
```

- [ ] **Step 2: テストが落ちることを確認**

```bash
cargo test -p sapphire-agent autonomous:: 2>&1 | tail -20
```

期待: `error[E0583]: file not found for module `autonomous``（`mod` 宣言を先に足していれば `error[E0425]: cannot find function `is_idle`` などのコンパイルエラー）。

- [ ] **Step 3: 実装する**

`server/src/serve/mod.rs` の `LlmTurnOutcome` のフィールドを `pub(crate)` にする（他は無変更）。

```rust
pub(crate) struct LlmTurnOutcome {
    pub(crate) text: Option<String>,
    pub(crate) was_first_turn: bool,
    pub(crate) stop: TurnStop,
}
```

`server/src/autonomous.rs` を新規作成する。

```rust
//! Autonomous sessions: the third loop.
//!
//! `heartbeat.rs` runs two loops, both on a clock. This one is not: it
//! wakes, asks whether anything else is happening, and if not picks the
//! highest-priority due task out of `<workspace>/autonomous/*.md` and
//! runs it as an ordinary session, until that task's `max_turns` is spent
//! or the model says it is done.
//!
//! It is a *poll*, not a cadence. Every cycle re-reads the task
//! directory, so editing a task file takes effect within `poll_seconds`
//! and without a restart. A cycle that finds nothing costs one directory
//! read and one meta-line scan per store.
//!
//! Reusing `run_llm_turn` is the whole trick: an autonomous turn is a
//! normal turn, so the system prompt, the tools, the persistence and the
//! compression all behave as they do for any other session. The only new
//! thing is which store it lands in and which row of the permission table
//! judges it.
//!
//! Nothing here is persisted except the status file: which session a task
//! is in, when it last ran, and how many turns it has spent are all
//! derived from the session store (design decisions 3 and 5), because a
//! second copy of that state is a second thing to keep true.

use crate::autonomous_config::{load_autonomous_dir, AutonomousTask};
use crate::config::{AutonomousOrigin, DEFAULT_NAMESPACE_NAME};
use crate::provider::{ChatMessage, Role};
use crate::serve::{AutonomousHost, ServeState};
use crate::session::{SessionRow, SessionStore};
use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use tracing::{info, warn};

/// The whole instruction for every turn after the first.
///
/// The body is deliberately not repeated — it is already in the session's
/// history, and re-sending it every turn invalidates the provider's
/// prompt cache for nothing.
///
/// The last line is a protocol rather than politeness: the loop closes the
/// session the moment a turn's final text is exactly `DONE`, so a task
/// that finishes early does not spend its remaining turns announcing it.
pub const CONTINUE_PROMPT: &str =
    "Continue the task. When it is finished, reply with exactly:\nDONE";

/// The first line of every user message this loop sends.
///
/// Not a search key — the session's `room_id` is (design decision 3) —
/// but the model has to be able to tell a system-fired start from the user
/// speaking.
pub fn marker(task: &str) -> String {
    format!("[Autonomous: {task}]")
}

/// `<workspace>/state/autonomous.json`, contents only.
///
/// `status` is `"idle"` or `"running"`, `reason` is the one-line human
/// answer to "why", and `updated_at` is how a human tells a live loop from
/// a stopped one. Nothing machine-reads this file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutonomousState {
    pub status: String,
    pub reason: String,
    pub updated_at: DateTime<Utc>,
}

/// Write the status file. Best-effort: a cycle that cannot write its own
/// status has still done (or not done) its work, and failing the cycle
/// over a bookkeeping file would be the tail wagging the dog.
fn write_state(workspace_dir: &Path, status: &str, reason: String) {
    let dir = workspace_dir.join("state");
    if let Err(e) = std::fs::create_dir_all(&dir) {
        warn!("autonomous: cannot create {}: {e}", dir.display());
        return;
    }
    let state = AutonomousState {
        status: status.to_string(),
        reason,
        updated_at: Utc::now(),
    };
    let path = dir.join("autonomous.json");
    let tmp = dir.join(".autonomous.json.tmp");
    let Ok(raw) = serde_json::to_string_pretty(&state) else {
        return;
    };
    // tmp + rename: a reader that catches the loop mid-write must never
    // see half a JSON document. Not routed through the framework
    // WorkspaceState — this is not a workspace document, and it is not
    // indexed.
    if std::fs::write(&tmp, raw)
        .and_then(|_| std::fs::rename(&tmp, &path))
        .is_err()
    {
        warn!("autonomous: cannot write {}", path.display());
    }
}

/// Idle iff the newest activity in the *other* stores is at least `idle`
/// old. Rows with no messages do not count as activity — an empty session
/// is not somebody talking (design decision 9).
pub fn is_idle(rows: &[SessionRow], now: DateTime<Utc>, idle: Duration) -> bool {
    match rows.iter().filter_map(|r| r.last_at).max() {
        Some(last) => now - last >= idle,
        None => true,
    }
}

/// The first enabled, due task. The order is the loader's (`priority`,
/// then `name`); this only filters, so exactly one place decides what
/// "first" means.
pub fn next_task<'a, F>(tasks: &'a [AutonomousTask], mut due: F) -> Option<&'a AutonomousTask>
where
    F: FnMut(&AutonomousTask) -> bool,
{
    tasks.iter().find(|t| t.enabled && due(t))
}

/// May `task` start now?
///
/// The anchor is the task's *latest* session activity, closed or not. A
/// closed session is the common case — `max_turns` closes one — and
/// exactly the case a cooldown has to remember, because the file is still
/// there to be found (design decision 5).
fn is_due(
    task: &AutonomousTask,
    rows: &[SessionRow],
    not_before: &HashMap<String, DateTime<Utc>>,
    now: DateTime<Utc>,
) -> bool {
    if let Some(at) = not_before.get(&task.name)
        && *at > now
    {
        return false;
    }
    let anchor = rows
        .iter()
        .filter(|r| r.meta.room_id == task.name)
        .map(|r| r.last_at.unwrap_or(r.meta.created_at))
        .max();
    match anchor {
        Some(a) => now - a >= Duration::days(task.cooldown_days as i64),
        None => true,
    }
}

/// Where this task's work goes next: the newest *open* session for it, or
/// `None`, meaning "start a new one".
fn resume_target(rows: &[SessionRow], task: &str) -> Option<String> {
    rows.iter()
        .filter(|r| r.meta.room_id == task && !r.is_closed)
        .max_by_key(|r| r.last_at.unwrap_or(r.meta.created_at))
        .map(|r| r.meta.session_id.clone())
}

/// How many turns this session has already spent. User messages, because
/// every turn of this loop appends exactly one.
fn turns_spent(store: &SessionStore, session_id: &str) -> usize {
    store
        .load_session(session_id)
        .map(|h| h.iter().filter(|m| m.role == Role::User).count())
        .unwrap_or(0)
}

/// The idle loop.
pub struct AutonomousLoop {
    state: Arc<ServeState>,
    /// `None` when this deployment has no chat channel configured — in
    /// which case there is no channel traffic to be idle *from*. Held here
    /// rather than on `ServeState` because only this loop wants it, and
    /// `ServeState` is built before the channel store exists.
    channel_session_store: Option<Arc<SessionStore>>,
    workspace_dir: PathBuf,
    idle: Duration,
    poll: Duration,
    /// In-process back-off: task name → the instant it may run again.
    ///
    /// The store's anchor is what survives a restart; this is what keeps
    /// `cooldown_days = 0` from meaning "a new session every cycle" within
    /// one run (design decision 12).
    not_before: HashMap<String, DateTime<Utc>>,
}

impl AutonomousLoop {
    pub fn new(state: Arc<ServeState>, channel_session_store: Option<Arc<SessionStore>>) -> Self {
        let workspace_dir = state.workspace.dir().to_path_buf();
        let cfg = &state.config.autonomous;
        Self {
            state,
            channel_session_store,
            workspace_dir,
            idle: Duration::minutes(cfg.idle_minutes as i64),
            poll: Duration::seconds(cfg.poll_seconds.max(1) as i64),
            not_before: HashMap::new(),
        }
    }

    /// Which row of the permission table this loop's turns are judged by.
    fn origin(&self) -> crate::tools::policy::Origin {
        match self.state.config.autonomous.origin {
            AutonomousOrigin::Channel => crate::tools::policy::Origin::Channel,
            AutonomousOrigin::Trusted => crate::tools::policy::Origin::Trusted,
        }
    }

    /// Every session this loop must treat as "somebody might be talking" —
    /// *except* its own store's. An autonomous session is not activity
    /// that should keep autonomous sessions from starting.
    fn busy_rows(&self) -> Vec<SessionRow> {
        let mut rows = Vec::new();
        if let Some(channel) = &self.channel_session_store {
            rows.extend(channel.session_rows());
        }
        rows.extend(self.state.cross_device_session_store.session_rows());
        rows.extend(self.state.device_default_session_store.session_rows());
        rows.extend(self.state.mcp_session_store.session_rows());
        rows.extend(self.state.acp_session_store.session_rows());
        rows
    }

    pub fn spawn(self) {
        tokio::spawn(self.run());
    }

    pub async fn run(mut self) {
        // `interval` panics on a zero period; `new` already floors it at a
        // second, and the first tick is discarded so startup does not
        // immediately start working (same as both heartbeat loops).
        let mut tick = tokio::time::interval(self.poll);
        tick.tick().await;
        loop {
            tick.tick().await;
            if let Err(e) = self.run_cycle().await {
                warn!("autonomous cycle failed: {e:#}");
            }
        }
    }

    /// One pass: idle? task? run it. Separated from `run` so a test can
    /// drive a cycle with no clock in the way.
    pub async fn run_cycle(&mut self) -> anyhow::Result<()> {
        let now = Utc::now();
        let tasks = load_autonomous_dir(&self.workspace_dir.join("autonomous"));
        if tasks.iter().all(|t| !t.enabled) {
            write_state(&self.workspace_dir, "idle", "no tasks".to_string());
            return Ok(());
        }

        if !is_idle(&self.busy_rows(), now, self.idle) {
            write_state(
                &self.workspace_dir,
                "idle",
                "busy: another session is active".to_string(),
            );
            return Ok(());
        }

        let store = Arc::clone(&self.state.autonomous_session_store);
        let rows = store.session_rows();
        let picked = next_task(&tasks, |t| is_due(t, &rows, &self.not_before, now)).cloned();
        let Some(task) = picked else {
            write_state(&self.workspace_dir, "idle", "nothing due".to_string());
            return Ok(());
        };

        self.run_task(&task, &store, &rows).await
    }

    /// The session-level unit of work: resume or create, then turn until
    /// done, capped, or interrupted by somebody else starting to talk.
    async fn run_task(
        &mut self,
        task: &AutonomousTask,
        store: &Arc<SessionStore>,
        rows: &[SessionRow],
    ) -> anyhow::Result<()> {
        let namespace = DEFAULT_NAMESPACE_NAME;
        let session_id = match resume_target(rows, &task.name) {
            Some(sid) => sid,
            None => store.create_autonomous_session(&task.name, namespace)?,
        };
        let mut turns = turns_spent(store, &session_id);

        // The session's own path, workspace-relative: the second line of
        // the first message, so a task that keeps a journal can name the
        // file it is writing (design decision 11). Derived from the store,
        // never from a template, so it cannot drift from the real name.
        let session_rel = store
            .absolute_path_for(&session_id)
            .and_then(|p| p.strip_prefix(&self.workspace_dir).ok().map(Path::to_path_buf))
            .map(|p| p.display().to_string())
            .unwrap_or_else(|| format!("sessions/{namespace}/autonomous/{session_id}.jsonl"));

        loop {
            let text = if turns == 0 {
                format!(
                    "{}\nSession: {session_rel}\n\n{}",
                    marker(&task.name),
                    task.body
                )
            } else {
                format!("{}\n{CONTINUE_PROMPT}", marker(&task.name))
            };
            write_state(
                &self.workspace_dir,
                "running",
                format!(
                    "task: {}, turn {}/{}",
                    task.name,
                    turns + 1,
                    task.max_turns
                ),
            );

            let outcome = crate::serve::run_llm_turn(
                Arc::clone(&self.state),
                session_id.clone(),
                ChatMessage::user(text),
                Arc::new(AutonomousHost {
                    origin: self.origin(),
                }),
                None,
            )
            .await;

            turns += 1;
            let said_done = outcome
                .text
                .as_deref()
                .is_some_and(|t| t.trim() == "DONE");
            // `text == None` means the provider failed or the tool-round
            // budget ran out. Either way the turn produced no answer, so
            // the session is spent; closing it makes the next cycle start
            // clean instead of appending to a conversation that already
            // went wrong.
            let spent = outcome.text.is_none();
            let capped = turns >= task.max_turns;

            if said_done || spent || capped {
                if let Err(e) = store.close_session(&session_id) {
                    warn!("autonomous: cannot close {}: {e}", task.name);
                }
                // Hold the task off even at `cooldown_days = 0`: the store
                // anchor clears instantly when the cooldown is zero, and
                // "as soon as the previous session ended" must not mean
                // "again in this same cycle".
                let backoff = Duration::days(task.cooldown_days as i64).max(self.poll);
                self.not_before
                    .insert(task.name.clone(), Utc::now() + backoff);
                info!(
                    "autonomous: {} finished after {turns} turn(s) (done={said_done}, capped={capped})",
                    task.name
                );
                write_state(
                    &self.workspace_dir,
                    "idle",
                    format!("task: {} finished after {turns} turn(s)", task.name),
                );
                return Ok(());
            }

            // Between turns only: interrupting a turn in flight is out of
            // scope (design decision 13). Somebody talking now means stop
            // *after* this turn and leave the session open, so the next
            // quiet moment continues it instead of starting over.
            if !is_idle(&self.busy_rows(), Utc::now(), self.idle) {
                write_state(
                    &self.workspace_dir,
                    "idle",
                    format!("busy: paused {}, {turns} turn(s) spent", task.name),
                );
                return Ok(());
            }
        }
    }
}
```

`server/src/main.rs` の `mod autonomous_config;` の直後に `mod autonomous;` を足す。

- [ ] **Step 4: テストが通ることを確認**

```bash
cargo test -p sapphire-agent autonomous:: 2>&1 | tail -20
```

期待: `test result: ok.` で6件。`serve::` の既存テストも緑のままであること（`LlmTurnOutcome` の可視性変更は挙動を変えない）。

- [ ] **Step 5: コミット**

```bash
git checkout -- Cargo.lock
cargo clippy --workspace -- -D warnings
git add server/src/autonomous.rs server/src/main.rs server/src/serve/mod.rs
git commit -m "feat(autonomous): run idle-time tasks as sessions"
```

---

### Task 6: `main.rs` に配線する

**Files:**
- Modify: `server/src/main.rs`（自律ストアの構築、`ServeState::new` の引数、ループの spawn）
- Modify: `server/src/tools/session_tools.rs`（テストヘルパ `sources()` のみ — `SessionSources::new` の引数が1本増えるため。本実装は Task 7）
- Test: 既存テストのコンパイル（配線なので新規テストは無い。挙動は Task 5 のサイクルテストが固定している）

**Interfaces:**
- Consumes: `AutonomousLoop::new`（Task 5）、`ServeState::new` の新引数（Task 4）
- Produces: なし（配線のみ）

- [ ] **Step 1: 自律ストアを構築する**

`server/src/main.rs` の channel ストア構築（`let channel_session_store = ...`）の直後に足す。**`sessions_base` を共有し、`kind` ディレクトリだけが違う**のは他のストアと同じ。

```rust
            // ── Autonomous session store (sessions/<ns>/autonomous/) ────────
            // Where the idle loop's sessions live. Built unconditionally —
            // construction touches nothing — so `session_list` can read what
            // last night's loop wrote even in a process that has autonomous
            // sessions turned off. Files are named `{agent-day}-{uuid}.jsonl`
            // so an overnight run reads as a run in a directory listing; the
            // session's own metadata is one line away and unchanged.
            let autonomous_session_store = Arc::new(
                SessionStore::with_workspace(
                    sessions_base.clone(),
                    "autonomous",
                    Arc::clone(&ws_state),
                    tool_payload_cache.clone(),
                )
                .with_dated_files(config.day_boundary_hour),
            );
```

- [ ] **Step 2: `ServeState::new` に渡す**

Task 4 で確定した引数順に合わせ、`device_default_session_store` の直後に1本足す。

```rust
                Arc::clone(&cross_device_session_store),
                Arc::clone(&device_default_session_store),
                Arc::clone(&autonomous_session_store),
                Arc::clone(&mcp_session_store),
```

- [ ] **Step 3: ループを spawn する**

`let serve_state = Arc::new(serve::ServeState::new(...));` と
`timer_manager.set_serve_state(Arc::downgrade(&serve_state));` の後、
**チャンネル設定の `if` の外**（`// ── Channel + Agent` ブロックの前）に足す。

```rust
            // ── Autonomous sessions (off unless configured) ─────────────────
            // Spawned outside the channel block on purpose: this loop needs
            // `ServeState` and nothing else, so an ACP-only or voice-only
            // deployment gets the feature too. `enabled = false` (the
            // default) means no loop, no status file, no directory.
            if config.autonomous.enabled {
                autonomous::AutonomousLoop::new(
                    Arc::clone(&serve_state),
                    Some(Arc::clone(&channel_session_store)),
                )
                .spawn();
            } else {
                tracing::info!("Autonomous sessions disabled by config");
            }
```

- [ ] **Step 4: ビルドと既存テスト**

```bash
cargo test -p sapphire-agent 2>&1 | tail -20
```

期待: `test result: ok.`。`ServeState::new` の引数追加でコンパイルエラーが出る場合は、呼び出し元は `main.rs` と `serve/mod.rs` のテスト fixture（`build_for_test_with`）の2箇所だけである。

- [ ] **Step 5: 設定を1つ書いて手で確かめる（任意・プロバイダが必要）**

自動テストは `StubProvider` でループの形だけを固定する。実際にモデルを1回回す確認は、
**llama.cpp を指す既存の設定**（`[profiles.background]`）に `[autonomous]` を足して行う。

```bash
# 1. 作業用ワークスペースを用意し、タスクを1つ置く
sapphire-agent-server init /tmp/autonomous-smoke
mkdir -p /tmp/autonomous-smoke/autonomous
cat > /tmp/autonomous-smoke/autonomous/smoke.md <<'EOF'
---
max_turns: 1
---
Read AGENTS.md and reply with one sentence about what this workspace is.
EOF

# 2. 設定に以下を足して起動する
#    [autonomous] enabled = true / idle_minutes = 1 / poll_seconds = 10 / origin = "channel"
sapphire-agent-server --config ~/.config/sapphire-agent/config.toml
```

期待: 起動ログに `Autonomous sessions disabled by config` が出**ない**こと。
`idle_minutes` 経過後に `/tmp/autonomous-smoke/state/autonomous.json` が現れ、
`reason` が `task: smoke, turn 1/1` → `task: smoke finished after 1 turn(s)` と進み、
`/tmp/autonomous-smoke/sessions/default/autonomous/` に `2026-09-12-<uuid>.jsonl` が
**1本だけ**出来ていること。`enabled` を書かずに起動した場合は `state/` も
`sessions/default/autonomous/` も作られないこと（受け入れ基準 1）。

- [ ] **Step 6: コミット**

```bash
git checkout -- Cargo.lock
cargo clippy --workspace -- -D warnings
git add server/src/main.rs server/src/tools/session_tools.rs
git commit -m "feat(autonomous): wire the idle loop into startup"
```

---

### Task 7: `session_list` / `session_read` に自律ストアを足す

**Files:**
- Modify: `server/src/tools/session_tools.rs`（`SessionSources` のフィールドと `new`、`all_rows`、`transcript`、`mod tests`）
- Test: `server/src/tools/session_tools.rs` の `mod tests`

**Interfaces:**
- Produces: `SessionSources::new(config, channel, cross_device, device_default, autonomous, acp)` — `autonomous: Arc<SessionStore>` が `device_default` の直後
- Consumes: Task 6 の `autonomous_session_store`

- [ ] **Step 1: 失敗するテストを書く**

```rust
    /// Autonomous sessions are ordinary sessions as far as reading back
    /// goes: last night's work is listed and readable from a chat, which
    /// is the *only* way a human sees it (nothing is posted to a channel).
    #[tokio::test]
    async fn an_autonomous_session_is_listed_and_readable() {
        let d = tempfile::tempdir().unwrap();
        let sources = sources(&d);
        let sid = sources
            .autonomous
            .create_autonomous_session("journal", "default")
            .unwrap();
        sources
            .autonomous
            .append(
                &sid,
                &ChatMessage::user("[Autonomous: journal]\n\nSummarise the day."),
            )
            .unwrap();
        sources
            .autonomous
            .append(&sid, &ChatMessage::assistant("Done in three lines."))
            .unwrap();

        let listed = SessionListTool::new(Arc::clone(&sources))
            .execute(&json!({}))
            .await
            .unwrap();
        assert!(listed.contains(&sid), "listing was {listed}");
        assert!(listed.contains("server/journal"), "listing was {listed}");

        let read = SessionReadTool::new(sources)
            .execute(&json!({"session_id": sid}))
            .await
            .unwrap();
        assert!(read.contains("Done in three lines."), "read was {read}");
    }
```

- [ ] **Step 2: テストが落ちることを確認**

```bash
cargo test -p sapphire-agent an_autonomous_session_is_listed 2>&1 | tail -20
```

期待: `error[E0609]: no field `autonomous` on type `SessionSources``（または `SessionSources::new` の引数不足）。

- [ ] **Step 3: 実装する**

`SessionSources` にフィールドを足す。

```rust
    /// The autonomous store (kind = `"autonomous"`). Included for the same
    /// reason the others are: the loop's sessions are ordinary sessions, and
    /// `session_list` is where a human finds out what happened overnight —
    /// nothing is posted to a chat (design decision 11).
    autonomous: Arc<SessionStore>,
```

`new` の引数と `Self { .. }` に `autonomous` を `device_default` の直後で足す。

```rust
    pub fn new(
        config: Config,
        channel: Option<Arc<SessionStore>>,
        cross_device: Arc<SessionStore>,
        device_default: Arc<SessionStore>,
        autonomous: Arc<SessionStore>,
        acp: Arc<AcpSessionStore>,
    ) -> Self {
```

`all_rows` に1行、`transcript` の探索配列に1本足す。

```rust
        rows.extend(self.cross_device.session_rows());
        rows.extend(self.device_default.session_rows());
        rows.extend(self.autonomous.session_rows());
        rows.extend(self.acp.session_rows());
```

```rust
            let stores = [
                self.sources.channel.as_ref(),
                Some(&self.sources.cross_device),
                Some(&self.sources.device_default),
                Some(&self.sources.autonomous),
            ];
```

`main.rs` の `SessionSources::new` 呼び出しにも `Arc::clone(&autonomous_session_store)` を足し、
`mod tests` の `sources()` を直す。

```rust
        Arc::new(SessionSources::new(
            config(),
            Some(Arc::new(SessionStore::new(base.clone(), "channel", None))),
            Arc::new(SessionStore::new(base.clone(), "cross-device", None)),
            Arc::new(SessionStore::new(base.clone(), "device-default", None)),
            Arc::new(SessionStore::new(base.clone(), "autonomous", None)),
            Arc::new(AcpSessionStore::new(base, None)),
        ))
```

- [ ] **Step 4: テストが通ることを確認**

```bash
cargo test -p sapphire-agent session_tools 2>&1 | tail -20
```

期待: `test result: ok.`。既存の `session_list` / `session_read` のテスト（namespace スコープ、
offset/limit、ACP 経路）が全部緑のままであること。

- [ ] **Step 5: コミット**

```bash
git checkout -- Cargo.lock
cargo clippy --workspace -- -D warnings
git add server/src/tools/session_tools.rs server/src/main.rs
git commit -m "feat(session-tools): let session_list read the autonomous store"
```

---

### Task 8: ワークスペース面（テンプレート・README）

**Files:**
- Create: `server/templates/workspace/example-autonomous.md`
- Modify: `server/src/cli_init.rs`（`FILE_TEMPLATES` / `EXPECTED_FILES`）
- Modify: `README.md`（`## Autonomous sessions` を追加）
- Test: `server/src/cli_init.rs` の `mod tests`（既存の2件が新しいファイルを要求する）

**Interfaces:**
- Consumes: なし（テンプレートと文書だけ）

- [ ] **Step 1: テンプレートを追加する**

`server/templates/workspace/example-autonomous.md`（`example-morning.md` と同じ体裁。
**既定は `enabled: false`** — `init` しただけのワークスペースが勝手に働き出さないこと）。

```markdown
---
enabled: false
priority: 10
cooldown_days: 7
max_turns: 3
---

# Example: look for refactoring work

Look over this workspace and the projects you know about, and pick one small
improvement worth doing. Do it if it is genuinely small — otherwise write it
down where you will find it again.

Work in this session until the task is finished. When you are done, reply
with exactly:

DONE

The path of this session's transcript is in the first message of this
conversation (the `Session:` line). If you keep a journal, record what you did
there — what happened, what you decided, and what is still open — so the next
run can read it instead of guessing.
```

- [ ] **Step 2: `cli_init` に登録する**

`server/src/cli_init.rs` の `FILE_TEMPLATES` の `heartbeat/example-morning.md` の直後と、
`EXPECTED_FILES` の同じ位置に足す。

```rust
    (
        "autonomous/example-autonomous.md",
        include_str!("../templates/workspace/example-autonomous.md"),
    ),
```

```rust
        "autonomous/example-autonomous.md",
```

- [ ] **Step 3: `README.md` に節を足す**

`## Skills` の前に `## Autonomous sessions` を置く。

```markdown
## Autonomous sessions

When nothing else has touched the agent for `idle_minutes`, it picks the
highest-priority due task out of `<workspace>/autonomous/*.md` and works on it
as an ordinary session, for up to that task's `max_turns`.

Three things separate this from a heartbeat task:

- **It starts on idleness, not on a clock.** A heartbeat fires at a time; this
  waits until no other session has moved.
- **It keeps going until the task is done.** A heartbeat is one prompt; this is
  a session, resumed across quiet moments and closed when the model answers
  `DONE` or `max_turns` runs out.
- **One task is one session.** `session_list` shows them as `server/<task>` and
  `session_read` shows the transcript, so last night's work is something you
  read rather than something that was posted at you.

Off by default:

```toml
[autonomous]
enabled = true
idle_minutes = 30    # how quiet everything else has to be
poll_seconds = 60    # how often the loop looks for work
origin = "channel"   # "channel" (default) | "trusted"
```

`origin = "channel"` runs tasks with the same permissions as a chat message:
reads and unapproved edits, no `shell`. Tasks that need `shell` (filing an
issue, committing) need `origin = "trusted"` **and**
`[tools] host_access.enabled = true` — without the latter, the host tools are
refused before permissions are consulted at all.

While a task runs, `<workspace>/state/autonomous.json` carries a one-line
reason (`task: journal, turn 2/3`) so you can tell what the agent is doing
without reading the session. Task definitions are re-read every cycle, so
editing one takes effect without a restart.
```

- [ ] **Step 4: テストが通ることを確認**

```bash
cargo test -p sapphire-agent cli_init 2>&1 | tail -20
```

期待: `test result: ok.`。`a_fresh_directory_gets_every_workspace_file` と、既存ファイルを
上書きしないテストが、新しい `autonomous/example-autonomous.md` を含めて緑であること。

- [ ] **Step 5: コミット**

```bash
git checkout -- Cargo.lock
cargo clippy --workspace -- -D warnings
git add server/templates/workspace/example-autonomous.md server/src/cli_init.rs README.md
git commit -m "docs(autonomous): ship an example task and document the loop"
```

---

### Task 9: 全体検証

**Files:** なし（検証と PR だけ）

- [ ] **Step 1: クレート全体でテストする**

```bash
cargo test -p sapphire-agent 2>&1 | tail -30
```

期待: `test result: ok.` が全バイナリ分並び、`failed` が0件。落ちた場合は直近のコミットを
戻すのではなく、その失敗を直すコミットを積む。

- [ ] **Step 2: ワークスペース全体（cli / desktop / core / client を含む）でテストする**

```bash
cargo test --workspace 2>&1 | tail -30
```

期待: `test result: ok.`。ここだけは時間がかかる（desktop の bevy を含むため）。
**`cargo test -p sapphire-agent` と交互に走らせない** — フィンガープリントが違うので
毎回リンクし直しになる。

- [ ] **Step 3: clippy と rustfmt**

```bash
cargo clippy --workspace -- -D warnings 2>&1 | tail -20
cargo fmt --all -- --check
```

期待: どちらも出力なし（clippy は warning が1つも無いこと）。`cargo fmt` が差分を出す場合は
`cargo fmt --all` してから、その差分を**直前のコミットに混ぜず** `style: rustfmt` として
別コミットにする（既存の慣習、`e04784d`）。

- [ ] **Step 4: 受け入れ基準を通しで確認する**

| 仕様の受け入れ基準 | 確認する場所 |
|---|---|
| 1. 既定で無効 | `config::tests::autonomous_is_disabled_by_default`、`main.rs` の `enabled` 分岐 |
| 2. 1セッションが作られ本文が最初の user メッセージになる | `autonomous::tests::one_cycle_runs_the_task_and_closes_the_session`、Task 6 Step 5 の手動確認 |
| 3. 動きがある間は始まらない／静かになったら同じセッションの続き | `autonomous::tests::a_busy_agent_starts_nothing`（開始しない側）、`resume_target` が `!is_closed` で絞る（再開側） |
| 4. `max_turns` 到達で閉じ、次は新規セッション | 同テストの `is_closed` 表明と `resume_target` |
| 5. `cooldown_days` の内側では due にならない | `autonomous::tests::the_store_anchor_is_what_makes_a_task_due`、`a_cooled_down_task_starts_nothing` |
| 6. タスク0件／壊れたファイルは飛ばす | `autonomous_config::tests` の6件 |
| 7. `state/autonomous.json` に理由が出る | サイクルテスト3件の `AutonomousState` 表明 |
| 8. `session_list` に `server/<task>`、`session_read` で読める | `session_tools::tests::an_autonomous_session_is_listed_and_readable` |
| 9. 既存の挙動が変わらない | `session::tests::a_plain_store_still_matches_its_filename_exactly`、`serve::` と `session_tools::` の既存テスト |
| 10. 既定 `channel` では `shell` が拒否、`trusted` + `host_access` で通る | `serve::tests::the_autonomous_host_carries_the_configured_origin`、`tools::policy` の既存 decide テスト |
| 11. 最初の user メッセージに `Session: <相対パス>` が入る | `autonomous::tests::one_cycle_runs_the_task_and_closes_the_session` |

- [ ] **Step 5: PR を作る**

```bash
git push -u origin feat/issue-248-autonomous-sessions
gh pr create --title "feat: autonomous sessions (#248)" --body "$(cat <<'EOF'
Implements #248.

The agent now works on its own when nothing else is happening: an idle loop
picks the highest-priority due task from `<workspace>/autonomous/*.md` and runs
it as an ordinary session, one session per task.

- new `SessionStore` kind `"autonomous"` (`sessions/<ns>/autonomous/`,
  `{agent-day}-{uuid}.jsonl`), reached through the existing `run_llm_turn` with
  an `AutonomousHost` that only carries the permission row
- the third loop (`autonomous.rs`), modelled on heartbeat's, gated on
  `[autonomous] enabled` — off by default
- `session_list` / `session_read` read the new store, so a night's work is
  listed and readable from a chat; nothing is posted to a channel
- nothing changes for `channel` / `cross-device` / `device-default` / `mcp` /
  `acp` sessions, and `Origin` / `decide` are untouched

Spec: `docs/superpowers/specs/2026-09-12-autonomous-sessions-design.md`
Plan: `docs/superpowers/plans/2026-09-12-autonomous-sessions-plan.md`
EOF
)"
```

期待: PR のURLが出る。`gh` が未認証なら `gh auth login` を先に行う（他の手順は `gh` を要らない）。

---

## スコープ外（別リポジトリ・別 Issue）

- **日次ノートへの注入の実装**。sapphire-agent 側は「最初の user メッセージに
  `Session: <相対パス>` を渡す」ところまで（Task 5）。日次ノート
  （`memory/<ns>/daily/YYYY-MM-DD.md`）にこのパスを書き戻す仕組みは
  **sapphire-journal 側**の仕事で、本計画には入らない。
- 自律ターンのチャット配信（仕様 決定 11）、進行中ターンのレジストリ（リスク 1）、
  暴走検知（リスク 3）。
