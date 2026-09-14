# 設定ファイル操作ツール（heartbeat / autonomous / agents） 実装計画

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** エージェントが自分の `heartbeat/*.md` / `autonomous/*.md` / `agents/*.md` を読み書きできる4本の管理ツール（`heartbeat_config` / `autonomous_config` / `agent_config` / `task_test`）を追加し、`[tools.admin].rooms` で許可されたルームからのみ使えるようにする。あわせて `agents/*.md` のホットリロードを入れる。

**Architecture:** 1つの `ConfigTool` を「どのディレクトリを見るか」だけ変えて3回登録する。`kind()` は `Edit` 1つ。ルーム許可は「そもそも登録しない（`rooms` が空）＋ 実行時にルーム id で拒否」の2段で表現する。ルーム id は**チャンネル経路が既に scoped している `timer::current_origin()` の `TimerOrigin::Chat { room_id }`** から取る（`TurnContext` も `agent.rs` も触らない）。無効化は frontmatter の `enabled:` **行だけ**を書き換える（コメント・フィールド順・body を1バイトも変えない）。`task_test` は本番と同じ `run_llm_turn` を本番と同じプロンプト組み立てで走らせ、**`test:<kind>:<name>` という別の room_id** のセッションに落とすので、テストしただけで本番タスクのクールダウンをリセットしない。サブエージェントには `enabled` もテスト用ツールも足さない（呼ばれなければ走らないので状態が存在しない）。

**Tech Stack:** Rust 2024, `serde` / `serde_yaml`, `tokio`, `async-trait`, `serde_json`

**Spec:** `docs/superpowers/specs/2026-09-14-config-tools-design.md`

## Global Constraints

- ブランチは `feat/issue-265-config-tools`（`main` = `00fb983` から作成し、本計画書と仕様書のコミットを最初に載せる）。
- テストは `cargo test -p sapphire-agent`。反復中は `--workspace` を使わない。最後に一度だけ `cargo test --workspace`。
- **`cargo check` と `cargo test` を混ぜない**（フィンガープリントが別で、交互に走らせると毎回リンクし直しになる）。
- cargo は**前景で `timeout: 600000`**。バックグラウンド実行・2本同時実行は禁止（小さい USB SSD 上にあり熱でスロットリングする）。
- **コミット前に `cargo clippy --workspace -- -D warnings`**（CI と同形。`--all-targets` は付けない）。
- **`Cargo.lock` をコミットしない。** 各コミット前に `git checkout -- Cargo.lock`。
- テストはソースファイル内の `mod tests` に置く。
- **`src/agent.rs` は編集しない。** ルーム判定はチャンネル経路が既に張っている `timer::scope_timer_origin` を読むだけで足りる（Task 5 の `current_call_room`）。ここには #201 のデッドロック修正が載っている。
- **`src/tools/policy.rs` は編集しない。** 既存の表（`Edit` は `Origin::Channel` で `Allow`）の上に成立する。
- 既存ツールの挙動は1つも変えない。`subagent` の変更は**定義リストを差し替え可能にする**ことだけ。
- `[tools.admin]` が無い既存デプロイは、**ビルドも起動もテストも今までどおり**動くこと（`rooms` が空＝4本とも未登録）。
- 実装は Step ごとに「失敗するテスト → 実装 → 通す」の順で進め、各 Task 末尾でコミットする。

---

## File Structure

| ファイル | 責務 |
|---|---|
| `server/src/frontmatter.rs`（変更） | `set_enabled(raw, bool) -> Option<String>`。`enabled:` 行だけを差し替える行編集。 |
| `server/src/config.rs`（変更） | `AdminToolsConfig`、`ToolsConfig.admin`、`config_tools_allowed_in`、`config_tools_enabled`。 |
| `server/src/config_layer.rs`（変更） | `[tools.admin]` がワークスペース層から設定不可であることのテスト1件。 |
| `server/src/heartbeat_config.rs`（変更） | `pub fn parse_definition`（loader の寛容な `parse_task` の薄いラッパー）。 |
| `server/src/autonomous_config.rs`（変更） | 同上。 |
| `server/src/agents.rs`（変更） | 同上（`AgentDef` を返す）。 |
| `server/src/autonomous.rs`（変更） | `is_due` の doc に「テストは別 room_id」を1行。挙動は不変。 |
| `server/src/tools/subagent.rs`（変更） | 定義リストを `RwLock` 越しに持ち、`set_agents` / `live_spec` を追加。 |
| `server/src/tools/mod.rs`（変更） | `pub mod config_tools;` と `ToolSet::replace_spec`。 |
| `server/src/tools/config_tools.rs`（**新規**） | `ConfigDir` / `ConfigTool` / `TaskTestTool` / `register_admin_tools` / `current_call_room` / `ROOM_REFUSAL` / `definition_path`。 |
| `server/src/main.rs`（変更） | subagent を `Arc` 化し、`register_admin_tools` を1回呼ぶ。 |
| `server/config.example.toml`（変更） | `[tools.admin]` の説明と例。 |
| `server/templates/workspace/config.toml`（変更） | 「ここでは設定しない」旨の1行。 |
| `README.md`（変更） | 「Editing the agent's own definitions」節。 |

---

## Task 1: `frontmatter::set_enabled` — `enabled:` 行だけを書き換える

全文をパースしてシリアライズし直すと、人間が書いたコメントとフィールド順が消える。行単位で編集する関数を1つ足す。

**Files:** Modify `server/src/frontmatter.rs`（`set_enabled` と `mod tests`）

**Interfaces — Produces:** `pub fn set_enabled(raw: &str, enabled: bool) -> Option<String>`。frontmatter の**トップレベル** `enabled:` 行を `enabled: <bool>` に置換。無ければ閉じ `---` の直前に挿入。frontmatter ブロックが無ければ `None`。**他の行・コメント・body・改行コードは不変。** Task 5 が呼ぶ。

**Step 1: 失敗するテストを書く** — `mod tests` に5件:

```rust
#[test]
fn set_enabled_replaces_only_the_key_and_keeps_the_rest() {
    let raw = "---\n# fires the morning call\nschedule: \"0 8 * * *\"\nenabled: true\nroom_id: \"!ops:x\"\n---\n\n# Morning\nCall the room.\n";
    let out = set_enabled(raw, false).unwrap();
    assert!(out.contains("enabled: false"), "{out}");
    assert!(out.contains("# fires the morning call"), "{out}");
    assert!(out.contains("room_id: \"!ops:x\""), "{out}");
    assert!(out.ends_with("# Morning\nCall the room.\n"), "{out}");
    // Still parses as the same document, with one value changed.
    let (fm, _) = split(&out).unwrap();
    assert_eq!(parse_mapping(fm).get("enabled").and_then(|v| v.as_bool()), Some(false));
}

#[test]
fn set_enabled_inserts_the_key_when_it_is_absent() {
    let raw = "---\nschedule: \"0 8 * * *\"\n---\n\nBody\n";
    let out = set_enabled(raw, false).unwrap();
    let (fm, body) = split(&out).unwrap();
    assert_eq!(parse_mapping(fm).get("enabled").and_then(|v| v.as_bool()), Some(false));
    assert_eq!(body, "\nBody\n");
    // And a second call replaces rather than duplicating.
    let again = set_enabled(&out, true).unwrap();
    assert_eq!(again.matches("enabled:").count(), 1, "{again}");
}

/// `voice:` may carry a nested `enabled` some day; this must not rewrite it.
#[test]
fn set_enabled_leaves_an_indented_enabled_alone() {
    let raw = "---\nschedule: \"0 8 * * *\"\nvoice:\n  device_id: \"01J\"\n  enabled: true\n---\n\nBody\n";
    let out = set_enabled(raw, false).unwrap();
    assert!(out.contains("  enabled: true"), "{out}");
    assert_eq!(out.matches("enabled: false").count(), 1, "{out}");
}

#[test]
fn set_enabled_is_none_without_frontmatter() {
    assert!(set_enabled("# Just markdown\n", true).is_none());
    assert!(set_enabled("", true).is_none());
    assert!(set_enabled("---\nnever closed\n", true).is_none());
}

/// CRLF files exist; rewriting them as LF would be a whole-file diff.
#[test]
fn set_enabled_preserves_crlf() {
    let raw = "---\r\nschedule: \"0 8 * * *\"\r\n---\r\nBody\r\n";
    let out = set_enabled(raw, false).unwrap();
    assert!(out.ends_with("---\r\nBody\r\n"), "{out:?}");
    assert!(out.contains("enabled: false\r\n"), "{out:?}");
}
```

**Step 2:** `cargo test -p sapphire-agent frontmatter::tests::set_enabled -- --nocapture` → FAIL（`cannot find function set_enabled`）

**Step 3: 実装する**（`split` の直後）:

```rust
/// Set the top-level `enabled:` key in `raw`'s frontmatter, and touch
/// nothing else.
///
/// `None` when `raw` has no frontmatter block — a file that is not a
/// definition at all is the caller's problem to refuse, not this
/// function's to repair by inventing one.
///
/// Deliberately not `parse_mapping` + `serialize`: that round trip drops
/// every comment and reorders the keys, so a model asked to flip one
/// switch would rewrite a human's file. A definition is a file people
/// hand-edit — the `schedule` line usually has a comment above it saying
/// why — and the tool's job is one line, not the document.
pub fn set_enabled(raw: &str, enabled: bool) -> Option<String> {
    let (fm, _) = split(raw)?;
    // Which delimiter the file already uses. Rewriting a CRLF file as LF
    // turns a one-line change into a whole-file diff.
    let head = if raw.starts_with("---\r\n") { "---\r\n" } else { "---\n" };
    let nl = if head == "---\r\n" { "\r\n" } else { "\n" };
    let value = if enabled { "true" } else { "false" };

    let mut out = String::with_capacity(raw.len() + 16);
    out.push_str(head);
    let mut replaced = false;
    for line in fm.split_inclusive('\n') {
        // Top-level only: a leading space is a nested key, not this one.
        if !replaced && line.trim_end_matches(['\n', '\r']).starts_with("enabled:") {
            out.push_str("enabled: ");
            out.push_str(value);
            out.push_str(nl);
            replaced = true;
        } else {
            out.push_str(line);
        }
    }
    if !replaced {
        if !out.ends_with(nl) {
            out.push_str(nl);
        }
        out.push_str("enabled: ");
        out.push_str(value);
        out.push_str(nl);
    }
    // From the closing `---` line onwards, verbatim.
    out.push_str(&raw[head.len() + fm.len()..]);
    Some(out)
}
```

**Step 4:** `cargo test -p sapphire-agent frontmatter::` → PASS（既存の `split_*` を含む）

**Step 5: コミット** `feat(frontmatter): add a line-level set_enabled for definition files`

---

## Task 2: `[tools.admin]` を設定に足す

**Files:** Modify `server/src/config.rs`、`server/src/config_layer.rs`、`server/config.example.toml`

**Interfaces — Produces:** `pub struct AdminToolsConfig { pub rooms: Vec<String> }`（`Default` = 空）、`Config::config_tools_allowed_in(&self, Option<&str>) -> bool`、`Config::config_tools_enabled(&self) -> bool`。Task 5/6/7 が消費。

- `ToolsConfig` の直前に `AdminToolsConfig` を置き、`ToolsConfig` に `#[serde(default)] pub admin: AdminToolsConfig` を1本追加。doc に「空＝4本とも未登録。`[tools] host_access` と同じ形」と、`room_profile.<n>.rooms` と同じ部屋 id 名前空間であることを書く。
- `Config` の impl に:

```rust
    pub fn config_tools_allowed_in(&self, room_id: Option<&str>) -> bool {
        match room_id {
            Some(r) => self.tools.admin.rooms.iter().any(|allowed| allowed == r),
            None => false,
        }
    }
    pub fn config_tools_enabled(&self) -> bool {
        !self.tools.admin.rooms.is_empty()
    }
```

`None` が常に false である理由を doc に書く: `/rpc`・`/acp` はセッション id から `room_id` を合成するので、運用者が宣言した部屋ではない。

- `config_layer.rs` の `host_only_tables_are_not_allowed` に2行追加:
  `assert!(!path_allowed(&["tools", "admin"]));` と `assert!(!path_allowed(&["tools", "admin", "rooms"]));`
- `config.example.toml` の `[tools]` の説明ブロックに、コメントアウトした `[tools.admin]` 例・「ホスト層限定（`[tools]` はワークスペース層の allowlist 外）」・「ここで許可した部屋に投稿できる者は、エージェントを通じて無人実行の内容を書き換えられる。部屋側の `allowed_users` と併用せよ」を書く。

**テスト方針**（`config::tests` に2件）:

```rust
#[test]
fn config_tools_are_off_until_a_room_is_named() {
    let cfg = parse("[anthropic]\napi_key = \"test\"\n");
    assert!(!cfg.config_tools_enabled());
    assert!(!cfg.config_tools_allowed_in(Some("!ops:x")));
    assert!(!cfg.config_tools_allowed_in(None));
}

#[test]
fn config_tools_are_allowed_in_a_named_room_and_nowhere_else() {
    let cfg = parse("[anthropic]\napi_key = \"test\"\n\n[tools.admin]\nrooms = [\"!ops:x\", \"!dev:y\"]\n");
    assert!(cfg.config_tools_enabled());
    assert!(cfg.config_tools_allowed_in(Some("!ops:x")));
    assert!(!cfg.config_tools_allowed_in(Some("!random:z")));
    assert!(!cfg.config_tools_allowed_in(None));   // /rpc, /acp, voice: never
}
```

**受け入れ基準:** 既存の設定ファイル（`[tools.admin]` 無し）がそのままパースでき、既存テストが無変更で通ること。

Run: `cargo test -p sapphire-agent config::tests::config_tools config_layer::tests::host_only -- --nocapture` → PASS

**コミット** `feat(config): add [tools.admin].rooms for the admin tool surface`

---

## Task 3: ローダーのパーサを書き込み側から使えるようにする

`parse_task` / `parse_agent` は「壊れたファイルは警告して飛ばす」— **読み込み**の寛容さ。書き込み側は同じ規則で**拒否**したい。規則を二重に書かないために、既存関数の薄い `pub` ラッパーを1本ずつ出す。**既存関数の中身をコピーしないこと**が本 Task の目的。

**Files:** Modify `server/src/heartbeat_config.rs`、`server/src/autonomous_config.rs`、`server/src/agents.rs`、`server/src/autonomous.rs`

**Interfaces — Produces:**
- `pub fn heartbeat_config::parse_definition(name: &str, raw: &str) -> Result<HeartbeatTask, String>`
- `pub fn autonomous_config::parse_definition(name: &str, raw: &str) -> Result<AutonomousTask, String>`
- `pub fn agents::parse_definition(name: &str, raw: &str) -> Result<AgentDef, String>`

**実装（3ファイル同形。`agents` だけ `AgentDef` を返す）:**

```rust
/// Parse one definition the way the loader does, but hand the failure
/// back instead of skipping the file.
///
/// `load_heartbeat_dir` swallows a broken file on purpose — one typo
/// must not take the other tasks down with it — but a caller that is
/// *about to write* the file has the opposite need: it must refuse what
/// the loader would silently drop, or the model writes a task that never
/// fires and cannot tell why.
///
/// A task whose `schedule:` does not parse is *not* refused here: the
/// loader keeps it and `next_due` skips it, which is a different
/// failure. `ConfigTool::validate` adds that check where it matters.
pub fn parse_definition(name: &str, raw: &str) -> Result<HeartbeatTask, String> {
    // `split` only for the message: no frontmatter at all is the one case
    // worth naming precisely, since that is what a model gets wrong when
    // it writes a bare markdown file.
    crate::frontmatter::split(raw)
        .ok_or_else(|| "no YAML frontmatter: the file must start with a `---` line".to_string())?;
    parse_task(name.to_string(), raw).ok_or_else(|| {
        "cannot be parsed as a task: check the frontmatter YAML and that the body is not empty"
            .to_string()
    })
}
```

`agents::parse_definition` の理由文は ``"cannot be parsed as an agent definition: `description` is required and the frontmatter must be valid YAML"``。

**`autonomous.rs`** は `is_due` の doc（「The anchor is the task's *latest* session activity…」の段落末）に1行足すだけ。挙動は変えない:

```rust
/// A session whose `room_id` is `task.name` is the only thing counted,
/// which is also why a test run must **not** reuse that name: the
/// `task_test` tool creates its sessions under `test:<kind>:<name>`.
```

**テスト方針**（各ファイルの `mod tests` に1件ずつ。いずれも「loader が黙って飛ばす入力で `Err` が返り、正常入力で `Ok` が返る」ことを見る）:

```rust
// heartbeat_config
assert!(parse_definition("morning", "---\nschedule: \"0 8 * * *\"\n---\nHi\n").is_ok());
let err = parse_definition("broken", "# no frontmatter\n").unwrap_err();
assert!(err.contains("frontmatter"), "{err}");
assert!(parse_definition("broken", "---\nschedule: [oops\n---\nHi\n").is_err());

// autonomous_config
assert!(parse_definition("journal", "---\npriority: 50\n---\nWrite it.\n").is_ok());
assert!(parse_definition("journal", "---\npriority: 50\n---\n\n").is_err());   // empty body

// agents
assert!(parse_definition("reviewer", "---\ntools: []\n---\nReview.\n").is_err());  // no description
let def = parse_definition("reviewer", "---\ndescription: Reviews.\n---\nReview.\n").unwrap();
assert_eq!((def.name.as_str(), def.prompt.as_str()), ("reviewer", "Review.\n"));
```

Run: `cargo test -p sapphire-agent parse_definition -- --nocapture` → PASS

**コミット** `feat(config): expose the definition parsers for the write side`

---

## Task 4: `subagent` を差し替え可能にする（ホットリロード）

`SubagentTool` は起動時に読んだ `Vec<AgentDef>` を**所有**しているので、あとから書いた定義が反映されない。ロック越しに持ち、専用の入り口から差し替えられるようにする。

**Files:** Modify `server/src/tools/subagent.rs`、`server/src/tools/mod.rs`

**Key constraint — `Tool::spec` が `&ToolSpec` を返す:** 状態で変わる spec は、`spec()` がガードを返せない以上ロックの中に置けない。だから **spec の正本は `ToolSet` が持つ `inner.specs`** であり、差し替えは `replace_spec` でそこを更新する。`SubagentTool::spec()` は登録時の初期値を返し続ける（既存テストはそのまま通る）。

**Interfaces — Produces:** `SubagentTool::set_agents(&self, Vec<AgentDef>)`、`SubagentTool::live_spec(&self) -> ToolSpec`、`ToolSet::replace_spec(&self, name: &str, spec: ToolSpec)`（`async`）。**変わらないもの**: `spec()` の型、`kind()` = `Other`、`dispatch`/`resume` の権限・分離・`NotOffered` ゲート、`ResumeGuard`、タイムアウト。

**変更内容:**

1. フィールドを `agents: std::sync::RwLock<Vec<AgentDef>>` に変更（doc に「読者は必要なものを clone して guard を await の前に落とすので `std` の `RwLock` で足りる」と書く）。`new` は `spec: build_spec(&agents)` のまま。
2. `fn agents(&self) -> Vec<AgentDef>`（read guard は clone 後に即 drop）、`pub fn set_agents(&self, agents: Vec<AgentDef>)`、`pub fn live_spec(&self) -> ToolSpec { build_spec(&self.agents()) }`。
3. `dispatch` / `resume` の `self.agents.iter().find(...)` を `let agents = self.agents();` 経由に変更（`server/src/tools/subagent.rs:454` と `:533` 付近の2箇所。`dispatch` の `known` メッセージも同じローカルから）。doc は変更しない。
4. `build_spec` の `Available agents:` ループの直後に、空リスト時の1行を追加:
   `"(none yet — create one with `agent_config` action `write`)\n"`
5. `server/src/tools/mod.rs`: `register_tool` の直後に `pub async fn replace_spec`。`inner.specs.iter_mut().find(|s| s.name == name)` で置換、見つからなければ `warn!("replace_spec: no registered tool named '{name}'")` して**push しない**（spec だけあって `execute` が dispatch できない状態を作らない）。doc に「`Tool::spec` は borrow を返すので、状態で変わる spec は set 側に持つしかない」「`Tool::execute` は Arc を clone して read guard を落としてから実行するので、`Tool::execute` の中から呼んでも自分の write が詰まらない」と書く。

**テスト方針:**

```rust
// subagent::tests
#[tokio::test]
async fn set_agents_makes_a_new_definition_callable() {
    let tool = SubagentTool::new(Vec::new());
    assert!(tool.live_spec().description.contains("none yet"));
    tool.set_agents(vec![AgentDef { name: "reviewer".into(), description: "Reviews things.".into(),
        tools: None, prompt: "Review.".into(), profile: None }]);
    assert!(tool.live_spec().description.contains("- reviewer: Reviews things."));
    // The tool's own `spec()` is the registration-time value and stays put.
    assert!(!tool.spec().description.contains("reviewer"));
}

// tools::tests
#[tokio::test]
async fn replace_spec_swaps_what_the_model_is_offered() {
    let set = ToolSet::new(vec![Box::new(crate::tools::subagent::SubagentTool::new(Vec::new()))
        as Box<dyn Tool>], Vec::new());
    let mut swapped = set.specs_filtered(|_| true).await[0].clone();
    swapped.description = "swapped".into();
    set.replace_spec("subagent", swapped).await;
    let after = set.specs_filtered(|_| true).await;
    assert_eq!(after.len(), 1, "replace, not append");
    assert_eq!(after[0].description, "swapped");
}

#[tokio::test]
async fn replace_spec_does_not_add_an_unknown_name() {
    let set = ToolSet::new(Vec::new(), Vec::new());
    set.replace_spec("nope", crate::provider::ToolSpec { name: "nope".into(),
        description: "x".into(), input_schema: serde_json::json!({}) }).await;
    assert!(set.specs_filtered(|_| true).await.is_empty());
}
```

**`mod tests` の `every_tool_declares_its_kind` は変更しない**（`rooms` 空の `default_tool_set` は設定ツールを登録しないので期待表はそのまま）。

**受け入れ基準:** 既存の `subagent` テスト全件と `tools` の policy テストが無変更で通ること。

Run: `cargo test -p sapphire-agent subagent:: replace_spec -- --nocapture` → PASS

**コミット** `feat(subagent): make the definition list swappable at run time`

---

## Task 5: `config_tools.rs` — `ConfigTool`（list / read / write / set_enabled）

本題。1つの型を3ディレクトリ分登録する。ここでルーム許可ゲート、パス検証、`set_enabled`、書き込み前検証をすべて実装する。

**Files:** Create `server/src/tools/config_tools.rs`、Modify `server/src/tools/mod.rs`（`pub mod config_tools;`）

**Interfaces:**
- **Consumes:** `frontmatter::set_enabled`（Task 1）、`Config::config_tools_allowed_in`（Task 2）、`*::parse_definition`（Task 3）、`SubagentTool::set_agents` / `live_spec`、`ToolSet::replace_spec`（Task 4）
- **Produces:** `pub enum ConfigDir { Heartbeat, Autonomous, Agents }`（`ALL` = `[Agents, Autonomous, Heartbeat]`、`dir_name`、`tool_name`、`supports_enabled`）、`pub struct ConfigTool` + `new(...)`、`pub(crate) fn current_call_room() -> Option<String>`、`pub(crate) const ROOM_REFUSAL: &str`。Task 6/7 が `ConfigDir` / `current_call_room` / `ROOM_REFUSAL` / `definition_path` / `declared_enabled` を再利用。

**実装:**

1. **`ConfigDir`**: `supports_enabled()` は `!matches!(self, Self::Agents)`。`tool_name()` は `heartbeat_config` / `autonomous_config` / `agent_config`。
2. **`fn definition_path(workspace_root: &Path, dir: &str, name: &str) -> Result<PathBuf>`**: `name` の `.md` を剥がした stem が空・`.` 始まり・`/`・`\`・`..` を含むなら拒否。それ以外は `<root>/<dir>/<stem>.md`。doc に「`file_write` は絶対パスと `~` を受けるが、これは受けない。意図が読めることがこのツールの値打ちだから」。
3. **`pub(crate) fn current_call_room()`**: `crate::timer::current_origin()` が `TimerOrigin::Chat { room_id }` なら `Some(room_id)`、他は `None`。doc に「チャンネル経路（`Agent::handle_message`）が全ツール呼び出しに `TimerOrigin::Chat` を scope している。heartbeat のチャット脚も同じ経路なので対象ルームが入る。`/rpc`・`/acp` はセッション id から合成した room_id、voice は `TimerOrigin::Voice` なので `None` ＝ 拒否。allow-list は“誰が書けるか分かっている場所”を名指すものであって、それらはそうではない」。
4. **`ROOM_REFUSAL`**: `"Permission denied: the config tools are not available in this room. An operator can allow them with `[tools.admin].rooms` in the host config."`（モデルがそのまま運用者に伝えれば設定名に辿り着く文面にする）。
5. **`fn declared_enabled(raw) -> Option<bool>`**: `split` して `enabled:` 行を探し、無ければ `Some(true)`（両ローダーの `serde(default = "default_true")` と同じ**実効値**を返す）。frontmatter 無しは `None`。
6. **`ConfigTool`** のフィールド: `dir`、`workspace_root: PathBuf`、`config: Config`、`ws: Arc<Mutex<WorkspaceState>>`（`memory_add` と同じくワークスペースの writer を通す）、`subagent: Option<Weak<SubagentTool>>`（`Agents` のみ）、`tool_set: Weak<ToolSet>`（`Agents` のみ）、`spec: ToolSpec`。
7. **`impl Tool`**: `kind()` は `ToolKind::Edit`。`execute` は**最初に `self.gate()`**、次に `action` で分岐。未知の action は `actions_help()` を付けて拒否（`Agents` では「サブエージェント定義は呼ばれた時に走るので on/off 状態が無い。変更は `write`」と明示）。
8. **`gate()`**: `config_tools_allowed_in(current_call_room().as_deref())` が false なら `ROOM_REFUSAL`。**`list` を含む全 action を通す。**
9. **`list()`**: `std::fs::read_dir` で `*.md` を列挙（ローダーと同じ素の `std::fs` 経路なので本番の一覧と一致する）。各行に実効 `enabled` を `enabled` / `disabled` として添える（`Agents` は付けない）。0件なら `"No <dir> definitions."`。
10. **`read()`**: 素の `std::fs::read_to_string` で全文を返す（`serde` を通さず原本を返すのが要件）。
11. **`write()`**: `definition_path` でパス検証 → **`validate()` を先に**呼ぶ（拒否時にファイルを残さない）→ `ws.lock().write_file(rel, content)`（`workspace_search` の索引に載る）→ `after_write()`。
12. **`set_enabled()`**: `supports_enabled()` でなければ拒否 → ファイルを読む → `frontmatter::set_enabled` → `None` なら「frontmatter が無いので `enabled:` を設定できない」 → `ws` を通して書き戻す。
13. **`validate()`**:
    - `Heartbeat` → `parse_definition` を通したうえで **`parsed_schedule().is_none()` なら拒否**（ローダーは cron が読めないタスクを残して `next_due` が飛ばす。書き込み側には同じ失敗なので二重ではなく別チェック）。
    - `Autonomous` → `parse_definition` のみ。
    - `Agents` → `parse_definition` を通し、さらに `config.validate_subagent_profiles(slice::from_ref(&def))` が非空なら拒否（起動時と同じ検査。実行中に書けるようになった今、ここを飛ばすと起動時検証を回避する穴になる）。`tools:` の綴り間違いは**拒否しない**（`subagent` は警告のみ）。
    - いずれも `anyhow!("refusing to write {name}.md: {e}")`。
14. **`after_write()`**: `Agents` 以外は即 return。`subagent` と `tool_set` が upgrade できたら `subagent.set_agents(agents::load_agents_dir(&dir_path()))` → `tool_set.replace_spec(SUBAGENT_TOOL_NAME, subagent.live_spec())`。upgrade できなければファイルは書けたまま何もしない（次回起動で反映）。
15. **`new(...)`**: `action_names` を `if dir.supports_enabled() { vec!["list","read","write","set_enabled"] } else { vec!["list","read","write"] }` で作り、description と `input_schema` の `enum` の両方に使う（`json!` の中に `if` を直接書かない）。description には **`set_enabled` が `enabled:` 行だけを変えコメント・他のキー・body を触らない**ことを明示する（リスク: モデルが `write` でコメントを消す経路の緩和）。

**テスト方針**（`config_tools::tests`。`in_room(room, fut)` は `crate::timer::scope_timer_origin(TimerOrigin::Chat { room_id }, fut).await` で本番と同じ経路を通す。環境変数やテスト専用フックは**足さない**）:

| テスト | 主張 |
|---|---|
| `definition_path_refuses_anything_but_a_bare_stem` | `""`, `"."`, `".."`, `"../etc/passwd"`, `"a/b"`, `"a\\b"`, `".hidden"` を拒否。`"morning_call.md"` は正規化して受理 |
| `declared_enabled_defaults_to_true` | key 無し→`Some(true)`、`true`/`false`→その値、frontmatter 無し→`None` |
| `agent_config_has_no_set_enabled_action` | `agent_config` の `set_enabled` はエラーで、文面に `set_enabled` と `agent_config` が含まれる |
| `heartbeat_write_refuses_a_schedule_the_agent_cannot_read` | cron でない `schedule:` を拒否し、**ファイルが残らない** |
| `autonomous_write_refuses_an_empty_body` | body 空を拒否し、ファイルが残らない |
| `list_reports_names_and_effective_enabled` | `enabled: false` と key 無しの2件を列挙し、`disabled` を付ける。`read` は原本をそのまま返す |
| `set_enabled_keeps_the_rest_of_the_file` | `# keep me` コメントが残り、body が不変 |
| `set_enabled_refuses_a_file_without_frontmatter` | 修復せず拒否 |
| `every_action_is_refused_outside_an_allowed_room` | `list`/`read`/`write`/`set_enabled` すべて `Permission denied` |
| `the_admin_tools_are_edits` | `ConfigDir::ALL` の全部が `ToolKind::Edit` |

`config_tool_for_test(dir, workspace)` は 3 ディレクトリを作成し、`Config::for_test()`（`server/src/config.rs:1305`）に `rooms = ["!ops:x"]` を入れて `ConfigTool::new(...)` を組む薄いヘルパ。`WorkspaceState` は `server/src/tools/mod.rs` の `test_workspace()` と同じ手順で作る。

**受け入れ基準:** 上表が全部通り、`rooms` 空では `ConfigTool` が1本も登録されない状態（Task 7 で検証）と矛盾しないこと。

Run: `cargo test -p sapphire-agent config_tools -- --nocapture` → PASS

**コミット** `feat(tools): add the heartbeat/autonomous/agents config tools`

---

## Task 6: `task_test` — 有効にする前に1回走らせる

autonomous を有効化する前に「動くか」「トークンを食べすぎないか」「正常終了するか」を確かめる道具。**本番と同じ `run_llm_turn`、同じプロンプト組み立て、`test:<kind>:<name>` という別の room_id** が要点。

**Files:** Modify `server/src/tools/config_tools.rs`（`TaskTestTool` とテスト）

**Interfaces:** Consumes `current_call_room` / `ROOM_REFUSAL` / `definition_path`（Task 5）、`parse_definition`（Task 3）、`config_tools_allowed_in`（Task 2）、`autonomous::marker` / `autonomous::CONTINUE_PROMPT`、`serve::run_llm_turn` / `serve::AutonomousHost` / `TurnStop`。Produces `pub struct TaskTestTool` + `new(state: Arc<ServeState>)`。

**実装:**

1. `const MAX_TEST_TURNS: usize = 3;`（「トークンを食べないか確かめる道具がトークンを食べてはならない」）。
2. `fn test_room_id(kind, name) -> String { format!("test:{kind}:{name}") }`。doc に「`autonomous::is_due` は `room_id == task.name` の最新セッションをクールダウン起点にするので、テストがその名前を名乗ると本番のクールダウンをリセットしてしまう」。
3. `fn capped_test_turns(requested: Option<usize>) -> usize`: `Some(0) => 1`、`Some(n) => n.min(MAX_TEST_TURNS)`、`None => 0`（0 は「定義自身の `max_turns` を使う」の意）。
4. `fn describe_stop(&TurnStop) -> &'static str`: `Replied` / `ProviderError` / `BudgetExhausted` を1行に訳す（`Debug` は `BudgetExhausted { partial_text: "..." }` を出してしまい、運用者に読ませる文にならない）。
5. `kind()` は `ToolKind::Edit`。`execute` は**最初に** `config_tools_allowed_in(current_call_room().as_deref())` を確認し、false なら `ROOM_REFUSAL`。`kind` は `"heartbeat" | "autonomous"`、他は拒否。
6. `origin()` は `config.autonomous.origin` を `policy::Origin` に写す（本番と同じ行。テストが本番より緩い権限で成功しては意味がない）。
7. `namespace()` は呼び出しルームの namespace（`config.namespace_for_room`）、無ければ既定。テストのレポートが運用者の見える場所に落ちる。
8. `create_test_session()` は `autonomous_session_store.create_autonomous_session(&test_room_id(kind, name), &namespace)`。
9. `turn()` は `serve::run_llm_turn(state, session_id, ChatMessage::user(text), Arc::new(AutonomousHost { origin }), None)`。
10. `run_heartbeat()`: 定義を読んで `parse_definition` → `"[Heartbeat: {name}]\n\n{body}"`（`Heartbeat::fire_task` と同じ前置き）で**1ターンだけ**走らせ、`finish(..., 1, ...)`。
11. `run_autonomous()`: 定義を読んで cap を決める（`cap == 0` なら `task.max_turns.min(MAX_TEST_TURNS)`）。1ターン目は `"{marker(name)}\nSession: {session_rel}\n\n{body}"`、2ターン目以降は `"{marker(name)}\n{CONTINUE_PROMPT}"`（本番の `run_task` と同じ組み立て）。`text.trim() == "DONE"` か `text.is_none()` で break。`last` に直前の答えを残す（最終ターンが provider 失敗でも何を言ったか報告する）。
12. `session_rel()`: `task_test` のレポートに載せる `<workspace>` 相対パス。
13. `finish()`: セッションを `close_session`（開いたまま残すと「実行中」に見える）してから、`"{kind} task '{name}' ran {turns} turn(s) ({stop}).\nSession: {rel}\nResult: {answer}"` を返す。**`enabled:` にもクールダウンにも触れない**旨を description に書く。配信は検証しない（`room_id:` / `voice:` を無視）ことも書く。

**テスト方針:**

| テスト | 主張 |
|---|---|
| `a_test_session_does_not_claim_the_task_name` | `test_room_id("autonomous","journal") == "test:autonomous:journal"` かつ `!= "journal"` |
| `a_disabled_task_can_be_tested_and_the_session_is_closed` | `enabled: false` の定義が走り、戻り値に本文が含まれ、セッションが `test:autonomous:journal` に1件だけ・`is_closed` |
| `max_turns_is_capped_for_a_test` | `Some(50) → MAX_TEST_TURNS`、`Some(0) → 1`、`None → 0` |
| `a_heartbeat_test_runs_exactly_one_turn` | 応答を2つ用意しても `1 turn` と報告し、セッションは1件 |
| `task_test_is_refused_without_a_chat_room` | scope 無し（`TimerOrigin::Chat` が無い）では `Permission denied` |

`test_response(text)` ヘルパは `server/src/autonomous.rs` の `mod tests` の `response` と同じ形で置く。

**受け入れ基準:** 上の表が全通過。特に **基準8（テストが本番のクールダウンをリセットしない）** は `a_test_session_does_not_claim_the_task_name` と、`is_due` の既存テストで担保される。

Run: `cargo test -p sapphire-agent task_test -- --nocapture` → PASS

**コミット** `feat(tools): add task_test to try a task without enabling it`

---

## Task 7: `main.rs` に配線する

`rooms` が空なら1本も登録しない。非空なら4本登録し、`agent_config` に `subagent` と `ToolSet` の弱参照を渡す。

**Files:** Modify `server/src/main.rs`、Modify `server/src/tools/config_tools.rs`（`register_admin_tools` とテスト）

**実装:**

1. `main()` の中の配線はテストできないので、**登録判断と本体を `config_tools.rs` の1関数に切り出す**:

```rust
/// Register the four admin tools when the deployment has named a room.
///
/// Extracted from `main.rs` so the condition is testable: `rooms` empty
/// must register nothing at all — not four tools that always refuse.
pub async fn register_admin_tools(
    tool_set: &Arc<crate::tools::ToolSet>,
    workspace_root: &Path,
    config: Config,
    ws: std::sync::Arc<std::sync::Mutex<sapphire_framework::workspace::WorkspaceState>>,
    subagent: Option<&Arc<crate::tools::subagent::SubagentTool>>,
    serve_state: Arc<crate::serve::ServeState>,
) {
    if !config.config_tools_enabled() {
        return;
    }
    // `agent_config` refreshes the spec `ToolSet::specs` advertises, and
    // `ToolSet` owns those specs — so it holds a `Weak` to the very set
    // it is registered into. Same shape `RefreshSystemPromptTool` uses
    // for `Agent`, and the same reason: a strong ref would be a cycle.
    let tool_set_weak = Arc::downgrade(tool_set);
    for dir in ConfigDir::ALL {
        tool_set
            .register_tool(Box::new(ConfigTool::new(
                dir,
                workspace_root.to_path_buf(),
                config.clone(),
                std::sync::Arc::clone(&ws),
                if dir == ConfigDir::Agents { subagent.map(Arc::downgrade) } else { None },
                tool_set_weak.clone(),
            )))
            .await;
    }
    tool_set.register_tool(Box::new(TaskTestTool::new(serve_state))).await;
}
```

2. `main.rs` の subagent ブロック（`:519` 付近）を差し替え:

```rust
            let agent_defs = agents::load_agents_dir(&workspace_dir.join("agents"));
            // Taken before `new` moves the list: an empty list is not an
            // error, it is the state `agent_config` exists to change.
            let no_agent_defs = agent_defs.is_empty();
            let profile_errors = config.validate_subagent_profiles(&agent_defs);
            if !profile_errors.is_empty() {
                anyhow::bail!(
                    "invalid subagent profile references:\n  - {}",
                    profile_errors.join("\n  - ")
                );
            }
            // Registered when there is something to delegate to, and also
            // when `[tools.admin].rooms` is set: `agent_config` can create
            // a definition while the process is running. `Arc` rather than
            // a bare `Box` because `agent_config` holds a `Weak` to it —
            // same shape `SkillTool` uses.
            let subagent = Arc::new(tools::subagent::SubagentTool::new(agent_defs));
            if !no_agent_defs || config.config_tools_enabled() {
                tool_set.register_tool(Box::new(Arc::clone(&subagent))).await;
            }
```

3. session tools ブロック（`:895` 付近）の直後に登録ブロックを追加:

```rust
            // ── Config tools ────────────────────────────────────────────────
            // The agent's own heartbeat / autonomous / subagent definitions,
            // editable from a chat room an operator names in
            // `[tools.admin].rooms`. Registered here rather than in
            // `default_tool_set` because `task_test` needs `serve_state`,
            // which is built above.
            tools::config_tools::register_admin_tools(
                &tool_set, &workspace_dir, config.clone(), Arc::clone(&ws_state),
                Some(&subagent), Arc::clone(&serve_state),
            )
            .await;
```

> **実装者への注記（唯一の引っかかり）:** `agent_config` は `Weak<ToolSet>` を必要とするが、`ToolSet` は**自分自身のツールとして登録される**。循環は `Weak` で切るのが既存の流儀（`RefreshSystemPromptTool` は `Weak<Agent>`、`TimerManager` は `Weak<ServeState>`）。`register_tool` が受け取るのが `Box<dyn Tool>` で `Arc<Tool>` を包めない場合は、既存の `SkillTool` の渡し方（`main.rs:544` 付近）を**先に確認してから**この Step を実装する。`Arc<SubagentTool>` を登録できないなら、`register_admin_tools` には `subagent: None` を渡し、`agent_config` の `after_write` の早期 return に「このデプロイでは差し替えられない。次回起動で反映される」旨の `info!` を1行足す。

**テスト方針**（`config_tools::tests` に2件。Task 5 の helper を再利用）:

```rust
#[tokio::test]
async fn no_room_means_no_registration() { /* rooms 空 → specs_filtered が空 */ }

#[tokio::test]
async fn one_room_registers_all_four() {
    // rooms = ["!ops:x"] → 名前4件が {"agent_config","autonomous_config",
    //                                   "heartbeat_config","task_test"} で、
    // set.kinds().await.len() == 4 （重複なし）
}
```

**受け入れ基準:** `rooms` 空で `specs_filtered` が空（＝モデルは存在を知らない）、非空でちょうど4本。既存ツールの一覧が変わらない。

Run: `cargo test -p sapphire-agent config_tools -- --nocapture` → PASS
Run: `cargo test -p sapphire-agent` → PASS（全件。`main.rs` はテストから叩けないのでビルド確認の意味が大きい）

**コミット** `feat(main): register the admin tools when a room is allow-listed`

---

## Task 8: ワークスペーステンプレートと README

**Files:** Modify `server/templates/workspace/config.toml`、`README.md`

1. `templates/workspace/config.toml` の `[autonomous]` の後ろに、`[tools.admin]` はここでは設定しない旨（「誰が運用者かは API キーや bind アドレスと同じホストの決定。ここに書くと起動時に警告付きで落とされる」）を1行足す。
2. `README.md` の `## Skills`（`:763`）の**前**に「## Editing the agent's own definitions」節を足す。内容: 4ツールと action の一覧（`agent_config` に `set_enabled` は無い）、`set_enabled` は `enabled:` 行だけを変えること、`[tools.admin] rooms` の例、空なら未登録であること、ホスト層限定（`[tools]` はワークスペース層 allowlist 外）で部屋の `allowed_users` と併用すべきこと、`task_test` は有効化前に1回試すためのもので3ターン上限・クールダウン不変・配信は検証しないこと、サブエージェントには `enabled` が無く（呼べば走る）、`agent_config` で書いた定義は再起動なしで使えること。

**テスト方針:** `cargo test -p sapphire-agent config -- --nocapture`（`config.example.toml` のパースと `templates/workspace/config.toml` の allowlist テスト）→ PASS。`cargo test -p sapphire-agent` → PASS。`cargo clippy --workspace -- -D warnings` → 警告ゼロ。

**コミット** `docs: document the admin tools and why they are room-scoped`

---

## Task 9: 最終検証

- **Step 1:** `cargo test --workspace` → PASS
- **Step 2: 受け入れ基準の通し確認**

| # | 基準 | 確認方法 |
|---|---|---|
| 1 | `rooms` 無しでは4本とも未登録 | `no_room_means_no_registration`、`every_tool_declares_its_kind` が無変更で通る |
| 2 | `rooms` 非空で4本登録・許可ルームのみ通る | `one_room_registers_all_four`、`every_action_is_refused_outside_an_allowed_room`、`task_test_is_refused_without_a_chat_room` |
| 3 | `list` が名前と enabled、`read` が全文 | `list_reports_names_and_effective_enabled` |
| 4 | `set_enabled` が行だけを書く | `frontmatter::tests::set_enabled_*`、`set_enabled_keeps_the_rest_of_the_file` |
| 5 | `false` の heartbeat が発火せず、`true` で再起動なしに発火 | `heartbeat_config` の既存テスト（`run_cron` が毎ループ読む）＋ 手動: 起動 → `set_enabled true` → 次の cron で発火 |
| 6 | `set_enabled false` の autonomous が選ばれない | `autonomous` の「enabled でないタスクは選ばれない」既存テスト |
| 7 | `enabled: false` のままテストでき、セッションが読める | `a_disabled_task_can_be_tested_and_the_session_is_closed` |
| 8 | テストが本番のクールダウンをリセットしない | `a_test_session_does_not_claim_the_task_name` |
| 9 | 壊れた定義を書かずに拒否、正常な定義は再起動なしで使える | `heartbeat_write_refuses_a_schedule_the_agent_cannot_read`、`autonomous_write_refuses_an_empty_body`、`agents::tests::parse_definition_reports_a_missing_description`、`set_agents_makes_a_new_definition_callable`、`replace_spec_swaps_what_the_model_is_offered` |
| 10 | `agent_config set_enabled` はエラー、`enabled` フラグは増えない | `agent_config_has_no_set_enabled_action` |
| 11 | `write` が壊れた内容を拒否 | 上記2件 ＋ `set_enabled_refuses_a_file_without_frontmatter` |
| 12 | `name` のパス脱出を拒否 | `definition_path_refuses_anything_but_a_bare_stem` |
| 13 | 既存の挙動が変わらない | `cargo test --workspace` 全件 ＋ `[tools.admin]` の無い設定で起動確認 |

- **Step 3: 手動の通し確認**（`config.example.toml` を写した一時ワークスペースで1回だけ、`sapphire-agent verify` を使う）:
  1. `rooms` をコメントアウトしたまま → 4ツールがプロンプトに出てこない。
  2. `rooms` にルームを入れて起動 → そのルームから `heartbeat_config action=list` が通る。
  3. 同じ設定で `/rpc` から呼ぶ → `Permission denied: the config tools are not available in this room.` が返る。
  4. `set_enabled` で人間のコメント付きタスクを `false` にし、ファイルを開いてコメントが残っていることを確認する。

- **Step 4:** `git checkout -- Cargo.lock && git status --short`。`Cargo.lock` 以外に差分が無ければこの Task はコミットを作らない。
