# システムプロンプトのピン留め 実装計画

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or "executing-plans" (LLM-friendly) to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** ワークスペースファイル（SOUL.md 等）の編集がシステムプロンプトに即反映されないよう、読み込みを「デイリーログ生成後」と `refresh_system_prompt` ツール呼び出しの2瞬間に限定する（プロンプトキャッシュ保護）。

**Architecture:** `Workspace::read_file` の mtime 即反映キャッシュを「ピン留め」方式（一度読むとクリアするまで不変）に置き換える。これが全パス（チャネル/ACP）共通の根本対策。チャネルパスの `SystemSnapshot`（日境界リフレッシュ）は維持し、`invalidate_system_prompts()` を「スナップショット全消去＋Workspace ピン留めキャッシュ全クリア」に拡張する。ACP パスはピン留めにより追加実装不要。

**Tech Stack:** Rust, tokio, axum (serve), tokio::sync::Mutex, 既存 Tool trait / ToolSet。

**Spec:** `docs/superpowers/specs/2026-09-07-system-prompt-pinning-design.md`

## Global Constraints

- 既存テスト `the_system_prompt_is_byte_identical_across_turns`（`server/src/workspace.rs` の tests）は変更しない・パスし続けること。
- 連結順・見出し・切り詰め `MAX_FILE_CHARS` などプロンプト生成ロジックは変更しない。
- 日境界リフレッシュの既存挙動（`SystemSnapshot` の「日付が変わったら再構築」）は変更しない。
- ACP パス（`serve/mod.rs::run_llm_turn`）へのスナップショット機構は新設しない。
- 各タスク末に `cargo test -p sapphire-agent-server`（該当テスト）と `cargo clippy --all-targets` がパスすること。fmt 済みであること。

## 設計判断（仕様書に補足する決定事項）

1. **ピン留めキャッシュの実装形**: `Workspace.cache` を `Mutex<HashMap<PathBuf, CachedFile>>`（`CachedFile{content,mtime}`）から、値を `Option<String>` とした **ピン留めマップ**へ単純化する（存在しないファイルも `Some`/`None` として保持＝欠落もピン留め）。mtime は廃止。クリアは `HashMap::clear()` の一括差し替えでアトミック。
2. **デイリーログ/ダイジェスト読み込みの一本化**: `periodic_log::read_body` / `read_digest_top_n` の直接 `std::fs::read_to_string` は、`Workspace` のピン留めキャッシュ**同一マップ**経由に置き換える（別マップを作らない）。`inject_periodic_logs` は async 化し、ピン留め読み込み経由でファイル本文/frontmatter を取得する。frontmatter の digest 抽出処理自体は現状のロジックを維持。
3. **ツールの接続先**: 無効化は `Agent::invalidate_system_prompts()`（スナップショット全消去＋Workspace キャッシュクリアの両方）に一本化し、`refresh_system_prompt` ツールはその関数を呼ぶだけ。ツールは `Weak<Agent>` を保持する（`Agent` は `Arc<ToolSet>` を持つため循環回避に Weak。`subagent` ツールと同様の登録パターンで、Agent 構築後に `register_tool` する）。なお `Agent` は Matrix/Discord チャネル設定時にのみ構築される（main.rs:979 条件）ため、**チャネルなし構成では本ツールは登録しない**（ACP パスは層1のピン留めで既に安定しておりスナップショット自体が存在しないため問題ない）。
4. **ツールの ToolKind**: デフォルト（`Other`＝最厳、ACP で許可確認対象）のままにする。

---

## Task 1: Workspace のピン留めキャッシュ

**Files:**
- Modify: `server/src/workspace.rs`（`CachedFile` 構造、`read_file`、`read_first_existing`、`inject_periodic_logs`、`build_chained_digest_block`、`build_memory_block`、tests）
- Modify: `server/src/periodic_log.rs`（`read_body` / `read_digest_top_n` をピン留めキャッシュ経由にするための読み出し口変更。Workspace 側にピン読み出し用メソッドを追加する場合はそちら中心に実装）
- Test: `server/src/workspace.rs` の `mod tests`

**Interfaces:**
- Produces: `Workspace::clear_pinned_cache(&self)`（`pub async fn`、ピン留めマップを全消去）。Task 2 が呼ぶ。
- Produces: `Workspace::read_file` は「初回読込→ピン留め、クリアまで不変、欠落も `None` としてピン留め」の挙動。

- [ ] **Step 1: 失敗するテストを書く**（`server/src/workspace.rs` の `mod tests` に追加）

```rust
/// ピン留めの本体性質：編集してもクリアするまでバイト列は不変。
#[tokio::test]
async fn the_system_prompt_pins_edits_until_cleared() {
    let dir = tempfile::TempDir::new().unwrap();
    let ws = workspace_with_agents_md(&dir);
    let chain = ["default".to_string()];

    let before = ws.build_system_prompt(Some("base"), 4, &chain, None).await;

    // 編集する（mtime が変わる）
    std::fs::write(
        dir.path().join("AGENTS.md"),
        "# how this works\n\nedited content.\n",
    )
    .unwrap();
    tokio::time::sleep(std::time::Duration::from_millis(5)).await;

    let pinned = ws.build_system_prompt(Some("base"), 4, &chain, None).await;
    assert_eq!(before, pinned, "ピン留め中は編集が反映されない");
    assert!(!pinned.contains("edited content."));

    ws.clear_pinned_cache().await;
    let refreshed = ws.build_system_prompt(Some("base"), 4, &chain, None).await;
    assert!(refreshed.contains("edited content."), "クリア後は反映される");
}

/// ピン留め中に新規作成されたファイルは、クリアするまで見えない。
#[tokio::test]
async fn newly_created_files_are_invisible_until_cleared() {
    let dir = tempfile::TempDir::new().unwrap();
    let ws = Workspace::new(dir.path().to_path_buf(), DigestConfig::default());
    let chain = ["default".to_string()];

    let before = ws.build_system_prompt(Some("base"), 4, &chain, None).await;
    std::fs::write(dir.path().join("SOUL.md"), "# Soul\n\nnew soul.\n").unwrap();

    let pinned = ws.build_system_prompt(Some("base"), 4, &chain, None).await;
    assert_eq!(before, pinned, "新規ファイルはクリアまで見えない");

    ws.clear_pinned_cache().await;
    let refreshed = ws.build_system_prompt(Some("base"), 4, &chain, None).await;
    assert!(refreshed.contains("new soul."));
}
```

- [ ] **Step 2: テスト実行して失敗を確認**

Run: `cargo test -p sapphire-agent-server the_system_prompt_pins_edits_until_cleared newly_created_files_are_invisible_until_cleared -- --nocapture`
Expected: FAIL（`clear_pinned_cache` が無い＝コンパイル失敗、または mtime 即反映で編集が即反映される）

- [ ] **Step 3: 実装**（`server/src/workspace.rs`）

`CachedFile` を廃し、`Workspace.cache: Mutex<HashMap<PathBuf, Option<String>>>` 相当のピン留めマップへ変更。`read_file` は以下の人（mtime 廃止、初回のみ disk 読込、欠落も `None` としてピン留め）:

```rust
/// Read a workspace file from the pinned cache. Once a path has been read
/// it is pinned until `clear_pinned_cache` — edits on disk are NOT reflected
/// until then. A missing file is also pinned (as `None`) so a file created
/// later is invisible until the next clear. Pinned because the system prompt
/// is rebuilt per turn and a byte change busts the provider prompt cache.
async fn read_file(&self, filename: &str) -> Option<String> {
    let path = self.dir.join(filename);
    let mut cache = self.cache.lock().await;
    if let Some(entry) = cache.get(&path) {
        return entry.clone();
    }
    let content = std::fs::read_to_string(&path)
        .ok()
        .map(|raw| truncate_chars(&raw, MAX_FILE_CHARS));
    cache.insert(path, content.clone());
    content
}

/// Drop every pinned file so the next read re-reads from disk. Called by
/// `Agent::invalidate_system_prompts` after the daily log is regenerated or
/// when the agent calls the `refresh_system_prompt` tool.
pub async fn clear_pinned_cache(&self) {
    self.cache.lock().await.clear();
}
```

`inject_periodic_logs`（`&self` → `&self` のまま async 化）と `build_chained_digest_block` は、`periodic_log::read_body` / `read_digest_top_n`（`std::fs::read_to_string` 直呼び）へ直接頼らず、`Workspace` のピン留め読み込み（`read_file` と同一マップ）経由で本文・frontmatter を取得するよう変更する。ファイル相対パスは既存の `periodic_log::log_abs_path` 相当を使い、読み出しだけ `self` のピンマップへ通す。digest の frontmatter 抽出ロジック自体は維持。`file_mtime` ヘルパは不要になるなら削除。

- [ ] **Step 4: テストパス確認**

Run: `cargo test -p sapphire-agent-server --lib workspace`  （既存 `the_system_prompt_is_byte_identical_across_turns` も含む全 workspace テスト）
Expected: PASS

- [ ] **Step 5: clippy/fmt してコミット**

```bash
cargo clippy -p sapphire-agent-server --all-targets -- -D warnings
cargo fmt -p sapphire-agent-server
git add server/src/workspace.rs server/src/periodic_log.rs
git commit -m "feat(workspace): pin system-prompt file reads until an explicit clear

mtime-based immediate reflection let any edit bust the provider prompt
cache on the next turn. Reads are now pinned per path (missing files
pinned as absent) until clear_pinned_cache is called."
```

---

## Task 2: `invalidate_system_prompts` をピン留めクリア付きに拡張

**Files:**
- Modify: `server/src/agent.rs`（`invalidate_system_prompts`、doc comment）
- Test: `server/src/agent.rs` の `mod tests`（既存 `Agent::new` テストヘルパがある tests を流用）

**Interfaces:**
- Consumes: Task 1 の `Workspace::clear_pinned_cache()`。
- Produces: `Agent::invalidate_system_prompts()` — スナップショット全消去 **＋** Workspace ピン留めキャッシュ全クリアの両方を行う（既存呼び出し元 heartbeat.rs は変更なし）。

- [ ] **Step 1: 実装**（`server/src/agent.rs` の `invalidate_system_prompts` を置換）

```rust
/// Drop all cached system-prompt snapshots AND the workspace's pinned file
/// cache, so the next build re-reads every file from disk. Called after a
/// daily/weekly/monthly/yearly log is regenerated (so the fresh log shows
/// up) and by the `refresh_system_prompt` tool. Pinned reads mean the
/// prompt is otherwise byte-stable; this is one of the two moments it
/// legitimately changes (the other being the day-boundary snapshot rebuild,
/// which re-reads only after a clear).
pub async fn invalidate_system_prompts(&self) {
    self.workspace.clear_pinned_cache().await;
    self.snapshots.lock().await.clear();
}
```

- [ ] **Step 2: 統合テスト（可能なら）を書く／既存テストを確認**

`Agent` を直接構築して検証するのが重い場合、Task 1 の workspace テストが本タスクの仕様（ピン留め→クリア→反映）を既に担保しているため、agent 側は「`invalidate_system_prompts` が両方を消す」ことを unit で確認できる最小テストを1つ追加（ Workspace を一時ディレクトリで構築 → ファイル配置 → build → 編集 → `invalidate_system_prompts` → build で反映確認）。既存 agent tests の workspace 構築ヘルパ（`Agent::new` 呼び出し付近）を流用。

- [ ] **Step 3: テスト・clippy・fmt・コミット**

```bash
cargo test -p sapphire-agent-server --lib agent
cargo clippy -p sapphire-agent-server --all-targets -- -D warnings
cargo fmt -p sapphire-agent-server
git add server/src/agent.rs
git commit -m "feat(agent): invalidate_system_prompts also clears the pinned file cache"
```

---

## Task 3: `refresh_system_prompt` ツール追加

**Files:**
- Modify: `server/src/tools/builtin_tools.rs`（`RefreshSystemPromptTool` 追加）
- Modify: `server/src/main.rs`（Agent 構築後に `register_tool` で登録。subagent ツール登録と同一箇所付近）
- Modify: `templates/workspace/AGENTS.md`（メモリ編集の即時反映されない旨を1行追記）
- Test: `server/src/tools/` の該当テストモジュール（既存 tool 登録テストに追記可）

**Interfaces:**
- Consumes: Task 2 の `Agent::invalidate_system_prompts()`。ツールの `execute` はこれを呼ぶだけ。
- Produces: `RefreshSystemPromptTool::new(std::sync::Weak<Agent>) -> Self`。ツール名 `refresh_system_prompt`、引数なし、`ToolKind` はデフォルト（Other）。

- [ ] **Step 1: 失敗するテストを書く**（ツール一覧表示とクリア挙動）

`builtin_tools.rs` の既存テストスタイルに倣い、（1）`ToolSet` に登録した `refresh_system_prompt` が `specs_filtered` で名前が見えること、（2）`execute` が成功テキストを返すこと、を最低1テストで確認する。`Agent` 依存がテストで重い場合は、`execute` 内部が `Weak::upgrade().invalidate_system_prompts()` を呼ぶだけの薄い実装であることを、upgrade 失敗時エラー文案のテストで担保する。

- [ ] **Step 2: 実装**（`builtin_tools.rs`）

```rust
/// Tool the agent calls to explicitly re-read workspace files into the
/// system prompt. System-prompt file reads are pinned (see
/// `Workspace::read_file`); this drops the pinned cache AND every channel
/// snapshot so the next turn rebuilds from the current files — one of the
/// two moments the prompt legitimately changes (the other is the
/// day-boundary snapshot rebuild). Delegates to `Agent::invalidate_system_prompts`.
pub struct RefreshSystemPromptTool {
    agent: std::sync::Weak<crate::agent::Agent>,
    spec: ToolSpec,
}

impl RefreshSystemPromptTool {
    pub fn new(agent: std::sync::Weak<crate::agent::Agent>) -> Self {
        Self {
            agent,
            spec: ToolSpec {
                name: "refresh_system_prompt".to_string().into(),
                description: "システムプロンプトの読み込みキャッシュをクリアし、\
ワークスペースファイルを再読み込みさせます。メモリ等を編集して即時反映させたい時に呼んでください。"
                    .to_string()
                    .into(),
                input_schema: serde_json::json!({"type": "object", "properties": {}}),
            },
        }
    }
}

#[async_trait]
impl Tool for RefreshSystemPromptTool {
    fn spec(&self) -> &ToolSpec {
        &self.spec
    }
    async fn execute(&self, _input: &serde_json::Value) -> Result<String> {
        let agent = self
            .agent
            .upgrade()
            .ok_or_else(|| anyhow::anyhow!("agent unavailable"))?;
        agent.invalidate_system_prompts().await;
        Ok("システムプロンプトの読み込みキャッシュをクリアしました".to_string())
    }
}
```

- [ ] **Step 3: 登録**（`main.rs`、Agent 構築直後・subagent ツール登録と同パターン）

```rust
tool_set
    .register_tool(Box::new(tools::builtin_tools::RefreshSystemPromptTool::new(
        Arc::downgrade(&agent),
    )))
    .await;
```

登録は Agent が構築されるチャネル設定時のみ行う（ACP のみ構成では Agent が無くスナップショットも存在しないため不要）。`visible_tool_predicate` の除外リストには加えない（全クライアントに提供する）。

- [ ] **Step 4: テスト・clippy・fmt・コミット**

```bash
cargo test -p sapphire-agent-server --lib tools
cargo clippy -p sapphire-agent-server --all-targets -- -D warnings
cargo fmt -p sapphire-agent-server
git add server/src/tools/builtin_tools.rs server/src/main.rs templates/workspace/AGENTS.md
git commit -m "feat(tools): add refresh_system_prompt tool + document pinned prompt"
```

---

## Task 4: ドキュメント（テンプレート AGENTS.md）

※ Task 3 Step 3 で `templates/workspace/AGENTS.md` へ1行追記する。単独タスクとして立てるほどではないが、追記文面は次で固定:

> メモリファイル（MEMORY.md / 日次ログ）の編集は即座にはシステムプロンプトへ反映されません。日次ログ生成時（day boundary）または `refresh_system_prompt` ツール呼び出し時に反映されます。即時反映したい場合は `refresh_system_prompt` を呼んでください。

（Task 3 に統合済み。別コミット不要。）
