# サブエージェント再帰起動（段数制限＋定義ごと許可リスト）実装計画書

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** サブエージェントの再帰的起動を可能にする。ネスト段数の上限を `[tools.subagent] max_depth`（既定 2）で設定可能にし、定義ファイルの `subagents:` 許可リストで「どのサブエージェントを委派先にできるか」を定義ごとに制限できるようにする。

**Architecture:** 現状 `subagent_tool_specs()` がネスト先のツールリストから `subagent` 自身を**常に除去**しているのを、「委派元の段数 `depth` に対し `depth < max_depth` なら除去せず、ネスト先用の `subagent` ツール仕様を再構築して含める」に置き換える。段数は `TurnContext` の新フィールド `subagent_depth` で伝え、委派元 0・ネスト先 +1。ネスト先に埋め込むエージェント一覧は委派元仕様から導出し、定義の `subagents:` があれば名前で絞る。実体を抑えている NotOffered ゲート（`TurnLoop::run`）は membership チェックだけなので無修正で通る。

**Tech Stack:** Rust（tokio, serde, serde_yaml）、既存の `subagent.rs` / `serve/mod.rs` / `agents.rs` / `config.rs` の既存機構。

**Spec:** `docs/superpowers/specs/2026-09-15-subagent-recursion-design.md`

## Global Constraints

- コード・コメント・コミットメッセージは英語（CONTRIBUTING.md 規約）。新規コメントは英語。
- 型・名前は既存の命名規則に従う（`max_depth` / `subagents` / `subagent_depth`）。
- テストは既存の様式に完全準拠: `ScriptedProvider` のスクリプト付き `ServeState::for_test_scripted_*` + `ChatLog` の `chat()` 呼び出し数検証、`subagent.rs` 側の純関数テスト。
- 既定値は `max_depth = 2`。`0` は委派自体を禁止（メインエージェントのターンからも `subagent` が見えない）。従来の「ネスト禁止」は `max_depth = 1`。
- 段数 `d` で動くエージェントは `d < max_depth` のときのみ委派できる（メインエージェントは `d = 0`）。
- `subagents:` の未記載/空リスト/リストのセマンティクスは既存 `tools:` と同型（未記載 = 委派元から見えている集合をそのまま継承、`[]` = 段数と無関係に委派不可、リスト = その名前に絞る）。
- リスト内の未知名は起動時に落ちず dispatch 時に `warn!` して無視（既存 `newly_unknown_tools` と同じ方針・同じ重複排除）。

## 設計の確定事項（タスク間で共有する前提）

1. **段数の伝達**: `TurnContext`（`server/src/serve/mod.rs`、約2293行）に `pub subagent_depth: u32` を追加。メインターン（`run_llm_turn` の `TurnLoop` 構築、約2936行）は `0`。`SubagentTool::run_and_store`（`subagent.rs` 約741行）のネスト `TurnLoop` 構築で `ctx.subagent_depth + 1`。テストヘルパー `turn_context()`（`subagent.rs` 約1461行）と `a_resumed_child_continues_its_own_history` の手製 `TurnContext`（約1509行）も同フィールド追加。

2. **`subagent_tool_specs` の新シグネチャ**:
   ```rust
   pub(crate) fn subagent_tool_specs(
       def: &AgentDef,
       parent_visible: &[ToolSpec],
       all_agents: &[AgentDef],
       depth: u32,
       max_depth: u32,
   ) -> Vec<ToolSpec>
   ```
   挙動:
   - 従来どおり `parent_visible` から `subagent` を除外し、`def.tools` で絞ったものが基底リスト。
   - `depth < max_depth` かつ `def.subagents != Some([])` なら、ネスト先用の `subagent` ツール仕様を末尾に含める。埋め込む一覧は:
     - `def.subagents == Some(list)` → `all_agents` から名前で絞った集合で `build_spec` を再構築
     - `def.subagents == None` → 委派元ターンに見えているエージェント集合を**そのまま継承**。実装上は `parent_visible` 中の `subagent`仕様（`name == SUBAGENT_TOOL_NAME`）の `description` に埋め込まれた一覧をそのまま使う —委派元仕様の spec クローンが最も忠実な継承で、パースは不要。委派元に `subagent` spec が無い場合（親自身が深さ上限で抑止されている場合）は、そもそも `parent_visible` に `subagent` が無く再構築元が無い。このケースは `depth < max_depth` と同時に起こり**得ない**（メインターンは `subagent` を常に持つ。起こるとすれば `subagents: []` の親の子で、それは上側の条件で既に除外される）ので分岐不要。
   - `depth >= max_depth` または `Some([])` → `subagent` はリストに含めない（現状と同様の除去）。

3. **`AgentDef` / `AgentMeta`**（`server/src/agents.rs`）に `subagents: Option<Vec<String>>` を追加（`tools` と対称。欠落 = `None`）。

4. **`SubagentConfig`**（`server/src/config.rs` 約835行）に `max_depth: u32`（`#[serde(default = "SubagentConfig::default_max_depth")]`、既定 2）。`Default` impl にも追加。読み出しは `run_and_store` から既存の `ctx.state.config.tools.subagent.turn_timeout()` と同じ経路（`ctx.state.config.tools.subagent.max_depth`）。

5. **既存テスト `a_subagent_cannot_invoke_subagent_by_name`**（`serve/mod.rs` 約4020行、`s-subagent-no-recursion`）は現状「ネストは常に拒否」を検証している。`max_depth` 既定値 2 では拒否されなくなるので、**`max_depth = 1` に明示設定した上で従来動作（4コールで終了）を検証するテストに書き換え**、既定値 2 でネストが実際に動く対テストを新規追加する。

---

## Task 1: `SubagentConfig.max_depth`（設定）

**Files:**
- Modify: `server/src/config.rs`（`SubagentConfig`、約835行〜。テストは既存の `a_zero_subagent_turn_timeout_means_no_deadline` の隣に追加）
- Modify: `server/config.example.toml`（`[tools.subagent]` 例、約423行付近）

**Interfaces:**
- Produces: `SubagentConfig { turn_timeout_secs, max_depth }`、`SubagentConfig::default_max_depth() -> u32`（値 2）。Task 3 が `ctx.state.config.tools.subagent.max_depth` を読む。

- [ ] Step 1: 失敗するテストを書く（`server/src/config.rs` の tests、既存 3427行付近の timeouts テスト群の隣）:

```rust
/// `max_depth` の既定は 2。main(0) → plan(1) → explorer(2) まで。
#[test]
fn subagent_max_depth_defaults_to_two() {
    let tools: crate::config::ToolsConfig =
        toml::from_str("[tools]\n[subagent]").unwrap();
    assert_eq!(tools.subagent.max_depth, 2);
}

/// 書かれた値がそのまま読める。`0`（委派禁止）も含む。
#[test]
fn subagent_max_depth_reads_from_toml() {
    let tools: crate::config::ToolsConfig =
        toml::from_str("[subagent]\nmax_depth = 0").unwrap();
    assert_eq!(tools.subagent.max_depth, 0);
}
```

  注: 既存テスト `a_zero_subagent_turn_timeout_means_no_deadline` は `[subagent]\nturn_timeout_secs = 0` に倣い、`ToolsConfig` へ `[subagent]` をトップレベルテーブルとして直接パースする形に統一した（`[tools]` ラッパは不要）。

- [ ] Step 2: `cargo test -p sapphire-agent-server --lib config::tests::subagent` で失敗（コンパイル不可を含む）を確認
- [ ] Step 3: `SubagentConfig` に実装:

```rust
    /// ネスト段数の上限。main(0) → plan(1) → explorer(2) までなら 2。
    /// `0` は委派そのものを禁止（メインエージェントのターンからも
    /// `subagent` ツールが見えない）。従来の「ネスト禁止」は 1。
    #[serde(default = "SubagentConfig::default_max_depth")]
    pub max_depth: u32,
```

```rust
    fn default_max_depth() -> u32 {
        2
    }
```

  `Default for SubagentConfig` にも `max_depth: Self::default_max_depth(),` を追加。
- [ ] Step 4: `cargo test -p sapphire-agent-server --lib config::tests` 全パス確認
- [ ] Step 5: `config.example.toml` の `[tools.subagent]` 例に追記:

```toml
# [tools.subagent]
# turn_timeout_secs = 3600                  # 0 = no limit
# max_depth = 2                             # ネスト段数の上限。0 = 委派不可、1 = 従来のネスト禁止
```

  （例示コメントは既存ファイルの英語コメント様式に合わせて英語で書く。既存例は英語）
- [ ] Step 6: `git add` & `git commit -m "feat(config): add [tools.subagent] max_depth (nesting depth cap, default 2)"`

## Task 2: `AgentDef.subagents`（定義パーサ）

**Files:**
- Modify: `server/src/agents.rs`（`AgentMeta` 約16行、`AgentDef` 約28行、`parse_agent` 約80行、tests）

**Interfaces:**
- Produces: `AgentDef { name, description, tools, subagents: Option<Vec<String>>, prompt, profile }`。Task 3 の `subagent_tool_specs` が `def.subagents` を読む。

- [ ] Step 1: 失敗するテストを `agents.rs` tests に追加（既存 `an_omitted_tools_list_is_none_not_empty` / `an_empty_tools_list_is_kept_as_empty` / `a_profile_is_parsed_onto_the_definition` と同型）:

```rust
/// `subagents` は `tools` と同型: 欠落は None（＝委派元から見えている
/// 集合をそのまま継承）、空リストは Some(empty)（＝委派不可）。
#[test]
fn subagents_field_parses_like_tools_does() {
    let d = tempfile::tempdir().unwrap();
    write(d.path(), "plain.md", "---\ndescription: Thinks.\n---\nThink.\n");
    write(
        d.path(),
        "limited.md",
        "---\ndescription: Plans.\nsubagents: [explorer]\n---\nPlan.\n",
    );
    let agents = load_agents_dir(d.path());
    let plain = agents.iter().find(|a| a.name == "plain").unwrap();
    let limited = agents.iter().find(|a| a.name == "limited").unwrap();
    assert_eq!(plain.subagents, None);
    assert_eq!(
        limited.subagents.as_deref(),
        Some(["explorer".to_string()].as_slice())
    );
}
```

- [ ] Step 2: `cargo test -p sapphire-agent-server --lib agents::tests` 失敗確認
- [ ] Step 3: `AgentMeta` に `#[serde(default)] pub subagents: Option<Vec<String>>`、`AgentDef` に `pub subagents: Option<Vec<String>>`（doc コメントは `tools` のものと対称に: 「`None` means inherit whatever the delegating turn can see. `Some(vec![])` means never delegate further, which is a legitimate definition.」）、`parse_agent` のコンストラクタに `subagents: meta.subagents` を追加
- [ ] Step 4: テスト全パス確認（既存テストが `AgentDef` リテラルの全箇所に影響するので `cargo test -p sapphire-agent-server --lib` を広く回す。テスト内の `AgentDef { ..defs()[0].clone() }` 形式はフィールド追加に追従する）
- [ ] Step 5: commit `"feat(agents): parse the subagents: allowlist on definitions"`

## Task 3: 段数伝達と `subagent_tool_specs` 一般化（中核）

**Files:**
- Modify: `server/src/serve/mod.rs`（`TurnContext` に `subagent_depth` 追加・約2293行、`run_llm_turn` の `TurnLoop` 構築に `subagent_depth: 0`・約2936行、既存テスト `a_subagent_cannot_invoke_subagent_by_name` の書き換え・約4016行〜）
- Modify: `server/src/tools/subagent.rs`（`subagent_tool_specs` 約158行の一般化、`run_and_store` 約707行の呼び出し変更、モジュールdocの「3つ目の性質」更新、テスト更新）

**Interfaces:**
- Consumes: Task 1 の `SubagentConfig.max_depth`、Task 2 の `AgentDef.subagents`
- Produces: `subagent_tool_specs(def, parent_visible, all_agents, depth, max_depth) -> Vec<ToolSpec>`（上記「設計の確定事項」2の挙動）、`TurnContext.subagent_depth: u32`

- [ ] Step 1（TDD）: `serve/mod.rs` の既存テスト `a_subagent_cannot_invoke_subagent_by_name` を「`max_depth = 1` なら従来どおり拒否」に書き換え。`ServeState` への設定は既存の `build_for_test_with` が `rounds` を引数に取る方式に倣うか、テスト内で `state.config.tools.subagent = SubagentConfig { turn_timeout_secs: 0, max_depth: 1 }` を直接代入できる型ならその場で設定（`ServeState.config` は `pub(crate)`）。テスト名は `a_subagent_cannot_nest_past_max_depth_one` に改名し、doc コメントは「段数上限がゲートでどう効くかを検証する」趣旨に更新。既存の4スクリプト・4コール検証（5コール目が走らないこと＝ネスト不発）はそのまま流用。
- [ ] Step 2: 同じスクリプト・同じエージェントで `max_depth = 2`（既定）ならネストが**実際に走る**対テスト `a_subagent_can_nest_up_to_max_depth_two` を追加。スクリプトは5コール目まで用意し、5つ目はネスト先（深さ2）の最終回答、4つ目は深さ1エージェントが `subagent` を再帰呼び出ししてツール結果を受け取るラウンド、親の最終回答が5つ目——という順序になるため、実際には**5コールで `chat_log.calls().len() == 5`** を検証する（深さ2の応答→深さ1の応答→親の応答で親ターンは3コール、全体で3+2ではなく、ネストが1段さらにネストして深さ2が動くため、スクリプトは親2＋深さ1の2ラウンド＋深さ2の1コール＝5コール）。
- [ ] Step 3: 失敗確認（現状は `subagent_tool_specs` が常に `subagent` を除去するため、Step 2 の新テストは「4コールで終わる」ので失敗する。Step 1 の改名テストは現状のコードでも通る）
- [ ] Step 4: `TurnContext` にフィールド追加:

```rust
    /// このターン自身のネスト段数。メインターンは 0、サブエージェントの
    /// ネストターンは委派元の +1（`SubagentTool::run_and_store` が
    /// 渡す）。`tools.subagent.max_depth` と合わせて、このターンが
    /// さらに委派できるかを `subagent_tool_specs` が判定する。
    pub subagent_depth: u32,
```

  （コメントは英語で書く — 既存コメント同様。上記は内容の概要）
- [ ] Step 5: `run_llm_turn` の `TurnLoop { ... }` に `subagent_depth: 0` を追加。テストヘルパー（`serve/mod.rs` と `subagent.rs` の `turn_context()`、`a_resumed_child_continues_its_own_history` の手製 `TurnContext`）にも `subagent_depth: 0` を追加
- [ ] Step 6: `subagent_tool_specs` を新シグネチャへ。新実装:

```rust
/// The tools a subagent may use, and the tool list its *own*
/// delegations would show — see this module's doc for what actually
/// enforces the cap.
///
/// `depth` is the delegating turn's own nesting depth (main turn
/// = 0) and `max_depth` the `[tools.subagent]` cap: the nested turn
/// sits at `depth + 1`, and it can delegate further exactly when
/// `depth + 1 <= max_depth`. `all_agents` is the currently-loaded
/// definition list — the source for rebuilding the `subagent` spec
/// embedded into the nested turn when a definition's `subagents:`
/// list narrows it.
pub(crate) fn subagent_tool_specs(
    def: &AgentDef,
    parent_visible: &[ToolSpec],
    all_agents: &[AgentDef],
    depth: u32,
    max_depth: u32,
) -> Vec<ToolSpec> {
    let nested_allowed = depth + 1 <= max_depth && def.subagents != Some(vec![]);
    let mut specs: Vec<ToolSpec> = parent_visible
        .iter()
        .filter(|s| s.name.as_ref() != SUBAGENT_TOOL_NAME)
        .filter(|s| match &def.tools {
            Some(allowed) => allowed.iter().any(|a| a == s.name.as_ref()),
            None => true,
        })
        .cloned()
        .collect();
    if nested_allowed {
        let nested_spec = match &def.subagents {
            // `None`: inherit the delegating turn's own view verbatim —
            // the spec embedded there already carries exactly the set
            // that turn can see.
            None => parent_visible.iter().find(|s| s.name.as_ref() == SUBAGENT_TOOL_NAME).cloned(),
            // `Some(list)`: rebuild narrowed to the listed names,
            // against the full registered list.
            Some(allowed) => {
                let visible: Vec<AgentDef> = all_agents
                    .iter()
                    .filter(|a| allowed.iter().any(|n| *n == a.name))
                    .cloned()
                    .collect();
                Some(build_spec(&visible))
            }
        };
        specs.extend(nested_spec);
    }
    specs
}
```

- [ ] Step 7: `run_and_store` の呼び出しを `subagent_tool_specs(def, &ctx.visible_specs, &self.agents(), ctx.subagent_depth, ctx.state.config.tools.subagent.max_depth)` に更新（`let specs = ...`・約707行）
- [ ] Step 8: `subagent.rs` の既存純関数テスト更新・追加:
  - `a_subagents_tool_list_never_contains_subagent` → 名称を現状の意味に合わせ、(a) `max_depth = 1` なら従来どおり `subagent` を含まない、(b) 既定 `max_depth = 2` かつ `depth = 0` なら**含む**、に更新。呼び出しに `&defs()`（テスト内既存の `defs()` ヘルパー）・depth・max_depth を追加
  - `a_definition_cannot_grant_itself_subagent` → `tools: Some(vec![SUBAGENT_TOOL_NAME, ...])` でも `subagent` はツールリストに**ツールの形で**入らないことを検証（深さにより別名で復活するだけ）、という趣旨に doc を更新して維持
  - 新規: `subagents: [explorer]` 定義の委派先仕様の説明に `explorer` のみ載る（`explorer: ` を含み `other-agent:` を含まない）、`subagents: []` は段数0でも `subagent` を含まない、`subagents: None` は委派元一覧を継承する
  - 新規: `depth + 1 > max_depth`（例: depth=1, max_depth=2）では `subagent` を含まない
- [ ] Step 9: `cargo test -p sapphire-agent-server --lib subagent` と `--lib serve` 全パス
- [ ] Step 10: `cargo clippy -- -D warnings` と `cargo fmt --check`
- [ ] Step 11: モジュールdocの「3. The tool list actually offered is enforced, not just built restricted.」の段落を新機構に合わせて更新（「常に除去」→「段数上限で条件付きに付与。ゲートが実体であることは変わらない——ゲートはその回の `tool_specs` の membership を見るだけであり、付与されたリストにも `subagent` はゲート経由でしか届かない」趣旨。既存 doc の `s-subagent-no-recursion` 参照テスト名も改名後の名前に更新）
- [ ] Step 12: commit `"feat(subagent): allow nesting up to max_depth, with per-definition subagents allowlists"`

## Task 4: ドキュメント（README）

**Files:**
- Modify: `README.md`（Subagents セクション、約107行〜）
- Modify: `README.ja.md`（サブエージェント言及、24行付近の箇条書きと同セクション）

- [ ] Step 1: `README.md` の Subagents セクションに英語で追記 — `[tools.subagent] max_depth`（既定 2、`0` で委派禁止、`1` が従来の「常に一层のみ」）、定義 frontmatter の `subagents:`（`tools:` と同型: 欠落 = 委派元から見えている集合を継承、`[]` = 委派不可、リスト = その名前に限定）、ネストしたサブエージェントも同じ段数上限と許可リストの連鎖で決まること、各ネスト段が個別に `turn_timeout_secs` の締切対象になること
- [ ] Step 2: `README.ja.md` の該当箇所に日本語で同内容
- [ ] Step 3: commit `"docs(readme): document max_depth and per-definition subagents allowlists"`

## Task 5: 統合検証（全体）

- [ ] `cargo test -p sapphire-agent-server`（workspace 全体でなく server でよい。CI は `--workspace` を使うが既存の流儀に合わせる）
- [ ] `cargo clippy -- -D warnings`、`cargo fmt --check`
- [ ] 既存テストの失敗がすべて新挙動への追随であることを確認（特に `serve/mod.rs` の既存 subagent テスト、`config.rs`、`agents.rs`）

## 対象外（変更しないもの）

- `subagent_cache` のハンドル・履歴保存機構（段数の保存不要。resume は仕様を毎回再計算する既存経路に乗るだけ）
- `turn_timeout_secs` のセマンティクス（各ネストターンが個別に締切対象、は既存挙動のまま）
- `agent_`/`user_` プレフィクス分離（別 issue #261 の方向性。名前が変わる場合は `SUBAGENT_TOOL_NAME` 参照箇所が追随するだけ）
