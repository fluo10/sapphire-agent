# サブエージェントのプロファイル（モデル）選択 実装計画

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** サブエージェント定義のフロントマターに `profile:` を1つ足し、指定された定義は親ではなくそのプロファイルのプロバイダで走るようにする。未指定の定義は現状どおり親追従。

**Architecture:** データは `AgentMeta`/`AgentDef` に `profile: Option<String>` を1本増やすだけ。実行時は `run_and_store` の `provider: &ctx.provider` の1箇所を「定義に `profile` があれば `ctx.state.registry.for_profile(&ctx.state.config, name)`、なければ `Arc::clone(&ctx.provider)`」に置き換える。プロファイル解決の意味論は既存の `ProviderRegistry::for_profile` 1本に集約したまま何もしない（未知名の扱いの差し替え可能性は `for_profile` 側にドキュメントで明文化する）。起動時検証は `Config::validate_subagent_profiles` を新設し、`main.rs` の登録箇所で `load_agents_dir` の宽松な読み込みとは**別に**走らせ、未知名があれば起動を落とす。

**Tech Stack:** Rust 2024, `serde` / `serde_yaml`, `tokio`, `async-trait`

**Spec:** `docs/superpowers/specs/2026-09-11-subagent-profiles-design.md`

## Global Constraints

- ブランチは `feat/subagent-profiles`（`main` から作成し、仕様書のコミットを最初に載せる）。
- テストコマンドは `cargo test -p sapphire-agent`。反復中は `--workspace` を使わない。最後に一度だけ `cargo test --workspace`。
- **`cargo check` と `cargo test` を混ぜない。** フィンガープリントが別なので交互に走らせると毎回リンクし直しになる。
- cargo は**前景で、`timeout: 600000` で**。`run_in_background` も `Monitor` も使わない。10分のツールタイムアウトに当たったらビルドは温まっているので同じコマンドを走らせ直す。**cargo を2本同時に走らせない**（このホストの OS は熱でスロットリングする小さな USB SSD 上にある）。
- **コミット前に `cargo clippy --workspace -- -D warnings` — CI と同じ形、`--all-targets` を付けない。** `.github/workflows/ci.yml` が走らせるのはこれで、`--all-targets` より厳しい（`--all-targets` はテストコードをコンパイルするので、その呼び出し元が未使用アイテムの警告を隠す）。
- **`Cargo.lock` をコミットしない。** 各コミット前に `git checkout -- Cargo.lock`。
- テストはソースファイル内の `mod tests` に置く。
- **`src/agent.rs` は編集しない。**
- 既存のサブエージェントの性質（親の `TurnHost`・同一 `Origin`・`NotOffered` ゲート・深さ1・非永続化）は1つも変えない。本計画が変えるのはプロバイダの**選択**だけ。

---

### Task 1: `profile` を定義のフロントマターに乗せる

`AgentMeta`/`AgentDef` にフィールドを1本増やし、パースして、壊れたテストを全て通すところまで。実行時の挙動はまだ変わらない（この Task の成果物は「データが乗ること」）。

**Files:**
- Modify: `server/src/agents.rs`（`AgentMeta`、`AgentDef`、`parse_agent`、`mod tests`）
- Modify: `server/src/tools/subagent.rs`（テスト内の `AgentDef` リテラル全箇所）
- Test: 同上 `mod tests`

**Interfaces:**
- Produces: `AgentDef { name, description, tools, prompt, profile }` — `pub profile: Option<String>`。`None` は「親のプロバイダをそのまま使う」を意味する（Task 3 の実装がこれを消費する）。

#### なぜテスト literal の編集が Task 1 に含まれるか

`AgentDef` は `Default` を derive しない（`name`/`description`/`prompt` に既定値はなく、既定値のない構造体として全フィールド明示で構築する流儀を崩さない）。したがってフィールド追加は `server/src/tools/subagent.rs` のテスト内リテラル9箇所（現行行番号 788, 882, 894, 912, 937, 957, 970, 1102, 1512 付近）全てに `profile: None,` を足さないと**その場でコンパイルが落ちる**。足さないまま Task 3 まで置く、という並びにできない。

- [ ] **Step 1: 失敗するテストを書く（パース）**

`server/src/agents.rs` の `mod tests` に追加。既存のテストと同じ `write` ヘルパを使う。

```rust
/// `profile:` is parsed onto the definition — it is the whole point of
/// the field: a definition pins its own provider with it.
#[test]
fn a_profile_is_parsed_onto_the_definition() {
    let d = tempfile::tempdir().unwrap();
    write(
        d.path(),
        "reviewer.md",
        "---\ndescription: Reviews.\nprofile: dev\n---\nReview.\n",
    );

    let agents = load_agents_dir(d.path());
    assert_eq!(agents.len(), 1);
    assert_eq!(agents[0].profile.as_deref(), Some("dev"));
}

/// Absent `profile:` means "the parent's provider, unchanged" — that
/// must be distinguishable from `Some`. Same shape as the `tools` test:
/// absent is `None`, never an empty string or anything else.
#[test]
fn an_omitted_profile_is_none_not_empty() {
    let d = tempfile::tempdir().unwrap();
    write(d.path(), "helper.md", "---\ndescription: Thinks.\n---\nThink.\n");

    let agents = load_agents_dir(d.path());
    assert_eq!(agents[0].profile, None);
}
```

- [ ] **Step 2: テストを実行して失敗を確認する**

Run: `cargo test -p sapphire-agent agents::tests::a_profile_is_parsed -- --nocapture`
Expected: FAIL（コンパイルエラー `missing field profile in ... AgentDef` を含む）

- [ ] **Step 3: `agents.rs` を実装する**

`AgentMeta` と `AgentDef` と `parse_agent`:

```rust
#[derive(Debug, Clone, Deserialize)]
struct AgentMeta {
    description: String,
    #[serde(default)]
    tools: Option<Vec<String>>,
    /// Name of the `[profiles.<name>]` entry this agent runs on. `None`
    /// means "the parent turn's provider, unchanged" — the default and
    /// the pre-feature behaviour.
    #[serde(default)]
    profile: Option<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AgentDef {
    /// The file stem — what the model passes as `agent`.
    pub name: String,
    pub description: String,
    /// `None` means "whatever the parent can see". `Some(vec![])` means
    /// no tools at all, which is a legitimate definition.
    pub tools: Option<Vec<String>>,
    /// The body, which becomes the whole system prompt.
    pub prompt: String,
    /// The name in `[profiles.<n>]` this agent runs on, or `None` for
    /// "the parent's provider". The name is validated against
    /// `Config::profiles` at startup (`Config::validate_subagent_profiles`)
    /// — a server config with an unknown name fails to boot rather than
    /// silently running the agent on a different model.
    pub profile: Option<String>,
}
```

`parse_agent` の `Some(AgentDef { ... })` に `profile: meta.profile,` を1行足す。

- [ ] **Step 4: `subagent.rs` のテストリテラル全箇所に `profile: None,` を足す**

現行9箇所（`defs()`、`resumable_defs()` を含む全 `crate::agents::AgentDef {` リテラル）。フィールド1つ分の機械的変更なので別コミットにせずこの Task の1コミットに含める。

- [ ] **Step 5: テストを通す**

Run: `cargo test -p sapphire-agent agents:: tests:: 2>/dev/null || cargo test -p sapphire-agent`
Expected: 全パス（パスしないと「テストを壊していない」が証明できない）

- [ ] **Step 6: clippy → コミット**

```bash
cargo clippy --workspace -- -D warnings
git checkout -- Cargo.lock
git add server/src/agents.rs server/src/tools/subagent.rs
git commit -m "feat(agents): parse an optional `profile:` onto agent definitions"
```

---

### Task 2: 未知の `profile` 参照を起動時に落とす

読み込み方針（1つ壊れても他は動かす）は変えない。検証は読み込みとは**別の場所** — 定義群と設定の両方を持つ `main.rs` の登録箇所 — で走らせる。エラーメッセージは `validate_profiles` と同様に全件列挙。

**Files:**
- Modify: `server/src/config.rs`（`Config::validate_subagent_profiles` 新設、`mod tests`）
- Modify: `server/src/main.rs`（subagents 登録ブロック、現行 510 行付近）

**Interfaces:**
- Consumes: Task 1 の `AgentDef.profile`
- Produces: `Config::validate_subagent_profiles(&self, defs: &[crate::agents::AgentDef]) -> Vec<String>` を Task 3 以降のドキュメントから参照する（実装上は Task 3 と同一 PR で、この Task 単体ではまだ誰からも呼ばれない — 呼ぶ側は Task 3）。

- [ ] **Step 1: 失敗するテストを書く**

`server/src/config.rs` の `mod tests` に、既存の `validate_profiles` テストと同じ `parse()` ヘルパの流儀で追加する。

```rust
/// An agent definition naming a profile that config never defines is a
/// static reference error in a server config: better to fail at startup
/// than to silently run the agent on `for_profile`'s fallback provider.
/// The error list enumerates every occurrence, like `validate_profiles`.
#[test]
fn a_subagent_definition_naming_an_unknown_profile_is_an_error() {
    let cfg = parse(
        r#"
[anthropic]
api_key = "test"

[profiles.dev]
provider = "anthropic"
"#,
    );
    let defs = vec![
        crate::agents::AgentDef {
            name: "reviewer".into(),
            description: "Reviews.".into(),
            tools: None,
            prompt: "Review.".into(),
            profile: Some("dev".into()),
        },
        crate::agents::AgentDef {
            name: "impl".into(),
            description: "Implements.".into(),
            tools: None,
            prompt: "Implement.".into(),
            profile: Some("missing".into()),
        },
        crate::agents::AgentDef {
            name: "helper".into(),
            description: "Thinks.".into(),
            tools: None,
            prompt: "Think.".into(),
            profile: None,
        },
    ];
    let errors = cfg.validate_subagent_profiles(&defs);
    assert_eq!(errors.len(), 1, "{errors:?}");
    assert!(errors[0].contains("impl"), "{errors[0]}");
    assert!(errors[0].contains("missing"), "{errors[0]}");
}

/// A defined profile — including one whose *provider* would fall back —
/// is not an error. Existence of the profile name is the whole check.
#[test]
fn a_subagent_definition_naming_a_defined_profile_is_fine() {
    let cfg = parse(
        r#"
[anthropic]
api_key = "test"

[profiles.dev]
provider = "anthropic"
"#,
    );
    let defs = vec![crate::agents::AgentDef {
        name: "reviewer".into(),
        description: "Reviews.".into(),
        tools: None,
        prompt: "Review.".into(),
        profile: Some("dev".into()),
    }];
    assert!(cfg.validate_subagent_profiles(&defs).is_empty());
}
```

- [ ] **Step 2: 実行して失敗（メソッド不存在でコンパイルエラー）を確認する**

Run: `cargo test -p sapphire-agent config::tests 2>&1 | tail -5`
Expected: `no method validate_subagent_profiles` コンパイルエラー

- [ ] **Step 3: 実装する**

`config.rs` の `validate_profiles` の直後に置く（同じ「設定ファイル内の静的な参照」検証の一族として）:

```rust
/// Validate the `profile:` references of loaded subagent definitions
/// against `profiles`. Called by `main` once both the definitions
/// (`load_agents_dir`) and this config exist — the loader stays
/// lenient (one broken definition must not take the others down) and
/// the strictness lives here instead: an unknown name in a *server*
/// config is a static reference error, and failing startup beats
/// silently running that agent on `for_profile`'s fallback provider.
/// The permissive variant the future client/local-loop layout wants
/// (warn + fallback) is a policy change at the *resolution* point
/// (`ProviderRegistry::for_profile`), not a reason to soften this check.
///
/// Returns human-readable error messages, one per definition that names
/// an unknown profile, in definition order.
pub fn validate_subagent_profiles(&self, defs: &[crate::agents::AgentDef]) -> Vec<String> {
    defs.iter()
        .filter_map(|d| d.profile.as_deref())
        .filter(|name| !self.profiles.contains_key(*name))
        .map(|name| format!("subagent definition references unknown profile '{name}'"))
        .collect()
}
```

`main.rs` のサブエージェント登録ブロック（既存コメント「Loaded from `<workspace>/agents/*.md`」に続く箇所）で検証を登録の**前**に挟む:

```rust
let agent_defs = agents::load_agents_dir(&workspace_dir.join("agents"));
let profile_errors = config.validate_subagent_profiles(&agent_defs);
if !profile_errors.is_empty() {
    anyhow::bail!(
        "invalid subagent profile references:\n  - {}",
        profile_errors.join("\n  - ")
    );
}
if !agent_defs.is_empty() {
    // 以下既存の register_tool のまま
```

エージェント定義が1つも無いデプロイでは `agent_defs` は空ベクトル、検証は恒真で通り、従来どおりツール自体を登録しない。

- [ ] **Step 4: テストを通す**

Run: `cargo test -p sapphire-agent config::tests`
Expected: 全パス

- [ ] **Step 5: clippy → コミット**

```bash
cargo clippy --workspace -- -D warnings
git checkout -- Cargo.lock
git add server/src/config.rs server/src/main.rs
git commit -m "feat(config): reject agent definitions naming an unknown profile at startup"
```

---

### Task 3: 定義の `profile` をプロバイダ解決に接続する

`run_and_store` の `provider: &ctx.provider` 直参照を置き換える。これがこの仕様の実質的な本体。`for_profile` の既存の意味論（未知名は anthropic にフォールバック、`fallback_provider` を巻く）はそのまま — 未知名は Task 2 で起動時に弾かれているので実行時には届かない。

**Files:**
- Modify: `server/src/tools/subagent.rs`（`run_and_store`、`mod tests` の `ScriptedProvider` とテスト）
- Modify: `server/src/provider/registry.rs`（`for_profile` の doc コメントのみ — 挙動は変えない）

**Interfaces:**
- Consumes: Task 1 の `AgentDef.profile`、既存の `ProviderRegistry::for_profile(&Config, &str) -> Arc<dyn Provider>`、`TurnContext.state`（`pub(crate)` の `config` と `registry`）
- Produces: なし（最終機能的成果物）

- [ ] **Step 1: 失敗するテストを書く**

`server/src/tools/subagent.rs` の `mod tests` に。`turn_context` ヘルパと `extract_handle`・`text_response`・`resumable_defs` は既存のものを使う。`ScriptedProvider` に受付件数を見るヘルパを1つ足す（`last_messages` は0件だと panic するので「呼ばれなかった」の検証に使えないため）:

```rust
/// How many `chat()` calls this double received. `last_messages`
/// panics on an empty log by design; "this provider received nothing"
/// needs a count instead.
fn call_count(&self) -> usize {
    self.calls.lock().unwrap().len()
}
```

テスト3本。`ServeState::for_test` のフィクスチャは stub provider を `"anthropic"` と `"stub"` の両名に登録し、`[profiles.dev] provider = "stub"` を含む（`serve/mod.rs` の `build_for_test_with` を読む）ので、「ctx 側のプロバイダ」と「プロファイル解決で引けるプロバイダ」を別々のダブリンで区別できる。

```rust
/// A definition with `profile:` runs on that profile's provider, not the
/// parent's — the point of the feature. The fixture's registry
/// (`serve::build_for_test_with`) registers one scripted provider under
/// both the `"anthropic"` and `"stub"` names, so profile resolution and
/// the ctx provider cannot be told apart by *identity* here — the
/// distinction is made by *recorded calls*: the ctx provider is a
/// separate record-only double whose log must stay empty.
#[tokio::test]
async fn a_definition_with_a_profile_runs_on_that_profiles_provider() {
    let tool = SubagentTool::new(vec![crate::agents::AgentDef {
        name: "impl".to_string(),
        description: "Implements a task.".to_string(),
        tools: Some(vec![]),
        prompt: "You are impl.".to_string(),
        profile: Some("dev".to_string()),
    }]);
    let state = crate::serve::ServeState::for_test_scripted(
        false,
        vec![text_response("profile answer")],
    );
    let parent_provider = ScriptedProvider::new(vec![]);
    let out = crate::serve::scope_turn_context(
        turn_context(
            std::sync::Arc::clone(&state),
            std::sync::Arc::clone(&parent_provider) as std::sync::Arc<dyn crate::provider::Provider>,
            Vec::new(),
        ),
        tool.execute(&serde_json::json!({"agent": "impl", "prompt": "go"})),
    )
    .await
    .unwrap();
    assert!(out.contains("profile answer"), "{out}");
    assert_eq!(
        parent_provider.call_count(),
        0,
        "the ctx (parent's) provider must not serve a profiled agent"
    );
}

/// The mirror image: no `profile:`, the parent's provider serves the
/// turn — the pre-feature default is unchanged behaviour, not an
/// accidental second path. Same fixture shape as above; here the ctx
/// provider IS the one that must receive the call.
#[tokio::test]
async fn a_definition_without_a_profile_runs_on_the_parents_provider() {
    let tool = SubagentTool::new(resumable_defs()); // profile: None
    let state = crate::serve::ServeState::for_test(false);
    let parent_provider = ScriptedProvider::new(vec![text_response("parent answer")]);
    let out = crate::serve::scope_turn_context(
        turn_context(
            std::sync::Arc::clone(&state),
            std::sync::Arc::clone(&parent_provider) as std::sync::Arc<dyn crate::provider::Provider>,
            Vec::new(),
        ),
        tool.execute(&serde_json::json!({"agent": "impl", "prompt": "go"})),
    )
    .await
    .unwrap();
    assert!(out.contains("parent answer"), "{out}");
    assert_eq!(parent_provider.call_count(), 1);
}

/// A handle stays resumable after the definition's `profile:` changes:
/// the cache stores model-agnostic `ChatMessage` history, so a resume
/// simply runs on whatever provider the reloaded definition resolves
/// to. This is the existing `an_agent_definition_that_disappeared_is_
/// reported` pattern — two tool instances sharing one cache-backed
/// state, standing in for "definitions reloaded between dispatch and
/// resume" — with the one differing field being `profile`: dispatch
/// runs with `profile: None` (the ctx provider), resume runs with
/// `profile: Some("dev")` and must still continue the stored history.
#[tokio::test]
async fn a_resumed_handle_survives_a_profile_change() {
    let state = crate::serve::ServeState::for_test_scripted(
        false,
        vec![text_response("dispatch answer"), text_response("resume answer")],
    );
    let base = crate::agents::AgentDef {
        name: "impl".to_string(),
        description: "Implements a task.".to_string(),
        tools: Some(vec![]),
        prompt: "You are impl.".to_string(),
        profile: None,
    };
    let dispatched = crate::serve::scope_turn_context(
        turn_context(
            std::sync::Arc::clone(&state),
            state.registry.anthropic(),
            Vec::new(),
        ),
        SubagentTool::new(vec![base.clone()])
            .execute(&serde_json::json!({"agent": "impl", "prompt": "first task"})),
    )
    .await
    .unwrap();
    let handle = extract_handle(&dispatched);

    let resumed_def = crate::agents::AgentDef {
        profile: Some("dev".to_string()),
        ..base
    };
    let resumed = crate::serve::scope_turn_context(
        turn_context(
            std::sync::Arc::clone(&state),
            state.registry.anthropic(),
            Vec::new(),
        ),
        SubagentTool::new(vec![resumed_def])
            .execute(&serde_json::json!({"resume": handle, "prompt": "second instruction"})),
    )
    .await
    .unwrap();
    assert!(resumed.contains("resume answer"), "{resumed}");
    assert!(resumed.contains(&handle), "still resumable: {resumed}");
}
```

- [ ] **Step 2: 実行して失敗を確認する**

Run: `cargo test -p sapphire-agent subagent::tests::a_definition_with_a_profile -- --nocapture`
Expected: FAIL — 現状は `profile` があっても ctx 側プロバイダが呼ばれる（call_count==1）か、テストが panic。

- [ ] **Step 3: `run_and_store` を実装する**

既存の「The parent's host, deliberately: …」コメントの直前に置き、`TurnLoop { provider: … }` を差し替える:

```rust
// A definition that pins a profile runs on that profile's provider —
// resolved through the very same `ProviderRegistry::for_profile` that
// room/session turns use, so profile-resolution semantics exist in one
// function, not two. The "unknown name" policy that future
// client/local-loop configs may want to change (warn + fallback
// instead of failing startup) is a change to that one function; a
// server config cannot reach it at runtime —
// `Config::validate_subagent_profiles` bails at startup first.
// `fallback_provider` wrapping happens inside `for_profile`, so a
// profiled subagent inherits the refusal-fallback behaviour for free.
let provider: std::sync::Arc<dyn crate::provider::Provider> = match def.profile.as_deref() {
    Some(name) => ctx.state.registry.for_profile(&ctx.state.config, name),
    None => std::sync::Arc::clone(&ctx.provider),
};
```

`TurnLoop` 構造体の `provider: &ctx.provider,` を `provider: &provider,` に変える。

`registry.rs` の `for_profile` の doc コメントに1段落足す（挙動変更なし）:

```rust
/// The single entry point for profile → provider resolution: room and
/// session turns go through here, and so do subagent definitions'
/// `profile:`. The "name not found → Anthropic fallback" behaviour
/// below is deliberately concentrated here: server configs make it
/// unreachable by rejecting unknown names at startup
/// (`Config::validate_subagent_profiles`); a future client/local-loop
/// config that wants warn-and-fallback instead changes only this point.
```

- [ ] **Step 4: テストを通す**

Run: `cargo test -p sapphire-agent subagent::tests`
Expected: 全パス（既存の resume / isolation / ゲート共通テスト含む）

- [ ] **Step 5: clippy → コミット**

```bash
cargo clippy --workspace -- -D warnings
git checkout -- Cargo.lock
git add server/src/tools/subagent.rs server/src/provider/registry.rs
git commit -m "feat(subagent): run profiled definitions on their profile's provider"
```

---

### Task 4: ドキュメント（README / README.ja）と統合確認

**Files:**
- Modify: `README.md`（"Subagents" 節、`tools` の bullet の直後）
- Modify: `README.ja.md`（対応する和文セクション）

**Interfaces:**
- Consumes: Task 1–3 の実装全体

- [ ] **Step 1: README.md の定義例と bullet を更新する**

```markdown
---
description: Reviews a diff. Reads and reports; does not edit.
tools: [client_file_read, workspace_search, memory_read]
profile: dev
---
```

bullet を1本追加（`tools` の bullet の直後に）:

```markdown
- **`profile`** is optional and names a `[profiles.<name>]` entry — the
  subagent then runs on *that* provider (with its `fallback_provider`
  included) instead of the parent's. Omit it and the subagent runs on
  whatever model the delegating turn runs on: both are legitimate, and
  "no profile" is the pre-existing behaviour unchanged. A name that
  config's `[profiles]` does not define fails startup outright — a
  typo'd profile name should not silently run the agent on a different
  model.
```

「What a subagent does not inherit」節の直後に1文: プロバイダは親から**引き継がれないことがある**唯一の要素であり、許可ゲート・ツール一覧・隔離はプロファイルの有無で一切変わらないことを1段落で書く（英語・日本語同じ内容）。

- [ ] **Step 2: README.ja.md に対応和文を書き、日本語版の定義例の frontmatter にも `profile: dev` を入れる**

例: 「**`profile`** は省略可。書くとその `[profiles.<name>]` のプロバイダ（`fallback_provider` 込み）で走る。省略時は委譲元のターンと同じモデルで走る — これが仕様変更前の既定動作そのまま。」

- [ ] **Step 3: 統合確認**

```bash
cargo test --workspace
```
Expected: 全パス。CI と同じ clippy は Task 3 時点で通っていること。

- [ ] **Step 4: コミット・PR・マージ後ブランチ削除**

```bash
git checkout -- Cargo.lock
git add README.md README.ja.md
git commit -m "docs: document the subagent `profile` frontmatter field"
git push -u origin feat/subagent-profiles
```
（PR 作成・レビュー・マージはプロセス側の手順。この計画の成果物はマージ済み `main`。）

---

## 計画全体の自己レビュー結果

- **仕様カバレッジ**: 決定1（frontmatter のみ）→ Task 1+3、決定2（サーバー起動時検証／解決入口集約）→ Task 2+3 の `for_profile` doc、決定3（検証の位置）→ Task 2、決定4（resume 復活）→ Task 3 の3本目テスト、決定5（実行時不変）→ 全 Task の「変えるのは選択だけ」制約。やらないこと4項目は計画にタスクなし（意図的）。
- **プレースホルダ**: Task 3 3本目のテストコードは既存テスト同型部分の省略記法付き（「dispatch 時 None → resume 時 Some のみ diff」と明記済み。実装者は既存テストをコピーしてその1行だけ変える）。
- **型の一致**: `AgentDef.profile: Option<String>` は Task 1 で定義し Task 2/3 で同一型で消費。`validate_subagent_profiles(&[AgentDef]) -> Vec<String>` は Task 2 定義 → Task 2 main.rs 呼び出しで同一シグネチャ。`call_count(&self) -> usize` は Task 3 で定義・同一ファイル内使用。
