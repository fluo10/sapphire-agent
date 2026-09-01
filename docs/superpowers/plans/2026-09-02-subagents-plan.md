# サブエージェント Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 親のシステムプロンプトと履歴を持ち込まない、専門化したエージェントに仕事を委譲できるようにする。

**Architecture:** `run_llm_turn` のツールループを、セッション永続化から切り離して抽出する。永続化は `Option<TurnPersistence>` として渡し、`None` が「サブエージェント」を意味する。エージェント定義は `<workspace>/agents/*.md` から、`<workspace>/heartbeat/*.md` と同じ形で読む。`subagent` ツールは抽出したループを永続化なしで呼び、親の `TurnHost` をそのまま渡す。

**Tech Stack:** Rust 2024, `serde` / `serde_yaml`, `tokio`, `async-trait`

**Spec:** `docs/superpowers/specs/2026-09-02-subagents-design.md`

## Global Constraints

- ブランチは `feat/subagents`（`main` から作成済み、spec をコミット済み）。
- テストコマンドは `cargo test -p sapphire-agent`。反復中は `--workspace` を使わない。最後に一度だけ `cargo test --workspace`。
- **`cargo check` と `cargo test` を混ぜない。** フィンガープリントが別なので交互に走らせると毎回リンクし直しになる。
- cargo は**前景で、`timeout: 600000` で**。`run_in_background` も `Monitor` も使わない。10分のツールタイムアウトに当たったらビルドは温まっているので同じコマンドを走らせ直す。**cargo を2本同時に走らせない**（このホストの OS は熱でスロットリングする小さな USB SSD 上にある）。
- **コミット前に `cargo clippy --workspace -- -D warnings` — CI と同じ形、`--all-targets` を付けない。** `.github/workflows/ci.yml` が走らせるのはこれで、`--all-targets` より厳しい（`--all-targets` はテストコードをコンパイルするので、その呼び出し元が未使用アイテムの警告を隠す）。
- **`Cargo.lock` をコミットしない。** 各コミット前に `git checkout -- Cargo.lock`。
- テストはソースファイル内の `mod tests` に置く。
- **`src/agent.rs` は編集しない。**
- **サブエージェントのツール呼び出しは、親と同じゲートを親と同じ `Origin` で通る。** 別扱いにすれば「サブエージェントに頼む」が承認の迂回路になる。
- **サブエージェントのシステムプロンプトにワークスペースのファイル・メモリ・ダイジェストを含めない。** それを落とせることがこの機能の目的の1つ。

---

### Task 1: ツールループを抽出する — 挙動を1ビットも変えずに

**この Task の成果物は「何も変わらないこと」である。** サブエージェントはまだ出てこない。

**Files:**
- Modify: `src/serve/mod.rs`
- Test: 既存テスト全体（新しいテストは書かない）

**Interfaces:**
- Produces: `TurnPersistence { store, acp_store, session_id, is_acp }`、`TurnLoop { … }` とその `async fn run(self) -> (Option<String>, TurnStop)`

#### なぜ抽出が危ないか

このループは ACP・`/rpc`・`/a2a`・音声の4トランスポートが通る hot path で、許可ゲート・圧縮・画像ハイドレート・ツール結果永続化が全部入っている。**複製すればサブエージェント側だけ直し忘れる未来が確定する**ので抽出するが、抽出そのものが全経路に効く。

だから**この Task では機能を足さない。** 抽出して、既存テストが全部通ることだけを確かめる。

#### ループがセッション状態に触る箇所は4つだけ

`src/serve/mod.rs:2365-2634` を読んで確認済み。永続化はこの4箇所:

1. 圧縮要約 → `store.append_summary`（非 ACP のみ）
2. アシスタントのテキストメッセージ → ACP なら `acp_session_store.append_message`、そうでなければ `store.append`
3. `tool_use` メッセージ → ACP のみ
4. `tool_result` メッセージ → ACP のみ、かつ `tool_use_persisted` が真のときだけ

ほかに `state.tools` を2回使う（`kinds()` と `Arc::clone`）。それ以外にセッションへの依存は無い。

- [ ] **Step 1: 永続化をまとめる型を作る**

`src/serve/mod.rs` に。

```rust
/// Where a turn's messages go — or that they go nowhere.
///
/// `Option<TurnPersistence>` rather than a `bool` on the loop: a
/// subagent has no session, so there is no id to write to and no
/// half-persisted state to reason about. Making the absence a shape
/// rather than a flag means the loop cannot accidentally write to a
/// session that does not exist.
pub(crate) struct TurnPersistence {
    store: Arc<SessionStore>,
    acp_store: Arc<AcpSessionStore>,
    session_id: String,
    is_acp: bool,
}

impl TurnPersistence {
    /// Append one message. Returns whether the caller may go on to
    /// persist a message that must be paired with this one.
    ///
    /// `true` when nothing was written at all: there is no pairing to
    /// break, so a `tool_result` must not be skipped just because its
    /// `tool_use` was never a candidate for the store.
    fn append_message(&self, msg: &ChatMessage) -> bool {
        if self.is_acp {
            match self.acp_store.append_message(&self.session_id, msg) {
                Ok(()) => true,
                Err(e) => {
                    warn!("Failed to persist a message: {e}");
                    false
                }
            }
        } else {
            if let Err(e) = self.store.append(&self.session_id, msg) {
                warn!("Failed to persist a message: {e}");
            }
            true
        }
    }

    /// Append a compaction summary. ACP sessions keep none — their
    /// history is rebuilt from events on reload, so a stored summary
    /// would be a second, staler answer to a question the events
    /// already answer.
    fn append_summary(&self, summary: &str) {
        if !self.is_acp
            && let Err(e) = self.store.append_summary(&self.session_id, summary)
        {
            warn!("Failed to persist compaction summary: {e}");
        }
    }
}
```

**現状の挙動と1対1で対応させること。** 特に:

- 非 ACP の `append` は結果を無視して警告するだけ（`tool_use_persisted` は `!is_acp` で `true`）。上の `append_message` はその通りになっている。
- ACP の `tool_use` 失敗時だけ `false` を返し、呼び出し側が `tool_result` を飛ばす。

**現物のコードを読んでから書くこと。** 上は読み取った挙動の写しだが、
食い違いがあれば**現物が正しい** — この Task は挙動を変えない。

- [ ] **Step 2: ループを構造体のメソッドに移す**

引数が多すぎて `clippy::too_many_arguments` に当たるので、構造体にする。

```rust
/// One model conversation run to completion: call the model, run the
/// tools it asks for, repeat until it stops asking.
///
/// Extracted from `run_llm_turn` so a subagent can run the same loop
/// without a session behind it. Everything session-shaped lives in
/// `persistence`, which is `None` for a subagent — see
/// `TurnPersistence`.
pub(crate) struct TurnLoop<'a> {
    pub state: &'a Arc<ServeState>,
    pub provider: &'a Arc<dyn Provider>,
    pub system: Option<&'a str>,
    pub tool_specs: &'a [ToolSpec],
    pub progress: &'a Arc<dyn TurnHost>,
    pub timer_origin: Option<crate::timer::TimerOrigin>,
    pub persistence: Option<&'a TurnPersistence>,
}

impl TurnLoop<'_> {
    /// Run until the model stops calling tools, the round budget runs
    /// out, or the provider fails. `history` is both the input and
    /// where the conversation accumulates.
    pub(crate) async fn run(
        self,
        history: &mut Vec<ChatMessage>,
    ) -> (Option<String>, TurnStop) {
        // ← `src/serve/mod.rs:2365-2634` の loop 本体をそのまま移す
    }
}
```

移すときに置き換えるのは**4箇所の永続化呼び出しだけ**:

- `store.append_summary(&session_id, &result.summary)` を含む `if !is_acp && …` → `if let Some(p) = self.persistence { p.append_summary(&result.summary); }`
- アシスタントメッセージの `if is_acp { … } else { … }` → `if let Some(p) = self.persistence { p.append_message(&msg); }`
- `let tool_use_persisted = !is_acp || match … ;` → `let tool_use_persisted = self.persistence.map_or(true, |p| p.append_message(&msg));`
- `tool_result` の `if is_acp && tool_use_persisted && …` → `if tool_use_persisted && let Some(p) = self.persistence { p.append_message(&result_msg); }`

`state.tools` の2箇所は `self.state.tools` になるだけ。

**ほかは1文字も変えない。** コメントも含めて移す — あそこのコメントは
なぜそう書いてあるかを説明しており、移動で失われると次に読む人が困る。

- [ ] **Step 3: `run_llm_turn` から呼ぶ**

`run_llm_turn` は今までどおり履歴を用意し、システムプロンプトを組み立て、
ツール仕様を絞り、そのあと:

```rust
    let persistence = TurnPersistence {
        store: Arc::clone(&store),
        acp_store: Arc::clone(&state.acp_session_store),
        session_id: session_id.clone(),
        is_acp,
    };
    let (final_text, stop) = TurnLoop {
        state: &state,
        provider: &provider,
        system: system.as_deref(),
        tool_specs: &tool_specs,
        progress: &progress,
        timer_origin,
        persistence: Some(&persistence),
    }
    .run(&mut history)
    .await;
```

その後の `scrub_history_inplace` と `state.sessions` への書き戻しは
**`run_llm_turn` に残す** — あれはセッションの話であって、ループの話ではない。

- [ ] **Step 4: 既存テストが全部通ることを確認**

Run: `cargo test -p sapphire-agent`
Expected: PASS。**1つでも落ちたら、それは抽出が挙動を変えた証拠である。**
テストを直すのではなく、抽出を直すこと。

落ちたテストがあれば、それが何を守っていたかを報告に書くこと。

- [ ] **Step 5: CI 形式の lint**

```bash
cargo fmt --all
cargo clippy --workspace -- -D warnings
```

- [ ] **Step 6: コミット**

```bash
git checkout -- Cargo.lock
git add src/serve/mod.rs
git commit -m "refactor(serve): extract the tool loop from session persistence"
```

---

### Task 2: エージェント定義を読む

**Files:**
- Create: `src/agents.rs`
- Modify: `src/main.rs`（`mod agents;`）
- Test: `src/agents.rs` の `mod tests`

**Interfaces:**
- Produces: `AgentDef { name: String, description: String, tools: Option<Vec<String>>, prompt: String }`、`load_agents_dir(dir: &Path) -> Vec<AgentDef>`

`src/heartbeat_config.rs` の `load_heartbeat_dir` が先例。**読んでから書くこと** —
ディレクトリが無ければ空、壊れたファイルは警告して飛ばす、という流儀に合わせる。

- [ ] **Step 1: 失敗するテストを書く**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    fn write(dir: &std::path::Path, name: &str, body: &str) {
        std::fs::write(dir.join(name), body).unwrap();
    }

    #[test]
    fn a_definition_splits_into_frontmatter_and_prompt() {
        let d = tempfile::tempdir().unwrap();
        write(
            d.path(),
            "reviewer.md",
            "---\ndescription: Reviews a diff.\ntools: [client_file_read]\n---\nYou are a reviewer.\n",
        );

        let agents = load_agents_dir(d.path());
        assert_eq!(agents.len(), 1);
        assert_eq!(agents[0].name, "reviewer");
        assert_eq!(agents[0].description, "Reviews a diff.");
        assert_eq!(
            agents[0].tools.as_deref(),
            Some(["client_file_read".to_string()].as_slice())
        );
        assert_eq!(agents[0].prompt.trim(), "You are a reviewer.");
    }

    /// `tools` absent means "inherit what the parent can see", which is
    /// a different thing from an empty list.
    #[test]
    fn an_omitted_tools_list_is_none_not_empty() {
        let d = tempfile::tempdir().unwrap();
        write(d.path(), "helper.md", "---\ndescription: Thinks.\n---\nThink.\n");

        let agents = load_agents_dir(d.path());
        assert_eq!(agents[0].tools, None);
    }

    /// An empty list is a valid definition: an agent with no tools
    /// answers from its prompt alone, which is enough for a summary or
    /// a judgement.
    #[test]
    fn an_empty_tools_list_is_kept_as_empty() {
        let d = tempfile::tempdir().unwrap();
        write(d.path(), "judge.md", "---\ndescription: Judges.\ntools: []\n---\nJudge.\n");

        let agents = load_agents_dir(d.path());
        assert_eq!(agents[0].tools.as_deref(), Some([].as_slice()));
    }

    /// One broken file must not take the others with it — the same
    /// rule `load_heartbeat_dir` follows.
    #[test]
    fn a_broken_definition_does_not_hide_the_others() {
        let d = tempfile::tempdir().unwrap();
        write(d.path(), "good.md", "---\ndescription: Fine.\n---\nFine.\n");
        write(d.path(), "no-frontmatter.md", "just a body\n");
        write(d.path(), "bad-yaml.md", "---\ndescription: [unclosed\n---\nx\n");

        let names: Vec<&str> = load_agents_dir(d.path())
            .iter()
            .map(|a| a.name.as_str())
            .collect();
        assert_eq!(names, vec!["good"]);
    }

    /// A description is what the parent model reads to decide whether
    /// to delegate. Without one the agent can never be chosen, so the
    /// definition is useless rather than merely incomplete.
    #[test]
    fn a_definition_without_a_description_is_skipped() {
        let d = tempfile::tempdir().unwrap();
        write(d.path(), "mystery.md", "---\ntools: []\n---\nHello.\n");
        assert!(load_agents_dir(d.path()).is_empty());
    }

    #[test]
    fn a_missing_directory_is_no_agents_rather_than_an_error() {
        let d = tempfile::tempdir().unwrap();
        assert!(load_agents_dir(&d.path().join("nope")).is_empty());
    }
}
```

- [ ] **Step 2: テストが落ちることを確認**

Run: `cargo test -p sapphire-agent agents`
Expected: FAIL — `load_agents_dir` が未定義。

- [ ] **Step 3: 実装する**

```rust
//! Subagent definitions, loaded from `<workspace>/agents/*.md`.
//!
//! Same shape as `<workspace>/heartbeat/*.md`: YAML frontmatter for the
//! metadata, the body for the prompt. Reusing that convention rather
//! than inventing a second one is the whole reason for the file layout.
//!
//! A definition's `description` is load-bearing in a way the others are
//! not: it is what the parent model reads to decide whether to delegate,
//! and it is the only thing it sees before choosing.

use serde::Deserialize;
use std::path::Path;
use tracing::warn;

#[derive(Debug, Clone, Deserialize)]
struct AgentMeta {
    description: String,
    #[serde(default)]
    tools: Option<Vec<String>>,
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
}

/// Load every definition under `dir`, skipping the ones that cannot be
/// read. A missing directory is no agents, not an error: an operator
/// who has not created any is in a normal state.
pub fn load_agents_dir(dir: &Path) -> Vec<AgentDef> {
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
                warn!("failed to read agent definition {}: {e}", path.display());
                continue;
            }
        };
        match parse_agent(name.to_string(), &raw) {
            Some(a) => out.push(a),
            None => warn!(
                "agent definition {} skipped (no/invalid frontmatter, or no description)",
                path.display()
            ),
        }
    }
    out.sort_by(|a, b| a.name.cmp(&b.name));
    out
}

fn parse_agent(name: String, raw: &str) -> Option<AgentDef> {
    let (fm, body) = crate::frontmatter::split(raw)?;
    let meta: AgentMeta = match serde_yaml::from_str(fm) {
        Ok(m) => m,
        Err(e) => {
            warn!("agent {name}: yaml parse error: {e}");
            return None;
        }
    };
    Some(AgentDef {
        name,
        description: meta.description,
        tools: meta.tools,
        prompt: body.trim_start_matches(['\n', '\r']).to_string(),
    })
}
```

`src/main.rs` に `mod agents;` を足す。

**`load_agents_dir` の結果を名前順に並べるのは意図的である** — ディレクトリの
走査順は OS 依存で、`subagent` の説明文がプロセスごとに変わるとモデルへの
入力が不安定になる。

- [ ] **Step 4: テストが通ることを確認**

Run: `cargo test -p sapphire-agent agents`
Expected: PASS（6テスト）。

- [ ] **Step 5: CI 形式の lint とコミット**

```bash
cargo fmt --all
cargo clippy --workspace -- -D warnings
git checkout -- Cargo.lock
git add src/agents.rs src/main.rs
git commit -m "feat(agents): load subagent definitions from the workspace"
```

---

### Task 3: `subagent` ツール

**Files:**
- Create: `src/tools/subagent.rs`
- Modify: `src/tools/mod.rs`
- Modify: `src/serve/mod.rs`（task_local でループを渡す）
- Test: `src/tools/subagent.rs` の `mod tests`

**Interfaces:**
- Consumes: Task 1 の `TurnLoop` / `TurnPersistence`、Task 2 の `AgentDef` / `load_agents_dir`
- Produces: `SubagentTool`（`Tool`、`ToolKind::Other`）

#### ツールがループに届く方法

`Tool::execute` は `ServeState` も `TurnHost` も持っていない。#198 が同じ問題を
`tokio::task_local` で解いており（`crate::tools::acp_client`）、**同じ乗り物を使う。**

`run_llm_turn` がツール実行を包むところで、サブエージェントに必要なもの
（`Arc<ServeState>`、`Arc<dyn Provider>`、`Arc<dyn TurnHost>`、可視ツール仕様、
`timer_origin`）をひとまとめにした値をスコープする。`subagent` ツールはそれを読む。

**サブエージェントは `ToolSet::execute` の中で走るので、`scope_acp_client` と
`scope_timer_origin` は既に張られている** — クライアント側ツールが何もしなくても動く。

- [ ] **Step 1: 失敗するテストを書く**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    fn defs() -> Vec<crate::agents::AgentDef> {
        vec![crate::agents::AgentDef {
            name: "reviewer".to_string(),
            description: "Reviews a diff.".to_string(),
            tools: Some(vec!["client_file_read".to_string()]),
            prompt: "You are a reviewer.".to_string(),
        }]
    }

    /// The description is the parent model's only basis for choosing,
    /// so every agent's own description has to reach it.
    #[test]
    fn the_tool_description_lists_every_agent() {
        let spec = SubagentTool::new(defs()).spec().clone();
        assert!(spec.description.contains("reviewer"), "{}", spec.description);
        assert!(
            spec.description.contains("Reviews a diff."),
            "{}",
            spec.description
        );
    }

    #[test]
    fn the_kind_is_other() {
        assert_eq!(SubagentTool::new(defs()).kind(), ToolKind::Other);
    }

    /// A name the operator never defined is a mistake the parent can
    /// recover from — list what exists rather than just refusing.
    #[tokio::test]
    async fn an_unknown_agent_names_the_ones_that_exist() {
        let err = SubagentTool::new(defs())
            .execute(&serde_json::json!({"agent": "nope", "prompt": "x"}))
            .await
            .unwrap_err()
            .to_string();
        assert!(err.contains("reviewer"), "{err}");
    }

    /// Outside a turn there is nothing to delegate with. Refusing here
    /// is what keeps the tool honest on any path that is not a live
    /// turn.
    #[tokio::test]
    async fn delegating_outside_a_turn_refuses() {
        let err = SubagentTool::new(defs())
            .execute(&serde_json::json!({"agent": "reviewer", "prompt": "x"}))
            .await
            .unwrap_err()
            .to_string();
        assert!(err.contains("no turn"), "{err}");
    }

    /// The point of the feature: a subagent's system prompt is its own
    /// definition and nothing else. A `SOUL.md` in the workspace must
    /// not reach it.
    #[test]
    fn the_system_prompt_is_the_definition_body_plus_the_date() {
        let sys = subagent_system_prompt(&defs()[0]);
        assert!(sys.contains("You are a reviewer."));
        assert!(
            sys.contains("Current Date and Time"),
            "the date is the one inherited fact: {sys}"
        );
        for absent in ["# Soul", "# Identity", "# User", "# Agent Instructions"] {
            assert!(!sys.contains(absent), "{absent} must not be inherited: {sys}");
        }
    }

    /// Depth is capped at one by construction: a subagent cannot see
    /// the tool that would let it delegate again.
    #[test]
    fn a_subagents_tool_list_never_contains_subagent() {
        let parent_visible = [
            spec_named("client_file_read"),
            spec_named(SUBAGENT_TOOL_NAME),
        ];
        let inherited = subagent_tool_specs(&defs()[0], &parent_visible);
        assert!(!inherited.iter().any(|s| s.name == SUBAGENT_TOOL_NAME));

        let unrestricted = crate::agents::AgentDef {
            tools: None,
            ..defs()[0].clone()
        };
        let inherited = subagent_tool_specs(&unrestricted, &parent_visible);
        assert!(!inherited.iter().any(|s| s.name == SUBAGENT_TOOL_NAME));
        assert!(inherited.iter().any(|s| s.name == "client_file_read"));
    }

    /// Even when the definition asks for it by name.
    #[test]
    fn a_definition_cannot_grant_itself_subagent() {
        let greedy = crate::agents::AgentDef {
            tools: Some(vec![SUBAGENT_TOOL_NAME.to_string(), "client_file_read".to_string()]),
            ..defs()[0].clone()
        };
        let parent_visible = [spec_named("client_file_read"), spec_named(SUBAGENT_TOOL_NAME)];
        let inherited = subagent_tool_specs(&greedy, &parent_visible);
        assert!(!inherited.iter().any(|s| s.name == SUBAGENT_TOOL_NAME));
    }

    /// An empty list is a definition, not an omission.
    #[test]
    fn an_empty_tools_list_yields_no_tools() {
        let toolless = crate::agents::AgentDef {
            tools: Some(vec![]),
            ..defs()[0].clone()
        };
        let parent_visible = [spec_named("client_file_read")];
        assert!(subagent_tool_specs(&toolless, &parent_visible).is_empty());
    }

    fn spec_named(name: &str) -> crate::provider::ToolSpec {
        crate::provider::ToolSpec {
            name: name.to_string().into(),
            description: "…".into(),
            input_schema: serde_json::json!({"type": "object"}),
        }
    }
}
```

`ToolSpec.name` は `Cow<'static, str>` なので、比較は `s.name.as_ref() != SUBAGENT_TOOL_NAME`
の形にすること。`Cow` と `&str` の直接比較に頼らない。

そして `src/serve/mod.rs` の `mod tests` に、**この機能の2つの核心的な性質**を足す。
どちらも spec のテスト節が名指ししているもので、上の単体テストでは押さえられない。

```rust
    /// A subagent runs under the parent's `Origin`, so it cannot do
    /// what the parent was refused. Anything else would make "ask a
    /// subagent" a way around the permission gate.
    #[tokio::test]
    async fn a_subagent_is_judged_by_the_parents_origin() {
        // 親を Origin::Channel の TurnHost で駆動し、サブエージェントに
        // Execute のツール（既存のテスト用 RiskyTool）を呼ばせる応答を
        // scripted provider に仕込む。
        //
        // 期待: そのツールは実行されない（RiskyTool の ran フラグが false）。
        // Origin::Channel は Execute を拒否するので、委譲を経由しても
        // 通ってはいけない。
        //
        // 既存の `the_channel_gate_refuses_shell_but_keeps_the_chat_tools`
        // （src/agent.rs）と、serve 側の scripted-provider テストの
        // 組み立てを読んでから書くこと。
    }

    /// The isolation, asserted rather than assumed: what the subagent
    /// said to itself must not reach the parent's history or its store.
    /// Only the final answer comes back, as the tool's result.
    #[tokio::test]
    async fn a_subagents_conversation_does_not_reach_the_parent() {
        // ACP セッションで、サブエージェントに固有の文字列を含む中間発話を
        // させてから最終回答を返させる。
        //
        // 期待:
        //   - state.sessions の親の履歴に、その中間発話の文字列が無い
        //   - acp_session_store.history(親) にも無い
        //   - 親の履歴には subagent の ToolResult があり、最終回答を含む
    }
```

**この2本は筋書きで書いてある。** 上の単体テストと違い、scripted provider と
`TurnHost` の組み立てが要り、その形は Task 1 の抽出結果と既存のテスト補助に
依存するため、実装者が現物に合わせて書く。**確かめるべき性質は
それぞれ3行のコメントに全部書いてある** — 迷ったら性質のほうを優先すること。

- [ ] **Step 2: テストが落ちることを確認**

Run: `cargo test -p sapphire-agent subagent`
Expected: FAIL — `SubagentTool` が未定義。

- [ ] **Step 3: 純粋な部分を先に実装する**

`src/tools/subagent.rs`。**先にテストできる形にしてから、ループを繋ぐ。**

```rust
pub(crate) const SUBAGENT_TOOL_NAME: &str = "subagent";

/// A subagent's whole system prompt.
///
/// The definition's body, plus the date — and nothing else. Not the
/// workspace files (`SOUL.md`, `IDENTITY.md`, `USER.md`, `AGENTS.md`,
/// `TOOLS.md`), not memory, not the day's cross-session digest, not the
/// room metadata, not the configured base prompt.
///
/// Dropping those is not an oversight, it is the feature: the main
/// agent carries them deliberately — it is someone to work *with* — and
/// a code review does not need yesterday's conversation. Inheriting
/// them by default would defeat the reason this exists.
///
/// The date is the one exception, because an agent that does not know
/// today's date cannot use a tool that writes one, and that is a fact
/// rather than a personality.
pub(crate) fn subagent_system_prompt(def: &crate::agents::AgentDef) -> String {
    let now = chrono::Local::now();
    format!(
        "{}\n\n# Current Date and Time\n\n{} ({})",
        def.prompt,
        now.format("%Y-%m-%d %H:%M:%S %z"),
        now.format("%A")
    )
}

/// The tools a subagent may use.
///
/// `None` in the definition inherits the parent's visible set; a list
/// selects from it. Either way `subagent` itself is removed, which is
/// what caps delegation depth at one — a bound by construction rather
/// than a counter that has to be threaded through.
pub(crate) fn subagent_tool_specs(
    def: &crate::agents::AgentDef,
    parent_visible: &[crate::provider::ToolSpec],
) -> Vec<crate::provider::ToolSpec> {
    parent_visible
        .iter()
        .filter(|s| s.name != SUBAGENT_TOOL_NAME)
        .filter(|s| match &def.tools {
            Some(allowed) => allowed.iter().any(|a| a == s.name.as_ref()),
            None => true,
        })
        .cloned()
        .collect()
}
```

ツール本体の `spec()` は、`description` に定義を列挙する:

```
Delegate a task to a specialised agent. The agent runs with its own
system prompt and its own conversation, and only its final answer comes
back — use this to keep a large investigation out of this conversation.

Available agents:
- reviewer: Reviews a diff.
- …
```

- [ ] **Step 4: ターンの文脈を task_local で渡す**

`src/serve/mod.rs` に、`crate::tools::acp_client` と同じ形で:

```rust
/// What a delegating tool needs to run a nested conversation.
///
/// Carried the same way the ACP client handle is (`tools::acp_client`),
/// and for the same reason: `Tool::execute` receives only its JSON
/// input, and threading a turn through the `Tool` trait would touch
/// every tool for the benefit of one.
pub(crate) struct TurnContext {
    pub state: Arc<ServeState>,
    pub provider: Arc<dyn Provider>,
    pub progress: Arc<dyn TurnHost>,
    pub visible_specs: Vec<ToolSpec>,
    pub timer_origin: Option<crate::timer::TimerOrigin>,
}
```

`scope_turn_context` / `current_turn_context` を `scope_acp_client` に倣って書き、
`run_llm_turn` のツール実行を包むところで、既存の2つのスコープと**入れ子にする**。

**`progress` は親のものをそのまま渡す。** これが承認を同じゲート・同じ
クライアントに通す仕組みで、別のものを渡せば迂回路になる。

- [ ] **Step 5: 委譲を実装する**

```rust
    async fn execute(&self, input: &serde_json::Value) -> anyhow::Result<String> {
        let name = input["agent"].as_str().context("missing 'agent'")?;
        let prompt = input["prompt"].as_str().context("missing 'prompt'")?;

        let Some(def) = self.agents.iter().find(|a| a.name == name) else {
            // Recoverable: the parent picked a name that does not
            // exist, and can pick again if it is told what does.
            let known: Vec<&str> = self.agents.iter().map(|a| a.name.as_str()).collect();
            anyhow::bail!("no agent named '{name}'. Available: {}", known.join(", "));
        };

        // Only a live turn has a model, a permission host and a visible
        // tool set to lend. Nothing else can delegate.
        let ctx = crate::serve::current_turn_context()
            .context("no turn to delegate from")?;

        let system = subagent_system_prompt(def);
        let specs = subagent_tool_specs(def, &ctx.visible_specs);
        let mut history = vec![crate::provider::ChatMessage::user(prompt)];

        let (text, stop) = crate::serve::TurnLoop {
            state: &ctx.state,
            provider: &ctx.provider,
            system: Some(&system),
            tool_specs: &specs,
            // The parent's host, deliberately: a permission request
            // from a subagent must reach the same person, judged by the
            // same origin. A different host here would make delegation
            // a way around the gate.
            progress: &ctx.progress,
            timer_origin: ctx.timer_origin.clone(),
            // No session behind it. The conversation exists for the
            // length of this call and is then dropped — that is what
            // "context isolation" means here.
            persistence: None,
        }
        .run(&mut history)
        .await;

        Ok(match stop {
            crate::serve::TurnStop::BudgetExhausted { partial_text } => format!(
                "[the subagent used its whole tool budget without finishing]\n\n{partial_text}"
            ),
            _ => text.unwrap_or_else(|| {
                "[the subagent produced no answer]".to_string()
            }),
        })
    }
```

`TurnStop` の variant 名と `BudgetExhausted` のフィールドは現物に合わせること
（`src/serve/mod.rs`）。`ctx.timer_origin` が `Clone` でなければ、
`TurnContext` に持たせる時点で `Clone` できる形にする。

- [ ] **Step 6: 登録する**

`build_default_tools`（`src/tools/mod.rs`）で、定義が1つ以上あるときだけ登録する。
1つも無ければツール自体を出さない — 選べる相手がいないツールを見せる理由が無い。

定義の読み込みは `src/main.rs` の起動時、ワークスペースのディレクトリから
`<workspace>/agents`。

- [ ] **Step 7: テストが通ることを確認**

Run: `cargo test -p sapphire-agent`
Expected: PASS。

- [ ] **Step 8: CI 形式の lint とコミット**

```bash
cargo fmt --all
cargo clippy --workspace -- -D warnings
git checkout -- Cargo.lock
git add src/tools/subagent.rs src/tools/mod.rs src/serve/mod.rs src/main.rs
git commit -m "feat(tools): delegate a task to a specialised subagent"
```

---

### Task 4: ドキュメントと全体確認

**Files:**
- Modify: `README.md`
- Modify: `config.example.toml`（該当すれば）
- Modify: `docs/superpowers/specs/2026-09-02-subagents-design.md`

- [ ] **Step 1: README を書く**

- 定義の置き場（`<workspace>/agents/<name>.md`）と、フロントマターの3項目。
  例を1つ載せる。
- **サブエージェントが継承しないもの**を名指しする（`SOUL.md` などのワークスペース
  ファイル、メモリ、今日のダイジェスト）。**そして、それを落とせることが目的である**と
  書く。ここを書き落とすと、次に読む人が「バグでは」と思う。
- 継承するのは日時だけであること。
- `tools` を省略すると親の可視集合、空リストはツール無し、どちらも有効。
- **`subagent` は `ToolKind::Other` なので Matrix / Discord からは呼べない** —
  実質 ACP 専用であること。
- 委譲は深さ1で、サブエージェントはさらに委譲できないこと。
- 承認はサブエージェントの呼び出しでも普通に出ること。
- プロジェクト規約（`CLAUDE.md`）の共有は [#199](https://github.com/fluo10/sapphire-agent/issues/199) 待ちであること。

- [ ] **Step 2: spec に実装時の訂正を追記する**

spec 本文は書き換えず、末尾に `## 実装時の訂正` を新設する。
**訂正が無ければ「無し」と1行書く** — 節ごと省かないこと。後から読む人が
「確認されなかった」のか「確認して何も無かった」のかを区別できる。

- [ ] **Step 3: ワークスペース全体の確認**

```bash
cargo test --workspace
cargo fmt --all --check
cargo clippy --workspace -- -D warnings
```

`cli_device::tests::add_turns_expires_in_into_an_absolute_time` は既知の
実時間フレーキー（[#197](https://github.com/fluo10/sapphire-agent/issues/197)）。
それだけが落ちたら単独で走らせ直して確認し、報告に書くこと。

- [ ] **Step 4: コミット**

```bash
git checkout -- Cargo.lock
git add README.md docs/superpowers/specs/2026-09-02-subagents-design.md
git commit -m "docs: describe subagents and what they deliberately do not inherit"
```
