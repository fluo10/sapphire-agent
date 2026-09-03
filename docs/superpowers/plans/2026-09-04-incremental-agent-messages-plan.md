# Incremental Agent Messages Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** エージェントが作業を続けたまま発言できるようにする — 中間テキストをその場で届け、10ラウンド固定の上限を経路ごとの設定に置き換える。

**Architecture:** `TurnHost` に既定 no-op の `message_chunk` を足し、`TurnLoop::run` がラウンドごとにテキストを渡す。ACP はこれに全面移行して終了時の一括送出をやめる。ラウンド上限はホストが返す `RoundBudget` を config に照らして解決する。Matrix/Discord は `TurnHost` を通らないので `agent.rs` のループ内で直接送信する。

**Tech Stack:** Rust 2024, tokio, `async_trait`, `agent-client-protocol` 2.0.0, serde/toml

**Spec:** `docs/superpowers/specs/2026-09-04-incremental-agent-messages-design.md`

**Branch:** `feat/incremental-agent-messages`（基点 `origin/main` = `a6cacdc`）

## Global Constraints

- ワークスペースは仮想マニフェスト。`default-members = ["server"]` なので、ルートでの `cargo test` は agent クレートに限定される。テストは常に `cargo test -p sapphire-agent` で回す
- コンベンショナルコミットのスコープを使う。agent 内部の変更は無スコープか `(serve)` / `(agent)` / `(tools)` / `(acp)`。`cliff.toml` の変更は不要（agent 内部スコープは素通りする）
- `server/config.example.toml` は `server/src/config.rs:2063` のテストでパースが検証される。設定を足したらここも直す
- **新しい `TurnHost` メソッドは必ず既定実装を持たせる** — さもないと `SseProgress` / `NullProgress` / テスト用ホストが軒並み壊れる
- 本番の `TurnHost` 実装者は4つだけ: `SseProgress`、`NullProgress`（`serve/mod.rs`）、`AcpProgress`（`serve/acp.rs`）、`ParentHostSansTurnError`（`tools/subagent.rs`）。`ChannelHost` と `AskExceptRiskyHost` は `mod tests` 内のテスト専用
- コメントは既存コードの密度に合わせる。このリポジトリは「なぜそうしたか」を長く書く慣習がある

---

## File Structure

| ファイル | 役割 | 変更内容 |
|---|---|---|
| `server/src/config.rs` | 設定スキーマ | `ToolRounds` を新設し `ToolsConfig` に生やす |
| `server/config.example.toml` | 設定の説明 | `[tools.tool_rounds]` の節を追加 |
| `server/src/serve/mod.rs` | 共有ターン実行器 | `RoundBudget`、`TurnHost::message_chunk` / `round_budget`、`TurnLoop::run` の上限解決とテキスト通知 |
| `server/src/serve/acp.rs` | ACP トランスポート | `AcpProgress` が2つを実装、終了時チャンク送出を削除 |
| `server/src/tools/subagent.rs` | サブエージェント | `ParentHostSansTurnError` が2つを非委譲で実装 |
| `server/src/agent.rs` | Matrix/Discord ループ | ラウンドごとの送信、const 廃止 |
| `server/templates/workspace/AGENTS.md` | モデルへの規約 | ターンの終わり方を明示する一節 |

タスクは依存の順に並べる。Task 1（設定）と Task 2（トレイト）が土台で、Task 3〜6 がそれを使う。Task 7 は独立。

---

### Task 1: `[tools.tool_rounds]` 設定を足す

ラウンド上限を config に露出させる。この時点ではまだ誰も読まない — スキーマとデフォルト値だけを確定させ、次のタスクが参照できるようにする。

**Files:**
- Modify: `server/src/config.rs`（`ToolsConfig` は 754 行付近）
- Modify: `server/config.example.toml`（`[tools]` の説明は 283 行付近、例は 343 行付近）
- Test: `server/src/config.rs`（同ファイル内の `mod tests`）

**Interfaces:**
- Consumes: なし
- Produces:
  - `pub struct ToolRounds { pub interactive: usize, pub unattended: usize }`（`Debug + Clone + PartialEq + Eq + Deserialize + Serialize`）
  - `impl Default for ToolRounds` → `{ interactive: 0, unattended: 25 }`
  - `ToolsConfig::tool_rounds: ToolRounds`（`#[serde(default)]`）
  - `ToolRounds::limit` は **Task 2 で足す**（`RoundBudget` がまだ無いため）

- [ ] **Step 1: 失敗するテストを書く**

`server/src/config.rs` の `mod tests` の末尾に追加する。

```rust
/// 既定値は「中断できる経路は無制限、できない経路は有限」。`0` が
/// 無制限を意味することと、省略時にその既定が入ることを両方留める。
#[test]
fn tool_rounds_default_to_unbounded_interactive_and_a_bounded_rest() {
    let rounds = crate::config::ToolRounds::default();
    assert_eq!(rounds.interactive, 0, "ACP は既定で無制限");
    assert_eq!(rounds.unattended, 25);
}

/// `[tools]` を書きつつ `tool_rounds` を省いた設定が、既定値で埋まる。
/// `#[serde(default)]` が無いとここで落ちる。
#[test]
fn tools_config_without_tool_rounds_falls_back_to_the_default() {
    let tools: crate::config::ToolsConfig =
        toml::from_str("tavily_api_key = \"tvly-x\"").unwrap();
    assert_eq!(tools.tool_rounds, crate::config::ToolRounds::default());
}

/// 書かれた値がそのまま読める。
#[test]
fn tool_rounds_round_trip_from_toml() {
    let tools: crate::config::ToolsConfig =
        toml::from_str("[tool_rounds]\ninteractive = 40\nunattended = 3").unwrap();
    assert_eq!(tools.tool_rounds.interactive, 40);
    assert_eq!(tools.tool_rounds.unattended, 3);
}
```

- [ ] **Step 2: 落ちることを確認する**

Run: `cargo test -p sapphire-agent tool_rounds`
Expected: FAIL — `ToolRounds` が存在せず `cannot find struct` でコンパイルエラー

- [ ] **Step 3: 最小の実装を書く**

`server/src/config.rs` の `ToolsConfig`（754 行付近）にフィールドを足す。

```rust
    /// How many tool rounds one turn may spend. See [`ToolRounds`].
    #[serde(default)]
    pub tool_rounds: ToolRounds,
```

`HostAccess` の定義の直後に `ToolRounds` を置く。

```rust
/// How many tool rounds a single turn may spend, per route.
///
/// Split in two because the routes are not alike in the one way that
/// matters: whether a human can stop a turn that has gone wrong. ACP has
/// `session/cancel` — the editor's Escape — so a turn there can run as
/// long as it is useful and be cut off the moment it is not. Matrix,
/// Discord, `/rpc`, A2A and the heartbeat have no such control, and the
/// heartbeat in particular runs with nobody watching, so a runaway loop
/// there spends money until the process is restarted. A single number
/// would have to be chosen for the worst of those, which is what the
/// hard-coded ten effectively did.
///
/// Context growth is not what these bound: `maybe_compress` runs every
/// round and trims the history on its own. What they bound is spend.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub struct ToolRounds {
    /// Routes with a way to cancel a turn in flight. ACP only.
    #[serde(default = "ToolRounds::default_interactive")]
    pub interactive: usize,
    /// Everything else: Matrix, Discord, `/rpc` (desktop chat and voice),
    /// A2A, heartbeat, and every subagent regardless of its parent.
    #[serde(default = "ToolRounds::default_unattended")]
    pub unattended: usize,
}

impl ToolRounds {
    /// `0` means unbounded, on either side.
    fn default_interactive() -> usize {
        0
    }
    fn default_unattended() -> usize {
        25
    }
}

impl Default for ToolRounds {
    fn default() -> Self {
        Self {
            interactive: Self::default_interactive(),
            unattended: Self::default_unattended(),
        }
    }
}
```

- [ ] **Step 4: テストが通ることを確認する**

Run: `cargo test -p sapphire-agent tool_rounds`
Expected: PASS（3 件）

- [ ] **Step 5: `config.example.toml` を更新する**

`server/config.example.toml` の `[tools]` 説明ブロック（283 行付近、`#   - \`tavily_api_key\`` の項の直後）に箇条書きを1つ足す。

```
#   - `[tools.tool_rounds]` — how many tool rounds one turn may spend,
#     split by whether the route can cancel a turn in flight. `interactive`
#     covers ACP, which has `session/cancel` (the editor's Escape), and
#     defaults to `0` — unbounded — so a long piece of delegated work runs
#     to its end instead of stopping every ten rounds. `unattended` covers
#     Matrix, Discord, `/rpc`, `/a2a` and the heartbeat, none of which can
#     be interrupted, and defaults to 25. `0` means unbounded on either
#     side; subagents are always judged `unattended`, whatever their
#     parent is. These bound spend, not context — `[compression]` is what
#     keeps the history in the window.
```

例のブロック（343 行付近、`# [tools.host_access]` の2行の直後）にも足す。

```
# [tools.tool_rounds]
# interactive = 0                           # ACP; 0 = unbounded
# unattended  = 25                          # everything else
#
```

- [ ] **Step 6: 例がパースすることを確認する**

Run: `cargo test -p sapphire-agent config_example`
Expected: PASS

- [ ] **Step 7: コミット**

```bash
git add server/src/config.rs server/config.example.toml
git commit -m "feat(config): make the per-turn tool-round budget configurable per route

The cap was a hard-coded ten in two places, chosen for the worst route
— the ones nobody can interrupt. ACP has session/cancel, so it does not
need to be judged by that, and a long piece of delegated work should not
stop every ten rounds.

Nothing reads this yet; the loops move over in the next commits."
```

---

### Task 2: `TurnHost` に `message_chunk` と `round_budget` を足す

トレイトに2つのフックを足し、`TurnLoop::run` がそれを使うようにする。ACP はまだ実装しないので、この時点で外から見える挙動は「上限が config から来るようになる」だけ。

**Files:**
- Modify: `server/src/serve/mod.rs`（`MAX_TOOL_ROUNDS` は 44 行、`TurnHost` は 2003 行付近、`TurnStop` は 2157 行付近、`TurnLoop::run` は 2388 行付近、`for_test_scripted` は 3514 行付近、`build_for_test` は 3543 行付近）
- Modify: `server/src/config.rs`（`impl ToolRounds`）
- Test: `server/src/serve/mod.rs`（同ファイル内の `mod tests`）

**Interfaces:**
- Consumes: `ToolRounds`（Task 1）
- Produces:
  - `pub(crate) enum RoundBudget { Interactive, Unattended }`（`Debug + Clone + Copy + PartialEq + Eq`）
  - `TurnHost::message_chunk(&self, text: &str)` — `async`、既定 no-op
  - `TurnHost::round_budget(&self) -> RoundBudget` — 非 `async`、既定 `Unattended`
  - `ToolRounds::limit(&self, budget: RoundBudget) -> Option<usize>` — `0` を `None` に写す
  - `ServeState::for_test_scripted_with_rounds(acp_enabled: bool, responses: Vec<ChatResponse>, rounds: ToolRounds) -> Arc<Self>`

- [ ] **Step 1: 失敗するテストを書く**

`server/src/serve/mod.rs` の `mod tests` の末尾に追加する。

```rust
/// `0` は無制限、それ以外はその数。二段階の写像を一箇所に留める。
#[test]
fn a_zero_budget_means_unbounded() {
    use crate::config::ToolRounds;
    let rounds = ToolRounds {
        interactive: 0,
        unattended: 25,
    };
    assert_eq!(rounds.limit(RoundBudget::Interactive), None);
    assert_eq!(rounds.limit(RoundBudget::Unattended), Some(25));
}

/// 中間テキストは溜められるのではなく、そのラウンドで渡される。ここが
/// この変更の全部である。ツールを呼ばない最後の応答のテキストも同じ口を
/// 通るので、ホストから見たテキストの並びは会話そのものになる。
#[tokio::test]
async fn text_reaches_the_host_round_by_round() {
    #[derive(Default)]
    struct ChunkRecorder {
        chunks: std::sync::Mutex<Vec<String>>,
    }
    #[async_trait::async_trait]
    impl TurnHost for ChunkRecorder {
        async fn tool_start(&self, _id: &str, _name: &str) {}
        async fn tool_end(&self, _id: &str, _name: &str) {}
        async fn turn_error(&self, _message: &str) {}
        async fn message_chunk(&self, text: &str) {
            self.chunks.lock().unwrap().push(text.to_string());
        }
    }

    let state = ServeState::for_test_scripted(
        false,
        vec![
            crate::provider::ChatResponse {
                text: Some("looking now".to_string()),
                tool_calls: vec![crate::provider::ToolCall {
                    id: "call-1".to_string(),
                    name: "echo".to_string(),
                    input: json!({ "text": "ping" }),
                }],
                stop_reason: None,
            },
            crate::provider::ChatResponse {
                text: Some("found it".to_string()),
                tool_calls: Vec::new(),
                stop_reason: None,
            },
        ],
    );
    let host = Arc::new(ChunkRecorder::default());
    let outcome = run_llm_turn(
        Arc::clone(&state),
        "s-chunks".to_string(),
        ChatMessage::user("go"),
        Arc::clone(&host) as Arc<dyn TurnHost>,
        None,
    )
    .await;

    assert_eq!(
        *host.chunks.lock().unwrap(),
        vec!["looking now".to_string(), "found it".to_string()]
    );
    // `outcome.text` は据え置き。これを読む `/rpc`・A2A・音声が壊れない
    // ことが、この変更が既定 no-op で足りる理由である。
    assert_eq!(outcome.text.as_deref(), Some("looking now\n\nfound it"));
}

/// 空のテキストは渡さない。ツールだけ呼ぶラウンドで空メッセージが
/// 流れると、チャンネル経路では空の吹き出しになる。
#[tokio::test]
async fn an_empty_text_is_not_handed_to_the_host() {
    #[derive(Default)]
    struct ChunkRecorder {
        chunks: std::sync::Mutex<Vec<String>>,
    }
    #[async_trait::async_trait]
    impl TurnHost for ChunkRecorder {
        async fn tool_start(&self, _id: &str, _name: &str) {}
        async fn tool_end(&self, _id: &str, _name: &str) {}
        async fn turn_error(&self, _message: &str) {}
        async fn message_chunk(&self, text: &str) {
            self.chunks.lock().unwrap().push(text.to_string());
        }
    }

    let state = ServeState::for_test_scripted(
        false,
        vec![
            crate::provider::ChatResponse {
                text: None,
                tool_calls: vec![crate::provider::ToolCall {
                    id: "call-1".to_string(),
                    name: "echo".to_string(),
                    input: json!({ "text": "ping" }),
                }],
                stop_reason: None,
            },
            crate::provider::ChatResponse {
                text: Some("done".to_string()),
                tool_calls: Vec::new(),
                stop_reason: None,
            },
        ],
    );
    let host = Arc::new(ChunkRecorder::default());
    let _ = run_llm_turn(
        Arc::clone(&state),
        "s-empty".to_string(),
        ChatMessage::user("go"),
        Arc::clone(&host) as Arc<dyn TurnHost>,
        None,
    )
    .await;

    assert_eq!(*host.chunks.lock().unwrap(), vec!["done".to_string()]);
}

/// 既定 `Unattended` のホストは、設定した数で打ち切られる。上限が
/// ハードコードではなく config から来ていることを、10 以外の数で示す。
#[tokio::test]
async fn an_unattended_host_stops_at_the_configured_budget() {
    use crate::config::ToolRounds;

    let script: Vec<crate::provider::ChatResponse> = (0..3)
        .map(|i| crate::provider::ChatResponse {
            text: Some(format!("step {i}")),
            tool_calls: vec![crate::provider::ToolCall {
                id: format!("call-{i}"),
                name: "echo".to_string(),
                input: json!({ "text": "ping" }),
            }],
            stop_reason: None,
        })
        .collect();
    let state = ServeState::for_test_scripted_with_rounds(
        false,
        script,
        ToolRounds {
            interactive: 0,
            unattended: 3,
        },
    );
    let outcome = run_llm_turn(
        Arc::clone(&state),
        "s-budget".to_string(),
        ChatMessage::user("go"),
        Arc::new(NullProgress) as Arc<dyn TurnHost>,
        None,
    )
    .await;

    assert!(
        matches!(outcome.stop, TurnStop::BudgetExhausted { .. }),
        "3 ラウンド使い切ったら打ち切られる"
    );
}

/// `interactive = 0` のホストは、既定の10ラウンドを超えても回り続ける。
/// スクリプトは12本 — 11本目に到達する時点で、旧来の上限は破れている。
#[tokio::test]
async fn an_interactive_host_runs_past_the_old_hard_coded_ten() {
    use crate::config::ToolRounds;

    struct InteractiveHost;
    #[async_trait::async_trait]
    impl TurnHost for InteractiveHost {
        async fn tool_start(&self, _id: &str, _name: &str) {}
        async fn tool_end(&self, _id: &str, _name: &str) {}
        async fn turn_error(&self, _message: &str) {}
        fn round_budget(&self) -> RoundBudget {
            RoundBudget::Interactive
        }
    }

    let mut script: Vec<crate::provider::ChatResponse> = (0..12)
        .map(|i| crate::provider::ChatResponse {
            text: None,
            tool_calls: vec![crate::provider::ToolCall {
                id: format!("call-{i}"),
                name: "echo".to_string(),
                input: json!({ "text": "ping" }),
            }],
            stop_reason: None,
        })
        .collect();
    script.push(crate::provider::ChatResponse {
        text: Some("finished".to_string()),
        tool_calls: Vec::new(),
        stop_reason: None,
    });

    let state = ServeState::for_test_scripted_with_rounds(
        false,
        script,
        ToolRounds {
            interactive: 0,
            unattended: 25,
        },
    );
    let outcome = run_llm_turn(
        Arc::clone(&state),
        "s-unbounded".to_string(),
        ChatMessage::user("go"),
        Arc::new(InteractiveHost) as Arc<dyn TurnHost>,
        None,
    )
    .await;

    assert_eq!(outcome.text.as_deref(), Some("finished"));
}
```

- [ ] **Step 2: 落ちることを確認する**

Run: `cargo test -p sapphire-agent -- a_zero_budget_means_unbounded text_reaches_the_host an_empty_text_is_not_handed an_unattended_host_stops an_interactive_host_runs_past`
Expected: FAIL — `RoundBudget`、`limit`、`message_chunk`、`round_budget`、`for_test_scripted_with_rounds` がいずれも存在せずコンパイルエラー

- [ ] **Step 3: `RoundBudget` と `ToolRounds::limit` を書く**

`server/src/serve/mod.rs` の `TurnStop` の定義の直前（2157 行付近）に置く。

```rust
/// Which round budget a turn is judged by.
///
/// A property of the *route*, not of the request: what separates the two
/// is whether a human can stop a turn that has gone wrong. Only ACP can
/// (`session/cancel`), so only `AcpProgress` returns `Interactive`.
///
/// Deliberately not a number. A host knows which kind of route it is; it
/// does not know what the operator configured, and threading the config
/// through every implementor to let each read the same field would put
/// the same decision in four places. `TurnLoop::run` resolves this
/// against `[tools.tool_rounds]` in one place instead — the same shape
/// `TurnHost::origin` uses for the permission table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RoundBudget {
    /// The turn can be cancelled in flight. ACP only.
    Interactive,
    /// It cannot. Everything else, and every subagent.
    Unattended,
}
```

`server/src/config.rs` の `impl ToolRounds` に足す。

```rust
    /// The cap for `budget`, or `None` when it is unbounded.
    ///
    /// `0` is the spelling for unbounded because a config file has to
    /// say it somehow, and `0` is the one value that is never a
    /// meaningful cap — a turn that may spend no tool rounds at all
    /// cannot call a tool, which is not a thing anyone wants to
    /// configure.
    pub fn limit(&self, budget: crate::serve::RoundBudget) -> Option<usize> {
        let n = match budget {
            crate::serve::RoundBudget::Interactive => self.interactive,
            crate::serve::RoundBudget::Unattended => self.unattended,
        };
        (n != 0).then_some(n)
    }
```

- [ ] **Step 4: トレイトに2つのメソッドを足す**

`server/src/serve/mod.rs` の `trait TurnHost`（2003 行付近）、`turn_error` の宣言の直後に置く。

```rust
    /// One piece of the model's prose, as soon as the round that produced
    /// it is done.
    ///
    /// Default no-op, and that default is load-bearing: `/rpc`, `/a2a` and
    /// the voice pipeline read the turn's whole reply from
    /// `LlmTurnOutcome::text` and are correct as they are. TTS in
    /// particular must not be handed a turn in pieces — it would speak the
    /// narration between tool calls as if it were the answer.
    ///
    /// Called for every non-empty text a round produces, including the
    /// final tool-less one, so a host that implements this sees the
    /// turn's prose in order and needs nothing from `outcome.text`.
    /// `ParentHostSansTurnError` deliberately does *not* forward it — see
    /// its doc.
    async fn message_chunk(&self, _text: &str) {}

    /// Which round budget this turn is judged by. See [`RoundBudget`].
    ///
    /// `Unattended` by default, which is the safe direction: a host that
    /// forgets to answer gets the bounded budget, not the unbounded one.
    fn round_budget(&self) -> RoundBudget {
        RoundBudget::Unattended
    }
```

- [ ] **Step 5: `TurnLoop::run` を書き換える**

`server/src/serve/mod.rs:2388` の `run` で、`let mut round = 0usize;`（2415 行付近）の直前に上限を解決する。

```rust
        // Resolved once per turn, not per round: the config cannot change
        // mid-turn, and reading it here keeps `RoundBudget` a routing
        // question rather than a numeric one. `None` is unbounded — the
        // check below simply never fires.
        let round_limit = state
            .config
            .tools
            .tool_rounds
            .limit(progress.round_budget());
```

上限判定（2416 行付近）を書き換える。

```rust
            if round_limit.is_some_and(|max| round >= max) {
                warn!("Reached max tool rounds ({round})");
                break (
                    None,
                    TurnStop::BudgetExhausted {
                        partial_text: accumulated_text.join("\n\n"),
                    },
                );
            }
```

ツールを呼ばない応答の分岐（2471 行付近）で、`accumulated_text.push` の隣に通知を足す。

```rust
                    if !text.is_empty() {
                        progress.message_chunk(&text).await;
                        accumulated_text.push(text);
                    }
```

ツールを呼ぶ応答の分岐（2486 行付近）も同じく。

```rust
                    if let Some(t) = resp.text.as_ref().filter(|s| !s.is_empty()) {
                        progress.message_chunk(t).await;
                        accumulated_text.push(t.clone());
                    }
```

`MAX_TOOL_ROUNDS`（44 行）はこのファイルからは使われなくなるが、**Task 4 まで消さない** — `acp.rs` のテストがまだ参照しているため、消すとコンパイルが通らない。

- [ ] **Step 6: テスト用コンストラクタを足す**

`server/src/serve/mod.rs` の `for_test_scripted`（3514 行付近）の隣に置く。

```rust
    /// Same as [`Self::for_test_scripted`], with an explicit round budget
    /// so a test can pin the cap instead of depending on the shipped
    /// default. Tests that do not care keep calling `for_test_scripted`.
    pub(crate) fn for_test_scripted_with_rounds(
        acp_enabled: bool,
        responses: Vec<crate::provider::ChatResponse>,
        rounds: crate::config::ToolRounds,
    ) -> Arc<Self> {
        Self::build_for_test_with(acp_enabled, StubProvider::new(responses), rounds)
    }
```

`build_for_test`（3543 行付近）を委譲に変え、本体を `build_for_test_with` に移す。

```rust
    fn build_for_test(acp_enabled: bool, provider: StubProvider) -> Arc<Self> {
        Self::build_for_test_with(acp_enabled, provider, crate::config::ToolRounds::default())
    }

    fn build_for_test_with(
        acp_enabled: bool,
        provider: StubProvider,
        rounds: crate::config::ToolRounds,
    ) -> Arc<Self> {
        // ... 既存の本体をそのまま移す ...
```

移した本体の中、`config.acp = Some(...)` を設定している行（3585 行付近）の隣に足す。

```rust
        config.tools.tool_rounds = rounds;
```

- [ ] **Step 7: テストが通ることを確認する**

Run: `cargo test -p sapphire-agent -- a_zero_budget_means_unbounded text_reaches_the_host an_empty_text_is_not_handed an_unattended_host_stops an_interactive_host_runs_past`
Expected: PASS（5 件）

- [ ] **Step 8: 全体が壊れていないことを確認する**

Run: `cargo test -p sapphire-agent`
Expected: PASS。ACP はまだ既定の `Unattended`（25）なので、`acp.rs` の既存の上限テスト（スクリプト10本）は上限に達せず、`end_turn` が返って落ちるはずである。**これは想定内**: そのテストは Task 4 で新しい形に置き換える。落ちたまま Task 3 に進んでよい

- [ ] **Step 9: コミット**

```bash
git add server/src/serve/mod.rs server/src/config.rs
git commit -m "feat(serve): hand the host each round's prose, and take the round cap from config

Two hooks on TurnHost, both defaulted so no existing implementor moves:
message_chunk, called with every non-empty text a round produces, and
round_budget, which says which of the two configured caps this route is
judged by. TurnLoop::run resolves the budget once per turn, in one place,
the way it already resolves the permission table from origin().

MAX_TOOL_ROUNDS is now unused here but stays until acp.rs stops
referencing it."
```

---

### Task 3: ACP を完全ストリーミングに移す

`AcpProgress` が両方のフックを実装し、ターン終了時の一括送出をやめる。ここで初めて外から見える挙動が変わる。

**Files:**
- Modify: `server/src/serve/acp.rs`（`impl super::TurnHost for AcpProgress` は 311 行付近、`BudgetExhausted` 分岐は 1520〜1545 行付近、最終チャンク送出は 1560〜1572 行付近）
- Test: `server/src/serve/acp.rs`（同ファイル内の `mod tests`）

**Interfaces:**
- Consumes: `TurnHost::message_chunk` / `round_budget`、`RoundBudget`（Task 2）
- Produces: なし（ACP の外部インタフェースは変わらない）

- [ ] **Step 1: 失敗するテストを書く**

`server/src/serve/acp.rs` の `mod tests` に追加する。`spawn` / `drive` / `text_prompt` は既存のヘルパ。

```rust
    /// 中間の散文がラウンドごとに届く。まとめて1通ではなく、モデルが
    /// 出した単位で並ぶ。これが「作業中に喋れる」の全部である。
    #[tokio::test]
    async fn prose_arrives_round_by_round_rather_than_all_at_the_end() {
        let state = ServeState::for_test_scripted(
            true,
            vec![
                crate::provider::ChatResponse {
                    text: Some("checking the config".to_string()),
                    tool_calls: vec![crate::provider::ToolCall {
                        id: "call-1".to_string(),
                        name: "echo".to_string(),
                        input: serde_json::json!({ "text": "ping" }),
                    }],
                    stop_reason: None,
                },
                crate::provider::ChatResponse {
                    text: Some("it was the timeout".to_string()),
                    tool_calls: Vec::new(),
                    stop_reason: None,
                },
            ],
        );
        let addr = spawn(state).await;
        let (_session_id, updates, reply) = drive(&addr, text_prompt("why is it slow")).await;

        let chunks: Vec<&str> = updates
            .iter()
            .filter(|u| u["sessionUpdate"] == "agent_message_chunk")
            .map(|u| u["content"]["text"].as_str().unwrap())
            .collect();
        assert_eq!(
            chunks,
            vec!["checking the config", "it was the timeout"],
            "got {chunks:?}"
        );
        assert_eq!(reply["result"]["stopReason"], "end_turn", "got {reply}");
    }

    /// 最終テキストが二重に届かない。`message_chunk` で流したものを
    /// ターン終了時にもう一度送っていたら、ここで2通になる。
    #[tokio::test]
    async fn the_final_reply_is_not_sent_twice() {
        let state = ServeState::for_test_scripted(
            true,
            vec![crate::provider::ChatResponse {
                text: Some("just this".to_string()),
                tool_calls: Vec::new(),
                stop_reason: None,
            }],
        );
        let addr = spawn(state).await;
        let (_session_id, updates, _reply) = drive(&addr, text_prompt("hello")).await;

        let chunks: Vec<&str> = updates
            .iter()
            .filter(|u| u["sessionUpdate"] == "agent_message_chunk")
            .map(|u| u["content"]["text"].as_str().unwrap())
            .collect();
        assert_eq!(chunks, vec!["just this"], "got {chunks:?}");
    }

    /// ACP は `interactive` 側で判定される。既定は 0 = 無制限なので、
    /// 旧来の10ラウンドを超えても止まらない。
    #[tokio::test]
    async fn an_acp_turn_runs_past_ten_rounds_by_default() {
        let mut script: Vec<crate::provider::ChatResponse> = (0..12)
            .map(|i| crate::provider::ChatResponse {
                text: None,
                tool_calls: vec![crate::provider::ToolCall {
                    id: format!("call-{i}"),
                    name: "echo".to_string(),
                    input: serde_json::json!({ "text": "ping" }),
                }],
                stop_reason: None,
            })
            .collect();
        script.push(crate::provider::ChatResponse {
            text: Some("finished".to_string()),
            tool_calls: Vec::new(),
            stop_reason: None,
        });
        let addr = spawn(ServeState::for_test_scripted(true, script)).await;
        let (_session_id, _updates, reply) = drive(&addr, text_prompt("do a lot")).await;

        assert_eq!(reply["result"]["stopReason"], "end_turn", "got {reply}");
    }
```

- [ ] **Step 2: 落ちることを確認する**

Run: `cargo test -p sapphire-agent -- prose_arrives_round_by_round the_final_reply_is_not_sent_twice an_acp_turn_runs_past_ten`
Expected: FAIL — 1つ目は `["checking the config\n\nit was the timeout"]` の1通、2つ目は `["just this", "just this"]` の2通、3つ目は `max_turn_requests`

- [ ] **Step 3: `AcpProgress` に2つを実装する**

`server/src/serve/acp.rs` の `impl super::TurnHost for AcpProgress`、`turn_error` の直後（343 行付近）に足す。

```rust
    /// Sent the moment the round that produced it is done, rather than
    /// held until the turn ends.
    ///
    /// One chunk per round, not per token: `Provider::chat` returns a
    /// whole response, so a round is the finest boundary that actually
    /// exists. Splitting further would invent boundaries the model never
    /// produced.
    ///
    /// This is the *only* way text reaches an ACP client now. The turn's
    /// `outcome.text` is deliberately ignored on this path — see the
    /// `session/prompt` handler — because everything in it has already
    /// come through here, and sending it again would duplicate the whole
    /// reply.
    async fn message_chunk(&self, text: &str) {
        self.notify(SessionUpdate::AgentMessageChunk(ContentChunk::new(
            ContentBlock::Text(TextContent::new(text.to_string())),
        )));
    }

    /// ACP is the one route that can stop a turn in flight:
    /// `session/cancel` is implemented, and the editor's Escape sends it.
    /// That is what makes an unbounded budget safe to offer here and
    /// nowhere else.
    fn round_budget(&self) -> super::RoundBudget {
        super::RoundBudget::Interactive
    }
```

- [ ] **Step 4: 終了時の送出を2箇所削除する**

`BudgetExhausted` 分岐（1520〜1545 行付近）を、コメントごと差し替える。

```rust
                                // Running out of tool rounds is not a
                                // failure, and ACP has the exact word for
                                // it: `MaxTurnRequests`, "the agent reached
                                // the maximum number of allowed agent
                                // requests between user turns". With
                                // `[tools.tool_rounds] interactive = 0` —
                                // the default — this never fires at all;
                                // an operator who sets a cap gets a
                                // routine ending rather than an error
                                // dialog, and the work done on the way is
                                // already in the client's hands because
                                // every round's prose went out through
                                // `message_chunk` as it happened.
                                if matches!(&outcome.stop, super::TurnStop::BudgetExhausted { .. })
                                {
                                    return answered(
                                        &session_id,
                                        responder.respond(PromptResponse::new(
                                            StopReason::MaxTurnRequests,
                                        )),
                                    );
                                }
```

`let Some(reply) = outcome.text else { ... }` の分岐は**残す** — `None` はプロバイダ失敗を意味し、JSON-RPC エラーを返す判定に使われている。束縛だけ `let Some(_reply) = outcome.text else` に変える。

最終チャンク送出（1560〜1572 行付近）の `if !reply.is_empty() { progress.notify(...) }` を削除し、コメントを置いて `answered(...)` だけを残す。

```rust
                            // No chunk here: every non-empty text this turn
                            // produced — the final one included — already
                            // went out from `AcpProgress::message_chunk` in
                            // the round that produced it. `outcome.text` is
                            // the same prose joined back together, kept for
                            // `/rpc`, `/a2a` and the voice pipeline, and
                            // sending it here as well would deliver the
                            // whole reply twice.
                            answered(
                                &session_id,
                                responder.respond(PromptResponse::new(StopReason::EndTurn)),
                            )
```

- [ ] **Step 5: 新しいテストが通ることを確認する**

Run: `cargo test -p sapphire-agent -- prose_arrives_round_by_round the_final_reply_is_not_sent_twice an_acp_turn_runs_past_ten`
Expected: PASS（3 件）

- [ ] **Step 6: コミット**

```bash
git add server/src/serve/acp.rs
git commit -m "feat(acp): stream the turn's prose as it is produced

AcpProgress implements message_chunk, so every round's text goes out in
the round that produced it, and the two end-of-turn sends are gone: they
would now deliver the whole reply a second time.

ACP also declares itself Interactive. It is the one route with
session/cancel, which is what makes the unbounded default safe here.

The budget test in this file still asserts the old batched shape and is
updated next."
```

---

### Task 4: 古い上限テストを新しい形に直し、`MAX_TOOL_ROUNDS` を消す

Task 2 と 3 で破れたテストを直し、`serve/` から const を落とす。

**Files:**
- Modify: `server/src/serve/acp.rs`（`exhausting_the_tool_budget_ends_the_turn_with_max_turn_requests` は 2355 行付近）
- Modify: `server/src/serve/mod.rs`（`MAX_TOOL_ROUNDS` は 44 行、参照するテストは 5456〜5470 行付近）
- Modify: `server/src/tools/subagent.rs`（モジュールコメント 59 行付近）

**Interfaces:**
- Consumes: `ServeState::for_test_scripted_with_rounds`（Task 2）
- Produces: なし

- [ ] **Step 1: 壊れているテストを確認する**

Run: `cargo test -p sapphire-agent -- exhausting_the_tool_budget`
Expected: FAIL — ACP が `Interactive`（無制限）になったので `end_turn` が返り、`max_turn_requests` を期待するアサートが落ちる

- [ ] **Step 2: テストを新しい形に書き換える**

`exhausting_the_tool_budget_ends_the_turn_with_max_turn_requests` を丸ごと差し替える。

```rust
    /// Running out of tool rounds is an ordinary ending, not a broken
    /// agent, and ACP has a stop reason for exactly it. The default no
    /// longer reaches it — `interactive = 0` is unbounded — so this pins
    /// the configured case: an operator who sets a cap gets
    /// `max_turn_requests`, not an internal-error dialog.
    ///
    /// The prose is pinned too, in its new shape: one chunk per round, as
    /// each round produced it, rather than one joined chunk at the end.
    #[tokio::test]
    async fn a_configured_budget_ends_the_turn_with_max_turn_requests() {
        const ROUNDS: usize = 4;

        let script: Vec<crate::provider::ChatResponse> = (0..ROUNDS)
            .map(|i| crate::provider::ChatResponse {
                text: Some(format!("step {i}")),
                tool_calls: vec![crate::provider::ToolCall {
                    id: format!("call-{i}"),
                    name: "echo".to_string(),
                    input: serde_json::json!({ "text": "ping" }),
                }],
                stop_reason: None,
            })
            .collect();
        let state = ServeState::for_test_scripted_with_rounds(
            true,
            script,
            crate::config::ToolRounds {
                interactive: ROUNDS,
                unattended: ROUNDS,
            },
        );
        let addr = spawn(state).await;
        let (_session_id, updates, reply) = drive(&addr, text_prompt("do a big refactor")).await;

        assert!(
            reply.get("error").is_none(),
            "a spent budget is not an error, got {reply}"
        );
        assert_eq!(
            reply["result"]["stopReason"], "max_turn_requests",
            "got {reply}"
        );

        // Every round's prose reached the editor, in its own chunk.
        let expected: Vec<String> = (0..ROUNDS).map(|i| format!("step {i}")).collect();
        let chunks: Vec<&str> = updates
            .iter()
            .filter(|u| u["sessionUpdate"] == "agent_message_chunk")
            .map(|u| u["content"]["text"].as_str().unwrap())
            .collect();
        assert_eq!(chunks, expected, "got {chunks:?}");
    }
```

- [ ] **Step 3: 通ることを確認する**

Run: `cargo test -p sapphire-agent -- a_configured_budget_ends_the_turn`
Expected: PASS

- [ ] **Step 4: `MAX_TOOL_ROUNDS` を消す**

`server/src/serve/mod.rs:44` の `const MAX_TOOL_ROUNDS: usize = 10;` を削除する。

同ファイル 5456〜5470 行付近のテストが、`MAX_TOOL_ROUNDS` より多い tool_use を含む履歴を組み立てるために参照している。`for i in 0..(MAX_TOOL_ROUNDS + 1)` を `for i in 0..11` に置き換え、その doc コメントの `MAX_TOOL_ROUNDS` への言及を「かつて固定だった10ラウンド」に書き換える。

`server/src/tools/subagent.rs:59` のモジュールコメントは所要時間の説明に `MAX_TOOL_ROUNDS` を使っている。サブエージェントは常に `Unattended` なので、`[tools.tool_rounds]` の `unattended` を指すよう書き換える。

- [ ] **Step 5: `serve/` に参照が残っていないことを確認する**

Run: `rg 'MAX_TOOL_ROUNDS' server/src/`
Expected: `server/src/agent.rs` の2箇所（17 行の const と 771 行の使用）のみ。Task 6 で消す

- [ ] **Step 6: 全体が通ることを確認する**

Run: `cargo test -p sapphire-agent`
Expected: PASS。Task 2 Step 8 で落ちていたテストがここで解消する

- [ ] **Step 7: コミット**

```bash
git add server/src/serve/acp.rs server/src/serve/mod.rs server/src/tools/subagent.rs
git commit -m "test(acp): pin the budget ending against a configured cap, not the default

The default no longer reaches it. The test now sets a cap of four and
asserts both halves that matter: max_turn_requests rather than an error,
and four separate chunks rather than one joined one.

MAX_TOOL_ROUNDS is gone from serve/; agent.rs still has its own copy."
```

---

### Task 5: サブエージェントの2つを非委譲で実装する

`ParentHostSansTurnError` は各メソッドを明示委譲する構造なので、新しい2つも明示的に決める。どちらも委譲しない。

**Files:**
- Modify: `server/src/tools/subagent.rs`（`ParentHostSansTurnError` は 261 行付近）
- Test: `server/src/tools/subagent.rs`（同ファイル内の `mod tests`）

**Interfaces:**
- Consumes: `TurnHost::message_chunk` / `round_budget`、`RoundBudget`（Task 2）
- Produces: なし

**このタスクだけ TDD の順序を採らない。** 理由を先に書く: `TurnHost` の新しい2つは既定実装を持ち、その既定（no-op と `Unattended`）が、たまたまこのラッパーに求める答えと一致する。したがって「実装前に落ちるテスト」は書けない — 何も書かなくてもテストは通ってしまう。

守っている危険は別物である。`ParentHostSansTurnError` は他の全メソッドを明示委譲しており、**書かれていないメソッドは見落としに見える**。誰かが「揃っていないから」と委譲を書き足した瞬間に、サブエージェントの散文が親のストリームに漏れ、入れ子の上限が消える。だからこのタスクは、実装（明示的に非委譲と書く）を先に入れ、テストはその決定を将来の編集から守る回帰テストとして後から足す。Step 4 でその回帰テストが本当に効くことを一度だけ手で確かめる。

- [ ] **Step 1: 非委譲であることをコードに書く**

`server/src/tools/subagent.rs` の `impl crate::serve::TurnHost for ParentHostSansTurnError`、`turn_error` の直後（277 行付近）に足す。既定に任せず明示的に書くのが要点。

```rust
    /// Swallowed, like `turn_error` and for a related reason: a
    /// subagent's prose is not the parent agent's speech. Forwarding it
    /// would put the delegate's narration into the editor under the
    /// parent's name, which misattributes it — and the parent has no way
    /// to correct the record, because by the time it sees the subagent's
    /// answer the chunks are already on screen.
    ///
    /// Nothing is lost: what the subagent concluded comes back as the
    /// `subagent` tool's result, which is where the parent reads it and
    /// where the user sees it attributed correctly.
    ///
    /// `tool_start`/`tool_end` still forward, deliberately — see the
    /// module doc. Those are what make a permission prompt for a
    /// subagent's call legible as coming from *this* session.
    async fn message_chunk(&self, _text: &str) {}

    /// Not delegated, and this one must not be: a nested turn under an
    /// unbounded parent would itself be unbounded, so a parent that
    /// delegates in a loop would have no cap anywhere. A subagent is
    /// always judged `unattended` — nobody can cancel it directly, only
    /// the whole parent turn — whatever route the parent came in on.
    fn round_budget(&self) -> crate::serve::RoundBudget {
        crate::serve::RoundBudget::Unattended
    }
```

- [ ] **Step 2: 回帰テストを書く**

`server/src/tools/subagent.rs` の `mod tests` に追加する。既存の `RecordingHost` は必要なメソッドを持たないので、テスト専用のホストをここで定義する。

```rust
    /// サブエージェントの散文は親のストリームに漏れない。漏れると、
    /// 委任先が言ったことが親エージェント自身の発言として編集画面に出る
    /// —— 誤帰属であり、報告はツール結果として戻ってくる。
    #[tokio::test]
    async fn a_subagents_prose_does_not_reach_the_parents_stream() {
        #[derive(Default)]
        struct ChunkRecorder {
            chunks: std::sync::Mutex<Vec<String>>,
        }
        #[async_trait]
        impl crate::serve::TurnHost for ChunkRecorder {
            async fn tool_start(&self, _id: &str, _name: &str) {}
            async fn tool_end(&self, _id: &str, _name: &str) {}
            async fn turn_error(&self, _message: &str) {}
            async fn message_chunk(&self, text: &str) {
                self.chunks.lock().unwrap().push(text.to_string());
            }
        }

        let parent = std::sync::Arc::new(ChunkRecorder::default());
        let wrapped = ParentHostSansTurnError(
            std::sync::Arc::clone(&parent) as std::sync::Arc<dyn crate::serve::TurnHost>,
        );

        crate::serve::TurnHost::message_chunk(&wrapped, "the subagent's own words").await;

        let seen = parent.chunks.lock().unwrap().clone();
        assert!(seen.is_empty(), "親には届かない: {seen:?}");
    }

    /// 親が無制限でも、サブエージェントは有限で回る。委譲していたら
    /// 入れ子が無制限になり、上限が二乗で消える。
    #[test]
    fn a_subagent_is_unattended_even_under_an_interactive_parent() {
        struct InteractiveParent;
        #[async_trait]
        impl crate::serve::TurnHost for InteractiveParent {
            async fn tool_start(&self, _id: &str, _name: &str) {}
            async fn tool_end(&self, _id: &str, _name: &str) {}
            async fn turn_error(&self, _message: &str) {}
            fn round_budget(&self) -> crate::serve::RoundBudget {
                crate::serve::RoundBudget::Interactive
            }
        }

        let wrapped = ParentHostSansTurnError(
            std::sync::Arc::new(InteractiveParent) as std::sync::Arc<dyn crate::serve::TurnHost>,
        );

        assert_eq!(
            crate::serve::TurnHost::round_budget(&wrapped),
            crate::serve::RoundBudget::Unattended
        );
    }
```

- [ ] **Step 3: テストが通ることを確認する**

Run: `cargo test -p sapphire-agent -- a_subagents_prose_does_not_reach a_subagent_is_unattended_even_under`
Expected: PASS（2 件）

- [ ] **Step 4: 委譲を書き足すと落ちることを確認する**

回帰テストが本当に守っているかを、一度だけ手で確かめる。Step 1 で書いた2つの本体を、それぞれ委譲に書き換えて回す。

1. `message_chunk` の本体を `self.0.message_chunk(_text).await;` にする（引数名は `text` に変える）
2. `round_budget` の本体を `self.0.round_budget()` にする

Run: `cargo test -p sapphire-agent -- a_subagents_prose_does_not_reach a_subagent_is_unattended_even_under`
Expected: FAIL（2 件とも）。確認できたら Step 1 の形に書き戻し、再度 PASS することを確認する

- [ ] **Step 5: コミット**

```bash
git add server/src/tools/subagent.rs
git commit -m "feat(tools): keep a subagent's prose and budget out of its parent's

ParentHostSansTurnError delegates every method explicitly, so the two
new ones need an explicit answer. Neither delegates.

message_chunk is swallowed: a delegate's narration under the parent's
name misattributes it, and its conclusion comes back as the tool result
anyway. round_budget is pinned to Unattended: a nested turn under an
unbounded parent would be unbounded, so the cap would vanish for anyone
who delegates in a loop.

The defaults happen to agree with both, so the tests here are
regression tests for a future edit that 'completes' the delegation
rather than a red-green pair."
```

---

---

### Task 6a: `agent.rs` のテスト土台を作る

`server/src/agent.rs` の `mod tests`（1077 行〜）には、`Agent` を組み立てるヘルパが**一つも無い**。あるのはツールポリシー関門のテスト4件だけで、スクリプト済みプロバイダも、送信を記録するチャンネルスタブも無い。Task 6b の振る舞いテストはこれ無しには書けないので、土台だけを先に作り、それ自体を1つのテストで証明する。

**このタスクは本番コードを変更しない。** `#[cfg(test)]` の中だけで完結する。

**Files:**
- Modify: `server/src/agent.rs`（`mod tests` は 1077 行〜。既存の `test_workspace()` は 1086 行付近）

**Interfaces:**
- Consumes: なし
- Produces（すべて `mod tests` の中）:
  - `struct RecordingChannel { sent: Arc<Mutex<Vec<OutgoingMessage>>>, fail_first: bool, calls: Mutex<usize> }`
    — `impl crate::channel::Channel for RecordingChannel`
  - `fn agent_with_scripted_provider(responses: Vec<crate::provider::ChatResponse>) -> (Arc<Agent>, Arc<Mutex<Vec<OutgoingMessage>>>)`
  - `fn agent_with_failing_first_send(responses: Vec<crate::provider::ChatResponse>) -> (Arc<Agent>, Arc<Mutex<Vec<OutgoingMessage>>>)`
  - `fn incoming(text: &str) -> crate::channel::IncomingMessage`

- [ ] **Step 1: 組み立てに必要なものを読む**

先に読むこと。**既にあるものを使い、無いものだけ作る。**

- `server/src/agent.rs:82` の `Agent::new` — 8引数（`Config`, `Arc<Channels>`, `Arc<ProviderRegistry>`, `Arc<Workspace>`, `Option<Arc<ToolSet>>`, `Arc<SessionStore>`, `Option<Arc<ImageCache>>`, `Option<Arc<DigestCache>>`）
- `server/src/agent.rs:1086` の `test_workspace()` — 既存。`AppContext` のキャッシュディレクトリ設定と `Workspace::from_root` をすでに済ませてある。**これを再利用する**
- `server/src/channel/mod.rs:117` の `trait Channel` — `send` / `start_typing` / `stop_typing` ほか。既定実装のあるメソッドは実装しない
- `server/src/channel/mod.rs:153` の `Channels::new(Vec<(String, Arc<dyn Channel>)>, HashMap<String, String>)` — ここにスタブを差す
- `server/src/serve/mod.rs` の `ServeState::build_for_test` — `Config::parse_for_test` とテンポラリディレクトリから組み立てる流儀。**同じ流儀に合わせる**
- `server/src/serve/mod.rs` の `StubProvider` — スクリプト済み `ChatResponse` を順に返すテスト用プロバイダが既にある。`ProviderRegistry` に差せるならこれを使い、`#[cfg(test)]` の可視性が届かない場合のみ同等のものを `agent.rs` 側に作る
- `server/src/session.rs` の `SessionStore` — テンポラリディレクトリから作れるか確認する

- [ ] **Step 2: 土台が動くことを示すテストを1つ書く**

```rust
    /// 土台そのもののテスト。`Agent` が組み立てられ、`handle_message` が
    /// 一周し、スタブチャンネルに返信が届くところまでを通す。Task 6b の
    /// 振る舞いテストは全部これに乗るので、ここが通らないうちは先へ進めない。
    #[tokio::test]
    async fn the_test_harness_drives_a_turn_end_to_end() {
        let (agent, sent) = agent_with_scripted_provider(vec![crate::provider::ChatResponse {
            text: Some("ok".to_string()),
            tool_calls: Vec::new(),
            stop_reason: None,
        }]);

        agent.handle_message(incoming("hello")).await.unwrap();

        let bodies: Vec<String> =
            sent.lock().unwrap().iter().map(|m| m.content.clone()).collect();
        assert_eq!(bodies, vec!["ok".to_string()]);
    }
```

- [ ] **Step 3: 落ちることを確認する**

Run: `cargo test -p sapphire-agent -- the_test_harness_drives_a_turn`
Expected: FAIL — `agent_with_scripted_provider` / `incoming` が存在せずコンパイルエラー

- [ ] **Step 4: スタブチャンネルを書く**

`mod tests` の中に置く。`Channel` トレイトの必須メソッドだけを実装し、既定実装のあるものは触らない。

```rust
    /// Records what the agent sent, so a test can assert on the shape of a
    /// turn's delivery rather than only its final text.
    ///
    /// `fail_first` exists for one test: a send that the channel refuses
    /// must not take the rest of the turn down with it. Counting calls
    /// rather than holding a queue of outcomes keeps the stub to the one
    /// behaviour that is actually needed.
    struct RecordingChannel {
        sent: Arc<Mutex<Vec<crate::channel::OutgoingMessage>>>,
        fail_first: bool,
        calls: Mutex<usize>,
    }
```

`impl crate::channel::Channel for RecordingChannel` の `send` は、`fail_first` が立っていて呼び出しが1回目なら `Err` を返し、それ以外は `sent` に積んで `Ok(())` を返す。`start_typing` / `stop_typing` は既定実装があるなら実装しない。

トレイトの他の必須メソッド（名前・受信ループなど）は、コンパイラが要求するものだけを最小限に埋める。**推測で埋めない** — `cargo build` のエラーが要求するものが正解である。

- [ ] **Step 5: `Agent` 組み立てヘルパを書く**

```rust
    /// Build an `Agent` whose provider replays `responses` in order and
    /// whose only channel records what it is asked to send.
    ///
    /// Mirrors `ServeState::build_for_test` (src/serve/mod.rs): leak the
    /// `TempDir` on purpose — this is a test binary and the OS reclaims
    /// the directory when it exits.
    fn agent_with_scripted_provider(
        responses: Vec<crate::provider::ChatResponse>,
    ) -> (Arc<Agent>, Arc<Mutex<Vec<crate::channel::OutgoingMessage>>>) {
        build_test_agent(responses, false)
    }

    /// Same, but the channel refuses the first send.
    fn agent_with_failing_first_send(
        responses: Vec<crate::provider::ChatResponse>,
    ) -> (Arc<Agent>, Arc<Mutex<Vec<crate::channel::OutgoingMessage>>>) {
        build_test_agent(responses, true)
    }
```

`build_test_agent(responses, fail_first)` が実際の組み立てを行う。`test_workspace()` を再利用し、`Config::parse_for_test` で最小の設定を作り、`Channels::new(vec![("test".into(), Arc::new(channel))], HashMap::new())` を差す。`tools` は `None` でよい — Task 6b のテストが呼ぶツールは `echo` だが、ツールが解決できない場合の挙動を確かめるのが目的ではないので、`ToolSet` が要るなら `crate::tools::default_tool_set` を使う。**どちらが要るかは Task 6b のテストが決める**: ツールコールを含む応答を流すので、そのツール名が解決できる `ToolSet` が要る。`ServeState::build_for_test` が `EchoTool` を差しているので、同じ `echo` が使えるようにすること。

`incoming` は最小の `IncomingMessage` を作る。`room_id` と `thread_id` はテスト間で固定でよい。

```rust
    fn incoming(text: &str) -> crate::channel::IncomingMessage {
        // フィールドは `crate::channel::IncomingMessage` の定義に合わせる
    }
```

- [ ] **Step 6: テストが通ることを確認する**

Run: `cargo test -p sapphire-agent -- the_test_harness_drives_a_turn`
Expected: PASS

- [ ] **Step 7: 既存テストが壊れていないことを確認する**

Run: `cargo test -p sapphire-agent`
Expected: PASS。ただし `acp.rs` の `exhausting_the_tool_budget_...` は Task 2〜4 の途中経過として落ちている場合がある。**それ以外**が通っていればよい

- [ ] **Step 8: コミット**

```bash
git add server/src/agent.rs
git commit -m "test(agent): give agent.rs a harness that can drive a whole turn

The tests in this file could only reach the tool-policy gate; there was
no way to build an Agent, script its provider, or see what it sent. The
next commit changes how a turn is delivered to a channel, which is not
observable without one.

A recording channel, a scripted provider, and one test that drives a
turn end to end through them. No production code moves here."
```

---

### Task 6b: ラウンドごとに送る

Task 6a の土台の上で、`agent.rs` のループを書き換える。

**Files:**
- Modify: `server/src/agent.rs`（const 17 行、`accumulated_text` 754 行付近、上限判定 771 行付近、分岐 842 / 850 行付近、`stop_typing` 982 行、末尾送信 984〜996 行付近）
- Test: `server/src/agent.rs`（`mod tests`）

**Interfaces:**
- Consumes: `ToolRounds::limit`、`RoundBudget`（Task 2）、`agent_with_scripted_provider` / `agent_with_failing_first_send` / `incoming`（Task 6a）
- Produces: `Agent::send_turn_message(&self, incoming: &IncomingMessage, text: &str)`（このタスク内でのみ使う）

- [ ] **Step 1: 失敗するテストを書く**

`server/src/agent.rs` の `mod tests` に3件追加する。ヘルパは Task 6a のものをそのまま使う。

```rust
    /// ツールを呼ぶターンは複数メッセージに分かれる。「○○するぞ」と
    /// 「○○したぞ」が1通に潰れていたのが、この変更の直す対象である。
    #[tokio::test]
    async fn a_tool_using_turn_is_delivered_as_several_messages() {
        let (agent, sent) = agent_with_scripted_provider(vec![
            crate::provider::ChatResponse {
                text: Some("調べます".to_string()),
                tool_calls: vec![crate::provider::ToolCall {
                    id: "call-1".to_string(),
                    name: "echo".to_string(),
                    input: serde_json::json!({ "text": "ping" }),
                }],
                stop_reason: None,
            },
            crate::provider::ChatResponse {
                text: Some("見つかりました".to_string()),
                tool_calls: Vec::new(),
                stop_reason: None,
            },
        ]);

        agent.handle_message(incoming("なぜ遅い")).await.unwrap();

        let bodies: Vec<String> =
            sent.lock().unwrap().iter().map(|m| m.content.clone()).collect();
        assert_eq!(
            bodies,
            vec!["調べます".to_string(), "見つかりました".to_string()]
        );
    }

    /// ツールを呼ばない普通の会話は、今までどおり1通のまま。この変更で
    /// 日常の会話が細切れになってはいけない。
    #[tokio::test]
    async fn an_ordinary_reply_is_still_one_message() {
        let (agent, sent) = agent_with_scripted_provider(vec![crate::provider::ChatResponse {
            text: Some("こんにちは".to_string()),
            tool_calls: Vec::new(),
            stop_reason: None,
        }]);

        agent.handle_message(incoming("やあ")).await.unwrap();

        let bodies: Vec<String> =
            sent.lock().unwrap().iter().map(|m| m.content.clone()).collect();
        assert_eq!(bodies, vec!["こんにちは".to_string()]);
    }

    /// 送信に失敗してもターンは落ちない。上限ぶんの連投がレート制限に
    /// 触れうる以上、1通の失敗でターン全体を失うのは高すぎる。
    #[tokio::test]
    async fn a_failed_send_does_not_end_the_turn() {
        let (agent, sent) = agent_with_failing_first_send(vec![
            crate::provider::ChatResponse {
                text: Some("最初".to_string()),
                tool_calls: vec![crate::provider::ToolCall {
                    id: "call-1".to_string(),
                    name: "echo".to_string(),
                    input: serde_json::json!({ "text": "ping" }),
                }],
                stop_reason: None,
            },
            crate::provider::ChatResponse {
                text: Some("最後".to_string()),
                tool_calls: Vec::new(),
                stop_reason: None,
            },
        ]);

        let result = agent.handle_message(incoming("やって")).await;

        assert!(result.is_ok(), "1通の失敗でターンを落とさない");
        let bodies: Vec<String> =
            sent.lock().unwrap().iter().map(|m| m.content.clone()).collect();
        assert_eq!(bodies, vec!["最後".to_string()], "後続は届く");
    }
```

- [ ] **Step 2: 落ちることを確認する**

Run: `cargo test -p sapphire-agent -- a_tool_using_turn_is_delivered an_ordinary_reply_is_still_one a_failed_send_does_not_end`
Expected: FAIL — 1件目は `["調べます\n\n見つかりました"]` の1通になる

- [ ] **Step 3: 送信ヘルパを足す**

`server/src/agent.rs` の `Agent` の `impl` に置く。

```rust
    /// Send one piece of the turn's prose, as the round that produced it
    /// finishes.
    ///
    /// A failure is logged and swallowed rather than ending the turn. A
    /// turn can now send up to `[tools.tool_rounds] unattended` messages,
    /// which is enough to meet a channel's rate limit, and losing the
    /// whole turn's remaining work to one refused send would be a much
    /// worse outcome than a gap in the narration.
    ///
    /// Typing is restarted after each send: the indicator stops on its
    /// own when a message lands, and a turn that goes quiet while it is
    /// still working is exactly what this change exists to prevent.
    async fn send_turn_message(&self, incoming: &IncomingMessage, text: &str) {
        let out = OutgoingMessage {
            content: text.to_string(),
            room_id: incoming.room_id.clone(),
            thread_id: incoming.thread_id.clone(),
        };
        if let Err(e) = self.channels.send(&out).await {
            warn!("Failed to send a turn message: {e:#}");
        }
        let _ = self.channels.start_typing(&incoming.room_id).await;
    }
```

- [ ] **Step 4: ループを書き換える**

`const MAX_TOOL_ROUNDS`（17 行）を削除し、`let mut round = 0usize;` の直前に上限解決を置く。

```rust
        // Matrix and Discord have no way to cancel a turn in flight, so
        // they are judged `unattended` — the same budget every route
        // without an interrupt gets. See `[tools.tool_rounds]`.
        let round_limit = self
            .config
            .tools
            .tool_rounds
            .limit(crate::serve::RoundBudget::Unattended);
```

上限判定（771 行付近）。

```rust
            if round_limit.is_some_and(|max| round >= max) {
                warn!("Reached max tool rounds ({round}), stopping");
                break;
            }
```

ツールを呼ばない応答の分岐（842 行付近）。

```rust
                    if !text.is_empty() {
                        self.send_turn_message(&incoming, &text).await;
                    }
                    break;
```

ツールを呼ぶ応答の分岐（850 行付近）。

```rust
                    if let Some(t) = resp.text.as_ref().filter(|s| !s.is_empty()) {
                        self.send_turn_message(&incoming, t).await;
                    }
```

`accumulated_text` はこのファイルから不要になる。宣言（754 行付近）を削除し、`let final_text = loop {` は `loop {` に、`break Some(accumulated_text.join("\n\n"))` は `break` になる。

- [ ] **Step 5: 末尾の一括送信を消す**

984 行付近の `if let Some(text) = final_text { ... }` から送信ブロックを削除する。

**プリフェッチの spawn は残す** — 次のターンのための処理で、送信とは無関係である。`final_text` の束縛が消えるので、プリフェッチは `if let` の外に出して条件なしで実行する。

`stop_typing`（982 行）はそのまま残す。ループ内の各送信後に `start_typing` を打ち直しているので、最後にここで止める必要がある。

- [ ] **Step 6: テストが通ることを確認する**

Run: `cargo test -p sapphire-agent -- a_tool_using_turn_is_delivered an_ordinary_reply_is_still_one a_failed_send_does_not_end the_test_harness_drives_a_turn`
Expected: PASS（4 件。6a の土台テストも引き続き通る）

- [ ] **Step 7: 参照が残っていないことを確認する**

Run: `rg 'MAX_TOOL_ROUNDS|accumulated_text' server/src/agent.rs`
Expected: 一致なし

- [ ] **Step 8: 全体が通ることを確認する**

Run: `cargo test -p sapphire-agent`
Expected: PASS

- [ ] **Step 9: コミット**

```bash
git add server/src/agent.rs
git commit -m "feat(agent): send each round's prose to the channel as it happens

'I'll look into it' and 'here's what I found' are two messages, and were
being collapsed into one that arrived only when the turn was over. On a
long turn that is silence exactly where progress is worth seeing.

The batching was never a decision — 1971e0d added accumulated_text as
the minimal fix for prose that was being dropped outright, and the
one-in-one-out shape predates the tool loop. A reply with no tool calls
is still a single message, unchanged.

A refused send is logged, not fatal: a turn may now send up to the
unattended budget's worth of messages, which can meet a channel's rate
limit, and losing the rest of the turn to one of them is worse than a
gap in the narration."
```

---

### Task 7: `AGENTS.md` にターンの終わり方を書く

停止条件はコードでは変わっていない。モデルにそれを知らせる。

**Files:**
- Modify: `server/templates/workspace/AGENTS.md`

**Interfaces:**
- Consumes: なし
- Produces: なし

- [ ] **Step 1: 節を足す**

`server/templates/workspace/AGENTS.md` の "Session Startup" 節の直後、"Memory" 節の前に置く。ホストに依らない規約なので `TOOLS.md`（そのホスト固有の話を書く場所）ではなくここに入る。

```markdown
## Speaking While You Work

A reply with no tool call ends your turn. That is the only thing that
ends it — there is no budget you have to race, and no "continue" you
need to be given.

So say what you are doing *alongside* the tool call that does it, not in
a message of its own. "Checking the config" followed by the read reaches
the person immediately and leaves you still working. "Checking the
config" on its own hands the turn back and waits for them to tell you to
go on.

Report as you go on anything that takes more than a moment. Each round's
words arrive as their own message, so there is no cost to saying where
you are — and on a long piece of work, silence is indistinguishable from
being stuck.

Save a bare reply for when you are genuinely finished, or genuinely need
an answer before you can continue. Those are the same thing to the
person reading: it is now their turn.
```

- [ ] **Step 2: 埋め込みが壊れていないことを確認する**

`server/src/cli_init.rs:23` が `include_str!` でこのファイルを取り込む。

Run: `cargo build -p sapphire-agent`
Expected: 成功

- [ ] **Step 3: コミット**

```bash
git add server/templates/workspace/AGENTS.md
git commit -m "docs(workspace): tell the agent how a turn actually ends

The stop condition has not changed: a reply with no tool call ends the
turn. What changed is that narration alongside a tool call now reaches
the person in the round that produced it, which makes 'say what you are
doing while you do it' a real option rather than a silent one.

Written where host-independent conventions go. TOOLS.md is for what only
this machine knows."
```

---

### Task 8: 通しで確認する

**Files:** なし（検証のみ）

**Interfaces:**
- Consumes: Task 1〜7 のすべて
- Produces: なし

- [ ] **Step 1: 全テストを回す**

Run: `cargo test -p sapphire-agent`
Expected: PASS

- [ ] **Step 2: clippy に新しい警告を足していないことを確認する**

Run: `cargo clippy -p sapphire-agent --all-targets 2>&1 | grep -c '^warning: '`

**`-D warnings` で「警告ゼロ」を期待してはいけない。** `origin/main`（a6cacdc）の時点で
既に6件の警告がある — `acp_session.rs:1474`、`serve/mod.rs:3379`（`ChatLog`）、
`session.rs:2146`、`session.rs:2320`、`skills.rs:460`、`tools/subagent.rs:1064`。
いずれも本ブランチが触っていない箇所で、直すのはこの変更の範囲外である。

Expected: 警告は**この6件のみ**。特に次の2つが消えていること:
- `constant MAX_TOOL_ROUNDS is never used` — Task 4 で const を削除して解消
- `variant RoundBudget::Interactive is never constructed` — Task 3 で構築され解消

`accumulated_text` の削除で未使用の変数や束縛が残っていれば、新しい警告として現れる。
7件目以降が出たらそれが本ブランチの負債である。

- [ ] **Step 3: 整形を確認する**

Run: `cargo fmt --check`
Expected: 差分なし

- [ ] **Step 4: 消えたはずの識別子が残っていないことを確認する**

Run: `rg 'MAX_TOOL_ROUNDS' server/`
Expected: 一致なし

- [ ] **Step 5: ワークスペース全体がビルドできることを確認する**

Run: `cargo build --workspace`
Expected: 成功（Linux/macOS）。Windows では `server/` はリンクしない — `CLAUDE.md` の "Two ONNX Runtimes" を参照

- [ ] **Step 6: すべてコミット済みであることを確認する**

Run: `git status --short`
Expected: 差分なし

---

## Self-Review

**1. Spec coverage**

| 仕様の節 | 対応タスク |
|---|---|
| §1 `TurnHost` にテキストフック | Task 2 |
| §2 ACP を完全ストリーミング化 | Task 3（実装）、Task 4（既存テストの更新） |
| §3 ラウンド上限を経路ごとの設定に | Task 1（設定）、Task 2（解決）、Task 4（`serve/` の const 削除）、Task 6b（`agent.rs` 側） |
| §4 サブエージェントの2つを非委譲 | Task 5 |
| §4 上限到達がユーザーを煩わせないこと | 既存の `answer_text` がそのまま担う。変更不要 — Task 5 のコメントで言及 |
| §5 Matrix/Discord のラウンドごと送信 | Task 6a（テスト土台）、Task 6b（本体＋振る舞いテスト） |
| §6 `AGENTS.md` の一節 | Task 7 |
| やらないこと: 停止条件、進捗報告ツール、`/rpc` SSE、音声 | どのタスクでも触らない。Task 2 Step 4 のトレイト doc が、音声を触らない理由をコードに残す |
| 既知の重複（2つのループ） | 直さない。Task 2 と Task 6b が同じ変更を両方に入れる形で現れる |

仕様の全節に対応タスクがある。

**2. Placeholder scan**

当初 Task 6 Step 1 が「既存のヘルパがあればそれを使い、無ければ作る」という条件付きだった。実行前の走査で `server/src/agent.rs` の `mod tests`（1077 行〜）を読んだ結果、`Agent` を組み立てるヘルパは**一つも無い**ことが判明したため、Task 6 を 6a（土台）と 6b（本体）に分割し、条件を除去した。6a Step 1 は「読むもの」を実ファイルの行番号で列挙してある。他に TBD / TODO / 「適切に処理する」の類は無い。

**3. Type consistency**

- `ToolRounds { interactive, unattended }` — Task 1 で定義、Task 2 / 4 / 6 で使用。フィールド名一貫
- `ToolRounds::limit(&self, RoundBudget) -> Option<usize>` — Task 2 Step 3 で定義、Task 2 Step 5（`TurnLoop::run`）と Task 6 Step 5（`agent.rs`）で使用。シグネチャ一貫
- `RoundBudget::{Interactive, Unattended}` — Task 2 Step 3 で定義、Task 3 / 5 / 6 で使用
- `TurnHost::message_chunk(&self, text: &str)` — Task 2 Step 4 で `async`・既定 no-op。Task 3 Step 3（`AcpProgress`）と Task 5 Step 1（`ParentHostSansTurnError`）が同じシグネチャで実装
- `TurnHost::round_budget(&self) -> RoundBudget` — 非 `async`。Task 3 Step 3 と Task 5 Step 1 が同じシグネチャで実装
- `ServeState::for_test_scripted_with_rounds(acp_enabled, responses, rounds)` — Task 2 Step 6 で定義、Task 2 Step 1 と Task 4 Step 2 で使用。引数順一貫
- `Agent::send_turn_message(&self, &IncomingMessage, &str)` — Task 6b Step 3 で定義、Step 4 で使用
- `agent_with_scripted_provider` / `agent_with_failing_first_send` / `incoming` — Task 6a で定義、Task 6b Step 1 のテストが使用

**4. 順序の依存**

Task 2 で ACP がまだ `Unattended` のまま `MAX_TOOL_ROUNDS`（10）から config の既定（25）に変わるため、`acp.rs` の既存の上限テスト（スクリプト10本）が上限に達しなくなり落ちる。Task 3 で ACP が `Interactive` になっても落ちたままで、Task 4 が新しい形に置き換えて解消する。

この一時的な破れは Task 2 Step 8、Task 3 Step 6 のコミットメッセージ、Task 4 Step 1 の期待値に明記してある。**Task 2・3・4 は続けて実施すること。** 途中で止めるとテストが赤いままコミットが積まれる。
