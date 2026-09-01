# ACP クライアント側ツール Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** エディタが開いているプロジェクトのファイルとシェルを、ACP 経由でエージェントが操作できるようにする。

**Architecture:** `Tool::execute` はセッションを知らないので、クライアントへの往復口は `tokio::task_local` で渡す — `scope_timer_origin` が timer ツールに呼び出し元を渡しているのと同じ仕組み。往復口自体は `AcpClient` トレイトにして、ツールは `acp.rs` に依存せずテストで差し替えられるようにする。ハンドルの出どころは既存の `TurnHost`（ACP ターンではそれが接続の包み）。

**Tech Stack:** Rust 2024, `agent-client-protocol` 2.0.0 / `-schema` 1.5.0, `tokio`, `async-trait`

**Spec:** `docs/superpowers/specs/2026-09-01-client-side-tools-design.md`

## Global Constraints

- ブランチは `feat/client-side-tools`（`main` から作成済み）。
- テストコマンドは `cargo test -p sapphire-agent`。反復中は `--workspace` を使わない。最後に一度だけ `cargo test --workspace`。
- **`cargo check` と `cargo test` を混ぜない。** フィンガープリントが別なので交互に走らせると毎回リンクし直しになる。
- cargo は**前景で、`timeout: 600000` で**。`run_in_background` も `Monitor` も使わない。10分のツールタイムアウトに当たったらビルドは温まっているので同じコマンドを走らせ直す。**cargo を2本同時に走らせない**（このホストの OS は熱でスロットリングする小さな USB SSD 上にある）。
- **`Cargo.lock` をコミットしない。** 各コミット前に `git checkout -- Cargo.lock`。
- テストはソースファイル内の `mod tests` に置く。
- **`src/agent.rs` は編集しない。**
- **SDK の送信要求をハンドラ内で await しない。** ディスパッチループが止まって応答が届かずデッドロックする。ツール実行は `run_llm_turn` の中＝ループの外なので安全だが、この規則は `acp.rs` を触るときに常に効く。既存の `approve`（`src/serve/acp.rs:325` 付近）のコメントがその根拠を書いている。
- **`terminal/release` はコマンドを殺す。** 「ハンドルを返す」ではない。接続が切れたことを理由に解放してはならない。
- **既存のサーバー側ツール（`file_*` / `dir_*` / `shell`）の実装は変えない。** 変えるのは「有効かどうか」と「一覧に出すかどうか」だけ。
- 日付は `Local::now().date_naive()` で作らない（境界時刻より前は前日）。この plan では日付を扱わないが、規則として残す。

## ファイル構成

| ファイル | 責務 |
|---|---|
| `src/tools/acp_client.rs`（新規） | `AcpClient` トレイトと task_local。ツールと `acp.rs` の間の境界 |
| `src/tools/client_tools.rs`（新規） | 6つのクライアント側ツール |
| `src/tools/mod.rs` | ツール一覧の絞り込み、新モジュールの宣言 |
| `src/serve/acp.rs` | `AcpClient` の実装、capability の記録 |
| `src/serve/mod.rs` | `TurnHost::acp_client()`、task_local のスコープ |
| `src/config.rs` | `[tools.host_access]` |

---

### Task 1: クライアントへの往復口を型にする

ツールが `acp.rs` を知らずにクライアントを呼べるようにする。**この Task には ACP の実物は出てこない** — トレイトと task_local と、テスト用の偽物だけ。

**Files:**
- Create: `src/tools/acp_client.rs`
- Modify: `src/tools/mod.rs`（`pub mod acp_client;`）
- Test: `src/tools/acp_client.rs` の `mod tests`

**Interfaces:**
- Produces: `trait AcpClient`（下記7メソッド）、`scope_acp_client<F>(client: Arc<dyn AcpClient>, fut: F) -> impl Future<Output = F::Output>`、`current_acp_client() -> Option<Arc<dyn AcpClient>>`、`TerminalHandle(String)`、`TerminalOutput { output, truncated, exit_status }`、`ExitStatus { exit_code: Option<u32>, signal: Option<String> }`

#### なぜトレイトか

ツールが `acp.rs` の `ConnectionTo` を直接握ると、テストに WebSocket が要る。トレイトにすれば偽物で駆動でき、`acp.rs` 側は実装を1箇所に閉じ込められる。

#### なぜ task_local か

`Tool::execute(&self, input: &Value)` にセッションの情報が無い。トレイトに引数を足すと全ツールが巻き込まれる。**`src/timer.rs` が同じ問題を同じ方法で解いている** — `scope_timer_origin` が `tools.execute(...)` を包み、timer ツールが `current_origin()` で読む。その形をそのまま踏襲する。

- [ ] **Step 1: 失敗するテストを書く**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    /// A stand-in for the editor. Records what it was asked and answers
    /// from a script, so the tools can be driven without a socket.
    #[derive(Default)]
    pub(crate) struct FakeClient {
        pub reads: Mutex<Vec<(String, Option<u32>, Option<u32>)>>,
        pub writes: Mutex<Vec<(String, String)>>,
        pub read_answer: Mutex<Option<Result<String, String>>>,
    }

    #[async_trait::async_trait]
    impl AcpClient for FakeClient {
        async fn read_text_file(
            &self,
            path: &str,
            line: Option<u32>,
            limit: Option<u32>,
        ) -> anyhow::Result<String> {
            self.reads
                .lock()
                .unwrap()
                .push((path.to_string(), line, limit));
            match self.read_answer.lock().unwrap().take() {
                Some(Ok(s)) => Ok(s),
                Some(Err(e)) => Err(anyhow::anyhow!(e)),
                None => Ok(String::new()),
            }
        }
        async fn write_text_file(&self, path: &str, content: &str) -> anyhow::Result<()> {
            self.writes
                .lock()
                .unwrap()
                .push((path.to_string(), content.to_string()));
            Ok(())
        }
        async fn create_terminal(
            &self,
            _command: &str,
            _args: &[String],
            _cwd: Option<&str>,
            _output_byte_limit: Option<u64>,
        ) -> anyhow::Result<TerminalHandle> {
            Ok(TerminalHandle("t1".to_string()))
        }
        async fn terminal_output(&self, _t: &TerminalHandle) -> anyhow::Result<TerminalOutput> {
            Ok(TerminalOutput::default())
        }
        async fn wait_for_terminal_exit(
            &self,
            _t: &TerminalHandle,
        ) -> anyhow::Result<ExitStatus> {
            Ok(ExitStatus::default())
        }
        async fn kill_terminal(&self, _t: &TerminalHandle) -> anyhow::Result<()> {
            Ok(())
        }
        async fn release_terminal(&self, _t: &TerminalHandle) -> anyhow::Result<()> {
            Ok(())
        }
    }

    /// Outside a scope there is no client — a channel or `/rpc` turn
    /// must not find one lying around from an earlier ACP turn.
    #[tokio::test]
    async fn there_is_no_client_outside_a_scope() {
        assert!(current_acp_client().is_none());
    }

    #[tokio::test]
    async fn a_scoped_client_is_visible_inside_and_gone_after() {
        let fake: Arc<dyn AcpClient> = Arc::new(FakeClient::default());
        scope_acp_client(Arc::clone(&fake), async {
            let seen = current_acp_client().expect("inside the scope");
            seen.write_text_file("/p/a.txt", "hi").await.unwrap();
        })
        .await;
        assert!(
            current_acp_client().is_none(),
            "the scope must not leak past its future"
        );
    }

    /// The scope has to survive being handed to a spawned task's await
    /// points, since a turn awaits the model between tool calls.
    #[tokio::test]
    async fn the_scope_survives_an_await() {
        let fake: Arc<dyn AcpClient> = Arc::new(FakeClient::default());
        scope_acp_client(fake, async {
            tokio::task::yield_now().await;
            assert!(current_acp_client().is_some(), "still scoped after a yield");
        })
        .await;
    }
}
```

- [ ] **Step 2: テストが落ちることを確認**

Run: `cargo test -p sapphire-agent acp_client`
Expected: FAIL — `AcpClient` が未定義。

- [ ] **Step 3: 実装を書く**

`src/tools/acp_client.rs`:

```rust
//! The agent's way of reaching the editor's machine.
//!
//! A tool's `execute` takes only its JSON input — it has no idea which
//! session called it, and the ACP connection lives per-connection in
//! `serve::acp`. Threading a session through `Tool` would touch every
//! tool for the benefit of six, so the handle is carried in a
//! `tokio::task_local` scoped around tool execution instead.
//!
//! `src/timer.rs` solves the same problem the same way: the turn loops
//! wrap `tools.execute(...)` in `scope_timer_origin` so the timer tool
//! can read where its call came from.
//!
//! This is a trait rather than the SDK's connection type so the tools
//! can be driven by a fake in tests, and so everything that knows about
//! ACP's wire types stays inside `serve::acp`.

use std::sync::Arc;

/// An opaque handle to a command running on the client's machine.
///
/// Opaque on purpose: the value is the client's, and nothing here may
/// parse or construct one except from what the client returned.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TerminalHandle(pub String);

impl std::fmt::Display for TerminalHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ExitStatus {
    pub exit_code: Option<u32>,
    pub signal: Option<String>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TerminalOutput {
    pub output: String,
    /// The client hit the byte limit we asked for and cut the output.
    pub truncated: bool,
    /// `None` while the command is still running.
    pub exit_status: Option<ExitStatus>,
}

/// What the editor can be asked to do on its own machine.
///
/// Every method maps 1:1 onto one ACP `agent → client` request. The
/// full set is exactly this — ACP has no directory listing, delete,
/// stat or rename, which is why there are no client-side tools for
/// those.
#[async_trait::async_trait]
pub trait AcpClient: Send + Sync {
    async fn read_text_file(
        &self,
        path: &str,
        line: Option<u32>,
        limit: Option<u32>,
    ) -> anyhow::Result<String>;

    async fn write_text_file(&self, path: &str, content: &str) -> anyhow::Result<()>;

    async fn create_terminal(
        &self,
        command: &str,
        args: &[String],
        cwd: Option<&str>,
        output_byte_limit: Option<u64>,
    ) -> anyhow::Result<TerminalHandle>;

    async fn terminal_output(&self, terminal: &TerminalHandle)
    -> anyhow::Result<TerminalOutput>;

    async fn wait_for_terminal_exit(
        &self,
        terminal: &TerminalHandle,
    ) -> anyhow::Result<ExitStatus>;

    /// Ends the command but keeps the handle usable, so the output can
    /// still be collected afterwards.
    async fn kill_terminal(&self, terminal: &TerminalHandle) -> anyhow::Result<()>;

    /// Frees the handle — **and kills the command if it is still
    /// running.** The protocol says so explicitly, which is why nothing
    /// may call this just because a connection went away.
    async fn release_terminal(&self, terminal: &TerminalHandle) -> anyhow::Result<()>;
}

tokio::task_local! {
    static ACP_CLIENT_TL: Arc<dyn AcpClient>;
}

/// Run `fut` with a client reachable from `current_acp_client`.
pub fn scope_acp_client<F: std::future::Future>(
    client: Arc<dyn AcpClient>,
    fut: F,
) -> impl std::future::Future<Output = F::Output> {
    ACP_CLIENT_TL.scope(client, fut)
}

/// The client for the turn currently executing, if it has one.
///
/// `None` on every non-ACP transport — `/rpc`, Matrix, Discord, voice —
/// which is what makes the client tools refuse there rather than
/// reaching for a connection that does not exist.
pub fn current_acp_client() -> Option<Arc<dyn AcpClient>> {
    ACP_CLIENT_TL.try_with(Arc::clone).ok()
}
```

`src/tools/mod.rs` に `pub mod acp_client;` を足す。

**`FakeClient` は Task 2 以降も使う。** `#[cfg(test)] pub(crate)` にして、`src/tools/client_tools.rs` のテストから `use crate::tools::acp_client::tests::FakeClient;` で参照できるようにすること。できない構成なら、共有できる場所（`acp_client.rs` の `#[cfg(test)] pub(crate) mod fake;` など）に置き直してよい — **重要なのは偽物が1つであることで、置き場所ではない。**

- [ ] **Step 4: テストが通ることを確認**

Run: `cargo test -p sapphire-agent acp_client`
Expected: PASS（3テスト）。

- [ ] **Step 5: コミット**

```bash
git checkout -- Cargo.lock
git add src/tools/acp_client.rs src/tools/mod.rs
git commit -m "feat(tools): a trait and a task-local for reaching the ACP client"
```

---

### Task 2: ホストを触るツールを opt-in にし、使えないツールを隠す

クライアント側ツールより先にこれを入れる。**現状の穴を塞ぐのが先**だからで、
クライアント側ツールの一覧絞り込みも同じ仕組みに乗る。

**Files:**
- Modify: `src/config.rs`
- Modify: `src/tools/policy.rs`
- Modify: `src/tools/mod.rs`
- Test: `src/tools/policy.rs` と `src/tools/mod.rs` の `mod tests`

**Interfaces:**
- Produces: `Config.tools.host_access.enabled: bool`（既定 `false`）、`policy::HOST_TOOLS: &[&str]`、`policy::host_tool_denied(name: &str, host_access_enabled: bool) -> bool`、`ToolSet::specs_filtered(&self, keep: impl Fn(&str) -> bool) -> Vec<ToolSpec>`

#### 今ある穴

`Origin::Channel` は `Execute` と `Other` だけを拒否している（`src/tools/policy.rs:184-190`）。
`file_write` は `Edit`、`file_delete` は `Delete` なので、**どちらも無条件に通る**。
つまり今、Discord から「このファイルを消して」が通る。この Task はそれも塞ぐ。

- [ ] **Step 1: 失敗するテストを書く**

`src/tools/policy.rs` の `mod tests` に:

```rust
    /// The seven tools that touch the agent's own machine. Off unless
    /// the operator turned them on — including for `Origin::Trusted`,
    /// which is the voice pipeline and the heartbeat.
    #[test]
    fn host_tools_are_denied_when_host_access_is_off() {
        for name in HOST_TOOLS {
            assert!(
                host_tool_denied(name, false),
                "{name} must be denied with host access off"
            );
        }
    }

    #[test]
    fn host_tools_are_allowed_through_when_host_access_is_on() {
        for name in HOST_TOOLS {
            assert!(
                !host_tool_denied(name, true),
                "{name} must fall through to the policy table when enabled"
            );
        }
    }

    /// The gate is about *which machine*, not about the tool's risk, so
    /// a workspace-scoped tool is never caught by it.
    #[test]
    fn workspace_tools_are_not_host_tools() {
        for name in ["memory_add", "workspace_search", "timer_set", "web_search"] {
            assert!(!host_tool_denied(name, false), "{name} is not a host tool");
        }
    }

    /// The hole this closes: `file_delete` is `Delete`, which
    /// `Origin::Channel` allows unasked, so a Discord message can
    /// delete a file on the agent's host today.
    #[test]
    fn a_channel_turn_cannot_reach_file_delete_with_host_access_off() {
        assert_eq!(
            decide(Origin::Channel, kind_of("file_delete", &kinds())),
            Decision::Allow,
            "the policy table alone still allows it — which is the point"
        );
        assert!(
            host_tool_denied("file_delete", false),
            "the host gate is what stops it"
        );
    }
```

`kinds()` は既存のテストヘルパに合わせること。無ければ `ToolKind::Delete` を直に渡す形でよい — 確かめたいのは「ポリシー表だけでは通る」ことと「ホストゲートが止める」ことの2点。

`src/tools/mod.rs` の `mod tests` に:

```rust
    #[tokio::test]
    async fn specs_filtered_keeps_only_what_the_predicate_allows() {
        let tools = ToolSet::new_empty_for_test();
        tools.register_tool(Box::new(NamedStub::new("keep_me"))).await;
        tools.register_tool(Box::new(NamedStub::new("drop_me"))).await;

        let names: Vec<String> = tools
            .specs_filtered(|n| n == "keep_me")
            .await
            .into_iter()
            .map(|s| s.name)
            .collect();
        assert_eq!(names, vec!["keep_me".to_string()]);
    }
```

`ToolSet::new_empty_for_test` と `NamedStub` が無ければ、既存の `Bare` スタブ（`src/tools/mod.rs:430` 付近）に倣って足すこと。

- [ ] **Step 2: テストが落ちることを確認**

Run: `cargo test -p sapphire-agent policy`
Expected: FAIL — `HOST_TOOLS` が未定義。

- [ ] **Step 3: 設定を足す**

`src/config.rs` に、既存の設定セクションの書き方に合わせて:

```rust
/// Whether the agent may touch the machine it runs on.
///
/// Off by default. On a self-hosted deployment this is the server, and
/// "read any file, run any command" is not something a Discord message
/// should reach by default. Turning it on is a deliberate act; running
/// the agent in a container is the recommended way to do it.
#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct HostAccess {
    #[serde(default)]
    pub enabled: bool,
}
```

`Config` の `tools` セクションに `host_access: HostAccess` として足す。`tools` セクションが無ければ作る。

- [ ] **Step 4: ゲートを足す**

`src/tools/policy.rs`:

```rust
/// The tools that operate on the agent's own filesystem and shell.
///
/// Listed by name rather than derived from `ToolKind`, because the
/// distinction is *which machine*, not how dangerous the operation is:
/// `memory_add` is also an `Edit`, and it is never in question.
pub const HOST_TOOLS: &[&str] = &[
    "file_read",
    "file_write",
    "file_append",
    "file_delete",
    "dir_list",
    "dir_walk",
    "shell",
];

/// Whether this call is refused before the policy table is consulted.
///
/// A gate in front of `decide` rather than a row inside it: `decide` is
/// a pure function of origin and kind, and this is a fact about the
/// deployment. Keeping them apart means the permission table still
/// reads as one thing.
pub fn host_tool_denied(name: &str, host_access_enabled: bool) -> bool {
    !host_access_enabled && HOST_TOOLS.contains(&name)
}
```

- [ ] **Step 5: ゲートを呼び出し側に繋ぐ**

`src/serve/mod.rs` の許可判定と `src/serve/acp.rs` の `approve` の**手前**に置く。
`grep -n "partition_without_asking\|fn approve" src/` で両方の入口を出し、
`host_tool_denied` が真なら `Refusal::Unavailable` 相当で断るようにする。

**`src/agent.rs` は編集しないこと。** channel 経路は `partition_without_asking` を
通るので、そちらに入れれば channel も塞がる。**もし channel 経路が
`src/agent.rs` 内でしかゲートを通らないなら、そのことを報告して止まること** —
`agent.rs` を触らずに塞げないなら、それは私が裁定すべき plan の欠陥である。

- [ ] **Step 6: 一覧の絞り込みを足す**

`src/tools/mod.rs`:

```rust
    /// The specs a particular turn should see.
    ///
    /// A tool the caller cannot use is worse than absent: the model
    /// spends a round trip discovering the refusal, and on an ACP
    /// session it may pick the host's `file_read` when it meant the
    /// editor's.
    pub async fn specs_filtered(&self, keep: impl Fn(&str) -> bool) -> Vec<ToolSpec> {
        self.inner
            .read()
            .await
            .specs
            .iter()
            .filter(|s| keep(&s.name))
            .cloned()
            .collect()
    }
```

`run_llm_turn` の `let tool_specs = state.tools.specs().await;` を
`specs_filtered` に置き換え、ホストツールは `host_access.enabled` のときだけ残す。
クライアント側ツールの条件は Task 4 で足す。

- [ ] **Step 7: テストが通ることを確認**

Run: `cargo test -p sapphire-agent`
Expected: PASS。既存テストがホストツールの利用を前提にしていたら、
**削らずに** `host_access.enabled = true` の設定で組み立て直すこと。
どのテストが何を前提にしていたかを報告に書く。

- [ ] **Step 8: コミット**

```bash
git checkout -- Cargo.lock
git add src/config.rs src/tools/policy.rs src/tools/mod.rs src/serve/mod.rs src/serve/acp.rs
git commit -m "feat(tools): make the agent's own filesystem and shell opt-in"
```

---

### Task 3: `AcpClient` を実物の接続で実装し、capability を記録する

**Files:**
- Modify: `src/serve/acp.rs`
- Modify: `src/serve/mod.rs`
- Test: `src/serve/acp.rs` の `mod tests`

**Interfaces:**
- Consumes: Task 1 の `AcpClient` / `TerminalHandle` / `TerminalOutput` / `ExitStatus`
- Produces: `AcpTurnHost` が `AcpClient` を実装、`TurnHost::acp_client(&self) -> Option<Arc<dyn AcpClient>>`（既定 `None`）、`AcpSession.client_capabilities: ClientCapabilities`

#### ハンドルの出どころ

新しい登録簿を作らない。**`run_llm_turn` は既に `progress: Arc<dyn TurnHost>` を
持っており、ACP ターンではそれが接続の包み**（`AcpTurnHost`）である。
`TurnHost` に既定 `None` のメソッドを1つ足すのは、`origin()` が既に取っている形と同じ。

- [ ] **Step 1: 失敗するテストを書く**

```rust
    /// The client's declared capabilities have to survive `initialize`,
    /// because which client tools exist is decided per session from
    /// them. They were being parsed and dropped.
    #[tokio::test]
    async fn initialize_records_the_clients_capabilities() {
        // 既存の initialize テスト（`initialize_answers_with_v1_capabilities`
        // 付近）の組み立てに倣い、fs.readTextFile だけ true、
        // fs.writeTextFile と terminal を false にした initialize を送る。
        // その後 session/new し、セッションに記録された capability が
        // read=true / write=false / terminal=false であることを確かめる。
        //
        // 記録先が private なら、確かめる手段は Task 4 の一覧絞り込み
        // ごしでよい。その場合はこのテストを Task 4 に移し、
        // 報告にそう書くこと。
    }

    /// A non-ACP turn must not find a client. `/rpc`, Matrix, Discord
    /// and voice have no editor on the other end.
    #[test]
    fn the_default_turn_host_offers_no_client() {
        assert!(NullProgress.acp_client().is_none());
    }
```

- [ ] **Step 2: テストが落ちることを確認**

Run: `cargo test -p sapphire-agent acp`
Expected: FAIL — `acp_client` が `TurnHost` に無い。

- [ ] **Step 3: `TurnHost` に足す**

`src/serve/mod.rs`:

```rust
    /// The editor on the other end, when there is one.
    ///
    /// `None` by default: `/rpc`, `/a2a`, Matrix, Discord and the voice
    /// pipeline have no client machine to reach. The client-side tools
    /// read this through a task-local and refuse when it is absent, so
    /// a default of `None` is what keeps them ACP-only.
    fn acp_client(&self) -> Option<Arc<dyn crate::tools::acp_client::AcpClient>> {
        None
    }
```

- [ ] **Step 4: capability を記録する**

`src/serve/acp.rs` の `initialize` ハンドラで `req.client_capabilities` を
接続スコープに保持し、`session/new` と adopt の両方で `AcpSession` に載せる。
`AcpSession.cwd` の隣に置く。

- [ ] **Step 5: `AcpClient` を実装する**

`AcpTurnHost` に `impl AcpClient`。各メソッドは spec の対応表どおり1本ずつ
`self.connection.send_request(...).block_task().await` する。`approve`
（`src/serve/acp.rs:325` 付近）が唯一の先例なので、**その形を読んでから書くこと** —
特に「ハンドラ内で await しない」根拠のコメントは、なぜこれが安全かを説明している。

エラーは `anyhow::Error` に変換して返す。**握り潰さない** — クライアント側の
権限やパスの問題は、モデルが読んで別の手を試せる情報である。

`TurnHost::acp_client` を `AcpTurnHost` で override し、`Some(Arc::new(...))` を返す。
`AcpTurnHost` 自身が `AcpClient` なら `Arc<Self>` をそのまま返せる形にしてよい。

- [ ] **Step 6: task_local を張る**

`src/serve/mod.rs` の `run_llm_turn` で、ツール実行を包む。
既存の `crate::timer::scope_timer_origin(o, fut).await` の隣で、
`progress.acp_client()` が `Some` のときだけ `scope_acp_client` を重ねる。

**両方のスコープが同時に要る**ので、入れ子にすること。片方だけになると
timer ツールかクライアントツールのどちらかが壊れる。

- [ ] **Step 7: テストが通ることを確認**

Run: `cargo test -p sapphire-agent`
Expected: PASS。

- [ ] **Step 8: コミット**

```bash
git checkout -- Cargo.lock
git add src/serve/acp.rs src/serve/mod.rs
git commit -m "feat(acp): implement the client handle and record what the client can do"
```

---

### Task 4: クライアント側のファイル読み書き

**Files:**
- Create: `src/tools/client_tools.rs`
- Modify: `src/tools/mod.rs`
- Modify: `src/serve/mod.rs`（一覧絞り込みの条件）
- Test: `src/tools/client_tools.rs` の `mod tests`

**Interfaces:**
- Consumes: Task 1 の `AcpClient` / `current_acp_client` / `FakeClient`、Task 2 の `specs_filtered`
- Produces: `ClientFileRead`、`ClientFileWrite`（どちらも `Tool` を実装）

- [ ] **Step 1: 失敗するテストを書く**

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::acp_client::{AcpClient, scope_acp_client};
    use serde_json::json;
    use std::sync::Arc;

    /// `line` and `limit` exist in ACP because a coding agent reads big
    /// files in pieces. Passing them through is the reason to prefer
    /// this over shelling out to `sed`.
    #[tokio::test]
    async fn read_passes_line_and_limit_through() {
        let fake = Arc::new(FakeClient::default());
        let client: Arc<dyn AcpClient> = Arc::clone(&fake) as Arc<dyn AcpClient>;
        scope_acp_client(client, async {
            ClientFileRead::new()
                .execute(&json!({"path": "/p/a.rs", "line": 10, "limit": 40}))
                .await
                .unwrap();
        })
        .await;
        assert_eq!(
            fake.reads.lock().unwrap().as_slice(),
            &[("/p/a.rs".to_string(), Some(10), Some(40))]
        );
    }

    /// Outside an ACP turn there is no editor. Refusing here is what
    /// keeps a Discord message from reaching a tool that would have
    /// nowhere to go.
    #[tokio::test]
    async fn a_client_tool_refuses_without_a_client() {
        let err = ClientFileRead::new()
            .execute(&json!({"path": "/p/a.rs"}))
            .await
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("no editor"),
            "the message should say why, got: {err}"
        );
    }

    /// The editor's refusal is information, not a failure to swallow:
    /// the model can read it and try something else.
    #[tokio::test]
    async fn the_clients_error_reaches_the_model() {
        let fake = Arc::new(FakeClient::default());
        *fake.read_answer.lock().unwrap() = Some(Err("permission denied".to_string()));
        let client: Arc<dyn AcpClient> = Arc::clone(&fake) as Arc<dyn AcpClient>;
        scope_acp_client(client, async {
            let err = ClientFileRead::new()
                .execute(&json!({"path": "/p/secret"}))
                .await
                .unwrap_err()
                .to_string();
            assert!(err.contains("permission denied"), "got: {err}");
        })
        .await;
    }

    #[tokio::test]
    async fn write_sends_the_path_and_content() {
        let fake = Arc::new(FakeClient::default());
        let client: Arc<dyn AcpClient> = Arc::clone(&fake) as Arc<dyn AcpClient>;
        scope_acp_client(client, async {
            ClientFileWrite::new()
                .execute(&json!({"path": "/p/b.rs", "content": "fn main() {}"}))
                .await
                .unwrap();
        })
        .await;
        assert_eq!(
            fake.writes.lock().unwrap().as_slice(),
            &[("/p/b.rs".to_string(), "fn main() {}".to_string())]
        );
    }

    #[test]
    fn the_kinds_match_what_the_permission_table_expects() {
        assert_eq!(ClientFileRead::new().kind(), ToolKind::Read);
        assert_eq!(ClientFileWrite::new().kind(), ToolKind::Edit);
    }
}
```

そして `src/serve/mod.rs` の `mod tests` に、一覧の絞り込みを確かめるものを3本。
**これが「モデルがどちらのマシンか取り違えない」ことを保証している唯一の仕掛け**なので、
条件を書くだけでなく確かめる。

```rust
    /// A client that says it can read but not write gets exactly one of
    /// the two file tools. Clients implement these independently, so
    /// the two flags are read separately rather than as one "fs" bit.
    #[tokio::test]
    async fn the_two_fs_tools_follow_their_own_capability_flags() {
        let names = tool_names_for_turn(TestCaps {
            fs_read: true,
            fs_write: false,
            terminal: false,
        })
        .await;
        assert!(names.contains(&"client_file_read".to_string()));
        assert!(!names.contains(&"client_file_write".to_string()));

        let names = tool_names_for_turn(TestCaps {
            fs_read: false,
            fs_write: true,
            terminal: false,
        })
        .await;
        assert!(!names.contains(&"client_file_read".to_string()));
        assert!(names.contains(&"client_file_write".to_string()));
    }

    /// Matrix, Discord, `/rpc` and voice have no editor. Offering them a
    /// tool that can only fail wastes a round trip and invites the model
    /// to pick the wrong machine.
    #[tokio::test]
    async fn a_non_acp_turn_is_offered_no_client_tools() {
        let names = tool_names_for_turn_without_a_client().await;
        assert!(
            !names.iter().any(|n| n.starts_with("client_")),
            "got: {names:?}"
        );
    }

    /// The host switch is off by default, so its seven tools are absent
    /// from an ordinary turn's list entirely — not offered and refused.
    #[tokio::test]
    async fn host_tools_are_absent_when_host_access_is_off() {
        let names = tool_names_for_turn_without_a_client().await;
        for name in crate::tools::policy::HOST_TOOLS {
            assert!(!names.contains(&name.to_string()), "{name} should be hidden");
        }
    }
```

`tool_names_for_turn` / `tool_names_for_turn_without_a_client` / `TestCaps` は
このタスクで作るヘルパ。`specs_filtered` に渡す述語を、`run_llm_turn` が組み立てるのと
**同じ関数**で作り、テストはその関数を呼ぶこと — 述語を組み立てる部分をテスト用に
書き直すと、確かめているのがテストの複製になってしまう。**述語の組み立てを
`fn visible_tool_predicate(...)` のような名前付き関数に切り出し、
`run_llm_turn` とテストの両方がそれを呼ぶ。**

- [ ] **Step 2: テストが落ちることを確認**

Run: `cargo test -p sapphire-agent client_tools`
Expected: FAIL — `ClientFileRead` が未定義。

- [ ] **Step 3: 2つのツールを実装する**

`src/tools/client_tools.rs`。既存のツール（`src/tools/builtin_tools.rs`）の
`Tool` 実装に倣う。要点だけ:

```rust
/// Read a file on the machine the editor is running on.
///
/// Distinct from `file_read`, which reads the machine the *agent* runs
/// on. In an ACP session only this one is offered, so the model cannot
/// pick the wrong machine.
pub struct ClientFileRead { spec: ToolSpec }
```

説明文（`description`）に**どちらのマシンか**を明記すること。モデルが読む唯一の
手がかりである。`limit` については「大きなファイルは `line` と `limit` で部分的に
読むこと。ACP は内容を丸ごと回線に流すので、全部読むのは高い」と書く。

`current_acp_client()` が `None` のときのエラー文は
`"no editor is connected to this session; this tool only works over ACP"` の
ように、**なぜ使えないかが分かる**ものにする（テストが `"no editor"` を見る）。

- [ ] **Step 4: 登録と絞り込みを繋ぐ**

`build_default_tools`（`src/tools/mod.rs:215` 付近）で2つを登録する。
`src/serve/mod.rs` の `specs_filtered` の述語に条件を足す:

- `client_file_read` は、そのターンにクライアントがあり `fs.read_text_file` が真のときだけ
- `client_file_write` は、そのターンにクライアントがあり `fs.write_text_file` が真のときだけ

capability は Task 3 で `AcpSession` に記録したもの。`run_llm_turn` から
読める形に渡す必要があるなら、`TurnHost` にもう1つ既定実装付きのメソッドを
足してよい（`fn client_fs_caps(&self) -> (bool, bool) { (false, false) }` など）。
**新しい登録簿は作らないこと** — Task 3 と同じ理由で、`TurnHost` が既に
そのターンの相手を知っている。

- [ ] **Step 5: テストが通ることを確認**

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
git add src/tools/client_tools.rs src/tools/mod.rs src/serve/mod.rs
git commit -m "feat(tools): read and write files on the editor's machine"
```

---

### Task 5: クライアント側シェル — 一発実行

**Files:**
- Modify: `src/tools/client_tools.rs`
- Modify: `src/serve/mod.rs`
- Test: `src/tools/client_tools.rs` の `mod tests`

**Interfaces:**
- Consumes: Task 1 の terminal 系メソッド
- Produces: `ClientShell`（`Tool`、`ToolKind::Execute`）

#### タイムアウトしても殺さない

既定120秒、`timeout_secs` で上書き可、上限600秒。

時間内に終われば `output` → `release` して結果を返す。**時間切れなら `release`
しない** — `release` はコマンドを殺すので、2分かけたビルドが捨てられ、
冪等でないコマンドなら再実行で二重に走る。ハンドルを結果に載せて返し、
モデルが `client_shell_output` で追えるようにする。

プロトコル自身は `terminal/kill` の説明で「タイムアウトしたら殺して出力を取る」を
示唆しているが、採らない。殺す判断はモデル（`client_shell_kill`）に委ねる。

- [ ] **Step 1: 失敗するテストを書く**

`FakeClient` に、終了を遅らせられる仕掛けを足す（`wait_for_terminal_exit` が
指定回数だけ「まだ」を返す、あるいは所定時間眠る）。既存の `FakeClient` を
拡張し、**Task 1 のテストを壊さないこと**。

```rust
    /// The whole point of the timeout: a build that outruns it keeps
    /// running, and the model is handed the handle instead of a
    /// corpse. Killing here would throw away the work and, for a
    /// non-idempotent command, run it twice.
    #[tokio::test]
    async fn a_timed_out_command_is_not_killed_and_hands_back_its_handle() {
        let fake = Arc::new(FakeClient::default());
        fake.make_exit_never_return();
        let client: Arc<dyn AcpClient> = Arc::clone(&fake) as Arc<dyn AcpClient>;

        let out = scope_acp_client(client, async {
            ClientShell::new()
                .execute(&json!({"command": "cargo", "args": ["test"], "timeout_secs": 1}))
                .await
                .unwrap()
        })
        .await;

        assert!(out.contains("still running"), "got: {out}");
        assert!(out.contains("t1"), "the handle must be in the result: {out}");
        assert!(
            fake.released.lock().unwrap().is_empty(),
            "release kills the command — it must not be called on a timeout"
        );
        assert!(fake.killed.lock().unwrap().is_empty(), "nor kill");
    }

    #[tokio::test]
    async fn a_command_that_finishes_in_time_is_released() {
        let fake = Arc::new(FakeClient::default());
        let client: Arc<dyn AcpClient> = Arc::clone(&fake) as Arc<dyn AcpClient>;
        scope_acp_client(client, async {
            ClientShell::new()
                .execute(&json!({"command": "ls", "args": []}))
                .await
                .unwrap();
        })
        .await;
        assert_eq!(fake.released.lock().unwrap().len(), 1, "the handle is freed");
    }

    /// The cap is handed to the client so the output is cut at the
    /// source rather than shipped across the wire and cut here.
    #[tokio::test]
    async fn the_output_cap_is_passed_to_the_client() {
        let fake = Arc::new(FakeClient::default());
        let client: Arc<dyn AcpClient> = Arc::clone(&fake) as Arc<dyn AcpClient>;
        scope_acp_client(client, async {
            ClientShell::new()
                .execute(&json!({"command": "ls", "args": []}))
                .await
                .unwrap();
        })
        .await;
        let (_, _, _, limit) = fake.creates.lock().unwrap()[0].clone();
        assert_eq!(limit, Some(crate::tools::OUTPUT_CAP_BYTES as u64));
    }

    #[tokio::test]
    async fn the_timeout_is_capped_at_ten_minutes() {
        // timeout_secs: 9999 を渡しても 600 を超えないことを確かめる。
        // 実際に待たずに済むよう、上限計算だけを切り出した関数
        // `clamp_timeout(requested: Option<u64>) -> Duration` を対象にする。
    }
```

- [ ] **Step 2: テストが落ちることを確認**

Run: `cargo test -p sapphire-agent client_tools`
Expected: FAIL — `ClientShell` が未定義。

- [ ] **Step 3: 実装する**

```rust
/// The cap on how long the one-shot form waits. Past this the command
/// keeps running and the caller gets its handle: see the module docs.
const DEFAULT_TIMEOUT_SECS: u64 = 120;
const MAX_TIMEOUT_SECS: u64 = 600;

fn clamp_timeout(requested: Option<u64>) -> std::time::Duration {
    std::time::Duration::from_secs(requested.unwrap_or(DEFAULT_TIMEOUT_SECS).min(MAX_TIMEOUT_SECS))
}
```

`execute` の流れ: `create_terminal(command, args, cwd, Some(OUTPUT_CAP_BYTES))` →
`tokio::time::timeout(clamp_timeout(..), client.wait_for_terminal_exit(&h))` →

- `Ok(status)`: `terminal_output` → `release_terminal` → 出力・終了コード・
  `truncated` だったらその旨を含む文字列を返す
- `Err(_)`（時間切れ）: **何も解放せず**、ハンドルを含む案内を返す

案内の文言は、**まだ走っていることが読み違えようのない**ものにすること:

```
[timed out after 120s — the command is STILL RUNNING as terminal t1.
 It was not killed. Use client_shell_output to check on it, or
 client_shell_kill to stop it. Do not re-run the command.]
```

最後の一文が要る。モデルが失敗と読んで再実行するのが、この設計で唯一の
新しい危険だと spec が名指ししている。

`cwd` は指定が無ければセッションの `cwd`。`TurnHost` 経由で渡す形にする
（Task 4 の capability と同じ流儀）。

- [ ] **Step 4: 登録と絞り込み**

`terminal` capability が真のときだけ一覧に出す。

- [ ] **Step 5: テストが通ることを確認**

Run: `cargo test -p sapphire-agent`
Expected: PASS。

- [ ] **Step 6: コミット**

```bash
git checkout -- Cargo.lock
git add src/tools/client_tools.rs src/tools/mod.rs src/serve/mod.rs
git commit -m "feat(tools): run a command on the editor's machine"
```

---

### Task 6: ライフサイクル版と、セッションに紐づくハンドル追跡

**Files:**
- Modify: `src/tools/client_tools.rs`
- Modify: `src/serve/mod.rs`
- Test: `src/tools/client_tools.rs` の `mod tests`

**Interfaces:**
- Produces: `ClientShellStart`（`Execute`）、`ClientShellOutput`（`Read`）、`ClientShellKill`（`Execute`）、`ServeState.acp_terminals: Mutex<HashMap<String, Vec<TerminalHandle>>>`

#### 追跡はセッションに紐づける、接続ではなく

**接続が切れてもターミナルは解放しない。** `release` はコマンドを殺すので、
回線が一瞬詰まっただけで相手の `cargo test` が死ぬ。そして ACP のターミナルは
`session_id` で識別され、セッションは `session/load` で再接続を跨いで生き残る。

したがって追跡は `ServeState` にセッション id をキーとして置く。接続ごとの
`AcpSession` に持たせると再接続で一覧を失い、残ったターミナルがこちらから
見えなくなる。

解放するのは3つの場合だけ: 一発実行が時間内に終わったとき、モデルが
`client_shell_kill` を呼んだとき、クライアントが「そのハンドルは無い」と
応答したとき（追跡から落とすだけ）。

上限は8。達したら新規作成を断り、**残っているハンドルを列挙する** —
モデルが片付ける先を知る必要がある。

- [ ] **Step 1: 失敗するテストを書く**

```rust
```rust
    /// The handle has to be recorded against the session, because that
    /// is what a reconnecting client's next turn will look it up by.
    #[tokio::test]
    async fn start_records_the_handle_against_the_session() {
        let (state, fake) = shell_test_state().await;
        let client: Arc<dyn AcpClient> = Arc::clone(&fake) as Arc<dyn AcpClient>;
        scope_acp_client(client, async {
            ClientShellStart::new()
                .execute(&json!({"command": "cargo", "args": ["test"]}))
                .await
                .unwrap();
        })
        .await;

        let held = state.acp_terminals.lock().await;
        assert_eq!(held.get(TEST_SESSION_ID).map(Vec::len), Some(1));
    }

    /// `kill` alone leaves the handle valid — the protocol says so, and
    /// says to release it afterwards. Doing only half would leak a
    /// handle against the cap forever.
    #[tokio::test]
    async fn kill_stops_the_command_and_then_frees_the_handle() {
        let (state, fake) = shell_test_state().await;
        let client: Arc<dyn AcpClient> = Arc::clone(&fake) as Arc<dyn AcpClient>;
        scope_acp_client(client, async {
            ClientShellStart::new()
                .execute(&json!({"command": "sleep", "args": ["999"]}))
                .await
                .unwrap();
            ClientShellKill::new()
                .execute(&json!({"terminal": "t1"}))
                .await
                .unwrap();
        })
        .await;

        assert_eq!(fake.killed.lock().unwrap().len(), 1, "the command is stopped");
        assert_eq!(fake.released.lock().unwrap().len(), 1, "and the handle freed");
        assert!(
            state
                .acp_terminals
                .lock()
                .await
                .get(TEST_SESSION_ID)
                .is_none_or(Vec::is_empty),
            "and it is no longer tracked"
        );
    }

    /// The cap has to name what is holding it. A bare "too many
    /// terminals" leaves the model with nothing to act on.
    #[tokio::test]
    async fn the_cap_refuses_and_lists_what_is_held() {
        let (_state, fake) = shell_test_state().await;
        fake.hand_out_distinct_handles();
        let client: Arc<dyn AcpClient> = Arc::clone(&fake) as Arc<dyn AcpClient>;

        let refusal = scope_acp_client(client, async {
            for _ in 0..MAX_TERMINALS_PER_SESSION {
                ClientShellStart::new()
                    .execute(&json!({"command": "sleep", "args": ["999"]}))
                    .await
                    .unwrap();
            }
            ClientShellStart::new()
                .execute(&json!({"command": "one", "args": ["too", "many"]}))
                .await
                .unwrap_err()
                .to_string()
        })
        .await;

        assert!(refusal.contains("t1"), "names a held handle: {refusal}");
        assert!(
            refusal.contains(&format!("t{MAX_TERMINALS_PER_SESSION}")),
            "names the last one too: {refusal}"
        );
        assert_eq!(
            fake.creates.lock().unwrap().len(),
            MAX_TERMINALS_PER_SESSION,
            "the refused call must not have reached the client"
        );
    }

    /// A handle the client has forgotten is dropped here too, so the
    /// cap does not fill with ghosts after a client restart.
    #[tokio::test]
    async fn an_unknown_handle_is_dropped_from_tracking() {
        let (state, fake) = shell_test_state().await;
        fake.make_output_fail_with("no such terminal");
        let client: Arc<dyn AcpClient> = Arc::clone(&fake) as Arc<dyn AcpClient>;

        let err = scope_acp_client(client, async {
            ClientShellStart::new()
                .execute(&json!({"command": "sleep", "args": ["999"]}))
                .await
                .unwrap();
            ClientShellOutput::new()
                .execute(&json!({"terminal": "t1"}))
                .await
                .unwrap_err()
                .to_string()
        })
        .await;

        assert!(err.contains("no such terminal"), "the client's words reach the model: {err}");
        assert!(
            state
                .acp_terminals
                .lock()
                .await
                .get(TEST_SESSION_ID)
                .is_none_or(Vec::is_empty),
            "a handle the client disowns stops counting against the cap"
        );
    }

    /// The property this whole task is shaped around. `terminal/release`
    /// kills the command, so a dropped socket must not trigger one — a
    /// network blip would otherwise kill the user's build.
    #[tokio::test]
    async fn a_connection_ending_releases_nothing() {
        let (state, fake) = shell_test_state().await;
        let client: Arc<dyn AcpClient> = Arc::clone(&fake) as Arc<dyn AcpClient>;
        scope_acp_client(client, async {
            ClientShellStart::new()
                .execute(&json!({"command": "cargo", "args": ["test"]}))
                .await
                .unwrap();
        })
        .await;

        simulate_connection_teardown(&state, TEST_SESSION_ID).await;

        assert!(
            fake.released.lock().unwrap().is_empty(),
            "release kills the command; a lost socket is not a reason to"
        );
        assert!(fake.killed.lock().unwrap().is_empty());
        assert_eq!(
            state.acp_terminals.lock().await.get(TEST_SESSION_ID).map(Vec::len),
            Some(1),
            "and the handle stays addressable for a client that reconnects"
        );
    }
```

`shell_test_state()` は `ServeState` と `Arc<FakeClient>` を返すヘルパ、
`TEST_SESSION_ID` はそのセッション id、`simulate_connection_teardown` は
`src/serve/acp.rs` の接続終了処理が呼ぶものを呼ぶ薄いラッパ。

**`simulate_connection_teardown` が「何もしない関数」になるなら、それでよい** —
確かめているのは*何が起きないか*なので、呼び出し先が空でもテストは意味を持つ。
ただし**接続終了処理そのものを呼ぶこと**。テストが独自に「何もしない」を書くと、
将来そこに解放を足した人を止められない。

`FakeClient` にこの Task で足すもの: `killed`、`released`、`creates`（引数の記録）、
`hand_out_distinct_handles()`、`make_output_fail_with(&str)`。
**Task 1 と Task 5 の既存テストを壊さないこと。**

- [ ] **Step 2: テストが落ちることを確認**

Run: `cargo test -p sapphire-agent client_tools`
Expected: FAIL。

- [ ] **Step 3: 3つのツールと追跡を実装する**

`src/tools/client_tools.rs` に:

```rust
/// How many terminals one session may hold at once.
///
/// A ceiling rather than a cleanup: nothing here is released on
/// disconnect, so without a cap a model that keeps starting commands
/// and never killing them would accumulate processes on the user's
/// machine indefinitely. Refusing the ninth — and naming the eight it
/// is holding — makes the model clean up rather than the agent guess
/// which one is safe to kill.
pub(crate) const MAX_TERMINALS_PER_SESSION: usize = 8;
```

`ServeState` に:

```rust
    /// Terminals the model started and has not cleaned up, per agent
    /// session id.
    ///
    /// Keyed by session rather than by connection on purpose. ACP
    /// terminals are addressed by `session_id`, a session outlives a
    /// connection via `session/load`, and `terminal/release` kills the
    /// command — so releasing these because a socket dropped would kill
    /// a build over a network blip. Nothing here is cleaned up on
    /// disconnect; see the module docs.
    pub(crate) acp_terminals: tokio::sync::Mutex<HashMap<String, Vec<TerminalHandle>>>,
```

- [ ] **Step 4: テストが通ることを確認**

Run: `cargo test -p sapphire-agent`
Expected: PASS。

- [ ] **Step 5: 接続終了時に何もしないことを確かめる**

`src/serve/acp.rs` の接続終了処理に、**ターミナルを解放するコードを足さないこと。**
そのうえで、そこにコメントを残す:

```rust
    // Terminals are deliberately NOT released here. `terminal/release`
    // kills the command, and a dropped socket is not a reason to kill a
    // build running on the user's machine. The handles stay tracked
    // against the session, so a client that reconnects and loads the
    // session can still reach them.
```

- [ ] **Step 6: フォーマットと lint**

```bash
cargo fmt --all
cargo clippy --workspace --all-targets -- -D warnings
```

- [ ] **Step 7: コミット**

```bash
git checkout -- Cargo.lock
git add src/tools/client_tools.rs src/serve/mod.rs src/serve/acp.rs
git commit -m "feat(tools): long-running commands on the editor's machine"
```

---

### Task 7: ドキュメントと全体確認

**Files:**
- Modify: `README.md`
- Modify: `docs/superpowers/specs/2026-09-01-client-side-tools-design.md`

- [ ] **Step 1: README を書く**

- クライアント側の6ツールと、**それぞれがどちらのマシンを触るか**
- ホストを触る7ツールは `[tools.host_access] enabled = false` が既定で、
  有効化は意図的な行為であること。コンテナでの実行が推奨であること
- **これが塞いだ穴**: 従来 Discord から `file_write` と `file_delete` が
  無条件に通っていたこと
- 一発シェルは既定120秒で、**タイムアウトしてもコマンドは殺されない**こと
- ディレクトリ列挙・削除はクライアント側に無く、シェルで行うこと。
  ACP にその面が無いため

- [ ] **Step 2: spec に実装時の訂正を追記する**

spec 本文は書き換えず、末尾に `## 実装時の訂正` を新設する。
**訂正が無ければ「無し」と1行書く** — 節ごと省かないこと。後から読む人が
「確認されなかった」のか「確認して何も無かった」のかを区別できる。

- [ ] **Step 3: ワークスペース全体の確認**

```bash
cargo test --workspace
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
```

- [ ] **Step 4: コミット**

```bash
git checkout -- Cargo.lock
git add README.md docs/superpowers/specs/2026-09-01-client-side-tools-design.md
git commit -m "docs: describe the client-side tools and the host-access switch"
```
