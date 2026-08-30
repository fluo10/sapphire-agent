# ツール実行の承認と、セッションモード Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** ACP の `session/request_permission` と `session/set_mode` を実装し、ツール実行に承認ゲートを入れる。

**Architecture:** ツールが ACP の `ToolKind` を自己申告し、純関数 `tools::policy::decide(origin, kind)` が `Allow` / `Ask` / `Deny` を返す。ゲートは 2 本あるツールループ（`run_llm_turn` と `agent.rs`）の両方に差すが、判定関数は 1 つだけ。`Ask` の対話は `TurnHost`（旧 `TurnProgress`）のデフォルト実装付きメソッドに乗るので、ACP 以外の実装は変更不要。

**Tech Stack:** Rust 2024, `agent-client-protocol` 2.0.0, `tokio`, `async_trait`, `serde_json`, `tempfile`

**Spec:** `docs/superpowers/specs/2026-08-30-acp-permissions-design.md`

## Global Constraints

- ブランチは `feat/acp-permissions`（作成済み、spec はコミット済み）。
- テストコマンドは `cargo test --workspace`。テストはソースファイル内の `mod tests` に置く（このリポジトリの既存作法）。
- モード ID は正確に `default` / `accept_edits` / `bypass` の 3 つ。`plan` は**実装しない**。
- 永続化先は `dirs.config_dir().join("acp-permissions.json")`。ワークスペース配下には置かない。
- `always_reject` は `always_allow` より優先（安全側）。
- `Tool::kind()` の既定は `ToolKind::Other`（最も厳しい側）。
- 判定表（`Origin` × `ToolKind`）は spec の表と完全に一致させること:

| ToolKind | ACP `default` | ACP `accept_edits` | ACP `bypass` | `Channel` | `Trusted` |
|---|---|---|---|---|---|
| `Read` / `Search` / `Fetch` / `Think` | Allow | Allow | Allow | Allow | Allow |
| `Edit` / `Delete` / `Move` | Ask | Allow | Allow | Allow | Allow |
| `Execute` / `Other` | Ask | Ask | Allow | Deny | Allow |

- `Deny` / 拒否のときもモデルには `tool_result` を返す。**ターンは止めない。** `StopReason::Refusal` は使わない。
- 既存挙動を変えてよいのは Matrix / Discord 経路のみ（`shell` と MCP ツールが拒否される）。`/rpc`・voice・heartbeat・`/a2a` は不変。

---

### Task 1: `Tool::kind()` と全ツールの分類

ACP の `ToolKind` をツール層に持ち込み、全ツールに割り当てる。既定は `Other` なので、割り当て忘れは「最も厳しい側に落ちる」＝安全側に失敗する。

**Files:**
- Modify: `src/tools/mod.rs`（trait にメソッド追加、`ToolKind` 再エクスポート、`ToolSet::kinds()`）
- Modify: `src/tools/builtin_tools.rs`
- Modify: `src/tools/workspace_tools.rs`
- Modify: `src/tools/timer_tools.rs`
- Modify: `src/tools/ambient_tools.rs`
- Modify: `src/serve/mod.rs`（`EchoTool` — **これを忘れると既存の ACP テストがハングする**）
- Test: `src/tools/mod.rs` の `mod tests`

**Interfaces:**
- Produces: `crate::tools::ToolKind`（`agent_client_protocol::schema::v1::ToolKind` の再エクスポート）、`Tool::kind(&self) -> ToolKind`、`ToolSet::kinds(&self) -> Vec<(String, ToolKind)>`

- [ ] **Step 1: `Tool` trait にメソッドを足す**

`src/tools/mod.rs`。既存の `use` 群の隣に再エクスポートを足し、trait にデフォルト実装付きメソッドを 1 つ追加する。

```rust
pub use agent_client_protocol::schema::v1::ToolKind;
```

```rust
#[async_trait]
pub trait Tool: Send + Sync {
    /// The spec advertised to the LLM.
    fn spec(&self) -> &ToolSpec;

    /// What this tool does, in ACP's vocabulary. Drives both the
    /// `session/update` display and the permission policy, so there is
    /// exactly one classification rather than two that drift apart.
    ///
    /// The default is `Other` — the strictest bucket — so a tool added
    /// without a `kind()` fails safe: it asks (ACP) or is refused
    /// (channels) rather than silently running unguarded.
    fn kind(&self) -> ToolKind {
        ToolKind::Other
    }

    /// Execute the tool with the given JSON input. Used by all tools
    /// that return only text — which is most of them.
    async fn execute(&self, input: &serde_json::Value) -> Result<String>;

    /// Execute the tool and return a `ToolOutput` carrying both text
    /// and any image attachments.
    async fn execute_full(&self, input: &serde_json::Value) -> Result<ToolOutput> {
        Ok(ToolOutput::from(self.execute(input).await?))
    }
}
```

- [ ] **Step 2: `ToolSet::kinds()` を足す**

`src/tools/mod.rs` の `impl ToolSet` 内、`specs()` の直後に置く。

```rust
    /// Every registered tool's name and kind. Exists so the policy test
    /// can pin the whole classification table in one assertion rather
    /// than constructing each tool by hand.
    pub async fn kinds(&self) -> Vec<(String, ToolKind)> {
        self.inner
            .read()
            .await
            .tools
            .iter()
            .map(|t| (t.spec().name.clone(), t.kind()))
            .collect()
    }
```

- [ ] **Step 3: 失敗するテストを書く**

`src/tools/mod.rs` の末尾に追加。実物の `default_tool_set` を組み立てて全ツールの分類を固定する。

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use sapphire_framework::workspace::{AppContext, Workspace, WorkspaceState};
    use std::sync::Mutex;

    static TEST_CTX: AppContext = AppContext::new("sapphire-agent").allow_external_paths();

    fn test_workspace() -> Arc<Mutex<WorkspaceState>> {
        // Leaked on purpose: this is a test binary and the OS reclaims
        // the directory when it exits.
        let dir = Box::leak(Box::new(tempfile::tempdir().unwrap()));
        std::fs::create_dir_all(dir.path().join(".sapphire-agent")).unwrap();
        let ws = Workspace::from_root(&TEST_CTX, dir.path()).unwrap();
        Arc::new(Mutex::new(WorkspaceState::open(ws).unwrap()))
    }

    /// Every tool declares what it does. A tool added without a `kind()`
    /// lands in `Other` — the strictest bucket — so this table failing
    /// on a newly added tool is the intended prompt to classify it.
    #[tokio::test]
    async fn every_tool_declares_its_kind() {
        let tools = default_tool_set(
            test_workspace(),
            Some("test-tavily-key".to_string()),
            &[],
            crate::timer::TimerManager::new(),
            Vec::new(),
        )
        .await;

        let mut got = tools.kinds().await;
        got.sort_by(|a, b| a.0.cmp(&b.0));
        let got_refs: Vec<(&str, ToolKind)> =
            got.iter().map(|(n, k)| (n.as_str(), *k)).collect();

        let want: Vec<(&str, ToolKind)> = vec![
            ("dir_list", ToolKind::Search),
            ("dir_walk", ToolKind::Search),
            ("file_append", ToolKind::Edit),
            ("file_delete", ToolKind::Delete),
            ("file_read", ToolKind::Read),
            ("file_write", ToolKind::Edit),
            ("memory_add", ToolKind::Edit),
            ("memory_append", ToolKind::Edit),
            ("memory_read", ToolKind::Read),
            ("memory_remove", ToolKind::Delete),
            ("memory_update", ToolKind::Edit),
            ("shell", ToolKind::Execute),
            ("timer_cancel", ToolKind::Delete),
            ("timer_preset", ToolKind::Edit),
            ("timer_set", ToolKind::Edit),
            ("timer_status", ToolKind::Search),
            ("weather", ToolKind::Fetch),
            ("web_search", ToolKind::Fetch),
            ("workspace_search", ToolKind::Search),
            ("workspace_sync", ToolKind::Other),
        ];

        assert_eq!(got_refs, want);
    }

    /// A tool that does not override `kind()` must land in the strictest
    /// bucket, so forgetting to classify one fails safe.
    #[test]
    fn the_default_kind_is_other() {
        struct Bare(ToolSpec);
        #[async_trait]
        impl Tool for Bare {
            fn spec(&self) -> &ToolSpec {
                &self.0
            }
            async fn execute(&self, _input: &serde_json::Value) -> Result<String> {
                Ok(String::new())
            }
        }
        let bare = Bare(ToolSpec {
            name: "bare".into(),
            description: String::new(),
            input_schema: serde_json::json!({}),
        });
        assert_eq!(bare.kind(), ToolKind::Other);
    }
}
```

- [ ] **Step 4: テストが落ちることを確認**

Run: `cargo test --workspace tools::tests`
Expected: FAIL。`every_tool_declares_its_kind` が「全部 `Other`」で不一致になる。`the_default_kind_is_other` は Step 1 の時点で PASS してよい。

もし `every_tool_declares_its_kind` が `want` の長さ違いで落ちる場合、`default_tool_set` が実際に登録するツールが変わっている。**`want` を実態に合わせるのではなく、まず実態を確認すること** — 想定外のツールが混ざっているなら分類が要る。

- [ ] **Step 5: 各ツールに `kind()` を実装する**

`impl Tool for X` ブロックそれぞれに 1 メソッド足す。割り当ては Step 3 の `want` テーブルが正。

```rust
    fn kind(&self) -> ToolKind {
        ToolKind::Read
    }
```

`want` に現れないツールも同じ規則で分類すること:

- `src/tools/ambient_tools.rs`: `transcript_read` → `ToolKind::Read`、`speaker_candidates` → `ToolKind::Search`、`speaker_promote` → `ToolKind::Edit`
- `src/tools/builtin_tools.rs`: `recall_image` → `ToolKind::Read`
- `mcp_reconnect` は `Other` のままでよいので**何も書かない**（既定に任せる）
- MCP サーバ由来のツール（`src/mcp_client/mod.rs` の `build_tools_for_client` が作るラッパ）も `Other` のまま。外部由来のものを最も厳しいバケツに置くのは意図的な設計判断なので、ここに `kind()` を足してはいけない。

- [ ] **Step 6: `EchoTool` に `kind()` を足す**

`src/serve/mod.rs` の `impl crate::tools::Tool for EchoTool` に追加する。

**これを省くと既存の ACP テストがハングする。** `EchoTool` は `kind()` 未実装だと `Other` になり、ACP の `default` モードでは `Ask` に落ちる。既存の `drive` テストヘルパは `session/request_permission` に応答しないので、テストが承認待ちで止まる。

```rust
    fn kind(&self) -> crate::tools::ToolKind {
        // A test fixture standing in for an ordinary read-only tool.
        // Deliberately NOT `Other`: that is the ask-me bucket, and the
        // pre-existing ACP tests drive turns with a helper that answers
        // no permission requests. Task 7 adds a separate fixture for
        // exercising the asking path.
        crate::tools::ToolKind::Read
    }
```

- [ ] **Step 7: テストが通ることを確認**

Run: `cargo test --workspace`
Expected: PASS（既存テストを含めて全部）。

- [ ] **Step 8: コミット**

```bash
git add src/tools/ src/serve/mod.rs
git commit -m "feat(tools): declare an ACP ToolKind for every tool"
```

---

### Task 2: 判定表 `tools::policy::decide`

純関数だけの新モジュール。2 本のツールループが同じものを呼ぶことで、判定表が分裂しないようにする。

**Files:**
- Create: `src/tools/policy.rs`
- Modify: `src/tools/mod.rs`（`pub mod policy;` を先頭のモジュール宣言群に追加）
- Test: `src/tools/policy.rs` の `mod tests`

**Interfaces:**
- Consumes: `crate::tools::ToolKind`（Task 1）
- Produces: `policy::SessionMode`（`ALL` / `id()` / `name()` / `description()` / `from_id()`）、`policy::Origin`、`policy::Decision`、`policy::Approval`（`allows()` / `is_sticky()`）、`policy::Refusal`、`policy::decide(Origin, ToolKind) -> Decision`、`policy::kind_of(&str, &[(String, ToolKind)]) -> ToolKind`、`policy::refusal_message(&str, Refusal) -> String`、`policy::partition_without_asking(Origin, &[ToolCall], &[(String, ToolKind)]) -> (Vec<ToolCall>, Vec<(String, String)>)`

- [ ] **Step 1: 失敗するテストを書く**

`src/tools/policy.rs` を新規作成し、**テストだけ**を先に書く。あわせて `src/tools/mod.rs` の先頭に `pub mod policy;` を足す。

```rust
#[cfg(test)]
mod tests {
    use super::*;

    const SAFE: [ToolKind; 4] = [
        ToolKind::Read,
        ToolKind::Search,
        ToolKind::Fetch,
        ToolKind::Think,
    ];
    const EDITING: [ToolKind; 3] = [ToolKind::Edit, ToolKind::Delete, ToolKind::Move];
    const RISKY: [ToolKind; 2] = [ToolKind::Execute, ToolKind::Other];

    const EVERY_ORIGIN: [Origin; 5] = [
        Origin::Acp(SessionMode::Default),
        Origin::Acp(SessionMode::AcceptEdits),
        Origin::Acp(SessionMode::Bypass),
        Origin::Channel,
        Origin::Trusted,
    ];

    #[test]
    fn safe_kinds_never_ask_on_any_origin() {
        for kind in SAFE {
            for origin in EVERY_ORIGIN {
                assert_eq!(decide(origin, kind), Decision::Allow, "{origin:?} {kind:?}");
            }
        }
    }

    #[test]
    fn editing_asks_only_in_acp_default() {
        for kind in EDITING {
            assert_eq!(decide(Origin::Acp(SessionMode::Default), kind), Decision::Ask);
            assert_eq!(
                decide(Origin::Acp(SessionMode::AcceptEdits), kind),
                Decision::Allow
            );
            assert_eq!(decide(Origin::Acp(SessionMode::Bypass), kind), Decision::Allow);
            assert_eq!(decide(Origin::Channel, kind), Decision::Allow);
            assert_eq!(decide(Origin::Trusted, kind), Decision::Allow);
        }
    }

    /// `shell` and every MCP tool live here. A channel must never reach
    /// them — that is the one behavioural change this feature makes to
    /// an existing transport.
    #[test]
    fn risky_kinds_are_refused_on_channels_and_asked_in_acp() {
        for kind in RISKY {
            assert_eq!(decide(Origin::Acp(SessionMode::Default), kind), Decision::Ask);
            assert_eq!(
                decide(Origin::Acp(SessionMode::AcceptEdits), kind),
                Decision::Ask
            );
            assert_eq!(decide(Origin::Acp(SessionMode::Bypass), kind), Decision::Allow);
            assert_eq!(decide(Origin::Channel, kind), Decision::Deny);
            assert_eq!(decide(Origin::Trusted, kind), Decision::Allow);
        }
    }

    /// `Trusted` is `/rpc`, voice, the heartbeat and `/a2a` — paths that
    /// were already authenticated and have no UI to ask through. Nothing
    /// they can call may change behaviour.
    #[test]
    fn trusted_allows_everything() {
        for kind in SAFE.iter().chain(&EDITING).chain(&RISKY) {
            assert_eq!(decide(Origin::Trusted, *kind), Decision::Allow);
        }
    }

    #[test]
    fn mode_ids_round_trip() {
        for mode in SessionMode::ALL {
            assert_eq!(SessionMode::from_id(mode.id()), Some(mode));
        }
        // `plan` is deliberately not implemented; it must not silently
        // resolve to some other mode.
        assert_eq!(SessionMode::from_id("plan"), None);
        assert_eq!(SessionMode::ALL.len(), 3);
    }

    #[test]
    fn only_always_variants_are_sticky() {
        assert!(!Approval::AllowOnce.is_sticky());
        assert!(Approval::AllowAlways.is_sticky());
        assert!(!Approval::RejectOnce.is_sticky());
        assert!(Approval::RejectAlways.is_sticky());

        assert!(Approval::AllowOnce.allows());
        assert!(Approval::AllowAlways.allows());
        assert!(!Approval::RejectOnce.allows());
        assert!(!Approval::RejectAlways.allows());
    }

    /// An unregistered name must not be treated as safe on its way to
    /// `ToolSet::execute`'s "Unknown tool" reply.
    #[test]
    fn an_unknown_tool_name_is_other() {
        let kinds = vec![("file_read".to_string(), ToolKind::Read)];
        assert_eq!(kind_of("file_read", &kinds), ToolKind::Read);
        assert_eq!(kind_of("no_such_tool", &kinds), ToolKind::Other);
    }

    /// Both refusal reasons name the tool, so the model can tell which
    /// of several calls was refused, and say why.
    #[test]
    fn refusal_messages_name_the_tool_and_the_reason() {
        let declined = refusal_message("shell", Refusal::UserDeclined);
        assert!(declined.contains("shell"), "got {declined}");
        assert!(declined.contains("declined"), "got {declined}");

        let unavailable = refusal_message("shell", Refusal::Unavailable);
        assert!(unavailable.contains("shell"), "got {unavailable}");
        assert_ne!(
            declined, unavailable,
            "the model should be able to tell a refusal from an unavailability"
        );
    }

    /// The channel path's whole gate, in one call: safe calls survive,
    /// risky ones are dropped and come back as refusals that still name
    /// their call id.
    #[test]
    fn partition_drops_risky_calls_and_reports_them() {
        let kinds = vec![
            ("file_read".to_string(), ToolKind::Read),
            ("shell".to_string(), ToolKind::Execute),
            ("mcp__x__y".to_string(), ToolKind::Other),
        ];
        let calls = vec![
            call("c1", "file_read"),
            call("c2", "shell"),
            call("c3", "mcp__x__y"),
        ];

        let (permitted, refused) = partition_without_asking(Origin::Channel, &calls, &kinds);

        let kept: Vec<&str> = permitted.iter().map(|c| c.id.as_str()).collect();
        assert_eq!(kept, vec!["c1"]);

        let refused_ids: Vec<&str> = refused.iter().map(|(id, _)| id.as_str()).collect();
        assert_eq!(refused_ids, vec!["c2", "c3"]);
        assert!(refused[0].1.contains("shell"), "got {}", refused[0].1);
    }

    /// A trusted origin refuses nothing, so the helper is a no-op there.
    #[test]
    fn partition_keeps_everything_for_a_trusted_origin() {
        let kinds = vec![("shell".to_string(), ToolKind::Execute)];
        let calls = vec![call("c1", "shell")];

        let (permitted, refused) = partition_without_asking(Origin::Trusted, &calls, &kinds);

        assert_eq!(permitted.len(), 1);
        assert!(refused.is_empty());
    }

    /// `Ask` cannot be honoured without a human, so an origin that
    /// somehow produces one is refused rather than waved through. This
    /// is what keeps a future policy change from silently opening the
    /// channel path.
    #[test]
    fn partition_refuses_rather_than_allows_an_ask() {
        let kinds = vec![("shell".to_string(), ToolKind::Execute)];
        let calls = vec![call("c1", "shell")];

        let (permitted, refused) = partition_without_asking(
            Origin::Acp(SessionMode::Default),
            &calls,
            &kinds,
        );

        assert!(permitted.is_empty(), "an Ask must not be treated as Allow");
        assert_eq!(refused.len(), 1);
    }

    fn call(id: &str, name: &str) -> crate::provider::ToolCall {
        crate::provider::ToolCall {
            id: id.to_string(),
            name: name.to_string(),
            input: serde_json::json!({}),
        }
    }
}
```

- [ ] **Step 2: テストが落ちる（コンパイルしない）ことを確認**

Run: `cargo test --workspace tools::policy`
Expected: FAIL — `decide` / `Origin` / `Decision` / `SessionMode` / `Approval` が未定義でコンパイルエラー。

- [ ] **Step 3: 実装を書く**

`src/tools/policy.rs` の先頭（`mod tests` の上）に。

```rust
//! Who may run which tool.
//!
//! One pure function, deliberately. There are two tool-calling loops in
//! this crate — `run_llm_turn` (`src/serve/mod.rs`) and the one inside
//! `Agent::handle_message` (`src/agent.rs`) — and the thing that must
//! not be duplicated between them is the *decision*. Merging the loops
//! themselves is a separate job, tracked in the design spec.
//!
//! Nothing here does I/O or knows about persistence: an `AllowAlways` is
//! recorded by the caller, not by `decide`.

use crate::tools::ToolKind;

/// The ACP session modes this agent offers.
///
/// Three, deliberately. `plan` is not "ask or don't ask" but "don't act
/// at all, produce a plan", which needs a different system prompt and a
/// way to present the plan — a separate feature.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionMode {
    Default,
    AcceptEdits,
    Bypass,
}

impl SessionMode {
    /// Every mode, in the order they are advertised to the client.
    pub const ALL: [SessionMode; 3] = [
        SessionMode::Default,
        SessionMode::AcceptEdits,
        SessionMode::Bypass,
    ];

    /// The wire id, as it appears in `session/set_mode`.
    pub fn id(self) -> &'static str {
        match self {
            SessionMode::Default => "default",
            SessionMode::AcceptEdits => "accept_edits",
            SessionMode::Bypass => "bypass",
        }
    }

    /// Human-readable name for the client's mode picker.
    pub fn name(self) -> &'static str {
        match self {
            SessionMode::Default => "Ask before editing",
            SessionMode::AcceptEdits => "Accept edits",
            SessionMode::Bypass => "Bypass permissions",
        }
    }

    pub fn description(self) -> &'static str {
        match self {
            SessionMode::Default => "Ask before writing files or running commands.",
            SessionMode::AcceptEdits => {
                "Write files without asking; still ask before running commands."
            }
            SessionMode::Bypass => "Run everything without asking.",
        }
    }

    /// `None` for an id this agent does not implement — `plan`, notably.
    pub fn from_id(id: &str) -> Option<SessionMode> {
        SessionMode::ALL.into_iter().find(|m| m.id() == id)
    }
}

/// Which transport asked for this tool call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Origin {
    /// An editor over `/acp`, in the session's current mode.
    Acp(SessionMode),
    /// Matrix or Discord. Never asked — a channel turn is asynchronous,
    /// so blocking it on a human reply could hang for hours, and routing
    /// the question through the LLM turn would let the model broker its
    /// own permission request.
    Channel,
    /// `/rpc`, voice, the heartbeat and `/a2a`: already authenticated,
    /// with no UI to ask through. Behaviour must not change for these.
    Trusted,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Decision {
    Allow,
    Ask,
    Deny,
}

/// The outcome of asking. Maps 1:1 onto ACP's `PermissionOptionKind`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Approval {
    AllowOnce,
    AllowAlways,
    RejectOnce,
    RejectAlways,
}

impl Approval {
    /// Whether the call runs.
    pub fn allows(self) -> bool {
        matches!(self, Approval::AllowOnce | Approval::AllowAlways)
    }

    /// Whether this answer should outlive the call that prompted it.
    pub fn is_sticky(self) -> bool {
        matches!(self, Approval::AllowAlways | Approval::RejectAlways)
    }
}

/// The whole policy. The table in the design spec is this function.
pub fn decide(origin: Origin, kind: ToolKind) -> Decision {
    // Group first, so that a `ToolKind` variant added upstream lands in
    // the strict bucket rather than silently becoming safe.
    let risky = matches!(kind, ToolKind::Execute | ToolKind::Other);
    let safe = matches!(
        kind,
        ToolKind::Read | ToolKind::Search | ToolKind::Fetch | ToolKind::Think
    );

    match origin {
        Origin::Trusted => Decision::Allow,
        Origin::Channel => {
            if risky {
                Decision::Deny
            } else {
                Decision::Allow
            }
        }
        Origin::Acp(SessionMode::Bypass) => Decision::Allow,
        Origin::Acp(_) if safe => Decision::Allow,
        Origin::Acp(SessionMode::AcceptEdits) => {
            if risky {
                Decision::Ask
            } else {
                Decision::Allow
            }
        }
        Origin::Acp(SessionMode::Default) => Decision::Ask,
    }
}

/// Why a call was refused. The model reads the difference: one is
/// worth rephrasing around, the other is not worth retrying at all.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Refusal {
    /// A human was asked and said no.
    UserDeclined,
    /// The policy refuses it outright on this transport; nobody was asked.
    Unavailable,
}

/// Look a tool's kind up by name.
///
/// An unregistered name yields `Other`. It will come back from
/// `ToolSet::execute` as "Unknown tool" anyway, but it must not travel
/// there classified as safe on the way.
pub fn kind_of(name: &str, kinds: &[(String, ToolKind)]) -> ToolKind {
    kinds
        .iter()
        .find(|(known, _)| known == name)
        .map(|(_, kind)| *kind)
        .unwrap_or(ToolKind::Other)
}

/// What the model is told in place of a result.
///
/// One function rather than a literal at each call site: both tool
/// loops refuse calls, and two copies of this wording would drift.
pub fn refusal_message(tool: &str, why: Refusal) -> String {
    match why {
        Refusal::UserDeclined => format!(
            "Permission denied: the user declined the '{tool}' call. \
             Do not retry it without being asked to."
        ),
        Refusal::Unavailable => format!(
            "Permission denied: the '{tool}' tool is not available on this \
             transport. Try another approach, or ask the user to run it."
        ),
    }
}

/// Split calls into the ones that may run and the ones that may not,
/// for an origin that has nobody to ask.
///
/// `Ask` is treated as a refusal, not as an allowance. Today only
/// `Origin::Channel` reaches this, and `decide` never returns `Ask` for
/// it — but a later policy change that did must not silently open the
/// channel path, so the unreachable case fails closed.
pub fn partition_without_asking(
    origin: Origin,
    calls: &[crate::provider::ToolCall],
    kinds: &[(String, ToolKind)],
) -> (Vec<crate::provider::ToolCall>, Vec<(String, String)>) {
    let mut permitted = Vec::with_capacity(calls.len());
    let mut refused = Vec::new();

    for call in calls {
        match decide(origin, kind_of(&call.name, kinds)) {
            Decision::Allow => permitted.push(call.clone()),
            Decision::Deny | Decision::Ask => refused.push((
                call.id.clone(),
                refusal_message(&call.name, Refusal::Unavailable),
            )),
        }
    }

    (permitted, refused)
}
```

- [ ] **Step 4: テストが通ることを確認**

Run: `cargo test --workspace tools::policy`
Expected: PASS（11 テスト）。

- [ ] **Step 5: コミット**

```bash
git add src/tools/policy.rs src/tools/mod.rs
git commit -m "feat(tools): one pure policy function for tool permission"
```

---

### Task 3: 承認の永続化 `PermissionStore`

`AllowAlways` / `RejectAlways` を host-local config ディレクトリに記録する。ワークスペースには置かない — 同期対象であり、承認は「このマシンを信用するか」というホストローカルな信頼判断だから。

**Files:**
- Create: `src/serve/acp_permissions.rs`
- Modify: `src/serve/mod.rs`（`mod acp_permissions;` を既存の `mod acp;` の隣に追加）
- Test: `src/serve/acp_permissions.rs` の `mod tests`

**Interfaces:**
- Consumes: `crate::tools::policy::Approval`（Task 2）
- Produces: `PermissionStore::open(PathBuf) -> PermissionStore`、`PermissionStore::default_path() -> PathBuf`、`PermissionStore::standing(&self, profile: &str, tool: &str) -> Option<bool>`、`PermissionStore::record(&self, profile: &str, tool: &str, approval: Approval)`

- [ ] **Step 1: 失敗するテストを書く**

`src/serve/acp_permissions.rs` を新規作成し、テストだけ先に書く。あわせて `src/serve/mod.rs` に `mod acp_permissions;` を足す。

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::tools::policy::Approval;

    fn temp_store() -> (tempfile::TempDir, PermissionStore) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("acp-permissions.json");
        let store = PermissionStore::open(path);
        (dir, store)
    }

    /// Nothing recorded yet means "no standing answer" — ask.
    #[test]
    fn an_unknown_tool_has_no_standing_answer() {
        let (_dir, store) = temp_store();
        assert_eq!(store.standing("zed", "file_write"), None);
    }

    /// Only the `Always` variants stick. A one-off answer must not
    /// silently become permanent.
    #[test]
    fn once_answers_are_not_recorded() {
        let (_dir, store) = temp_store();
        store.record("zed", "file_write", Approval::AllowOnce);
        store.record("zed", "file_delete", Approval::RejectOnce);
        assert_eq!(store.standing("zed", "file_write"), None);
        assert_eq!(store.standing("zed", "file_delete"), None);
    }

    #[test]
    fn always_answers_survive_a_reopen() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("acp-permissions.json");

        let store = PermissionStore::open(path.clone());
        store.record("zed", "file_write", Approval::AllowAlways);
        store.record("zed", "file_delete", Approval::RejectAlways);

        // A fresh process would see this.
        let reopened = PermissionStore::open(path);
        assert_eq!(reopened.standing("zed", "file_write"), Some(true));
        assert_eq!(reopened.standing("zed", "file_delete"), Some(false));
    }

    /// Grants are per room profile: a token pinned to one profile must
    /// not inherit another profile's standing answers.
    #[test]
    fn profiles_do_not_share_grants() {
        let (_dir, store) = temp_store();
        store.record("zed", "shell", Approval::AllowAlways);
        assert_eq!(store.standing("zed", "shell"), Some(true));
        assert_eq!(store.standing("matrix", "shell"), None);
    }

    /// Reject wins. A tool listed in both must not run — the safe side
    /// of a contradiction is refusal.
    #[test]
    fn reject_takes_precedence_over_allow() {
        let (_dir, store) = temp_store();
        store.record("zed", "shell", Approval::AllowAlways);
        store.record("zed", "shell", Approval::RejectAlways);
        assert_eq!(store.standing("zed", "shell"), Some(false));
    }

    /// A grant that is later revoked stops applying.
    #[test]
    fn a_later_answer_replaces_an_earlier_one() {
        let (_dir, store) = temp_store();
        store.record("zed", "file_write", Approval::RejectAlways);
        assert_eq!(store.standing("zed", "file_write"), Some(false));
        store.record("zed", "file_write", Approval::AllowAlways);
        assert_eq!(store.standing("zed", "file_write"), Some(true));
    }

    /// An unreadable record is not a reason to refuse to start. It is
    /// treated as empty, which means "ask" — the safe direction.
    #[test]
    fn a_corrupt_file_is_treated_as_empty() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("acp-permissions.json");
        std::fs::write(&path, b"{ this is not json").unwrap();

        let store = PermissionStore::open(path);
        assert_eq!(store.standing("zed", "file_write"), None);

        // And it must still be writable afterwards.
        store.record("zed", "file_write", Approval::AllowAlways);
        assert_eq!(store.standing("zed", "file_write"), Some(true));
    }

    /// The directory may not exist yet on a fresh host.
    #[test]
    fn a_missing_directory_is_created_on_write() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("nested").join("acp-permissions.json");

        let store = PermissionStore::open(path.clone());
        store.record("zed", "file_write", Approval::AllowAlways);

        assert!(path.exists(), "the record file should have been created");
        assert_eq!(
            PermissionStore::open(path).standing("zed", "file_write"),
            Some(true)
        );
    }
}
```

- [ ] **Step 2: テストが落ちる（コンパイルしない）ことを確認**

Run: `cargo test --workspace acp_permissions`
Expected: FAIL — `PermissionStore` が未定義。

- [ ] **Step 3: 実装を書く**

`src/serve/acp_permissions.rs` の先頭に。

```rust
//! Standing answers to `session/request_permission`, per room profile.
//!
//! Kept beside the host-local config rather than in the workspace. The
//! workspace is a synced artefact, and "always allow `file_write` for
//! this editor" is a statement about *this machine's* trust in *this
//! client* — the same category as the credentials, bind addresses and
//! machine paths `main.rs` already keeps host-local.
//!
//! Grants are per tool name. Argument-level grants ("always, for paths
//! under this directory") would need a path-normalisation design and
//! are deliberately out of scope.

use crate::tools::policy::Approval;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;
use std::sync::Mutex;
use tracing::warn;

#[derive(Debug, Default, Serialize, Deserialize)]
struct Persisted {
    #[serde(default)]
    profiles: BTreeMap<String, ProfileGrants>,
}

#[derive(Debug, Default, Serialize, Deserialize)]
struct ProfileGrants {
    #[serde(default)]
    always_allow: BTreeSet<String>,
    #[serde(default)]
    always_reject: BTreeSet<String>,
}

/// The recorded answers, and the file they live in.
pub(crate) struct PermissionStore {
    path: PathBuf,
    /// A blocking mutex: every critical section here is a map lookup or
    /// a small synchronous write, never held across an await.
    state: Mutex<Persisted>,
}

impl PermissionStore {
    /// `~/.config/sapphire-agent/acp-permissions.json`, matching how
    /// `Config::default_path` resolves the config file itself.
    pub(crate) fn default_path() -> PathBuf {
        if let Some(dirs) = directories::ProjectDirs::from("", "", "sapphire-agent") {
            dirs.config_dir().join("acp-permissions.json")
        } else {
            PathBuf::from("acp-permissions.json")
        }
    }

    /// Load what is on disk. Never fails: a missing file is an empty
    /// record, and an unreadable one is logged and treated as empty.
    /// Losing the grants means asking again, which is the safe
    /// direction; refusing to start the agent over it is not.
    pub(crate) fn open(path: PathBuf) -> Self {
        let state = match std::fs::read(&path) {
            Ok(bytes) => match serde_json::from_slice::<Persisted>(&bytes) {
                Ok(parsed) => parsed,
                Err(e) => {
                    warn!(
                        "ACP: ignoring unreadable permission record at {}: {e}. \
                         Standing answers are lost; the client will be asked again.",
                        path.display()
                    );
                    Persisted::default()
                }
            },
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Persisted::default(),
            Err(e) => {
                warn!(
                    "ACP: could not read the permission record at {}: {e}",
                    path.display()
                );
                Persisted::default()
            }
        };
        Self {
            path,
            state: Mutex::new(state),
        }
    }

    /// `Some(true)` = always allow, `Some(false)` = always reject,
    /// `None` = nothing recorded, so ask.
    ///
    /// Reject wins over allow: a tool somehow present in both lists is
    /// refused, because that is the safe side of a contradiction.
    pub(crate) fn standing(&self, profile: &str, tool: &str) -> Option<bool> {
        let state = self.state.lock().expect("permission store poisoned");
        let grants = state.profiles.get(profile)?;
        if grants.always_reject.contains(tool) {
            Some(false)
        } else if grants.always_allow.contains(tool) {
            Some(true)
        } else {
            None
        }
    }

    /// Record an answer. One-off answers are dropped: only the
    /// `Always` variants are meant to outlive the call.
    pub(crate) fn record(&self, profile: &str, tool: &str, approval: Approval) {
        if !approval.is_sticky() {
            return;
        }
        {
            let mut state = self.state.lock().expect("permission store poisoned");
            let grants = state.profiles.entry(profile.to_string()).or_default();
            // The newest answer replaces the older one, so remove from
            // both lists before inserting into one.
            grants.always_allow.remove(tool);
            grants.always_reject.remove(tool);
            if approval.allows() {
                grants.always_allow.insert(tool.to_string());
            } else {
                grants.always_reject.insert(tool.to_string());
            }
        }
        self.flush();
    }

    /// Write the whole record out. Temp file then rename, so a crash
    /// mid-write cannot leave a half-written record that the next start
    /// would discard entirely.
    fn flush(&self) {
        let json = {
            let state = self.state.lock().expect("permission store poisoned");
            match serde_json::to_vec_pretty(&*state) {
                Ok(v) => v,
                Err(e) => {
                    warn!("ACP: could not serialise the permission record: {e}");
                    return;
                }
            }
        };

        if let Some(parent) = self.path.parent()
            && let Err(e) = std::fs::create_dir_all(parent)
        {
            warn!(
                "ACP: could not create {} for the permission record: {e}",
                parent.display()
            );
            return;
        }

        let tmp = self.path.with_extension("json.tmp");
        if let Err(e) = std::fs::write(&tmp, &json) {
            warn!("ACP: could not write {}: {e}", tmp.display());
            return;
        }
        if let Err(e) = std::fs::rename(&tmp, &self.path) {
            warn!(
                "ACP: could not replace {} with {}: {e}",
                self.path.display(),
                tmp.display()
            );
            let _ = std::fs::remove_file(&tmp);
        }
    }
}
```

**注意:** `use std::path::PathBuf;` だけでよい — `Path` は使わない。`cargo clippy --workspace --all-targets -- -D warnings` が未使用 import を弾くので、それに従うこと。

- [ ] **Step 4: テストが通ることを確認**

Run: `cargo test --workspace acp_permissions`
Expected: PASS（8 テスト）。

- [ ] **Step 5: コミット**

```bash
git add src/serve/acp_permissions.rs src/serve/mod.rs
git commit -m "feat(acp): persist standing permission answers per room profile"
```

---

### Task 4: `TurnProgress` を `TurnHost` にし、`origin()` と `approve()` を足す

trait が「報告」だけでなくなるので改名する。既存の 3 実装のうち `SseProgress` と `NullProgress` は**デフォルト実装のおかげで中身を変えない**（改名の追従のみ）。

**Files:**
- Modify: `src/serve/mod.rs`（trait 定義、`SseProgress` / `NullProgress` の impl、`run_llm_turn` の引数型）
- Modify: `src/serve/acp.rs`（`impl super::TurnProgress for AcpProgress` の追従、doc コメント内の言及）
- Modify: `src/serve/a2a.rs`（`run_llm_turn` 呼び出しの型注釈があれば）
- Test: `src/serve/mod.rs` の既存 `mod tests`

**Interfaces:**
- Consumes: `crate::tools::policy::{Origin, Approval}`（Task 2）、`crate::tools::ToolKind`（Task 1）
- Produces: `trait TurnHost`（`tool_start` / `tool_end` / `turn_error` / `origin` / `approve`）

- [ ] **Step 1: 失敗するテストを書く**

`src/serve/mod.rs` の既存 `mod tests` に。既存の実装が `Trusted` のままであることを固定する — これが「`/rpc` と voice の挙動は変わらない」の機械的な保証になる。

```rust
    /// The pre-existing transports keep today's behaviour. `Trusted` is
    /// what makes that true: `decide` allows everything for it, so no
    /// `/rpc`, voice or heartbeat turn can start asking for permission.
    #[tokio::test]
    async fn existing_transports_are_trusted() {
        use crate::tools::policy::Origin;

        let (tx, _rx) = mpsc::channel(4);
        let sse = SseProgress::new(tx, serde_json::json!(1));
        assert_eq!(sse.origin(), Origin::Trusted);
        assert_eq!(NullProgress.origin(), Origin::Trusted);
    }

    /// A host that cannot ask must not block the call. The default is
    /// to let it through, which is what keeps the existing transports
    /// behaving exactly as before.
    #[tokio::test]
    async fn the_default_approval_allows_once() {
        use crate::provider::ToolCall;
        use crate::tools::{ToolKind, policy::Approval};

        let call = ToolCall {
            id: "c1".to_string(),
            name: "shell".to_string(),
            input: serde_json::json!({}),
        };
        assert_eq!(
            NullProgress.approve(&call, ToolKind::Execute).await,
            Approval::AllowOnce
        );
    }
```

- [ ] **Step 2: テストが落ちることを確認**

Run: `cargo test --workspace existing_transports_are_trusted the_default_approval_allows_once`
Expected: FAIL — `origin` / `approve` が未定義。

- [ ] **Step 3: trait を書き換える**

`src/serve/mod.rs` の `pub(crate) trait TurnProgress` を置き換える。

```rust
/// The per-transport hook a turn reports through — and, now, asks
/// through.
///
/// Renamed from `TurnProgress`: it is no longer only about reporting.
/// Both new methods carry defaults, so a transport that has no way to
/// ask a human implements neither and keeps behaving exactly as it did.
#[async_trait::async_trait]
pub(crate) trait TurnHost: Send + Sync {
    async fn tool_start(&self, id: &str, name: &str);
    async fn tool_end(&self, id: &str, name: &str);
    async fn turn_error(&self, message: &str);

    /// Which row of the permission table this turn is judged by.
    ///
    /// `Trusted` by default: `/rpc`, voice and the heartbeat were
    /// authenticated before the turn started and have no UI to ask
    /// through, so they must keep running everything.
    fn origin(&self) -> crate::tools::policy::Origin {
        crate::tools::policy::Origin::Trusted
    }

    /// Called only when `decide` returned `Ask`.
    ///
    /// The default answers `AllowOnce` — a host that never returns an
    /// asking `origin()` can never reach this, so the default exists
    /// for safety rather than for use.
    async fn approve(
        &self,
        call: &crate::provider::ToolCall,
        kind: crate::tools::ToolKind,
    ) -> crate::tools::policy::Approval {
        let _ = (call, kind);
        crate::tools::policy::Approval::AllowOnce
    }
}
```

- [ ] **Step 4: 改名に追従する**

`TurnProgress` を `TurnHost` に置換する。対象は 8 箇所:

- `src/serve/mod.rs`: trait 定義、`impl TurnProgress for SseProgress`、`impl TurnProgress for NullProgress`、`run_llm_turn` の `progress: Arc<dyn TurnProgress>`、doc コメント 2 箇所
- `src/serve/acp.rs`: `impl super::TurnProgress for AcpProgress`、`Arc::clone(&progress) as Arc<dyn super::TurnProgress>`、doc コメント 1 箇所

`SseProgress` と `NullProgress` の impl は **`origin` も `approve` も実装しない**。デフォルトのままが正しい。

- [ ] **Step 5: テストが通ることを確認**

Run: `cargo test --workspace`
Expected: PASS。この時点ではまだゲートが差さっていないので、挙動は完全に変わらない。

- [ ] **Step 6: コミット**

```bash
git add src/serve/
git commit -m "refactor(serve): TurnProgress becomes TurnHost, with origin and approve"
```

---

### Task 5: `run_llm_turn` にゲートを差す

`/acp`・`/rpc`・voice・heartbeat・`/a2a` が通るループ。承認は直列、実行は従来どおり並行。

**Files:**
- Modify: `src/serve/mod.rs`（`run_llm_turn` のツール実行ブロック）
- Test: `src/serve/mod.rs` の `mod tests`

**Interfaces:**
- Consumes: `TurnHost::origin` / `TurnHost::approve`（Task 4）、`policy::decide`（Task 2）、`ToolSet::kinds`（Task 1）
- Produces: なし（内部変更）

- [ ] **Step 1: 失敗するテストを書く**

`src/serve/mod.rs` の `mod tests` に。`Origin::Channel` を返す `TurnHost` を用意して、`shell` 相当が拒否され、ターンが続くことを確認する。

```rust
    /// A refused tool still gets a `tool_result`, and the turn carries
    /// on. Refusing must not look to the model like the tool vanished,
    /// and must not end the turn — the model may have another route.
    #[tokio::test]
    async fn a_refused_tool_returns_a_result_and_the_turn_continues() {
        use crate::tools::policy::Origin;

        struct ChannelHost;
        #[async_trait::async_trait]
        impl TurnHost for ChannelHost {
            async fn tool_start(&self, _id: &str, _name: &str) {}
            async fn tool_end(&self, _id: &str, _name: &str) {}
            async fn turn_error(&self, _message: &str) {}
            fn origin(&self) -> Origin {
                Origin::Channel
            }
        }

        // `risky` is Execute, so `Origin::Channel` refuses it. The
        // second scripted response is the model carrying on afterwards.
        let state = ServeState::for_test_scripted(
            true,
            vec![
                crate::provider::ChatResponse {
                    text: None,
                    tool_calls: vec![crate::provider::ToolCall {
                        id: "call-1".to_string(),
                        name: "risky".to_string(),
                        input: json!({}),
                    }],
                    stop_reason: None,
                },
                crate::provider::ChatResponse {
                    text: Some("could not run that".to_string()),
                    tool_calls: Vec::new(),
                    stop_reason: None,
                },
            ],
        );
        let risky = RiskyTool::new();
        let ran = risky.ran_flag();
        state.tools.register_tool(Box::new(risky)).await;

        let outcome = run_llm_turn(
            Arc::clone(&state),
            "s-refused".to_string(),
            ChatMessage::user("run it"),
            Arc::new(ChannelHost),
            None,
        )
        .await;

        assert_eq!(outcome.text.as_deref(), Some("could not run that"));
        assert!(
            !ran.load(std::sync::atomic::Ordering::SeqCst),
            "a refused tool must not have executed"
        );
    }
    }

    /// The same call is allowed once the origin is a trusted one, so
    /// the refusal above is the policy talking and not a broken tool.
    #[tokio::test]
    async fn a_trusted_origin_runs_the_same_tool() {
        let state = ServeState::for_test_scripted(
            true,
            vec![
                crate::provider::ChatResponse {
                    text: None,
                    tool_calls: vec![crate::provider::ToolCall {
                        id: "call-1".to_string(),
                        name: "risky".to_string(),
                        input: json!({}),
                    }],
                    stop_reason: None,
                },
                crate::provider::ChatResponse {
                    text: Some("ran it".to_string()),
                    tool_calls: Vec::new(),
                    stop_reason: None,
                },
            ],
        );
        state.tools.register_tool(Box::new(RiskyTool::new())).await;

        let outcome = run_llm_turn(
            Arc::clone(&state),
            "s-allowed".to_string(),
            ChatMessage::user("run it"),
            Arc::new(NullProgress),
            None,
        )
        .await;

        assert_eq!(outcome.text.as_deref(), Some("ran it"));
    }
```

あわせて `src/serve/mod.rs` のテスト用フィクスチャ群（`EchoTool` の隣）に `RiskyTool` を足す。実行されたかどうかをインスタンス単位のフラグで観測する。

```rust
/// A stand-in for `shell`: `ToolKind::Execute`, so the policy asks or
/// refuses. Carries a per-instance "did I run" flag, which is how the
/// gate tests tell "refused" from "ran and returned an error".
///
/// Per instance, not a `static`: `cargo test` runs tests in parallel and
/// several tasks construct one of these, so a process-global flag would
/// make the assertions depend on scheduling.
#[cfg(test)]
pub(crate) struct RiskyTool {
    spec: crate::provider::ToolSpec,
    ran: Arc<std::sync::atomic::AtomicBool>,
}

#[cfg(test)]
impl RiskyTool {
    pub(crate) fn new() -> Self {
        Self {
            spec: crate::provider::ToolSpec {
                name: "risky".into(),
                description: "Pretend to run a command.".into(),
                input_schema: json!({ "type": "object", "properties": {} }),
            },
            ran: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        }
    }

    /// A handle the test keeps after the tool is boxed into the ToolSet.
    pub(crate) fn ran_flag(&self) -> Arc<std::sync::atomic::AtomicBool> {
        Arc::clone(&self.ran)
    }
}

#[cfg(test)]
#[async_trait::async_trait]
impl crate::tools::Tool for RiskyTool {
    fn spec(&self) -> &crate::provider::ToolSpec {
        &self.spec
    }

    fn kind(&self) -> crate::tools::ToolKind {
        crate::tools::ToolKind::Execute
    }

    async fn execute(&self, _input: &Value) -> anyhow::Result<String> {
        self.ran.store(true, std::sync::atomic::Ordering::SeqCst);
        Ok("ran".to_string())
    }
}
```

登録するときは、ボックス化する前にフラグのハンドルを取ること（Step 1 のテストがその形で書いてある）。

- [ ] **Step 3: ゲートを実装する**

`src/serve/mod.rs` の `run_llm_turn` 内、`// Notify client of each tool starting` のループの直後、`// Execute all tools concurrently` の直前に挿入する。

```rust
                // Permission gate.
                //
                // Serial on purpose. `decide` is a cheap pure call, but
                // `approve` puts a dialog in front of a human, and
                // firing several at once would stack them on the poor
                // soul in the editor. Execution below stays concurrent.
                let kinds = state.tools.kinds().await;
                let origin = progress.origin();
                let mut permitted: Vec<crate::provider::ToolCall> = Vec::new();
                let mut refused: Vec<(String, String)> = Vec::new();
                for call in &tool_calls {
                    use crate::tools::policy::{Decision, Refusal, kind_of, refusal_message};

                    let kind = kind_of(&call.name, &kinds);
                    let verdict = crate::tools::policy::decide(origin, kind);
                    let refusal = match verdict {
                        Decision::Allow => None,
                        Decision::Deny => {
                            Some(refusal_message(&call.name, Refusal::Unavailable))
                        }
                        Decision::Ask => {
                            if progress.approve(call, kind).await.allows() {
                                None
                            } else {
                                Some(refusal_message(&call.name, Refusal::UserDeclined))
                            }
                        }
                    };

                    match refusal {
                        None => permitted.push(call.clone()),
                        Some(reason) => {
                            info!("Refused tool {} (id={}): {verdict:?}", call.name, call.id);
                            refused.push((call.id.clone(), reason));
                        }
                    }
                }
```

続けて、既存の `join_all` ブロックを `tool_calls.iter()` ではなく `permitted.iter()` を回すように変え、拒否分を結果に合流させる。既存の

```rust
                let results: Vec<(String, crate::tools::ToolOutput)> =
                    futures_util::future::join_all(tool_calls.iter().map(|c| {
```

を

```rust
                let mut results: Vec<(String, crate::tools::ToolOutput)> =
                    futures_util::future::join_all(permitted.iter().map(|c| {
```

に変え（`let` を `let mut` にするのを忘れないこと）、その `.await;` の直後に:

```rust
                // A refused call still owes the model a tool_result:
                // every tool_use in the assistant message must be
                // answered, and the reason is more useful to the model
                // than silence. The turn is NOT ended — the model may
                // have another route, and ACP's `Refusal` stop reason
                // means "the agent declined", which is a different
                // thing from "the user declined".
                for (id, reason) in refused {
                    results.push((id, crate::tools::ToolOutput::from(reason)));
                }
```

- [ ] **Step 4: テストが通ることを確認**

Run: `cargo test --workspace`
Expected: PASS。既存テストも全部通ること — `EchoTool` は Task 1 で `Read` にしてあるので `NullProgress`（`Trusted`）でも `SseProgress` でも素通りする。

- [ ] **Step 5: コミット**

```bash
git add src/serve/mod.rs
git commit -m "feat(serve): gate tool execution on the permission policy"
```

---

### Task 6: `agent.rs` のループにゲートを差す（Matrix / Discord）

こちらは `TurnHost` を使っていない。`Origin::Channel` は `Ask` を返さないので `approve()` は呼ばれず、拒否されたものを実行に回さないだけ。並行実行の形も変えない。

Task 2 の `partition_without_asking` がこのゲートの本体なので、ここでの仕事は「それを正しい場所で呼び、拒否分を結果に合流させる」ことに尽きる。

**Files:**
- Modify: `src/agent.rs`（ツール実行ブロック）
- Test: `src/agent.rs` の `mod tests`

**Interfaces:**
- Consumes: `policy::partition_without_asking` / `policy::Origin`（Task 2）、`ToolSet::kinds`（Task 1）
- Produces: なし（内部変更）

- [ ] **Step 1: 失敗するテストを書く**

`src/agent.rs` の `mod tests` に。実物の `ToolSet` を組み立て、そこから取った `kinds()` を通してゲートを駆動する — これは production の `partition_without_asking` を production の分類データで動かすテストであって、判定表の再宣言ではない。

```rust
    /// The channel path refuses `Execute` and `Other` outright. This is
    /// the one behavioural change the permission work makes to an
    /// existing transport: `shell` and every MCP tool stop being
    /// reachable from Matrix and Discord, while everything the chat
    /// bots actually use keeps working.
    #[tokio::test]
    async fn the_channel_gate_refuses_shell_but_keeps_the_chat_tools() {
        use crate::provider::ToolCall;
        use crate::tools::policy::{Origin, partition_without_asking};

        let tools = crate::tools::default_tool_set(
            test_workspace(),
            Some("test-tavily-key".to_string()),
            &[],
            crate::timer::TimerManager::new(),
            Vec::new(),
        )
        .await;
        let kinds = tools.kinds().await;

        let calls: Vec<ToolCall> = [
            "shell",
            "web_search",
            "memory_add",
            "file_read",
            "workspace_sync",
            "mcp__somewhere__do_thing",
        ]
        .iter()
        .enumerate()
        .map(|(i, name)| ToolCall {
            id: format!("c{i}"),
            name: (*name).to_string(),
            input: serde_json::json!({}),
        })
        .collect();

        let (permitted, refused) =
            partition_without_asking(Origin::Channel, &calls, &kinds);

        let kept: Vec<&str> = permitted.iter().map(|c| c.name.as_str()).collect();
        assert_eq!(
            kept,
            vec!["web_search", "memory_add", "file_read"],
            "the chat bots' own tools must keep working"
        );

        let blocked: Vec<&str> = refused
            .iter()
            .map(|(id, _)| {
                calls
                    .iter()
                    .find(|c| &c.id == id)
                    .map(|c| c.name.as_str())
                    .unwrap()
            })
            .collect();
        assert_eq!(
            blocked,
            vec!["shell", "workspace_sync", "mcp__somewhere__do_thing"],
            "Execute and Other must not be reachable over chat"
        );

        // Every refusal names its tool, so the model can say which call
        // it lost and why.
        for (id, reason) in &refused {
            let name = calls.iter().find(|c| &c.id == id).unwrap().name.as_str();
            assert!(reason.contains(name), "{reason} should name {name}");
        }
    }
```

`test_workspace()` は Task 1 で `src/tools/mod.rs` のテストに書いたものと同じ形。`src/agent.rs` のテストからは見えないので、こちらの `mod tests` にも同じヘルパを置く。

```rust
    fn test_workspace() -> Arc<std::sync::Mutex<sapphire_framework::workspace::WorkspaceState>> {
        use sapphire_framework::workspace::{AppContext, Workspace, WorkspaceState};
        static TEST_CTX: AppContext = AppContext::new("sapphire-agent").allow_external_paths();

        let dir = Box::leak(Box::new(tempfile::tempdir().unwrap()));
        std::fs::create_dir_all(dir.path().join(".sapphire-agent")).unwrap();
        let ws = Workspace::from_root(&TEST_CTX, dir.path()).unwrap();
        Arc::new(std::sync::Mutex::new(WorkspaceState::open(ws).unwrap()))
    }
```

- [ ] **Step 2: テストが落ちることを確認**

Run: `cargo test --workspace the_channel_gate_refuses_shell`
Expected: PASS。`partition_without_asking` は Task 2 で入っているので、このテストはゲートを差す前から通る。

**これはゲートが入った証拠ではない。** このテストが押さえているのは「production の分類 × production の判定でどのツールが落ちるか」であって、`agent.rs` がそれを実際に呼んでいるかではない。呼び出しの配線は Step 3 で入れ、Step 4 の全体テストで既存のチャネル経路が壊れていないことを見る。配線そのものの回帰は最終レビューの担当。

- [ ] **Step 3: ゲートを配線する**

`src/agent.rs` の `let mut handles = Vec::with_capacity(tool_calls.len());` の直前に挿入する。

```rust
                    // Permission gate for the channel path.
                    //
                    // `Origin::Channel` never returns `Ask` — a channel
                    // turn is asynchronous, so there is nobody to hold
                    // it open for — which is why this needs no `approve`
                    // and no serialisation. Refused calls simply never
                    // reach `tokio::spawn`.
                    let kinds = tools.kinds().await;
                    let (tool_calls, refused) = crate::tools::policy::partition_without_asking(
                        crate::tools::policy::Origin::Channel,
                        &tool_calls,
                        &kinds,
                    );
                    for (id, reason) in &refused {
                        info!("Refused tool call {id} on the channel path: {reason}");
                    }
```

そのあと、既存の結果収集ループ

```rust
                    let mut results = Vec::with_capacity(handles.len());
                    for handle in handles {
                        match handle.await {
                            Ok(r) => results.push(r),
                            Err(e) => warn!("Tool task panicked: {e}"),
                        }
                    }
```

の直後に、拒否分を合流させる。

```rust
                    // Every tool_use owes the model a tool_result, and
                    // the reason is more useful than silence. The turn
                    // continues — the model may have another route.
                    for (id, reason) in refused {
                        results.push((id, crate::tools::ToolOutput::from(reason)));
                    }
```

**確認事項が 2 つある。**

1. `tools` がこのスコープで `Arc<ToolSet>` として束縛されているか。無ければ `let tools = Arc::clone(&self.tools);` 相当を上に足すこと。
2. `partition_without_asking` は `tool_calls` をシャドウする。**この直前で `tool_calls` が `ChatMessage::assistant_with_tools` にクローンして渡されていることを確認すること。** 渡していれば、拒否分を落としても履歴の tool_use ブロックは元のまま残る（そして拒否分の tool_result が下で合流するので、tool_use と tool_result は一対一で対応し続ける）。もしクローンではなくムーブしていたら、シャドウする前に履歴用のクローンを取ること。**ここを間違えると tool_use と tool_result の数が合わず、次のプロバイダ呼び出しが 400 で落ちる。**

- [ ] **Step 4: テストが通ることを確認**

Run: `cargo test --workspace`
Expected: PASS。

- [ ] **Step 5: コミット**

```bash
git add src/agent.rs
git commit -m "feat(agent): refuse Execute and Other tools on the channel path"
```

---

### Task 7: `AcpProgress` が `session/request_permission` を出す

ここで初めて人間に問いが届く。

**Files:**
- Modify: `src/serve/mod.rs`（`ServeState` に `permissions` フィールド、テストフィクスチャ）
- Modify: `src/serve/acp.rs`（`AcpProgress` のフィールド追加、`origin` / `approve` 実装、`AcpSession` に `mode`）
- Test: `src/serve/acp.rs` の `mod tests`

**Interfaces:**
- Consumes: `PermissionStore`（Task 3）、`TurnHost::approve`（Task 4）、`policy::{Approval, Origin, SessionMode}`（Task 2）
- Produces: `ServeState.permissions: Arc<PermissionStore>`、`AcpSession.mode: SessionMode`

- [ ] **Step 1: `ServeState` に store を足す**

`src/serve/mod.rs` の `ServeState` に:

```rust
    /// Standing answers to `session/request_permission`. Shared across
    /// connections because the record is host-wide, keyed by room
    /// profile inside.
    pub(crate) permissions: Arc<acp_permissions::PermissionStore>,
```

本番の構築箇所では `Arc::new(acp_permissions::PermissionStore::open(acp_permissions::PermissionStore::default_path()))` を渡す。`build_for_test` では tempdir 配下を使う:

```rust
            permissions: Arc::new(acp_permissions::PermissionStore::open(
                base.join("acp-permissions.json"),
            )),
```

- [ ] **Step 2: 失敗するテストを書く**

`src/serve/acp.rs` の `mod tests` に。まず承認要求に答えるドライバを足す。既存の `drive` は agent → client の**リクエスト**に答えないので、それ用のヘルパが要る。

```rust
    /// Like `drive`, but answers any `session/request_permission` the
    /// agent sends with `option_id`. Returns the updates, the final
    /// reply, and how many permission requests arrived.
    async fn drive_answering(
        addr: &str,
        prompt: serde_json::Value,
        option_id: &str,
    ) -> (Vec<serde_json::Value>, serde_json::Value, usize) {
        let mut ws = connect(addr).await;
        for request in [initialize_request(0), new_session_request(1)] {
            ws.send(Message::Text(request.to_string().into()))
                .await
                .unwrap();
        }

        let mut updates = Vec::new();
        let mut asked = 0usize;

        loop {
            let frame = next_frame(&mut ws).await;
            let Message::Text(t) = frame else { continue };
            let v: serde_json::Value = serde_json::from_str(&t).unwrap();

            if v["id"] == 1 {
                let id = v["result"]["sessionId"].as_str().unwrap().to_string();
                ws.send(Message::Text(
                    prompt_request(2, &id, prompt.clone()).to_string().into(),
                ))
                .await
                .unwrap();
            } else if v["method"] == "session/request_permission" {
                asked += 1;
                let answer = serde_json::json!({
                    "jsonrpc": "2.0",
                    "id": v["id"],
                    "result": {
                        "outcome": { "outcome": "selected", "optionId": option_id }
                    }
                });
                ws.send(Message::Text(answer.to_string().into()))
                    .await
                    .unwrap();
            } else if v["method"] == "session/update" {
                updates.push(v["params"]["update"].clone());
            } else if v["id"] == 2 {
                return (updates, v, asked);
            }
        }
    }
```

続けてテスト本体。`RiskyTool`（Task 5 で追加済み）を使う。ここでは実行の有無ではなく**返ってきたテキストと承認要求の回数**で判定するので、`ran_flag()` は使わない。

```rust
    /// An `Execute` tool in the default mode puts the question to the
    /// user, and an allow lets it run.
    #[tokio::test]
    async fn an_execute_tool_asks_and_runs_when_allowed() {
        let state = ServeState::for_test_scripted(
            true,
            vec![
                crate::provider::ChatResponse {
                    text: None,
                    tool_calls: vec![crate::provider::ToolCall {
                        id: "call-1".to_string(),
                        name: "risky".to_string(),
                        input: serde_json::json!({}),
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
        state
            .tools
            .register_tool(Box::new(super::RiskyTool::new()))
            .await;
        let addr = spawn(state).await;

        let (_updates, reply, asked) =
            drive_answering(&addr, text_prompt("run it"), "allow_once").await;

        assert_eq!(asked, 1, "exactly one permission request, got {asked}");
        assert_eq!(reply["result"]["stopReason"], "end_turn", "got {reply}");
    }

    /// A refusal does not end the turn: the model gets a tool_result
    /// saying so and answers normally. Showing the user an error dialog
    /// because they declined would be wrong.
    #[tokio::test]
    async fn a_refusal_does_not_end_the_turn() {
        let state = ServeState::for_test_scripted(
            true,
            vec![
                crate::provider::ChatResponse {
                    text: None,
                    tool_calls: vec![crate::provider::ToolCall {
                        id: "call-1".to_string(),
                        name: "risky".to_string(),
                        input: serde_json::json!({}),
                    }],
                    stop_reason: None,
                },
                crate::provider::ChatResponse {
                    text: Some("understood".to_string()),
                    tool_calls: Vec::new(),
                    stop_reason: None,
                },
            ],
        );
        state
            .tools
            .register_tool(Box::new(super::RiskyTool::new()))
            .await;
        let addr = spawn(state).await;

        let (updates, reply, asked) =
            drive_answering(&addr, text_prompt("run it"), "reject_once").await;

        assert_eq!(asked, 1);
        assert_eq!(
            reply["result"]["stopReason"], "end_turn",
            "a declined tool is not a failed turn, got {reply}"
        );
        let chunks: Vec<&str> = updates
            .iter()
            .filter(|u| u["sessionUpdate"] == "agent_message_chunk")
            .filter_map(|u| u["content"]["text"].as_str())
            .collect();
        assert_eq!(chunks, vec!["understood"]);
    }

    /// A `Read` tool is never put to the user. This is what keeps the
    /// feature usable: a dialog per `file_read` would be intolerable.
    #[tokio::test]
    async fn a_safe_tool_is_not_put_to_the_user() {
        let state = ServeState::for_test_scripted(
            true,
            vec![
                crate::provider::ChatResponse {
                    text: None,
                    tool_calls: vec![crate::provider::ToolCall {
                        id: "call-1".to_string(),
                        name: "echo".to_string(),
                        input: serde_json::json!({ "text": "ping" }),
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
        let addr = spawn(state).await;

        let (_updates, reply, asked) =
            drive_answering(&addr, text_prompt("echo"), "allow_once").await;

        assert_eq!(asked, 0, "a Read tool must not ask");
        assert_eq!(reply["result"]["stopReason"], "end_turn");
    }

    /// `allow_always` is remembered, so the second call in the same
    /// turn does not ask again.
    #[tokio::test]
    async fn allow_always_is_not_asked_twice() {
        let state = ServeState::for_test_scripted(
            true,
            vec![
                crate::provider::ChatResponse {
                    text: None,
                    tool_calls: vec![crate::provider::ToolCall {
                        id: "call-1".to_string(),
                        name: "risky".to_string(),
                        input: serde_json::json!({}),
                    }],
                    stop_reason: None,
                },
                crate::provider::ChatResponse {
                    text: None,
                    tool_calls: vec![crate::provider::ToolCall {
                        id: "call-2".to_string(),
                        name: "risky".to_string(),
                        input: serde_json::json!({}),
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
        state
            .tools
            .register_tool(Box::new(super::RiskyTool::new()))
            .await;
        let addr = spawn(state).await;

        let (_updates, reply, asked) =
            drive_answering(&addr, text_prompt("run it twice"), "allow_always").await;

        assert_eq!(asked, 1, "the second call must use the recorded answer");
        assert_eq!(reply["result"]["stopReason"], "end_turn", "got {reply}");
    }
```

- [ ] **Step 3: テストが落ちることを確認**

Run: `cargo test --workspace acp::tests::an_execute_tool_asks_and_runs_when_allowed`
Expected: FAIL — 承認要求が送られないので `asked == 0`。

- [ ] **Step 4: `AcpSession` にモードを持たせ、`AcpProgress` に文脈を渡す**

`src/serve/acp.rs`。`AcpSession` に:

```rust
    /// The permission mode this session is in. Per session, not per
    /// connection: two sessions on one socket are judged separately.
    mode: crate::tools::policy::SessionMode,
```

`session/new` ハンドラの `AcpSession { ... }` に `mode: crate::tools::policy::SessionMode::Default,` を足す。

`AcpProgress` に 3 つ足す:

```rust
    /// The room profile this connection's bearer token resolved to.
    /// Standing answers are recorded under it.
    profile: String,
    /// The session's mode as of the moment this turn started. Copied
    /// rather than shared: a `set_mode` arriving mid-turn must not
    /// change the rules under a call already being judged.
    mode: crate::tools::policy::SessionMode,
    permissions: Arc<super::acp_permissions::PermissionStore>,
```

`AcpProgress::new` の引数を増やし、`session/prompt` ハンドラの生成箇所で、セッションを引くのと同じロックの中で `mode` を読んで渡すこと（`looked_up` のクロージャで `session.mode` も返す）。

- [ ] **Step 5: `origin` と `approve` を実装する**

`impl super::TurnHost for AcpProgress` に追加する。

```rust
    fn origin(&self) -> crate::tools::policy::Origin {
        crate::tools::policy::Origin::Acp(self.mode)
    }

    /// Put the call to the user, unless a standing answer already
    /// settles it.
    ///
    /// The standing answer is consulted here rather than in `decide`
    /// because `decide` is a pure function over the policy table and
    /// knows nothing about what this host has been told before.
    async fn approve(
        &self,
        call: &crate::provider::ToolCall,
        kind: crate::tools::ToolKind,
    ) -> crate::tools::policy::Approval {
        use crate::tools::policy::Approval;

        match self.permissions.standing(&self.profile, &call.name) {
            Some(true) => return Approval::AllowAlways,
            Some(false) => return Approval::RejectAlways,
            None => {}
        }

        let request = RequestPermissionRequest::new(
            self.session_id.clone(),
            ToolCallUpdate::new(
                ToolCallId::new(call.id.as_str()),
                ToolCallUpdateFields::new()
                    .title(call.name.clone())
                    .kind(kind)
                    .raw_input(call.input.clone()),
            ),
            vec![
                PermissionOption::new(
                    "allow_once",
                    "Allow once",
                    PermissionOptionKind::AllowOnce,
                ),
                PermissionOption::new(
                    "allow_always",
                    "Always allow this tool",
                    PermissionOptionKind::AllowAlways,
                ),
                PermissionOption::new(
                    "reject_once",
                    "Reject",
                    PermissionOptionKind::RejectOnce,
                ),
                PermissionOption::new(
                    "reject_always",
                    "Never allow this tool",
                    PermissionOptionKind::RejectAlways,
                ),
            ],
        );

        let answer = match self.connection.send_request(request).await {
            Ok(a) => a,
            Err(e) => {
                // The client went away, or refused the method. Either
                // way nobody said yes, and running unguarded because
                // the question failed to arrive is the wrong direction
                // to fail in.
                warn!(
                    "ACP: could not ask about '{}' on session {}: {e}. Treating as declined.",
                    call.name, self.session_id
                );
                return Approval::RejectOnce;
            }
        };

        let approval = match answer.outcome {
            // The turn is being cancelled; the cancel path answers the
            // prompt with `Cancelled`, so this call must simply not run.
            RequestPermissionOutcome::Cancelled => Approval::RejectOnce,
            RequestPermissionOutcome::Selected(selected) => {
                match selected.option_id.0.as_ref() {
                    "allow_once" => Approval::AllowOnce,
                    "allow_always" => Approval::AllowAlways,
                    "reject_always" => Approval::RejectAlways,
                    "reject_once" => Approval::RejectOnce,
                    other => {
                        warn!(
                            "ACP: unknown permission option '{other}' for '{}'; \
                             treating as declined.",
                            call.name
                        );
                        Approval::RejectOnce
                    }
                }
            }
        };

        if approval.is_sticky() {
            self.permissions.record(&self.profile, &call.name, approval);
        }
        approval
    }
```

必要な `use` を `src/serve/acp.rs` の import に足す:

```rust
use agent_client_protocol::schema::v1::{
    PermissionOption, PermissionOptionKind, RequestPermissionOutcome, RequestPermissionRequest,
};
```

- [ ] **Step 6: テストが通ることを確認**

Run: `cargo test --workspace`
Expected: PASS。既存の ACP テストが**ハングしないこと**を必ず確認すること。ハングしたら Task 1 Step 6 の `EchoTool::kind()` が入っていない。

- [ ] **Step 7: コミット**

```bash
git add src/serve/
git commit -m "feat(acp): ask the client before running a gated tool"
```

---

### Task 8: セッションモード（`session/new` の modes と `session/set_mode`）

**Files:**
- Modify: `src/serve/acp.rs`（`session/new` の応答、`session/set_mode` ハンドラ）
- Test: `src/serve/acp.rs` の `mod tests`

**Interfaces:**
- Consumes: `AcpSession.mode`（Task 7）、`policy::SessionMode`（Task 2）
- Produces: `session/set_mode` エンドポイント、`SessionUpdate::CurrentModeUpdate` 通知

- [ ] **Step 1: 失敗するテストを書く**

`src/serve/acp.rs` の `mod tests` に。

```rust
    fn set_mode_request(id: i64, session_id: &str, mode_id: &str) -> serde_json::Value {
        serde_json::json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": "session/set_mode",
            "params": { "sessionId": session_id, "modeId": mode_id }
        })
    }

    /// The client learns the modes when the session is created, and
    /// starts in the one that asks.
    #[tokio::test]
    async fn session_new_advertises_the_three_modes() {
        let state = ServeState::for_test_scripted(true, Vec::new());
        let addr = spawn(state).await;
        let replies = conversation(
            &addr,
            vec![initialize_request(0), new_session_request(1)],
        )
        .await;

        let modes = &replies[1]["result"]["modes"];
        assert_eq!(modes["currentModeId"], "default", "got {modes}");
        let ids: Vec<&str> = modes["availableModes"]
            .as_array()
            .expect("availableModes is an array")
            .iter()
            .map(|m| m["id"].as_str().unwrap())
            .collect();
        assert_eq!(ids, vec!["default", "accept_edits", "bypass"]);
    }

    /// Switching modes is acknowledged and announced.
    #[tokio::test]
    async fn set_mode_switches_and_notifies() {
        let state = ServeState::for_test_scripted(true, Vec::new());
        let addr = spawn(state).await;

        let mut ws = connect(&addr).await;
        for request in [initialize_request(0), new_session_request(1)] {
            ws.send(Message::Text(request.to_string().into()))
                .await
                .unwrap();
        }

        let mut session_id = None;
        let mut saw_mode_update = false;
        loop {
            let Message::Text(t) = next_frame(&mut ws).await else {
                continue;
            };
            let v: serde_json::Value = serde_json::from_str(&t).unwrap();
            if v["id"] == 1 {
                let id = v["result"]["sessionId"].as_str().unwrap().to_string();
                ws.send(Message::Text(
                    set_mode_request(2, &id, "bypass").to_string().into(),
                ))
                .await
                .unwrap();
                session_id = Some(id);
            } else if v["method"] == "session/update"
                && v["params"]["update"]["sessionUpdate"] == "current_mode_update"
            {
                assert_eq!(v["params"]["update"]["currentModeId"], "bypass");
                saw_mode_update = true;
            } else if v["id"] == 2 {
                assert!(v["error"].is_null(), "set_mode failed: {v}");
                break;
            }
        }
        assert!(session_id.is_some());
        assert!(saw_mode_update, "a mode change must be announced");
    }

    /// `plan` is not implemented, and must not silently resolve to
    /// something else — a client told "fine" would believe the agent is
    /// planning when it is about to act.
    ///
    /// Hand-rolled rather than via `roundtrip`, because a session lives
    /// only as long as the connection that minted it: a second
    /// connection would fail on the session id, not on the mode id, and
    /// the test would pass for the wrong reason.
    #[tokio::test]
    async fn an_unknown_mode_is_invalid_params() {
        let state = ServeState::for_test_scripted(true, Vec::new());
        let addr = spawn(state).await;

        let mut ws = connect(&addr).await;
        for request in [initialize_request(0), new_session_request(1)] {
            ws.send(Message::Text(request.to_string().into()))
                .await
                .unwrap();
        }

        loop {
            let Message::Text(t) = next_frame(&mut ws).await else {
                continue;
            };
            let v: serde_json::Value = serde_json::from_str(&t).unwrap();
            if v["id"] == 1 {
                let id = v["result"]["sessionId"].as_str().unwrap().to_string();
                ws.send(Message::Text(
                    set_mode_request(2, &id, "plan").to_string().into(),
                ))
                .await
                .unwrap();
            } else if v["id"] == 2 {
                assert_eq!(v["error"]["code"], -32602, "got {v}");
                assert!(
                    v["error"]["data"]
                        .as_str()
                        .is_some_and(|d| d.contains("plan")),
                    "the error should name the mode it rejected, got {v}"
                );
                break;
            }
        }
    }
```

- [ ] **Step 2: テストが落ちることを確認**

Run: `cargo test --workspace acp::tests::session_new_advertises_the_three_modes`
Expected: FAIL — `modes` が `null`。

- [ ] **Step 3: `session/new` で modes を返す**

`src/serve/acp.rs` の `session/new` ハンドラの応答を差し替える。

```rust
                    let modes = SessionModeState::new(
                        crate::tools::policy::SessionMode::Default.id(),
                        crate::tools::policy::SessionMode::ALL
                            .into_iter()
                            .map(|m| SessionMode::new(m.id(), m.name()).description(m.description()))
                            .collect(),
                    );

                    responder.respond(NewSessionResponse::new(session_id).modes(modes))
```

import に `SessionMode`（ACP のもの）と `SessionModeState` を足す。**名前が衝突する**ので、こちらのポリシー側は `crate::tools::policy::SessionMode` とフルパスで書くか、`use agent_client_protocol::schema::v1::SessionMode as AcpSessionMode;` と別名にすること。後者を推奨する。

- [ ] **Step 4: `session/set_mode` ハンドラを足す**

`session/cancel` の `on_receive_notification` の隣に、`on_receive_request` として足す。

```rust
        .on_receive_request(
            {
                let sessions = Arc::clone(&sessions);
                async move |req: SetSessionModeRequest,
                            responder,
                            connection: ConnectionTo<Client>| {
                    let Some(mode) =
                        crate::tools::policy::SessionMode::from_id(req.mode_id.0.as_ref())
                    else {
                        // `plan` lands here. Answering with an error is
                        // the honest reply: silently picking another
                        // mode would leave the user believing the agent
                        // is planning when it is about to act.
                        return responder.respond_with_error(Error::invalid_params().data(
                            format!("unknown mode '{}'", req.mode_id),
                        ));
                    };

                    {
                        let mut guard = sessions.inner.lock().await;
                        let Some(session) = guard.get_mut(&req.session_id) else {
                            return responder.respond_with_error(Error::invalid_params().data(
                                format!("unknown session '{}'", req.session_id),
                            ));
                        };
                        session.mode = mode;
                    }

                    // Announce it: a client that changed the mode from
                    // one surface should see it reflected on the others.
                    if let Err(e) = connection.send_notification(SessionNotification::new(
                        req.session_id.clone(),
                        SessionUpdate::CurrentModeUpdate(CurrentModeUpdate::new(mode.id())),
                    )) {
                        warn!("ACP: dropped a current_mode_update: {e}");
                    }

                    responder.respond(SetSessionModeResponse::new())
                }
            },
            on_receive_request!(),
        )
```

import に `CurrentModeUpdate`、`SetSessionModeRequest`、`SetSessionModeResponse` を足す。`SetSessionModeResponse::new()` が引数を取るかは `cargo check` で確認すること。

- [ ] **Step 5: テストが通ることを確認**

Run: `cargo test --workspace`
Expected: PASS。

- [ ] **Step 6: `bypass` が実際に承認を飛ばすことを確認するテストを足す**

モードが表示だけでなく判定に効いていることを固定する。

```rust
    /// The mode is not decoration: `bypass` stops the asking.
    #[tokio::test]
    async fn bypass_mode_does_not_ask() {
        let state = ServeState::for_test_scripted(
            true,
            vec![
                crate::provider::ChatResponse {
                    text: None,
                    tool_calls: vec![crate::provider::ToolCall {
                        id: "call-1".to_string(),
                        name: "risky".to_string(),
                        input: serde_json::json!({}),
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
        state
            .tools
            .register_tool(Box::new(super::RiskyTool::new()))
            .await;
        let addr = spawn(state).await;

        let mut ws = connect(&addr).await;
        for request in [initialize_request(0), new_session_request(1)] {
            ws.send(Message::Text(request.to_string().into()))
                .await
                .unwrap();
        }

        let mut asked = 0usize;
        loop {
            let Message::Text(t) = next_frame(&mut ws).await else {
                continue;
            };
            let v: serde_json::Value = serde_json::from_str(&t).unwrap();
            if v["id"] == 1 {
                let id = v["result"]["sessionId"].as_str().unwrap().to_string();
                ws.send(Message::Text(
                    set_mode_request(2, &id, "bypass").to_string().into(),
                ))
                .await
                .unwrap();
                ws.send(Message::Text(
                    prompt_request(3, &id, text_prompt("run it")).to_string().into(),
                ))
                .await
                .unwrap();
            } else if v["method"] == "session/request_permission" {
                asked += 1;
                panic!("bypass mode must not ask, got {v}");
            } else if v["id"] == 3 {
                assert_eq!(v["result"]["stopReason"], "end_turn", "got {v}");
                break;
            }
        }
        assert_eq!(asked, 0);
    }
```

Run: `cargo test --workspace acp::tests::bypass_mode_does_not_ask`
Expected: PASS。

- [ ] **Step 7: コミット**

```bash
git add src/serve/acp.rs
git commit -m "feat(acp): session modes, advertised on new and switchable on set_mode"
```

---

### Task 9: 仕上げ — lint、ドキュメント、通し確認

**Files:**
- Modify: `README.md`（ACP 節に承認とモードの記述）
- Modify: `config.example.toml`（承認記録ファイルの場所を注記）

- [ ] **Step 1: lint と format**

```bash
cargo fmt --all
cargo clippy --workspace --all-targets -- -D warnings
```

Expected: 警告ゼロ。

- [ ] **Step 2: 全テスト**

```bash
cargo test --workspace
```

Expected: PASS。**ハングした場合**は `cargo test --workspace -- --nocapture --test-threads=1` で犯人を特定する。承認待ちで止まるテストは、`kind()` が `Other` に落ちているツールを `Ask` 経路で踏んでいる。

- [ ] **Step 3: README を更新**

`README.md` の ACP について書いてある節に、次の 3 点を足す:

- `default` / `accept_edits` / `bypass` の 3 モードがあり、Zed のモード切替から選べること
- 承認は ACP クライアントにだけ出ること。Matrix / Discord からは `shell` と MCP ツールが**使えない**こと
- 「常に許可」の記録先が `~/.config/sapphire-agent/acp-permissions.json` で、消せば全部聞き直しになること

- [ ] **Step 4: `config.example.toml` に注記**

ACP 関連の節に、承認記録は config ではなく `acp-permissions.json` に入る旨をコメントで 2〜3 行。設定項目は増えないので、キーは足さない。

- [ ] **Step 5: コミット**

```bash
git add README.md config.example.toml
git commit -m "docs: permission modes and where standing answers are kept"
```

---

## 実装後に残ること（spec の「別イシューに切り出すもの」より）

- 2 本のツールループの統合
- `plan` モード（TODO / プラン対応と一緒に）
- 引数単位の承認
- server-side `shell` の撤去（ACP `terminal/*` が動いてから）
- 拒否されたツールの `session/update` ステータス。現状は `tool_end` が `completed` を送るので、Zed には「完了」と見える。`failed` にするには `TurnHost` に口を増やす必要があり、今回は見送っている。
