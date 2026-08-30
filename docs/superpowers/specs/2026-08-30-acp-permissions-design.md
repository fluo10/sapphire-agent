# ツール実行の承認と、セッションモード

- 日付: 2026-08-30
- 対象: `src/tools/{mod,policy}.rs`, `src/tools/*_tools.rs`,
  `src/serve/{mod,acp}.rs`, `src/agent.rs`, `src/config.rs`
- 前提: `2026-08-24-acp-websocket-transport-design.md`（`/acp` エンドポイント）
- ACP: `session/request_permission`, `session/set_mode`, `CurrentModeUpdate`
  （いずれも v1 stable。`agent-client-protocol` 2.0.0）

## 背景

いま agent には**ツール実行の承認という概念が存在しない**。`shell` も
`file_delete` も、モデルが呼べば無条件に走る。Matrix / Discord から届いた
発言で本番ホスト上のシェルが撃てる状態で、これは `/acp` を Zed に向けて
開けた時点で「エディタから無承認で任意コマンドが実行できる」に変わる。

ACP には `session/request_permission` があり、Zed 側の UI もある。エージェント
側にそれを受ける仕組みが無いだけ。

もう一つ、**ツール実行ループが 2 本ある**という既存事情がこの設計を規定する。

| ループ | 場所 | 通る経路 |
|---|---|---|
| `run_llm_turn` | `src/serve/mod.rs:1824` | `/rpc`, voice, heartbeat, `/a2a`, `/acp` |
| `Agent::handle_message` 内 | `src/agent.rs:837-949` | Matrix, Discord |

`MAX_TOOL_ROUNDS` の定義まで両方に重複している（`serve/mod.rs:40` と
`agent.rs:17`）。ループの統合は本設計のスコープ外だが、**判定表が 2 つに
分裂することだけは避ける**。

## 決めたこと

1. **承認対象はツール側の分類で決める。** ACP の `ToolKind` を表示と
   ポリシーの両方に兼用し、分類を 2 つ持たない。
2. **モードは `default` / `accept_edits` / `bypass` の 3 つ。** `plan` は
   入れない。
3. **ゲートは `TurnProgress` に生やす。** デフォルト実装が `Allow` なので
   既存の実装は 1 行も変えなくてよい。
4. **判定は純関数 `tools::policy::decide` に置く。** 2 本のループが同じ
   関数を呼ぶ。
5. **チャネル（Matrix / Discord）では対話的な承認をしない。** 静的ポリシーで
   `Execute` / `Other` を拒否する。
6. **`AllowAlways` は host-local config ディレクトリに永続化する。**
   ワークスペースには置かない。

### 2. `plan` を入れない理由

`plan` は「聞くか聞かないか」ではなく「**実行せずに計画を出す**」という別次元の
動作で、system prompt の差し替え・書き込み系ツールの封じ込め・`session/update:
plan` による提示が要る。承認とは独立に設計できるので、TODO / プラン対応と
一緒にやる。

### 3. ゲートの形

```rust
pub(crate) trait TurnHost: Send + Sync {
    async fn tool_start(&self, id: &str, name: &str);
    async fn tool_end(&self, id: &str, name: &str);
    async fn turn_error(&self, message: &str);

    /// このターンがどの経路から来たか。判定表の行を選ぶ。
    fn origin(&self) -> Origin { Origin::Trusted }

    /// `decide()` が `Ask` を返したときだけ呼ばれる。既定は
    /// 「聞けないので通す」。ACP 以外は実装しない。
    async fn approve(&self, call: &ToolCall, kind: ToolKind) -> Approval {
        let _ = (call, kind);
        Approval::AllowOnce
    }
}

pub enum Approval { AllowOnce, AllowAlways, RejectOnce, RejectAlways }
```

`origin()` の既定が `Trusted` なのは、既存の実装（`SseProgress` /
`NullProgress`）が触られずに現状の挙動を保つため。`AcpProgress` は
`Origin::Acp(mode)` を返し、`agent.rs` は `TurnHost` を使わないので
`Origin::Channel` を直に `decide()` へ渡す。

`Approval` は ACP の `PermissionOptionKind` と 1:1 で対応する。`AllowAlways` /
`RejectAlways` を受け取った `AcpProgress` が永続化ファイルを更新する
（判定の純関数側は永続化を知らない）。

### 3（続き）. `ToolGate` を別 trait にしない理由

`TurnProgress` の実装は 3 つ（`SseProgress` / `NullProgress` / `AcpProgress`）、
`run_llm_turn` の呼び出しは 5 箇所しかない。独立した trait を第 2 引数で
渡す形にすると全呼び出し元に「ゲート無し」を配って回る必要があり、得るものは
名前の綺麗さだけ。

ただし `TurnProgress` が「報告」だけでなくなるので、**`TurnHost` にリネーム
する**（機械的、8 箇所）。

`ToolSet` に持たせる案は却下。`ToolSet` はプロセス共有なので、セッション単位の
モードを持てない。Matrix 経路のモードを Zed が書き換えてしまう。

### 5. チャネルで聞かない理由

理由は 2 つあり、どちらも単独で決定的。

- **チャネルのターンは非同期。** 承認待ちでターンをブロックすると、人間が
  気づかない限り数時間ぶら下がる。`run_llm_turn` は「キャンセルされた future が
  drop されると履歴の書き戻しが走らない」という既知の欠陥を抱えており
  （`src/serve/acp.rs` 冒頭）、長時間ぶら下がるターンとは相性が悪い。
- **承認を LLM のターン内に通すと、モデルが自分自身の許可申請を仲介する。**
  「ユーザーは yes と言いました」とモデルが判断すれば通る。out-of-band に
  するには `Channel` trait に問い合わせ口を足す必要があり、`listen` の
  ストリームを分岐させる実装が Matrix / Discord それぞれに要る。

Discord のボタン（serenity の `builder` feature + `InteractionCreate`）は UX
としては最良だが Discord 専用で、Matrix と voice には別途答えが要る。ブロック
問題も残る。

## 分類

`Tool` trait にメソッドを **1 つだけ**足す。既定は最も厳しい側に倒す。

```rust
fn kind(&self) -> ToolKind { ToolKind::Other }
```

| ToolKind | ツール |
|---|---|
| `Read` | `file_read`, `memory_read`, `transcript_read`, `recall_image` |
| `Search` | `dir_list`, `dir_walk`, `workspace_search`, `speaker_candidates`, `timer_status` |
| `Fetch` | `web_search`, `weather` |
| `Edit` | `file_write`, `file_append`, `memory_add`, `memory_update`, `memory_append`, `speaker_promote`, `timer_set`, `timer_preset` |
| `Delete` | `file_delete`, `memory_remove`, `timer_cancel` |
| `Execute` | `shell` |
| `Other` | `mcp_reconnect`, `workspace_sync`, **および全 MCP ツール** |

MCP ツールが `Other` に落ちるのは意図的。外部由来のものは最も厳しいバケツに
入れる。`build_tools_for_client` が作るラッパは `kind()` を実装しない
＝ 既定の `Other` になる。

## 判定表

```rust
// src/tools/policy.rs — 純関数。両方のループから呼ばれる。
pub enum Origin { Acp(SessionMode), Channel, Trusted }
pub enum Decision { Allow, Ask, Deny }

pub fn decide(origin: Origin, kind: ToolKind) -> Decision;
```

| ToolKind | ACP `default` | ACP `accept_edits` | ACP `bypass` | `Channel` | `Trusted` |
|---|---|---|---|---|---|
| `Read` / `Search` / `Fetch` / `Think` | Allow | Allow | Allow | Allow | Allow |
| `Edit` / `Delete` / `Move` | **Ask** | Allow | Allow | Allow | Allow |
| `Execute` / `Other` | **Ask** | **Ask** | Allow | **Deny** | Allow |

- `Trusted` は `/rpc`・voice・heartbeat・`/a2a`。既に認証済みのローカル経路で、
  **挙動は現状から一切変わらない**。
- `Channel` は Matrix / Discord。`web_search` は `Fetch` なので通る。`shell` と
  MCP ツールは通らなくなる。**これが本設計における唯一の既存挙動の変更**。
- `Deny` のときもモデルには `tool_result` で理由を返す。モデルは「この経路では
  使えない」と理解して別の手段を試せる。

## ゲートの位置

2 本のループで形が違うので、別々に書く。

### `run_llm_turn`（`/acp` を含む）

ツールは `join_all` で並行実行されている（`serve/mod.rs:2009`）。承認をその
まま並行にすると Zed に複数ダイアログが同時に出るので、**承認だけ直列、実行は
従来どおり並行**にする。

```
1. progress.tool_start(...)                    ← 現状のまま
2. 各 call を順に decide() → Ask なら approve()  ← 新規（直列）
3. Allow のものだけ join_all で並行実行           ← 現状のまま
4. Deny のものは実行せず合成 ToolOutput
5. progress.tool_end(...)                      ← 現状のまま
```

`tool_start` は承認前に出す。Zed 側で「何を承認しようとしているか」が
ツールリストに出ている状態にしたいため。

### `agent.rs`（Matrix / Discord）

こちらは `tokio::spawn` した handle を順に `await` する形（`agent.rs:937-967`）で、
そもそも `TurnProgress` を使っていない。`Origin::Channel` は `Ask` を返さないので、
**`approve()` は呼ばれない**。`decide()` が `Deny` を返した call を `tokio::spawn`
に渡さず、合成 `ToolOutput` に差し替えるだけ。直列化も不要で、並行実行の形は
変わらない。

## モード

`AcpSession` に `mode: SessionModeId` を持たせる。

- `session/new` の応答で `modes: SessionModeState` を返す（`current_mode_id` は
  `default`）。`NewSessionResponse.modes` は素の `Option` で capability gate は
  無いため、`initialize` 側の変更は不要。
- `session/set_mode` ハンドラを追加。セッションのモードを差し替え、
  `SessionUpdate::CurrentModeUpdate` を通知する。
- 未知の `mode_id` は `invalid_params`。
- モードは**セッション単位**。同一接続の別セッションには波及しない。

## 永続化

`dirs.config_dir().join("acp-permissions.json")`。ワークスペースではない
（同期対象であり、承認は「このマシンを信用するか」というホストローカルな信頼
判断のため。`main.rs:148` の「Credentials, MCP servers, bind addresses and
machine paths are host-local」と揃える）。

```json
{
  "profiles": {
    "zed": {
      "always_allow": ["file_write"],
      "always_reject": ["file_delete"]
    }
  }
}
```

- ルームプロファイル単位・**ツール名単位**。引数単位（「このパスなら常に」）は
  スコープ外。
- `always_reject` が `always_allow` より優先（安全側）。
- 起動時にロード、書き込みは一時ファイル → rename の atomic replace。
- ファイルが無い・壊れている場合は空として扱い、警告を出して続行する。承認
  記録が読めないことは起動を止める理由にならない。

## テスト

ユニット:

- 全 25 ツールの `kind()` を表で固定する。ツールを足して `kind()` を書き忘れると
  `Other` ＝ 最も厳しい側に落ちるので、失敗方向は安全。
- `decide()` の判定表を `Origin` × `ToolKind` の全組み合わせで固定。
- 永続化の読み書き、`always_reject` 優先、壊れたファイルからの復帰。

ACP E2E（既存の `conversation` ヘルパを使う）:

- `Ask` のツールで `session/request_permission` が飛ぶ
- `AllowOnce` で実行される / `RejectOnce` で `tool_result` が denied になる
- `AllowAlways` で 2 回目は聞かれない、かつファイルに書かれる
- `session/set_mode` で `accept_edits` に移ると `Edit` が聞かれなくなる
- 未知の `mode_id` が `invalid_params` になる

回帰:

- `SseProgress` 経路（`/rpc`・voice）は承認を求めず、`shell` が通る
- `agent.rs` 側のループで `shell` が `Deny` になり、`web_search` は通る

## 別イシューに切り出すもの

- **2 本のツールループの統合。** 本設計は判定を共有するだけで、ループ自体は
  重複したまま残る。
- **`plan` モード。** TODO / プラン対応（`session/update: plan`）と一緒に。
- **引数単位の承認。**「このディレクトリ配下の `file_write` なら常に許可」。
  ACP の `PermissionOption` は表現できるが、パス正規化の設計が要る。
- **server-side `shell` の撤去。** ACP の `terminal/*` が動いてからの話。本設計の
  「チャネルからは `Execute` を拒否」はその第一歩にあたる。
