# シェル・ファイル系ツール統合（セッション種別ルーティング）実装計画書

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** シェル・ファイル系ツールを1セットに統合し、実行先（エディタのマシン／エージェント自身のマシン）をセッション種別で動的に決める。`client_*` プレフィクスを廃止し、ACP に無い操作（`file_delete` / `dir_list` / `dir_walk`）は terminal 経由でクライアント上に実装する。

**Architecture:** 既存の task-local（`server/src/tools/acp_client.rs` の `scope_acp_client` / `current_acp_client`）を唯一のルーティング判定に使う。`server/src/serve/mod.rs` の `TurnLoop::run` は既にツール実行をこのスコープで包んでいるので、**新しいルーティング機構は追加しない**。各ツールは `execute` の先頭で `current_acp_client()` を読み、`Some` なら ACP 経路、`None` ならエージェント自身の経路に分岐する。可視性は `visible_tool_predicate` が「非 ACP なら `host_access` のみ」「ACP なら capability のみ」で決め、フォールバックは行わない。

**Tech Stack:** Rust（tokio, async-trait, serde_json）、ACP（`agent_client_protocol`）、既存の `WorkspaceState` / `ToolSet` / `Tool` trait。

**Spec:** `docs/superpowers/specs/2026-09-16-tool-unification-design.md`

## Global Constraints

- コード・コメント・コミットメッセージは英語（`CONTRIBUTING.md` 規約）。ドキュメント（`docs/superpowers/**`）は日本語。
- コミットは conventional commits。スコープは `tools` / `serve` / `policy` / `docs` など既存慣行に従う（`CLAUDE.md` のスコープ表）。
- ツール名は統合後の10個から動かさない: `file_read`, `file_write`, `file_append`, `file_delete`, `dir_list`, `dir_walk`, `shell`, `shell_start`, `shell_output`, `shell_kill`。
- `ToolKind` は現行の割り当てを維持する（`policy::decide` には手を入れない）。
- フォールバック禁止: ACP セッションで capability が無い／クライアントがエラーを返した場合、エージェント自身のマシンへは絶対に切り替えない。
- 既存テスト様式に従う: `acp_client::tests::FakeClient` ＋ `scope_acp_client`、`ServeState::for_test_scripted*` ＋ `ScriptedProvider`、`policy.rs` の純関数テスト。
- 各タスクの最後に `cargo fmt --all` を掛けてからコミットする。

## 設計の確定事項（タスク間で共有する前提）

1. **ルーティング**: 各ツールの `execute` 冒頭で
   ```rust
   match crate::tools::acp_client::current_acp_client() {
       Some(client) => client_side_impl(&client, input).await,   // エディタのマシン
       None => agent_side_impl(input).await,                     // エージェント自身のマシン
   }
   ```
   判定はターン単位の task-local なので、サブエージェントのツール実行にも同じ判定が届く
   （`SubagentHost::acp_client()` が親に転送しているため）。

2. **モジュール配置**（ファイルは増やさない）:
   - `server/src/tools/builtin_tools.rs` — 統合後の7ツール（`FileReadTool`, `FileWriteTool`,
     `FileAppendTool`, `FileDeleteTool`, `DirListTool`, `DirWalkTool`, `ShellTool`）。
     各 `execute` が分岐する。エージェント側の実装は現行コードをそのまま残す。
   - `server/src/tools/client_tools.rs` — ACP 側の実装関数群（`fn client_read(client, input)`,
     `client_write`, `client_append`, `client_delete`, `client_dir_list`, `client_dir_walk`）、
     既存の実行ヘルパ（`format_finished` / `format_timed_out` / `format_exit_status` /
     `cap_error` / `clamp_timeout`）、およびライフサイクル3ツール
     （`ShellStartTool` / `ShellOutputTool` / `ShellKillTool`）。旧 `ClientFileRead` /
     `ClientFileWrite` / `ClientShell` の struct は消える。
   - `server/src/tools/client_exec.rs` — 変更なし（`run_client_command` は `shell` と
     統合後の3ツールがそのまま使う）。

3. **入力スキーマはホスト側現行版が唯一の正**。ACP 経路への写像は実装側で行う:
   `file_read` の `offset`→`line`、`limit`→`limit`。`file_append` は ACP では
   read（全体）＋連結＋write。

4. **terminal 経由の3ツールの出力形式はエージェント側と同一**にする。整形は Rust 側で行い、
   スクリプトは `D\t<path>` / `F\t<path>` の行を返すだけにする。

5. **`host_tool_denied(name, host_access_enabled, routed_to_client)`** の新シグネチャ。
   `routed_to_client` は `progress.acp_client().is_some()`。ACP セッションでは常に
   `false` を返す（＝拒否しない）。

---

## File Structure

| ファイル | 責務 | 変更 |
|---|---|---|
| `server/src/tools/builtin_tools.rs` | 統合後の7ツール。`execute` でセッション種別分岐 | 大幅改修 |
| `server/src/tools/client_tools.rs` | ACP 側実装関数群、端末ライフサイクル3ツール、整形ヘルパ | 再編（旧 `ClientFileRead`/`ClientFileWrite`/`ClientShell` を削除） |
| `server/src/tools/client_exec.rs` | `run_client_command`（create→wait→output→release、セッション上限の予約） | 変更なし |
| `server/src/tools/mod.rs` | `default_tool_set()` の登録、`kinds()` の分類テスト | 登録名の差し替え |
| `server/src/tools/policy.rs` | `HOST_TOOLS`、`host_tool_denied`、`partition_without_asking` | 引数追加＋テスト |
| `server/src/serve/mod.rs` | `visible_tool_predicate`、許可ゲート、`run_llm_turn` の可視性テスト群 | 改修 |
| `server/src/agent.rs` | Matrix/Discord 経路の `specs_filtered` と `partition_without_asking` 呼び出し | 引数追随（挙動不変） |
| `server/src/serve/acp.rs` | capability ログのコメント中のツール名 | 文言追随のみ |
| `server/src/tools/subagent.rs` | `SubagentHost` の doc と、旧名を使うテストリテラル | テスト追随 |
| `server/src/agents.rs` | 定義ファイル例のテストリテラル | 追随 |
| `server/src/tools/acp_client.rs` | `AcpClient` trait の doc（「client-side tools」→ 統合後の説明） | コメントのみ |
| `server/config.example.toml` | `host_access` の意味、旧 `client_*` 名の説明 | 追随 |
| `server/templates/workspace/config.toml` | `origin = "trusted"` の説明 | 追随 |
| `README.md` / `README.ja.md` | ツール一覧、「Client-side tools: whose machine」節 | 書き換え |

---

## Task 1: ルーティング基盤と `file_read` / `file_write` / `file_append` / `file_delete` / `shell`

**Files:**
- Modify: `server/src/tools/builtin_tools.rs`（`FileReadTool` 約24行、`FileWriteTool` 約462行、`FileDeleteTool` 約543行、`FileAppendTool` 約615行、`ShellTool` 約915行）
- Modify: `server/src/tools/client_tools.rs`（ACP 側実装関数を追加、旧 `ClientFileRead`/`ClientFileWrite`/`ClientShell` を削除）
- Modify: `server/src/tools/mod.rs`（登録）

**Interfaces:**
- Produces: `client_tools::{client_read, client_write, client_append, client_delete}(client: &Arc<dyn AcpClient>, input: &Value) -> Result<String>`
- Produces: `ShellTool::execute` の分岐（ACP 経路は `client_exec::run_client_command` を `timeout` 秒で呼ぶ）
- Consumes: `crate::tools::acp_client::{current_acp_client, AcpClient}`, `crate::tools::client_exec::run_client_command`

- [ ] **Step 1: 失敗するテストを書く**（`server/src/tools/builtin_tools.rs` の tests モジュール）

```rust
/// The point of the unification: with an ACP client scoped, `file_read`
/// reads the *editor's* machine. The workspace and the fake client are
/// given deliberately different contents so the assertion cannot pass by
/// accident.
#[tokio::test]
async fn file_read_reads_the_clients_machine_inside_an_acp_session() {
    let (state, tool) = file_read_tool_for_test();
    // The agent's own file, which must NOT be what comes back.
    write_workspace_file(&state, "note.txt", "agent side\n");

    let client = std::sync::Arc::new(crate::tools::acp_client::tests::FakeClient::new());
    client.queue_read_result("client side\n");
    let out = crate::tools::acp_client::scope_acp_client(
        std::sync::Arc::clone(&client) as std::sync::Arc<dyn crate::tools::acp_client::AcpClient>,
        tool.execute(&serde_json::json!({"path": "note.txt"})),
    )
    .await
    .unwrap();

    assert_eq!(out, "client side\n");
    assert!(
        client.read_paths().contains(&"note.txt".to_string()),
        "the read must have gone to the client, got {:?}",
        client.read_paths()
    );
}

/// Without a client scoped, the same tool reads the agent's own machine —
/// unchanged behaviour.
#[tokio::test]
async fn file_read_reads_the_agents_machine_outside_an_acp_session() {
    let (state, tool) = file_read_tool_for_test();
    write_workspace_file(&state, "note.txt", "agent side\n");
    let out = tool
        .execute(&serde_json::json!({"path": "note.txt"}))
        .await
        .unwrap();
    assert!(out.contains("agent side"));
}
```

  注: `FakeClient` に `queue_read_result` / `read_paths()`（呼ばれたパスの記録）を追加する
  （`server/src/tools/acp_client.rs` の `FakeClient`、既存の `queue_terminal_stdout` /
  `queue_terminal_result` と同型）。`file_read_tool_for_test` / `write_workspace_file` は
  既存の `AppContext` / `WorkspaceState` テストヘルパ（`builtin_tools.rs` tests 内の setup）を
  使って書く。

- [ ] **Step 2: 失敗を確認**

```sh
cargo test -p sapphire-agent-server --lib tools::builtin_tools::tests::file_read_reads
```
  期待: コンパイルエラー（`queue_read_result` 未定義、`client_tools::client_read` 未定義）。

- [ ] **Step 3: `client_tools.rs` に ACP 側実装を追加**

```rust
// ---------------------------------------------------------------------------
// ACP-side implementations for the unified tools
// ---------------------------------------------------------------------------
//
// These are not tools any more: `file_read`/`file_write`/`file_append`/
// `file_delete` (`src/tools/builtin_tools.rs`) pick between these and their
// agent-side bodies by reading `current_acp_client()`. They live here so
// everything that knows about ACP's wire surface stays in one module.

/// `file_read` against the editor's machine. The tool's own `offset`/
/// `limit` map onto ACP's `line`/`limit`, which exist for exactly this
/// reason: the full file does not have to cross the wire to read a range.
pub(crate) async fn client_read(
    client: &std::sync::Arc<dyn AcpClient>,
    input: &serde_json::Value,
) -> Result<String> {
    let path = input["path"].as_str().context("missing 'path'")?;
    let line = input["offset"].as_u64().map(|v| v as u32);
    let limit = input["limit"].as_u64().map(|v| v as u32);
    client.read_text_file(path, line, limit).await
}

/// `file_write` against the editor's machine.
pub(crate) async fn client_write(
    client: &std::sync::Arc<dyn AcpClient>,
    input: &serde_json::Value,
) -> Result<String> {
    let path = input["path"].as_str().context("missing 'path'")?;
    let content = input["content"].as_str().context("missing 'content'")?;
    client.write_text_file(path, content).await?;
    Ok(format!(
        "Written: {path} ({} bytes)",
        content.len()
    ))
}

/// `file_append` against the editor's machine.
///
/// ACP has no append, so this is read → concatenate → write. Two
/// consequences the tool description has to carry, because a model that
/// does not know them will use this where it should use a shell:
/// the whole file crosses the wire twice, and the pair is not atomic —
/// a write by another process between the read and the write is lost.
/// A missing parent directory is a `fs/write_text_file` error, not a
/// silent `mkdir`.
pub(crate) async fn client_append(
    client: &std::sync::Arc<dyn AcpClient>,
    input: &serde_json::Value,
) -> Result<String> {
    let path = input["path"].as_str().context("missing 'path'")?;
    let content = input["content"].as_str().context("missing 'content'")?;
    let existing = match client.read_text_file(path, None, None).await {
        Ok(existing) => existing,
        // A missing file is the ordinary "create it" case, not an error —
        // `file_append`'s agent-side contract says it creates the file.
        // Every other read failure is reported as-is.
        Err(_) => String::new(),
    };
    let mut merged = existing;
    merged.push_str(content);
    client.write_text_file(path, &merged).await?;
    Ok(format!("Appended: {path} (+{} bytes)", content.len()))
}
```

- [ ] **Step 4: `builtin_tools.rs` の5ツールに分岐を入れる**

  `FileReadTool::execute` の冒頭:

```rust
if let Some(client) = crate::tools::acp_client::current_acp_client() {
    return crate::tools::client_tools::client_read(&client, input).await;
}
// … 既存のエージェント側実装はそのまま
```

  `FileWriteTool` / `FileAppendTool` も同型（`client_write` / `client_append`）。
  `FileDeleteTool` は `client_delete` を呼ぶ（Step 5）。
  `ShellTool::execute` の冒頭:

```rust
if let Some(client) = crate::tools::acp_client::current_acp_client() {
    // Same one-shot contract as the agent-side body: wait up to
    // `timeout`, and on timeout hand back a *running* handle rather than
    // killing the command — see `client_exec::run_client_command` and
    // the old `client_shell` doc for why releasing would re-run
    // non-idempotent commands.
    let command = input["command"].as_str().context("missing 'command'")?;
    let timeout = crate::tools::client_tools::clamp_timeout(input["timeout"].as_u64());
    let cwd = input["workdir"].as_str();
    // ACP takes a command plus argv, not a shell string — so the agent-side
    // `shell` tool's command line goes through `sh -c` explicitly here,
    // which is what the agent-side body already does with `$SHELL -c`.
    let args = vec!["-c".to_string(), command.to_string()];
    let run = crate::tools::client_exec::run_client_command(
        &client,
        "sh",
        &args,
        cwd,
        timeout,
    )
    .await?;
    return match run.timed_out_handle {
        Some(h) => Ok(crate::tools::client_tools::format_timed_out(&h, timeout)),
        None => {
            let status = run
                .status
                .expect("run_client_command always sets `status` when it does not time out");
            let mut out = crate::tools::client_tools::format_finished(&run.output, &status);
            if let Some(warning) = run.release_warning {
                out.push('\n');
                out.push_str(&warning);
            }
            Ok(out)
        }
    };
}
```

  注: `cwd` は `workdir`。未指定なら `None` → `AcpClientHandle` がセッション cwd を既定にする
  （既存 `client_shell` と同じ）。

- [ ] **Step 5: `client_delete` を terminal 経由で実装**（Task 2 の共通ヘルパを先に置く）

```rust
/// The waiting budget for the short, local commands the unified tools run
/// on the client (`rm`, `find`, `test`). Same value as `skill_tools`'
/// `LOCAL_TIMEOUT`: these cost about as little as reading a file.
pub(crate) const CLIENT_LOCAL_TIMEOUT: std::time::Duration =
    std::time::Duration::from_secs(30);

/// Run one `bash -c <script> <argv0> <args...>` on the client and require
/// it to have finished with exit code 0, returning its stdout.
///
/// Paths travel as **positional arguments**, never interpolated into the
/// script: `terminal/create` takes a command and an argv separately, so
/// there is no shell quoting to get wrong and a path with spaces or a
/// quote in it is safe by construction.
///
/// A timeout is reported as an error naming the terminal handle rather
/// than swallowing it: a handle left tracked counts against the session's
/// 8-terminal cap (`MAX_TERMINALS_PER_SESSION`), so the model has to be
/// told which one to free with `shell_kill`.
pub(crate) async fn run_client_bash(
    client: &std::sync::Arc<dyn AcpClient>,
    script: &str,
    argv0: &str,
    args: &[String],
) -> Result<String> {
    let mut full = Vec::with_capacity(args.len() + 1);
    full.push(argv0.to_string());
    full.extend(args.iter().cloned());
    let run = run_client_command(client, "bash", &{
        let mut v = vec!["-c".to_string(), script.to_string()];
        v.extend(full);
        v
    }, None, CLIENT_LOCAL_TIMEOUT)
    .await?;
    if let Some(handle) = run.timed_out_handle {
        anyhow::bail!(
            "timed out after {}s on the editor's machine; the command is still \
             running as terminal {handle}. Use shell_output to check on it, or \
             shell_kill to stop it.",
            CLIENT_LOCAL_TIMEOUT.as_secs()
        );
    }
    let status = run
        .status
        .expect("run_client_command always sets `status` when it does not time out");
    if status.signal.is_some() || status.exit_code != Some(0) {
        anyhow::bail!(
            "the command failed on the editor's machine: {}",
            format_exit_status(&status).trim()
        );
    }
    Ok(run.output.output)
}

const DELETE_SH: &str = r#"
if [ -d "$1" ]; then
  echo "is a directory" >&2
  exit 1
fi
if [ ! -e "$1" ]; then
  echo "no such file" >&2
  exit 1
fi
rm -- "$1"
"#;

/// `file_delete` against the editor's machine. ACP has no delete, so this
/// is `rm` over the terminal. The `-d` check is what keeps `dir_list`'s
/// sibling `-r` flag undeclared: `file_delete`'s agent-side contract is
/// "files, never directories".
pub(crate) async fn client_delete(
    client: &std::sync::Arc<dyn AcpClient>,
    input: &serde_json::Value,
) -> Result<String> {
    let path = input["path"].as_str().context("missing 'path'")?;
    run_client_bash(client, DELETE_SH, path, &[]).await?;
    Ok(format!("Deleted: {path}"))
}
```

- [ ] **Step 6: 旧 struct を削除し、`default_tool_set()` を差し替える**

  `client_tools.rs` から `ClientFileRead` / `ClientFileWrite` / `ClientShell` を削除し、
  `no_editor_error()` も削除する（統合後は「エディタが無い」＝非 ACP 経路であり、エラーではない）。
  `server/src/tools/mod.rs` の `default_tool_set()`（約305行）で
  `Box::new(ClientFileRead::new())` / `ClientFileWrite` / `ClientShell` の3行を消し、
  `ClientShellStart` / `ClientShellOutput` / `ClientShellKill` の登録は残す。
  `kinds()` の分類テスト（約484行）を統合後の名前集合に更新する。

- [ ] **Step 7: テストが通ることを確認**

```sh
cargo test -p sapphire-agent-server --lib tools::builtin_tools::tests
cargo test -p sapphire-agent-server --lib tools::client_tools::tests
cargo test -p sapphire-agent-server --lib tools::tests
```
  期待: 全パス。

- [ ] **Step 8: コミット**

```sh
cargo fmt --all
git add server/src/tools/builtin_tools.rs server/src/tools/client_tools.rs \
        server/src/tools/client_exec.rs server/src/tools/acp_client.rs server/src/tools/mod.rs
git commit -m "feat(tools): route file_read/file_write/file_append/file_delete/shell by session type"
```

---

## Task 2: `dir_list` / `dir_walk` の terminal 経由実装（#262 吸収）

**Files:**
- Modify: `server/src/tools/client_tools.rs`（`LIST_SH` / `WALK_SH` / `parse_find_output`）
- Modify: `server/src/tools/builtin_tools.rs`（`DirListTool` 約696行、`DirWalkTool` 約770行 に分岐）

**Interfaces:**
- Consumes: `client_tools::run_client_bash`, `CLIENT_LOCAL_TIMEOUT`（Task 1）
- Produces: `client_tools::{client_dir_list, client_dir_walk}(client, input) -> Result<String>`

- [ ] **Step 1: 失敗するテストを書く**

```rust
/// `#262`'s feature, over the interface the old design rejected: the
/// listing is a shell command on the client, and the *shape* the model
/// sees is this side's — so a `dir_list` result reads identically
/// whichever machine it came from.
#[tokio::test]
async fn dir_list_shapes_the_clients_find_output_like_the_agents_own() {
    let client = std::sync::Arc::new(FakeClient::new());
    // Deliberately unsorted, and with a directory in the middle: the
    // sort and the trailing slash are this side's job, not the script's.
    client.queue_terminal_stdout("F\t/z/b.txt\nD\t/z/sub\nF\t/z/a.txt\n");
    let out = scope_acp_client(
        std::sync::Arc::clone(&client) as std::sync::Arc<dyn AcpClient>,
        crate::tools::client_tools::client_dir_list(&client, &serde_json::json!({"path": "/z"})),
    )
    .await
    .unwrap();

    assert_eq!(out, "/z/a.txt\n/z/b.txt\n/z/sub/");
    let script = client.last_terminal_command().expect("a terminal was created");
    assert!(script.contains("-mindepth 1"), "got: {script}");
    assert!(script.contains("-maxdepth 1"), "got: {script}");
}

/// The walk honours `max_depth`/`max_entries` and reports truncation with
/// the same marker the agent-side walk uses — the model must not have to
/// learn a second vocabulary for "there was more".
#[tokio::test]
async fn dir_walk_truncates_with_the_same_marker_as_the_agent_side() {
    let client = std::sync::Arc::new(FakeClient::new());
    let mut out = String::new();
    for i in 0..3 {
        out.push_str(&format!("F\t/z/f{i}\n"));
    }
    client.queue_terminal_stdout(&out);

    let text = scope_acp_client(
        std::sync::Arc::clone(&client) as std::sync::Arc<dyn AcpClient>,
        crate::tools::client_tools::client_dir_walk(
            &client,
            &serde_json::json!({"path": "/z", "max_depth": 2, "max_entries": 2}),
        ),
    )
    .await
    .unwrap();

    assert!(text.contains("/z/f0"), "got: {text}");
    assert!(
        text.contains("[truncated — more than 2 entries"),
        "got: {text}"
    );
    let script = client.last_terminal_command().unwrap();
    assert!(script.contains("-maxdepth 3"), "max_depth + 1, got: {script}");
    assert!(script.contains("head -n 3"), "max_entries + 1, got: {script}");
}
```

  注: `FakeClient` に `queue_terminal_stdout`（既存）と `last_terminal_command()` を追加する。
  既存の `queue_terminal_stdout` は「1回の `terminal/output` に対する固定出力」なので、
  そのまま使える。

- [ ] **Step 2: 失敗を確認**

```sh
cargo test -p sapphire-agent-server --lib tools::client_tools::tests::dir_
```

- [ ] **Step 3: スクリプトと解釈を実装**

```rust
/// Print one `D<TAB>path` / `F<TAB>path` line per entry.
///
/// `LC_ALL=C` is not cosmetic: the agent-side tools sort with Rust's
/// `PathBuf` ordering, and a locale-aware `sort` would disagree with it
/// on names containing case or punctuation — two machines, two orders,
/// one tool. `-print0`/`-0` is deliberately not used: it would buy
/// newline-in-filename support at the cost of assuming `sort -z`, which
/// BSD and GNU spell the same way but fewer clients ship.
const CLASSIFY_LINE: &str = r#"
while IFS= read -r p; do
  if [ -d "$p" ]; then printf 'D\t%s\n' "$p"; else printf 'F\t%s\n' "$p"; fi
done
"#;

const LIST_SH: &str = r#"
set -e
find "$1" -mindepth 1 -maxdepth 1 | LC_ALL=C sort | {
  while IFS= read -r p; do
    if [ -d "$p" ]; then printf 'D\t%s\n' "$p"; else printf 'F\t%s\n' "$p"; fi
  done
}
"#;

const WALK_SH: &str = r#"
set -e
find "$1" -mindepth 1 -maxdepth "$2" | LC_ALL=C sort | head -n "$3" | {
  while IFS= read -r p; do
    if [ -d "$p" ]; then printf 'D\t%s\n' "$p"; else printf 'F\t%s\n' "$p"; fi
  done
}
"#;

/// Turn the script's `D`/`F` lines into what `dir_list`/`dir_walk`
/// return on the agent's own machine: sorted, directories with a
/// trailing slash, and `(empty) <path>` when there is nothing.
///
/// `limit` is `max_entries + 1` (or `usize::MAX` for a listing that
/// cannot truncate): one more than the caller will show, so "there was
/// more" is decidable without a second round trip.
fn shape_entries(stdout: &str, path: &str, limit: usize) -> (Vec<String>, bool) {
    let mut entries: Vec<(String, bool)> = stdout
        .lines()
        .filter_map(|line| {
            let (kind, name) = line.split_once('\t')?;
            let is_dir = kind == "D";
            let shown = if is_dir {
                format!("{name}/")
            } else {
                name.to_string()
            };
            Some((shown, is_dir))
        })
        .collect();
    entries.sort_unstable_by(|a, b| a.0.cmp(&b.0));
    let truncated = entries.len() > limit;
    entries.truncate(limit);
    if entries.is_empty() {
        return (vec![format!("(empty) {path}")], false);
    }
    (entries.into_iter().map(|(s, _)| s).collect(), truncated)
}

pub(crate) async fn client_dir_list(
    client: &std::sync::Arc<dyn AcpClient>,
    input: &serde_json::Value,
) -> Result<String> {
    let path = input["path"].as_str().context("missing 'path'")?;
    let stdout = run_client_bash(client, LIST_SH, path, &[]).await?;
    let (entries, _) = shape_entries(&stdout, path, usize::MAX);
    Ok(entries.join("\n"))
}

pub(crate) async fn client_dir_walk(
    client: &std::sync::Arc<dyn AcpClient>,
    input: &serde_json::Value,
) -> Result<String> {
    let path = input["path"].as_str().context("missing 'path'")?;
    let max_depth = input["max_depth"].as_u64().unwrap_or(5).min(20);
    let max_entries = input["max_entries"].as_u64().unwrap_or(500).clamp(1, 5000) as usize;
    let stdout = run_client_bash(
        client,
        WALK_SH,
        path,
        &[
            // `max_depth = 0` means "direct children only", matching the
            // agent side — hence `+ 1`, since `find -maxdepth` counts the
            // starting point as depth 0.
            (max_depth + 1).to_string(),
            (max_entries + 1).to_string(),
        ],
    )
    .await?;
    let (mut entries, truncated) = shape_entries(&stdout, path, max_entries);
    if truncated {
        entries.push(format!(
            "[truncated — more than {max_entries} entries; raise max_entries or narrow path]"
        ));
    }
    Ok(entries.join("\n"))
}
```

- [ ] **Step 4: `dir_list` / `dir_walk` に分岐を入れる**

  `DirListTool::execute` / `DirWalkTool::execute` の冒頭に、`file_read` と同じ形の
  `if let Some(client) = current_acp_client() { return client_dir_list(&client, input).await; }`
  を追加する（`dir_walk` も同型）。

- [ ] **Step 5: テストと fmt**

```sh
cargo test -p sapphire-agent-server --lib tools::client_tools::tests
cargo test -p sapphire-agent-server --lib tools::builtin_tools::tests
cargo fmt --all
```

- [ ] **Step 6: コミット**

```sh
git add server/src/tools/client_tools.rs server/src/tools/builtin_tools.rs server/src/tools/acp_client.rs
git commit -m "feat(tools): implement dir_list/dir_walk on the client over the terminal"
```

---

## Task 3: `policy.rs` / `visible_tool_predicate` / `host_access` の改修

**Files:**
- Modify: `server/src/tools/policy.rs`（`HOST_TOOLS` 約192行、`host_tool_denied` 約203行、`partition_without_asking` 約266行、tests）
- Modify: `server/src/serve/mod.rs`（`visible_tool_predicate` 約2182行、許可ゲート 約2553〜2580行、可視性テスト群 約5440行〜）
- Modify: `server/src/agent.rs`（`specs_filtered` 約562行、`partition_without_asking` 約759行）

**Interfaces:**
- Produces: `policy::host_tool_denied(name: &str, host_access_enabled: bool, routed_to_client: bool) -> bool`
- Produces: `policy::partition_without_asking(origin, calls, kinds, host_access_enabled, routed_to_client)`
- Produces: `serve::visible_tool_predicate(host_access_enabled, has_client, client_fs_read, client_fs_write, client_terminal)`（シグネチャ不変、本体のみ）

- [ ] **Step 1: 失敗するテストを書く**（`server/src/tools/policy.rs` tests）

```rust
/// The meaning change #270 makes: with host access off, a call routed to
/// the editor is not the deployment's own disk and has nothing to do with
/// this gate. The visibility rule for that turn is its declared
/// capabilities, checked elsewhere.
#[test]
fn a_call_routed_to_the_client_is_never_denied_by_the_host_gate() {
    for name in HOST_TOOLS {
        assert!(!host_tool_denied(name, false, true), "{name}");
    }
}

/// Unchanged: with no client on the other end the gate behaves exactly as
/// it did, for every origin.
#[test]
fn a_call_routed_to_the_agent_is_still_denied_with_host_access_off() {
    for name in HOST_TOOLS {
        assert!(host_tool_denied(name, false, false), "{name}");
        assert!(!host_tool_denied(name, true, false), "{name}");
    }
}
```

  `server/src/serve/mod.rs` の可視性テスト（`client_filtering_test_set` 約5444行、
  `tool_names_for_turn` 約5464行）も統合後の名前で書き換える:

```rust
    fn client_filtering_test_set() -> ToolSet {
        let names = [
            "file_read", "file_write", "file_append", "file_delete",
            "dir_list", "dir_walk", "shell",
            "skill", "skill_install", "skill_update", "skill_uninstall",
        ];
        let mut tools: Vec<Box<dyn crate::tools::Tool>> = names
            .iter()
            .map(|name| Box::new(NamedStubTool::new(name)) as Box<dyn crate::tools::Tool>)
            .collect();
        tools.push(Box::new(crate::tools::client_tools::ShellStartTool::new()));
        tools.push(Box::new(crate::tools::client_tools::ShellOutputTool::new()));
        tools.push(Box::new(crate::tools::client_tools::ShellKillTool::new()));
        ToolSet::new(tools, Vec::new())
    }
```

  期待を固定するテストを3本置く:

```rust
/// Inside an ACP session the seven names are gated on capabilities
/// alone — `host_access` is not consulted at all, because every one of
/// them reaches the editor.
#[tokio::test]
async fn an_acp_turn_sees_the_seven_names_by_capability_only() {
    let names = tool_names_for_turn(TestCaps { fs_read: true, fs_write: true, terminal: true }).await;
    for n in ["file_read", "file_write", "file_append", "file_delete", "dir_list", "dir_walk", "shell"] {
        assert!(names.contains(&n.to_string()), "missing {n}: {names:?}");
    }

    // `fs.read` only: the read pair survives, the write pair and the
    // terminal-dependent names do not. No fallback to the agent's disk.
    let names = tool_names_for_turn(TestCaps { fs_read: true, fs_write: false, terminal: false }).await;
    assert!(names.contains(&"file_read".to_string()));
    for n in ["file_write", "file_append", "file_delete", "dir_list", "dir_walk", "shell"] {
        assert!(!names.contains(&n.to_string()), "unexpected {n}: {names:?}");
    }
}

/// Outside an ACP session the same seven names answer to the deployment
/// switch and nothing else, and the three lifecycle names are never
/// offered — there is no handle store on the agent's own side.
#[tokio::test]
async fn a_non_acp_turn_is_governed_by_host_access_alone() {
    let off = tool_names_for_turn_with_host_access(false).await;
    for n in crate::tools::policy::HOST_TOOLS {
        assert!(!off.contains(&n.to_string()), "{n} should be hidden");
    }
    let on = tool_names_for_turn_with_host_access(true).await;
    for n in crate::tools::policy::HOST_TOOLS {
        assert!(on.contains(&n.to_string()), "{n} should be present");
    }
    for n in ["shell_start", "shell_output", "shell_kill"] {
        assert!(!on.contains(&n.to_string()), "{n} has no agent-side body");
        assert!(!off.contains(&n.to_string()), "{n} has no agent-side body");
    }
}
```

- [ ] **Step 2: 失敗を確認**

```sh
cargo test -p sapphire-agent-server --lib policy::tests
cargo test -p sapphire-agent-server --lib serve::tests
```

- [ ] **Step 3: `policy.rs` を実装**

```rust
/// Whether this call is refused before the policy table is consulted.
///
/// A gate in front of `decide` rather than a row inside it: `decide` is a
/// pure function of origin and kind, and this is a fact about the
/// deployment. Keeping them apart means the permission table still reads
/// as one thing.
///
/// `routed_to_client` is what #270 added, and it is what changed this
/// gate's meaning. The seven names below are "the agent's own filesystem
/// and shell" *only when that is where the call is going*: inside an ACP
/// session the same names reach the editor's machine, which this
/// deployment's `host_access` switch has no business speaking for. A
/// caller that knows the turn has an ACP client passes `true` and the gate
/// declines to fire; a caller that does not (the channel path, whose
/// turns never have one) passes `false` and gets exactly the old
/// behaviour.
pub fn host_tool_denied(name: &str, host_access_enabled: bool, routed_to_client: bool) -> bool {
    !host_access_enabled && !routed_to_client && HOST_TOOLS.contains(&name)
}
```

  `partition_without_asking` の末尾に `routed_to_client: bool` を追加し、内部の
  `host_tool_denied(&call.name, host_access_enabled)` にそのまま渡す。

- [ ] **Step 4: `visible_tool_predicate` を実装**（Spec §5 の本体をそのまま）

  doc コメントも書き換える: 「client-side tools は capability で絞る」ではなく
  「one set of names, whose *machine* the session type decides」であること、
  非 ACP では `host_access` だけがゲートであることを書く。

- [ ] **Step 5: 呼び出し側を追随させる**

  `server/src/serve/mod.rs` の許可ゲート（約2553〜2580行）:

```rust
let routed_to_client = acp_client.is_some();
...
} else if host_tool_denied(&call.name, host_access_enabled, routed_to_client) {
    Some(refusal_message(&call.name, Refusal::Unavailable))
} else {
```

  `server/src/agent.rs` の `partition_without_asking` 呼び出し（約759行）は
  `false` を追加で渡し、doc コメントを「a channel turn never has an ACP client,
  so the host gate speaks for every one of these names here」に直す。
  `specs_filtered(visible_tool_predicate(host_access_enabled, false, false, false, false))`
  は**そのまま**（引数は変わらない）。

- [ ] **Step 6: テスト**

```sh
cargo test -p sapphire-agent-server --lib policy::tests
cargo test -p sapphire-agent-server --lib serve::tests
cargo test -p sapphire-agent-server --lib agent::tests
cargo fmt --all
```
  期待: パス。`agent::tests::the_channel_gate_also_refuses_host_tools_when_host_access_is_off`
  が**期待値を変えずに**通ることが、チャネル挙動が変わっていないことの証拠になる。

- [ ] **Step 7: コミット**

```sh
git add server/src/tools/policy.rs server/src/serve/mod.rs server/src/agent.rs
git commit -m "feat(policy): restrict [tools.host_access] to calls routed to the agent's own machine"
```

---

## Task 4: 旧名を使うテスト・リテラルの追随

**Files:**
- Modify: `server/src/tools/subagent.rs`（約1033, 1159, 1177, 1265, 1271, 1286, 1304, 1308, 1479, 1480, 1744, 1745, 1781 行、`SubagentHost` の doc 約317行）
- Modify: `server/src/agents.rs`（約150〜159行の定義ファイル例）
- Modify: `server/src/serve/acp.rs`（capability ログの doc 約254行）
- Modify: `server/src/tools/acp_client.rs`（trait doc 約74〜79行、`TerminalRegistry` doc 約57行、`try_reserve_terminal_slot` doc 約125行）

- [ ] **Step 1: 旧名を機械的に置換**

```sh
grep -rn "client_file_read\|client_file_write\|client_shell" server/src/
```
  置換規則: `client_file_read`→`file_read`、`client_file_write`→`file_write`、
  `client_shell_start`→`shell_start`、`client_shell_output`→`shell_output`、
  `client_shell_kill`→`shell_kill`、`client_shell`→`shell`
  （**長い名前から先に**置換する。`client_shell` を先に置換すると
  `shell_start` になるべきものが `shell_start` に潰れず、逆順では壊れる）。
  `subagent.rs` の `tools: Some(vec!["file_read".to_string()])` のようなテストリテラルは
  そのまま統合後の名前として妥当（ACP セッションのサブエージェントが読む名前）。

- [ ] **Step 2: ACP セッションからの委派がクライアントへ届くことを固定するテストを追加**

  `server/src/tools/subagent.rs` の tests に1本追加する（`FakeClient` と
  `scope_acp_client` + `scope_turn_context` を重ね、サブエージェントのツール実行が
  クライアント側のメソッドに到達することを確認する。既存の
  `turn_context()` ヘルパ（約1601行）と `FakeClient` を再利用する）:

```rust
/// A subagent delegated from an ACP session reaches the same machine its
/// parent does. `SubagentHost::acp_client()` forwards to the parent's
/// host, and `TurnLoop::run` is what scopes the task-local — so this is
/// the assertion that delegation cannot silently become "the agent's own
/// disk instead of the editor's".
#[tokio::test]
async fn a_delegated_subagents_tool_calls_still_reach_the_editor() {
    let client = std::sync::Arc::new(FakeClient::new());
    let reaches_editor = scope_turn_context(
        turn_context_with_client(std::sync::Arc::clone(&client)),
        scope_acp_client(
            std::sync::Arc::clone(&client) as std::sync::Arc<dyn AcpClient>,
            async {
                // The delegating turn's own `TurnHost::acp_client` is what a
                // subagent's `SubagentHost` forwards, so reading it from
                // inside the same scopes is what the child's tools see.
                crate::tools::acp_client::current_acp_client().is_some()
            },
        ),
    )
    .await;
    assert!(reaches_editor);
}
```
  注: `turn_context_with_client` は既存の `turn_context()` が `NullProgress` を積んでいるなら
  その箇所だけ `AcpProgress` 相当のスタブに差し替えたヘルパを1つ足す。既存ヘルパで足りるなら
  追加しない（`acp_client()` が `None` を返すスタブでは意味のあるテストにならないため、
  足りない場合のみ追加する）。

- [ ] **Step 3: `SubagentHost` の doc を追随**

  「`acp_client()` を転送するのは、委派先が親と同じマシンで動くため」に書き換える。

- [ ] **Step 4: テスト**

```sh
grep -rn "client_file_read\|client_file_write\|client_shell" server/src/ ; # 0件を確認
cargo test -p sapphire-agent-server --lib tools::subagent::tests
cargo test -p sapphire-agent-server --lib agents::tests
cargo test -p sapphire-agent-server --lib serve::acp::tests
cargo fmt --all
```

- [ ] **Step 5: コミット**

```sh
git add server/src/tools/subagent.rs server/src/agents.rs server/src/serve/acp.rs server/src/tools/acp_client.rs
git commit -m "test(tools): follow the unified shell/file tool names through subagents and stubs"
```

---

## Task 5: 説明文・設定例・ドキュメントの追随

**Files:**
- Modify: `server/src/tools/builtin_tools.rs`（7ツールの `description`）
- Modify: `server/src/tools/client_tools.rs`（`shell_start` / `shell_output` / `shell_kill` の `description`）
- Modify: `server/config.example.toml`（約319〜330行、約421行）
- Modify: `server/templates/workspace/config.toml`（約42行）
- Modify: `README.md`（16〜17行、502〜578行の節、122行の例）
- Modify: `README.ja.md`（同じ箇所）

- [ ] **Step 1: 共通の前置きを7ツールの説明文に入れる**

```rust
/// The sentence every shell/file tool's description now opens with.
///
/// One sentence in one place, rather than seven copies that drift: which
/// machine a call reaches is a property of the *session*, not of the tool,
/// and a description that described a fixed machine would be wrong in one
/// of the two cases every time.
pub(crate) const REACHES_SENTENCE: &str =
    "Reaches whichever machine this conversation is about: inside an ACP \
     session, the machine the connected editor is running on; otherwise \
     this agent's own machine. ";
```

  各 `description` は `format!("{REACHES_SENTENCE}…")` の形にする。
  `file_delete` / `dir_list` / `dir_walk` の説明文には、ACP 経路が
  bash の `rm` / `find` を使うことを1文で書く。旧 `client_*` の
  「Only available inside an ACP session whose editor supports …; refuses
  otherwise.」は**削除**する（可視性ルールにより、その場合はそもそも提示されない）。

- [ ] **Step 2: `config.example.toml` の `host_access` 説明を書き換える**

```toml
#   - `[tools.host_access]` — the switch that lets the agent touch its
#     own filesystem and shell: `file_read`, `file_write`, `file_append`,
#     `file_delete`, `dir_list`, `dir_walk`, `shell`. `enabled = false` by
#     default, for every origin — `/rpc`, voice, `/a2a`, Matrix/Discord and
#     ACP alike.
#
#     Inside an ACP session these same seven names reach the *editor's*
#     machine instead, and this switch does not speak for it: what that
#     session may call is decided by the capabilities the connected
#     editor declared at `initialize` (`fs.read_text_file`,
#     `fs.write_text_file`, `terminal/*`), and a missing capability
#     hides the tool rather than falling back to the agent's own disk.
#     Turning this on is therefore only about the agent's own machine;
#     running the agent in a container is the recommended way to do it
#     once it is on.
```

  約423行の例ブロック（`# [tools.host_access]`）はそのまま。
  `server/templates/workspace/config.toml` の `origin = "trusted"` 行（約42行）の
  コメント `# "trusted" (everything, needs [tools] host_access) | "channel" (no shell)`
  は現状のまま正しい（チャネル経路の挙動は変わらない）。

- [ ] **Step 3: `README.md` を書き換える**

  - 16〜17行の2項目を1項目に統合する:

```markdown
- **Shell and file tools**: `file_read`, `file_write`, `file_append`, `file_delete`,
  `dir_list`, `dir_walk`, `shell`, plus `shell_start` / `shell_output` / `shell_kill`
  for long-running commands. **Which machine they reach is decided by the session, not
  by the name**: inside an ACP session they act on the machine the connected editor is
  running on, over `/acp`'s `fs/*` and `terminal/*` requests — and only for the
  capabilities that editor declared at `initialize`, with no fallback to the agent's own
  disk when one is missing. Everywhere else (Matrix, Discord, `/rpc`, voice, `/a2a`,
  heartbeat, autonomous, subagents) they act on the agent's own machine, and are opt-in:
  `[tools.host_access] enabled = false` by default, for every origin. Turning that on is
  a deliberate act; running the agent in a container is the recommended way to do it
  once it is.
```

  - 「Client-side tools: whose machine」節（502〜578行）を「Shell and file tools: whose
    machine」に改題し、表を統合後の10ツールに置き換える。ACP 側の呼び出し列は
    本計画 Task 1/2 の写像（`file_append` は read+write、`file_delete`/`dir_list`/
    `dir_walk` は terminal 経由）を書く。
  - 528〜578行の「タイムアウトは殺さない」「端末はセッション毎」「上限8」の記述は
    そのまま残す（挙動は不変）。ツール名だけ差し替える。
  - 569〜578行の「The agent's own filesystem and shell are opt-in」段落は、
    「opt-in なのはエージェント自身のマシンに届くときだけ」に書き換える。
  - 122行の `tools: [client_file_read, workspace_search, memory_read]` を
    `tools: [file_read, workspace_search, memory_read]` に。
- [ ] **Step 4: `README.ja.md` を同じ内容で追随させる**（16〜17行、クライアント側ツール節、
  サブエージェント定義例）。

- [ ] **Step 5: 残存チェック**

```sh
grep -rn "client_file_read\|client_file_write\|client_shell" README.md README.ja.md \
  server/config.example.toml server/templates server/src/
```
  期待: 0件。

- [ ] **Step 6: コミット**

```sh
git add README.md README.ja.md server/config.example.toml server/templates \
        server/src/tools/builtin_tools.rs server/src/tools/client_tools.rs
git commit -m "docs(tools): describe the unified shell/file tools and the narrowed host_access"
```

---

## Task 6: 全体検証

**Files:** 変更なし（検証のみ。落ちた箇所を直す場合はそのファイル）

- [ ] **Step 1: fmt / clippy**

```sh
cargo fmt --all -- --check
cargo clippy --workspace --all-targets -- -D warnings
```
  期待: どちらも無出力で終了コード0。

- [ ] **Step 2: テスト**

```sh
cargo test --workspace
```
  期待: 全パス。落ちたテストが「統合で名前が変わったこと」だけを理由にしているなら、
  テスト側を統合後の名前・統合後の意味に合わせて直す（**実装を旧名に戻さない**）。

- [ ] **Step 3: 受け入れ基準の手動確認**

```sh
# 1. 登録されている名前
grep -n '"file_read"\|"file_write"\|"file_append"\|"file_delete"\|"dir_list"\|"dir_walk"\|"shell"\|"shell_start"\|"shell_output"\|"shell_kill"' server/src/tools/mod.rs

# 2. 旧名がソース・ドキュメントに残っていないこと
grep -rn "client_file_read\|client_file_write\|client_shell\|client_" server/src/ README.md README.ja.md server/config.example.toml
```
  期待: 1 は10名すべて、2 は0件（`AcpClient` / `client` 変数名など英語の一般語は除く）。
  2 で残るのは `acp_client`（ACP 接続そのものを指す正当な名前）と
  `client_capabilities` / `client_fs_caps` / `client_terminal_cap`（capability を指す
  既存名）だけであること。

- [ ] **Step 4: 仕様書の受け入れ基準1〜7を通しで確認し、満たさない項目を修正**

  特に:
  - ACP セッションで `host_access = true` でも `file_read` がエディタ側を返す
    （Task 1 Step 1 のテストが固定済み）。
  - `dir_list` の出力形式が両経路で同一（Task 2 Step 1 のテストが固定済み）。
  - 非 ACP ＋ `host_access = false` で7名が一覧に現れない
    （Task 3 Step 1 のテストが固定済み）。

- [ ] **Step 5: 最終コミット**（Step 1〜4 で修正が生じた場合のみ）

```sh
git add -A
git commit -m "fix(tools): finish the unified shell/file tool routing"
```

---

## 参考: 各タスク完了時点で動作していること

| タスク完了時 | 動作 |
|---|---|
| T1 | `file_read`/`file_write`/`file_append`/`file_delete`/`shell` がセッション種別でルーティングされる（可視性はまだ旧ルール） |
| T2 | `dir_list`/`dir_walk` も同様（#262 の機能が terminal 経由で入る） |
| T3 | 可視性と `host_access` の意味が仕様どおりになる。`shell_start`/`output`/`kill` が非 ACP で消える |
| T4 | サブエージェントとテストが統合後の名前で通る |
| T5 | 説明文・README・設定例が追随し、旧名の記述が消える |
| T6 | `fmt` / `clippy` / `test --workspace` が通り、受け入れ基準1〜7を満たす |
