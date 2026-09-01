# ACP セッションのツール呼び出しを永続化する Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** ACP セッションを開き直したときにエージェントが「何をしたか」を取り戻せるようにし、同時に全ツール出力に上限を掛けて1件あたりの常駐量を縛る。

**Architecture:** 変更は3箇所に閉じる。ツール出力の上限は全ツールの唯一の合流点（`ToolSet::execute`）で掛ける。永続化は `run_llm_turn` の2つのスキップ地点で ACP セッションのときだけ行い、ストア側の変換は既にある。そしてキャッシュが無いときに `tool_use`/`tool_result` の対が壊れる既存の穴を塞ぐ。**メモリ上の履歴は一切加工しない** — 参照化は spec で取り下げられた。

**Tech Stack:** Rust 2024, `serde` / `serde_json`, `tokio`, `tempfile`

**Spec:** `docs/superpowers/specs/2026-09-01-tool-call-persistence-design.md`

## Global Constraints

- ブランチは `feat/tool-call-persistence`（`main` から作成済み、spec 改訂を cherry-pick 済み）。
- テストコマンドは `cargo test -p sapphire-agent`。反復中は `--workspace` を使わない。最後に一度だけ `cargo test --workspace`。
- **`cargo check` と `cargo test` を混ぜない。** フィンガープリントが別なので交互に走らせると毎回リンクし直しになる。
- cargo は**前景で、`timeout: 600000` で**。`run_in_background` も `Monitor` も使わない。**cargo を2本同時に走らせない**（このホストの OS は熱でスロットリングする小さな USB SSD 上にある）。10分のツールタイムアウトに当たったら、ビルドは温まっているので同じコマンドを走らせ直せばよい。
- **`Cargo.lock` をコミットしない。** `Cargo.toml` が `cron = "0.16"` / `tower-http = "0.6"` を宣言しているのに committed lockfile が 0.17 / 0.7 を持つため、cargo を走らせるたびに書き換わる。各コミット前に `git checkout -- Cargo.lock`。
- テストはソースファイル内の `mod tests` に置く。
- **`src/agent.rs` は編集しない。**（ただし Task 1 の変更は共有の合流点を通って channel の挙動にも及ぶ。意図した結果であり、Task 1 に明記してある。）
- **既存4ストア（`channel` / `cross-device` / `device-default` / `mcp`）の永続化は変えない。** ツール呼び出しを保存するのは ACP セッションだけ。既存ストアの対応は [#194](https://github.com/fluo10/sapphire-agent/issues/194)。
- **メモリ上の履歴を加工する処理を足さない。** `state.sessions` に入るものとモデルが見るものは常に同じ。加工が入るのはストアへの書き込み経路だけ。
- **日付は `Local::now().date_naive()` で作らない。** 境界時刻より前は前日に属するので、`crate::session::local_date_for_timestamp(Local::now(), boundary_hour)` を使う。この規則を破ったコードが CI を落としたばかり。

## spec の訂正が1つ

spec の決定1は「組み込みツールも MCP ツールも、**拒否されたときの理由文字列も**、すべて `ToolOutput` として `ToolSet::execute` を通る」と書いているが、**拒否理由は通らない。**

`src/serve/mod.rs:2326` と `src/agent.rs:996` が
`results.push((id, crate::tools::ToolOutput::from(reason)))` で直接組み立てている。

実害は無い — 拒否理由は `crate::tools::policy::refusal_message` が返す定型文で、
構造上短い。**しかし「全部がそこを通る」という前提で後から何かを足すと外れる**ので、
Task 1 でその旨をコード上のコメントに残す。

---

### Task 1: ツール出力の上限を、全ツールの合流点で掛ける

`truncate_output` を `ToolSet::execute` に移し、シェル専用だった上限を全ツールに広げる。

**Files:**
- Modify: `src/tools/mod.rs`
- Modify: `src/tools/builtin_tools.rs`
- Test: `src/tools/mod.rs` の `mod tests`

**Interfaces:**
- Produces: `crate::tools::truncate_output(&str) -> String`（`pub(crate)`）。`ToolSet::execute` が返す `ToolOutput.text` は 50 000 字以下であることが保証される。

#### この Task が変える挙動の範囲

`ToolSet::execute` は `src/serve/mod.rs:2305`（ACP と `/rpc`）と
`src/agent.rs:971`（Matrix / Discord）の両方から呼ばれる。**したがって上限は
全トランスポートに掛かる。** `src/agent.rs` のファイルは編集しないが、挙動は変わる。

これは意図した結果である。ツール結果が無制限にメモリへ載る問題は channel セッションでも
同じで、合流点が共有されている以上、片方だけ縛る理由が無い。

**シェルの既存の切り詰めは削る。** 今は stdout と stderr を*別々に*切ってから
1つの文字列に組み立てているので、残すと合流点でもう一度切られて省略マーカーが入れ子になる。

削った結果の変化を明記しておく: 両方のストリームが巨大な場合、50 000 字の窓は
**stdout の先頭と stderr の末尾**を残し、その間（stdout の末尾と stderr の先頭）が
落ちる。失敗したコマンドは stderr が末尾にあるので残り、成功したコマンドは
stdout の先頭が残る。どちらも欲しい側が残る。

- [ ] **Step 1: 失敗するテストを書く**

`src/tools/mod.rs` の末尾に `mod tests` があればそこへ、無ければ新設する。

```rust
#[cfg(test)]
mod truncation_tests {
    use super::*;

    #[test]
    fn short_output_is_returned_unchanged() {
        let s = "a short result";
        assert_eq!(truncate_output(s), s);
    }

    /// The cap exists so one `file_read` of a large file cannot put the
    /// whole file into the in-memory history, the model's input, and the
    /// cache all at once.
    #[test]
    fn long_output_is_cut_to_the_cap_with_a_marker() {
        let s = "x".repeat(120_000);
        let out = truncate_output(&s);
        assert!(
            out.len() < s.len(),
            "a 120k result must not come back whole"
        );
        assert!(out.contains("chars truncated"), "got {}", &out[..80]);
    }

    /// Head and tail are both kept: the head is where a file's shape is,
    /// the tail is where a failing command's error is.
    #[test]
    fn both_ends_survive_truncation() {
        let s = format!("{}{}{}", "H".repeat(30_000), "M".repeat(60_000), "T".repeat(30_000));
        let out = truncate_output(&s);
        assert!(out.starts_with('H'), "the head is kept");
        assert!(out.ends_with('T'), "the tail is kept");
        assert!(
            !out.contains(&"M".repeat(1000)),
            "the middle is what gets dropped"
        );
    }

    /// Cutting at a byte index inside a multi-byte character would
    /// panic. `floor_char_boundary` is what prevents it, so a result
    /// that is entirely multi-byte is the case worth pinning.
    #[test]
    fn a_multibyte_result_is_cut_on_a_character_boundary() {
        let s = "日".repeat(60_000); // 3 bytes each — well past the cap
        let out = truncate_output(&s);
        assert!(out.contains("chars truncated"));
        assert!(out.starts_with('日') && out.ends_with('日'));
    }

    /// The cap has to be a cap: a result that has already been cut must
    /// come back unchanged rather than picking up a second marker.
    ///
    /// The old constants did not satisfy this — head 20 000 + tail
    /// 30 000 + the marker itself exceeds 50 000, so a second pass cut
    /// again and nested the markers. That only stayed invisible because
    /// truncation happened in exactly one place.
    #[test]
    fn truncating_an_already_truncated_result_changes_nothing() {
        let once = truncate_output(&"x".repeat(200_000));
        assert!(once.len() <= 50_000, "the cap is a cap: {} bytes", once.len());
        assert_eq!(truncate_output(&once), once);
        assert_eq!(once.matches("chars truncated").count(), 1);
    }
}
```

- [ ] **Step 2: テストが落ちることを確認**

Run: `cargo test -p sapphire-agent truncation_tests`
Expected: FAIL — `truncate_output` は `builtin_tools` の private 関数なので、`tools::mod` からは見えない。

- [ ] **Step 3: 関数を移す**

`src/tools/builtin_tools.rs` から `truncate_output` の定義（`fn truncate_output` から
その閉じ括弧まで、doc コメント含む）を削除し、`src/tools/mod.rs` に移す。
可視性を `pub(crate)` にする。

```rust
/// Cap a tool result at 50 000 chars, keeping head + tail.
///
/// Applied at `ToolSet::execute` rather than inside each tool, because
/// that is the one place every builtin and every MCP tool's output
/// passes through — per-tool truncation would be forgotten by the next
/// tool someone adds.
///
/// The cap is what bounds a tool result's cost in all three places it
/// lands: the in-memory history, the model's input, and (for ACP
/// sessions) the tool-result cache on disk.
///
/// Head and tail both survive because they carry different things: a
/// file's shape is at the top, and a failing command's error is at the
/// bottom.
///
/// The head, tail and marker budgets add up to `MAX` rather than
/// overshooting it, so truncating an already-truncated result is a
/// no-op. The previous constants (20 000 + 30 000, plus the marker on
/// top) exceeded the cap, which meant a second pass cut again and
/// nested the markers — harmless only for as long as truncation
/// happened in exactly one place, which is what this change ends.
pub(crate) fn truncate_output(s: &str) -> String {
    const MAX: usize = 50_000;
    /// Room for `\n\n[... 1234567 chars truncated ...]\n\n`, generously.
    const MARKER_BUDGET: usize = 200;
    const HEAD: usize = 19_920;
    const TAIL: usize = MAX - MARKER_BUDGET - HEAD;

    if s.len() <= MAX {
        return s.to_string();
    }
    let head_end = s.floor_char_boundary(HEAD);
    let tail_start = s.floor_char_boundary(s.len() - TAIL);
    format!(
        "{}\n\n[... {} chars truncated ...]\n\n{}",
        &s[..head_end],
        tail_start - head_end,
        &s[tail_start..]
    )
}
```

`tail_start - head_end` は、`s.len() - HEAD - TAIL` より正確である。
`floor_char_boundary` が境界を手前にずらした分を含むので、実際に落ちた量を報告する。

`floor_char_boundary` は unstable な場合がある。`src/tools/builtin_tools.rs` の
先頭に `#![feature(...)]` や `round_char_boundary` の宣言があるか確認し、
**元の実装が使っていたものをそのまま持ってくること。** 元が
`s.floor_char_boundary(..)` を使えていたのなら、同じクレート内なので移しても使える。

- [ ] **Step 4: 合流点で適用する**

`src/tools/mod.rs` の `ToolSet::execute`:

```rust
    /// Execute a tool call. The returned `ToolOutput` carries the
    /// text result plus any image attachments the tool produced; the
    /// caller is responsible for assembling them into a tool_result
    /// user message.
    ///
    /// The text is capped here — see `truncate_output`. Note this is
    /// not *quite* every tool_result the model sees: a call refused by
    /// the permission gate never reaches this function, and its
    /// `ToolOutput` is built directly by the caller
    /// (`src/serve/mod.rs`, `src/agent.rs`). Those strings come from
    /// `policy::refusal_message` and are short by construction, but
    /// anything added on that path is NOT capped by this.
    pub async fn execute(&self, call: &ToolCall) -> ToolOutput {
        let inner = self.inner.read().await;
        for tool in &inner.tools {
            if tool.spec().name == call.name {
                let mut output = match tool.execute_full(&call.input).await {
                    Ok(output) => output,
                    Err(e) => ToolOutput::from(format!("Error: {e:#}")),
                };
                output.text = truncate_output(&output.text);
                return output;
            }
        }
        ToolOutput::from(format!("Unknown tool: {}", call.name))
    }
```

`ToolOutput::images` は切らない — 画像は別経路（`image_cache`）で扱う。

- [ ] **Step 5: シェルの二重適用を削る**

`src/tools/builtin_tools.rs:1047-1048`:

```rust
                let stdout = String::from_utf8_lossy(&output.stdout).to_string();
                let stderr = String::from_utf8_lossy(&output.stderr).to_string();
```

`truncate_output` の呼び出しを外すだけで、他は変えない。`use` が未使用になったら消す。

- [ ] **Step 6: テストが通ることを確認**

Run: `cargo test -p sapphire-agent`
Expected: PASS。シェルの切り詰めを検証していた既存テストがあれば、**削らずに**
`ToolSet::execute` 経由を見るように直すか、上限がまだ効いていることを別の形で
確かめること。落ちたテストが何を守っていたのかを報告に書く。

- [ ] **Step 7: コミット**

```bash
git checkout -- Cargo.lock
git add src/tools/mod.rs src/tools/builtin_tools.rs
git commit -m "feat(tools): cap every tool result, not just the shell's"
```

---

### Task 2: キャッシュが無いときに `tool_use`/`tool_result` の対を壊さない

**Files:**
- Modify: `src/acp_session.rs`
- Test: `src/acp_session.rs` の `mod tests`

**Interfaces:**
- Produces: `StoredPart::ToolResultRef { tool_use_id: String, sha256: Option<String> }`

#### 何が壊れているか

`store_part`（`src/acp_session.rs:287`）は、キャッシュが `None` のとき
`StoredPart::Other` を書き、**`tool_use_id` を落とす。** `load_part` はそれを
ただのテキストマーカーとして読み戻すので、`tool_use` に対応する `tool_result` が
無い履歴ができる。**Anthropic API はそれを拒否する** — 縮退ではなく破綻。

現在のコメントは「対はそのまま保たれる」と書いているが、**それは誤り**である。
読み出し側のミス（ハッシュはあるが内容が消えた）は対を保つが、書き込み側の
不在は保たない。コメントも直すこと。

今は本番の書き手がいないので、この経路はまだ踏まれていない。Task 3 がそれを変える。

**移行の心配は無い。** ツール結果を書いたセッションファイルはまだ存在しない。
加えて `Option<String>` は serde が素の文字列を `Some` として読むので、
仮に古い行があっても読める。

- [ ] **Step 1: 失敗するテストを書く**

`src/acp_session.rs` の `mod tests` に追記する。既存の `store_without_cache()`
ヘルパと `tool_result_message(tool_use_id, content)` ヘルパを使う。

```rust
    /// The regression this task exists for. A tool result stored with no
    /// cache must still read back as a `ToolResult` carrying its
    /// `tool_use_id` — an unpaired `tool_use` is rejected by the API,
    /// which is a broken session rather than a degraded one.
    #[test]
    fn a_result_stored_without_a_cache_keeps_its_pairing() {
        let (_d, store) = store_without_cache();
        store.create("s1", "default", "/p").unwrap();
        store
            .append_message("s1", &tool_result_message("c1", "content that cannot be cached"))
            .unwrap();

        let history = store.history("s1").expect("the session loads");
        assert_eq!(
            history[0].parts[0],
            ContentPart::ToolResult {
                tool_use_id: "c1".to_string(),
                content: MISSING_RESULT.to_string(),
            },
            "the id survives even though the content never had anywhere to go"
        );
    }

    /// The two ways a result can be absent — never cached, or cached and
    /// later evicted — must read back identically. A reader has no
    /// reason to tell them apart: both mean "the pairing is here, the
    /// content is not".
    #[test]
    fn never_cached_and_evicted_read_back_the_same() {
        let (dir_a, cached) = store();
        cached.create("s1", "default", "/p").unwrap();
        cached
            .append_message("s1", &tool_result_message("c1", "gone later"))
            .unwrap();
        std::fs::remove_dir_all(dir_a.path().join("cache")).unwrap();
        let evicted = cached.history("s1").unwrap();

        let (_dir_b, uncached) = store_without_cache();
        uncached.create("s1", "default", "/p").unwrap();
        uncached
            .append_message("s1", &tool_result_message("c1", "never stored"))
            .unwrap();
        let never = uncached.history("s1").unwrap();

        assert_eq!(evicted[0].parts, never[0].parts);
    }

    /// A cache that is present still stores the hash, not the content.
    #[test]
    fn a_cached_result_still_records_its_hash() {
        let (_d, store) = store();
        store.create("s1", "default", "/p").unwrap();
        store
            .append_message("s1", &tool_result_message("c1", "the real output"))
            .unwrap();

        let raw = std::fs::read_to_string(store.path_for_test("s1")).unwrap();
        assert!(!raw.contains("the real output"), "got {raw}");
        assert!(raw.contains("tool_result_ref"), "got {raw}");
        assert_eq!(
            store.history("s1").unwrap()[0].parts[0],
            ContentPart::ToolResult {
                tool_use_id: "c1".to_string(),
                content: "the real output".to_string(),
            }
        );
    }
```

`store_without_cache()` と `tool_result_message()` が存在しない、または名前が違う
場合は現物に合わせること。無ければ既存の `store()` に倣って足す。

- [ ] **Step 2: テストが落ちることを確認**

Run: `cargo test -p sapphire-agent acp_session`
Expected: FAIL — `a_result_stored_without_a_cache_keeps_its_pairing` が
`ContentPart::Text(...)` を返す。

- [ ] **Step 3: `sha256` を `Option` にする**

`StoredPart` の定義:

```rust
    ToolResultRef {
        tool_use_id: String,
        /// `None` means the result had nowhere to be stored — the cache
        /// was unavailable when this line was written. Distinct from a
        /// hash whose entry has since been evicted, but only in how it
        /// arose: a reader treats both as "the pairing is here, the
        /// content is not", and `load_part` produces the same
        /// `MISSING_RESULT` for each.
        ///
        /// `Option<String>` also reads a bare string as `Some`, so a
        /// line written before this field became optional still loads.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        sha256: Option<String>,
    },
```

- [ ] **Step 4: 書き込み側を直す**

```rust
            ContentPart::ToolResult {
                tool_use_id,
                content,
            } => {
                // A cache miss on read degrades gracefully; a missing
                // cache on write must not be allowed to degrade any
                // further. Writing `Other` here would drop the
                // `tool_use_id`, leaving a `tool_use` with no matching
                // `tool_result` — which the API rejects outright, so the
                // session would fail to load rather than load thinner.
                let sha256 = match &self.cache {
                    Some(cache) => Some(cache.put(content)?),
                    None => {
                        warn!(
                            "Tool-result cache unavailable; recording '{tool_use_id}' \
                             with no content"
                        );
                        None
                    }
                };
                StoredPart::ToolResultRef {
                    tool_use_id: tool_use_id.clone(),
                    sha256,
                }
            }
```

- [ ] **Step 5: 読み出し側を直す**

```rust
            StoredPart::ToolResultRef {
                tool_use_id,
                sha256,
            } => ContentPart::ToolResult {
                tool_use_id: tool_use_id.clone(),
                // Absent either way — no hash was ever written, or the
                // hash's entry is gone. The model can call the tool
                // again if it needs to; what it cannot recover from is
                // an unpaired `tool_use`.
                content: sha256
                    .as_ref()
                    .and_then(|sha| self.cache.as_ref()?.get(sha))
                    .unwrap_or_else(|| MISSING_RESULT.to_string()),
            },
```

- [ ] **Step 6: テストが通ることを確認**

Run: `cargo test -p sapphire-agent acp_session`
Expected: PASS。

- [ ] **Step 7: コミット**

```bash
git checkout -- Cargo.lock
git add src/acp_session.rs
git commit -m "fix(sessions): keep the tool_use/tool_result pairing when the cache is gone"
```

---

### Task 3: ACP セッションのツール呼び出しを永続化する

**Files:**
- Modify: `src/serve/mod.rs`
- Test: `src/serve/mod.rs` の `mod tests`

**Interfaces:**
- Consumes: Task 2 の `StoredPart::ToolResultRef`、既存の `AcpSessionStore::append_message`

- [ ] **Step 1: 失敗するテストを書く**

`src/serve/mod.rs` の `mod tests` に追記する。既存の `ServeState::for_test` と
`NullProgress` を使う（名前が違えば現物に合わせる）。

既存の道具立てをそのまま使う。`ServeState::for_test_scripted(acp_enabled, responses)`
がプロバイダの応答列を差し込み、`RiskyTool` が実際に走るテスト用ツール、
`NullProgress` が `TurnHost` の既定実装（`origin()` は `Origin::Trusted` なので
許可ゲートは通る）。同じ組み合わせが `src/serve/mod.rs:2988` 付近の既存テストで
使われているので、そこを読んでから書くこと。

```rust
    /// The point of the task: after a turn that called a tool, reopening
    /// the session shows what the agent did, not just what was said.
    #[tokio::test]
    async fn an_acp_turn_persists_its_tool_calls() {
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
                    text: Some("done".to_string()),
                    tool_calls: Vec::new(),
                    stop_reason: None,
                },
            ],
        );
        state.tools.register_tool(Box::new(RiskyTool::new())).await;

        let sid = "acp-tools".to_string();
        state.acp_sessions.lock().await.insert(sid.clone());
        state
            .acp_session_store
            .create(&sid, "default", "/work")
            .unwrap();

        run_llm_turn(
            Arc::clone(&state),
            sid.clone(),
            ChatMessage::user("run it"),
            Arc::new(NullProgress),
            None,
        )
        .await;

        let history = state.acp_session_store.history(&sid).expect("the session");
        let uses: Vec<&str> = history
            .iter()
            .flat_map(|m| m.parts.iter())
            .filter_map(|p| match p {
                ContentPart::ToolUse { id, .. } => Some(id.as_str()),
                _ => None,
            })
            .collect();
        let results: Vec<&str> = history
            .iter()
            .flat_map(|m| m.parts.iter())
            .filter_map(|p| match p {
                ContentPart::ToolResult { tool_use_id, .. } => Some(tool_use_id.as_str()),
                _ => None,
            })
            .collect();

        assert_eq!(uses, vec!["call-1"], "the tool_use was persisted");
        assert_eq!(
            uses, results,
            "every tool_use has its matching tool_result, in order — an \
             unpaired one is rejected by the API on reload"
        );
    }

    /// `/rpc` and the other transports keep today's behaviour: their
    /// stores have no reference form for a tool result, so writing one
    /// raw would put the content in the workspace and the retrieve
    /// index. Tracked as #194.
    #[tokio::test]
    async fn an_rpc_turn_still_does_not_persist_tool_calls() {
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
                    text: Some("done".to_string()),
                    tool_calls: Vec::new(),
                    stop_reason: None,
                },
            ],
        );
        state.tools.register_tool(Box::new(RiskyTool::new())).await;

        // Not registered in `acp_sessions`, so this is an /rpc session.
        let sid = "rpc-tools".to_string();

        run_llm_turn(
            Arc::clone(&state),
            sid.clone(),
            ChatMessage::user("run it"),
            Arc::new(NullProgress),
            None,
        )
        .await;

        let stored = state
            .cross_device_session_store
            .load_session(&sid)
            .unwrap_or_default();
        assert!(
            !stored.iter().flat_map(|m| m.parts.iter()).any(|p| matches!(
                p,
                ContentPart::ToolUse { .. } | ContentPart::ToolResult { .. }
            )),
            "the /rpc store must still hold no tool traffic"
        );
    }
```

- [ ] **Step 2: テストが落ちることを確認**

Run: `cargo test -p sapphire-agent serve`
Expected: FAIL — `an_acp_turn_persists_its_tool_calls` が空の `uses` を見る。

- [ ] **Step 3: `tool_use` を永続化する**

`src/serve/mod.rs:2234-2239` を置き換える。

```rust
                let msg = ChatMessage::assistant_with_tools(resp.text.clone(), tool_calls.clone());
                history.push(msg.clone());
                // ACP sessions persist tool traffic; the other stores do
                // not (see #194). Their line format has no reference
                // form for a result, so writing one raw would put the
                // content into the workspace and the retrieve index —
                // which is exactly what the ACP store's external cache
                // exists to avoid.
                if is_acp
                    && let Err(e) = state.acp_session_store.append_message(&session_id, &msg)
                {
                    warn!("Failed to persist a tool_use message: {e}");
                }
```

- [ ] **Step 4: `tool_result` を永続化する**

`src/serve/mod.rs:2340-2343` を置き換える。

```rust
                let result_msg = ChatMessage::tool_results_with_images(text_results, images);
                history.push(result_msg.clone());
                // Must follow the `tool_use` append above and must not
                // be skipped independently of it: a `tool_use` with no
                // matching `tool_result` is rejected by the API, so a
                // half-persisted pair is worse than neither.
                if is_acp
                    && let Err(e) = state
                        .acp_session_store
                        .append_message(&session_id, &result_msg)
                {
                    warn!("Failed to persist a tool_result message: {e}");
                }
```

**`is_acp` は既にこの関数の先頭で計算されている** — 再計算しないこと。

- [ ] **Step 5: テストが通ることを確認**

Run: `cargo test -p sapphire-agent`
Expected: PASS。

- [ ] **Step 6: コミット**

```bash
git checkout -- Cargo.lock
git add src/serve/mod.rs
git commit -m "feat(acp): persist tool_use and tool_result for ACP sessions"
```

---

### Task 4: メモリ常駐量を測れるようにする

アイドルセッションの破棄は別に設計する。その判断の材料をログに出す。

**Files:**
- Modify: `src/serve/mod.rs`
- Test: `src/serve/mod.rs` の `mod tests`

**Interfaces:**
- Produces: `ServeState::session_residency(&self) -> SessionResidency`、`SessionResidency { sessions: usize, messages: usize, text_bytes: usize, tool_result_bytes: usize, largest: Option<(String, usize)> }`

#### なぜ「最大の1本」まで出すのか

合計だけでは「多数のセッションが溜まる」のか「1本が長く伸びる」のかが区別できない。
前者なら破棄が効き、後者なら破棄は効かず別の手が要る。**その2つを分ける値が、
次の設計判断の唯一の入力になる。**

- [ ] **Step 1: 失敗するテストを書く**

```rust
    /// The measurement exists to distinguish "many sessions accumulate"
    /// from "one session grows", because only the first is fixed by
    /// dropping idle sessions. A total alone cannot tell them apart, so
    /// the largest single session is reported too.
    #[tokio::test]
    async fn residency_separates_the_total_from_the_largest_session() {
        let state = ServeState::for_test(true);
        {
            let mut sessions = state.sessions.lock().await;
            sessions.insert(
                "small".to_string(),
                vec![ChatMessage::user("hi")],
            );
            sessions.insert(
                "big".to_string(),
                vec![ChatMessage {
                    role: Role::User,
                    parts: vec![ContentPart::ToolResult {
                        tool_use_id: "c1".to_string(),
                        content: "x".repeat(5_000),
                    }],
                    input_kind: None,
                    user_id: None,
                }],
            );
        }

        let r = state.session_residency().await;
        assert_eq!(r.sessions, 2);
        assert_eq!(r.messages, 2);
        assert_eq!(r.text_bytes, 2, "only the 'hi'");
        assert_eq!(r.tool_result_bytes, 5_000);
        assert_eq!(
            r.largest.as_ref().map(|(id, _)| id.as_str()),
            Some("big"),
            "the largest session is named so one long thread is \
             distinguishable from many short ones"
        );
    }

    #[tokio::test]
    async fn residency_of_an_empty_map_is_all_zero() {
        let state = ServeState::for_test(true);
        state.sessions.lock().await.clear();
        let r = state.session_residency().await;
        assert_eq!(r.sessions, 0);
        assert_eq!(r.messages, 0);
        assert_eq!(r.largest, None);
    }
```

- [ ] **Step 2: テストが落ちることを確認**

Run: `cargo test -p sapphire-agent residency`
Expected: FAIL — `session_residency` が未定義。

- [ ] **Step 3: 実装する**

`src/serve/mod.rs` に。

```rust
/// What `state.sessions` is holding, for the idle-eviction decision that
/// has not been made yet.
///
/// `largest` is the point: a total cannot distinguish many accumulated
/// sessions from one very long one, and only the first of those is fixed
/// by dropping idle sessions.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct SessionResidency {
    pub sessions: usize,
    pub messages: usize,
    pub text_bytes: usize,
    pub tool_result_bytes: usize,
    /// `(session_id, bytes)` for the heaviest single session.
    pub largest: Option<(String, usize)>,
}

impl ServeState {
    pub(crate) async fn session_residency(&self) -> SessionResidency {
        let sessions = self.sessions.lock().await;
        let mut out = SessionResidency {
            sessions: sessions.len(),
            messages: 0,
            text_bytes: 0,
            tool_result_bytes: 0,
            largest: None,
        };
        for (id, history) in sessions.iter() {
            out.messages += history.len();
            let mut this_session = 0usize;
            for msg in history {
                for part in &msg.parts {
                    match part {
                        ContentPart::Text(t) => {
                            out.text_bytes += t.len();
                            this_session += t.len();
                        }
                        ContentPart::ToolResult { content, .. } => {
                            out.tool_result_bytes += content.len();
                            this_session += content.len();
                        }
                        // Tool inputs and images are not what this
                        // measurement is deciding about, and counting
                        // them would blur the two numbers that matter.
                        _ => {}
                    }
                }
            }
            if out.largest.as_ref().is_none_or(|(_, b)| this_session > *b) {
                out.largest = Some((id.clone(), this_session));
            }
        }
        out
    }
}
```

`is_none_or` が使えない場合は `map_or(true, ...)` に置き換える。

- [ ] **Step 4: 掃き出しタスクから呼ぶ**

`spawn_acp_digest_sweep` のループ内、`tokio::time::sleep(period).await;` の直後、
`digest_cache` の `let Some(...) else { continue }` より**前**に置く。
ダイジェストキャッシュが無くても測定は続けたいので。

```rust
            tokio::time::sleep(period).await;

            // Rides on this timer rather than adding one. Debug level:
            // it is input for a design decision, not an operational
            // signal anyone needs at info.
            let r = state.session_residency().await;
            debug!(
                "session residency: {} session(s), {} message(s), {} B text, \
                 {} B tool results; largest {}",
                r.sessions,
                r.messages,
                r.text_bytes,
                r.tool_result_bytes,
                match &r.largest {
                    Some((id, bytes)) => format!("{id} at {bytes} B"),
                    None => "none".to_string(),
                }
            );
```

`debug!` が `use` されていなければ足す。

- [ ] **Step 5: テストが通ることを確認**

Run: `cargo test -p sapphire-agent`
Expected: PASS。

- [ ] **Step 6: コミット**

```bash
git checkout -- Cargo.lock
git add src/serve/mod.rs
git commit -m "feat(serve): log what state.sessions is holding, for the eviction decision"
```

---

### Task 5: ドキュメントと全体確認

**Files:**
- Modify: `README.md`
- Modify: `docs/superpowers/specs/2026-09-01-tool-call-persistence-design.md`

**Interfaces:** なし。

- [ ] **Step 1: README を直す**

前回の作業で README には「ツール呼び出しは永続化されない、再読み込みでは
話した内容は戻るが何をしたかは戻らない、#191 を参照」と書いた。**それが今は
偽になる。** 置き換えて、次を書く。

- ACP セッションは `tool_use` と `tool_result` を保存する。JSONL が持つのは
  ツール名・入力・結果の**ハッシュ**で、結果の本体は
  `<cache_dir>/sapphire-agent/tool-results/` にある。
- キャッシュを消してもセッションは読める。結果がプレースホルダになるだけで、
  `tool_use` と `tool_result` の対は保たれる（対が崩れた履歴は API が拒否するので、
  これは縮退ではなく必須条件）。
- **ツール出力は全ツール 50 000 字で切り詰められる**（先頭20 000 + 末尾30 000）。
  以前はシェルの stdout/stderr だけだった。**この上限は全トランスポートに掛かる** —
  Matrix と Discord も含む。
- `/rpc` / device-default / MCP セッションはツール呼び出しを保存しない（[#194](https://github.com/fluo10/sapphire-agent/issues/194)）。
- 開き直したスレッドではツール呼び出しがエディタに表示されない。モデルは見えている
  が UI には出ない（[#192](https://github.com/fluo10/sapphire-agent/issues/192)）。

**#191 への「保存していない」という参照は消すこと。** そのイシューは閉じる対象になる。

- [ ] **Step 2: spec に訂正を追記する**

spec 本文は書き換えず、末尾に `## 実装時の訂正` を新設して足す。

```markdown
## 実装時の訂正

### 拒否理由は `ToolSet::execute` を通らない

決定1は「組み込みツールも MCP ツールも、拒否されたときの理由文字列も、すべて
`ToolOutput` としてそこを通る」と書いたが、拒否理由は通らない。
`src/serve/mod.rs` と `src/agent.rs` が
`results.push((id, ToolOutput::from(reason)))` で直接組み立てている。

実害は無い — 理由は `policy::refusal_message` の定型文で構造上短い。ただし
「全部がそこを通る」という前提で後から何かを足すと外れるので、
`ToolSet::execute` の doc コメントに但し書きを残した。

### 上限は全トランスポートに掛かる

`ToolSet::execute` は `src/agent.rs` からも呼ばれるので、50 000 字の上限は
Matrix と Discord のツール結果にも掛かる。spec は ACP の文脈で書いていたが、
合流点が共有されている以上、片方だけ縛る理由が無い。意図した結果である。

### `truncate_output` は冪等ではなかった

spec は「シェルの既存の呼び出しは削る — 二重に掛けると省略マーカーが入れ子になる」
と書いた。正しいが、原因は呼び出しの重複ではなく関数そのものにあった。

旧定数は head 20 000 + tail 30 000 で、そこに省略マーカーが**上乗せ**されるため、
出力が上限の 50 000 を約35字超える。つまり切り詰め済みの結果をもう一度通すと
また切られる。呼び出しが1箇所しかなかったので露見していなかっただけ。

head・tail・マーカーの予算が合計で上限に収まるように定数を組み直し、
「一度切った結果をもう一度通しても変わらない」ことをテストで固定した。
上限は上限であるべきで、それは呼び出し側の規律ではなく関数の性質であるほうがよい。
```

- [ ] **Step 3: ワークスペース全体の確認**

```bash
cargo test --workspace
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
```

Expected: all pass.

- [ ] **Step 4: コミット**

```bash
git checkout -- Cargo.lock
git add README.md docs/superpowers/specs/2026-09-01-tool-call-persistence-design.md
git commit -m "docs: describe tool-call persistence and the new output cap"
```
