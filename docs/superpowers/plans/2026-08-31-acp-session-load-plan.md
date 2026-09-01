# Zed から過去のセッションを開く Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Zed から過去のセッションを一覧して開き、会話を続けられるようにする。

**Architecture:** `run_llm_turn` は履歴をメモリに持たなければ `store.load_session()` で遅延ハイドレートするので、継続そのものは実行層の変更を必要としない。足りないのは、ACP アダプタが新規 id を mint する以外の道を持たないことと、クライアントに会話を見せる replay。境界は namespace で、`list` と `load` の**両方**に置く。

**Tech Stack:** Rust 2024, `agent-client-protocol` 2.0.0, `tokio`, `serde_json`, `tempfile`

**Spec:** `docs/superpowers/specs/2026-08-31-acp-session-load-design.md`

## Global Constraints

- ブランチは `feat/acp-session-load`（作成済み、spec はコミット済み）。
- テストコマンドは `cargo test -p sapphire-agent`。反復中は `--workspace` を使わない（サブクレートを巻き込むと1サイクル7分になる）。コミット前に一度だけ `cargo test --workspace`。
- **`cargo check` と `cargo test` を混ぜない。** フィンガープリントが別なので、交互に走らせると毎回リンクし直しになる（実測: 混ぜると約3分、混ぜないと1秒未満）。
- cargo は**フォアグラウンドで、長いタイムアウトで**実行する。`run_in_background` を使わない。
- テストはソースファイル内の `mod tests` に置く（このリポジトリの作法）。
- 実装するのは `session/load` / `session/list` / `session/resume` の3つ。**`close` と `delete` は実装しない。**
- **capability はそれを実装したタスクで初めて立てる。** `load_session(true)` を `session/load` より先に返すのは嘘になる。
- `session/list` のフィルタ条件は3つすべて: namespace 一致（`None` は除外）、`closed_at` なし、`cwd` 指定時は一致。
- **`load` は namespace を照合して拒否する。** 一覧のフィルタだけでは境界が片側にしかない。
- **未知の id と他 namespace の id は同じ文言**を返す。区別すると id を列挙できる。
- `cursor` / `next_cursor` は使わない。`next_cursor: None` で全件返す。
- replay はストアの生履歴。`ContentPart::Text` のみ。画像はこの版では飛ばす。

---

### Task 1: `SessionMeta.cwd` と `SessionStore::session_header()`

`cwd` を永続化する場所と、`load` が namespace を照合するための読み口を作る。`src/session.rs` だけで完結する。

**Files:**
- Modify: `src/session.rs`（`SessionMeta` に1フィールド、`ensure_session` に1引数、`session_header()` と `list_session_headers()` を新設）
- Modify: `src/serve/mod.rs`（`ensure_session` の唯一の呼び出し元）
- Test: `src/session.rs` の `mod tests`

**Interfaces:**
- Produces: `SessionMeta.cwd: Option<String>`、`SessionStore::session_header(&self, session_id: &str) -> Option<(SessionMeta, bool)>`、`SessionStore::list_session_headers(&self) -> Vec<(SessionMeta, bool)>`、`SessionStore::ensure_session(&self, session_id, key, channel, public_id_override, namespace, cwd: Option<String>)`

`bool` は `is_closed`（`closed_at` マーカーの有無）。`load_session_file` の3番目の戻り値がそれで、公開された読み口が無い。**meta と closed を別々のメソッドにしない**のは、一覧が1セッションあたりファイルを2回読むことになるため。

- [ ] **Step 1: 失敗するテストを書く**

`src/session.rs` の `mod tests` に追加する。このファイルにテストモジュールが無ければ、末尾に `#[cfg(test)] mod tests { use super::*; ... }` を新設すること。

```rust
    /// `cwd` はメタ行に載り、読み戻せる。ACP の `session/list` が
    /// `SessionInfo.cwd` を埋めるために要る。
    #[test]
    fn cwd_round_trips_through_the_meta_line() {
        let dir = tempfile::tempdir().unwrap();
        let store = SessionStore::new(dir.path().to_path_buf(), "rpc");
        let key: ConversationKey = ("s-cwd".to_string(), None);

        store
            .ensure_session(
                "s-cwd",
                &key,
                "rpc",
                None,
                "default",
                Some("/home/u/project".to_string()),
            )
            .unwrap();

        let (meta, closed) = store.session_header("s-cwd").expect("the session exists");
        assert!(!closed, "a fresh session is not closed");
        assert_eq!(meta.cwd.as_deref(), Some("/home/u/project"));
        assert_eq!(meta.namespace.as_deref(), Some("default"));
    }

    /// `cwd` を渡さない経路（/rpc、voice、チャット）は `None` のまま。
    /// このフィールド以前に作られたファイルも同じ形になる。
    #[test]
    fn a_session_without_a_cwd_reads_back_as_none() {
        let dir = tempfile::tempdir().unwrap();
        let store = SessionStore::new(dir.path().to_path_buf(), "rpc");
        let key: ConversationKey = ("s-nocwd".to_string(), None);

        store
            .ensure_session("s-nocwd", &key, "rpc", None, "default", None)
            .unwrap();

        assert_eq!(store.session_header("s-nocwd").unwrap().0.cwd, None);
    }

    /// 知らない id には `None`。`session/load` はこれを「拒否」に
    /// 変換する。
    #[test]
    fn session_header_is_none_for_an_unknown_id() {
        let dir = tempfile::tempdir().unwrap();
        let store = SessionStore::new(dir.path().to_path_buf(), "rpc");
        assert!(store.session_header("no-such-session").is_none());
    }
```

- [ ] **Step 2: テストが落ちる（コンパイルしない）ことを確認**

Run: `cargo test -p sapphire-agent session::tests::cwd_round_trips`
Expected: FAIL — `session_header` が未定義で、`ensure_session` の引数が足りない。

- [ ] **Step 3: `SessionMeta` にフィールドを足す**

`src/session.rs` の `SessionMeta`、`room_profile` の下、`title` の上に。

```rust
    /// The client's workspace root for this session, when one was
    /// reported. Only ACP sets it: `session/new` and `session/load`
    /// carry a `cwd`, and `SessionInfo.cwd` is required when the editor
    /// lists sessions.
    ///
    /// `None` for every other path — `/rpc`, voice, chat — and for every
    /// file written before this field existed. `session/list` treats
    /// `None` as "belongs to no project", so those sessions are absent
    /// when the editor filters by `cwd` and present when it does not.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub cwd: Option<String>,
```

- [ ] **Step 4: 3つの `SessionMeta { .. }` リテラルを直す**

`cwd` フィールドを足したので、構造体リテラルが全部コンパイルエラーになる。`cargo test -p sapphire-agent --no-run` が場所を教えてくれる。`create_session`（約350行）と `ensure_session`（約684行）と、テスト内のリテラルがあればそれも。

- `create_session` → `cwd: None,`（チャット経路。cwd の概念がない）
- `ensure_session` → `cwd,`（Step 5 で足す引数をそのまま）

- [ ] **Step 5: `ensure_session` に引数を足す**

```rust
    pub fn ensure_session(
        &self,
        session_id: &str,
        key: &ConversationKey,
        channel: &str,
        public_id_override: Option<String>,
        namespace: &str,
        cwd: Option<String>,
    ) -> anyhow::Result<Option<String>> {
```

`SessionMeta` リテラルの `cwd: None,` を `cwd,` に変える。

- [ ] **Step 6: `session_header()` と `list_session_headers()` を足す**

`load_session` の隣に。両方とも `(SessionMeta, is_closed)` を返す — 呼ぶ側は
どちらの情報も要るし、分けるとファイルを二度読むことになる。

```rust
    /// This session's metadata and whether it has been closed.
    ///
    /// `session/load` needs the namespace before it decides whether the
    /// caller may open the session at all. The closed flag rides along
    /// because the only way to get either is to read the file, and
    /// splitting them into two methods would read it twice.
    pub fn session_header(&self, session_id: &str) -> Option<(SessionMeta, bool)> {
        let path = self.resolve_path(session_id)?;
        load_session_file(&path).map(|(meta, _, is_closed, _)| (meta, is_closed))
    }

    /// Every session's metadata and closed flag, oldest first.
    ///
    /// `list_sessions` drops the closed flag, and `session/list` has to
    /// exclude archived conversations — so this is the same walk keeping
    /// the one field that was being thrown away.
    pub fn list_session_headers(&self) -> Vec<(SessionMeta, bool)> {
        let mut headers: Vec<(SessionMeta, bool)> =
            collect_session_files(&self.base_dir, self.kind)
                .into_iter()
                .filter_map(|p| {
                    load_session_file(&p).map(|(meta, _, is_closed, _)| (meta, is_closed))
                })
                .collect();
        headers.sort_by_key(|(m, _)| m.created_at);
        headers
    }
```

- [ ] **Step 7: 唯一の呼び出し元を直す**

`src/serve/mod.rs` の約1927行。ここは `/acp` を含む `run_llm_turn` の経路で、cwd は Task 2 で入るので今は `None`。

```rust
            .ensure_session(&session_id, &key, "rpc", pending_pub_id, &namespace, None)
```

- [ ] **Step 8: テストが通ることを確認**

Run: `cargo test -p sapphire-agent`
Expected: PASS。既存テストを含めて全部。

- [ ] **Step 9: コミット**

```bash
git add src/session.rs src/serve/mod.rs
git commit -m "feat(sessions): record the client's cwd, and read a session's meta without its messages"
```

---

### Task 2: ACP の `cwd` を保存まで届ける

`session/new` はファイルを作らない — `run_llm_turn` の `ensure_session` が初回ターンで遅延生成する。したがって `cwd` はその瞬間まで持ち回る必要がある。`pending_sessions`（public_id を同じように持ち回っている）と同じ形にする。

**Files:**
- Modify: `src/serve/mod.rs`（`ServeState` に `pending_cwd`、`run_llm_turn` で消費）
- Modify: `src/serve/acp.rs`（`session/new` で登録）
- Test: `src/serve/acp.rs` の `mod tests`

**Interfaces:**
- Consumes: `SessionStore::ensure_session(..., cwd)`、`SessionStore::session_header()`（Task 1）
- Produces: `ServeState.pending_cwd: tokio::sync::Mutex<HashMap<String, String>>`

- [ ] **Step 1: 失敗するテストを書く**

`src/serve/acp.rs` の `mod tests` に。`drive` は `session/new` に `test_cwd()` を渡しているので、そのセッションのメタに cwd が入るはず。

```rust
    /// `session/new` の `cwd` は、そのセッションが初めて永続化される
    /// ときにメタ行へ載る。`session/list` がプロジェクトで絞るための
    /// 唯一の材料なので、ここで落ちると一覧が常に空になる。
    #[tokio::test]
    async fn a_new_sessions_cwd_reaches_the_store() {
        let state = ServeState::for_test_scripted(
            true,
            vec![crate::provider::ChatResponse {
                text: Some("ok".to_string()),
                tool_calls: Vec::new(),
                stop_reason: None,
            }],
        );
        let store = Arc::clone(&state.cross_device_session_store);
        let addr = spawn(state).await;

        let (session_id, _updates, _reply) = drive(&addr, text_prompt("hi")).await;

        let meta = store
            .session_header(&session_id)
            .map(|(m, _)| m)
            .expect("the turn persisted the session");
        assert_eq!(meta.cwd.as_deref(), Some(test_cwd()));
    }
```

- [ ] **Step 2: テストが落ちることを確認**

Run: `cargo test -p sapphire-agent acp::tests::a_new_sessions_cwd_reaches_the_store`
Expected: FAIL — `meta.cwd` が `None`。

- [ ] **Step 3: `ServeState` にフィールドを足す**

`src/serve/mod.rs`、`pending_sessions` の隣に。

```rust
    /// The client `cwd` for a session that has been created but not yet
    /// written to disk.
    ///
    /// `session/new` mints an id and nothing else; the JSONL file is
    /// created lazily by `ensure_session` on the first turn. The cwd
    /// arrives at `session/new` and is needed at `ensure_session`, so it
    /// waits here in between — the same shape `pending_sessions` uses to
    /// carry a reserved public_id across the same gap.
    pub(crate) pending_cwd: tokio::sync::Mutex<HashMap<String, String>>,
```

`ServeState::new` の `Self { .. }` と `build_for_test` の両方に
`pending_cwd: tokio::sync::Mutex::new(HashMap::new()),` を足す。

- [ ] **Step 4: `run_llm_turn` で消費する**

`src/serve/mod.rs`、`pending_pub_id` を取っている行のすぐ下。

```rust
        let pending_pub_id = state.pending_sessions.lock().await.remove(&session_id);
        let pending_cwd = state.pending_cwd.lock().await.remove(&session_id);
        if let Err(e) = store
            .ensure_session(
                &session_id,
                &key,
                "rpc",
                pending_pub_id,
                &namespace,
                pending_cwd,
            )
            .map(|_| ())
```

- [ ] **Step 5: `session/new` で登録する**

`src/serve/acp.rs` の `session/new` ハンドラ、`session_room_profiles` に入れている箇所の隣。

```rust
                    // The file does not exist yet — `ensure_session`
                    // creates it on the first turn — so the cwd waits in
                    // `pending_cwd` until then.
                    state.pending_cwd.lock().await.insert(
                        agent_session_id.clone(),
                        req.cwd.to_string_lossy().to_string(),
                    );
```

`AcpSession.cwd` に付いている `#[allow(dead_code)]` を外す（この後 `load` でも使う）。

- [ ] **Step 6: テストが通ることを確認**

Run: `cargo test -p sapphire-agent`
Expected: PASS。

- [ ] **Step 7: コミット**

```bash
git add src/serve/
git commit -m "feat(acp): carry a new session's cwd through to the store"
```

---

### Task 3: `session/list`

**Files:**
- Modify: `src/serve/acp.rs`（`initialize` の capability、`session/list` ハンドラ)
- Test: `src/serve/acp.rs` の `mod tests`

**Interfaces:**
- Consumes: `SessionStore::list_session_headers() -> Vec<(SessionMeta, bool)>`、`SessionMeta.cwd`（Task 1）、`SessionStore::absolute_path_for(&str) -> Option<PathBuf>`
- Produces: `session/list` エンドポイント

- [ ] **Step 1: 失敗するテストを書く**

`src/serve/acp.rs` の `mod tests` に。テストは `create_session` でストアに直接セッションを作る（`pub fn create_session(&self, key, channel, namespace) -> Result<String>`）。接続の namespace は `state.config.namespace_for_room_profile("developer")` で引ける。

```rust
    fn list_request(id: i64, cwd: Option<&str>) -> serde_json::Value {
        let params = match cwd {
            Some(cwd) => serde_json::json!({ "cwd": cwd }),
            None => serde_json::json!({}),
        };
        serde_json::json!({
            "jsonrpc": "2.0", "id": id, "method": "session/list", "params": params
        })
    }

    /// The boundary. A token pinned to one room profile must not see
    /// another profile's conversations, and a file too old to say which
    /// namespace it belongs to is not shown either — an unknown owner is
    /// not the same as "mine".
    #[tokio::test]
    async fn list_only_returns_this_namespaces_sessions() {
        let state = ServeState::for_test_scripted(true, Vec::new());
        let store = Arc::clone(&state.cross_device_session_store);
        let ours = state
            .config
            .namespace_for_room_profile("developer")
            .to_string();

        let mine = store
            .create_session(&("r-mine".to_string(), None), "rpc", &ours)
            .unwrap();
        let theirs = store
            .create_session(&("r-theirs".to_string(), None), "rpc", "someone-else")
            .unwrap();

        let addr = spawn(state).await;
        let replies = conversation(
            &addr,
            vec![initialize_request(0), list_request(1, None)],
        )
        .await;

        let ids: Vec<&str> = replies[1]["result"]["sessions"]
            .as_array()
            .expect("sessions is an array")
            .iter()
            .map(|s| s["sessionId"].as_str().unwrap())
            .collect();
        assert!(ids.contains(&mine.as_str()), "got {ids:?}");
        assert!(
            !ids.contains(&theirs.as_str()),
            "another namespace leaked into the list: {ids:?}"
        );
        assert_eq!(replies[1]["result"]["nextCursor"], serde_json::Value::Null);
    }

    /// A file written before `namespace` existed cannot say whose it is.
    /// An unknown owner is not the same as "mine", so it is not listed.
    /// Written by hand because `create_session` always records one.
    #[tokio::test]
    async fn list_omits_sessions_with_no_namespace() {
        let state = ServeState::for_test_scripted(true, Vec::new());
        let store = Arc::clone(&state.cross_device_session_store);
        let ours = state
            .config
            .namespace_for_room_profile("developer")
            .to_string();
        let mine = store
            .create_session(&("r-mine".to_string(), None), "rpc", &ours)
            .unwrap();

        // A legacy meta line: no `namespace` key at all.
        let legacy_dir = store
            .absolute_path_for(&mine)
            .unwrap()
            .parent()
            .unwrap()
            .to_path_buf();
        let legacy = legacy_dir.join("00000000-0000-7000-8000-00000000dead.jsonl");
        std::fs::write(
            &legacy,
            format!(
                "{}\n",
                serde_json::json!({"meta": {
                    "session_id": "00000000-0000-7000-8000-00000000dead",
                    "room_id": "r-legacy",
                    "thread_id": null,
                    "channel": "rpc",
                    "created_at": "2020-01-01T00:00:00Z"
                }})
            ),
        )
        .unwrap();

        let addr = spawn(state).await;
        let replies = conversation(
            &addr,
            vec![initialize_request(0), list_request(1, None)],
        )
        .await;

        let ids: Vec<&str> = replies[1]["result"]["sessions"]
            .as_array()
            .unwrap()
            .iter()
            .map(|s| s["sessionId"].as_str().unwrap())
            .collect();
        assert_eq!(ids, vec![mine.as_str()], "a namespace-less file leaked: {ids:?}");
    }

    /// A closed session is archived, not current. Listing it would offer
    /// the user a thread the agent has already summarised and moved on
    /// from.
    #[tokio::test]
    async fn list_omits_closed_sessions() {
        let state = ServeState::for_test_scripted(true, Vec::new());
        let store = Arc::clone(&state.cross_device_session_store);
        let ours = state
            .config
            .namespace_for_room_profile("developer")
            .to_string();

        let open = store
            .create_session(&("r-open".to_string(), None), "rpc", &ours)
            .unwrap();
        let closed = store
            .create_session(&("r-closed".to_string(), None), "rpc", &ours)
            .unwrap();
        store.close_session(&closed).unwrap();

        let addr = spawn(state).await;
        let replies = conversation(
            &addr,
            vec![initialize_request(0), list_request(1, None)],
        )
        .await;

        let ids: Vec<&str> = replies[1]["result"]["sessions"]
            .as_array()
            .unwrap()
            .iter()
            .map(|s| s["sessionId"].as_str().unwrap())
            .collect();
        assert_eq!(ids, vec![open.as_str()], "got {ids:?}");
    }

    /// The editor filters by project. A session with no recorded cwd
    /// belongs to no project, so it is absent when a cwd is asked for
    /// and present when one is not — which is how conversations from
    /// before cwd was recorded stay reachable.
    #[tokio::test]
    async fn list_honours_the_cwd_filter() {
        let state = ServeState::for_test_scripted(true, Vec::new());
        let store = Arc::clone(&state.cross_device_session_store);
        let ours = state
            .config
            .namespace_for_room_profile("developer")
            .to_string();

        let no_cwd = store
            .create_session(&("r-nocwd".to_string(), None), "rpc", &ours)
            .unwrap();
        let here = store
            .ensure_session(
                "s-here",
                &("r-here".to_string(), None),
                "rpc",
                None,
                &ours,
                Some("/projects/here".to_string()),
            )
            .map(|_| "s-here".to_string())
            .unwrap();
        store
            .ensure_session(
                "s-elsewhere",
                &("r-elsewhere".to_string(), None),
                "rpc",
                None,
                &ours,
                Some("/projects/elsewhere".to_string()),
            )
            .unwrap();

        let addr = spawn(state).await;
        let replies = conversation(
            &addr,
            vec![
                initialize_request(0),
                list_request(1, Some("/projects/here")),
                list_request(2, None),
            ],
        )
        .await;

        let ids = |r: &serde_json::Value| -> Vec<String> {
            r["result"]["sessions"]
                .as_array()
                .unwrap()
                .iter()
                .map(|s| s["sessionId"].as_str().unwrap().to_string())
                .collect()
        };

        assert_eq!(ids(&replies[1]), vec![here.clone()], "filtered by cwd");
        let all = ids(&replies[2]);
        assert!(all.contains(&no_cwd), "unfiltered must include the cwd-less session: {all:?}");
        assert_eq!(all.len(), 3, "got {all:?}");
    }

    #[tokio::test]
    async fn initialize_advertises_listing() {
        let state = ServeState::for_test_scripted(true, Vec::new());
        let addr = spawn(state).await;
        let reply = roundtrip(&addr, initialize_request(0)).await;
        assert!(
            !reply["result"]["agentCapabilities"]["sessionCapabilities"]["list"].is_null(),
            "got {reply}"
        );
    }
```

- [ ] **Step 2: テストが落ちることを確認**

Run: `cargo test -p sapphire-agent acp::tests::list_`
Expected: FAIL — `session/list` が `method not found`。

- [ ] **Step 3: capability を立てる**

`src/serve/acp.rs` の `initialize` ハンドラ。`loadSession` はまだ `false` のまま（Task 4 まで嘘をつかない）。

```rust
                responder.respond(
                    InitializeResponse::new(version).agent_capabilities(
                        AgentCapabilities::new()
                            .load_session(false)
                            .session_capabilities(
                                SessionCapabilities::new().list(SessionListCapabilities::new()),
                            ),
                    ),
                )
```

import に `SessionCapabilities`、`SessionListCapabilities` を足す。

- [ ] **Step 4: ハンドラを足す**

`session/new` のハンドラの後ろ、`session/cancel` の前に。

```rust
        .on_receive_request(
            {
                let state = Arc::clone(&state);
                let profile_name = profile_name.clone();
                async move |req: ListSessionsRequest, responder, _connection| {
                    let namespace = state
                        .config
                        .namespace_for_room_profile(&profile_name)
                        .to_string();
                    let store = Arc::clone(&state.cross_device_session_store);
                    let wanted_cwd = req.cwd.as_ref().map(|c| c.to_string_lossy().to_string());

                    let sessions: Vec<SessionInfo> = store
                        .list_session_headers()
                        .into_iter()
                        // A closed session is archived, not current.
                        .filter(|(_, is_closed)| !is_closed)
                        .map(|(meta, _)| meta)
                        .filter(|meta| {
                            // Three filters, and the namespace one is a
                            // boundary rather than a convenience. A file
                            // too old to name its namespace is not shown:
                            // an unknown owner is not the same as "mine".
                            meta.namespace.as_deref() == Some(namespace.as_str())
                        })
                        .filter(|meta| match &wanted_cwd {
                            Some(wanted) => meta.cwd.as_deref() == Some(wanted.as_str()),
                            None => true,
                        })
                        .filter_map(|meta| {
                            let path = store.absolute_path_for(&meta.session_id)?;
                            let updated_at = std::fs::metadata(&path)
                                .and_then(|m| m.modified())
                                .ok()
                                .map(|t| {
                                    chrono::DateTime::<chrono::Utc>::from(t)
                                        .to_rfc3339()
                                });
                            let cwd = meta
                                .cwd
                                .as_deref()
                                .map(PathBuf::from)
                                .unwrap_or_else(PathBuf::new);
                            let mut info = SessionInfo::new(
                                SessionId::new(meta.session_id.clone()),
                                cwd,
                            );
                            info = info.title(meta.title.clone());
                            if let Some(updated_at) = updated_at {
                                info = info.updated_at(updated_at);
                            }
                            Some(info)
                        })
                        .collect();

                    // No pagination: the whole list, every time. A
                    // cursor earns its keep when a namespace holds
                    // thousands of sessions, and none does yet.
                    responder.respond(ListSessionsResponse::new(sessions))
                }
            },
            on_receive_request!(),
        )
```

import に `ListSessionsRequest`、`ListSessionsResponse`、`SessionInfo` を足す。

**注意:** `list_sessions()` ではなく **`list_session_headers()`**（Task 1）を使う。前者は `closed_at` の情報を捨てるので、閉じたセッションを除外できない。

- [ ] **Step 5: `SessionInfo` のビルダー名を確認する**

`SessionInfo::new(session_id, cwd)` と `.title(..)` / `.updated_at(..)` の正確な形は、`agent-client-protocol-schema` の `v1/agent.rs` にある。`cargo test -p sapphire-agent --no-run` が違えばすぐ教えてくれるので、コンパイラに従うこと。`title` と `updated_at` は `Option` を取る `impl IntoOption<..>` の可能性が高い。

- [ ] **Step 6: テストが通ることを確認**

Run: `cargo test -p sapphire-agent`
Expected: PASS。

- [ ] **Step 7: コミット**

```bash
git add src/serve/acp.rs src/session.rs
git commit -m "feat(acp): list this namespace's sessions"
```

---

### Task 4: `session/load` と replay

**Files:**
- Modify: `src/serve/acp.rs`（`initialize` の `load_session(true)`、`session/load` ハンドラ）
- Test: `src/serve/acp.rs` の `mod tests`

**Interfaces:**
- Consumes: `SessionStore::session_header()`（Task 1）、`SessionStore::load_session()`
- Produces: `session/load` エンドポイント

- [ ] **Step 1: 失敗するテストを書く**

```rust
    fn load_request(id: i64, session_id: &str) -> serde_json::Value {
        serde_json::json!({
            "jsonrpc": "2.0", "id": id, "method": "session/load",
            "params": { "sessionId": session_id, "cwd": test_cwd(), "mcpServers": [] }
        })
    }

    /// Loading replays the conversation as session/update notifications,
    /// and does so BEFORE answering — a client that got the reply first
    /// would render an empty thread and then have messages appear under
    /// it.
    #[tokio::test]
    async fn load_replays_the_conversation_before_replying() {
        let state = ServeState::for_test_scripted(true, Vec::new());
        let store = Arc::clone(&state.cross_device_session_store);
        let ours = state
            .config
            .namespace_for_room_profile("developer")
            .to_string();
        let sid = store
            .create_session(&("r".to_string(), None), "rpc", &ours)
            .unwrap();
        store
            .append(&sid, &crate::provider::ChatMessage::user("first"))
            .unwrap();
        store
            .append(&sid, &crate::provider::ChatMessage::assistant("second"))
            .unwrap();

        let addr = spawn(state).await;
        let mut ws = connect(&addr).await;
        for request in [initialize_request(0), load_request(1, &sid)] {
            ws.send(Message::Text(request.to_string().into()))
                .await
                .unwrap();
        }

        let mut replayed: Vec<(String, String)> = Vec::new();
        loop {
            let Message::Text(t) = next_frame(&mut ws).await else {
                continue;
            };
            let v: serde_json::Value = serde_json::from_str(&t).unwrap();
            if v["method"] == "session/update" {
                let u = &v["params"]["update"];
                replayed.push((
                    u["sessionUpdate"].as_str().unwrap().to_string(),
                    u["content"]["text"].as_str().unwrap_or_default().to_string(),
                ));
            } else if v["id"] == 1 {
                assert!(v["error"].is_null(), "load failed: {v}");
                break;
            }
        }

        assert_eq!(
            replayed,
            vec![
                ("user_message_chunk".to_string(), "first".to_string()),
                ("agent_message_chunk".to_string(), "second".to_string()),
            ],
            "the replay must arrive in order, before the reply"
        );
    }

    /// The boundary that filtering the list cannot provide: `load` takes
    /// an id directly, so a session that never appears in any list is
    /// still reachable by anyone who learns its id.
    #[tokio::test]
    async fn load_refuses_another_namespaces_session() {
        let state = ServeState::for_test_scripted(true, Vec::new());
        let store = Arc::clone(&state.cross_device_session_store);
        let theirs = store
            .create_session(&("r".to_string(), None), "rpc", "someone-else")
            .unwrap();

        let addr = spawn(state).await;
        let replies = conversation(
            &addr,
            vec![initialize_request(0), load_request(1, &theirs)],
        )
        .await;
        assert_eq!(replies[1]["error"]["code"], -32602, "got {}", replies[1]);
    }

    /// The two refusals must be indistinguishable. If "not yours" reads
    /// differently from "no such session", the pair enumerates ids.
    #[tokio::test]
    async fn an_unknown_and_a_forbidden_session_look_the_same() {
        let state = ServeState::for_test_scripted(true, Vec::new());
        let store = Arc::clone(&state.cross_device_session_store);
        let theirs = store
            .create_session(&("r".to_string(), None), "rpc", "someone-else")
            .unwrap();

        let addr = spawn(state).await;
        let replies = conversation(
            &addr,
            vec![
                initialize_request(0),
                load_request(1, &theirs),
                load_request(2, "01900000-0000-7000-8000-000000000000"),
            ],
        )
        .await;

        assert_eq!(replies[1]["error"], replies[2]["error"], "the two refusals differ");
    }

    /// A loaded session is a real session: prompting it continues the
    /// conversation rather than starting a new one. This is what proves
    /// the adapter registered the *existing* id, not a fresh one.
    #[tokio::test]
    async fn a_loaded_session_continues_its_history() {
        let state = ServeState::for_test_scripted(
            true,
            vec![crate::provider::ChatResponse {
                text: Some("third".to_string()),
                tool_calls: Vec::new(),
                stop_reason: None,
            }],
        );
        let store = Arc::clone(&state.cross_device_session_store);
        let ours = state
            .config
            .namespace_for_room_profile("developer")
            .to_string();
        let sid = store
            .create_session(&("r".to_string(), None), "rpc", &ours)
            .unwrap();
        store
            .append(&sid, &crate::provider::ChatMessage::user("first"))
            .unwrap();

        let addr = spawn(state).await;
        let mut ws = connect(&addr).await;
        for request in [
            initialize_request(0),
            load_request(1, &sid),
            prompt_request(2, &sid, text_prompt("second")),
        ] {
            ws.send(Message::Text(request.to_string().into()))
                .await
                .unwrap();
        }
        loop {
            let Message::Text(t) = next_frame(&mut ws).await else {
                continue;
            };
            let v: serde_json::Value = serde_json::from_str(&t).unwrap();
            if v["id"] == 2 {
                assert_eq!(v["result"]["stopReason"], "end_turn", "got {v}");
                break;
            }
        }

        let history = store.load_session(&sid).expect("the session still exists");
        let texts: Vec<String> = history
            .iter()
            .flat_map(|m| &m.parts)
            .filter_map(|p| match p {
                crate::provider::ContentPart::Text(t) => Some(t.clone()),
                _ => None,
            })
            .collect();
        assert_eq!(texts, vec!["first", "second", "third"], "got {texts:?}");
    }

    #[tokio::test]
    async fn initialize_advertises_loading() {
        let state = ServeState::for_test_scripted(true, Vec::new());
        let addr = spawn(state).await;
        let reply = roundtrip(&addr, initialize_request(0)).await;
        assert_eq!(reply["result"]["agentCapabilities"]["loadSession"], true);
    }
```

- [ ] **Step 2: テストが落ちることを確認**

Run: `cargo test -p sapphire-agent acp::tests::load_`
Expected: FAIL — `session/load` が `method not found`。

- [ ] **Step 3: ハンドラを足す**

`session/list` の後ろに。

```rust
        .on_receive_request(
            {
                let sessions = Arc::clone(&sessions);
                let state = Arc::clone(&state);
                let profile_name = profile_name.clone();
                async move |req: LoadSessionRequest,
                            responder,
                            connection: ConnectionTo<Client>| {
                    let id = req.session_id.to_string();
                    let namespace = state
                        .config
                        .namespace_for_room_profile(&profile_name)
                        .to_string();
                    let store = Arc::clone(&state.cross_device_session_store);

                    // One refusal for both "no such session" and "not
                    // yours". Distinguishing them would let a caller
                    // enumerate which ids exist.
                    let refuse = || {
                        Error::invalid_params()
                            .data(format!("no session '{id}' available on this connection"))
                    };
                    let Some((meta, _closed)) = store.session_header(&id) else {
                        return responder.respond_with_error(refuse());
                    };
                    if meta.namespace.as_deref() != Some(namespace.as_str()) {
                        warn!(
                            "ACP: refused session/load for {id}: it belongs to namespace {:?}, \
                             not {namespace}",
                            meta.namespace
                        );
                        return responder.respond_with_error(refuse());
                    }

                    // Register the EXISTING id rather than minting one.
                    // `run_llm_turn` hydrates history from the store for
                    // an id it has not seen, so continuing the
                    // conversation needs nothing further.
                    state
                        .session_room_profiles
                        .lock()
                        .await
                        .insert(id.clone(), profile_name.clone());
                    sessions.inner.lock().await.insert(
                        req.session_id.clone(),
                        AcpSession {
                            agent_session_id: id.clone(),
                            cwd: req.cwd.clone(),
                            turns: HashMap::new(),
                            mode: crate::tools::policy::SessionMode::Default,
                        },
                    );

                    // Replay BEFORE answering: the ACP specification
                    // orders it that way, and a client that got the
                    // reply first would render an empty thread and then
                    // watch messages appear underneath it.
                    for message in store.load_session(&id).unwrap_or_default() {
                        let text: String = message
                            .parts
                            .iter()
                            .filter_map(|part| match part {
                                crate::provider::ContentPart::Text(t) => Some(t.as_str()),
                                _ => None,
                            })
                            .collect::<Vec<_>>()
                            .join("\n");
                        if text.is_empty() {
                            continue;
                        }
                        let chunk = ContentChunk::new(ContentBlock::Text(TextContent::new(text)));
                        let update = match message.role {
                            crate::provider::Role::User => SessionUpdate::UserMessageChunk(chunk),
                            _ => SessionUpdate::AgentMessageChunk(chunk),
                        };
                        if let Err(e) = connection.send_notification(SessionNotification::new(
                            req.session_id.clone(),
                            update,
                        )) {
                            warn!("ACP: dropped a replay update for {id}: {e}");
                        }
                    }

                    responder.respond(LoadSessionResponse::new().modes(mode_state()))
                }
            },
            on_receive_request!(),
        )
```

import に `LoadSessionRequest`、`LoadSessionResponse` を足す。

- [ ] **Step 4: `mode_state()` を切り出す**

`session/new` が `SessionModeState` を組み立てているコードを、モジュール内の自由関数に出して両方から呼ぶ。`session/new` 側もこれを使うように書き換えること。

```rust
/// The mode list every session starts with.
///
/// A loaded session starts in `default` like a new one: the mode is a
/// statement about how the editor wants the agent to behave right now,
/// not a property of the conversation, so it is not persisted.
fn mode_state() -> SessionModeState {
    SessionModeState::new(
        crate::tools::policy::SessionMode::Default.id(),
        crate::tools::policy::SessionMode::ALL
            .into_iter()
            .map(|m| AcpSessionMode::new(m.id(), m.name()).description(m.description()))
            .collect(),
    )
}
```

- [ ] **Step 5: capability を `true` にする**

`initialize` の `.load_session(false)` を `.load_session(true)` に変える。

- [ ] **Step 6: テストが通ることを確認**

Run: `cargo test -p sapphire-agent`
Expected: PASS。既存の `initialize_answers_with_v1_capabilities` が `loadSession: false` を期待していたら、`true` に更新すること。

- [ ] **Step 7: コミット**

```bash
git add src/serve/acp.rs
git commit -m "feat(acp): load a past session and replay its conversation"
```

---

### Task 5: `session/resume`

`load` から replay を抜くだけ。

**Files:**
- Modify: `src/serve/acp.rs`
- Test: `src/serve/acp.rs` の `mod tests`

**Interfaces:**
- Consumes: Task 4 の `session/load` ハンドラの構造
- Produces: `session/resume` エンドポイント

- [ ] **Step 1: 失敗するテストを書く**

```rust
    fn resume_request(id: i64, session_id: &str) -> serde_json::Value {
        serde_json::json!({
            "jsonrpc": "2.0", "id": id, "method": "session/resume",
            "params": { "sessionId": session_id, "cwd": test_cwd(), "mcpServers": [] }
        })
    }

    /// `resume` is `load` without the replay — for a client that wants
    /// the session back without paying to redraw a long conversation.
    /// It must still continue the history.
    #[tokio::test]
    async fn resume_continues_without_replaying() {
        let state = ServeState::for_test_scripted(
            true,
            vec![crate::provider::ChatResponse {
                text: Some("second".to_string()),
                tool_calls: Vec::new(),
                stop_reason: None,
            }],
        );
        let store = Arc::clone(&state.cross_device_session_store);
        let ours = state
            .config
            .namespace_for_room_profile("developer")
            .to_string();
        let sid = store
            .create_session(&("r".to_string(), None), "rpc", &ours)
            .unwrap();
        store
            .append(&sid, &crate::provider::ChatMessage::user("first"))
            .unwrap();

        let addr = spawn(state).await;
        let mut ws = connect(&addr).await;
        for request in [initialize_request(0), resume_request(1, &sid)] {
            ws.send(Message::Text(request.to_string().into()))
                .await
                .unwrap();
        }

        let mut updates_before_reply = 0usize;
        loop {
            let Message::Text(t) = next_frame(&mut ws).await else {
                continue;
            };
            let v: serde_json::Value = serde_json::from_str(&t).unwrap();
            if v["method"] == "session/update" {
                updates_before_reply += 1;
            } else if v["id"] == 1 {
                assert!(v["error"].is_null(), "resume failed: {v}");
                break;
            }
        }
        assert_eq!(updates_before_reply, 0, "resume must not replay");

        // ...and the history is still there for the next turn.
        ws.send(Message::Text(
            prompt_request(2, &sid, text_prompt("x")).to_string().into(),
        ))
        .await
        .unwrap();
        loop {
            let Message::Text(t) = next_frame(&mut ws).await else {
                continue;
            };
            let v: serde_json::Value = serde_json::from_str(&t).unwrap();
            if v["id"] == 2 {
                assert_eq!(v["result"]["stopReason"], "end_turn", "got {v}");
                break;
            }
        }
        let history = store.load_session(&sid).unwrap();
        assert!(history.len() >= 3, "the resumed session kept its history");
    }

    #[tokio::test]
    async fn initialize_advertises_resume() {
        let state = ServeState::for_test_scripted(true, Vec::new());
        let addr = spawn(state).await;
        let reply = roundtrip(&addr, initialize_request(0)).await;
        assert!(
            !reply["result"]["agentCapabilities"]["sessionCapabilities"]["resume"].is_null(),
            "got {reply}"
        );
    }
```

- [ ] **Step 2: テストが落ちることを確認**

Run: `cargo test -p sapphire-agent acp::tests::resume_`
Expected: FAIL — `session/resume` が `method not found`。

- [ ] **Step 3: 共通部分を関数に切り出す**

`session/load` のハンドラから、検証と登録の部分を自由関数にする。replay の有無だけが違う。

```rust
/// Validate an existing session id and adopt it onto this connection.
///
/// Shared by `session/load` and `session/resume`, which differ only in
/// whether they replay afterwards. `Err` carries the refusal to answer
/// with — one wording for both "no such session" and "not yours", so
/// the pair cannot be used to enumerate ids.
async fn adopt_session(
    state: &Arc<ServeState>,
    sessions: &Arc<AcpSessions>,
    profile_name: &str,
    session_id: &SessionId,
    cwd: PathBuf,
) -> Result<String, Error> {
    let id = session_id.to_string();
    let namespace = state
        .config
        .namespace_for_room_profile(profile_name)
        .to_string();
    let refuse = || {
        Error::invalid_params().data(format!("no session '{id}' available on this connection"))
    };

    let Some((meta, _closed)) = state.cross_device_session_store.session_header(&id) else {
        return Err(refuse());
    };
    if meta.namespace.as_deref() != Some(namespace.as_str()) {
        warn!(
            "ACP: refused adopting {id}: it belongs to namespace {:?}, not {namespace}",
            meta.namespace
        );
        return Err(refuse());
    }

    state
        .session_room_profiles
        .lock()
        .await
        .insert(id.clone(), profile_name.to_string());
    sessions.inner.lock().await.insert(
        session_id.clone(),
        AcpSession {
            agent_session_id: id.clone(),
            cwd,
            turns: HashMap::new(),
            mode: crate::tools::policy::SessionMode::Default,
        },
    );
    Ok(id)
}
```

`session/load` をこれを使う形に書き換える。

- [ ] **Step 4: `session/resume` ハンドラを足す**

```rust
        .on_receive_request(
            {
                let sessions = Arc::clone(&sessions);
                let state = Arc::clone(&state);
                let profile_name = profile_name.clone();
                async move |req: ResumeSessionRequest, responder, _connection| {
                    // Same adoption as `load`, no replay. The ACP
                    // specification frames `resume` as the fallback for
                    // agents that cannot load at all; offered here so a
                    // client can skip redrawing a long conversation.
                    match adopt_session(
                        &state,
                        &sessions,
                        &profile_name,
                        &req.session_id,
                        req.cwd.clone(),
                    )
                    .await
                    {
                        Ok(_) => responder.respond(ResumeSessionResponse::new().modes(mode_state())),
                        Err(e) => responder.respond_with_error(e),
                    }
                }
            },
            on_receive_request!(),
        )
```

import に `ResumeSessionRequest`、`ResumeSessionResponse` を足す。capability に `.resume(SessionResumeCapabilities::new())` を足す（import も）。

- [ ] **Step 5: テストが通ることを確認**

Run: `cargo test -p sapphire-agent`
Expected: PASS。

- [ ] **Step 6: コミット**

```bash
git add src/serve/acp.rs
git commit -m "feat(acp): resume a session without replaying it"
```

---

### Task 6: 同じセッションを二重に開いたときの警告、そしてドキュメント

`load` は「別の接続から同じセッションを開く」を可能にする。`run_llm_turn` の履歴レースは直さないが、**踏んだことが観測できる**ようにはする。

**Files:**
- Modify: `src/serve/mod.rs`（`ServeState` に `open_acp_sessions`）
- Modify: `src/serve/acp.rs`（登録と解放、警告）
- Modify: `README.md`
- Test: `src/serve/acp.rs` の `mod tests`

**Interfaces:**
- Consumes: `adopt_session`（Task 5）
- Produces: `ServeState.open_acp_sessions: tokio::sync::Mutex<HashMap<String, usize>>`

- [ ] **Step 1: 失敗するテストを書く**

```rust
    /// Two connections can now hold the same session, which makes the
    /// history race in `run_llm_turn` reachable across connections
    /// rather than only within one. The race is not fixed here — this
    /// only makes hitting it observable, because a corrupted transcript
    /// with no log line is undebuggable.
    #[tokio::test]
    async fn opening_a_session_twice_is_counted() {
        let state = ServeState::for_test_scripted(true, Vec::new());
        let store = Arc::clone(&state.cross_device_session_store);
        let counts = Arc::clone(&state.open_acp_sessions);
        let ours = state
            .config
            .namespace_for_room_profile("developer")
            .to_string();
        let sid = store
            .create_session(&("r".to_string(), None), "rpc", &ours)
            .unwrap();

        let addr = spawn(state).await;
        let mut a = connect(&addr).await;
        let mut b = connect(&addr).await;
        for ws in [&mut a, &mut b] {
            for request in [initialize_request(0), load_request(1, &sid)] {
                ws.send(Message::Text(request.to_string().into()))
                    .await
                    .unwrap();
            }
            loop {
                let Message::Text(t) = next_frame(ws).await else {
                    continue;
                };
                let v: serde_json::Value = serde_json::from_str(&t).unwrap();
                if v["id"] == 1 {
                    assert!(v["error"].is_null(), "load failed: {v}");
                    break;
                }
            }
        }

        assert_eq!(
            counts.lock().await.get(&sid).copied(),
            Some(2),
            "both connections should be counted as holding the session"
        );
    }
```

`ServeState.open_acp_sessions` はテストから読むので `pub(crate)` かつ `Arc` で包む（`Arc<tokio::sync::Mutex<HashMap<String, usize>>>`）。

- [ ] **Step 2: テストが落ちることを確認**

Run: `cargo test -p sapphire-agent acp::tests::opening_a_session_twice`
Expected: FAIL — `open_acp_sessions` が未定義。

- [ ] **Step 3: `ServeState` にカウンタを足す**

```rust
    /// How many live ACP connections hold each session.
    ///
    /// `session/load` lets two editors open one conversation. The
    /// history race in `run_llm_turn` — clone at the top, write the
    /// whole vector back at the end — then becomes reachable across
    /// connections instead of only within one. Fixing it needs a
    /// per-session lock around a whole turn, which is a separate job;
    /// this makes hitting it visible in the log, because a transcript
    /// that diverged with nothing written down cannot be debugged.
    pub(crate) open_acp_sessions: Arc<tokio::sync::Mutex<HashMap<String, usize>>>,
```

`ServeState::new` と `build_for_test` の両方に `open_acp_sessions: Arc::new(tokio::sync::Mutex::new(HashMap::new())),`。

- [ ] **Step 4: `adopt_session` で数え、警告する**

`adopt_session` の末尾、`Ok(id)` の直前に。

```rust
    {
        let mut open = state.open_acp_sessions.lock().await;
        let count = open.entry(id.clone()).or_insert(0);
        *count += 1;
        if *count > 1 {
            warn!(
                "ACP: session {id} is now open on {count} connections. Concurrent prompts on \
                 one session are not history-safe: run_llm_turn clones the history at the top \
                 and writes it back whole, so the last turn to finish wins in memory while both \
                 have already appended to the transcript."
            );
        }
    }
```

`session/new` も数える（新規セッションも接続が持つ）。

- [ ] **Step 5: 接続が終わるときに減らす**

`serve_connection` の末尾、`connection_cancel.cancel()` の隣。この接続が持っていたセッションを列挙して減らす。

```rust
    {
        let held: Vec<String> = sessions
            .inner
            .lock()
            .await
            .values()
            .map(|s| s.agent_session_id.clone())
            .collect();
        let mut open = state.open_acp_sessions.lock().await;
        for id in held {
            if let Some(count) = open.get_mut(&id) {
                *count = count.saturating_sub(1);
                if *count == 0 {
                    open.remove(&id);
                }
            }
        }
    }
```

- [ ] **Step 6: テストが通ることを確認**

Run: `cargo test -p sapphire-agent`
Expected: PASS。

- [ ] **Step 7: README を更新する**

`README.md` の Zed / ACP の節に「過去のセッションを開く」を足す。書くこと:

- Zed のスレッド一覧に、そのトークンのルームプロファイルに属するセッションが出る
- **この変更以前の会話と `/rpc` 由来の会話は `cwd` を持たないので、Zed がプロジェクトで絞る一覧には出ない。**絞らない一覧には出る
- 開いたセッションは会話が復元され、続きを書ける
- **ツール呼び出しは復元されない** — JSONL に保存していないため。Known limitations にも既にある「usage が無い」の隣に置く
- 同じセッションを2つのウィンドウで開いて同時に喋らせると履歴が壊れる

- [ ] **Step 8: 最終確認**

```bash
cargo fmt --all
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace
```

Expected: すべて通る。警告ゼロ。

- [ ] **Step 9: コミット**

```bash
git add src/serve/ README.md
git commit -m "feat(acp): count the connections holding a session, and document loading"
```

---

## 実装後に残ること（spec の「別イシューに切り出すもの」より）

- **ツール結果のキャッシュと忠実な復元。** JSONL に `tool_use` / `tool_result` の参照を入れ、ワークスペース外のキャッシュから復元する。再開用要約はロード時の遅延処理にし、キャッシュから復元できなかった場合だけ走らせる。これが入ると replay にツール呼び出しが出る。
- **セッション単位のロック。** Task 6 が観測できるようにしただけの履歴レース。
- **`session/close` と `session/delete`。**
- **ページネーション。** `cursor` / `next_cursor`。
- **replay の画像。** `ImageRef` をキャッシュから引いて流す。`promptCapabilities.image` と一緒に。
