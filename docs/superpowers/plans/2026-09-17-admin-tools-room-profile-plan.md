# 管理ツールの許可単位を room_profile 名へ（実装計画書）

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development
> (recommended) or superpowers:executing-plans to implement this plan task-by-task.
> Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 管理ツール（`heartbeat_config` / `autonomous_config` / `agent_config` /
`task_test`）の許可単位を room ID から room_profile 名へ置き換え、ACP（Zed）セッションから
自分の room_profile が列挙されていれば使えるようにする。旧 `[tools.admin].rooms` は
起動時に loud に失敗させて `room_profiles` へ誘導する。

**Architecture:** 設定キーを `[tools.admin].room_profiles`（Vec<String>、room_profile 名）へ
置換し、gate の入力だけを差し替える。チャンネル経路は `TimerOrigin::Chat { room_id }` を
`Config::room_profile_name_for_room` で解決し、ACP 経路は
`ServeState.session_room_profiles` に pin 済みの profile 名を、`config_tools.rs` に新設する
task-local `ADMIN_ROOM_PROFILE_TL`（`TurnLoop::run` が `scope_admin_room_profile` で包む）に
載せてツール実行まで運ぶ。`run_llm_turn` の既存 `is_acp` が唯一の分岐点。voice・`/rpc`・
A2A・無人ループは値が `None` のままなので従来どおり拒否される。

**Tech Stack:** Rust（tokio task-local, serde/serde_ignored, async-trait, serde_json）、
既存の `Config` / `ToolSet` / `Tool` / `TurnLoop` / `TurnContext` 機構。

**Spec:** `docs/superpowers/specs/2026-09-17-admin-tools-room-profile-design.md`
（台帳コピー: `.superpowers/sdd/2026-09-17-admin-tools-room-profile/2026-09-17-admin-tools-room-profile-design.md`）

## Global Constraints

- コード・コメント・コミットメッセージは英語（`CLAUDE.md` / `CONTRIBUTING.md` 規約）。
  ドキュメント（`docs/superpowers/**`、本ファイル）は日本語。
- コミットは conventional commits。スコープは `config` / `tools` / `serve` / `docs` /
  `test`（`CLAUDE.md` のスコープ表に従う。いずれも `cliff.toml` の skip 一覧に無いので
  エージェント側 changelog に載る）。
- 新キーの意味論は provider / memory namespace の room_profile 解決と一致させる。
  解決順は 明示一致 → `[room_profile.default]` → 暗黙の `"default"`。
- fail-closed を崩さない: 値が `None` の経路（voice / `/rpc` / A2A / 無人）は常に拒否。
- gate は「登録条件」と「実行時判定」の二重許可のまま（登録条件を緩めない）。
- 各タスクの最後に `cargo fmt --all` を掛けてからコミットする。
- 既存テスト様式に従う: `Config::parse_for_test`、`Config::for_test`、
  `ServeState::for_test_scripted*` + `StubProvider`、`scope_timer_origin` /
  `scope_turn_context`、ACP の `drive` ヘルパ。

## File Structure

| ファイル | 責務（本計画での変更点） |
|---|---|
| `server/src/config.rs` | `AdminToolsConfig` のキー置換（`room_profiles` + 捕捉専用 `legacy_rooms`）、`admin_allows_room_profile` / `room_profile_name_for_room` / `config_tools_enabled`、`validate_profiles` への未知 profile 検証、`migration_errors` への旧キー誘導。旧 `config_tools_allowed_in` は削除。 |
| `server/src/tools/config_tools.rs` | 許可判定を `current_admin_room_profile_for(config)` に一本化。ACP 用 task-local `ADMIN_ROOM_PROFILE_TL` + `scope_admin_room_profile` / `current_admin_room_profile`。`current_call_room` を削除。`ROOM_REFUSAL`・`ConfigTool::spec()`・`TaskTestTool::spec()`・モジュール doc を新キーに更新。`TaskTestTool::namespace` を profile 名解決に変更。gate/namespace のテスト。 |
| `server/src/serve/mod.rs` | `TurnContext` と `TurnLoop` に `admin_room_profile: Option<String>`、`run_llm_turn` が `is_acp` から値を決めて流す、`TurnLoop::run` がツール実行を `scope_admin_room_profile` で包む。テスト用 `#[cfg(test)] pub(crate) struct AdminProfileProbe` を追加。非 ACP 経路の `None` 検証テスト。 |
| `server/src/tools/subagent.rs` | nested `TurnLoop` が `ctx.admin_room_profile` を継承。テスト用 `TurnContext` リテラル 2 箇所に `admin_room_profile: None`。 |
| `server/src/tools/skill_tools.rs` | テスト用 `TurnContext` リテラル 1 箇所に `admin_room_profile: None`。 |
| `server/src/serve/acp.rs` | ACP の `session/prompt` ターンで probe が pin 済み profile 名（fixture の `"developer"`）を見ることを end-to-end で固定。 |
| `server/src/main.rs` | `[tools.admin].rooms` に触れるコメント 3 箇所を `room_profiles` に更新（挙動の変更なし）。 |
| `server/config.example.toml` | `[tools.admin]` 例を `room_profiles` に差し替え、`"default"` の広さを明記。 |
| `server/templates/workspace/config.toml` | host-local であることの説明中のキー名を更新。 |
| `README.md` | 管理ツール節（room 許可リストの説明）を room_profile 許可リストへ書き換え、ACP から使えることと `"default"` の注意を追記。 |
| `README.ja.md` | 上記節の要約行のキー名を更新。 |

## 設計の確定事項（タスク間で共有する前提）

1. **設定キー**: `[tools.admin]` は
   `room_profiles = ["ops", "developer"]`（room_profile 名）。
   `#[serde(default, rename = "rooms")] legacy_rooms: Vec<String>` が旧キーを捕捉し、
   非空なら `migration_errors()` が起動を止める。二重読み（fixture 用の後方互換）は**しない**。
2. **判定 API**:
   - `Config::admin_allows_room_profile(&self, name: &str) -> bool` — 列挙の membership。
   - `Config::room_profile_name_for_room(&self, room_id: &str) -> &str` — `room_profile_for`
     の解決結果を名前に落とす（`DEFAULT_PROFILE_NAME` で必ず `Some`）。
   - `Config::config_tools_enabled(&self) -> bool` — `room_profiles` 非空。
   - `Config::config_tools_allowed_in(Option<&str>)` は**削除**する。
3. **gate 入力**（`config_tools.rs`）:
   ```rust
   pub(crate) fn current_admin_room_profile_for(config: &Config) -> Option<String>
   ```
   - `TimerOrigin::Chat { room_id }` → `room_profile_name_for_room(&room_id)` を `Some`
   - `TimerOrigin::Voice { .. }` → `None`
   - `current_origin()` が `None` → task-local `current_admin_room_profile()`
     （ACP ターンだけが `Some`、`/rpc`/A2A/無人は `None`）
4. **許可**: `current_admin_room_profile_for(&config).is_some_and(|n| config.admin_allows_room_profile(&n))`。
   `ConfigTool::gate()` と `TaskTestTool::execute()` の両方で同じ式を使う。
5. **伝達**: `TurnContext` / `TurnLoop` の新フィールド `admin_room_profile: Option<String>`。
   `run_llm_turn` が `is_acp` のとき `session_room_profiles` の pin を clone、それ以外 `None`。
   `TurnLoop::run` はツール実行の既存 `scope_turn_context` の直前に 1 回だけ
   `scope_admin_room_profile(...)` を掛ける（`match (origin, client)` の外側）。
   サブエージェントの nested `TurnLoop` は `ctx.admin_room_profile.clone()` で継承。
6. **`task_test` の namespace**: `namespace_for_room_profile` に置き換える
   （チャンネル経路の結果は定義上不変）。
7. **テスト用 `TurnContext` リテラルは 5 箇所**あり、すべて `admin_room_profile` の追加が要る:
   `serve/mod.rs`（production 構築）、`tools/subagent.rs`（`turn_context_with_host` と
   `parent_ctx`）、`tools/skill_tools.rs`（`test_turn_context`）、
   `tools/config_tools.rs`（`after_write` テスト内）。
   **`TurnLoop` リテラルは 2 箇所**: `serve/mod.rs`（production）と `tools/subagent.rs`（nested）。
8. **変更しないもの**: `ToolKind`（4 ツールとも `Edit`）、`register_admin_tools` の呼び出し条件、
   `agents/*.md` の `tools:` 制限、`policy::decide`、`ROOM_REFUSAL` 以外の拒否文言。

---

## Task 1: 設定キーを `room_profiles` へ置換する

**Files:**
- Modify: `server/src/config.rs`（`AdminToolsConfig` L793、`config_tools_allowed_in` /
  `config_tools_enabled` L1683-1694、`validate_profiles` L1496、`migration_errors` L1635、tests L2093-2109）
- Modify: `server/src/tools/config_tools.rs`（gate L311 と L873、fixture L991、および
  `Todo` の機械的追従のみ。ACP 経路は Task 2）

**Interfaces:**
- Produces: `AdminToolsConfig { room_profiles, legacy_rooms }`、
  `Config::admin_allows_room_profile(&str) -> bool`、
  `Config::room_profile_name_for_room(&str) -> &str`、`Config::config_tools_enabled() -> bool`。
  Task 2 の `current_admin_room_profile_for(config)` がこの 3 つを使う。

- [ ] Step 1: 失敗するテストを `server/src/config.rs` の `mod tests`（既存
  `config_tools_are_off_until_a_room_is_named` L2093 の位置）で、既存 2 テストを
  以下の 5 テストに**置き換える**:

```rust
    /// The default is off: no room profile is named, so none of the four
    /// tools is ever registered.
    #[test]
    fn config_tools_are_off_until_a_room_profile_is_named() {
        let cfg = parse("[anthropic]\napi_key = \"test\"\n");
        assert!(!cfg.config_tools_enabled());
        assert!(!cfg.admin_allows_room_profile("ops"));
        assert!(!cfg.admin_allows_room_profile("default"));
    }

    /// A named room profile is the grant, and the name is the room_profile
    /// key — not a room id.
    #[test]
    fn config_tools_are_allowed_in_a_named_room_profile() {
        let cfg = parse(
            "[anthropic]\napi_key = \"test\"\n\n[tools.admin]\n\
             room_profiles = [\"ops\", \"developer\"]\n",
        );
        assert!(cfg.config_tools_enabled());
        assert!(cfg.admin_allows_room_profile("ops"));
        assert!(cfg.admin_allows_room_profile("developer"));
        assert!(!cfg.admin_allows_room_profile("guest"));
        assert!(
            !cfg.admin_allows_room_profile("!ops:x"),
            "a room id is not a room_profile name"
        );
    }

    /// A channel room resolves to the profile the rest of the config uses:
    /// an explicit listing wins, `[room_profile.default]` catches the rest,
    /// and with neither defined the implicit `"default"` applies.
    #[test]
    fn a_room_resolves_to_the_same_profile_everywhere() {
        let cfg = parse(
            "[anthropic]\napi_key = \"test\"\n\n[profiles.dev]\nprovider = \"stub\"\n\n\
             [room_profile.ops]\nprofile = \"dev\"\nrooms = [\"!ops:x\"]\n\n\
             [room_profile.default]\nprofile = \"dev\"\n",
        );
        assert_eq!(cfg.room_profile_name_for_room("!ops:x"), "ops");
        assert_eq!(cfg.room_profile_name_for_room("!unlisted:y"), "default");

        let bare = parse("[anthropic]\napi_key = \"test\"\n");
        assert_eq!(
            bare.room_profile_name_for_room("!anything:z"),
            "default",
            "with no room_profile table at all the implicit default still names a profile"
        );
    }

    /// Listing the implicit `"default"` profile is what an operator writes to
    /// allow every room no profile claims — a broad grant, and the reason the
    /// docs say so.
    #[test]
    fn default_profile_is_allowed_by_name() {
        let cfg = parse(
            "[anthropic]\napi_key = \"test\"\n\n[tools.admin]\nroom_profiles = [\"default\"]\n",
        );
        assert!(cfg.config_tools_enabled());
        assert!(cfg.admin_allows_room_profile("default"));
    }

    /// `roles` is gone: a config that still sets it must not start, because
    /// silently dropping it makes the admin tools vanish with no symptom that
    /// names the config file.
    #[test]
    fn the_removed_admin_rooms_key_is_a_migration_error() {
        let cfg = parse(
            "[anthropic]\napi_key = \"test\"\n\n[tools.admin]\nrooms = [\"!ops:x\"]\n",
        );
        let errors = cfg.migration_errors();
        assert_eq!(errors.len(), 1, "{errors:?}");
        assert!(errors[0].contains("[tools.admin].rooms"), "{errors:?}");
        assert!(errors[0].contains("room_profiles"), "{errors:?}");
    }

    /// A typo in the allow list must fail `verify` and startup, not silently
    /// grant nothing.
    #[test]
    fn an_unknown_room_profile_in_the_allow_list_is_reported() {
        let cfg = parse(
            "[anthropic]\napi_key = \"test\"\n\n[tools.admin]\nroom_profiles = [\"typo\"]\n",
        );
        let errors = cfg.validate_profiles();
        assert!(
            errors
                .iter()
                .any(|e| e.contains("[tools.admin].room_profiles") && e.contains("typo")),
            "{errors:?}"
        );

        // `"default"` is always valid, defined or not.
        let ok = parse(
            "[anthropic]\napi_key = \"test\"\n\n[tools.admin]\nroom_profiles = [\"default\"]\n",
        );
        assert!(ok.validate_profiles().is_empty(), "{:?}", ok.validate_profiles());
    }
```

  注: 上記コメント中の `roles` は誤記ではなく読み替え不要 — 正しくは
  `rooms`。Step 3 で書き直す際に `/// rooms is gone:` とすること。

- [ ] Step 2: `cargo test -p sapphire-agent-server --lib config::tests` を実行し、
  コンパイル失敗（`room_profiles` / `admin_allows_room_profile` 未定義）を確認。

- [ ] Step 3: `server/src/config.rs` を実装する。

  `AdminToolsConfig`（L793 付近）を差し替え:

```rust
#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct AdminToolsConfig {
    /// Room-profile names whose sessions may use the config tools
    /// (`heartbeat_config`, `autonomous_config`, `agent_config`,
    /// `task_test`). A room profile is the unit every other access decision
    /// already uses — the provider it runs on, its memory namespace, the
    /// devices bound to it — so an operator writing this table does not have
    /// to translate a room id into a profile. Empty (the default) leaves the
    /// four tools unregistered.
    #[serde(default)]
    pub room_profiles: Vec<String>,
    /// Removed: replaced by `room_profiles`, which names room profiles rather
    /// than rooms. Retained only so `Config::migration_errors` can name the
    /// leftover key and stop startup instead of letting the grant vanish
    /// silently. Never populated by anything else.
    #[serde(default, rename = "rooms", skip_serializing_if = "Vec::is_empty")]
    pub legacy_rooms: Vec<String>,
}
```

  `config_tools_allowed_in` / `config_tools_enabled`（L1683-1694）を差し替え:

```rust
    /// True when `name` is one of the room profiles declared in
    /// `[tools.admin].room_profiles`.
    ///
    /// The name is a `[room_profile.<n>]` key, resolved from the calling
    /// transport rather than typed by the caller: a channel room through
    /// [`Self::room_profile_name_for_room`], an ACP session through the
    /// profile its bearer token pinned. The transports that name no profile —
    /// voice, `/rpc`, A2A, the unattended loops — never reach this with a
    /// value, which is the point: the endpoint a client reaches first, before
    /// any profile is pinned, must not be the one that can author unattended
    /// work.
    pub fn admin_allows_room_profile(&self, name: &str) -> bool {
        self.tools
            .admin
            .room_profiles
            .iter()
            .any(|allowed| allowed == name)
    }

    /// The room profile a channel `room_id` runs under, by name.
    ///
    /// The same resolution [`Self::profile_for`] and
    /// [`Self::namespace_for_room`] use — an explicit listing in
    /// `[room_profile.<n>].rooms` wins, then `[room_profile.default]` catches
    /// every unmatched room, then the implicit `"default"` applies — so a
    /// room's admin grant cannot disagree with the provider it runs on.
    pub fn room_profile_name_for_room(&self, room_id: &str) -> &str {
        self.room_profile_for(room_id)
            .map(|(name, _)| name)
            .unwrap_or(DEFAULT_PROFILE_NAME)
    }

    /// True when `[tools.admin].room_profiles` names at least one room
    /// profile, i.e. the config-file management tools are registered at all.
    pub fn config_tools_enabled(&self) -> bool {
        !self.tools.admin.room_profiles.is_empty()
    }
```

  `validate_profiles()` の room 重複チェックの後（L1538 付近、`for room in &rp.rooms` ループが
  閉じた後）に追記:

```rust
        // The admin allow list names room profiles. A typo there would grant
        // nothing and say nothing — the same hour-costing shape
        // `validate_subagent_profiles` exists to close.
        for name in &self.tools.admin.room_profiles {
            if name != DEFAULT_PROFILE_NAME && !self.room_profiles.contains_key(name) {
                errors.push(format!(
                    "[tools.admin].room_profiles references unknown room_profile '{name}'"
                ));
            }
        }
```

  `migration_errors()`（L1635）の先頭、`let mut errors = Vec::new();` の直後に追記:

```rust
        if !self.tools.admin.legacy_rooms.is_empty() {
            errors.push(
                "[tools.admin].rooms was replaced by `[tools.admin].room_profiles`, which names \
                 room profiles instead of room ids. Delete the `rooms` line and write the room \
                 profile names you mean, e.g. `room_profiles = [\"ops\"]` — the profile whose \
                 `[room_profile.<name>]` block the room is listed under in that block's `rooms` \
                 array. Leaving `rooms` in place would drop the grant without saying so, so \
                 startup refuses until it is gone."
                    .to_string(),
            );
        }
```

- [ ] Step 4: `server/src/tools/config_tools.rs` の既存 gate を新 API へ機械的に追従させ、
  ビルドを緑に保つ（ACP 経路は Task 2）:
  - `ConfigTool::gate()`（L311）:

```rust
    fn gate(&self) -> Result<()> {
        let allowed = current_call_room()
            .map(|room| self.config.room_profile_name_for_room(&room).to_string())
            .is_some_and(|name| self.config.admin_allows_room_profile(&name));
        if allowed {
            Ok(())
        } else {
            Err(anyhow!(ROOM_REFUSAL))
        }
    }
```

  - `TaskTestTool::execute`（L873）:

```rust
        let allowed = current_call_room()
            .map(|room| self.state.config.room_profile_name_for_room(&room).to_string())
            .is_some_and(|name| self.state.config.admin_allows_room_profile(&name));
        if !allowed {
            anyhow::bail!(ROOM_REFUSAL);
        }
```

  - fixture（L991）を room_profile ベースに変更:

```rust
    /// The tool as the deployment wires it: `[tools.admin].room_profiles =
    /// ["ops"]`, `[room_profile.ops]` claiming `!ops:x`, the three
    /// directories present, no subagent / tool set back-reference (that is
    /// the agents-directory wiring, tested where it exists).
    fn fixture(dir: ConfigDir) -> Fixture {
        let (d, root, ws) = test_workspace();
        let mut config = Config::for_test();
        config.tools.admin.room_profiles = vec!["ops".to_string()];
        config.room_profiles.insert(
            "ops".to_string(),
            crate::config::RoomProfileConfig {
                profile: "dev".to_string(),
                rooms: vec!["!ops:x".to_string()],
                ..Default::default()
            },
        );
        let tool = ConfigTool::new(dir, root.clone(), config, ws, None, Weak::new());
        Fixture {
            _dir: d,
            root,
            tool,
        }
    }
```

  - `test_state`（L1240）:

```rust
    /// A `ServeState` whose scripted provider answers with `responses`, with
    /// the `ops` room profile allow-listed so the tool's gate is passable.
    fn test_state(responses: Vec<ChatResponse>) -> Arc<ServeState> {
        let mut state = ServeState::for_test_scripted(false, responses);
        {
            let config = &mut Arc::get_mut(&mut state)
                .expect("uniquely owned immediately after construction")
                .config;
            config.tools.admin.room_profiles = vec!["ops".to_string()];
            config.room_profiles.insert(
                "ops".to_string(),
                crate::config::RoomProfileConfig {
                    profile: "dev".to_string(),
                    rooms: vec!["!ops:x".to_string()],
                    ..Default::default()
                },
            );
        }
        state
    }
```

  - `no_room_means_no_registration`（L1416）と `one_room_registers_all_four`（L1439）と
    `a_written_agent_definition_is_offered_and_callable`（L1489）の
    `config.tools.admin.rooms = ...` を `config.tools.admin.room_profiles = ...` に改名し、
    `room_profiles` に `ops` を入れる（前者は `Vec::new()`）。

- [ ] Step 5: `cargo test -p sapphire-agent-server --lib config::tests` と
  `cargo test -p sapphire-agent-server --lib tools::config_tools::tests` を実行し全緑を確認。
  期待出力: 各フィルタで `test result: ok.` と失敗 0。

- [ ] Step 6: `cargo fmt --all` の後コミット:

```sh
git add server/src/config.rs server/src/tools/config_tools.rs
git commit -m "feat(config): replace [tools.admin].rooms with room_profiles"
```

---

## Task 2: gate を room_profile 名に一本化し、ACP の task-local 経路を追加する

**Files:**
- Modify: `server/src/tools/config_tools.rs`（モジュール doc L1-25、`current_call_room` L132、
  `ROOM_REFUSAL` L146、`spec()` 説明文 L237-240・L649、`ConfigTool::gate` L311、
  `TaskTestTool::namespace` L693、`TaskTestTool::execute` L873、tests）

**Interfaces:**
- Consumes: Task 1 の `Config::admin_allows_room_profile` / `room_profile_name_for_room`。
- Produces: `config_tools::scope_admin_room_profile(Option<String>, fut)`、
  `config_tools::current_admin_room_profile() -> Option<String>`、
  `config_tools::current_admin_room_profile_for(&Config) -> Option<String>`。
  Task 3 の `TurnLoop::run` が `scope_admin_room_profile` を、
  テストの probe が `current_admin_room_profile` を呼ぶ。

- [ ] Step 1: 失敗するテストを `server/src/tools/config_tools.rs` の `mod tests` に追加。
  `in_room`（L1006）の直後にヘルパを 1 つ足す:

```rust
    /// Every call goes through the ACP path's scope, the way an editor's
    /// prompt does: `run_llm_turn` scopes the profile the session pinned
    /// around tool execution, and no `TimerOrigin` is set on that path.
    async fn from_room_profile<F: Future>(profile: Option<&str>, fut: F) -> F::Output {
        scope_admin_room_profile(profile.map(str::to_string), fut).await
    }
```

  テストを追加（`every_action_is_refused_outside_an_allowed_room` の隣）:

```rust
    /// The ACP path: an editor's session carries the room profile its bearer
    /// token pinned, with no room id anywhere. Listing that profile is what
    /// lets `agent_config` manage subagent definitions from Zed — the whole
    /// reason this grant is no longer written in room ids.
    #[tokio::test]
    async fn an_acp_session_is_allowed_by_its_pinned_room_profile() {
        let f = fixture(ConfigDir::Agents);
        let listing = from_room_profile(
            Some("ops"),
            f.tool.execute(&json!({"action": "list"})),
        )
        .await
        .unwrap();
        assert!(listing.contains("No agents definitions."), "{listing}");
    }

    /// Every other profile-shaped input refuses: a profile the operator did
    /// not list, and a turn with no profile at all (`/rpc`, A2A, the
    /// unattended loops).
    #[tokio::test]
    async fn an_acp_session_off_the_list_is_refused() {
        let f = fixture(ConfigDir::Agents);
        for profile in [Some("guest"), None] {
            let err = from_room_profile(profile, f.tool.execute(&json!({"action": "list"})))
                .await
                .unwrap_err()
                .to_string();
            assert!(err.contains("Permission denied"), "{profile:?}: {err}");
            assert!(err.contains("room_profiles"), "{profile:?}: {err}");
        }
    }

    /// A channel turn resolves its room to a profile the same way its
    /// provider does, so `[room_profile.default]` captures an unlisted room
    /// there exactly as it does for the provider. A room in a *different*
    /// profile than the allow list is a refusal.
    #[tokio::test]
    async fn a_channel_room_is_gated_by_its_resolved_room_profile() {
        let (d, root, ws) = test_workspace();
        let mut config = Config::for_test();
        config.tools.admin.room_profiles = vec!["ops".to_string()];
        for (name, room) in [("ops", "!ops:x"), ("guest", "!lounge:y")] {
            config.room_profiles.insert(
                name.to_string(),
                crate::config::RoomProfileConfig {
                    profile: "dev".to_string(),
                    rooms: vec![room.to_string()],
                    ..Default::default()
                },
            );
        }
        let tool = ConfigTool::new(ConfigDir::Agents, root, config, ws, None, Weak::new());

        in_room("!ops:x", tool.execute(&json!({"action": "list"})))
            .await
            .unwrap();
        let err = in_room("!lounge:y", tool.execute(&json!({"action": "list"})))
            .await
            .unwrap_err()
            .to_string();
        assert!(err.contains("Permission denied"), "{err}");
        let _dir = d;
    }

    /// Voice is a device, not a room or an editor. It carries no profile a
    /// grant could name, and an `ops` allow list must not change that.
    #[tokio::test]
    async fn a_voice_turn_is_refused_even_when_a_profile_is_listed() {
        let f = fixture(ConfigDir::Agents);
        let err = scope_timer_origin(
            TimerOrigin::Voice {
                device_id: "speaker".to_string(),
            },
            f.tool.execute(&json!({"action": "list"})),
        )
        .await
        .unwrap_err()
        .to_string();
        assert!(err.contains("Permission denied"), "{err}");
    }
```

  加えて、テストセッションの namespace が profile 由来になることを固定（`task_test` 群の隣）:

```rust
    /// The test session lands under the calling room profile's memory
    /// namespace, resolved from the profile rather than from a room id — so
    /// an ACP call writes its report where the operator already reads.
    #[tokio::test]
    async fn a_test_session_uses_the_room_profiles_namespace() {
        let mut state = ServeState::for_test_scripted(false, vec![test_response("ok")]);
        {
            let config = &mut Arc::get_mut(&mut state)
                .expect("uniquely owned immediately after construction")
                .config;
            config.tools.admin.room_profiles = vec!["ops".to_string()];
            config.room_profiles.insert(
                "ops".to_string(),
                crate::config::RoomProfileConfig {
                    profile: "dev".to_string(),
                    rooms: vec!["!ops:x".to_string()],
                    memory_namespace: Some("ops_ns".to_string()),
                    ..Default::default()
                },
            );
        }
        let tool = TaskTestTool::new(Arc::clone(&state));
        from_room_profile(Some("ops"), tool.execute(&json!({"kind": "heartbeat", "name": "x"})))
            .await
            .unwrap();
        // The session file, not only the report line: the namespace is what
        // routes it.
        let rows = state.autonomous_session_store.session_rows();
        assert_eq!(rows.len(), 1, "{rows:?}");
        assert_eq!(rows[0].meta.namespace, "ops_ns");
    }
```

  注: 上記の `memory_namespace` と `session_rows()[0].meta.namespace` のフィールド名は
  実装時に `RoomProfileConfig::memory_namespace` と `SessionMeta` の対応フィールド
  （`server/src/session_store.rs` の `session_rows` が返す型）を確認して合わせること。
  名前が違う場合は assert を該当フィールドに読み替える（検証対象は
  「namespace が `ops` profile のものになっていること」であって、フィールド名ではない）。
  また `session_rows()` の戻り値に namespace が無い場合は、
  `autonomous_session_store.absolute_path_for(session_id)` が
  `sessions/ops_ns/autonomous/...` を含むことの assert に置き換える。

- [ ] Step 2: `cargo test -p sapphire-agent-server --lib tools::config_tools::tests` を実行し、
  コンパイル失敗（`scope_admin_room_profile` 未定義）を確認。

- [ ] Step 3: `server/src/tools/config_tools.rs` を実装する。

  (a) モジュール doc（L1-25）の該当行を差し替え:

```rust
//! - **A room profile must be named.** `[tools.admin].room_profiles` is a
//!   host-layer list of room-profile names, and every action refuses in a
//!   session whose profile is not on it. A channel room resolves to its
//!   profile the same way its provider does; an ACP session carries the
//!   profile its bearer token pinned at `session/new` (or `load`/`resume`),
//!   which is what lets an editor manage definitions at all — ACP has no
//!   room id. The transports that name no profile (`/rpc`, A2A, voice, the
//!   unattended loops) get `None`, which is a refusal.
```

  (b) `current_call_room()`（L113-136）を削除し、同じ位置に task-local と 3 つの関数を置く:

```rust
tokio::task_local! {
    /// The room profile an ACP turn runs under, scoped around tool execution
    /// by `TurnLoop::run` from the profile pinned at `session/new` (or
    /// `load`/`resume`). `None` on every other serve-path turn.
    static ADMIN_ROOM_PROFILE_TL: Option<String>;
}

/// Run `fut` with the turn's room profile reachable from
/// [`current_admin_room_profile`]. Called once per round by
/// `TurnLoop::run`, wrapping the same tool execution the timer and ACP-client
/// scopes wrap.
pub(crate) fn scope_admin_room_profile<F: std::future::Future>(
    profile: Option<String>,
    fut: F,
) -> impl std::future::Future<Output = F::Output> {
    ADMIN_ROOM_PROFILE_TL.scope(profile, fut)
}

/// The room profile scoped around the tool call currently executing, if the
/// turn came in on a transport that pins one — today only ACP.
///
/// `None` outside `scope_admin_room_profile` and `None` for a `/rpc`, A2A or
/// unattended turn, which pins no profile. The two are the same answer on
/// purpose: neither is a place an operator declared.
pub(crate) fn current_admin_room_profile() -> Option<String> {
    ADMIN_ROOM_PROFILE_TL.try_with(Clone::clone).ok().flatten()
}

/// The room profile this call's admin grant is judged by, or `None` when the
/// transport names no profile an operator could have allowed.
///
/// A channel turn (`Agent::handle_message`, and the heartbeat's chat leg
/// through it) carries `TimerOrigin::Chat`, so its room resolves to a profile
/// exactly as its provider does — the grant follows the room, which is why a
/// heartbeat fired *into* a listed room can call these tools with nobody
/// present. Voice is a device, not a room or an editor. With no timer origin
/// the turn is on the serve path: an ACP turn scopes the profile its bearer
/// token pinned, while `/rpc`, A2A and the unattended loops scope nothing.
pub(crate) fn current_admin_room_profile_for(config: &Config) -> Option<String> {
    match crate::timer::current_origin() {
        Some(crate::timer::TimerOrigin::Chat { room_id }) => {
            Some(config.room_profile_name_for_room(&room_id).to_string())
        }
        Some(crate::timer::TimerOrigin::Voice { .. }) => None,
        None => current_admin_room_profile(),
    }
}
```

  (c) `ROOM_REFUSAL`（L140-146）:

```rust
/// What every action answers with when the calling room profile is not on
/// `[tools.admin].room_profiles`.
///
/// The wording names the config key on purpose: a model that is refused has
/// no other way to find out, and an operator being told by the agent *which*
/// setting to change is the whole reason the refusal is not just "denied".
pub(crate) const ROOM_REFUSAL: &str = "Permission denied: the config tools are not available for this room profile. An operator can allow them with `[tools.admin].room_profiles` in the host config.";
```

  (d) `ConfigTool::spec()` の説明文（L237-240）の末尾文を差し替え:

```rust
        description.push_str(
            " These tools are available only in sessions whose room profile an operator \
             listed in `[tools.admin].room_profiles`; anywhere else, every action including \
             `list` is refused.",
        );
```

  (e) `TaskTestTool::spec()` の説明文（L649）の末尾文を差し替え:

```rust
                 These tools are available only in sessions whose room profile an operator \
                 listed in `[tools.admin].room_profiles`."
```

  (f) `ConfigTool::gate()` を一本化:

```rust
    /// The profile gate. A refusal here is the only permission check these
    /// tools have — they are `ToolKind::Edit` and deliberately outside
    /// `ToolPolicy`, because the grant that matters is the host's room-profile
    /// list, not a per-tool policy the workspace can set.
    fn gate(&self) -> Result<()> {
        if current_admin_room_profile_for(&self.config)
            .is_some_and(|name| self.config.admin_allows_room_profile(&name))
        {
            Ok(())
        } else {
            Err(anyhow!(ROOM_REFUSAL))
        }
    }
```

  (g) `TaskTestTool::namespace()`（L687-698）:

```rust
    /// The namespace the test session lands in: the calling room profile's, so
    /// the report is readable where the operator already is. `None` only
    /// happens on a path the gate has already refused.
    fn namespace(&self) -> &str {
        match current_admin_room_profile_for(&self.state.config) {
            Some(name) => self.state.config.namespace_for_room_profile(&name),
            None => DEFAULT_NAMESPACE_NAME,
        }
    }
```

  (h) `TaskTestTool::execute`（L866-877）の冒頭 gate:

```rust
        // Before the arguments are even read: the room-profile gate every
        // config tool shares. A run is the most powerful thing on this
        // surface — it drives a whole turn — so it is refused as firmly as a
        // write.
        if !current_admin_room_profile_for(&self.state.config)
            .is_some_and(|name| self.state.config.admin_allows_room_profile(&name))
        {
            anyhow::bail!(ROOM_REFUSAL);
        }
```

  (i) tests の import に `scope_admin_room_profile` を加える（`use super::*;` が既にあるので
  不要なはずだが、`Future` は既に import 済み。変更なければそのまま）。

- [ ] Step 4: `cargo test -p sapphire-agent-server --lib` を実行し、
  `tools::config_tools::tests` が全緑であることを確認。期待出力:
  `test result: ok.` と、`an_acp_session_is_allowed_by_its_pinned_room_profile` を含む
  新規 5 テストが `ok` で並ぶ。

- [ ] Step 5: `cargo fmt --all` の後コミット:

```sh
git add server/src/tools/config_tools.rs
git commit -m "feat(tools): gate the admin tools on the calling session's room profile"
```

---

## Task 3: ACP セッションの profile をツール実行まで運ぶ

**Files:**
- Modify: `server/src/serve/mod.rs`（`TurnContext` L2349、`TurnLoop` L2414、
  production `TurnContext` 構築 L2714、`run_llm_turn` L2876、production `TurnLoop` 構築 L3053、
  tests の `TurnContext` リテラルがある場合は該当箇所）
- Modify: `server/src/tools/subagent.rs`（nested `TurnLoop` L806、
  テスト用 `TurnContext` L1779 と L1828）
- Modify: `server/src/tools/skill_tools.rs`（テスト用 `TurnContext` L1860）
- Modify: `server/src/tools/config_tools.rs`（テスト用 `TurnContext` L1552）

**Interfaces:**
- Consumes: Task 2 の `config_tools::scope_admin_room_profile`。
- Produces: `TurnContext.admin_room_profile: Option<String>`、
  `TurnLoop.admin_room_profile: Option<String>`。Task 4 の probe がこの伝達の
  end-to-end を固定する。

- [ ] Step 1: `server/src/serve/mod.rs` の `TurnContext` にフィールドを追加する。
  `pub subagent_depth: u32,` の直後:

```rust
    /// The room profile this turn's admin-tool grant hangs on, or `None` on a
    /// transport that names no profile an operator could have allowed
    /// (`/rpc`, A2A, voice, the unattended loops). Set by `run_llm_turn` from
    /// the profile an ACP session pinned, and read through
    /// `config_tools::current_admin_room_profile_for` while a tool call runs.
    ///
    /// Copied straight into the nested `TurnLoop` a delegated subagent builds,
    /// so a delegation keeps the grant its parent ran under — the grant is the
    /// session's, not the caller's, the same way `timer_origin` is carried.
    /// A definition that must not hold it narrows its own `tools:` instead.
    pub admin_room_profile: Option<String>,
```

- [ ] Step 2: `TurnLoop` にも同名フィールドを追加（`pub subagent_depth: u32,` の直後）:

```rust
    /// The admin-tool grant this loop's tool calls are judged by. Copied into
    /// every round's `TurnContext` and scoped around tool execution the same
    /// way `timer_origin` is. See `TurnContext::admin_room_profile`.
    pub admin_room_profile: Option<String>,
```

- [ ] Step 3: `run_llm_turn`（L2876）で値を決める。`let is_acp = state.is_acp(&session_id).await;`
  の直後:

```rust
    // The room profile this turn's admin-tool grant hangs on. An ACP session
    // carries the one its bearer token pinned at `session/new` (or
    // `load`/`resume`); `/rpc`, A2A, voice and the unattended loops name no
    // profile an operator declared, so they get `None` and the admin tools
    // refuse there. Channel rooms never reach this function — `Agent::
    // handle_message` runs its own loop, where the room travels as
    // `TimerOrigin::Chat` instead.
    let admin_room_profile = if is_acp {
        state
            .session_room_profiles
            .lock()
            .await
            .get(&session_id)
            .cloned()
    } else {
        None
    };
```

- [ ] Step 4: production `TurnLoop` 構築（L3053）に
  `admin_room_profile: admin_room_profile.clone(),` を追加。

- [ ] Step 5: production `TurnContext` 構築（L2714）に
  `admin_room_profile: self.admin_room_profile.clone(),` を追加。

- [ ] Step 6: `TurnLoop::run` のツール実行を包む。L2746 の
  `let fut = scope_turn_context(turn_ctx, fut);` の**直前**に 1 行追加する
  （`match (origin, client)` の外側。arms を増やさないため）:

```rust
                                // The admin tools read this through
                                // `config_tools::current_admin_room_profile()`.
                                // Scoped here rather than inside the match
                                // below so the four (origin, client) arms do
                                // not each need their own copy; `None` is a
                                // meaningful value (no profile to allow), not
                                // "unset".
                                let fut = crate::tools::config_tools::scope_admin_room_profile(
                                    admin_room_profile.clone(),
                                    fut,
                                );
```

  同アームの少し上、`let timer_origin = self.timer_origin.clone();`（L2706 付近）の隣に:

```rust
                    let admin_room_profile = self.admin_room_profile.clone();
```

- [ ] Step 7: `server/src/tools/subagent.rs` の nested `TurnLoop`（L806）に
  `admin_room_profile: ctx.admin_room_profile.clone(),` を追加（`timer_origin:` の直後）。

- [ ] Step 8: 残るテスト用 `TurnContext` リテラル 4 箇所に
  `admin_room_profile: None,` を追加:
  - `server/src/tools/subagent.rs` L1779（`turn_context_with_host`）と L1828（`parent_ctx`）
  - `server/src/tools/skill_tools.rs` L1860（`test_turn_context`）
  - `server/src/tools/config_tools.rs` L1552（`after_write` テスト）
  それぞれ `subagent_depth: 0,` の直後に置く。

- [ ] Step 9: `cargo test -p sapphire-agent-server --lib` を実行。
  期待出力: `test result: ok.` のみ（E0063 missing field の解消確認）。
  続けて `cargo clippy -p sapphire-agent-server --all-targets` が警告 0 で終わることを確認。

- [ ] Step 10: `cargo fmt --all` の後コミット:

```sh
git add server/src/serve/mod.rs server/src/tools/subagent.rs server/src/tools/skill_tools.rs server/src/tools/config_tools.rs
git commit -m "feat(serve): carry a turn's pinned room profile to its tool calls"
```

---

## Task 4: ACP 経路を end-to-end で固定する

**Files:**
- Modify: `server/src/serve/mod.rs`（`#[cfg(test)] pub(crate) struct AdminProfileProbe` を
  `StubProvider` の隣 L3450 付近に追加、tests に非 ACP の対照テスト）
- Modify: `server/src/serve/acp.rs`（tests に ACP の probe テスト）

**Interfaces:**
- Consumes: Task 3 の `TurnContext`/`TurnLoop` 伝達、Task 2 の
  `config_tools::current_admin_room_profile()`。
- Produces: `serve::AdminProfileProbe`（`#[cfg(test)] pub(crate)`）と
  `AdminProfileProbe::new() -> Arc<Self>` / `seen() -> Option<Option<String>>`。

- [ ] Step 1: `server/src/serve/mod.rs` の `StubProvider` 定義の隣に probe を追加:

```rust
/// A `ToolKind::Read` tool registered under a name the ACP turn's visibility
/// predicate passes through, whose only job is to record which room profile
/// the surrounding tool execution was scoped with.
///
/// The task-local is only observable from inside a tool, which is exactly
/// where the gate reads it, so a probe is the honest instrument: a test that
/// asserted on the scope directly would pin the plumbing rather than the
/// behaviour. Registered by name (`heartbeat_config`) that
/// `visible_tool_predicate` neither gates on `host_access` nor on a client
/// capability, so it is offered on both the ACP and the non-ACP path.
#[cfg(test)]
pub(crate) struct AdminProfileProbe {
    seen: std::sync::Mutex<Option<Option<String>>>,
    spec: crate::provider::ToolSpec,
}

#[cfg(test)]
impl AdminProfileProbe {
    pub(crate) fn new() -> Arc<Self> {
        Arc::new(Self {
            seen: std::sync::Mutex::new(None),
            spec: crate::provider::ToolSpec {
                name: "heartbeat_config".into(),
                description: "Records the room profile this turn scoped.".into(),
                input_schema: serde_json::json!({"type": "object", "properties": {}}),
            },
        })
    }

    /// `Some(value)` once the tool ran, `None` before it did.
    pub(crate) fn seen(&self) -> Option<Option<String>> {
        self.seen.lock().unwrap().clone()
    }
}

#[cfg(test)]
#[async_trait::async_trait]
impl crate::tools::Tool for AdminProfileProbe {
    fn spec(&self) -> &crate::provider::ToolSpec {
        &self.spec
    }

    fn kind(&self) -> crate::tools::ToolKind {
        crate::tools::ToolKind::Read
    }

    async fn execute(&self, _input: &serde_json::Value) -> anyhow::Result<String> {
        *self.seen.lock().unwrap() =
            Some(crate::tools::config_tools::current_admin_room_profile());
        Ok("probed".to_string())
    }
}
```

- [ ] Step 2: 失敗するテストを `server/src/serve/acp.rs` の `mod tests` に追加
  （`drive` ヘルパが使える位置）:

```rust
    /// The grant this feature exists for, pinned end to end: an editor's
    /// `session/prompt` turn scopes the room profile its bearer token pinned,
    /// so `agent_config` — which ACP has no room id to allow — can be judged
    /// against `[tools.admin].room_profiles`. The probe stands in for one of
    /// the admin tools: it is offered and judged on the same path.
    #[tokio::test]
    async fn an_acp_turn_scopes_its_pinned_room_profile() {
        let state = ServeState::for_test_scripted(
            true,
            vec![
                crate::provider::ChatResponse {
                    prompt_usage: None,
                    text: None,
                    tool_calls: vec![crate::provider::ToolCall {
                        id: "call-1".to_string(),
                        name: "heartbeat_config".to_string(),
                        input: serde_json::json!({"action": "list"}),
                    }],
                    stop_reason: None,
                },
                crate::provider::ChatResponse {
                    prompt_usage: None,
                    text: Some("done".to_string()),
                    tool_calls: Vec::new(),
                    stop_reason: None,
                },
            ],
        );
        let probe = crate::serve::AdminProfileProbe::new();
        state.tools.register_tool(Box::new(Arc::clone(&probe))).await;

        let addr = spawn(Arc::clone(&state)).await;
        let (_, updates, reply) = drive(&addr, text_prompt("manage the definitions")).await;
        assert!(
            reply["error"].is_null(),
            "the turn must not fail: {reply}"
        );
        assert_eq!(
            probe.seen(),
            Some(Some("developer".to_string())),
            "the fixture's `sa-acp-token` resolves to room profile 'developer', so that is \
             the name the admin gate must see; updates: {updates:?}"
        );
    }
```

  注: `drive` は `initialize_request(0)` と `new_session_request(1)` を同一接続で送る
  ヘルパで、bearer token は `sa-acp-token`（fixture）。`text_prompt` は同モジュールに既存。
  名前や引数が実装と食い違う場合は既存の `session_new_returns_a_session_id` と
  `drive` の呼び出し例に合わせること。

- [ ] Step 3: `server/src/serve/mod.rs` の tests に非 ACP の対照を追加
  （`a_refused_tool_returns_a_result_and_the_turn_continues` L3855 の隣）:

```rust
    /// The control case for the ACP test above: `/rpc` and A2A call
    /// `run_llm_turn` with no ACP session behind them, so no profile is
    /// scoped and the admin tools refuse. Asserting `None` — not merely "not
    /// 'developer'" — keeps the two paths distinguishable.
    #[tokio::test]
    async fn a_non_acp_turn_scopes_no_room_profile() {
        let state = ServeState::for_test_scripted(
            true,
            vec![
                crate::provider::ChatResponse {
                    prompt_usage: None,
                    text: None,
                    tool_calls: vec![crate::provider::ToolCall {
                        id: "call-1".to_string(),
                        name: "heartbeat_config".to_string(),
                        input: json!({"action": "list"}),
                    }],
                    stop_reason: None,
                },
                crate::provider::ChatResponse {
                    prompt_usage: None,
                    text: Some("done".to_string()),
                    tool_calls: Vec::new(),
                    stop_reason: None,
                },
            ],
        );
        let probe = AdminProfileProbe::new();
        state.tools.register_tool(Box::new(Arc::clone(&probe))).await;

        run_llm_turn(
            Arc::clone(&state),
            "s-plain".to_string(),
            ChatMessage::user("go"),
            Arc::new(NullProgress),
            None,
        )
        .await;

        assert_eq!(probe.seen(), Some(None), "no profile may be scoped here");
    }
```

- [ ] Step 4: `cargo test -p sapphire-agent-server --lib serve::` を実行。
  期待出力: 新規 2 テストを含め `test result: ok.`。ACP テストが
  `Some(Some("developer"))` を観測すること（失敗時は
  `session_room_profiles` の pin が `session/prompt` より先に行われているかを確認）。

- [ ] Step 5: `cargo fmt --all` の後コミット:

```sh
git add server/src/serve/mod.rs server/src/serve/acp.rs
git commit -m "test(serve): pin the admin grant's ACP room-profile path end to end"
```

---

## Task 5: ドキュメントとコメントの更新

**Files:**
- Modify: `README.md`（管理ツール節 L840-890）
- Modify: `README.ja.md`（L94）
- Modify: `server/config.example.toml`（L362-442）
- Modify: `server/templates/workspace/config.toml`（L44-48）
- Modify: `server/src/main.rs`（コメント L515-535、L925-935）

**Interfaces:**
- Consumes: なし（文書のみ）。挙動は Task 1〜4 で確定済み。

- [ ] Step 1: `README.md` の管理ツール節の
  「**A room has to be named, in the host config.**」段落（L868-890 付近）を差し替える:

```markdown
**A room profile has to be named, in the host config.** The tools are registered
only when `[tools.admin].room_profiles` lists at least one room profile, and every
action — including the read-only `list` — is refused in a session whose profile is
not on it:

```toml
# host-local config, never the workspace's
[tools.admin]
room_profiles = ["ops", "developer"]   # [room_profile.<name>] keys, not room ids
```

With `room_profiles` absent or empty the four tools are **not registered at all** —
the model is never told they exist. A tool that is offered and then always refuses
is the worse answer: it invites a retry loop against a wall.

A **channel** turn is judged by the profile its room resolves to — the same
resolution that picks its provider and memory namespace, so an explicit listing in a
`[room_profile.<name>].rooms` array wins and `[room_profile.default]` catches every
room no profile claims. An **ACP** session (Zed and other editors) is judged by the
profile its bearer token pinned at `session/new`, which is what makes these tools
usable from an editor at all: ACP has no room id. Note what `room_profiles =
["default"]` therefore means — every room no other profile claims — and list the
specific profiles you mean unless that is what you want.

Voice, `/rpc`, A2A and the unattended loops (heartbeat, autonomous) name no profile
an operator declared, so the tools are refused there even when the list is not empty.
A heartbeat or autonomous task that fires *into* a room of a listed profile is the
exception, and the reason a listed profile's rooms should not also be a task's
delivery target: the grant follows the room, not the caller, so such a turn can
author unattended work with nobody watching.

`[tools]` is deliberately outside the workspace layer's allowlist, so a synced
workspace `config.toml` cannot grant itself this — a `[tools.admin]` written there is
dropped and named in a startup warning, the same as any other key that layer may not
set. Who may run these tools is the same kind of host-local decision as the API key or
the bind address, and it has to be: a listed profile can rewrite, through the agent,
what runs unattended. **Pair it with that profile's own `[room_profile.<name>].devices`
or the room's `allowed_users`**, or everyone who can reach a session of that profile
can author code-adjacent work that runs with nobody watching. `[tools.admin].rooms`
from an earlier build is rejected at startup and names `room_profiles` in the error.
```

  あわせて同節のツール列挙・`set_enabled` の説明中に `rooms` への言及が残っていれば
  `room_profiles` に直す（`grep -n "admin" README.md` で確認）。

- [ ] Step 2: `README.ja.md` L94 を差し替え:

```markdown
- エージェント自身の定義編集 — heartbeat / autonomous / agents の定義ファイルを読み書きする4ツールと `[tools.admin].room_profiles` の room_profile 許可リスト（ACP セッションからも使えます）
```

- [ ] Step 3: `server/config.example.toml` の
  「`[tools.admin].rooms` — the rooms whose calls may use ...」の説明（L362-378）と
  例（L436-440）を差し替える:

```toml
#   - `[tools.admin].room_profiles` — the room profiles whose sessions may
#     use the config-file management tools (`heartbeat`, `autonomous`,
#     `agents`), which add and edit the definitions under the workspace
#     that the unattended loops later run. Values are `[room_profile.<name>]`
#     keys, not room ids: a channel turn is judged by the profile its room
#     resolves to, an ACP session by the profile its bearer token pinned, and
#     voice / `/rpc` / A2A / the unattended loops by nothing (refused). Empty
#     (the default) leaves all of them unregistered, the same shape as
#     `[tools.host_access]`. Host-layer only: `[tools]` is outside the
#     workspace-layer allowlist, so a synced workspace config cannot grant
#     itself this.
#
#     `room_profiles = ["default"]` is the broad one: it names the implicit
#     profile that every room no other profile claims falls through to, so
#     list the profiles you actually mean. Anyone who can reach a session of
#     a listed profile can rewrite, through the agent, what runs unattended
#     — pair this with the profile's `devices` or the room's `allowed_users`.
```

```toml
# # Example: let the `ops` and `developer` room profiles manage the
# # heartbeat/autonomous/agents definitions. Empty by default (all four
# # tools unregistered).
# [tools.admin]
# room_profiles = ["ops", "developer"]
```

- [ ] Step 4: `server/templates/workspace/config.toml` L44-48 の
  `[tools.admin]` への言及はキー名を含まないので原則そのままだが、
  「`[tools.admin]` belongs in the host config」の文はそのまま維持し、
  同ファイル内で `rooms` を名指ししている箇所があれば `room_profiles` に直す
  （`grep -n "admin" server/templates/workspace/config.toml` で確認）。

- [ ] Step 5: `server/src/main.rs` のコメント 3 箇所（L515-535 の subagent 登録理由、
  L925-935 の `register_admin_tools` 呼び出し箇所）の
  `` `[tools.admin].rooms` `` を `` `[tools.admin].room_profiles` `` に置換し、
  「set」の意味が「1 つ以上の profile が列挙されている」であることが読み取れるよう
  `if !no_agent_defs || config.config_tools_enabled()` の説明を微修正する:

```rust
            // Registered when there is something to delegate to, and also
            // when `[tools.admin].room_profiles` names a profile: `agent_config`
            // can create a definition while the process is running. `Arc` rather
            // than a bare `Box` because `agent_config` holds a `Weak` to it —
            // same shape `SkillTool` uses.
```

- [ ] Step 6: `grep -rn "tools\.admin\]\.rooms\|tools\.admin\].rooms\|admin\.rooms" server/src README.md README.ja.md server/config.example.toml server/templates` の出力が
  `config.rs` の `legacy_rooms` 関連（`rename = "rooms"` と migration メッセージ）だけに
  なることを確認。残りは旧キーを名指す意図的な記述のみ許容する。

- [ ] Step 7: `cargo test -p sapphire-agent-server --lib` を再実行（コメントのみの変更なので
  緑のままであることの確認）。期待出力: `test result: ok.`。

- [ ] Step 8: 仕様書と計画書を追跡対象へ移し、コミット:

```sh
mkdir -p docs/superpowers/specs docs/superpowers/plans
cp .superpowers/sdd/2026-09-17-admin-tools-room-profile/2026-09-17-admin-tools-room-profile-design.md \
   docs/superpowers/specs/2026-09-17-admin-tools-room-profile-design.md
cp .superpowers/sdd/2026-09-17-admin-tools-room-profile/2026-09-17-admin-tools-room-profile-plan.md \
   docs/superpowers/plans/2026-09-17-admin-tools-room-profile-plan.md
git add server/src/main.rs server/config.example.toml server/templates/workspace/config.toml README.md README.ja.md docs/superpowers
git commit -m "docs: document [tools.admin].room_profiles"
```

---

## 検証（全タスク完了後）

- [ ] `cargo fmt --all --check` が差分なしで終わる。
- [ ] `cargo clippy -p sapphire-agent-server --all-targets` が警告 0。
- [ ] `cargo test -p sapphire-agent-server` が全緑。
- [ ] 手動確認（任意、実 config で）:
  1. `room_profiles = ["ops"]` + `[room_profile.ops].rooms = ["!ops:x"]` で起動し、
     `!ops:x` からの `heartbeat_config list` が成功、別 room からは拒否される。
  2. `[tools.admin].rooms = ["!ops:x"]` を残した config で起動が失敗し、
     エラーが `room_profiles` を名指しする。
  3. `room_profiles = ["typo"]` で `sapphire-agent-server verify` が
     `[tools.admin].room_profiles references unknown room_profile 'typo'` を含む報告を出す。
