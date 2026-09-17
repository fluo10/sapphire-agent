# 管理ツールの許可単位を room ID から room_profile 名へ（仕様書）

- **対象コード**: `server/src/config.rs`、`server/src/tools/config_tools.rs`、
  `server/src/serve/mod.rs`、`server/src/serve/acp.rs`（テスト）、
  `server/src/tools/subagent.rs`、`server/src/tools/skill_tools.rs`（テスト）、
  `server/src/main.rs`、`server/config.example.toml`、
  `server/templates/workspace/config.toml`、`README.md`、`README.ja.md`
- **計画書**: `2026-09-17-admin-tools-room-profile-plan.md`（本ディレクトリ）。
  追跡対象のコピーは `docs/superpowers/plans/2026-09-17-admin-tools-room-profile-plan.md`、
  本仕様書のコピーは `docs/superpowers/specs/2026-09-17-admin-tools-room-profile-design.md`
  （`.superpowers/` は `.gitignore` 対象なので、コミットされるのは `docs/` 側）
- **前提**: PR #265 系で入った設定ファイル操作ツールと `[tools.admin].rooms`。
  未リリース（`0.8.0` タグには含まれず、09-14 以降の main のみ）
- **作業ブランチ**: `feat/admin-tools-room-profile`

## 背景と目的

管理ツール（`heartbeat_config` / `autonomous_config` / `agent_config` /
`task_test`）の許可は現在 `[tools.admin].rooms`（room ID の完全一致）で表現されている。
判定は `Config::config_tools_allowed_in(room_id)`、呼び出し元の room は
`TimerOrigin::Chat { room_id }` だけを読む `config_tools::current_call_room()` に固定されている。

しかしサブエージェント定義を実際に管理したいのは **Zed（ACP）のセッションから**である。
ACP は bearer token → device → room_profile で認証され、`session/new` の時点で
`ServeState.session_room_profiles` にセッション→room_profile 名が pin される。ACP には
room ID が存在しないため、現行の room 許可リストでは **エディタからサブエージェント定義を
管理できない**。`/rpc` と voice も room を持たないので同様に常時拒否される。

運用者にとって自然な単位は「どの room_profile か」である。room_profile は既に provider・
memory namespace・voice pipeline・device 束縛の単位であり、「誰がその profile のセッションを
開けるか」は `[room_profile.<n>].devices` と `[profile.<n>]` で決まっている。管理ツールの
許可だけを room ID という別の語彙で書かせる理由が薄い。

**目的**: 許可単位を room_profile 名に置き換え、ACP セッションからも
（そのセッションの profile が列挙されていれば）管理ツールを使えるようにする。

## 決めたこと

### 1. `[tools.admin].rooms` → `[tools.admin].room_profiles`（置換）

```toml
[tools.admin]
# 管理ツール（heartbeat_config / autonomous_config / agent_config /
# task_test）を使ってよい room_profile 名。空（既定）なら4本は登録されない。
room_profiles = ["ops", "developer"]
```

- 値は `[room_profile.<n>]` の**キー名**（room_profile 名）。room ID ではない。
- 旧 `rooms` は受け付けない。残留していれば起動時に**明示的に失敗**させる（決定 9）。
- 併存期間（両キーを読む移行モード）は設けない。`[tools.admin]` 自体が未リリースで、
  二重キーの意味論（和集合か排他か）を決める運用実績が無い。YAGNI。

### 2. 許可判定は「そのターンが走っている room_profile 名」が列挙に含まれるか

| transport | 判定に使う room_profile 名 | 結果 |
|---|---|---|
| チャンネル（Matrix / Discord、heartbeat の chat leg） | `TimerOrigin::Chat { room_id }` を `Config::room_profile_name_for_room` で解決した名前 | 列挙にあれば許可 |
| ACP（Zed 等） | `ServeState.session_room_profiles[session_id]`（`session/new` で pin） | 列挙にあれば許可（**必須要件**） |
| voice | — | 常に拒否（呼び手は device） |
| `/rpc`、A2A | — | 常に拒否（従来どおり） |
| 無人ターン（heartbeat の voice leg、autonomous） | — | 常に拒否（`/rpc` と同じ扱い） |
| サブエージェントのネストターン | 委派元ターンの値 | 継承する（決定 8） |

`room_profile_name_for_room` は `room_profile_for` を名前に落とすだけの薄い関数で、
解決順は 明示一致 → `[room_profile.default]` → 暗黙の `"default"`。したがって
**`room_profiles = ["default"]` は「どの room_profile にも明示的に属さない全 room」を意味する**
（provider / memory namespace の解決と同じ意味論）。強力な書き方になり得るため
README と `config.example.toml` に明記する。

### 3. 伝達は新しい task-local 一本（ACP 経路だけ値が入る）

チャンネル経路は既に `TimerOrigin::Chat` を持っているので、room → profile 名の解決は
gate の中で行う（`config` は両ツールが保持している）。新しい伝達が要るのは ACP だけで、
「このターンはどの room_profile で走っているか」をツール実行時まで運ぶ必要がある。

- `server/src/tools/config_tools.rs` に task-local を新設:
  `static ADMIN_ROOM_PROFILE_TL: Option<String>` と
  `scope_admin_room_profile(Option<String>, fut)` / `current_admin_room_profile()`。
- `TurnLoop` に `admin_room_profile: Option<String>` を追加し、`TurnLoop::run` のツール実行を
  既存の `scope_turn_context` の直前に**1回だけ** scope する（`match (origin, client)` の
  外側。arms が増えないように）。ACP クライアント用の scope も `timer` の scope も
  `TimerOrigin` を ACP 向けに拡張しない。`TimerOrigin` は
  「timer がどこへ発火するか」の型であり、`TimerManager::dispatch_fire` が variant で分岐
  している。カテゴリ違いの同居は事故の元。
- `run_llm_turn` が値を決める。`is_acp` は同関数で既に計算済み（L2886）。
  ACP なら `session_room_profiles` の pin を clone、それ以外は `None`。
- `TurnContext` に同名フィールドを追加し、委派（`subagent.rs` のネスト `TurnLoop`）が
  親の値をそのまま引き継ぐ（`timer_origin` と同じ扱い）。

### 4. 登録条件は「1つ以上の profile が列挙されていること」

`Config::config_tools_enabled()` = `!tools.admin.room_profiles.is_empty()`。
`main.rs` の `register_admin_tools` はこの述語をそのまま使うので変更しない。
「登録されているが常に拒否」より「最初から存在しない」を保つ（現行設計のまま）。

### 5. 起動時検証: 未知の room_profile 名は fatal

`Config::validate_profiles()` に 1 件追加する。`room_profiles` の各要素は `"default"` か、
実際に定義された `[room_profile.<n>]` のキーでなければならない。未知名は typo であり、
放置すると「grant したつもりが効かない」という `validate_subagent_profiles` と同型の
時間を溶かすバグになる。エラー文言に `[tools.admin].room_profiles` を名指しする。
効く経路は 2 つ: `sapphire-agent-server verify` の報告と、serve 起動時の
`ProviderRegistry::from_config` 経由の bail。

### 6. `task_test` の namespace は profile 名から解決する

`TaskTestTool::namespace()` は現在 `current_call_room()` → `config.namespace_for_room(room)`
である。ACP からの呼び出しには room_id が無いため `namespace_for_room_profile(name)` に
置き換える。`namespace_for_room(r) == namespace_for_room_profile(room_profile_name_for_room(r))`
は定義上一致するのでチャンネル経路の挙動は変わらない。

### 7. 拒否文言・ツール説明・ドキュメントを新キーに合わせる

- `config_tools::ROOM_REFUSAL` を `[tools.admin].room_profiles` を名指す文言にする。
- `ConfigTool::spec()` / `TaskTestTool::spec()` の説明文、`config_tools.rs` のモジュール doc を
  `[tools.admin].room_profiles` に更新。
- `README.md` の管理ツール節、`README.ja.md`、`config.example.toml`、
  `server/templates/workspace/config.toml`（該当は L46 の 1 行、キー名自体は含まない）、
  `main.rs` のコメント 3 箇所を更新。
- `config_tools::current_call_room()` は使われなくなるので削除する。`TimerOrigin::Chat` を
  読むという実装詳細が gate から消え、判定は `current_admin_room_profile_for(config)` に
  一本化される。

### 8. サブエージェントは委派元の grant を継承する

管理 profile のターンから委派されたサブエージェントは、その nested ターンでも同じ
room_profile で走っている（`timer_origin` を引き継ぐのと同じ理屈）。継承させる。実体は
`TurnContext.admin_room_profile` を nested `TurnLoop` が複製するだけ。run できるかは
そのターンの `tool_specs` に含まれるか（委派元の可視集合と定義の `tools:`）でも決まる。
grant は「呼び手」ではなく「そのターンが走っている room_profile」に紐づく、という原則の
一貫した帰結。制限したい場合の既存手段は定義の `tools:` 絞り込み。

### 9. 旧 `rooms` キーは起動時に loud に失敗させる

`AdminToolsConfig` に捕捉専用フィールドを残す:

```rust
/// Removed: replaced by `room_profiles`.
#[serde(default, rename = "rooms", skip_serializing_if = "Vec::is_empty")]
pub legacy_rooms: Vec<String>,
```

`Config::migration_errors()` がこれを検出し、「`[tools.admin].rooms` は
`room_profiles`（room_profile 名の配列）に置き換わった。削除して書き直せ」と名指しして
停止させる。あわせて `main.rs` の bail 前置き文
（現在は "uses settings that were removed in the device-registry migration" 固定）を
キー種別に依存しない文言へ一般化する。この前置きを assert するテストは無い。

採用理由: `[tools.admin]` は admin surface の grant であり、黙って読み落とすと
「管理ツールが忽然と消える」という症状が config を指さない失敗になる。同ファイルに
前例（`migration_errors`）がある。未知キー警告（`serde_ignored`）でも方向は fail-closed だが、
名前を伏せたまま消えるため不十分と判断した。

## 要件

### 機能要件

1. R1: host config の `[tools.admin].room_profiles`（`Vec<String>`、room_profile 名）で
   許可対象を指定できる。空（既定）なら 4 ツールは登録されない。
2. R2: チャンネルからの呼び出しは、その room が属する room_profile 名が列挙に含まれるときだけ
   許可される。属さない room は拒否される。
3. R3: ACP セッションからの呼び出しは、そのセッションに pin された room_profile 名が
   列挙に含まれるときだけ許可される（**本改修の主目的**）。
4. R4: voice・`/rpc`・A2A・無人からは、列挙に何が書かれていても拒否される。
5. R5: 拒否文言は `[tools.admin].room_profiles` を名指しする。
6. R6: 旧 `rooms` キーが残っていれば起動が失敗し、`room_profiles` を案内する。
7. R7: `room_profiles` の未知の profile 名は `verify` で報告され、serve 起動を失敗させる。
8. R8: `task_test` のテストセッションは、呼び出し元 room_profile の memory namespace に置かれる
   （ACP 経路でも）。
9. R9: 登録条件・実行時ゲートの二重許可・定義 stem 検証・`set_enabled` の行編集・
   `ToolKind::Edit` 割り当て・`[tools]` が workspace layer 外であること、といった既存の
   不変事項は維持される。

### 非機能要件

- N1: コード・コメント・コミットメッセージは英語。`docs/superpowers/**` のドキュメントは日本語。
- N2: ACP の許可判定は既存の `session_room_profiles` を唯一の情報源とし、新しいセッション
  対応表を増やさない。ACP か否かの判定は既存の `ServeState::is_acp` だけが担い、
  値を受け取る側で再判定しない。
- N3: 新キーの意味論は provider / namespace の room_profile 解決と一致させる。
- N4: 既存テスト様式を踏襲する（`Config::parse_for_test` / `for_test`、
  `ServeState::for_test_scripted*` + `StubProvider`、`scope_timer_origin` /
  `scope_turn_context`、ACP の `spawn` / `drive` ヘルパ、probe ツールは
  `RiskyTool` と同じ「インスタンス毎の記録 + handle」パターン）。
- N5: `cargo fmt --all`、`cargo clippy -p sapphire-agent-server --all-targets`（警告 0）、
  `cargo test -p sapphire-agent-server` が通ること。

## 設計

### 設定型（`server/src/config.rs`）

```rust
#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct AdminToolsConfig {
    /// Room-profile names whose sessions may use the config tools.
    #[serde(default)]
    pub room_profiles: Vec<String>,
    /// Removed: replaced by `room_profiles`. Captured only so
    /// `Config::migration_errors` can name the leftover key.
    #[serde(default, rename = "rooms", skip_serializing_if = "Vec::is_empty")]
    pub legacy_rooms: Vec<String>,
}

impl Config {
    /// True when `name` is one of the room profiles declared in
    /// `[tools.admin].room_profiles`.
    pub fn admin_allows_room_profile(&self, name: &str) -> bool {
        self.tools.admin.room_profiles.iter().any(|allowed| allowed == name)
    }

    /// The room profile a channel `room_id` runs under, by name. The same
    /// resolution `profile_for` / `namespace_for_room` use.
    pub fn room_profile_name_for_room(&self, room_id: &str) -> &str {
        self.room_profile_for(room_id)
            .map(|(name, _)| name)
            .unwrap_or(DEFAULT_PROFILE_NAME)
    }

    /// True when `[tools.admin].room_profiles` names at least one profile.
    pub fn config_tools_enabled(&self) -> bool {
        !self.tools.admin.room_profiles.is_empty()
    }
}
```

`config_tools_allowed_in(Option<&str>)` は削除する（room ID 完全一致という役目が消え、
名前だけ似た関数を残すと誤用を招く）。`room_profile_for` は `Option` を返すので
`room_profile_name_for_room` が必ず `Some` 側の名前を返す点に注意（暗黙 default を含む）。

`validate_profiles()` に追加（room 重複チェックのループの後）:

```rust
for name in &self.tools.admin.room_profiles {
    if name != DEFAULT_PROFILE_NAME && !self.room_profiles.contains_key(name) {
        errors.push(format!(
            "[tools.admin].room_profiles references unknown room_profile '{name}'"
        ));
    }
}
```

`migration_errors()` の先頭に追加（`legacy_rooms` 非空で 1 件）。`main.rs` の bail 前置きは
`"config at {path} uses settings that were removed:\n\n  - {errors}\n"` に一般化する。

### gate（`server/src/tools/config_tools.rs`）

```rust
tokio::task_local! {
    /// The room profile an ACP turn runs under, scoped around tool execution
    /// by `TurnLoop::run` from the profile pinned at `session/new`. `None` on
    /// every other serve-path turn.
    static ADMIN_ROOM_PROFILE_TL: Option<String>;
}

pub(crate) fn scope_admin_room_profile<F: Future>(profile: Option<String>, fut: F)
    -> impl Future<Output = F::Output> { ADMIN_ROOM_PROFILE_TL.scope(profile, fut) }

pub(crate) fn current_admin_room_profile() -> Option<String> {
    ADMIN_ROOM_PROFILE_TL.try_with(Clone::clone).ok().flatten()
}

/// The room profile this call's admin grant is judged by, or `None` when the
/// transport names no profile an operator could have allowed.
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

`ConfigTool::gate()` と `TaskTestTool::execute()` の冒頭ゲートは同じ式:

```rust
if !current_admin_room_profile_for(&config)
    .is_some_and(|name| config.admin_allows_room_profile(&name))
{
    anyhow::bail!(ROOM_REFUSAL);
}
```

`TaskTestTool::namespace()`:

```rust
match current_admin_room_profile_for(&self.state.config) {
    Some(name) => self.state.config.namespace_for_room_profile(&name),
    None => DEFAULT_NAMESPACE_NAME,
}
```

### 伝達（`server/src/serve/mod.rs`, `server/src/tools/subagent.rs`）

- `TurnContext` に `pub admin_room_profile: Option<String>`（`subagent_depth` の）。
- `TurnLoop` に同名フィールド。`run_llm_turn`（L2876）が `let is_acp = ...`（L2886）の直後で:

```rust
let admin_room_profile = if is_acp {
    state.session_room_profiles.lock().await.get(&session_id).cloned()
} else {
    None
};
```

- production `TurnLoop` リテラル（L3053 付近）と各ラウンドの `TurnContext` リテラル
  （L2714 付近）に渡す。
- `TurnLoop::run` のラウンド内、`let timer_origin = self.timer_origin.clone();` の隣で
  `let admin_room_profile = self.admin_room_profile.clone();`、そして
  `let fut = scope_turn_context(turn_ctx, fut);` の直前に
  `let fut = crate::tools::config_tools::scope_admin_room_profile(admin_room_profile.clone(), fut);`
- `subagent.rs` のネスト `TurnLoop` に `admin_room_profile: ctx.admin_room_profile.clone()`。
- テスト用 `TurnContext` リテラル 4 箇所（`serve/mod.rs` 以外）に `admin_room_profile: None`。

### テスト方針

- **config.rs**: キー置換、`admin_allows_room_profile`、`room_profile_name_for_room`
  （明示一致 / `[room_profile.default]` / 暗黙 default）、`config_tools_enabled`、
  旧 `rooms` の migration error、未知 profile の validate error。
- **config_tools.rs**: 実際に読まれる経路で gate を検証する。
  - チャンネル形: 既存の `in_room` ヘルパ（`scope_timer_origin(Chat)`）。
  - ACP 形: `scope_admin_room_profile(Some("ops"), ...)` で許可、
    `Some("guest")` / `None` で拒否。
  - voice 形: `scope_timer_origin(Voice)` で拒否（列挙に `ops` があっても）。
  - `task_test` の namespace が profile の namespace になり、実際のセッションパスに現れること。
  - 登録条件（空 → 未登録、1 名 → 4 本）。
- **serve/mod.rs**: `#[cfg(test)] AdminProfileProbe`（`RiskyTool` と同じパターン）を追加し、
  非 ACP（`/rpc` 相当）の `run_llm_turn` では `None` が scope されることを固定。
- **serve/acp.rs**: 同じ probe を `session/prompt` の実ターンで走らせ、bearer token が
  pin した profile 名（fixture の `"developer"`）が scope されることを end-to-end で固定。
  probe は `ToolKind::Read` かつ `visible_tool_predicate` を素通しする名前にする。

## 受け入れ基準

1. `room_profiles = ["ops"]` + `[room_profile.ops].rooms = ["!ops:x"]` の deployment で、
   `!ops:x` からの `heartbeat_config list` が成功し、別 room からは `Permission denied` で失敗する。
2. 同じ deployment で、`developer` profile に束縛された device の ACP セッション
   （`session/prompt`）から `agent_config write` が成功する。
3. `room_profiles = ["ops"]` のまま、`ops` 以外の profile の ACP セッション・voice・`/rpc`・
   無人からの呼び出しは拒否される。
4. `room_profiles = ["default"]` は「どの profile にも明示的に属さない全 room」を許可する
   （README に明記され、テストで固定されている）。
5. config に `rooms = [...]` が残っていると起動が失敗し、エラーが `room_profiles` を名指しする。
6. `room_profiles = ["typo"]`（未定義名）は `verify` で報告され、serve 起動が失敗する。
7. `room_profiles` が空なら 4 ツールは `ToolSet` に存在しない。
8. `cargo test -p sapphire-agent-server` が緑。

## リスクと未解決事項

- **`"default"` の広さ**（最重要）: `[room_profile.default]` も暗黙 default も
  「明示的に他の profile に属さないすべての room」を吸収するため、
  `room_profiles = ["default"]` は事実上ほぼ全ルーム許可になる。意味論としては
  provider / namespace の解決と一致しており正しいが、運用者が「1 プロファイルのつもり」で
  書くと過剰許可になる。README・`config.example.toml`・`ROOM_REFUSAL` で明示し、
  受け入れ基準 4 でテストする。
- **破的変更**: 旧 `rooms` を書いた config は起動しなくなる。未リリース機能であり、
  影響は main 追跡者に限られる。`migration_errors` の文言に代替キーの書式例を含める。
- **委派の継承**: 管理 profile のターンから委派されたサブエージェントも管理ツールを呼べる。
  「権限が孫まで伝わる」ことを `TurnContext::admin_room_profile` の doc と README に明記する。
- **ACP pin のタイミング**: `session_room_profiles` への挿入は `session/prompt` より前に
  行われる（`session/new` と採用経路）。将来 pin を遅延させると本機能は fail-closed
  （拒否側）に倒れるので安全側。
- **`/rpc` と ACP の区別**: 両者とも `timer_origin` は `None` で、区別は
  `run_llm_turn` の `is_acp`（`acp_sessions` への登録有無）だけが担う。判定箇所を
  増やさないこと（片方だけ直す改修事故を防ぐ）。
- **未解決**: なし（すべて本仕様で決定済み。実装中に新しい設計判断が必要になった場合は、
  実装者ではなく本仕様と計画書の改訂として扱う）。