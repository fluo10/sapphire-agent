# 自律セッション（autonomous session）の導入

- **Issue**: #248「feat: ホストアイドル時に自発活動する自律セッション（autonomous session）の導入」
- **対象**: `server/src/autonomous.rs`（新規）、`server/src/autonomous_config.rs`（新規）、
  `server/src/session.rs`、`server/src/serve/mod.rs`、`server/src/config.rs`、
  `server/src/main.rs`、`server/src/tools/session_tools.rs`、`server/config.example.toml`、
  `server/src/cli_init.rs`、`server/templates/workspace/`
- **前提**: PR #256（subagent profiles）がマージ済み（`main` = `55363aa`）
- **計画書**: `docs/superpowers/plans/2026-09-12-autonomous-sessions-plan.md`

## なぜ

llama.cpp の計算資源は、フミトが寝ている間も仕事中も遊んでいる。heartbeat は
「時刻が来たら1回撃つ」ことはできるが、**撃った仕事が終わった後に手が空く** —
タスクが短ければ1ターンで終わって深夜の残り時間が余り、大きければ朝まで走り続けて
他セッションの応答を待たせる。粒度の制御も、やること自体の選定も、heartbeat には無い。

**heartbeat との違いは3点で、どれも「ループ」の形から来る。**

1. **アイドルで起動する**。時刻ではなく、他セッションが静かなことを条件に動く。
2. **タスクが尽きるまで続ける**。1回撃って終わりではない。キューを優先度順に処理し、
   空になったら止まる。
3. **1タスク1セッション**。活動の単位が heartbeat の「1プロンプト」ではなく
   「1セッション」になるので、後から会話として振り返れる。

副次的な目的として、「自律的に動いている感」がある — つまり**起きたことを人間が
後から読める**ことが要る。これはセッション JSONL と、タスク自身が書く journal /
GitHub 側の成果物で満たす（決定 11）。

**背景として先に置く将来方針**: sapphire-agent には「shell からのワークスペース内
ファイルアクセスを禁止し、ワークスペース内のファイル編集はツール専用にして、
そのツールも room/profile ごとの許可制にする」という方針が別 Issue として起票されて
いる。本設計はその方針に依存しないし、`Origin` / 権限系の設計は何も変えない。
ただし自律セッションは無人で `shell` を回す唯一の経路になるため、**その方針が入る
日には自律セッションの権限は真っ先に見直し対象になる**（決定 10 の注記を参照）。

## 決めたこと

### 1. `kind = "autonomous"` を4つ目の SessionStore として足す

`channel` / `cross-device` / `device-default` に続く4つ目。`sessions/<namespace>/autonomous/`
に置き、`ServeState` が `Arc<SessionStore>` を1本持ち、`store_for_session` が
device-default より先に判定する。

**「自律セッションは普通のセッションである」ことが設計の軸**。専用のターン実行器を
作らない — 既存の `run_llm_turn` をそのまま使う。そのために必要なのは
「どの store に書くか」を教えることだけなので、`store_for_session` に1分岐足すのが
変更の全体になる。ACP と同じ理由で物理的に別ディレクトリにする: 振り返りで
`session_list` に混ざるのは望ましいが、ファイルとしては chat と混ざってほしくない。

- `channel` は `"server"` のまま（既存 chat セッションと同じ）。専用チャンネル名は作らない。
- `namespace` は既定 namespace 固定。room_profile はピンしない
  （`session_room_profiles` に触らないので `run_llm_turn` は
  `registry.background_provider(&config)` に落ちる — **llama.cpp を指すのは
  `[profiles.background]` の役目で、自律セッション用の新しい設定項目は作らない**）。

### 2. セッションファイル名は `{date}-{session-id}.jsonl`

`{date}-{room_id}-{session-id}.jsonl` ではなく中央を省く。`room_id` はファイル名に
出ないが、メタ行には入る（決定 3）。

`SessionStore` は今 `{session_id}.jsonl` 固定で、`resolve_path` も完全一致で
探している。ここに**ファイル命名の種別**を1つ足す:

```rust
enum FileNaming {
    /// `{session_id}.jsonl`. Every store but the autonomous one.
    Plain,
    /// `{date}-{session_id}.jsonl`, `date` being the agent-day the file
    /// was created in.
    Dated { boundary_hour: u8 },
}
```

`path_for_new` が `FileNaming` を見てファイル名を決め、`resolve_path` は
`Dated` のときだけ「`-{session_id}.jsonl` で終わる名前」で照合する。session_id は
v7 UUID なので接尾辞一致は曖昧にならない。**`Plain` の store は1バイトも挙動が
変わらない**（`name == "{session_id}.jsonl"` のまま）。

### 3. `room_id` にタスク名を入れる

自律セッションでは `room_id` に使う値が無い。**そこにタスク名（`autonomous/<name>.md`
のファイルステム）を入れる。**

これは見た目の都合ではなく、**永続状態を増やさないための鍵**である（決定 5）。
`SessionMeta` はセッションファイルの1行目に入っているので、`session_rows()` が
1ファイル1行読みで返す `meta.room_id` だけで

- どのセッションがどのタスクのものか
- そのタスクの最後の活動がいつか
- まだ開いているか（`is_closed`）

が全部分かる。本文を読んでマーカー文字列を探す必要がない。副産物として
`session_list` のラベルが `autonomous/<task>` になる（`session_label` が
`channel/room_id` を出すため）。

`room_id` が空でないことは `session_label` の分岐にも効く — 空だと
`"autonomous session"` になり、どのタスクか読めなくなる。

### 4. 状態ファイルは `<workspace>/state/autonomous.json` の1枚だけ

```json
{
  "status": "running",
  "reason": "task: journal, turn 2/3",
  "updated_at": "2026-09-12T03:14:07Z"
}
```

- `status`: `"idle"` | `"running"`。
- `reason`: 人間が読むための1行。**書き込むだけで、読み戻さない**。
  `"no tasks"` / `"busy: channel/room-1"` / `"task: journal"` / `"disabled"` など。
- `updated_at`: 最終更新。プロセスが生きているかの確認に使う。

**session_id もタスクの進捗もここには書かない**。それらは store から導出する
（決定 3・5）。書くと store と二重管理になり、片方だけ古くなる。

書き込みは tmp ファイル + rename。読む側は居ないが、途中で落ちた半端な JSON を
残さないため。`<workspace>/state/` は新設のサブディレクトリで、`init` の
`EMPTY_DIRS` には入れない（タスクが1つも無ければ1度も作られないのが正しい姿で、
空ディレクトリを約束する意味がない）。heartbeat の state.json とは無関係。

### 5. 状態は全てセッションストアから導出する（永続状態を増やさない）

タスク `T` について:

- **クールダウンの起点** = `room_id == T` のセッションのうち最新の `last_at`（無ければ `created_at`。閉じたセッションも含めて探す）。
heartbeat の `[Heartbeat: <name>]` に倣った `[Autonomous: <name>]` は本文の先頭に
入れるが、それは**モデル向けの合図**であって検索キーではない（検索キーは決定 3 の
`room_id`）。両方持つ理由を書いておく: プロンプトに「これはシステムからの起動だ」と
書くことと、store から安く引くことは別の要求である。

さらに最初の user メッセージの2行目には `Session: <workspace からの相対パス>` を
入れる。`room_id` と `[Autonomous: <name>]` が「どのタスクか」を表すのに対し、
これは「どのファイルに残ったか」を表す。タスク本文が journal に書き写すための1行で、
パスは store から導出するので実際のファイル名と食い違わない。

### 6. コンテキストは isolated、ただしワークスペース知識は読む

`run_llm_turn` は `Workspace::build_system_prompt` を通るので、`SOUL.md` /
`IDENTITY.md` / `USER.md` / `AGENTS.md` / メモリが普通に入る。**これは意図どおり** —
自律セッションは「別の人格」ではなく「同じエージェントが一人で働いている」状態
だからである。落とすのは**他セッションの会話履歴**だけで、それは新規セッションで
始めることが自動的に満たす（`state.sessions` にエントリが無ければ
`store.load_session` が空を返す）。

サブエージェント（`agents/*.md`）がワークスペースのプロンプトを**継承しない**のとは
逆である。あちらは「要らない部分を落とす」ために存在する道具であり、こちらは
「同じエージェントの続き」である。この違いは意図的で、両方の設計文書に書いてある。

`lightContext` のような「何を読むかを絞る設定」は**作らない**（YAGNI）。必要になった
時点で `memory_namespace` の `skills` ゲートと同じ形で足せる。

### 7. タスク定義は `<workspace>/autonomous/*.md`

`heartbeat/*.md`・`agents/*.md` と同じ frontmatter + body。**3つ目の規約ではなく、
1つの規約の3つ目の利用者**。ローダーは `agents.rs` の形をそのまま踏襲する
（ディレクトリが無ければ空、拡張子 `md` だけ、壊れたファイルは警告して飛ばす、
名前順にソート）。

```yaml
---
enabled: true        # optional, default true
priority: 10         # optional, default 100。小さいほど先
cooldown_days: 7     # optional, default 0。0 = 前のセッションが終わったらすぐ次
max_turns: 3         # optional, default 3。1セッションのターン上限
---
（タスクの指示文。本文がそのまま最初の user メッセージになる）
```

- `priority` に既定値（100）を置くのは、**未指定のタスクも必ず順序の中に居る**必要が
  あるから。`Option` にすると「優先度が無い」を毎回どこかで解釈することになる。
  100 は「まだ考えていないタスク」を、`priority: 10` と書いたタスクの後ろに置く。
- `max_turns: 0` は `1` に切り上げる。0 の意味を「1ターンも走らせない」にすると、
  そのタスクは永久に何もせずセッションだけ作って閉じる。
- 同点は名前昇順で決める（`sort_by` が安定なだけでは、`read_dir` の順に依存して
  しまう）。
- **sapphire-journal との蜜結合はしない**。ジャーナル由来の仕事は
  「自分のタグの `task_status: open` 条目を1回N件処理し、進捗は journal 側に
  書き戻す」という**1つのタスクファイル**に集約する。進捗・再開点・履歴が
  journal 側に一元化され、エージェント側には何も残らない。

タスクの例（初期テンプレートとして1つ置く）:

- ジャーナルの自分のタグ付き open タスクを1回N件処理する（`cooldown_days: 1`）
- 古いジャーナルを sapphire-journal へ移行する（1回10件、`cooldown_days: 1`）
- 参考プロジェクトの更新を調べて issue 化する（`cooldown_days: 7`）
- リファクタ・改修案の issue 化（`cooldown_days: 30`）

### 8. 起動スタイルはタスク単位ループ（heartbeat の第3のループ）

`heartbeat.rs` は day-boundary ループと cron ループの2本を `spawn` している。
自律ループはそこに足すのではなく**独立した3本目**として `autonomous.rs` に置く。

`main.rs` の起動位置は `serve_state` 構築の後、`serve::run` の前。**channel 設定の
有無に依存させない** — heartbeat は `Agent` が要るので
`if config.matrix.is_some() || config.discord.is_some()` の中に居るが、自律ループは
`ServeState` だけで動くので、chat を持たない ACP 専用 deployment でも動くのが正しい。

```text
loop {
    sleep(poll_seconds)

    tasks = load_autonomous_dir(<workspace>/autonomous)   // 毎回読み直す
    enabled = tasks.filter(enabled)
    if enabled.is_empty() { status = idle("no tasks"); continue }

    picked = next_task(enabled, due)                       // priority, name 順
    if picked.is_none() { status = idle("nothing due"); continue }

    // 1サイクル = 1タスク。新規セッションで開始し、最大 max_turns ターン走る
    session = create_session(picked)                       // 決定 5
    loop {
        run_llm_turn(session, "[Autonomous: <name>]\n\n<body>")
        turns += 1
        if done or turns >= picked.max_turns { store.close_session(session); break }
    }
}
```

**アイドル検知は開始条件にだけ効く。** 一度始まったセッションは DONE または `max_turns` まで走り切る。走っているセッションを他セッションの動きで途中で止めて続きから再開する、という中座・再開機構は今回スコープ外である（決定 13）。一度始まったセッションは走り切り、すべてのセッションは走り切り時点で閉じられる。次サイクルで再び due になったときは、常に新しいセッションで始まる。

### 9. アイドル判定は他ストアの `last_at` の最大値

**自律ストア自身を除く**全ストア（channel / cross-device / device-default / mcp / acp）
の `session_rows()` から `last_at` の最大値を取り、`now - idle_minutes` より古ければ
アイドル。1行も無ければアイドル。`idle_minutes` の既定は 30（Issue コメントの
「30分の無稼働でアイドル判定」）。

`ambient` の `DeviceState` は使わない。あれは音声デバイスの状態で、Matrix の
メッセージ1通も検知しない — S4 まで `Idle` 固定のまま放置されている。

**既知の限界（受け入れる）**: `last_at` はセッションに**保存された**最後のメッセージの
時刻である。長いターンの最中（user メッセージは既に書かれ、assistant の返答がまだ）は
「静か」に見える。つまり**他セッションが長いターンを走っている間に自律セッションが
始まりうる**。完全に防ぐには進行中ターンのレジストリが要る（`state.sessions` の
エントリの有無で近似できるが、ACP と `/rpc` の deferred session で例外が出る）。
v1 では `idle_minutes` を上げるのが運用上の答えで、既定 30 分は
「人間が1回喋って、返事を待っている」程度の間隔より十分長い。ここは
**リスク節に残す**（解決しない）。

### 10. 権限は `[autonomous] origin`、既定は `"channel"`

```toml
[autonomous]
enabled = true
idle_minutes = 30
poll_seconds = 60
origin = "channel"   # "channel" | "trusted"
```

- `"channel"` → `Origin::Channel`。読み取りは許可、実行系（`Execute`/`Other`）は拒否、
  編集は無承認で許可。heartbeat の chat leg と同じ扱い。**既定**。
  `autonomous/*.md` を書けるのは管理者だけ、という前提が崩れた deployment でも、
  無人実行が既定で `shell` に届かないようにする。
- `"trusted"` → `Origin::Trusted`。`/rpc`・voice・`/a2a` と同じ行で、種類を問わず許可。
  GitHub の issue 起票や commit など `shell` が要るタスクは、これを**明示的に**選ぶ。

`AutonomousHost::origin()` がこれ1つを返すだけで、`decide` も `Origin` も
**一切変えない**。既定を `"channel"` にしておくのは、この前提（`autonomous/*.md` を書けるのは
管理者だけ）が崩れた deployment でも無人実行が `shell` に届かないようにするためで、
`shell` が要るタスクだけが `"trusted"` を明示的に選ぶ。

**運用上の必須条件**: `"trusted"` でも `[tools] host_access.enabled = true` でなければ
`host_tool_denied` が `decide` より先に `shell` / `file_write` を拒否する
（`Origin::Trusted` の `decide` は `Allow` を返すが、そもそも呼ばれない）。
自律セッションに shell を回させるには**両方**が要る。`config.example.toml` の
`[autonomous]` の説明にこれを書く。

**将来方針との関係**: 「shell からのワークスペース内アクセス禁止・ファイル編集は
ツール専用・room/profile ごとの許可制」が入ると、`Origin` の行だけでは
「どのディレクトリを触れるか」を表現できなくなる。その時点で自律セッションは
（a）room_profile 相当の何かをピンする、（b）専用の `Origin` 変種を持つ、の
どちらかが要る。**今は作らない**が、`AutonomousHost` を `Origin` を1つ持つだけの
小さな型にしておくのは、その日が来たときに差し替える場所を1箇所にしておくため。

### 11. 配信はしない（v1）。振り返りは `session_list` / `session_read`

自律ターンの最終テキストを Matrix / Discord へ投稿する経路は**作らない**。

- 投稿すると `Agent` に依存し、channel 未設定の deployment でループが動かなくなる。
- 投稿先の room が「他セッションの活動」になり、アイドル判定の入力に混ざる
  （自分で自分を起こす循環）。
- Issue 本文の「専用チャンネル」は、実質「後から読める」ことが目的である。
  それは**セッション JSONL と、タスク自身が書く journal / GitHub 側の成果物**で
  満たす。ジャーナルへの記録はタスク本文（エージェントへの指示）の責任である。

**日次ノートへの記録もエージェント任せにしない。** 最初の user メッセージの2行目に
`Session: <workspace からの相対パス>` を必ず入れる（決定 5）。日次ノート
（`memory/<ns>/daily/YYYY-MM-DD.md`）に run を残す側は、この1行を読んでセッションを
名指しできる。**注入する側（sapphire-journal）の実装は本 Issue のスコープ外**で、
ここでは「その1行を必ず渡す」ところまでを約束する。

代わりに `session_list` / `session_read` に自律ストアを足す。`SessionSources` に
`Arc<SessionStore>` を1本増やし、`all_rows()` と `transcript()` の探索先に加える。
これで ACP の会話からも「昨夜エージェントが何をしたか」を一覧 → 本文で読める。

### 12. `cooldown_days` は前回アンカーからのレート制限、`max_turns` 到達でセッションを閉じる

- **クールダウン**: `now - anchor >= cooldown_days 日` のときだけ due。`anchor` は
  決定 5 のとおり store から引く。`cooldown_days = 0` は「前のセッションが終わったら
  すぐ次」。タスク種別ごとに「7日」「30日」を書けるのが狙い。
- **セッションは走り切り時点で閉じる**: DONE でも `max_turns` 到達でも、セッションは `close_session` で閉じられる。中座・再開機構が無いので開いたままのセッションは存在せず、**再開の概念自体が無い**（次サイクルは常に新規セッション。決定 5・13）。
- **プロセス内のバックオフ**: セッションを閉じた直後、そのタスク名を
  `not_before: HashMap<String, Instant>` に `now + cooldown` で入れる。
  これが無いと `cooldown_days = 0` + `max_turns = 3` が
  「セッション作成 → 3ターン → 即座に次」の連続になり、セッション一覧が
  一晩で何百本にもなる。**store 側のアンカーが再起動をまたぐ正しさを担い、
  この map が1プロセス内の行儀を担う**。`cooldown_days = 0` は
  「一晩中続けて働け」という意味であり、それ自体は正当な設定なので禁止しない。
- 日付境界（`day_boundary_hour`）をまたいでもセッションは回転しない。要約もしない。
  `maybe_handle_day_boundary` は `agent.rs` の経路で、自律ループは通らない。
- 自律セッションは daily log の catch-up に数えない。catch-up は channel store と
  ACP store しか見ておらず、自律ストアは別ディレクトリだからである。**意図的な
  措置として受け入れる** — 自律活動の記録は journal 側の責任（決定 11）。

### 13. スコープ外

- **アイドル検知による自律セッションの中座・再開の機構**。アイドル検知は開始条件としてのみスコープ内（決定 8・9）。走っているセッションが他セッションの動きで途中で止まり、静かになった続きから同じセッションへ再開する、という機構は作らない。**理由**: サブエージェントが外部APIプロバイダ実行へ移行済みで、並行作業によるプロンプトキャッシュのドロップが起こりにくくなったため、この機構の本来の動機（llama.cpp のキュー滞留とプロンプトキャッシュ退避）自体が消えている。
- **llama.cpp のパラレルリクエスト / コンテキスト共有 / プロンプトキャッシュ**。
  サブエージェントが API 実行に移行済みで解消済み。session_id / kind を区別する
  ことだけは行う（セッション混在の防止）。
- `lightContext` 的な読み込み量の調整（決定 6）。
- チャットへの配信（決定 11）。
- 専用の namespace / room_profile（決定 1）。
- 権限モデルの変更（決定 10）。
- 自律セッションの保持ポリシー（削除・圧縮）。`sessions/<ns>/autonomous/` は
  `session_list` から見える普通のセッションなので、既存の `[compression]` が効く。

## 具体的な設計

### ファイル構成

| 場所 | 役割 |
|---|---|
| `<workspace>/autonomous/*.md` | タスク定義。frontmatter = 制御情報、body = 指示文 |
| `<workspace>/sessions/<ns>/autonomous/{date}-{uuid}.jsonl` | セッション。1タスク1セッション |
| `<workspace>/state/autonomous.json` | 状態の1枚（決定 4） |

### 新しい型

```rust
// autonomous_config.rs
pub struct AutonomousTask {
    pub name: String,        // ファイルステム = セッションの room_id
    pub enabled: bool,
    pub priority: i64,
    pub cooldown_days: u32,
    pub max_turns: usize,
    pub body: String,
}
pub fn load_autonomous_dir(dir: &Path) -> Vec<AutonomousTask>

// config.rs
pub struct AutonomousConfig {
    pub enabled: bool,        // default false
    pub idle_minutes: u64,    // default 30
    pub poll_seconds: u64,    // default 60
    pub origin: AutonomousOrigin,  // default Trusted
}
pub enum AutonomousOrigin { Trusted, Channel }

// autonomous.rs
pub struct AutonomousLoop { state: Arc<ServeState>, workspace_dir: PathBuf }
pub fn is_idle(rows: &[SessionRow], now: DateTime<Utc>, idle: Duration) -> bool
pub fn next_task<'a, F: FnMut(&AutonomousTask) -> bool>(
    tasks: &'a [AutonomousTask], due: F,
) -> Option<&'a AutonomousTask>
pub struct AutonomousState { status, reason, updated_at }  // state/autonomous.json

// serve/mod.rs
pub(crate) struct AutonomousHost { origin: crate::tools::policy::Origin }  // TurnHost
```

### 受け入れ基準

1. **既定で無効**。`[autonomous] enabled` を書かない限り、ループは起動せず、
   ファイルもディレクトリも作られない。
2. `[autonomous] enabled = true` + `autonomous/*.md` を1つ置くと、他セッションが
   `idle_minutes` 静かだったときに1セッションが作られ、そのタスクの本文が
   最初の user メッセージとして渡り、モデルの返答が
   `sessions/<ns>/autonomous/{date}-{uuid}.jsonl` に残る。
3. 他セッションに動きがある間は**始まらない**。一度始まったセッションは走り切る（途中中座・続きからの再開はスコープ外: 決定 13）。
4. `max_turns` に達したセッションは閉じられ、次にそのタスクが due になったときは
   新しいセッションで始まる。
5. `cooldown_days` の内側では同じタスクが二度 due にならない（store のアンカーで
   判定されるので、プロセスを再起動しても同じ）。
6. タスクが0件／全部 enabled=false なら何も起きない。壊れた frontmatter のファイルは
   警告して飛ばされ、**他のタスクは動く**。
7. `state/autonomous.json` に `idle` / `running` とその理由が書かれ、
   `reason` から「今何をしているか」「なぜ止まっているか」が読める。
8. `session_list` に `server/<task>` が出て、`session_read` で本文が読める。
9. ACP・chat・`/rpc` の既存の挙動が変わらない。`Plain` な store のファイル名・
   解決順が完全に同一。
10. `origin = "channel"` のときは自律ターンから `shell` が拒否され、
    `origin = "trusted"` + `host_access.enabled = true` のときは通る。

### 実装上の注意

- `tokio::time::interval` は `Duration::ZERO` で **panic する**。
  `poll_seconds` は `max(1)` してから渡す。
- ループの最初の `tick.tick().await` は捨てる（起動直後に走らない）。
  heartbeat の両ループと同じ。
- `state/autonomous.json` の書き込みは `std::fs`。`SessionStore::notify_updated` の
  ような workspace への通知は行わない（ワークスペースの文書ではない）。
- `AutonomousHost` は `round_budget()` を**実装しない**。既定の
  `RoundBudget::Unattended` がそのまま正しい（人間が止められない経路なので
  有限・既定 25）。実装しないことを doc コメントに書く。
- ターン数のために `load_session` を毎回呼ぶ。自律ストアは小さいので許容する。

## リスクと未解決

1. **長いターンの最中に他セッションが「静か」に見える**（決定 9）。v1 は
   `idle_minutes` を上げるのが答え。進行中ターンのレジストリを足せば厳密にできるが、
   ACP の deferred session と `/rpc` の保留セッションの扱いを先に決める必要がある。
2. **`origin = "trusted"` の穴は塞がっていない**。管理者以外が chat に書ける
   deployment にした瞬間、`autonomous/*.md` を書き換えて次サイクルに無人実行させる
   経路が開く。`Origin::Channel` の doc が名指ししている穴そのもので、決定 10 の
   将来方針（ファイル編集の許可制）が入るまでの既知のリスク。決定 10 で既定を
`"channel"` にしたので、今この穴を持つのは `"trusted"` を明示的に選んだ
deployment だけである。
3. **止まらないタスク**。`max_turns` はセッションの上限だが、クールダウンが短ければ
   セッションが次々作られる。`cooldown_days = 0` は「連続稼働」の意図的な設定なので
   禁止しないが、`state/autonomous.json` とセッション一覧がその唯一の可視化で
   ある。暴走の検知（単位時間あたりのセッション数）は今回作らない。
4. **アイドル判定の対象に `mcp` を含める**。外部 AI が MCP 経由で書いている間は
   静かとは言えないため含めるのが正しいと判断したが、`mcp` の `write_report` は
   バッチ的な書き込みで、その直後に自律セッションが30分動かなくなる。
   運用で気になれば `idle_minutes` を下げる。
5. **`autonomous` の設定をワークスペース層に許すか**。`heartbeat_enabled` と同じ
   「エージェントの振る舞い」なので許可する（`config_layer` の allowlist に
   `["autonomous"]` を足す）。許可すると、ワークスペースを持てる者が
   `enabled = true` と `idle_minutes = 0` を書ける。これは
   `heartbeat_enabled` が既に持っている権限と同じなので新しい穴ではない、という
   判断である。
