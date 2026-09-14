# 設定ファイル操作ツール（heartbeat / autonomous / agents）

- **Issue**: #265「設定用ツールの追加・enabled フラグ・テスト用ツール・ホットリロード」
- **対象**: `server/src/tools/config_tools.rs`（新規）、`server/src/tools/mod.rs`、
  `server/src/serve/mod.rs`、`server/src/agent.rs`、`server/src/config.rs`、
  `server/src/config_layer.rs`、`server/src/main.rs`、`server/src/frontmatter.rs`、
  `server/src/tools/subagent.rs`、`server/src/agents.rs`、
  `server/config.example.toml`、`server/templates/workspace/config.toml`、`README.md`
- **前提**: PR #260（autonomous sessions, `main` = `00fb983`）がマージ済み
- **計画書**: `docs/superpowers/plans/2026-09-14-config-tools-plan.md`
- **関連**: #257（シェルからのワークスペース内ファイルアクセス制限とツール別許可制の導入）。
  本 Issue はその**前提**である。

## なぜ

#257 は「ワークスペース内のファイル編集をツール専用にし、`shell` からも
`file_write` からもワークスペースの中身を直接いじれなくする」方向へ進む。
そのとき、**heartbeat / autonomous / agents の定義ファイルはエージェントにとって
編集不能な設定**になる。ところがこれらは「エージェントの振る舞い」の一部であり、
運用の現場では次のようなことが起きる:

- 深夜に走ったタスクが暴れていたので、**その場で `enabled: false` にしたい**。
- うまくいったタスクを**もう少しだけ条件を変えて再開したい**。
- 新しいサブエージェントを**使いたいと思ったその場で定義したい**。

いずれも今は「人間がワークスペースのファイルを直接開いて編集する」しかない。
つまり **#257 が入るとできなくなる操作が、実際に必要な操作**である。だから
#257 の前に、その操作を専用ツールとして先に用意する。

もう1つの動機が **autonomous を有効化する前の検証**である。autonomous は無人で
ループを回すので、いきなり有効にすると「機能しない」「トークンを食べすぎる」
「正常終了（タスク達成もしくは中止）しない」という3つの失敗を、人間が寝ている間に
起こしうる。有効化の前に**普通のセッションから1回叩いて結果を読める**必要がある。
これが決定 6 のテスト用ツールである。

サブエージェント（`agents/*.md`）については、フミトさんからの追加要件のとおり
**設定操作ツールの対象に含めるが、`enabled` フラグもテスト用ツールも足さない**。
サブエージェントは「呼べば普通に走る」ので、テストは `subagent` を普通に呼べば
よく、有効化という状態遷移そのものが存在しない（決定 7）。

## 決めたこと

### 1. 3つの管理ツールを1つの実装で作る

`<workspace>/{heartbeat,autonomous,agents}/` の3ディレクトリは、どれも
`agents.rs` が確立した「frontmatter + body」の同じ規約を読む（`heartbeat_config.rs`
の doc が名指ししているとおり、*3つ目の規約ではなく1つの規約の3つ目の利用者*）。
書き込み側も同じ規約の利用者にするべきなので、**1つの構造体をディレクトリ違いで
3回登録する**。

```text
heartbeat_config   action = list | read | write | set_enabled
autonomous_config  action = list | read | write | set_enabled
agent_config       action = list | read | write
```

- ツール名は「そのディレクトリの設定」を表す名詞にする。`memory_add` /
  `session_list` / `skill_install` のような動詞形にしなかったのは、1ツールが
  4つの動詞を持つため — `memory_add` 相当を4本に割ると12本になる。
- `action` は必須。`list` はファイル名と（heartbeat/autonomous のみ）`enabled` を
  返し、`read` はファイル全文を返す。**モデルがまず `list` / `read` で現状を読んで
  から書く**という順序を自然に誘導する。
- **`set_enabled` は frontmatter を行単位で書き換える**（決定 3）。
- **`agent_config` に `set_enabled` は無い。** 未知の action は
  「このツールでは使えない」と返す（実行時エラーではなく説明文で返す。
  モデルが再試行で直せる種類の誤りである）。

### 2. 「管理者向けツール」はルーム許可制で表現する

Issue の「特定のチャンネル（ルーム）でしか使えないように設定できる」を、
新しい設定テーブル1つで表現する。

```toml
[tools.admin]
# これら3(+1)本のツールを使ってよいチャンネル側のルーム id。
# 空（既定）なら、ツールは**登録すらされない**。
rooms = ["!ops:example.com"]
```

- **既定で無効**。`rooms` を書かない限り、4本のツールは `ToolSet` に存在しない。
  「登録されているが全部拒否される」より「最初から居ない」ほうが良いのは、
  `ToolSet::specs_filtered` の doc が書いているとおり**使えないツールは
  存在しないツールより悪い**（モデルが1往復を無駄にする）からである。
  これは `host_access` と同じ形の既定（決定 5）。
- 許可判定は**呼び出し元のルーム id** で行う。`/rpc`・`/acp`・voice のように
  ルームを持たない経路からは**使えない**。管理者が設定をいじる場所を
  「人間が複数いて、誰が書けるか分かっているチャットルーム」に固定するのが狙いで、
  これは #257 が導入する「経路ごとの許可制」と同じ思想である（決定 8）。
- ルーム id は**既に `[room_profile.<n>].rooms` に入っている値と同じ名前空間**。
  新しい概念ではない。`config.room_profile_for(room_id)` が引ける値である。
- **ホスト専用**。`[tools.admin]` は workspace 層の allowlist に入れない
  （`[tools]` 全体が既に入っていない）。理由は `host_access` と同じで、
  「誰が管理者か」をワークスペースの中身で決められるべきではない。

### 3. `set_enabled` は行編集、`write` は全文置換

2つの書き込みモードを分ける。

| action | 意味 | 実装 |
|---|---|---|
| `write` | ファイル全体（frontmatter + body）を置換。無ければ作る | 文字列をそのまま書く |
| `set_enabled` | frontmatter の `enabled:` 行だけを足す／書き換える | **行単位の編集** |

`set_enabled` を「パースしてシリアライズし直す」で実装しないのは、
**frontmatter のコメントとフィールド順が消える**からである。heartbeat のタスク
ファイルは人間が手で書くもので、`schedule` の意図を書いたコメントが入っている
ことがある。モデルが「無効化」という1つの意図を実行しただけで、人間が書いた
コメントが全部消えるのは**モデルが人間のノートを書き換えた**のと同じである。

規則（`frontmatter.rs` に `set_enabled(raw, enabled) -> Option<String>` として置く）:

- frontmatter ブロック（先頭 `---` 行から次の `---` 行まで）を探す。無ければ `None`。
- その中に **トップレベルの** `enabled:` 行があれば、その行だけを置き換える。
  `enabled` がインデントされている（`voice:` の中など）場合はトップレベルでは
  ないので触らない — 行頭が `enabled` で始まることを条件にする。
- 無ければ frontmatter の**末尾**（閉じ `---` の直前）に挿入する。
- body と改行は1バイトも変えない。**`write` 以外の書き込みは、対象行以外を
  触らないことが保証される。**

`read` が返すのはファイル全文なので、モデルは必要なら `write` で全文を書き直せる。
人間がコメントを消したいときに消す手段は残っている（`write`）。問題は
**消したくないときに消えないこと**である。

### 4. ファイル名の検証を1箇所に集約する

管理ツールは**ワークスペース内の決まったディレクトリの外に出てはならない**。
`config_tools.rs` に1つ検証関数を置き、3ツールすべてがこれを通る。

```rust
/// `<workspace>/<dir>/<name>.md` を返す。`name` はファイルステムであって
/// パスではない。`..` / 区切り文字 / 先頭ドットを弾く。
fn definition_path(workspace_root: &Path, dir: &str, name: &str) -> Result<PathBuf>
```

弾くもの:

- 空文字、`.` で始まる名前（`.ssh` などを作らせない）、
- `/` `\` `..` を含む名前（`../../etc/x` を弾く。`expand_path` を通さない）、
- `md` 以外の拡張子を明示された場合（`name` はステムを受け取るので、`foo.md` と
  来たら `.md` を落として `foo` にする。拡張子を保ったまま渡す呼び出しは
  受け付けない）、
- 名前が空になるケース。

`file_write` は「絶対パスも `~` もワークスペース外も受ける」ツールで、
その自由度はここでは**要らない**。#257 の目的は「エージェントがワークスペースを
いじる手段を、意図が読める粒度に置き換えること」であり、管理ツールはその
**意図が読める側**の代表例である。だから最小の権限（決まったディレクトリの
`*.md` だけ）を持つ。

### 5. ツールは `ToolKind::Edit`、許可の判断はルーム許可制が担う

`kind()` は `Edit` を返す。

`Origin::Channel` の `decide` は `Edit` を**無承認で許可**する。つまり
許可されたルームからの呼び出しは通る。これが意図どおりである理由:

- 管理ツールが使いたい場所は、まさに**チャットルーム**である（決定 2）。
  ここで `Other` を返すと `Origin::Channel` は `Deny` するので、ツールは
  「特定のルームでしか使えない」どころか**どのルームからも使えない**。
- したがってこのツールの安全装置は `decide` の行ではなく
  **`[tools.admin].rooms` の許可リスト**である。既定は空＝ツール未登録。

`ToolKind::Read` にしない理由も同じで、`read` だけを許可しても
`write`/`set_enabled` は同じツールなので**一緒に通ってしまう**。
action ごとに `kind()` を変えることはできない（`kind` はツール単位）ので、
このツールの権限は「1つ」として扱い、`Edit` より緩い扱いを主張しない。

**注意（リスク節 1）**: 許可リストに入れたルームに、運用者が信用しない人間が
書き込める場合、その人間はエージェント経由で heartbeat / autonomous の定義を
書き換えられる。autonomous は無人で走るので、これは #257 が塞ごうとしている穴と
同種である。既定オフと `allowed_users` の併用が運用上の答えで、ここは
**仕様として受け入れる**（`Origin::Channel` の doc が既に名指ししている穴と
同じ形の既知のリスク）。

### 6. テスト用ツール `task_test` は「1回だけ、普通のセッションで走らせる」

```text
task_test  kind = "heartbeat" | "autonomous"   name = <ファイルステム>
           max_turns = <省略時は定義の値、上限 3>  (autonomous のみ)
```

動き:

1. 定義ファイルを読む（**`enabled: false` でも読む。これが目的である**）。
2. **テスト専用のセッションを autonomous ストアに作る**。`room_id` は
   `"test:<kind>:<name>"`、`channel` は `"server"`。
3. `run_llm_turn` を最大 `max_turns` 回呼ぶ（heartbeat は 1 回固定。heartbeat の
   タスクは「1プロンプト」だから）。
4. 結果（最終テキスト・ターン数・セッションの相対パス）を返し、セッションを閉じる。

**`room_id` をタスク名にしないことが要点である。** autonomous の `is_due` は
`room_id == task.name` のセッションの最新活動をクールダウンのアンカーに使う
（設計 2026-09-12 決定 5・12）。テストがタスク名を名乗ると、**テストしただけで
本番のクールダウンがリセットされる**。`test:` を前置した別名にすることで、
テストはクールダウンに一切影響しない。この規則は autonomous の仕様に対する
**追加の不変条件**なので、`autonomous.rs` の `is_due` の doc に1行足す。

- テストセッションが autonomous ストアに落ちるのは、`session_list` /
  `session_read` から**後で読める**ためでもある（自律セッションの振り返りと
  同じ道具を使う）。`session_for_session` の分岐は `absolute_path_for` を
  見るので、テストセッションも正しく自律ストアに解決される。
- **配信はしない。** heartbeat タスクには `room_id:` / `voice:` の配信先があるが、
  テストは配信経路を試すものではない（Matrix のルームに居ないと動かないタスクを
  「テストできない」にしないため）。テストが確かめるのは**プロンプトの振る舞い**
  と**終了の仕方**であり、配信の確認は本番の1回目にやる。この線引きはツールの
  description に書く。
- `max_turns` の上限を 3 に固定するのは、テストが**トークンを食べすぎないこと**を
  確かめるための道具だから。上限を撤廃すると、テスト自体が確かめたい問題を
  起こす。
- `enabled: false` のままでも走るので、**有効化 → テスト → 無効化**という順序を
  強制しない。決定 1 の `set_enabled` と組み合わせて「まずテスト、良ければ
  `set_enabled true`」が自然な運用手順になる。
- 権限は同じ `[tools.admin].rooms` で gate する（決定 2）。「管理サーフェスは
  スイッチ1つ」という一貫性を選んだ。テストも無人実行の1形態である以上、
  別扱いにする理由が薄い。

**トークン量の報告はしない。** `run_llm_turn` の結果は今 `Option<String>` の
最終テキスト・`was_first_turn`・`stop` しか返さず、usage は
`ProviderResponse` の時点で落ちている。usage を通すには
`LlmTurnOutcome` とプロバイダ境界に手を入れることになり、**「トークンを
食べすぎないか」はターン数・所要時間と `session_read` で読める本文の長さで
おおよそ判断できる**ので、v1 ではやらない（YAGNI）。必要になった時点で
`PromptUsage` を `LlmTurnOutcome` に足す（プロバイダが usage を返さない場合が
あるので `Option` になる）。

### 7. サブエージェントには `enabled` もテストツールも足さない

フミトさんの追加要件のとおり。

- `agents/*.md` には `enabled` フラグを**追加しない**。サブエージェントは
  バックグラウンドで勝手に起動するものではなく、`subagent` が呼ばれたときに
  だけ走る。**有効・無効という状態が存在しない**ものに状態を足すと、
  「無効なのに呼ばれたらどうするか」（黙って何もしない／エラーを返す）という
  余計な問いが生まれる。呼ばれなければ走らないのだから、無効化は
  「定義ファイルを消す」か「`tools: []` にする」で足りる。
- テスト用ツールも作らない。**テストは `subagent` を普通に呼ぶ**。テスト専用の
  経路を足すと、「テストで通ったのに本番で失敗する」余地が生まれるだけで、
  サブエージェントには試したい配信経路もスケジュールも無い。
- したがって `agent_config` は `list` / `read` / `write` の3つだけ（決定 1）。

### 8. `write` は書く前に検証する（agents のみ）

`agent_config` の `write` は、書く前に**その場でパースして検証する**。

- frontmatter が無い／`description` が無い／YAML が壊れている
  → 書かずにエラーを返す（`load_agents_dir` の「壊れたファイルは警告して
  飛ばす」は**読み込み**の寛容さであって、**これから書く**ファイルに対して
  適用する規則ではない。壊れた定義を書けば、そのサブエージェントは
  次に `subagent` を呼ぶまで誰にも見えないまま無言で存在しなくなる）。
- `profile:` が `config.profiles` に無い → 書かずにエラーを返す。既存の
  `Config::validate_subagent_profiles` を**同じ関数として再利用**する。
  起動時検証（2026-09-11 決定 2・3）は「タイプミスしたプロファイル名で
  静かに別モデルへ流れる」のを防ぐために起動を落とすと決めた。エージェントが
  実行時に定義を書けるようになる以上、**同じ規則を書き込み時にも適用しないと
  穴になる**（起動時検証をすり抜けて、次回起動まで気づかれない定義が残る）。
- `tools:` の未知の名前は**エラーにしない**。既存の `subagent` は
  「1行のタイプミスでそのエージェントの残りを奪わない」方針で警告のみ
  （`newly_unknown_tools` の doc）。その方針を書き込み時にもそのまま使う。

heartbeat / autonomous の `write` は frontmatter の検証をもっと軽くする:
**パースできなければエラー**（壊れたタスクを書くと `load_*_dir` が警告して
飛ばすので、書き手には「無言で消える」に見える）。`heartbeat` の `schedule:`
が cron として解釈できなければエラー（`parsed_schedule()` が `None` を返す）。
autonomous の body が空ならエラー（ローダーが skip する規則と同じ）。

### 9. ホットリロード（低優先度）の現状を正しく書く

Issue の3点目「heartbeat / autonomous ファイルのホットリロード」について、
**現状を調査した結果**は次のとおりで、思われているより進んでいる。

| 対象 | 現状 | 結論 |
|---|---|---|
| `heartbeat/*.md` のタスク定義 | `run_cron` が**毎ループ `load_heartbeat_dir` を呼び直す**（`heartbeat.rs:336`） | **実装済み** |
| `autonomous/*.md` のタスク定義 | `run_cycle` が**毎サイクル `load_autonomous_dir` を呼び直す**（`autonomous.rs:241`） | **実装済み** |
| `agents/*.md` の定義 | 起動時に1回読み、`SubagentTool` が `Vec<AgentDef>` を**所有** | **未実装（これが唯一の穴）** |
| host 設定のスイッチ（`heartbeat_enabled` / `[autonomous] enabled`） | 起動時に1回読む | **意図的にやらない** |

したがってこの Issue のホットリロード作業は **`agents/*.md` のそれだけ**であり、
これは決定 1・8 の `agent_config` と**同じ機能の裏表**である:

- エージェントが自分でサブエージェント定義を書けるようになる（決定 1）。
- その定義が**再起動まで反映されない**なら、ツールは半分しか働いていない。
- 反映されるなら、決定 8 の書き込み時検証が**必須**になる（起動時検証が
  走らないから）。

`SubagentTool` の `agents: Vec<AgentDef>` を `Arc<RwLock<Vec<AgentDef>>>` にし、
`agents.rs` のローダーを呼び直す `reload()` を持たせる。`agent_config` の
`write` が成功したら `reload()` する（`tool_set` は `Arc` なので、
`AgentConfigTool` が `Weak<SubagentTool>` を持つ — `RefreshSystemPromptTool` が
`Weak<Agent>` を持つ形と同じ）。

host 設定のスイッチは**やらない**。`heartbeat_enabled = false` をエージェントが
自分で立てられるようにすると、エージェントが**自分の監視を止める**経路ができる
（`[autonomous] enabled` も同様）。この2つは運用者の決定であって、エージェントの
決定ではない。タスク定義ファイル（`enabled:` フラグ）と違って**エージェントに
渡す理由が無い**ので、読み直さない。

### 10. ワークスペーステンプレートに「管理者向け」の案内を足す

- `server/config.example.toml`: `[tools.admin]` の説明。**ホスト専用**であること、
  既定で無効であること、**許可したルームに書き込める人間はエージェントの
  無人実行を書き換えられる**ことを明記する（リスク 1）。
- `server/templates/workspace/config.toml`: `[tools.admin]` は**書けない**ことを
  1行で書く（ワークスペース層の allowlist に入れないため、書くと
  `rejected` として報告される）。既存の `[tools]` の扱いと同じ説明に揃える。
- `README.md`: `## Config tools` を短く。4本の名前、`[tools.admin].rooms`、
  「サブエージェントには `enabled` が無い」理由を2行で。

## 具体的な設計

### 新しい設定

```rust
// config.rs
#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct AdminToolsConfig {
    /// Channel-side room ids (`[room_profile.<n>].rooms` と同じ名前空間)
    /// in which the config tools are offered. Empty (the default) means
    /// the tools are not registered at all.
    #[serde(default)]
    pub rooms: Vec<String>,
}

// ToolsConfig に1フィールド:
#[serde(default)]
pub admin: AdminToolsConfig,
```

`Config` に参照用の1メソッドを足す（判定をここに集約する）:

```rust
impl Config {
    /// Whether the config tools may be used from `room_id`.
    ///
    /// `None` は「ルームを持たない経路」（`/rpc`・`/acp`・voice）で、
    /// 常に false。空の `rooms` も false — ツール自体が登録されないが、
    /// 登録後に設定を変えられた場合の二重の安全として同じ規則を持つ。
    pub fn config_tools_allowed_in(&self, room_id: Option<&str>) -> bool {
        match room_id {
            Some(r) => self.tools.admin.rooms.iter().any(|allowed| allowed == r),
            None => false,
        }
    }
}
```

### 新しいモジュール `server/src/tools/config_tools.rs`

```rust
/// どのディレクトリを操作するか。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConfigDir { Heartbeat, Autonomous, Agents }

impl ConfigDir {
    fn dir_name(self) -> &'static str;         // "heartbeat" | "autonomous" | "agents"
    fn tool_name(self) -> &'static str;        // "heartbeat_config" | ...
    fn supports_enabled(self) -> bool;         // Heartbeat | Autonomous
    /// ローダーを通した検証。パースエラーは anyhow::Error。
    fn validate(self, name: &str, raw: &str, profiles: &HashMap<String, ProfileConfig>) -> Result<()>;
}

/// 3本のツールの実装。`Tool` を1回実装して3回登録する。
pub struct ConfigTool {
    dir: ConfigDir,
    workspace_root: PathBuf,
    config: Config,          // profiles の検証とルーム判定に使う
    /// 書き込み成功後に呼ぶ。`Agents` 以外では `None`。
    subagent: Option<std::sync::Weak<crate::tools::subagent::SubagentTool>>,
    spec: ToolSpec,
}

/// この呼び出しが来たルーム。`None` はルームを持たない経路。
fn current_call_room() -> Option<String>;
```

`current_call_room()` は**2つの出所をこの順で見る**:

1. `crate::serve::current_turn_context()` の `room_id`（新設フィールド。決定 2・
   下記）。`/acp` と `/rpc` の `run_llm_turn` 経路がここを通る。
2. `crate::timer::current_origin()` が `TimerOrigin::Chat { room_id }` ならその
   値。Matrix / Discord の `Agent::handle_message` 経路はこれを scoped して
   いる（`agent.rs:758` 付近）ので、`agent.rs` を編集せずにルームが取れる。

どちらでも `None` なら「ルームを持たない経路」として拒否する。`agent.rs` に
手を入れないのは、**チャット経路が既にルームを知っている唯一の場所**だからで、
もう1本の task_local を足すことは二重管理になる。

### `TurnContext` に `room_id` を足す

```rust
pub(crate) struct TurnContext {
    // ...既存...
    pub session_id: Option<String>,
    /// このターンの元になったルーム id。`run_llm_turn` がセッションの
    /// メタから引いて入れる。`None` は「ルームを持たない」— ACP の遅延
    /// セッションや、まだメタを持たないセッション。設定ツールの
    /// ルーム許可判定だけが読む。
    pub room_id: Option<String>,
}
```

`room_id` は `run_llm_turn` の 3a 節（`session_room_metadata` /
`store_for_session` の近く）で、既に取っている `room_info` と同じ場所から
引けるはずで、**新しい探索を足さない**。取れない場合（メタが無い＝ファイルが
まだ無い）は `None` にして、ツールは拒否する（安全側）。

### `task_test` の実装

```rust
pub struct TaskTestTool {
    state: Arc<crate::serve::ServeState>,
    workspace_root: PathBuf,
    spec: ToolSpec,
}
```

- 判定は `state.config.config_tools_allowed_in(current_call_room())`。
- セッションは `state.autonomous_session_store.create_autonomous_session(&room_id, ns)`
  を**そのまま使う**。`create_autonomous_session` は今 `room_id` を引数に取るので、
  テスト名を渡すだけでよい（`session.rs` の変更は不要）。
- ターン実行は `crate::serve::run_llm_turn(state, session_id, ChatMessage::user(text),
  Arc::new(AutonomousHost { origin }), None)`。origin は本番と同じ規則
  （`[autonomous] origin`、既定 `channel`）を使う。**テストだけ緩い権限で
  走らせない**のが要点 — 本番で `shell` が要るなら、テストも同じ origin で
  「拒否される」ことを確認できるべきだから。
- `heartbeat_config` のテストは1ターン固定。body をそのまま user メッセージに
  し、先頭に `[Heartbeat: <name>]` を付ける（本番の `fire_task` と同じ前置。
  `heartbeat.rs` の `format!("[Heartbeat: {name}]\n\n{body}")` と揃える）。
- autonomous のテストは `[Autonomous: <name>]` + body、2ターン目以降は
  `CONTINUE_PROMPT`（本番の `run_task` と同じ規則。**同じ規則を使うことが
  「テストで通った＝本番で通る」の根拠**）。

### ツールの登録（`main.rs`）

```rust
// 設定ツール。ルーム許可リストが空なら4本とも登録しない。
if !config.tools.admin.rooms.is_empty() {
    let subagent_weak = /* SubagentTool 登録時に Arc を作って保持 */;
    for dir in [ConfigDir::Heartbeat, ConfigDir::Autonomous, ConfigDir::Agents] { ... }
    tool_set.register_tool(Box::new(TaskTestTool::new(...))).await;
}
```

`SubagentTool` は現在 `Box::new(SubagentTool::new(defs))` を直接登録しているので、
`Arc<SubagentTool>` を作り、`Box::new(Arc::clone(&t))` で登録する形へ変える
（`SkillTool` が既にその形になっている — `main.rs:544` 付近）。これで
`Weak<SubagentTool>` を `agent_config` に渡せる。

### `visible_tool_predicate` との関係

**設定ツールは `visible_tool_predicate` では隠さない。** ルーム許可判定は
その関数の引数（bool 5つ）では表現できない — ルーム id は文字列であり、
「どのターンか」を知っているのは `run_llm_turn` の内側である。

代わりに登録段階で隠す: `rooms` が空なら登録されない。`rooms` が非空でも、
**許可されていないルームからは呼べるが拒否される**（`ToolSet::specs_filtered`
の思想からは一段劣るが、`NotOffered` にするには `run_llm_turn` の
spec フィルタ closure にルーム情報を渡す必要があり、`visible_tool_predicate` の
シグネチャを6引数に増やすことになる。閉じた経路を1つ増やすより、実行時の
明快な拒否メッセージのほうが、**設定ミスに気づきやすい**という判断。

拒否メッセージは専用の文言にする（`Refusal` の3種では伝わらない）:

```
Permission denied: the config tools are not available in this room.
An operator can allow them with `[tools.admin].rooms`.
```

## 受け入れ基準

1. `[tools.admin].rooms` を書かない限り、`heartbeat_config` /
   `autonomous_config` / `agent_config` / `task_test` は **`ToolSet` に
   登録されない**（`every_tool_declares_its_kind` の期待表にも現れない）。
2. `rooms = ["!ops:x"]` を書くと4本が登録され、`!ops:x` からの呼び出しは
   通り、他のルームからは拒否メッセージが返る。`/rpc`・`/acp`・voice からも
   拒否される。
3. `heartbeat_config action=list` が `<workspace>/heartbeat/*.md` のファイル名と
   `enabled` を返す。`read` が全文を返す。
4. `set_enabled` は `enabled:` 行だけを書き換え、**body と他の frontmatter
   行・コメントが1バイトも変わらない**。`enabled:` が無ければ閉じ `---` の
   直前に挿入される。`enabled:` がインデントされた行（`voice:` の中）は
   触らない。
5. `set_enabled` で `false` にした heartbeat タスクは次に cron が来ても発火せず、
   `true` に戻すと**再起動なしで**発火する（`run_cron` が毎ループ読むため。
   本 Issue のテストで固定する）。
6. `autonomous_config action=set_enabled false` のあと、自律ループはその
   タスクを選ばない。`true` に戻すと `poll_seconds` 以内に選ばれうる。
7. `task_test kind=autonomous name=<enabled: false のタスク>` が
   **無効なまま**テストを1回走らせ、最終テキスト・ターン数・セッションの相対パスを
   返す。そのセッションは `session_read` で読める。
8. **テストは本番のクールダウンをリセットしない。** `task_test` を実行した後も
   `is_due` のアンカーがテスト前と同じであること（テスト用セッションの
   `room_id` が `test:<kind>:<name>` であるため）。
9. `agent_config action=write` は、`description` の無い定義・壊れた YAML・
   未知の `profile:` を**書かずにエラーを返す**。正常な定義は書け、
   **再起動なしで** `subagent` から呼べる（ホットリロード）。
10. `agent_config action=set_enabled` は「このツールでは使えない」を返し、
    ファイルを書かない。`agents/*.md` に `enabled` フラグは追加されない。
11. `heartbeat_config action=write` は、cron として解釈できない `schedule:`、
    frontmatter の無い内容、空の body を書かずにエラーを返す。
    `autonomous_config action=write` も同じ（body 空・frontmatter 無し）。
12. `name` に `..` / `/` / 先頭 `.` / 空文字を渡すと拒否され、ワークスペースの
    外にも `<workspace>/{heartbeat,autonomous,agents}/` の外にもファイルが
    作られない。
13. 既存の挙動が変わらない: `file_write` / `shell` / `subagent` / heartbeat /
    autonomous の既存テストが全通過。`[tools.admin]` を持たない既存の設定
    ファイルがそのまま動く。

## テスト

`config_tools.rs` の `mod tests` に置く（ソース内テストの既存方針）。

- **ルーム判定**: `config_tools_allowed_in` が空リスト・一致・不一致・`None` の
  4通りで正しい。`current_call_room()` が TurnContext 経由 / timer origin 経由 /
  どちらも無い、の3通りを返す。
- **`set_enabled` の行編集**: コメント入りの frontmatter を持つ実ファイルで、
  書き換え後の全文が「`enabled:` 行以外は同一」であることを文字列比較で固定する。
  挿入ケース、インデントされた `enabled:` を無視するケース、frontmatter 無しで
  `None` を返すケース。
- **パス検証**: `..`・`/`・先頭 `.`・空を弾く。`foo.md` が `foo` に正規化される。
- **検証**: heartbeat の不正 cron、空 body、frontmatter 無し。autonomous の
  空 body。agents の description 欠落・未知プロファイル（`validate_subagent_profiles`
  を再利用しているので、プロファイルのケースは既存関数のテストを共有する）。
- **`task_test`**: `StubProvider` を刺した `ServeState` で1サイクル走らせ、
  (a) セッションが自律ストアに作られ閉じられる、(b) `room_id` が
  `test:autonomous:<name>` である、(c) **`is_due` が本番タスクのアンカーを
  見ていない**（テスト前後で `is_due` の結果が変わらない）、(d) `enabled: false`
  のタスクでも走る、(e) heartbeat は1ターンで終わる。
- **ホットリロード**: `SubagentTool` に定義を1つ書いて `reload()` した後、
  `subagent` の spec 説明文にその名前が現れる（`build_spec` の出力で確認）。
- **登録**: `rooms` が空なら4本が `ToolSet` に無い。非空なら4本あり、
  `kinds()` がすべて `Edit` を返す（`every_tool_declares_its_kind` の
  期待表への追記は**別テスト**にする。`default_tool_set` は `[tools.admin]` を
  知らないので、既存の期待表は空リストのケースのまま無変更で通るはずである）。

## リスクと未解決

1. **許可ルームは信頼境界そのものである。** `[tools.admin].rooms` に入れた
   ルームに書き込める人間は、エージェント経由で heartbeat / autonomous の
   定義を書き換えられる。autonomous は無人で走り、`[autonomous] origin =
   "trusted"` + `host_access.enabled` なら `shell` にも届く。これは
   `Origin::Channel` の doc が既に名指ししている穴と同型で、#257 の
   「ファイル編集の許可制」が入るまでの既知のリスク。既定オフと
   `allowed_users` の併用が運用上の答え。**解決しない**（決定 2・5）。
2. **`write` はモデルが書いた frontmatter をそのまま受け取る。** 決定 3 の
   行編集は `set_enabled` にしか効かない。`write` で定義を書き直せば、
   人間が書いたコメントは消える。これは「全文置換」という action の意味
   そのものなので受け入れるが、ツールの description に
   「`set_enabled` で足りるなら `write` を使わない」と書く。
3. **テストと本番の規則が同一であることの担保はテストでしかない。** 
   `task_test` は本番の `run_task` と別の実装である（同じ
   `run_llm_turn` と同じプロンプト組み立てを使うが、コードは共有していない）。
   プロンプト組み立て（`marker` / `CONTINUE_PROMPT`）を
   `autonomous.rs` の公開関数から呼ぶことでずれを小さくするが、
   完全な共有は v1 ではやらない。
4. **`agents/*.md` のホットリロードは定義の増減を反映するが、
   `SubagentCache` に残った子会話のハンドルは古い定義の名前を指しうる。**
   2026-09-11 決定 4 のとおり resume は履歴がモデル非依存なので動く。
   定義を消した後の resume はハンドルが残っていれば成功する（それは
   望ましい — 消したのは定義であって会話ではない）。
5. **`[tools.admin]` をワークスペース層から書かせない判断は、
   「リモートワークスペース同期で管理ルームを配りたい」という将来要求と
   衝突しうる。** そのときは `[room_profile.<n>].rooms` と違って
   「誰が管理者か」の委譲になるので、別の設計（ホスト設定の明示的な
   オプトイン）が要る。今は入れない。
6. **`session_label` の見た目**: `task_test` のセッションは
   `server/test:autonomous:<name>` というラベルになる。読みにくいが、
   `room_id` に `:` を使うのは既存の自律セッション（`room_id` = タスク名）と
   同じ流儀で、テスト由来であることが一目で分かる利点を取る。
7. **`[autonomous] enabled = false` の deployment で `task_test
   kind=autonomous` を使えるか。** 使えるようにする（`ServeState` に
   autonomous ストアは常にある。ループが動いているかは関係ない）。
   ループが無効でもタスクの中身を試せるのはテストの目的に合う。