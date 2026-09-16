# シェル・ファイル系ツールのセッション種別ルーティング統合

- **Issue**: [#270](https://github.com/fluo10/sapphire-agent/issues/270)
  （[#261](https://github.com/fluo10/sapphire-agent/issues/261) のプレフィクス分離案と
  [#262](https://github.com/fluo10/sapphire-agent/issues/262) の「読み取り専用の
  クライアント側ディレクトリ一覧」を吸収する。本実装の完了時に両者を close する）
- **対象**: `server/src/tools/builtin_tools.rs`、`server/src/tools/client_tools.rs`、
  `server/src/tools/client_exec.rs`、`server/src/tools/skill_tools.rs`（共有ヘルパの
  移送のみ）、`server/src/tools/policy.rs`、`server/src/tools/mod.rs`、
  `server/src/serve/mod.rs`、`server/src/agent.rs`（コメントのみ）、
  `server/config.example.toml`、`server/templates/workspace/config.toml`、
  `README.md` / `README.ja.md`
- **前提**: `2026-09-01-client-side-tools-design.md`（ACP クライアント側ツール一式）が
  実装済み。`AcpClient` trait、`scope_acp_client` / `current_acp_client`、
  `TerminalRegistry`（セッション毎端末上限8）、`client_exec::run_client_command` が
  そのまま土台になる。

## なぜ

いま同じ仕事をするツールが2セットある。

| セット | ツール | 触るマシン | 可視性 |
|---|---|---|---|
| ホスト側 | `file_read`, `file_write`, `file_append`, `file_delete`, `dir_list`, `dir_walk`, `shell` | エージェント自身 | `[tools.host_access] enabled` が真のときだけ |
| クライアント側 | `client_file_read`, `client_file_write`, `client_shell`, `client_shell_start`, `client_shell_output`, `client_shell_kill` | 接続中のエディタ | ACP セッションかつ当該 capability 宣言時のみ |

ACP セッションで両方が同時に見えることがある（`host_access = true` のとき）。どちらが
どちらのマシンかは**ツール説明文だけ**が区別しており、モデルが `file_read` を選んで
エージェント自身のファイルを読む／`client_file_read` を選ぶ、という選択を毎回させている。
人間にとっては「この会話はエディタのプロジェクトについての会話だ」という1つの事実で
しかないものが、モデルには6つの名前の選択として届いている。

`#262`（クライアント側 `dir_list`）が示すとおり、クライアント側の面はファイル読み書き
だけでは完結しない。ACP に存在しない操作まで含めて「どちらのマシンか」を1つの軸に
まとめ直すのが本変更である。

## 決めたこと

### 1. ツール名は現行の7つを維持し、クライアント側は同じ名前に統合する

| 統合後の名前 | 由来 |
|---|---|
| `file_read` | ホスト側 `file_read` ＋ 旧 `client_file_read` |
| `file_write` | ホスト側 `file_write` ＋ 旧 `client_file_write` |
| `file_append` | ホスト側 `file_append`（ACP では read+連結+write で再現） |
| `file_delete` | ホスト側 `file_delete`（ACP では terminal 経由） |
| `dir_list` | ホスト側 `dir_list`（ACP では terminal 経由） |
| `dir_walk` | ホスト側 `dir_walk`（ACP では terminal 経由） |
| `shell` | ホスト側 `shell` ＋ 旧 `client_shell` |
| `shell_start` | 旧 `client_shell_start` |
| `shell_output` | 旧 `client_shell_output` |
| `shell_kill` | 旧 `client_shell_kill` |

**`client_` / `agent_` / `user_` プレフィクスは使わない。** どのマシンに届くかは
セッション種別が決めるのであって、名前が決めることではない。上の3つの
`shell_*` は「タイムアウトしない長寿命コマンドを扱う」ための兄弟ツールで、
`client_shell_*` から `client_` を落とすだけの改名である（この3つだけは
非 ACP セッションでは提供しない — §4）。

`ToolKind` は現行どおり: `file_read` = `Read`、`file_write` / `file_append` = `Edit`、
`file_delete` = `Delete`、`dir_list` / `dir_walk` = `Search`、`shell` / `shell_start` /
`shell_kill` = `Execute`、`shell_output` = `Read`。権限表（`policy::decide`）には
一切手を入れない。

### 2. どちらのマシンに届くかは、セッション種別だけが決める

```
current_acp_client() -> Some(client)  =>  クライアント（エディタ）のマシン
                     -> None          =>  エージェント自身のマシン
```

判定は**ツール実行時**に、既存の task-local（`server/src/tools/acp_client.rs` の
`ACP_CLIENT_TL` / `scope_acp_client` / `current_acp_client`）を読んで行う。
その task-local は既に `TurnLoop::run` のツール実行（`server/src/serve/mod.rs`、
許可されたコールを `join_all` で回す箇所）で張られており、ACP セッションの
ターンでのみ `Some` になる。**新しいルーティング機構は作らない** —
`scope_acp_client` の適用範囲を `client_shell` 群から統合後の全ツールへ
広げる（というより、呼び出し側の分岐を追加しない）だけである。

| セッション | 例 | 届く先 |
|---|---|---|
| ACP | `/acp`（Zed 等） | 接続中エディタのマシン（`fs/*`、`terminal/*`） |
| 非 ACP | Matrix、Discord、`/rpc`、voice、`/a2a`、heartbeat、autonomous | エージェント自身のマシン |

**サブエージェントも同じ規則に従う。** ネストしたターンの `TurnHost` は
`SubagentHost`（`server/src/tools/subagent.rs`）で、`acp_client()` を親に転送して
いる。ツール実行を包む `scope_acp_client` は `TurnLoop::run` 側にあるため、
ACP セッションから委派されたサブエージェントのツール呼び出しもクライアントへ届く。
これは「委派は権限ゲートを迂回する経路になってはならない」という既存方針と
同じ向きの性質であり、仕様として明記する。

### 3. ツール別の実行内容

| ツール | 非 ACP（エージェント自身） | ACP（クライアント） | ACP で使う呼び出し |
|---|---|---|---|
| `file_read` | 現行のまま（`WorkspaceState::read_file`、`N\|content` 形式、`offset`/`limit`、device path 拒否） | `fs/read_text_file`。`offset`→`line`、`limit`→`limit` に写像。戻り値はそのまま | 1 × `fs/read_text_file` |
| `file_write` | 現行のまま（`write_file`、親ディレクトリ自動作成、`refuse_if_sensitive`） | `fs/write_text_file`（既存内容を完全置換） | 1 × `fs/write_text_file` |
| `file_append` | 現行のまま（`append_file`） | `fs/read_text_file`（`line`/`limit` なし＝全体）→ 連結 → `fs/write_text_file` | 1 read ＋ 1 write |
| `file_delete` | 現行のまま（`delete_file`、`refuse_if_sensitive`） | terminal 上で `rm` | 1 × terminal |
| `dir_list` | 現行のまま（`list_dir`、ソート済み、ディレクトリは末尾 `/`） | terminal 上で `find -maxdepth 1` | 1 × terminal |
| `dir_walk` | 現行のまま（`walk_recurse`、`max_depth`/`max_entries`） | terminal 上で `find -maxdepth N` | 1 × terminal |
| `shell` | 現行のまま（`$SHELL` または `/bin/sh` を `-c`、`timeout`、`kill_on_drop`） | `terminal/create` → `wait_for_terminal_exit` → `terminal/output` → `release`（既存 `run_client_command`） | 上記4つ |
| `shell_start` | **提供しない** | `terminal/create` ＋ セッション別 tracking | 1 × `terminal/create` |
| `shell_output` | **提供しない** | `terminal/output` | 1 × `terminal/output` |
| `shell_kill` | **提供しない** | `terminal/kill` ＋ `terminal/release`、無条件で untrack | 2 × terminal |

入力スキーマは**現行のホスト側スキーマを唯一の正**とする（`path` / `offset` / `limit` /
`content` / `max_depth` / `max_entries` / `command` / `shell` / `timeout` / `workdir`）。
ACP 側に都合のよい別名（`line` など）はモデルに見せない。`shell` の
`timeout`（既定60、上限600秒）は ACP 経路では `run_client_command` の待ち時間に
そのまま対応し、**タイムアウトしても殺さない**という既存の判断（`client_shell` の
設計）を維持する。タイムアウト時の文言は `shell_output` / `shell_kill` を指す。

### 4. capability による可視性（フォールバックはしない）

ツールは**セッション種別で可視性を決め、capability で更に絞る**。

非 ACP セッション（`has_client == false`）:

- 上の7つ（`file_read` 〜 `shell`）は `[tools.host_access] enabled = true` のときだけ見える。
- `shell_start` / `shell_output` / `shell_kill` は**決して見えない**（エージェント側には
  長寿命ハンドルを保持する機構が無い）。

ACP セッション（`has_client == true`）:

| ツール | 必要な capability |
|---|---|
| `file_read` | `fs.read_text_file` |
| `file_write` | `fs.write_text_file` |
| `file_append` | `fs.read_text_file` **かつ** `fs.write_text_file` |
| `file_delete`, `dir_list`, `dir_walk`, `shell`, `shell_start`, `shell_output`, `shell_kill` | `terminal` |

- **`host_access` は ACP セッションの可否に影響しない。** ACP セッションで
  エージェント自身のファイルシステムに触る経路は無い。これが本設計の中心で、
  「エディタで開いているプロジェクトを触る」という1つの意味に閉じる。
- **capability が無ければ、そのツールは提示されない。** エージェント自身の
  マシンへフォールバックしない（確定方針）。理由は2つ: (a) モデルが
  「たまたま別のマシンを触った」ことに気づけない経路を作らない、(b) `#262` が
  却下した「プロトコルの裏付けが無い操作を黙って代用する」構図と同じだから。
- 実行時にエージェント側へ落ちることは無い。`current_acp_client()` が `None` なら
  常にエージェント経路である（＝そのターンが非 ACP だったということ）。
  したがって旧 `no_editor_error`（「no editor is connected…」）は7ツールから消える。
- **部分的 capability は部分的にしか見えない。** `fs.read` だけ宣言したエディタには
  `file_read` だけ、`fs.write` だけなら `file_write` だけ、`terminal` が無ければ
  `file_delete` / `dir_list` / `dir_walk` / `shell` 群も無い。往復して知るのでは
  なく、最初から一覧に無い（既存方針の維持）。
- 接続断・RPC エラーはクライアントのエラーとしてモデルに返る（フォールバック無し）。

### 5. `host_access` の意味を変える

現状 `policy::host_tool_denied(name, host_access_enabled)` は `HOST_TOOLS`
（7つ）を全 origin で拒否する。統合後は**「エージェント自身のマシンへルーティング
されるコール」にだけ効くゲート**になる。

```rust
pub fn host_tool_denied(name: &str, host_access_enabled: bool, routed_to_client: bool) -> bool {
    !host_access_enabled && !routed_to_client && HOST_TOOLS.contains(&name)
}
```

- `routed_to_client` は「このターンに ACP クライアントが付いているか」。
  ACP セッションでは常に `false` を返す（＝ゲートは何も拒否しない。可視性は
  capability が、実行先は `current_acp_client()` が決める）。
- `visible_tool_predicate(host_access_enabled, has_client, client_fs_read,
  client_fs_write, client_terminal)` の本体:

```rust
move |name: &str| {
    if crate::tools::policy::host_tool_denied(name, host_access_enabled, has_client) {
        return false;
    }
    // 非 ACP: ここまで生き残ったツールはエージェント自身のもの。
    // host_access が唯一のゲートである。
    if !has_client {
        return true;
    }
    match name {
        "file_read" => client_fs_read,
        "file_write" => client_fs_write,
        "file_append" => client_fs_read && client_fs_write,
        "file_delete" | "dir_list" | "dir_walk" | "shell" | "shell_start"
        | "shell_output" | "shell_kill" => client_terminal,
        // 既存の skill 群は変更しない。
        "skill" | "skill_install" | "skill_update" | "skill_uninstall" => client_terminal,
        _ => true,
    }
}
```

- `TurnLoop::run` の許可ゲート（`server/src/serve/mod.rs`）も同じ
  `routed_to_client` を渡す。`routed_to_client = progress.acp_client().is_some()`
  をラウンドごとに1回計算し、`host_tool_denied` と `partition_without_asking` に渡す。
- `partition_without_asking(origin, calls, kinds, host_access_enabled, routed_to_client)`
  は Matrix/Discord 経路（`server/src/agent.rs`）から `routed_to_client = false` で
  呼ばれる — **チャネル経路の挙動は変わらない**。
- `Refusal::Unavailable` の文言は変更しない。

### 6. ACP に無い操作の実装（terminal 経由、bash 前提）

ACP の agent→client 面は `session/request_permission`、`fs/*`、`terminal/*` のみで、
ディレクトリ列挙・削除・stat は無い（既存 design の結論を維持）。統合にあたり、
`#262` が「クライアント側 `dir_list` は有用」と示した分を terminal 経由で実装する。

- 実行形式は `bash -c <script> <argv0> <args...>`。パスは**位置引数**で渡す
  （ACP の `terminal/create` はコマンドと引数を分けて送るので、シェルクォートの
  必要が無い）。`bash` と `find` / `sort` / `head` / `rm` があることを前提とする
  （無ければコマンドが失敗し、その終了コードがエラーとしてモデルに返る）。
- **`file_delete`**: `[ -e ]` / `[ -d ]` を先に検査し、ディレクトリなら拒否、
  無ければ非ゼロ終了。存在するファイルに対して `rm -- "$1"`。
- **`dir_list`**: `find "$1" -mindepth 1 -maxdepth 1 | LC_ALL=C sort` の各要素を
  `[ -d ]` で判定し、`D\t<path>` / `F\t<path>` の行として出力。
- **`dir_walk`**: 同じ判定で `find "$1" -mindepth 1 -maxdepth $((max_depth + 1))`
  （エージェント側の「`max_depth = 0` は直下のみ」に合わせた +1）を通し、
  `LC_ALL=C sort | head -n $((max_entries + 1))` で打ち切ってから各要素を判定。
  取得件数が `max_entries` を超えたら、エージェント側と同じ
  `[truncated — more than N entries; raise max_entries or narrow path]` を付ける。
- 出力の解釈はツール側で行い、**モデルに見える形式はエージェント側と同一**にする
  （ソート済み、ディレクトリは末尾 `/`、空なら `(empty) <path>`）。
  これによりプロンプト上の「dir_list はこう返る」がセッション種別で変わらない。
- 出力は 50,000 バイト（`OUTPUT_CAP_BYTES`）を `terminal/create` の
  `output_byte_limit` として要求する（既存 `run_client_command` の挙動）。
- 待ち時間は短い固定値（`CLIENT_LOCAL_TIMEOUT = 30s`、`skill_tools` の
  `LOCAL_TIMEOUT` と同じ値）。**タイムアウトした場合は端末ハンドルを名指しした
  エラー**を返し、`shell_kill` で解放できることを伝える（黙ってハンドルを
  リークさせない。セッション上限8に数えられるため）。
- **ファイル名に改行を含むパスは非対応**（行区切りで受け渡すため）。ACP 経路の
  既知の制約として明記する。エージェント側は Rust 実装なので影響しない。
- `file_append` の ACP 経路は read＋連結＋write で、**原子的ではない**うえ
  ファイル全体が往復する。親ディレクトリが無ければ `fs/write_text_file` の失敗が
  そのまま返る（`mkdir` はしない）。エージェント側 `append_file` との差として明記する。
- `refuse_if_sensitive`（`/etc` 等、`config.toml`、`acp-permissions.json`）は
  **エージェント側経路にだけ適用する**。クライアントのパスはクライアントの管轄で、
  ACP 側の権限プロンプトが受け持つ。ACP 経路にこのガードを持ち込むと、
  「エディタで開いているプロジェクトの `/etc` 風パス」を誤って拒否しうる。

### 7. 説明文

統合後の10ツール全ての説明文を、同じ1文で始まる形に書き換える。

> Reaches whichever machine this conversation is about: inside an ACP session, the
> machine the connected editor is running on; otherwise this agent's own machine.

そのうえでツール固有の説明を続ける。旧 `client_*` の説明にあった
「Only available inside an ACP session whose editor supports `fs/read_text_file`;
refuses otherwise.」は**削除する**（今の可視性ルールでは、非対応のエディタには
そもそも提示されないため、この文は嘘になる）。ACP のみに存在する
`shell_start` / `shell_output` / `shell_kill` は、その旨（エディタ上でのみ動く）を
明記したままとする。`file_delete` / `dir_list` / `dir_walk` の ACP 経路が
bash を前提とすることも説明文に1文入れる。

### 8. 非互換（意図的なもの）

- **旧名は消える**: `client_file_read` → `file_read`、`client_file_write` →
  `file_write`、`client_shell` → `shell`、`client_shell_start` → `shell_start`、
  `client_shell_output` → `shell_output`、`client_shell_kill` → `shell_kill`。
- サブエージェント定義（`<workspace>/agents/*.md` の `tools:`）や設定の
  許可リストに旧名が書かれていると、その名前は**単に見えなくなる**（既存の
  `newly_unknown_tools` が `warn!` する。沈黙ではなく警告される）。
- `acp-permissions.json` の `always_allow` / `always_reject` は**ツール名**で
  キーされている。旧名の grant は死んだ項目になり、新名は一度だけ尋ねられる。
  マイグレーションコードは書かない（自動移行は「知らない名前の権限を引き継ぐ」
  ことになり、統合で意味が変わったツールに対して危険側に倒れる）。
- ACP セッションで `host_access = true` にしていた運用は、**エージェント自身の
  マシンに触れなくなる**。これが本変更の目的そのもの。エージェント自身の
  ファイルを触りたい会話は `/rpc` や Matrix から行う。
- `#261`（プレフィクス分離）と `#262`（読み取り専用クライアント側 dir ツール）は
  本変更に吸収され、実装完了時に close する。

## テスト方針

- **ルーティング（ACP 経路）**: `acp_client::tests::FakeClient` を作り、
  `scope_acp_client` で包んだ状態で各ツールを実行し、クライアント側の
  メソッド（`read_text_file` / `write_text_file` / `create_terminal`）が呼ばれ、
  **エージェント側のファイルシステムに触れていないこと**を検証する。
  `file_read` は「ワークスペース上のファイル内容と、FakeClient が返す内容を
  別物にしておき、結果が FakeClient 側になる」形で固定する。
- **terminal 経由の3ツール**: FakeClient に流すスクリプト文字列を検査し
  （`rm --`、`-maxdepth`、`max_entries + 1`）、FakeClient が返す
  `D\t…` / `F\t…` の標準出力が、エージェント側と同じ整形結果になることを
  検証する。空ディレクトリの `(empty) <path>`、`max_entries` 超過時の
  truncation マーカーも個別に固定する。
- **ルーティング（非 ACP 経路）**: 既存の `builtin_tools` のテストは
  `scope_acp_client` を張らないので、そのままエージェント経路の回帰テストになる。
- **可視性**: `server/src/serve/mod.rs` の `client_filtering_test_set` /
  `tool_names_for_turn` を統合後の名前で書き換える。ACP ＋ capability の
  組み合わせ（`fs.read` のみ、`fs.write` のみ、`terminal` のみ、全部）で
  期待される名前集合を固定する。非 ACP では `host_access` on/off の2通りで
  「7つが見える／見えない」を固定する。
- **host_access の意味変更**: `policy.rs` の `host_tool_denied` テストを新引数で
  書き換える（`routed_to_client = true` なら `enabled = false` でも拒否しない）。
  `partition_without_asking` のチャネル経路テストは `false` を渡す形にして
  **期待値は現状のまま**通ることを確認する（チャネル挙動が変わっていないことの証拠）。
- **サブエージェント**: ACP セッションから委派したサブエージェントのツール呼び出しが
  クライアントへ届くことを1本のテストで固定する（`SubagentHost` が `acp_client()`
  を転送していることの回帰防止）。

## 受け入れ基準

1. `server/src/tools/mod.rs` の `default_tool_set()` に登録されるシェル・ファイル系
   ツールが10個（`file_read`、`file_write`、`file_append`、`file_delete`、
   `dir_list`、`dir_walk`、`shell`、`shell_start`、`shell_output`、`shell_kill`）で、
   名前に `client_` / `agent_` / `user_` を含むものが1つも無い。
2. ACP セッションで `host_access = true` でも、`file_read` は**エディタ側**の
   内容を返す。
3. 非 ACP セッション（`/rpc`、Matrix、voice、`/a2a`、heartbeat、autonomous）で
   `host_access = true` なら `file_read` は**エージェント自身**のファイルを返し、
   `host_access = false` なら7ツールは一覧に現れない。
4. ACP セッションで `fs.read` を宣言しないエディタには `file_read` が見えず、
   呼んでもエージェント自身のファイルを読まない（フォールバック無し）。
5. `dir_list` / `dir_walk` が ACP セッションで動き、出力形式が非 ACP と同一である
   （ソート済み、ディレクトリは末尾 `/`、空なら `(empty) <path>`、打ち切り時は
   同じマーカー）。
6. `cargo fmt --all -- --check`、`cargo clippy --workspace --all-targets`、
   `cargo test --workspace` が通る。
7. `README.md` / `README.ja.md` / `server/config.example.toml` /
   `server/templates/workspace/config.toml` に旧 `client_*` 名の記述が残っていない。

## リスクと未解決点

- **クライアント側の `bash` / `find` / `sort` / `head` / `rm` 前提**。無い環境では
  `dir_list` / `dir_walk` / `file_delete` は失敗する（フォールバックしないという
  決定の帰結）。エラーメッセージは「クライアント側でコマンドが失敗した」ことが
  読める形にする。`file_delete` は `rm` 単体なので実質 POSIX 前提と同じ。
- **`find -maxdepth` と `--` の移植性**。macOS の BSD `find` も
  `-mindepth` / `-maxdepth` を持つ。`head -n` は BSD/GNU 共通。`LC_ALL=C` は両方で
  効く。改行を含むパスは非対応（§6）。
- **改行入りパス**は ACP 経路の `dir_list` / `dir_walk` で区別できない。行区切りを
  やめて NUL 区切りにするとクライアント側 `sort -z` の移植性を新たに背負うため、
  今回は行区切りを選び、制約として明記する。
- **`file_append` の非原子性**（read＋write の間の他プロセス更新を失う）。エージェント
  側 `append_file` との差として説明文に残す。
- **`shell_start` / `shell_output` / `shell_kill` の改名はユーザーの既存プロンプトや
  記憶に影響する**。README とテンプレートの追随で緩和するが、既存セッションの
  履歴に残る旧名の呼び出しは（モデルが再度呼ばない限り）そのまま残る。
- **`host_access` の意味変更は設定例のコメントを書き換えないと誤読される**。
  `config.example.toml` の該当ブロック（約319〜330行、約421行）を必ず追随させる。
