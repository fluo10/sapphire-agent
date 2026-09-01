# クライアント側のファイルシステムとシェルを ACP 経由で使う

- **対象**: `src/serve/acp.rs`、`src/tools/`（新規モジュール）、`src/tools/policy.rs`、
  `src/config.rs`、`src/serve/mod.rs`
- **前提**: `session/request_permission`（#186）がマージ済み — エージェント→クライアント
  方向の往復は実証済み

## なぜ

エディタで開いているプロジェクトを触れないコーディングエージェントは、
コーディングエージェントではない。今 `file_read` も `shell` も**サーバー自身のホスト**を
見ており、Zed が動いている端末とは別のマシンでありうる。

そして ACP はこのためのメソッドを持っている。使っていないだけ。

## ACP の agent→client 面は、これで全部である

`agent-client-protocol` 2.0.0 / `-schema` 1.5.0 で確認した。

| メソッド | 主な引数 / 応答 |
|---|---|
| `session/request_permission` | **実装済み**（#186） |
| `fs/read_text_file` | `{session_id, path, line?, limit?}` |
| `fs/write_text_file` | `{session_id, path, content}` |
| `terminal/create` | `{session_id, command, args, env, cwd?, output_byte_limit?}` → `{terminal_id}` |
| `terminal/output` | → `{output, truncated, exit_status?}` |
| `terminal/wait_for_exit` | → `{exit_status}` |
| `terminal/kill` / `terminal/release` | — |

**ディレクトリ列挙・削除・stat・rename・glob に相当するものは無い。** これは
「まだ無い」ではなく「無い」— この設計はその前提で立てる。

## 決めたこと

### 1. クライアント側ツールは6つ

| ツール | ACP 呼び出し | `ToolKind` |
|---|---|---|
| `client_file_read` | `fs/read_text_file` | `Read` |
| `client_file_write` | `fs/write_text_file` | `Edit` |
| `client_shell` | create → wait_for_exit → output → release | `Execute` |
| `client_shell_start` | create | `Execute` |
| `client_shell_output` | output | `Read` |
| `client_shell_kill` | kill + release | `Execute` |

`client_file_read` は `line` / `limit` をそのまま通す。ACP がこの2つを持っているのは
偶然ではなく、コーディングエージェントが大きなファイルを部分的に読むためで、
サーバー側の `file_read` には無い利点である。

### 2. クライアント側のディレクトリ列挙・削除・追記は作らない

ACP にその面が無い。そして `ls` / `find` / `rm` / `>>` はシェルコマンドである。

シェルの上に構造化ラッパを重ねれば、**プロトコルの裏付けが無い第二の規約**を
発明することになる — 出力の書式を自分で決め、クライアントごとの `ls` の差異を
吸収し、それを未来にわたって保守する。説明文で「一覧はシェルで」と誘導するほうが
安く、壊れにくい。

サーバー側の `dir_list` / `dir_walk` / `file_delete` / `file_append` は残る。
それらはサーバー自身のホストを見るツールとして、引き続き意味がある。

### 3. 一発シェルはタイムアウトするが、**殺さない**

`client_shell` は既定120秒、呼び出しごとに上書き可、上限600秒。

**タイムアウトしてもプロセスは走らせたまま、ハンドルを返す。**

```
[timed out after 120s — still running as terminal a1b2c3.
 Use client_shell_output to check on it, or client_shell_kill to stop it.]
```

殺して返すと、モデルは同じコマンドをライフサイクル版で**やり直す**。2分かけた
ビルドを捨ててもう一度払うことになり、冪等でないコマンド（`git push`、
マイグレーション、ファイルを書くスクリプト）なら**二重実行**になる。

タイムアウトは「失敗」ではなく「一発で待ちきれなかった」であり、
一発ツールがその場でライフサイクル版に化けるのが、実際に起きていることに一番近い。

**戻り値がハンドルになりうる以上、メッセージは「まだ走っている」ことを明示する。**
モデルが失敗と読み違えて再実行するのが、この設計で唯一の新しい危険。

**プロトコル自身の示唆とは異なる選択である。** スキーマの `terminal/kill` の説明は
*"helpful when implementing command timeouts which terminate the command as soon as
elapsed, and then get the final output"* — つまり「タイムアウトしたら殺して出力を取る」を
想定している。それを採らないのは上の理由による。`terminal/kill` は
`TerminalId` を有効なまま残すので、殺す形も選べるが、殺す判断はモデル
（`client_shell_kill`）か人間に委ねるほうが、こちらが勝手に決めるより良い。

### 4. サーバー側のホストを触るツールは opt-in にする

`file_read` / `file_write` / `file_append` / `file_delete` / `dir_list` / `dir_walk` /
`shell` の7つに、設定での有効化を要求する。**既定は全 origin で不許可。**

```toml
[tools.host_access]
# The agent's own filesystem and shell. Off unless you say otherwise:
# on a self-hosted deployment this is the machine the agent runs on.
enabled = false
```

これは既存の `decide()` の**手前**に置くゲートである。無効なら origin に関わらず
`Deny`、有効なら今までどおり `decide()` に落ちる。`src/tools/policy.rs` の
判断表には手を入れない。

**これは現状の穴も塞ぐ。** 今 `Origin::Channel` は `Execute` と `Other` だけを
拒否しており、`file_write`（`Edit`）と `file_delete`（`Delete`）は**無条件に通る**。
つまり Discord から「このファイルを消して」が今は通る。ポリシー自身のコメントが
heartbeat について同じ懸念を書いているが、チャットからの直接の経路は塞がれていない。

将来 Docker コンテナを用意し、ホストアクセスを許可する場合はそちらを推奨する、
という方針だが、この spec には含めない。

### 5. 使えないツールはモデルに見せない

ツール一覧をセッションごとに絞る。

- **ホスト側ツール**: 設定で有効なときだけ。
- **クライアント側ツール**: ACP セッションで、かつ当該 capability が宣言されている
  ときだけ。`fs.read_text_file` / `fs.write_text_file` / `terminal` は
  **個別に**見る（クライアントは片方だけ実装しうる）。

断るために往復を1回使うのは無駄で、ACP セッションではクライアント側だけが見える
ので、モデルが「どちらのマシンか」を取り違える余地がなくなる。

`ToolSet::specs()` は今プロセスに1つなので、ここは実際の変更になる。

### 6. `initialize` の capability を記録する

今は `req.client_capabilities` を捨てている。`AcpSession` に記録し、5 の絞り込みが
読む。今 `AcpSession.cwd` に付いている `#[allow(dead_code)]` も、
`terminal/create` の `cwd` として使われることで外れる。

### 7. 出力の上限はクライアントに渡す

`terminal/create` の `output_byte_limit` に `OUTPUT_CAP_BYTES`（50 000）を渡す。
**手元で切るのではなく、そもそも回線に流れない。** `TerminalOutputResponse.truncated`
が来たら、その旨をモデルへの結果に添える。

`fs/read_text_file` には上限引数が無いが、返した内容は `ToolSet::execute` の
合流点で既に切り詰められる — 全ツール共通の上限がそのまま効くので、
このツールのために足すものは無い。ただし**回線には全部流れてくる**ので、
大きなファイルには `line` / `limit` を使うよう説明文で誘導する。

**クライアント側ツールは設定（4）の対象外である。** 有効かどうかを決めるのは
クライアントが宣言した capability だけで、設定項目は増やさない。触られるのは
クライアント自身のマシンであり、それを許すかどうかはクライアントが決めることである。

### 8. 接続が切れてもターミナルは解放しない

**`terminal/release` はコマンドも殺す。** スキーマの `terminal/kill` の説明が
そう書いている — *"While `terminal/release` will also kill the command, this method
will keep the `TerminalId` valid"*。解放は「ハンドルを返す」ではなく
「片付けて終わらせる」である。

したがって接続終了時に一括解放してはならない。**回線が一瞬詰まっただけで、
相手の走っている `cargo test` が死ぬ。**

そして「ターミナルは接続に属する」という考え自体が誤りである。ACP のターミナルは
`session_id` で識別され、セッションは `session/load` で再接続を跨いで生き残る。
プロセスが走っているのは**クライアントのマシン**で、クライアントの UI に見えており、
片付ける立場としてはこちらよりクライアントのほうが上である。こちらの WebSocket が
しゃっくりしたことを根拠に、自分のものでない資源を壊してはいけない。

**解放するのは3つの場合だけ:**

- 一発 `client_shell` が時間内に終わったとき（`wait_for_exit` の直後）
- モデルが `client_shell_kill` を呼んだとき
- クライアントが「そのハンドルは無い」と応答したとき（追跡から落とすだけ）

**ハンドルはセッションに紐づけて `ServeState` で追跡する。** 接続ごとの
`AcpSession` に持たせると再接続で一覧を失い、残ったターミナルがこちらから
見えなくなる。セッション id をキーにすれば、再接続したモデルが自分の
`client_shell_output` や `client_shell_kill` をそのまま続けられる。

**モデルが後始末を忘れる件**には上限で対処する。セッションごとに保持中の
ターミナルは8まで。達したら新規作成を断り、残っているハンドルを結果に列挙する
（モデルが片付ける先が分かる形で）。上限が再接続を跨いで意味を持つのも、
追跡がセッション側にあるからである。

再接続後にハンドルがまだ有効かはクライアント次第で、プロトコルは規定していない。
無効になっていれば `client_shell_output` がエラーを返し、こちらは追跡から落とす —
モデルにはそのエラーがそのまま伝わるので、走り直すか諦めるかを判断できる。

**PR #188 で入れて撤回した「接続終了時に空セッションを閉じる」と、これは同じ誤りである。**
あのときも接続の消失を、接続より長生きするものを片付けてよい合図と読み違えた。
今回は書く前に気づけた。

## データフロー

```
モデルが client_shell を呼ぶ
  → 許可ゲート（Execute → Origin::Acp(mode) の判断表、#186 のまま）
  → terminal/create {command, args, cwd: session.cwd, output_byte_limit: 50_000}
       ← terminal_id
  → terminal/wait_for_exit を timeout 付きで待つ
       ├─ 時間内に終了 → terminal/output → terminal/release → 出力を返す
       └─ タイムアウト → release せず、ハンドルを結果に入れて返す
                          （セッションの保持リストに残る）
```

## エラー処理

- **クライアントが要求を拒否した**（ACP エラー応答）: ツール結果にそのエラーを載せて
  モデルに返す。ターンは続く。クライアント側の権限やパスの問題は、
  モデルが読んで別の手を試せる情報である。
- **capability が宣言されていない**: そのツールはそもそも一覧に無い（5）。
  それでも名前で呼ばれたら `kind_of` が `Other` を返し、既存の経路で拒否される。
- **接続が切れた最中の呼び出し**: 要求が返らない。既存のターン取り消し
  （`connection_cancel`）がターンごと畳む。
- **`terminal/release` が失敗**: 警告して続行。相手側の資源であり、こちらから
  できることはない。

## テスト

- `client_file_read` が `line` / `limit` を素通しし、返った内容が上限を超えたら
  切り詰められる
- `client_file_write` がクライアントに要求を出し、拒否応答をモデルへの結果に載せる
- **一発シェルがタイムアウトしたときプロセスを殺さず、ハンドルを含む結果を返す**
- 一発シェルが時間内に終わったら `release` まで済ませ、ハンドルを残さない
- **接続が切れても、保持中のターミナルは解放されない**（`release` はコマンドを
  殺すので、これは「片付けない」ことを確かめるテストである）
- 再接続して同じセッションを `session/load` すると、切れる前のハンドルが
  まだ追跡されており、`client_shell_output` で参照できる
- クライアントが「そのハンドルは無い」と応答したら、追跡から落ち、
  そのエラーがモデルへの結果に載る
- ターミナル数の上限に達したら新規作成を断り、残っているハンドルを列挙する
- **ホスト側ツールは設定が無効なとき、どの origin からも拒否される** — 特に
  `Origin::Channel` からの `file_delete` が拒否される（現状は通る）
- ホスト側ツールは設定が無効なとき、ツール一覧に現れない
- クライアント側ツールは、capability を宣言していないクライアントのセッションでは
  ツール一覧に現れない — `fs` と `terminal` を個別に落として、それぞれ確かめる
- 非 ACP セッション（`/rpc`・channel）のツール一覧にクライアント側ツールが無い

## 実装の順序

**fs を先に、terminal を後に。** capability の記録（6）とツール一覧の絞り込み（5）と
ホストツールの opt-in（4）は fs と一緒に入れる — terminal もそれらの上に乗る。

fs は2ツールで往復も1回、ライフサイクルも無い。承認ゲートとの噛み合わせを
そこで一度確かめれば、terminal はその繰り返しに、ライフサイクル管理が加わるだけになる。

## やらないこと

- **Docker コンテナ**。ホストアクセスを許可する場合の推奨環境として将来用意するが、
  この spec の対象外。
- **サーバー側ツールの撤去**。当初は「ACP が動いたら消す」という方針だったが、
  OpenClaw 的なエージェントとして自身の環境を触れること自体に意味があるため、
  残したうえで既定オフにする。パスガード（`refuse_if_sensitive_in`、5回書き直した）も
  残る — ホストツールが有効なとき、それが最後の防壁であることは変わらない。
- **引数レベルの承認**（「このディレクトリ以下なら常に許可」）。#186 が
  パス正規化の設計が要るとして除外しており、その判断は変わらない。

## 実装時の訂正

- **ターミナルの追跡先**: 本文の「`Mutex<HashMap>` を `ServeState` に持つ」という
  素描は、`Arc` でラップした `TerminalRegistry`（`src/tools/acp_client.rs`）に
  代わった。実体は引き続き `ServeState.acp_terminals` が1つだけ持ち、
  `AcpClient` トレイト経由で `src/tools/` に渡す — これで `src/tools/` が
  `serve::ServeState` に依存しなくて済む。セッション id をキーにする、という
  守るべき性質そのものは変わっていない。

- **「クライアントがハンドル不明と応答した」場合だけ追跡解除、とは厳密には
  実装できなかった**: `AcpClient` の各メソッドは `anyhow::Result` を返すため、
  「このハンドルは無い」という応答と「一時的な通信エラー」を型で区別できない。
  本文の第8節は前者だけを追跡解除の理由に挙げていたが、実際には
  `client_shell_output` の失敗はエラーの種類を問わずすべて追跡を保持する
  （生きているかもしれないプロセスを一時エラーで見失うほうが取り返しが
  つかない）。一方 `client_shell_kill` は kill / release の成否によらず
  常に追跡解除する（上限に達したときの案内が「kill しろ」なので、
  二重計上はいつでも kill で回復できる）。この非対称は意図したもので、
  両ツール（`src/tools/client_tools.rs` の `ClientShellOutput` /
  `ClientShellKill`）の doc コメントに理由を書いてある。

- **一発 `client_shell` のタイムアウトも8個の上限に数える**: 本文はタスク5
  （一発シェル）とタスク6（追跡・上限）に分けて書いており、その境界のまま
  実装すると、タイムアウトで残ったハンドルが上限にも「保持中一覧」にも
  現れないという抜けが生まれる（実装計画のプリフライトスキャンで検出、
  `progress.md` の「5 → 6」の行を参照）。実装時に埋め、`client_shell` も
  `create_terminal` を呼ぶ前に `client_shell_start` と同じ上限チェックを通す。

- **`cargo clippy --workspace --all-targets -- -D warnings` ではなく、
  `--all-targets` を外した形が基準**: 計画の初期段階ではこちらを CI のゲートと
  誤って書いていたが、`.github/workflows/ci.yml` が実際に走らせるのは
  `--all-targets` を付けない形で、そちらのほうが厳しい
  （`#[cfg(test)]` の呼び出し箇所が unused-item 警告を隠してしまうため）。
  以後のタスクはこちらを基準にした。

- **`Origin::Channel` のゲートは `src/agent.rs` にも触れる必要があった**:
  本文は「対象」に `src/agent.rs` を挙げていなかったが、
  `partition_without_asking` の呼び出し口は `src/agent.rs:950` の
  1箇所しか無く、そこを変えずに「チャンネルからのホストツールを既定で塞ぐ」を
  実現する経路は無かった。呼び出し箇所のみを変更し、周辺のチャンネル処理は
  触れていない。
