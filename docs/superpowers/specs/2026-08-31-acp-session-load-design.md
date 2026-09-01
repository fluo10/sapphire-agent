# Zed から過去のセッションを開く（`session/load` · `list` · `resume`）

- 日付: 2026-08-31
- 対象: `src/serve/acp.rs`, `src/serve/mod.rs`, `src/session.rs`
- 前提: `2026-08-30-acp-permissions-design.md`（モードとゲート）、
  `2026-08-24-acp-websocket-transport-design.md`（`/acp` エンドポイント）
- ACP: `session/load`, `session/list`, `session/resume`（いずれも v1 stable）

## 背景

ACP セッションは**接続の寿命しか持たない**。`AcpSessions` は接続ごとの
`HashMap` で、ソケットが死ねば消える（`src/serve/acp.rs`）。`initialize` は
`loadSession(false)` を返している。つまり Zed を再起動すると、昨日の会話は
二度と開けない。

一方、必要なものは**ほぼ全部すでにある**。

- `src/session.rs` に `list_sessions` / `load_session` / `close_session` /
  `delete_session` / `set_title` が揃っている。
- `run_llm_turn` は履歴をメモリに持っていなければ
  `store.load_session(&session_id)` で**遅延ハイドレートする**
  （`src/serve/mod.rs`）。

したがって「過去セッションの継続」は**実行層の変更をまったく必要としない**。
足りないのは、ACP アダプタが「新規 id を mint する」以外の道を持たないこと
と、クライアントに会話を見せる replay だけ。

## 決めたこと

1. **`load` / `list` / `resume` の 3 つを実装する。** `close` と `delete` は
   出さない。
2. **一覧はトークンの namespace に限る。**
3. **`load` も namespace を照合して拒否する。** 一覧のフィルタだけでは不十分。
4. **replay はストアの生履歴。** モデルの記憶ではなく、起きたこと。
5. **`cwd` を `SessionMeta` に永続化する。** ACP が作った/開いたセッションのみ。
6. **並行ターンの履歴レースは直さない。** 明記して次に送る。

### 1. `close` / `delete` を出さない理由

`close_session` は `closed_at` マーカーの追記なので可逆だが、`delete_session`
は JSONL を実際に削除する。エディタのボタン一つで本番ワークスペースの会話が
消えるのは、この機能の目的（過去を開く）とは別の判断が要る。

### 3. 一覧のフィルタだけでは不十分な理由

`session/list` を namespace で絞っても、`session/load` が id を直接受け取る
以上、**一覧に出ないセッションでも id さえ知っていれば開ける**。id は
UUIDv7 なので推測はできないが、ログや別プロファイルの一覧から漏れる経路が
ある。境界は両方に置く。

### 4. モデルの記憶ではなく起きたこと

`load_session` は生の全メッセージを返し、`SummaryLine` を無視する。一方
メモリ上の履歴は `maybe_compress` で圧縮されていることがある。したがって
圧縮済みセッションでは **画面の表示 > モデルの文脈** になる。

これは replay の欠陥ではなく圧縮の本質。ユーザーは自分が書いたことを読み
たいのであって、モデルの要約を読みたいのではない。

## セッションの同一性

`session/new` は今も新規 id を mint する。`load` / `resume` は mint せず、
渡された id を接続のマップに登録する。

```
session/load  → 既存 id を検証して登録、replay して応答
session/resume → 既存 id を検証して登録、replay せず応答
session/new   → 変更なし
```

登録時に `state.session_room_profiles` にも接続のプロファイルを入れる。
これは `run_llm_turn` がプロバイダと namespace チェーンを解決する経路。

モードは常に `default` から始まる。前回のモードは永続化しない——モードは
「今このエディタでどう振る舞ってほしいか」であって、会話の属性ではない。

## `session/list`

`SessionMeta` → `SessionInfo` の対応:

| ACP | sapphire | 備考 |
|---|---|---|
| `session_id` | `session_id` | |
| `title` | `title` | `session_title` 行から。無ければ `None` |
| `updated_at` | — | JSONL の mtime を ISO 8601 で |
| `cwd` | — | 新フィールド（下記） |
| `additional_directories` | — | 常に空 |

フィルタ:

- `SessionMeta.namespace` が接続の namespace と一致するもののみ。
  `namespace` が `None`（このフィールド以前の古いファイル）は**除外する**
  — どの namespace のものか分からないものを見せる側に倒さない。
- `ListSessionsRequest.cwd` が指定されたら、`cwd` が一致するもののみ。
- `closed_at` があるセッションは除外する。

`cursor` / `next_cursor` は使わない。全件返し、`next_cursor: None`。
セッション数が問題になるまで YAGNI。

## `cwd` の永続化

`SessionMeta` に `cwd: Option<String>` を足す。`session/new` と `session/load`
が受け取った `cwd` を、そのセッションのメタ行に記録する。

**既存のセッションと `/rpc` 由来のセッションは `cwd: None` になる。**
`ListSessionsRequest.cwd` が指定された場合、それらは一覧に出ない。つまり
**この変更以前の会話は、Zed がプロジェクトで絞る限り見えない。**

受け入れる。代案（`None` はフィルタに関わらず常に出す）は、別プロジェクトの
会話が毎回混ざるほうを選ぶことになり、そちらのほうが害が大きい。`cwd` を
指定しない `session/list` では従来どおり出る。

## `session/load` の手順

```
1. id を SessionMeta で解決。無ければ invalid_params
2. namespace を照合。違えば invalid_params（存在は漏らさない同じ文言）
3. 接続のマップに登録、mode = default、
   state.session_room_profiles にプロファイルを入れる
4. store.load_session() の履歴を session/update で流す
     Role::User      → SessionUpdate::UserMessageChunk
     Role::Assistant → SessionUpdate::AgentMessageChunk
   ContentPart::Text のみ。Image / ImageRef はこの版では飛ばす
5. LoadSessionResponse::new().modes(...) で応答
```

replay は**応答より前**（ACP の規定）。`session/new` と同じ `SessionModeState`
を返す。

ツール呼び出しは replay されない。JSONL に保存していないため
（`run_llm_turn` の「tool_use は意図的に保存しない」）。これは次の spec の
主題であり、ここでは扱わない。

### 存在を漏らさない

手順 1 と 2 は**同じエラー文言**を返す。「そのセッションは存在するが君の
ものではない」と「存在しない」を区別すると、id の有無を列挙できる。

## `session/resume`

手順 4 を抜くだけ。`sessionCapabilities.resume` を立てる。

ACP の `resume` は「load を実装できないエージェント向けの代替」と定義されて
いるので、`load` がある以上クライアントは使わないかもしれない。それでも
出すのは、replay が高い（履歴が長い）ときにクライアントが選べるようにする
ため。

## capabilities

`initialize` の応答:

```rust
AgentCapabilities::new()
    .load_session(true)
    .session_capabilities(
        SessionCapabilities::new()
            .list(SessionListCapabilities::new())
            .resume(SessionResumeCapabilities::new()),
    )
```

## 並行性 — 直さない

`load` があると、**2 つの接続が同じセッションを開ける**。

`acp.rs` 冒頭が既に記録しているとおり、`run_llm_turn` はセッションの履歴を
先頭でクローンし、最後に丸ごと書き戻す。同一セッションの並行ターンは
メモリ上で last-writer-wins であり、両方が JSONL には user メッセージを
追記済み。したがって永続の記録とメモリの履歴が食い違う。

今までこれは同一接続でしか踏めなかった。`load` はそれを**別接続からでも
踏めるようにする**。

直すには `state.sessions` にセッション単位のロックが要り、`run_llm_turn` の
ターン全体をその中で回すことになる。今回のスコープ外。**このセッションを
すでに別の接続が開いている場合に警告ログを出す**ところまでをこの spec に
含め、実際の直しは別イシューにする。

## テスト

- `list` が他 namespace のセッションを返さない
- `list` が `namespace: None` の古いファイルを返さない
- `list` が `closed_at` のあるセッションを返さない
- `list` が `cwd` フィルタを honour する
- **`load` が他 namespace の id を拒否する** — 一覧のフィルタだけでは
  不十分であることの証明。これが無いと境界が片側にしかない
- 存在しない id と他 namespace の id で**同じ文言**が返る
- `load` の replay が応答より前に届く
- `load` した履歴の内容と順序が JSONL と一致する
- `load` 後の `session/prompt` が履歴を引き継ぐ（`load_session` の
  遅延ハイドレートと同じ id を使っていることの確認）
- `resume` は replay を送らないが、後続の `prompt` は履歴を引き継ぐ
- `initialize` が `loadSession: true` と `list` / `resume` を返す
- 回帰: `session/new` は従来どおり新規 id を mint する

## 別イシューに切り出すもの

- **ツール結果のキャッシュと忠実な復元。** JSONL に `tool_use` と
  `tool_result` の参照を入れ、ワークスペース外のキャッシュ（`image_cache`
  と同型）から復元する。あわせて再開用要約の生成を shutdown / 定期から
  **ロード時の遅延処理**に移し、キャッシュから復元できなかった場合だけ
  走らせる。これが入ると replay にツール呼び出しが出る。
- **セッション単位のロック。** 上記の並行性。
- **`session/close` と `session/delete`。**
- **ページネーション。** `cursor` / `next_cursor`。
- **replay の画像。** `ImageRef` をキャッシュから引いて `ContentBlock::Image`
  で流す。`promptCapabilities.image` と一緒にやるのが自然。
