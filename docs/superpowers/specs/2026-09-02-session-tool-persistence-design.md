# 全セッションストアでツール呼び出しを永続化し、LLM 履歴を compaction チェックポイントから復元する

- **対象**: `src/session.rs`、`src/acp_session.rs`、`src/agent.rs`、
  `src/serve/mod.rs`（`run_llm_turn` / `TurnLoop`）、`src/serve/acp.rs`、
  `src/context_compression.rs`、`src/periodic_log.rs`、`src/provider/`、`src/main.rs`
- **イシュー**: [#194](https://github.com/fluo10/sapphire-agent/issues/194)、
  [#190](https://github.com/fluo10/sapphire-agent/issues/190)、
  [#195](https://github.com/fluo10/sapphire-agent/issues/195)
- **前提**: [#191](https://github.com/fluo10/sapphire-agent/issues/191)（PR #196）がマージ済み。
  全トランスポートで `ToolSet::execute` がツール結果を 50 000 バイトに切り詰めているので、
  キャッシュに入る 1 件あたりの大きさは既に縛られている。

## なぜ

**Discord と Matrix のセッションは「何をしたか」を復元できない。** これはエージェントアプリで、
チャットからツールの実行を指示するのは普通の使い方なのに、`run_llm_turn` も `Agent::persist` も
`tool_use` / `tool_result` を捨てている。復元された履歴は「話したこと」しか持たない — ACP が
#191 以前に抱えていたのと同じ欠落が、日常的に使われている側に残っている。

**再起動復元が要約に依存している。** 生履歴を一切読み直さない代わりに、`restart_summaries` が
シャットダウン時に生成した要約をシステムプロンプトへ貼り込んでいる
（[agent.rs:768](../../../src/agent.rs)）。要約はモデル呼び出しを 1 回消費し、ターン構造を捨て、
「サーバー再起動のため直前のやり取り自体は失われています」と言い訳をさせている。ツール呼び出しが
永続化されれば、これは丸ごと不要になる。

**同じ無駄が ACP 側にもある。** `maybe_compress` は ACP セッションでも走るが、生成した要約は
捨てられている（[serve/mod.rs:2384](../../../src/serve/mod.rs)）。理由は「イベントから再構築するので
要約は二重解答になる」。だが再構築されるのは**全履歴**なので、再起動のたびに全部を replay して
初回ターンで同じ compaction をやり直している。

**`AcpSessionStore::history()` が別々の消費者に同じ答えを返している。** `session/load` は Zed の
表示用に全リプレイを要求する（Zed 側は会話を保持していない）。`run_llm_turn` は LLM の
コンテキストとして同じものを使う。前者は全部要るが、後者は要らない。

**チャンネルの intra-day digest が retrieve 索引を汚している（#190）。** `append_intraday_digest` は
セッション自身の JSONL に書いて `notify_updated` を呼ぶ。セッションファイルは
`<workspace>/sessions/` 配下、つまり索引の中にある。digest は会話が伸びるたびに再生成される
near-duplicate なので、長く続く部屋は同じ午後の言い換えを一つの索引済み文書の中に溜め続ける。

## やらないこと

**画像の復元はしない。** `scrub_images_for_storage` は画像をテキストマーカーに落とし、誰も
hydrate しない。生履歴を復元するようになると、この欠落が初めて見えるようになる。ACP も
`StoredPart::Other` で同じなので新しい不整合ではないが、Discord は画像が貼られる場所なので
**別イシューとして起票する**（`ImageCache` と `ContentPart::ImageRef` は既にあるので、
`scrub_images_for_storage` を `ImageRef` を書く側に倒すだけの話になるはず）。

**`ToolResultCache` の prune はしない（#193）。** 隣接する運用作業であって、この spec の変更が
壊れているかどうかとは独立に判断できる。#194 のレビューを prune の設計議論で希釈しない。
結果は 50 000 バイト上限かつ content-addressed（同一内容は 1 ファイル）なので、実測が無い
段階で見積もると 1 日数 MB オーダー。`~/.cache` 配下なので手動削除も安全。

**`SummaryLine` の累積には手を入れない。** compaction が走ったときだけ（数時間〜数日に 1 回）
増えるので、digest（30 分ごと）とは蓄積速度が 2 桁違う。しかも今回この行は復元の起点として
load-bearing になるので、同じ PR に削除ロジックを入れるのはリスクが高い。観測してから別途。

**ファイル形式そのものを ACP に収束させない。** `SessionStore` の行の語彙は
`meta` / `session_title` / `closed_at` / `summary_at` / `digest_at` と ACP より豊かで、パーサも
`value.get("closed_at").is_some()` 式のフィールド sniffing。単一の tagged enum に寄せるなら
4 種類すべての既存ファイルの移行が要る。やるとしても独立したイシュー。

**`parent`（連鎖）は入れない。** 見返りは ACP 自身が書いている通り「sync が実装されたときの
オフライン分岐検出」で、今は誰も読んでいない。チャンネルは書き手が常に 1 プロセスなので
見返りはゼロ。一方コストは、`append` が tip をロック下で解決する必要が生まれること
（[acp_session.rs:227](../../../src/acp_session.rs) が強いられている設計）。`SessionStore::append` は
#194 自身が「このリポジトリで最もリスクの高い編集」と呼んでいる場所で、今回すでに
ツール永続化と対の all-or-nothing ゲートを入れる。そこに一層重ねる理由が今は無い。

後から入れられることは決定 3.1 で保証する。

## 決めたこと

### 1. `ContentPart::ToolResultRef` を追加する

```rust
ToolResultRef {
    tool_use_id: String,
    /// `None` は「置き場が無かった」— 書き込み時にキャッシュが使えなかった。
    /// evict された hash と読み手にとっては同じことで、どちらも MISSING_RESULT になる。
    sha256: Option<String>,
}
```

4 ストアは `StoredMessage.parts: Vec<ContentPart>` をそのまま serde しているので、追加は純粋に
加算的。既存 JSONL はそのまま読め、`StoredMessage` と `load_session_file` は無変更。

`ContentPart` にストレージ専用の variant を置くのは既に前例がある（`ImageRef` は
「JSONL 上の正規の画像表現」として定義されている）。ただし `ImageRef` と違い `ToolResultRef` は
メモリ上に残らない — モデルが見るときには実体でなければならないので、参照はディスクから出ない。

**provider に漏れる懸念は網羅 match が塞ぐ。** [anthropic.rs:295](../../../src/provider/anthropic.rs) と
[openai_compatible.rs:254](../../../src/provider/openai_compatible.rs) には `_` アームが無いので、
variant を足した時点で全 provider に対応が強制される。そこで
`ToolResult { tool_use_id, content: MISSING_RESULT }` に落とす。hydrate を書き忘れた読み取り経路が
仮にあっても、API 的には常に valid なままになる。

`MISSING_RESULT` は `acp_session` から共有の場所へ移す。

### 2. `ToolUse.input` にも上限を掛ける

ツール結果はキャッシュへ逃がせるが、`input` は JSONL に直書きされる。`<workspace>/sessions` は
retrieve 索引の中なので、巨大な `input`（`file_write` の本文など）がそのまま索引に入る。
ACP は [acp_session.rs:357](../../../src/acp_session.rs) の `elide_oversized_input` でこれを塞いでいるので、
同じものを共有ヘルパーに切り出して 4 ストアにも適用する。

切り詰めではなく elide なのは、truncate した JSON はパースできず、再読み込み時に `input` が
同じ形の valid な JSON である必要があるため。

### 3. `StoredMessage` に `id` を入れる（`parent` は入れない）

```rust
#[serde(skip_serializing_if = "Option::is_none", default)]
pub id: Option<Uuid>,   // 新規書き込みは常に Uuid::now_v7()
```

決定 4 のチェックポイントが安定したカーソルを必要とし、timestamp はカーソルとして劣るため。

- `Utc::now()` の粒度が粗い環境ではクロックが同じ値を返し得る。連続する append が同一 timestamp に
  なると「これより後」の判定が壊れる。
- NTP がクロックを巻き戻すと timestamp が単調でなくなり、チェックポイント以降のメッセージが
  replay から静かに落ちる。

`id` ならどちらも起きない。加えて、ACP 側のチェックポイントと**同じ機構**（`covers_through: Uuid`）で
書けるようになる — 同じ概念に 2 つの実装を持たずに済む。

`None` は既存行のみ。読み手はファイル順にフォールバックする。

### 3.1. `parent` を後から入れられる状態を保つ

`/rpc`（`cross-device`）は grain-id で過去のセッションを開き直せるので、同じセッションから
2 方向に伸ばしたい要求はいずれ出てくる。そのとき `parent` が必要になる。今は入れないが、
**入れられなくなることはない**ことを規則として固定しておく。

移行規則:

> `parent` を持たない行は、**ファイル順で直前のメッセージ行の `id`** を親とみなす。

タイムスタンプ順ではなくファイル順なのは、決定 3 と同じ理由 — クロックの粒度と巻き戻しに
影響されないため。`SessionStore` の JSONL は append-only なので、ファイル順がそのまま
権威ある順序になる。

**この規則は構造的に正しい。** `parent` を記録する仕組みが無い時代のファイルには分岐が
存在し得ない — 分岐は複数の書き手が互いを知らずに append することで生まれるもので、
どちらの書き手も親を記録していない以上、ファイルは線形にしかならない。つまり「後から
再構築できない情報」は失われていない。

ACP が「`id` と `parent` は今誰も読んでいないが後から再構築できないので今入れる」と
書いているのと矛盾しない。ACP がそう言えるのは**1 イベント 1 ファイルへの分割**を見ているからで、
その場合ファイル順という手掛かりが消える。`SessionStore` は 1 セッション 1 ファイルのままなので、
手掛かりは残り続ける。

非メッセージ行（`SummaryLine` / `TitleLine` / `ClosedLine`）には今回 `id` を入れない。
`parent` が来たときに同じ規則で引き取れるので、使う当てのないフィールドを先に足さない。

### 4. compaction チェックポイントを 5 ストア共通の概念にする

書き込み API を両ストアで対称にする:

```rust
append_summary(session_id, summary: &str, keep_recent: usize)
```

`keep_recent` は `maybe_compress` が手元に残した末尾メッセージ数
（`CompressionResult` に追加する `to_keep.len()`。境界 compaction は履歴を丸ごとスタブで
置換するので `0`）。**カーソルの算出はストア自身が自分のファイルを数えて行う** —
ACP が `append_line` で tip を自分で解決しているのと同じ「ストアが自分のファイルの権威」という形。
呼び出し側が in-memory の index とディスク上の行番号を対応付ける必要がなくなる。

- `SessionStore`: `SummaryLine` に `covers_through: Option<Uuid>` を追加。既存の
  `up_to_timestamp` は informational のまま触らない。
- `AcpSessionStore`: `Line::Summary { id, parent, at, summary, covers_through: Uuid }` と
  対応する `EventBody::Summary` を追加。連鎖イベントとして書くので `history()` は
  Message だけを拾う既存の filter でそのまま無視する。

**`covered <= 0` は到達不能。** `maybe_compress` は `split >= 1` のときしか発火せず、ディスク上の
メッセージ件数は常にメモリ上の件数以上（メモリ側は前回の compaction で削られている）。
防御的に 1 で clamp する。

**後方互換**: `covers_through: None`（既存の `SummaryLine`）は「**その行より後（ファイル順）の
メッセージを replay**」と定義する。シャットダウン要約だけを持つ既存ファイルでは replay 対象が
空になり、現在の挙動とほぼ一致する。

**移行期の既知の劣化**: `id` を持たないメッセージがカーソルの対象になった場合、
`covers_through` は書けず `None` に落ちる。その一度だけ `keep_recent` 分の末尾が replay から
外れる。既定の `session_policy = Reset` ではファイルが日次ローテートするので、この窓は最大 1 日。

**チェックポイントは対を割らない。** `find_safe_split_point` は tool_use / tool_result のペアを
跨いで切らないので、`keep_recent` から算出したカーソルは構造的に安全な位置に落ちる。
`None` フォールバック側の切り口は保証が無いが、決定 6 の孤児修復が拾う。

### 5. 読み取り経路ごとに何を返すかを決める

| 経路 | ツール結果 | チェックポイント切り詰め |
|---|---|---|
| `SessionStore::load_all`（チャンネル起動） | hydrate | **する** |
| `SessionStore::load_session`（`/rpc` 再開） | hydrate | **する** |
| `SessionStore::load_session_full`（`recall_memory`） | hydrate | しない |
| `SessionStore::sessions_for_day*`（日次ログ） | **しない** | しない |
| `AcpSessionStore::history`（`session/load` 表示・digest sweep） | hydrate | しない |
| `AcpSessionStore::history_for_model`（`run_llm_turn`） | hydrate | **する** |

日次ログは `ToolResultRef` のまま受け取り、`format_sessions` が Text 部分だけを拾う既存の挙動で
落ちる。「evict された結果のプレースホルダー文が恒久記録に混ざる」ことは構造的に起きない。

`session/load` の全リプレイは維持する — Zed は会話を保持していないので、表示には全部要る。
分けるのは LLM に渡す方だけ。

**復元スタブは 1 種類に統一する。** `compaction_stub(summary) -> Vec<ChatMessage>` を
`context_compression` に置き、`maybe_compress`、境界 compaction、復元の 3 箇所で使う。
境界 compaction の「prior-day」という文言は失われるが、スタブの文面は load-bearing ではなく、
「復元時にどちらのスタブを使うべきか分からない」状態を作る方が悪い。

### 6. 孤児修復を共有し、#195 を直す

チャンネル側も今回はじめて「tool_use だけディスクに残って tool_result が残らない」状態を
持ち得るようになる（2 回の append の間でプロセス死）。[acp_session.rs:570](../../../src/acp_session.rs) の
位置ベースの修復を共有関数として切り出し、LLM 向け履歴を作る 3 経路すべてに適用する。

同時に #195 を直す: tool_use メッセージが**部分的にしか**応答されていない場合、新しい
メッセージを挿入するのではなく、直後が既に tool_result メッセージならそこに placeholder を
**マージ**する。現状の書き手からは到達不能なバグだが、壊れていると分かっている関数を
Discord / Matrix の常用経路へ持って行くことになるので、移動と同じコミットで直す。

### 7. 書き込み時の対の all-or-nothing を両経路で守る

`TurnLoop` には既にある（tool_use の append が失敗したら tool_result の append もしない）。
`Agent::handle_message` は 2 回の `persist()` が独立しているので、同じゲートを入れる。
片方だけがディスクに残ると、in-memory 履歴（どちらにせよ正しい）と違って**そのセッションは
永久に壊れる**。

### 8. チャンネルの intra-day digest を `DigestCache` へ移す（#190）

`DigestCache` は既に session_id キーでストア非依存、atomic rename、`prune_before` 実装済み。

- `Agent` に `Option<Arc<DigestCache>>` を持たせ、idle flush とシャットダウンの発行先を
  `cache.put(&session_id, &summary, None)` に変える。キャッシュが `None` なら digest を
  諦める（その部屋が today ブロックから抜けるだけ）。
- `SessionStore::append_intraday_digest` は削除する。
- `SessionStore::intraday_digests_for_day(date, boundary, cache: Option<&DigestCache>)`:
  meta はセッションファイルから、本文はキャッシュから引く（ACP 版と同形）。
  **キャッシュに無ければファイル内の既存 `IntradayDigestLine` にフォールバックする** —
  アップグレード直後にその日の digest が全部消えるのを防ぐための移行措置。
  `load_meta_and_latest_intraday_digest` はそのために残す。
- セッション id は UUID なので `DigestCache::path_for` の文字種チェックを通る。
- heartbeat の prune（[heartbeat.rs:233](../../../src/heartbeat.rs)）は自動的にチャンネル分も
  カバーするようになる。

### 9. `agent.rs` から退役するもの

- `restart_summaries` と `<prior-session-recap>` の注入
- `pending_fallback` と `bootstrap()` の要約合成 — 部屋フィルタごと不要になる
- `summarize_on_shutdown` の `append_summary` 呼び出し。digest 発行だけ残す
- `persist()` の `ToolUse` / `ToolResult` ストリップ → `session_store.append` に一本化
- `load_all` の戻り値は `(active, histories)` になる

**残るもの**: compaction の `SummaryLine` 書き込み。これが「コンテキスト長節約のための要約」
そのもので、今回はむしろ復元の起点として役割が増える。消えるのは再起動復元専用だった
シャットダウン要約だけ。

### 10. 配線

`tool_result_cache` は現在 main.rs:637 で生成され acp_session_store にムーブされている。
4 ストアの構築（main.rs:465 / 475 / 486 / 723）はそれより**前**にあるので、**生成を前倒しする**。

`SessionStore::with_workspace`（本番用、呼び出しは main.rs の 4 箇所のみ）に cache 引数を
追加し、`SessionStore::new`（テスト専用）は cache なしのまま残す — 忘れたらコンパイルが
通らない側にだけ強制がかかる。

`digest_cache`（main.rs:674）は `serve_state` にムーブされる前に `Arc::clone` して `Agent` へ渡す。

## データフロー

**書き込み（全トランスポート共通）**

```
ToolSet::execute → truncate_output (50 000 B)
  → ChatMessage::tool_results_with_images
  → store.append / acp_store.append_message
      ToolResult → cache.put(content) → ToolResultRef { tool_use_id, sha256 }
      ToolUse    → elide_oversized_input(input)
      Image      → テキストマーカー（今回対象外）
```

**読み込み（LLM 向け）**

```
load_all / load_session / history_for_model
  → 最新の SummaryLine / Summary イベントを探す
  → covers_through 以降のメッセージだけを取る
  → compaction_stub(summary) を先頭に付ける
  → ToolResultRef → cache.get(sha) → ToolResult
                    miss なら MISSING_RESULT
  → 孤児修復（位置ベース、#195 修正込み）
```

**読み込み（表示・記録向け）**

```
session/load       → history()            全リプレイ、hydrate あり
日次ログ            → sessions_for_day*    切り詰めなし、hydrate なし（Text のみ拾われる）
recall_memory      → load_session_full    切り詰めなし、hydrate あり
```

**起動時 I/O が青天井にならない理由**: hydrate は 1 ツール結果につき小さなファイル 1 つを
読む。チェックポイント切り詰めが対象をおおむね「直近 20 メッセージ + 前回 compaction 以降」に
縛るので、何ヶ月も続いている部屋でも起動コストは有界になる。本番が Ceph 上で動いている以上、
これは付随的な利点ではなく必要条件。

## エラー処理

| 状況 | 挙動 |
|---|---|
| 書き込み時にキャッシュが使えない | `ToolResultRef { sha256: None }` + warn。対は保たれる |
| 読み込み時に hash が無い / 壊れている | `MISSING_RESULT` プレースホルダー。ミスであってエラーではない |
| tool_use は書けたが tool_result が書けなかった | 次回読み込みで孤児修復が placeholder を差し込む |
| tool_use の append が失敗 | tool_result の append もしない（決定 7） |
| `covers_through` が連鎖上に見つからない | warn して全リプレイに落とす |
| `covers_through: None`（既存行） | その行より後をファイル順で replay |
| digest キャッシュが `None` | その部屋の digest を諦める。today ブロックから抜けるだけ |
| ToolResultRef が provider に届いた | `MISSING_RESULT` 入りの `ToolResult` に落とす。API は valid のまま |

## テスト

**ストア単位（`session.rs` / `acp_session.rs`）**

- ツール結果が round-trip する（append → load で内容が戻る）
- キャッシュミスが `MISSING_RESULT` になる
- 書き込み時にキャッシュ不在 → `sha256: None` → 読み込みで `MISSING_RESULT`
- 巨大な `ToolUse.input` が elide され、なお valid JSON として読み戻せる
- チェックポイント切り詰め: `keep_recent` 個の末尾 + スタブだけが返る
- `covers_through: None` の既存ファイル: その行より後だけが返る
- `SummaryLine` を持たない既存ファイル: 全リプレイ
- `id` を持たない既存メッセージがカーソル対象 → `None` に落ちるが壊れない
- `AcpSessionStore::history()` は `Summary` イベントを無視して全リプレイのまま

**孤児修復（共有関数）**

- tool_use のみ / tool_result のみ の各孤児
- **#195**: assistant が tool_use を 2 つ持ち、直後のメッセージが片方しか答えていない場合、
  余計なメッセージを挿入せず、実結果と placeholder を持つ 1 メッセージにマージされる

**経路単位**

- 日次ログがツール結果を含まない（`ToolResultRef` が Text フィルタで落ちる）
- `session/load` の editor リプレイが全履歴を出す
- チャンネル再起動: ツール結果込みの履歴が復元され、`<prior-session-recap>` が出ない

**digest（#190）**

- チャンネル digest がキャッシュに書かれ、セッションファイルに書かれない
- キャッシュに無い場合、既存 `IntradayDigestLine` にフォールバックする
- `build_today_digest_for_namespace` が 5 ストア分を混ぜて namespace で振り分ける

## 提出単位

ブランチ 1 本、PR 4 本。#194 が求めていた「バイセクトしやすい分割」に沿う。

1. `feat(sessions)`: チャンネル digest を `DigestCache` へ（#190、決定 8）
2. `refactor(sessions)`: 孤児修復・`elide_oversized_input`・`MISSING_RESULT` を共有の場所へ
   切り出し、#195 を直す（決定 6、決定 2 の切り出しまで）。ACP の挙動は #195 の修正以外
   変わらない、純粋な移動の差分
3. `feat(sessions)`: 4 ストアでツール呼び出しを永続化（#194 本体、決定 1・3・7・10 と、
   決定 2 の 4 ストアへの適用）
4. `feat(sessions)`: LLM 履歴を compaction チェックポイントから replay（決定 4・5・9）

## 未解決のまま残すもの

- **画像の復元** — 起票する。`ImageCache` と `ImageRef` は既にあるので、
  `scrub_images_for_storage` を `ImageRef` を書く側に倒す話になるはず
- **`ToolResultCache` の prune**（#193）— この変更で埋まる速度が上がる
- **`SummaryLine` の累積** — 起票する。観測してから
- **ファイル形式の ACP への収束** — 起票する。4 種類すべての既存ファイルの移行が要る
- **`parent` の導入** — `/rpc` に分岐が要るようになったとき、または remote-workspace sync が
  実装されたとき。決定 3.1 の移行規則で既存ファイルを引き取れる
- **アイドルセッションの破棄** — #191 の spec から引き継ぎ。今回も対象外
