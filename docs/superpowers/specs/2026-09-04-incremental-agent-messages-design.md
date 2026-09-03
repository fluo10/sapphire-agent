# 作業中のメッセージを、終わりを待たずに届ける

- **対象**: `src/serve/mod.rs`（`TurnHost` / `TurnLoop::run`）、`src/serve/acp.rs`、
  `src/agent.rs`、`src/tools/subagent.rs`、`src/config.rs`、
  `templates/workspace/AGENTS.md`
- **前提**: ACP の `session/cancel`（`turn_cancel`）は実装済み

## なぜ

エージェントには「作業を続ける」と「発言する」を同時に成立させる手段が無い。

| モデルの応答 | 現在の扱い |
|---|---|
| テキストのみ | ループを `break`。ターン終了 |
| テキスト + ツールコール | ループ継続。テキストは `accumulated_text` に溜まり、ターン終了まで届かない |
| ツールコールのみ | ループ継続。無言 |

**喋れば止まり、進めば黙る。** 「○○するぞ」と一言だけ返せばそこでターンが終わり、
続きを促す必要がある。ツールコールと一緒に喋れば止まらないが、その散文は
ターンが終わるまで相手に見えない。長い作業ほど、途中経過が最も要るときに
最も見えなくなる。

### まとめ送信は設計されたものではない

Matrix/Discord が返信を1通にまとめているのは判断の結果ではなく、消失バグの
最小修正の残骸である。

```
1971e0d fix(agent): preserve assistant text from tool-call rounds

  When a model turn returned both text and tool_use, the text was stored
  in history but never sent to the channel; the subsequent round often
  yielded an empty final text, so users saw no reply. Mirror serve.rs by
  accumulating per-round text and joining it at the end of the loop.
```

それ以前、ツールコールと同時に出たテキストは捨てられていた。最終ラウンドが
空テキストを返すと返信が丸ごと消えるため、溜めて末尾で join する修正が入った。
ラウンドごとに送る選択肢は検討されていない。

さらに遡ると `38f9f2f`（初期実装）は Matrix + Anthropic の単発
リクエスト/レスポンスで、ツールループ自体が無い。「1入力 = 1出力」はその頃の形が
残っているだけで、`740e9dc` でツールループが入ったときに見直されていない。

まとめ送信の利点は、通知が1回で済むこととレート制限に当たりにくいことだけである。

### 10ラウンド上限は設定できない

`MAX_TOOL_ROUNDS: usize = 10` が `src/serve/mod.rs:44` と `src/agent.rs:17` に
ベタ書きされており、config に露出していない。`src/serve/acp.rs` の
`MaxTurnRequests` 分岐のコメントは、10ラウンドが「エディタが普通のプロンプトで
到達する日常的な終わり方」だと明言している。任せて一気に作業させる用途には足りない。

## 参考にした先行実装: Claude Code

- **ラウンド上限は無い。** tool_use を含まない応答が返るまでループが回る
- **停止条件は本実装と同一。** `end_turn` ツールは存在せず、テキストのみの応答が
  `stop_reason: end_turn` になる
- 歯止めはユーザーの中断（Esc）、コンテキスト自動コンパクション、ツール許可プロンプト
- 途中経過が見えるのは、text ブロックと tool_use ブロックが同じ assistant
  メッセージに共存し、それがライブに流れるからである

停止条件を変える必要は無い、というのがここから得られる結論である。**足りないのは
テキストの届き方だけ**であり、進捗報告用ツールのような新しい語彙は要らない。

## 決めたこと

### 1. `TurnHost` にテキストフックを足す

```rust
/// 既定 no-op。ストリーミングできるホストだけが実装する。
async fn message_chunk(&self, _text: &str) {}
```

`TurnLoop::run` から2箇所で呼ぶ。

- ツールコール付き応答のテキスト（`mod.rs:2486` 付近）
- 最終のテキストのみ応答（`mod.rs:2472` 付近）

`origin()` / `client_fs_caps()` / `acp_client()` と同じ「既定実装つきの
ホスト方針フック」の形をとる。既定が no-op なので、実装しないホストは無改修で
今の挙動を保つ。

`accumulated_text` と `outcome.text` は**残す**。`/rpc`・A2A・音声・タイトル生成が
最終テキストを受け取る唯一の経路であり、ここを壊す理由が無い。

粒度はラウンド単位である。`Provider::chat` は応答を丸ごと返す非ストリーミング
なので、「ツールを呼ぶ直前に、そこまでの散文が届く」が到達できる限界。
トークン単位にはならない。

### 2. ACP は完全にストリーミングへ移す

`AcpProgress::message_chunk` が `SessionUpdate::AgentMessageChunk` を即送出する。
そのうえで **ACP はターン終了時のチャンク送出をやめる**。

- `acp.rs:1567` の `outcome.text` 一括送出を削除 — 既に流したものと二重になる
- `acp.rs:1531` の `BudgetExhausted { partial_text }` 送出も削除 — 同じ理由

`StopReason` の返し方（`EndTurn` / `MaxTurnRequests` / `Cancelled` /
internal error）は変えない。結果として役割が分かれる。

| ホスト | テキストの受け取り方 |
|---|---|
| ACP | `message_chunk` のみ。`outcome.text` は見ない |
| `/rpc`・A2A・音声 | `outcome.text` のみ。`message_chunk` は実装しない |
| Matrix/Discord | ループ内で直接送信（`TurnHost` を通らない。§5） |

### 3. ラウンド上限を、経路ごとの設定にする

中断できる経路とできない経路を区別する。ACP には `session/cancel` があり、
Claude Code の Esc と同じ歯止めが既にある。Matrix / Discord / 音声 / heartbeat
には中断手段が無く、特に heartbeat は無人で回る。

`origin()` と同じパターンで、ホストが方針を返し、`TurnLoop::run` が config に
照らして解決する。

```rust
/// 既定 `Unattended`。中断手段のあるホストだけが Interactive を返す。
fn round_budget(&self) -> RoundBudget { RoundBudget::Unattended }
```

`AcpProgress` だけが `Interactive` を返す。

```toml
[tools.tool_rounds]
interactive = 0    # ACP
unattended = 25    # 上記以外すべて
```

どちらも **`0` を無制限**とする。`unattended` が受け持つのは Matrix / Discord /
`/rpc`（desktop チャットと音声）/ A2A / heartbeat である。`/rpc` は対話的だが
ターンを中断する手段を持たないので `Unattended` 側に入る。

`src/serve/mod.rs:44` と `src/agent.rs:17` の const は廃止する。`agent.rs` は
`TurnHost` を使わないので `unattended` を直接読む。const を消すと道連れになる箇所が
2つある — `acp.rs:2360`/`2384` のテストが `super::super::MAX_TOOL_ROUNDS` を参照し、
`subagent.rs:59` のモジュールコメントが所要時間の説明に使っている。

`acp.rs:1520` の「10ラウンドは日常的な終わり方」というコメントは前提が崩れるので
書き換える。

コンテキスト肥大は既存の `maybe_compress` が毎ラウンド見ているため、
上限を外しても対処は要らない。効くのは課金と暴走だけである。

### 4. サブエージェントの2つのメソッドは、委譲しない

`ParentHostSansTurnError`（`src/tools/subagent.rs:261`）は各メソッドを明示委譲する
構造なので、新しい2つも明示的に決める必要がある。どちらも委譲しない。

- **`round_budget` → `Unattended` を返す。** 親が無制限のときに入れ子も無制限だと、
  上限が二乗で消える。サブエージェントは常に有限であること
- **`message_chunk` → 握り潰す**（`turn_error` と同じ）。サブエージェントの散文が
  親のメッセージとして Zed に出るのは誤帰属であり、その報告はツール結果として戻る

`tool_start` / `tool_end` が親に届く扱いは変えない。あれはサブエージェントの
ツール許可プロンプトを「このセッションから来たもの」として読ませるために必要である。

**サブエージェントの上限到達はユーザーを煩わせない。** `answer_text`
（`subagent.rs:695`）が `BudgetExhausted` を親への**ツール結果**に変換する。

```rust
TurnStop::BudgetExhausted { partial_text } => {
    format!("[the subagent used its whole tool budget without finishing]\n\n{partial_text}")
}
```

親のターンは終わらない。親のループはこれを `tool_result` として受け取り次の
ラウンドに進むだけで、ユーザーには何も届かず確認も求められない。`resume` 引数
（`subagent.rs:205`）にハンドルを渡せば、最初からやり直さず続きから再開できる。

留意点が2つある。

- 続行するかは**モデルの判断**であって機構ではない。文言は明示的だが強制はしない
- サブエージェント呼び出しは**親のラウンドを1消費する**。親が ACP
  （`interactive = 0`）なら影響しないが、親が `unattended` のときリトライが
  その上限を食う

### 5. Matrix/Discord もラウンドごとに送る

`agent.rs` のループ内で、ツールコール付き応答のテキストが空でなければその場で
`channels.send()` する。最終のテキストのみ応答も、`break` する地点で同じように
1通として送る。すなわち送信箇所がループ内の2つに移り、ループ後段の
`accumulated_text.join("\n\n")` を1通で送る処理（`agent.rs:985` 付近）は消える。
`accumulated_text` 自体も `agent.rs` からは不要になる。

- **タイピング表示** — 現在 `stop_typing` はループ終了後に1回だけ（`agent.rs:982`）。
  送信のたびに表示が切れるので、送信後に typing を打ち直す
- **空テキストは送らない** — 既存の `filter(|s| !s.is_empty())` を維持する。
  ツールだけ呼ぶラウンドで空メッセージは飛ばない。ツールを1つも呼ばない普通の
  会話は今と完全に同じ1メッセージのまま
- **送信失敗はターンを落とさない** — 最悪 `unattended` 回数ぶんの連投になり
  Discord のチャンネル制限に触れうる。失敗はログに留め、ループは続ける

設定トグルは付けない。うるさいと分かってから足す。

### 6. `AGENTS.md` に一節を足す

作業を続ける間はツールコールと同時に喋ること、単独のテキスト応答はターンの
終わりを意味すること。ホストに依らない規約なので `TOOLS.md` ではなく
`AGENTS.md` に置く。Claude Code のモデルが自然にやっている振る舞いを、
明示するだけである。

## やらないこと

- **停止条件は変えない。** テキストのみの応答でもループを続け、明示的な
  `end_turn` ツールで終わらせる案は却下する。共有実行器の意味論が変わり、
  Matrix / Discord / 音声 / A2A のすべてが「`end_turn` を呼び忘れたら上限まで
  空回りする」リスクを負う。停止性をプロンプトの規律に預けることになる
- **進捗報告用ツールは作らない。** §1 でテキストがライブに流れれば、モデルが
  ツールコールと一緒に喋る癖がそのまま「止まらない状況報告」になる。呼ぶツールが
  無いのに喋りたい場面は、それは本当にターンの終わりである
- **`/rpc` の SSE は今回触らない。** 既に `tool_start` / `tool_end` を流しており
  `message_chunk` を足すのは安いが、desktop 側の対応が要る。別途
- **音声経路は据え置き。** TTS は `mod.rs:2989` で `outcome.text` を受けており、
  随時にすると発話がラウンドごとに切れる。`SseProgress` に `message_chunk` を
  実装しないだけで済むので、工数はゼロ

## 既知の重複（今回は直さない）

`agent.rs:761` のループと `serve/mod.rs:2415` の `TurnLoop::run` は、ほぼ同一
コードの二重実装である。`1971e0d` のコミットメッセージ自身が "Mirror serve.rs" と
言っており、同じ修正が手で写されている。本変更も両方に入れることになる。

統合は正しい方向だが、`agent.rs` は履歴を `Mutex<HashMap>` で持ち `TurnHost` を
使わないなど前提が違い、この変更に混ぜるには大きすぎる。別の課題として記録する。

## 中断と歯止め

`interactive = 0` を選んだとき、ターンを止める手段は **ACP の `session/cancel`
だけ**になる。これは意図した設計であり、Claude Code の Esc に対応する。
`unattended` 側に上限を残すのは、そこに同等の中断手段が無いからである。

## テスト

- 中間テキストがラウンドごとにホストへ届く
- ACP で最終テキストが二重に届かない（`message_chunk` と終了時チャンクの重複）
- `interactive = 0` で10ラウンドを超えても止まらない
- `unattended = 25` が上限で打ち切られ、ACP からは `MaxTurnRequests` になる
- サブエージェントの散文が親のストリームに漏れない
- サブエージェントは親が `Interactive` でも有限ラウンドで止まる
- Matrix/Discord: ツールを呼ばない会話は1メッセージのまま
- Matrix/Discord: ツールを呼ぶターンで複数メッセージに分かれる
- Matrix/Discord: 送信失敗がターンを落とさない
