# サブエージェントの再帰的起動（ネスト段数制限と定義ごとの許可リスト）

- **Issue**: 番号なし（動機は下記「なぜ」参照）
- **対象**: `server/src/tools/subagent.rs`、`server/src/serve/mod.rs`（`TurnContext` /
  `TurnLoop::run` / ゲート関連）、`server/src/agents.rs`、`server/src/config.rs`、
  `server/config.example.toml`、`server/templates/workspace/config.toml`、
  `README.md` / `README.ja.md`

## なぜ

APIコスト削減のため、コードリサーチ型の簡単なタスクを安価・無料モデルに
任せたい、という動機から。既に定義の `profile:` でサブエージェントごとの
プロバイダ固定はできるが、現状のサブエージェントは**再帰が構造上閉じて
いる**ため、plan / implement のようなサブエージェントから explorer を
起動できない。

現状の機構（`server/src/tools/subagent.rs` モジュールdocの「3つ目の性質」）:

- `subagent_tool_specs()` がネスト先のツールリストから `subagent` 自身を
  **常に除去**する。
- 除去はモデルへのヒントに過ぎず、実際に再帰を閉じているのは
  `TurnLoop::run` の許可ゲート（`Refusal::NotOffered`、`crate::tools::policy`）。
  ネストターン自身の `tool_specs` に含まれない名前を呼んでも拒否される。

この「常に除去」を段数制限に置き換えるのが本変更の中心。

## 決めたこと

### 1. ネスト段数の上限を config で設定する（A案）

既存の `[tools.subagent]`（`SubagentConfig`、`turn_timeout_secs` のある場所）に
`max_depth` を追加する。

```toml
[tools.subagent]
turn_timeout_secs = 3600   # 既存
max_depth = 2              # 新規: ネスト段数の上限。0 =委派不可（ネスト禁止）
```

- 段数 `d` で動くエージェントは `d < max_depth` のときのみ委派できる。
  メインエージェントは `d = 0`。
- **既定値 2** = メイン(0) → plan(1) → explorer(2) まで。explorer からは
  さらにネストできない。
- 従来の「ネスト禁止」は `max_depth = 1`。
- `max_depth = 0` は委派そのものが不可（メインエージェントのターンからも
  `subagent` ツールが見えない）。
- 型は `u32`、既定値 2。上限クランプや検証エラーは設けない（大きな値は
  実質無制限として素通り）。

### 2. 定義ファイルの `subagents:` 許可リスト（B案）

`agents/<name>.md` の frontmatter に `subagents:` を追加する。セマンティクスは
既存の `tools:` と**完全に同型**にする:

| 書き方 | 意味 |
|---|---|
| 未記載 | 委派元ターンから見えているエージェント集合をそのまま継承（=Aのみの挙動） |
| `subagents: [explorer]` | 委派先ターンからはその名前のものだけが見える |
| `subagents: []` | そのエージェントからは一切ネストしない（深さと無関係に `subagent` ツール自体を渡さない） |

- `tools:` と同様、リスト内の未知名は起動時に落ちず、dispatch 時に `warn!`
  して無視する（既存の `newly_unknown_tools` と同じ方針・同じ重複排除の考え方。
  実装は同機能を `subagents:` にも適用する）。
- `subagents:` に `subagent` と同じツール名を書くのは no-op（`tools:` と同じ
  扱い。ゲートが実体なので、リストに無くても実害ではなく、そもそも名前リストは
  エージェント名空間の話でツール名空間とは別物）。
- 段数制限と許可リストの合成は**各段で局所的に**決まる。`subagents:` が
  あればそのリストを、無ければ委派元ターンの見えている集合を、委派先の
  見せる `subagent` 仕様のエージェント一覧として埋める。連鎖全体の制限は
  この継承／制限の連鎖から自然に導かれるので、別途チェーン全体の計算は
  しない。

### 3. 機構の中身: `subagent_tool_specs` の一般化

`subagent_tool_specs(def, parent_visible)` は現状「`parent_visible` から
`subagent` を除き、`def.tools` で絞る」だけ。これを:

1. 従来どおり `parent_visible` から `subagent` を除外し、`def.tools` で絞る。
2.委派元ターンの段数 `d` に対して `d < max_depth` かつ
   `def.subagents != Some([])` なら、**委派先ターン用の `subagent` ツール仕様を
   再構築して末尾に含める**。この仕様に埋める一覧は、
   `def.subagents: Some(list)` なら登録済みエージェントとの交差集合として
   該当名だけ、`None` なら委派元ターンに見えているエージェント集合
   （委派元仕様に埋め込まれた一覧が事実上のソース）をそのまま、という
   集合で再構築する（`build_spec` の再利用）。
3. 段数制限で禁止された場合は現状と同様 `subagent` 自体をリストに含めない。
   ゲート（NotOffered）はリスト membership を見るだけなので無修正で通る。

深さの情報元:委派元の `TurnContext` に段数を持たせ、ネスト `TurnLoop` 生成時
（`SubagentTool::run_and_store`）に +1 したものをネスト側ターンへ渡す。既存の
`visible_specs: Arc::clone` の流れ（ターンごとに1回構築、ターン中は共有）に
そのまま乗せる。見えているエージェント集合は委派先仕様の説明文に埋め込まれた
一覧から導出するため、`TurnContext` の追加フィールドは段数のみで足りる
見込み。

### 4. resume との交差

resume はツールリストを常に再計算する設計（モジュールdocの「3つ目の性質」）なので、
許可リスト・段数も同じ再計算経路に乗るだけで、格別の扱いを必要としない。
既存の「resume_recomputes_the_tool_list_so_the_depth_cap_still_holds」テストが
これを保証する形になる。ネストターンで保存されたハンドルも同一経路
（`SubagentCache`）で、段数情報の保存は不要（毎ターン再計算のため）。

### 5. テスト方針

既存テスト様式（scripted StubProvider + `ChatLog` の call 数検証）を継ぐ:

- 既存の `a_subagent_cannot_invoke_subagent_by_name`（serve/mod.rs、
  `s-subagent-no-recursion`）は `max_depth = 1` の明示下での従来動作検証に
  置き換える。同テスト構成で `max_depth = 2` にすると5つ目の `chat()`
  呼び出しが走る（＝ネストのネストが実際に動く）ことを対で検証。
- `subagent_tool_specs` の純関数テストを `subagent.rs` に追加:
  - 段数制限で `subagent` が仕様リストに入る／出ない
  - `subagents: [explorer]` で委派先仕様のエージェント一覧が該当名だけに絞られる
  - `subagents: []` で段数に関わらず `subagent` 自体がリストされない
  - `subagents: None` で全集合が継承される
- `config.rs`: `max_depth` の既定値 2、明示指定、`0` のパーステスト
  （既存の `turn_timeout_secs` テストと同型）。
- `agents.rs`: `subagents:` フロンターマの解析（未記載 = None、空リスト =
  Some(empty)、リスト = 名前順そのまま）。
- 未知名 warn の重複排除は既存の `newly_unknown_tools` 経由であることを
  関数単体で検証。

### 6. ドキュメント

- `server/config.example.toml` と `server/templates/workspace/config.toml` の
  `[tools.subagent]` 例に `max_depth` を追記（コメント込み）。
- `README.md`（英語）と `README.ja.md` のサブエージェント節に、ネスト段数と
  `subagents:` の1段落を追記。既存の「プロファイルは親から引き継がれない
  唯一の要素…」という記述に続けて、段数制限と許可リストのセマンティクスを
  同じ文体で書く。
- `server/src/tools/subagent.rs` モジュールdocの「第3の性質」（再帰を閉じる
  機構）の記述を、新機構に合わせて更新する。

## 対象外（非対象）

- ネスト段ごとの `turn_timeout_secs` の意味変更はしない（各段のネストターンが
  それぞれ個別に締切の対象になる、という既存挙動のまま。最悪の待ち時間は
  段数分の締切の合計になりうる、という点はREADME側の記述で触れる）。
- `subagent_cache` のハンドル・履歴保存機構は変更しない。
- ツール名のプレフィクス分離（`agent_/user_`）の方向性は本変更の範囲外。
  将来的にツール名が変わる場合も、本仕様は機能単位（「`subagent` ツール」）で
  記述しているので、名前自身の文字列参照箇所が追随するだけ。
