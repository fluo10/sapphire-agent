# デバイス台帳による認証と、room_profile の紐づけ

- 日付: 2026-08-29
- 対象: `src/config.rs`, `src/config_layer.rs`, `src/main.rs`（CLI）,
  `src/serve/{mod,a2a,acp,mcp}.rs`, `src/ambient/{auth,ingest,startup}.rs`
- 前提: `sapphire-framework` の `2026-08-29-device-user-registry-design.md`
  （`sapphire-framework-registry` と `KeyEntry.device_id`）

## 背景

いま agent には**認証機構が 2 本**ある。

| 経路 | 現状 |
|---|---|
| ambient ingest | トークン → `KeyStore` → `key.id` → `[device.*].key_id` の逆引き → デバイス名（`src/ambient/auth.rs`） |
| `/a2a` `/acp` `/mcp` | トークンを `[room_profile.*].api_keys` の**平文と総当たり比較** → プロファイル名（`src/config.rs` の `resolve_a2a_token`） |

後者は生のトークンを設定ファイルに置く。前者は設定に秘密を置かない正しい形だが、
デバイス → 鍵という向きなので、鍵ファイルがホストごとに存在する事実と噛み合わない。

さらに `[device.*]` はメイン設定ファイルの中にあり、コマンドで生成・更新する対象としては
置き場所が悪い（人間が書いたコメントを毎回消すことになる）。

## 決めたこと

1. **デバイス台帳を `<workspace>/.sapphire-agent/devices.toml` へ出す。** ユーザー台帳も同じ場所に
   `users.toml`。framework の registry を使う。
2. **`device` / `user` サブコマンドを足す。** `device add` が台帳の行と鍵を一度に作る。
3. **認証を 1 本に畳む。** ambient と `/a2a` `/acp` `/mcp` が同じ `DeviceAuth` を通る。
4. **room_profile の紐づけはホスト設定側**に `[room_profile.*].devices` として持つ。
5. **`[device.*]` と `[room_profile.*].api_keys` を廃止**し、残っていたら起動時にエラーで落とす。

### 4. room_profile をホスト設定側に持つ理由

`devices.toml` は**コマンドが全上書きするファイル**（framework の `keys.toml` と同じ作法）。そこに
人間が手で意味を書き込むフィールドがあると、コメントが消える・手編集とコマンドが競合するという
`keys.toml` が既に抱えている問題をもう一段持ち込むことになる。

加えて room_profile 名は `config.toml` 内の `[room_profile.*]` への参照なので、参照する側と
される側が同じファイルに居るほうが整合を取りやすい。

よって `devices.toml` は「名前・説明・user_id」だけの台帳に留め、ルーティングの決定は
人間が所有する `config.toml` に置く：

```toml
[room_profile.work]
profile = "sonnet"
rooms   = ["!abc:example.org"]
devices = ["a3f9k2p", "b7x2m9q"]   # ← api_keys を置き換える
```

## CLI

```
sapphire-agent device add --name <NAME> [--description <DESC>] [--user <SELECTOR>] [--expires-in <DUR>]
sapphire-agent device list
sapphire-agent device rotate <SELECTOR> [--expires-in <DUR>]
sapphire-agent device retire <SELECTOR> [--purge]

sapphire-agent user add --name <NAME> [--description <DESC>]
sapphire-agent user list
```

`--expires-in` は `sapphire-journal-server` の `parse_duration` と同じ書式（`90d` / `12h` / `30m`、
単位必須）。出力も journal-server の `gen-key` に合わせ、**トークンだけ stdout・メタデータは
stderr**。パイプで拾える。

トークンの接頭辞は `"sat"`（journal-server が `"sjt"`）。framework の `mint_token` は
`<prefix>_<random>` を作るので、`config.example.toml` に出てくる手書きの `sa-dev-...` とは
形が変わる。

### `device add` の手順と、2 ファイルにまたがる書き込み

ワークスペースは他と同じく `Config::resolved_workspace_dir(&config_path)` で解決し、鍵ファイルは
`[keys].file`（未設定なら `DeviceRegistry::default_key_file` が返す既定の場所）を使う。

1. `<workspace>/.sapphire-agent/devices.toml` を読む
2. `--name` の重複を検査
3. grain-id を採番してデバイス行を追記・保存
4. ホストローカルの鍵ファイル（`[keys].file`）に `generate(prefix, None, label=name, expires_at)`、
   `device_id` に採番した ID を入れて保存
5. トークンを出力

3 と 4 の間で落ちると片方だけ残る。**デバイス行を先に書く**順序にする — 鍵の無いデバイス行は
完全に不活性（誰も認証できない）。逆順だと、中断のたびに孤児の鍵が溜まって誰も掃除しない。

その上で `device add` を**再開可能**にする。同名のデバイスが既に居て、かつこのホストの鍵ファイルに
その `device_id` の鍵が無ければ、鍵だけ発行する。鍵もあるならエラーにして `device rotate` を
案内する。この分岐が無いと、中断状態から抜ける手段が無くなる（`rotate` は既存の鍵を要求するため）。

### `device add` の直後は設定が不正になる

room_profile の紐づけはホスト設定側にあるので、`device add` を実行しただけでは新しいデバイスが
どの room_profile にも属さず、次の起動が下の検査で落ちる。これは意図した順序（コマンドが
`config.toml` を書き換えない）だが、放置すると必ず踏む。

`device add` は完了時に stderr へ、貼り付けられる形の次の一手を出す：

```
id a3f9k2p  created 2026-08-29T11:00:00Z
next: add this device to a room profile in ~/.config/sapphire-agent/config.toml

    [room_profile.<name>]
    devices = ["a3f9k2p"]
```

## 認証

起動時に 1 回組んで `Arc` で共有する：

```rust
pub struct DeviceAuth {
    keys: KeyStore,                                    // ホストローカル
    devices: registry::Devices,                        // ワークスペース
    room_profile_by_device: HashMap<GrainId, String>,  // ホスト config から反転して構築
}

impl DeviceAuth {
    /// トークン → (デバイス, room_profile 名)
    pub fn resolve(&self, token: &str) -> Option<Resolved<'_>>;
}
```

`ServeState`（`src/serve/mod.rs`）が持ち、ambient の `IngestState`（`src/ambient/ingest.rs`）は
今の `DeviceRegistry` の代わりに同じ `Arc` を受け取る。`src/ambient/auth.rs` の `DeviceRegistry` は
この型に吸収されて消える。

差し替わる呼び出しは 3 箇所（`serve/a2a.rs`, `serve/acp.rs`, `serve/mcp.rs` の
`state.config.resolve_a2a_token(&bearer)`）。`Config::resolve_a2a_token` は削除する。

新しい経路では**デバイスも取れる**ので、ambient が既にやっているようにデバイス名を
システムプロンプトへ回せる。今回は配線だけして、実際に載せるかは別の判断とする。

### 解決の失敗はすべて 401

トークンが鍵ファイルに無い／期限切れ／`device_id` が無い／その ID が台帳に無い／台帳のエントリが
retired、のいずれも `None` に潰す。呼び出し側は全部に 401 を返し、区別はログに出す。
`src/ambient/auth.rs` の `resolve` が既にこの方針を取っているので、それを引き継ぐ。

**「台帳に無い鍵を通す」トグル（`allow_unknown_device`）は持たない。** framework 側の spec が
可能性として挙げていたが、fail-closed のほうが安全で、必要になってから足せる。

## 検査（`validate_profiles`）

既存の `api_keys` 重複検査の位置に、次を置く。

- **すべてのデバイス ID は、ちょうど 1 つの room_profile の `devices` に現れなければならない。**
  現れないものはエラー。
- 同じ device ID が 2 つ以上の room_profile に現れたらエラー。
- `devices.toml` に存在しない ID を指していたらエラー。

`verify` はデバイスと room_profile の対応を出力する。手で書く紐づけなので確認手段が要る。

### room_profile を必須にする理由

現状 `DeviceConfig.room_profile` は**宣言されているが誰も読んでいない**（`src/` で読むのは
テストのフィクスチャ 2 箇所のみ）。ambient パイプラインは構造的に LLM ターンを起こさず
（`src/ambient/worker.rs` の冒頭コメント、`Disposition::RecordAndConverse` は `#[allow(dead_code)]`）、
トランスクリプトは名前空間を持たない単一プールに落ちる。つまり**今日の ingest 経路は room_profile
を一度も参照しない**。

しかしそれは ambient が record-only という**途中状態**だからで、設計の到達点ではない。S4 で
ペンダントが会話を始めた瞬間、LLM プロファイルとメモリ名前空間への経路は room_profile 以外に無い。
`DeviceConfig.room_profile` はまさにそのために置かれている。

ここで任意にしておくと、S4 の時点で「room_profile が必須になりました」という 2 度目の破壊的な
設定変更を強いることになる。今 required にすればデバイス 1 台あたり 1 行で済む。副次的に、
`device add` したのにルーティングを書き忘れた、という実際に起きるミスをそのまま拾う。

会話させないセンサー用途のデバイスでも、room_profile の割り当てに害は無い — 「将来の会話が
どこへ行くか」「そのトランスクリプトがどのメモリ名前空間に属するか」を宣言するだけ。

## config_layer の allowlist

`room_profile.*.devices` を `WORKSPACE_ALLOWLIST` に**入れる**。

`room_profile.*.rooms` は既に許可、`room_profile.*.api_keys` は不許可（資格情報のため）。
`devices` は資格情報ではなく `rooms` と同じルーティング情報。ワークスペース層が汚染された場合に
できるのは「デバイスを別の room_profile へ回す」ことで、これは `rooms` で既にできることと同程度。

`devices.toml` 自体は `.sapphire-agent/config.toml` の隣に置かれるが、allowlist フィルタは通らない。
それでも**アクセス権は一切与えられない** — トークンはホストローカルの鍵ファイルにあり、鍵側が
`device_id` を名指しする向きなので、台帳は鍵が既に知っている ID にメタデータを供給するだけ。
ただし `user_id` は将来の帰属表示を左右するので、帰属の詐称だけは理屈上できる。単一運用者の
同期ワークスペースという脅威モデルでは許容する。

## 移行（破壊的変更 2 つ）

`config.toml` から 2 つ消える。

1. **`[device.<name>]` ブロック** → `devices.toml` + `[room_profile.*].devices` へ
2. **`[room_profile.*].api_keys`** → `devices` へ

どちらも**黙って無視させず、起動時に明示的なエラーで落とす**。`standby_mode` が同じ扱いを
受けている先例（`src/main.rs`）に倣う。理由も同じで、黙って無視すると症状が「壊れた設定」ではなく
「壊れたデバイス」に見える — `api_keys` を無視すれば `/acp` が全クライアントを 401 で弾き、
`[device.*]` を無視すれば ambient が全セグメントを拒否する。どちらも設定ファイルを見に行く動機が
湧きにくい壊れ方。

`RoomProfileConfig` は `deny_unknown_fields` を付けていないので、残った `api_keys` は今のままだと
静かに捨てられる。`standby_mode` と同型に、フィールドを残したまま起動時に検査する。

**移行コマンドは作らない。** 旧 `[device.*]` から `devices.toml` を生成する `device import` は
書けるが、対象は 1 人の運用者の数台で、一度きり。エラーメッセージに具体的な手順（この
`[device.pendant]` に対して `sapphire-agent device add --name pendant` を実行し、出たトークンを
デバイスに設定し直す）を書くほうが、使い捨てのコードを残すより安い。旧トークンは手書きの平文で、
新しい鍵ファイルの ID 体系に紐づけ直せないため、**トークンは再発行になる**。

## テスト

- `DeviceAuth::resolve` — 正常系、鍵ファイルに無い、期限切れ、`device_id` 無し、台帳に無い、
  retired。`src/ambient/auth.rs` の既存テスト群を移植・拡張する
- `validate_profiles` — 未割り当てデバイス、重複割り当て、存在しない ID
- 起動時の移行検査 — 残存 `[device.*]` と残存 `api_keys` がそれぞれエラーになり、
  メッセージが対象の名前を含むこと
- `config_layer` — `room_profile.*.devices` がワークスペース層から通ること
- `device add` — 台帳と鍵の両方が書かれること、名前重複の拒否、中断状態からの再開
- CLI パース（`src/main.rs` の既存テストの形に合わせる）

## 別イシューに切り出すもの

**ambient のトランスクリプトが名前空間を持たない。** `TranscriptStore` は
`<cache_root>/transcripts/<日付>.jsonl` の単一プール（`src/ambient/startup.rs`）で、memory
namespace とは無関係。room_profile が名前空間を運ぶ以上、メモリ名前空間の異なる 2 つの
room_profile が ambient のトランスクリプトを共有することになる（仕事用の会話とプライベートの
会話が同じプールに落ちる）。S4 で会話が始まると顕在化するが、本 spec の範囲では直さない。
