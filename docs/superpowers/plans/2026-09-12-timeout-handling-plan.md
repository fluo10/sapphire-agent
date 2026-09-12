# Issue #258: プロバイダSSEストリームとサブエージェントターンのタイムアウト実装

## 背景・問題（Issue #258）

1. **プロバイダHTTPにタイムアウトがない**: `server/src/provider/openai_compatible.rs` と `server/src/provider/anthropic.rs` の両方が `reqwest::Client::new()`（デフォルト=タイムアウトなし）を使用し、SSEストリームを `while let Some(chunk) = stream.next().await` で読んでいる。アップストリーム（OpenRouter等）がヘッダ送信後にストリームが停滞すると、セマンティックな期限が存在しないため `.await` が永久に待つ。実例: 2026-09-12 plannerサブエージェントが15:02開始→15:23最後にログ停止、ターンが途中のまま永久ハング。
2. **サブエージェントターン全体のタイムアウト機構がない**: `server/src/tools/subagent.rs` の `run_and_store` はプロバイダ呼び出しを含むターン全体を無期限に待つ。唯一の救出経路（セッションごとキャンセル）もACP側が待たないため運用上機能しない。結果「黙って死ぬ」。

## 解決方針

1. **プロバイダ層**: 両プロバイダの `Client::new()` を `Client::builder()` へ変更し `connect_timeout` を付与。SSEチャンク読み (`stream.next()`) を `tokio::time::timeout` で包み、チャンク間アイドルが設定値を超えたらエラーで return。レスポンス全体長とは別枠にしたいので total timeout ではなく **idle timeout** を採用。値は設定可能（`stream_idle_timeout_secs` デフォルト300、`connect_timeout_secs` デフォルト15程度）にする。設定先は `OpenAICompatibleConfig` と `AnthropicConfig` の両方。
2. **サブエージェント層**: `run_and_store` のターン実行を `tokio::time::timeout` で包む。期限超過時はそれまでの履歴をサブエージェントキャッシュへ保存した上で、明示的なタイムアウトエラーを親へのtool errorとして返す。設定可能（`turn_timeout_secs`、デフォルト900）にする。設定先はサブエージェント設定（既存の設定構造を確認して適切な場所に。`SubagentCacheConfig` ではなくサブエージェント自体の設定が望ましいが、既存 config 構造に無い場合はサブエージェント設定を新設 or 既存の適切な箇所へ）。

## File Structure（変更対象候補）

- `server/src/provider/openai_compatible.rs` — Client::new() → Client::builder + connect_timeout、stream.next() を timeout で包む
- `server/src/provider/anthropic.rs` — 同上
- `server/src/config.rs` — `OpenAICompatibleConfig` / `AnthropicConfig` にタイムアウト設定フィールド追加（デフォルト関数付き）
- `server/src/tools/subagent.rs` — `run_and_store` をターンタイムアウトで包む（期限超過時に履歴保存＋エラー返却）
- 設定構造が置かれている場所（config.rs）と、serve側での設定読み出し経路を確認すること

## Global Constraints

- 既存の `Default` 実装 / serde default を壊さない（既存の設定ファイルが無変更で動くこと。new フィールドは全て `#[serde(default = "...")]`）。
- `tokio` の `time` feature が必要なら Cargo.toml で有効にする（既に有効な可能性が高いが確認）。
- 既存テストを全てパスさせる。ハングを再現するテストを最低1つ追加する（provider側は応答を滞留させるスタブHTTP、subagent側はハングするStubProviderを用い、タイムアウトでエラーが返ることを確認。既存の `HangingChat`/`StubProvider` 構造（`server/src/serve/mod.rs` の cfg(test) 内）を流用できる）。
- 既存のコメントスタイル（なぜそうするかを説明する詳細コメント）に合わせる。
- rustfmt に適合させる。全テストを通す。
