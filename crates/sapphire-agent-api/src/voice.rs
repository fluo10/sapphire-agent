//! Client-side helpers for the `voice/pipeline_run` MCP method.
//!
//! Wraps the request/response shape used by the server in `serve.rs`:
//! audio in (base64 s16le mono 16 kHz) → SSE progress events
//! (`stt_final`, `assistant_text`, `audio_chunk`) → final JSON result
//! (`transcript`, `assistant_text`).
//!
//! Audio I/O is intentionally not handled here — that's the
//! satellite's job (see `sapphire-call`'s voice subcommand). This
//! module deals only in `Vec<i16>` PCM and JSON-RPC over HTTP/SSE.

use anyhow::Result;
use base64::Engine;
use futures_util::StreamExt;
use serde_json::{Value, json};
use std::sync::atomic::{AtomicU64, Ordering};
use tokio::sync::mpsc;

static REQUEST_ID: AtomicU64 = AtomicU64::new(1);

fn next_id() -> u64 {
    REQUEST_ID.fetch_add(1, Ordering::Relaxed)
}

/// Pipeline sample rate. Must match the server-side constant in
/// `sapphire-agent::voice::PIPELINE_SAMPLE_RATE`. Duplicated here so
/// the API crate doesn't depend on the agent binary.
pub const PIPELINE_SAMPLE_RATE: u32 = 16_000;

/// Wake-word configuration fetched from the server's `voice/config`
/// method. `phrase` is present when wake-word mode is available for
/// the session's room profile; `model` carries the engine-specific
/// payload (a sherpa bundle name OR an inline openWakeWord ONNX blob).
#[derive(Debug, Clone, Default)]
pub struct WakeWordConfig {
    /// Display phrase. For `sherpa_onnx` this is also the
    /// natural-language string the client tokenises against the
    /// bundle's `tokens.txt`. For `open_wake_word` it's just a label
    /// printed when wake fires.
    pub phrase: Option<String>,
    /// `None` when the room profile has no wake-word set; otherwise
    /// the engine-specific model handle.
    pub model: Option<WakeWordModel>,
}

/// Engine-tagged wake-word model payload.
#[derive(Debug, Clone)]
pub enum WakeWordModel {
    /// Sherpa-onnx KWS bundle name. Satellites resolve it against
    /// the sherpa-onnx GitHub releases tag on first use.
    SherpaBundle { name: String },
    /// openWakeWord classifier ONNX, distributed inline so the
    /// satellite never has to talk to the user's training box. The
    /// satellite caches by `sha256`.
    OnnxInline {
        filename: String,
        sha256: String,
        /// Decoded model bytes. `voice_config()` does the base64
        /// decode so callers don't have to.
        bytes: Vec<u8>,
    },
}

/// Fetch the wake-word configuration the server has bound to the
/// given `room_profile`. Returns an empty `WakeWordConfig` when the
/// room profile has no wake-word set.
///
/// No MCP session is needed — voice clients identify themselves by
/// `device_id` + `room_profile` per request rather than by a
/// long-lived session token.
pub async fn voice_config(
    client: &reqwest::Client,
    base: &str,
    room_profile: &str,
) -> Result<WakeWordConfig> {
    let base = base.trim_end_matches('/');
    let body = json!({
        "jsonrpc": "2.0",
        "id": next_id(),
        "method": "voice/config",
        "params": { "room_profile": room_profile },
    });
    let val: Value = client
        .post(format!("{base}/mcp"))
        .json(&body)
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;
    if let Some(err) = val.get("error") {
        let msg = err["message"].as_str().unwrap_or("unknown error");
        anyhow::bail!("voice/config: {msg}");
    }
    let result = &val["result"];
    let phrase = result["wake_word"].as_str().map(String::from);
    let model = parse_wake_word_model(&result["wake_word_model"])?;
    Ok(WakeWordConfig { phrase, model })
}

fn parse_wake_word_model(value: &Value) -> Result<Option<WakeWordModel>> {
    if value.is_null() {
        return Ok(None);
    }
    let format = match value["format"].as_str() {
        Some(f) => f,
        None => return Ok(None),
    };
    match format {
        "sherpa_bundle" => {
            let name = value["name"]
                .as_str()
                .ok_or_else(|| anyhow::anyhow!("voice/config: sherpa_bundle missing 'name'"))?;
            Ok(Some(WakeWordModel::SherpaBundle {
                name: name.to_string(),
            }))
        }
        "onnx_inline" => {
            let filename = value["filename"]
                .as_str()
                .unwrap_or("wake.onnx")
                .to_string();
            let sha256 = value["sha256"]
                .as_str()
                .ok_or_else(|| anyhow::anyhow!("voice/config: onnx_inline missing 'sha256'"))?
                .to_string();
            let data_b64 = value["data_b64"]
                .as_str()
                .ok_or_else(|| anyhow::anyhow!("voice/config: onnx_inline missing 'data_b64'"))?;
            let bytes = base64::engine::general_purpose::STANDARD
                .decode(data_b64.as_bytes())
                .map_err(|e| anyhow::anyhow!("voice/config: onnx_inline base64 decode: {e}"))?;
            Ok(Some(WakeWordModel::OnnxInline {
                filename,
                sha256,
                bytes,
            }))
        }
        other => anyhow::bail!("voice/config: unknown wake_word_model format '{other}'"),
    }
}

/// Streaming events emitted while a `voice/pipeline_run` call is in
/// flight. Consumers typically:
///   * push `AudioChunk` PCM into a playback queue as soon as it
///     arrives (low-latency speech)
///   * display `SttFinal` / `AssistantText` in a transcript view
///   * stop reading on `Done` or `Error`
#[derive(Debug, Clone)]
pub enum VoiceEvent {
    /// Pipeline stage started (e.g. "stt", "intent", "tts").
    StageStart { stage: String },
    /// Pipeline stage ended.
    StageEnd { stage: String },
    /// Final transcript from STT.
    SttFinal { text: String },
    /// Assistant's text reply (echo of what gets synthesized to audio).
    AssistantText { text: String },
    /// One chunk of synthesized speech (mono s16le @ 16 kHz).
    AudioChunk { pcm: Vec<i16> },
    /// Tool call started during the LLM turn.
    ToolStart { name: String },
    /// Tool call finished during the LLM turn.
    ToolEnd { name: String },
    /// Pipeline completed; carries the same transcript + reply text the
    /// caller can also reconstruct from prior `SttFinal` / `AssistantText`.
    Done {
        transcript: String,
        assistant_text: String,
    },
    /// Pipeline failed — message describes the JSON-RPC error.
    Error { message: String },
}

/// Run one voice pipeline pass: upload `pcm` as a single utterance,
/// stream progress events into `event_tx`, and return when the
/// server emits its final JSON-RPC result. Closing `event_tx` is the
/// caller's responsibility — it happens automatically when the
/// returned future is dropped, but explicit closure helps the
/// playback consumer exit.
///
/// `device_id` + `room_profile` are the conversation-routing key. The
/// server derives a deterministic session id from this pair so the
/// satellite can resume the same conversation across restarts /
/// network blips without juggling explicit session tokens.
pub async fn voice_pipeline_run(
    client: &reqwest::Client,
    base: &str,
    device_id: &str,
    room_profile: &str,
    pcm: &[i16],
    language: Option<&str>,
    event_tx: mpsc::Sender<VoiceEvent>,
) -> Result<()> {
    let base = base.trim_end_matches('/');

    // Encode PCM (mono s16le) as base64.
    let mut bytes = Vec::with_capacity(pcm.len() * 2);
    for s in pcm {
        bytes.extend_from_slice(&s.to_le_bytes());
    }
    let audio_b64 = base64::engine::general_purpose::STANDARD.encode(&bytes);

    let mut params = json!({
        "audio": audio_b64,
        "device_id": device_id,
        "room_profile": room_profile,
    });
    if let Some(l) = language {
        params["language"] = json!(l);
    }
    let body = json!({
        "jsonrpc": "2.0",
        "id": next_id(),
        "method": "voice/pipeline_run",
        "params": params,
    });

    let resp = client
        .post(format!("{base}/mcp"))
        .header("accept", "text/event-stream")
        .json(&body)
        .send()
        .await?
        .error_for_status()?;

    let mut stream = resp.bytes_stream();
    let mut buf = String::new();
    while let Some(chunk) = stream.next().await {
        buf.push_str(&String::from_utf8_lossy(&chunk?));
        while let Some(pos) = buf.find("\n\n") {
            let raw: String = buf.drain(..pos + 2).collect();
            let Some(value) = parse_sse_data(&raw) else {
                continue;
            };
            if dispatch_event(&value, &event_tx).await {
                return Ok(());
            }
        }
    }
    Ok(())
}

fn parse_sse_data(raw: &str) -> Option<Value> {
    let data_line = raw.lines().find(|l| l.starts_with("data:"))?;
    let data = data_line.strip_prefix("data:").unwrap_or("").trim();
    serde_json::from_str(data).ok()
}

/// Map one parsed JSON-RPC frame to a `VoiceEvent` and push it. Returns
/// `true` when the stream should be considered complete (final result
/// or error received).
async fn dispatch_event(val: &Value, tx: &mpsc::Sender<VoiceEvent>) -> bool {
    // Notification frames.
    if let Some(method) = val["method"].as_str() {
        let params = &val["params"];
        let evt = match method {
            "notifications/progress" => {
                let kind = params["kind"].as_str().unwrap_or("");
                match kind {
                    "stage" => {
                        let stage = params["stage"].as_str().unwrap_or("").to_string();
                        match params["status"].as_str() {
                            Some("start") => Some(VoiceEvent::StageStart { stage }),
                            Some("end") => Some(VoiceEvent::StageEnd { stage }),
                            _ => None,
                        }
                    }
                    "stt_final" => params["text"].as_str().map(|s| VoiceEvent::SttFinal {
                        text: s.to_string(),
                    }),
                    "assistant_text" => {
                        params["text"].as_str().map(|s| VoiceEvent::AssistantText {
                            text: s.to_string(),
                        })
                    }
                    "audio_chunk" => params["data"].as_str().and_then(decode_audio_chunk),
                    _ => None,
                }
            }
            "tool_start" => params["name"].as_str().map(|s| VoiceEvent::ToolStart {
                name: s.to_string(),
            }),
            "tool_end" => params["name"].as_str().map(|s| VoiceEvent::ToolEnd {
                name: s.to_string(),
            }),
            _ => None,
        };
        if let Some(evt) = evt {
            let _ = tx.send(evt).await;
        }
        return false;
    }

    // Final result / error frames carry an `id`.
    if val.get("result").is_some() {
        let result = &val["result"];
        let transcript = result["transcript"].as_str().unwrap_or("").to_string();
        let assistant_text = result["assistant_text"].as_str().unwrap_or("").to_string();
        let _ = tx
            .send(VoiceEvent::Done {
                transcript,
                assistant_text,
            })
            .await;
        return true;
    }
    if let Some(err) = val.get("error") {
        let message = err["message"]
            .as_str()
            .unwrap_or("unknown error")
            .to_string();
        let _ = tx.send(VoiceEvent::Error { message }).await;
        return true;
    }
    false
}

fn decode_audio_chunk(b64: &str) -> Option<VoiceEvent> {
    let bytes = base64::engine::general_purpose::STANDARD
        .decode(b64.as_bytes())
        .ok()?;
    if bytes.len() % 2 != 0 {
        return None;
    }
    let pcm: Vec<i16> = bytes
        .chunks_exact(2)
        .map(|c| i16::from_le_bytes([c[0], c[1]]))
        .collect();
    Some(VoiceEvent::AudioChunk { pcm })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn audio_chunk_round_trip() {
        let pcm: Vec<i16> = vec![0, 100, -100, 32767, -32768];
        let bytes: Vec<u8> = pcm.iter().flat_map(|s| s.to_le_bytes()).collect();
        let b64 = base64::engine::general_purpose::STANDARD.encode(&bytes);
        let evt = decode_audio_chunk(&b64).unwrap();
        match evt {
            VoiceEvent::AudioChunk { pcm: out } => assert_eq!(out, pcm),
            _ => panic!("wrong variant"),
        }
    }

    #[tokio::test]
    async fn dispatch_stt_final_emits_event() {
        let (tx, mut rx) = mpsc::channel(4);
        let frame = json!({
            "jsonrpc": "2.0",
            "method": "notifications/progress",
            "params": {"kind": "stt_final", "text": "hello"},
        });
        let done = dispatch_event(&frame, &tx).await;
        assert!(!done);
        match rx.recv().await.unwrap() {
            VoiceEvent::SttFinal { text } => assert_eq!(text, "hello"),
            other => panic!("unexpected: {other:?}"),
        }
    }

    #[tokio::test]
    async fn dispatch_final_result_signals_done() {
        let (tx, mut rx) = mpsc::channel(4);
        let frame = json!({
            "jsonrpc": "2.0",
            "id": 1,
            "result": {"transcript": "hi", "assistant_text": "hello there"},
        });
        let done = dispatch_event(&frame, &tx).await;
        assert!(done);
        match rx.recv().await.unwrap() {
            VoiceEvent::Done {
                transcript,
                assistant_text,
            } => {
                assert_eq!(transcript, "hi");
                assert_eq!(assistant_text, "hello there");
            }
            other => panic!("unexpected: {other:?}"),
        }
    }
}
