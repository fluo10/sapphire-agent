//! Generic Gradio Web UI client used as a TTS provider.
//!
//! Talks to any Gradio 4.x app's call API (`POST /gradio_api/call/<fn>`
//! followed by `GET /gradio_api/call/<fn>/<event_id>` for the result),
//! retrieves the synthesized audio file URL from the response JSON via
//! a user-supplied JSON Pointer, fetches the WAV bytes, and streams
//! 16 kHz mono s16le PCM chunks into the pipeline.
//!
//! Streaming is currently "synthesize, then pace" — the Gradio call API
//! is request/response, so the first chunk does not leave the server
//! until the full synthesis is done. Chunks are paced at 20 ms each so
//! the consumer can play them smoothly without buffering the entire
//! reply before audio starts.

use std::io::Cursor;
use std::time::Duration;

use async_trait::async_trait;
use futures_util::StreamExt;
use serde_json::Value;
use tokio::sync::mpsc;

use crate::voice::{PIPELINE_SAMPLE_RATE, TtsProvider};

/// Initial timeout for the `POST /gradio_api/call/...` round-trip that
/// schedules the synthesis job. Stricter than the SSE wait below since
/// only an event_id comes back here.
const SCHEDULE_TIMEOUT: Duration = Duration::from_secs(30);
/// Upper bound for the SSE wait that polls for the synthesis result.
/// Generous to allow slow first-token-on-cold-start on the Gradio side.
const SYNTHESIZE_TIMEOUT: Duration = Duration::from_secs(300);

pub(crate) struct GradioTts {
    name: String,
    base_url: String,
    fn_name: String,
    payload_template: String,
    audio_field: String,
    client: reqwest::Client,
}

impl GradioTts {
    pub(crate) fn new(
        name: String,
        base_url: String,
        fn_name: String,
        payload_template: String,
        audio_field: String,
    ) -> anyhow::Result<Self> {
        let client = reqwest::Client::builder()
            .timeout(SYNTHESIZE_TIMEOUT)
            .build()?;
        let fn_name = fn_name.trim_start_matches('/').to_string();
        let base_url = base_url.trim_end_matches('/').to_string();
        if !audio_field.starts_with('/') {
            anyhow::bail!(
                "tts_provider '{name}': audio_field must be a JSON Pointer starting with '/' \
                 (got: '{audio_field}')"
            );
        }
        Ok(Self {
            name,
            base_url,
            fn_name,
            payload_template,
            audio_field,
            client,
        })
    }

    /// Substitute `{{text}}` in the payload template with `text` (escaped
    /// for JSON string context), parse the result, and POST it.
    async fn schedule(&self, text: &str) -> anyhow::Result<String> {
        let escaped = serde_json::to_string(text)?;
        // serde_json::to_string wraps the value in quotes; strip them so
        // the template can place the literal inside its own quotes.
        let unquoted = &escaped[1..escaped.len() - 1];
        let body = self.payload_template.replace("{{text}}", unquoted);
        let parsed: Value = serde_json::from_str(&body).map_err(|e| {
            anyhow::anyhow!(
                "tts_provider '{}': payload template did not parse as JSON after substitution: {e}\nbody:\n{body}",
                self.name
            )
        })?;

        let url = format!("{}/gradio_api/call/{}", self.base_url, self.fn_name);
        let resp = self
            .client
            .post(&url)
            .timeout(SCHEDULE_TIMEOUT)
            .json(&parsed)
            .send()
            .await?
            .error_for_status()?;
        let json: Value = resp.json().await?;
        json.get("event_id")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .ok_or_else(|| {
                anyhow::anyhow!("Gradio schedule response missing event_id: {json}")
            })
    }

    /// Poll the SSE stream until either a `complete` event arrives (then
    /// returns the parsed JSON payload) or an `error` event ends the
    /// stream.
    async fn await_completion(&self, event_id: &str) -> anyhow::Result<Value> {
        let url = format!(
            "{}/gradio_api/call/{}/{}",
            self.base_url, self.fn_name, event_id
        );
        let resp = self.client.get(&url).send().await?.error_for_status()?;
        let mut stream = resp.bytes_stream();
        let mut buf = String::new();
        let mut current_event = String::new();
        while let Some(chunk) = stream.next().await {
            let bytes = chunk?;
            buf.push_str(std::str::from_utf8(&bytes)?);
            // SSE frames are separated by blank lines.
            while let Some(idx) = buf.find("\n\n") {
                let frame: String = buf.drain(..idx + 2).collect();
                let mut event_name: Option<&str> = None;
                let mut data: Option<&str> = None;
                for line in frame.lines() {
                    if let Some(rest) = line.strip_prefix("event:") {
                        event_name = Some(rest.trim());
                    } else if let Some(rest) = line.strip_prefix("data:") {
                        data = Some(rest.trim());
                    }
                }
                if let Some(name) = event_name {
                    current_event = name.to_string();
                }
                let Some(data) = data else { continue };
                match current_event.as_str() {
                    "complete" => {
                        return serde_json::from_str(data).map_err(|e| {
                            anyhow::anyhow!("Gradio 'complete' payload not JSON: {e}\n{data}")
                        });
                    }
                    "error" => anyhow::bail!("Gradio reported error: {data}"),
                    // "generating" / "heartbeat" / etc. — ignore intermediates.
                    _ => continue,
                }
            }
        }
        anyhow::bail!("Gradio SSE stream ended without a 'complete' event")
    }

    /// Resolve `audio_field` against the response JSON to a single URL or
    /// path string. Objects with `url` / `path` are unwrapped automatically.
    fn resolve_audio_ref(&self, response: &Value) -> anyhow::Result<String> {
        let target = response.pointer(&self.audio_field).ok_or_else(|| {
            anyhow::anyhow!(
                "tts_provider '{}': audio_field '{}' not found in response: {response}",
                self.name,
                self.audio_field
            )
        })?;
        match target {
            Value::String(s) => Ok(s.clone()),
            Value::Object(map) => {
                if let Some(Value::String(s)) = map.get("url") {
                    return Ok(s.clone());
                }
                if let Some(Value::String(s)) = map.get("path") {
                    return Ok(s.clone());
                }
                anyhow::bail!(
                    "tts_provider '{}': audio_field resolved to an object with no \
                     'url' or 'path' string field: {target}",
                    self.name
                )
            }
            _ => anyhow::bail!(
                "tts_provider '{}': audio_field resolved to a non-string non-object value: {target}",
                self.name
            ),
        }
    }

    async fn fetch_audio_bytes(&self, audio_ref: &str) -> anyhow::Result<Vec<u8>> {
        let url = if audio_ref.starts_with("http://") || audio_ref.starts_with("https://") {
            audio_ref.to_string()
        } else if audio_ref.starts_with('/') {
            format!("{}{}", self.base_url, audio_ref)
        } else {
            // Bare path returned by Gradio's `gr.Audio` — Gradio serves
            // these via `/gradio_api/file=<path>`.
            format!("{}/gradio_api/file={}", self.base_url, audio_ref)
        };
        let bytes = self
            .client
            .get(&url)
            .send()
            .await?
            .error_for_status()?
            .bytes()
            .await?;
        Ok(bytes.to_vec())
    }
}

#[async_trait]
impl TtsProvider for GradioTts {
    fn name(&self) -> &str {
        &self.name
    }

    fn sample_rate(&self) -> u32 {
        PIPELINE_SAMPLE_RATE
    }

    async fn synthesize_stream(
        &self,
        text: &str,
        pcm_tx: mpsc::Sender<Vec<i16>>,
    ) -> anyhow::Result<()> {
        let event_id = self.schedule(text).await?;
        let response = self.await_completion(&event_id).await?;
        let audio_ref = self.resolve_audio_ref(&response)?;
        let bytes = self.fetch_audio_bytes(&audio_ref).await?;

        // Decode WAV. hound handles 16-bit PCM directly; floats are
        // converted in spawn_blocking since hound is sync and decoding
        // a few MB of WAV shouldn't block the runtime.
        let (samples, sample_rate, channels) =
            tokio::task::spawn_blocking(move || decode_wav(&bytes))
                .await
                .map_err(|e| anyhow::anyhow!("WAV decode task panicked: {e}"))??;

        let mono = if channels == 1 {
            samples
        } else {
            // Average channels for stereo+ → mono.
            samples
                .chunks(channels as usize)
                .map(|frame| {
                    let sum: i32 = frame.iter().map(|s| *s as i32).sum();
                    (sum / channels as i32) as i16
                })
                .collect()
        };

        let resampled = if sample_rate == PIPELINE_SAMPLE_RATE {
            mono
        } else {
            linear_resample(&mono, sample_rate, PIPELINE_SAMPLE_RATE)
        };

        // Pace 20 ms chunks (320 samples @ 16 kHz). No actual sleep —
        // backpressure on the bounded mpsc paces the producer naturally.
        let chunk_size = (PIPELINE_SAMPLE_RATE as usize) / 50;
        for chunk in resampled.chunks(chunk_size) {
            if pcm_tx.send(chunk.to_vec()).await.is_err() {
                // Receiver dropped — caller cancelled.
                break;
            }
        }
        Ok(())
    }
}

/// Decode a WAV blob into interleaved i16 samples. Returns
/// `(samples, sample_rate, channels)`.
fn decode_wav(bytes: &[u8]) -> anyhow::Result<(Vec<i16>, u32, u16)> {
    let mut reader = hound::WavReader::new(Cursor::new(bytes))
        .map_err(|e| anyhow::anyhow!("WAV header parse failed: {e}"))?;
    let spec = reader.spec();
    let samples: Vec<i16> = match spec.sample_format {
        hound::SampleFormat::Int => {
            let bits = spec.bits_per_sample;
            reader
                .samples::<i32>()
                .map(|s| {
                    s.map(|v| {
                        // Normalize any int bit depth to i16.
                        if bits == 16 {
                            v as i16
                        } else {
                            let shift = (bits as i32) - 16;
                            if shift > 0 {
                                (v >> shift) as i16
                            } else {
                                (v << (-shift)) as i16
                            }
                        }
                    })
                })
                .collect::<Result<Vec<_>, _>>()
                .map_err(|e| anyhow::anyhow!("WAV sample read failed: {e}"))?
        }
        hound::SampleFormat::Float => reader
            .samples::<f32>()
            .map(|s| s.map(|v| (v.clamp(-1.0, 1.0) * (i16::MAX as f32)) as i16))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| anyhow::anyhow!("WAV f32 sample read failed: {e}"))?,
    };
    Ok((samples, spec.sample_rate, spec.channels))
}

/// Linear-interpolation resampler. Quick and simple; adequate for
/// speech in v1. If TTS quality matters, swap in rubato later.
fn linear_resample(input: &[i16], src_rate: u32, dst_rate: u32) -> Vec<i16> {
    if input.is_empty() || src_rate == dst_rate {
        return input.to_vec();
    }
    let ratio = src_rate as f64 / dst_rate as f64;
    let out_len = ((input.len() as f64) / ratio).round() as usize;
    let mut out = Vec::with_capacity(out_len);
    for i in 0..out_len {
        let src_pos = i as f64 * ratio;
        let idx = src_pos.floor() as usize;
        let frac = src_pos - idx as f64;
        let s0 = input[idx.min(input.len() - 1)] as f64;
        let s1 = input[(idx + 1).min(input.len() - 1)] as f64;
        let interpolated = s0 * (1.0 - frac) + s1 * frac;
        out.push(interpolated.round().clamp(i16::MIN as f64, i16::MAX as f64) as i16);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resample_downsample_halves_length_approximately() {
        let input: Vec<i16> = (0..32000).map(|i| (i % 1000) as i16).collect();
        let out = linear_resample(&input, 32000, 16000);
        // Allow ±1 rounding tolerance.
        assert!(
            (out.len() as isize - 16000).abs() <= 1,
            "got len {}",
            out.len()
        );
    }

    #[test]
    fn resample_no_op_when_rates_match() {
        let input: Vec<i16> = vec![100, 200, 300];
        let out = linear_resample(&input, 16000, 16000);
        assert_eq!(out, input);
    }

    #[test]
    fn resolve_audio_ref_string() {
        let g = GradioTts::new(
            "test".into(),
            "http://localhost".into(),
            "predict".into(),
            "{}".into(),
            "/data/0".into(),
        )
        .unwrap();
        let json = serde_json::json!({"data": ["http://x/y.wav"]});
        assert_eq!(g.resolve_audio_ref(&json).unwrap(), "http://x/y.wav");
    }

    #[test]
    fn resolve_audio_ref_object_with_url() {
        let g = GradioTts::new(
            "test".into(),
            "http://localhost".into(),
            "predict".into(),
            "{}".into(),
            "/data/0".into(),
        )
        .unwrap();
        let json = serde_json::json!({"data": [{"url": "/file=foo.wav", "path": "/tmp/foo.wav"}]});
        assert_eq!(g.resolve_audio_ref(&json).unwrap(), "/file=foo.wav");
    }

    #[test]
    fn payload_substitution_quotes_safely() {
        let g = GradioTts::new(
            "test".into(),
            "http://localhost".into(),
            "predict".into(),
            r#"{"data": ["{{text}}"]}"#.into(),
            "/data/0".into(),
        )
        .unwrap();
        // Escaped via serde_json so quotes/backslashes are preserved.
        let escaped = serde_json::to_string("hello \"world\"").unwrap();
        let unquoted = &escaped[1..escaped.len() - 1];
        let body = g.payload_template.replace("{{text}}", unquoted);
        let parsed: Value = serde_json::from_str(&body).unwrap();
        assert_eq!(parsed["data"][0].as_str().unwrap(), "hello \"world\"");
    }
}
