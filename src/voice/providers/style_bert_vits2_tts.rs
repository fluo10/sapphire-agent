//! Style-Bert-VITS2 TTS client (https://github.com/litagin02/Style-Bert-VITS2).
//!
//! Server protocol: `POST /voice` with query parameters →
//! `200 audio/wav` body. Single round-trip, no SSE / queue dance to
//! deal with (unlike Gradio). Per-call style parameters can override
//! the deployment defaults.

use std::time::Duration;

use async_trait::async_trait;
use tokio::sync::mpsc;

use crate::config::StyleBertVits2Config;
use crate::voice::providers::wav_stream;
use crate::voice::{PIPELINE_SAMPLE_RATE, TtsProvider};

const SYNTHESIZE_TIMEOUT: Duration = Duration::from_secs(120);

pub(crate) struct StyleBertVits2Tts {
    name: String,
    base_url: String,
    cfg: StyleBertVits2Config,
    client: reqwest::Client,
}

impl StyleBertVits2Tts {
    pub(crate) fn new(name: String, cfg: StyleBertVits2Config) -> anyhow::Result<Self> {
        let client = reqwest::Client::builder()
            .timeout(SYNTHESIZE_TIMEOUT)
            .build()?;
        let base_url = cfg.base_url.trim_end_matches('/').to_string();
        Ok(Self {
            name,
            base_url,
            cfg,
            client,
        })
    }
}

#[async_trait]
impl TtsProvider for StyleBertVits2Tts {
    fn name(&self) -> &str {
        &self.name
    }

    fn sample_rate(&self) -> u32 {
        // We resample on the way out, so the pipeline always sees
        // 16 kHz regardless of SBV2's native output (usually 44.1 kHz).
        PIPELINE_SAMPLE_RATE
    }

    async fn synthesize_stream(
        &self,
        text: &str,
        pcm_tx: mpsc::Sender<Vec<i16>>,
    ) -> anyhow::Result<()> {
        let url = format!("{}/voice", self.base_url);
        let mut query: Vec<(&str, String)> = vec![
            ("text", text.to_string()),
            ("model_id", self.cfg.model_id.to_string()),
            ("speaker_id", self.cfg.speaker_id.to_string()),
            ("sdp_ratio", self.cfg.sdp_ratio.to_string()),
            ("noise", self.cfg.noise.to_string()),
            ("noisew", self.cfg.noisew.to_string()),
            ("length", self.cfg.length.to_string()),
            ("style_weight", self.cfg.style_weight.to_string()),
        ];
        if let Some(s) = &self.cfg.style {
            query.push(("style", s.clone()));
        }
        if let Some(l) = &self.cfg.language {
            query.push(("language", l.clone()));
        }

        let resp = self
            .client
            .post(&url)
            .query(&query)
            .send()
            .await
            .map_err(|e| anyhow::anyhow!("SBV2 POST {url} failed: {e}"))?;
        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            anyhow::bail!("SBV2 returned {status}: {body}");
        }
        let bytes = resp
            .bytes()
            .await
            .map_err(|e| anyhow::anyhow!("SBV2 body read: {e}"))?
            .to_vec();

        // Decode WAV → mono → 16 kHz → 20 ms chunks via the shared
        // helper. Same path Gradio TTS / OpenAI TTS use.
        wav_stream::stream_wav(bytes, pcm_tx).await
    }
}
