//! Local STT via whisper.cpp through the [`whisper_rs`] crate.
//!
//! Gated behind the `voice-whisper` cargo feature so the default build
//! does not pay whisper.cpp's C compile cost. Users who only need cloud
//! STT skip this module entirely; users who want offline STT enable the
//! feature and point a `[stt_provider.<n>] type = "whisper_rs"` block
//! at a ggml/gguf model file on disk.

use std::sync::Arc;

use async_trait::async_trait;
use whisper_rs::{
    FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters,
    convert_integer_to_float_audio,
};

use crate::voice::{PIPELINE_SAMPLE_RATE, SttProvider};

pub(crate) struct WhisperRsStt {
    name: String,
    ctx: Arc<WhisperContext>,
}

impl WhisperRsStt {
    pub(crate) fn new(name: String, model_path: &str) -> anyhow::Result<Self> {
        let expanded = shellexpand::tilde(model_path).into_owned();
        let ctx = WhisperContext::new_with_params(&expanded, WhisperContextParameters::default())
            .map_err(|e| anyhow::anyhow!("whisper-rs: failed to load model '{expanded}': {e}"))?;
        Ok(Self {
            name,
            ctx: Arc::new(ctx),
        })
    }
}

#[async_trait]
impl SttProvider for WhisperRsStt {
    fn name(&self) -> &str {
        &self.name
    }

    fn sample_rate(&self) -> u32 {
        PIPELINE_SAMPLE_RATE
    }

    async fn transcribe(&self, pcm: &[i16], language: Option<&str>) -> anyhow::Result<String> {
        let pcm_owned: Vec<i16> = pcm.to_vec();
        let ctx = Arc::clone(&self.ctx);
        let lang = language.map(String::from);

        // whisper.cpp is synchronous and CPU-bound — keep it off the tokio
        // worker pool.
        let text = tokio::task::spawn_blocking(move || -> anyhow::Result<String> {
            let mut samples = vec![0.0f32; pcm_owned.len()];
            convert_integer_to_float_audio(&pcm_owned, &mut samples)
                .map_err(|e| anyhow::anyhow!("whisper-rs: i16→f32 conversion failed: {e}"))?;

            let mut state = ctx
                .create_state()
                .map_err(|e| anyhow::anyhow!("whisper-rs: create_state failed: {e}"))?;

            let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
            if let Some(l) = lang.as_deref() {
                params.set_language(Some(l));
            }
            // Silence whisper.cpp's own stdout — we surface results via the
            // SSE progress stream instead.
            params.set_print_special(false);
            params.set_print_progress(false);
            params.set_print_realtime(false);
            params.set_print_timestamps(false);

            state
                .full(params, &samples[..])
                .map_err(|e| anyhow::anyhow!("whisper-rs: transcription failed: {e}"))?;

            let mut text = String::new();
            for segment in state.as_iter() {
                let s = segment
                    .to_str()
                    .map_err(|e| anyhow::anyhow!("whisper-rs: segment decode failed: {e}"))?;
                text.push_str(s);
            }
            Ok(text.trim().to_string())
        })
        .await
        .map_err(|e| anyhow::anyhow!("whisper-rs: blocking task panicked: {e}"))??;

        Ok(text)
    }
}
