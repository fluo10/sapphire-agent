//! Local STT via whisper.cpp through the [`whisper_rs`] crate.
//!
//! Gated behind the `voice-whisper` cargo feature so the default build
//! does not pay whisper.cpp's C compile cost. Users who only need cloud
//! STT skip this module entirely; users who want offline STT enable the
//! feature and point a `[stt_provider.<n>] type = "whisper_rs"` block
//! at a ggml/gguf model file on disk.
//!
//! Models are auto-downloaded from
//! `https://huggingface.co/ggerganov/whisper.cpp` when the configured
//! path's basename matches a known whisper model name and the file
//! does not yet exist. See [`KNOWN_MODELS`].

use std::path::{Path, PathBuf};
use std::sync::Arc;

use async_trait::async_trait;
use tracing::info;
use whisper_rs::{
    FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters,
    convert_integer_to_float_audio,
};

use crate::voice::{PIPELINE_SAMPLE_RATE, SttProvider};

/// ggml model filenames the auto-downloader recognises. Each maps 1:1
/// to a file under `https://huggingface.co/ggerganov/whisper.cpp/resolve/main/`.
/// Anything outside this list is treated as user-managed (error if missing).
const KNOWN_MODELS: &[&str] = &[
    "ggml-tiny.bin",
    "ggml-tiny.en.bin",
    "ggml-base.bin",
    "ggml-base.en.bin",
    "ggml-small.bin",
    "ggml-small.en.bin",
    "ggml-medium.bin",
    "ggml-medium.en.bin",
    "ggml-large-v1.bin",
    "ggml-large-v2.bin",
    "ggml-large-v3.bin",
    "ggml-large-v3-turbo.bin",
];

pub(crate) struct WhisperRsStt {
    name: String,
    ctx: Arc<WhisperContext>,
}

impl WhisperRsStt {
    pub(crate) fn new(name: String, model_path: &str) -> anyhow::Result<Self> {
        let expanded = shellexpand::tilde(model_path).into_owned();
        let resolved = ensure_model_present(Path::new(&expanded))
            .map_err(|e| anyhow::anyhow!("whisper-rs '{name}': {e:#}"))?;
        let ctx = WhisperContext::new_with_params(
            resolved.to_string_lossy().as_ref(),
            WhisperContextParameters::default(),
        )
        .map_err(|e| {
            anyhow::anyhow!("whisper-rs: failed to load model '{}': {e}", resolved.display())
        })?;
        Ok(Self {
            name,
            ctx: Arc::new(ctx),
        })
    }
}

/// Ensure the whisper model at `path` is present, downloading it from
/// the HuggingFace whisper.cpp repo if its basename is in
/// [`KNOWN_MODELS`]. Returns the resolved path (same as the input, after
/// directory creation).
fn ensure_model_present(path: &Path) -> anyhow::Result<PathBuf> {
    if path.exists() {
        return Ok(path.to_path_buf());
    }

    let basename = path
        .file_name()
        .and_then(|s| s.to_str())
        .ok_or_else(|| anyhow::anyhow!("model path has no filename: {}", path.display()))?;

    if !KNOWN_MODELS.contains(&basename) {
        anyhow::bail!(
            "model file not found at '{}', and basename '{basename}' is not a known \
             whisper.cpp model. Either place the file manually, or use one of: {}",
            path.display(),
            KNOWN_MODELS.join(", ")
        );
    }

    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| {
            anyhow::anyhow!("failed to create model dir '{}': {e}", parent.display())
        })?;
    }

    let url = format!(
        "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/{basename}"
    );
    info!(
        "Downloading whisper model from {url} to {} (one-time, may take several minutes)",
        path.display()
    );

    // Stream to a temp file alongside the target so a half-finished
    // download doesn't masquerade as a complete model on the next boot.
    let tmp = path.with_extension("bin.partial");
    let resp = reqwest::blocking::Client::builder()
        // Models are 75 MB (tiny) to 3 GB (large-v3); slow links need
        // hours. No timeout — let the user Ctrl-C if it stalls.
        .timeout(None)
        .build()?
        .get(&url)
        .send()?
        .error_for_status()?;
    let mut reader = resp;
    let mut file = std::fs::File::create(&tmp)
        .map_err(|e| anyhow::anyhow!("failed to create '{}': {e}", tmp.display()))?;
    std::io::copy(&mut reader, &mut file)
        .map_err(|e| anyhow::anyhow!("download to '{}' failed: {e}", tmp.display()))?;
    std::fs::rename(&tmp, path).map_err(|e| {
        anyhow::anyhow!(
            "failed to move '{}' → '{}': {e}",
            tmp.display(),
            path.display()
        )
    })?;
    info!("Whisper model ready at {}", path.display());
    Ok(path.to_path_buf())
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
