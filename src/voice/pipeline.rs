//! In-process voice pipeline runner for non-SSE callers.
//!
//! The HTTP voice/pipeline_run endpoint streams every stage (STT
//! transcript, assistant text, TTS audio chunks) over SSE so the
//! satellite can start playback before TTS finishes. Discord voice
//! doesn't have an SSE channel back to the speaker — songbird only
//! plays a fully-assembled audio source — so it needs the same
//! pipeline but with audio accumulated into a buffer instead of
//! streamed event-by-event.
//!
//! Rather than duplicate the STT → LLM → TTS plumbing, this module
//! reuses [`crate::serve::run_llm_turn`] with a discarded SSE sender
//! and buffers TTS PCM into a `Vec<i16>` returned to the caller.

use std::convert::Infallible;
use std::sync::Arc;

use anyhow::{Result, anyhow, bail};
use axum::response::sse::Event;
use serde_json::Value;
use tokio::sync::mpsc;
use tracing::{info, warn};

use crate::serve::{ServeState, run_llm_turn};

/// Output of one full voice turn: what the user said, what the
/// assistant replied (text), and the synthesized TTS audio at the
/// pipeline sample rate (16 kHz mono s16le).
pub struct VoiceTurnOutcome {
    pub transcript: String,
    pub assistant_text: String,
    pub audio_pcm_16k: Vec<i16>,
}

/// Run one voice turn end-to-end against an already-pinned session.
///
/// The caller must have inserted `(session_id → room_profile)` into
/// `state.session_room_profiles` so `run_llm_turn` resolves the
/// right LLM profile / memory namespace, and so we can look up the
/// session's voice_pipeline. For Discord voice that pinning happens
/// once when the bot joins the channel.
pub async fn run_voice_turn_buffered(
    state: Arc<ServeState>,
    session_id: String,
    pcm_16k: Vec<i16>,
    language: Option<String>,
) -> Result<VoiceTurnOutcome> {
    let voice_registry = state
        .voice
        .as_ref()
        .ok_or_else(|| anyhow!("voice pipeline unavailable: no STT/TTS providers configured"))?
        .clone();

    let rp_name = state
        .session_room_profiles
        .lock()
        .await
        .get(&session_id)
        .cloned()
        .ok_or_else(|| anyhow!("session '{session_id}' has no room_profile pinned"))?;
    let pipeline = state
        .config
        .voice_pipeline_for_room_profile(&rp_name)
        .ok_or_else(|| anyhow!("room_profile '{rp_name}' has no voice_pipeline configured"))?
        .clone();

    let stt = voice_registry
        .stt(&pipeline.stt_provider)
        .ok_or_else(|| anyhow!("stt_provider '{}' not instantiated", pipeline.stt_provider))?;
    let tts = voice_registry
        .tts(&pipeline.tts_provider)
        .ok_or_else(|| anyhow!("tts_provider '{}' not instantiated", pipeline.tts_provider))?;

    info!(
        "voice turn (buffered): STT via '{}' ({} samples, lang={:?})",
        stt.name(),
        pcm_16k.len(),
        language.as_deref().or(pipeline.language.as_deref()),
    );

    let lang = language.as_deref().or(pipeline.language.as_deref());
    let transcript = stt
        .transcribe(&pcm_16k, lang)
        .await
        .map_err(|e| anyhow!("STT failed: {e:#}"))?;
    if transcript.trim().is_empty() {
        bail!("STT returned empty transcript — skipping LLM turn");
    }

    // run_llm_turn needs an SSE sender; we don't care about its
    // events here, so we hand it a channel and immediately drop the
    // receiver. The send().await calls inside short-circuit when
    // the receiver is gone, so we incur no real work for them.
    let (tx, rx) = mpsc::channel::<Result<Event, Infallible>>(8);
    drop(rx);
    // run_llm_turn uses req_id only to shape error_event payloads.
    // No real client is listening here, so a placeholder is fine.
    let dummy_req_id = Value::Null;
    let outcome = run_llm_turn(
        Arc::clone(&state),
        session_id.clone(),
        transcript.clone(),
        dummy_req_id,
        tx,
    )
    .await;
    let reply_text = outcome
        .text
        .ok_or_else(|| anyhow!("LLM turn produced no reply text"))?;

    info!(
        "voice turn (buffered): TTS via '{}' ({} chars)",
        tts.name(),
        reply_text.len(),
    );

    let (pcm_tx, mut pcm_rx) = mpsc::channel::<Vec<i16>>(32);
    let reply_for_tts = reply_text.clone();
    let synth_handle =
        tokio::spawn(async move { tts.synthesize_stream(&reply_for_tts, pcm_tx).await });
    let mut audio: Vec<i16> = Vec::new();
    while let Some(chunk) = pcm_rx.recv().await {
        audio.extend_from_slice(&chunk);
    }
    match synth_handle.await {
        Ok(Ok(())) => {
            if audio.is_empty() {
                warn!(
                    "TTS returned no audio (provider: {})",
                    pipeline.tts_provider
                );
                bail!(
                    "TTS provider '{}' produced no audio",
                    pipeline.tts_provider
                );
            }
        }
        Ok(Err(e)) => bail!("TTS synthesis failed: {e:#}"),
        Err(join_err) => bail!("TTS task panicked: {join_err}"),
    }

    Ok(VoiceTurnOutcome {
        transcript,
        assistant_text: reply_text,
        audio_pcm_16k: audio,
    })
}
