//! Async RPC bridge between tokio and the bevy main thread.
//!
//! Bevy systems run sync, but the RPC client (`sapphire-agent-rpc`) is
//! all `async`. We park a multi-threaded tokio runtime in a bevy
//! `Resource`, expose `submit_*` helpers that spawn async tasks on it,
//! and hand a `tokio::sync::mpsc::UnboundedReceiver<BridgeEvent>` to
//! the UI side which drains it once per frame via `try_recv`.
//!
//! Pure-ish module: imports nothing from bevy. The bevy `Resource`
//! wrapper lives in `main.rs` so this stays portable to other GUI
//! frameworks if we ever swap bevy out.

use std::sync::Arc;

use anyhow::Result;
use sapphire_agent_rpc::{
    ChatEvent, ChatModality, DeviceMetadata, VoiceEvent, chat_stream, initialize,
    voice_pipeline_run,
};
use tokio::runtime::Runtime;
use tokio::sync::mpsc;

/// Endpoint coordinates needed to talk to the agent. Cloned cheaply
/// into each spawned task.
#[derive(Debug, Clone)]
pub struct Endpoint {
    pub base: String,
    pub token: String,
}

/// Events flowing tokio → bevy. The UI converts these into `ChatEntry`
/// updates and `Session` state transitions.
//
// `AssistantTextPreview.text` and `VoiceDone.{transcript,assistant_text}`
// are intentionally captured but not yet consumed — the streaming-text
// affordance and a future "voice turn ended" toast both want them, and
// pulling them out now would mean re-plumbing the dispatch on the way
// back. The `#[allow(dead_code)]` keeps clippy quiet without hiding
// the fields from `Debug` output.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub enum BridgeEvent {
    /// `initialize` succeeded; chat can begin.
    SessionReady {
        session_id: String,
        display_id: String,
    },
    /// `initialize` failed — the user has to fix Settings and retry.
    SessionFailed { message: String },
    /// A chat turn produced text (final result). Always the last event
    /// for a successful turn.
    ChatText { text: String },
    /// Server emitted a tool call notification. Reserved for future
    /// "AI is using tool X" affordance; ignored by the chat panel for
    /// now but parsed so the dispatch is exhaustive.
    ToolStart { name: String },
    ToolEnd { name: String },
    /// Server emitted `assistant_text` as a progress notification —
    /// fires when the GUI requested audio, so text can land before
    /// audio finishes streaming. Ignored when `ChatText` follows it
    /// (both carry the same string); included so future split-text-
    /// while-audio-streams UX has a hook.
    AssistantTextPreview { text: String },
    /// One PCM chunk (mono s16le @ 16 kHz). Reserved for the audio
    /// playback wiring (phase 2 of the desktop work).
    AudioChunk { pcm: Vec<i16> },
    /// Audio was requested but the server couldn't produce it.
    TtsError { message: String },
    /// JSON-RPC error during a chat turn.
    ChatError { message: String },
    /// `voice/pipeline_run` STT finished — server side has a transcript.
    VoiceTranscript { text: String },
    /// `voice/pipeline_run` produced an assistant reply (text part).
    /// Audio is delivered via the same `AudioChunk` events as the
    /// chat-turn TTS path.
    VoiceAssistantText { text: String },
    /// Voice pipeline completed; carries the final reply text.
    VoiceDone {
        transcript: String,
        assistant_text: String,
    },
    /// Voice pipeline failed.
    VoiceError { message: String },
}

/// Owns the tokio runtime + outgoing event channel.
pub struct RpcBridge {
    runtime: Arc<Runtime>,
    client: reqwest::Client,
    event_tx: mpsc::UnboundedSender<BridgeEvent>,
}

impl RpcBridge {
    pub fn new() -> Result<(Self, mpsc::UnboundedReceiver<BridgeEvent>)> {
        // Multi-threaded so an in-flight chat (which holds open an SSE
        // stream) doesn't block a separate audio playback task on the
        // same thread.
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .thread_name("sapphire-call-desktop-rt")
            .build()?;
        let (event_tx, event_rx) = mpsc::unbounded_channel();
        Ok((
            Self {
                runtime: Arc::new(runtime),
                client: reqwest::Client::new(),
                event_tx,
            },
            event_rx,
        ))
    }

    /// Open or resume a session against `endpoint`. Emits exactly one
    /// of `SessionReady` / `SessionFailed`.
    pub fn submit_initialize(
        &self,
        endpoint: Endpoint,
        resume_session: Option<String>,
        device: Option<DeviceMetadata>,
    ) {
        let client = self.client.clone();
        let tx = self.event_tx.clone();
        self.runtime.spawn(async move {
            let evt = match initialize(&client, &endpoint.base, resume_session, &endpoint.token, device.as_ref()).await
            {
                Ok((session_id, display_id, _is_new)) => BridgeEvent::SessionReady {
                    session_id,
                    display_id,
                },
                Err(e) => BridgeEvent::SessionFailed {
                    message: format!("{e:#}"),
                },
            };
            let _ = tx.send(evt);
        });
    }

    /// Send a chat turn. `want_audio` toggles `modalities=["text","audio"]`.
    pub fn submit_chat(
        &self,
        endpoint: Endpoint,
        session_id: String,
        content: String,
        want_audio: bool,
    ) {
        let client = self.client.clone();
        let tx = self.event_tx.clone();
        self.runtime.spawn(async move {
            // The chat_stream future drives an inner mpsc; we
            // re-publish into the bridge channel so callers only deal
            // with one receiver.
            let (inner_tx, mut inner_rx) = mpsc::channel::<ChatEvent>(32);
            let modalities: &[ChatModality] = if want_audio {
                &[ChatModality::Text, ChatModality::Audio]
            } else {
                &[]
            };
            let pump = async {
                while let Some(evt) = inner_rx.recv().await {
                    let bevt = match evt {
                        ChatEvent::ToolStart { name } => BridgeEvent::ToolStart { name },
                        ChatEvent::ToolEnd { name } => BridgeEvent::ToolEnd { name },
                        ChatEvent::AssistantText { text } => {
                            BridgeEvent::AssistantTextPreview { text }
                        }
                        ChatEvent::AudioChunk { pcm } => BridgeEvent::AudioChunk { pcm },
                        ChatEvent::TtsError { message } => BridgeEvent::TtsError { message },
                        ChatEvent::Done { content } => BridgeEvent::ChatText { text: content },
                        ChatEvent::Error { message } => BridgeEvent::ChatError { message },
                    };
                    if tx.send(bevt).is_err() {
                        break;
                    }
                }
            };
            let run = chat_stream(
                &client,
                &endpoint.base,
                &endpoint.token,
                &session_id,
                &content,
                modalities,
                inner_tx,
            );
            let (_, run_result) = tokio::join!(pump, run);
            if let Err(e) = run_result {
                let _ = tx.send(BridgeEvent::ChatError {
                    message: format!("{e:#}"),
                });
            }
        });
    }

    /// Upload one mic utterance to `voice/pipeline_run`. STT happens
    /// server-side; progress + final result land in the same bridge
    /// event channel as chat events (`VoiceTranscript`,
    /// `VoiceAssistantText`, `AudioChunk`, `VoiceDone` / `VoiceError`).
    pub fn submit_voice(
        &self,
        endpoint: Endpoint,
        device_id: String,
        pcm_16khz: Vec<i16>,
        device: Option<DeviceMetadata>,
    ) {
        let client = self.client.clone();
        let tx = self.event_tx.clone();
        self.runtime.spawn(async move {
            let (inner_tx, mut inner_rx) = mpsc::channel::<VoiceEvent>(64);
            let pump = async {
                while let Some(evt) = inner_rx.recv().await {
                    let bevt = match evt {
                        VoiceEvent::SttFinal { text } => BridgeEvent::VoiceTranscript { text },
                        VoiceEvent::AssistantText { text } => {
                            BridgeEvent::VoiceAssistantText { text }
                        }
                        VoiceEvent::AudioChunk { pcm } => BridgeEvent::AudioChunk { pcm },
                        VoiceEvent::ToolStart { name } => BridgeEvent::ToolStart { name },
                        VoiceEvent::ToolEnd { name } => BridgeEvent::ToolEnd { name },
                        VoiceEvent::Done {
                            transcript,
                            assistant_text,
                        } => BridgeEvent::VoiceDone {
                            transcript,
                            assistant_text,
                        },
                        VoiceEvent::Error { message } => BridgeEvent::VoiceError { message },
                        // Stage events are useful for debugging but
                        // don't drive the UI yet.
                        VoiceEvent::StageStart { .. } | VoiceEvent::StageEnd { .. } => continue,
                    };
                    if tx.send(bevt).is_err() {
                        break;
                    }
                }
            };
            let run = voice_pipeline_run(
                &client,
                &endpoint.base,
                &endpoint.token,
                &device_id,
                &pcm_16khz,
                None,
                device.as_ref(),
                inner_tx,
            );
            let (_, run_result) = tokio::join!(pump, run);
            if let Err(e) = run_result {
                let _ = tx.send(BridgeEvent::VoiceError {
                    message: format!("{e:#}"),
                });
            }
        });
    }
}
