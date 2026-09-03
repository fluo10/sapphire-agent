use crate::config_layer::{self, Layer};
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Config {
    /// Matrix channel configuration. Both `matrix` and `discord` may be
    /// configured at once — when set, both run concurrently in the
    /// same `serve` process. Both may also be omitted, in which case the
    /// agent serves only the HTTP API.
    #[serde(default)]
    pub matrix: Option<MatrixConfig>,
    /// Discord channel configuration. May coexist with `matrix`.
    #[serde(default)]
    pub discord: Option<DiscordConfig>,
    pub anthropic: AnthropicConfig,
    /// Context compression configuration.
    #[serde(default)]
    pub compression: CompressionConfig,
    /// Tool configuration (search APIs, etc.).
    #[serde(default)]
    pub tools: ToolsConfig,
    /// HTTP API server configuration.
    #[serde(default)]
    pub serve: Option<ServeConfig>,
    /// A2A (Agent2Agent Protocol) server configuration. Mounted on the
    /// same axum app as `serve`; `enabled = false` (or absent) leaves
    /// the `/a2a` and `/.well-known/agent-card.json` routes off.
    #[serde(default)]
    pub a2a: Option<A2aConfig>,
    /// ACP (Agent Client Protocol) endpoint configuration. Mounted on the
    /// same axum app as `serve`; `enabled = false` (or absent) leaves the
    /// `GET /acp` route off. Host-local — see [`AcpConfig`].
    #[serde(default)]
    pub acp: Option<AcpConfig>,
    /// Workspace-external image cache. Holds raw bytes for vision
    /// inputs by SHA-256 so in-memory `ChatMessage` history and JSONL
    /// session files only carry compact references. When unset, the
    /// cache uses `dirs::cache_dir() / "sapphire-agent" / "images"`.
    /// Set `enabled = false` to fall back to the PR1 text-marker shape
    /// (no re-display of past images, but no cache directory either).
    #[serde(default)]
    pub image_cache: ImageCacheConfig,
    /// Directory containing AGENT.md and MEMORY.md.
    /// Defaults to the config file's parent directory.
    pub workspace_dir: Option<String>,
    /// Directory for persisted JSONL sessions.
    /// Defaults to `<workspace_dir>/sessions`.
    pub sessions_dir: Option<String>,
    /// Hour (0–23, local time) at which a new "day" begins.
    /// Used for session resets and daily log generation. Default: 0 (midnight).
    #[serde(default)]
    pub day_boundary_hour: u8,
    /// Default session policy applied at the day boundary when no room
    /// profile sets its own policy. Default: `reset` (back-compat).
    #[serde(default)]
    pub session_policy: SessionPolicy,
    /// Additional LLM providers beyond the built-in `anthropic` one.
    /// Keyed by user-chosen name (e.g. `"local"`, `"openai"`).
    #[serde(default)]
    pub providers: HashMap<String, ProviderConfig>,
    /// Named profiles that bind a use-case (e.g. `"casual"`, `"opus"`,
    /// `"local"`) to a provider name and optional refusal-fallback
    /// provider. A profile is a *pure* LLM preset — it does **not**
    /// know about memory namespaces or rooms; pairing happens via
    /// `[room_profile.<n>]`.
    #[serde(default)]
    pub profiles: HashMap<String, ProfileConfig>,
    /// Room profiles: bundle a chat profile + memory namespace +
    /// session policy and apply to a list of rooms / API channel
    /// targets. Each room_id appears in at most one room profile.
    #[serde(default, rename = "room_profile")]
    pub room_profiles: HashMap<String, RoomProfileConfig>,
    /// Memory namespaces. Each namespace owns its own subtree under
    /// `memory/<namespace>/` (daily/weekly/monthly/yearly logs and
    /// MEMORY.md). Profiles pin their writes to one namespace, and
    /// rooms reading the system prompt also pull in the parent
    /// namespaces declared via `include`.
    ///
    /// The `"default"` namespace is implicitly present (with `include = []`)
    /// even when no `[memory_namespace.*]` block is configured, so that
    /// every config has a valid root.
    #[serde(default, rename = "memory_namespace")]
    pub memory_namespaces: HashMap<String, MemoryNamespaceConfig>,
    /// Whether to generate a daily log at the day boundary. Default: true.
    #[serde(default = "default_true")]
    pub daily_log_enabled: bool,
    /// Whether to compact MEMORY.md at the day boundary. Default: true.
    #[serde(default = "default_true")]
    pub memory_compaction_enabled: bool,
    /// Whether to enable heartbeat (day-boundary + cron) tasks. Default: true.
    /// Set to false in test environments to avoid duplicate heartbeat tasks
    /// when both test and production instances share the same config.
    #[serde(default = "default_true")]
    pub heartbeat_enabled: bool,
    /// How often the agent re-indexes the workspace, in minutes. Unset or
    /// `0` disables the periodic re-index entirely. Each tick runs
    /// `WorkspaceState::sync`, an mtime-based refresh of the retrieve
    /// cache — this is what picks up files edited outside the agent.
    ///
    /// There is no git leg any more: the framework removed local-workspace
    /// auto-sync, so nothing is committed or pushed on this cadence.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sync_interval_minutes: Option<u32>,
    /// Removed in the sapphire-framework `main` migration. Retained only so an
    /// existing config that still sets it fails loudly instead of silently
    /// coming up as a second active agent: cold standby was the primary's
    /// config with this flag flipped, so ignoring it would start duplicate
    /// channel listeners and race on MEMORY.md in a shared workspace.
    #[serde(default)]
    pub standby_mode: Option<bool>,
    /// How many minutes of inactivity (no incoming user message) before
    /// the agent emits a same-day digest line summarising the session
    /// so far. The digest is read back across sessions and injected
    /// into the system prompt of newly opened rooms in the same memory
    /// namespace — this is what makes a morning voice chat visible in
    /// an afternoon text chat without waiting for the day-boundary
    /// daily log.
    ///
    /// `None` (default 30) keeps the feature on; explicit `0` disables.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub intraday_idle_minutes: Option<u32>,
    /// Periodic log digest configuration (weekly / monthly / yearly).
    #[serde(default)]
    pub digest: DigestConfig,
    /// Workspace-external cache of resumable subagent child
    /// conversations. See `subagent_cache::SubagentCache`'s module doc
    /// for why this lives outside the workspace rather than in the
    /// session store.
    #[serde(default)]
    pub subagent_cache: SubagentCacheConfig,
    /// Voice pipeline presets, referenced by `[room_profile.<n>].voice_pipeline`.
    #[serde(default, rename = "voice_pipeline")]
    pub voice_pipelines: HashMap<String, VoicePipelineConfig>,
    /// Named STT providers, referenced by `[voice_pipeline.<n>].stt_provider`.
    #[serde(default, rename = "stt_provider")]
    pub stt_providers: HashMap<String, SttProviderConfig>,
    /// Named TTS providers, referenced by `[voice_pipeline.<n>].tts_provider`.
    #[serde(default, rename = "tts_provider")]
    pub tts_providers: HashMap<String, TtsProviderConfig>,
    /// Global voice settings — `wake_word_model` etc. Same for every
    /// satellite regardless of which room_profile they connect to.
    #[serde(default)]
    pub voice: VoiceConfig,
    /// Timer / Pomodoro presets. Single-slot in-memory timers fired from
    /// the `timer_*` tools — Pomodoro cycles drop into [[timer.preset]]
    /// blocks so the user can say "ポモドーロ開始" instead of redeclaring
    /// "25分集中 + 5分休憩を3回" every time.
    #[serde(default)]
    pub timer: TimerConfig,
    /// Ambient (always-on) audio capture ingest configuration.
    #[serde(default)]
    pub ambient: AmbientConfig,
    /// Location of the `sapphire-framework` key file used to authenticate
    /// ambient capture devices.
    #[serde(default)]
    pub keys: KeysConfig,
    /// Removed in the device-registry migration; kept only so a config that
    /// still has `[device.*]` blocks parses far enough for
    /// `Config::migration_errors` to name each leftover block by its key and
    /// fail loudly, rather than the loader rejecting the whole file with a
    /// generic parse error. Never populated by anything else.
    #[serde(default, rename = "device")]
    pub devices: HashMap<String, DeviceConfig>,
}

fn default_true() -> bool {
    true
}

/// Action taken at the day boundary for a given conversation.
///
/// - `Reset`: close the session and clear in-memory caches (legacy behavior).
///   The next message starts a fresh session.
/// - `Compact`: keep the same session alive, but force-summarize the current
///   in-memory history and replace it with a summary stub. The SummaryLine is
///   appended to the session JSONL. Session continuity is preserved.
/// - `None`: no day-boundary action.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum SessionPolicy {
    #[default]
    Reset,
    Compact,
    None,
}

/// Bundle of (chat profile, memory namespace, session policy)
/// applied to a set of rooms.
///
/// Each `room_id` may appear in at most one room profile. Rooms that
/// don't appear in any room profile fall back to `[room_profile.default]`
/// if defined, otherwise the built-in defaults (Anthropic provider,
/// `"default"` namespace, global `session_policy`).
#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct RoomProfileConfig {
    /// Name of the LLM profile (in `[profiles.<n>]`) that drives chat
    /// turns for rooms in this room profile. Required.
    pub profile: String,
    /// Memory namespace these rooms read and write under. Defaults to
    /// the implicit `"default"` namespace.
    #[serde(default)]
    pub memory_namespace: Option<String>,
    /// Override the day-boundary session policy for these rooms.
    /// Falls through to `Config.session_policy` when absent.
    #[serde(default)]
    pub session_policy: Option<SessionPolicy>,
    /// Channel-side room ids this profile applies to. Matrix room ids,
    /// Discord channel ids, etc. Empty `[]` means the room profile is
    /// usable from API sessions only — no channel rooms map to it.
    #[serde(default)]
    pub rooms: Vec<String>,
    /// Voice pipeline preset (in `[voice_pipeline.<n>]`) used when the
    /// MCP `voice/pipeline_run` method targets this room profile.
    /// Absent means voice is disabled for this room profile.
    #[serde(default)]
    pub voice_pipeline: Option<String>,
    /// Removed in the device-registry migration; replaced by `devices`.
    /// Retained so a config that still sets it fails loudly — see
    /// `Config::migration_errors`.
    #[serde(default)]
    pub api_keys: Vec<String>,
    /// Device ids (from the workspace `devices.toml`) that run under this room
    /// profile. A device id appears in exactly one room profile; the binding is
    /// what gives an authenticated device its LLM profile and memory namespace.
    /// Replaces `api_keys`, which held raw tokens in this file.
    #[serde(default)]
    pub devices: Vec<String>,
}

/// A2A (Agent2Agent Protocol) server settings.
///
/// The A2A endpoints (`/a2a` JSON-RPC and `/.well-known/agent-card.json`)
/// are mounted on the same axum app as the legacy `/rpc` API server
/// (driven by `[serve]`). Disabled by default — set `enabled = true` to
/// turn the routes on. `Authorization: Bearer <token>` is resolved through
/// `DeviceAuth` — token -> device -> room profile — to determine which
/// profile (and therefore which provider/memory_namespace) the request
/// runs under.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct A2aConfig {
    /// Whether to mount the A2A routes on the axum app. Default: false.
    #[serde(default)]
    pub enabled: bool,
    /// Public URL of this agent's A2A endpoint, published in the Agent
    /// Card under `supportedInterfaces[].url`. When absent, the card
    /// emits an empty URL and clients have to know the endpoint out of
    /// band. Set this to e.g. `"https://agent.example/a2a"` once you
    /// know the externally-visible address.
    #[serde(default)]
    pub public_url: Option<String>,
    /// Human-readable name of this agent in the Agent Card. Default:
    /// `"sapphire-agent"`.
    #[serde(default)]
    pub agent_name: Option<String>,
    /// One-line description of this agent in the Agent Card. Default:
    /// a generic personal-assistant description.
    #[serde(default)]
    pub agent_description: Option<String>,
}

/// `[acp]` — the Agent Client Protocol endpoint at `GET /acp`.
///
/// Host-local by construction: `src/config_layer.rs` is default-deny, so the
/// workspace layer cannot turn an endpoint on for every host that syncs it.
#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct AcpConfig {
    /// Serve `/acp`. Off by default; the endpoint 404s while disabled.
    #[serde(default)]
    pub enabled: bool,
}

/// Image cache settings. See [`Config::image_cache`] for an overview
/// of how this interacts with persistence and provider calls.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ImageCacheConfig {
    /// When false, no cache directory is opened and the image scrubbing
    /// path becomes a no-op. JSONL still gets the SHA-256 text marker
    /// from `SessionStore::append`; in-memory history keeps full base64
    /// (same shape as before PR2).
    #[serde(default = "default_image_cache_enabled")]
    pub enabled: bool,
    /// Override the default cache directory. `None` resolves to
    /// `dirs::cache_dir() / "sapphire-agent" / "images"` at startup.
    /// Path may be relative; resolved against the process cwd at open
    /// time (typical configs use absolute paths).
    #[serde(default)]
    pub dir: Option<PathBuf>,
}

impl Default for ImageCacheConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            dir: None,
        }
    }
}

fn default_image_cache_enabled() -> bool {
    true
}

/// Ambient (always-on) audio capture ingest. Opt-in: with `enabled =
/// false`, `ambient::startup::build` returns `None` and `/audio/ingest` is
/// never mounted, so it 404s like any other unmatched path rather than
/// answering 401 and telling a probing device its key is wrong. The
/// handler's own `enabled` check survives as defence in depth for a caller
/// that mounts the routes anyway.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct AmbientConfig {
    #[serde(default)]
    pub enabled: bool,
    /// Override the cache root. `None` resolves to
    /// `dirs::cache_dir() / "sapphire-agent" / "ambient"` at startup.
    #[serde(default)]
    pub cache_dir: Option<PathBuf>,
    /// Days of raw audio to keep. Transcripts are never swept.
    #[serde(default = "default_audio_retention_days")]
    pub audio_retention_days: u32,
    /// Name of the `[stt_provider.*]` block used for ambient transcription.
    #[serde(default)]
    pub stt_provider: String,
    /// Segments with less gated speech than this get no speaker attribution.
    /// Embeddings from very short utterances are unreliable and are the main
    /// driver of speaker-id inflation.
    #[serde(default = "default_min_embed_ms")]
    pub min_embed_ms: u32,
    /// `SpeakerEmbeddingManager::search` threshold.
    #[serde(default = "default_match_threshold")]
    pub match_threshold: f32,
    #[serde(default = "default_promote_after_seconds")]
    pub promote_after_seconds: u32,
    #[serde(default = "default_promote_after_days")]
    pub promote_after_days: u32,
    /// Admission queue depth. A full queue answers 429, which a device
    /// handles exactly like being offline.
    #[serde(default = "default_max_queue")]
    pub max_queue: usize,
    /// Directory holding the Silero VAD model. sherpa-onnx publishes it only
    /// as a bare `.onnx` file (e.g. `silero_vad.onnx`), not a downloadable
    /// bundle — get it from the sherpa-onnx GitHub releases page
    /// (`asr-models` tag) and point this at the directory you saved it in.
    #[serde(default)]
    pub vad_model_dir: Option<String>,
    /// Directory holding the speaker embedding model. Like the VAD model,
    /// sherpa-onnx publishes these only as bare `.onnx` files — get one from
    /// the sherpa-onnx GitHub releases page (`speaker-recongition-models`
    /// tag, upstream's own spelling) and point this at the directory you
    /// saved it in.
    #[serde(default)]
    pub embedding_model_dir: Option<String>,
    /// VAD speech probability threshold.
    #[serde(default = "default_vad_threshold")]
    pub vad_threshold: f32,
    /// Inference threads for the embedding model.
    #[serde(default = "default_embedding_num_threads")]
    pub embedding_num_threads: i32,
}

fn default_audio_retention_days() -> u32 {
    7
}
fn default_min_embed_ms() -> u32 {
    1500
}
fn default_match_threshold() -> f32 {
    0.55
}
fn default_promote_after_seconds() -> u32 {
    60
}
fn default_promote_after_days() -> u32 {
    2
}
fn default_max_queue() -> usize {
    1000
}
fn default_vad_threshold() -> f32 {
    0.5
}
fn default_embedding_num_threads() -> i32 {
    2
}

impl Default for AmbientConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            cache_dir: None,
            audio_retention_days: default_audio_retention_days(),
            stt_provider: String::new(),
            min_embed_ms: default_min_embed_ms(),
            match_threshold: default_match_threshold(),
            promote_after_seconds: default_promote_after_seconds(),
            promote_after_days: default_promote_after_days(),
            max_queue: default_max_queue(),
            vad_model_dir: None,
            embedding_model_dir: None,
            vad_threshold: default_vad_threshold(),
            embedding_num_threads: default_embedding_num_threads(),
        }
    }
}

/// Location of the `sapphire-framework` key file. Host-local: it names
/// the only place tokens are stored, so it must never be settable from
/// the workspace config layer.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct KeysConfig {
    #[serde(default)]
    pub file: Option<PathBuf>,
}

/// Removed in the device-registry migration. Retained only so an existing
/// config that still has `[device.*]` blocks fails loudly with instructions
/// instead of silently coming up with ambient ingest rejecting every segment.
/// Every field is optional on purpose: this type only has to parse, never to
/// carry a value. See `Config::migration_errors`.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct DeviceConfig {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub key_id: Option<uuid::Uuid>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub room_profile: Option<String>,
}

/// Voice-mode global settings — everything that's the same for every
/// satellite regardless of which room_profile they connect to.
/// Currently just the wake-word ONNX path; future global voice
/// knobs (default language, sample rate overrides, etc.) land here.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct VoiceConfig {
    /// Path to an openWakeWord-trained `.onnx` classifier. Loaded
    /// once at startup, distributed to satellites inline in the
    /// `voice/config` response. AI-name wake words can't realistically
    /// be served by pre-trained KWS bundles (their vocabulary is
    /// finite), so custom openWakeWord ONNXes are the only path.
    #[serde(default)]
    pub wake_word_model: Option<String>,
}

/// Definition of an additional LLM provider.
///
/// Tagged by `type` to allow future provider kinds. Currently only
/// `openai_compatible` is supported.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "type")]
pub enum ProviderConfig {
    /// llama.cpp `llama-server`, OpenAI proper, Ollama, vLLM, etc.
    #[serde(rename = "openai_compatible")]
    OpenAiCompatible(crate::provider::openai_compatible::OpenAICompatibleConfig),
}

/// Pure LLM preset — provider plus optional refusal-fallback provider.
///
/// Profiles intentionally know **nothing** about memory namespaces or
/// rooms. They are referenced by:
///   - `[room_profile.<n>].profile` for chat turns
///   - `[memory_namespace.<n>].background_profile` for daily-log /
///     digest / compaction work
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ProfileConfig {
    /// Name of the provider to use. Either `"anthropic"` (built-in) or a
    /// key from the top-level `[providers]` table.
    pub provider: String,
    /// Optional fallback provider used when the primary refuses a request
    /// (e.g. NSFW content). Wired up by the routing layer.
    #[serde(default)]
    pub fallback_provider: Option<String>,
}

/// Definition of a memory namespace — a subtree under `memory/<name>/`
/// that owns its own MEMORY.md and periodic logs. The `include` list
/// names parent namespaces whose memory should also be visible to
/// rooms using this namespace; reads chain through the include DAG,
/// writes go only to the leaf namespace.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct MemoryNamespaceConfig {
    /// Names of parent namespaces whose memory should be merged in
    /// when assembling the system prompt for this namespace. Forms a
    /// DAG; cycles are rejected at startup.
    #[serde(default)]
    pub include: Vec<String>,
    /// Profile used by background tasks (daily-log generation, periodic
    /// digests, MEMORY.md compaction) when working under this
    /// namespace. Lets a per-namespace policy pick a permissive local
    /// model up front instead of relying on a refusal-fallback hop —
    /// e.g. an NSFW namespace can route directly to its local provider
    /// while the default namespace stays on Anthropic.
    ///
    /// Resolution order for a given namespace:
    ///   1. `memory_namespace.<n>.background_profile` (this field)
    ///   2. global `[profiles.background]` (back-compat with PR #68)
    ///   3. plain Anthropic
    #[serde(default)]
    pub background_profile: Option<String>,
    /// Whether conversations under this namespace are offered the
    /// skill tools. Off by default: `using-superpowers` asks the model
    /// to check for a relevant skill before answering at all, which is
    /// right for development and wrong for an ordinary conversation.
    #[serde(default)]
    pub skills: bool,
}

/// Voice pipeline preset — references a named STT provider and TTS provider
/// plus per-pipeline defaults (language, capture limits). Bound to a
/// `[room_profile.<n>]` via that profile's `voice_pipeline` field.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct VoicePipelineConfig {
    /// Name of the entry in `[stt_provider.<n>]`.
    pub stt_provider: String,
    /// Name of the entry in `[tts_provider.<n>]`.
    pub tts_provider: String,
    /// BCP-47 language hint passed to STT when the caller omits one.
    /// `None` lets the provider auto-detect (whisper) or use its own default.
    #[serde(default)]
    pub language: Option<String>,
    /// Hard cap on a single utterance, in milliseconds. Helps reject
    /// runaway clients that forget to stop. Default: 30 seconds.
    #[serde(default = "default_capture_max_ms")]
    pub capture_max_ms: u32,
}

fn default_capture_max_ms() -> u32 {
    30_000
}

/// STT provider definition. Tagged by `type` so future providers (e.g.
/// Deepgram, AssemblyAI) can be added without breaking config.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "type")]
pub enum SttProviderConfig {
    /// Local STT via the official sherpa-onnx Rust crate.
    ///
    /// Requires building with `--features voice-sherpa`. Model family
    /// (SenseVoice, Whisper, Paraformer, …) is determined by `kind`;
    /// the bundle is auto-downloaded from sherpa-onnx GitHub releases
    /// when `model` is a known bundle name and `model_dir` is absent.
    #[serde(rename = "sherpa_onnx")]
    SherpaOnnx(SherpaSttConfig),
    /// OpenAI Whisper API (audio/transcriptions).
    #[serde(rename = "openai_whisper_api")]
    OpenAiWhisperApi {
        /// Environment variable holding the API key.
        api_key_env: String,
        /// Optional base URL override (for OpenAI-compatible endpoints
        /// like Groq, OpenRouter). Defaults to OpenAI's public endpoint.
        #[serde(default)]
        base_url: Option<String>,
        /// Model name. Defaults to `whisper-1` when omitted.
        #[serde(default)]
        model: Option<String>,
    },
    /// Deterministic mock — always returns the same configured text.
    /// Useful for testing the pipeline plumbing without any model setup.
    #[serde(rename = "mock")]
    Mock {
        /// Text to return for every transcription. Default: `"test transcript"`.
        #[serde(default = "default_mock_transcript")]
        transcript: String,
    },
}

fn default_mock_transcript() -> String {
    "test transcript".to_string()
}

/// Configuration for the sherpa-onnx STT provider.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct SherpaSttConfig {
    /// Model family. Each family has a different on-disk layout that
    /// sherpa-onnx expects; this tells the wrapper which fields to set.
    pub kind: SherpaSttKind,
    /// Either a known bundle name (auto-downloaded to the cache dir)
    /// or an explicit path to an extracted model directory. When both
    /// `model` and `model_dir` are set, `model_dir` wins.
    #[serde(default)]
    pub model: Option<String>,
    /// Explicit path to an extracted model directory. Takes precedence
    /// over `model` when both are present.
    #[serde(default)]
    pub model_dir: Option<String>,
    /// BCP-47 language hint passed to model families that accept one
    /// (SenseVoice, Whisper). Ignored by others.
    #[serde(default)]
    pub language: Option<String>,
    /// Number of CPU threads used for inference. Default: 2.
    #[serde(default = "default_sherpa_num_threads")]
    pub num_threads: i32,
    /// ONNX runtime provider (`cpu`, `cuda`, `coreml`). Default: `cpu`.
    #[serde(default = "default_sherpa_provider")]
    pub provider: String,
}

fn default_sherpa_num_threads() -> i32 {
    2
}

fn default_sherpa_provider() -> String {
    "cpu".to_string()
}

/// Model families supported by the sherpa-onnx STT provider.
#[derive(Debug, Clone, Copy, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SherpaSttKind {
    /// SenseVoice — multilingual (zh/en/ja/ko/yue), recommended default.
    SenseVoice,
    /// OpenAI Whisper running on the sherpa-onnx runtime.
    Whisper,
}

/// TTS provider definition. Tagged by `type` so we can add `piper_shell`,
/// `elevenlabs`, etc. without breaking config.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "type")]
pub enum TtsProviderConfig {
    /// OpenAI Audio Speech (`POST /v1/audio/speech`). Works against
    /// OpenAI's public endpoint and any self-hosted server that
    /// speaks the same request shape (e.g. forks of Irodori-TTS
    /// exposing an OpenAI-compatible TTS API).
    #[serde(rename = "openai_tts")]
    OpenAiTts {
        /// Environment variable holding the API key. Optional —
        /// when omitted, no `Authorization` header is sent, which
        /// is what self-hosted endpoints without auth want. For
        /// OpenAI's real endpoint this must be set.
        #[serde(default)]
        api_key_env: Option<String>,
        /// Optional base URL override. Defaults to OpenAI's public
        /// endpoint (`https://api.openai.com`).
        #[serde(default)]
        base_url: Option<String>,
        /// Model name. Defaults to `tts-1` when omitted.
        #[serde(default)]
        model: Option<String>,
        /// Voice name. Defaults to `alloy`. For OpenAI: one of
        /// `alloy`, `echo`, `fable`, `onyx`, `nova`, `shimmer`.
        /// Self-hosted endpoints accept whatever voice id they
        /// expose (the field is passed through verbatim).
        #[serde(default)]
        voice: Option<String>,
    },
    /// Synthetic mock — returns a fixed-length sine wave. Useful for
    /// testing the pipeline plumbing without any model setup.
    #[serde(rename = "mock")]
    Mock {
        /// Duration of the generated tone in milliseconds. Default: 200ms.
        #[serde(default = "default_mock_duration_ms")]
        duration_ms: u32,
        /// Tone frequency in Hz. Default: 440Hz.
        #[serde(default = "default_mock_freq_hz")]
        frequency_hz: u32,
    },
    /// Local TTS via the official sherpa-onnx Rust crate. Requires
    /// building with `--features voice-sherpa`. Bundle is auto-downloaded
    /// from sherpa-onnx GitHub releases when `model` is a known name
    /// and `model_dir` is absent.
    #[serde(rename = "sherpa_onnx")]
    SherpaOnnx(SherpaTtsConfig),
}

fn default_mock_duration_ms() -> u32 {
    200
}

fn default_mock_freq_hz() -> u32 {
    440
}

/// Configuration for the sherpa-onnx TTS provider.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct SherpaTtsConfig {
    /// Model family — determines how the on-disk files are wired up.
    pub kind: SherpaTtsKind,
    /// Bundle name (auto-downloaded) or path. Either `model` or
    /// `model_dir` must be set; `model_dir` wins when both are.
    #[serde(default)]
    pub model: Option<String>,
    #[serde(default)]
    pub model_dir: Option<String>,
    /// Speaker id for multi-speaker models. Default: 0.
    #[serde(default)]
    pub speaker_id: i32,
    /// Synthesis speed (1.0 = normal, <1.0 = slower, >1.0 = faster).
    #[serde(default = "default_tts_speed")]
    pub speed: f32,
    #[serde(default = "default_sherpa_num_threads")]
    pub num_threads: i32,
    #[serde(default = "default_sherpa_provider")]
    pub provider: String,
}

fn default_tts_speed() -> f32 {
    1.0
}

/// Model families supported by the sherpa-onnx TTS provider.
#[derive(Debug, Clone, Copy, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SherpaTtsKind {
    /// VITS — broad language coverage, single ONNX model file.
    Vits,
    /// Matcha — Flow Matching, needs a separate vocoder.
    Matcha,
    /// Kokoro — multilingual flow-matching, voice embeddings file.
    Kokoro,
}

/// Built-in name of the Anthropic provider — referenced by profiles.
pub const ANTHROPIC_PROVIDER_NAME: &str = "anthropic";

/// Conventional name of the default profile.
pub const DEFAULT_PROFILE_NAME: &str = "default";

/// Conventional name of the profile used by background tasks (daily-log,
/// memory compaction, periodic digests). When this profile is defined the
/// background tasks honour its `provider` and `fallback_provider`; when
/// it isn't, those tasks run on the built-in Anthropic provider with no
/// fallback.
pub const BACKGROUND_PROFILE_NAME: &str = "background";

/// Implicit name of the root memory namespace. Always present, even when
/// no `[memory_namespace.*]` block is configured — backstop so every
/// profile / room resolves to a valid namespace.
pub const DEFAULT_NAMESPACE_NAME: &str = "default";

/// Configuration for the HTTP API server (serve command).
#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct ServeConfig {
    #[serde(default = "default_serve_host")]
    pub host: String,
    #[serde(default = "default_serve_port")]
    pub port: u16,
}

fn default_serve_host() -> String {
    "127.0.0.1".to_string()
}

fn default_serve_port() -> u16 {
    9000
}

/// Configuration for built-in tools.
#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct ToolsConfig {
    /// Tavily API key for `web_search`. If absent the tool is not registered.
    pub tavily_api_key: Option<String>,
    /// External MCP servers to connect to. Each server's tools are registered
    /// with the naming convention `mcp__<name>__<tool_name>`.
    #[serde(default)]
    pub mcp_servers: Vec<McpServerConfig>,
    /// Whether the agent may touch its own filesystem and shell.
    #[serde(default)]
    pub host_access: HostAccess,
}

/// Whether the agent may touch the machine it runs on.
///
/// Off by default. On a self-hosted deployment this is the server, and
/// "read any file, run any command" is not something a Discord message
/// should reach by default. Turning it on is a deliberate act; running
/// the agent in a container is the recommended way to do it.
#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct HostAccess {
    #[serde(default)]
    pub enabled: bool,
}

/// Configuration for a single external MCP server.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct McpServerConfig {
    /// Human-readable name (used in tool prefix: `mcp__<name>__<tool>`).
    pub name: String,
    /// How far this server's tools are trusted. See `McpTrust`.
    #[serde(default)]
    pub trust: McpTrust,
    /// Transport configuration.
    #[serde(flatten)]
    pub transport: McpTransportConfig,
}

/// How far the operator trusts an outbound MCP server's tools.
///
/// Every tool a server lists is classified by this one value, which
/// becomes the tool's `ToolKind` and so decides what the permission
/// policy does with it. Coarse on purpose, and declared here rather than
/// read from the server, for two reasons:
///
/// - **The operator, not the server.** MCP's own annotations
///   (`readOnlyHint`, `destructiveHint`) are finer, but they are
///   *self-reported*. The channel restriction exists precisely because a
///   channel turn carries untrusted input and cannot be approved
///   interactively; letting the far side declare its own tools safe
///   would invert the thing being defended. This field is written by the
///   person who already decided to connect to that server.
/// - **Per server, not per tool.** A list of tool names in the config
///   goes stale the moment the server adds a tool, and it goes stale
///   *silently*: the new tool falls back to `Other`, is refused, and
///   looks like the server being broken. Per-server is coarser but
///   cannot rot.
///
/// Annotations could later refine classification *within* a trusted
/// server — reads as `Read`, writes as `Edit`, inside a server marked
/// `edit`. That only makes sense once this gate exists.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum McpTrust {
    /// Tools are `Other` — the strictest bucket. Channels refuse them
    /// and ACP asks. The default, so adding this field changes nothing
    /// for an existing config.
    #[default]
    None,
    /// Tools are `Read`: safe on every origin, never asked.
    Read,
    /// Tools are `Edit`: allowed unasked on channels and trusted
    /// transports, asked in ACP's `default` mode — the same treatment
    /// `file_write` gets.
    Edit,
}

/// Transport configuration for connecting to an MCP server.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "type")]
pub enum McpTransportConfig {
    /// Streamable HTTP transport.
    #[serde(rename = "http")]
    Http {
        /// Server URL (e.g. `http://localhost:3000/mcp`).
        url: String,
        /// Optional API key / bearer token.
        #[serde(default)]
        api_key: Option<String>,
    },
    /// stdio transport — spawn a child process and communicate via stdin/stdout.
    #[serde(rename = "stdio")]
    Stdio {
        /// Command to execute (e.g. `"npx"`, `"uvx"`, `"/path/to/server"`).
        command: String,
        /// Command arguments.
        #[serde(default)]
        args: Vec<String>,
        /// Additional environment variables passed to the child process.
        #[serde(default)]
        env: std::collections::HashMap<String, String>,
    },
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct MatrixConfig {
    pub homeserver: String,
    pub access_token: String,
    pub user_id: String,
    pub device_id: String,
    /// Rooms the bot listens to. Accepts either a TOML array
    /// (`room_ids = ["!a:srv", "!b:srv"]`) or — for backward compatibility —
    /// a single string key named `room_id`.
    #[serde(default, alias = "room_id", deserialize_with = "deserialize_room_ids")]
    pub room_ids: Vec<String>,
    #[serde(default)]
    pub allowed_users: Vec<String>,
    /// E2EE recovery key (optional)
    pub recovery_key: Option<String>,
    /// Directory for matrix-sdk state/crypto store. Defaults to
    /// `~/.local/share/sapphire-agent/matrix`.
    pub state_dir: Option<String>,
}

/// Accept either `"!a:srv"` (legacy single string) or `["!a:srv", "!b:srv"]`
/// for the `room_ids` / legacy `room_id` field.
fn deserialize_room_ids<'de, D>(deserializer: D) -> std::result::Result<Vec<String>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum OneOrMany {
        One(String),
        Many(Vec<String>),
    }
    match OneOrMany::deserialize(deserializer)? {
        OneOrMany::One(s) => Ok(vec![s]),
        OneOrMany::Many(v) => Ok(v),
    }
}

impl MatrixConfig {
    /// Primary room — first configured room. Used as the default target for
    /// heartbeat tasks that don't name a specific room.
    pub fn primary_room_id(&self) -> Option<&str> {
        self.room_ids.first().map(|s| s.as_str())
    }

    pub fn resolved_state_dir(&self) -> PathBuf {
        if let Some(dir) = &self.state_dir {
            PathBuf::from(shellexpand::tilde(dir).as_ref())
        } else if let Some(dirs) = directories::ProjectDirs::from("", "", "sapphire-agent") {
            dirs.data_local_dir().join("matrix")
        } else {
            PathBuf::from(".sapphire-agent/matrix")
        }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct DiscordConfig {
    pub bot_token: String,
    /// Text channel IDs the bot listens to. Empty = all channels the bot can see.
    #[serde(default)]
    pub channel_ids: Vec<String>,
    /// Discord user IDs allowed to interact. Empty = all users.
    #[serde(default)]
    pub allowed_users: Vec<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct AnthropicConfig {
    /// Anthropic API key. Optional — when omitted (or commented out)
    /// the value is read from the `ANTHROPIC_API_KEY` environment
    /// variable at provider-construction time. Keeping the field
    /// optional lets test configs sit in the repo with no secret
    /// material on disk.
    #[serde(default)]
    pub api_key: Option<String>,
    #[serde(default = "default_model")]
    pub model: String,
    /// Cheaper model for casual (non-coding) conversations.
    /// If set, the agent uses this model by default and switches to `model`
    /// when the message appears to be coding-related.
    pub light_model: Option<String>,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: u32,
    pub system_prompt: Option<String>,
}

/// Env var consulted when `[anthropic].api_key` is absent.
pub const ANTHROPIC_API_KEY_ENV: &str = "ANTHROPIC_API_KEY";

impl AnthropicConfig {
    /// Return the effective API key, falling back to
    /// [`ANTHROPIC_API_KEY_ENV`] when the config field is absent or
    /// blank. Errors with a clear message when neither is set so the
    /// failure surfaces at startup rather than as an opaque 401 from
    /// the API.
    pub fn resolve_api_key(&self) -> Result<String> {
        let from_config = self
            .api_key
            .as_deref()
            .map(str::trim)
            .filter(|s| !s.is_empty());
        if let Some(key) = from_config {
            return Ok(key.to_string());
        }
        match std::env::var(ANTHROPIC_API_KEY_ENV) {
            Ok(v) if !v.trim().is_empty() => Ok(v),
            _ => Err(anyhow::anyhow!(
                "no Anthropic API key found: set [anthropic].api_key in config or \
                 the {ANTHROPIC_API_KEY_ENV} environment variable"
            )),
        }
    }
}

/// Context compression configuration (provider-agnostic).
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct CompressionConfig {
    /// Whether context compression is enabled. Default: true.
    #[serde(default = "default_true")]
    pub enabled: bool,
    /// Context window size in tokens. Defaults to 200,000.
    #[serde(default = "default_context_window")]
    pub context_window: usize,
    /// Fraction of context window at which compression triggers (0.0–1.0).
    /// Defaults to 0.80.
    #[serde(default = "default_compression_threshold")]
    pub threshold: f64,
    /// Number of recent messages to preserve verbatim during compression.
    /// Defaults to 20.
    #[serde(default = "default_preserve_recent")]
    pub preserve_recent: usize,
}

impl Default for CompressionConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            context_window: default_context_window(),
            threshold: default_compression_threshold(),
            preserve_recent: default_preserve_recent(),
        }
    }
}

fn default_model() -> String {
    "claude-opus-4-6".to_string()
}

fn default_max_tokens() -> u32 {
    8192
}

fn default_context_window() -> usize {
    200_000
}

fn default_compression_threshold() -> f64 {
    0.80
}

fn default_preserve_recent() -> usize {
    20
}

// ---------------------------------------------------------------------------
// Digest config
// ---------------------------------------------------------------------------

/// Timer presets (Pomodoro etc.). Each preset names a sequence of
/// `[label, minutes]` steps and a repeat count. Single-shot timers
/// don't need a preset — they take `minutes` + `message` directly via
/// the `timer_set` tool.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct TimerConfig {
    /// Named presets referenced by the `timer_preset` tool.
    #[serde(default, rename = "preset")]
    pub presets: Vec<TimerPreset>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct TimerPreset {
    /// Unique preset name (matched case-insensitively against the
    /// `name` argument of `timer_preset`).
    pub name: String,
    /// How many times to repeat the `steps` sequence. Default 1.
    #[serde(default = "default_timer_cycles")]
    pub cycles: u32,
    /// Ordered steps fired once per cycle.
    pub steps: Vec<TimerStep>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct TimerStep {
    /// Short label surfaced in the fire prompt (e.g. "Focus", "Break").
    pub label: String,
    /// Step duration in minutes. Fractional minutes are allowed.
    pub minutes: f64,
}

fn default_timer_cycles() -> u32 {
    1
}

/// Frontmatter-digest injection & generation config.
///
/// At each day boundary the agent generates weekly, monthly, and yearly log
/// files under `memory/{weekly,monthly,yearly}/`. Each file carries a YAML
/// `digest:` array of importance-ordered bullets. The top-N items per file
/// are injected into the system prompt so the agent retains long-horizon
/// context without paying full-body token cost.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct DigestConfig {
    /// Top-N digest items injected per daily log (used for "This Week's
    /// Digests" — days before yesterday within the current ISO week).
    #[serde(default = "default_digest_daily_items")]
    pub daily_items: usize,
    /// Top-N items injected per weekly log (used for "This Month's Digests").
    #[serde(default = "default_digest_weekly_items")]
    pub weekly_items: usize,
    /// Top-N items injected per monthly log (used for "This Year's Digests").
    #[serde(default = "default_digest_monthly_items")]
    pub monthly_items: usize,
    /// Top-N items injected per yearly log (used for "Past Years' Digests").
    #[serde(default = "default_digest_yearly_items")]
    pub yearly_items: usize,
    /// Generate a weekly log at each Monday day-boundary. Default: true.
    #[serde(default = "default_true")]
    pub weekly_enabled: bool,
    /// Generate a monthly log on the 1st of each month. Default: true.
    #[serde(default = "default_true")]
    pub monthly_enabled: bool,
    /// Generate a yearly log on Jan 1. Default: true.
    #[serde(default = "default_true")]
    pub yearly_enabled: bool,
}

impl Default for DigestConfig {
    fn default() -> Self {
        Self {
            daily_items: default_digest_daily_items(),
            weekly_items: default_digest_weekly_items(),
            monthly_items: default_digest_monthly_items(),
            yearly_items: default_digest_yearly_items(),
            weekly_enabled: true,
            monthly_enabled: true,
            yearly_enabled: true,
        }
    }
}

fn default_digest_daily_items() -> usize {
    3
}

fn default_digest_weekly_items() -> usize {
    3
}

fn default_digest_monthly_items() -> usize {
    5
}

fn default_digest_yearly_items() -> usize {
    5
}

/// Configuration for `subagent_cache::SubagentCache`, the
/// workspace-external store that lets a resumed subagent pick its
/// child conversation back up.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct SubagentCacheConfig {
    /// Cap on a single child's serialized history, in bytes. A `put`
    /// over this cap is refused wholesale rather than truncated — see
    /// `SubagentCache::put`'s doc.
    #[serde(default = "default_subagent_cache_max_history_bytes")]
    pub max_history_bytes: usize,
    /// Days a child conversation may go untouched before it is pruned.
    #[serde(default = "default_subagent_cache_retain_days")]
    pub retain_days: u32,
}

impl Default for SubagentCacheConfig {
    fn default() -> Self {
        Self {
            max_history_bytes: default_subagent_cache_max_history_bytes(),
            retain_days: default_subagent_cache_retain_days(),
        }
    }
}

fn default_subagent_cache_max_history_bytes() -> usize {
    8_388_608
}

fn default_subagent_cache_retain_days() -> u32 {
    7
}

impl Config {
    /// Minimal valid config for tests outside this module. `anthropic` is
    /// the only field without a `#[serde(default)]`, so this is the
    /// smallest TOML that parses; every other field falls back to its own
    /// default. Kept in one place so a change to a currently-required
    /// field doesn't have to be replicated across every test module that
    /// needs a `Config`.
    #[cfg(test)]
    pub(crate) fn for_test() -> Self {
        toml::from_str(
            r#"
[anthropic]
api_key = "test"
"#,
        )
        .expect("minimal config should parse")
    }

    /// Resolve the workspace directory: explicit config > config file's parent directory.
    pub fn resolved_workspace_dir(&self, config_path: &Path) -> PathBuf {
        resolve_workspace_dir(self.workspace_dir.as_deref(), config_path)
    }

    /// Resolve the sessions directory for JSONL persistence.
    ///
    /// Explicit config value > `<workspace_dir>/sessions` (default).
    pub fn resolved_sessions_dir(&self, workspace_dir: &Path) -> PathBuf {
        if let Some(dir) = &self.sessions_dir {
            PathBuf::from(shellexpand::tilde(dir).as_ref())
        } else {
            workspace_dir.join("sessions")
        }
    }

    /// Find the room profile a `room_id` belongs to.
    ///
    /// Order: explicit listing in `[room_profile.<n>].rooms` >
    /// conventional `[room_profile.default]` (catches all unmatched
    /// rooms) > `None`.
    pub fn room_profile_for(&self, room_id: &str) -> Option<(&str, &RoomProfileConfig)> {
        for (name, rp) in &self.room_profiles {
            if rp.rooms.iter().any(|r| r == room_id) {
                return Some((name.as_str(), rp));
            }
        }
        self.room_profiles
            .get_key_value(DEFAULT_PROFILE_NAME)
            .map(|(k, v)| (k.as_str(), v))
    }

    /// Look up a room profile by name. Used by API sessions, which pin
    /// a room_profile name at `initialize` time.
    pub fn room_profile(&self, name: &str) -> Option<&RoomProfileConfig> {
        self.room_profiles.get(name)
    }

    /// Resolve the session policy for a given `room_id`, falling back to
    /// the global default when no room profile sets one.
    pub fn session_policy_for(&self, room_id: &str) -> SessionPolicy {
        self.room_profile_for(room_id)
            .and_then(|(_, rp)| rp.session_policy)
            .unwrap_or(self.session_policy)
    }

    /// Effective idle threshold (in minutes) before a same-day digest
    /// flush fires. `None` means the feature is disabled — either by an
    /// explicit `0` or by future per-profile opt-out.
    pub fn intraday_idle_threshold_minutes(&self) -> Option<u32> {
        match self.intraday_idle_minutes {
            Some(0) => None,
            Some(n) => Some(n),
            None => Some(30),
        }
    }

    /// Resolve the LLM profile name for a given `room_id`.
    ///
    /// Order: room profile that contains this room > `[profiles.default]`
    /// if defined > `None` (caller falls back to the built-in Anthropic
    /// provider).
    pub fn profile_for(&self, room_id: &str) -> Option<&str> {
        if let Some((_, rp)) = self.room_profile_for(room_id) {
            return Some(rp.profile.as_str());
        }
        if self.profiles.contains_key(DEFAULT_PROFILE_NAME) {
            return Some(DEFAULT_PROFILE_NAME);
        }
        None
    }

    /// Resolve a profile name to its primary provider name.
    ///
    /// Returns `None` if the profile is not defined. Caller is expected to
    /// fall back to the built-in anthropic provider in that case.
    #[allow(dead_code)]
    pub fn provider_for_profile(&self, profile_name: &str) -> Option<&str> {
        self.profiles.get(profile_name).map(|p| p.provider.as_str())
    }

    /// Validate that every profile points to a known provider, and that
    /// every room's `profile` references a defined profile. Returns
    /// human-readable error messages for each issue found.
    pub fn validate_profiles(&self) -> Vec<String> {
        let mut errors = Vec::new();
        let known_provider = |name: &str| -> bool {
            name == ANTHROPIC_PROVIDER_NAME || self.providers.contains_key(name)
        };
        for (pname, prof) in &self.profiles {
            if !known_provider(&prof.provider) {
                errors.push(format!(
                    "profile '{pname}' references unknown provider '{}'",
                    prof.provider
                ));
            }
            if let Some(fb) = &prof.fallback_provider
                && !known_provider(fb)
            {
                errors.push(format!(
                    "profile '{pname}' references unknown fallback_provider '{fb}'"
                ));
            }
        }
        // Room profile references and uniqueness of room_ids across profiles.
        let mut seen_rooms: HashMap<String, String> = HashMap::new();
        for (rp_name, rp) in &self.room_profiles {
            if !self.profiles.contains_key(&rp.profile) {
                errors.push(format!(
                    "room_profile '{rp_name}' references unknown profile '{}'",
                    rp.profile
                ));
            }
            if let Some(ns) = &rp.memory_namespace
                && !self.namespace_is_defined(ns)
            {
                errors.push(format!(
                    "room_profile '{rp_name}' references unknown memory_namespace '{ns}'"
                ));
            }
            for room in &rp.rooms {
                if let Some(prev) = seen_rooms.get(room) {
                    errors.push(format!(
                        "room '{room}' appears in multiple room_profiles: '{prev}' and '{rp_name}'"
                    ));
                } else {
                    seen_rooms.insert(room.clone(), rp_name.clone());
                }
            }
        }
        // Memory namespace include references and cycle detection.
        for (ns_name, ns_cfg) in &self.memory_namespaces {
            for parent in &ns_cfg.include {
                if !self.namespace_is_defined(parent) {
                    errors.push(format!(
                        "memory_namespace '{ns_name}' includes unknown namespace '{parent}'"
                    ));
                }
            }
            if let Some(prof) = &ns_cfg.background_profile
                && !self.profiles.contains_key(prof)
            {
                errors.push(format!(
                    "memory_namespace '{ns_name}' references unknown background_profile '{prof}'"
                ));
            }
        }
        for ns_name in self.memory_namespaces.keys() {
            if let Some(cycle) = self.namespace_cycle_starting_at(ns_name) {
                errors.push(format!(
                    "memory_namespace cycle detected: {}",
                    cycle.join(" -> ")
                ));
            }
        }
        // Voice pipeline references.
        for (rp_name, rp) in &self.room_profiles {
            if let Some(vp) = &rp.voice_pipeline
                && !self.voice_pipelines.contains_key(vp)
            {
                errors.push(format!(
                    "room_profile '{rp_name}' references unknown voice_pipeline '{vp}'"
                ));
            }
        }
        // Global [voice].wake_word_model must point at a real file so
        // typos surface at server startup rather than as a 500 on the
        // first satellite voice/config call.
        if let Some(path) = &self.voice.wake_word_model {
            let expanded = shellexpand::tilde(path);
            if !std::path::Path::new(expanded.as_ref()).is_file() {
                errors.push(format!(
                    "voice.wake_word_model = '{path}' is not an existing file"
                ));
            }
        }
        for (vp_name, vp) in &self.voice_pipelines {
            if !self.stt_providers.contains_key(&vp.stt_provider) {
                errors.push(format!(
                    "voice_pipeline '{vp_name}' references unknown stt_provider '{}'",
                    vp.stt_provider
                ));
            }
            if !self.tts_providers.contains_key(&vp.tts_provider) {
                errors.push(format!(
                    "voice_pipeline '{vp_name}' references unknown tts_provider '{}'",
                    vp.tts_provider
                ));
            }
        }
        errors
    }

    /// Settings that were removed when device-based auth landed.
    ///
    /// Reported as a hard error at start-up rather than ignored, following the
    /// `standby_mode` precedent in `main.rs`. Ignoring them would turn a broken
    /// *config* into what looks like a broken *device*: dropping `api_keys`
    /// makes `/acp` 401 every client, and dropping `[device.*]` makes ambient
    /// refuse every segment. Neither symptom sends anyone to the config file.
    pub fn migration_errors(&self) -> Vec<String> {
        let mut errors = Vec::new();

        let mut names: Vec<&String> = self.devices.keys().collect();
        names.sort();
        for name in names {
            errors.push(format!(
                "[device.{name}] no longer lives in this file. Delete this block from \
                 config.toml first — startup refuses to run any command, including \
                 `sapphire-agent device add`, while it is still present. Devices moved to the \
                 workspace table at <workspace>/.sapphire-agent/devices.toml. Once the block is \
                 gone, run `sapphire-agent device add --name {name}`, put the printed token on \
                 the device, and add the printed id to a `[room_profile.<name>].devices` array. \
                 The old token cannot be carried over: it was hand-written plaintext with no \
                 entry in the key file."
            ));
        }

        let mut rp_names: Vec<&String> = self
            .room_profiles
            .iter()
            .filter(|(_, rp)| !rp.api_keys.is_empty())
            .map(|(name, _)| name)
            .collect();
        rp_names.sort();
        for name in rp_names {
            errors.push(format!(
                "[room_profile.{name}].api_keys was replaced by `devices`. Delete this \
                 `api_keys` line from config.toml first — startup refuses to run any command, \
                 including `sapphire-agent device add`, while it is still present. Raw tokens \
                 no longer live in this file; run `sapphire-agent device add --name <device>` \
                 for each client and list the printed ids in `[room_profile.{name}].devices`."
            ));
        }

        errors
    }

    /// True if `name` is either the implicit `"default"` namespace or has a
    /// `[memory_namespace.<name>]` block.
    fn namespace_is_defined(&self, name: &str) -> bool {
        name == DEFAULT_NAMESPACE_NAME || self.memory_namespaces.contains_key(name)
    }

    /// DFS from `start` looking for back-edges. Returns the cyclic path
    /// (start -> ... -> start) on detection, otherwise `None`.
    fn namespace_cycle_starting_at(&self, start: &str) -> Option<Vec<String>> {
        let mut stack: Vec<String> = vec![start.to_string()];
        let mut on_stack = std::collections::HashSet::new();
        on_stack.insert(start.to_string());

        fn dfs(
            cfg: &Config,
            node: &str,
            stack: &mut Vec<String>,
            on_stack: &mut std::collections::HashSet<String>,
        ) -> Option<Vec<String>> {
            let parents: Vec<String> = cfg
                .memory_namespaces
                .get(node)
                .map(|c| c.include.clone())
                .unwrap_or_default();
            for parent in parents {
                if on_stack.contains(&parent) {
                    let mut cycle: Vec<String> = stack.to_vec();
                    cycle.push(parent);
                    return Some(cycle);
                }
                stack.push(parent.clone());
                on_stack.insert(parent.clone());
                if let Some(c) = dfs(cfg, &parent, stack, on_stack) {
                    return Some(c);
                }
                stack.pop();
                on_stack.remove(&parent);
            }
            None
        }

        dfs(self, start, &mut stack, &mut on_stack)
    }

    /// Resolve `name` to its include-chain in DFS pre-order: the namespace
    /// itself first, then each parent in include order, with parents'
    /// parents flattened in. Duplicates are removed (first occurrence
    /// wins). The implicit `"default"` namespace, when not configured,
    /// resolves to a single-entry chain `["default"]`.
    pub fn resolve_namespace_chain(&self, name: &str) -> Vec<String> {
        let mut out: Vec<String> = Vec::new();
        let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
        self.namespace_chain_walk(name, &mut out, &mut seen);
        out
    }

    fn namespace_chain_walk(
        &self,
        name: &str,
        out: &mut Vec<String>,
        seen: &mut std::collections::HashSet<String>,
    ) {
        if !seen.insert(name.to_string()) {
            return;
        }
        out.push(name.to_string());
        if let Some(cfg) = self.memory_namespaces.get(name) {
            for parent in &cfg.include {
                self.namespace_chain_walk(parent, out, seen);
            }
        }
    }

    /// Profile name to use for background tasks (daily-log generation,
    /// digests, memory compaction) under `namespace`. Returns `None` when
    /// neither the namespace's own `background_profile` nor the global
    /// `[profiles.background]` is configured — caller should then fall
    /// back to the built-in Anthropic provider.
    pub fn background_profile_for_namespace(&self, namespace: &str) -> Option<&str> {
        if let Some(name) = self
            .memory_namespaces
            .get(namespace)
            .and_then(|c| c.background_profile.as_deref())
        {
            return Some(name);
        }
        if self.profiles.contains_key(BACKGROUND_PROFILE_NAME) {
            return Some(BACKGROUND_PROFILE_NAME);
        }
        None
    }

    /// Resolve the memory namespace declared by a room profile (by
    /// name). Falls back to `"default"` if the room profile is unknown
    /// or doesn't set one.
    pub fn namespace_for_room_profile(&self, name: &str) -> &str {
        self.room_profiles
            .get(name)
            .and_then(|rp| rp.memory_namespace.as_deref())
            .unwrap_or(DEFAULT_NAMESPACE_NAME)
    }

    /// Resolve the memory namespace for a given `room_id`. Rooms not
    /// present in any room profile fall through to `"default"`.
    pub fn namespace_for_room(&self, room_id: &str) -> &str {
        self.room_profile_for(room_id)
            .and_then(|(_, rp)| rp.memory_namespace.as_deref())
            .unwrap_or(DEFAULT_NAMESPACE_NAME)
    }

    /// Every memory namespace name relevant to this config: the implicit
    /// `"default"`, every `[memory_namespace.<name>]` key, and every
    /// namespace named by a room profile. Used by background catch-up
    /// loops to know what subtrees to enumerate.
    pub fn all_memory_namespaces(&self) -> Vec<String> {
        let mut out: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
        out.insert(DEFAULT_NAMESPACE_NAME.to_string());
        out.extend(self.memory_namespaces.keys().cloned());
        for rp in self.room_profiles.values() {
            if let Some(ns) = &rp.memory_namespace {
                out.insert(ns.clone());
            }
        }
        out.into_iter().collect()
    }

    /// Voice pipeline preset for the given room profile name, if any.
    /// Returns `None` when the room profile is unknown or has no
    /// `voice_pipeline` set.
    pub fn voice_pipeline_for_room_profile(&self, name: &str) -> Option<&VoicePipelineConfig> {
        self.room_profiles
            .get(name)
            .and_then(|rp| rp.voice_pipeline.as_ref())
            .and_then(|vp_name| self.voice_pipelines.get(vp_name))
    }

    /// Resolve the default config path: `~/.config/sapphire-agent/config.toml`
    pub fn default_path() -> PathBuf {
        if let Some(dirs) = directories::ProjectDirs::from("", "", "sapphire-agent") {
            dirs.config_dir().join("config.toml")
        } else {
            PathBuf::from("config.toml")
        }
    }
}

/// Resolve the workspace directory from an explicit setting, falling back to the
/// config file's own directory.
///
/// Free-standing because the layered loader needs it before a `Config` exists:
/// the workspace directory has to be known to find the workspace config, so it
/// can only ever come from the host layer.
pub fn resolve_workspace_dir(explicit: Option<&str>, config_path: &Path) -> PathBuf {
    match explicit {
        Some(dir) => PathBuf::from(shellexpand::tilde(dir).as_ref()),
        None => config_path
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .to_path_buf(),
    }
}

/// Path of the workspace-level config: `{workspace_dir}/.sapphire-agent/config.toml`.
///
/// This mirrors the framework's `Workspace::config_path()` convention. It is not
/// called through the framework because reaching a `Workspace` value goes via
/// `from_root`, which fails when the marker directory is absent — and the agent
/// resolves its workspace through the marker-free path.
pub fn workspace_config_path(workspace_dir: &Path) -> PathBuf {
    workspace_dir.join(".sapphire-agent").join("config.toml")
}

/// Path to the workspace device table.
///
/// Mirrors `workspace_config_path`. The framework has `Workspace::devices_path`
/// for the same convention, but the agent resolves its workspace as a plain
/// `PathBuf` and never builds a framework `Workspace` for config purposes —
/// that constructor canonicalizes and requires the marker directory to already
/// exist, neither of which is true when `device add` runs on a fresh checkout.
pub fn workspace_devices_path(workspace_dir: &Path) -> PathBuf {
    workspace_dir.join(".sapphire-agent").join("devices.toml")
}

/// Path to the workspace user table. See `workspace_devices_path`.
pub fn workspace_users_path(workspace_dir: &Path) -> PathBuf {
    workspace_dir.join(".sapphire-agent").join("users.toml")
}

/// A `Config` plus what the layering did to produce it.
#[derive(Debug)]
pub struct LoadedConfig {
    pub config: Config,
    /// Workspace-layer keys refused by the allowlist. Reported at startup.
    pub rejected: Vec<String>,
    /// Which layer supplied each setting, for `verify`.
    pub provenance: BTreeMap<String, Layer>,
    /// Path of the workspace-level config, when one was actually read (the file
    /// existed and was readable). `None` when no workspace layer was loaded, so
    /// callers can name the file in diagnostics without recomputing the path.
    pub workspace_path: Option<PathBuf>,
}

impl Config {
    /// Load the host config, then layer the workspace config under it.
    ///
    /// The workspace layer is opt-in by existence: with no
    /// `{workspace_dir}/.sapphire-agent/config.toml` this behaves exactly like
    /// a single-file load of the host config.
    ///
    /// A malformed **host** config is an error, as it always was. The workspace
    /// layer must never be able to stop the agent starting: from the point the
    /// workspace syncs from a server it is remote input, and one bad file must
    /// not take down every host. That covers three distinct ways the workspace
    /// file can be bad:
    ///
    /// - **TOML syntax** the parser rejects — demoted to a warning, workspace
    ///   treated as absent.
    /// - **The merged document fails to deserialize** into `Config` (a wrong
    ///   type, a required field missing from a table the workspace introduced)
    ///   — even though every key involved was allowlisted.
    /// - **The merged config fails semantic validation**
    ///   (`Config::validate_profiles`) — deserializes fine, references
    ///   something that does not exist.
    ///
    /// The latter two fall back to the host layer alone: the merged document is
    /// discarded, the host `toml::Value` is deserialized on its own, and
    /// `rejected`/`provenance` are reset to reflect a host-only load. A host
    /// layer that fails on its own (with or without a workspace layer present)
    /// is still a hard error — it is local, hand-written, and belongs to the
    /// person reading the message.
    pub fn load_layered(host_path: &Path) -> Result<LoadedConfig> {
        let host_text = std::fs::read_to_string(host_path)
            .with_context(|| format!("Failed to read config file: {}", host_path.display()))?;
        let host: toml::Value = toml::from_str(&host_text)
            .with_context(|| format!("Failed to parse config file: {}", host_path.display()))?;

        let workspace_dir = resolve_workspace_dir(
            host.get("workspace_dir").and_then(toml::Value::as_str),
            host_path,
        );
        let ws_path = workspace_config_path(&workspace_dir);

        let mut workspace_path = None;
        let mut workspace_present = false;
        let workspace = match std::fs::read_to_string(&ws_path) {
            Ok(text) => {
                workspace_path = Some(ws_path.clone());
                match toml::from_str::<toml::Value>(&text) {
                    Ok(value) => {
                        workspace_present = true;
                        value
                    }
                    Err(e) => {
                        tracing::warn!(
                            "Ignoring malformed workspace config at {}: {e}",
                            ws_path.display()
                        );
                        toml::Value::Table(toml::map::Map::new())
                    }
                }
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                toml::Value::Table(toml::map::Map::new())
            }
            Err(e) => {
                tracing::warn!(
                    "Ignoring unreadable workspace config at {}: {e}",
                    ws_path.display()
                );
                toml::Value::Table(toml::map::Map::new())
            }
        };

        let outcome = config_layer::merge_layers(workspace, host.clone());
        let deserialized: std::result::Result<Config, toml::de::Error> =
            outcome.merged.clone().try_into();

        // Drop the workspace layer and continue on the host config alone — but
        // only once the host config is known to be good on its own. Blaming the
        // workspace file before checking that would misattribute a host-side
        // bug to it: a host with its own broken profile reference fails the
        // merged check too, and the mere existence of an unrelated workspace
        // file would otherwise put its name in the message.
        let fall_back_to_host_only = |reason: String| -> Result<LoadedConfig> {
            let host_config: Config = host.clone().try_into().with_context(|| {
                format!(
                    "Failed to parse config file: {} (the workspace layer had already been \
                     dropped because it {reason})",
                    host_path.display()
                )
            })?;
            let host_errors = host_config.validate_profiles();
            if !host_errors.is_empty() {
                anyhow::bail!(
                    "invalid configuration in {}: {}",
                    host_path.display(),
                    host_errors.join("; ")
                );
            }
            tracing::warn!(
                "Workspace config at {} produces an invalid merged configuration ({reason}); \
                 falling back to the host config alone at {}.",
                ws_path.display(),
                host_path.display()
            );
            let empty = toml::Value::Table(toml::map::Map::new());
            let provenance = config_layer::provenance_of(&empty, &host);
            Ok(LoadedConfig {
                config: host_config,
                rejected: Vec::new(),
                provenance,
                workspace_path: workspace_path.clone(),
            })
        };

        match deserialized {
            Ok(config) => {
                let profile_errors = config.validate_profiles();
                if workspace_present && !profile_errors.is_empty() {
                    fall_back_to_host_only(format!(
                        "fails validation: {}",
                        profile_errors.join("; ")
                    ))
                } else {
                    Ok(LoadedConfig {
                        config,
                        rejected: outcome.rejected,
                        provenance: outcome.provenance,
                        workspace_path,
                    })
                }
            }
            Err(e) if workspace_present => {
                fall_back_to_host_only(format!("fails to deserialize: {e}"))
            }
            Err(e) => Err(e).with_context(|| "Failed to parse config file"),
        }
    }
}

/// Parse a TOML string into a [`Config`]. Test-only; shared by `mod
/// tests` below and by fixtures elsewhere in the crate (e.g.
/// `ServeState::build_for_test` in `src/serve/mod.rs`) that need a
/// minimal, hand-written config without going through a file on disk.
#[cfg(test)]
impl Config {
    pub(crate) fn parse_for_test(s: &str) -> Config {
        toml::from_str(s).expect("config should parse")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(s: &str) -> Config {
        Config::parse_for_test(s)
    }

    const MINIMAL: &str = r#"
[anthropic]
api_key = "test"
"#;

    /// The default is `none`: an existing config gains nothing, and the
    /// server's tools stay `Other` — refused on a channel. This is the
    /// property that makes the field safe to add.
    #[test]
    fn an_mcp_server_without_trust_is_untrusted() {
        let cfg = parse(
            r#"
[anthropic]
api_key = "test"

[[tools.mcp_servers]]
name = "ledger"
type = "http"
url  = "http://127.0.0.1:3838/mcp"
"#,
        );
        let server = &cfg.tools.mcp_servers[0];
        assert_eq!(server.trust, McpTrust::None);
    }

    /// The operator's declaration, spelled the way `config.example.toml`
    /// spells it. `trust` sits beside the flattened transport fields, so
    /// this also pins that the two do not collide during deserialisation.
    #[test]
    fn trust_is_read_from_the_server_entry() {
        let cfg = parse(
            r#"
[anthropic]
api_key = "test"

[[tools.mcp_servers]]
name  = "reader"
type  = "http"
url   = "http://127.0.0.1:3838/mcp"
trust = "read"

[[tools.mcp_servers]]
name    = "writer"
type    = "stdio"
command = "ledger-mcp"
trust   = "edit"
"#,
        );
        let servers = &cfg.tools.mcp_servers;
        assert_eq!(servers[0].trust, McpTrust::Read);
        assert_eq!(servers[1].trust, McpTrust::Edit);
        // The transport still deserialises alongside it.
        assert!(matches!(
            servers[1].transport,
            McpTransportConfig::Stdio { .. }
        ));
    }

    #[test]
    fn no_profiles_means_no_resolution() {
        let cfg = parse(MINIMAL);
        assert!(cfg.profile_for("!any:srv").is_none());
        assert!(cfg.validate_profiles().is_empty());
    }

    #[test]
    fn default_profile_is_used_when_room_unspecified() {
        let cfg = parse(
            r#"
[anthropic]
api_key = "test"

[profiles.default]
provider = "anthropic"
"#,
        );
        assert_eq!(cfg.profile_for("!some:srv"), Some("default"));
    }

    #[test]
    fn room_profile_assigns_profile_to_listed_rooms() {
        let cfg = parse(
            r#"
[anthropic]
api_key = "test"

[providers.local]
type = "openai_compatible"
base_url = "http://127.0.0.1:8080/v1"
model = "gemma-4-31b-it"

[profiles.default]
provider = "anthropic"

[profiles.nsfw]
provider = "local"

[room_profile.private_nsfw]
profile = "nsfw"
rooms   = ["!nsfw:srv"]
"#,
        );
        assert_eq!(cfg.profile_for("!nsfw:srv"), Some("nsfw"));
        // Unmatched room falls through to [profiles.default].
        assert_eq!(cfg.profile_for("!other:srv"), Some("default"));
        assert_eq!(cfg.provider_for_profile("nsfw"), Some("local"));
        assert_eq!(cfg.provider_for_profile("default"), Some("anthropic"));
        assert!(cfg.validate_profiles().is_empty());
    }

    #[test]
    fn default_room_profile_catches_unmatched_rooms() {
        let cfg = parse(
            r#"
[anthropic]
api_key = "test"

[profiles.casual]
provider = "anthropic"

[profiles.opus]
provider = "anthropic"

[room_profile.default]
profile = "casual"
rooms   = []

[room_profile.dev]
profile = "opus"
rooms   = ["!dev:srv"]
"#,
        );
        assert_eq!(cfg.profile_for("!dev:srv"), Some("opus"));
        // An unmatched room falls through to room_profile.default.
        assert_eq!(cfg.profile_for("!chat:srv"), Some("casual"));
    }

    #[test]
    fn validate_rejects_room_listed_in_two_profiles() {
        let cfg = parse(
            r#"
[anthropic]
api_key = "test"

[profiles.a]
provider = "anthropic"

[profiles.b]
provider = "anthropic"

[room_profile.first]
profile = "a"
rooms   = ["!shared:srv"]

[room_profile.second]
profile = "b"
rooms   = ["!shared:srv"]
"#,
        );
        let errors = cfg.validate_profiles();
        assert!(
            errors.iter().any(|e| e.contains("multiple room_profiles")),
            "expected duplicate-room error, got: {errors:?}"
        );
    }

    #[test]
    fn validate_flags_unknown_provider_in_profile() {
        let cfg = parse(
            r#"
[anthropic]
api_key = "test"

[profiles.default]
provider = "ghost"
"#,
        );
        let errors = cfg.validate_profiles();
        assert_eq!(errors.len(), 1, "got: {errors:?}");
        assert!(errors[0].contains("ghost"));
    }

    #[test]
    fn validate_flags_unknown_fallback_provider() {
        let cfg = parse(
            r#"
[anthropic]
api_key = "test"

[profiles.default]
provider = "anthropic"
fallback_provider = "ghost"
"#,
        );
        let errors = cfg.validate_profiles();
        assert_eq!(errors.len(), 1);
        assert!(errors[0].contains("fallback"));
        assert!(errors[0].contains("ghost"));
    }

    #[test]
    fn validate_flags_unknown_profile_in_room_profile() {
        let cfg = parse(
            r#"
[anthropic]
api_key = "test"

[room_profile.x]
profile = "missing"
rooms   = ["!x:srv"]
"#,
        );
        let errors = cfg.validate_profiles();
        assert!(
            errors.iter().any(|e| e.contains("missing")),
            "got: {errors:?}"
        );
    }

    #[test]
    fn migration_errors_name_a_leftover_device_block() {
        let cfg: Config = toml::from_str(
            r#"
[anthropic]
api_key = "test"

[device.pendant]
key_id = "550e8400-e29b-41d4-a716-446655440000"
"#,
        )
        .unwrap();

        let errors = cfg.migration_errors();

        assert_eq!(errors.len(), 1, "{errors:?}");
        assert!(errors[0].contains("pendant"), "{errors:?}");
        // The message has to say what to run, not just what is wrong: the
        // token cannot be carried over, so the operator must re-issue it.
        assert!(errors[0].contains("device add"), "{errors:?}");
    }

    #[test]
    fn migration_errors_name_a_leftover_api_keys_array() {
        let cfg: Config = toml::from_str(
            r#"
[anthropic]
api_key = "test"

[profiles.sonnet]
provider = "anthropic"

[room_profile.work]
profile = "sonnet"
api_keys = ["sa-acp-token"]
"#,
        )
        .unwrap();

        let errors = cfg.migration_errors();

        assert_eq!(errors.len(), 1, "{errors:?}");
        assert!(errors[0].contains("work"), "{errors:?}");
        assert!(errors[0].contains("devices"), "{errors:?}");
    }

    #[test]
    fn a_migrated_config_has_no_migration_errors() {
        let cfg: Config = toml::from_str(
            r#"
[anthropic]
api_key = "test"

[profiles.sonnet]
provider = "anthropic"

[room_profile.work]
profile = "sonnet"
devices = ["a3f9k2p"]
"#,
        )
        .unwrap();

        assert!(cfg.migration_errors().is_empty());
    }

    #[test]
    fn shipped_example_parses() {
        // Sanity check: the example file we ship in the repo must parse and
        // validate without errors so first-time users aren't greeted with a
        // confusing TOML error. Loaded through `load_layered` from a tempdir,
        // which also exercises the no-workspace-layer path — the case that has
        // to behave exactly as a single-file load always did.
        let src = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("config.example.toml");
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        std::fs::copy(&src, &path).expect("copy the shipped example");

        let cfg = Config::load_layered(&path)
            .expect("config.example.toml should parse")
            .config;
        assert!(
            cfg.validate_profiles().is_empty(),
            "validation errors: {:?}",
            cfg.validate_profiles()
        );
        // This is the one test guaranteeing the shipped example is clean;
        // it must also check the device-registry migration gate, not just
        // profile validation, or a `[device.*]` / `api_keys` leftover in the
        // example would go unnoticed until a user's own config hit it.
        assert!(
            cfg.migration_errors().is_empty(),
            "migration errors: {:?}",
            cfg.migration_errors()
        );
    }

    #[test]
    fn namespace_default_resolves_when_unconfigured() {
        let cfg = parse(MINIMAL);
        assert_eq!(
            cfg.resolve_namespace_chain(DEFAULT_NAMESPACE_NAME),
            vec!["default".to_string()]
        );
        assert_eq!(cfg.namespace_for_room("!any:srv"), "default");
        assert!(cfg.validate_profiles().is_empty());
    }

    #[test]
    fn namespace_chain_includes_parents_in_dfs_preorder() {
        let cfg = parse(
            r#"
[anthropic]
api_key = "test"

[memory_namespace.user]
include = ["default"]

[memory_namespace.user_nsfw]
include = ["user"]
"#,
        );
        assert_eq!(
            cfg.resolve_namespace_chain("user_nsfw"),
            vec![
                "user_nsfw".to_string(),
                "user".to_string(),
                "default".to_string()
            ]
        );
        assert_eq!(
            cfg.resolve_namespace_chain("user"),
            vec!["user".to_string(), "default".to_string()]
        );
    }

    #[test]
    fn namespace_chain_dedupes_diamond() {
        // a includes b and c; b and c both include d. d should appear once.
        let cfg = parse(
            r#"
[anthropic]
api_key = "test"

[memory_namespace.b]
include = ["d"]

[memory_namespace.c]
include = ["d"]

[memory_namespace.d]

[memory_namespace.a]
include = ["b", "c"]
"#,
        );
        let chain = cfg.resolve_namespace_chain("a");
        assert_eq!(chain.iter().filter(|n| *n == "d").count(), 1);
        assert_eq!(chain[0], "a");
    }

    #[test]
    fn namespace_cycle_is_rejected() {
        let cfg = parse(
            r#"
[anthropic]
api_key = "test"

[memory_namespace.a]
include = ["b"]

[memory_namespace.b]
include = ["a"]
"#,
        );
        let errors = cfg.validate_profiles();
        assert!(
            errors.iter().any(|e| e.contains("cycle")),
            "expected cycle error, got: {errors:?}"
        );
    }

    #[test]
    fn namespace_unknown_include_is_rejected() {
        let cfg = parse(
            r#"
[anthropic]
api_key = "test"

[memory_namespace.user]
include = ["ghost"]
"#,
        );
        let errors = cfg.validate_profiles();
        assert!(
            errors.iter().any(|e| e.contains("ghost")),
            "expected unknown-namespace error, got: {errors:?}"
        );
    }

    #[test]
    fn room_profile_assigns_memory_namespace() {
        let cfg = parse(
            r#"
[anthropic]
api_key = "test"

[memory_namespace.user_nsfw]
include = ["default"]

[profiles.nsfw]
provider = "anthropic"

[room_profile.private_nsfw]
profile          = "nsfw"
memory_namespace = "user_nsfw"
rooms            = ["!nsfw:srv"]
"#,
        );
        assert!(cfg.validate_profiles().is_empty());
        assert_eq!(cfg.namespace_for_room("!nsfw:srv"), "user_nsfw");
        assert_eq!(cfg.namespace_for_room("!other:srv"), "default");
        assert_eq!(cfg.namespace_for_room_profile("private_nsfw"), "user_nsfw");
    }

    #[test]
    fn room_profile_unknown_memory_namespace_is_rejected() {
        let cfg = parse(
            r#"
[anthropic]
api_key = "test"

[profiles.x]
provider = "anthropic"

[room_profile.bad]
profile          = "x"
memory_namespace = "ghost"
rooms            = ["!x:srv"]
"#,
        );
        let errors = cfg.validate_profiles();
        assert!(
            errors.iter().any(|e| e.contains("ghost")),
            "expected unknown-namespace error, got: {errors:?}"
        );
    }

    #[test]
    fn background_profile_resolves_from_namespace_first() {
        let cfg = parse(
            r#"
[anthropic]
api_key = "test"

[profiles.bg_global]
provider = "anthropic"

[profiles.bg_nsfw]
provider = "anthropic"

[profiles.background]
provider = "anthropic"

[memory_namespace.user_nsfw]
include            = ["default"]
background_profile = "bg_nsfw"
"#,
        );
        assert!(cfg.validate_profiles().is_empty());
        // Namespace-local override wins.
        assert_eq!(
            cfg.background_profile_for_namespace("user_nsfw"),
            Some("bg_nsfw")
        );
        // No namespace override → falls back to [profiles.background].
        assert_eq!(
            cfg.background_profile_for_namespace("default"),
            Some("background")
        );
    }

    #[test]
    fn background_profile_is_none_when_unconfigured() {
        let cfg = parse(MINIMAL);
        assert!(cfg.background_profile_for_namespace("default").is_none());
    }

    #[test]
    fn background_profile_falls_back_to_global() {
        let cfg = parse(
            r#"
[anthropic]
api_key = "test"

[profiles.background]
provider = "anthropic"
"#,
        );
        assert_eq!(
            cfg.background_profile_for_namespace("anything"),
            Some("background")
        );
    }

    #[test]
    fn unknown_background_profile_is_rejected() {
        let cfg = parse(
            r#"
[anthropic]
api_key = "test"

[memory_namespace.user]
background_profile = "ghost"
"#,
        );
        let errors = cfg.validate_profiles();
        assert!(
            errors.iter().any(|e| e.contains("ghost")),
            "expected ghost error, got: {errors:?}"
        );
    }

    #[test]
    fn all_memory_namespaces_unions_sources() {
        let cfg = parse(
            r#"
[anthropic]
api_key = "test"

[memory_namespace.user]
include = ["default"]

[profiles.nsfw]
provider         = "anthropic"
memory_namespace = "user_nsfw"

[memory_namespace.user_nsfw]
include = ["user"]
"#,
        );
        let all = cfg.all_memory_namespaces();
        assert!(all.contains(&"default".to_string()));
        assert!(all.contains(&"user".to_string()));
        assert!(all.contains(&"user_nsfw".to_string()));
    }

    #[test]
    fn voice_pipeline_config_parses_and_validates() {
        let cfg = parse(
            r#"
[anthropic]
api_key = "test"

[profiles.casual]
provider = "anthropic"

[voice_pipeline.default]
stt_provider = "sense_voice"
tts_provider = "irodori"
language     = "ja"

[stt_provider.sense_voice]
type  = "sherpa_onnx"
kind  = "sense_voice"
model = "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17"

[tts_provider.irodori]
type        = "openai_tts"
base_url    = "https://irodori-tts-api.home.fireturtle.net"
model       = "tts-1"
voice       = "alloy"

[room_profile.home]
profile        = "casual"
voice_pipeline = "default"
rooms          = []
"#,
        );
        assert!(
            cfg.validate_profiles().is_empty(),
            "errors: {:?}",
            cfg.validate_profiles()
        );
        let vp = cfg
            .voice_pipeline_for_room_profile("home")
            .expect("voice pipeline resolved");
        assert_eq!(vp.stt_provider, "sense_voice");
        assert_eq!(vp.tts_provider, "irodori");
        assert_eq!(vp.language.as_deref(), Some("ja"));
        assert_eq!(vp.capture_max_ms, 30_000); // default
    }

    #[test]
    fn sherpa_stt_config_round_trips_with_defaults() {
        let cfg = parse(
            r#"
[anthropic]
api_key = "test"

[stt_provider.sense_voice]
type   = "sherpa_onnx"
kind   = "sense_voice"
model  = "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17"
"#,
        );
        let stt = cfg
            .stt_providers
            .get("sense_voice")
            .expect("provider parses");
        match stt {
            SttProviderConfig::SherpaOnnx(s) => {
                assert!(matches!(s.kind, SherpaSttKind::SenseVoice));
                assert_eq!(
                    s.model.as_deref(),
                    Some("sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17")
                );
                assert_eq!(s.num_threads, 2);
                assert_eq!(s.provider, "cpu");
                assert!(s.language.is_none());
            }
            _ => panic!("expected SherpaOnnx variant"),
        }
    }

    #[test]
    fn sherpa_tts_config_round_trips_with_defaults() {
        let cfg = parse(
            r#"
[anthropic]
api_key = "test"

[tts_provider.vits_ja]
type        = "sherpa_onnx"
kind        = "vits"
model       = "vits-someone-2024"
speaker_id  = 3
speed       = 1.2
"#,
        );
        let tts = cfg.tts_providers.get("vits_ja").expect("provider parses");
        match tts {
            TtsProviderConfig::SherpaOnnx(s) => {
                assert!(matches!(s.kind, SherpaTtsKind::Vits));
                assert_eq!(s.speaker_id, 3);
                assert_eq!(s.speed, 1.2);
                assert_eq!(s.num_threads, 2);
                assert_eq!(s.provider, "cpu");
            }
            _ => panic!("expected SherpaOnnx variant"),
        }
    }

    #[test]
    fn voice_pipeline_rejects_unknown_stt() {
        let cfg = parse(
            r#"
[anthropic]
api_key = "test"

[voice_pipeline.default]
stt_provider = "ghost"
tts_provider = "irodori"

[tts_provider.irodori]
type     = "openai_tts"
base_url = "http://localhost:8000"
"#,
        );
        let errors = cfg.validate_profiles();
        assert!(
            errors.iter().any(|e| e.contains("ghost")),
            "got: {errors:?}"
        );
    }

    #[test]
    fn room_profile_voice_pipeline_must_exist() {
        let cfg = parse(
            r#"
[anthropic]
api_key = "test"

[profiles.casual]
provider = "anthropic"

[room_profile.home]
profile        = "casual"
voice_pipeline = "ghost"
rooms          = []
"#,
        );
        let errors = cfg.validate_profiles();
        assert!(
            errors.iter().any(|e| e.contains("ghost")),
            "got: {errors:?}"
        );
    }

    #[test]
    fn voice_wake_word_model_defaults_to_none() {
        let cfg = parse(
            r#"
[anthropic]
api_key = "test"
"#,
        );
        assert!(cfg.voice.wake_word_model.is_none());
    }

    #[test]
    fn voice_wake_word_model_rejects_missing_file() {
        let cfg = parse(
            r#"
[anthropic]
api_key = "test"

[voice]
wake_word_model = "/nonexistent/saphina.onnx"
"#,
        );
        let errors = cfg.validate_profiles();
        assert!(
            errors
                .iter()
                .any(|e| e.contains("/nonexistent/saphina.onnx")),
            "got: {errors:?}"
        );
    }

    #[test]
    fn provider_config_parses_openai_compatible() {
        let cfg = parse(
            r#"
[anthropic]
api_key = "test"

[providers.local]
type = "openai_compatible"
base_url = "http://127.0.0.1:8080/v1"
model = "gemma-4-31b-it"
"#,
        );
        let local = cfg.providers.get("local").expect("local provider present");
        match local {
            ProviderConfig::OpenAiCompatible(c) => {
                assert_eq!(c.base_url, "http://127.0.0.1:8080/v1");
                assert_eq!(c.model, "gemma-4-31b-it");
                assert!(c.api_key.is_none());
            }
        }
    }

    #[test]
    fn load_layered_without_a_workspace_config_matches_plain_load() {
        let dir = tempfile::tempdir().unwrap();
        let host_path = dir.path().join("config.toml");
        std::fs::write(
            &host_path,
            "day_boundary_hour = 5\n\n[anthropic]\napi_key = \"sk-test\"\n",
        )
        .unwrap();

        let loaded = Config::load_layered(&host_path).unwrap();
        assert_eq!(loaded.config.day_boundary_hour, 5);
        assert!(loaded.rejected.is_empty());
        // With no workspace file every value can only have come from the host.
        assert!(
            loaded
                .provenance
                .values()
                .all(|l| *l == crate::config_layer::Layer::Host)
        );
    }

    #[test]
    fn load_layered_merges_the_workspace_config() {
        let dir = tempfile::tempdir().unwrap();
        let host_path = dir.path().join("config.toml");
        std::fs::write(
            &host_path,
            "day_boundary_hour = 6\n\n[anthropic]\napi_key = \"sk-host\"\n",
        )
        .unwrap();
        let marker = dir.path().join(".sapphire-agent");
        std::fs::create_dir_all(&marker).unwrap();
        std::fs::write(
            marker.join("config.toml"),
            "day_boundary_hour = 4\n\n[anthropic]\nsystem_prompt = \"shared\"\napi_key = \"sk-should-not-travel\"\n",
        )
        .unwrap();

        let loaded = Config::load_layered(&host_path).unwrap();
        // Host wins where both set a key.
        assert_eq!(loaded.config.day_boundary_hour, 6);
        // Workspace supplies what the host omits.
        assert_eq!(
            loaded.config.anthropic.system_prompt.as_deref(),
            Some("shared")
        );
        // The host's secret is untouched and the workspace's is refused.
        assert_eq!(loaded.config.anthropic.api_key.as_deref(), Some("sk-host"));
        assert_eq!(loaded.rejected, vec!["anthropic.api_key".to_string()]);
        assert_eq!(
            loaded.provenance.get("anthropic.system_prompt"),
            Some(&crate::config_layer::Layer::Workspace)
        );
    }

    #[test]
    fn workspace_dir_from_the_host_layer_locates_the_workspace_config() {
        let dir = tempfile::tempdir().unwrap();
        let ws = dir.path().join("elsewhere");
        std::fs::create_dir_all(ws.join(".sapphire-agent")).unwrap();
        std::fs::write(
            ws.join(".sapphire-agent").join("config.toml"),
            "session_policy = \"none\"\n",
        )
        .unwrap();

        let host_path = dir.path().join("config.toml");
        std::fs::write(
            &host_path,
            format!(
                "workspace_dir = \"{}\"\n\n[anthropic]\napi_key = \"sk-test\"\n",
                ws.display().to_string().replace('\\', "\\\\")
            ),
        )
        .unwrap();

        let loaded = Config::load_layered(&host_path).unwrap();
        assert_eq!(loaded.config.session_policy, SessionPolicy::None);
    }

    #[test]
    fn a_workspace_config_that_fails_to_deserialize_falls_back_to_the_host_layer() {
        // The workspace's bad value for `day_boundary_hour` is fully allowlisted,
        // and the host sets the same key, so `host wins` on the merge already
        // neutralises it before deserialization ever runs. This pins that the
        // fallback decision is made against the *merged* document, not the raw
        // workspace document — a naive implementation that validated the
        // workspace layer standalone would wrongly fall back here even though
        // the merged config is perfectly fine.
        let dir = tempfile::tempdir().unwrap();
        let host_path = dir.path().join("config.toml");
        std::fs::write(
            &host_path,
            "day_boundary_hour = 5\n\n[anthropic]\napi_key = \"sk-test\"\n",
        )
        .unwrap();
        let marker = dir.path().join(".sapphire-agent");
        std::fs::create_dir_all(&marker).unwrap();
        std::fs::write(marker.join("config.toml"), "day_boundary_hour = \"nine\"\n").unwrap();

        let loaded = Config::load_layered(&host_path).unwrap();
        assert_eq!(loaded.config.day_boundary_hour, 5);
    }

    #[test]
    fn a_workspace_config_that_fails_validation_falls_back_to_the_host_layer() {
        // `room_profile.*.profile` is allowlisted, but the workspace's chosen
        // profile does not exist. The merged document deserializes fine — it's
        // `Config::validate_profiles` that catches this, one layer past
        // `try_into()`, and it must fall back exactly as a deserialize failure
        // does.
        let dir = tempfile::tempdir().unwrap();
        let host_path = dir.path().join("config.toml");
        std::fs::write(&host_path, "[anthropic]\napi_key = \"sk-test\"\n").unwrap();
        let marker = dir.path().join(".sapphire-agent");
        std::fs::create_dir_all(&marker).unwrap();
        std::fs::write(
            marker.join("config.toml"),
            "[room_profile.x]\nprofile = \"does-not-exist\"\n",
        )
        .unwrap();

        let loaded = Config::load_layered(&host_path).unwrap();
        assert!(
            loaded.config.validate_profiles().is_empty(),
            "{:?}",
            loaded.config.validate_profiles()
        );
    }

    #[test]
    fn a_host_config_that_fails_validation_is_fatal_and_blames_the_host_file() {
        // The bad profile reference is the *host's*, and the workspace file is
        // innocent — it only sets an unrelated key. Falling back would return a
        // config that is still invalid while naming the workspace file as the
        // culprit, so the fallback has to re-validate and let a host-side
        // failure stay fatal.
        let dir = tempfile::tempdir().unwrap();
        let host_path = dir.path().join("config.toml");
        std::fs::write(
            &host_path,
            "[anthropic]\napi_key = \"sk-test\"\n\n[room_profile.x]\nprofile = \"does-not-exist\"\n",
        )
        .unwrap();
        let marker = dir.path().join(".sapphire-agent");
        std::fs::create_dir_all(&marker).unwrap();
        std::fs::write(marker.join("config.toml"), "day_boundary_hour = 4\n").unwrap();

        let err = Config::load_layered(&host_path)
            .expect_err("a host config that fails validation must not load");
        let rendered = format!("{err:#}");
        assert!(
            rendered.contains(&host_path.display().to_string()),
            "error should name the host config, got: {rendered}"
        );
        assert!(
            rendered.contains("does-not-exist"),
            "error should carry the validation failure, got: {rendered}"
        );
    }

    #[test]
    fn acp_absent_means_disabled() {
        let cfg = parse(
            r#"
[anthropic]
api_key = "test"
"#,
        );
        assert!(cfg.acp.is_none());
        assert!(!cfg.acp.as_ref().is_some_and(|c| c.enabled));
    }

    #[test]
    fn acp_block_parses_and_defaults_to_disabled() {
        let cfg = parse(
            r#"
[anthropic]
api_key = "test"

[acp]
"#,
        );
        assert!(!cfg.acp.as_ref().expect("[acp] parsed").enabled);

        let cfg = parse(
            r#"
[anthropic]
api_key = "test"

[acp]
enabled = true
"#,
        );
        assert!(cfg.acp.as_ref().expect("[acp] parsed").enabled);
    }

    #[test]
    fn ambient_config_defaults_to_disabled_with_documented_values() {
        let cfg: crate::config::AmbientConfig = toml::from_str("").unwrap();
        assert!(!cfg.enabled, "ambient must be opt-in");
        assert_eq!(cfg.audio_retention_days, 7);
        assert_eq!(cfg.min_embed_ms, 1500);
        assert_eq!(cfg.match_threshold, 0.55);
        assert_eq!(cfg.promote_after_seconds, 60);
        assert_eq!(cfg.promote_after_days, 2);
        assert_eq!(cfg.max_queue, 1000);
        assert!(cfg.cache_dir.is_none());
    }

    #[test]
    fn ambient_model_fields_default_to_unset() {
        let cfg: crate::config::AmbientConfig = toml::from_str("").unwrap();
        assert!(cfg.vad_model_dir.is_none());
        assert!(cfg.embedding_model_dir.is_none());
        assert_eq!(cfg.vad_threshold, 0.5);
        assert_eq!(cfg.embedding_num_threads, 2);
    }

    #[test]
    fn ambient_model_fields_parse_from_toml() {
        let cfg: crate::config::AmbientConfig = toml::from_str(
            r#"
embedding_model_dir = "/models/3dspeaker"
vad_model_dir = "/models/silero"
vad_threshold = 0.6
embedding_num_threads = 4
"#,
        )
        .unwrap();
        assert_eq!(
            cfg.embedding_model_dir.as_deref(),
            Some("/models/3dspeaker")
        );
        assert_eq!(cfg.vad_model_dir.as_deref(), Some("/models/silero"));
        assert_eq!(cfg.vad_threshold, 0.6);
        assert_eq!(cfg.embedding_num_threads, 4);
    }
}
