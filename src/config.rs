use anyhow::{Context, Result};
use sapphire_workspace::SyncConfig;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Config {
    /// Matrix channel configuration. Both `matrix` and `discord` may be
    /// configured at once — when set, both run concurrently in the
    /// same `serve` process. At least one of them is required (unless
    /// `standby_mode = true`).
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
    /// Cold-standby mode: only perform git sync, skip channel listening and
    /// heartbeat tasks. Useful for maintaining a backup node that stays in
    /// sync without actively processing messages. Default: false.
    #[serde(default)]
    pub standby_mode: bool,
    /// Workspace sync configuration.
    ///
    /// The workspace-level config (`{workspace_dir}/.sapphire-agent/config.toml`)
    /// provides shared defaults. This per-user `[sync]` section, when present,
    /// takes precedence — allowing each user to override the workspace defaults.
    #[serde(default)]
    pub sync: Option<SyncConfig>,
    /// How often the agent runs the periodic workspace sync cycle, in
    /// minutes. Unset or `0` disables periodic sync entirely. Each tick
    /// runs `WorkspaceState::periodic_sync`, which does a git sync **and**
    /// an mtime-based refresh of the retrieve cache — one cadence drives
    /// both.
    ///
    /// Lives at the config root (not inside `[sync]`) because the cadence
    /// spans both `sapphire-sync` and `sapphire-retrieve`; nesting it
    /// under `[sync]` would have implied a sync-only knob and forced a
    /// duplicate for the retrieve side. Upstream relocated it out of
    /// `SyncConfig` for the same reason in sapphire-workspace 0.10.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub sync_interval_minutes: Option<u32>,
    /// Periodic log digest configuration (weekly / monthly / yearly).
    #[serde(default)]
    pub digest: DigestConfig,
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
}

fn default_true() -> bool {
    true
}

/// Action taken at the day boundary for a given conversation.
///
/// - `Reset`: close the session and clear in-memory caches (legacy behavior).
///   The next message starts a fresh session; prior-run summary is injected via
///   `restart_summaries`.
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
///
/// Future-extension fields (planned for a follow-up release):
///   - `api_enabled: bool` — gate API access per room profile
///   - `api_keys: Vec<String>` — bearer tokens accepted by the API for
///     this room profile
///
/// See https://github.com/fluo10/sapphire-agent/issues/73
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
    /// Generic Gradio Web UI client. Works against any Gradio-hosted TTS
    /// app (Irodori-TTS, Style-Bert-VITS2, etc.) — the user supplies the
    /// endpoint name and a payload template.
    #[serde(rename = "gradio")]
    Gradio {
        /// Gradio base URL (without the `/gradio_api/...` suffix), e.g.
        /// `http://localhost:7860`.
        base_url: String,
        /// API endpoint name as exposed by the Gradio app. Becomes
        /// `{base_url}/gradio_api/call/{fn_name}` — a leading `/` is
        /// stripped. Examples: `generate`, `predict`.
        fn_name: String,
        /// Payload template (JSON, serialized as a string). `{{text}}`
        /// is substituted with the utterance text at call time. Must
        /// resolve to a `{"data": [...]}` shape per Gradio's API.
        payload: String,
        /// RFC 6901 JSON Pointer selecting the audio location in the
        /// SSE `complete` event's parsed payload.
        ///
        /// Gradio 4.x emits a **bare JSON array** (not `{"data": [...]}`),
        /// where each element is a component update of shape
        /// `{"__type__": "update", "value": {...}, "visible": ...}`.
        /// For an Audio component the inner `value` is a FileData
        /// object with `url`, `path`, `orig_name`, etc.
        ///
        /// The pointer must resolve to one of:
        ///   * a string (treated as URL or path),
        ///   * an object with a string `url` or `path` field (auto-unwrapped).
        ///
        /// Typical values: `/0/value` (lets the resolver auto-unwrap
        /// the FileData's `url`), `/0/value/url` (explicit).
        audio_field: String,
    },
    /// OpenAI's `audio/speech` endpoint (`tts-1` / `tts-1-hd`).
    #[serde(rename = "openai_tts")]
    OpenAiTts {
        /// Environment variable holding the API key.
        api_key_env: String,
        /// Optional base URL override. Defaults to OpenAI's public endpoint.
        #[serde(default)]
        base_url: Option<String>,
        /// Model name. Defaults to `tts-1` when omitted.
        #[serde(default)]
        model: Option<String>,
        /// Voice name (`alloy`, `echo`, `fable`, `onyx`, `nova`, `shimmer`).
        /// Defaults to `alloy`.
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
    /// Style-Bert-VITS2 FastAPI server
    /// (https://github.com/litagin02/Style-Bert-VITS2). Simpler API
    /// than Gradio — single `POST /voice` returns a WAV — and the
    /// training/merging ecosystem is more mature than Irodori-TTS's,
    /// which makes it the practical choice for custom Japanese voices.
    #[serde(rename = "style_bert_vits2")]
    StyleBertVits2(StyleBertVits2Config),
}

/// Configuration for the Style-Bert-VITS2 TTS provider.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct StyleBertVits2Config {
    /// Base URL where the SBV2 FastAPI server is reachable, e.g.
    /// `"http://localhost:5000"`.
    pub base_url: String,
    /// Numeric `model_id` registered with the server. See
    /// `GET /models/info` on the server for the list.
    #[serde(default)]
    pub model_id: i32,
    /// Numeric `speaker_id` within the chosen model.
    #[serde(default)]
    pub speaker_id: i32,
    /// Style name (defaults to the model's default style when omitted).
    #[serde(default)]
    pub style: Option<String>,
    /// Style strength (1.0 = neutral; higher exaggerates).
    #[serde(default = "default_sbv2_style_weight")]
    pub style_weight: f32,
    /// BCP-47 language ("JP", "EN", "ZH" per SBV2 convention).
    #[serde(default)]
    pub language: Option<String>,
    /// Speaking rate (1.0 = normal; <1.0 slower, >1.0 faster).
    #[serde(default = "default_sbv2_length")]
    pub length: f32,
    /// SDP ratio (0.0–1.0). Default 0.2 per SBV2 README.
    #[serde(default = "default_sbv2_sdp_ratio")]
    pub sdp_ratio: f32,
    /// Noise scale (prosody jitter). Default 0.6.
    #[serde(default = "default_sbv2_noise")]
    pub noise: f32,
    /// Noise-w scale (cadence jitter). Default 0.8.
    #[serde(default = "default_sbv2_noisew")]
    pub noisew: f32,
}

fn default_sbv2_style_weight() -> f32 {
    1.0
}
fn default_sbv2_length() -> f32 {
    1.0
}
fn default_sbv2_sdp_ratio() -> f32 {
    0.2
}
fn default_sbv2_noise() -> f32 {
    0.6
}
fn default_sbv2_noisew() -> f32 {
    0.8
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
}

/// Configuration for a single external MCP server.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct McpServerConfig {
    /// Human-readable name (used in tool prefix: `mcp__<name>__<tool>`).
    pub name: String,
    /// Transport configuration.
    #[serde(flatten)]
    pub transport: McpTransportConfig,
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

impl Config {
    pub fn load(path: &Path) -> Result<Self> {
        let content = std::fs::read_to_string(path)
            .with_context(|| format!("Failed to read config file: {}", path.display()))?;
        let config: Config =
            toml::from_str(&content).with_context(|| "Failed to parse config file")?;
        Ok(config)
    }

    /// Resolve the workspace directory: explicit config > config file's parent directory.
    pub fn resolved_workspace_dir(&self, config_path: &Path) -> PathBuf {
        if let Some(dir) = &self.workspace_dir {
            PathBuf::from(shellexpand::tilde(dir).as_ref())
        } else {
            config_path
                .parent()
                .unwrap_or_else(|| Path::new("."))
                .to_path_buf()
        }
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
    pub fn provider_for_profile(&self, profile_name: &str) -> Option<&str> {
        self.profiles
            .get(profile_name)
            .map(|p| p.provider.as_str())
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
            if let Some(fb) = &prof.fallback_provider {
                if !known_provider(fb) {
                    errors.push(format!(
                        "profile '{pname}' references unknown fallback_provider '{fb}'"
                    ));
                }
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
            if let Some(ns) = &rp.memory_namespace {
                if !self.namespace_is_defined(ns) {
                    errors.push(format!(
                        "room_profile '{rp_name}' references unknown memory_namespace '{ns}'"
                    ));
                }
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
            if let Some(prof) = &ns_cfg.background_profile {
                if !self.profiles.contains_key(prof) {
                    errors.push(format!(
                        "memory_namespace '{ns_name}' references unknown background_profile '{prof}'"
                    ));
                }
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
            if let Some(vp) = &rp.voice_pipeline {
                if !self.voice_pipelines.contains_key(vp) {
                    errors.push(format!(
                        "room_profile '{rp_name}' references unknown voice_pipeline '{vp}'"
                    ));
                }
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
                    let mut cycle: Vec<String> = stack.iter().cloned().collect();
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

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(s: &str) -> Config {
        toml::from_str(s).expect("config should parse")
    }

    const MINIMAL: &str = r#"
[anthropic]
api_key = "test"
"#;

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
    fn shipped_example_parses() {
        // Sanity check: the example file we ship in the repo must parse
        // and validate without errors so first-time users aren't greeted
        // with a confusing TOML error.
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("config.example.toml");
        let cfg = Config::load(&path).expect("config.example.toml should parse");
        assert!(
            cfg.validate_profiles().is_empty(),
            "validation errors: {:?}",
            cfg.validate_profiles()
        );
    }

    /// Smoke check that every TOML under `test-configs/` parses and
    /// validates. These files are documentation templates the user
    /// copies into place for manual end-to-end runs; if they break
    /// we want to know before they're copy-pasted, not after.
    #[test]
    fn test_configs_parse_and_validate() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("test-configs");
        let entries: Vec<std::path::PathBuf> = std::fs::read_dir(&dir)
            .unwrap_or_else(|e| panic!("read_dir({}) failed: {e}", dir.display()))
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|x| x == "toml"))
            .collect();
        assert!(
            !entries.is_empty(),
            "test-configs/ must contain at least one .toml"
        );
        for path in entries {
            let cfg = Config::load(&path)
                .unwrap_or_else(|e| panic!("{} failed to parse: {e:#}", path.display()));
            let errs = cfg.validate_profiles();
            assert!(
                errs.is_empty(),
                "{} validation errors: {:?}",
                path.display(),
                errs
            );
        }
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
            vec!["user_nsfw".to_string(), "user".to_string(), "default".to_string()]
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
type        = "gradio"
base_url    = "http://localhost:7860"
fn_name     = "/predict"
payload     = '{"data":["{{text}}"]}'
audio_field = "/data/0"

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
                assert_eq!(s.model.as_deref(), Some("sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17"));
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
        let tts = cfg
            .tts_providers
            .get("vits_ja")
            .expect("provider parses");
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
type        = "gradio"
base_url    = "http://localhost:7860"
fn_name     = "/predict"
payload     = '{}'
audio_field = "/data/0"
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
            errors.iter().any(|e| e.contains("/nonexistent/saphina.onnx")),
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
}
