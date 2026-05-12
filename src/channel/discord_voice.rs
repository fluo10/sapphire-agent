//! Discord voice channel listener.
//!
//! Auto-joins the channel IDs listed under `[discord].voice_channel_ids`,
//! subscribes to per-speaker audio via songbird, and gates processing
//! on at least one human user being present in the channel. When the
//! bot is alone — no humans in the voice chat — incoming voice ticks
//! are discarded without spending CPU on VAD / STT.
//!
//! Audio receive path (VAD / STT / LLM / TTS / playback) is wired
//! up incrementally — this commit gets the bot joining channels and
//! tracking presence reliably; subsequent commits hang the actual
//! voice pipeline off the speaking-ssrc dispatch.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::Context as _;
use serenity::all::{ChannelId, ChannelType, Context as SerenityContext, GuildId, UserId, VoiceState};
use serenity::async_trait;
use songbird::events::context_data::VoiceTick;
use songbird::events::{CoreEvent, Event, EventContext, EventHandler as VoiceEventHandler};
use songbird::model::payload::Speaking;
use tokio::sync::Mutex;
use tracing::{debug, info, warn};

#[cfg(feature = "voice-sherpa")]
use crate::voice::vad::{VAD_WINDOW_SAMPLES, build_default as build_silero_default};
#[cfg(feature = "voice-sherpa")]
use sherpa_onnx::VoiceActivityDetector;

/// Discord ships voice as 48 kHz stereo s16le; our pipeline runs at
/// 16 kHz mono. We downmix + resample inline before the VAD sees it.
const DISCORD_SAMPLE_RATE: u32 = 48_000;
const PIPELINE_SAMPLE_RATE: u32 = 16_000;

/// Per-voice-channel runtime state. Tracks which humans are currently
/// in the channel (so we can gate audio processing) and the
/// SSRC↔UserId map songbird requires to associate audio with speakers.
pub struct ChannelState {
    /// Discord voice channel id this state belongs to.
    pub channel_id: ChannelId,
    /// Non-bot user ids currently connected to the channel.
    pub present_users: HashSet<UserId>,
    /// ssrc → user_id mapping populated by SpeakingStateUpdate events.
    pub ssrc_to_user: HashMap<u32, UserId>,
    /// Per-speaker audio + VAD state. Built lazily on first
    /// VoiceTick from a speaker — songbird re-issues new ssrcs when
    /// users disconnect / reconnect, so we don't preallocate.
    pub speakers: HashMap<u32, SpeakerState>,
    /// `true` when at least one human is present. Cheap atomic check
    /// for the audio receive callback.
    pub has_human: Arc<AtomicBool>,
}

impl ChannelState {
    pub fn new(channel_id: ChannelId) -> Self {
        Self {
            channel_id,
            present_users: HashSet::new(),
            ssrc_to_user: HashMap::new(),
            speakers: HashMap::new(),
            has_human: Arc::new(AtomicBool::new(false)),
        }
    }

    fn recompute_presence(&self) {
        self.has_human
            .store(!self.present_users.is_empty(), Ordering::SeqCst);
    }
}

/// Per-speaker buffer + VAD state. Each Discord user gets one.
pub struct SpeakerState {
    /// Pending samples waiting for the next 512-sample VAD window.
    pub pending: Vec<f32>,
    /// Silero VAD instance — one per speaker so internal smoothing
    /// state doesn't bleed across users.
    #[cfg(feature = "voice-sherpa")]
    pub vad: VoiceActivityDetector,
}

impl SpeakerState {
    #[cfg(feature = "voice-sherpa")]
    pub fn new() -> anyhow::Result<Self> {
        Ok(Self {
            pending: Vec::with_capacity(VAD_WINDOW_SAMPLES * 2),
            vad: build_silero_default()?,
        })
    }
    #[cfg(not(feature = "voice-sherpa"))]
    pub fn new() -> anyhow::Result<Self> {
        Ok(Self {
            pending: Vec::new(),
        })
    }
}

/// Shared state across the whole voice subsystem. Keyed by voice
/// channel id so multiple channels coexist cleanly.
///
/// Cloning is cheap (just bumps the Arc); pass clones to every event
/// handler that needs read or write access.
#[derive(Clone, Default)]
pub struct VoiceContext {
    /// Channel id → state. The outer Mutex protects the map shape;
    /// each entry is its own state struct (Mutex internally locked
    /// during updates by the VoiceReceiver).
    pub channels: Arc<Mutex<HashMap<ChannelId, Arc<Mutex<ChannelState>>>>>,
    /// Voice channel ids the operator listed in
    /// `[discord].voice_channel_ids`. Looked up at ready-time so the
    /// bot only joins channels it's been told to.
    pub configured_ids: Arc<HashSet<u64>>,
    /// Bot's own user id, populated by `on_ready`. Filter for
    /// `voice_state_update` so the bot's own join/leave doesn't
    /// toggle the "humans present" gate.
    pub bot_user_id: Arc<Mutex<Option<UserId>>>,
}

impl VoiceContext {
    pub fn new(configured_ids: Vec<u64>) -> Self {
        Self {
            channels: Arc::new(Mutex::new(HashMap::new())),
            configured_ids: Arc::new(configured_ids.into_iter().collect()),
            bot_user_id: Arc::new(Mutex::new(None)),
        }
    }
}

/// songbird event handler installed per-channel. Currently logs
/// presence transitions and speaker-level audio activity; future
/// commits replace the log calls with VAD / STT dispatch.
struct VoiceReceiver {
    channel_id: ChannelId,
    state: Arc<Mutex<ChannelState>>,
    has_human: Arc<AtomicBool>,
}

#[async_trait]
impl VoiceEventHandler for VoiceReceiver {
    async fn act(&self, ctx: &EventContext<'_>) -> Option<Event> {
        match ctx {
            EventContext::SpeakingStateUpdate(Speaking { ssrc, user_id, .. }) => {
                if let Some(uid) = user_id {
                    let mut st = self.state.lock().await;
                    st.ssrc_to_user.insert(*ssrc, UserId::new(uid.0));
                    debug!(
                        "discord_voice {}: ssrc {ssrc} → user {}",
                        self.channel_id, uid.0
                    );
                }
            }
            EventContext::VoiceTick(VoiceTick { speaking, .. }) => {
                if !self.has_human.load(Ordering::SeqCst) {
                    return None;
                }
                let mut st = self.state.lock().await;
                for (ssrc, data) in speaking {
                    let Some(decoded) = data.decoded_voice.as_ref() else {
                        continue;
                    };
                    // 48 kHz stereo i16 → 16 kHz mono f32.
                    let mono16k = downmix_and_resample(decoded);
                    on_speaker_audio(&mut st, *ssrc, &mono16k, self.channel_id).await;
                }
            }
            _ => {}
        }
        None
    }
}

/// React to a Discord `voice_state_update` event. The bot's own
/// user id (populated by `on_ready`) is used to filter out the
/// bot's own join/leave so it doesn't toggle the "humans present"
/// gate.
pub async fn on_voice_state_update(voice: &VoiceContext, update: VoiceStateUpdate) {
    let bot_uid = *voice.bot_user_id.lock().await;
    if Some(update.user_id) == bot_uid {
        return;
    }
    let channels = voice.channels.lock().await;
    for (ch_id, state_arc) in channels.iter() {
        let mut ch_state = state_arc.lock().await;
        let user = update.user_id;
        let was_present = ch_state.present_users.contains(&user);
        let is_present = update.channel_id == Some(*ch_id);
        match (was_present, is_present) {
            (false, true) => {
                ch_state.present_users.insert(user);
                ch_state.recompute_presence();
                info!(
                    "discord_voice {ch_id}: user {user} joined (humans now {})",
                    ch_state.present_users.len()
                );
            }
            (true, false) => {
                ch_state.present_users.remove(&user);
                ch_state.recompute_presence();
                info!(
                    "discord_voice {ch_id}: user {user} left (humans now {})",
                    ch_state.present_users.len()
                );
            }
            _ => {}
        }
    }
}

/// Light wrapper over the data we care about from a Discord
/// `voice_state_update` payload, so the call site doesn't have to
/// pull all of serenity's VoiceState fields.
#[derive(Debug, Clone, Copy)]
pub struct VoiceStateUpdate {
    pub user_id: UserId,
    pub channel_id: Option<ChannelId>,
    pub guild_id: Option<GuildId>,
}

impl From<&VoiceState> for VoiceStateUpdate {
    fn from(v: &VoiceState) -> Self {
        Self {
            user_id: v.user_id,
            channel_id: v.channel_id,
            guild_id: v.guild_id,
        }
    }
}

/// Build a songbird Config with PCM decode enabled. Without this,
/// `VoiceTick.speaking[ssrc].decoded_voice` is `None` and we can't
/// feed audio to the VAD downstream.
pub fn songbird_config() -> songbird::Config {
    songbird::Config::default().decode_mode(songbird::driver::DecodeMode::Decode(
        songbird::driver::DecodeConfig::default(),
    ))
}

/// On serenity `ready`, walk every configured voice channel id and
/// have songbird join it. Resolves guild_id via the Discord API
/// because configs only know about channel ids.
pub async fn on_ready(
    ctx: &SerenityContext,
    bot_user_id: UserId,
    voice: &VoiceContext,
) {
    *voice.bot_user_id.lock().await = Some(bot_user_id);

    let manager = match songbird::get(ctx).await {
        Some(m) => m,
        None => {
            warn!("discord_voice: songbird manager not registered on the serenity client");
            return;
        }
    };

    for &raw_id in voice.configured_ids.iter() {
        let channel_id = ChannelId::new(raw_id);
        let channel = match channel_id.to_channel(&ctx.http).await {
            Ok(c) => c,
            Err(e) => {
                warn!("discord_voice {channel_id}: failed to fetch channel info: {e}");
                continue;
            }
        };
        let guild_channel = match channel.guild() {
            Some(gc) => gc,
            None => {
                warn!("discord_voice {channel_id}: not a guild channel (DMs unsupported)");
                continue;
            }
        };
        if guild_channel.kind != ChannelType::Voice && guild_channel.kind != ChannelType::Stage {
            warn!(
                "discord_voice {channel_id}: not a voice channel (kind={:?}); skipping",
                guild_channel.kind
            );
            continue;
        }
        let guild_id = guild_channel.guild_id;

        match manager.join(guild_id, channel_id).await {
            Ok(call) => {
                let ch_state = Arc::new(Mutex::new(ChannelState::new(channel_id)));
                voice
                    .channels
                    .lock()
                    .await
                    .insert(channel_id, Arc::clone(&ch_state));
                register_handlers(&call, ch_state, channel_id).await;
                info!("discord_voice: joined {channel_id} in guild {guild_id}");
            }
            Err(e) => {
                warn!("discord_voice {channel_id}: failed to join: {e}");
            }
        }
    }
}

/// Install our event handlers on a fresh songbird `Call` after the
/// bot joins a voice channel.
async fn register_handlers(
    call: &Arc<tokio::sync::Mutex<songbird::Call>>,
    state: Arc<Mutex<ChannelState>>,
    channel_id: ChannelId,
) {
    let has_human = {
        let st = state.lock().await;
        Arc::clone(&st.has_human)
    };
    let receiver = VoiceReceiver {
        channel_id,
        state,
        has_human,
    };
    let mut call = call.lock().await;
    call.add_global_event(
        CoreEvent::SpeakingStateUpdate.into(),
        receiver_clone(&receiver),
    );
    call.add_global_event(CoreEvent::VoiceTick.into(), receiver);
}

fn receiver_clone(r: &VoiceReceiver) -> VoiceReceiver {
    VoiceReceiver {
        channel_id: r.channel_id,
        state: Arc::clone(&r.state),
        has_human: Arc::clone(&r.has_human),
    }
}

/// Convenience: anyhow-typed channel-id parse so the discord.rs
/// caller doesn't have to repeat the conversion logic.
pub fn parse_channel_ids(raw: &[String]) -> anyhow::Result<Vec<u64>> {
    raw.iter()
        .map(|s| {
            s.parse::<u64>()
                .with_context(|| format!("invalid Discord voice channel id '{s}'"))
        })
        .collect()
}

/// Downmix Discord's 48 kHz stereo i16 (interleaved) to 16 kHz mono
/// f32 normalised to [-1, 1]. Linear-interpolation resample, same
/// approach the satellite + sherpa-onnx TTS provider use — quality
/// is fine for speech.
fn downmix_and_resample(stereo_48k: &[i16]) -> Vec<f32> {
    // Stereo → mono (i16).
    let mut mono: Vec<i16> = Vec::with_capacity(stereo_48k.len() / 2);
    for frame in stereo_48k.chunks_exact(2) {
        let sum = frame[0] as i32 + frame[1] as i32;
        mono.push((sum / 2) as i16);
    }
    // 48 kHz → 16 kHz (ratio 3:1).
    let ratio = DISCORD_SAMPLE_RATE as f64 / PIPELINE_SAMPLE_RATE as f64;
    let out_len = ((mono.len() as f64) / ratio).round() as usize;
    let mut out = Vec::with_capacity(out_len);
    if mono.is_empty() {
        return out;
    }
    for i in 0..out_len {
        let src_pos = i as f64 * ratio;
        let idx = src_pos.floor() as usize;
        let frac = src_pos - idx as f64;
        let s0 = mono[idx.min(mono.len() - 1)] as f64;
        let s1 = mono[(idx + 1).min(mono.len() - 1)] as f64;
        let v = (s0 * (1.0 - frac) + s1 * frac) / (i16::MAX as f64);
        out.push(v as f32);
    }
    out
}

/// Push fresh 16 kHz mono audio from one speaker into their per-ssrc
/// state and drain any complete utterances out of the VAD. Drops
/// audio silently when the `voice-sherpa` feature is off (Silero VAD
/// requires sherpa-onnx).
async fn on_speaker_audio(
    ch: &mut ChannelState,
    ssrc: u32,
    samples_16k_f32: &[f32],
    channel_id: ChannelId,
) {
    let speaker = match ch.speakers.entry(ssrc) {
        std::collections::hash_map::Entry::Occupied(o) => o.into_mut(),
        std::collections::hash_map::Entry::Vacant(v) => match SpeakerState::new() {
            Ok(s) => v.insert(s),
            Err(e) => {
                warn!("discord_voice {channel_id}: failed to init speaker {ssrc}: {e:#}");
                return;
            }
        },
    };
    speaker.pending.extend_from_slice(samples_16k_f32);

    #[cfg(feature = "voice-sherpa")]
    {
        while speaker.pending.len() >= VAD_WINDOW_SAMPLES {
            let frame: Vec<f32> = speaker.pending.drain(..VAD_WINDOW_SAMPLES).collect();
            speaker.vad.accept_waveform(&frame);
        }
        while !speaker.vad.is_empty() {
            let Some(segment) = speaker.vad.front() else {
                break;
            };
            let samples = segment.samples().to_vec();
            speaker.vad.pop();
            let user_str = ch
                .ssrc_to_user
                .get(&ssrc)
                .map(|u| format!("user {u}"))
                .unwrap_or_else(|| format!("ssrc {ssrc}"));
            info!(
                "discord_voice {channel_id}: utterance from {user_str} ({:.2}s, {} samples) \
                 — TODO: dispatch to voice pipeline",
                samples.len() as f32 / PIPELINE_SAMPLE_RATE as f32,
                samples.len()
            );
            // TODO(next commit): convert `samples` (f32 mono 16k) to
            // i16 and hand to the shared run_voice_turn. Audio reply
            // comes back as PCM chunks that we feed into a songbird
            // track for playback.
        }
    }

    #[cfg(not(feature = "voice-sherpa"))]
    {
        let _ = (samples_16k_f32, channel_id, speaker);
        // Without voice-sherpa: VAD is not available, so we just
        // buffer + drop. Log throttling to avoid spam.
    }
}
