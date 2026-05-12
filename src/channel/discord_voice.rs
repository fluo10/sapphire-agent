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
            has_human: Arc::new(AtomicBool::new(false)),
        }
    }

    fn recompute_presence(&self) {
        self.has_human
            .store(!self.present_users.is_empty(), Ordering::SeqCst);
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
                // TODO(next commit): for each speaking ssrc, look up
                // user_id, push `decoded_voice` (48 kHz stereo i16)
                // into a per-user ring buffer, feed Silero VAD, on
                // utterance complete call into the shared voice
                // pipeline (STT → LLM → TTS).
                if !speaking.is_empty() {
                    let st = self.state.lock().await;
                    let speakers: Vec<String> = speaking
                        .keys()
                        .map(|ssrc| match st.ssrc_to_user.get(ssrc) {
                            Some(uid) => format!("user {} (ssrc {ssrc})", uid),
                            None => format!("ssrc {ssrc}"),
                        })
                        .collect();
                    debug!(
                        "discord_voice {}: speaking — {}",
                        self.channel_id,
                        speakers.join(", ")
                    );
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
