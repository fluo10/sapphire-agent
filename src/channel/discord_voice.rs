//! Discord voice channel listener (scaffold).
//!
//! Auto-joins the channel IDs listed under `[discord].voice_channel_ids`,
//! subscribes to per-speaker audio via songbird, and gates processing
//! on at least one human user being present in the channel. When the
//! bot is alone — no humans in the voice chat — incoming voice ticks
//! are discarded without spending CPU on VAD / STT.
//!
//! **Status: scaffolding only.** This commit gets the bot to join
//! voice channels and surfaces audio events via tracing logs so the
//! basic wiring can be verified end-to-end. The actual VAD →
//! utterance assembly → voice pipeline → playback path lands in
//! subsequent commits. Until then, the bot silently observes —
//! nothing is forwarded to the LLM and nothing is played back.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use serenity::all::{ChannelId, GuildId, UserId, VoiceState};
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
#[derive(Clone, Default)]
pub struct VoiceState_ {
    pub channels: Arc<Mutex<HashMap<ChannelId, ChannelState>>>,
}

/// songbird event handler installed per-channel. Currently logs
/// presence transitions and speaker-level audio activity; future
/// commits replace the log calls with VAD / STT dispatch.
pub struct VoiceReceiver {
    pub channel_id: ChannelId,
    pub state: Arc<Mutex<ChannelState>>,
    pub has_human: Arc<AtomicBool>,
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

/// React to a Discord `voice_state_update` event. Returns whether the
/// channel's presence state changed (caller logs the transition).
pub async fn on_voice_state_update(
    voice_state: &VoiceStateUpdate,
    bot_user_id: UserId,
    state: &Mutex<HashMap<ChannelId, ChannelState>>,
) {
    let mut channels = state.lock().await;
    for ch_state in channels.values_mut() {
        // A user is now considered "in" this channel if their new
        // voice_state names it. We don't track per-user move history;
        // just snapshot membership from each event.
        let user = voice_state.user_id;
        let was_present = ch_state.present_users.contains(&user);
        let is_present = voice_state.channel_id == Some(ch_state.channel_id);
        let counts_as_human = user != bot_user_id;
        if !counts_as_human {
            continue;
        }
        match (was_present, is_present) {
            (false, true) => {
                ch_state.present_users.insert(user);
                ch_state.recompute_presence();
                info!(
                    "discord_voice {}: user {} joined (presence={})",
                    ch_state.channel_id,
                    user,
                    ch_state.present_users.len()
                );
            }
            (true, false) => {
                ch_state.present_users.remove(&user);
                ch_state.recompute_presence();
                info!(
                    "discord_voice {}: user {} left (presence={})",
                    ch_state.channel_id,
                    user,
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

/// Install our event handlers on a fresh songbird `Call` after the
/// bot joins a voice channel.
pub async fn register_handlers(
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
    call.add_global_event(CoreEvent::SpeakingStateUpdate.into(), receiver_clone(&receiver));
    call.add_global_event(CoreEvent::VoiceTick.into(), receiver);
}

/// `VoiceReceiver` is not `Clone` by design (it owns Arcs internally
/// already, but songbird's `add_global_event` needs us to register
/// multiple times with separate handler instances). This is a hand
/// clone for that purpose.
fn receiver_clone(r: &VoiceReceiver) -> VoiceReceiver {
    VoiceReceiver {
        channel_id: r.channel_id,
        state: Arc::clone(&r.state),
        has_human: Arc::clone(&r.has_human),
    }
}
