use crate::channel::discord_voice::{self, VoiceContext, VoiceStateUpdate};
use crate::channel::{Attachment, Channel, IncomingMessage, MAX_ATTACHMENT_BYTES, OutgoingMessage};
use crate::config::DiscordConfig;
use crate::serve::ServeState;
use anyhow::{Context, Result};
use async_trait::async_trait;
use serenity::Client;
use serenity::all::{
    ChannelId, Context as SerenityCtx, EventHandler, GatewayIntents, Message, Ready, VoiceState,
};
use serenity::http::Http;
use songbird::SerenityInit;
use std::collections::HashSet;
use std::sync::Arc;
use tokio::sync::{OnceCell, mpsc};
use tracing::{debug, info, warn};

// ---------------------------------------------------------------------------
// Internal event handler
// ---------------------------------------------------------------------------

struct DiscordHandler {
    tx: mpsc::Sender<IncomingMessage>,
    allowed_channel_ids: HashSet<u64>,
    allowed_user_ids: HashSet<u64>,
    /// `Some` when voice channels are configured; drives the voice
    /// auto-join logic on the `ready` event and the presence map on
    /// `voice_state_update`.
    voice: Option<VoiceContext>,
}

#[serenity::async_trait]
impl EventHandler for DiscordHandler {
    async fn ready(&self, ctx: SerenityCtx, ready: Ready) {
        info!("Discord bot connected as {}", ready.user.name);
        if let Some(voice) = &self.voice {
            discord_voice::on_ready(&ctx, ready.user.id, voice).await;
        }
    }

    async fn voice_state_update(
        &self,
        _ctx: SerenityCtx,
        _old: Option<VoiceState>,
        new: VoiceState,
    ) {
        let Some(voice) = &self.voice else {
            return;
        };
        discord_voice::on_voice_state_update(voice, VoiceStateUpdate::from(&new)).await;
    }

    async fn message(&self, _ctx: SerenityCtx, msg: Message) {
        // Ignore bots (includes self)
        if msg.author.bot {
            return;
        }

        let channel_id = msg.channel_id.get();
        if !self.allowed_channel_ids.is_empty() && !self.allowed_channel_ids.contains(&channel_id) {
            debug!("Ignoring message from channel {channel_id} (not in allowed list)");
            return;
        }

        let user_id = msg.author.id.get();
        if !self.allowed_user_ids.is_empty() && !self.allowed_user_ids.contains(&user_id) {
            debug!("Ignoring message from user {user_id} (not in allowed list)");
            return;
        }

        let content = msg.content.trim().to_string();
        let attachments = download_image_attachments(&msg).await;

        // Skip messages that have neither text nor any usable image attachment.
        if content.is_empty() && attachments.is_empty() {
            return;
        }

        let incoming = IncomingMessage {
            id: msg.id.to_string(),
            sender: msg.author.id.to_string(),
            content,
            room_id: msg.channel_id.to_string(),
            timestamp: msg.timestamp.unix_timestamp() as u64 * 1000,
            thread_id: None,
            attachments,
        };

        if let Err(e) = self.tx.send(incoming).await {
            warn!("Failed to forward Discord message: {e}");
        }
    }
}

// ---------------------------------------------------------------------------
// DiscordChannel
// ---------------------------------------------------------------------------

pub struct DiscordChannel {
    token: String,
    channel_ids: HashSet<u64>,
    voice_channel_ids: Vec<u64>,
    allowed_user_ids: HashSet<u64>,
    /// Filled by `listen()` once the gateway client is built; used by `send()`.
    http: Arc<OnceCell<Arc<Http>>>,
    /// Shared server runtime so the Discord voice path can run the
    /// same STT → LLM → TTS pipeline as the HTTP `voice/pipeline_run`
    /// endpoint. `None` when voice is not configured.
    serve_state: Option<Arc<ServeState>>,
}

impl std::fmt::Debug for DiscordChannel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DiscordChannel")
            .field("channel_ids", &self.channel_ids)
            .finish_non_exhaustive()
    }
}

impl DiscordChannel {
    pub fn new(cfg: &DiscordConfig, serve_state: Option<Arc<ServeState>>) -> Result<Self> {
        let channel_ids = cfg
            .channel_ids
            .iter()
            .map(|s| s.parse::<u64>().context("Invalid Discord channel_id"))
            .collect::<Result<HashSet<_>>>()?;

        let voice_channel_ids = discord_voice::parse_channel_ids(&cfg.voice_channel_ids)?;

        let allowed_user_ids = cfg
            .allowed_users
            .iter()
            .map(|s| {
                s.parse::<u64>()
                    .context("Invalid Discord user ID in allowed_users")
            })
            .collect::<Result<HashSet<_>>>()?;

        Ok(Self {
            token: cfg.bot_token.clone(),
            channel_ids,
            voice_channel_ids,
            allowed_user_ids,
            http: Arc::new(OnceCell::new()),
            serve_state,
        })
    }

    fn get_http_or_new(&self) -> Arc<Http> {
        if let Some(http) = self.http.get() {
            Arc::clone(http)
        } else {
            // listen() hasn't started yet; create a standalone HTTP client.
            Arc::new(Http::new(&self.token))
        }
    }
}

// ---------------------------------------------------------------------------
// Channel impl
// ---------------------------------------------------------------------------

#[async_trait]
impl Channel for DiscordChannel {
    fn name(&self) -> &str {
        "discord"
    }

    async fn send(&self, message: &OutgoingMessage) -> Result<()> {
        let http = self.get_http_or_new();
        let channel_id: u64 = message
            .room_id
            .parse()
            .context("Discord room_id is not a valid channel ID")?;
        let channel_id = ChannelId::new(channel_id);

        for chunk in split_for_discord(&message.content) {
            channel_id
                .say(http.as_ref(), chunk)
                .await
                .context("Failed to send Discord message")?;
        }
        Ok(())
    }

    /// Long-running: connects to the Discord gateway and forwards incoming
    /// messages through `tx`. Reconnects with exponential backoff on disconnect.
    async fn listen(&self, tx: mpsc::Sender<IncomingMessage>) -> Result<()> {
        // MESSAGE_CONTENT is a privileged intent — enable it in Discord Developer Portal
        // (Bot → Privileged Gateway Intents → Message Content Intent).
        // GUILD_VOICE_STATES is required to receive voice_state_update
        // events (presence in voice channels). Always enabled — it's
        // not privileged and costs nothing when voice isn't used.
        let intents = GatewayIntents::GUILD_MESSAGES
            | GatewayIntents::DIRECT_MESSAGES
            | GatewayIntents::MESSAGE_CONTENT
            | GatewayIntents::GUILD_VOICE_STATES;

        // Voice context, once and only once. Cloned into each handler
        // instance on reconnect; the underlying state Arc survives so
        // presence + channel-state survive transient disconnects.
        let voice_ctx = if self.voice_channel_ids.is_empty() {
            None
        } else {
            Some(VoiceContext::new(
                self.voice_channel_ids.clone(),
                self.serve_state.clone(),
            ))
        };

        let min_backoff = std::time::Duration::from_secs(1);
        let max_backoff = std::time::Duration::from_secs(300);
        let stable_threshold = std::time::Duration::from_secs(60);
        let mut backoff = min_backoff;

        loop {
            let handler = DiscordHandler {
                tx: tx.clone(),
                allowed_channel_ids: self.channel_ids.clone(),
                allowed_user_ids: self.allowed_user_ids.clone(),
                voice: voice_ctx.clone(),
            };

            let mut builder = Client::builder(&self.token, intents).event_handler(handler);
            if voice_ctx.is_some() {
                // Register songbird against this serenity client so
                // event handlers can fetch the manager via
                // `songbird::get(&ctx).await` on `ready`.
                builder = builder.register_songbird_from_config(discord_voice::songbird_config());
            }
            let mut client = builder
                .await
                .context("Failed to build Discord client")?;

            // Share the HTTP client so send() can use it immediately.
            let _ = self.http.set(Arc::clone(&client.http));

            info!("Starting Discord gateway...");
            let started = std::time::Instant::now();
            match client.start().await {
                Ok(()) => {
                    warn!("Discord gateway exited without error; reconnecting in {backoff:?}");
                }
                Err(e) => {
                    warn!("Discord gateway exited with error: {e}; reconnecting in {backoff:?}");
                }
            }
            tokio::time::sleep(backoff).await;
            if started.elapsed() >= stable_threshold {
                backoff = min_backoff;
            } else {
                backoff = (backoff * 2).min(max_backoff);
            }
            info!("Reconnecting Discord gateway...");
        }
    }

    async fn start_typing(&self, room_id: &str) -> Result<()> {
        let http = self.get_http_or_new();
        let channel_id: u64 = room_id.parse().context("Invalid Discord channel ID")?;
        // broadcast_typing lasts ~10 seconds and auto-expires; no explicit stop needed.
        ChannelId::new(channel_id)
            .broadcast_typing(http.as_ref())
            .await
            .context("Failed to send typing indicator")?;
        Ok(())
    }

    // stop_typing is a no-op: Discord typing expires automatically.
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Download every `image/*` attachment on `msg`. Oversized attachments
/// (>5 MB) and non-image attachments are skipped with a warning so the
/// conversation continues without them.
async fn download_image_attachments(msg: &Message) -> Vec<Attachment> {
    let mut out = Vec::new();
    for att in &msg.attachments {
        let Some(ct) = att.content_type.as_deref() else {
            continue;
        };
        const SUPPORTED: &[&str] = &["image/jpeg", "image/png", "image/gif", "image/webp"];
        if !SUPPORTED.contains(&ct) {
            warn!(
                "Discord image '{}' has unsupported MIME type '{}'; skipping",
                att.filename, ct
            );
            continue;
        }
        if (att.size as usize) > MAX_ATTACHMENT_BYTES {
            warn!(
                "Discord image '{}' is {} bytes (>5MB); skipping",
                att.filename, att.size
            );
            continue;
        }
        match att.download().await {
            Ok(bytes) if bytes.len() <= MAX_ATTACHMENT_BYTES => {
                out.push(Attachment {
                    media_type: ct.to_string(),
                    data: bytes,
                });
            }
            Ok(bytes) => warn!(
                "Discord image '{}' decoded to {} bytes (>5MB); skipping",
                att.filename,
                bytes.len()
            ),
            Err(e) => warn!(
                "Failed to download Discord attachment '{}': {e}",
                att.filename
            ),
        }
    }
    out
}

/// Split content into chunks that fit within Discord's 2000-character limit.
fn split_for_discord(content: &str) -> Vec<String> {
    const LIMIT: usize = 1990; // leave a small margin
    if content.len() <= LIMIT {
        return vec![content.to_owned()];
    }

    let mut chunks = Vec::new();
    let mut remaining = content;

    while remaining.len() > LIMIT {
        // Find a safe UTF-8 boundary at or before LIMIT.
        let mut split = LIMIT;
        while !remaining.is_char_boundary(split) {
            split -= 1;
        }
        // Prefer splitting at a newline to keep code blocks intact.
        if let Some(nl) = remaining[..split].rfind('\n') {
            split = nl + 1;
        }
        chunks.push(remaining[..split].to_owned());
        remaining = remaining[split..].trim_start_matches('\n');
    }

    if !remaining.is_empty() {
        chunks.push(remaining.to_owned());
    }

    chunks
}
