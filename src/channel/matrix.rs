use crate::channel::{Channel, IncomingMessage, OutgoingMessage};
use crate::config::MatrixConfig;
use anyhow::{Context, Result};
use async_trait::async_trait;
use matrix_sdk::{
    Client, SessionMeta, SessionTokens,
    authentication::matrix::MatrixSession,
    config::SyncSettings,
    ruma::{
        OwnedEventId, OwnedRoomId, OwnedUserId,
        events::relation::Thread,
        events::room::message::{
            MessageType, OriginalSyncRoomMessageEvent, Relation, RoomMessageEventContent,
        },
    },
};
use std::collections::HashSet;
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::{OnceCell, mpsc};
use tracing::{debug, info, warn};

pub struct MatrixChannel {
    homeserver: String,
    access_token: String,
    user_id: String,
    device_id: String,
    room_id: String,
    allowed_users: HashSet<String>,
    recovery_key: Option<String>,
    state_dir: PathBuf,
    client: Arc<OnceCell<Client>>,
}

impl std::fmt::Debug for MatrixChannel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MatrixChannel")
            .field("homeserver", &self.homeserver)
            .field("user_id", &self.user_id)
            .field("room_id", &self.room_id)
            .finish_non_exhaustive()
    }
}

impl MatrixChannel {
    pub fn new(cfg: &MatrixConfig) -> Self {
        Self {
            homeserver: cfg.homeserver.clone(),
            access_token: cfg.access_token.clone(),
            user_id: cfg.user_id.clone(),
            device_id: cfg.device_id.clone(),
            room_id: cfg.room_id.clone(),
            allowed_users: cfg.allowed_users.iter().cloned().collect(),
            recovery_key: cfg.recovery_key.clone(),
            state_dir: cfg.resolved_state_dir(),
            client: Arc::new(OnceCell::new()),
        }
    }

    async fn get_or_init_client(&self) -> Result<&Client> {
        self.client
            .get_or_try_init(|| async { self.build_client().await })
            .await
    }

    async fn build_client(&self) -> Result<Client> {
        std::fs::create_dir_all(&self.state_dir)
            .with_context(|| format!("Failed to create state dir: {}", self.state_dir.display()))?;

        let client = Client::builder()
            .homeserver_url(&self.homeserver)
            .sqlite_store(&self.state_dir, None)
            .build()
            .await
            .context("Failed to build Matrix client")?;

        let user_id: OwnedUserId = self.user_id.parse().context("Invalid user_id in config")?;

        let session = MatrixSession {
            meta: SessionMeta {
                user_id,
                device_id: self.device_id.as_str().into(),
            },
            tokens: SessionTokens {
                access_token: self.access_token.clone(),
                refresh_token: None,
            },
        };
        client
            .restore_session(session)
            .await
            .context("Failed to restore Matrix session")?;

        if let Some(key) = &self.recovery_key {
            if let Err(e) = client.encryption().recovery().recover(key).await {
                warn!("E2EE recovery failed (key may already be active): {e}");
            } else {
                info!("E2EE recovery key applied successfully");
            }
        }

        info!(
            "Matrix client initialized for {} on {}",
            self.user_id, self.homeserver
        );
        Ok(client)
    }

    async fn get_room(&self, room_id_str: &str) -> Result<matrix_sdk::Room> {
        let client = self.get_or_init_client().await?;
        let room_id: OwnedRoomId = room_id_str.parse().context("Invalid room_id")?;
        client.get_room(&room_id).context("Room not found")
    }
}

#[async_trait]
impl Channel for MatrixChannel {
    fn name(&self) -> &str {
        "matrix"
    }

    async fn send(&self, message: &OutgoingMessage) -> Result<()> {
        let room = self.get_room(&message.room_id).await?;

        let mut content = RoomMessageEventContent::text_markdown(&message.content);

        if let Some(thread_id) = &message.thread_id {
            let thread_root: OwnedEventId = thread_id.parse().context("Invalid thread_id")?;
            content.relates_to = Some(Relation::Thread(Thread::plain(
                thread_root.clone(),
                thread_root,
            )));
        }

        room.send(content).await.context("Failed to send message")?;
        Ok(())
    }

    async fn listen(&self, tx: mpsc::Sender<IncomingMessage>) -> Result<()> {
        let client = self.get_or_init_client().await?;
        let room_id_str = self.room_id.clone();
        let allowed_users = self.allowed_users.clone();
        let bot_user_id = self.user_id.clone();

        client.add_event_handler({
            let tx = tx.clone();
            move |event: OriginalSyncRoomMessageEvent, room: matrix_sdk::Room| {
                let tx = tx.clone();
                let room_id_str = room_id_str.clone();
                let allowed_users = allowed_users.clone();
                let bot_user_id = bot_user_id.clone();

                async move {
                    if room.room_id().as_str() != room_id_str {
                        return;
                    }
                    if event.sender.as_str() == bot_user_id {
                        return;
                    }
                    if !allowed_users.is_empty() && !allowed_users.contains(event.sender.as_str()) {
                        debug!("Ignoring message from non-allowed user: {}", event.sender);
                        return;
                    }

                    let MessageType::Text(ref text_content) = event.content.msgtype else {
                        return;
                    };

                    let thread_id = event.content.relates_to.as_ref().and_then(|rel| {
                        if let Relation::Thread(thread) = rel {
                            Some(thread.event_id.to_string())
                        } else {
                            None
                        }
                    });

                    let msg = IncomingMessage {
                        id: event.event_id.to_string(),
                        sender: event.sender.to_string(),
                        content: text_content.body.clone(),
                        room_id: room_id_str,
                        timestamp: 0,
                        thread_id,
                    };

                    if let Err(e) = tx.send(msg).await {
                        warn!("Failed to forward Matrix message: {e}");
                    }
                }
            }
        });

        info!("Starting Matrix sync loop...");
        client
            .sync(SyncSettings::default().timeout(std::time::Duration::from_secs(30)))
            .await
            .context("Matrix sync loop exited with error")?;

        Ok(())
    }

    async fn start_typing(&self, room_id: &str) -> Result<()> {
        let room = self.get_room(room_id).await?;
        room.typing_notice(true)
            .await
            .context("Failed to send typing notice")?;
        Ok(())
    }

    async fn stop_typing(&self, room_id: &str) -> Result<()> {
        let room = self.get_room(room_id).await?;
        room.typing_notice(false)
            .await
            .context("Failed to stop typing notice")?;
        Ok(())
    }
}
