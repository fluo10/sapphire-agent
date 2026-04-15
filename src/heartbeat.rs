//! Heartbeat: periodic background tasks.
//!
//! Two parallel loops:
//!
//! 1. **Day-boundary loop**: at every `day_boundary_hour:00:00` local time,
//!    optionally generates a daily log and compacts MEMORY.md.
//! 2. **Cron loop**: scans `<workspace>/heartbeat/*.md` and fires `prompt`
//!    triggers on the agent according to each task's cron schedule.

use crate::agent::Agent;
use crate::daily_log::{catchup_pending_logs, generate_daily_log};
use crate::heartbeat_config::{load_heartbeat_dir, next_due};
use crate::memory_compaction::compact_memory;
use crate::provider::Provider;
use crate::session::SessionStore;
use chrono::{Duration, Local, NaiveTime, Timelike};
use sapphire_workspace::WorkspaceState;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration as StdDuration;
use tracing::{info, warn};

/// How often the catchup loop scans for missing daily logs.
const CATCHUP_INTERVAL: StdDuration = StdDuration::from_secs(60 * 60);

pub struct Heartbeat {
    pub workspace_dir: PathBuf,
    pub ws_state: Arc<Mutex<WorkspaceState>>,
    pub day_boundary_hour: u8,
    pub daily_log_enabled: bool,
    pub memory_compaction_enabled: bool,
    pub session_store: Arc<SessionStore>,
    pub provider: Arc<dyn Provider>,
    pub agent: Arc<Agent>,
    pub default_room_id: Option<String>,
}

impl Heartbeat {
    /// Spawn the day-boundary, cron, and catchup loops as independent tasks.
    pub fn spawn(self) {
        let me = Arc::new(self);
        let a = Arc::clone(&me);
        tokio::spawn(async move { a.run_day_boundary().await });
        let b = Arc::clone(&me);
        tokio::spawn(async move { b.run_cron().await });
        if me.daily_log_enabled {
            let c = Arc::clone(&me);
            tokio::spawn(async move { c.run_catchup().await });
        }
    }

    /// Periodically scan for past dates that have session messages but no
    /// daily log file, and generate any that are missing. Recovers from
    /// transient failures during the day-boundary fire (e.g. network blip
    /// during the LLM call) without waiting for the next process restart.
    async fn run_catchup(self: Arc<Self>) {
        let mut tick = tokio::time::interval(CATCHUP_INTERVAL);
        tick.tick().await; // skip immediate fire — startup catchup already ran in main
        loop {
            tick.tick().await;
            catchup_pending_logs(
                &self.session_store,
                self.provider.as_ref(),
                &self.ws_state,
                &self.workspace_dir,
                self.day_boundary_hour,
            )
            .await;
        }
    }

    async fn run_day_boundary(self: Arc<Self>) {
        loop {
            let sleep_dur = self.time_until_next_boundary();
            info!(
                "Heartbeat: next day-boundary in {:.0}s (boundary hour: {}:00 local)",
                sleep_dur.as_secs_f64(),
                self.day_boundary_hour
            );
            tokio::time::sleep(sleep_dur).await;

            let yesterday = crate::session::local_date_for_timestamp(
                Local::now() - Duration::seconds(1),
                self.day_boundary_hour,
            );

            if self.daily_log_enabled {
                info!("Heartbeat: generating daily log for {yesterday}");
                if let Err(e) = generate_daily_log(
                    &self.session_store,
                    self.provider.as_ref(),
                    &self.ws_state,
                    yesterday,
                    self.day_boundary_hour,
                )
                .await
                {
                    warn!("Heartbeat: failed to generate daily log for {yesterday}: {e:#}");
                }
            }

            if self.memory_compaction_enabled {
                info!("Heartbeat: compacting MEMORY.md");
                compact_memory(self.provider.as_ref(), &self.workspace_dir).await;
            }
        }
    }

    async fn run_cron(self: Arc<Self>) {
        let dir = self.workspace_dir.join("heartbeat");
        loop {
            // Re-read the directory every iteration so edits take effect.
            let tasks = load_heartbeat_dir(&dir);
            let enabled: Vec<_> = tasks.into_iter().filter(|t| t.meta.enabled).collect();

            let now = Local::now();
            let (next_at, due_names): (
                chrono::DateTime<Local>,
                Vec<(String, String, Option<String>)>,
            ) = match next_due(&enabled, now) {
                None => {
                    // No tasks (or none with valid schedules); poll the
                    // directory periodically in case files are added.
                    tokio::time::sleep(StdDuration::from_secs(60)).await;
                    continue;
                }
                Some((at, due)) => {
                    let extracted = due
                        .into_iter()
                        .map(|t| (t.name.clone(), t.body.clone(), t.meta.room_id.clone()))
                        .collect();
                    (at, extracted)
                }
            };

            let wait = (next_at - now)
                .to_std()
                .unwrap_or(StdDuration::from_secs(0));
            info!(
                "Heartbeat cron: next fire in {:.0}s ({} task(s))",
                wait.as_secs_f64(),
                due_names.len()
            );
            tokio::time::sleep(wait).await;

            for (name, body, task_room) in due_names {
                let room = task_room.or_else(|| self.default_room_id.clone());
                match room {
                    Some(room) => {
                        info!("Heartbeat cron: firing task {name} -> {room}");
                        if let Err(e) = self.agent.trigger(&name, &body, &room).await {
                            warn!("Heartbeat cron: task {name} failed: {e:#}");
                        }
                    }
                    None => {
                        warn!(
                            "Heartbeat cron: task {name} has no room_id and no default; skipping"
                        );
                    }
                }
            }
        }
    }

    /// Compute how long to sleep until the next day-boundary time.
    fn time_until_next_boundary(&self) -> StdDuration {
        let now = Local::now();
        let boundary =
            NaiveTime::from_hms_opt(self.day_boundary_hour as u32, 0, 0).expect("valid hour 0–23");

        let now_time = now.time();
        let secs_until = if now_time < boundary {
            (boundary - now_time).num_seconds()
        } else {
            let secs_today = now_time.num_seconds_from_midnight() as i64;
            let boundary_secs = boundary.num_seconds_from_midnight() as i64;
            86_400 - secs_today + boundary_secs
        };

        StdDuration::from_secs(secs_until.max(1) as u64)
    }
}
