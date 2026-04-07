//! Heartbeat: periodic background tasks.
//!
//! Currently handles daily log generation at `day_boundary_hour` (local time).
//! Designed as a foundation for future HEARTBEAT / Cron-style periodic execution.

use crate::daily_log::generate_daily_log;
use crate::provider::Provider;
use crate::session::SessionStore;
use chrono::{Duration, Local, NaiveTime, Timelike};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration as StdDuration;
use tracing::{info, warn};

pub struct Heartbeat {
    pub day_boundary_hour: u8,
    pub session_store: Arc<SessionStore>,
    pub provider: Arc<dyn Provider>,
    pub workspace_dir: PathBuf,
}

impl Heartbeat {
    /// Run the heartbeat loop indefinitely.
    ///
    /// Sleeps until the next `day_boundary_hour:00:00` local time, then
    /// generates a daily log for the just-ended day, and repeats.
    pub async fn run(self) {
        loop {
            let sleep_dur = self.time_until_next_boundary();
            info!(
                "Heartbeat: next daily log in {:.0}s (boundary hour: {}:00 local)",
                sleep_dur.as_secs_f64(),
                self.day_boundary_hour
            );
            tokio::time::sleep(sleep_dur).await;

            // The day that just ended is "yesterday" relative to the boundary
            let yesterday = crate::session::local_date_for_timestamp(
                Local::now() - Duration::seconds(1),
                self.day_boundary_hour,
            );

            info!("Heartbeat: generating daily log for {yesterday}");
            if let Err(e) = generate_daily_log(
                &self.session_store,
                self.provider.as_ref(),
                &self.workspace_dir,
                yesterday,
                self.day_boundary_hour,
            )
            .await
            {
                warn!("Heartbeat: failed to generate daily log for {yesterday}: {e:#}");
            }
        }
    }

    /// Compute how long to sleep until the next boundary time.
    fn time_until_next_boundary(&self) -> StdDuration {
        let now = Local::now();
        let boundary = NaiveTime::from_hms_opt(self.day_boundary_hour as u32, 0, 0)
            .expect("valid hour 0–23");

        let now_time = now.time();
        let secs_until = if now_time < boundary {
            // Boundary is later today
            (boundary - now_time).num_seconds()
        } else {
            // Boundary already passed today; target is tomorrow
            let secs_today = now_time.num_seconds_from_midnight() as i64;
            let boundary_secs = boundary.num_seconds_from_midnight() as i64;
            86_400 - secs_today + boundary_secs
        };

        // Guard against edge cases (DST, clock jumps): minimum 1 second
        StdDuration::from_secs(secs_until.max(1) as u64)
    }
}
