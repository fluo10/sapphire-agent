//! The `device` and `user` subcommands.
//!
//! `device add` is the only place in the agent that mints a token. It writes
//! two files that are not in the same place — the device row goes into the
//! workspace table, the key into the host-local key file — so the order
//! matters: the row goes first, because a row with no key is inert while an
//! orphan key is litter nobody sweeps. `add` is therefore resumable: it
//! finishes a row that has no key rather than dead-ending on the duplicate
//! name.

use std::path::Path;

use anyhow::{Context as _, Result, anyhow, bail};
use chrono::{DateTime, Duration, Utc};
use clap::Subcommand;
use sapphire_framework::registry::{Devices, Users};
use sapphire_framework::remote_server::KeyStore;

/// Prefix on tokens this agent mints (sapphire-agent token).
pub const TOKEN_PREFIX: &str = "sat";

#[derive(Subcommand, Debug)]
pub enum DeviceCommand {
    /// Register a device and mint the key it authenticates with.
    Add {
        #[arg(long, value_name = "DEVICE_NAME")]
        name: String,
        /// A note for you — what this device is.
        #[arg(long, value_name = "TEXT")]
        description: Option<String>,
        /// Whose device this is: a user id or name from users.toml.
        #[arg(long, value_name = "SELECTOR")]
        user: Option<String>,
        /// Expire the key after this long, e.g. `90d`, `12h`.
        #[arg(long, value_name = "DURATION")]
        expires_in: Option<String>,
    },
    /// List devices, their users, and whether they hold a key on this host.
    List,
    /// Re-issue a device's token, keeping its id and its row.
    ///
    /// `--expires-in` REPLACES the expiry rather than keeping it: omitting the
    /// flag makes the key non-expiring.
    Rotate {
        /// The device's id, or its name.
        selector: String,
        #[arg(long, value_name = "DURATION")]
        expires_in: Option<String>,
    },
    /// Stop a device: revoke its key, and mark the row retired.
    Retire {
        /// The device's id, or its name.
        selector: String,
        /// Delete the row outright instead of retiring it. Device ids get
        /// written into content (a journal entry's `updated_by`, say), so this
        /// makes those references unresolvable. Retiring is the default for
        /// that reason.
        #[arg(long)]
        purge: bool,
    },
}

#[derive(Subcommand, Debug)]
pub enum UserCommand {
    /// Register a user.
    Add {
        #[arg(long, value_name = "USER_NAME")]
        name: String,
        #[arg(long, value_name = "TEXT")]
        description: Option<String>,
    },
    /// List users.
    List,
}

/// Turn `90d` / `12h` / `30m` into a [`Duration`].
///
/// The unit is mandatory. A bare `90` is refused — nothing should have to guess
/// whether a credential lives for ninety seconds or ninety days.
///
/// `try_days` rather than `days`: the latter panics on out-of-range input, so a
/// typo in `--expires-in` would abort the process instead of erroring.
pub fn parse_duration(s: &str) -> Result<Duration> {
    let split = s
        .find(|c: char| !c.is_ascii_digit())
        .with_context(|| format!("duration needs a unit (d/h/m): {s:?}"))?;
    let (value, unit) = s.split_at(split);
    if value.is_empty() {
        bail!("duration must start with digits before the unit: {s:?}");
    }
    let n: i64 = value
        .parse()
        .with_context(|| format!("bad duration: {s:?}"))?;
    let d = match unit {
        "d" => Duration::try_days(n),
        "h" => Duration::try_hours(n),
        "m" => Duration::try_minutes(n),
        other => bail!("unknown duration unit {other:?} in {s:?} (use d, h or m)"),
    };
    d.ok_or_else(|| anyhow!("duration is out of range: {s:?}"))
}

/// Relative expiry to an absolute instant.
///
/// `checked_add_signed`, because `Utc::now() + d` panics on a time it cannot
/// represent — chrono's `Duration` range is much wider than `DateTime`'s, so a
/// value that survives `parse_duration` can still blow up here.
fn absolute_expiry(expires_in: Option<&str>) -> Result<Option<DateTime<Utc>>> {
    expires_in
        .map(parse_duration)
        .transpose()?
        .map(|d| {
            Utc::now()
                .checked_add_signed(d)
                .ok_or_else(|| anyhow!("expiry is too far in the future: {d}"))
        })
        .transpose()
}

pub fn run_device(
    command: DeviceCommand,
    devices_file: &Path,
    users_file: &Path,
    keys_file: &Path,
) -> Result<()> {
    let mut devices = Devices::load(devices_file)
        .with_context(|| format!("loading device table {}", devices_file.display()))?;
    let mut keys = KeyStore::load(keys_file)
        .with_context(|| format!("loading key file {}", keys_file.display()))?;

    match command {
        DeviceCommand::Add {
            name,
            description,
            user,
            expires_in,
        } => {
            // Resolve everything that can fail before writing anything.
            let expires_at = absolute_expiry(expires_in.as_deref())?;
            let user_id = match user {
                Some(selector) => {
                    let users = Users::load(users_file)
                        .with_context(|| format!("loading user table {}", users_file.display()))?;
                    Some(users.resolve(&selector)?.id)
                }
                None => None,
            };

            // Take an owned copy before the match: `resolve` borrows `devices`
            // immutably and the other arm needs it mutably, so holding the
            // reference across the match does not borrow-check.
            let existing = devices.resolve(&name).ok().cloned();
            let device = match existing {
                // A retired row keeps the name forever as far as `Devices::add`
                // is concerned, so without this check a retired device is a
                // silent dead end: this call would mint a token and report
                // success, but `DeviceAuth::resolve` rejects retired devices
                // and `DeviceAuth::open` exempts them from the room_profile
                // binding check, so every request against the new token 401s
                // and startup never complains either.
                Some(existing) if existing.is_retired() => {
                    bail!(
                        "device {name:?} is retired; minting it a new key would not help — a \
                         retired device is rejected by every authenticated endpoint regardless \
                         of whether it holds a live token. sapphire-agent cannot un-retire a \
                         device in place yet (there is no framework API to clear `retired_at` \
                         without reassigning the row's id), so either add the device under a \
                         different name, or accept a new id for it: `sapphire-agent device \
                         retire {name} --purge` followed by `sapphire-agent device add --name \
                         {name}` again"
                    );
                }
                // The row already exists, is not retired, and does not hold a
                // key. Either this is a resumed `add` whose key write did not
                // happen, or the name is genuinely taken.
                Some(existing) => {
                    if keys
                        .entries()
                        .iter()
                        .any(|k| k.device_id == Some(existing.id))
                    {
                        bail!(
                            "device {name:?} already exists and already holds a key on this \
                             host; use `sapphire-agent device rotate {name}` to re-issue its \
                             token"
                        );
                    }
                    // A row with no key was never usable: nothing could have
                    // authenticated as this device, so nothing could have
                    // written content carrying its id. That makes it safe to
                    // apply `--description`/`--user` here rather than
                    // silently dropping them as before — `Devices` exposes no
                    // in-place field update (only add/retire/purge), so the
                    // only way to change them is to purge the stray row and
                    // re-add it. The new row gets a fresh id, which is fine
                    // only because the old one was never handed to anyone.
                    if description.is_some() || user_id.is_some() {
                        devices.purge(&name)?;
                        devices.add(
                            &name,
                            description.or_else(|| existing.description.clone()),
                            user_id.or(existing.user_id),
                        )?
                    } else {
                        existing
                    }
                }
                None => devices.add(&name, description, user_id)?,
            };

            let entry = keys.generate(
                TOKEN_PREFIX,
                None,
                Some(device.id),
                Some(device.name.clone()),
                expires_at,
            )?;

            println!("{}", entry.token);
            eprintln!(
                "id {}  created {}{}",
                device.id,
                device.created_at.to_rfc3339(),
                entry
                    .expires_at
                    .map(|e| format!("  expires {}", e.to_rfc3339()))
                    .unwrap_or_default()
            );
            // Routing lives in config.toml, which this command does not touch,
            // so the config is invalid until the operator adds this line. Say
            // exactly what to paste rather than letting the next start-up
            // explain it.
            eprintln!(
                "\nnext: bind it to a room profile in your config.toml\n\n    \
                 [room_profile.<name>]\n    devices = [\"{}\"]\n",
                device.id
            );
        }
        DeviceCommand::List => {
            for d in devices.entries() {
                let has_key = keys.entries().iter().any(|k| k.device_id == Some(d.id));
                println!(
                    "{}  {}  {}  {}  {}  {}",
                    d.id,
                    d.name,
                    d.user_id
                        .map(|u| u.to_string())
                        .unwrap_or_else(|| "-".to_owned()),
                    if has_key { "key" } else { "no-key" },
                    if d.is_retired() { "retired" } else { "active" },
                    d.description.as_deref().unwrap_or("-"),
                );
            }
        }
        DeviceCommand::Rotate {
            selector,
            expires_in,
        } => {
            let expires_at = absolute_expiry(expires_in.as_deref())?;
            let device = devices.resolve(&selector)?.clone();
            if device.is_retired() {
                bail!(
                    "device {selector:?} ({}) is retired; rotating it would print a fresh \
                     token that authenticates to nothing — a retired device is rejected by \
                     every authenticated endpoint regardless of which token it holds. \
                     sapphire-agent cannot un-retire a device in place yet; add a replacement \
                     under a different name, or purge this row \
                     (`sapphire-agent device retire {selector} --purge`) and add it again",
                    device.name
                );
            }
            // Find the key by `device_id`, not by `device.name`: `add` sets
            // `label = device.name` at mint time, but `devices.toml` invites
            // hand-editing and nothing keeps a later rename in sync with the
            // key file's label, which is what `KeyStore`'s selector matches.
            // Passing the key's own id sidesteps the label entirely.
            let key_id = keys
                .entries()
                .iter()
                .find(|k| k.device_id == Some(device.id))
                .map(|k| k.id.to_string())
                .ok_or_else(|| {
                    anyhow!(
                        "device {selector:?} ({}) has no key on this host; use `sapphire-agent \
                         device add --name {}` to mint one instead of rotating a key that \
                         doesn't exist",
                        device.name,
                        device.name
                    )
                })?;
            let entry = keys.rotate(TOKEN_PREFIX, &key_id, expires_at)?;
            println!("{}", entry.token);
            eprintln!("rotated {} ({})", device.id, device.name);
            eprintln!("a running agent keeps accepting the old token until it restarts");
        }
        DeviceCommand::Retire { selector, purge } => {
            let device = devices.resolve(&selector)?.clone();
            // Same device_id-based lookup as `Rotate` above, and for the same
            // reason: a renamed device's key still carries the pre-rename
            // label, so resolving by name/label here would leave the key
            // live while reporting success.
            let key_id = keys
                .entries()
                .iter()
                .find(|k| k.device_id == Some(device.id))
                .map(|k| k.id.to_string());
            let had_key = key_id.is_some();
            // Revoke first: the point of retiring is to stop the device, and a
            // crash between the two writes must not leave a live key behind.
            if let Some(key_id) = key_id {
                keys.revoke(&key_id)?;
            }
            if purge {
                devices.purge(&selector)?;
                eprintln!("purged {} ({})", device.id, device.name);
            } else {
                devices.retire(&selector)?;
                eprintln!("retired {} ({})", device.id, device.name);
            }
            // Retiring exists to stop a device *now*; say so plainly rather
            // than letting the operator assume it already has, the way
            // `rotate` already warns about the same stale-snapshot gap.
            if had_key {
                eprintln!("a running agent keeps accepting the old token until it restarts");
            }
        }
    }
    Ok(())
}

pub fn run_user(command: UserCommand, users_file: &Path) -> Result<()> {
    let mut users = Users::load(users_file)
        .with_context(|| format!("loading user table {}", users_file.display()))?;
    match command {
        UserCommand::Add { name, description } => {
            let user = users.add(&name, description)?;
            println!("{}", user.id);
            eprintln!("added {} ({})", user.id, user.name);
        }
        UserCommand::List => {
            for u in users.entries() {
                println!(
                    "{}  {}  {}",
                    u.id,
                    u.name,
                    if u.is_retired() { "retired" } else { "active" }
                );
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;

    struct Files {
        _dir: tempfile::TempDir,
        devices: PathBuf,
        users: PathBuf,
        keys: PathBuf,
    }

    fn files() -> Files {
        let dir = tempfile::tempdir().unwrap();
        Files {
            devices: dir.path().join("devices.toml"),
            users: dir.path().join("users.toml"),
            keys: dir.path().join("keys.toml"),
            _dir: dir,
        }
    }

    fn add(f: &Files, name: &str) -> anyhow::Result<()> {
        run_device(
            DeviceCommand::Add {
                name: name.into(),
                description: None,
                user: None,
                expires_in: None,
            },
            &f.devices,
            &f.users,
            &f.keys,
        )
    }

    #[test]
    fn add_writes_both_the_device_row_and_a_key_bound_to_it() {
        let f = files();

        add(&f, "pendant").unwrap();

        let devices = Devices::load(&f.devices).unwrap();
        let device = devices.resolve("pendant").unwrap();
        let keys = KeyStore::load(&f.keys).unwrap();
        assert_eq!(keys.entries().len(), 1);
        let key = &keys.entries()[0];
        assert_eq!(
            key.device_id,
            Some(device.id),
            "the key must name the device"
        );
        assert!(key.token.starts_with("sat_"));
        assert_eq!(key.label.as_deref(), Some("pendant"));
    }

    /// `add` writes the device row first, so an interrupted run leaves an inert
    /// row rather than an orphan key nobody sweeps up. Re-running must finish
    /// the job instead of dead-ending on the duplicate name — otherwise there
    /// is no way out of the partial state (`rotate` needs an existing key).
    #[test]
    fn add_finishes_a_device_row_that_has_no_key_yet() {
        let f = files();
        let id = Devices::load(&f.devices)
            .unwrap()
            .add("pendant", None, None)
            .unwrap()
            .id;

        add(&f, "pendant").unwrap();

        let keys = KeyStore::load(&f.keys).unwrap();
        assert_eq!(keys.entries().len(), 1);
        assert_eq!(
            keys.entries()[0].device_id,
            Some(id),
            "reuses the existing row"
        );
        assert_eq!(Devices::load(&f.devices).unwrap().entries().len(), 1);
    }

    #[test]
    fn add_refuses_a_device_that_already_has_a_key() {
        let f = files();
        add(&f, "pendant").unwrap();

        let err = add(&f, "pendant").unwrap_err();

        let msg = format!("{err:#}");
        assert!(
            msg.contains("rotate"),
            "must point at the way forward: {msg}"
        );
    }

    #[test]
    fn add_binds_a_user_when_asked() {
        let f = files();
        run_user(
            UserCommand::Add {
                name: "fluo10".into(),
                description: None,
            },
            &f.users,
        )
        .unwrap();

        run_device(
            DeviceCommand::Add {
                name: "pendant".into(),
                description: Some("首から下げるやつ".into()),
                user: Some("fluo10".into()),
                expires_in: None,
            },
            &f.devices,
            &f.users,
            &f.keys,
        )
        .unwrap();

        let users = Users::load(&f.users).unwrap();
        let devices = Devices::load(&f.devices).unwrap();
        let device = devices.resolve("pendant").unwrap();
        assert_eq!(device.user_id, Some(users.resolve("fluo10").unwrap().id));
        assert_eq!(device.description.as_deref(), Some("首から下げるやつ"));
    }

    #[test]
    fn add_errors_on_an_unknown_user_without_writing_anything() {
        let f = files();

        let err = run_device(
            DeviceCommand::Add {
                name: "pendant".into(),
                description: None,
                user: Some("nobody".into()),
                expires_in: None,
            },
            &f.devices,
            &f.users,
            &f.keys,
        )
        .unwrap_err();

        assert!(format!("{err:#}").contains("nobody"));
        assert!(
            Devices::load(&f.devices).unwrap().entries().is_empty(),
            "the user is resolved before anything is written"
        );
    }

    #[test]
    fn add_turns_expires_in_into_an_absolute_time() {
        let f = files();

        run_device(
            DeviceCommand::Add {
                name: "pendant".into(),
                description: None,
                user: None,
                expires_in: Some("90d".into()),
            },
            &f.devices,
            &f.users,
            &f.keys,
        )
        .unwrap();

        let keys = KeyStore::load(&f.keys).unwrap();
        let expires = keys.entries()[0]
            .expires_at
            .expect("an expiry was asked for");
        let expected = chrono::Utc::now() + chrono::Duration::days(90);
        assert!((expires - expected).num_seconds().abs() < 5);
    }

    #[test]
    fn add_errors_instead_of_panicking_on_an_absurd_expiry() {
        let f = files();

        let result = run_device(
            DeviceCommand::Add {
                name: "pendant".into(),
                description: None,
                user: None,
                expires_in: Some("99999999999d".into()),
            },
            &f.devices,
            &f.users,
            &f.keys,
        );

        assert!(result.is_err());
    }

    #[test]
    fn rotate_replaces_the_token_and_keeps_the_device() {
        let f = files();
        add(&f, "pendant").unwrap();
        let before = KeyStore::load(&f.keys).unwrap().entries()[0].clone();

        run_device(
            DeviceCommand::Rotate {
                selector: "pendant".into(),
                expires_in: None,
            },
            &f.devices,
            &f.users,
            &f.keys,
        )
        .unwrap();

        let after = &KeyStore::load(&f.keys).unwrap().entries()[0].clone();
        assert_ne!(after.token, before.token);
        assert_eq!(after.device_id, before.device_id);
    }

    #[test]
    fn retire_marks_the_device_and_revokes_its_key() {
        let f = files();
        add(&f, "pendant").unwrap();

        run_device(
            DeviceCommand::Retire {
                selector: "pendant".into(),
                purge: false,
            },
            &f.devices,
            &f.users,
            &f.keys,
        )
        .unwrap();

        let devices = Devices::load(&f.devices).unwrap();
        // The row stays: device ids get written into content elsewhere.
        assert!(devices.resolve("pendant").unwrap().is_retired());
        // The key does not: retiring a device must actually stop it.
        assert!(KeyStore::load(&f.keys).unwrap().entries().is_empty());
    }

    #[test]
    fn retire_with_purge_removes_the_row_too() {
        let f = files();
        add(&f, "pendant").unwrap();

        run_device(
            DeviceCommand::Retire {
                selector: "pendant".into(),
                purge: true,
            },
            &f.devices,
            &f.users,
            &f.keys,
        )
        .unwrap();

        assert!(Devices::load(&f.devices).unwrap().entries().is_empty());
    }

    /// `retire`'s non-purging default keeps the row and revokes its key —
    /// exactly the "row exists, no key" state that a resumed `add` treats as
    /// a crashed add. Without this check, `add` on a retired name mints a
    /// token and reports success, but `DeviceAuth` rejects the device on
    /// both sides (`resolve` for the auth check, `open` for the routing
    /// check), so every request against the fresh token 401s silently.
    #[test]
    fn add_refuses_a_retired_device() {
        let f = files();
        add(&f, "pendant").unwrap();
        run_device(
            DeviceCommand::Retire {
                selector: "pendant".into(),
                purge: false,
            },
            &f.devices,
            &f.users,
            &f.keys,
        )
        .unwrap();

        let err = add(&f, "pendant").unwrap_err();

        let msg = format!("{err:#}");
        assert!(msg.contains("retired"), "must say why: {msg}");
        // No key must have been minted for the still-retired row.
        assert_eq!(KeyStore::load(&f.keys).unwrap().entries().len(), 0);
    }

    /// Same hole as `add`, by a different door: `rotate` would otherwise
    /// happily print a fresh token for a device that every authenticated
    /// endpoint still rejects.
    #[test]
    fn rotate_refuses_a_retired_device() {
        let f = files();
        add(&f, "pendant").unwrap();
        run_device(
            DeviceCommand::Retire {
                selector: "pendant".into(),
                purge: false,
            },
            &f.devices,
            &f.users,
            &f.keys,
        )
        .unwrap();

        let err = run_device(
            DeviceCommand::Rotate {
                selector: "pendant".into(),
                expires_in: None,
            },
            &f.devices,
            &f.users,
            &f.keys,
        )
        .unwrap_err();

        assert!(format!("{err:#}").contains("retired"));
    }

    /// A resumed `add` (row exists, no key yet) used to silently drop
    /// `--description`/`--user` on the floor — there is no `device set-user`
    /// or `device describe` to recover them afterwards. They must land on
    /// the row instead.
    #[test]
    fn resumed_add_applies_description_and_user() {
        let f = files();
        run_user(
            UserCommand::Add {
                name: "fluo10".into(),
                description: None,
            },
            &f.users,
        )
        .unwrap();
        // Simulate the crash: a row with no key, as `device add` leaves
        // behind when it dies between writing the row and minting the key.
        Devices::load(&f.devices)
            .unwrap()
            .add("pendant", None, None)
            .unwrap();

        run_device(
            DeviceCommand::Add {
                name: "pendant".into(),
                description: Some("首から下げるやつ".into()),
                user: Some("fluo10".into()),
                expires_in: None,
            },
            &f.devices,
            &f.users,
            &f.keys,
        )
        .unwrap();

        let users = Users::load(&f.users).unwrap();
        let devices = Devices::load(&f.devices).unwrap();
        let device = devices.resolve("pendant").unwrap();
        assert_eq!(device.description.as_deref(), Some("首から下げるやつ"));
        assert_eq!(device.user_id, Some(users.resolve("fluo10").unwrap().id));
        // Exactly one row and one key survive the resume.
        assert_eq!(devices.entries().len(), 1);
        let keys = KeyStore::load(&f.keys).unwrap();
        assert_eq!(keys.entries().len(), 1);
        assert_eq!(keys.entries()[0].device_id, Some(device.id));
    }

    /// `retire` and `rotate` used to resolve the key by device *name*, while
    /// `add`/`list` key by `device_id`. After a rename, the key file's
    /// `label` (set once, at mint time, to the device's name-at-the-time)
    /// no longer matches — `retire` would fail with "no key matches" before
    /// `devices.retire` even ran (device neither stopped nor retired, key
    /// still live), and `rotate` would fail the same way with no argument
    /// left that could reach the key. Resolving by `device_id` instead fixes
    /// both.
    #[test]
    fn retire_finds_the_key_after_the_device_was_renamed() {
        let f = files();
        add(&f, "pendant").unwrap();
        rename_device(&f.devices, "pendant", "lanyard");

        run_device(
            DeviceCommand::Retire {
                selector: "lanyard".into(),
                purge: false,
            },
            &f.devices,
            &f.users,
            &f.keys,
        )
        .unwrap();

        assert!(Devices::load(&f.devices).unwrap().resolve("lanyard").unwrap().is_retired());
        assert!(
            KeyStore::load(&f.keys).unwrap().entries().is_empty(),
            "the key must be revoked even though its label still says the old name"
        );
    }

    #[test]
    fn rotate_finds_the_key_after_the_device_was_renamed() {
        let f = files();
        add(&f, "pendant").unwrap();
        rename_device(&f.devices, "pendant", "lanyard");
        let before = KeyStore::load(&f.keys).unwrap().entries()[0].clone();

        run_device(
            DeviceCommand::Rotate {
                selector: "lanyard".into(),
                expires_in: None,
            },
            &f.devices,
            &f.users,
            &f.keys,
        )
        .unwrap();

        let after = KeyStore::load(&f.keys).unwrap().entries()[0].clone();
        assert_eq!(after.id, before.id, "same key, not a new one");
        assert_ne!(after.token, before.token);
    }

    /// Hand-edit `devices.toml` to rename a device without touching its id —
    /// exactly the operation the file's own header documents as supported
    /// ("Hand-editing is fine"). `Devices` has no rename method of its own.
    fn rename_device(devices_file: &Path, old_name: &str, new_name: &str) {
        let text = std::fs::read_to_string(devices_file).unwrap();
        let renamed = text.replacen(
            &format!("name = \"{old_name}\""),
            &format!("name = \"{new_name}\""),
            1,
        );
        assert_ne!(text, renamed, "the rename must actually match something");
        std::fs::write(devices_file, renamed).unwrap();
    }

    #[test]
    fn user_add_rejects_a_duplicate_name() {
        let f = files();
        run_user(
            UserCommand::Add {
                name: "fluo10".into(),
                description: None,
            },
            &f.users,
        )
        .unwrap();

        let err = run_user(
            UserCommand::Add {
                name: "fluo10".into(),
                description: None,
            },
            &f.users,
        )
        .unwrap_err();

        assert!(format!("{err:#}").contains("fluo10"));
    }

    #[test]
    fn parse_duration_accepts_days_hours_and_minutes() {
        assert_eq!(parse_duration("90d").unwrap(), chrono::Duration::days(90));
        assert_eq!(parse_duration("12h").unwrap(), chrono::Duration::hours(12));
        assert_eq!(
            parse_duration("30m").unwrap(),
            chrono::Duration::minutes(30)
        );
    }

    #[test]
    fn parse_duration_rejects_junk() {
        // A unit is mandatory: `90` could be seconds or days, and neither
        // reading is safe to guess for a credential's lifetime.
        for s in ["", "90", "d90", "-1d", "90y", "1000000000000d"] {
            assert!(parse_duration(s).is_err(), "{s} passed");
        }
    }
}
