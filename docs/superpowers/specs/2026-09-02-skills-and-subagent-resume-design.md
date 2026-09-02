# Skills and subagent resume — design

**Goal:** run [obra/superpowers](https://github.com/obra/superpowers) inside
`sapphire-agent`, so that spec-driven development works the same way through Zed
over ACP as it does through Claude Code today.

**Status:** approved in brainstorming, 2026-09-02.

---

## Why this is not a plugin system, and not a fork

The request arrived as a fork in the road — implement a plugin system, or
hardcode superpowers as a built-in — with the concern that hardcoding buys a
standing obligation to track upstream. That concern is correct, and both
branches turn out to be avoidable.

**superpowers already targets hosts that are not Claude Code.** The 6.3.0
distribution ships `.codex-plugin/`, `.cursor-plugin/`, `.devin-plugin/`,
`.kimi-plugin/` and `gemini-extension.json` alongside `.claude-plugin/`, and
`skills/using-superpowers/references/` carries adaptation notes for Codex, Pi,
Antigravity, Hermes and Gemini. The host contract the Codex manifest declares is
two keys:

```json
"skills": "./skills/",
"hooks": {}
```

The Hermes reference spells out the fallback for a host with no skill-loading
machinery at all: read `SKILL.md` directly, and it names this "the same
mechanism used by other harnesses without native skill loading." A host is
expected to be able to participate with a directory and a file read.

**The coupling to Claude Code's tool surface is close to zero.** Across all 14
skills, the strings "Read tool", "Edit tool", "Write tool" and "Skill tool"
appear **0** times; "Task tool" appears once and "TodoWrite" twice. Skills speak
in actions — "dispatch a subagent", "read a file" — which is precisely why
per-harness mapping tables are only a page long. The executable payload is six
bash scripts; Node appears only in brainstorming's optional visual companion and
in `writing-skills`' graph rendering, neither of which SDD needs.

**The receiving shape already exists here.** `<workspace>/agents/<name>.md` —
YAML frontmatter plus body, loaded by `src/agents.rs`, surfaced through a tool
whose description enumerates what is available — is the same shape a skill has.
`load_agents_dir` is most of a `load_skills_dir`; the difference is that a skill
owns a directory (`<name>/SKILL.md` plus siblings) rather than a single file.

**And the cost of hardcoding is measurable.** superpowers is 94 Markdown files
and 1.2 MB, and it moved five times in eight weeks (6.0.3 on 2026-06-18, 6.1.0
on 06-30, 6.1.1 on 07-02, 6.2.0 on 07-23, 6.3.0 on 08-12). Vendoring an
MIT-licensed project on that cadence into an unrelated repository is a permanent
merge debt for no benefit.

So this spec implements **one generic mechanism — a skills directory** — which
is the contract superpowers' own non-Claude manifests ask for. Updating becomes
`git pull` in a checkout. Nothing about superpowers is named in the code.

---

## Decisions taken

| Question | Decision |
|---|---|
| Where does the checkout live? | **Client side.** The SDD scripts (`review-package`, `task-brief`, `sdd-workspace`) run `git` against the repository under review, and `find-polluter.sh` runs its test suite. They must execute where the repository is, which is the editor's machine. |
| How does the agent find it? | **A convention resolved on the client**, with a client-side environment override. The server stores no path: it serves clients on several machines and operating systems, so a path in its config is right for at most one of them. |
| How does anything get into it? | **`skill_install` / `skill_uninstall`.** A `data_dir` is managed by applications, not by hand, so the agent populates it on request. Sources are restricted to `https://` and every install is a permissioned action. |
| Who may see skills? | **Per memory namespace.** A development namespace opts in; the everyday one does not. This avoids writing "development means ACP" into the code — an overstatement made once already in the subagents spec and corrected during review. |
| Scope of this spec | **Skills, plus subagent resume.** SDD's fix rounds 1-3 resume a specific implementer; without that the loop silently degrades to a fresh implementer every round. |
| Where do resumed children live? | **A cache**, alongside `digest_cache` and `tool_result_cache`. |
| Over the cache's size cap? | **Refuse to persist and say so** in the handle line, rather than truncating a history or writing without limit. |

---

## Part 1 — Skills

### Configuration

**The server holds no client path.** It cannot: one agent serves clients on
different machines and different operating systems, so any absolute path in its
own config is right for at most one of them and silently wrong for the rest.
Server-side configuration is a single switch:

```toml
[memory_namespace.dev]
skills = true          # default false

[subagent_cache]
# Optional. Largest history that will be persisted for resume.
max_history_bytes = 8_388_608   # 8 MiB
retain_days = 7
```

Enablement hangs off `MemoryNamespaceConfig`, which `resolve_namespace_chain`
already computes per turn. A namespace with `skills = true` offers the tool; the
rest do not. `visible_tool_predicate` takes only booleans today, so it gains one
more — resolved from the turn's namespace by the caller, in the same place the
client capabilities are resolved.

### Where the skills directory is, without the server knowing

The location is a **convention resolved on the client**, mirroring the
`directories::BaseDirs::data_dir()` semantics `init_app_ctx` already uses for
this crate's own directories on the server:

| Platform | Directory |
|---|---|
| Windows | `%APPDATA%\sapphire-agent\skills` |
| macOS | `~/Library/Application Support/sapphire-agent/skills` |
| Linux / BSD | `${XDG_DATA_HOME:-~/.local/share}/sapphire-agent/skills` |

A checkout is data, not configuration, which is why this follows `data_dir`
rather than `config_dir`.

Candidates are tried **in order, first hit wins** — the idiom
`Workspace::read_first_existing` already applies to `AGENTS.md` / `AGENT.md`:

1. `$SAPPHIRE_AGENT_SKILLS_DIR`, if set
2. `$APPDATA/sapphire-agent/skills`
3. `$HOME/Library/Application Support/sapphire-agent/skills`
4. `${XDG_DATA_HOME:-$HOME/.local/share}/sapphire-agent/skills`

The environment variable is the escape hatch for a checkout that already lives
somewhere else. It is set on the client, by the person whose machine it is, in
whatever form that OS uses — which is the only place that question can be
answered correctly.

The resolution happens in the same client-side invocation that builds the index,
so the agent learns the path by being told it, never by assuming it. The
resolved directory is cached for the session and reused for every subsequent
`skill(name)`. A `skill(name)` that arrives before any index call resolves
first.

### Installing and removing skills

A `data_dir` is a place applications manage, not a place a person conveniently
`cd`s into and clones a repository. Resolving the convention without also being
able to populate it would leave the one manual step in the least comfortable
possible location, so installation is a tool the model can be asked to run.

Three tools, split by what they do rather than for symmetry:

- **`skill_install(url)`** — `ToolKind::Execute`. Clones into the skills
  directory, creating it first if no candidate exists.
- **`skill_update(name?)`** — `ToolKind::Execute`. `git pull --ff-only` on one
  installed entry, or on every one when the name is omitted. Reports which
  entries moved and to what.
- **`skill_uninstall(name)`** — `ToolKind::Delete`. Removes one installed entry.

**Update is addressed by name, not by URL.** Folding it into `skill_install`
would have meant that keeping skills current required remembering the URL each
entry came from, which is exactly the friction that makes people stop updating.
Tracking upstream is the reason this design does not vendor superpowers, so the
update path has to be the cheapest one in the set: "update my skills" with no
argument at all.

`--ff-only` is deliberate. A checkout that has diverged — most likely because
someone edited a skill in place — fails the fast-forward and says so, rather
than producing a merge nobody asked for.

**The stored remote is re-validated before the pull.** `skill_install` only ever
writes an `https` remote, but `.git/config` is an ordinary file on the person's
machine, and `git pull` against an `ext::` remote executes a command. Checking
the resolved remote against the same rules the install applies costs nothing and
removes the asymmetry.

All three are gated by the same namespace switch as `skill`, and all three go
through the ordinary permission path — `Execute` is asked about in `Default` and
`AcceptEdits`, so neither an install nor an update is silent outside `Bypass`.

Installation is `git`, not a tarball download: `git` is already required by the
SDD scripts, and it is what makes an update a `pull`.

#### What the URL is allowed to be

The model chooses this string, so `git clone` is an arbitrary-code delivery path
unless it is constrained. Only `https://` is accepted. Rejected explicitly:

| Rejected | Why |
|---|---|
| `ext::…` | `git` treats it as a command to execute. Remote code execution by design. |
| `file://`, `git://`, `ssh://`, `user@host:path` | Not `https`; the scp-like form is easy to mistake for a path. |
| Anything beginning with `-` | It is an argument, not a URL — `--upload-pack=…` is the classic. |

The URL is passed after a `--` separator so it cannot be reparsed as an option
even if a check is ever missed.

#### Where it lands

The destination directory name is **derived by the agent** from the URL's final
path segment, never taken from the model. It is then constrained to
`[A-Za-z0-9._-]`, rejected if empty, if it is `.` or `..`, if it begins with
`-`, or if it matches a Windows reserved device name (`CON`, `PRN`, `AUX`,
`NUL`, `COM1`-`COM9`, `LPT1`-`LPT9`).

That last clause is here because ruling that an alphanumeric allow-list is
sufficient was wrong once already, during the client-side tools work: `CON` is
alphanumeric.

`skill_uninstall` takes a name rather than a path, and applies the same rules
before it is joined to the resolved directory, so a removal cannot address
anything that is not a direct child of the skills directory.

**A checkout with uncommitted changes is not removed.** If `git status
--porcelain` is non-empty, uninstall refuses and says so; the person may have
edited a skill in place. An explicit `force` overrides.

#### The two layouts the index accepts

A checkout is not itself a skill. superpowers declares `"skills": "./skills/"`,
so its skills are one level deeper than the clone root. The index therefore
walks two shapes:

- `<dir>/<name>/SKILL.md` — a skill directory placed there directly
- `<dir>/<repo>/skills/<name>/SKILL.md` — a cloned bundle

This accepts both without parsing anyone's plugin manifest, and it means a
hand-written local skill can sit beside an installed bundle.

### The `skill` tool

One tool, two modes.

- `skill()` — no argument. Returns the index: every skill's `name` and
  `description`, which is 2.9 KB across the 14 skills shipped today.
- `skill(name)` — returns that skill's `SKILL.md` body, prefixed with the
  skill's **absolute directory** (see below).

This mirrors `subagent`, which already lets the model choose from descriptions
and then fetch one by name. `ToolKind::Read`.

**The response header matters.** Skills reference siblings by relative path —
`./implementer-prompt.md`, `references/codex-tools.md`, `scripts/task-brief`. The
model can only resolve those if it knows where the skill lives, so `skill(name)`
prefixes the body with the skill's absolute directory on the editor's machine.
Without that header every sibling reference in every skill is dead.

### How it reads

Bodies go through ACP `fs/read_text_file`. **If that fails, retry through the
client terminal with `cat`.** This is not belt-and-braces: an editor may scope
`fs/read_text_file` to the open project, and the skills checkout is deliberately
outside it. The terminal has no such scoping, so the fallback is the path that
works in the case we expect to hit.

The index is different, because **ACP has no directory listing** — no list, glob
or stat exists in the agent→client surface. So the index comes from one terminal
invocation of a fixed shell script that walks the candidate list above, prints
the directory it settled on, and then prints each `SKILL.md`'s `name` and
`description`.

**That script is a compile-time constant with nothing interpolated into it.**
The candidates are environment expansions the client's own shell performs; the
agent contributes no string to the command line at all. Building shell commands
by concatenation is how the workspace path guard was defeated four times, so
this design removes the possibility rather than guarding it.

**This makes `skill` depend on the client's terminal capability.** That is not a
new dependency being introduced — superpowers' six scripts all run under bash,
so a client without a shell cannot run SDD at all. The design states an existing
requirement rather than adding one.

There is deliberately **no configurable index command**. A shell snippet in
server-side config would be both a fresh injection surface and a second way to
put a client-shaped value on the server, which is the mistake this section
exists to undo. A client that cannot run the fixed script cannot run superpowers
either.

### Where the discipline goes

`using-superpowers` requires the model to check for a relevant skill *before*
responding. That instruction lives in the **`skill` tool's own description**, not
in `TOOLS.md`.

The reason is the gate. `TOOLS.md` is one file for the whole workspace and is
injected into every namespace's system prompt; a tool's description reaches only
the conversations where the tool is offered. Putting the discipline in the
description makes it follow the namespace gate for free — and keeps
"invoke a skill before you answer" out of an ordinary evening conversation.

The action-to-tool mapping — the equivalent of `references/hermes-tools.md` —
does go in `TOOLS.md`, which `WORKSPACE_FILES` already injects under `# Tools`.
It is inert where skills are off, so it needs no gate. No new mechanism is
required for it.

### Failure modes

| Condition | Behaviour |
|---|---|
| The namespace has not opted in | The tool is not offered. Same path as the gate. |
| No candidate directory exists on the client | Recoverable error naming the candidates tried, the `$SAPPHIRE_AGENT_SKILLS_DIR` override, and `skill_install` — which is the fix in the common case of a machine that has never had skills installed. |
| The directory resolves but is empty | Same, pointing at `skill_install`. |
| `skill_install` given a non-`https` or option-shaped URL | Refused before any process starts, naming which rule it broke. |
| `skill_install` on an already-installed URL | Refused, naming the entry and pointing at `skill_update`. |
| `skill_update` on a diverged or locally-edited checkout | The fast-forward fails; reported per entry, and with no name given the other entries still update. |
| `skill_update` where the stored remote is not `https` | Refused for that entry, naming the remote. |
| `skill_update` with no name and nothing installed | Recoverable error pointing at `skill_install`. |
| `skill_uninstall` on a checkout with local modifications | Refused, naming the modified files. `force` overrides. |
| `skill_uninstall` on an unknown name | Recoverable error listing what is installed. |
| `fs/read_text_file` refuses the path | Retry through the terminal; only then report. |
| Unknown skill name | Recoverable error listing the known names — the convention `subagent` already follows. |
| No client (a non-ACP transport in an enabled namespace) | Not offered. Skills are client-side by construction. |

### Testing

`AcpClient` has a `FakeClient` in `src/tools/acp_client.rs::tests`, so the whole
surface closes in unit tests: parsing the resolver's output into a directory
plus an index, the absolute-directory header, the `fs`-to-terminal fallback,
unknown skill names, a client where no candidate directory exists, reuse of the
resolved directory across calls in one session, and the gate on and off.

The candidate list itself is shell running on someone else's machine, so it is
covered by an executable test of the script against a fixture tree rather than
by asserting on Rust that never evaluates it.

Install and uninstall are mostly guard code, and the guards are what get tested
directly: every rejected URL form from the table above; a derived destination
name for ordinary and awkward URLs (trailing slash, `.git` suffix, a repo named
`con`); a name containing a separator or `..` refused before any join; uninstall
refused on a dirty checkout and permitted under `force`; and the index finding
skills through both accepted layouts, including one of each side by side.

---

## Part 2 — Subagent resume

Today `subagent` is one-shot: `{agent, prompt}` in, an answer out, no child kept.
SDD's fix loop wants "rounds 1-3 resume the implementer; rounds 4-5 dispatch a
fresh one on a more capable model." Without resume, every round is a fresh
implementer that has to rediscover its own task.

### The handle

Dispatch returns the child's answer prefixed with a short opaque handle. SDD
already expects exactly this: *"Record the implementer's agent identity from the
dispatch result — fix-loop rounds 1-3 resume this agent."*

### Schema

`agent` and `resume` are mutually exclusive.

- `subagent(agent, prompt)` — new child.
- `subagent(resume, prompt)` — continue an existing one.
- Neither, or both, is a recoverable error.

### Storage

A new `src/subagent_cache.rs`, sibling to `digest_cache.rs` and
`tool_result_cache.rs`: outside the workspace, one file per handle, overwritten
in place. It holds the agent's *name*, the child's `Vec<ChatMessage>`, and
timestamps. `ChatMessage` already derives `Serialize`/`Deserialize`, so no new
type work is needed.

It is a cache and not the session store for the reason the digests are:
`<workspace>/sessions` is in the retrieve index, and a subagent's full internal
transcript is the single most effective way to skew a search over it. It also
keeps the promise made in the subagents spec — that a subagent's conversation
reaches neither the parent's history nor the store — while making resume
possible.

Being on disk, resume survives a restart. That is a side effect rather than a
goal, but a welcome one given how often this agent is redeployed.

### The definition is re-read on resume

Only the agent's *name* is stored. The definition is reloaded from
`<workspace>/agents/` at resume time.

This picks up edits to the `.md`, but the load-bearing reason is different: the
offered-tool list is **recomputed**, so the depth cap added in #202 cannot be
smuggled through a stale stored list. Restoring a child's tool set from disk
would make resume the hole that the offer gate exists to close.

If the definition has since been deleted or renamed, resume fails with a
recoverable error.

### Size

A child's history will not be described as "bounded by construction." That claim
was made once in the subagents spec and was wrong. `MAX_TOOL_ROUNDS` is 10 and a
tool result is capped near 50 KB, which bounds a history to a few megabytes in
practice — but the number of tool calls *within* a round is chosen by the model
and executed concurrently through `join_all`, so there is no structural ceiling.

So there is an explicit cap — `max_history_bytes`, 8 MiB by default, measured on
the serialized history. **Over it, the history is not persisted and the handle
line says the child is not resumable.** The child's answer is returned normally,
so the failure costs the fix loop a resume, never an answer.

Truncating instead was rejected: dropping old messages can leave a `tool_use`
whose `tool_result` is gone, which makes the history unloadable. That is the
exact invariant Task 1 of the subagents branch existed to protect.

### Concurrency

A resume against a handle that is currently running is refused with a
recoverable error, so two turns cannot interleave writes into one history.

Parallel *dispatch* already works and needs nothing here: a turn's tool calls run
concurrently under `join_all`, and #202 made tool execution lock-free.

### Pruning

`prune_before(cutoff)`, matching `DigestCache`, run from the existing periodic
sweep. Default retention: 7 days.

### Failure modes

| Condition | Behaviour |
|---|---|
| Handle unknown (pruned, never existed, another host) | Recoverable error saying to dispatch a fresh child. SDD already does this in rounds 4-5, so the skill needs no adaptation. |
| Handle busy | Recoverable error. |
| Agent definition gone | Recoverable error. |
| History over the cap | Answer returned; not persisted; handle line says so. |

### Testing

Scripted `StubProvider` plus `FakeClient`, as the subagents branch used: a
resumed child continues its history; resume recomputes the tool list and the
depth cap still holds; an unknown handle is recoverable; a busy handle is
refused; the cache round-trips and prunes; an over-cap history is returned but
not written.

---

## Out of scope

- **Per-dispatch model override.** SDD asks for one on every dispatch. It is a
  no-op against a single self-hosted llama.cpp server, and subagents were
  explicitly not adopted for model switching. File an issue.
- **A todo tool.** Skills say "create a todo per item"; there is no such tool
  here. SDD's real durable state is its ledger file, which the skill treats as
  the recovery map, so this degrades to prose. File an issue.
- **A plugin system** — hooks, slash commands, a marketplace, bundled MCP
  servers. Nothing in superpowers needs it; `"hooks": {}` is what its own
  non-Claude manifest declares.
- **A registry, search, or version pinning for installs.** `skill_install` takes
  a URL the person names and tracks that checkout's default branch. Discovering
  skills, pinning a tag, or rolling one back are all `git` operations the person
  can run in the same directory, and none of them is needed to make superpowers
  work.
- **Server-side skills.** Rejected during design: scripts must run where the
  repository is.
- **brainstorming's visual companion.** It runs a Node server and expects a
  browser the user can reach. It may work, since it would run on the editor's
  machine, but it is not a goal and is not tested here.
- **#199** — sharing a client project's `CLAUDE.md` — remains open and is what
  would let a subagent inherit project conventions.

## What could be wrong

- **The editor's terminal may not run bash the way this assumes**, especially on
  Windows, where it depends on a Git Bash or WSL `bash` being on `PATH`. There
  is no configuration escape from this by design; the honest fallback is that
  the model reads the skills through `client_shell` itself.
- **A private repository can hang the install.** `git clone` over `https` against
  a repository needing credentials will try to prompt, in a terminal the model
  cannot answer. The install therefore runs with prompting disabled
  (`GIT_TERMINAL_PROMPT=0`) and under the one-shot terminal's timeout, so the
  failure is a clear refusal rather than a stall. Private sources are not a
  goal.
- **The candidate list is a reimplementation of `directories`' rules in shell**,
  and it can drift from that crate's behaviour — most plausibly on a Linux
  desktop with an unusual `XDG_DATA_HOME`, or on Windows where a roaming
  profile makes `%APPDATA%` not what the user expected.
  `$SAPPHIRE_AGENT_SKILLS_DIR` exists so that drift is always recoverable
  without a release.
- **`fs/read_text_file` may be project-scoped**, which is anticipated by the
  terminal fallback but has not been confirmed against Zed.
- **Upstream may reorganise.** The contract implemented here —
  `<dir>/<name>/SKILL.md` with `name` and `description` frontmatter — is the one
  superpowers' own non-Claude manifests declare, which makes it the most stable
  part of the layout, but it is still someone else's repository moving twice a
  month.
