# Design: ambient audio ingest — `/audio/ingest`, speaker identity, transcript cache

Date: 2026-08-26

## Context

The goal is an AI secretary that listens to my day and writes what matters into
`sapphire-journal`: transcribe ambient audio, tell speakers apart, summarise, and
push previously-unrecorded facts and new tasks to the journal over MCP.

The eventual hardware is a portable, battery-powered always-on microphone —
ultimately a portable AI speakerphone that I can address by name. That device does
not exist yet in a form I can buy, so the first target is a microcontroller
(ESP32-S3 class, e.g. Seeed XIAO ESP32S3 Sense: built-in PDM mic, microSD, LiPo
charging) running purpose-built firmware, not `sapphire-call`.

Manual file handoff was explicitly rejected as an operating model — it does not
survive contact with a weekly cadence. But the same ingest endpoint serves manual
drops during development, so the pipeline can be tuned before hardware arrives.

### Decomposition

This spec covers **S1 only**.

| | Scope | Repo |
|---|---|---|
| **S1** | Ingest endpoint, audio/transcript cache, STT, speaker identity | `sapphire-agent` |
| S2 | Daily summarisation → journal MCP writes | `sapphire-agent` |
| S3 | Always-on capture, offline spool, replay | firmware / `sapphire-call` |
| S4 | Server-side wake word, downlink audio, live conversation | both |

Implementation order is S1 → S2 → S3. S1+S2 deliver end-to-end value with manual
input; S3 replaces the manual input with automation; S4 layers realtime on top.

### Constraints that shaped this design

1. **The client may be a microcontroller with no OS.** `sherpa-onnx` requires
   ONNX Runtime, which requires an OS and tens of megabytes of heap. An RP2040 has
   264 KB of SRAM; an ESP32-S3 has 8 MB PSRAM but still no ONNX Runtime. Nothing in
   the wire protocol may assume the client can run our stack. `curl` must be able to
   speak it.
2. **Battery cost is radio-on time, not compute.** On this class of device the WiFi
   radio draws an order of magnitude more than the CPU (~15–25 mA associated,
   100–250 mA transmitting, versus ~1–2 mA for a classical VAD). Every protocol
   decision optimises for "let the radio sleep", not for latency.
3. **Ambient recording of a whole day is maximally sensitive.** Local STT is the
   default. Audio is cached outside the workspace with a retention limit.
4. **STT, LLM and TTS each cost seconds.** Sub-second transport latency is noise.
   This is why the uplink is plain HTTP POST rather than a WebSocket.

### What already exists and is reused

- `src/image_cache.rs` — workspace-external, SHA-256 content-addressed cache with
  scrub/hydrate helpers. The audio cache follows its shape and its rationale.
- `src/voice/` — the `SttProvider` trait and its sherpa-onnx implementation, plus a
  `MockStt` provider that makes most of this testable without models.
- `sherpa-onnx` 1.13 already exposes `SpeakerEmbeddingExtractor` (embeddings) and
  `SpeakerEmbeddingManager` (`add` / `search` with a threshold) — online speaker
  enrolment needs no new dependency.
- `src/voice/providers/sherpa_download.rs` — model bundle fetch/extract, reused for
  the speaker embedding model.
- The bearer scheme behind `/a2a`, `/mcp` and `/acp` — device auth reuses it.
- `src/mcp_client/` — how S2 will reach `sapphire-journal-mcp`. Not touched here.

The new subsystem lives in `src/ambient/` and uses the `(ambient)` commit scope.
Per `CLAUDE.md`, agent-internal scopes need no `cliff.toml` change. It is
deliberately separate from `src/voice/`: `voice` is the **interactive** pipeline
(audio in → LLM turn → audio out); `ambient` **records without answering**.

## Architecture

```text
[capture device]                        [sapphire-agent]
  coarse VAD segments
        |
        +-- online  --> POST /audio/ingest --+
        |                                    v
        +-- offline --> spool to SD          admission (auth, idempotency, enqueue)
              |                              returns immediately; no LLM turn
              +-- replay via same endpoint --+
                                             v
                                       bounded work queue
                                             v
                                   +--- background worker ---+
                                   | Silero VAD re-gate      |
                                   | STT (local by default)  |
                                   | speaker embedding       |
                                   | match registered/known  |
                                   | else enrol as candidate |
                                   +------------+------------+
                                                v
                                   audio cache (7d) / transcripts (kept)
                                                v
                                        S2: daily summary -> journal MCP
```

**Admission and processing are separate.** Reconnecting after a day offline dumps a
whole day of spooled segments at once. The endpoint accepts and returns; a bounded
queue feeds a background worker. The device is never held waiting on STT, which is
what lets its radio go back to sleep.

## Ingest endpoint

```http
POST /audio/ingest?segment=<id>&started_at=<unix_ms>&rate=16000&live=1
Authorization: Bearer <device api key>
Content-Type: audio/L16
<raw bytes>
```

The body is raw audio. This is **not** JSON-RPC on `/rpc`: base64 inflates the
payload by 33% and forces a microcontroller to build JSON and base64-encode a
multi-kilobyte blob. Since radio-on time is the dominant battery cost (constraint 2),
both the bytes and the CPU matter, and neither buys anything.

- **`Content-Type`** carries the format: `audio/L16` (raw s16le) and `audio/wav` in
  v1. This is the forward-compatibility seam — adding `audio/opus` later cuts
  transmitted bytes by an order of magnitude without changing the protocol. Opus is
  deferred until the ESP32-S3 draw is measured, because it costs a decoder
  dependency server-side and encoder cycles device-side.
- **`rate`** must be 16000 in v1 (matching `voice::PIPELINE_SAMPLE_RATE`); mono only.
  Anything else is rejected rather than resampled.
- **`segment`** is the idempotency key. A repeat returns `200` and discards the body.
  Replay and live delivery share one path, so duplicates are a normal condition, not
  an error.
- **`started_at`** is **when the audio was recorded**, not when it arrived. This is
  what makes an offline segment and a live segment the same kind of object.
- **`live`** marks realtime audio. Default `0`.

### Device identity comes from the API key

There is no `device` parameter. One API key per device, bound server-side:

```toml
[device.pendant]
api_key      = "sa-dev-<long random>"
label        = "the one on the lanyard"
room_profile = "default"   # which profile a conversation runs under (S4)
```

The config key (`pendant`) is the stable device id recorded in transcripts.

Rationale over the existing `sapphire-call` model (client-generated UUID v7 in
`sapphire-call-core/src/device_id.rs`, sent per request):

- A device cannot impersonate another device; identity is not self-declared.
- Firmware needs one constant, with no UUID generation and no NVS persistence.
- Transcripts carry `"device":"pendant"` instead of a UUID.

Both models coexist — the existing voice satellite path is unchanged. **One key per
device is a requirement, not a suggestion:** a shared key would collide in the
`segment` idempotency namespace.

### `live` is explicit, not inferred

Freshness is not derived from `started_at`. An ESP32 has no RTC; while offline it
reconstructs wall-clock from elapsed time since the last NTP sync, and that drifts.
Deriving liveness from a drifting clock would put S4's safety property on a
foundation that fails silently. An explicit flag costs the firmware one byte.

The property this protects: **wake-word detection runs only on `live=1` segments.**
The agent must never answer something I said six hours ago because the recording of
it just arrived.

### `POST /audio/hello` (optional)

Startup handshake. The device reports firmware version and supported formats; the
server returns ingest parameters (accepted content types, maximum segment length,
whether a downlink exists). It exists so firmware need not hard-code server
capabilities.

**It is optional by design.** Going straight to `/audio/ingest` works. Preserving the
"one `curl` line" property matters more than a tidy handshake, because the same
endpoint serves manual drops, a Termux script, and firmware that may not be ours.

Liveness state (last segment received, online/offline, firmware version) is held in
memory, never written to config.

## Segment router

A per-device state machine sits in front of the pipeline. **Only the "record only"
branch is built now**; the state stays pinned to `Idle`.

```text
segment admitted
   |- live=0 --> record only
   `- live=1 --> device state
         |- Idle       --> record only  (+ wake-word detection, S4)
         `- Conversing --> record + conversation turn (S4)
```

The seam is built now because S4 changes what a segment *means*, and retrofitting
that decision point later means threading it through the whole pipeline. Adding a
branch to an existing fork is cheap; introducing the fork is not.

## Processing pipeline

1. **Silero VAD re-gate.** The device's VAD is classical (energy or WebRTC-style) and
   tuned to over-capture; sherpa's Silero pass trims leading/trailing silence and
   drops noise-only segments. A segment with no speech produces no transcript.
2. **STT** via the existing `SttProvider`. Local (sherpa-onnx) by default,
   swappable to an API by config. Feasible because VAD removes silence and online
   audio is processed incrementally — only the offline portion arrives in a batch.
3. **Speaker embedding** via `SpeakerEmbeddingExtractor`, **skipped for segments
   shorter than `min_embed_ms` (default 1500)**. Embeddings from very short
   utterances are unreliable and are the main driver of speaker-id inflation. Such
   segments get `"speaker": null` and keep their transcript.
4. **Match** with `SpeakerEmbeddingManager::search` against registered speakers *and*
   existing candidates.
5. **Enrol on miss.** A new grain-id is minted, and the embedding, a representative
   clip and statistics are stored as a candidate. Later segments from that voice
   match at step 4.

`OfflineSpeakerDiarization` is **not used**. It clusters within a single input, but
VAD has already cut segments to single utterances, so embedding-plus-match is
sufficient. If real recordings show overlapping speakers inside one segment, it can
be added then.

## Cache layout and retention

Rooted at `dirs::cache_dir()/sapphire-agent/ambient/`, outside the workspace, on the
same principle as `image_cache`.

```text
ambient/
  audio/<sha256>                    # raw segments; swept after audio_retention_days
  transcripts/YYYY-MM-DD.jsonl      # one line per segment, time-ordered; kept
  speakers/
    registered/<sha256>.<model>.emb # embeddings of workspace reference audio
    candidates/<grain-id>/
      centroid.emb                  # running mean embedding
      clip.wav                      # representative clip, exported on promotion
      stats.json                    # cumulative speech seconds, days seen, first seen
```

A transcript line:

```json
{"segment":"...","device":"pendant","started_at":"2026-08-26T14:03:11+09:00",
 "speech_ms":4200,"speaker":"me","speaker_score":0.87,"text":"...","audio":"<sha256>"}
```

- **`started_at`** is the device's recording start for the whole segment, before
  re-gating — it is what the device sent, unmodified.
- **`speech_ms`** is the speech duration *after* the Silero re-gate, not the length of
  the submitted segment. It is also the value accumulated into a candidate's
  `promote_after_seconds` total and compared against `min_embed_ms`, so all three
  measure the same thing.
- **`speaker_score`** is the `SpeakerEmbeddingManager` match score, not an STT
  confidence. It is absent when `speaker` is `null`.

**`speaker` holds an id, never a display name.** Names resolve from the workspace at
read time, so renaming `voices/blithe-otter-42/` to `voices/tanaka-san/` makes every
past transcript read back under the new name with no rewrite pass.

**Day boundaries follow the existing `day_boundary_hour` config**, not UTC midnight
and not local midnight. The `YYYY-MM-DD.jsonl` filename and the "days seen" count
behind `promote_after_days` both use it, so a conversation at 02:00 lands in the same
day as the evening before it, exactly like the agent's daily logs.

Retention: audio `audio_retention_days` (default 7), transcripts unbounded. At an
estimated 2–4 hours of actual speech per day, 16 kHz s16le is roughly 200–450 MB/day,
so a week is 1.5–3 GB; transcripts are text and effectively free. A week is enough to
re-listen or re-run STT; beyond that only text remains.

## Speaker registry and promotion

The workspace holds reference audio — the deliberate exception to "no media in the
workspace", because it is input I curate, not derived data:

```text
voices/
  me/                *.wav
  agent/             *.wav    # samples of the agent's own TTS
  tanaka-san/        *.wav
  blithe-otter-42/   clip.wav # auto-promoted; rename to finish registering
```

The directory name is both the speaker id and the display name. Embeddings are keyed
by (reference file sha256 × model id) in the cache, so renaming triggers no
recomputation, and **changing the embedding model recomputes automatically** — no
model-dependent data lives in the workspace.

### Why candidates are not written straight to the workspace

A day of ambient audio contains television, shop staff, train announcements and
passers-by. Writing every first-seen voice into the workspace would bury the handful
of people I actually want to name under hundreds of one-off entries, and voice
variation (mic distance, whispering, background noise) splits one person across
several ids.

Two tiers:

- **Candidate** — cache only. Id, centroid, representative clip, statistics.
- **Promoted** — exported to `voices/<grain-id>/` once it clears
  `promote_after_seconds` (default 60) **and** `promote_after_days` (default 2).
  Transient voices do not survive both thresholds.

Manual promotion is also available for candidates below the thresholds.

Online enrolment gives cross-day stability that a per-day batch clustering pass
cannot: the id persists in the manager, so the same voice matches tomorrow.

## Agent-facing tools

Three built-in tools in `src/tools/`. This is the S1/S2 boundary.

| Tool | Purpose |
|---|---|
| `transcript_read` | Read transcripts by time range, optionally filtered by speaker. S2's summarisation calls this. |
| `speaker_candidates` | List unpromoted candidates: id, cumulative seconds, days seen, sample utterances. |
| `speaker_promote` | Export a candidate to the workspace, optionally naming it in the same call. |

`speaker_promote` taking a name means registering a speaker can be done by telling
the agent "that was Tanaka-san" in chat, without touching files.

## Configuration

```toml
[ambient]
enabled              = true
cache_dir            = "..."        # default: OS cache dir
audio_retention_days = 7
stt_provider         = "sherpa_ja"  # name from [stt_provider.*]
min_embed_ms         = 1500
match_threshold      = 0.55
promote_after_seconds = 60
promote_after_days    = 2
max_queue             = 1000

[device.pendant]
api_key      = "sa-dev-<long random>"
label        = "the one on the lanyard"
room_profile = "default"
```

## Error handling

- **Queue full** returns `429`. The device spools and retries — the same behaviour as
  being offline, so no new branch in firmware.
- **STT or embedding failure** drops that segment only; the audio stays cached. The
  pipeline does not stop.
- **Unreadable reference audio** logs a warning and disables that speaker. Ingest
  continues.
- **Unknown or unbound bearer** returns `401`.

## Testing

TDD. `MockStt` already exists, so most of this runs without models.

- Endpoint: auth, idempotent double-POST of one `segment`, bad `Content-Type`,
  non-16 kHz rejection, `429` on a full queue.
- Promotion policy boundaries: 59 s vs 60 s, 1 day vs 2 days.
- Rename transparency: promote a candidate, rename the directory, confirm past
  transcripts read back under the new name.
- Retention: audio older than the limit is swept; transcripts survive.
- Re-gate: a silence-only segment produces no transcript line.
- Router: a `live=0` segment never reaches the conversation branch.

## Out of scope

S2 (summarisation, journal MCP writes), S3 (firmware and `sapphire-call` capture),
S4 (wake word, downlink, live conversation), opus, multi-speaker separation within a
single segment.

## Reserved for future work

`GET /audio/events` is **reserved but not implemented** — the downlink that carries
server-initiated audio to the device, for S4.

Deferring the downlink transport costs S1 nothing. What S1 must fix now, and does,
is device identity, the `live` flag, and the router seam. The current preference is
SSE, because `sapphire-call` already receives server-pushed audio chunks that way
(`voice/subscribe`, `VoicePushItem::AudioChunk`), so it is the least new machinery;
a downlink is one-directional and does not need WebSocket's duplex. This also stays
within the battery budget: the downlink is only held open while online, when the
radio is already awake for the uplink, and offline means no reply is wanted anyway.

### Firmware notes for S3

- **300–500 ms pre-roll buffer.** Server-side wake-word detection fails if the
  device's VAD clips the start of the word. The device must send audio from *before*
  the VAD trigger. This does not change the protocol, but it is a hard firmware
  requirement.
- One API key per device (see above).
- Offline spool needs real storage: 16 MB of flash holds about eight minutes of raw
  16 kHz audio. A microSD slot is not optional for a day of offline recording.

## Rejected alternatives

- **JSON-RPC on `/rpc` with base64 audio.** 33% more bytes and JSON assembly on an
  MCU, buying nothing; radio-on time is the battery cost.
- **WebSocket uplink.** Bidirectional framing, masking and ping/pong for a device
  that is idle most of the time, and offline replay would need a second path anyway.
  STT/LLM/TTS latency dwarfs the transport difference.
- **Silero VAD on the device.** Impossible: ONNX Runtime needs an OS. A classical VAD
  on-device plus Silero server-side gets both cheap radio gating and accurate
  segmentation.
- **Deferred nightly clustering of unknown speakers.** Online enrolment is simpler
  and gives better cross-day stability, which was the whole point of the batch pass.
- **Writing every first-seen speaker to the workspace.** Ambient audio floods it with
  television and passers-by.
- **Client-declared device id.** Self-declared identity is spoofable and costs the
  firmware UUID generation and persistence.
