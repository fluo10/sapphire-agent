# Voice pipeline testing

Two paths for trying the voice pipeline end-to-end:

1. **Mock smoke test** — fastest. Verifies the MCP wire format, SSE
   streaming, and the `sapphire-call ↔ sapphire-agent` plumbing without
   downloading any models or starting Irodori-TTS. ~2 minutes.
2. **Real test** — SenseVoice STT + Irodori-TTS. ~30 minutes the first
   time (sherpa-onnx C++ build, model downloads, Irodori-TTS setup).

The mock path is the right first move when something's wrong — it
isolates "is the wiring broken" from "is sherpa/Irodori broken".

---

## 1. Mock smoke test

### One-time setup

You need an Anthropic API key. The LLM step runs for real even in mock
mode; the mocks only stub out STT (constant transcript) and TTS (sine
wave).

```sh
export ANTHROPIC_API_KEY=sk-ant-...
```

### Build

```sh
cargo build --bin sapphire-agent --bin sapphire-call
```

No `--features voice-sherpa` needed — the mock providers are part of
the default build.

### Run

Terminal A (server):

```sh
cargo run --bin sapphire-agent -- \
    --config test-configs/voice-mock.toml serve
```

Terminal B (satellite, microphone + speaker required):

```sh
cargo run --bin sapphire-call -- voice --room-profile voice_test
```

The first satellite run downloads the Silero VAD ONNX (~2 MB) to
`~/.local/share/sapphire-call/voice-models/`.

### What to expect

1. Speak any utterance.
2. Server log shows
   `voice/pipeline_run: STT via 'mock_stt' (...samples...)` and
   `STT via 'mock_tts' (...chars)`.
3. Satellite stderr prints `> Say hi briefly.` (the mock STT result)
   followed by the LLM's actual reply text.
4. Speaker plays a 400 ms 440 Hz beep (the mock TTS output).

If you reach step 4 the MCP wire format, the SSE stream, the audio
chunk encoding, and the satellite's playback queue are all working.

---

## 2. Real test: SenseVoice STT + Irodori-TTS

### One-time setup

#### a. sherpa-onnx-enabled agent build

Compiles sherpa-onnx + ONNX runtime through CMake. **~10 minutes
cold**, mostly bottlenecked on cmake/onnxruntime, not Rust.

```sh
cargo build --release --features voice-sherpa --bin sapphire-agent
```

#### b. Satellite build

Compiles sherpa-onnx a second time (workspace member). Same ~10 min
cold on the satellite side — distribute via prebuilt binaries in
production.

```sh
cargo build --release --bin sapphire-call
```

#### c. Irodori-TTS server

In a separate directory:

```sh
git clone https://github.com/Aratako/Irodori-TTS
cd Irodori-TTS
pip install -r requirements.txt
# Run the Gradio entrypoint — check the repo for the exact command;
# it spawns at http://localhost:7860 with a `generate` endpoint.
python app.py
```

Confirm it's up:

```sh
curl http://localhost:7860/info  # should return JSON describing /generate
```

#### d. Anthropic API key

```sh
export ANTHROPIC_API_KEY=sk-ant-...
```

### Run

Terminal A (server):

```sh
cargo run --release --features voice-sherpa --bin sapphire-agent -- \
    --config test-configs/voice-irodori.toml serve
```

First boot auto-downloads the SenseVoice bundle (~230 MB) from
sherpa-onnx GitHub releases to
`~/.local/share/sapphire-agent/voice-models/`.

Terminal B (satellite):

```sh
cargo run --release --bin sapphire-call -- voice \
    --room-profile voice_irodori --language ja
```

First boot auto-downloads Silero VAD (~2 MB) to
`~/.local/share/sapphire-call/voice-models/`.

### What to expect

1. Server prints `voice/pipeline_run: STT via 'sense_voice' ...` after
   the first utterance reaches the pipeline.
2. Satellite prints `> <transcribed Japanese text>` once SenseVoice
   finishes (sub-second on Apple Silicon, longer on CPU-only x86).
3. Server prints `TTS via 'irodori' ...`; satellite plays the synth
   reply through the speaker.

### Tuning knobs

- **VAD too sensitive / not sensitive enough** — edit
  `crates/sapphire-call/src/voice/mod.rs`, the `silero_config` values
  inside `build_vad()`:
    - `threshold` (default 0.5): lower → more permissive
    - `min_silence_duration` (0.25): higher → longer pause needed to
      close an utterance
- **Irodori payload schema drift** — if the upstream Gradio app
  changes its argument count, edit
  `test-configs/voice-irodori.toml`'s `[tts_provider.irodori].payload`
  template. The user message goes into `{{text}}` (JSON-escaped).

---

## 3. Wake-word mode (optional)

Once the basic real path works, add a wake-word gate so the satellite
only forwards utterances after a configured phrase fires.

```sh
cargo run --release --bin sapphire-call -- voice \
    --room-profile voice_irodori --language ja \
    --wake-word-model sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01
```

First run downloads the KWS bundle (~50 MB). The bundle ships its own
`keywords.txt`; override with `--keywords-file <path>` to use custom
phrases. Note that keywords use sherpa-onnx's tokenised format — one
line per phrase, tokens separated by spaces, then `@<readable label>`.

---

## 4. Common gotchas

| Symptom | Cause | Fix |
|---|---|---|
| `no default input device` | macOS: app lacks mic permission | System Settings → Privacy & Security → Microphone |
| Silent reply, no errors | Output device muted or output_rate mismatch in the satellite's log | Check `output: <rate> Hz` line; raise system volume |
| `[error: Internal error: Provider error: ...]` from satellite | Anthropic key absent / invalid on the server side | Verify `ANTHROPIC_API_KEY` env var on the **server** terminal |
| `Failed to read voice_pipeline 'irodori'` validation error at server startup | room_profile.voice_pipeline points at undefined block | Match name with `[voice_pipeline.<n>]` |
| `failed to fetch Silero VAD model` | No network / proxy | Pre-download the model manually and place at `~/.local/share/sapphire-call/voice-models/silero_vad.onnx` |
| First sherpa-onnx build OOM-kills CI | C++ compile is memory-heavy | Use a runner with ≥4 GB / 2 cores or set `CARGO_BUILD_JOBS=1` |

---

## 5. What's still in unit-test territory

The TOML schema, MCP wire format helpers, VAD math, and resampler are
covered by `cargo test --workspace`. The whole MCP-server-to-cpal
round-trip is **not** unit-tested — the integration boundary is the
manual flow above. Bringing that into automated CI would require
spawning a real axum process plus a mock LLM that talks the
JSON-streaming Anthropic protocol, which is more bookkeeping than it
saves at the project's current size.
