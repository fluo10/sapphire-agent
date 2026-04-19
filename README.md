# sapphire-agent

A personal AI assistant agent that lives in a [`sapphire-workspace`](https://crates.io/crates/sapphire-workspace) and talks to me through Matrix and Discord.

> **Status: personal project.** This is something I built for my own use. It only has to work in my environment, and that is the only environment I test it in. You are welcome to use it, fork it, or send pull requests, but I am not going to maintain providers, channels, or features I do not personally use. If your use case overlaps with mine, great; if not, fork freely.
>
> The reason this exists at all is that other agents I tried (openclaw, zeroclaw, …) either did not support what I needed, were not actually tested for the parts I cared about, or did not accept fixes. So I wrote my own. Please calibrate expectations accordingly.

## What it does

- **Channels**: Matrix (E2EE via `matrix-sdk`) and Discord (`serenity`).
- **Provider**: Anthropic Messages API with SSE streaming and a multi-round tool-use loop.
- **Workspace**: backed by [`sapphire-workspace`](https://crates.io/crates/sapphire-workspace) — file index, full-text + vector search (LanceDB), git sync.
- **Built-in tools**: `file_read`, `file_write`, `file_append`, `file_delete`, `dir_list`, `dir_walk`, `web_search`, `weather`, `shell`, plus workspace memory / search / sync tools.
- **Sessions**: human-readable [`grain-id`](https://crates.io/crates/grain-id) aliases, auto-generated titles, history dump on resume.
- **Background**: heartbeat cron tasks, periodic memory compaction, periodic workspace sync, daily logs.
- **Commands**:
  - `call` — interactive REPL (reedline)
  - `serve` — MCP Streamable HTTP API
  - `run` — start the channel listeners
  - `verify` — validate config and report loaded workspace files

## Install

```sh
cargo install sapphire-agent
```

Or from source:

```sh
git clone https://github.com/fluo10/sapphire-agent
cd sapphire-agent
cargo build --release
```

## Configure

Copy `config.example.toml` to your config directory (`~/.config/sapphire-agent/config.toml` on Linux) and fill in the Anthropic API key, workspace path, and whichever channels you actually want.

Then:

```sh
sapphire-agent verify   # sanity-check config and workspace
sapphire-agent run      # start the channel listeners
sapphire-agent call     # one-off interactive session
```

## License

Licensed under either of:

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT License ([LICENSE-MIT](LICENSE-MIT))

at your option.
