//! Client library for `sapphire-agent`.
//!
//! Provides an HTTP client for the sapphire-agent JSON-RPC 2.0 API and an
//! interactive REPL that can be embedded in any binary (`sapphire-agent call`
//! or the standalone `sapphire-call`).

use anyhow::{Context, Result};
use futures_util::StreamExt;
use reedline::{
    Prompt, PromptEditMode, PromptHistorySearch, PromptHistorySearchStatus, Reedline, Signal,
};
use serde_json::{Value, json};
use std::borrow::Cow;
use std::io::{Write, stderr, stdout};
use std::sync::atomic::{AtomicU64, Ordering};

static REQUEST_ID: AtomicU64 = AtomicU64::new(1);

/// Minimal prompt for reedline: just "> ".
struct SimplePrompt;

impl Prompt for SimplePrompt {
    fn render_prompt_left(&self) -> Cow<'_, str> {
        Cow::Borrowed("> ")
    }
    fn render_prompt_right(&self) -> Cow<'_, str> {
        Cow::Borrowed("")
    }
    fn render_prompt_indicator(&self, _mode: PromptEditMode) -> Cow<'_, str> {
        Cow::Borrowed("")
    }
    fn render_prompt_multiline_indicator(&self) -> Cow<'_, str> {
        Cow::Borrowed("::: ")
    }
    fn render_prompt_history_search_indicator(
        &self,
        history_search: PromptHistorySearch,
    ) -> Cow<'_, str> {
        let prefix = match history_search.status {
            PromptHistorySearchStatus::Passing => "",
            PromptHistorySearchStatus::Failing => "failing ",
        };
        Cow::Owned(format!(
            "({}reverse-search: {}) ",
            prefix, history_search.term
        ))
    }
}

fn next_id() -> u64 {
    REQUEST_ID.fetch_add(1, Ordering::Relaxed)
}

/// Run the interactive call client.
///
/// This is the shared entry point used by both `sapphire-agent call` and
/// the standalone `sapphire-call` binary.
pub async fn run(
    server: String,
    session: Option<String>,
    list: bool,
    message: Option<String>,
    history: bool,
    json: bool,
    profile: Option<String>,
) -> Result<()> {
    let base = server.trim_end_matches('/').to_string();
    let client = reqwest::Client::new();

    // -- --list mode ----------------------------------------------------------
    if list {
        let (mcp_session_id, _, _) = initialize_session(&client, &base, session, None).await?;
        list_sessions(&client, &base, &mcp_session_id, json).await?;
        return Ok(());
    }

    // -- Initialize session ---------------------------------------------------
    let (mut mcp_session_id, actual_session_id, is_new) =
        initialize_session(&client, &base, session, profile.as_deref()).await?;

    // -- --history dump-only mode ---------------------------------------------
    if history {
        dump_history(&client, &base, &mcp_session_id, json, true).await?;
        return Ok(());
    }

    // -- --message one-shot mode ----------------------------------------------
    if let Some(text) = message {
        let trimmed = text.trim();
        if trimmed.is_empty() {
            anyhow::bail!("--message requires non-empty text");
        }
        send_chat(&client, &base, &mcp_session_id, trimmed, json).await?;
        if !json {
            println!();
        }
        return Ok(());
    }

    // -- REPL -----------------------------------------------------------------
    // --json is a no-op in REPL mode (text output is always human-readable).
    let _ = json;
    println!("sapphire-agent call  (session: {actual_session_id})");
    if !is_new {
        println!("[resumed existing session]\n");
        if let Err(e) = dump_history(&client, &base, &mcp_session_id, false, false).await {
            eprintln!("[warning: failed to load history: {e:#}]");
        }
    }
    println!("Commands: /clear  /help  /quit\n");

    let mut line_editor = Reedline::create();
    let prompt = SimplePrompt;

    loop {
        let line = match line_editor.read_line(&prompt) {
            Ok(Signal::Success(buf)) => buf,
            Ok(Signal::CtrlC) | Ok(Signal::CtrlD) => break,
            Ok(_) => continue,
            Err(e) => {
                eprintln!("Error reading input: {e}");
                break;
            }
        };

        let trimmed = line.trim();
        match trimmed {
            "/quit" | "/exit" => break,
            "/help" => {
                println!("  /clear   Start a new session");
                println!("  /quit    Exit");
                continue;
            }
            "/clear" => {
                match initialize_session(&client, &base, None, profile.as_deref()).await {
                    Ok((new_mcp_id, new_session_id, _)) => {
                        mcp_session_id = new_mcp_id;
                        println!("[new session: {new_session_id}]");
                    }
                    Err(e) => eprintln!("[error starting new session: {e:#}]"),
                }
                continue;
            }
            "" => continue,
            _ => {}
        }

        if let Err(e) = send_chat(&client, &base, &mcp_session_id, trimmed, false).await {
            eprintln!("[error: {e:#}]");
        }
        println!();
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Session initialization
// ---------------------------------------------------------------------------

async fn initialize_session(
    client: &reqwest::Client,
    base: &str,
    session: Option<String>,
    profile: Option<&str>,
) -> Result<(String, String, bool)> {
    let session_id_req = session.as_deref().unwrap_or("new");
    let mut params = json!({ "session_id": session_id_req });
    if let Some(p) = profile {
        params["profile"] = json!(p);
    }
    let body = json!({
        "jsonrpc": "2.0",
        "id": next_id(),
        "method": "initialize",
        "params": params,
    });

    let resp = client
        .post(format!("{base}/mcp"))
        .json(&body)
        .send()
        .await
        .context("Failed to connect to server. Is `sapphire-agent serve` running?")?
        .error_for_status()?;

    let mcp_session_id = resp
        .headers()
        .get("mcp-session-id")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string())
        .unwrap_or_else(|| session_id_req.to_string());

    let val: Value = resp.json().await?;
    if let Some(err) = val.get("error") {
        let msg = err["message"].as_str().unwrap_or("unknown error");
        anyhow::bail!("{msg}");
    }
    let result = &val["result"];
    // Prefer human-readable grain-id for display; fall back to UUID
    let display_id = result["public_id"]
        .as_str()
        .or_else(|| result["session_id"].as_str())
        .unwrap_or(&mcp_session_id)
        .to_string();
    let is_new = result["is_new"].as_bool().unwrap_or(true);

    Ok((mcp_session_id, display_id, is_new))
}

// ---------------------------------------------------------------------------
// History dump (on session resume)
// ---------------------------------------------------------------------------

async fn dump_history(
    client: &reqwest::Client,
    base: &str,
    mcp_session_id: &str,
    json_mode: bool,
    standalone: bool,
) -> Result<()> {
    let body = json!({
        "jsonrpc": "2.0",
        "id": next_id(),
        "method": "get_session",
        "params": null,
    });

    let val: Value = client
        .post(format!("{base}/mcp"))
        .header("mcp-session-id", mcp_session_id)
        .json(&body)
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;

    if let Some(err) = val.get("error") {
        let msg = err["message"].as_str().unwrap_or("unknown error");
        anyhow::bail!("{msg}");
    }

    let messages = val["result"]["messages"]
        .as_array()
        .cloned()
        .unwrap_or_default();

    if json_mode {
        let out = json!({ "messages": messages });
        println!("{}", serde_json::to_string(&out)?);
        return Ok(());
    }

    if messages.is_empty() {
        return Ok(());
    }

    if !standalone {
        println!("──── history ────");
    }
    for msg in &messages {
        let role = msg["role"].as_str().unwrap_or("?");
        let parts = msg["parts"].as_array().cloned().unwrap_or_default();
        for part in parts {
            match part["type"].as_str() {
                Some("text") => {
                    let text = part["text"].as_str().unwrap_or("");
                    if text.is_empty() {
                        continue;
                    }
                    match role {
                        "user" => println!("> {text}"),
                        "assistant" => println!("{text}\n"),
                        _ => println!("{text}"),
                    }
                }
                Some("tool_use") => {
                    let name = part["name"].as_str().unwrap_or("?");
                    println!("[tool: {name}]");
                }
                Some("tool_result") => {
                    // Skip tool results in the dump to keep it readable
                }
                _ => {}
            }
        }
    }
    if !standalone {
        println!("──── end of history ────\n");
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Session listing
// ---------------------------------------------------------------------------

async fn list_sessions(
    client: &reqwest::Client,
    base: &str,
    mcp_session_id: &str,
    json_mode: bool,
) -> Result<()> {
    let body = json!({
        "jsonrpc": "2.0",
        "id": next_id(),
        "method": "list_sessions",
        "params": null,
    });

    let val: Value = client
        .post(format!("{base}/mcp"))
        .header("mcp-session-id", mcp_session_id)
        .json(&body)
        .send()
        .await?
        .error_for_status()?
        .json()
        .await?;

    let sessions = val["result"]["sessions"]
        .as_array()
        .cloned()
        .unwrap_or_default();

    if json_mode {
        let out = json!({ "sessions": sessions });
        println!("{}", serde_json::to_string(&out)?);
        return Ok(());
    }

    if sessions.is_empty() {
        println!("No sessions found.");
    } else {
        println!("{:<10}  {:<30}  Created at", "ID", "Title");
        println!("{}", "-".repeat(64));
        for s in &sessions {
            let pub_id = s["public_id"].as_str().unwrap_or("-");
            let title = s["title"].as_str().unwrap_or("(untitled)");
            let created = s["created_at"].as_str().unwrap_or("?");
            println!("{pub_id:<10}  {title:<30}  {created}");
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Send a chat message and stream back the SSE response
// ---------------------------------------------------------------------------

async fn send_chat(
    client: &reqwest::Client,
    base: &str,
    mcp_session_id: &str,
    content: &str,
    json_mode: bool,
) -> Result<()> {
    let id = next_id();
    let body = json!({
        "jsonrpc": "2.0",
        "id": id,
        "method": "chat",
        "params": { "content": content },
    });

    let resp = client
        .post(format!("{base}/mcp"))
        .header("mcp-session-id", mcp_session_id)
        .header("accept", "text/event-stream")
        .json(&body)
        .send()
        .await?
        .error_for_status()?;

    let mut stream = resp.bytes_stream();
    let mut buf = String::new();

    // For JSON mode, accumulate events until the final result and emit one
    // structured object at the end.
    let mut json_tools: Vec<Value> = Vec::new();
    let mut json_content: Option<String> = None;
    let mut json_error: Option<String> = None;

    'outer: while let Some(chunk) = stream.next().await {
        buf.push_str(&String::from_utf8_lossy(&chunk?));
        while let Some(pos) = buf.find("\n\n") {
            let raw = buf[..pos].to_string();
            buf.drain(..pos + 2);
            if let Some(data) = parse_sse_data(&raw) {
                if json_mode {
                    if collect_event(&data, &mut json_tools, &mut json_content, &mut json_error) {
                        break 'outer;
                    }
                } else if handle_event(&data) {
                    return Ok(());
                }
            }
        }
    }

    if json_mode {
        let mut out = json!({
            "content": json_content.unwrap_or_default(),
            "tools": json_tools,
        });
        if let Some(err) = json_error {
            out["error"] = Value::String(err);
        }
        println!("{}", serde_json::to_string(&out)?);
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// SSE parsing helpers
// ---------------------------------------------------------------------------

/// Extract the `data:` field from a raw SSE message block, parse as JSON.
fn parse_sse_data(raw: &str) -> Option<Value> {
    let data_line = raw.lines().find(|l| l.starts_with("data:"))?;
    let data = data_line.strip_prefix("data:").unwrap_or("").trim();
    serde_json::from_str(data).ok()
}

/// Collect SSE events into JSON output buffers. Returns true when done.
fn collect_event(
    val: &Value,
    tools: &mut Vec<Value>,
    content: &mut Option<String>,
    error: &mut Option<String>,
) -> bool {
    if let Some(method) = val["method"].as_str() {
        if method == "tool_start" {
            let name = val["params"]["name"].as_str().unwrap_or("?").to_string();
            let id = val["params"]["id"].as_str().unwrap_or("").to_string();
            tools.push(json!({ "id": id, "name": name }));
        }
        false
    } else if val.get("result").is_some() {
        *content = val["result"]["content"].as_str().map(|s| s.to_string());
        true
    } else if let Some(err) = val.get("error") {
        *error = err["message"].as_str().map(|s| s.to_string());
        true
    } else {
        false
    }
}

/// Handle a JSON-RPC 2.0 SSE event. Returns true when the final result arrives.
fn handle_event(val: &Value) -> bool {
    if let Some(method) = val["method"].as_str() {
        // Notification
        match method {
            "tool_start" => {
                let name = val["params"]["name"].as_str().unwrap_or("?");
                eprint!("[{name}] ");
                let _ = stderr().flush();
            }
            "tool_end" => {
                eprint!("done ");
                let _ = stderr().flush();
            }
            _ => {}
        }
        false
    } else if val.get("result").is_some() {
        // Final chat response
        if let Some(content) = val["result"]["content"].as_str()
            && !content.is_empty()
        {
            eprintln!();
            print!("{content}");
            let _ = stdout().flush();
        }
        true // done
    } else if val.get("error").is_some() {
        let msg = val["error"]["message"].as_str().unwrap_or("unknown error");
        eprintln!("\n[error: {msg}]");
        true // treat as done
    } else {
        false
    }
}
