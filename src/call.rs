//! Interactive REPL client for `sapphire-agent call`.
//!
//! Connects to a running `sapphire-agent serve` instance via MCP Streamable HTTP
//! (JSON-RPC 2.0 over HTTP + SSE).

use anyhow::{Context, Result};
use futures_util::StreamExt;
use serde_json::{Value, json};
use std::io::{Write, stdin, stdout, stderr};
use std::sync::atomic::{AtomicU64, Ordering};

static REQUEST_ID: AtomicU64 = AtomicU64::new(1);

fn next_id() -> u64 {
    REQUEST_ID.fetch_add(1, Ordering::Relaxed)
}

pub async fn run(server: String, session: Option<String>, list: bool) -> Result<()> {
    let base = server.trim_end_matches('/').to_string();
    let client = reqwest::Client::new();

    // ── Initialize session ──────────────────────────────────────────────────
    let (mut mcp_session_id, actual_session_id, is_new) =
        initialize_session(&client, &base, session).await?;

    // ── --list mode ─────────────────────────────────────────────────────────
    if list {
        list_sessions(&client, &base, &mcp_session_id).await?;
        return Ok(());
    }

    // ── REPL ────────────────────────────────────────────────────────────────
    println!("sapphire-agent call  (session: {actual_session_id})");
    if !is_new {
        println!("[resumed existing session]");
    }
    println!("Commands: /clear  /help  /quit\n");

    loop {
        print!("> ");
        stdout().flush()?;

        let mut line = String::new();
        match stdin().read_line(&mut line) {
            Ok(0) => break, // EOF
            Ok(_) => {}
            Err(e) => {
                eprintln!("Error reading stdin: {e}");
                break;
            }
        }

        let trimmed = line.trim();
        match trimmed {
            "/quit" | "/exit" => break,
            "/help" => {
                println!("  /clear   Start a new session");
                println!("  /quit    Exit");
                continue;
            }
            "/clear" => {
                match initialize_session(&client, &base, None).await {
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

        if let Err(e) = send_chat(&client, &base, &mcp_session_id, trimmed).await {
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
) -> Result<(String, String, bool)> {
    let session_id_req = session.as_deref().unwrap_or("new");
    let body = json!({
        "jsonrpc": "2.0",
        "id": next_id(),
        "method": "initialize",
        "params": { "session_id": session_id_req },
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
    let result = &val["result"];
    let actual_session_id = result["session_id"]
        .as_str()
        .unwrap_or(&mcp_session_id)
        .to_string();
    let is_new = result["is_new"].as_bool().unwrap_or(true);

    Ok((mcp_session_id, actual_session_id, is_new))
}

// ---------------------------------------------------------------------------
// Session listing
// ---------------------------------------------------------------------------

async fn list_sessions(
    client: &reqwest::Client,
    base: &str,
    mcp_session_id: &str,
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
        .map(|v| v.as_slice())
        .unwrap_or(&[]);

    if sessions.is_empty() {
        println!("No sessions found.");
    } else {
        println!("{:<38}  {}", "Session ID", "Created at");
        println!("{}", "-".repeat(62));
        for s in sessions {
            let id = s["session_id"].as_str().unwrap_or("?");
            let created = s["created_at"].as_str().unwrap_or("?");
            println!("{id:<38}  {created}");
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

    while let Some(chunk) = stream.next().await {
        buf.push_str(&String::from_utf8_lossy(&chunk?));
        while let Some(pos) = buf.find("\n\n") {
            let raw = buf[..pos].to_string();
            buf.drain(..pos + 2);
            if let Some(data) = parse_sse_data(&raw) {
                if handle_event(&data) {
                    return Ok(());
                }
            }
        }
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
        if let Some(content) = val["result"]["content"].as_str() {
            if !content.is_empty() {
                // Print a newline after tool activity (if any) before the response
                let needs_newline = val.get("_tool_active").is_none(); // always true; handled below
                let _ = needs_newline; // suppress warning
                eprint!("\n");
                print!("{content}");
                let _ = stdout().flush();
            }
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
