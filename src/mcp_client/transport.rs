//! MCP transport abstraction — HTTP (Streamable HTTP) and stdio.

use anyhow::{Context, Result, bail};
use async_trait::async_trait;
use futures_util::StreamExt;
use serde_json::{Value, json};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::sync::Mutex;
use tracing::{debug, warn};

// ---------------------------------------------------------------------------
// Transport trait
// ---------------------------------------------------------------------------

/// Callback invoked when the server sends a JSON-RPC request (with `id` and
/// `method`) back to the client during an ongoing request.  The callback
/// returns the JSON-RPC response object to send back.
pub type ServerRequestHandler = Arc<dyn Fn(&str, &Value) -> Value + Send + Sync>;

/// Callback invoked when the server sends a JSON-RPC notification (no `id`,
/// has `method`).  Used to detect `notifications/tools/list_changed` etc.
pub type NotificationHandler = Arc<dyn Fn(&str, &Value) + Send + Sync>;

#[async_trait]
pub trait McpTransport: Send + Sync {
    /// Send a JSON-RPC request and receive the final response.
    ///
    /// For HTTP, the response may arrive as a single JSON body **or** as an
    /// SSE stream containing interleaved server-initiated requests.  For stdio
    /// the response arrives as newline-delimited JSON on stdout.
    ///
    /// `on_server_request` is called for any server-initiated request that
    /// arrives before the final response.  The returned value is sent back to
    /// the server.
    ///
    /// `on_notification` is called for any notification (no `id`, has `method`).
    async fn request(
        &self,
        body: &Value,
        on_server_request: &ServerRequestHandler,
        on_notification: &NotificationHandler,
    ) -> Result<Value>;

    /// Gracefully shut down the transport (close connection / kill child).
    async fn shutdown(&self) -> Result<()>;
}

// ---------------------------------------------------------------------------
// HTTP (Streamable HTTP) transport
// ---------------------------------------------------------------------------

pub struct HttpTransport {
    url: String,
    api_key: Option<String>,
    http: reqwest::Client,
    session_id: Mutex<Option<String>>,
}

impl HttpTransport {
    pub fn new(url: String, api_key: Option<String>) -> Self {
        Self {
            url,
            api_key,
            http: reqwest::Client::new(),
            session_id: Mutex::new(None),
        }
    }

    /// Build a POST request with standard headers.
    fn build_request(&self, body: &Value, session_id: &Option<String>) -> reqwest::RequestBuilder {
        let mut req = self
            .http
            .post(&self.url)
            .header("content-type", "application/json")
            .header("accept", "application/json, text/event-stream");

        if let Some(key) = &self.api_key {
            req = req.header("authorization", format!("Bearer {key}"));
        }
        if let Some(sid) = session_id {
            req = req.header("mcp-session-id", sid.as_str());
        }
        req.json(body)
    }

    /// Send a JSON-RPC response back to the server (for server-initiated requests).
    async fn send_response(&self, response: &Value, session_id: &Option<String>) -> Result<()> {
        let mut req = self
            .http
            .post(&self.url)
            .header("content-type", "application/json");
        if let Some(sid) = session_id {
            req = req.header("mcp-session-id", sid.as_str());
        }
        req.json(response)
            .send()
            .await
            .context("Failed to send response to MCP server")?;
        Ok(())
    }

    /// Parse an SSE `data:` line into JSON.
    fn parse_sse_data(raw: &str) -> Option<Value> {
        let data_line = raw.lines().find(|l| l.starts_with("data:"))?;
        let data = data_line.strip_prefix("data:").unwrap_or("").trim();
        serde_json::from_str(data).ok()
    }
}

#[async_trait]
impl McpTransport for HttpTransport {
    async fn request(
        &self,
        body: &Value,
        on_server_request: &ServerRequestHandler,
        on_notification: &NotificationHandler,
    ) -> Result<Value> {
        let session_id = self.session_id.lock().await.clone();
        let resp = self
            .build_request(body, &session_id)
            .send()
            .await
            .context("Failed to send request to MCP server")?;

        // Capture session id from response header.
        if let Some(sid) = resp.headers().get("mcp-session-id")
            && let Ok(s) = sid.to_str()
        {
            *self.session_id.lock().await = Some(s.to_string());
        }

        let content_type = resp
            .headers()
            .get("content-type")
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .to_string();

        if content_type.contains("text/event-stream") {
            // SSE stream — read events until we get the final result.
            let current_sid = self.session_id.lock().await.clone();
            let req_id = body.get("id").cloned().unwrap_or(Value::Null);
            let mut stream = resp.bytes_stream();
            let mut buf = String::new();

            loop {
                match stream.next().await {
                    None => bail!("SSE stream ended without a final result"),
                    Some(Err(e)) => bail!("SSE stream error: {e}"),
                    Some(Ok(chunk)) => {
                        buf.push_str(&String::from_utf8_lossy(&chunk));
                        while let Some(pos) = buf.find("\n\n") {
                            let raw = buf[..pos].to_string();
                            buf.drain(..pos + 2);

                            let Some(data) = Self::parse_sse_data(&raw) else {
                                continue;
                            };

                            // Server-initiated request: has both `id` and `method`.
                            if data.get("method").is_some()
                                && data.get("id").is_some()
                                && data.get("result").is_none()
                            {
                                let method = data["method"].as_str().unwrap_or("");
                                let params = data.get("params").cloned().unwrap_or(Value::Null);
                                let mut response = on_server_request(method, &params);
                                // Attach the request id.
                                if let Value::Object(ref mut map) = response {
                                    map.insert("id".to_string(), data["id"].clone());
                                    map.entry("jsonrpc".to_string())
                                        .or_insert_with(|| json!("2.0"));
                                }
                                if let Err(e) = self.send_response(&response, &current_sid).await {
                                    warn!("Failed to send server-request response: {e}");
                                }
                                continue;
                            }

                            // Final result or error for our request.
                            if data.get("id") == Some(&req_id)
                                && (data.get("result").is_some() || data.get("error").is_some())
                            {
                                return Ok(data);
                            }

                            // Notification (no id) — dispatch to handler.
                            if let Some(method) = data.get("method").and_then(|m| m.as_str()) {
                                let params = data.get("params").cloned().unwrap_or(Value::Null);
                                on_notification(method, &params);
                            } else {
                                debug!("SSE notification (unrecognized): {data}");
                            }
                        }
                    }
                }
            }
        } else {
            // Plain JSON response.
            let data: Value = resp.json().await.context("Failed to parse JSON response")?;
            Ok(data)
        }
    }

    async fn shutdown(&self) -> Result<()> {
        // HTTP transport is stateless per-request; nothing to shut down.
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// stdio transport
// ---------------------------------------------------------------------------

pub struct StdioTransport {
    stdin: Mutex<tokio::process::ChildStdin>,
    reader: Mutex<BufReader<tokio::process::ChildStdout>>,
    child: Mutex<tokio::process::Child>,
}

impl StdioTransport {
    pub async fn new(
        command: &str,
        args: &[String],
        env: &HashMap<String, String>,
    ) -> Result<Self> {
        let mut cmd = tokio::process::Command::new(command);
        cmd.args(args)
            .envs(env)
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::null());

        let mut child = cmd
            .spawn()
            .with_context(|| format!("Failed to spawn MCP server process: {command}"))?;

        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| anyhow::anyhow!("Failed to open stdin of child process"))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| anyhow::anyhow!("Failed to open stdout of child process"))?;

        Ok(Self {
            stdin: Mutex::new(stdin),
            reader: Mutex::new(BufReader::new(stdout)),
            child: Mutex::new(child),
        })
    }
}

#[async_trait]
impl McpTransport for StdioTransport {
    async fn request(
        &self,
        body: &Value,
        on_server_request: &ServerRequestHandler,
        on_notification: &NotificationHandler,
    ) -> Result<Value> {
        let req_id = body.get("id").cloned().unwrap_or(Value::Null);

        // Write the request as a single JSON line to stdin.
        {
            let mut stdin = self.stdin.lock().await;
            let mut line = serde_json::to_string(body)?;
            line.push('\n');
            stdin
                .write_all(line.as_bytes())
                .await
                .context("Failed to write to MCP server stdin")?;
            stdin.flush().await?;
        }

        // Read lines from stdout until we get the matching response.
        let mut reader = self.reader.lock().await;
        let mut line = String::new();

        loop {
            line.clear();
            let n = reader
                .read_line(&mut line)
                .await
                .context("Failed to read from MCP server stdout")?;
            if n == 0 {
                bail!("MCP server process closed stdout unexpectedly");
            }

            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }

            let data: Value = match serde_json::from_str(trimmed) {
                Ok(v) => v,
                Err(e) => {
                    debug!("Ignoring non-JSON line from MCP server: {e}");
                    continue;
                }
            };

            // Server-initiated request: has both `id` and `method`.
            if data.get("method").is_some()
                && data.get("id").is_some()
                && data.get("result").is_none()
            {
                let method = data["method"].as_str().unwrap_or("");
                let params = data.get("params").cloned().unwrap_or(Value::Null);
                let mut response = on_server_request(method, &params);
                if let Value::Object(ref mut map) = response {
                    map.insert("id".to_string(), data["id"].clone());
                    map.entry("jsonrpc".to_string())
                        .or_insert_with(|| json!("2.0"));
                }
                // Send response back via stdin.
                let mut stdin = self.stdin.lock().await;
                let mut resp_line = serde_json::to_string(&response)?;
                resp_line.push('\n');
                stdin.write_all(resp_line.as_bytes()).await?;
                stdin.flush().await?;
                continue;
            }

            // Final result or error for our request.
            if data.get("id") == Some(&req_id)
                && (data.get("result").is_some() || data.get("error").is_some())
            {
                return Ok(data);
            }

            // Notification — dispatch to handler.
            if let Some(method) = data.get("method").and_then(|m| m.as_str()) {
                let params = data.get("params").cloned().unwrap_or(Value::Null);
                on_notification(method, &params);
            } else {
                debug!("stdio message (unrecognized): {data}");
            }
        }
    }

    async fn shutdown(&self) -> Result<()> {
        // Drop stdin to signal the child, then kill it.
        drop(self.stdin.lock().await);
        let mut child = self.child.lock().await;
        let _ = child.kill().await;
        Ok(())
    }
}
