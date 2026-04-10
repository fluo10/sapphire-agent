//! Built-in MCP client for connecting to external MCP servers.
//!
//! Each configured MCP server's tools are registered in the agent's `ToolSet`
//! using the naming convention `mcp__<server_name>__<tool_name>`.

pub mod transport;

use crate::config::{McpServerConfig, McpTransportConfig};
use crate::provider::ToolSpec;
use crate::tools::Tool;
use anyhow::{Context, Result, bail};
use async_trait::async_trait;
use serde_json::{Value, json};
use std::borrow::Cow;
use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::{info, warn};
use transport::{HttpTransport, McpTransport, ServerRequestHandler, StdioTransport};

// ---------------------------------------------------------------------------
// Remote tool metadata
// ---------------------------------------------------------------------------

/// A tool specification retrieved from a remote MCP server.
pub struct RemoteToolSpec {
    pub name: String,
    pub description: String,
    pub input_schema: Value,
}

// ---------------------------------------------------------------------------
// MCP Client
// ---------------------------------------------------------------------------

/// Client for a single external MCP server.
pub struct McpClient {
    name: String,
    transport: Box<dyn McpTransport>,
    workspace_root: String,
    request_id: Mutex<u64>,
}

impl McpClient {
    /// Create a new client and establish the transport.
    pub async fn new(config: &McpServerConfig, workspace_root: &str) -> Result<Self> {
        let transport: Box<dyn McpTransport> = match &config.transport {
            McpTransportConfig::Http { url, api_key } => {
                Box::new(HttpTransport::new(url.clone(), api_key.clone()))
            }
            McpTransportConfig::Stdio { command, args, env } => {
                Box::new(StdioTransport::new(command, args, env).await?)
            }
        };

        Ok(Self {
            name: config.name.clone(),
            transport,
            workspace_root: workspace_root.to_string(),
            request_id: Mutex::new(1),
        })
    }

    /// Get the next request ID.
    async fn next_id(&self) -> u64 {
        let mut id = self.request_id.lock().await;
        let current = *id;
        *id += 1;
        current
    }

    /// Build the server-request handler that handles Elicitation, Roots, and
    /// Sampling callbacks from the MCP server.
    fn server_request_handler(&self) -> ServerRequestHandler {
        let workspace_root = self.workspace_root.clone();
        Arc::new(move |method: &str, params: &Value| -> Value {
            match method {
                "roots/list" => {
                    json!({
                        "result": {
                            "roots": [{
                                "uri": format!("file://{workspace_root}"),
                                "name": "workspace"
                            }]
                        }
                    })
                }
                "elicitation/create" => {
                    // Auto-accept with the message echoed back.
                    let message = params
                        .get("message")
                        .and_then(|v| v.as_str())
                        .unwrap_or("");
                    json!({
                        "result": {
                            "action": "accept",
                            "content": message
                        }
                    })
                }
                "sampling/createMessage" => {
                    json!({
                        "error": {
                            "code": -32601,
                            "message": "Sampling is not supported by this client"
                        }
                    })
                }
                _ => {
                    json!({
                        "error": {
                            "code": -32601,
                            "message": format!("Unknown method: {method}")
                        }
                    })
                }
            }
        })
    }

    /// Send a JSON-RPC request through the transport.
    async fn send(&self, method: &str, params: Value) -> Result<Value> {
        let id = self.next_id().await;
        let body = json!({
            "jsonrpc": "2.0",
            "id": id,
            "method": method,
            "params": params,
        });

        let handler = self.server_request_handler();
        let response = self.transport.request(&body, &handler).await?;

        if let Some(err) = response.get("error") {
            let msg = err["message"].as_str().unwrap_or("unknown error");
            let code = err["code"].as_i64().unwrap_or(-1);
            bail!("MCP server error {code}: {msg}");
        }

        Ok(response.get("result").cloned().unwrap_or(Value::Null))
    }

    /// Initialize the MCP session (handshake).
    pub async fn connect(&self) -> Result<()> {
        let params = json!({
            "protocolVersion": "2025-03-26",
            "capabilities": {
                "roots": { "listChanged": false },
                "elicitation": {}
            },
            "clientInfo": {
                "name": "sapphire-agent",
                "version": env!("CARGO_PKG_VERSION")
            }
        });

        let result = self.send("initialize", params).await?;
        info!(
            "MCP '{}': connected (server: {})",
            self.name,
            result
                .get("serverInfo")
                .and_then(|s| s.get("name"))
                .and_then(|n| n.as_str())
                .unwrap_or("unknown")
        );

        // Send initialized notification (no id, no response expected).
        // For stdio, we write directly; for HTTP we send as a POST.
        let notification = json!({
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
        });
        // Best-effort notification — ignore errors.
        let handler = self.server_request_handler();
        let _ = self.transport.request(&notification, &handler).await;

        Ok(())
    }

    /// List tools available on the remote MCP server.
    pub async fn list_tools(&self) -> Result<Vec<RemoteToolSpec>> {
        let result = self.send("tools/list", json!({})).await?;
        let tools = result
            .get("tools")
            .and_then(|v| v.as_array())
            .cloned()
            .unwrap_or_default();

        let specs: Vec<RemoteToolSpec> = tools
            .into_iter()
            .filter_map(|t| {
                let name = t.get("name")?.as_str()?.to_string();
                let description = t
                    .get("description")
                    .and_then(|d| d.as_str())
                    .unwrap_or("")
                    .to_string();
                let input_schema = t.get("inputSchema").cloned().unwrap_or(json!({}));
                Some(RemoteToolSpec {
                    name,
                    description,
                    input_schema,
                })
            })
            .collect();

        info!("MCP '{}': found {} tools", self.name, specs.len());
        Ok(specs)
    }

    /// Call a tool on the remote MCP server.
    pub async fn call_tool(&self, name: &str, arguments: &Value) -> Result<Value> {
        let params = json!({
            "name": name,
            "arguments": arguments,
        });
        self.send("tools/call", params).await
    }

    /// Shut down the transport.
    pub async fn shutdown(&self) -> Result<()> {
        self.transport.shutdown().await
    }
}

// ---------------------------------------------------------------------------
// McpTool — wraps a single remote tool as a local Tool impl
// ---------------------------------------------------------------------------

/// A Tool implementation that delegates to a remote MCP server.
pub struct McpTool {
    client: Arc<McpClient>,
    spec: ToolSpec,
    remote_tool_name: String,
}

#[async_trait]
impl Tool for McpTool {
    fn spec(&self) -> &ToolSpec {
        &self.spec
    }

    async fn execute(&self, input: &serde_json::Value) -> Result<String> {
        let result = self.client.call_tool(&self.remote_tool_name, input).await?;

        // MCP tools/call returns { content: [...] } where each item has
        // type "text" with a text field.  Concatenate all text content.
        if let Some(contents) = result.get("content").and_then(|c| c.as_array()) {
            let texts: Vec<&str> = contents
                .iter()
                .filter_map(|c| {
                    if c.get("type").and_then(|t| t.as_str()) == Some("text") {
                        c.get("text").and_then(|t| t.as_str())
                    } else {
                        None
                    }
                })
                .collect();
            if !texts.is_empty() {
                return Ok(texts.join("\n"));
            }
        }

        // Fallback: pretty-print the raw result.
        Ok(serde_json::to_string_pretty(&result)?)
    }
}

// ---------------------------------------------------------------------------
// Factory: connect to configured MCP servers and collect their tools
// ---------------------------------------------------------------------------

/// Connect to all configured MCP servers and return their tools wrapped as
/// local `Tool` implementations.
pub async fn create_mcp_tools(
    configs: &[McpServerConfig],
    workspace_root: &str,
) -> Vec<Box<dyn Tool>> {
    let mut tools: Vec<Box<dyn Tool>> = Vec::new();

    for config in configs {
        let client = match McpClient::new(config, workspace_root).await {
            Ok(c) => Arc::new(c),
            Err(e) => {
                warn!("MCP '{}': failed to create client: {e:#}", config.name);
                continue;
            }
        };

        if let Err(e) = client.connect().await {
            warn!("MCP '{}': failed to connect: {e:#}", config.name);
            continue;
        }

        match client.list_tools().await {
            Ok(remote_tools) => {
                for rt in remote_tools {
                    let tool_name = format!("mcp__{}__{}", config.name, rt.name);
                    tools.push(Box::new(McpTool {
                        client: Arc::clone(&client),
                        spec: ToolSpec {
                            name: Cow::Owned(tool_name),
                            description: Cow::Owned(rt.description),
                            input_schema: rt.input_schema,
                        },
                        remote_tool_name: rt.name,
                    }));
                }
            }
            Err(e) => {
                warn!("MCP '{}': failed to list tools: {e:#}", config.name);
            }
        }
    }

    tools
}
