pub mod builtin_tools;
pub mod timer_tools;
pub mod workspace_tools;

use crate::config::McpServerConfig;
use crate::mcp_client::{self, McpClient, build_tools_for_client};
use crate::provider::{ToolCall, ToolSpec};
use anyhow::Result;
use async_trait::async_trait;
use std::sync::{Arc, Mutex};
use tokio::sync::RwLock;
use tracing::{info, warn};

/// Output of a tool execution.
///
/// Most tools return only text — for those, construct via `String` (impl
/// `From<String>` below). Tools that also need to deliver image bytes to
/// a multimodal model (e.g. `recall_image`, fetching a past image from
/// the cache) populate `images` so the agent runtime can attach them to
/// the user message carrying the tool_result block.
#[derive(Debug, Clone, Default)]
pub struct ToolOutput {
    /// Text content of the tool result.
    pub text: String,
    /// Optional inline image attachments — `(media_type, data_base64)`.
    /// Attached as `ContentPart::Image` parts on the tool_result user
    /// message; only providers that understand image input use them.
    pub images: Vec<(String, String)>,
}

impl From<String> for ToolOutput {
    fn from(text: String) -> Self {
        Self {
            text,
            images: Vec::new(),
        }
    }
}

impl From<&str> for ToolOutput {
    fn from(text: &str) -> Self {
        Self::from(text.to_string())
    }
}

/// A tool the agent can invoke.
#[async_trait]
pub trait Tool: Send + Sync {
    /// The spec advertised to the LLM.
    fn spec(&self) -> &ToolSpec;

    /// Execute the tool with the given JSON input. Used by all tools
    /// that return only text — which is most of them.
    async fn execute(&self, input: &serde_json::Value) -> Result<String>;

    /// Execute the tool and return a `ToolOutput` carrying both text
    /// and any image attachments. The default impl wraps `execute`;
    /// override when the tool needs to return image bytes (e.g.
    /// `recall_image`).
    async fn execute_full(&self, input: &serde_json::Value) -> Result<ToolOutput> {
        Ok(ToolOutput::from(self.execute(input).await?))
    }
}

/// A collection of tools with their specs.
///
/// Tools and specs are behind a `RwLock` so that MCP server tool lists can be
/// refreshed at runtime when a `notifications/tools/list_changed` is received.
pub struct ToolSet {
    inner: RwLock<ToolSetInner>,
    /// MCP clients whose `tools_changed` flag is checked before each turn.
    mcp_clients: Vec<Arc<McpClient>>,
}

struct ToolSetInner {
    tools: Vec<Box<dyn Tool>>,
    specs: Vec<ToolSpec>,
}

impl ToolSet {
    pub fn new(tools: Vec<Box<dyn Tool>>, mcp_clients: Vec<Arc<McpClient>>) -> Self {
        let specs = tools.iter().map(|t| t.spec().clone()).collect();
        Self {
            inner: RwLock::new(ToolSetInner { tools, specs }),
            mcp_clients,
        }
    }

    /// Return a snapshot of the current tool specs.
    pub async fn specs(&self) -> Vec<ToolSpec> {
        self.inner.read().await.specs.clone()
    }

    /// Execute a tool call. The returned `ToolOutput` carries the
    /// text result plus any image attachments the tool produced; the
    /// caller is responsible for assembling them into a tool_result
    /// user message.
    pub async fn execute(&self, call: &ToolCall) -> ToolOutput {
        let inner = self.inner.read().await;
        for tool in &inner.tools {
            if tool.spec().name == call.name {
                return match tool.execute_full(&call.input).await {
                    Ok(output) => output,
                    Err(e) => ToolOutput::from(format!("Error: {e:#}")),
                };
            }
        }
        ToolOutput::from(format!("Unknown tool: {}", call.name))
    }

    /// Check all MCP clients for `tools_changed` flags and refresh their
    /// tools if needed.  Should be called before each LLM turn.
    pub async fn refresh_if_needed(&self) {
        for client in &self.mcp_clients {
            if !client.take_tools_changed() {
                continue;
            }
            if let Err(e) = self.refresh_client_tools(client).await {
                warn!("MCP '{}': failed to refresh tools: {e:#}", client.name());
            }
        }
    }

    /// Re-list a client's tools and swap them into the ToolSet.
    async fn refresh_client_tools(&self, client: &Arc<McpClient>) -> Result<()> {
        info!("MCP '{}': refreshing tool list", client.name());
        let remote_tools = client.list_tools().await?;
        let new_tools = build_tools_for_client(client, remote_tools);
        let prefix = format!("mcp__{}__", client.name());

        let mut inner = self.inner.write().await;
        inner.tools.retain(|t| !t.spec().name.starts_with(&prefix));
        inner.specs.retain(|s| !s.name.starts_with(&prefix));
        for tool in new_tools {
            inner.specs.push(tool.spec().clone());
            inner.tools.push(tool);
        }
        info!(
            "MCP '{}': tool list refreshed ({} total tools)",
            client.name(),
            inner.tools.len()
        );
        Ok(())
    }

    /// Names of configured MCP servers (for tool discovery / error messages).
    pub fn mcp_server_names(&self) -> Vec<String> {
        self.mcp_clients
            .iter()
            .map(|c| c.name().to_string())
            .collect()
    }

    /// Reconnect one MCP server by name and refresh its tool list.
    /// Returns a human-readable status summary.
    pub async fn reconnect_mcp_server(&self, name: &str) -> Result<String> {
        let client = self
            .mcp_clients
            .iter()
            .find(|c| c.name() == name)
            .ok_or_else(|| anyhow::anyhow!("unknown MCP server: {name}"))?;

        client.reconnect().await?;
        self.refresh_client_tools(client).await?;
        Ok(format!(
            "Reconnected MCP server '{name}' and refreshed its tools."
        ))
    }

    /// Register an additional tool after construction.
    pub async fn register_tool(&self, tool: Box<dyn Tool>) {
        let mut inner = self.inner.write().await;
        inner.specs.push(tool.spec().clone());
        inner.tools.push(tool);
    }
}

/// Build the default tool set backed by a sapphire-workspace WorkspaceState.
///
/// `tavily_api_key`: if provided, the `web_search` tool is included.
/// `mcp_servers`: external MCP servers whose tools are registered with the
/// naming convention `mcp__<name>__<tool_name>`.
/// `timer_manager` + `timer_presets`: drive the `timer_*` tools. Manager
/// is shared with `main` so the agent/serve fire dispatchers can be
/// wired in after construction.
pub async fn default_tool_set(
    state: Arc<Mutex<sapphire_workspace::WorkspaceState>>,
    tavily_api_key: Option<String>,
    mcp_servers: &[McpServerConfig],
    timer_manager: Arc<crate::timer::TimerManager>,
    timer_presets: Vec<crate::config::TimerPreset>,
) -> Arc<ToolSet> {
    use builtin_tools::*;
    use timer_tools::*;
    use workspace_tools::*;

    let workspace_root = state
        .lock()
        .expect("WorkspaceState mutex poisoned")
        .workspace
        .root
        .clone();

    let mut tools: Vec<Box<dyn Tool>> = vec![
        Box::new(MemoryAddTool::new(Arc::clone(&state))),
        Box::new(MemoryReadTool::new(Arc::clone(&state))),
        Box::new(MemoryAppendTool::new(Arc::clone(&state))),
        Box::new(MemoryUpdateTool::new(Arc::clone(&state))),
        Box::new(MemoryRemoveTool::new(Arc::clone(&state))),
        Box::new(WorkspaceSearchTool::new(Arc::clone(&state))),
        Box::new(WorkspaceSyncTool::new(Arc::clone(&state))),
        Box::new(FileReadTool::new(Arc::clone(&state))),
        Box::new(FileWriteTool::new(Arc::clone(&state))),
        Box::new(FileAppendTool::new(Arc::clone(&state))),
        Box::new(FileDeleteTool::new(Arc::clone(&state))),
        Box::new(DirListTool::new(Arc::clone(&state))),
        Box::new(DirWalkTool::new(Arc::clone(&state))),
        Box::new(ShellTool::new(workspace_root.clone())),
        Box::new(WeatherTool::new()),
        Box::new(TimerSetTool::new(Arc::clone(&timer_manager))),
        Box::new(TimerPresetTool::new(
            Arc::clone(&timer_manager),
            timer_presets,
        )),
        Box::new(TimerCancelTool::new(Arc::clone(&timer_manager))),
        Box::new(TimerStatusTool::new(Arc::clone(&timer_manager))),
    ];

    if let Some(key) = tavily_api_key {
        tools.push(Box::new(WebSearchTool::new(key)));
    }

    // External MCP server tools
    let mut mcp_clients = Vec::new();
    if !mcp_servers.is_empty() {
        let workspace_root_str = workspace_root.to_string_lossy();
        let (mcp_tools, clients) =
            mcp_client::create_mcp_tools(mcp_servers, &workspace_root_str).await;
        tools.extend(mcp_tools);
        mcp_clients = clients;
    }

    let tool_set = Arc::new(ToolSet::new(tools, mcp_clients));

    // Register the reconnect tool only if at least one MCP server is configured.
    if !mcp_servers.is_empty() {
        let reconnect = Box::new(builtin_tools::McpReconnectTool::new(Arc::downgrade(
            &tool_set,
        )));
        tool_set.register_tool(reconnect).await;
    }

    tool_set
}
