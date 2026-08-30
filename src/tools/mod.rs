pub mod ambient_tools;
pub mod builtin_tools;
pub mod timer_tools;
pub mod workspace_tools;

use crate::config::McpServerConfig;
use crate::mcp_client::{self, McpClient, build_tools_for_client};
use crate::provider::{ToolCall, ToolSpec};
use anyhow::Result;
use async_trait::async_trait;
pub use agent_client_protocol::schema::v1::ToolKind;
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

    /// What this tool does, in ACP's vocabulary. Drives both the
    /// `session/update` display and the permission policy, so there is
    /// exactly one classification rather than two that drift apart.
    ///
    /// The default is `Other` — the strictest bucket — so a tool added
    /// without a `kind()` fails safe: it asks (ACP) or is refused
    /// (channels) rather than silently running unguarded.
    fn kind(&self) -> ToolKind {
        ToolKind::Other
    }

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

    /// Every registered tool's name and kind. Exists so the policy test
    /// can pin the whole classification table in one assertion rather
    /// than constructing each tool by hand.
    pub async fn kinds(&self) -> Vec<(String, ToolKind)> {
        self.inner
            .read()
            .await
            .tools
            .iter()
            .map(|t| (t.spec().name.to_string(), t.kind()))
            .collect()
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

/// Build the default tool set backed by a sapphire-framework WorkspaceState.
///
/// `tavily_api_key`: if provided, the `web_search` tool is included.
/// `mcp_servers`: external MCP servers whose tools are registered with the
/// naming convention `mcp__<name>__<tool_name>`.
/// `timer_manager` + `timer_presets`: drive the `timer_*` tools. Manager
/// is shared with `main` so the agent/serve fire dispatchers can be
/// wired in after construction.
pub async fn default_tool_set(
    state: Arc<Mutex<sapphire_framework::workspace::WorkspaceState>>,
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

#[cfg(test)]
mod tests {
    use super::*;
    use sapphire_framework::workspace::{AppContext, Workspace, WorkspaceState};
    use std::sync::Mutex;

    static TEST_CTX: AppContext = AppContext::new("sapphire-agent").allow_external_paths();

    fn test_workspace() -> Arc<Mutex<WorkspaceState>> {
        // AppContext panics on any cache_dir() access until set_cache_dir has
        // been called once; Workspace::from_root reads it to compute the
        // workspace's cache path. set_cache_dir is "first writer wins" and
        // silently ignores later calls, so it is safe to call on every
        // invocation of this helper (matches sapphire-framework's own
        // workspace_state.rs test pattern).
        TEST_CTX.set_cache_dir(std::env::temp_dir().join("sapphire-agent-tools-test-cache"));
        // Leaked on purpose: this is a test binary and the OS reclaims
        // the directory when it exits.
        let dir = Box::leak(Box::new(tempfile::tempdir().unwrap()));
        std::fs::create_dir_all(dir.path().join(".sapphire-agent")).unwrap();
        let ws = Workspace::from_root(&TEST_CTX, dir.path()).unwrap();
        Arc::new(Mutex::new(WorkspaceState::open(ws).unwrap()))
    }

    /// Every tool declares what it does. A tool added without a `kind()`
    /// lands in `Other` — the strictest bucket — so this table failing
    /// on a newly added tool is the intended prompt to classify it.
    #[tokio::test]
    async fn every_tool_declares_its_kind() {
        let tools = default_tool_set(
            test_workspace(),
            Some("test-tavily-key".to_string()),
            &[],
            crate::timer::TimerManager::new(),
            Vec::new(),
        )
        .await;

        let mut got = tools.kinds().await;
        got.sort_by(|a, b| a.0.cmp(&b.0));
        let got_refs: Vec<(&str, ToolKind)> =
            got.iter().map(|(n, k)| (n.as_str(), *k)).collect();

        let want: Vec<(&str, ToolKind)> = vec![
            ("dir_list", ToolKind::Search),
            ("dir_walk", ToolKind::Search),
            ("file_append", ToolKind::Edit),
            ("file_delete", ToolKind::Delete),
            ("file_read", ToolKind::Read),
            ("file_write", ToolKind::Edit),
            ("memory_add", ToolKind::Edit),
            ("memory_append", ToolKind::Edit),
            ("memory_read", ToolKind::Read),
            ("memory_remove", ToolKind::Delete),
            ("memory_update", ToolKind::Edit),
            ("shell", ToolKind::Execute),
            ("timer_cancel", ToolKind::Delete),
            ("timer_preset", ToolKind::Edit),
            ("timer_set", ToolKind::Edit),
            ("timer_status", ToolKind::Search),
            ("weather", ToolKind::Fetch),
            ("web_search", ToolKind::Fetch),
            ("workspace_search", ToolKind::Search),
            ("workspace_sync", ToolKind::Other),
        ];

        assert_eq!(got_refs, want);
    }

    /// A tool that does not override `kind()` must land in the strictest
    /// bucket, so forgetting to classify one fails safe.
    #[test]
    fn the_default_kind_is_other() {
        struct Bare(ToolSpec);
        #[async_trait]
        impl Tool for Bare {
            fn spec(&self) -> &ToolSpec {
                &self.0
            }
            async fn execute(&self, _input: &serde_json::Value) -> Result<String> {
                Ok(String::new())
            }
        }
        let bare = Bare(ToolSpec {
            name: "bare".into(),
            description: String::new().into(),
            input_schema: serde_json::json!({}),
        });
        assert_eq!(bare.kind(), ToolKind::Other);
    }
}
