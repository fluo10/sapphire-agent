pub mod builtin_tools;
pub mod workspace_tools;

use crate::config::McpServerConfig;
use crate::provider::{ToolCall, ToolSpec};
use anyhow::Result;
use async_trait::async_trait;
use std::sync::{Arc, Mutex};

/// A tool the agent can invoke.
#[async_trait]
pub trait Tool: Send + Sync {
    /// The spec advertised to the LLM.
    fn spec(&self) -> &ToolSpec;

    /// Execute the tool with the given JSON input.
    async fn execute(&self, input: &serde_json::Value) -> Result<String>;
}

/// A collection of tools with their specs.
pub struct ToolSet {
    tools: Vec<Box<dyn Tool>>,
    specs: Vec<ToolSpec>,
}

impl ToolSet {
    pub fn new(tools: Vec<Box<dyn Tool>>) -> Self {
        let specs = tools.iter().map(|t| t.spec().clone()).collect();
        Self { tools, specs }
    }

    pub fn specs(&self) -> &[ToolSpec] {
        &self.specs
    }

    /// Execute a tool call; returns a human-readable result string.
    pub async fn execute(&self, call: &ToolCall) -> String {
        for tool in &self.tools {
            if tool.spec().name == call.name {
                return match tool.execute(&call.input).await {
                    Ok(result) => result,
                    Err(e) => format!("Error: {e:#}"),
                };
            }
        }
        format!("Unknown tool: {}", call.name)
    }
}

/// Build the default tool set backed by a sapphire-workspace WorkspaceState.
///
/// `tavily_api_key`: if provided, the `web_search` tool is included.
/// `mcp_servers`: external MCP servers whose tools are registered with the
/// naming convention `mcp__<name>__<tool_name>`.
pub async fn default_tool_set(
    state: Arc<Mutex<sapphire_workspace::WorkspaceState>>,
    tavily_api_key: Option<String>,
    mcp_servers: &[McpServerConfig],
) -> ToolSet {
    use builtin_tools::*;
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
        Box::new(WorkspaceReadTool::new(Arc::clone(&state))),
        Box::new(WorkspaceWriteTool::new(Arc::clone(&state))),
        Box::new(WorkspaceSearchTool::new(Arc::clone(&state))),
        Box::new(WorkspaceSyncTool::new(Arc::clone(&state))),
        Box::new(ReadFileTool::new()),
        Box::new(WriteFileTool::new(Arc::clone(&state))),
        Box::new(DeleteFileTool::new(Arc::clone(&state))),
        Box::new(TerminalTool::new(workspace_root.clone())),
    ];

    if let Some(key) = tavily_api_key {
        tools.push(Box::new(WebSearchTool::new(key)));
    }

    // External MCP server tools
    if !mcp_servers.is_empty() {
        let workspace_root_str = workspace_root.to_string_lossy();
        let mcp_tools =
            crate::mcp_client::create_mcp_tools(mcp_servers, &workspace_root_str).await;
        tools.extend(mcp_tools);
    }

    ToolSet::new(tools)
}
