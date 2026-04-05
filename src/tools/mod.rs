pub mod workspace_tools;

use crate::provider::{ToolCall, ToolSpec};
use anyhow::Result;
use std::sync::{Arc, Mutex};

/// A tool the agent can invoke.
pub trait Tool: Send + Sync {
    /// The spec advertised to the LLM.
    fn spec(&self) -> &ToolSpec;

    /// Execute the tool with the given JSON input.
    fn execute(&self, input: &serde_json::Value) -> Result<String>;
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
    pub fn execute(&self, call: &ToolCall) -> String {
        for tool in &self.tools {
            if tool.spec().name == call.name {
                return match tool.execute(&call.input) {
                    Ok(result) => result,
                    Err(e) => format!("Error: {e:#}"),
                };
            }
        }
        format!("Unknown tool: {}", call.name)
    }
}


/// Build the default tool set backed by a sapphire-workspace WorkspaceState.
pub fn default_tool_set(
    state: Arc<Mutex<sapphire_workspace::WorkspaceState>>,
) -> ToolSet {
    use workspace_tools::*;
    ToolSet::new(vec![
        Box::new(MemoryAppendTool::new(Arc::clone(&state))),
        Box::new(MemoryWriteTool::new(Arc::clone(&state))),
        Box::new(WorkspaceReadTool::new(Arc::clone(&state))),
        Box::new(WorkspaceWriteTool::new(Arc::clone(&state))),
        Box::new(WorkspaceSearchTool::new(Arc::clone(&state))),
        Box::new(WorkspaceSyncTool::new(Arc::clone(&state))),
    ])
}
