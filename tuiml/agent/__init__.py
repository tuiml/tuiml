"""
LLM Integration for TuiML.

Provides MCP (Model Context Protocol) server and tools for LLM integration.
Only 11 workflow/discovery tools are exposed as MCP tools, but ALL 200+
components are accessible through them. New algorithms registered via
decorators or the hub are automatically discoverable.

Quick Start
-----------
Run the MCP server::

    python -m tuiml.agent.mcp.server

Or configure in Claude Desktop (claude_desktop_config.json)::

    {
        "mcpServers": {
            "tuiml": {
                "command": "tuiml-mcp"
            }
        }
    }

Exposed MCP Tools
-----------------
**Workflow Tools:**
- tuiml_train - Train ML models with full workflow
- tuiml_predict - Make predictions
- tuiml_evaluate - Evaluate models
- tuiml_experiment - Compare algorithms
- tuiml_upload_data - Upload dataset content

**Discovery Tools:**
- tuiml_list - List all components (algorithms, preprocessors, etc.)
- tuiml_describe - Get component details and parameter schema
- tuiml_search - Search components by keyword

All 200+ algorithms, preprocessors, datasets, features, and splitters
are accessible through these tools. When a new algorithm is added to
the hub or registered with a decorator, it is automatically available.

Example Usage
-------------
>>> from tuiml.agent import execute_tool
>>>
>>> # Train any algorithm by name
>>> result = execute_tool(
...     "tuiml_train",
...     algorithm="RandomForestClassifier",
...     data="iris",
...     target="class",
...     cv=10
... )
>>>
>>> # List available algorithms
>>> result = execute_tool("tuiml_list", category="algorithm")
"""

# Core exports
from tuiml.agent.tools import (
    execute_tool,
    get_workflow_tools,
    WORKFLOW_TOOLS,
    DISCOVERY_TOOLS,
)

from tuiml.agent.registry import (
    get_all_tools,
    get_tool,
    list_tools_by_category,
    get_tool_count,
    ToolDefinition,
)

# MCP server availability check
try:
    from mcp.server import Server
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False

def get_mcp_server():
    """Get the MCP server (lazy import to avoid circular import)."""
    if not MCP_AVAILABLE:
        raise ImportError("MCP package not installed. Install with: pip install mcp")
    from tuiml.agent.mcp.server import create_server
    return create_server()

def run_mcp_server():
    """Run the MCP server."""
    from tuiml.agent.mcp.server import main
    main()

def get_tools_for_llm(format: str = "mcp") -> list:
    """
    Get all tool schemas formatted for LLM consumption.

    Returns only the workflow/discovery tools (not 200+ component tools).
    Components are accessible via tuiml_train, tuiml_list, etc.

    Parameters
    ----------
    format : str, default="mcp"
        Output format. Currently only "mcp" is supported.

    Returns
    -------
    list
        List of tool schemas ready for LLM tool calling.

    Examples
    --------
    >>> tools = get_tools_for_llm()
    >>> len(tools)
    11
    """
    tools = []

    for name, schema in get_workflow_tools().items():
        tools.append({
            "name": name,
            "description": schema["description"],
            "inputSchema": schema["inputSchema"]
        })

    return tools

# ---------------------------------------------------------------------------
# Framework-agnostic helpers (re-exported from tuiml.agent._core)
# ---------------------------------------------------------------------------
from tuiml.agent._core import invoke, callables, load_skill


# ---------------------------------------------------------------------------
# One-liner agent (Pydantic-AI substrate)
# ---------------------------------------------------------------------------

def agent(model: "Optional[str]" = None, **kwargs):  # type: ignore[name-defined]
    """Return a ready-to-run Pydantic-AI ``Agent`` pre-loaded with every
    TuiML tool and the canonical ``SKILL.md`` system prompt.

    Requires ``pip install tuiml[pydantic-ai]``.

    Example
    -------
    >>> import tuiml
    >>> result = tuiml.agent().run_sync(
    ...     "Train RandomForestClassifier on iris and report accuracy."
    ... )
    >>> print(result.output)
    """
    from tuiml.agent.pydantic_ai import agent as _agent
    return _agent(model=model, **kwargs)


__all__ = [
    # Tool execution
    "execute_tool",
    "get_workflow_tools",
    "get_all_tools",
    "get_tool",
    "list_tools_by_category",
    "get_tool_count",
    "get_tools_for_llm",
    # Tool schemas
    "WORKFLOW_TOOLS",
    "DISCOVERY_TOOLS",
    "ToolDefinition",
    # MCP server
    "get_mcp_server",
    "run_mcp_server",
    "MCP_AVAILABLE",
    # Framework-agnostic helpers
    "invoke",
    "callables",
    "load_skill",
    # One-liner agent
    "agent",
]
