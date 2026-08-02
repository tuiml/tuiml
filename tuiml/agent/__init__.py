"""
LLM Integration for TuiML.

Provides MCP (Model Context Protocol) server and tools for LLM integration.
Workflow/discovery tools are exposed as MCP tools, and ALL 200+
components are accessible through them. New algorithms registered via
decorators or the component registry are automatically discoverable.

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
- tuiml_benchmark - Compare algorithms
- tuiml_upload_data - Upload dataset content

**Discovery Tools:**
- tuiml_list - List all components; pass search= to filter by keyword
- tuiml_describe - Get component details and parameter schema

All 200+ algorithms, preprocessors, datasets, features, and splitters
are accessible through these tools. When a new algorithm is added to
the component registry or registered with a decorator, it is
automatically available.

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

from tuiml.agent.tools._components import (
    get_all_tools,
    get_tool,
    list_tools_by_category,
    get_tool_count,
    ToolDefinition,
)

# MCP server availability check. Checked here rather than imported from
# tuiml.agent.mcp.server, which imports back from tuiml.agent.tools; the
# server module is only reachable lazily from inside the functions below.
try:
    from mcp.server import Server
    # A 1.x SDK imports cleanly but cannot serve: it registers handlers by
    # decorator, and tuiml.agent.mcp.server is written against the 2.x
    # constructor API. ``Server.list_tools`` is the 1.x-only decorator.
    MCP_AVAILABLE = not hasattr(Server, "list_tools")
except ImportError:
    MCP_AVAILABLE = False

def get_mcp_server():
    """Get the MCP server (lazy import to avoid circular import).

    Returns
    -------
    Server
        Configured MCP server exposing the TuiML workflow and
        discovery tools.

    Raises
    ------
    ImportError
        If the 2.x ``mcp`` package is not installed.
    """
    if not MCP_AVAILABLE:
        raise ImportError(
            "TuiML's MCP server requires the 2.x MCP SDK. "
            "Install it with: pip install 'mcp>=2'"
        )
    from tuiml.agent.mcp.server import create_server
    return create_server()

def run_mcp_server():
    """Run the MCP server over stdio transport.

    Returns
    -------
    None
        Blocks until the server process exits.
    """
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
        List of tool schema dicts, one per workflow/discovery tool, each
        with keys ``"name"``, ``"description"``, and ``"inputSchema"``
        (a JSON Schema object).

    Examples
    --------
    >>> from tuiml.agent import get_tools_for_llm
    >>> tools = get_tools_for_llm()
    >>> len(tools)
    len(tools) > 0
    True
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
# Framework-agnostic helpers (re-exported from tuiml.agent.adapters._base)
# ---------------------------------------------------------------------------
from tuiml.agent.adapters._base import invoke, callables, load_skill


# ---------------------------------------------------------------------------
# One-liner agent (Pydantic-AI substrate)
# ---------------------------------------------------------------------------

def agent(model: "Optional[str]" = None, **kwargs):  # type: ignore[name-defined]
    """Return a ready-to-run Pydantic-AI ``Agent`` pre-loaded with every
    TuiML tool and the canonical ``SKILL.md`` system prompt.

    Requires ``pip install tuiml[pydantic-ai]``.

    Parameters
    ----------
    model : str, optional
        A Pydantic-AI model string, e.g. ``"anthropic:claude-sonnet-4-6"``
        or ``"openai:gpt-4o"``. Defaults to ``"anthropic:claude-sonnet-4-6"``.
    **kwargs
        Passed through to ``pydantic_ai.Agent``.

    Returns
    -------
    pydantic_ai.Agent
        Agent configured with all TuiML workflow tools and the canonical
        system prompt.

    Examples
    --------
    >>> from tuiml.agent import agent
    >>> result = agent().run_sync(
    ...     "Train RandomForestClassifier on iris and report accuracy."
    ... )
    >>> print(result.output)
    """
    from tuiml.agent.adapters.pydantic_ai import agent as _agent
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
