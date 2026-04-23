"""
Entry point for running TuiML MCP server.

Usage:
    uv run python -m tuiml.agent.mcp
    python -m tuiml.agent.mcp
"""

from tuiml.agent.mcp.server import main

if __name__ == "__main__":
    main()
