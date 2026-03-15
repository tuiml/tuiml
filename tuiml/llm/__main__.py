"""
Entry point for running TuiML MCP server.

Usage:
    uv run python -m tuiml.llm
    python -m tuiml.llm
"""

from tuiml.llm.server import main

if __name__ == "__main__":
    main()
