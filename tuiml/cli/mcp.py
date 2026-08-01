"""``tuiml mcp``: run the TuiML MCP server in the foreground (for debugging).

Equivalent to invoking the ``tuiml-mcp`` binary or ``python -m tuiml.agent.mcp``
directly, but exposed under the unified ``tuiml`` CLI so users don't have to
remember the second executable name. Useful when an AI client launches
``tuiml-mcp`` from a config entry and you want to inspect its stdout/stderr by
hand.
"""
from __future__ import annotations

import click


@click.command()
def mcp() -> None:
    """Run the TuiML MCP server in the foreground (stdio transport).

    AI clients normally spawn ``tuiml-mcp`` themselves and talk to it over
    stdio, which hides its output. Running the server by hand keeps it in the
    foreground where the handshake, log lines, and startup errors are visible;
    press Ctrl-C to stop it. This is the same server as the ``tuiml-mcp``
    executable and ``python -m tuiml.agent.mcp``.

    ``tuiml setup`` registers the server with the AI clients it finds. To wire
    one up by hand, add an entry like this to the client's MCP config
    (``claude_desktop_config.json`` for Claude Desktop)::

        {"mcpServers": {"tuiml": {"command": "tuiml-mcp"}}}

    Examples
    --------
    Start the server and watch its stdio traffic:

    $ tuiml mcp

    Check which clients already have a server running instead of starting one:

    $ tuiml status

    Follow the tool calls a connected client is making:

    $ tuiml trace -f
    """
    # Import lazily so `tuiml --help` doesn't pay the import cost.
    from tuiml.agent.mcp.server import main as mcp_main
    mcp_main()
