"""The MCP server that puts TuiML in front of an AI agent.

:mod:`tuiml.agent.mcp.server` speaks the Model Context Protocol over stdio,
advertising the tools defined in :mod:`tuiml.agent.tools` so a client such as
Claude Desktop, Codex or OpenClaw can train, evaluate, benchmark and serve
models by name. Clients spawn it as a child process; it is not a long-running
daemon you start yourself.

Running it
----------
``tuiml setup`` wires it into every AI client it detects, which is the
supported path. The console script it registers is::

    tuiml-mcp

Run that by hand only to check the server starts: with no MCP client on the
other end of stdio it will simply wait.

Notes
-----
The server dispatches through :func:`tuiml.agent.tools.execute_tool` and then
records each successful call for the notebook exporter, so a whole agent
session can be replayed as a runnable notebook. Calling ``execute_tool``
directly skips that recording.

See Also
--------
:mod:`tuiml.agent.tools` : The tool definitions the server advertises.
:mod:`tuiml.agent.adapters` : The same tools for non-MCP frameworks.
"""
