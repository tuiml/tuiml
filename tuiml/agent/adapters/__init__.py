"""The TuiML tools, for agent frameworks that do not speak MCP.

:mod:`tuiml.agent.mcp` covers clients that support the Model Context
Protocol. These adapters cover the rest: if you are building an agent in
Python rather than configuring one, they hand you the same tools in the shape
your framework expects, with no MCP server involved.

Adapters
--------
- :mod:`~tuiml.agent.adapters.openai` — OpenAI function-calling schemas.
- :mod:`~tuiml.agent.adapters.anthropic` — Anthropic tool definitions.
- :mod:`~tuiml.agent.adapters.langchain` — LangChain ``BaseTool`` objects.
- :mod:`~tuiml.agent.adapters.crewai` — CrewAI tools.
- :mod:`~tuiml.agent.adapters.pydantic_ai` — Pydantic-AI tools.

Each exposes ``get_tools()``, returning every tool — the workflow tools plus
``tuiml_list`` and ``tuiml_describe``, so an agent can discover what exists
rather than being told in its prompt.

Installation
------------
Each framework is an optional extra, imported lazily::

    pip install 'tuiml[openai]'        # or anthropic, langchain,
                                       # pydantic-ai, crewai
    pip install 'tuiml[frameworks]'    # all five at once

Examples
--------
>>> from tuiml.agent.adapters.openai import get_tools   # doctest: +SKIP
>>> tools = get_tools()                                 # doctest: +SKIP

See Also
--------
:mod:`tuiml.agent.tools` : The tool definitions all five adapters expose.
:mod:`tuiml.agent.mcp` : The MCP server, for clients that speak MCP.
"""
