"""OpenAI adapter — TuiML tools in OpenAI's function-calling shape.

Works with both the Chat Completions API (``client.chat.completions.create``)
and the new Agents SDK (``agents.Agent(tools=[...])``). For the Agents SDK, the
recommended path is usually the native MCP integration
(``MCPServerStdio("tuiml-mcp")``); this adapter exists for the Chat Completions
and Responses paths that don't yet understand MCP.

Example
-------
>>> from tuiml.agent import openai as tuiml_openai
>>> from tuiml.agent import invoke as tuiml_invoke
>>> tools = tuiml_openai.get_tools()
>>> # pass tools=tools to client.chat.completions.create(...)
>>> # when the model returns a tool_call, dispatch it via tuiml_invoke(name, **args)
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

from tuiml.agent._core import invoke, iter_tools, load_skill


def get_tools() -> List[Dict[str, Any]]:
    """Return every TuiML workflow tool as an OpenAI function-calling dict.

    Shape::

        {"type": "function",
         "function": {"name": ..., "description": ..., "parameters": {...}}}

    OpenAI accepts this format on both Chat Completions and the Responses API.
    """
    out: List[Dict[str, Any]] = []
    for name, description, schema in iter_tools():
        out.append({
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": schema or {"type": "object", "properties": {}},
            },
        })
    return out


def dispatch_tool_call(tool_call: Any) -> Dict[str, Any]:
    """Execute an OpenAI tool-call object and return the structured result.

    Accepts either a Chat Completions ``tool_calls[i]`` entry (a Pydantic-like
    object with ``.function.name`` and ``.function.arguments``) or a raw dict
    of the same shape.
    """
    if hasattr(tool_call, "function"):
        name = tool_call.function.name
        raw_args = tool_call.function.arguments
    else:
        fn = tool_call.get("function", tool_call)
        name = fn["name"]
        raw_args = fn.get("arguments", "{}")
    args = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
    return invoke(name, **args)


def system_prompt() -> str:
    """Return the canonical TuiML system prompt (SKILL.md) for use as
    ``messages[0] = {"role": "system", "content": tuiml_openai.system_prompt()}``.
    """
    return load_skill()


__all__ = ["get_tools", "dispatch_tool_call", "system_prompt"]
