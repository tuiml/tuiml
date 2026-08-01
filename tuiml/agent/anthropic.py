"""Anthropic adapter, TuiML tools in the Messages API tools shape.

Anthropic's native MCP support already lets Claude Desktop / Claude Code
consume the ``tuiml-mcp`` server directly. This adapter is for code that calls
the Messages API (``client.messages.create``) and wants to hand-roll the
tool-use loop, e.g., server-side agents, custom UIs, or Anthropic SDK usage
outside an MCP-aware client.

Example
-------
>>> from tuiml.agent import anthropic as tuiml_anthropic
>>> from tuiml.agent import invoke as tuiml_invoke
>>> tools = tuiml_anthropic.get_tools()
>>> # In your tool-use loop: when response.stop_reason == "tool_use",
>>> # iterate response.content for tool_use blocks and call tuiml_anthropic.dispatch_tool_use(block).
"""

from __future__ import annotations

from typing import Any, Dict, List

from tuiml.agent._core import invoke, iter_tools, load_skill


def get_tools() -> List[Dict[str, Any]]:
    """Return every TuiML workflow tool as an Anthropic Messages tool dict.

    Returns
    -------
    list of dict
        One dict per workflow tool in the Messages API tools shape::

            {"name": ..., "description": ..., "input_schema": {...}}

        where ``"input_schema"`` is a JSON Schema object. Pass this list
        directly to ``client.messages.create(tools=tools, ...)``.

    Examples
    --------
    >>> from tuiml.agent import anthropic as tuiml_anthropic
    >>> tools = tuiml_anthropic.get_tools()
    >>> response = client.messages.create(
    ...     model="claude-sonnet-4-6",
    ...     system=tuiml_anthropic.system_prompt(),
    ...     tools=tools,
    ...     messages=[{"role": "user", "content": "Train a model on iris"}],
    ...     max_tokens=1024,
    ... )
    """
    out: List[Dict[str, Any]] = []
    for name, description, schema in iter_tools():
        out.append({
            "name": name,
            "description": description,
            "input_schema": schema or {"type": "object", "properties": {}},
        })
    return out


def dispatch_tool_use(block: Any) -> Dict[str, Any]:
    """Execute an Anthropic ``tool_use`` content block and return the result.

    Parameters
    ----------
    block : Any
        Either an SDK block object (with ``.name`` and ``.input``) or the
        raw dict form ``{"type": "tool_use", "name": ..., "input": ...}``.

    Returns
    -------
    dict
        The TuiML structured result dict for the tool call (with a
        ``"status"`` key plus tool-specific fields).
    """
    if hasattr(block, "name"):
        name = block.name
        args = block.input or {}
    else:
        name = block["name"]
        args = block.get("input") or {}
    return invoke(name, **args)


def system_prompt() -> str:
    """Return the canonical TuiML system prompt (SKILL.md) to pass as the
    ``system=`` parameter on ``client.messages.create``.

    Returns
    -------
    str
        The contents of the canonical ``SKILL.md`` system prompt.
    """
    return load_skill()


__all__ = ["get_tools", "dispatch_tool_use", "system_prompt"]
