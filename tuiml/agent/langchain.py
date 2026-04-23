"""LangChain / LangGraph adapter — TuiML tools as ``StructuredTool`` objects.

LangGraph consumes LangChain tools directly (``ToolNode(tools=[...])``), so
this module covers both frameworks.

Example
-------
>>> from tuiml.agent.langchain import get_tools, system_prompt
>>> from langgraph.prebuilt import create_react_agent
>>> from langchain_anthropic import ChatAnthropic
>>>
>>> agent = create_react_agent(
...     model=ChatAnthropic(model="claude-sonnet-4-6"),
...     tools=get_tools(),
...     prompt=system_prompt(),
... )
>>> agent.invoke({"messages": [("user", "Train a random forest on iris")]})
"""

from __future__ import annotations

from typing import Any, List

from tuiml.agent._core import build_args_model, invoke, iter_tools, load_skill, require


def get_tools() -> List[Any]:
    """Return every TuiML workflow tool wrapped in a LangChain ``StructuredTool``."""
    require("langchain_core", "langchain")
    from langchain_core.tools import StructuredTool  # type: ignore

    out: List[Any] = []
    for name, description, schema in iter_tools():
        args_model = build_args_model(name, schema)
        out.append(_make_structured_tool(StructuredTool, name, description, args_model))
    return out


def _make_structured_tool(StructuredTool: Any, name: str, description: str, args_model: Any) -> Any:
    """Helper that captures ``name`` in its own closure so each tool routes
    to the right executor."""

    def _run(**kwargs: Any) -> Any:
        return invoke(name, **kwargs)

    return StructuredTool.from_function(
        func=_run,
        name=name,
        description=description,
        args_schema=args_model,
    )


def system_prompt() -> str:
    """Return the canonical TuiML system prompt (SKILL.md) for ``prompt=``."""
    return load_skill()


__all__ = ["get_tools", "system_prompt"]
