"""Pydantic-AI adapter — TuiML tools + a one-liner ``tuiml.agent()`` agent.

Pydantic-AI is model-agnostic (works with Anthropic, OpenAI, Google, Groq,
etc.), uses Pydantic models for schemas natively, and is our chosen substrate
for the one-liner agent. This module exposes both the plain tools list and a
pre-wired ``Agent`` instance.

Example — plug tools into your own agent
----------------------------------------
>>> from tuiml.agent.pydantic_ai import get_tools, system_prompt
>>> from pydantic_ai import Agent
>>> agent = Agent("anthropic:claude-sonnet-4-6",
...               tools=get_tools(), system_prompt=system_prompt())
>>> agent.run_sync("Train a random forest on iris and report accuracy")

Example — one-liner
-------------------
>>> import tuiml
>>> tuiml.agent().run_sync("Predict churn on customers.csv")
"""

from __future__ import annotations

from typing import Any, List, Optional

from tuiml.agent._core import build_args_model, invoke, iter_tools, load_skill, require


def get_tools() -> List[Any]:
    """Return every TuiML workflow tool as a ``pydantic_ai.Tool``."""
    require("pydantic_ai", "pydantic-ai")
    from pydantic_ai import Tool  # type: ignore

    out: List[Any] = []
    for name, description, schema in iter_tools():
        args_model = build_args_model(name, schema)
        out.append(_make_tool(Tool, name, description, args_model))
    return out


def _make_tool(Tool: Any, name: str, description: str, args_model: Any) -> Any:
    """Build one ``pydantic_ai.Tool`` using a closure that captures ``name``."""
    from pydantic_ai import RunContext  # type: ignore

    async def _run(ctx: RunContext[Any], **kwargs: Any) -> Any:  # noqa: ARG001
        return invoke(name, **kwargs)

    # Pydantic-AI's Tool accepts either a function (infers schema from
    # signature/docstring) or an explicit schema. Prefer the explicit path
    # so we stay aligned with MCP/OpenAI/Anthropic.
    return Tool(
        _run,
        name=name,
        description=description,
    )


def system_prompt() -> str:
    """Return the canonical TuiML system prompt (SKILL.md)."""
    return load_skill()


def agent(model: Optional[str] = None, **kwargs: Any) -> Any:
    """Return a ready-to-run ``pydantic_ai.Agent`` pre-loaded with every
    TuiML tool and the canonical system prompt.

    Parameters
    ----------
    model : str, optional
        A Pydantic-AI model string, e.g. ``"anthropic:claude-sonnet-4-6"`` or
        ``"openai:gpt-4o"``. Defaults to ``"anthropic:claude-sonnet-4-6"``;
        the Pydantic-AI environment must be configured with the matching
        provider API key (``ANTHROPIC_API_KEY``, ``OPENAI_API_KEY``, …).
    **kwargs
        Passed through to ``pydantic_ai.Agent``.

    Example
    -------
    >>> import tuiml
    >>> result = tuiml.agent().run_sync("Compare RandomForestClassifier and XGBoost on iris.")
    >>> print(result.output)
    """
    require("pydantic_ai", "pydantic-ai")
    from pydantic_ai import Agent  # type: ignore

    return Agent(
        model or "anthropic:claude-sonnet-4-6",
        tools=get_tools(),
        system_prompt=system_prompt(),
        **kwargs,
    )


__all__ = ["get_tools", "system_prompt", "agent"]
