"""CrewAI adapter, TuiML tools as ``BaseTool`` subclasses.

Example
-------
>>> from tuiml.agent.crewai import get_tools
>>> from crewai import Agent, Task, Crew
>>>
>>> analyst = Agent(role="Data Scientist", goal="build good models",
...                 tools=get_tools(), backstory="...", verbose=True)
"""

from __future__ import annotations

from typing import Any, List

from tuiml.agent._core import build_args_model, invoke, iter_tools, load_skill, require


def get_tools() -> List[Any]:
    """Return every TuiML workflow tool as a CrewAI ``BaseTool`` instance.

    Returns
    -------
    list of crewai.tools.BaseTool
        One BaseTool instance per workflow tool, each with a Pydantic args
        schema and a ``_run`` that routes to the TuiML executor. Pass
        directly as ``Agent(tools=get_tools(), ...)``.

    Raises
    ------
    ImportError
        If ``crewai`` is not installed.

    Examples
    --------
    >>> from tuiml.agent.crewai import get_tools
    >>> tools = get_tools()
    >>> [t.name for t in tools][:2]  # doctest: +SKIP
    ['tuiml_train', 'tuiml_predict']
    """
    require("crewai", "crewai")
    from crewai.tools import BaseTool  # type: ignore

    out: List[Any] = []
    for name, description, schema in iter_tools():
        args_model = build_args_model(name, schema)
        out.append(_make_crewai_tool(BaseTool, name, description, args_model))
    return out


def _make_crewai_tool(BaseTool: Any, name: str, description: str, args_model: Any) -> Any:
    """Build a per-tool ``BaseTool`` subclass and return a singleton instance.

    Parameters
    ----------
    BaseTool : type
        The ``crewai.tools.BaseTool`` class (passed in to keep the
        import lazy).
    name : str
        Workflow tool name (e.g. "tuiml_train").
    description : str
        Tool description shown to the model.
    args_model : type
        Pydantic model describing the tool's arguments.

    Returns
    -------
    BaseTool
        Instance of a dynamically created subclass whose ``_run`` invokes
        the named TuiML executor.
    """
    def _run(self: Any, **kwargs: Any) -> Any:  # noqa: ARG001
        return invoke(name, **kwargs)

    tool_cls = type(
        f"{name}Tool",
        (BaseTool,),
        {
            "name": name,
            "description": description,
            "args_schema": args_model,
            "_run": _run,
        },
    )
    return tool_cls()


def system_prompt() -> str:
    """Return the canonical TuiML system prompt (SKILL.md). Typically passed
    into ``Agent(backstory=...)`` or ``Crew(system_prompt=...)``.

    Returns
    -------
    str
        The contents of the canonical ``SKILL.md`` system prompt.
    """
    return load_skill()


__all__ = ["get_tools", "system_prompt"]
