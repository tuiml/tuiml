"""Grep algorithm sources."""

from typing import Any, Dict

from .._spec import ToolSpec


def execute_search_source(**kwargs) -> Dict[str, Any]:
    """Search algorithm source code for a text query.

    Backs the ``tuiml_search_source`` tool; delegates to
    ``tuiml.agent.user_algorithms.search_source``.

    Parameters
    ----------
    query : str
        Text to search for. Required (arrives via ``**kwargs``, like all
        parameters below).
    name : str, default=None
        Restrict the search to one algorithm.
    builtin : bool, default=True
        Search built-in TuiML algorithm sources.
    user : bool, default=True
        Search user-created algorithm sources.

    Returns
    -------
    result : dict
        Result dict from ``user_algorithms.search_source`` with
        ``status`` and the matches, or an error dict when ``query`` is
        missing.
    """
    from tuiml.agent import user_algorithms
    if "query" not in kwargs:
        return {"status": "error", "error_type": "ValueError",
                "error": "missing required field: query"}
    return user_algorithms.search_source(
        query=kwargs["query"],
        name=kwargs.get("name"),
        builtin=bool(kwargs.get("builtin", True)),
        user=bool(kwargs.get("user", True)),
    )


SPEC = ToolSpec(
    name='tuiml_search_source',
    description="Grep algorithm source files by regex; returns matching lines with path and line number.",
    input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Regex pattern."},
                "name": {"type": "string", "description": "Limit to one user algorithm."},
                "builtin": {"type": "boolean", "default": True},
                "user": {"type": "boolean", "default": True}
            },
            "required": ["query"],
        },
    # No dedicated output schema; falls back to COMPONENT_OUTPUT_SCHEMA.
    output_schema=None,
    execute=execute_search_source,
    group='code',
    read_only=True, destructive=False,
    idempotent=True, open_world=False,
    reproducible=False,
)
