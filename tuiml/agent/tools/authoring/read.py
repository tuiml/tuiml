"""Read algorithm source."""

from typing import Any, Dict

from .._spec import ToolSpec


def execute_read_algorithm(**kwargs) -> Dict[str, Any]:
    """Read the source code of a user or built-in algorithm.

    Backs the ``tuiml_read_algorithm`` tool; delegates to
    ``tuiml.agent.user_algorithms.read_source``.

    Parameters
    ----------
    name : str
        Algorithm class name. Required (arrives via ``**kwargs``, like
        all parameters below).
    version : str, default=None
        Specific version to read; defaults to the latest.
    builtin : bool, default=False
        Read a built-in TuiML algorithm instead of a user algorithm.

    Returns
    -------
    result : dict
        Result dict from ``user_algorithms.read_source`` with ``status``
        and the source code, or an error dict when ``name`` is missing.
    """
    from tuiml.agent import user_algorithms
    if "name" not in kwargs:
        return {"status": "error", "error_type": "ValueError",
                "error": "missing required field: name"}
    return user_algorithms.read_source(
        name=kwargs["name"],
        version=kwargs.get("version"),
        builtin=bool(kwargs.get("builtin", False)),
    )


SPEC = ToolSpec(
    name='tuiml_read_algorithm',
    description="Return the full source code of any algorithm, user-authored or built-in. "
        "For user algorithms pass the directory name (class name). "
        "For built-in algorithms set builtin=true and pass the class name "
        "(e.g. 'RandomForestClassifier') or file stem (e.g. 'random_forest'). "
        "Source is returned both raw and with line numbers for easy reference. "
        "Built-in algorithms are read-only; use tuiml_create_algorithm to fork them.",
    input_schema={
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Algorithm name (class name or directory name).",
                },
                "version": {
                    "type": "string",
                    "description": "Specific version to read (e.g. '1.0.2'). Defaults to latest.",
                },
                "builtin": {
                    "type": "boolean",
                    "default": False,
                    "description": "Set true to read a built-in tuiml algorithm instead of a user algorithm.",
                },
            },
            "required": ["name"],
        },
    # No dedicated output schema; falls back to COMPONENT_OUTPUT_SCHEMA.
    output_schema=None,
    execute=execute_read_algorithm,
    group='code',
    read_only=True, destructive=False,
    idempotent=True, open_world=False,
    reproducible=False,
)
