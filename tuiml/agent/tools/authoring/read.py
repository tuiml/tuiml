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
    description="Full source of a user algorithm (by name) or, with builtin=true, a built-in one "
        "(class name, namespaced hub key such as foundation.ChronosForecaster or sklearn.SVC, "
        "or file stem). Returned raw and with line numbers.",
    input_schema={
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "version": {"type": "string", "description": "Defaults to latest."},
                "builtin": {"type": "boolean", "default": False}
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
