"""List algorithm source files."""

from typing import Any, Dict

from .._spec import ToolSpec


def execute_list_algorithm_files(**kwargs) -> Dict[str, Any]:
    """List algorithm source files, built-in and/or user-created.

    Backs the ``tuiml_list_files`` tool; delegates to
    ``tuiml.agent.user_algorithms.list_algorithm_files``.

    Parameters
    ----------
    builtin : bool, default=True
        Include built-in TuiML algorithm files (arrives via ``**kwargs``,
        like all parameters below).
    user : bool, default=True
        Include user-created algorithm files.

    Returns
    -------
    result : dict
        Result dict from ``user_algorithms.list_algorithm_files`` with
        ``status`` and the file listing.
    """
    from tuiml.agent import user_algorithms
    return user_algorithms.list_algorithm_files(
        builtin=bool(kwargs.get("builtin", True)),
        user=bool(kwargs.get("user", True)),
    )


SPEC = ToolSpec(
    name='tuiml_list_files',
    description="List algorithm source files (built-in and/or user) with paths; use before tuiml_read_algorithm.",
    input_schema={
            "type": "object",
            "properties": {
                "builtin": {"type": "boolean", "default": True},
                "user": {"type": "boolean", "default": True}
            },
        },
    # No dedicated output schema; falls back to COMPONENT_OUTPUT_SCHEMA.
    output_schema=None,
    execute=execute_list_algorithm_files,
    group='code',
    read_only=True, destructive=False,
    idempotent=True, open_world=False,
    reproducible=False,
)
