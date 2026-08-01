"""Delete a user algorithm."""

from typing import Any, Dict

from .._spec import ToolSpec


def execute_delete_user_algorithm(**kwargs) -> Dict[str, Any]:
    """Delete a user-created algorithm (or one of its versions).

    Backs the ``tuiml_delete_algorithm`` tool; delegates to
    ``tuiml.agent.user_algorithms.delete``.

    Parameters
    ----------
    name : str
        Algorithm class name to delete. Required (arrives via
        ``**kwargs``, like all parameters below).
    version : str, default=None
        Specific version to delete; when omitted, all versions are
        removed.

    Returns
    -------
    result : dict
        Result dict from ``user_algorithms.delete`` with ``status`` and
        deletion details, or an error dict when ``name`` is missing.
    """
    from tuiml.agent import user_algorithms
    if "name" not in kwargs:
        return {"status": "error", "error_type": "ValueError",
                "error": "missing required field: name"}
    return user_algorithms.delete(name=kwargs["name"], version=kwargs.get("version"))


SPEC = ToolSpec(
    name='tuiml_delete_algorithm',
    description="Delete a user algorithm from disk. Pass only `name` to remove every "
        "version; pass both to remove a single version. Registry entries for "
        "already-loaded classes remain until the MCP server restarts. "
        "",
    input_schema={
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "version": {
                    "type": "string",
                    "description": "If omitted, all versions are removed.",
                },
            },
            "required": ["name"],
        },
    # No dedicated output schema; falls back to COMPONENT_OUTPUT_SCHEMA.
    output_schema=None,
    execute=execute_delete_user_algorithm,
    group='workflow',
    read_only=False, destructive=True,
    idempotent=False, open_world=False,
)
