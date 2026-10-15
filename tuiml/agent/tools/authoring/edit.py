"""Targeted source edit."""

from typing import Any, Dict

from .._spec import ToolSpec


def execute_edit_algorithm(**kwargs) -> Dict[str, Any]:
    """Apply a string replacement edit to a user algorithm's source.

    Backs the ``tuiml_edit_algorithm`` tool; delegates to
    ``tuiml.agent.user_algorithms.edit_algorithm``.

    Parameters
    ----------
    name : str
        Algorithm class name to edit. Required (arrives via ``**kwargs``,
        like all parameters below).
    old_string : str
        Exact source text to replace. Required.
    new_string : str
        Replacement text. Required.
    version : str, default=None
        Specific version to edit; defaults to the latest.
    bump_version : bool, default=False
        Save the edit as a new bumped version instead of in place.

    Returns
    -------
    result : dict
        Result dict from ``user_algorithms.edit_algorithm`` with
        ``status`` and edit details, or an error dict listing missing
        required fields.
    """
    from tuiml.agent import user_algorithms
    required = [k for k in ("name", "old_string", "new_string") if k not in kwargs]
    if required:
        return {"status": "error", "error_type": "ValueError",
                "error": f"missing required fields: {', '.join(required)}"}
    return user_algorithms.edit_algorithm(
        name=kwargs["name"],
        old_string=kwargs["old_string"],
        new_string=kwargs["new_string"],
        version=kwargs.get("version"),
        bump_version=bool(kwargs.get("bump_version", False)),
    )


SPEC = ToolSpec(
    name='tuiml_edit_algorithm',
    description="str_replace edit on a user algorithm: old_string must occur exactly once. The result is "
        "AST-validated and re-registered. Read with tuiml_read_algorithm first; set bump_version to keep "
        "the previous version. Built-ins are read-only (fork via tuiml_create_algorithm).",
    input_schema={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "User algorithm name."},
                "old_string": {"type": "string", "description": "Exact, unique text to replace."},
                "new_string": {"type": "string"},
                "version": {"type": "string", "description": "Defaults to latest."},
                "bump_version": {"type": "boolean", "default": False, "description": "Save as a new patch version."}
            },
            "required": ["name", "old_string", "new_string"],
        },
    # No dedicated output schema; falls back to COMPONENT_OUTPUT_SCHEMA.
    output_schema=None,
    execute=execute_edit_algorithm,
    group='code',
    read_only=False, destructive=True,
    idempotent=False, open_world=False,
)
