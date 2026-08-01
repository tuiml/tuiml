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
    description="Apply a targeted str_replace edit to a user algorithm. "
        "Replaces exactly one occurrence of old_string with new_string, "
        "fails loudly if old_string is not found or appears more than once "
        "(make it more specific with surrounding context). "
        "The edited source is AST-validated and the algorithm is re-registered "
        "so all MCP tools immediately see the change. "
        "Workflow: tuiml_read_algorithm → identify the text to change → tuiml_edit_algorithm. "
        "Set bump_version=true to save as a new patch version instead of overwriting. "
        "Built-in algorithms cannot be edited, fork them first with tuiml_create_algorithm. "
        "",
    input_schema={
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "User algorithm name (directory name / class name).",
                },
                "old_string": {
                    "type": "string",
                    "description": "The exact text to find and replace. Must be unique in the file.",
                },
                "new_string": {
                    "type": "string",
                    "description": "The replacement text.",
                },
                "version": {
                    "type": "string",
                    "description": "Target a specific version. Defaults to latest.",
                },
                "bump_version": {
                    "type": "boolean",
                    "default": False,
                    "description": "Save the edit as a new patch version instead of overwriting the current one.",
                },
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
