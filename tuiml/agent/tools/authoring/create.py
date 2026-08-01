"""Create and register a user algorithm."""

from typing import Any, Dict

from .._spec import ToolSpec


def execute_create_algorithm(**kwargs) -> Dict[str, Any]:
    """Create and register a new user algorithm from source code.

    Backs the ``tuiml_create_algorithm`` tool; delegates to
    ``tuiml.agent.user_algorithms.create``.

    Parameters
    ----------
    name : str
        Class name of the new algorithm. Required (arrives via
        ``**kwargs``, like all parameters below).
    kind : str
        Algorithm kind (e.g. ``'classifier'`` or ``'regressor'``).
        Required.
    code : str
        Full Python source of the algorithm. Required.
    version : str, default='1.0.0'
        Version to register the algorithm under.
    description : str, default=None
        Short description stored with the algorithm.
    force : bool, default=False
        Overwrite an existing algorithm of the same name/version.

    Returns
    -------
    result : dict
        Result dict from ``user_algorithms.create`` with ``status`` and
        creation details, or an error dict listing missing required
        fields.
    """
    from tuiml.agent import user_algorithms
    required = [k for k in ("name", "kind", "code") if k not in kwargs]
    if required:
        return {"status": "error", "error_type": "ValueError",
                "error": f"missing required fields: {', '.join(required)}"}
    return user_algorithms.create(
        name=kwargs["name"],
        kind=kwargs["kind"],
        code=kwargs["code"],
        version=kwargs.get("version", "1.0.0"),
        description=kwargs.get("description"),
        force=bool(kwargs.get("force", False)),
    )


SPEC = ToolSpec(
    name='tuiml_create_algorithm',
    description="Persist, validate, and register a new agent-authored algorithm. "
        "The source is AST-validated (forbidden modules: subprocess, socket, "
        "os, urllib, requests, …; forbidden calls: eval, exec, open, __import__) "
        "and saved to ~/.tuiml/user_algorithms/<name>/<version>/algorithm.py. "
        "After registration, the algorithm is available via its class name to "
        "every existing MCP tool (tuiml_train, tuiml_benchmark, tuiml_describe). "
        "Each version is also registered under a pinned alias "
        "<ClassName>_v<major>_<minor>_<patch> so you can A/B compare versions "
        "inside a single tuiml_benchmark. "
        "",
    input_schema={
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Directory name, usually equal to the class name (Python identifier).",
                },
                "kind": {
                    "type": "string",
                    "enum": ["classifier", "regressor"],
                    "description": "Task kind. Must match the imported class's base type.",
                },
                "code": {
                    "type": "string",
                    "description": "Full Python source. Must define exactly one @classifier or @regressor class.",
                },
                "version": {
                    "type": "string",
                    "description": "Semver for this submission, e.g. '1.0.0', '1.0.1'.",
                    "default": "1.0.0",
                },
                "description": {
                    "type": "string",
                    "description": "Optional short description (falls back to the class docstring).",
                },
                "force": {
                    "type": "boolean",
                    "description": "Overwrite an existing file at <name>/<version>/. Bump the version instead when possible.",
                    "default": False,
                },
            },
            "required": ["name", "kind", "code"],
        },
    # No dedicated output schema; falls back to COMPONENT_OUTPUT_SCHEMA.
    output_schema=None,
    execute=execute_create_algorithm,
    group='workflow',
    read_only=False, destructive=False,
    idempotent=False, open_world=False,
)
