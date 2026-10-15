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
    description="Validate (AST: no os/subprocess/socket/network modules, no eval/exec/open) and register "
        "an agent-authored algorithm under ~/.tuiml/user_algorithms/<name>/<version>/. It then works in "
        "every tool by class name, plus a pinned alias <Class>_v<M>_<m>_<p> for A/B benchmarks.",
    input_schema={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Python identifier, usually the class name."},
                "kind": {"type": "string", "enum": ["classifier", "regressor"], "description": "Must match the class's base type."},
                "code": {"type": "string", "description": "Full source defining exactly one @classifier or @regressor class."},
                "version": {"type": "string", "default": "1.0.0", "description": "Semver."},
                "description": {"type": "string", "description": "Falls back to the class docstring."},
                "force": {"type": "boolean", "default": False, "description": "Overwrite an existing version (prefer bumping)."}
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
