"""Starter template for a new algorithm."""

from typing import Any, Dict

from .._spec import ToolSpec


def execute_algorithm_skeleton(**kwargs) -> Dict[str, Any]:
    """Return a starter code skeleton for a new user algorithm.

    Backs the ``tuiml_get_skeleton`` tool; delegates to
    ``tuiml.agent.user_algorithms.skeleton``.

    Parameters
    ----------
    kind : str, default='classifier'
        Kind of algorithm to scaffold (e.g. ``'classifier'`` or
        ``'regressor'``). Arrives via ``**kwargs``, like all parameters
        below.
    class_name : str, default='MyAlgorithm'
        Name of the generated class.
    version : str, default='1.0.0'
        Initial version string embedded in the skeleton.
    description : str, default='Describe what your algorithm does.'
        One-line description embedded in the skeleton docstring.

    Returns
    -------
    result : dict
        Result dict from ``user_algorithms.skeleton`` with ``status`` and
        the generated skeleton code.
    """
    from tuiml.agent import user_algorithms
    return user_algorithms.skeleton(
        kind=kwargs.get("kind", "classifier"),
        class_name=kwargs.get("class_name", "MyAlgorithm"),
        version=kwargs.get("version", "1.0.0"),
        description=kwargs.get("description", "Describe what your algorithm does."),
    )


SPEC = ToolSpec(
    name='tuiml_get_skeleton',
    description="Return a ready-to-edit Python source template for a new @classifier "
        "or @regressor class. Agents should call this, fill in fit() and "
        "predict(), then pass the completed source to tuiml_create_algorithm. "
        "",
    input_schema={
            "type": "object",
            "properties": {
                "kind": {
                    "type": "string",
                    "enum": ["classifier", "regressor"],
                    "description": "Task kind the new algorithm targets.",
                },
                "class_name": {
                    "type": "string",
                    "description": "Python identifier for the new class, e.g. 'MyGradientBoosting'.",
                    "default": "MyAlgorithm",
                },
                "version": {
                    "type": "string",
                    "description": "Initial semver, e.g. '1.0.0'.",
                    "default": "1.0.0",
                },
                "description": {
                    "type": "string",
                    "description": "One-line docstring for the class.",
                    "default": "Describe what your algorithm does.",
                },
            },
            "required": ["kind"],
        },
    # No dedicated output schema; falls back to COMPONENT_OUTPUT_SCHEMA.
    output_schema=None,
    execute=execute_algorithm_skeleton,
    group='workflow',
    read_only=True, destructive=False,
    idempotent=True, open_world=False,
    # Scaffolding only: the filled-in source arrives via tuiml_create_algorithm,
    # which is what the notebook exports. An empty template is not a workflow step.
    reproducible=False,
)
