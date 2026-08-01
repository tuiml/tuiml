"""Component introspection."""

from typing import Any, Dict

from .._spec import ToolSpec


def execute_describe(**kwargs) -> Dict[str, Any]:
    """Describe a single component: its parameters, tags and description.

    Backs the ``tuiml_describe`` tool. Resolves the name first as an
    algorithm in the component registry, then as a built-in dataset, then
    as a preprocessing / feature / splitting component tool.

    Parameters
    ----------
    name : str
        Name of the component to describe (arrives via ``**kwargs``).

    Returns
    -------
    result : dict
        On success: ``status`` (``'success'``), ``type``, ``name``,
        ``description`` and ``parameters`` (JSON Schema); algorithms also
        include ``tags`` and ``version``, datasets include their info
        fields. On failure: ``status`` (``'error'``), ``error``,
        ``suggestion``, ``recovery_tool`` and ``recovery_params``.
    """
    try:
        name = kwargs['name']

        # 1. Try as algorithm from the component registry (covers all registered
        #    algorithms including community uploads)
        try:
            from tuiml.registry import registry as hub_registry, ComponentType
            import tuiml.algorithms  # noqa: F401 - trigger registration

            component = hub_registry.get(name)
            if component:
                schema = {}
                if hasattr(component, 'get_parameter_schema'):
                    schema = component.get_parameter_schema()

                info = {}
                if hasattr(hub_registry, 'get_info'):
                    try:
                        info = hub_registry.get_info(name)
                    except Exception:
                        pass

                return {
                    'status': 'success',
                    'type': info.get('type', 'algorithm'),
                    'name': name,
                    'description': (component.__doc__ or '').split('\n')[0].strip(),
                    'parameters': schema,
                    'tags': info.get('tags', []),
                    'version': info.get('version', ''),
                }
        except (ImportError, ValueError, KeyError):
            pass

        # 2. Try as dataset
        try:
            from tuiml.datasets.builtin import get_dataset_info
            info = get_dataset_info(name)
            return {
                'status': 'success',
                'type': 'dataset',
                'name': name,
                **info
            }
        except (ValueError, KeyError, ImportError):
            pass

        # 3. Try from component tool registry (preprocessing, features, splitting)
        from .._components import get_all_tools
        tools = get_all_tools()

        for prefix in ['tuiml_preprocessing_', 'tuiml_feature_', 'tuiml_splitting_']:
            tool = tools.get(f"{prefix}{name}")
            if tool:
                return {
                    'status': 'success',
                    'type': tool.category,
                    'name': name,
                    'description': tool.description,
                    'parameters': tool.input_schema
                }

        return {
            'status': 'error',
            'error': f"Component '{name}' not found",
            'suggestion': "Use 'tuiml_list' with search= to find components by keyword, or browse all components",
            'recovery_tool': 'tuiml_list',
            'recovery_params': {'search': name}
        }
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e),
            'error_type': type(e).__name__
        }


SPEC = ToolSpec(
    name='tuiml_describe',
    description="Get detailed information and parameter schema for any TuiML component.",
    input_schema={
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Component name (e.g., 'RandomForestClassifier', 'SimpleImputer', 'iris')"
                }
            },
            "required": ["name"]
        },
    output_schema={
            "type": "object",
            "properties": {
                "status": {"type": "string", "enum": ["success", "error"]},
                "type": {"type": "string"},
                "name": {"type": "string"},
                "description": {"type": "string"},
                "parameters": {"type": "object"},
                "error": {"type": "string"}
            },
            "required": ["status"]
        },
    execute=execute_describe,
    group='discovery',
    read_only=True, destructive=False,
    idempotent=True, open_world=True,
    reproducible=False,
)
