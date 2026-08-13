"""
Auto-generate MCP tools from the TuiML component registry.

This module dynamically creates tool definitions for ALL registered
components (algorithms, preprocessors, datasets, features) so LLMs
can call any TuiML component directly.
"""

import threading
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field

@dataclass
class ToolDefinition:
    """Definition of an MCP tool.

    Parameters
    ----------
    name : str
        Unique tool name (e.g. ``tuiml_algorithm_RandomForestClassifier``).
    description : str
        One-line human/LLM-readable description of the tool.
    category : str
        Tool category: ``'algorithm'``, ``'preprocessing'``, ``'dataset'``,
        ``'feature'``, or ``'splitting'``.
    input_schema : Dict[str, Any]
        JSON Schema describing the tool's input parameters.
    executor : Callable
        Callable that executes the tool given a params dict.
    """
    name: str
    description: str
    category: str
    input_schema: Dict[str, Any]
    executor: Callable

def _make_component_executor(component_class):
    """Create an executor with a proper closure to avoid the lambda capture bug.

    Parameters
    ----------
    component_class : type
        Component class to instantiate when the executor is called.

    Returns
    -------
    executor : Callable
        Function taking a params dict and returning ``component_class(**params)``.
    """
    def executor(params):
        return component_class(**params)
    return executor

def _make_dataset_executor(dataset_name):
    """Create a dataset-loading executor with a proper closure.

    Parameters
    ----------
    dataset_name : str
        Name of the built-in dataset to load.

    Returns
    -------
    executor : Callable
        Function taking a (ignored) params dict and returning the dataset
        info dict from ``_load_dataset``.
    """
    def executor(params):
        return _load_dataset(dataset_name)
    return executor

def _get_algorithm_tools() -> Dict[str, ToolDefinition]:
    """Generate tools for all algorithms in the component registry.

    Returns
    -------
    tools : Dict[str, ToolDefinition]
        Mapping of tool name to definition for every registered classifier,
        regressor, clusterer, anomaly detector, and associator.
    """
    tools = {}

    try:
        from tuiml.registry import registry, ComponentType
        # Import algorithms to trigger registration with the component registry
        import tuiml.algorithms  # noqa: F401

        # One MCP tool per registered algorithm, across every algorithm
        # component type. Anomaly detectors and associators register under
        # their own types, so listing only classifier/regressor/clusterer
        # would hide them from agents.
        for ctype, label in (
            (ComponentType.CLASSIFIER, "classifier"),
            (ComponentType.REGRESSOR, "regressor"),
            (ComponentType.CLUSTERER, "clusterer"),
            (ComponentType.ANOMALY, "anomaly detector"),
            (ComponentType.ASSOCIATOR, "associator"),
        ):
            for info in registry.list(ctype):
                name = info.get("name", info.get("class_name", ""))
                if not name:
                    continue
                component = registry.get(name)
                if not component:
                    continue
                schema = {}
                if hasattr(component, 'get_parameter_schema'):
                    schema = component.get_parameter_schema()

                tools[f"tuiml_algorithm_{name}"] = ToolDefinition(
                    name=f"tuiml_algorithm_{name}",
                    description=component.__doc__.split('\n')[0] if component.__doc__ else f"Create {name} {label}",
                    category="algorithm",
                    input_schema=_schema_to_json_schema(schema),
                    executor=_make_component_executor(component)
                )

    except ImportError:
        pass

    return tools

def _get_preprocessing_tools() -> Dict[str, ToolDefinition]:
    """Generate tools for all preprocessing components.

    Returns
    -------
    tools : Dict[str, ToolDefinition]
        Mapping of tool name to definition for every exported preprocessor.
    """
    tools = {}

    try:
        from tuiml import preprocessing

        # Get all exported preprocessors from __all__
        for name in preprocessing.__all__:
            component = getattr(preprocessing, name, None)
            if component and hasattr(component, '__init__'):
                schema = {}
                if hasattr(component, 'get_parameter_schema'):
                    schema = component.get_parameter_schema()

                tools[f"tuiml_preprocessing_{name}"] = ToolDefinition(
                    name=f"tuiml_preprocessing_{name}",
                    description=component.__doc__.split('\n')[0] if component.__doc__ else f"Create {name} preprocessor",
                    category="preprocessing",
                    input_schema=_schema_to_json_schema(schema),
                    executor=_make_component_executor(component)
                )
    except ImportError:
        pass

    return tools

def _get_dataset_tools() -> Dict[str, ToolDefinition]:
    """Generate tools for all dataset operations.

    Returns
    -------
    tools : Dict[str, ToolDefinition]
        Mapping of tool name to definition: one discovery tool
        (``tuiml_dataset_list``) plus one loader per built-in dataset.
    """
    tools = {}

    try:
        from tuiml.datasets.builtin import DATASET_REGISTRY, get_dataset_info
        from tuiml.datasets import load_dataset

        # Discovery tool
        tools["tuiml_dataset_list"] = ToolDefinition(
            name="tuiml_dataset_list",
            description="List all available built-in datasets with metadata",
            category="dataset",
            input_schema={
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "enum": ["classification", "regression", "association", "text_classification"],
                        "description": "Filter by task type"
                    }
                },
                "required": []
            },
            executor=lambda params: get_dataset_info() if not params.get("task") else {
                k: v for k, v in DATASET_REGISTRY.items() if v.get("task") == params.get("task")
            }
        )

        # Individual dataset loaders
        for name, info in DATASET_REGISTRY.items():
            tools[f"tuiml_dataset_{name}"] = ToolDefinition(
                name=f"tuiml_dataset_{name}",
                description=info.get("description", f"Load {name} dataset"),
                category="dataset",
                input_schema={
                    "type": "object",
                    "properties": {},
                    "required": []
                },
                executor=_make_dataset_executor(name)
            )

    except ImportError:
        pass

    return tools

def _get_feature_tools() -> Dict[str, ToolDefinition]:
    """Generate tools for feature selection/extraction/generation components.

    Returns
    -------
    tools : Dict[str, ToolDefinition]
        Mapping of tool name to definition for every feature class found in
        ``tuiml.features.selection``, ``.extraction``, and ``.generation``.
    """
    tools = {}

    try:
        from tuiml.features import selection, extraction, generation
        import inspect

        # Helper to check if something is a feature class
        def is_feature_class(obj):
            return (inspect.isclass(obj) and
                    hasattr(obj, 'fit') and
                    hasattr(obj, 'transform') and
                    not obj.__name__.startswith('_'))

        # Process selection module - use __all__ if available
        selection_names = getattr(selection, '__all__', dir(selection))
        for name in selection_names:
            if name.startswith('_'):
                continue
            component = getattr(selection, name, None)
            if component and is_feature_class(component):
                schema = {}
                if hasattr(component, 'get_parameter_schema'):
                    schema = component.get_parameter_schema()

                tools[f"tuiml_feature_{name}"] = ToolDefinition(
                    name=f"tuiml_feature_{name}",
                    description=component.__doc__.split('\n')[0] if component.__doc__ else f"Create {name} feature selector",
                    category="feature",
                    input_schema=_schema_to_json_schema(schema),
                    executor=_make_component_executor(component)
                )

        # Process extraction module
        extraction_names = getattr(extraction, '__all__', dir(extraction))
        for name in extraction_names:
            if name.startswith('_'):
                continue
            component = getattr(extraction, name, None)
            if component and is_feature_class(component):
                schema = {}
                if hasattr(component, 'get_parameter_schema'):
                    schema = component.get_parameter_schema()

                tools[f"tuiml_feature_{name}"] = ToolDefinition(
                    name=f"tuiml_feature_{name}",
                    description=component.__doc__.split('\n')[0] if component.__doc__ else f"Create {name} feature extractor",
                    category="feature",
                    input_schema=_schema_to_json_schema(schema),
                    executor=_make_component_executor(component)
                )

        # Process generation module
        generation_names = getattr(generation, '__all__', dir(generation))
        for name in generation_names:
            if name.startswith('_'):
                continue
            component = getattr(generation, name, None)
            if component and is_feature_class(component):
                schema = {}
                if hasattr(component, 'get_parameter_schema'):
                    schema = component.get_parameter_schema()

                tools[f"tuiml_feature_{name}"] = ToolDefinition(
                    name=f"tuiml_feature_{name}",
                    description=component.__doc__.split('\n')[0] if component.__doc__ else f"Create {name} feature generator",
                    category="feature",
                    input_schema=_schema_to_json_schema(schema),
                    executor=_make_component_executor(component)
                )

    except ImportError:
        pass

    return tools

def _get_evaluation_tools() -> Dict[str, ToolDefinition]:
    """Generate tools for evaluation (data splitting) components.

    Returns
    -------
    tools : Dict[str, ToolDefinition]
        Mapping of tool name to definition for every exported splitter class.
    """
    tools = {}

    try:
        from tuiml.evaluation import splitting
        import inspect

        # Use __all__ to get all exported splitters
        for name in splitting.__all__:
            if name.startswith('_'):
                continue
            component = getattr(splitting, name, None)
            # Skip functions and base classes
            if not inspect.isclass(component):
                continue
            if name in ('BaseSplitter',):
                continue

            schema = {}
            if hasattr(component, 'get_parameter_schema'):
                schema = component.get_parameter_schema()

            tools[f"tuiml_splitting_{name}"] = ToolDefinition(
                name=f"tuiml_splitting_{name}",
                description=component.__doc__.split('\n')[0] if component.__doc__ else f"Create {name} splitter",
                category="splitting",
                input_schema=_schema_to_json_schema(schema),
                executor=_make_component_executor(component)
            )
    except ImportError:
        pass

    return tools

def _python_type_to_json_type(t: Any) -> str:
    """Convert a Python type to a JSON Schema type string.

    Parameters
    ----------
    t : Any
        Python type (``int``, ``float``, ``bool``, ``str``, ``list``,
        ``dict``), a JSON Schema type string, or None.

    Returns
    -------
    json_type : str
        JSON Schema type name; ``'string'`` for unknown types.
    """
    if t is None or t == "null":
        return "null"
    if isinstance(t, str):
        return t
    if t is int or t == int:
        return "integer"
    if t is float or t == float:
        return "number"
    if t is bool or t == bool:
        return "boolean"
    if t is str or t == str:
        return "string"
    if t is list or t == list:
        return "array"
    if t is dict or t == dict:
        return "object"
    # Default to string for unknown types
    return "string"

def _schema_to_json_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a TuiML parameter schema to JSON Schema format.

    Parameters
    ----------
    schema : Dict[str, Any]
        TuiML parameter schema as returned by ``get_parameter_schema()``:
        maps parameter name to a dict with keys like ``type``, ``default``,
        ``description``, ``enum``, ``minimum``, ``maximum``.

    Returns
    -------
    json_schema : Dict[str, Any]
        JSON Schema object with keys ``type`` (``'object'``), ``properties``,
        and ``required`` (parameters without a default).
    """
    if not schema:
        return {"type": "object", "properties": {}, "required": []}

    properties = {}
    required = []

    for param_name, param_info in schema.items():
        prop = {
            "description": str(param_info.get("description", ""))
        }

        # Handle type - convert Python types to JSON Schema strings
        param_type = param_info.get("type", "string")
        if isinstance(param_type, list):
            prop["type"] = [_python_type_to_json_type(t) for t in param_type]
        else:
            prop["type"] = _python_type_to_json_type(param_type)

        # Handle enum - ensure all values are JSON serializable
        if "enum" in param_info:
            prop["enum"] = [str(v) if not isinstance(v, (str, int, float, bool, type(None))) else v
                          for v in param_info["enum"]]

        # Handle default - ensure JSON serializable
        if "default" in param_info:
            default_val = param_info["default"]
            if default_val is None or isinstance(default_val, (str, int, float, bool, list, dict)):
                prop["default"] = default_val
            else:
                prop["default"] = str(default_val)

        # Handle min/max
        if "minimum" in param_info:
            prop["minimum"] = param_info["minimum"]
        if "maximum" in param_info:
            prop["maximum"] = param_info["maximum"]

        properties[param_name] = prop

        # Mark as required if no default value provided
        if "default" not in param_info:
            required.append(param_name)

    return {
        "type": "object",
        "properties": properties,
        "required": required
    }

def _load_dataset(name: str) -> Dict[str, Any]:
    """Load a built-in dataset and return summary info.

    Parameters
    ----------
    name : str
        Name of the built-in dataset to load.

    Returns
    -------
    info : Dict[str, Any]
        Dict with keys ``name``, ``shape`` (list or None), ``task``,
        ``description``, and ``loaded`` (bool).
    """
    from tuiml.datasets import load_dataset
    from tuiml.datasets.builtin import DATASET_REGISTRY

    dataset = load_dataset(name)
    info = DATASET_REGISTRY.get(name, {})

    return {
        "name": name,
        "shape": list(dataset.X.shape) if hasattr(dataset, 'X') else None,
        "task": info.get("task"),
        "description": info.get("description"),
        "loaded": True
    }

# =============================================================================
# Main Registry
# =============================================================================

_TOOL_REGISTRY: Optional[Dict[str, ToolDefinition]] = None
_REGISTRY_LOCK = threading.Lock()

def get_all_tools() -> Dict[str, ToolDefinition]:
    """
    Get all registered tools.

    Thread-safe: the registry may be preloaded from a background thread
    (see ``tuiml.agent.mcp.server.run_server``) while tool calls arrive
    concurrently, so the global is only assigned once fully built.

    Returns
    -------
    tools : Dict[str, ToolDefinition]
        Dictionary mapping tool names to their definitions.
    """
    global _TOOL_REGISTRY

    if _TOOL_REGISTRY is None:
        with _REGISTRY_LOCK:
            if _TOOL_REGISTRY is None:
                registry: Dict[str, ToolDefinition] = {}
                registry.update(_get_algorithm_tools())
                registry.update(_get_preprocessing_tools())
                registry.update(_get_dataset_tools())
                registry.update(_get_feature_tools())
                registry.update(_get_evaluation_tools())
                _TOOL_REGISTRY = registry

    return _TOOL_REGISTRY

def get_tool(name: str) -> Optional[ToolDefinition]:
    """Get a specific tool by name.

    Parameters
    ----------
    name : str
        Tool name (e.g. ``tuiml_algorithm_RandomForestClassifier``).

    Returns
    -------
    tool : Optional[ToolDefinition]
        The tool definition, or None if no tool has that name.
    """
    tools = get_all_tools()
    return tools.get(name)

def list_tools_by_category(category: str) -> List[ToolDefinition]:
    """List all tools in a category.

    Parameters
    ----------
    category : str
        Category to filter by: ``'algorithm'``, ``'preprocessing'``,
        ``'dataset'``, ``'feature'``, or ``'splitting'``.

    Returns
    -------
    tools : List[ToolDefinition]
        All tool definitions whose category matches.
    """
    return [
        tool for tool in get_all_tools().values()
        if tool.category == category
    ]

def get_tool_count() -> Dict[str, int]:
    """Get the count of tools by category.

    Returns
    -------
    counts : Dict[str, int]
        Mapping of category name to number of registered tools.
    """
    tools = get_all_tools()
    counts = {}
    for tool in tools.values():
        counts[tool.category] = counts.get(tool.category, 0) + 1
    return counts
