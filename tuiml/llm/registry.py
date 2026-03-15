"""
Auto-generate MCP tools from TuiML hub registry.

This module dynamically creates tool definitions for ALL registered
components (algorithms, preprocessors, datasets, features) so LLMs
can call any TuiML component directly.
"""

from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field

@dataclass
class ToolDefinition:
    """Definition of an MCP tool."""
    name: str
    description: str
    category: str
    input_schema: Dict[str, Any]
    executor: Callable

def _make_component_executor(component_class):
    """Create executor with proper closure to avoid lambda capture bug."""
    def executor(params):
        return component_class(**params)
    return executor

def _make_dataset_executor(dataset_name):
    """Create dataset executor with proper closure."""
    def executor(params):
        return _load_dataset(dataset_name)
    return executor

def _get_algorithm_tools() -> Dict[str, ToolDefinition]:
    """Generate tools for all registered algorithms."""
    tools = {}

    try:
        from tuiml.hub import registry, ComponentType
        # Import algorithms to trigger registration with hub
        import tuiml.algorithms  # noqa: F401

        # Classifiers
        for info in registry.list(ComponentType.CLASSIFIER):
            name = info.get("name", info.get("class_name", ""))
            if not name:
                continue
            component = registry.get(name)
            if component:
                schema = {}
                if hasattr(component, 'get_parameter_schema'):
                    schema = component.get_parameter_schema()

                tools[f"tuiml_algorithm_{name}"] = ToolDefinition(
                    name=f"tuiml_algorithm_{name}",
                    description=component.__doc__.split('\n')[0] if component.__doc__ else f"Create {name} classifier",
                    category="algorithm",
                    input_schema=_schema_to_json_schema(schema),
                    executor=_make_component_executor(component)
                )

        # Regressors
        for info in registry.list(ComponentType.REGRESSOR):
            name = info.get("name", info.get("class_name", ""))
            if not name:
                continue
            component = registry.get(name)
            if component:
                schema = {}
                if hasattr(component, 'get_parameter_schema'):
                    schema = component.get_parameter_schema()

                tools[f"tuiml_algorithm_{name}"] = ToolDefinition(
                    name=f"tuiml_algorithm_{name}",
                    description=component.__doc__.split('\n')[0] if component.__doc__ else f"Create {name} regressor",
                    category="algorithm",
                    input_schema=_schema_to_json_schema(schema),
                    executor=_make_component_executor(component)
                )

        # Clusterers
        for info in registry.list(ComponentType.CLUSTERER):
            name = info.get("name", info.get("class_name", ""))
            if not name:
                continue
            component = registry.get(name)
            if component:
                schema = {}
                if hasattr(component, 'get_parameter_schema'):
                    schema = component.get_parameter_schema()

                tools[f"tuiml_algorithm_{name}"] = ToolDefinition(
                    name=f"tuiml_algorithm_{name}",
                    description=component.__doc__.split('\n')[0] if component.__doc__ else f"Create {name} clusterer",
                    category="algorithm",
                    input_schema=_schema_to_json_schema(schema),
                    executor=_make_component_executor(component)
                )

    except ImportError:
        pass

    return tools

def _get_preprocessing_tools() -> Dict[str, ToolDefinition]:
    """Generate tools for all preprocessing components."""
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
    """Generate tools for all dataset operations."""
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
    """Generate tools for feature selection/extraction."""
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
    """Generate tools for evaluation components."""
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
    """Convert Python type to JSON Schema type string."""
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
    """Convert TuiML parameter schema to JSON Schema format."""
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
    """Load a dataset and return info."""
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

def get_all_tools() -> Dict[str, ToolDefinition]:
    """
    Get all registered tools.

    Returns
    -------
    tools : Dict[str, ToolDefinition]
        Dictionary mapping tool names to their definitions.
    """
    global _TOOL_REGISTRY

    if _TOOL_REGISTRY is None:
        _TOOL_REGISTRY = {}
        _TOOL_REGISTRY.update(_get_algorithm_tools())
        _TOOL_REGISTRY.update(_get_preprocessing_tools())
        _TOOL_REGISTRY.update(_get_dataset_tools())
        _TOOL_REGISTRY.update(_get_feature_tools())
        _TOOL_REGISTRY.update(_get_evaluation_tools())

    return _TOOL_REGISTRY

def get_tool(name: str) -> Optional[ToolDefinition]:
    """Get a specific tool by name."""
    tools = get_all_tools()
    return tools.get(name)

def list_tools_by_category(category: str) -> List[ToolDefinition]:
    """List all tools in a category."""
    return [
        tool for tool in get_all_tools().values()
        if tool.category == category
    ]

def get_tool_count() -> Dict[str, int]:
    """Get count of tools by category."""
    tools = get_all_tools()
    counts = {}
    for tool in tools.values():
        counts[tool.category] = counts.get(tool.category, 0) + 1
    return counts
