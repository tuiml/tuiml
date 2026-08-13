"""Component listing."""

from typing import Any, Dict

from .._spec import ToolSpec


def execute_list(**kwargs) -> Dict[str, Any]:
    """List available components (algorithms, preprocessors, datasets, ...).

    Backs the ``tuiml_list`` tool. Supports filtering, pagination, and a
    special ``category='custom'`` mode that lists user-created algorithms
    with their research-log summaries.

    Parameters
    ----------
    category : str, default='all'
        Component category to list (``'all'``, ``'algorithm'``,
        ``'preprocessing'``, ``'custom'``, ...). Arrives via ``**kwargs``,
        like all parameters below.
    search : str, default=None
        Case-insensitive substring filter on name/description.
    type : str, default=None
        Algorithm type filter: ``'classifier'``, ``'regressor'``,
        ``'clusterer'``, ``'anomaly'``, ``'associator'``, or ``'timeseries'``.
    limit : int, default=50
        Maximum number of entries to return.
    offset : int, default=0
        Pagination offset.
    include_runs : bool, default=False
        For ``category='custom'``: include per-version run details.

    Returns
    -------
    result : dict
        On success: ``status`` (``'success'``), ``total``, ``count``,
        ``limit``, ``offset``, ``has_more``, and ``components`` (each with
        ``name``, ``description``, ``category`` and, for algorithms,
        ``type`` and ``tags``); ``category='custom'`` returns
        ``algorithms`` and a ``hint`` instead of ``components``. On
        failure: ``status`` (``'error'``), ``error``, ``error_type`` and
        ``suggestion``.
    """
    from .._components import get_all_tools, list_tools_by_category

    try:
        category = kwargs.get('category', 'all')
        search = kwargs.get('search')
        algo_type = kwargs.get('type')
        limit = kwargs.get('limit', 50)
        offset = kwargs.get('offset', 0)
        include_runs = bool(kwargs.get('include_runs', False))

        # category='custom', delegate to user_algorithms (absorbs tuiml_list_user_algorithms
        # and tuiml_research_log)
        if category == 'custom':
            from tuiml.agent import user_algorithms
            result = user_algorithms.research_log()
            if result.get('status') != 'success':
                return result
            algorithms = result.get('algorithms', [])
            if search:
                algorithms = [a for a in algorithms if search.lower() in a['name'].lower()]
            total = len(algorithms)
            paginated = algorithms[offset:offset + limit]
            if not include_runs:
                # Strip run details for a fast listing, keep versions + best scores
                for alg in paginated:
                    for v in alg.get('versions', []):
                        v.pop('path', None)
            return {
                'status': 'success',
                'category': 'custom',
                'total': total,
                'count': len(paginated),
                'limit': limit,
                'offset': offset,
                'has_more': (offset + limit) < total,
                'algorithms': paginated,
                'hint': "Use tuiml_train or tuiml_benchmark with any class_name or versioned_alias shown above.",
            }

        if category == 'all':
            tools = get_all_tools()
        else:
            tools = {t.name: t for t in list_tools_by_category(category)}

        # Filter by search
        if search:
            tools = {
                name: tool for name, tool in tools.items()
                if search.lower() in name.lower() or search.lower() in tool.description.lower()
            }

        # Build component list with type/tags from the component registry for algorithms
        from tuiml.registry import registry as hub_registry
        import tuiml.algorithms  # noqa: F401 - trigger registration

        components_list = []
        for t in tools.values():
            entry = {'name': t.name, 'description': t.description, 'category': t.category}

            # For algorithm tools, enrich with type and tags from the component registry
            if t.category == 'algorithm':
                # Strip prefix to get the class name
                class_name = t.name
                for prefix in ('tuiml_algorithm_',):
                    if class_name.startswith(prefix):
                        class_name = class_name[len(prefix):]
                try:
                    info = hub_registry.get_info(class_name)
                    entry['type'] = info.get('type', '')
                    entry['tags'] = info.get('tags', [])
                except (KeyError, Exception):
                    pass

            components_list.append(entry)

        # Filter by algorithm type (classifier, regressor, clusterer, anomaly,
        # associator, timeseries). 'anomaly' matches either the registered
        # component type or the tag, so a wrapper that registers as a plain
        # classifier but carries the tag is still found.
        if algo_type:
            if algo_type == 'anomaly':
                components_list = [
                    c for c in components_list
                    if c.get('type') == 'anomaly'
                    or 'anomaly-detection' in c.get('tags', [])
                ]
            elif algo_type == 'timeseries':
                components_list = [
                    c for c in components_list
                    if 'timeseries' in c.get('tags', [])
                ]
            else:
                components_list = [
                    c for c in components_list
                    if c.get('type') == algo_type
                ]

        total = len(components_list)

        # Apply pagination
        paginated = components_list[offset:offset + limit]

        # Format result
        result = {
            'status': 'success',
            'total': total,
            'count': len(paginated),
            'limit': limit,
            'offset': offset,
            'has_more': (offset + limit) < total,
            'components': paginated
        }

        return result
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e),
            'error_type': type(e).__name__,
            'suggestion': 'Check that the category parameter is valid. Use category="all" to list all components.'
        }


SPEC = ToolSpec(
    name='tuiml_list',
    description="List TuiML components (algorithms, preprocessors, datasets, features) "
        "or custom user-authored algorithms. Use category='custom' to list "
        "algorithms created via tuiml_create_algorithm, shows all versions, "
        "best scores, and run history. Pass include_runs=true for full experiment "
        "history (useful for auto-research: see what was tried and what to improve next).",
    input_schema={
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "enum": ["algorithm", "preprocessing", "dataset", "feature", "splitting", "custom", "all"],
                    "default": "all",
                    "description": "Category to list. Use 'custom' for user-authored algorithms."
                },
                "type": {
                    "type": "string",
                    "enum": ["classifier", "regressor", "clusterer", "anomaly", "associator", "timeseries"],
                    "description": "Filter algorithms by type (ignored for category='custom')."
                },
                "search": {
                    "type": "string",
                    "description": "Search keyword to filter results."
                },
                "include_runs": {
                    "type": "boolean",
                    "default": False,
                    "description": "For category='custom': include full experiment run history and best scores per version."
                },
                "limit": {
                    "type": "integer",
                    "default": 50,
                    "minimum": 1,
                    "maximum": 200,
                    "description": "Maximum number of results to return (default: 50)."
                },
                "offset": {
                    "type": "integer",
                    "default": 0,
                    "minimum": 0,
                    "description": "Number of results to skip for pagination."
                }
            },
            "required": []
        },
    output_schema={
            "type": "object",
            "properties": {
                "status": {"type": "string", "enum": ["success", "error"]},
                "total": {"type": "integer", "description": "Total number of components"},
                "count": {"type": "integer", "description": "Number of components returned"},
                "limit": {"type": "integer"},
                "offset": {"type": "integer"},
                "has_more": {"type": "boolean"},
                "components": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "description": {"type": "string"},
                            "category": {"type": "string"}
                        }
                    }
                },
                "error": {"type": "string"}
            },
            "required": ["status"]
        },
    execute=execute_list,
    group='discovery',
    read_only=True, destructive=False,
    idempotent=True, open_world=False,
    reproducible=False,
)
