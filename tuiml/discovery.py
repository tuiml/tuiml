"""Find and inspect the registered components.

The hub registry holds every algorithm, preprocessor, and feature component
by name. These functions expose it for browsing: list by type, fetch a
component\'s description and parameter schema, or search by keyword.
"""

from typing import Dict, List, Optional

def list_algorithms(type: Optional[str] = None) -> List[Dict]:
    """List available algorithms in the registry.

    Parameters
    ----------
    type : str, optional
        Filter by algorithm type:
        
        - ``"classifier"``: Classification algorithms
        - ``"regressor"``: Regression algorithms
        - ``"clusterer"``: Clustering algorithms
        - ``"anomaly"``: Anomaly detectors
        - ``"associator"``: Association rule miners
        - ``None``: List all algorithms

    Returns
    -------
    list of dict
        Metadata for matching algorithms (name, description, tags).

    Examples
    --------
    >>> import tuiml
    >>> classifiers = tuiml.list_algorithms(type="classifier")   # doctest: +SKIP
    >>> [algo["name"] for algo in classifiers[:3]]               # doctest: +SKIP
    ['NaiveBayesClassifier', 'DecisionTreeClassifier', 'RandomForestClassifier']

    The names go straight into a :func:`tuiml.train` spec:

    >>> tuiml.train({"model": {"name": classifiers[0]["name"]},
    ...              "data": {"source": "iris"}})                # doctest: +SKIP
    """
    from tuiml.registry import registry, ComponentType

    if type:
        type_map = {
            "classifier": ComponentType.CLASSIFIER,
            "regressor": ComponentType.REGRESSOR,
            "clusterer": ComponentType.CLUSTERER,
            "anomaly": ComponentType.ANOMALY,
            "associator": ComponentType.ASSOCIATOR,
        }
        component_type = type_map.get(type.lower())
        if component_type is None:
            raise ValueError(
                f"Invalid algorithm type '{type}'. Valid types: "
                f"'classifier', 'regressor', 'clusterer', 'anomaly', 'associator'."
            )
        return registry.list(component_type)

    # Return all algorithms
    # Every algorithm component type, so anomaly detectors and associators are
    # listed alongside the rest instead of being invisible to callers.
    results = []
    for ctype in [ComponentType.CLASSIFIER, ComponentType.REGRESSOR,
                  ComponentType.CLUSTERER, ComponentType.ANOMALY,
                  ComponentType.ASSOCIATOR]:
        results.extend(registry.list(ctype))
    return results

def describe_algorithm(name: str) -> Dict:
    """Get detailed information about a specific algorithm.

    Parameters
    ----------
    name : str
        Name of the algorithm (e.g., ``"RandomForestClassifier"``).

    Returns
    -------
    dict
        Metadata dictionary containing:

        - ``description``: Full docstring documentation
        - ``parameters``: JSON schema for hyperparameters
        - ``type``: Component type (classifier, etc.)

    Examples
    --------
    >>> import tuiml
    >>> info = tuiml.describe_algorithm("RandomForestClassifier")   # doctest: +SKIP
    >>> sorted(info["parameters"])                                  # doctest: +SKIP
    ['bootstrap', 'max_depth', 'max_features', 'n_estimators', ...]
    """
    from tuiml.registry import registry

    try:
        component = registry.get(name)
    except KeyError:
        raise ValueError(
            f"Algorithm '{name}' not found in hub. "
            f"Use list_algorithms() to see available options."
        )
    return {
        "name": name,
        "description": component.__doc__,
        "parameters": getattr(component, "get_parameter_schema", lambda: {})(),
        "type": getattr(component, "_component_type", None),
    }

def search_algorithms(query: str, limit: Optional[int] = None) -> List[Dict]:
    """Search for components by keyword in name, tags, or description.

    Results are ranked by relevance, best match first. Multi-word queries are
    matched token-wise, so ``"random forest"`` finds ``RandomForestClassifier``
    as well as namespaced wrappers such as ``sklearn.RandomForestClassifier``.

    Parameters
    ----------
    query : str
        Search query (e.g., ``"random forest"``, ``"linear"``).
    limit : int, optional, default=None
        Maximum number of results to return. ``None`` returns all matches.

    Returns
    -------
    list of dict
        Metadata for matching components, best match first.

    Notes
    -----
    This searches every registered component type, not just algorithms -
    transformers and feature selectors are included in the results.

    Examples
    --------
    >>> import tuiml
    >>> results = tuiml.search_algorithms("random forest", limit=3)
    >>> [algo["name"] for algo in results]
    ['RandomForestClassifier', 'RandomForestRegressor', 'capymoa.AdaptiveRandomForest']
    """
    from tuiml.registry import registry

    return registry.search(query, limit=limit)

# Pipeline presets, each maps a name to an ordered step list in the same
# {"name": ..., "params": {...}} shape as an explicit pipeline.
