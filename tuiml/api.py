"""High-level API for TuiML - One-liner functions for common ML tasks."""

from typing import Callable, Union, Optional, List, Dict, Any
import os
import threading
import numpy as np
import pandas as pd

from tuiml.workflow import Workflow

# Module-level state for tracking running servers
_SERVERS: Dict[str, Dict] = {}

# ===== Main API Functions =====

def _reject_foreign_estimator(model: Any) -> None:
    """Raise if ``model`` is a foreign estimator object rather than a TuiML one.

    TuiML is native-first: algorithms are addressed by registry name, and
    external libraries are reached through curated wrappers registered under a
    namespaced key (``sklearn.SVC``, ``capymoa.HoeffdingTree``). Passing a raw
    third-party estimator *instance* is not supported, so this fails early with
    a message naming the wrapper to use instead of failing obscurely later.

    Parameters
    ----------
    model : Any
        The value passed as the ``model`` / algorithm argument.

    Raises
    ------
    TypeError
        If ``model`` looks like a non-TuiML estimator object.
    """
    from tuiml.base.algorithms import Algorithm

    if isinstance(model, (str, dict, type)) or isinstance(model, Algorithm):
        return
    # Duck-typed foreign estimator: has fit(), but is not a TuiML Algorithm.
    if not callable(getattr(model, "fit", None)):
        return

    cls_name = type(model).__name__
    from tuiml.hub import registry

    candidates = [
        name for name in registry.list_names()
        if "." in name and name.split(".", 1)[1] == cls_name
    ]
    if candidates:
        hint = (
            f"Use the registered wrapper instead:  "
            f'tuiml.train("{candidates[0]}", ...)'
        )
    else:
        hint = (
            f"No wrapper is registered for {cls_name}. Use a native TuiML "
            f"algorithm (tuiml.list_algorithms()), or add a wrapper under "
            f"tuiml/sklearn/ or tuiml/capymoa/."
        )
    raise TypeError(
        f"{cls_name} is not a TuiML algorithm. TuiML addresses algorithms by "
        f"registry name; raw third-party estimator objects are not accepted. "
        f"{hint}"
    )


def _resolve_data_spec(data, target, features):
    """Normalize the ``data`` argument into ``(source, target, features)``.

    The ``data`` argument may be a **data spec** dict or a bare source. A spec
    dict groups everything about the data in one place:

    - ``{"source": "sales.csv", "target": "label", "features": [...]}`` — a file
      path or builtin name plus its target column (and optional feature subset).
    - ``{"X": X_array, "y": y_array}`` — in-memory arrays, already split.

    Anything that is not such a dict (a path string, builtin name, ``DataFrame``,
    or ``ndarray``) is returned unchanged, paired with the separately supplied
    ``target``/``features`` arguments.

    Parameters
    ----------
    data : str, DataFrame, ndarray, or dict
        The raw ``data`` argument passed to :func:`train`.
    target : str, ndarray, or None
        The separately supplied target (used only for bare-``data`` forms).
    features : list of str or None
        The separately supplied feature subset.

    Returns
    -------
    source : object
        The data source to hand to :class:`~tuiml.workflow.Workflow`.
    target : object
        The resolved target column name / array (or ``None``).
    features : list of str or None
        The resolved feature subset.
    """
    if not isinstance(data, dict):
        return data, target, features

    spec = dict(data)
    features = spec.get("features", features)

    # In-memory arrays, already split.
    if "X" in spec:
        return spec["X"], spec.get("y", target), features

    # File path / builtin name / DataFrame reference.
    source = spec.get("source")
    if source is None and ("path" in spec or "data" in spec):
        raise ValueError(
            'Data specs use the key "source" for the file path or builtin '
            'name, e.g. {"source": "sales.csv", "target": "label"}.'
        )
    target = spec.get("target", target)
    return source, target, features


def _parse_component_spec(spec, kind: str):
    """Split a component spec into ``(name, params)``.

    Components — the model and each pipeline step — share one shape:
    ``{"name": "<ClassName>", "params": {...}}``. A bare string means the
    class name with default params. Any other key in the dict is rejected so
    that misplaced hyperparameters fail loudly instead of being ignored.

    Parameters
    ----------
    spec : str, dict, class, or instance
        The component specification.
    kind : str
        Label used in error messages (e.g., ``"model"``, ``"pipeline step"``).

    Returns
    -------
    name : str, class, or instance
        The component name (or the class/instance itself, passed through).
    params : dict
        The component's constructor parameters.
    """
    if isinstance(spec, str):
        return spec, {}
    if isinstance(spec, dict):
        name = spec.get("name")
        if not name:
            raise ValueError(
                f'A {kind} spec dict needs a "name" key, e.g. '
                f'{{"name": "RandomForestClassifier", "params": {{...}}}}.'
            )
        params = spec.get("params") or {}
        if not isinstance(params, dict):
            raise ValueError(f'The {kind} "params" value must be a dict.')
        extra = set(spec) - {"name", "params"}
        if extra:
            raise ValueError(
                f"Unexpected keys {sorted(extra)} in {kind} spec for "
                f"'{name}'. Hyperparameters go inside \"params\": "
                f'{{"name": "{name}", "params": {{...}}}}.'
            )
        return name, params
    # Class object or configured instance — passed through as-is.
    return spec, {}


_EVALUATION_KEYS = {"cv", "test_size", "stratify", "metrics"}


def train(
    model: Union[str, Dict, Any] = None,
    data: Union[str, pd.DataFrame, np.ndarray, Dict] = None,
    target: Union[str, np.ndarray] = None,
    *,
    # Restrict the feature matrix to these named columns
    features: Optional[List[str]] = None,
    # Ordered preprocessing / feature-engineering steps (or a preset name)
    pipeline: Optional[Union[str, List[Union[str, Dict]]]] = None,
    # Grouped evaluation options: cv OR test_size/stratify, plus metrics
    evaluation: Optional[Dict[str, Any]] = None,
    # Reproducibility
    random_seed: Optional[int] = None,
) -> Workflow:
    """Train a machine learning model with a complete workflow.

    This is the **main high-level function** for training models in TuiML.
    A training run is described by four self-contained specs:

    - ``model`` — what to train: ``{"name": ..., "params": {...}}``
    - ``data`` — what to train on: ``{"source": ..., "target": ...}``
    - ``pipeline`` — ordered feature-engineering steps applied before the
      model (imputation, scaling, encoding, feature generation, extraction,
      selection, resampling), each ``{"name": ..., "params": {...}}``
    - ``evaluation`` — how to score it: ``{"cv": ...}`` or
      ``{"test_size": ..., "stratify": ...}``, plus ``"metrics"``

    The whole run can also be passed as **one declarative spec**: a single
    dict with those keys, or the path to a ``.json`` file containing it.

    Parameters
    ----------
    model : str, dict, class, or instance
        Model specification — or the full experiment spec. Accepts:

        - ``{"name": "RandomForestClassifier", "params": {"n_estimators":
          100}}`` — **component spec** (recommended): class name plus its
          hyperparameters under ``"params"``.
        - ``"RandomForestClassifier"`` — class name only, default params.
        - ``RandomForestClassifier(n_estimators=100)`` — a configured
          instance (opt into an import for editor autocomplete).
        - a class object — instantiated with default parameters.
        - ``{"model": ..., "data": ..., ...}`` — a **full experiment spec**;
          every other argument comes from the dict itself.
        - ``"experiment.json"`` — path to a JSON file with a full spec.

    data : str, DataFrame, ndarray, or dict
        Training data. Accepts:

        - ``{"source": "sales.csv", "target": "label"}`` — **data spec**
          (recommended for files): file path (csv, arff, parquet, json,
          excel — auto-detected) or builtin dataset name, plus the target
          column. Optional ``"features": [...]`` restricts input columns.
        - ``{"X": X_array, "y": y_array}`` — in-memory arrays, already split.
        - ``"path/to/file.csv"`` / ``"iris"`` — a bare file path or builtin
          name (use the ``target`` argument to name the label column).
        - ``DataFrame`` / ``ndarray`` — in-memory data (with ``target``).

    target : str or ndarray, optional
        Target for the bare-``data`` forms (ignored when ``data`` is a spec
        dict that already carries ``"target"``/``"y"``):

        - ``"column_name"`` — the label column in a file/DataFrame.
        - ``ndarray`` — a separate target array.

    features : list of str, optional
        Restrict the feature matrix to these named columns. When ``None``
        (default), every non-target column is used. May also be supplied
        inside the data spec as ``"features"``.

    pipeline : str, list, or None, default=None
        Ordered feature-engineering steps applied before the model. Every
        kind of transform is a step — imputation, scaling, encoding,
        feature generation, feature extraction, feature selection, and
        resampling — resolved by name from the hub:

        - ``[{"name": "SimpleImputer", "params": {"strategy": "mean"}},
          {"name": "StandardScaler"},
          {"name": "PCAExtractor", "params": {"n_components": 5}},
          {"name": "SelectKBestSelector", "params": {"k": 3}}]``
        - ``["SimpleImputer", "MinMaxScaler"]`` — bare names, default params
        - ``"minimal"`` / ``"fast"`` / ``"standard"`` / ``"full"`` /
          ``"imbalanced"`` — preset name (see :data:`PRESETS`)
        - ``None`` — no pipeline

    evaluation : dict, optional
        Grouped evaluation options. Keys:

        - ``"cv"`` — number of cross-validation folds. When present, k-fold
          CV is used and the holdout keys are ignored.
        - ``"test_size"`` — holdout test proportion (default 0.2).
        - ``"stratify"`` — keep class balance in the split (default True).
        - ``"metrics"`` — ``"auto"`` (default) or a list of metric function
          names from ``tuiml.evaluation.metrics``.

        ``None`` means a stratified 80/20 holdout with auto metrics.

    random_seed : int or None, default=None
        Random seed for reproducibility. Falls back to the global seed,
        then 42.

    Returns
    -------
    Workflow
        The fitted pipeline, which behaves like a model:

        - ``model_`` — the fitted final model
        - ``metrics_`` — the computed metrics
        - ``predict(X)`` / ``score(X, y)`` / ``save(path)``

    Examples
    --------
    Quick — model name and a data spec:

    >>> result = tuiml.train("RandomForestClassifier", {"source": "iris"})

    One declarative spec — a single serializable dict, ideal for agents:

    >>> result = tuiml.train({
    ...     "model": {"name": "RandomForestClassifier",
    ...               "params": {"n_estimators": 100}},
    ...     "data": {"source": "sales.csv", "target": "label"},
    ...     "pipeline": [
    ...         {"name": "SimpleImputer", "params": {"strategy": "mean"}},
    ...         {"name": "StandardScaler"},
    ...         {"name": "SelectKBestSelector", "params": {"k": 10}},
    ...     ],
    ...     "evaluation": {"cv": 10, "metrics": ["accuracy_score"]},
    ... })

    The same spec from a JSON file:

    >>> result = tuiml.train("experiment.json")

    In-memory arrays, already split:

    >>> result = tuiml.train(
    ...     {"name": "SVC", "params": {"kernel": "rbf", "C": 1.0}},
    ...     {"X": X, "y": y},
    ... )

    Type-safe — pass a configured instance (opt into an import for
    autocomplete):

    >>> from tuiml.algorithms.trees import RandomForestClassifier
    >>> result = tuiml.train(RandomForestClassifier(n_estimators=100),
    ...                      {"source": "iris"})
    """
    # A full experiment spec may arrive as the first argument: a dict whose
    # keys mirror this function's parameters, or the path to a .json file
    # containing that dict. A *component* spec dict has a "name" key instead,
    # so the two dict forms cannot collide.
    if data is None:
        spec = None
        if isinstance(model, str) and model.endswith(".json"):
            import json
            with open(model, "r") as f:
                spec = json.load(f)
        elif isinstance(model, dict) and "name" not in model:
            spec = dict(model)
        if spec is not None:
            return train(**spec)

    # Resolve the data spec: {"source": ..., "target": ..., "features": ...}
    # or {"X": ..., "y": ...}. Bare paths / DataFrames / arrays pass through
    # with the separate ``target``/``features`` arguments.
    data, target, features = _resolve_data_spec(data, target, features)

    # Validate required params
    if model is None:
        raise ValueError("A model must be specified (name, spec dict, or instance).")
    if data is None:
        raise ValueError("Data must be provided.")

    # Extract model name and params (handles str / spec dict / class / instance)
    _reject_foreign_estimator(model)
    algo_name, algo_params = _parse_component_spec(model, "model")

    # Unpack the evaluation spec
    evaluation = dict(evaluation or {})
    unknown = set(evaluation) - _EVALUATION_KEYS
    if unknown:
        raise ValueError(
            f"Unexpected evaluation keys {sorted(unknown)}. "
            f"Allowed: {sorted(_EVALUATION_KEYS)}."
        )
    cv = evaluation.get("cv")
    test_size = evaluation.get("test_size", 0.2)
    stratify = evaluation.get("stratify", True)
    metrics = evaluation.get("metrics", "auto")

    # Resolve a pipeline preset name to its step list
    if isinstance(pipeline, str):
        if pipeline not in PRESETS:
            raise ValueError(
                f"Unknown pipeline preset '{pipeline}'. "
                f"Available presets: {sorted(PRESETS)}."
            )
        pipeline = PRESETS[pipeline]

    # A spec is just a Workflow written as data: the pipeline steps followed by
    # the model as the final step.
    steps = list(pipeline or [])
    steps.append({"name": algo_name, "params": algo_params}
                 if isinstance(algo_name, str) else algo_name)

    return Workflow(steps).fit(
        data,
        target=target,
        features=features,
        cv=cv,
        test_size=None if cv else test_size,
        stratify=stratify,
        metrics=metrics,
        random_seed=random_seed,
    )

def experiment(
    algorithms: Union[Dict[str, Any], List[Union[str, tuple, Any]]],
    datasets: Union[Dict[str, tuple], List[Union[str, Dict]]],
    *,
    pipeline: Optional[Union[List[Dict], str]] = None,
    cv: int = 10,
    metrics: List[str] = None,
    n_jobs: int = 1,
    verbose: int = 0,
    random_seed: Optional[int] = None,
    progress_callback: Optional[Callable] = None,
    experiment_type: Optional[str] = None,
):
    """Run experiments to compare multiple algorithms on multiple datasets.

    This function facilitates large-scale benchmarking by executing multiple 
    algorithms across various datasets using cross-validation. It uses exact 
    class names for maximum scalability and transparency.

    Parameters
    ----------
    algorithms : dict or list
        Algorithm specifications. Accepts flexible formats:

        - ``{"RF": RandomForestClassifier(), "SVM": SVM()}`` — Dict of name to instance
        - ``["RandomForestClassifier", "SVC"]`` — List of class names (uses defaults)
        - ``[{"name": "RandomForestClassifier", "params": {...}}, ...]`` —
          List of component specs (same shape as :func:`train`)
        - ``[("RF", {"name": "RandomForestClassifier", "params": {...}}), ...]``
          — ``(label, component)`` tuples when you want a custom display name;
          the component uses any of the forms above

    datasets : dict or list
        Dataset specifications:
        
        - ``{"iris": (X, y)}`` — Dict of name to (features, target) tuples
        - ``["iris", "wine"]`` — List of dataset names (loads from registry)
        - ``[{"source": "data.csv", "target": "class"}, ...]`` — List of data specs

    pipeline : str, list, or None, default=None
        Pipeline steps (same ``{"name": ..., "params": {...}}`` shape as
        :func:`train`) or a preset name, applied to all datasets.

    cv : int, default=10
        Number of cross-validation folds.

    metrics : list of str, optional
        Metrics to compute for each algorithm/dataset pair. 
        Exact function names from ``tuiml.evaluation.metrics`` (e.g.,
        ``"accuracy_score"``). Defaults are chosen per task type.

    n_jobs : int, default=1
        Number of parallel jobs to run. Use ``-1`` for all available CPUs.

    verbose : int, default=0
        Verbosity level for progress reporting.

    experiment_type : str, optional
        ``"classification"``, ``"regression"``, or ``"clustering"``. When
        omitted, the type is inferred from the models; set it explicitly for
        mixed or ambiguous collections.

    Returns
    -------
    Experiment
        Experiment object containing results, comparison tables, and 
        statistical tests (e.g., Nemenyi test).

    Examples
    --------
    Compare algorithms using class names:

    >>> exp = tuiml.experiment(
    ...     algorithms=["RandomForestClassifier", "SVC", "NaiveBayesClassifier"],
    ...     datasets=["iris", "wine"],
    ...     cv=10
    ... )

    With specific model parameters and custom metrics:

    >>> exp = tuiml.experiment(
    ...     algorithms={
    ...         "RF_100": RandomForestClassifier(n_trees=100),
    ...         "SVM_RBF": SVM(kernel="rbf")
    ...     },
    ...     datasets={"iris": (X_iris, y_iris)},
    ...     metrics=["accuracy_score", "f1_score"]
    ... )
    >>> print(exp.summary())
    """
    from tuiml.evaluation import run_experiment
    from tuiml.hub import registry
    import tuiml.algorithms  # noqa: F401 - trigger registration

    # Convert algorithms to dict format
    models_dict = {}
    if isinstance(algorithms, dict):
        for m in algorithms.values():
            _reject_foreign_estimator(m)
        models_dict = dict(algorithms)
    elif isinstance(algorithms, list):
        for item in algorithms:
            if isinstance(item, str):
                try:
                    model_class = registry.get(item)
                except KeyError:
                    raise ValueError(
                        f"Algorithm '{item}' not found in hub. "
                        f"Use list_algorithms() to see available options."
                    )
                models_dict[item] = model_class()
            elif isinstance(item, dict):
                # The same {"name": ..., "params": {...}} spec train() takes.
                name, params = _parse_component_spec(item, "algorithm")
                try:
                    model_class = registry.get(name)
                except KeyError:
                    raise ValueError(
                        f"Algorithm '{name}' not found in hub. "
                        f"Use list_algorithms() to see available options."
                    )
                models_dict[name] = model_class(**params)
            elif isinstance(item, tuple):
                if len(item) != 2:
                    raise ValueError(
                        "Algorithm tuples are (label, component), e.g. "
                        '("RF", {"name": "RandomForestClassifier", '
                        '"params": {"n_estimators": 100}}).'
                    )
                label, component = item
                if isinstance(component, dict) and "name" not in component:
                    raise ValueError(
                        f'The component for "{label}" needs the '
                        f'{{"name": ..., "params": {{...}}}} shape; a bare '
                        f"params dict does not say which algorithm to build."
                    )
                if isinstance(component, (str, dict)):
                    spec_name, spec_params = _parse_component_spec(
                        component, "algorithm"
                    )
                    try:
                        model_class = registry.get(spec_name)
                    except KeyError:
                        raise ValueError(
                            f"Algorithm '{spec_name}' not found in hub. "
                            f"Use list_algorithms() to see available options."
                        )
                    models_dict[label] = model_class(**spec_params)
                else:
                    _reject_foreign_estimator(component)
                    models_dict[label] = component
            else:
                _reject_foreign_estimator(item)
                model_name = item.__class__.__name__
                models_dict[model_name] = item

    # Convert datasets to dict format — uses TuiML loaders (csv, arff,
    # parquet, json, excel, numpy) via auto-detect, not raw pandas.
    from tuiml.datasets import load_dataset, load as load_file
    datasets_dict = {}
    if isinstance(datasets, dict):
        for name, value in datasets.items():
            if isinstance(value, str):
                # Could be a built-in name or file path
                if os.path.exists(value):
                    ds = load_file(value)
                else:
                    ds = load_dataset(value)
                datasets_dict[name] = (ds.X, ds.y)
            elif isinstance(value, tuple):
                datasets_dict[name] = value
            else:
                datasets_dict[name] = value
    elif isinstance(datasets, list):
        for item in datasets:
            if isinstance(item, str):
                # Built-in name or file path
                if os.path.exists(item):
                    ds = load_file(item)
                else:
                    ds = load_dataset(item)
                datasets_dict[item] = (ds.X, ds.y)
            elif isinstance(item, dict):
                source = item.get('source')
                if source is None:
                    raise ValueError(
                        'Dataset specs use the key "source" for the file '
                        'path, e.g. {"source": "data.csv", "target": "class"}.'
                    )
                target = item.get('target')
                name = item.get('name', source)
                # load() auto-detects format and returns Dataset
                ds = load_file(source, target_column=target)
                datasets_dict[name] = (ds.X, ds.y)
            elif isinstance(item, tuple) and len(item) == 2:
                datasets_dict[f"dataset_{len(datasets_dict)}"] = item

    # Resolve seed: explicit argument, then global seed, then 42
    if random_seed is None:
        from tuiml.utils.seed import get_global_seed
        random_seed = get_global_seed()
    if random_seed is None:
        random_seed = 42

    # A shared pipeline becomes part of each model: wrapping every model in a
    # Workflow means the steps are re-fitted inside each CV fold on that
    # fold's training data only. Transforming the datasets up front would let
    # every validation fold leak into the preprocessing (and resampling
    # would fabricate synthetic points that then land in validation folds).
    if pipeline:
        if isinstance(pipeline, str):
            if pipeline not in PRESETS:
                raise ValueError(
                    f"Unknown pipeline preset '{pipeline}'. "
                    f"Available presets: {sorted(PRESETS)}."
                )
            steps = PRESETS[pipeline]
        else:
            steps = list(pipeline)

        models_dict = {
            name: Workflow(steps + [model])
            for name, model in models_dict.items()
        }

    # Run experiment
    exp = run_experiment(
        models=models_dict,
        datasets=datasets_dict,
        n_folds=cv,
        metrics=metrics,
        random_state=random_seed,
        n_jobs=n_jobs,
        verbose=verbose,
        progress_callback=progress_callback,
        experiment_type=experiment_type,
    )

    return exp

def list_algorithms(type: Optional[str] = None) -> List[Dict]:
    """List available algorithms in the registry.

    Parameters
    ----------
    type : str, optional
        Filter by algorithm type:
        
        - ``"classifier"`` — Classification algorithms
        - ``"regressor"`` — Regression algorithms
        - ``"clusterer"`` — Clustering algorithms
        - ``None`` — List all algorithms

    Returns
    -------
    list of dict
        Metadata for matching algorithms (name, description, tags).

    Examples
    --------
    >>> classifiers = tuiml.list_algorithms(type="classifier")
    >>> for algo in classifiers:
    ...     print(f"{algo['name']}: {algo['description']}")
    """
    from tuiml.hub import registry, ComponentType

    if type:
        type_map = {
            "classifier": ComponentType.CLASSIFIER,
            "regressor": ComponentType.REGRESSOR,
            "clusterer": ComponentType.CLUSTERER,
        }
        component_type = type_map.get(type.lower())
        if component_type is None:
            raise ValueError(
                f"Invalid algorithm type '{type}'. "
                f"Valid types: 'classifier', 'regressor', 'clusterer'."
            )
        return registry.list(component_type)

    # Return all algorithms
    results = []
    for ctype in [ComponentType.CLASSIFIER, ComponentType.REGRESSOR, ComponentType.CLUSTERER]:
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

        - ``description`` — Full docstring documentation
        - ``parameters`` — JSON schema for hyperparameters
        - ``type`` — Component type (classifier, etc.)

    Examples
    --------
    >>> info = tuiml.describe_algorithm("RandomForestClassifier")
    >>> print(info["parameters"])
    """
    from tuiml.hub import registry

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
    This searches every registered component type, not just algorithms —
    transformers and feature selectors are included in the results.

    Examples
    --------
    >>> results = tuiml.search_algorithms("random forest", limit=3)
    >>> [algo["name"] for algo in results]
    ['RandomForestClassifier', 'RandomForestRegressor', 'capymoa.AdaptiveRandomForest']
    """
    from tuiml.hub import registry

    return registry.search(query, limit=limit)

# Pipeline presets — each maps a name to an ordered step list in the same
# {"name": ..., "params": {...}} shape as an explicit pipeline.
PRESETS = {
    "minimal": [],
    "fast": [
        {"name": "SimpleImputer", "params": {"strategy": "most_frequent"}},
    ],
    "standard": [
        {"name": "SimpleImputer", "params": {"strategy": "mean"}},
        {"name": "MinMaxScaler"},
        {"name": "OneHotEncoder"},
    ],
    "full": [
        {"name": "SimpleImputer", "params": {"strategy": "median"}},
        {"name": "StandardScaler"},
        {"name": "OneHotEncoder"},
        {"name": "SelectKBestSelector", "params": {"k": 10}},
    ],
    "imbalanced": [
        {"name": "SimpleImputer", "params": {"strategy": "mean"}},
        {"name": "MinMaxScaler"},
        {"name": "SMOTESampler"},
    ],
}

# ===== Model Serving Functions =====

def serve(
    model_or_path,
    host: str = "127.0.0.1",
    port: int = 8000,
    model_id: str = "default",
    background: bool = True,
):
    """Serve a trained model via REST API.

    Accepts a file path, a fitted ``Workflow``, or a model object and starts
    a uvicorn server exposing prediction endpoints.

    Parameters
    ----------
    model_or_path : str, Workflow, or model object
        The model to serve:

        - ``str`` — Path to a saved model file
        - ``Workflow`` — a fitted pipeline from ``train()`` or ``Workflow.fit()``
        - model object — Any object with a ``predict()`` method

    host : str, default="127.0.0.1"
        Host to bind the server to.

    port : int, default=8000
        Port to listen on.

    model_id : str, default="default"
        Identifier for the model in the server.

    background : bool, default=True
        If True, run the server in a daemon thread and return immediately.
        If False, block until the server is stopped.

    Returns
    -------
    dict or None
        If ``background=True``, returns a dict with ``server_id``, ``url``,
        and ``endpoints``. If ``background=False``, blocks and returns None.

    Examples
    --------
    Serve from a fitted Workflow:

    >>> model = tuiml.train("NaiveBayesClassifier", {"source": "iris"})
    >>> info = tuiml.serve(model, port=9999)
    >>> print(info["url"])
    http://127.0.0.1:9999

    Serve from a file path:

    >>> info = tuiml.serve("model.pkl", port=8000)

    Blocking mode:

    >>> tuiml.serve(result, background=False)
    """
    import tempfile

    from tuiml.serving import ModelServer
    from tuiml.utils.serialization import save_model

    server = ModelServer()

    if isinstance(model_or_path, str):
        # File path — load directly
        server.load_model(model_id, model_or_path)
    elif isinstance(model_or_path, Workflow):
        # Fitted Workflow — serialize the whole pipeline so the served model
        # applies its transformations too.
        if not model_or_path._is_fitted:
            raise RuntimeError("This Workflow is not fitted yet. Call fit() first.")
        tmp = tempfile.NamedTemporaryFile(suffix=".pkl", delete=False)
        tmp.close()
        save_model(model_or_path, tmp.name)
        server.load_model(model_id, tmp.name)
    else:
        # Assume it's a model object with predict()
        if not hasattr(model_or_path, 'predict'):
            raise ValueError("Object does not have a predict() method.")
        tmp = tempfile.NamedTemporaryFile(suffix=".pkl", delete=False)
        tmp.close()
        save_model(model_or_path, tmp.name)
        server.load_model(model_id, tmp.name)

    app = server.create_app()
    server_id = f"{host}:{port}"

    if not background:
        import uvicorn
        _SERVERS[server_id] = {
            "server_id": server_id,
            "host": host,
            "port": port,
            "model_id": model_id,
            "url": f"http://{host}:{port}",
            "server_obj": server,
        }
        try:
            uvicorn.run(app, host=host, port=port, log_level="info")
        finally:
            _SERVERS.pop(server_id, None)
        return None

    # Background mode — run in daemon thread
    import uvicorn
    import time
    import asyncio

    config = uvicorn.Config(app, host=host, port=port, log_level="warning")
    uvicorn_server = uvicorn.Server(config)

    def run_server():
        """Run uvicorn server with proper event loop handling."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(uvicorn_server.serve())
        finally:
            loop.close()

    thread = threading.Thread(target=run_server, daemon=True)
    thread.start()

    # Wait for uvicorn to actually bind before reporting success. Without this
    # a failure in the background thread — most often the port already being in
    # use — would go unnoticed and this call would hand back a URL for a server
    # that never started.
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        if getattr(uvicorn_server, "started", False):
            break
        if not thread.is_alive():
            raise RuntimeError(
                f"The server failed to start on {host}:{port}. The port is "
                f"most likely already in use — pick another port, or call "
                f"tuiml.stop_server() to shut down servers started earlier."
            )
        time.sleep(0.05)
    else:
        uvicorn_server.should_exit = True
        raise RuntimeError(
            f"The server did not become ready on {host}:{port} within 10s."
        )

    info = {
        "server_id": server_id,
        "host": host,
        "port": port,
        "model_id": model_id,
        "url": f"http://{host}:{port}",
        "endpoints": {
            "predict": f"http://{host}:{port}/models/{model_id}/predict",
            "health": f"http://{host}:{port}/health",
            "docs": f"http://{host}:{port}/docs",
        },
        "server_obj": server,
        "uvicorn_server": uvicorn_server,
        "thread": thread,
    }
    _SERVERS[server_id] = info

    # Return a clean dict without internal objects
    return {
        "server_id": server_id,
        "host": host,
        "port": port,
        "model_id": model_id,
        "url": f"http://{host}:{port}",
        "endpoints": info["endpoints"],
    }


def stop_server(server_id: Optional[str] = None) -> None:
    """Stop running model server(s).

    Parameters
    ----------
    server_id : str, optional
        The server ID (``"host:port"``) to stop. If None, stops all
        running servers.

    Examples
    --------
    Stop a specific server:

    >>> tuiml.stop_server("127.0.0.1:9999")

    Stop all servers:

    >>> tuiml.stop_server()
    """
    if server_id is not None:
        info = _SERVERS.pop(server_id, None)
        if info and "uvicorn_server" in info:
            info["uvicorn_server"].should_exit = True
    else:
        for sid in list(_SERVERS.keys()):
            info = _SERVERS.pop(sid, None)
            if info and "uvicorn_server" in info:
                info["uvicorn_server"].should_exit = True


def server_status() -> List[Dict]:
    """Get status of running model servers.

    Returns
    -------
    list of dict
        List of server info dicts, each containing ``server_id``,
        ``host``, ``port``, ``model_id``, and ``url``.

    Examples
    --------
    >>> tuiml.server_status()
    [{'server_id': '127.0.0.1:9999', 'url': 'http://127.0.0.1:9999', ...}]
    """
    return [
        {
            "server_id": info["server_id"],
            "host": info["host"],
            "port": info["port"],
            "model_id": info["model_id"],
            "url": info["url"],
        }
        for info in _SERVERS.values()
    ]
