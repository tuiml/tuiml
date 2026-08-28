"""Train models from one declarative spec.

The high-level entry point of the library: :func:`train` takes a single
spec dict describing the model, the data, the pipeline, and the evaluation,
builds the corresponding :class:`~tuiml.workflow.Workflow`, and returns it
fitted. :data:`PRESETS` provides ready-made pipeline step lists.
"""

from typing import Any, Dict, Union

from tuiml._specs import _reject_foreign_estimator, _resolve_data_spec
from tuiml.workflow import Workflow

_SPEC_KEYS = {"model", "data", "target", "features", "pipeline", "evaluation", "random_seed"}
_EVALUATION_KEYS = {"cv", "test_size", "stratify", "metrics"}


def train(spec: Union[Dict[str, Any], str]) -> "Workflow":
    """Train a model from one declarative spec.

    This is the high-level entry point. The whole run is described by a
    single dict (or a path to a JSON file containing it). Every component
    is written the same way, as ``{"name": ..., "params": {...}}``, so a
    spec can be stored, diffed, sent over the wire, and replayed.

    To work with imported classes and configured instances instead, use
    :class:`tuiml.Workflow` directly. It takes the same pipeline as a list
    of objects.

    Parameters
    ----------
    spec : dict or str
        The experiment spec, or a path to a ``.json`` file containing it.
        Keys:

        ``model`` (required)
            What to train, as ``{"name": "RandomForestClassifier", "params":
            {"n_estimators": 100}}``. ``"params"`` may be omitted for
            defaults; hyperparameters outside it are rejected.
        ``data`` (required)
            What to train on. ``{"source": "sales.csv", "target": "label"}``
            for a file (csv, arff, parquet, json, excel, auto-detected) or a
            builtin dataset name, ``{"X": X_array, "y": y_array}`` for
            in-memory arrays, or a bare source such as ``"iris"``.
        ``target``, ``features`` (optional)
            Companions for a bare ``data`` source: the target column name
            (or array) and a column subset to keep.
        ``pipeline`` (optional)
            Ordered steps applied before the model. Each step is a
            component spec; imputation, scaling, encoding, feature
            generation, extraction, selection, and resampling are all
            steps. A preset name (``"minimal"``, ``"fast"``, ``"standard"``,
            ``"full"``, ``"imbalanced"``) expands to a predefined list.
        ``evaluation`` (optional)
            How to score the run. ``{"cv": 10}`` for k-fold, or
            ``{"test_size": 0.2, "stratify": True}`` for a holdout split,
            plus ``"metrics"``: a list of metric function names from
            ``tuiml.evaluation.metrics`` (default ``"auto"``). Omitted
            entirely: a stratified 80/20 holdout with auto metrics.
        ``random_seed`` (optional)
            Seed for splits, folds, and any component that accepts one.
            Falls back to the global seed, then 42.

    Returns
    -------
    Workflow
        The fitted pipeline. It behaves like a model: ``metrics_`` holds
        the evaluation scores, and ``predict(X)``, ``score(X, y)``,
        ``save(path)``, ``serve(port=...)`` are all available on it.

    Raises
    ------
    ValueError
        If required keys are missing, an unknown key is present, or a
        component spec carries loose hyperparameter keys.

    Examples
    --------
    >>> import tuiml
    >>> model = tuiml.train({
    ...     "model": {"name": "RandomForestClassifier",
    ...               "params": {"n_estimators": 100}},
    ...     "data": {"source": "sales.csv", "target": "label"},
    ...     "pipeline": [
    ...         {"name": "SimpleImputer", "params": {"strategy": "mean"}},
    ...         {"name": "StandardScaler"},
    ...         {"name": "SelectKBestSelector", "params": {"k": 10}},
    ...     ],
    ...     "evaluation": {"cv": 10, "metrics": ["accuracy_score"]},
    ...     "random_seed": 42,
    ... })
    >>> model.metrics_                                    # doctest: +SKIP
    {'cv_accuracy_score_mean': 0.94, 'cv_accuracy_score_std': 0.02}

    The same spec from a JSON file:

    >>> model = tuiml.train("experiment.json")            # doctest: +SKIP

    In-memory arrays:

    >>> model = tuiml.train({
    ...     "model": {"name": "NaiveBayesClassifier"},
    ...     "data": {"X": X, "y": y},
    ... })                                                # doctest: +SKIP
    """
    from tuiml.workflow import Workflow, _build_step

    if isinstance(spec, str):
        import json
        with open(spec, "r") as f:
            spec = json.load(f)

    if not isinstance(spec, dict):
        raise TypeError(
            "train() takes one spec dict (or a path to a JSON file), e.g. "
            'train({"model": "RandomForestClassifier", "data": {"source": '
            '"iris"}}). To pass configured instances, use tuiml.Workflow.'
        )

    unknown = set(spec) - _SPEC_KEYS
    if unknown:
        raise ValueError(
            f"Unknown spec keys {sorted(unknown)}. "
            f"Allowed: {sorted(_SPEC_KEYS)}."
        )

    def _require_component(value, where):
        """Every component uses one shape: {"name": ..., "params": {...}}."""
        if isinstance(value, dict) and value.get("name"):
            return
        if isinstance(value, str):
            hint = f'Bare names are not accepted; write {{"name": {value!r}}}.'
        else:
            hint = (
                "To use configured instances, build the pipeline with "
                "tuiml.Workflow instead."
            )
        raise ValueError(
            f'The {where} must be {{"name": ..., "params": {{...}}}}, '
            f"got {value!r}. {hint}"
        )

    model = spec.get("model")
    if model is None:
        raise ValueError(
            'The spec needs a "model" key, e.g. '
            '{"name": "RandomForestClassifier", "params": {...}}.'
        )
    # Before _require_component, not after. _require_component accepts only a
    # {"name": ...} dict, and _reject_foreign_estimator returns immediately for
    # dicts, so running it afterwards meant it could never fire. Someone who
    # passes an SVC() instance should be told which wrapper to name, not given
    # the generic "must be {name: ...}" message.
    _reject_foreign_estimator(model)
    _require_component(model, '"model" spec')
    if spec.get("data") is None:
        raise ValueError(
            'The spec needs a "data" key, e.g. '
            '{"source": "sales.csv", "target": "label"}.'
        )

    data, target, features = _resolve_data_spec(
        spec["data"], spec.get("target"), spec.get("features")
    )
    if data is None:
        raise ValueError(
            'The data spec resolved to nothing. Use {"source": ...} for a '
            'file or builtin name, or {"X": ..., "y": ...} for arrays.'
        )
    # Unpack the evaluation options
    evaluation = dict(spec.get("evaluation") or {})
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

    # Resolve the seed early so every component built below receives it.
    random_seed = spec.get("random_seed")
    if random_seed is None:
        from tuiml.utils.seed import get_global_seed
        random_seed = get_global_seed()
    if random_seed is None:
        random_seed = 42

    # Resolve a pipeline preset name to its step list.
    pipeline = spec.get("pipeline")
    if isinstance(pipeline, str):
        if pipeline not in PRESETS:
            raise ValueError(
                f"Unknown pipeline preset '{pipeline}'. "
                f"Available presets: {sorted(PRESETS)}."
            )
        pipeline = PRESETS[pipeline]

    # A spec is a Workflow written as data: build each step, then the model
    # as the final step, and hand the instances to the object pipeline.
    for position, step in enumerate(pipeline or []):
        _require_component(step, f"pipeline step at index {position}")
    steps = [_build_step(step, random_seed) for step in pipeline or []]
    steps.append(_build_step(model, random_seed))

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
