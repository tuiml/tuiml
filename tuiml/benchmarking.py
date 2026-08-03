"""Benchmark models across datasets.

:class:`Benchmark` takes a set of model specs and a set of dataset specs;
:meth:`Benchmark.run` scores every model on every dataset with
cross-validation or repeated holdout, optionally tuning each model inside
the training folds. Results live on the instance: tidy per-fold scores,
ranked tables, significance tests, paper-ready exports, and comparison
plots.

Everything is declarative, in the same ``{"name": ..., "params": {...}}``
convention as :func:`tuiml.train`, so a whole benchmark can be stored as JSON
and replayed.
"""

import os
from itertools import product
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from tuiml.workflow import Workflow, _build_step

_MODEL_KEYS = {"name", "params", "tune", "pipeline", "label"}
_DATASET_SOURCE_KEYS = {"source", "target", "features", "name"}
_DATASET_ARRAY_KEYS = {"name", "X", "y"}
_EVALUATION_KEYS = {"cv", "test_size", "stratify", "metrics", "repeats"}
_TUNE_KEYS = {"method", "space", "iterations"}

_DEFAULT_METRICS = {
    "classification": ["accuracy_score", "f1_score"],
    "regression": ["r2_score", "root_mean_squared_error"],
    "clustering": ["silhouette_score", "calinski_harabasz_score"],
}


def _higher_is_better(metric: str) -> bool:
    """Whether larger values of a metric mean a better model.

    Parameters
    ----------
    metric : str
        Metric function name.

    Returns
    -------
    bool
        False for error and loss style metrics, True otherwise.
    """
    lowered = metric.lower()
    return not any(word in lowered for word in ("error", "loss", "davies_bouldin"))


def _resolve_dataset(spec: Any, index: int):
    """Load one dataset spec into ``(name, X, y, feature_names)``.

    Parameters
    ----------
    spec : dict
        Either ``{"source": ..., "target": ..., "features": ..., "name": ...}``
        for a file or builtin dataset, or ``{"name": ..., "X": ..., "y": ...}``
        for in-memory arrays.
    index : int
        Position in the datasets list, used in error messages and default
        names.

    Returns
    -------
    tuple
        ``(display_name, X, y, feature_names)``.

    Raises
    ------
    ValueError
        For bare strings, unknown keys, missing columns, or a target passed
        with a builtin dataset.
    """
    if isinstance(spec, str):
        raise ValueError(
            f"Dataset at index {index}: bare names are not accepted; write "
            f'{{"source": {spec!r}}}.'
        )
    if not isinstance(spec, dict):
        raise ValueError(
            f"Dataset at index {index} must be a spec dict, "
            f"got {type(spec).__name__}."
        )

    if "X" in spec or "y" in spec:
        unknown = set(spec) - _DATASET_ARRAY_KEYS
        if unknown:
            raise ValueError(
                f"Unexpected dataset spec keys {sorted(unknown)} at index "
                f'{index}. In-memory specs take "name", "X", and "y".'
            )
        if "X" not in spec:
            raise ValueError(f'Dataset at index {index} has "y" but no "X".')
        name = spec.get("name", f"dataset_{index}")
        y = np.asarray(spec["y"]) if spec.get("y") is not None else None
        return name, np.asarray(spec["X"]), y, None

    unknown = set(spec) - _DATASET_SOURCE_KEYS
    if unknown:
        raise ValueError(
            f"Unexpected dataset spec keys {sorted(unknown)} at index {index}. "
            f'Allowed: "source", "target", "features", "name" '
            f'(or "name", "X", "y" for arrays).'
        )
    source = spec.get("source")
    if source is None:
        raise ValueError(
            f'Dataset at index {index} needs a "source" key: a file path or '
            f"a builtin dataset name."
        )

    from tuiml.datasets import load as load_file, load_dataset

    target = spec.get("target")
    if os.path.exists(str(source)):
        ds = load_file(source, target_column=target) if target else load_file(source)
    else:
        if target is not None:
            raise ValueError(
                f"Builtin dataset '{source}' defines its own target; remove "
                f'the "target" key.'
            )
        ds = load_dataset(source)

    X, feature_names = ds.X, list(ds.feature_names or []) or None
    features = spec.get("features")
    if features:
        if not feature_names:
            raise ValueError(
                f"Dataset '{source}' has no column names to select from."
            )
        missing = [f for f in features if f not in feature_names]
        if missing:
            raise ValueError(
                f"features not found in the columns of '{source}' "
                f"{feature_names}: {missing}"
            )
        indices = [feature_names.index(f) for f in features]
        X = np.asarray(X)[:, indices]
        feature_names = list(features)

    return spec.get("name", str(source)), X, ds.y, feature_names


def _validate_dataset_spec(spec: Any, index: int) -> None:
    """Structurally check one dataset spec without loading any data.

    The membership of ``"features"`` in the file's columns can only be
    checked at load time; everything about the spec's shape is checked here
    so a typo fails at construction.

    Parameters
    ----------
    spec : dict
        The dataset spec.
    index : int
        Position in the datasets list, for error messages.

    Raises
    ------
    ValueError
        For bare strings, non-dicts, unknown keys, a missing source, or a
        target passed with a builtin dataset.
    """
    if isinstance(spec, str):
        raise ValueError(
            f"Dataset at index {index}: bare names are not accepted; write "
            f'{{"source": {spec!r}}}.'
        )
    if not isinstance(spec, dict):
        raise ValueError(
            f"Dataset at index {index} must be a spec dict, "
            f"got {type(spec).__name__}."
        )
    if "X" in spec or "y" in spec:
        unknown = set(spec) - _DATASET_ARRAY_KEYS
        if unknown:
            raise ValueError(
                f"Unexpected dataset spec keys {sorted(unknown)} at index "
                f'{index}. In-memory specs take "name", "X", and "y".'
            )
        if "X" not in spec:
            raise ValueError(f'Dataset at index {index} has "y" but no "X".')
        return
    unknown = set(spec) - _DATASET_SOURCE_KEYS
    if unknown:
        raise ValueError(
            f"Unexpected dataset spec keys {sorted(unknown)} at index {index}. "
            f'Allowed: "source", "target", "features", "name" '
            f'(or "name", "X", "y" for arrays).'
        )
    source = spec.get("source")
    if source is None:
        raise ValueError(
            f'Dataset at index {index} needs a "source" key: a file path or '
            f"a builtin dataset name."
        )
    if spec.get("target") is not None and not os.path.exists(str(source)):
        raise ValueError(
            f"Builtin dataset '{source}' defines its own target; remove "
            f'the "target" key.'
        )


def _validate_model_spec(spec: Any, index: int) -> Dict[str, Any]:
    """Check one model spec and return it.

    Parameters
    ----------
    spec : dict
        ``{"name": ...}`` plus optional ``params``, ``tune``, ``pipeline``,
        ``label``.
    index : int
        Position in the models list, for error messages.

    Returns
    -------
    dict
        The validated spec.

    Raises
    ------
    ValueError
        For non-dict entries, unknown keys, or a malformed tune block.
    """
    if not isinstance(spec, dict) or not spec.get("name"):
        raise ValueError(
            f'Model at index {index} must be {{"name": ..., "params": '
            f"{{...}}}}, got {spec!r}. Bare names and instances are not "
            f"accepted here; instances belong to tuiml.Workflow."
        )
    unknown = set(spec) - _MODEL_KEYS
    if unknown:
        raise ValueError(
            f"Unexpected keys {sorted(unknown)} in model spec "
            f"'{spec['name']}'. Allowed: {sorted(_MODEL_KEYS)}."
        )
    tune = spec.get("tune")
    if tune is not None:
        unknown = set(tune) - _TUNE_KEYS
        if unknown:
            raise ValueError(
                f"Unexpected tune keys {sorted(unknown)} for "
                f"'{spec['name']}'. Allowed: {sorted(_TUNE_KEYS)}."
            )
        if not tune.get("space"):
            raise ValueError(
                f"The tune block for '{spec['name']}' needs a \"space\" dict "
                f"of parameter name to candidate values."
            )
        method = tune.get("method", "grid")
        if method not in ("grid", "random"):
            raise ValueError(
                f"Unknown tune method '{method}' for '{spec['name']}'. "
                f'Supported: "grid", "random".'
            )
    return spec


def _candidate_grid(space: Dict[str, List], method: str, iterations: int,
                    rng: np.random.RandomState) -> List[Dict[str, Any]]:
    """Expand a search space into candidate parameter dicts.

    Parameters
    ----------
    space : dict
        Parameter name to list of candidate values.
    method : {"grid", "random"}
        Full cartesian product, or a random sample of it.
    iterations : int
        Sample size for the random method.
    rng : np.random.RandomState
        Source of randomness for the random method.

    Returns
    -------
    list of dict
        Candidate parameter combinations.
    """
    keys = sorted(space)
    combos = [dict(zip(keys, values))
              for values in product(*(space[k] for k in keys))]
    if method == "random" and len(combos) > iterations:
        picks = rng.choice(len(combos), size=iterations, replace=False)
        combos = [combos[i] for i in picks]
    return combos


def _make_splits(X, y, task, cv, test_size, stratify, repeats, seed):
    """Build the ``(train_idx, test_idx)`` splits for one dataset.

    Parameters
    ----------
    X, y : np.ndarray
        The dataset.
    task : str
        Benchmark task; stratification only applies to classification.
    cv : int or None
        Number of folds for k-fold.
    test_size : float or None
        Holdout fraction, used when ``cv`` is None.
    stratify : bool
        Keep class balance where the task allows it.
    repeats : int
        Repeat the whole scheme this many times with shifted seeds.
    seed : int
        Base random seed.

    Returns
    -------
    list of tuple
        ``(train_indices, test_indices)`` pairs.
    """
    from tuiml.evaluation.splitting import KFold, StratifiedKFold, train_test_split

    use_stratify = stratify and task == "classification" and y is not None
    splits = []
    for repeat in range(max(1, repeats)):
        repeat_seed = seed + repeat
        if cv:
            splitter = (StratifiedKFold if use_stratify else KFold)(
                n_splits=cv, shuffle=True, random_state=repeat_seed
            )
            splits.extend(splitter.split(X, y))
        else:
            indices = np.arange(len(X))
            train_idx, test_idx = train_test_split(
                indices, test_size=test_size, random_state=repeat_seed,
                stratify=y if use_stratify else None,
            )
            splits.append((train_idx, test_idx))
    return splits


def _tune_on_fold(build_workflow, tune, X_tr, y_tr, metric, seed, rng):
    """Pick the best hyperparameters using only this fold's training data.

    An inner 3-fold split scores every candidate; the winner is what the
    outer fold trains with. Tuning never sees the outer validation data,
    which is what keeps the reported scores honest.

    Parameters
    ----------
    build_workflow : callable
        Factory producing a fresh pipeline given extra model params.
    tune : dict
        The model's tune block: method, space, iterations.
    X_tr, y_tr : np.ndarray
        The outer fold's training data.
    metric : str
        Metric name used for selection (the first requested one).
    seed : int
        Seed for the inner folds.
    rng : np.random.RandomState
        Source of randomness for random search.

    Returns
    -------
    dict
        The winning parameter combination.
    """
    from tuiml.evaluation.splitting import KFold

    candidates = _candidate_grid(
        tune["space"], tune.get("method", "grid"),
        int(tune.get("iterations", 25)), rng,
    )
    if len(candidates) == 1:
        return candidates[0]

    higher = _higher_is_better(metric)
    inner = list(
        KFold(n_splits=3, shuffle=True, random_state=seed).split(X_tr, y_tr)
    )
    best_params, best_score = candidates[0], None
    for candidate in candidates:
        inner_scores = []
        for inner_train, inner_val in inner:
            workflow = build_workflow(candidate).fit(
                X_tr[inner_train], y_tr[inner_train]
            )
            result = workflow.evaluate(
                X_tr[inner_val], y_tr[inner_val], metrics=[metric]
            )
            if metric in result:
                inner_scores.append(result[metric])
        if not inner_scores:
            continue
        score = float(np.mean(inner_scores))
        if best_score is None or (
            score > best_score if higher else score < best_score
        ):
            best_params, best_score = candidate, score
    return best_params


def _report(progress_callback, verbose, dataset, model, done, total):
    """Emit progress to the callback and, when verbose, to stdout."""
    if progress_callback:
        progress_callback({"type": "benchmark_progress", "dataset": dataset,
                           "model": model, "fold": done, "folds": total})
    if verbose:
        print(f"  [{dataset}] {model}: fold {done}/{total}")




class Benchmark:
    """A declarative model comparison: configure it, then :meth:`run` it.

    The constructor validates the whole configuration up front, so a typo
    fails immediately, not minutes into the run. :meth:`run` executes the
    comparison and stores the results on the instance: :attr:`scores_`,
    :attr:`best_params_`, and every table, test, export, and plot method.

    Parameters
    ----------
    models : list of dict
        Model specs, each ``{"name": ..., "params": {...}}`` plus optional
        keys:

        - ``"tune"``: ``{"method": "grid" | "random", "space": {param:
          [values]}, "iterations": 25}``. Tuning runs on the training side
          of every fold with an inner 3-fold split, so reported scores stay
          honest.
        - ``"pipeline"``: per-model step list overriding the shared
          ``pipeline`` (for example scaling for an SVM but not for trees).
        - ``"label"``: display name; defaults to the model name.
    datasets : list of dict
        Dataset specs, always in spec form:

        - ``{"source": "iris"}``: a builtin dataset (builtins define their
          own target).
        - ``{"source": "sales.csv", "target": "label", "features": [...],
          "name": "sales"}``: a data file; ``"features"`` restricts the
          columns, ``"name"`` sets the display name.
        - ``{"name": "custom", "X": X_array, "y": y_array}``: in-memory
          arrays.
    pipeline : str or list of dict, optional
        Steps applied before every model, as component specs, or a preset
        name from :data:`tuiml.training.PRESETS`.
    evaluation : dict, optional
        ``{"cv": 10}`` for k-fold (the default), or ``{"test_size": 0.2}``
        for holdout. ``"repeats"`` repeats either scheme with fresh seeds,
        which also gives holdout a spread for statistics. ``"stratify"``
        (default True) keeps class balance, and ``"metrics"`` lists metric
        function names from ``tuiml.evaluation.metrics``.
    random_seed : int, optional
        Seeds the splits and every component that accepts a seed, making
        the run reproducible. Falls back to the global seed, then 42.
    task : {"classification", "regression", "clustering"}, optional
        Overrides the task inferred from the models.

    Attributes
    ----------
    scores_ : pd.DataFrame
        Tidy per-fold scores with columns ``dataset``, ``model``,
        ``metric``, ``fold``, ``value``. Set by :meth:`run`.
    best_params_ : pd.DataFrame
        Tuning choices per dataset, model, and fold. Set by :meth:`run`.
    task : str
        The resolved task.
    metrics : list of str
        The metric names to compute; the first is the primary one.
    random_seed : int
        The resolved seed.

    Examples
    --------
    Step 1, import and configure:

    >>> from tuiml import Benchmark
    >>> bench = Benchmark(
    ...     models=[
    ...         {"name": "NaiveBayesClassifier"},
    ...         {"name": "RandomForestClassifier",
    ...          "tune": {"method": "grid",
    ...                   "space": {"n_estimators": [50, 100]}}},
    ...     ],
    ...     datasets=[
    ...         {"source": "iris"},
    ...         {"source": "sales.csv", "target": "label"},
    ...     ],
    ...     evaluation={"cv": 10, "metrics": ["accuracy_score"]},
    ...     random_seed=42,
    ... )

    Step 2, execute:

    >>> bench.run()                                        # doctest: +SKIP

    Step 3, read the results:

    >>> print(bench.summary())                             # doctest: +SKIP
    >>> bench.table()                                      # doctest: +SKIP
    >>> bench.compare(baseline="NaiveBayesClassifier")     # doctest: +SKIP
    >>> bench.to_latex()                                   # doctest: +SKIP
    """

    def __init__(
        self,
        models: List[Dict[str, Any]],
        datasets: List[Dict[str, Any]],
        *,
        pipeline: Optional[Union[str, List[Dict]]] = None,
        evaluation: Optional[Dict[str, Any]] = None,
        random_seed: Optional[int] = None,
        task: Optional[str] = None,
    ):
        from tuiml.registry import registry, ComponentType
        import tuiml.algorithms  # noqa: F401 (ensures registration)
        from tuiml.training import PRESETS

        if not models:
            raise ValueError("Benchmark needs at least one model spec.")
        if not datasets:
            raise ValueError("Benchmark needs at least one dataset spec.")
        self._model_specs = [_validate_model_spec(m, i)
                             for i, m in enumerate(models)]
        for i, spec in enumerate(datasets):
            _validate_dataset_spec(spec, i)
        self._dataset_specs = list(datasets)

        # Evaluation options
        evaluation = dict(evaluation or {})
        unknown = set(evaluation) - _EVALUATION_KEYS
        if unknown:
            raise ValueError(
                f"Unexpected evaluation keys {sorted(unknown)}. "
                f"Allowed: {sorted(_EVALUATION_KEYS)}."
            )
        cv = evaluation.get("cv")
        test_size = evaluation.get("test_size")
        if cv and test_size:
            raise ValueError('Pass either "cv" or "test_size", not both.')
        if not cv and not test_size:
            cv = 10
        self._cv = cv
        self._test_size = test_size
        self._repeats = int(evaluation.get("repeats", 1))
        self._stratify = evaluation.get("stratify", True)

        # Seed: explicit, then global, then 42
        if random_seed is None:
            from tuiml.utils.seed import get_global_seed
            random_seed = get_global_seed()
        if random_seed is None:
            random_seed = 42
        self.random_seed = random_seed

        # Task: explicit override, else inferred from the model components
        if task is None:
            kinds = set()
            for spec in self._model_specs:
                try:
                    cls = registry.get(spec["name"])
                except KeyError:
                    raise ValueError(
                        f"Algorithm '{spec['name']}' not found in hub. "
                        f"Use list_algorithms() to see available options."
                    )
                component_type = getattr(cls, "_component_type", None)
                if component_type == ComponentType.REGRESSOR:
                    kinds.add("regression")
                elif component_type == ComponentType.CLUSTERER:
                    kinds.add("clustering")
                else:
                    kinds.add("classification")
            task = kinds.pop() if len(kinds) == 1 else "classification"
        elif task not in _DEFAULT_METRICS:
            raise ValueError(
                f"Unknown task '{task}'. Allowed: {sorted(_DEFAULT_METRICS)}."
            )
        self.task = task
        self.metrics = list(evaluation.get("metrics") or _DEFAULT_METRICS[task])

        # Shared pipeline: preset name or step list
        if isinstance(pipeline, str):
            if pipeline not in PRESETS:
                raise ValueError(
                    f"Unknown pipeline preset '{pipeline}'. "
                    f"Available presets: {sorted(PRESETS)}."
                )
            pipeline = PRESETS[pipeline]
        self._pipeline = pipeline

    def run(self, verbose: int = 0,
            progress_callback: Optional[Callable] = None) -> "Benchmark":
        """Execute the benchmark and store the results on this instance.

        Parameters
        ----------
        verbose : int, default=0
            Print progress while running.
        progress_callback : callable, optional
            Called after every fold with a progress dict.

        Returns
        -------
        self : Benchmark
            With :attr:`scores_` and :attr:`best_params_` populated.
        """
        rng = np.random.RandomState(self.random_seed)
        records: List[Dict[str, Any]] = []
        tuned_records: List[Dict[str, Any]] = []
        random_seed = self.random_seed
        task = self.task
        metrics = self.metrics

        for ds_index, ds_spec in enumerate(self._dataset_specs):
            ds_name, X, y, _ = _resolve_dataset(ds_spec, ds_index)
            X = np.asarray(X)

            for spec in self._model_specs:
                label = spec.get("label", spec["name"])
                steps = spec.get("pipeline", self._pipeline) or []

                def build_workflow(extra_params: Optional[Dict] = None) -> Workflow:
                    """A fresh pipeline for this model with optional overrides."""
                    model_spec = {
                        "name": spec["name"],
                        "params": {**(spec.get("params") or {}),
                                   **(extra_params or {})},
                    }
                    built = [_build_step(s, random_seed) for s in steps]
                    built.append(_build_step(model_spec, random_seed))
                    return Workflow(built)

                if task == "clustering":
                    # Unsupervised: fit on all data, score quality once.
                    workflow = build_workflow().fit(X, random_seed=random_seed)
                    for metric in metrics:
                        value = (workflow.metrics_ or {}).get(metric)
                        if value is not None:
                            records.append({"dataset": ds_name, "model": label,
                                            "metric": metric, "fold": 0,
                                            "value": float(value)})
                    _report(progress_callback, verbose, ds_name, label, 1, 1)
                    continue

                splits = _make_splits(X, y, task, self._cv, self._test_size,
                                      self._stratify, self._repeats, random_seed)
                for fold, (train_idx, test_idx) in enumerate(splits):
                    X_tr, y_tr = X[train_idx], y[train_idx]
                    X_te, y_te = X[test_idx], y[test_idx]

                    chosen: Dict[str, Any] = {}
                    if spec.get("tune"):
                        chosen = _tune_on_fold(
                            build_workflow, spec["tune"], X_tr, y_tr,
                            metrics[0], random_seed, rng,
                        )
                        tuned_records.append({"dataset": ds_name,
                                              "model": label, "fold": fold,
                                              "params": chosen})

                    workflow = build_workflow(chosen).fit(X_tr, y_tr)
                    fold_scores = workflow.evaluate(X_te, y_te, metrics=metrics)
                    for metric, value in fold_scores.items():
                        if isinstance(value, (int, float, np.floating)):
                            records.append({"dataset": ds_name, "model": label,
                                            "metric": metric, "fold": fold,
                                            "value": float(value)})
                    _report(progress_callback, verbose, ds_name, label,
                            fold + 1, len(splits))

        self.scores_ = pd.DataFrame.from_records(
            records, columns=["dataset", "model", "metric", "fold", "value"]
        )
        self.best_params_ = pd.DataFrame.from_records(
            tuned_records, columns=["dataset", "model", "fold", "params"]
        )
        return self

    def _check_run(self) -> None:
        """Raise a clear error when results are read before :meth:`run`."""
        if not hasattr(self, "scores_"):
            raise RuntimeError(
                "This Benchmark has not run yet. Call .run() first."
            )

    # ----- core views ----------------------------------------------------

    def _metric(self, metric: Optional[str]) -> str:
        """Resolve the metric argument, defaulting to the primary metric."""
        self._check_run()
        if metric is None:
            return self.metrics[0]
        if metric not in self.metrics:
            raise ValueError(
                f"Metric '{metric}' was not computed. "
                f"Available: {self.metrics}."
            )
        return metric

    def table(self, metric: Optional[str] = None,
              formatted: bool = True) -> pd.DataFrame:
        """Datasets by models table of scores for one metric.

        Parameters
        ----------
        metric : str, optional
            Which metric to tabulate; defaults to the primary one.
        formatted : bool, default=True
            True gives ``"mean ± std"`` strings; False gives numeric means.

        Returns
        -------
        pd.DataFrame
            One row per dataset, one column per model.
        """
        metric = self._metric(metric)
        subset = self.scores_[self.scores_["metric"] == metric]
        means = subset.pivot_table(index="dataset", columns="model",
                                   values="value", aggfunc="mean", sort=False)
        if not formatted:
            return means
        stds = subset.pivot_table(index="dataset", columns="model",
                                  values="value", aggfunc="std",
                                  sort=False).fillna(0.0)
        formatted_cells = means.copy().astype(object)
        for dataset in means.index:
            for model in means.columns:
                formatted_cells.loc[dataset, model] = (
                    f"{means.loc[dataset, model]:.4f} "
                    f"± {stds.loc[dataset, model]:.4f}"
                )
        return formatted_cells

    def best(self, metric: Optional[str] = None) -> pd.DataFrame:
        """The winning model per dataset, plus the overall mean rank.

        Parameters
        ----------
        metric : str, optional
            Which metric decides; defaults to the primary one.

        Returns
        -------
        pd.DataFrame
            Columns ``best_model`` and ``best_score`` indexed by dataset,
            with an ``overall`` row naming the best mean rank.
        """
        metric = self._metric(metric)
        higher = _higher_is_better(metric)
        means = self.table(metric, formatted=False)
        best_models = means.idxmax(axis=1) if higher else means.idxmin(axis=1)
        best_scores = means.max(axis=1) if higher else means.min(axis=1)
        ranks = means.rank(axis=1, ascending=not higher)
        mean_ranks = ranks.mean(axis=0)
        result = pd.DataFrame({"best_model": best_models,
                               "best_score": best_scores})
        result.loc["overall"] = [mean_ranks.idxmin(), float(mean_ranks.min())]
        return result


    def compare(self, baseline: str, metric: Optional[str] = None,
                significance_level: float = 0.05) -> pd.DataFrame:
        """Paired significance tests of every model against a baseline.

        Parameters
        ----------
        baseline : str
            The model label to compare against.
        metric : str, optional
            Which metric to test; defaults to the primary one.
        significance_level : float, default=0.05
            Alpha for the paired t-test.

        Returns
        -------
        pd.DataFrame
            One row per (dataset, model): means, p-value, significance,
            and the winner (or ``"tie"``).
        """
        from tuiml.evaluation.statistics import paired_t_test

        metric = self._metric(metric)
        higher = _higher_is_better(metric)
        subset = self.scores_[self.scores_["metric"] == metric]
        if baseline not in set(subset["model"]):
            raise ValueError(
                f"Baseline '{baseline}' is not in this benchmark. "
                f"Models: {sorted(set(subset['model']))}."
            )

        rows = []
        for dataset, per_ds in subset.groupby("dataset", sort=False):
            base = per_ds[per_ds["model"] == baseline] \
                .sort_values("fold")["value"].values
            for model, per_model in per_ds.groupby("model", sort=False):
                if model == baseline:
                    continue
                other = per_model.sort_values("fold")["value"].values
                n = min(len(base), len(other))
                if n < 2:
                    rows.append({
                        "dataset": dataset, "model": model,
                        "mean": float(np.mean(other)),
                        "baseline_mean": float(np.mean(base)),
                        "p_value": float("nan"), "significant": False,
                        "winner": "n/a (needs repeated folds)",
                    })
                    continue
                stats = paired_t_test(other[:n], base[:n],
                                      significance_level=significance_level,
                                      higher_better=higher)
                from tuiml.evaluation.statistics import SignificanceLevel
                significant = stats.significance != SignificanceLevel.TIE
                if stats.significance == SignificanceLevel.WIN:
                    winner = model
                elif stats.significance == SignificanceLevel.LOSS:
                    winner = baseline
                else:
                    winner = "tie"
                rows.append({
                    "dataset": dataset, "model": model,
                    "mean": float(np.mean(other)),
                    "baseline_mean": float(np.mean(base)),
                    "p_value": float(stats.p_value),
                    "significant": significant, "winner": winner,
                })
        return pd.DataFrame(rows)

    # ----- reports and exports -------------------------------------------

    def summary(self) -> str:
        """A readable text report of the primary metric per dataset.

        Returns
        -------
        str
            The report.
        """
        self._check_run()
        lines = [
            f"Benchmark: {self.task}, "
            f"{self.scores_['model'].nunique()} models x "
            f"{self.scores_['dataset'].nunique()} datasets, "
            f"seed {self.random_seed}"
        ]
        primary = self.metrics[0]
        primary_scores = self.scores_[self.scores_["metric"] == primary]
        for dataset, per_ds in primary_scores.groupby("dataset", sort=False):
            lines.append("")
            lines.append(f"{dataset}  ({primary})")
            lines.append("-" * 50)
            grouped = per_ds.groupby("model", sort=False)["value"]
            means = grouped.mean()
            stds = grouped.std().fillna(0.0)
            best = (means.idxmax() if _higher_is_better(primary)
                    else means.idxmin())
            for model, mean in means.items():
                marker = "  <- best" if model == best else ""
                lines.append(
                    f"  {model}: {mean:.4f} ± {stds[model]:.4f}{marker}"
                )
        if len(self.best_params_):
            lines.append("")
            lines.append("Tuned models: see .best_params_ for chosen values.")
        return "\n".join(lines)

    def to_markdown(self, metric: Optional[str] = None) -> str:
        """The scores table as GitHub-flavored markdown, winners in bold.

        Parameters
        ----------
        metric : str, optional
            Which metric to export; defaults to the primary one.

        Returns
        -------
        str
            A markdown table.
        """
        metric = self._metric(metric)
        formatted = self.table(metric)
        means = self.table(metric, formatted=False)
        higher = _higher_is_better(metric)
        header = "| Dataset | " + " | ".join(formatted.columns) + " |"
        divider = "|" + "---|" * (len(formatted.columns) + 1)
        rows = []
        for dataset in formatted.index:
            best = (means.loc[dataset].idxmax() if higher
                    else means.loc[dataset].idxmin())
            cells = [
                f"**{formatted.loc[dataset, model]}**" if model == best
                else str(formatted.loc[dataset, model])
                for model in formatted.columns
            ]
            rows.append(f"| {dataset} | " + " | ".join(cells) + " |")
        return "\n".join([header, divider] + rows)

    def to_latex(self, metric: Optional[str] = None) -> str:
        """The scores table as a LaTeX table, winners in bold.

        Parameters
        ----------
        metric : str, optional
            Which metric to export; defaults to the primary one.

        Returns
        -------
        str
            A LaTeX table environment.
        """
        metric = self._metric(metric)
        means = self.table(metric, formatted=False)
        subset = self.scores_[self.scores_["metric"] == metric]
        stds = subset.pivot_table(index="dataset", columns="model",
                                  values="value", aggfunc="std",
                                  sort=False).fillna(0.0)
        higher = _higher_is_better(metric)
        columns = list(means.columns)
        lines = [
            "\\begin{table}[htbp]",
            f"\\caption{{{metric.replace('_', ' ')} comparison}}",
            "\\centering",
            "\\begin{tabular}{l" + "c" * len(columns) + "}",
            "\\hline",
            "Dataset & "
            + " & ".join(c.replace("_", "\\_") for c in columns) + " \\\\",
            "\\hline",
        ]
        for dataset in means.index:
            best = (means.loc[dataset].idxmax() if higher
                    else means.loc[dataset].idxmin())
            cells = []
            for model in columns:
                cell = (f"${means.loc[dataset, model]:.4f} "
                        f"\\pm {stds.loc[dataset, model]:.4f}$")
                cells.append(f"\\textbf{{{cell}}}" if model == best else cell)
            lines.append(f"{dataset} & " + " & ".join(cells) + " \\\\")
        lines += ["\\hline", "\\end{tabular}", "\\end{table}"]
        return "\n".join(lines)

    def to_csv(self, path: Optional[str] = None) -> Optional[str]:
        """Write (or return) the tidy per-fold scores as CSV.

        Parameters
        ----------
        path : str, optional
            File to write. When omitted, the CSV text is returned.

        Returns
        -------
        str or None
            The CSV text when no path was given.
        """
        self._check_run()
        return self.scores_.to_csv(path, index=False)

    def to_html(self, metric: Optional[str] = None) -> str:
        """The scores table as an HTML table.

        Parameters
        ----------
        metric : str, optional
            Which metric to export; defaults to the primary one.

        Returns
        -------
        str
            An HTML table.
        """
        return self.table(metric).to_html()

    # ----- plots ----------------------------------------------------------

    def _per_dataset_means(self, metric: Optional[str]) -> Dict[str, np.ndarray]:
        """Model name to per-dataset mean scores, for comparison plots."""
        means = self.table(self._metric(metric), formatted=False)
        return {model: means[model].values for model in means.columns}

    def plot_critical_difference(self, metric: Optional[str] = None, **kwargs):
        """Critical difference diagram over datasets (Demsar style).

        Parameters
        ----------
        metric : str, optional
            Which metric to rank by; defaults to the primary one.
        **kwargs
            Forwarded to
            :func:`tuiml.evaluation.visualization.plot_critical_difference`.
        """
        from tuiml.evaluation.visualization import plot_critical_difference

        kwargs.setdefault(
            "lower_better", not _higher_is_better(self._metric(metric))
        )
        return plot_critical_difference(self._per_dataset_means(metric), **kwargs)

    def plot_boxplot(self, metric: Optional[str] = None, **kwargs):
        """Per-model box plots of every fold score.

        Parameters
        ----------
        metric : str, optional
            Which metric to plot; defaults to the primary one.
        **kwargs
            Forwarded to
            :func:`tuiml.evaluation.visualization.plot_boxplot_comparison`.
        """
        from tuiml.evaluation.visualization import plot_boxplot_comparison

        metric = self._metric(metric)
        subset = self.scores_[self.scores_["metric"] == metric]
        folds = {model: group["value"].values
                 for model, group in subset.groupby("model", sort=False)}
        return plot_boxplot_comparison(folds, **kwargs)

    def plot_ranking(self, metric: Optional[str] = None, **kwargs):
        """Mean-rank table plot across datasets.

        Parameters
        ----------
        metric : str, optional
            Which metric to rank by; defaults to the primary one.
        **kwargs
            Forwarded to
            :func:`tuiml.evaluation.visualization.plot_ranking_table`.
        """
        from tuiml.evaluation.visualization import plot_ranking_table

        metric = self._metric(metric)
        means = self.table(metric, formatted=False)
        kwargs.setdefault("dataset_names", list(means.index))
        kwargs.setdefault(
            "lower_better", not _higher_is_better(metric)
        )
        scores = {model: means[model].values for model in means.columns}
        return plot_ranking_table(scores, **kwargs)

    def __repr__(self) -> str:
        if not hasattr(self, "scores_"):
            return (f"Benchmark({len(self._model_specs)} models x "
                    f"{len(self._dataset_specs)} datasets, "
                    f"metrics={self.metrics}, not run)")
        models = self.scores_["model"].nunique()
        datasets = self.scores_["dataset"].nunique()
        return (f"Benchmark({self.task}, {models} models x {datasets} "
                f"datasets, metrics={self.metrics})")
