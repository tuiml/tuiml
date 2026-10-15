"""Helpers shared by more than one tool executor.

Model persistence and dataset resolution: every executor that takes a
``model_id`` / ``model_path`` / ``data`` argument routes through here, so the
resolution rules stay identical across tools.
"""

import os
from typing import List

from ._state import _DATASET_INDEX, _MODEL_INDEX, _MODELS_DIR


def _save_model_to_disk(model, model_id: str, save_path: str = None) -> str:
    """Save model to disk and return the file path.

    Parameters
    ----------
    model : object
        Fitted model (or Workflow) to serialize.
    model_id : str
        Identifier used to name the file when no explicit path is given.
    save_path : str, default=None
        Explicit destination path. When None, the model is written to
        ``~/.tuiml/models/<model_id>.joblib``.

    Returns
    -------
    path : str
        Path of the file the model was written to.
    """
    from tuiml.utils.serialization import save_model

    if save_path:
        os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
        save_model(model, save_path)
        return save_path
    else:
        path = os.path.join(_MODELS_DIR, f'{model_id}.joblib')
        save_model(model, path)
        return path


def _load_model_from_disk(model_id: str = None, model_path: str = None):
    """Load model from disk by model_id or explicit path.

    Parameters
    ----------
    model_id : str, default=None
        Identifier previously returned by a training tool; resolved via the
        in-memory model index.
    model_path : str, default=None
        Explicit path to a serialized model file, used when the model_id is
        unknown or not indexed.

    Returns
    -------
    model : object or None
        The deserialized model, or None if neither lookup succeeded.
    """
    from tuiml.utils.serialization import load_model

    if model_id and model_id in _MODEL_INDEX:
        return load_model(_MODEL_INDEX[model_id])
    elif model_path and os.path.exists(model_path):
        return load_model(model_path)
    return None


def _load_data(data_source: str):
    """Load data from an uploaded dataset_id, file path, or built-in dataset name.

    Resolution order:
      1. Uploaded dataset_id (registered via tuiml_upload_data)
      2. Existing file path on disk
      3. Built-in dataset name (iris, diabetes, ...)

    Parameters
    ----------
    data_source : str
        Dataset identifier: an uploaded dataset_id, a path to a data file
        on disk, or the name of a built-in dataset.

    Returns
    -------
    dataset : Dataset
        Loaded dataset with ``X``, ``y`` and ``feature_names`` attributes.
    """
    from tuiml.datasets import load, load_dataset

    # 1. Uploaded dataset registered by name
    if data_source in _DATASET_INDEX:
        path = _DATASET_INDEX[data_source]
        if os.path.exists(path):
            return load(path)
        # Stale entry, drop and fall through
        _DATASET_INDEX.pop(data_source, None)

    # 2. File path on disk
    if os.path.exists(data_source):
        return load(data_source)

    # 3. Built-in dataset name
    return load_dataset(data_source)


def _get_model_tags(model) -> List[str]:
    """Get tags from a model if available.

    Parameters
    ----------
    model : object
        Model instance (or Workflow wrapping one) to inspect.

    Returns
    -------
    tags : list of str
        Instance-level or class-level ``_tags``, or an empty list.
    """
    target = getattr(model, 'model', model)
    tags = getattr(target, '_tags', [])
    if not tags:
        # Try class-level tags
        tags = getattr(target.__class__, '_tags', [])
    return tags or []


def _normalise_scores(results, metric=None):
    """Coerce any supported results shape to ``{algorithm: [scores...]}``.

    Accepts the flat shape unchanged, a whole ``tuiml_benchmark`` result
    (its ``results`` key is used), or that ``results`` sub-dict, which is
    shaped ``{dataset: {algorithm: {metric: {"mean", "std", "scores"}}}}``.
    With one dataset the per-fold ``scores`` are returned; with several, one
    value per dataset (the ``mean``), so that a Friedman test treats datasets
    as blocks, the standard multi-dataset comparison.

    Parameters
    ----------
    results : dict
        One of the shapes above.
    metric : str, default=None
        Metric to extract from a benchmark result. Defaults to the first
        metric present.

    Returns
    -------
    dict of str -> list of float
        Algorithm name -> scores.

    Raises
    ------
    ValueError
        If ``results`` is empty, or ``metric`` is not in the benchmark.
    """
    if not isinstance(results, dict) or not results:
        raise ValueError("results must be a non-empty dict")
    if "results" in results and isinstance(results["results"], dict):
        results = results["results"]
    first = next(iter(results.values()))
    if not isinstance(first, dict):
        return {name: [float(v) for v in scores] for name, scores in results.items()}

    # Benchmark shape: dataset -> algorithm -> metric -> {mean, std, scores}
    datasets = list(results)
    metrics_seen = []
    for algos in results.values():
        for per_metric in algos.values():
            for m in per_metric:
                if m not in metrics_seen:
                    metrics_seen.append(m)
    if not metrics_seen:
        raise ValueError("benchmark results contain no metrics")
    if metric is None:
        metric = metrics_seen[0]
    elif metric not in metrics_seen:
        raise ValueError(f"metric '{metric}' not in benchmark results; available: {metrics_seen}")

    out = {}
    if len(datasets) == 1:
        for algo, per_metric in results[datasets[0]].items():
            entry = per_metric[metric]
            out[algo] = [float(v) for v in entry.get("scores", [entry.get("mean")])]
        return out
    for ds in datasets:
        for algo, per_metric in results[ds].items():
            out.setdefault(algo, []).append(float(per_metric[metric]["mean"]))
    return out


def _unknown_metrics(metrics):
    """Return an error dict for metric names that do not exist, else None.

    Parameters
    ----------
    metrics : list of str or None
        Metric function names as passed by the agent.

    Returns
    -------
    dict or None
        A ``status: error`` result naming the bad metrics, close matches
        and every valid name; None when all names resolve (or none given).
    """
    if not metrics:
        return None
    import difflib
    from tuiml.evaluation import metrics as metrics_module

    valid = sorted(n for n in dir(metrics_module)
                   if n.endswith(("_score", "_error", "_loss", "_corrcoef")) and not n.startswith("_"))
    names = [metrics] if isinstance(metrics, str) else list(metrics)
    bad = [m for m in names if getattr(metrics_module, m, None) is None]
    if not bad:
        return None
    aliases = {"rmse": "root_mean_squared_error", "mse": "mean_squared_error",
               "mae": "mean_absolute_error", "r2": "r2_score", "auc": "roc_auc_score",
               "roc_auc": "roc_auc_score", "f1": "f1_score", "acc": "accuracy_score",
               "accuracy": "accuracy_score", "precision": "precision_score",
               "recall": "recall_score", "mcc": "matthews_corrcoef", "kappa": "cohen_kappa_score",
               "silhouette": "silhouette_score"}
    hints = {}
    for m in bad:
        close = [aliases[m.lower()]] if m.lower() in aliases else []
        close += [c for c in difflib.get_close_matches(m, valid, n=3, cutoff=0.5) if c not in close]
        close += [v for v in valid if m.lower() in v and v not in close][:3]
        hints[m] = close
    hint_txt = "; ".join(f"'{m}' -> {', '.join(c) or '?'}" for m, c in hints.items())
    return {
        "status": "error",
        "error": f"Unknown metric(s): {bad}. Did you mean: {hint_txt}",
        "error_type": "ValueError",
        "suggestion": f"Use exact metric function names: {valid}",
    }
