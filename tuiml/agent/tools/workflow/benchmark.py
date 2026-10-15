"""Multi-algorithm comparison."""

from typing import Any, Dict

from .._spec import ToolSpec
from .._shared import _load_data, _unknown_metrics


def execute_benchmark(**kwargs) -> Dict[str, Any]:
    """Execute a model comparison via ``tuiml.Benchmark``.

    Backs the ``tuiml_benchmark`` tool: runs every algorithm on every
    dataset with cross-validation and aggregates per-fold scores.

    Parameters
    ----------
    data : str or list of str
        Dataset name(s) to benchmark on: dataset_ids, file paths, or
        built-in names (arrives via ``**kwargs``, like all parameters
        below).
    algorithms : list
        Algorithms to compare; entries are names or
        ``{"name": ..., "params": {...}}`` dicts.
    cv : int, default=10
        Number of cross-validation folds.
    metrics : list of str, default=None
        Metrics to compute; defaults to the benchmark's auto selection.
    random_seed : int, default=None
        Random seed for reproducible folds.
    _progress_callback : callable, default=None
        Internal per-fold progress hook; stripped from recorded args.

    Returns
    -------
    result : dict
        On success: ``status`` (``'success'``), ``summary`` (text),
        ``table_markdown``, ``results`` (nested ``{dataset: {model:
        {metric: {mean, std, scores}}}}``), ``algorithms``, ``datasets``,
        ``cv_folds``, ``random_seed``, and optionally ``progress_log`` and
        ``research_log_updates``. On failure: ``status`` (``'error'``)
        and ``error``.
    """
    import numpy as np
    import tuiml

    try:
        progress_callback = kwargs.pop('_progress_callback', None)
        progress_log = []

        def _on_progress(info):
            """Collect a progress event and forward it to the caller's callback."""
            progress_log.append(info)
            if progress_callback:
                progress_callback(info)

        # Datasets: resolve dataset_ids / paths / builtin names through the
        # shared loader, then hand benchmark() in-memory specs.
        data_input = kwargs['data']
        data_names = [data_input] if isinstance(data_input, str) else list(data_input)
        datasets = []
        for name in data_names:
            ds = _load_data(name)
            datasets.append({"name": str(name), "X": ds.X, "y": ds.y})

        # Models: translate the tool-level vocabulary (bare names, flat
        # param dicts) into strict {"name", "params"} specs.
        models = []
        for item in kwargs['algorithms']:
            if isinstance(item, str):
                models.append({"name": item})
            elif isinstance(item, dict):
                entry = dict(item)
                name = entry.pop('name', None)
                params = entry.pop('params', entry)
                spec = {"name": name}
                if params:
                    spec["params"] = params
                models.append(spec)
            else:
                return {
                    'status': 'error',
                    'error': (
                        "algorithms entries must be names or "
                        '{"name": ..., "params": {...}} dicts.'
                    ),
                }

        evaluation = {"cv": kwargs.get('cv', 10)}
        if kwargs.get('metrics'):
            bad_metrics = _unknown_metrics(kwargs['metrics'])
            if bad_metrics:
                return bad_metrics
            evaluation["metrics"] = list(kwargs['metrics'])

        bench = tuiml.Benchmark(
            models=models,
            datasets=datasets,
            evaluation=evaluation,
            random_seed=kwargs.get('random_seed'),
        ).run(progress_callback=_on_progress)

        # Nested results dict, kept in the shape earlier consumers expect:
        # {dataset: {model: {metric: {mean, std, scores}}}}
        results_data = {}
        grouped = bench.scores_.groupby(['dataset', 'model', 'metric'], sort=False)
        for (dataset, model, metric), group in grouped:
            values = [float(v) for v in group['value']]
            results_data.setdefault(dataset, {}).setdefault(model, {})[metric] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'scores': values,
            }

        result = {
            'status': 'success',
            'summary': bench.summary(),
            'table_markdown': bench.to_markdown(),
            'results': results_data,
            'algorithms': [m['name'] for m in models],
            'datasets': data_names,
            'cv_folds': kwargs.get('cv', 10),
            'random_seed': bench.random_seed,
        }
        if progress_log:
            result['progress_log'] = progress_log

        # Best-effort research-log hook: append a run entry to the matching
        # user algorithm's runs.jsonl. Silently no-ops when no algorithm in
        # this experiment is a user algorithm, or when the feature flag is off.
        try:
            from tuiml.agent import user_algorithms as _user_algorithms
            appended = _user_algorithms.record_experiment_runs(result)
            if appended:
                result["research_log_updates"] = appended
        except Exception:
            pass

        return result
    except Exception as e:
        return {'status': 'error', 'error': str(e)}


SPEC = ToolSpec(
    name='tuiml_benchmark',
    description="Compare several algorithms on one or more datasets with k-fold CV. "
        "Supports classification, regression, clustering and forecasting (forecasters get "
        "expanding-window time-ordered folds on the target column; namespaced keys such as "
        "foundation.ChronosForecaster work). Returns per-dataset mean/std "
        "scores you can pass to tuiml_test_statistics and tuiml_plot.",
    input_schema={
            "type": "object",
            "properties": {
                "algorithms": {"type": "array", "items": {"type": "string"}, "description": "Exact class names, e.g. [\"RandomForestClassifier\", \"SVC\", \"NaiveBayesClassifier\"]."},
                "data": {"description": "One dataset name/path, or a list of them, e.g. [\"iris\", \"glass\", \"breast-cancer\"]."},
                "target": {"type": "string", "description": "Target column for supervised files; omit for built-in datasets or clustering."},
                "cv": {"type": "integer", "default": 10, "description": "CV folds (ignored for clustering)."},
                "metrics": {"type": "array", "items": {"type": "string"}, "description": "Exact metric function names, auto-selected when omitted. Classification: accuracy_score, f1_score, precision_score, recall_score, roc_auc_score. Regression: r2_score, root_mean_squared_error, mean_absolute_error, mean_squared_error. Clustering: silhouette_score, calinski_harabasz_score, davies_bouldin_score. Forecasting: mean_absolute_error, root_mean_squared_error, mean_squared_error."},
                "random_seed": {"type": "integer"}
            },
            "required": ["algorithms", "data"]
        },
    output_schema={
            "type": "object",
            "properties": {
                "status": {"type": "string", "enum": ["success", "error"]},
                "results": {"type": "object", "description": "By dataset and model"}
            },
            "required": ["status"]
        },
    execute=execute_benchmark,
    group='workflow',
    read_only=False, destructive=False,
    idempotent=False, open_world=True,
    seeded=True,
)
