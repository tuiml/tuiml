"""Hyperparameter optimization."""

import uuid
from typing import Any, Dict

from .._spec import ToolSpec
from .._state import _MODEL_INDEX
from .._shared import _save_model_to_disk, _load_data


def execute_tune(**kwargs) -> Dict[str, Any]:
    """Hyperparameter optimization for any registered algorithm.

    Backs the ``tuiml_tune`` tool. Runs grid, random, or Bayesian search
    over a parameter space, then saves and indexes the best estimator.

    Parameters
    ----------
    algorithm : str
        Registered algorithm class name to tune (arrives via ``**kwargs``,
        like all parameters below).
    data : str
        Dataset to tune on: dataset_id, file path, or built-in name.
    method : str
        Search strategy: ``'grid'``, ``'random'``, or ``'bayesian'``.
    param_grid : dict
        Parameter grid / distributions / space, depending on ``method``.
    cv : int, default=5
        Number of cross-validation folds.
    scoring : str, default='accuracy'
        Scoring metric name.
    n_iter : int, default=10
        Number of sampled candidates (random search only).
    n_iterations : int, default=50
        Number of optimization iterations (Bayesian search only).
    random_seed : int, default=None
        Random seed for reproducible search.
    _progress_callback : callable, default=None
        Internal per-iteration progress hook; stripped from recorded args.

    Returns
    -------
    result : dict
        On success: ``status`` (``'success'``), ``method``,
        ``best_params``, ``best_score``, ``cv_results`` (summary with
        ``n_candidates``, ``best_rank`` and ``top_5``), ``model_id``,
        ``model_path``, and optionally ``progress_log``. On failure:
        ``status`` (``'error'``), ``error`` and optionally
        ``suggestion`` / ``error_type``.
    """
    import numpy as np

    try:
        algorithm_name = kwargs['algorithm']
        progress_callback = kwargs.pop('_progress_callback', None)

        from tuiml.registry import registry
        import tuiml.algorithms  # noqa: F401 - trigger registration

        algo_cls = registry.get(algorithm_name)
        if algo_cls is None:
            return {
                'status': 'error',
                'error': f"Algorithm '{algorithm_name}' not found.",
                'suggestion': "Use tuiml_list with category='algorithm' to see available algorithms."
            }

        dataset = _load_data(kwargs['data'])
        X = np.asarray(dataset.X, dtype=float)
        y = np.asarray(dataset.y)

        method = kwargs['method']
        param_grid = kwargs['param_grid']
        cv = kwargs.get('cv', 5)
        scoring = kwargs.get('scoring', 'accuracy')
        random_seed = kwargs.get('random_seed')

        # Collect progress messages
        progress_log = []

        def _on_progress(info):
            """Collect a progress event and forward it to the caller's callback."""
            progress_log.append(info)
            if progress_callback:
                progress_callback(info)

        estimator = algo_cls()

        if method == 'grid':
            from tuiml.evaluation.tuning import GridSearchCV
            tuner = GridSearchCV(
                estimator=estimator,
                param_grid=param_grid,
                cv=cv,
                scoring=scoring,
                random_seed=random_seed,
                progress_callback=_on_progress,
            )
        elif method == 'random':
            from tuiml.evaluation.tuning import RandomSearchCV
            n_iter = kwargs.get('n_iter', 10)
            tuner = RandomSearchCV(
                estimator=estimator,
                param_distributions=param_grid,
                n_iter=n_iter,
                cv=cv,
                scoring=scoring,
                random_seed=random_seed,
                progress_callback=_on_progress,
            )
        elif method == 'bayesian':
            from tuiml.evaluation.tuning import BayesianSearchCV
            n_iterations = kwargs.get('n_iterations', 50)
            tuner = BayesianSearchCV(
                estimator=estimator,
                param_space=param_grid,
                n_iterations=n_iterations,
                cv=cv,
                scoring=scoring,
                random_seed=random_seed,
                progress_callback=_on_progress,
            )
        else:
            return {
                'status': 'error',
                'error': f"Unknown tuning method: '{method}'",
                'suggestion': "Available methods: 'grid', 'random', 'bayesian'"
            }

        tuner.fit(X, y)

        # Save best estimator
        model_id = uuid.uuid4().hex[:12]
        model_path = _save_model_to_disk(tuner.best_estimator_, model_id)
        _MODEL_INDEX[model_id] = model_path

        # Summarize cv_results
        cv_results_summary = {}
        if hasattr(tuner, 'cv_results_') and tuner.cv_results_:
            cv_res = tuner.cv_results_
            if 'params' in cv_res and 'mean_test_score' in cv_res:
                cv_results_summary['n_candidates'] = len(cv_res['params'])
                cv_results_summary['best_rank'] = int(cv_res.get('rank_test_score', [1])[0]) if 'rank_test_score' in cv_res else 1
                # Top 5 parameter sets
                top_indices = np.argsort(cv_res['mean_test_score'])[::-1][:5]
                cv_results_summary['top_5'] = [
                    {
                        'params': cv_res['params'][i],
                        'mean_score': float(cv_res['mean_test_score'][i]),
                        'std_score': float(cv_res['std_test_score'][i]) if 'std_test_score' in cv_res else 0.0,
                    }
                    for i in top_indices
                ]

        result = {
            'status': 'success',
            'method': method,
            'best_params': tuner.best_params_,
            'best_score': float(tuner.best_score_),
            'cv_results': cv_results_summary,
            'model_id': model_id,
            'model_path': model_path,
        }
        if progress_log:
            result['progress_log'] = [
                {
                    'iteration': p.get('iteration'),
                    'total': p.get('total'),
                    'mean_score': round(p.get('mean_score', 0), 4),
                    'best_score': round(p.get('best_score', 0), 4),
                    'params': p.get('params'),
                }
                for p in progress_log
            ]
        return result
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e),
            'error_type': type(e).__name__
        }


SPEC = ToolSpec(
    name='tuiml_tune',
    description="Hyperparameter optimization for any algorithm. Supports grid search, "
        "random search, and Bayesian optimization. Returns best parameters, "
        "best score, and a trained model with optimal settings.",
    input_schema={
            "type": "object",
            "properties": {
                "algorithm": {
                    "type": "string",
                    "description": "Algorithm class name (e.g., 'RandomForestClassifier', 'SVM')"
                },
                "data": {
                    "type": "string",
                    "description": "Data file path or built-in dataset name"
                },
                "target": {
                    "type": "string",
                    "description": "Target column name"
                },
                "method": {
                    "type": "string",
                    "enum": ["grid", "random", "bayesian"],
                    "description": "Tuning method: 'grid' (exhaustive), 'random' (sampled), 'bayesian' (GP-based)"
                },
                "param_grid": {
                    "type": "object",
                    "description": (
                        "Parameter search space. For grid: {'param': [val1, val2]}. "
                        "For random/bayesian: {'param': [low, high, 'int']} or {'param': [val1, val2]}."
                    )
                },
                "cv": {
                    "type": "integer",
                    "default": 5,
                    "description": "Number of cross-validation folds"
                },
                "scoring": {
                    "type": "string",
                    "description": "Scoring metric (e.g., 'accuracy', 'r2', 'neg_mse')"
                },
                "n_iter": {
                    "type": "integer",
                    "default": 10,
                    "description": "Number of iterations for random search"
                },
                "n_iterations": {
                    "type": "integer",
                    "default": 50,
                    "description": "Number of iterations for Bayesian search"
                },
                "random_seed": {
                    "type": "integer",
                    "description": "Random seed for reproducibility"
                }
            },
            "required": ["algorithm", "data", "target", "method", "param_grid"]
        },
    output_schema={
            "type": "object",
            "properties": {
                "status": {"type": "string", "enum": ["success", "error"]},
                "method": {"type": "string"},
                "best_params": {"type": "object"},
                "best_score": {"type": "number"},
                "cv_results": {"type": "object"},
                "model_id": {"type": "string"},
                "model_path": {"type": "string"},
                "random_seed": {"type": "integer"},
                "error": {"type": "string"}
            },
            "required": ["status"]
        },
    execute=execute_tune,
    group='workflow',
    read_only=False, destructive=False,
    idempotent=False, open_world=True,
    seeded=True,
)
