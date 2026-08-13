"""Model training."""

import json
import uuid
from typing import Any, Dict

from .._spec import ToolSpec
from .._state import _MODEL_INDEX
from .._shared import _save_model_to_disk, _load_model_from_disk, _load_data


def execute_train(**kwargs) -> Dict[str, Any]:
    """Execute the training workflow behind the ``tuiml_train`` tool.

    Supports staged execution (``init`` / ``fit`` / ``partial_fit`` /
    ``cross_validate``) as well as the default full pipeline via
    ``tuiml.train()``. The trained model is saved to disk and indexed so
    other tools can load it by ``model_id``.

    Parameters
    ----------
    algorithm : str
        Registered algorithm class name to train (arrives via ``**kwargs``,
        like all parameters below). Required unless a ``model_id`` /
        ``model_path`` is given for a stage that loads an existing model.
    algorithm_params : dict, default=None
        Constructor parameters for the algorithm.
    data : str
        Dataset to train on: uploaded dataset_id, file path, or built-in
        dataset name. Required for the default path and for the ``fit`` /
        ``partial_fit`` stages.
    stage : str, default=None
        Optional atomic stage: ``'init'`` (instantiate and save an unfitted
        model), ``'fit'``, ``'partial_fit'`` (incremental training), or
        ``'cross_validate'``. When None the full train pipeline runs.
    stage_kwargs : dict, default=None
        Extra keyword arguments for the selected stage (e.g. ``classes``
        for ``partial_fit``, ``cv`` for ``cross_validate``).
    model_id : str, default=None
        Existing model to continue training (``fit`` / ``partial_fit``).
    model_path : str, default=None
        Explicit path to an existing serialized model.
    save_path : str, default=None
        Where to save the trained model; defaults to ``~/.tuiml/models/``.
    preset : str, default=None
        Named preprocessing preset used when no explicit steps are given.
    preprocessing : list, default=None
        Preprocessing steps, each a name or ``{"name", **params}`` dict.
    feature_selection : dict, default=None
        Feature-selection step appended to the pipeline.
    cv : int, default=None
        Number of cross-validation folds.
    test_size : float, default=None
        Holdout fraction for evaluation.
    stratify : bool, default=None
        Whether to stratify the evaluation split.
    metrics : list of str, default=None
        Metrics to compute during evaluation.
    random_seed : int, default=None
        Random seed; falls back to the global seed, then 42.

    Returns
    -------
    result : dict
        On success: ``status`` (``'success'``), ``model_id``, ``model_path``
        and ``model_class``; the full-pipeline path also includes
        ``metrics``, ``cv_results`` and ``metadata``. On failure:
        ``status`` (``'error'``), ``error``, and optionally ``error_type``,
        ``suggestion``, ``recovery_tool`` and ``recovery_params``.
    """
    import tuiml
    import numpy as np

    stage = kwargs.pop('stage', None)
    stage_kwargs = kwargs.pop('stage_kwargs', None) or {}
    model_id = kwargs.pop('model_id', None)
    model_path = kwargs.pop('model_path', None)

    # 1. Handle Stage: init
    if stage == 'init':
        algorithm = kwargs.get('algorithm')
        if not algorithm:
            return {
                'status': 'error',
                'error': "Missing required parameter 'algorithm' for stage 'init'"
            }
        algo_params = kwargs.pop('algorithm_params', {}) or {}
        algo_params.update(stage_kwargs)

        from tuiml.registry import registry
        import tuiml.algorithms  # noqa
        try:
            model_cls = registry.get(algorithm)
        except KeyError:
            return {
                'status': 'error',
                'error': f"Algorithm not found: {algorithm}",
                'error_type': 'KeyError',
                'suggestion': "Use 'tuiml_list' with category='algorithm' to see available algorithms"
            }

        random_seed = kwargs.get('random_seed')
        if random_seed is None:
            from tuiml.utils.seed import get_global_seed
            random_seed = get_global_seed() or 42

        from tuiml.workflow import _inject_seed
        algo_params = _inject_seed(model_cls, algo_params, random_seed)

        try:
            model = model_cls(**algo_params)
        except Exception as e:
            return {
                'status': 'error',
                'error': f"Failed to instantiate {algorithm}: {e}",
                'error_type': type(e).__name__
            }

        save_path = kwargs.pop('save_path', None)
        out_model_id = uuid.uuid4().hex[:12]
        out_model_path = _save_model_to_disk(model, out_model_id, save_path)
        _MODEL_INDEX[out_model_id] = out_model_path

        return {
            'status': 'success',
            'model_id': out_model_id,
            'model_path': out_model_path,
            'model_class': model.__class__.__name__
        }

    # 2. Handle Stage: fit
    elif stage == 'fit':
        model = None
        if model_id or model_path:
            model = _load_model_from_disk(model_id, model_path)
            if model is None:
                return {
                    'status': 'error',
                    'error': f"Could not load model from model_id='{model_id}' or model_path='{model_path}'"
                }
        else:
            algorithm = kwargs.get('algorithm')
            if not algorithm:
                return {
                    'status': 'error',
                    'error': "Provide either 'algorithm' to train a new model, or 'model_id'/'model_path' to load an existing model."
                }
            from tuiml.registry import registry
            import tuiml.algorithms  # noqa
            try:
                model_cls = registry.get(algorithm)
            except KeyError:
                return {
                    'status': 'error',
                    'error': f"Algorithm not found: {algorithm}",
                    'error_type': 'KeyError'
                }
            algo_params = kwargs.pop('algorithm_params', {}) or {}
            algo_params.update(stage_kwargs)
            random_seed = kwargs.get('random_seed')
            if random_seed is None:
                from tuiml.utils.seed import get_global_seed
                random_seed = get_global_seed() or 42
            from tuiml.workflow import _inject_seed
            algo_params = _inject_seed(model_cls, algo_params, random_seed)
            try:
                model = model_cls(**algo_params)
            except Exception as e:
                return {
                    'status': 'error',
                    'error': f"Failed to instantiate {algorithm}: {e}",
                    'error_type': type(e).__name__
                }

        data_arg = kwargs.get('data')
        if not data_arg:
            return {
                'status': 'error',
                'error': "Missing required parameter 'data' for stage 'fit'"
            }
        try:
            dataset = _load_data(data_arg)
        except Exception as e:
            return {
                'status': 'error',
                'error': f"Could not resolve data='{data_arg}': {e}",
                'error_type': type(e).__name__
            }

        X, y = dataset.X, dataset.y
        import inspect
        fit_sig = inspect.signature(model.fit)
        fit_params = list(fit_sig.parameters.keys())
        expects_y = 'y' in fit_params

        try:
            if expects_y and y is not None:
                model.fit(X, y)
            else:
                model.fit(X)
        except Exception as e:
            return {
                'status': 'error',
                'error': f"Fit failed: {e}",
                'error_type': type(e).__name__
            }

        save_path = kwargs.pop('save_path', None)
        out_model_id = model_id or uuid.uuid4().hex[:12]
        out_model_path = _save_model_to_disk(model, out_model_id, save_path)
        _MODEL_INDEX[out_model_id] = out_model_path

        return {
            'status': 'success',
            'model_id': out_model_id,
            'model_path': out_model_path,
            'model_class': model.__class__.__name__
        }

    # 3. Handle Stage: partial_fit
    elif stage == 'partial_fit':
        classes_arg = stage_kwargs.pop('classes', None)
        model = None
        if model_id or model_path:
            model = _load_model_from_disk(model_id, model_path)
            if model is None:
                return {
                    'status': 'error',
                    'error': f"Could not load model from model_id='{model_id}' or model_path='{model_path}'"
                }
        else:
            algorithm = kwargs.get('algorithm')
            if not algorithm:
                return {
                    'status': 'error',
                    'error': "Provide either 'algorithm' to train a new model, or 'model_id'/'model_path' to load an existing model."
                }
            from tuiml.registry import registry
            import tuiml.algorithms  # noqa
            try:
                model_cls = registry.get(algorithm)
            except KeyError:
                return {
                    'status': 'error',
                    'error': f"Algorithm not found: {algorithm}",
                    'error_type': 'KeyError'
                }
            algo_params = kwargs.pop('algorithm_params', {}) or {}
            algo_params.update(stage_kwargs)
            random_seed = kwargs.get('random_seed')
            if random_seed is None:
                from tuiml.utils.seed import get_global_seed
                random_seed = get_global_seed() or 42
            from tuiml.workflow import _inject_seed
            algo_params = _inject_seed(model_cls, algo_params, random_seed)
            try:
                model = model_cls(**algo_params)
            except Exception as e:
                return {
                    'status': 'error',
                    'error': f"Failed to instantiate {algorithm}: {e}",
                    'error_type': type(e).__name__
                }

        if not hasattr(model, 'partial_fit'):
            return {
                'status': 'error',
                'error': f"Algorithm '{model.__class__.__name__}' does not support incremental training (partial_fit)"
            }

        data_arg = kwargs.get('data')
        if not data_arg:
            return {
                'status': 'error',
                'error': "Missing required parameter 'data' for stage 'partial_fit'"
            }
        try:
            dataset = _load_data(data_arg)
        except Exception as e:
            return {
                'status': 'error',
                'error': f"Could not resolve data='{data_arg}': {e}",
                'error_type': type(e).__name__
            }

        X, y = dataset.X, dataset.y

        # Parse classes if passed
        classes = classes_arg if classes_arg is not None else stage_kwargs.get('classes')
        if classes is not None:
            if isinstance(classes, str):
                try:
                    classes = json.loads(classes)
                except json.JSONDecodeError:
                    classes = [c.strip() for c in classes.split(',')]
            classes = np.asarray(classes)

        import inspect
        pf_sig = inspect.signature(model.partial_fit)
        pf_params = list(pf_sig.parameters.keys())
        expects_y = 'y' in pf_params
        expects_classes = 'classes' in pf_params

        pf_kwargs = {}
        if expects_classes and classes is not None:
            pf_kwargs['classes'] = classes

        try:
            if expects_y and y is not None:
                model.partial_fit(X, y, **pf_kwargs)
            else:
                model.partial_fit(X, **pf_kwargs)
        except Exception as e:
            return {
                'status': 'error',
                'error': f"partial_fit failed: {e}",
                'error_type': type(e).__name__
            }

        save_path = kwargs.pop('save_path', None)
        out_model_id = model_id or uuid.uuid4().hex[:12]
        out_model_path = _save_model_to_disk(model, out_model_id, save_path)
        _MODEL_INDEX[out_model_id] = out_model_path

        return {
            'status': 'success',
            'model_id': out_model_id,
            'model_path': out_model_path,
            'model_class': model.__class__.__name__
        }

    # 4. Handle Stage: cross_validate (or normal fallback)
    elif stage == 'cross_validate':
        cv_folds = kwargs.get('cv') or stage_kwargs.get('cv') or 5
        kwargs['cv'] = cv_folds

        if model_id or model_path:
            model = _load_model_from_disk(model_id, model_path)
            if model is None:
                return {
                    'status': 'error',
                    'error': f"Could not load model from model_id='{model_id}' or model_path='{model_path}'"
                }
            kwargs['algorithm'] = model.__class__.__name__
            if hasattr(model, 'get_params'):
                kwargs['algorithm_params'] = model.get_params()

    # Normal execution path (either default or cross_validate stage).
    # Translate the tool-level vocabulary (algorithm/algorithm_params/preset/
    # preprocessing/feature_selection + flat evaluation options) into
    # tuiml.train()'s spec convention: {"name", "params"} components, one
    # ordered "pipeline" list, and a grouped "evaluation" dict.
    algo_params = kwargs.pop('algorithm_params', {}) or {}
    save_path = kwargs.pop('save_path', None)
    algo_name = kwargs.pop('algorithm', None)
    if not algo_name:
        return {
            'status': 'error',
            'error': "Missing required parameter 'algorithm'"
        }
    kwargs['model'] = {'name': algo_name, 'params': algo_params}

    def _nest_step(step):
        """Convert a flat tool-level step ({"name", **params}) to spec form."""
        if isinstance(step, str):
            return {'name': step}
        step = dict(step)
        name = step.pop('name', None)
        params = step.pop('params', step)
        return {'name': name, 'params': params}

    preset = kwargs.pop('preset', None)
    steps = [_nest_step(s) for s in kwargs.pop('preprocessing', None) or []]
    fs = kwargs.pop('feature_selection', None)
    if fs:
        steps.append(_nest_step(fs))
    if steps:
        kwargs['pipeline'] = steps
    elif preset:
        kwargs['pipeline'] = preset

    evaluation = {
        key: kwargs.pop(key)
        for key in ('cv', 'test_size', 'stratify', 'metrics')
        if kwargs.get(key) is not None
    }
    if evaluation:
        kwargs['evaluation'] = evaluation

    # Pre-resolve data via the shared loader
    data_arg = kwargs.get('data')
    if isinstance(data_arg, str):
        try:
            kwargs['data'] = _load_data(data_arg)
        except Exception as e:
            return {
                'status': 'error',
                'error': f"Could not resolve data='{data_arg}': {e}",
                'error_type': type(e).__name__,
                'suggestion': (
                    'Use a built-in dataset name (e.g., "iris"), a dataset_id from '
                    'tuiml_upload_data, or an existing file path.'
                )
            }

    try:
        # train() returns the fitted Workflow, which is itself the model: it
        # carries the fitted transformations, so saving it keeps predictions
        # consistent with training.
        result = tuiml.train({k: v for k, v in kwargs.items() if v is not None})

        model_id = uuid.uuid4().hex[:12]
        model_path = _save_model_to_disk(result, model_id, save_path)
        _MODEL_INDEX[model_id] = model_path

        return {
            'status': 'success',
            'model_id': model_id,
            'model_path': model_path,
            'metrics': result.metrics_,
            'cv_results': result.cv_results_,
            'model_class': type(result.model_).__name__,
            'metadata': result.metadata_
        }
    except KeyError as e:
        return {
            'status': 'error',
            'error': f"Algorithm not found: {kwargs.get('algorithm')}",
            'error_type': 'KeyError',
            'suggestion': "Use 'tuiml_list' with category='algorithm' to see available algorithms",
            'recovery_tool': 'tuiml_list',
            'recovery_params': {'category': 'algorithm'}
        }
    except ValueError as e:
        return {
            'status': 'error',
            'error': str(e),
            'error_type': 'ValueError',
            'suggestion': "Check parameter types and values. Use 'tuiml_describe' to see the algorithm's parameter schema",
            'recovery_tool': 'tuiml_describe',
            'recovery_params': {'name': kwargs.get('algorithm')}
        }
    except FileNotFoundError as e:
        return {
            'status': 'error',
            'error': f"Data file not found: {kwargs.get('data')}",
            'error_type': 'FileNotFoundError',
            'suggestion': 'Check the file path or use a built-in dataset name (e.g., "iris", "wine")'
        }
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e),
            'error_type': type(e).__name__
        }


SPEC = ToolSpec(
    name='tuiml_train',
    description="Train a machine learning model with evaluation. Two evaluation modes:\n"
        "1. Holdout (default): splits data into train/test sets using test_size. "
        "Returns metrics on the test set and predictions.\n"
        "2. Cross-validation: set cv=5 or cv=10 for k-fold CV. "
        "Returns mean/std metrics across folds.\n"
        "If neither cv nor test_size is provided, defaults to holdout with test_size=0.2.\n"
        "Supports classifiers, regressors, and clusterers.",
    input_schema={
            "type": "object",
            "properties": {
                "algorithm": {
                    "type": "string",
                    "description": (
                        "Algorithm class name. Examples:\n"
                        "- Classifiers: 'RandomForestClassifier', 'SVM', 'NaiveBayesClassifier', 'DecisionTreeClassifier'\n"
                        "- Regressors: 'LinearRegression', 'RandomForestRegressor'\n"
                        "- Clusterers: 'KMeansClusterer', 'GaussianMixtureClusterer', 'DBSCANClusterer'\n"
                        "- Optional sklearn backends (needs tuiml[sklearn]): 'sklearn.RandomForestClassifier', 'sklearn.SVC', 'sklearn.Lasso'"
                    )
                },
                "data": {
                    "type": "string",
                    "description": "Data file path or built-in dataset name (e.g., 'iris', 'wine')"
                },
                "target": {
                    "type": "string",
                    "description": "Target column (required for supervised, optional for clustering)"
                },
                "features": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Optional: restrict the feature matrix to these named columns. "
                        "When omitted, every non-target column is used as a feature."
                    )
                },
                "preprocessing": {
                    "type": "array",
                    "items": {
                        "oneOf": [
                            {"type": "string"},
                            {
                                "type": "object",
                                "properties": {"name": {"type": "string"}},
                                "required": ["name"],
                                "additionalProperties": True
                            }
                        ]
                    },
                    "description": (
                        "Preprocessing steps as names or objects with params.\n"
                        "Examples: ['SimpleImputer', 'StandardScaler'] or "
                        "[{'name': 'SimpleImputer', 'strategy': 'median'}, 'MinMaxScaler']"
                    )
                },
                "feature_selection": {
                    "oneOf": [
                        {"type": "string"},
                        {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                            },
                            "required": ["name"],
                            "additionalProperties": True
                        }
                    ],
                    "description": (
                        "Feature selection method. String name or object with params.\n"
                        "Examples: 'SelectKBestSelector' or {'name': 'SelectKBestSelector', 'k': 10}"
                    )
                },
                "cv": {
                    "type": "integer",
                    "description": (
                        "Number of cross-validation folds (e.g. 5 or 10). "
                        "OPTIONAL: if omitted, uses holdout train/test split instead. "
                        "Only used for supervised learning (ignored for clustering)."
                    )
                },
                "test_size": {
                    "type": "number",
                    "default": 0.2,
                    "description": (
                        "Proportion of data for the test set (0.0-1.0). "
                        "Used in holdout mode (when cv is NOT set). Default 0.2 (80/20 split). "
                        "Ignored when cv is set."
                    )
                },
                "metrics": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Metrics to compute. Use exact function names. Must match algorithm type:\n"
                        "- Classification: ['accuracy_score', 'f1_score', 'precision_score', 'recall_score', 'roc_auc_score', 'balanced_accuracy_score', 'log_loss', 'matthews_corrcoef']\n"
                        "- Regression: ['r2_score', 'root_mean_squared_error', 'mean_absolute_error', 'mean_squared_error']\n"
                        "- Clustering: ['silhouette_score', 'calinski_harabasz_score', 'davies_bouldin_score']"
                    )
                },
                "preset": {
                    "type": "string",
                    "enum": ["minimal", "fast", "standard", "full", "imbalanced"],
                    "description": "Preprocessing preset"
                },
                "algorithm_params": {
                    "type": "object",
                    "description": "Algorithm hyperparameters (e.g., {'n_clusters': 3})"
                },
                "save_path": {
                    "type": "string",
                    "description": "Custom path to save the model file (optional). If omitted, saved to temp directory."
                },
                "random_seed": {
                    "type": "integer",
                    "description": "Random seed for reproducibility"
                },
                "stage": {
                    "type": "string",
                    "description": "Atomic training stage: 'init', 'fit', 'partial_fit', 'cross_validate'"
                },
                "stage_kwargs": {
                    "type": "object",
                    "description": "Arbitrary stage-specific keyword arguments (e.g. classes)"
                },
                "model_id": {
                    "type": "string",
                    "description": "Unique identifier of a previously initialized/saved model"
                },
                "model_path": {
                    "type": "string",
                    "description": "File path of a previously initialized/saved model"
                }
            },
            "required": []
        },
    output_schema={
            "type": "object",
            "properties": {
                "status": {"type": "string", "enum": ["success", "error"]},
                "model_id": {
                    "type": "string",
                    "description": "Model ID - use with tuiml_predict and tuiml_evaluate"
                },
                "model_path": {
                    "type": "string",
                    "description": "File path where the model is saved on disk"
                },
                "metrics": {
                    "type": "object",
                    "description": "Performance metrics (accuracy, f1, etc.)"
                },
                "cv_results": {
                    "type": "object",
                    "description": "Cross-validation fold results"
                },
                "model_class": {
                    "type": "string",
                    "description": "Name of the trained model class"
                },
                "metadata": {"type": "object"},
                "random_seed": {"type": "integer"},
                "error": {"type": "string"}
            },
            "required": ["status"]
        },
    execute=execute_train,
    group='workflow',
    read_only=False, destructive=False,
    idempotent=False, open_world=True,
    seeded=True,
)
