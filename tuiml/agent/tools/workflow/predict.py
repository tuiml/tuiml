"""Prediction and forecasting."""

from typing import Any, Dict

from .._spec import ToolSpec
from .._shared import _load_model_from_disk, _load_data, _get_model_tags


def execute_predict(**kwargs) -> Dict[str, Any]:
    """Execute prediction with support for timeseries and anomaly models.

    Backs the ``tuiml_predict`` tool. Timeseries models forecast ``steps``
    ahead; anomaly detectors additionally report anomaly counts and score
    statistics; all other models run standard ``predict``.

    Parameters
    ----------
    model_id : str
        Identifier of a trained model from ``tuiml_train`` (arrives via
        ``**kwargs``, like all parameters below). One of ``model_id`` /
        ``model_path`` is required.
    model_path : str, default=None
        Explicit path to a serialized model file.
    data : str
        Dataset to predict on (dataset_id, file path, or built-in name).
        Not needed for timeseries forecasting.
    stage : str, default=None
        Optional stage: ``'forecast'`` (timeseries) or ``'predict_proba'``
        (class probabilities). When None, dispatch is based on model tags.
    stage_kwargs : dict, default=None
        Extra stage arguments (e.g. ``steps`` for forecasting).
    steps : int, default=10
        Number of future steps to forecast for timeseries models.
    output_path : str, default=None
        When given, predictions are also written to this file.

    Returns
    -------
    result : dict
        On success: ``status`` (``'success'``), ``num_predictions``,
        ``predictions_preview`` (first 10), and optionally ``model_type``,
        ``steps``, ``output_path``; anomaly models add ``n_anomalies``,
        ``n_normal``, ``anomaly_ratio``, ``anomaly_scores_preview`` and
        ``score_stats``. On failure: ``status`` (``'error'``), ``error``,
        ``error_type`` and optionally ``suggestion``.
    """
    import numpy as np

    try:
        model_id = kwargs.get('model_id')
        model_path = kwargs.get('model_path')
        stage = kwargs.pop('stage', None)
        stage_kwargs = kwargs.pop('stage_kwargs', None) or {}

        model = _load_model_from_disk(model_id, model_path)
        if model is None:
            return {
                'status': 'error',
                'error': 'Model not found. Provide model_id (from tuiml_train) or a valid model_path.',
                'error_type': 'ValueError',
                'suggestion': 'Train a model first with tuiml_train which returns a model_id and model_path'
            }

        tags = _get_model_tags(model)

        # 1. Handle Stage: forecast
        if stage == 'forecast':
            steps = kwargs.get('steps') or stage_kwargs.get('steps') or 10
            try:
                predictions = model.predict(steps)
            except Exception as e:
                return {
                    'status': 'error',
                    'error': f"Forecasting failed: {e}",
                    'error_type': type(e).__name__
                }
            predictions = np.asarray(predictions)
            result = {
                'status': 'success',
                'model_type': 'timeseries',
                'num_predictions': len(predictions),
                'predictions_preview': predictions[:10].tolist(),
                'steps': steps
            }
            if kwargs.get('output_path'):
                np.savetxt(kwargs['output_path'], predictions)
                result['output_path'] = kwargs['output_path']
            return result

        # 2. Handle Stage: predict_proba
        elif stage == 'predict_proba':
            if not hasattr(model, 'predict_proba'):
                return {
                    'status': 'error',
                    'error': f"Model '{model.__class__.__name__}' does not support class probability prediction (predict_proba)"
                }

            data_arg = kwargs.get('data')
            if not data_arg:
                return {
                    'status': 'error',
                    'error': "Missing required parameter 'data' for stage 'predict_proba'"
                }
            dataset = _load_data(data_arg)
            try:
                probabilities = model.predict_proba(dataset.X)
            except Exception as e:
                return {
                    'status': 'error',
                    'error': f"Probability prediction failed: {e}",
                    'error_type': type(e).__name__
                }
            probabilities = np.asarray(probabilities)
            result = {
                'status': 'success',
                'num_predictions': len(probabilities),
                'predictions_preview': probabilities[:10].tolist()
            }
            if kwargs.get('output_path'):
                np.savetxt(kwargs['output_path'], probabilities)
                result['output_path'] = kwargs['output_path']
            return result

        # 3. Handle Stage: predict (or default fallback)
        # Timeseries models
        if 'timeseries' in tags and stage is None:
            steps = kwargs.get('steps', 10)
            predictions = model.predict(steps)
            predictions = np.asarray(predictions)
            result = {
                'status': 'success',
                'model_type': 'timeseries',
                'num_predictions': len(predictions),
                'predictions_preview': predictions[:10].tolist(),
                'steps': steps
            }
            if kwargs.get('output_path'):
                np.savetxt(kwargs['output_path'], predictions)
                result['output_path'] = kwargs['output_path']
            return result

        # Anomaly detection models
        if 'anomaly-detection' in tags:
            dataset = _load_data(kwargs['data'])
            predictions = model.predict(dataset.X)
            predictions = np.asarray(predictions)
            result = {
                'status': 'success',
                'model_type': 'anomaly',
                'num_predictions': len(predictions),
                'predictions_preview': predictions[:10].tolist(),
                'n_anomalies': int(np.sum(predictions == -1)),
                'n_normal': int(np.sum(predictions == 1)),
                'anomaly_ratio': float(np.mean(predictions == -1))
            }
            # Get anomaly scores if available
            if hasattr(model, 'decision_function'):
                try:
                    scores = model.decision_function(dataset.X)
                    scores = np.asarray(scores)
                    result['anomaly_scores_preview'] = scores[:10].tolist()
                    result['score_stats'] = {
                        'mean': float(np.mean(scores)),
                        'std': float(np.std(scores)),
                        'min': float(np.min(scores)),
                        'max': float(np.max(scores))
                    }
                except Exception:
                    pass
            if kwargs.get('output_path'):
                np.savetxt(kwargs['output_path'], predictions)
                result['output_path'] = kwargs['output_path']
            return result

        # Standard supervised/clustering prediction
        dataset = _load_data(kwargs['data'])
        predictions = model.predict(dataset.X)

        result = {
            'status': 'success',
            'num_predictions': len(predictions),
            'predictions_preview': predictions[:10].tolist()
        }

        if kwargs.get('output_path'):
            np.savetxt(kwargs['output_path'], predictions)
            result['output_path'] = kwargs['output_path']

        return result
    except FileNotFoundError as e:
        return {
            'status': 'error',
            'error': f"File not found: {str(e)}",
            'error_type': 'FileNotFoundError',
            'suggestion': 'Check the file path or use model_id from tuiml_train instead'
        }
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e),
            'error_type': type(e).__name__
        }


SPEC = ToolSpec(
    name='tuiml_predict',
    description="Make predictions using a trained model on new data. Supports supervised models, "
        "timeseries models (use 'steps' parameter), and anomaly detection models.",
    input_schema={
            "type": "object",
            "properties": {
                "model_id": {
                    "type": "string",
                    "description": "Model ID returned by tuiml_train (preferred)"
                },
                "model_path": {
                    "type": "string",
                    "description": "Path to saved model file (.pkl) (alternative to model_id)"
                },
                "data": {
                    "type": "string",
                    "description": "Path to data file for prediction"
                },
                "steps": {
                    "type": "integer",
                    "description": "Number of forecast steps (timeseries models only)"
                },
                "output_path": {
                    "type": "string",
                    "description": "Path to save predictions (optional)"
                },
                "stage": {
                    "type": "string",
                    "description": "Atomic prediction stage: 'predict', 'predict_proba', 'forecast'"
                },
                "stage_kwargs": {
                    "type": "object",
                    "description": "Arbitrary stage-specific keyword arguments"
                }
            },
            "required": []
        },
    output_schema={
            "type": "object",
            "properties": {
                "status": {"type": "string", "enum": ["success", "error"]},
                "num_predictions": {"type": "integer"},
                "predictions_preview": {
                    "type": "array",
                    "description": "First 10 predictions"
                },
                "output_path": {"type": "string"},
                "error": {"type": "string"}
            },
            "required": ["status"]
        },
    execute=execute_predict,
    group='workflow',
    read_only=True, destructive=False,
    idempotent=True, open_world=False,
)
