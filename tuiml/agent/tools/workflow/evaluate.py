"""Model evaluation and reports."""

from typing import Any, Dict

from .._spec import ToolSpec
from .._shared import _load_model_from_disk, _load_data, _get_model_tags


def execute_evaluate(**kwargs) -> Dict[str, Any]:
    """Execute evaluation with support for timeseries and anomaly models.

    Backs the ``tuiml_evaluate`` tool. Detects the model family
    (classifier, regressor, clusterer, timeseries, anomaly) and computes
    the appropriate metrics; the ``report`` stage additionally builds a
    formatted text report.

    Parameters
    ----------
    model_id : str
        Identifier of a trained model from ``tuiml_train`` (arrives via
        ``**kwargs``, like all parameters below). One of ``model_id`` /
        ``model_path`` is required.
    model_path : str, default=None
        Explicit path to a serialized model file.
    data : str
        Dataset to evaluate on (dataset_id, file path, or built-in name).
    stage : str, default=None
        Optional stage: ``'report'`` for a human-readable evaluation
        report, or ``'metrics'`` / None for a plain metrics dict.
    stage_kwargs : dict, default=None
        Extra stage arguments (currently unused by the stages).
    metrics : str or list, default='auto'
        Metrics passed to ``model.evaluate`` on the standard path.

    Returns
    -------
    result : dict
        On success: ``status`` (``'success'``) and ``metrics``; the
        ``report`` stage adds ``report`` (formatted text) and
        ``model_type``; timeseries evaluation adds ``train_size``,
        ``test_size`` and ``forecast_preview``. On failure: ``status``
        (``'error'``), ``error``, ``error_type`` and optionally
        ``suggestion``.
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
        dataset = _load_data(kwargs['data'])

        # Check model type
        is_timeseries = 'timeseries' in tags
        is_anomaly = 'anomaly-detection' in tags
        is_classifier = False
        is_regressor = False
        is_clustering = False

        if not is_timeseries and not is_anomaly:
            try:
                from tuiml.registry import registry
                algo_info = registry.get_info(model.__class__.__name__)
                algo_type = algo_info.get('type')
                if algo_type == 'classifier':
                    is_classifier = True
                elif algo_type == 'regressor':
                    is_regressor = True
                elif algo_type in ('clusterer', 'clustering'):
                    is_clustering = True
            except Exception:
                if hasattr(model, 'predict_proba'):
                    is_classifier = True
                elif hasattr(model, 'labels_'):
                    is_clustering = True
                else:
                    is_regressor = True

        # ---- Handle Stage: report ----
        if stage == 'report':
            # 1. Timeseries Report
            if is_timeseries:
                from tuiml.evaluation.metrics import mean_absolute_error, mean_squared_error, r2_score
                y = np.asarray(dataset.y) if dataset.y is not None else np.asarray(dataset.X).ravel()
                split_idx = int(len(y) * 0.8)
                y_train, y_test = y[:split_idx], y[split_idx:]
                model.fit(y_train)
                forecast = np.asarray(model.predict(len(y_test)))
                mae = mean_absolute_error(y_test, forecast)
                mse = mean_squared_error(y_test, forecast)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_test, forecast)
                report_str = (
                    f"==================================================\n"
                    f"Time-Series Forecasting Report ({model.__class__.__name__})\n"
                    f"==================================================\n"
                    f"Training Samples : {len(y_train)}\n"
                    f"Testing Samples  : {len(y_test)}\n"
                    f"--------------------------------------------------\n"
                    f"Mean Absolute Error (MAE)      : {mae:.4f}\n"
                    f"Mean Squared Error (MSE)       : {mse:.4f}\n"
                    f"Root Mean Squared Error (RMSE) : {rmse:.4f}\n"
                    f"R² (Coefficient of Determination): {r2:.4f}\n"
                    f"=================================================="
                )
                return {
                    'status': 'success',
                    'model_type': 'timeseries',
                    'report': report_str,
                    'metrics': {
                        'mean_absolute_error': float(mae),
                        'mean_squared_error': float(mse),
                        'root_mean_squared_error': float(rmse),
                        'r2_score': float(r2)
                    }
                }

            # 2. Anomaly Detection Report
            elif is_anomaly:
                predictions = np.asarray(model.predict(dataset.X))
                n_anomalies = int(np.sum(predictions == -1))
                n_normal = int(np.sum(predictions == 1))
                total = len(predictions)
                anomaly_ratio = n_anomalies / total if total > 0 else 0.0

                report_str = (
                    f"==================================================\n"
                    f"Anomaly Detection Report ({model.__class__.__name__})\n"
                    f"==================================================\n"
                    f"Total Samples Tested  : {total}\n"
                    f"Normal Instances Detected : {n_normal} ({100*(1-anomaly_ratio):.2f}%)\n"
                    f"Anomalies Detected        : {n_anomalies} ({100*anomaly_ratio:.2f}%)\n"
                    f"Anomaly Ratio             : {anomaly_ratio:.4f}\n"
                )

                metrics = {
                    'n_anomalies': n_anomalies,
                    'n_normal': n_normal,
                    'anomaly_ratio': anomaly_ratio
                }

                if hasattr(model, 'decision_function'):
                    scores = np.asarray(model.decision_function(dataset.X))
                    metrics['score_mean'] = float(np.mean(scores))
                    metrics['score_std'] = float(np.std(scores))
                    report_str += f"Anomaly Score Mean        : {metrics['score_mean']:.4f}\n"
                    report_str += f"Anomaly Score Std         : {metrics['score_std']:.4f}\n"

                if dataset.y is not None:
                    from tuiml.evaluation.metrics import accuracy_score, precision_score, recall_score, f1_score
                    y_true = np.asarray(dataset.y)
                    metrics['accuracy'] = float(accuracy_score(y_true, predictions))
                    metrics['precision'] = float(precision_score(y_true, predictions))
                    metrics['recall'] = float(recall_score(y_true, predictions))
                    metrics['f1'] = float(f1_score(y_true, predictions))
                    report_str += f"--------------------------------------------------\n"
                    report_str += f"Supervised Evaluation (using ground truth):\n"
                    report_str += f"  Accuracy  : {metrics['accuracy']:.4f}\n"
                    report_str += f"  Precision : {metrics['precision']:.4f}\n"
                    report_str += f"  Recall    : {metrics['recall']:.4f}\n"
                    report_str += f"  F1-Score  : {metrics['f1']:.4f}\n"

                report_str += f"=================================================="
                return {
                    'status': 'success',
                    'model_type': 'anomaly',
                    'report': report_str,
                    'metrics': metrics
                }

            # 3. Classifier Report
            elif is_classifier:
                from tuiml.evaluation.metrics import classification_report
                y_pred = model.predict(dataset.X)
                report_str = classification_report(np.asarray(dataset.y), np.asarray(y_pred))
                report_header = (
                    f"==================================================\n"
                    f"Classification Report ({model.__class__.__name__})\n"
                    f"==================================================\n"
                )
                report_str = report_header + report_str + "=================================================="

                # Also compute standard dict metrics
                metrics = model.evaluate(dataset.X, dataset.y, metrics='auto')
                return {
                    'status': 'success',
                    'model_type': 'classifier',
                    'report': report_str,
                    'metrics': metrics
                }

            # 4. Regressor Report
            elif is_regressor:
                from tuiml.evaluation.metrics import mean_absolute_error, mean_squared_error, r2_score
                y_pred = np.asarray(model.predict(dataset.X))
                y_true = np.asarray(dataset.y)
                mae = mean_absolute_error(y_true, y_pred)
                mse = mean_squared_error(y_true, y_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_true, y_pred)
                report_str = (
                    f"==================================================\n"
                    f"Regression Evaluation Report ({model.__class__.__name__})\n"
                    f"==================================================\n"
                    f"Mean Absolute Error (MAE)      : {mae:.4f}\n"
                    f"Mean Squared Error (MSE)       : {mse:.4f}\n"
                    f"Root Mean Squared Error (RMSE) : {rmse:.4f}\n"
                    f"R² (Coefficient of Determination): {r2:.4f}\n"
                    f"=================================================="
                )
                return {
                    'status': 'success',
                    'model_type': 'regressor',
                    'report': report_str,
                    'metrics': {
                        'mean_absolute_error': float(mae),
                        'mean_squared_error': float(mse),
                        'root_mean_squared_error': float(rmse),
                        'r2_score': float(r2)
                    }
                }

            # 5. Clusterer Report
            elif is_clustering:
                from tuiml.evaluation.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
                labels = model.predict(dataset.X) if hasattr(model, 'predict') else model.labels_
                sil = silhouette_score(dataset.X, labels)
                db = davies_bouldin_score(dataset.X, labels)
                ch = calinski_harabasz_score(dataset.X, labels)
                report_str = (
                    f"==================================================\n"
                    f"Clustering Evaluation Report ({model.__class__.__name__})\n"
                    f"==================================================\n"
                    f"Silhouette Coefficient         : {sil:.4f}  (closer to 1 is better)\n"
                    f"Davies-Bouldin Index           : {db:.4f}  (closer to 0 is better)\n"
                    f"Calinski-Harabasz Score        : {ch:.4f}  (higher is better)\n"
                    f"=================================================="
                )
                return {
                    'status': 'success',
                    'model_type': 'clustering',
                    'report': report_str,
                    'metrics': {
                        'silhouette_score': float(sil),
                        'davies_bouldin_score': float(db),
                        'calinski_harabasz_score': float(ch)
                    }
                }

        # ---- Handle Stage: metrics (or fallback default) ----
        # Timeseries evaluation
        if is_timeseries:
            from tuiml.evaluation.metrics import mean_absolute_error, mean_squared_error
            y = np.asarray(dataset.y) if dataset.y is not None else np.asarray(dataset.X).ravel()
            split_idx = int(len(y) * 0.8)
            y_train, y_test = y[:split_idx], y[split_idx:]
            model.fit(y_train)
            forecast = model.predict(len(y_test))
            forecast = np.asarray(forecast)

            metrics = {
                'mean_absolute_error': float(mean_absolute_error(y_test, forecast)),
                'mean_squared_error': float(mean_squared_error(y_test, forecast)),
                'root_mean_squared_error': float(np.sqrt(mean_squared_error(y_test, forecast))),
            }
            try:
                from tuiml.evaluation.metrics import r2_score
                metrics['r2_score'] = float(r2_score(y_test, forecast))
            except Exception:
                pass

            return {
                'status': 'success',
                'model_type': 'timeseries',
                'metrics': metrics,
                'train_size': int(split_idx),
                'test_size': int(len(y_test)),
                'forecast_preview': forecast[:10].tolist()
            }

        # Anomaly detection evaluation
        if is_anomaly:
            predictions = np.asarray(model.predict(dataset.X))
            n_anomalies = int(np.sum(predictions == -1))
            n_total = len(predictions)

            result = {
                'status': 'success',
                'model_type': 'anomaly',
                'metrics': {
                    'n_anomalies': n_anomalies,
                    'n_normal': int(n_total - n_anomalies),
                    'anomaly_ratio': float(n_anomalies / n_total) if n_total > 0 else 0.0
                }
            }

            if hasattr(model, 'decision_function'):
                try:
                    scores = np.asarray(model.decision_function(dataset.X))
                    result['metrics']['score_mean'] = float(np.mean(scores))
                    result['metrics']['score_std'] = float(np.std(scores))
                except Exception:
                    pass

            if dataset.y is not None:
                try:
                    from tuiml.evaluation.metrics import accuracy_score, precision_score, recall_score, f1_score
                    y_true = np.asarray(dataset.y)
                    result['metrics']['accuracy'] = float(accuracy_score(y_true, predictions))
                    result['metrics']['precision'] = float(precision_score(y_true, predictions))
                    result['metrics']['recall'] = float(recall_score(y_true, predictions))
                    result['metrics']['f1'] = float(f1_score(y_true, predictions))
                except Exception:
                    pass

            return result

        # Standard supervised/clustering evaluation
        metrics = model.evaluate(
            dataset.X, dataset.y,
            metrics=kwargs.get('metrics', 'auto')
        )
        return {'status': 'success', 'metrics': metrics}
    except FileNotFoundError as e:
        return {
            'status': 'error',
            'error': f"File not found: {str(e)}",
            'error_type': 'FileNotFoundError',
            'suggestion': 'Check file paths or use model_id from tuiml_train instead'
        }
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e),
            'error_type': type(e).__name__
        }


SPEC = ToolSpec(
    name='tuiml_evaluate',
    description="Evaluate a trained model on test data and compute metrics.",
    input_schema={
            "type": "object",
            "properties": {
                "model_id": {
                    "type": "string",
                    "description": "Model ID returned by tuiml_train (preferred)"
                },
                "model_path": {
                    "type": "string",
                    "description": "Path to saved model file (alternative to model_id)"
                },
                "data": {
                    "type": "string",
                    "description": "Path to test data file"
                },
                "target": {
                    "type": "string",
                    "description": "Target column name"
                },
                "metrics": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Metrics to compute"
                },
                "stage": {
                    "type": "string",
                    "description": "Atomic evaluation stage: 'metrics', 'report'"
                },
                "stage_kwargs": {
                    "type": "object",
                    "description": "Arbitrary stage-specific keyword arguments"
                }
            },
            "required": ["data"]
        },
    output_schema={
            "type": "object",
            "properties": {
                "status": {"type": "string", "enum": ["success", "error"]},
                "metrics": {
                    "type": "object",
                    "description": "Evaluation metrics"
                },
                "error": {"type": "string"}
            },
            "required": ["status"]
        },
    execute=execute_evaluate,
    group='workflow',
    read_only=True, destructive=False,
    idempotent=True, open_world=False,
)
