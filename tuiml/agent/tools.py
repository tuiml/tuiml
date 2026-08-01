"""
Core workflow tools for LLM integration.

Provides high-level task-oriented tools that LLMs can use
to perform complete ML workflows.
"""

from typing import Dict, List, Any, Optional
from pathlib import Path
import json
import os
import uuid
import tempfile

# Persistent TuiML state directory (survives MCP server restarts, unlike /tmp).
_TUIML_HOME = os.path.join(os.path.expanduser('~'), '.tuiml')
_MODELS_DIR = os.path.join(_TUIML_HOME, 'models')
_UPLOADS_DIR = os.path.join(_TUIML_HOME, 'uploads')
os.makedirs(_MODELS_DIR, exist_ok=True)
os.makedirs(_UPLOADS_DIR, exist_ok=True)

# Maps model_id -> file path on disk
_MODEL_INDEX: Dict[str, str] = {}

# Maps dataset_id (user-provided name or auto-generated) -> file path on disk.
# Scanned at import time so uploads persist across MCP server restarts.
_DATASET_INDEX: Dict[str, str] = {}
for _f in os.listdir(_UPLOADS_DIR):
    _full = os.path.join(_UPLOADS_DIR, _f)
    if os.path.isfile(_full):
        _DATASET_INDEX[os.path.splitext(_f)[0]] = _full

# Serving state: tracks running API servers
_SERVING_SERVERS: Dict[str, Dict[str, Any]] = {}  # server_id -> {thread, server, port, ...}

# Session call log, populated by record_session_call() which server.py calls
# after every successful tool invocation. Used by tuiml_export_notebook.
import threading as _threading
_SESSION_CALLS: List[Dict] = []          # [{tool, args}, ...]
_SESSION_LOCK = _threading.Lock()
_MODEL_ID_TO_VAR: Dict[str, str] = {}   # model_id -> "result_N"
_TRAIN_CALL_SEQ: List[int] = []         # indices into _SESSION_CALLS for train calls

# Tools that produce no reproducible Python code (discovery / admin)
_SESSION_SKIP = {
    'tuiml_export_notebook', 'tuiml_list', 'tuiml_search', 'tuiml_describe',
    'tuiml_server_status', 'tuiml_system_info', 'tuiml_restart', 'tuiml_self_update',
    'tuiml_read_data', 'tuiml_list_files', 'tuiml_search_source',
    'tuiml_read_algorithm',
}


def record_session_call(tool_name: str, args: dict, result: dict) -> None:
    """Record a *successful* MCP tool call for notebook export.

    Called by server.py's call_tool handler after every invocation. Only
    successful calls are stored: failed calls (bad algorithm name, schema
    mismatch, etc.) return ``{"status": "error"}`` rather than raising, and
    must not become notebook cells, otherwise the exported notebook would
    raise when re-run. Strips internal kwargs (_progress_callback, etc.)
    before storing so the notebook sees only user-visible arguments.

    Parameters
    ----------
    tool_name : str
        Name of the MCP tool that was invoked (e.g. ``"tuiml_train"``).
    args : dict
        Arguments the tool was called with. Keys starting with an
        underscore are stripped before recording.
    result : dict
        The result dict returned by the tool executor. Only calls whose
        result has ``status == "success"`` are recorded.

    Returns
    -------
    None
        The call is appended to the module-level session log as a side
        effect; nothing is returned.
    """
    if tool_name in _SESSION_SKIP:
        return
    # Skip anything that didn't succeed. Tool executors signal failure via a
    # status field instead of raising, so a missing/non-success status means
    # the call produced no reproducible result worth exporting.
    if not isinstance(result, dict) or result.get('status') != 'success':
        return
    clean_args = {k: v for k, v in args.items() if not k.startswith('_')}
    # Capture the effective random seed. execute_tool resolves the seed (explicit
    # arg → global seed → default) and writes it back into the *result*, not the
    # args. Fold it into the recorded args so the exported notebook reproduces the
    # exact run even when the seed was auto-resolved rather than passed explicitly.
    if isinstance(result, dict) and result.get('random_seed') is not None \
            and 'random_seed' not in clean_args:
        clean_args['random_seed'] = result['random_seed']
    with _SESSION_LOCK:
        idx = len(_SESSION_CALLS)
        _SESSION_CALLS.append({'tool': tool_name, 'args': clean_args})
        if tool_name == 'tuiml_train' and isinstance(result, dict) and result.get('model_id'):
            n = len(_TRAIN_CALL_SEQ) + 1
            _MODEL_ID_TO_VAR[result['model_id']] = f'result_{n}'
            _TRAIN_CALL_SEQ.append(idx)


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

# =============================================================================
# Tool Schemas (JSON Schema format for MCP)
# =============================================================================

# Output Schemas for MCP Tools
OUTPUT_SCHEMAS = {
    "tuiml_train": {
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
    "tuiml_predict": {
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
    "tuiml_evaluate": {
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
    "tuiml_benchmark": {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["success", "error"]},
            "summary": {"type": "string"},
            "results": {
                "type": "object",
                "description": "Results by dataset and model"
            },
            "algorithms": {"type": "array", "items": {"type": "string"}},
            "datasets": {"type": "array", "items": {"type": "string"}},
            "cv_folds": {"type": "integer"},
            "error": {"type": "string"},
            "suggested_metrics": {"type": "array", "items": {"type": "string"}},
            "algorithm_types": {"type": "array", "items": {"type": "string"}},
            "random_seed": {"type": "integer"}
        },
        "required": ["status"]
    },
    "tuiml_upload_data": {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["success", "error"]},
            "dataset_id": {
                "type": "string",
                "description": "Stable name to pass as `data` to tuiml_train / tuiml_predict / tuiml_evaluate"
            },
            "file_path": {"type": "string"},
            "rows": {"type": "integer"},
            "features": {"type": "integer"},
            "feature_names": {"type": "array", "items": {"type": "string"}},
            "message": {"type": "string"},
            "error": {"type": "string"}
        },
        "required": ["status"]
    },
    "tuiml_save_model": {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["success", "error"]},
            "model_id": {"type": "string"},
            "source": {"type": "string"},
            "destination": {"type": "string"},
            "message": {"type": "string"},
            "error": {"type": "string"}
        },
        "required": ["status"]
    },
    "tuiml_serve_model": {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["success", "error"]},
            "server_id": {"type": "string"},
            "model_id": {"type": "string"},
            "url": {"type": "string", "description": "Base URL of the serving API"},
            "endpoints": {"type": "object", "description": "Map of endpoint names to URLs"},
            "example_curl": {"type": "string"},
            "error": {"type": "string"}
        },
        "required": ["status"]
    },
    "tuiml_stop_server": {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["success", "error"]},
            "message": {"type": "string"},
            "stopped": {"type": "array"},
            "error": {"type": "string"}
        },
        "required": ["status"]
    },
    "tuiml_server_status": {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["success", "error"]},
            "count": {"type": "integer"},
            "servers": {"type": "array"},
            "error": {"type": "string"}
        },
        "required": ["status"]
    },
    "tuiml_list": {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["success", "error"]},
            "total": {"type": "integer", "description": "Total number of components"},
            "count": {"type": "integer", "description": "Number of components returned"},
            "limit": {"type": "integer"},
            "offset": {"type": "integer"},
            "has_more": {"type": "boolean"},
            "components": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "description": {"type": "string"},
                        "category": {"type": "string"}
                    }
                }
            },
            "error": {"type": "string"}
        },
        "required": ["status"]
    },
    "tuiml_describe": {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["success", "error"]},
            "type": {"type": "string"},
            "name": {"type": "string"},
            "description": {"type": "string"},
            "parameters": {"type": "object"},
            "error": {"type": "string"}
        },
        "required": ["status"]
    },
    "tuiml_search": {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["success", "error"]},
            "query": {"type": "string"},
            "count": {"type": "integer"},
            "results": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "description": {"type": "string"},
                        "category": {"type": "string"}
                    }
                }
            },
            "error": {"type": "string"}
        },
        "required": ["status"]
    },
    "tuiml_plot": {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["success", "error"]},
            "plot_type": {"type": "string"},
            "description": {"type": "string"},
            "error": {"type": "string"}
        },
        "required": ["status"]
    },
    "tuiml_profile_data": {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["success", "error"]},
            "name": {"type": "string"},
            "shape": {"type": "array", "items": {"type": "integer"}},
            "n_samples": {"type": "integer"},
            "n_features": {"type": "integer"},
            "feature_names": {"type": "array", "items": {"type": "string"}},
            "dtypes": {"type": "object"},
            "missing_values": {"type": "object"},
            "numeric_stats": {"type": "object"},
            "class_distribution": {"type": "object"},
            "target_column": {"type": "string"},
            "error": {"type": "string"}
        },
        "required": ["status"]
    },
    "tuiml_generate_data": {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["success", "error"]},
            "generator": {"type": "string"},
            "file_path": {"type": "string"},
            "shape": {"type": "array", "items": {"type": "integer"}},
            "feature_names": {"type": "array", "items": {"type": "string"}},
            "target_names": {"type": "array", "items": {"type": "string"}},
            "preview": {"type": "object"},
            "random_seed": {"type": "integer"},
            "error": {"type": "string"}
        },
        "required": ["status"]
    },
    "tuiml_preprocess": {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["success", "error"]},
            "file_path": {"type": "string"},
            "files": {
                "type": "object",
                "description": "Mapping of split names/folds to file paths"
            },
            "stage": {"type": "string"},
            "split_type": {"type": "string"},
            "n_splits": {"type": "integer"},
            "original_shape": {"type": "array", "items": {"type": "integer"}},
            "new_shape": {"type": "array", "items": {"type": "integer"}},
            "steps_applied": {"type": "array", "items": {"type": "string"}},
            "error": {"type": "string"}
        },
        "required": ["status"]
    },
    "tuiml_select_features": {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["success", "error"]},
            "method": {"type": "string"},
            "n_original": {"type": "integer"},
            "n_selected": {"type": "integer"},
            "selected_features": {"type": "array", "items": {"type": "string"}},
            "scores": {"type": "object"},
            "file_path": {"type": "string"},
            "error": {"type": "string"}
        },
        "required": ["status"]
    },
    "tuiml_test_statistics": {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["success", "error"]},
            "test": {"type": "string"},
            "statistic": {"type": "number"},
            "p_value": {"type": "number"},
            "significant": {"type": "boolean"},
            "details": {"type": "object"},
            "error": {"type": "string"}
        },
        "required": ["status"]
    },
    "tuiml_tune": {
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
    "tuiml_read_data": {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["success", "error"]},
            "name": {"type": "string"},
            "shape": {"type": "array", "items": {"type": "integer"}},
            "columns": {"type": "array", "items": {"type": "string"}},
            "n_rows_returned": {"type": "integer"},
            "rows": {"type": "array", "items": {"type": "object"}},
            "error": {"type": "string"}
        },
        "required": ["status"]
    },
    "tuiml_export_notebook": {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["success", "error"]},
            "path": {"type": "string", "description": "Absolute path to the written .ipynb file"},
            "cells": {"type": "integer", "description": "Total number of notebook cells"},
            "workflow_calls": {"type": "integer", "description": "Number of MCP calls translated"},
            "message": {"type": "string"},
            "error": {"type": "string"}
        },
        "required": ["status"]
    }
}

# Component tool output schema (generic for all component tools)
COMPONENT_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "status": {"type": "string", "enum": ["success", "error"]},
        "result": {"type": "string", "description": "String representation of the component"},
        "type": {"type": "string", "description": "Component class name"},
        "error": {"type": "string"}
    },
    "required": ["status"]
}

WORKFLOW_TOOLS = {
    "tuiml_train": {
        "name": "tuiml_train",
        "description": (
            "Train a machine learning model with evaluation. Two evaluation modes:\n"
            "1. Holdout (default): splits data into train/test sets using test_size. "
            "Returns metrics on the test set and predictions.\n"
            "2. Cross-validation: set cv=5 or cv=10 for k-fold CV. "
            "Returns mean/std metrics across folds.\n"
            "If neither cv nor test_size is provided, defaults to holdout with test_size=0.2.\n"
            "Supports classifiers, regressors, and clusterers."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "algorithm": {
                    "type": "string",
                    "description": (
                        "Algorithm class name. Examples:\n"
                        "- Classifiers: 'RandomForestClassifier', 'SVM', 'NaiveBayesClassifier', 'C45TreeClassifier'\n"
                        "- Regressors: 'LinearRegression', 'M5ModelTreeRegressor'\n"
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
        }
    },

    "tuiml_predict": {
        "name": "tuiml_predict",
        "description": (
            "Make predictions using a trained model on new data. Supports supervised models, "
            "timeseries models (use 'steps' parameter), and anomaly detection models."
        ),
        "inputSchema": {
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
        }
    },

    "tuiml_evaluate": {
        "name": "tuiml_evaluate",
        "description": "Evaluate a trained model on test data and compute metrics.",
        "inputSchema": {
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
        }
    },

    "tuiml_benchmark": {
        "name": "tuiml_benchmark",
        "description": "Compare multiple algorithms on one or more datasets with cross-validation and statistical tests. Supports supervised learning (classification, regression) and unsupervised learning (clustering). Pass a single dataset name or a list of dataset names to benchmark across multiple datasets.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "algorithms": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of algorithm class names to compare (e.g., ['RandomForestClassifier', 'SVM'] for classification, ['KMeansClusterer', 'GaussianMixtureClusterer'] for clustering)"
                },
                "data": {
                    "oneOf": [
                        {"type": "string"},
                        {"type": "array", "items": {"type": "string"}}
                    ],
                    "description": "Dataset name(s) or file path(s). Single string (e.g., 'iris') or list of names (e.g., ['iris', 'wine', 'breast_cancer']) to compare across multiple datasets."
                },
                "target": {
                    "type": "string",
                    "description": "Target column name (for supervised learning)"
                },
                "cv": {
                    "type": "integer",
                    "default": 10,
                    "description": "Number of CV folds (ignored for clustering)"
                },
                "metrics": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Metrics to compute. Use exact function names. IMPORTANT: Must match algorithm type:\n"
                        "- Classification: ['accuracy_score', 'f1_score', 'precision_score', 'recall_score', 'roc_auc_score']\n"
                        "- Regression: ['r2_score', 'root_mean_squared_error', 'mean_absolute_error', 'mean_squared_error']\n"
                        "- Clustering: ['silhouette_score', 'calinski_harabasz_score', 'davies_bouldin_score']\n"
                        "If omitted, appropriate metrics are automatically selected based on algorithm type."
                    )
                },
                "random_seed": {
                    "type": "integer",
                    "description": "Random seed for reproducibility"
                }
            },
            "required": ["algorithms", "data", "target"]
        }
    },

    "tuiml_upload_data": {
        "name": "tuiml_upload_data",
        "description": (
            "Register a dataset for use with other TuiML tools. "
            "Provide either a file_path to an existing file on disk (preferred for large datasets), "
            "or content as raw text for small inline datasets. "
            "Supported formats: CSV, TSV, ARFF, Parquet, Excel (xlsx/xls), JSON, JSONL, NumPy (npy/npz). "
            "Returns a validated path for use with tuiml_train, tuiml_preprocess, etc."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": (
                        "Path to an existing dataset file on disk. "
                        "Supported: .csv, .tsv, .arff, .parquet, .pq, .xlsx, .xls, .json, .jsonl, .ndjson, .npy, .npz"
                    )
                },
                "content": {
                    "type": "string",
                    "description": "Raw text content for small inline datasets (use with 'format')"
                },
                "format": {
                    "type": "string",
                    "enum": ["csv", "tsv", "arff", "json", "jsonl"],
                    "default": "csv",
                    "description": "File format, only needed with 'content'; auto-detected from file_path extension"
                },
                "name": {
                    "type": "string",
                    "description": "Optional name for the dataset (without extension)"
                }
            },
            "required": []
        }
    },

    "tuiml_save_model": {
        "name": "tuiml_save_model",
        "description": "Copy a trained model to a custom path. Use this when the user wants to save or download a model to a specific location.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "model_id": {
                    "type": "string",
                    "description": "Model ID returned by tuiml_train"
                },
                "destination": {
                    "type": "string",
                    "description": "Destination file path (e.g., './my_model.joblib', '/home/user/models/rf.joblib')"
                }
            },
            "required": ["model_id", "destination"]
        }
    },

    "tuiml_serve_model": {
        "name": "tuiml_serve_model",
        "description": (
            "Start a REST API server to serve a trained model for predictions. "
            "Returns the URL with endpoints: POST /predict, POST /models/{id}/predict, "
            "GET /health, GET /models, GET /docs (Swagger UI)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "model_id": {
                    "type": "string",
                    "description": "Model ID returned by tuiml_train"
                },
                "model_path": {
                    "type": "string",
                    "description": "Path to model file (alternative to model_id)"
                },
                "port": {
                    "type": "integer",
                    "default": 8000,
                    "minimum": 1024,
                    "maximum": 65535,
                    "description": "Port to serve on (default: 8000)"
                },
                "host": {
                    "type": "string",
                    "default": "127.0.0.1",
                    "description": "Host to bind to (default: 127.0.0.1)"
                }
            },
            "required": []
        }
    },

    "tuiml_stop_server": {
        "name": "tuiml_stop_server",
        "description": "Stop a running model serving API server.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "server_id": {
                    "type": "string",
                    "description": "Server ID returned by tuiml_serve_model. If omitted, stops all servers."
                }
            },
            "required": []
        }
    },

    "tuiml_server_status": {
        "name": "tuiml_server_status",
        "description": "Get status of running model serving API servers.",
        "inputSchema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },

    "tuiml_plot": {
        "name": "tuiml_plot",
        "description": (
            "Generate a visualization/plot for model analysis. Returns the plot as an "
            "inline image. Supported plot types: confusion_matrix, roc_curve, pr_curve, "
            "learning_curve, tree, feature_importance."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "plot_type": {
                    "type": "string",
                    "enum": [
                        "confusion_matrix",
                        "roc_curve",
                        "pr_curve",
                        "learning_curve",
                        "tree",
                        "feature_importance",
                        "cd_diagram",
                        "boxplot_comparison",
                        "heatmap",
                        "ranking_table"
                    ],
                    "description": (
                        "Type of plot to generate:\n"
                        "- confusion_matrix: Heatmap of predicted vs actual classes (requires model_id + data + target)\n"
                        "- roc_curve: ROC curve with AUC for binary or multiclass classifiers (requires model_id + data + target)\n"
                        "- pr_curve: Precision-Recall curve with AP for binary classifiers (requires model_id + data + target)\n"
                        "- learning_curve: Training vs validation score over dataset sizes (requires algorithm + data + target)\n"
                        "- tree: Decision tree structure visualization (requires model_id)\n"
                        "- feature_importance: Bar chart of feature importances (requires model_id)\n"
                        "- cd_diagram: Critical difference diagram for algorithm comparison (requires benchmark_results)\n"
                        "- boxplot_comparison: Box plot comparing algorithm scores (requires benchmark_results)\n"
                        "- heatmap: Heatmap of algorithm scores across datasets (requires benchmark_results)\n"
                        "- ranking_table: Ranking table of algorithms (requires benchmark_results)"
                    )
                },
                "model_id": {
                    "type": "string",
                    "description": "Model ID from tuiml_train (required for most plot types)"
                },
                "model_path": {
                    "type": "string",
                    "description": "Path to saved model file (alternative to model_id)"
                },
                "data": {
                    "type": "string",
                    "description": "Data file path or built-in dataset name (required for confusion_matrix, roc_curve, pr_curve, learning_curve)"
                },
                "target": {
                    "type": "string",
                    "description": "Target column name (required for confusion_matrix, roc_curve, pr_curve, learning_curve)"
                },
                "algorithm": {
                    "type": "string",
                    "description": "Algorithm class name (required for learning_curve)"
                },
                "title": {
                    "type": "string",
                    "description": "Custom plot title (optional)"
                },
                "normalize": {
                    "type": "boolean",
                    "default": False,
                    "description": "Normalize confusion matrix to show percentages (confusion_matrix only)"
                },
                "benchmark_results": {
                    "type": "object",
                    "description": "Algorithm CV scores for comparison plots: { 'AlgoName': [score1, score2, ...], ... }"
                }
            },
            "required": ["plot_type"]
        }
    },

    "tuiml_profile_data": {
        "name": "tuiml_profile_data",
        "description": (
            "Inspect a dataset before training, shape, dtypes, missing values, "
            "basic statistics, and class distribution. Works with file paths or "
            "built-in dataset names."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "data": {
                    "type": "string",
                    "description": "Data file path or built-in dataset name (e.g., 'iris', 'wine', '/path/to/data.csv')"
                },
                "target": {
                    "type": "string",
                    "description": "Target column name (optional, used for class distribution)"
                }
            },
            "required": ["data"]
        }
    },

    "tuiml_generate_data": {
        "name": "tuiml_generate_data",
        "description": (
            "Generate synthetic datasets for testing and demos. Supports classification "
            "(RandomRBF, Agrawal, LED, Hyperplane), regression (Friedman, MexicanHat, Sine), "
            "and clustering (Blobs, Moons, Circles, SwissRoll) generators."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "generator": {
                    "type": "string",
                    "enum": [
                        "RandomRBF", "Agrawal", "LED", "Hyperplane",
                        "Friedman", "MexicanHat", "Sine",
                        "Blobs", "Moons", "Circles", "SwissRoll"
                    ],
                    "description": "Generator class name"
                },
                "n_samples": {
                    "type": "integer",
                    "default": 100,
                    "description": "Number of samples to generate"
                },
                "n_features": {
                    "type": "integer",
                    "description": "Number of features (not all generators support this)"
                },
                "n_classes": {
                    "type": "integer",
                    "description": "Number of classes (classification generators only)"
                },
                "n_clusters": {
                    "type": "integer",
                    "description": "Number of clusters (clustering generators only)"
                },
                "noise": {
                    "type": "number",
                    "description": "Noise level (regression generators only)"
                },
                "random_seed": {
                    "type": "integer",
                    "description": "Random seed for reproducibility"
                },
                "generator_params": {
                    "type": "object",
                    "description": "Additional generator-specific parameters"
                }
            },
            "required": ["generator"]
        }
    },

    "tuiml_preprocess": {
        "name": "tuiml_preprocess",
        "description": (
            "Apply preprocessing steps to a dataset and return the result as a new file. "
            "Supports running standard pipelines or single atomic stages like split, impute, "
            "balance, scale, encode, and discretize."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "data": {
                    "type": "string",
                    "description": "Data file path or built-in dataset name"
                },
                "target": {
                    "type": "string",
                    "description": "Target column name (excluded from preprocessing, re-appended to output)"
                },
                "steps": {
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
                        "Examples: ['StandardScaler', 'SimpleImputer'] or "
                        "[{'name': 'SimpleImputer', 'strategy': 'median'}, 'MinMaxScaler']"
                    )
                },
                "stage": {
                    "type": "string",
                    "description": "Atomic preprocessing stage to execute: 'split', 'impute', 'balance', 'scale', 'encode', 'discretize'"
                },
                "stage_kwargs": {
                    "type": "object",
                    "description": "Arbitrary keyword arguments for the selected stage (e.g. kfold, test_size, strategy, method)"
                },
                "output": {
                    "type": "string",
                    "description": "Output path to save the generated file(s)"
                },
                "save_as": {
                    "type": "string",
                    "description": "Custom output file path (optional, alias for output)"
                }
            },
            "required": ["data"]
        }
    },

    "tuiml_select_features": {
        "name": "tuiml_select_features",
        "description": (
            "Run feature selection on a dataset and return selected feature names/indices. "
            "Supports filter methods (SelectKBestSelector, SelectPercentileSelector, "
            "VarianceThresholdSelector, SelectFprSelector, SelectThresholdSelector), "
            "correlation-based (CFSSelector), and wrapper methods (WrapperSelector)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
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
                    "enum": [
                        "SelectKBestSelector", "SelectPercentileSelector",
                        "VarianceThresholdSelector", "CFSSelector",
                        "WrapperSelector", "SelectFprSelector", "SelectThresholdSelector"
                    ],
                    "description": "Feature selection method"
                },
                "k": {
                    "type": "integer",
                    "description": "Number of top features to select (SelectKBestSelector)"
                },
                "threshold": {
                    "type": "number",
                    "description": "Threshold for VarianceThresholdSelector or SelectThresholdSelector"
                },
                "method_params": {
                    "type": "object",
                    "description": "Additional method-specific parameters"
                }
            },
            "required": ["data", "target", "method"]
        }
    },

    "tuiml_test_statistics": {
        "name": "tuiml_test_statistics",
        "description": (
            "Run statistical significance tests on experiment results (cross-validation scores). "
            "Supports Friedman test, Nemenyi post-hoc, Wilcoxon signed-rank, paired t-test, "
            "one-way ANOVA, Friedman aligned ranks, and Quade test."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "test": {
                    "type": "string",
                    "enum": [
                        "friedman", "nemenyi", "wilcoxon",
                        "paired_t", "anova", "friedman_aligned", "quade"
                    ],
                    "description": (
                        "Statistical test to run:\n"
                        "- friedman: Non-parametric test for 3+ algorithms\n"
                        "- nemenyi: Post-hoc pairwise test after Friedman\n"
                        "- wilcoxon: Non-parametric pairwise test (2 algorithms)\n"
                        "- paired_t: Parametric pairwise test (2 algorithms)\n"
                        "- anova: Parametric test for 3+ groups\n"
                        "- friedman_aligned: More powerful variant of Friedman\n"
                        "- quade: Non-parametric test accounting for dataset difficulty"
                    )
                },
                "results": {
                    "type": "object",
                    "description": "Algorithm CV scores: { 'AlgorithmName': [score1, score2, ...], ... }"
                },
                "significance_level": {
                    "type": "number",
                    "default": 0.05,
                    "description": "Significance level (alpha), default 0.05"
                },
                "higher_better": {
                    "type": "boolean",
                    "default": True,
                    "description": "Whether higher scores are better (default True)"
                }
            },
            "required": ["test", "results"]
        }
    },

    "tuiml_tune": {
        "name": "tuiml_tune",
        "description": (
            "Hyperparameter optimization for any algorithm. Supports grid search, "
            "random search, and Bayesian optimization. Returns best parameters, "
            "best score, and a trained model with optimal settings."
        ),
        "inputSchema": {
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
        }
    },

    "tuiml_read_data": {
        "name": "tuiml_read_data",
        "description": (
            "Read and preview actual rows from a dataset. Returns sample rows as a list of "
            "dictionaries. Supports head, tail, random sample, or specific row indices."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "data": {
                    "type": "string",
                    "description": "Data file path or built-in dataset name (e.g., 'iris', '/tmp/tuiml_preprocessed/file.csv')"
                },
                "n_rows": {
                    "type": "integer",
                    "default": 10,
                    "description": "Number of rows to return (default: 10, max: 100)"
                },
                "mode": {
                    "type": "string",
                    "enum": ["head", "tail", "sample", "indices"],
                    "default": "head",
                    "description": (
                        "How to select rows:\n"
                        "- head: First n_rows (default)\n"
                        "- tail: Last n_rows\n"
                        "- sample: Random sample of n_rows\n"
                        "- indices: Specific row indices (provide 'indices' parameter)"
                    )
                },
                "indices": {
                    "type": "array",
                    "items": {"type": "integer"},
                    "description": "Specific row indices to return (only used when mode='indices')"
                },
                "columns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Subset of columns to return (optional, returns all if omitted)"
                },
                "include_target": {
                    "type": "boolean",
                    "default": True,
                    "description": "Include the target column in the output (default: True)"
                },
                "target": {
                    "type": "string",
                    "description": "Target column name (used to label the target in the output)"
                }
            },
            "required": ["data"]
        }
    },
    "tuiml_system_info": {
        "name": "tuiml_system_info",
        "description": (
            "Report details about the TuiML installation on this machine: "
            "installed version, install method (uv tool / pip / editable), "
            "package location, Python executable, platform, and the latest "
            "version available on PyPI. Agents can use this to decide whether "
            "an update is worth running via tuiml_self_update."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "check_latest": {
                    "type": "boolean",
                    "description": "Query PyPI for the latest released version. Defaults to true.",
                    "default": True,
                }
            },
        },
    },
    "tuiml_get_skeleton": {
        "name": "tuiml_get_skeleton",
        "description": (
            "Return a ready-to-edit Python source template for a new @classifier "
            "or @regressor class. Agents should call this, fill in fit() and "
            "predict(), then pass the completed source to tuiml_create_algorithm. "
            ""
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "kind": {
                    "type": "string",
                    "enum": ["classifier", "regressor"],
                    "description": "Task kind the new algorithm targets.",
                },
                "class_name": {
                    "type": "string",
                    "description": "Python identifier for the new class, e.g. 'MyGradientBoosting'.",
                    "default": "MyAlgorithm",
                },
                "version": {
                    "type": "string",
                    "description": "Initial semver, e.g. '1.0.0'.",
                    "default": "1.0.0",
                },
                "description": {
                    "type": "string",
                    "description": "One-line docstring for the class.",
                    "default": "Describe what your algorithm does.",
                },
            },
            "required": ["kind"],
        },
    },
    "tuiml_create_algorithm": {
        "name": "tuiml_create_algorithm",
        "description": (
            "Persist, validate, and register a new agent-authored algorithm. "
            "The source is AST-validated (forbidden modules: subprocess, socket, "
            "os, urllib, requests, …; forbidden calls: eval, exec, open, __import__) "
            "and saved to ~/.tuiml/user_algorithms/<name>/<version>/algorithm.py. "
            "After registration, the algorithm is available via its class name to "
            "every existing MCP tool (tuiml_train, tuiml_benchmark, tuiml_describe). "
            "Each version is also registered under a pinned alias "
            "<ClassName>_v<major>_<minor>_<patch> so you can A/B compare versions "
            "inside a single tuiml_benchmark. "
            ""
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Directory name, usually equal to the class name (Python identifier).",
                },
                "kind": {
                    "type": "string",
                    "enum": ["classifier", "regressor"],
                    "description": "Task kind. Must match the imported class's base type.",
                },
                "code": {
                    "type": "string",
                    "description": "Full Python source. Must define exactly one @classifier or @regressor class.",
                },
                "version": {
                    "type": "string",
                    "description": "Semver for this submission, e.g. '1.0.0', '1.0.1'.",
                    "default": "1.0.0",
                },
                "description": {
                    "type": "string",
                    "description": "Optional short description (falls back to the class docstring).",
                },
                "force": {
                    "type": "boolean",
                    "description": "Overwrite an existing file at <name>/<version>/. Bump the version instead when possible.",
                    "default": False,
                },
            },
            "required": ["name", "kind", "code"],
        },
    },
    "tuiml_delete_algorithm": {
        "name": "tuiml_delete_algorithm",
        "description": (
            "Delete a user algorithm from disk. Pass only `name` to remove every "
            "version; pass both to remove a single version. Registry entries for "
            "already-loaded classes remain until the MCP server restarts. "
            ""
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "version": {
                    "type": "string",
                    "description": "If omitted, all versions are removed.",
                },
            },
            "required": ["name"],
        },
    },
    "tuiml_self_update": {
        "name": "tuiml_self_update",
        "description": (
            "Upgrade the TuiML installation to the latest PyPI release. "
            "Auto-detects the installer (uv tool install vs pip) and runs the "
            "appropriate upgrade command. Returns the command, its stdout / "
            "stderr, and the resulting version. Editable / dev installs are "
            "refused, those need a git pull instead.\n\n"
            "IMPORTANT: the running MCP process is still using the old package "
            "in memory. Call tuiml_restart immediately after, or restart the "
            "client manually, for the new version to take effect."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "dry_run": {
                    "type": "boolean",
                    "description": "Print the command that would be run without executing it.",
                    "default": False,
                },
                "target_version": {
                    "type": "string",
                    "description": "Upgrade to a specific version (e.g. '0.1.3'). Defaults to latest.",
                },
            },
        },
    },
    "tuiml_restart": {
        "name": "tuiml_restart",
        "description": (
            "Restart every running tuiml-mcp process so AI clients pick up "
            "freshly installed code (e.g. right after tuiml_self_update). "
            "Sends SIGTERM (then SIGKILL after a grace period) to every "
            "tuiml-mcp child; each parent client (Claude Desktop, Cursor, "
            "Codex, ...) automatically respawns its child on the next "
            "request. The current MCP process schedules a self-exit AFTER "
            "this response is sent, so the agent should expect a brief "
            "reconnect."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "include_self": {
                    "type": "boolean",
                    "description": (
                        "Also exit the current MCP process after responding. "
                        "True is the usual case, it forces the calling "
                        "client to respawn with the new code too. Set False "
                        "to restart only the other clients' instances."
                    ),
                    "default": True,
                },
            },
        },
    },
    "tuiml_export_notebook": {
        "name": "tuiml_export_notebook",
        "description": (
            "Export the current MCP chat session as a reproducible Jupyter notebook (.ipynb). "
            "Training, experiment, tuning, plotting, and data-prep steps performed in this "
            "session are translated to equivalent Python API calls so the user can re-run the "
            "full workflow without the AI client. "
            "Call this at the end of a session when the user wants to save their work as a notebook."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": (
                        "Destination file path for the notebook. "
                        "Defaults to ~/tuiml_session.ipynb if omitted."
                    ),
                },
                "title": {
                    "type": "string",
                    "description": "Optional custom title for the notebook header cell.",
                },
            },
            "required": [],
        },
    },
}

CODE_TOOLS = {
    "tuiml_read_algorithm": {
        "name": "tuiml_read_algorithm",
        "description": (
            "Return the full source code of any algorithm, user-authored or built-in. "
            "For user algorithms pass the directory name (class name). "
            "For built-in algorithms set builtin=true and pass the class name "
            "(e.g. 'RandomForestClassifier') or file stem (e.g. 'random_forest'). "
            "Source is returned both raw and with line numbers for easy reference. "
            "Built-in algorithms are read-only; use tuiml_create_algorithm to fork them."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Algorithm name (class name or directory name).",
                },
                "version": {
                    "type": "string",
                    "description": "Specific version to read (e.g. '1.0.2'). Defaults to latest.",
                },
                "builtin": {
                    "type": "boolean",
                    "default": False,
                    "description": "Set true to read a built-in tuiml algorithm instead of a user algorithm.",
                },
            },
            "required": ["name"],
        },
    },
    "tuiml_list_files": {
        "name": "tuiml_list_files",
        "description": (
            "List all algorithm source files, built-in and/or user-authored. "
            "Returns file paths, categories, and metadata. Use this before "
            "tuiml_read_algorithm to discover what's available and find the right name."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "builtin": {
                    "type": "boolean",
                    "default": True,
                    "description": "Include built-in tuiml algorithm files.",
                },
                "user": {
                    "type": "boolean",
                    "default": True,
                    "description": "Include user-authored algorithm files.",
                },
            },
        },
    },
    "tuiml_search_source": {
        "name": "tuiml_search_source",
        "description": (
            "Grep for a pattern inside algorithm source files. "
            "Returns matching lines with file path and line number, "
            "use this to locate a specific function, variable, or logic before editing. "
            "Accepts a regex pattern."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Regex pattern to search for.",
                },
                "name": {
                    "type": "string",
                    "description": "Scope search to one user algorithm by name. Omit to search all.",
                },
                "builtin": {
                    "type": "boolean",
                    "default": True,
                    "description": "Search built-in algorithm files.",
                },
                "user": {
                    "type": "boolean",
                    "default": True,
                    "description": "Search user-authored algorithm files.",
                },
            },
            "required": ["query"],
        },
    },
    "tuiml_edit_algorithm": {
        "name": "tuiml_edit_algorithm",
        "description": (
            "Apply a targeted str_replace edit to a user algorithm. "
            "Replaces exactly one occurrence of old_string with new_string, "
            "fails loudly if old_string is not found or appears more than once "
            "(make it more specific with surrounding context). "
            "The edited source is AST-validated and the algorithm is re-registered "
            "so all MCP tools immediately see the change. "
            "Workflow: tuiml_read_algorithm → identify the text to change → tuiml_edit_algorithm. "
            "Set bump_version=true to save as a new patch version instead of overwriting. "
            "Built-in algorithms cannot be edited, fork them first with tuiml_create_algorithm. "
            ""
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "User algorithm name (directory name / class name).",
                },
                "old_string": {
                    "type": "string",
                    "description": "The exact text to find and replace. Must be unique in the file.",
                },
                "new_string": {
                    "type": "string",
                    "description": "The replacement text.",
                },
                "version": {
                    "type": "string",
                    "description": "Target a specific version. Defaults to latest.",
                },
                "bump_version": {
                    "type": "boolean",
                    "default": False,
                    "description": "Save the edit as a new patch version instead of overwriting the current one.",
                },
            },
            "required": ["name", "old_string", "new_string"],
        },
    },
}

DISCOVERY_TOOLS = {
    "tuiml_list": {
        "name": "tuiml_list",
        "description": (
            "List TuiML components (algorithms, preprocessors, datasets, features) "
            "or custom user-authored algorithms. Use category='custom' to list "
            "algorithms created via tuiml_create_algorithm, shows all versions, "
            "best scores, and run history. Pass include_runs=true for full experiment "
            "history (useful for auto-research: see what was tried and what to improve next)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "category": {
                    "type": "string",
                    "enum": ["algorithm", "preprocessing", "dataset", "feature", "splitting", "custom", "all"],
                    "default": "all",
                    "description": "Category to list. Use 'custom' for user-authored algorithms."
                },
                "type": {
                    "type": "string",
                    "enum": ["classifier", "regressor", "clusterer", "anomaly", "timeseries"],
                    "description": "Filter algorithms by type (ignored for category='custom')."
                },
                "search": {
                    "type": "string",
                    "description": "Search keyword to filter results."
                },
                "include_runs": {
                    "type": "boolean",
                    "default": False,
                    "description": "For category='custom': include full experiment run history and best scores per version."
                },
                "limit": {
                    "type": "integer",
                    "default": 50,
                    "minimum": 1,
                    "maximum": 200,
                    "description": "Maximum number of results to return (default: 50)."
                },
                "offset": {
                    "type": "integer",
                    "default": 0,
                    "minimum": 0,
                    "description": "Number of results to skip for pagination."
                }
            },
            "required": []
        }
    },

    "tuiml_describe": {
        "name": "tuiml_describe",
        "description": "Get detailed information and parameter schema for any TuiML component.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Component name (e.g., 'RandomForestClassifier', 'SimpleImputer', 'iris')"
                }
            },
            "required": ["name"]
        }
    },

}

# =============================================================================
# Tool Executors
# =============================================================================

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
        
        from tuiml.workflow import _inject_seed_to_algorithm
        algo_params = _inject_seed_to_algorithm(model_cls, algo_params, random_seed)
        
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
            from tuiml.workflow import _inject_seed_to_algorithm
            algo_params = _inject_seed_to_algorithm(model_cls, algo_params, random_seed)
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
            from tuiml.workflow import _inject_seed_to_algorithm
            algo_params = _inject_seed_to_algorithm(model_cls, algo_params, random_seed)
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

def execute_list(**kwargs) -> Dict[str, Any]:
    """List available components (algorithms, preprocessors, datasets, ...).

    Backs the ``tuiml_list`` tool. Supports filtering, pagination, and a
    special ``category='custom'`` mode that lists user-created algorithms
    with their research-log summaries.

    Parameters
    ----------
    category : str, default='all'
        Component category to list (``'all'``, ``'algorithm'``,
        ``'preprocessing'``, ``'custom'``, ...). Arrives via ``**kwargs``,
        like all parameters below.
    search : str, default=None
        Case-insensitive substring filter on name/description.
    type : str, default=None
        Algorithm type filter: ``'classifier'``, ``'regressor'``,
        ``'clusterer'``, ``'anomaly'``, or ``'timeseries'``.
    limit : int, default=50
        Maximum number of entries to return.
    offset : int, default=0
        Pagination offset.
    include_runs : bool, default=False
        For ``category='custom'``: include per-version run details.

    Returns
    -------
    result : dict
        On success: ``status`` (``'success'``), ``total``, ``count``,
        ``limit``, ``offset``, ``has_more``, and ``components`` (each with
        ``name``, ``description``, ``category`` and, for algorithms,
        ``type`` and ``tags``); ``category='custom'`` returns
        ``algorithms`` and a ``hint`` instead of ``components``. On
        failure: ``status`` (``'error'``), ``error``, ``error_type`` and
        ``suggestion``.
    """
    from tuiml.agent.registry import get_all_tools, list_tools_by_category

    try:
        category = kwargs.get('category', 'all')
        search = kwargs.get('search')
        algo_type = kwargs.get('type')
        limit = kwargs.get('limit', 50)
        offset = kwargs.get('offset', 0)
        include_runs = bool(kwargs.get('include_runs', False))

        # category='custom', delegate to user_algorithms (absorbs tuiml_list_user_algorithms
        # and tuiml_research_log)
        if category == 'custom':
            from tuiml.agent import user_algorithms
            result = user_algorithms.research_log()
            if result.get('status') != 'success':
                return result
            algorithms = result.get('algorithms', [])
            if search:
                algorithms = [a for a in algorithms if search.lower() in a['name'].lower()]
            total = len(algorithms)
            paginated = algorithms[offset:offset + limit]
            if not include_runs:
                # Strip run details for a fast listing, keep versions + best scores
                for alg in paginated:
                    for v in alg.get('versions', []):
                        v.pop('path', None)
            return {
                'status': 'success',
                'category': 'custom',
                'total': total,
                'count': len(paginated),
                'limit': limit,
                'offset': offset,
                'has_more': (offset + limit) < total,
                'algorithms': paginated,
                'hint': "Use tuiml_train or tuiml_benchmark with any class_name or versioned_alias shown above.",
            }

        if category == 'all':
            tools = get_all_tools()
        else:
            tools = {t.name: t for t in list_tools_by_category(category)}

        # Filter by search
        if search:
            tools = {
                name: tool for name, tool in tools.items()
                if search.lower() in name.lower() or search.lower() in tool.description.lower()
            }

        # Build component list with type/tags from the component registry for algorithms
        from tuiml.registry import registry as hub_registry
        import tuiml.algorithms  # noqa: F401 - trigger registration

        components_list = []
        for t in tools.values():
            entry = {'name': t.name, 'description': t.description, 'category': t.category}

            # For algorithm tools, enrich with type and tags from the component registry
            if t.category == 'algorithm':
                # Strip prefix to get the class name
                class_name = t.name
                for prefix in ('tuiml_algorithm_',):
                    if class_name.startswith(prefix):
                        class_name = class_name[len(prefix):]
                try:
                    info = hub_registry.get_info(class_name)
                    entry['type'] = info.get('type', '')
                    entry['tags'] = info.get('tags', [])
                except (KeyError, Exception):
                    pass

            components_list.append(entry)

        # Filter by algorithm type (classifier, regressor, clusterer, anomaly, timeseries)
        if algo_type:
            if algo_type == 'anomaly':
                components_list = [
                    c for c in components_list
                    if 'anomaly-detection' in c.get('tags', [])
                ]
            elif algo_type == 'timeseries':
                components_list = [
                    c for c in components_list
                    if 'timeseries' in c.get('tags', [])
                ]
            else:
                components_list = [
                    c for c in components_list
                    if c.get('type') == algo_type
                ]

        total = len(components_list)

        # Apply pagination
        paginated = components_list[offset:offset + limit]

        # Format result
        result = {
            'status': 'success',
            'total': total,
            'count': len(paginated),
            'limit': limit,
            'offset': offset,
            'has_more': (offset + limit) < total,
            'components': paginated
        }

        return result
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e),
            'error_type': type(e).__name__,
            'suggestion': 'Check that the category parameter is valid. Use category="all" to list all components.'
        }

def execute_describe(**kwargs) -> Dict[str, Any]:
    """Describe a single component: its parameters, tags and description.

    Backs the ``tuiml_describe`` tool. Resolves the name first as an
    algorithm in the component registry, then as a built-in dataset, then
    as a preprocessing / feature / splitting component tool.

    Parameters
    ----------
    name : str
        Name of the component to describe (arrives via ``**kwargs``).

    Returns
    -------
    result : dict
        On success: ``status`` (``'success'``), ``type``, ``name``,
        ``description`` and ``parameters`` (JSON Schema); algorithms also
        include ``tags`` and ``version``, datasets include their info
        fields. On failure: ``status`` (``'error'``), ``error``,
        ``suggestion``, ``recovery_tool`` and ``recovery_params``.
    """
    try:
        name = kwargs['name']

        # 1. Try as algorithm from the component registry (covers all registered
        #    algorithms including community uploads)
        try:
            from tuiml.registry import registry as hub_registry, ComponentType
            import tuiml.algorithms  # noqa: F401 - trigger registration

            component = hub_registry.get(name)
            if component:
                schema = {}
                if hasattr(component, 'get_parameter_schema'):
                    schema = component.get_parameter_schema()

                info = {}
                if hasattr(hub_registry, 'get_info'):
                    try:
                        info = hub_registry.get_info(name)
                    except Exception:
                        pass

                return {
                    'status': 'success',
                    'type': info.get('type', 'algorithm'),
                    'name': name,
                    'description': (component.__doc__ or '').split('\n')[0].strip(),
                    'parameters': schema,
                    'tags': info.get('tags', []),
                    'version': info.get('version', ''),
                }
        except (ImportError, ValueError, KeyError):
            pass

        # 2. Try as dataset
        try:
            from tuiml.datasets.builtin import get_dataset_info
            info = get_dataset_info(name)
            return {
                'status': 'success',
                'type': 'dataset',
                'name': name,
                **info
            }
        except (ValueError, KeyError, ImportError):
            pass

        # 3. Try from component tool registry (preprocessing, features, splitting)
        from tuiml.agent.registry import get_all_tools
        tools = get_all_tools()

        for prefix in ['tuiml_preprocessing_', 'tuiml_feature_', 'tuiml_splitting_']:
            tool = tools.get(f"{prefix}{name}")
            if tool:
                return {
                    'status': 'success',
                    'type': tool.category,
                    'name': name,
                    'description': tool.description,
                    'parameters': tool.input_schema
                }

        return {
            'status': 'error',
            'error': f"Component '{name}' not found",
            'suggestion': "Use 'tuiml_search' to find components by keyword, or 'tuiml_list' to browse all components",
            'recovery_tool': 'tuiml_search',
            'recovery_params': {'query': name}
        }
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e),
            'error_type': type(e).__name__
        }

def execute_search(**kwargs) -> Dict[str, Any]:
    """Search components by keyword in their name or description.

    Backs the ``tuiml_search`` tool.

    Parameters
    ----------
    query : str
        Case-insensitive search term (arrives via ``**kwargs``, like all
        parameters below).
    category : str, default='all'
        Restrict matches to one component category.

    Returns
    -------
    result : dict
        On success: ``status`` (``'success'``), ``query``, ``count`` and
        ``results`` (list of ``{name, description, category}``), plus a
        ``suggestion`` when nothing matched. On failure: ``status``
        (``'error'``), ``error`` and ``error_type``.
    """
    from tuiml.agent.registry import get_all_tools

    try:
        query = kwargs['query'].lower()
        category = kwargs.get('category', 'all')

        tools = get_all_tools()

        matches = []
        for name, tool in tools.items():
            if category != 'all' and tool.category != category:
                continue

            if query in name.lower() or query in tool.description.lower():
                matches.append({
                    'name': tool.name,
                    'description': tool.description,
                    'category': tool.category
                })

        result = {
            'status': 'success',
            'query': kwargs['query'],
            'count': len(matches),
            'results': matches
        }

        if len(matches) == 0:
            result['suggestion'] = "No matches found. Try a broader search term or use 'tuiml_list' to browse all components"

        return result
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e),
            'error_type': type(e).__name__
        }

def execute_upload_data(**kwargs) -> Dict[str, Any]:
    """Register a dataset file or inline content for use with other tools.

    Backs the ``tuiml_upload_data`` tool. The dataset is copied into
    ``~/.tuiml/uploads/``, validated by loading it, and indexed under a
    stable ``dataset_id`` so later tools can reference it by name.

    Parameters
    ----------
    file_path : str, default=None
        Path to a data file on disk (CSV, TSV, ARFF, Parquet, JSON,
        Excel, NPY/NPZ). One of ``file_path`` / ``content`` is required
        (both arrive via ``**kwargs``, like all parameters below).
    content : str, default=None
        Inline dataset text (e.g. CSV content) written to a new file.
    name : str, default=None
        Name to register the dataset under; defaults to the file's
        basename or an auto-generated id for inline content.
    format : str, default='csv'
        File extension to use in content mode.

    Returns
    -------
    result : dict
        On success: ``status`` (``'success'``), ``dataset_id``,
        ``file_path``, ``rows``, ``features``, ``feature_names`` and
        ``message``. On failure: ``status`` (``'error'``), ``error``,
        ``error_type`` and optionally ``suggestion``.
    """
    import shutil

    try:
        src_path = kwargs.get('file_path')
        content = kwargs.get('content')

        if not src_path and not content:
            return {
                'status': 'error',
                'error': "Provide either 'file_path' (path to CSV/ARFF on disk) or 'content' (inline text).",
                'error_type': 'ValueError'
            }

        upload_dir = _UPLOADS_DIR
        os.makedirs(upload_dir, exist_ok=True)

        if src_path:
            # --- File path mode: validate and copy/link the file ---
            src_path = os.path.expanduser(src_path)
            if not os.path.isfile(src_path):
                return {
                    'status': 'error',
                    'error': f"File not found: {src_path}",
                    'error_type': 'FileNotFoundError'
                }

            ext = os.path.splitext(src_path)[1].lower()
            supported = {'.csv', '.tsv', '.arff', '.parquet', '.pq', '.json', '.jsonl', '.ndjson', '.xlsx', '.xls', '.npy', '.npz'}
            if ext not in supported:
                return {
                    'status': 'error',
                    'error': f"Unsupported file type '{ext}'. Supported: {sorted(supported)}",
                    'error_type': 'ValueError'
                }

            name = kwargs.get('name') or os.path.splitext(os.path.basename(src_path))[0]
            dest_path = os.path.join(upload_dir, f'{name}{ext}')
            shutil.copy2(src_path, dest_path)
            file_path = dest_path
        else:
            # --- Content mode: write inline text to file ---
            file_format = kwargs.get('format', 'csv')
            name = kwargs.get('name', f'uploaded_{uuid.uuid4().hex[:8]}')
            file_path = os.path.join(upload_dir, f'{name}.{file_format}')
            with open(file_path, 'w') as f:
                f.write(content)

        # Validate the file can be loaded
        try:
            dataset = _load_data(file_path)
            n_rows, n_cols = dataset.X.shape if hasattr(dataset, 'X') else (None, None)
            feature_names = list(dataset.feature_names) if hasattr(dataset, 'feature_names') and dataset.feature_names is not None else None
            dataset_id = os.path.splitext(os.path.basename(file_path))[0]
            _DATASET_INDEX[dataset_id] = file_path
            return {
                'status': 'success',
                'dataset_id': dataset_id,
                'file_path': file_path,
                'rows': n_rows,
                'features': n_cols,
                'feature_names': feature_names,
                'message': (
                    f'Dataset registered ({n_rows} rows, {n_cols} features). '
                    f'Pass data="{dataset_id}" (or the full file_path) to other tools.'
                )
            }
        except Exception as e:
            os.remove(file_path)
            return {
                'status': 'error',
                'error': f'Invalid dataset: {str(e)}',
                'error_type': type(e).__name__,
                'suggestion': 'Ensure the file is a valid CSV (with header row) or ARFF file.'
            }

    except Exception as e:
        return {
            'status': 'error',
            'error': str(e),
            'error_type': type(e).__name__
        }

def execute_save_model(**kwargs) -> Dict[str, Any]:
    """Copy a trained model to a user-specified location.

    Backs the ``tuiml_save_model`` tool.

    Parameters
    ----------
    model_id : str
        Identifier of a trained model from ``tuiml_train`` (arrives via
        ``**kwargs``, like all parameters below).
    destination : str
        Target file path to copy the serialized model to.

    Returns
    -------
    result : dict
        On success: ``status`` (``'success'``), ``model_id``, ``source``,
        ``destination`` (absolute path) and ``message``. On failure:
        ``status`` (``'error'``), ``error``, ``error_type`` and
        optionally ``suggestion``.
    """
    import shutil

    try:
        model_id = kwargs['model_id']
        destination = kwargs['destination']

        if model_id not in _MODEL_INDEX:
            return {
                'status': 'error',
                'error': f"Model '{model_id}' not found",
                'error_type': 'KeyError',
                'suggestion': 'Train a model first with tuiml_train which returns a model_id'
            }

        source = _MODEL_INDEX[model_id]
        os.makedirs(os.path.dirname(os.path.abspath(destination)) or '.', exist_ok=True)
        shutil.copy2(source, destination)

        return {
            'status': 'success',
            'model_id': model_id,
            'source': source,
            'destination': os.path.abspath(destination),
            'message': f'Model saved to {os.path.abspath(destination)}'
        }
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e),
            'error_type': type(e).__name__
        }

def execute_serve_model(**kwargs) -> Dict[str, Any]:
    """Start a REST API server to serve a trained model.

    Backs the ``tuiml_serve_model`` tool. Runs a ``tuiml.serving``
    ``ModelServer`` (FastAPI + uvicorn, installed via the
    ``tuiml[serving]`` extra) in a background thread and registers it in
    the in-process server table.

    Parameters
    ----------
    model_id : str, default=None
        Identifier of a trained model from ``tuiml_train``. One of
        ``model_id`` / ``model_path`` is required (both arrive via
        ``**kwargs``, like all parameters below).
    model_path : str, default=None
        Explicit path to a serialized model file.
    port : int, default=8000
        TCP port to bind; must be free.
    host : str, default='127.0.0.1'
        Interface to bind the server to.

    Returns
    -------
    result : dict
        On success: ``status`` (``'success'``), ``server_id``,
        ``model_id``, ``url``, ``endpoints`` (name -> URL map) and
        ``example_curl``. On failure: ``status`` (``'error'``),
        ``error``, ``error_type`` and optionally ``suggestion``.
    """
    import threading
    import socket

    try:
        model_id = kwargs.get('model_id')
        model_path = kwargs.get('model_path')
        port = kwargs.get('port', 8000)
        host = kwargs.get('host', '127.0.0.1')

        # Resolve model file path
        if model_id and model_id in _MODEL_INDEX:
            serve_path = _MODEL_INDEX[model_id]
        elif model_path and os.path.exists(model_path):
            serve_path = model_path
            model_id = model_id or os.path.splitext(os.path.basename(model_path))[0]
        else:
            return {
                'status': 'error',
                'error': 'Model not found. Provide model_id (from tuiml_train) or a valid model_path.',
                'error_type': 'ValueError',
                'suggestion': 'Train a model first with tuiml_train'
            }

        # Check port availability
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex((host, port)) == 0:
                return {
                    'status': 'error',
                    'error': f'Port {port} is already in use',
                    'error_type': 'OSError',
                    'suggestion': f'Use a different port or stop the existing server with tuiml_stop_server'
                }

        # Check dependencies
        try:
            from tuiml.serving.server import ModelServer
            import uvicorn
        except ImportError:
            return {
                'status': 'error',
                'error': 'Serving dependencies not installed',
                'error_type': 'ImportError',
                'suggestion': 'Install with: pip install "tuiml[serving]" (requires fastapi and uvicorn)'
            }

        # Create server and load model
        model_server = ModelServer()
        model_server.load_model(model_id, serve_path)
        app = model_server.create_app()

        # Configure uvicorn to run in background thread
        config = uvicorn.Config(
            app,
            host=host,
            port=port,
            log_level="warning",
        )
        server = uvicorn.Server(config)

        server_id = uuid.uuid4().hex[:8]
        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()

        # Wait briefly for server to start
        import time
        time.sleep(1.0)

        base_url = f"http://{host}:{port}"

        _SERVING_SERVERS[server_id] = {
            'server': server,
            'thread': thread,
            'model_server': model_server,
            'model_id': model_id,
            'model_path': serve_path,
            'host': host,
            'port': port,
            'url': base_url,
        }

        return {
            'status': 'success',
            'server_id': server_id,
            'model_id': model_id,
            'url': base_url,
            'endpoints': {
                'predict': f'{base_url}/predict',
                'predict_model': f'{base_url}/models/{model_id}/predict',
                'predict_proba': f'{base_url}/models/{model_id}/predict_proba',
                'health': f'{base_url}/health',
                'models': f'{base_url}/models',
                'docs': f'{base_url}/docs',
            },
            'example_curl': (
                f'curl -X POST {base_url}/predict '
                f'-H "Content-Type: application/json" '
                f'-d \'{{"features": [[5.1, 3.5, 1.4, 0.2]]}}\''
            ),
        }
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e),
            'error_type': type(e).__name__
        }


def execute_stop_server(**kwargs) -> Dict[str, Any]:
    """Stop one or all running model serving servers.

    Backs the ``tuiml_stop_server`` tool.

    Parameters
    ----------
    server_id : str, default=None
        Identifier of the server to stop (arrives via ``**kwargs``).
        When omitted, every running server is stopped.

    Returns
    -------
    result : dict
        On success: ``status`` (``'success'``) and ``message``; stopping
        all servers also returns ``stopped`` (list of
        ``{server_id, model_id, port}``). On failure: ``status``
        (``'error'``), ``error`` and optionally ``suggestion`` /
        ``error_type``.
    """
    try:
        server_id = kwargs.get('server_id')

        if server_id:
            # Stop specific server
            if server_id not in _SERVING_SERVERS:
                return {
                    'status': 'error',
                    'error': f"Server '{server_id}' not found",
                    'suggestion': 'Use tuiml_server_status to see running servers'
                }
            info = _SERVING_SERVERS.pop(server_id)
            info['server'].should_exit = True
            info['thread'].join(timeout=5)
            return {
                'status': 'success',
                'message': f"Server {server_id} stopped (was serving {info['model_id']} on port {info['port']})"
            }
        else:
            # Stop all servers
            stopped = []
            for sid, info in list(_SERVING_SERVERS.items()):
                info['server'].should_exit = True
                info['thread'].join(timeout=5)
                stopped.append({'server_id': sid, 'model_id': info['model_id'], 'port': info['port']})
            _SERVING_SERVERS.clear()
            return {
                'status': 'success',
                'message': f'Stopped {len(stopped)} server(s)',
                'stopped': stopped
            }
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e),
            'error_type': type(e).__name__
        }


def execute_server_status(**kwargs) -> Dict[str, Any]:
    """Get status of running model serving servers.

    Backs the ``tuiml_server_status`` tool. Takes no arguments (any
    ``**kwargs`` are ignored).

    Returns
    -------
    result : dict
        ``status`` (``'success'``), ``count``, and ``servers`` -- a list
        of ``{server_id, model_id, model_path, url, port, running}``.
    """
    servers = []
    for sid, info in _SERVING_SERVERS.items():
        servers.append({
            'server_id': sid,
            'model_id': info['model_id'],
            'model_path': info['model_path'],
            'url': info['url'],
            'port': info['port'],
            'running': info['thread'].is_alive(),
        })
    return {
        'status': 'success',
        'count': len(servers),
        'servers': servers
    }


def execute_plot(**kwargs) -> Dict[str, Any]:
    """Generate a visualization and return it as a saved PNG plus base64.

    Backs the ``tuiml_plot`` tool. Supported plot types: confusion_matrix,
    roc_curve, pr_curve, learning_curve, tree, feature_importance,
    cd_diagram, boxplot_comparison, heatmap, ranking_table.

    Parameters
    ----------
    plot_type : str
        Which plot to produce (arrives via ``**kwargs``, like all
        parameters below).
    title : str, default=None
        Custom plot title; each plot type has a sensible default.
    model_id : str, default=None
        Trained model to plot from (model-based plot types).
    model_path : str, default=None
        Explicit path to a serialized model file.
    data : str, default=None
        Dataset to evaluate/plot against (dataset_id, path, or built-in
        name); required by data-driven plot types.
    normalize : bool, default=False
        Normalize counts in the confusion matrix.
    algorithm : str, default=None
        Algorithm name; required for ``learning_curve``.
    benchmark_results : dict, default=None
        ``{algorithm_name: [scores...]}`` mapping; required for the
        comparison plots (cd_diagram, boxplot_comparison, heatmap,
        ranking_table).

    Returns
    -------
    result : dict
        On success: ``status`` (``'success'``), ``plot_type``,
        ``description``, ``path`` (saved PNG on disk), ``_image_base64``
        and ``_image_mime``. On failure: ``status`` (``'error'``),
        ``error`` and optionally ``error_type`` / ``suggestion``.
    """
    import base64
    import numpy as np

    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt

        plot_type = kwargs['plot_type']
        title = kwargs.get('title')

        # Save plot to a persistent, discoverable directory so the AI can
        # reference the file path in markdown reports (in addition to seeing
        # the inline image). Override via $TUIML_PLOT_DIR.
        from pathlib import Path
        plot_dir = Path(os.environ.get('TUIML_PLOT_DIR',
                                       str(Path.home() / '.tuiml' / 'plots')))
        plot_dir.mkdir(parents=True, exist_ok=True)
        plot_path = str(plot_dir / f'{plot_type}_{uuid.uuid4().hex[:8]}.png')

        if plot_type == 'confusion_matrix':
            from tuiml.evaluation.visualization import plot_confusion_matrix
            import tuiml

            model = _load_model_from_disk(kwargs.get('model_id'), kwargs.get('model_path'))
            if model is None:
                return {'status': 'error', 'error': 'Model not found. Provide model_id or model_path.'}

            dataset = _load_data(kwargs['data'])
            predictions = model.predict(dataset.X)
            plot_confusion_matrix(
                dataset.y, predictions,
                title=title or 'Confusion Matrix',
                save_path=plot_path,
                normalize=kwargs.get('normalize', False),
            )
            description = 'Confusion matrix showing predicted vs actual class labels.'

        elif plot_type == 'roc_curve':
            from tuiml.evaluation.visualization import plot_roc_curve
            import tuiml

            model = _load_model_from_disk(kwargs.get('model_id'), kwargs.get('model_path'))
            if model is None:
                return {'status': 'error', 'error': 'Model not found. Provide model_id or model_path.'}

            dataset = _load_data(kwargs['data'])

            if not hasattr(model, 'predict_proba'):
                return {
                    'status': 'error',
                    'error': 'Model does not support predict_proba, required for ROC curve.',
                    'suggestion': 'Use a classifier that supports probability estimates (e.g., RandomForestClassifier, NaiveBayesClassifier).'
                }
            probas = model.predict_proba(dataset.X)
            # Multiclass (>2 columns): pass the full proba matrix + class
            # labels so plot_roc_curve draws per-class OvR curves + macro avg.
            # Binary (1- or 2-D): pass positive-class probabilities.
            classes = getattr(model, 'classes_', None)
            if probas.ndim == 2 and probas.shape[1] > 2:
                y_score = probas
                desc_suffix = ' (one-vs-rest per class, plus macro-average)'
            elif probas.ndim == 2:
                y_score = probas[:, 1]
                desc_suffix = ''
            else:
                y_score = probas
                desc_suffix = ''

            plot_roc_curve(
                dataset.y, y_score,
                title=title or 'ROC Curve',
                save_path=plot_path,
                classes=list(classes) if classes is not None else None,
            )
            description = f'ROC curve with AUC score.{desc_suffix}'

        elif plot_type == 'pr_curve':
            from tuiml.evaluation.visualization import plot_pr_curve
            import tuiml

            model = _load_model_from_disk(kwargs.get('model_id'), kwargs.get('model_path'))
            if model is None:
                return {'status': 'error', 'error': 'Model not found. Provide model_id or model_path.'}

            dataset = _load_data(kwargs['data'])

            if not hasattr(model, 'predict_proba'):
                return {
                    'status': 'error',
                    'error': 'Model does not support predict_proba, required for PR curve.',
                    'suggestion': 'Use a classifier that supports probability estimates.'
                }
            probas = model.predict_proba(dataset.X)
            if probas.ndim == 2:
                y_score = probas[:, 1]
            else:
                y_score = probas

            plot_pr_curve(
                dataset.y, y_score,
                title=title or 'Precision-Recall Curve',
                save_path=plot_path,
            )
            description = 'Precision-Recall curve with Average Precision score.'

        elif plot_type == 'learning_curve':
            from tuiml.evaluation.visualization import plot_learning_curve
            from tuiml.registry import registry
            from tuiml.evaluation.metrics import accuracy_score
            import tuiml.algorithms  # noqa: F401 - trigger registration

            algorithm_name = kwargs.get('algorithm')
            if not algorithm_name:
                return {'status': 'error', 'error': 'algorithm parameter is required for learning_curve.'}

            dataset = _load_data(kwargs['data'])

            algo_cls = registry.get(algorithm_name)
            if algo_cls is None:
                return {'status': 'error', 'error': f"Algorithm '{algorithm_name}' not found."}

            # Compute learning curve manually with k-fold CV
            from tuiml.evaluation.splitting import KFold
            n_splits = 5
            train_fractions = np.linspace(0.1, 1.0, 10)
            n_samples = len(dataset.y)
            from tuiml.utils.seed import get_global_seed
            seed = get_global_seed()
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed if seed is not None else 42)

            all_train_sizes = []
            all_train_scores = []  # shape: (n_sizes, n_splits)
            all_test_scores = []

            for frac in train_fractions:
                fold_train_scores = []
                fold_test_scores = []
                for train_idx, test_idx in kf.split(dataset.X):
                    X_train_full, y_train_full = dataset.X[train_idx], dataset.y[train_idx]
                    X_test, y_test = dataset.X[test_idx], dataset.y[test_idx]

                    # Subsample training set
                    subset_size = max(2, int(len(X_train_full) * frac))
                    X_train = X_train_full[:subset_size]
                    y_train = y_train_full[:subset_size]

                    model_lc = algo_cls()
                    model_lc.fit(X_train, y_train)

                    train_pred = model_lc.predict(X_train)
                    test_pred = model_lc.predict(X_test)
                    fold_train_scores.append(accuracy_score(y_train, train_pred))
                    fold_test_scores.append(accuracy_score(y_test, test_pred))

                # Use the actual training size for the first fold as representative
                all_train_sizes.append(max(2, int(len(dataset.X) * (n_splits - 1) / n_splits * frac)))
                all_train_scores.append(fold_train_scores)
                all_test_scores.append(fold_test_scores)

            train_sizes_arr = np.array(all_train_sizes)
            train_scores_arr = np.array(all_train_scores)
            test_scores_arr = np.array(all_test_scores)

            plot_learning_curve(
                train_sizes_arr,
                train_scores_arr,
                test_scores_arr,
                title=title or 'Learning Curve',
                save_path=plot_path,
                metric_name='Accuracy',
            )
            description = f'Learning curve for {algorithm_name} showing training vs validation accuracy.'

        elif plot_type == 'tree':
            from tuiml.evaluation.visualization import plot_tree

            model = _load_model_from_disk(kwargs.get('model_id'), kwargs.get('model_path'))
            if model is None:
                return {'status': 'error', 'error': 'Model not found. Provide model_id or model_path.'}

            # Get feature names from the model if available
            feature_names = None
            if hasattr(model, 'feature_names_'):
                feature_names = model.feature_names_

            plot_tree(
                model,
                feature_names=feature_names,
                filled=True,
                rounded=True,
                title=title or 'Decision Tree',
                save_path=plot_path,
            )
            description = 'Decision tree structure visualization.'

        elif plot_type == 'feature_importance':
            model = _load_model_from_disk(kwargs.get('model_id'), kwargs.get('model_path'))
            if model is None:
                return {'status': 'error', 'error': 'Model not found. Provide model_id or model_path.'}

            importances = None

            # Try direct attribute first
            if hasattr(model, 'feature_importances_') and model.feature_importances_ is not None:
                importances = np.array(model.feature_importances_)

            # Try wrapped inner model (e.g., XGBoost, GradientBoosting store
            # the sklearn-compatible model in model.model_)
            if importances is None and hasattr(model, 'model_'):
                inner = model.model_
                if hasattr(inner, 'feature_importances_') and inner.feature_importances_ is not None:
                    importances = np.array(inner.feature_importances_)

            # Try coef_ for linear models (Logistic Regression, SVM, etc.)
            if importances is None:
                coef = getattr(model, 'coef_', None)
                if coef is None and hasattr(model, 'model_'):
                    coef = getattr(model.model_, 'coef_', None)
                if coef is not None:
                    coef = np.array(coef)
                    if coef.ndim > 1:
                        importances = np.mean(np.abs(coef), axis=0)
                    else:
                        importances = np.abs(coef)

            def _count_feature_usage(node, counts):
                """Recursively count feature usage in a tree node."""
                if node is None or getattr(node, 'is_leaf', True):
                    return
                feat = getattr(node, 'feature_index', None)
                if feat is not None and feat >= 0:
                    samples = getattr(node, 'n_samples', 1)
                    counts[feat] += samples
                _count_feature_usage(getattr(node, 'left', None), counts)
                _count_feature_usage(getattr(node, 'right', None), counts)

            # For ensemble models, compute from estimators' trees
            if importances is None and hasattr(model, 'estimators_') and model.estimators_:
                n_features = getattr(model, 'n_features_', None)
                if n_features and all(hasattr(est, 'tree_') for est in model.estimators_):
                    total = np.zeros(n_features)
                    for est in model.estimators_:
                        _count_feature_usage(est.tree_, total)
                    if total.sum() > 0:
                        importances = total / total.sum()

            # For single tree models
            if importances is None and hasattr(model, 'tree_'):
                n_features = getattr(model, 'n_features_', None)
                if n_features:
                    total = np.zeros(n_features)
                    _count_feature_usage(model.tree_, total)
                    if total.sum() > 0:
                        importances = total / total.sum()

            if importances is None:
                return {
                    'status': 'error',
                    'error': 'Cannot compute feature importances from this model.',
                    'suggestion': 'Use a tree-based model (e.g., RandomForestClassifier, XGBoostClassifier, C45TreeClassifier).'
                }

            importances = np.array(importances)
            feature_names = None
            if hasattr(model, 'feature_names_'):
                feature_names = model.feature_names_
            if feature_names is None and hasattr(model, 'model_'):
                inner = model.model_
                if hasattr(inner, 'feature_names_in_'):
                    feature_names = list(inner.feature_names_in_)
            if feature_names is None:
                feature_names = [f'Feature {i}' for i in range(len(importances))]

            # Sort by importance
            indices = np.argsort(importances)[::-1]
            sorted_names = [feature_names[i] for i in indices]
            sorted_importances = importances[indices]

            from tuiml.evaluation.visualization import setup_figure, style_axis, get_colors
            fig, ax = setup_figure(figsize=(10, max(6, len(sorted_names) * 0.35)))
            colors = get_colors(len(sorted_names))

            ax.barh(range(len(sorted_names)), sorted_importances[::-1], color=colors[0])
            ax.set_yticks(range(len(sorted_names)))
            ax.set_yticklabels(sorted_names[::-1])
            style_axis(
                ax,
                title=title or 'Feature Importance',
                xlabel='Importance',
                ylabel=None,
                legend=False,
            )
            fig.tight_layout()
            fig.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            description = 'Feature importance bar chart from model.'

        elif plot_type in ('cd_diagram', 'boxplot_comparison', 'heatmap', 'ranking_table'):
            benchmark_results = kwargs.get('benchmark_results')
            if not benchmark_results:
                return {
                    'status': 'error',
                    'error': f"'{plot_type}' requires benchmark_results parameter with algorithm CV scores.",
                    'suggestion': "Provide benchmark_results: { 'AlgoName': [score1, score2, ...], ... }"
                }

            scores_dict = {
                name: np.array(scores) for name, scores in benchmark_results.items()
            }

            if plot_type == 'cd_diagram':
                from tuiml.evaluation.visualization import plot_critical_difference
                plot_critical_difference(
                    scores=scores_dict,
                    title=title or 'Critical Difference Diagram',
                    save_path=plot_path,
                )
                description = 'Critical difference diagram showing statistically significant differences between algorithms.'

            elif plot_type == 'boxplot_comparison':
                from tuiml.evaluation.visualization import plot_boxplot_comparison
                plot_boxplot_comparison(
                    scores=scores_dict,
                    save_path=plot_path,
                )
                description = 'Box plot comparison of algorithm cross-validation scores.'

            elif plot_type == 'heatmap':
                from tuiml.evaluation.visualization import plot_heatmap
                plot_heatmap(
                    scores=scores_dict,
                    save_path=plot_path,
                )
                description = 'Heatmap of algorithm scores across datasets.'

            elif plot_type == 'ranking_table':
                from tuiml.evaluation.visualization import plot_ranking_table
                plot_ranking_table(
                    scores=scores_dict,
                    title=title or 'Algorithm Ranking',
                    save_path=plot_path,
                )
                description = 'Ranking table of algorithm performance.'

        else:
            return {'status': 'error', 'error': f"Unknown plot_type: '{plot_type}'"}

        # Close any remaining figures to free memory
        plt.close('all')

        # Read the saved plot and base64 encode it (so the AI can see it
        # inline). We keep the file on disk so the AI can also embed it
        # in markdown reports via the returned `path`.
        with open(plot_path, 'rb') as f:
            image_bytes = f.read()
        image_b64 = base64.b64encode(image_bytes).decode('utf-8')

        return {
            'status': 'success',
            'plot_type': plot_type,
            'description': description,
            'path': plot_path,
            '_image_base64': image_b64,
            '_image_mime': 'image/png',
        }

    except Exception as e:
        plt.close('all')
        return {
            'status': 'error',
            'error': str(e),
            'error_type': type(e).__name__
        }


def execute_data_profile(**kwargs) -> Dict[str, Any]:
    """Profile a dataset: shape, dtypes, missing values, stats, class distribution.

    Backs the ``tuiml_profile_data`` tool.

    Parameters
    ----------
    data : str
        Dataset to profile: dataset_id, file path, or built-in name
        (arrives via ``**kwargs``, like all parameters below).
    target : str, default=None
        Target column name, echoed back as ``target_column`` when the
        dataset has labels.

    Returns
    -------
    result : dict
        On success: ``status`` (``'success'``), ``name``, ``shape``,
        ``n_samples``, ``n_features``, ``feature_names``, ``dtypes``
        (feature -> 'numeric'/'categorical'), ``missing_values``,
        ``numeric_stats`` (per-feature mean/std/min/max/median) and,
        when labels exist, ``class_distribution``. On failure: ``status``
        (``'error'``), ``error`` and ``error_type``.
    """
    import numpy as np

    try:
        dataset = _load_data(kwargs['data'])
        X = np.asarray(dataset.X)
        y = dataset.y
        feature_names = list(dataset.feature_names) if hasattr(dataset, 'feature_names') and dataset.feature_names is not None else [f'feature_{i}' for i in range(X.shape[1])]

        result = {
            'status': 'success',
            'name': kwargs['data'],
            'shape': list(X.shape),
            'n_samples': int(X.shape[0]),
            'n_features': int(X.shape[1]),
            'feature_names': feature_names,
        }

        # Dtypes
        dtypes = {}
        for i, name in enumerate(feature_names):
            col = X[:, i]
            try:
                col.astype(float)
                dtypes[name] = 'numeric'
            except (ValueError, TypeError):
                dtypes[name] = 'categorical'
        result['dtypes'] = dtypes

        # Missing values
        missing = {}
        for i, name in enumerate(feature_names):
            col = X[:, i]
            n_missing = int(np.sum(np.isnan(col))) if np.issubdtype(col.dtype, np.number) else 0
            if n_missing > 0:
                missing[name] = n_missing
        result['missing_values'] = missing

        # Numeric stats
        numeric_stats = {}
        for i, name in enumerate(feature_names):
            if dtypes.get(name) == 'numeric':
                col = X[:, i].astype(float)
                valid = col[~np.isnan(col)]
                if len(valid) > 0:
                    numeric_stats[name] = {
                        'mean': float(np.mean(valid)),
                        'std': float(np.std(valid)),
                        'min': float(np.min(valid)),
                        'max': float(np.max(valid)),
                        'median': float(np.median(valid)),
                    }
        result['numeric_stats'] = numeric_stats

        # Class distribution (if target provided)
        target_col = kwargs.get('target')
        if y is not None:
            y_arr = np.asarray(y)
            unique, counts = np.unique(y_arr, return_counts=True)
            result['class_distribution'] = {str(u): int(c) for u, c in zip(unique, counts)}
            if target_col:
                result['target_column'] = target_col

        return result
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e),
            'error_type': type(e).__name__
        }


def execute_generate_data(**kwargs) -> Dict[str, Any]:
    """Generate synthetic data using a generator class.

    Backs the ``tuiml_generate_data`` tool. The generated dataset is
    written to a temporary CSV whose path can be passed to other tools.

    Parameters
    ----------
    generator : str
        Generator class name, one of: RandomRBF, Agrawal, LED,
        Hyperplane, Friedman, MexicanHat, Sine, Blobs, Moons, Circles,
        SwissRoll (arrives via ``**kwargs``, like all parameters below).
    n_samples : int, default=None
        Number of samples to generate.
    n_features : int, default=None
        Number of features (generator dependent).
    n_classes : int, default=None
        Number of classes (classification generators).
    n_clusters : int, default=None
        Number of clusters (clustering generators).
    noise : float, default=None
        Noise level (generator dependent).
    random_seed : int, default=None
        Random seed; mapped to the generator's ``random_state``.
    generator_params : dict, default=None
        Additional constructor parameters merged on top.

    Returns
    -------
    result : dict
        On success: ``status`` (``'success'``), ``generator``,
        ``file_path`` (CSV on disk), ``shape``, ``feature_names``,
        ``preview`` (first 5 rows of up to 6 columns) and optionally
        ``target_names``. On failure: ``status`` (``'error'``),
        ``error`` and optionally ``suggestion`` / ``error_type``.
    """
    import numpy as np

    try:
        generator_name = kwargs['generator']

        from tuiml.datasets.generators import (
            RandomRBF, Agrawal, LED, Hyperplane,
            Friedman, MexicanHat, Sine,
            Blobs, Moons, Circles, SwissRoll,
        )

        generators = {
            'RandomRBF': RandomRBF, 'Agrawal': Agrawal, 'LED': LED, 'Hyperplane': Hyperplane,
            'Friedman': Friedman, 'MexicanHat': MexicanHat, 'Sine': Sine,
            'Blobs': Blobs, 'Moons': Moons, 'Circles': Circles, 'SwissRoll': SwissRoll,
        }

        gen_cls = generators.get(generator_name)
        if gen_cls is None:
            return {
                'status': 'error',
                'error': f"Generator '{generator_name}' not found.",
                'suggestion': f"Available generators: {list(generators.keys())}"
            }

        # Build constructor params
        params = {}
        extra_params = kwargs.get('generator_params', {})
        if 'random_seed' in kwargs:
            kwargs['random_state'] = kwargs.pop('random_seed')
            
        for key in ('n_samples', 'n_features', 'n_classes', 'n_clusters', 'noise', 'random_state'):
            if key in kwargs and kwargs[key] is not None:
                params[key] = kwargs[key]
        params.update(extra_params)

        gen = gen_cls(**params)
        data = gen.generate()

        # Save to CSV temp file
        upload_dir = os.path.join(tempfile.gettempdir(), 'tuiml_generated')
        os.makedirs(upload_dir, exist_ok=True)
        file_path = os.path.join(upload_dir, f'{generator_name.lower()}_{uuid.uuid4().hex[:8]}.csv')

        import pandas as pd
        feature_names = list(data.feature_names) if data.feature_names else [f'x{i}' for i in range(data.X.shape[1])]
        df = pd.DataFrame(data.X, columns=feature_names)
        if data.y is not None:
            df['target'] = data.y
        df.to_csv(file_path, index=False)

        # Preview: first 5 rows
        preview = {col: df[col].head(5).tolist() for col in df.columns[:6]}

        result = {
            'status': 'success',
            'generator': generator_name,
            'file_path': file_path,
            'shape': [int(data.X.shape[0]), int(data.X.shape[1])],
            'feature_names': feature_names,
            'preview': preview,
        }
        if data.target_names:
            result['target_names'] = list(data.target_names)

        return result
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e),
            'error_type': type(e).__name__
        }


def execute_preprocess(**kwargs) -> Dict[str, Any]:
    """Apply preprocessing steps or a specific atomic stage to a dataset.

    Backs the ``tuiml_preprocess`` tool. Two modes: ``steps`` runs an
    ordered pipeline of named preprocessors; ``stage`` runs one atomic
    operation (``split``, ``impute``, ``balance``, ``scale``, ``encode``,
    ``discretize``) with a default class per stage. Output is written to
    CSV so downstream tools can consume it by path.

    Parameters
    ----------
    data : str
        Dataset to preprocess: dataset_id, file path, or built-in name
        (arrives via ``**kwargs``, like all parameters below).
    steps : list, default=None
        Pipeline mode: preprocessor steps, each a class name or
        ``{"name", **params}`` dict. One of ``steps`` / ``stage`` is
        required.
    stage : str, default=None
        Atomic stage mode: ``'split'``, ``'impute'``, ``'balance'``,
        ``'scale'``, ``'encode'``, or ``'discretize'``.
    stage_kwargs : dict, default=None
        Stage options, e.g. ``method`` to pick a specific class, or for
        ``split``: ``n_splits``/``kfold``, ``test_size``, ``train_size``,
        ``shuffle``, ``stratify``, ``random_seed``.
    output : str, default=None
        Output file or directory (alias: ``save_as``); defaults to a
        temp location.
    save_as : str, default=None
        Alias for ``output``.
    target : str, default='target'
        Column name for the label in the output CSV.
    random_seed : int, default=None
        Random seed forwarded to seed-aware stages.

    Returns
    -------
    result : dict
        On success: ``status`` (``'success'``), ``original_shape``, and
        either ``file_path``, ``new_shape`` and ``steps_applied``, or --
        for ``stage='split'`` -- ``split_type`` (``'kfold'`` /
        ``'holdout'``) and ``files`` (train/test CSV paths, per fold for
        k-fold). On failure: ``status`` (``'error'``), ``error`` and
        optionally ``error_type`` / ``suggestion``.
    """
    import numpy as np
    import pandas as pd

    try:
        dataset = _load_data(kwargs['data'])
        X = np.asarray(dataset.X, dtype=float)
        y = dataset.y
        original_shape = list(X.shape)
        feature_names = list(dataset.feature_names) if hasattr(dataset, 'feature_names') and dataset.feature_names is not None else [f'feature_{i}' for i in range(X.shape[1])]

        stage = kwargs.get('stage')
        steps = kwargs.get('steps')

        if stage is None and steps is None:
            return {
                'status': 'error',
                'error': "Either 'steps' or 'stage' must be specified for preprocessing."
            }

        if stage is not None:
            # Atomic stage execution
            stage = stage.strip().lower()
            if stage == 'split':
                from pathlib import Path
                # Get splitting arguments from stage_kwargs
                stage_kwargs = kwargs.get('stage_kwargs') or {}
                # Support common aliases
                n_splits = stage_kwargs.get('kfold') or stage_kwargs.get('n_splits')
                test_size = stage_kwargs.get('test_size')
                train_size = stage_kwargs.get('train_size')
                shuffle = stage_kwargs.get('shuffle', True)
                stratify_flag = stage_kwargs.get('stratify', False)
                # Use random_seed or random_state
                seed = stage_kwargs.get('random_seed') or stage_kwargs.get('random_state') or kwargs.get('random_seed')
                
                # Determine output location
                output = kwargs.get('output') or kwargs.get('save_as')
                if output:
                    output_path = Path(output)
                    if output_path.suffix == '' or output_path.is_dir():
                        os.makedirs(output_path, exist_ok=True)
                        prefix = ""
                        out_dir = output_path
                    else:
                        os.makedirs(output_path.parent, exist_ok=True)
                        prefix = output_path.name.rsplit('.', 1)[0] + "_"
                        out_dir = output_path.parent
                else:
                    import tempfile
                    out_dir = Path(tempfile.mkdtemp(prefix='tuiml_split_'))
                    prefix = ""

                out_cols = feature_names[:X.shape[1]] if len(feature_names) >= X.shape[1] else [f'feature_{i}' for i in range(X.shape[1])]
                
                # If n_splits is provided, we perform K-Fold / StratifiedKFold split
                if n_splits is not None:
                    n_splits = int(n_splits)
                    from tuiml.evaluation.splitting import KFold, StratifiedKFold
                    if stratify_flag and y is not None:
                        splitter = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=seed)
                    else:
                        splitter = KFold(n_splits=n_splits, shuffle=shuffle, random_state=seed)
                    
                    split_files = {}
                    for i, (train_idx, test_idx) in enumerate(splitter.split(X, y)):
                        X_train, X_test = X[train_idx], X[test_idx]
                        
                        df_train = pd.DataFrame(X_train, columns=out_cols)
                        df_test = pd.DataFrame(X_test, columns=out_cols)
                        
                        if y is not None:
                            target_name = kwargs.get('target', 'target')
                            df_train[target_name] = y[train_idx]
                            df_test[target_name] = y[test_idx]
                        
                        train_path = out_dir / f"{prefix}train_{i}.csv"
                        test_path = out_dir / f"{prefix}test_{i}.csv"
                        
                        df_train.to_csv(train_path, index=False)
                        df_test.to_csv(test_path, index=False)
                        
                        split_files[f"fold_{i}"] = {
                            "train": str(train_path),
                            "test": str(test_path)
                        }
                    
                    return {
                        'status': 'success',
                        'stage': 'split',
                        'split_type': 'kfold',
                        'n_splits': n_splits,
                        'files': split_files,
                        'original_shape': original_shape
                    }
                else:
                    # Simple holdout split (train_test_split)
                    from tuiml.evaluation.splitting import train_test_split
                    
                    # Prepare arguments for train_test_split
                    split_args = [X]
                    if y is not None:
                        split_args.append(y)
                    
                    stratify_arr = y if (stratify_flag and y is not None) else None
                    
                    splits = train_test_split(
                        *split_args,
                        test_size=test_size,
                        train_size=train_size,
                        shuffle=shuffle,
                        stratify=stratify_arr,
                        random_state=seed
                    )
                    
                    if y is not None:
                        X_train, X_test, y_train, y_test = splits
                    else:
                        X_train, X_test = splits
                        y_train, y_test = None, None
                        
                    df_train = pd.DataFrame(X_train, columns=out_cols)
                    df_test = pd.DataFrame(X_test, columns=out_cols)
                    
                    if y is not None:
                        target_name = kwargs.get('target', 'target')
                        df_train[target_name] = y_train
                        df_test[target_name] = y_test
                        
                    train_path = out_dir / f"{prefix}train.csv"
                    test_path = out_dir / f"{prefix}test.csv"
                    
                    df_train.to_csv(train_path, index=False)
                    df_test.to_csv(test_path, index=False)
                    
                    return {
                        'status': 'success',
                        'stage': 'split',
                        'split_type': 'holdout',
                        'files': {
                            'train': str(train_path),
                            'test': str(test_path)
                        },
                        'original_shape': original_shape,
                        'train_shape': list(X_train.shape),
                        'test_shape': list(X_test.shape)
                    }

            else:
                # Other atomic preprocessing stages
                import tuiml.preprocessing as pp_module
                
                stage_kwargs = kwargs.get('stage_kwargs') or {}
                method = stage_kwargs.get('method')
                
                # Filter out params that shouldn't be passed directly to initialization
                estimator_params = {k: v for k, v in stage_kwargs.items() if k != 'method'}
                
                # Check for random_seed / random_state / seed and inject if appropriate
                seed = stage_kwargs.get('random_seed') or stage_kwargs.get('random_state') or kwargs.get('random_seed')
                
                # Map stage to default class and suffix
                if stage == 'impute':
                    class_name = 'SimpleImputer'
                    if method:
                        if method.lower() in ('knn', 'knnimputer'):
                            class_name = 'KNNImputer'
                        elif method.lower() in ('simple', 'simpleimputer'):
                            class_name = 'SimpleImputer'
                        else:
                            class_name = method
                elif stage == 'balance':
                    class_name = 'SMOTESampler'
                    if method:
                        class_name = method
                elif stage == 'scale':
                    class_name = 'StandardScaler'
                    if method:
                        class_name = method
                elif stage == 'encode':
                    class_name = 'OneHotEncoder'
                    if method:
                        class_name = method
                elif stage == 'discretize':
                    class_name = 'EqualWidthDiscretizer'
                    if method:
                        class_name = method
                else:
                    return {
                        'status': 'error',
                        'error': f"Unknown preprocessing stage: '{stage}'"
                    }
                
                # Helper to perform case-insensitive attribute lookup
                def resolve_class_name(name):
                    """Resolve a preprocessor class name case-insensitively in tuiml.preprocessing.

                    Parameters
                    ----------
                    name : str
                        Candidate class name to look up.

                    Returns
                    -------
                    resolved : str or None
                        The exact attribute name in ``tuiml.preprocessing``,
                        or None if no match exists.
                    """
                    if hasattr(pp_module, name):
                        return name
                    for attr in dir(pp_module):
                        if attr.lower() == name.lower():
                            return attr
                    return None
                
                resolved_name = resolve_class_name(class_name)
                if not resolved_name:
                    # Try matching with suffix based on stage
                    suffix = ""
                    if stage == 'balance': suffix = 'sampler'
                    elif stage == 'scale': suffix = 'scaler'
                    elif stage == 'encode': suffix = 'encoder'
                    elif stage == 'discretize': suffix = 'discretizer'
                    elif stage == 'impute': suffix = 'imputer'
                    
                    if suffix and not class_name.lower().endswith(suffix):
                        resolved_name = resolve_class_name(class_name + suffix)
                
                if resolved_name:
                    preprocessor_cls = getattr(pp_module, resolved_name)
                    class_name = resolved_name
                else:
                    # Fallback to registry lookup
                    from tuiml.registry import registry
                    try:
                        preprocessor_cls = registry.get(class_name)
                    except Exception:
                        preprocessor_cls = None
                
                if preprocessor_cls is None:
                    return {
                        'status': 'error',
                        'error': f"Preprocessor class '{class_name}' for stage '{stage}' not found."
                    }
                
                # If random_seed is supported, inject it
                import inspect
                init_sig = inspect.signature(preprocessor_cls.__init__)
                if seed is not None:
                    if 'random_state' in init_sig.parameters and 'random_state' not in estimator_params:
                        estimator_params['random_state'] = seed
                    elif 'random_seed' in init_sig.parameters and 'random_seed' not in estimator_params:
                        estimator_params['random_seed'] = seed
                
                preprocessor = preprocessor_cls(**estimator_params)
                
                if hasattr(preprocessor, 'fit_resample') and y is not None:
                    X, y = preprocessor.fit_resample(X, y)
                else:
                    from tuiml.base.preprocessing import InstanceTransformer
                    if isinstance(preprocessor, InstanceTransformer):
                        result = preprocessor.fit_transform(X, y)
                        X, y = result[0], result[1]
                    else:
                        from tuiml.base.preprocessing import SupervisedTransformer
                        if isinstance(preprocessor, SupervisedTransformer) and y is not None:
                            X = preprocessor.fit_transform(X, y)
                        else:
                            X = preprocessor.fit_transform(X)
                
                # Save result
                output = kwargs.get('output') or kwargs.get('save_as')
                if output:
                    from pathlib import Path
                    output_path = Path(output)
                    if output_path.is_dir() or output.endswith('/') or output.endswith('\\'):
                        os.makedirs(output_path, exist_ok=True)
                        file_path = str(output_path / f"preprocessed_{uuid.uuid4().hex[:8]}.csv")
                    else:
                        os.makedirs(output_path.parent, exist_ok=True)
                        file_path = str(output_path)
                else:
                    import tempfile
                    upload_dir = os.path.join(tempfile.gettempdir(), 'tuiml_preprocessed')
                    os.makedirs(upload_dir, exist_ok=True)
                    file_path = os.path.join(upload_dir, f'preprocessed_{uuid.uuid4().hex[:8]}.csv')
                
                out_cols = feature_names[:X.shape[1]] if len(feature_names) >= X.shape[1] else [f'feature_{i}' for i in range(X.shape[1])]
                df = pd.DataFrame(X, columns=out_cols)
                if y is not None:
                    target_name = kwargs.get('target', 'target')
                    df[target_name] = y
                df.to_csv(file_path, index=False)
                
                return {
                    'status': 'success',
                    'stage': stage,
                    'file_path': file_path,
                    'original_shape': original_shape,
                    'new_shape': list(X.shape),
                    'steps_applied': [class_name],
                }

        else:
            # Standard step-by-step pipeline execution
            steps_applied = []
            from tuiml.registry import registry
            import tuiml.preprocessing  # noqa: F401 - trigger registration

            for step in steps:
                if isinstance(step, str):
                    name, params = step, {}
                elif isinstance(step, dict):
                    name = step.get('name')
                    params = {k: v for k, v in step.items() if k != 'name'}
                else:
                    continue

                # Resolve preprocessor class
                preprocessor_cls = None
                try:
                    preprocessor_cls = registry.get(name)
                except (KeyError, Exception):
                    pass

                if preprocessor_cls is None:
                    # Fallback: try direct import
                    try:
                        from tuiml import preprocessing as pp_module
                        preprocessor_cls = getattr(pp_module, name, None)
                    except ImportError:
                        pass

                if preprocessor_cls is None:
                    return {
                        'status': 'error',
                        'error': f"Preprocessor '{name}' not found.",
                        'suggestion': "Use tuiml_list with category='preprocessing' to see available preprocessors."
                    }

                preprocessor = preprocessor_cls(**params)
                if hasattr(preprocessor, 'fit_resample') and y is not None:
                    X, y = preprocessor.fit_resample(X, y)
                else:
                    from tuiml.base.preprocessing import InstanceTransformer
                    if isinstance(preprocessor, InstanceTransformer):
                        result = preprocessor.fit_transform(X, y)
                        X, y = result[0], result[1]
                    else:
                        X = preprocessor.fit_transform(X)

                steps_applied.append(name)

            # Save result to CSV
            save_as = kwargs.get('save_as') or kwargs.get('output')
            if save_as:
                file_path = save_as
                os.makedirs(os.path.dirname(os.path.abspath(save_as)) or '.', exist_ok=True)
            else:
                import tempfile
                upload_dir = os.path.join(tempfile.gettempdir(), 'tuiml_preprocessed')
                os.makedirs(upload_dir, exist_ok=True)
                file_path = os.path.join(upload_dir, f'preprocessed_{uuid.uuid4().hex[:8]}.csv')

            # Build output DataFrame
            out_cols = feature_names[:X.shape[1]] if len(feature_names) >= X.shape[1] else [f'feature_{i}' for i in range(X.shape[1])]
            df = pd.DataFrame(X, columns=out_cols)
            if y is not None:
                target_name = kwargs.get('target', 'target')
                df[target_name] = y
            df.to_csv(file_path, index=False)

            return {
                'status': 'success',
                'file_path': file_path,
                'original_shape': original_shape,
                'new_shape': list(X.shape),
                'steps_applied': steps_applied,
            }
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e),
            'error_type': type(e).__name__
        }


def execute_select_features(**kwargs) -> Dict[str, Any]:
    """Run feature selection on a dataset.

    Backs the ``tuiml_select_features`` tool. The reduced dataset is
    written to a temporary CSV for use by downstream tools.

    Parameters
    ----------
    data : str
        Dataset to select features from: dataset_id, file path, or
        built-in name (arrives via ``**kwargs``, like all parameters
        below).
    method : str
        Selector class name: SelectKBestSelector,
        SelectPercentileSelector, VarianceThresholdSelector, CFSSelector,
        WrapperSelector, SelectFprSelector, or SelectThresholdSelector.
    method_params : dict, default=None
        Extra constructor parameters for the selector.
    k : int, default=None
        Number of features to keep (selectors that accept ``k``).
    threshold : float, default=None
        Threshold value (selectors that accept ``threshold``).
    target : str, default='target'
        Column name for the label in the output CSV.

    Returns
    -------
    result : dict
        On success: ``status`` (``'success'``), ``method``,
        ``n_original``, ``n_selected``, ``selected_features``,
        ``file_path`` (reduced CSV) and, when available, ``scores``
        (feature -> score). On failure: ``status`` (``'error'``),
        ``error`` and optionally ``suggestion`` / ``error_type``.
    """
    import numpy as np

    try:
        dataset = _load_data(kwargs['data'])
        X = np.asarray(dataset.X, dtype=float)
        y = np.asarray(dataset.y) if dataset.y is not None else None
        feature_names = list(dataset.feature_names) if hasattr(dataset, 'feature_names') and dataset.feature_names is not None else [f'feature_{i}' for i in range(X.shape[1])]

        method_name = kwargs['method']

        from tuiml.features.selection import (
            SelectKBestSelector, SelectPercentileSelector,
            VarianceThresholdSelector, CFSSelector,
            WrapperSelector, SelectFprSelector, SelectThresholdSelector
        )

        selectors = {
            'SelectKBestSelector': SelectKBestSelector,
            'SelectPercentileSelector': SelectPercentileSelector,
            'VarianceThresholdSelector': VarianceThresholdSelector,
            'CFSSelector': CFSSelector,
            'WrapperSelector': WrapperSelector,
            'SelectFprSelector': SelectFprSelector,
            'SelectThresholdSelector': SelectThresholdSelector,
        }

        selector_cls = selectors.get(method_name)
        if selector_cls is None:
            return {
                'status': 'error',
                'error': f"Feature selection method '{method_name}' not found.",
                'suggestion': f"Available methods: {list(selectors.keys())}"
            }

        # Build params
        params = kwargs.get('method_params', {})
        if 'k' in kwargs and kwargs['k'] is not None:
            params['k'] = kwargs['k']
        if 'threshold' in kwargs and kwargs['threshold'] is not None:
            params['threshold'] = kwargs['threshold']

        selector = selector_cls(**params)
        X_selected = selector.fit_transform(X, y)

        # Get selected feature names
        try:
            selected_indices = selector.get_support(indices=True)
            selected_names = [feature_names[i] for i in selected_indices]
        except Exception:
            selected_names = [f'feature_{i}' for i in range(X_selected.shape[1])]

        result = {
            'status': 'success',
            'method': method_name,
            'n_original': int(X.shape[1]),
            'n_selected': int(X_selected.shape[1]),
            'selected_features': selected_names,
        }

        # Include scores if available
        if hasattr(selector, 'scores_') and selector.scores_ is not None:
            scores_arr = np.asarray(selector.scores_)
            result['scores'] = {
                feature_names[i]: float(scores_arr[i])
                for i in range(min(len(scores_arr), len(feature_names)))
            }

        # Save filtered dataset to temp CSV
        import pandas as pd
        upload_dir = os.path.join(tempfile.gettempdir(), 'tuiml_selected')
        os.makedirs(upload_dir, exist_ok=True)
        file_path = os.path.join(upload_dir, f'selected_{uuid.uuid4().hex[:8]}.csv')

        df = pd.DataFrame(X_selected, columns=selected_names)
        if y is not None:
            target_name = kwargs.get('target', 'target')
            df[target_name] = y
        df.to_csv(file_path, index=False)
        result['file_path'] = file_path

        return result
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e),
            'error_type': type(e).__name__
        }


def execute_statistical_test(**kwargs) -> Dict[str, Any]:
    """Run statistical significance tests on experiment results.

    Backs the ``tuiml_test_statistics`` tool. Supported tests: friedman,
    nemenyi, wilcoxon, paired_t, anova, friedman_aligned, quade. The
    pairwise tests (wilcoxon, paired_t) compare the first two algorithms
    in ``results``.

    Parameters
    ----------
    test : str
        Name of the test to run (arrives via ``**kwargs``, like all
        parameters below).
    results : dict
        ``{algorithm_name: [scores...]}`` mapping of per-fold scores.
    significance_level : float, default=0.05
        Alpha level for significance.
    higher_better : bool, default=True
        Whether higher scores are better (pairwise tests only).

    Returns
    -------
    result : dict
        On success: ``status`` (``'success'``), ``test`` and
        ``significant``; most tests add ``statistic`` and ``p_value``;
        pairwise tests add ``algorithms`` and a ``details`` dict of
        means/stds; nemenyi returns per-pair significance in
        ``details``. On failure: ``status`` (``'error'``), ``error`` and
        optionally ``suggestion`` / ``error_type``.
    """
    import numpy as np

    try:
        test_name = kwargs['test']
        raw_results = kwargs['results']
        alpha = kwargs.get('significance_level', 0.05)
        higher_better = kwargs.get('higher_better', True)

        # Convert results to numpy arrays
        results = {name: np.array(scores, dtype=float) for name, scores in raw_results.items()}

        from tuiml.evaluation.statistics import (
            friedman_test, nemenyi_post_hoc, wilcoxon_signed_rank_test,
            paired_t_test, one_way_anova, friedman_aligned_ranks_test, quade_test,
        )

        if test_name == 'friedman':
            statistic, p_value, significant = friedman_test(results, significance_level=alpha)
            return {
                'status': 'success',
                'test': 'friedman',
                'statistic': float(statistic),
                'p_value': float(p_value),
                'significant': bool(significant),
            }

        elif test_name == 'nemenyi':
            pairwise = nemenyi_post_hoc(results, significance_level=alpha)
            details = {f"{k[0]} vs {k[1]}": bool(v) for k, v in pairwise.items()}
            return {
                'status': 'success',
                'test': 'nemenyi',
                'significant': any(pairwise.values()),
                'details': details,
            }

        elif test_name in ('wilcoxon', 'paired_t'):
            # Pairwise tests: use first two algorithms
            names = list(results.keys())
            if len(names) < 2:
                return {
                    'status': 'error',
                    'error': 'Pairwise tests require at least 2 algorithms in results.',
                }
            x, y = results[names[0]], results[names[1]]

            if test_name == 'wilcoxon':
                stats = wilcoxon_signed_rank_test(x, y, significance_level=alpha, higher_better=higher_better)
            else:
                stats = paired_t_test(x, y, significance_level=alpha, higher_better=higher_better)

            return {
                'status': 'success',
                'test': test_name,
                'algorithms': [names[0], names[1]],
                'statistic': float(stats.t_statistic),
                'p_value': float(stats.p_value),
                'significant': stats.is_significant(),
                'details': {
                    f'{names[0]}_mean': float(stats.x_mean),
                    f'{names[1]}_mean': float(stats.y_mean),
                    f'{names[0]}_std': float(stats.x_std),
                    f'{names[1]}_std': float(stats.y_std),
                    'diff_mean': float(stats.diff_mean),
                    'significance': stats.significance.name,
                }
            }

        elif test_name == 'anova':
            groups = list(results.values())
            f_stat, p_value, significant = one_way_anova(*groups, significance_level=alpha)
            return {
                'status': 'success',
                'test': 'anova',
                'statistic': float(f_stat),
                'p_value': float(p_value),
                'significant': bool(significant),
            }

        elif test_name == 'friedman_aligned':
            statistic, p_value, significant = friedman_aligned_ranks_test(results, significance_level=alpha)
            return {
                'status': 'success',
                'test': 'friedman_aligned',
                'statistic': float(statistic),
                'p_value': float(p_value),
                'significant': bool(significant),
            }

        elif test_name == 'quade':
            statistic, p_value, significant = quade_test(results, significance_level=alpha)
            return {
                'status': 'success',
                'test': 'quade',
                'statistic': float(statistic),
                'p_value': float(p_value),
                'significant': bool(significant),
            }

        else:
            return {
                'status': 'error',
                'error': f"Unknown test: '{test_name}'",
                'suggestion': "Available tests: friedman, nemenyi, wilcoxon, paired_t, anova, friedman_aligned, quade"
            }
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e),
            'error_type': type(e).__name__
        }


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


def execute_read_data(**kwargs) -> Dict[str, Any]:
    """Read and preview actual rows from a dataset using ``Dataset.to_pandas()``.

    Backs the ``tuiml_read_data`` tool.

    Parameters
    ----------
    data : str
        Dataset to read: dataset_id, file path, or built-in name
        (arrives via ``**kwargs``, like all parameters below).
    n_rows : int, default=10
        Number of rows to return, capped at 100.
    mode : str, default='head'
        Row selection mode: ``'head'``, ``'tail'``, ``'sample'``, or
        ``'indices'``.
    indices : list of int, default=None
        Explicit row indices, used with ``mode='indices'``.
    columns : list of str, default=None
        Subset of columns to include; unknown names are dropped.
    target : str, default=None
        Target column name, kept in the output when ``include_target``.
    include_target : bool, default=True
        Whether to include the target column in the preview.

    Returns
    -------
    result : dict
        On success: ``status`` (``'success'``), ``name``, ``shape``,
        ``columns``, ``n_rows_returned``, and ``rows`` (list of
        column -> value dicts with floats rounded to 6 places). On
        failure: ``status`` (``'error'``), ``error`` and ``error_type``.
    """
    try:
        dataset = _load_data(kwargs['data'])
        include_target = kwargs.get('include_target', True)
        df = dataset.to_pandas(include_target=include_target)

        n_rows = min(kwargs.get('n_rows', 10), 100)
        mode = kwargs.get('mode', 'head')

        # Filter columns if requested
        requested_cols = kwargs.get('columns')
        if requested_cols:
            # Keep only columns that exist
            valid = [c for c in requested_cols if c in df.columns]
            # Always include target if present and requested
            target_name = kwargs.get('target')
            if include_target and target_name and target_name in df.columns and target_name not in valid:
                valid.append(target_name)
            df = df[valid]

        # Select rows based on mode
        if mode == 'head':
            subset = df.head(n_rows)
        elif mode == 'tail':
            subset = df.tail(n_rows)
        elif mode == 'sample':
            from tuiml.utils.seed import get_global_seed
            seed = get_global_seed()
            subset = df.sample(n=min(n_rows, len(df)), random_state=seed if seed is not None else 42)
        elif mode == 'indices':
            indices = kwargs.get('indices', [])
            indices = [i for i in indices if 0 <= i < len(df)]
            subset = df.iloc[indices]
        else:
            subset = df.head(n_rows)

        # Convert to list of dicts, rounding floats for readability
        rows = []
        for _, row in subset.iterrows():
            d = {}
            for col in subset.columns:
                val = row[col]
                if hasattr(val, 'item'):
                    val = val.item()
                if isinstance(val, float):
                    val = round(val, 6)
                d[col] = val
            rows.append(d)

        return {
            'status': 'success',
            'name': kwargs['data'],
            'shape': [len(df), len(df.columns)],
            'columns': list(subset.columns),
            'n_rows_returned': len(rows),
            'rows': rows,
        }
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e),
            'error_type': type(e).__name__
        }


def execute_algorithm_skeleton(**kwargs) -> Dict[str, Any]:
    """Return a starter code skeleton for a new user algorithm.

    Backs the ``tuiml_get_skeleton`` tool; delegates to
    ``tuiml.agent.user_algorithms.skeleton``.

    Parameters
    ----------
    kind : str, default='classifier'
        Kind of algorithm to scaffold (e.g. ``'classifier'`` or
        ``'regressor'``). Arrives via ``**kwargs``, like all parameters
        below.
    class_name : str, default='MyAlgorithm'
        Name of the generated class.
    version : str, default='1.0.0'
        Initial version string embedded in the skeleton.
    description : str, default='Describe what your algorithm does.'
        One-line description embedded in the skeleton docstring.

    Returns
    -------
    result : dict
        Result dict from ``user_algorithms.skeleton`` with ``status`` and
        the generated skeleton code.
    """
    from tuiml.agent import user_algorithms
    return user_algorithms.skeleton(
        kind=kwargs.get("kind", "classifier"),
        class_name=kwargs.get("class_name", "MyAlgorithm"),
        version=kwargs.get("version", "1.0.0"),
        description=kwargs.get("description", "Describe what your algorithm does."),
    )


def execute_create_algorithm(**kwargs) -> Dict[str, Any]:
    """Create and register a new user algorithm from source code.

    Backs the ``tuiml_create_algorithm`` tool; delegates to
    ``tuiml.agent.user_algorithms.create``.

    Parameters
    ----------
    name : str
        Class name of the new algorithm. Required (arrives via
        ``**kwargs``, like all parameters below).
    kind : str
        Algorithm kind (e.g. ``'classifier'`` or ``'regressor'``).
        Required.
    code : str
        Full Python source of the algorithm. Required.
    version : str, default='1.0.0'
        Version to register the algorithm under.
    description : str, default=None
        Short description stored with the algorithm.
    force : bool, default=False
        Overwrite an existing algorithm of the same name/version.

    Returns
    -------
    result : dict
        Result dict from ``user_algorithms.create`` with ``status`` and
        creation details, or an error dict listing missing required
        fields.
    """
    from tuiml.agent import user_algorithms
    required = [k for k in ("name", "kind", "code") if k not in kwargs]
    if required:
        return {"status": "error", "error_type": "ValueError",
                "error": f"missing required fields: {', '.join(required)}"}
    return user_algorithms.create(
        name=kwargs["name"],
        kind=kwargs["kind"],
        code=kwargs["code"],
        version=kwargs.get("version", "1.0.0"),
        description=kwargs.get("description"),
        force=bool(kwargs.get("force", False)),
    )


def execute_list_user_algorithms(**kwargs) -> Dict[str, Any]:
    """List all user-created algorithms.

    Delegates to ``tuiml.agent.user_algorithms.list_all``. Takes no
    arguments (any ``**kwargs`` are ignored).

    Returns
    -------
    result : dict
        Result dict from ``user_algorithms.list_all`` with ``status`` and
        the registered user algorithms.
    """
    from tuiml.agent import user_algorithms
    return user_algorithms.list_all()


def execute_read_algorithm(**kwargs) -> Dict[str, Any]:
    """Read the source code of a user or built-in algorithm.

    Backs the ``tuiml_read_algorithm`` tool; delegates to
    ``tuiml.agent.user_algorithms.read_source``.

    Parameters
    ----------
    name : str
        Algorithm class name. Required (arrives via ``**kwargs``, like
        all parameters below).
    version : str, default=None
        Specific version to read; defaults to the latest.
    builtin : bool, default=False
        Read a built-in TuiML algorithm instead of a user algorithm.

    Returns
    -------
    result : dict
        Result dict from ``user_algorithms.read_source`` with ``status``
        and the source code, or an error dict when ``name`` is missing.
    """
    from tuiml.agent import user_algorithms
    if "name" not in kwargs:
        return {"status": "error", "error_type": "ValueError",
                "error": "missing required field: name"}
    return user_algorithms.read_source(
        name=kwargs["name"],
        version=kwargs.get("version"),
        builtin=bool(kwargs.get("builtin", False)),
    )


def execute_list_algorithm_files(**kwargs) -> Dict[str, Any]:
    """List algorithm source files, built-in and/or user-created.

    Backs the ``tuiml_list_files`` tool; delegates to
    ``tuiml.agent.user_algorithms.list_algorithm_files``.

    Parameters
    ----------
    builtin : bool, default=True
        Include built-in TuiML algorithm files (arrives via ``**kwargs``,
        like all parameters below).
    user : bool, default=True
        Include user-created algorithm files.

    Returns
    -------
    result : dict
        Result dict from ``user_algorithms.list_algorithm_files`` with
        ``status`` and the file listing.
    """
    from tuiml.agent import user_algorithms
    return user_algorithms.list_algorithm_files(
        builtin=bool(kwargs.get("builtin", True)),
        user=bool(kwargs.get("user", True)),
    )


def execute_search_source(**kwargs) -> Dict[str, Any]:
    """Search algorithm source code for a text query.

    Backs the ``tuiml_search_source`` tool; delegates to
    ``tuiml.agent.user_algorithms.search_source``.

    Parameters
    ----------
    query : str
        Text to search for. Required (arrives via ``**kwargs``, like all
        parameters below).
    name : str, default=None
        Restrict the search to one algorithm.
    builtin : bool, default=True
        Search built-in TuiML algorithm sources.
    user : bool, default=True
        Search user-created algorithm sources.

    Returns
    -------
    result : dict
        Result dict from ``user_algorithms.search_source`` with
        ``status`` and the matches, or an error dict when ``query`` is
        missing.
    """
    from tuiml.agent import user_algorithms
    if "query" not in kwargs:
        return {"status": "error", "error_type": "ValueError",
                "error": "missing required field: query"}
    return user_algorithms.search_source(
        query=kwargs["query"],
        name=kwargs.get("name"),
        builtin=bool(kwargs.get("builtin", True)),
        user=bool(kwargs.get("user", True)),
    )


def execute_edit_algorithm(**kwargs) -> Dict[str, Any]:
    """Apply a string replacement edit to a user algorithm's source.

    Backs the ``tuiml_edit_algorithm`` tool; delegates to
    ``tuiml.agent.user_algorithms.edit_algorithm``.

    Parameters
    ----------
    name : str
        Algorithm class name to edit. Required (arrives via ``**kwargs``,
        like all parameters below).
    old_string : str
        Exact source text to replace. Required.
    new_string : str
        Replacement text. Required.
    version : str, default=None
        Specific version to edit; defaults to the latest.
    bump_version : bool, default=False
        Save the edit as a new bumped version instead of in place.

    Returns
    -------
    result : dict
        Result dict from ``user_algorithms.edit_algorithm`` with
        ``status`` and edit details, or an error dict listing missing
        required fields.
    """
    from tuiml.agent import user_algorithms
    required = [k for k in ("name", "old_string", "new_string") if k not in kwargs]
    if required:
        return {"status": "error", "error_type": "ValueError",
                "error": f"missing required fields: {', '.join(required)}"}
    return user_algorithms.edit_algorithm(
        name=kwargs["name"],
        old_string=kwargs["old_string"],
        new_string=kwargs["new_string"],
        version=kwargs.get("version"),
        bump_version=bool(kwargs.get("bump_version", False)),
    )


def execute_research_log(**kwargs) -> Dict[str, Any]:
    """Return the research log (versions and recorded runs) for user algorithms.

    Delegates to ``tuiml.agent.user_algorithms.research_log``.

    Parameters
    ----------
    name : str, default=None
        Restrict the log to one algorithm; when omitted, all user
        algorithms are included (arrives via ``**kwargs``).

    Returns
    -------
    result : dict
        Result dict from ``user_algorithms.research_log`` with ``status``
        and the per-algorithm version/run history.
    """
    from tuiml.agent import user_algorithms
    return user_algorithms.research_log(name=kwargs.get("name"))


def execute_delete_user_algorithm(**kwargs) -> Dict[str, Any]:
    """Delete a user-created algorithm (or one of its versions).

    Backs the ``tuiml_delete_algorithm`` tool; delegates to
    ``tuiml.agent.user_algorithms.delete``.

    Parameters
    ----------
    name : str
        Algorithm class name to delete. Required (arrives via
        ``**kwargs``, like all parameters below).
    version : str, default=None
        Specific version to delete; when omitted, all versions are
        removed.

    Returns
    -------
    result : dict
        Result dict from ``user_algorithms.delete`` with ``status`` and
        deletion details, or an error dict when ``name`` is missing.
    """
    from tuiml.agent import user_algorithms
    if "name" not in kwargs:
        return {"status": "error", "error_type": "ValueError",
                "error": "missing required field: name"}
    return user_algorithms.delete(name=kwargs["name"], version=kwargs.get("version"))


def _detect_install_method() -> Dict[str, Any]:
    """Inspect sys.prefix / sys.executable to guess how tuiml was installed."""
    import sys
    # Don't resolve(): that follows the python symlink out of the tool venv.
    prefix = sys.prefix.replace("\\", "/")
    exe = sys.executable.replace("\\", "/")

    # Editable install wins over path-matching: check first so a dev checkout
    # imported into any venv is still reported as editable-dev.
    try:
        import tuiml as _pkg
        pkg_dir = Path(_pkg.__file__).resolve().parent
        if (pkg_dir.parent / "pyproject.toml").exists():
            return {"method": "editable-dev", "writable": False,
                    "upgrade_hint": "cd <checkout> && git pull"}
    except Exception:
        pass

    # uv tool install puts the venv under .../uv/tools/tuiml/
    if "/uv/tools/tuiml" in prefix or "/uv/tools/tuiml" in exe:
        return {"method": "uv-tool", "writable": True,
                "upgrade_hint": "uv tool install --reinstall --force tuiml"}

    # Default: assume pip / uv pip
    return {"method": "pip", "writable": True,
            "upgrade_hint": f"{sys.executable} -m pip install --upgrade tuiml"}


def _query_latest_pypi_version(package: str = "tuiml", timeout: float = 5.0) -> Dict[str, Any]:
    """Look up the latest released version of a package on PyPI.

    Parameters
    ----------
    package : str, default='tuiml'
        PyPI package name to query.
    timeout : float, default=5.0
        HTTP timeout in seconds.

    Returns
    -------
    result : dict
        On success: ``ok`` (True), ``version`` and ``released`` (upload
        timestamp). On failure: ``ok`` (False) and ``error``.
    """
    import json as _json
    import urllib.request
    url = f"https://pypi.org/pypi/{package}/json"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            data = _json.loads(resp.read().decode("utf-8"))
        return {"ok": True, "version": data["info"]["version"],
                "released": data["releases"].get(data["info"]["version"], [{}])[0].get("upload_time")}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def execute_system_info(**kwargs) -> Dict[str, Any]:
    """Report installation details for the running TuiML install.

    Backs the ``tuiml_system_info`` tool.

    Parameters
    ----------
    check_latest : bool, default=True
        Also query PyPI for the latest released version (arrives via
        ``**kwargs``).

    Returns
    -------
    result : dict
        On success: ``status`` (``'success'``), ``version``,
        ``install_method``, ``upgrade_hint``, ``package_path``,
        ``site_packages``, ``python_executable``, ``python_version``,
        ``platform``, and -- when ``check_latest`` -- ``latest_version``
        and ``update_available`` (or ``latest_version_error``). On
        failure: ``status`` (``'error'``), ``error`` and ``error_type``.
    """
    import sys
    import platform as _plat
    try:
        import tuiml as _pkg
        pkg_dir = Path(_pkg.__file__).resolve().parent
        version = getattr(_pkg, "__version__", "unknown")
    except Exception as e:
        return {"status": "error", "error": f"cannot import tuiml: {e}",
                "error_type": type(e).__name__}

    install = _detect_install_method()
    result: Dict[str, Any] = {
        "status": "success",
        "version": version,
        "install_method": install["method"],
        "upgrade_hint": install["upgrade_hint"],
        "package_path": str(pkg_dir),
        "site_packages": str(pkg_dir.parent),
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
        "platform": _plat.platform(),
    }

    if kwargs.get("check_latest", True):
        pypi = _query_latest_pypi_version()
        if pypi["ok"]:
            latest = pypi["version"]
            result["latest_version"] = latest
            result["update_available"] = (latest != version)
            if pypi.get("released"):
                result["latest_released"] = pypi["released"]
        else:
            result["latest_version_error"] = pypi["error"]

    return result


def execute_self_update(**kwargs) -> Dict[str, Any]:
    """Upgrade tuiml to the latest PyPI version using the detected installer.

    Backs the ``tuiml_self_update`` tool. Editable/dev checkouts are
    refused; use ``git pull`` there instead.

    Parameters
    ----------
    target_version : str, default=None
        Specific version to install; defaults to the latest release
        (arrives via ``**kwargs``, like all parameters below).
    dry_run : bool, default=False
        Only report the command that would run; make no changes.

    Returns
    -------
    result : dict
        On success: ``status`` (``'success'``), ``install_method``,
        ``command``, ``returncode``, ``stdout``, ``stderr``,
        ``version_before``, ``version_after``, ``restart_required`` and
        ``note`` (dry runs return ``dry_run`` and ``command`` only). On
        failure: ``status`` (``'error'``), ``error`` and ``error_type``.
    """
    import subprocess
    import sys

    install = _detect_install_method()
    method = install["method"]

    if method == "editable-dev":
        return {
            "status": "error",
            "error": "refusing to upgrade an editable / dev checkout, run `git pull` in the source tree instead",
            "error_type": "EditableInstallError",
            "install_method": method,
        }

    target = kwargs.get("target_version")
    spec = f"tuiml=={target}" if target else "tuiml"

    if method == "uv-tool":
        cmd = ["uv", "tool", "install", "--reinstall", "--force", spec]
    else:  # pip
        cmd = [sys.executable, "-m", "pip", "install", "--upgrade", spec]

    if kwargs.get("dry_run"):
        return {
            "status": "success",
            "dry_run": True,
            "install_method": method,
            "command": cmd,
            "note": "no changes made; set dry_run=false to actually run",
        }

    try:
        before = execute_system_info(check_latest=False).get("version")
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    except subprocess.TimeoutExpired:
        return {"status": "error", "error": "upgrade timed out after 300s",
                "error_type": "TimeoutExpired", "command": cmd}
    except FileNotFoundError as e:
        return {"status": "error",
                "error": f"installer not found on PATH: {e.filename or str(e)}",
                "error_type": "FileNotFoundError", "command": cmd}

    # After upgrade the on-disk package has changed, but the running process is
    # still holding the old module objects. Re-reading the installed version
    # requires a subprocess.
    try:
        probe = subprocess.run(
            [sys.executable, "-c",
             "import importlib, importlib.metadata as m; "
             "importlib.invalidate_caches(); "
             "print(m.version('tuiml'))"],
            capture_output=True, text=True, timeout=15,
        )
        after = probe.stdout.strip() or None
    except Exception:
        after = None

    ok = (proc.returncode == 0)
    return {
        "status": "success" if ok else "error",
        "install_method": method,
        "command": cmd,
        "returncode": proc.returncode,
        "stdout": proc.stdout[-4000:],
        "stderr": proc.stderr[-4000:],
        "version_before": before,
        "version_after": after,
        "restart_required": ok,
        "note": "Call tuiml_restart to reload the MCP servers (or restart the client manually).",
    }


def execute_restart(**kwargs) -> Dict[str, Any]:
    """Restart every running tuiml-mcp child process.

    When called from an MCP context the current server is one of those
    children. We schedule a deferred self-exit (after a short delay)
    so this response can be flushed back to the agent before the
    process dies; the parent client (Claude Desktop, Cursor, …) will
    auto-respawn the child with the newly installed code on its next
    request.

    Parameters
    ----------
    include_self : bool, default=True
        Also schedule a deferred exit of the current server process
        (arrives via ``**kwargs``).

    Returns
    -------
    result : dict
        ``status`` (``'success'``), ``killed_other`` (count),
        ``failed``, ``self_exit_scheduled`` and ``note``.
    """
    from tuiml.agent.restart_util import find_mcp_processes, kill_mcp_processes

    include_self = kwargs.get("include_self", True)
    others = find_mcp_processes(exclude_self=True)
    result = kill_mcp_processes(
        procs=others,
        grace_seconds=2.0,
        include_self=include_self,
        self_delay_seconds=0.5,
    )

    return {
        "status": "success",
        "killed_other": result["killed"],
        "failed": result["failed"],
        "self_exit_scheduled": result["self_exit_scheduled"],
        "note": (
            "Clients automatically respawn their tuiml-mcp child on the next "
            "request. If you called this right after tuiml_self_update, the "
            "new version will be loaded then."
        ),
    }


# =============================================================================
# Notebook Export
# =============================================================================

def _nb_markdown(lines: List[str]) -> Dict:
    """Build a Jupyter markdown cell dict from source lines."""
    return {"cell_type": "markdown", "id": uuid.uuid4().hex[:8],
            "metadata": {}, "source": lines}


def _nb_code(lines: List[str]) -> Dict:
    """Build a Jupyter code cell dict from source lines."""
    return {"cell_type": "code", "id": uuid.uuid4().hex[:8],
            "execution_count": None, "metadata": {}, "outputs": [],
            "source": lines}


def _repr_arg(v: Any) -> str:
    """Compact repr for a single arg value."""
    if isinstance(v, str):
        return repr(v)
    if isinstance(v, list):
        return repr(v)
    if isinstance(v, dict):
        return repr(v)
    return repr(v)


def _call_to_kwargs_str(args: dict, skip: set = None, indent: int = 4) -> str:
    """Format a dict of args as indented keyword arguments.

    Parameters
    ----------
    args : dict
        Argument name -> value mapping to render.
    skip : set, default=None
        Keys to omit (None-valued entries are always omitted).
    indent : int, default=4
        Number of spaces to indent each line.

    Returns
    -------
    text : str
        Newline-joined ``name=value,`` lines.
    """
    pad = ' ' * indent
    skip = skip or set()
    lines = []
    for k, v in args.items():
        if k in skip or v is None:
            continue
        lines.append(f"{pad}{k}={_repr_arg(v)},")
    return '\n'.join(lines)


def _data_load_lines(data: str) -> List[str]:
    """Return notebook code lines that load `data` into variable `_dataset`.

    Parameters
    ----------
    data : str
        File path (loaded via pandas) or built-in dataset name (loaded
        via ``load_dataset``).

    Returns
    -------
    lines : list of str
        Source lines for a notebook code cell.
    """
    if data and os.path.isfile(os.path.expanduser(data)):
        return [
            f"import pandas as _pd\n",
            f"_df = _pd.read_csv({repr(data)})\n",
            f"_X = _df.iloc[:, :-1].values\n",
            f"_y = _df.iloc[:, -1].values",
        ]
    # assume builtin dataset name
    return [f"_dataset = load_dataset({repr(data)})"]


def _resolve_model_var(model_id: Optional[str], fallback: str = "model_1") -> str:
    """Map a model_id back to the Python variable name used in the notebook.

    Parameters
    ----------
    model_id : str or None
        Model identifier recorded during the session.
    fallback : str, default='model_1'
        Variable name returned when the id is unknown.

    Returns
    -------
    var : str
        Notebook variable name (e.g. ``'result_2'``).
    """
    if model_id and model_id in _MODEL_ID_TO_VAR:
        return _MODEL_ID_TO_VAR[model_id]
    return fallback


def _translate_call(call: Dict, train_counter: List[int]) -> tuple:
    """Translate one recorded session call into notebook cells.

    Parameters
    ----------
    call : dict
        Recorded call with ``'tool'`` and ``'args'`` keys.
    train_counter : list of int
        Single-element mutable counter of train calls seen so far, used
        to number ``result_N`` / ``model_N`` variables.

    Returns
    -------
    md_lines : list of str or None
        Markdown cell source, or None when the tool is not translatable.
    code_lines : list of str or None
        Code cell source, or None when the tool is not translatable.
    """
    tool = call['tool']
    args = call['args']

    # ── tuiml_profile_data ───────────────────────────────────────────────────
    if tool == 'tuiml_profile_data':
        data = args.get('data', '')
        target = args.get('target')
        md = [
            f"## Data Profiling, `{data}`\n",
            f"> `tuiml_profile_data(data={repr(data)}`",
        ]
        code = _data_load_lines(data) + [
            "\n",
            "import pandas as pd\n",
            "_df_profile = pd.DataFrame(_dataset.X, columns=_dataset.feature_names)\n",
        ]
        if target:
            code.append(f"_df_profile[{repr(target)}] = _dataset.y\n")
        code += [
            "print(f'Shape: {_df_profile.shape}')\n",
            "print(f'Missing values: {_df_profile.isnull().sum().sum()}')\n",
        ]
        if target:
            code.append(f"print('Class distribution:\\n', _df_profile[{repr(target)}].value_counts())\n")
        code.append("_df_profile.describe()")
        return md, code

    # ── tuiml_train ──────────────────────────────────────────────────────────
    if tool == 'tuiml_train':
        train_counter[0] += 1
        n = train_counter[0]
        algo = args.get('algorithm', 'UnknownAlgorithm')
        data = args.get('data', '')
        target = args.get('target')
        # tuiml.train() uses the spec convention: the model is
        # {"name": algo, "params": {...}}, the data is {"source", "target"},
        # steps live in one ordered "pipeline" list, and evaluation options
        # are grouped into an "evaluation" dict. Translate the tool-level
        # vocabulary into that shape so the generated call is valid,
        # idiomatic Python.
        model_spec = {"name": algo}
        algo_params = args.get('algorithm_params')
        if isinstance(algo_params, dict) and algo_params:
            model_spec["params"] = algo_params
        data_spec = {"source": data}
        if target is not None:
            data_spec["target"] = target
        if args.get('features') is not None:
            data_spec["features"] = args['features']

        def _nest_step(step):
            """Convert a flat tool-level step ({"name", **params}) to spec form."""
            if isinstance(step, str):
                return {"name": step}
            step = dict(step)
            name = step.pop('name', None)
            params = step.pop('params', step)
            nested = {"name": name}
            if params:
                nested["params"] = params
            return nested

        steps = [_nest_step(s) for s in args.get('preprocessing') or []]
        if args.get('feature_selection'):
            steps.append(_nest_step(args['feature_selection']))
        pipeline = steps or args.get('preset')
        evaluation = {
            k: args[k]
            for k in ('cv', 'test_size', 'stratify', 'metrics')
            if args.get(k) is not None
        }
        lines = [
            f"result_{n} = tuiml.train({{\n",
            f"    \"model\": {model_spec!r},\n",
            f"    \"data\": {data_spec!r},\n",
        ]
        if pipeline:
            lines.append(f"    \"pipeline\": {pipeline!r},\n")
        if evaluation:
            lines.append(f"    \"evaluation\": {evaluation!r},\n")
        if args.get('random_seed') is not None:
            lines.append(f"    \"random_seed\": {args['random_seed']!r},\n")
        lines.append("})\n")
        md = [
            f"## Train `{algo}` (step {n})\n",
            f"> `tuiml_train(algorithm={repr(algo)}, data={repr(data)}, ...)`",
        ]
        code = lines + [
            f"model_{n} = result_{n}.model_\n",
            f"print('Metrics:', result_{n}.metrics_)",
        ]
        return md, code

    # ── tuiml_predict ────────────────────────────────────────────────────────
    if tool == 'tuiml_predict':
        model_id = args.get('model_id')
        var = _resolve_model_var(model_id)
        result_var = var  # e.g. result_1
        model_var = var.replace('result_', 'model_')
        data = args.get('data', '')
        md = [
            f"## Predict with `{result_var}`\n",
            f"> `tuiml_predict(model_id=..., data={repr(data)})`",
        ]
        code = _data_load_lines(data) + [
            "\n",
            # Predict via the fitted Workflow so the training-time
            # transformations are re-applied (a bare model would see raw,
            # untransformed inputs and produce wrong predictions).
            f"predictions = {result_var}.predict(_dataset.X)\n",
            "print('Predictions (first 10):', predictions[:10])",
        ]
        return md, code

    # ── tuiml_evaluate ───────────────────────────────────────────────────────
    if tool == 'tuiml_evaluate':
        model_id = args.get('model_id')
        var = _resolve_model_var(model_id)
        model_var = var.replace('result_', 'model_')
        data = args.get('data', '')
        target = args.get('target')
        metrics = args.get('metrics')
        md = [
            f"## Evaluate `{model_var}`\n",
            f"> `tuiml_evaluate(model_id=..., data={repr(data)})`",
        ]
        code = _data_load_lines(data) + [
            "\n",
            # Evaluate via the fitted Workflow: it re-applies the fitted
            # transformations before scoring, so the metrics reflect the real
            # pipeline.
            f"eval_metrics = {var}.evaluate(\n",
            f"    _dataset.X, _dataset.y,\n",
        ]
        if metrics:
            code.append(f"    metrics={repr(metrics)},\n")
        code += [")\n", "print('Eval metrics:', eval_metrics)"]
        return md, code

    # ── tuiml_benchmark ─────────────────────────────────────────────────────
    if tool == 'tuiml_benchmark':
        algos = args.get('algorithms', [])
        data_arg = args.get('data', [])
        if isinstance(data_arg, str):
            data_arg = [data_arg]
        cv = args.get('cv', 10)
        metrics = args.get('metrics')
        md = [
            f"## Benchmark, {', '.join(str(a) for a in algos)}\n",
            f"> `tuiml_benchmark(algorithms={algos}, data={data_arg}, cv={cv})`",
        ]
        code = []
        for ds_name in data_arg:
            safe = ds_name.replace('-', '_').replace('/', '_')
            code.append(f"_{safe} = load_dataset({repr(ds_name)})\n")
        model_specs = [
            a if isinstance(a, dict) else {"name": a} for a in algos
        ]
        dataset_specs = "[" + ", ".join(
            f'{{"name": {repr(d)}, '
            f'"X": _{d.replace("-","_").replace("/","_")}.X, '
            f'"y": _{d.replace("-","_").replace("/","_")}.y}}'
            for d in data_arg
        ) + "]"
        evaluation = {"cv": cv}
        if metrics:
            evaluation["metrics"] = list(metrics)
        code += [
            "\n",
            "bench = tuiml.Benchmark(\n",
            f"    models={model_specs!r},\n",
            f"    datasets={dataset_specs},\n",
            f"    evaluation={evaluation!r},\n",
        ]
        seed = args.get('random_seed')
        if seed is not None:
            code.append(f"    random_seed={seed!r},\n")
        code += [
            ").run()\n",
            "print(bench.summary())\n",
            "bench.table()",
        ]
        return md, code

    # ── tuiml_tune ───────────────────────────────────────────────────────────
    if tool == 'tuiml_tune':
        algo = args.get('algorithm', '')
        data = args.get('data', '')
        method = args.get('method', 'random')
        param_grid = args.get('param_grid', {})
        cv = args.get('cv', 5)
        scoring = args.get('scoring', 'accuracy_score')
        n_iter = args.get('n_iter', 10)
        n_iterations = args.get('n_iterations', 50)
        # Prefer an explicit random_state, then the effective session seed folded
        # in by record_session_call, then the default, so tuning reproduces.
        random_state = args.get('random_state', args.get('random_seed', 42))
        cls_map = {'grid': 'GridSearchCV', 'random': 'RandomSearchCV', 'bayesian': 'BayesianSearchCV'}
        tuner_cls = cls_map.get(method, 'RandomSearchCV')
        param_kw = 'param_grid' if method == 'grid' else ('param_space' if method == 'bayesian' else 'param_distributions')
        n_kw_line = (f"    n_iter={n_iter},\n" if method == 'random'
                     else f"    n_iterations={n_iterations},\n" if method == 'bayesian' else "")
        md = [
            f"## Hyperparameter Tuning, `{algo}` ({method} search)\n",
            f"> `tuiml_tune(algorithm={repr(algo)}, method={repr(method)}, ...)`",
        ]
        code = [
            "from tuiml.registry import registry as _registry\n",
            "import tuiml.algorithms as _\n",
            f"from tuiml.evaluation.tuning import {tuner_cls}\n",
            "\n",
            *_data_load_lines(data), "\n",
            f"_cls = _registry.get({repr(algo)})\n",
            f"tuner = {tuner_cls}(\n",
            f"    estimator=_cls(),\n",
            f"    {param_kw}={repr(param_grid)},\n",
            f"    cv={cv},\n",
            f"    scoring={repr(scoring)},\n",
            n_kw_line,
            f"    random_state={random_state},\n",
            ")\n",
            "tuner.fit(_dataset.X, _dataset.y)\n",
            "print('Best params:', tuner.best_params_)\n",
            "print('Best score: ', tuner.best_score_)",
        ]
        return md, code

    # ── tuiml_plot ───────────────────────────────────────────────────────────
    if tool == 'tuiml_plot':
        plot_type = args.get('plot_type', '')
        model_id = args.get('model_id')
        var = _resolve_model_var(model_id)
        model_var = var.replace('result_', 'model_')
        data = args.get('data', '')
        target = args.get('target', '')
        algo = args.get('algorithm', '')
        title = args.get('title') or f"{plot_type.replace('_', ' ').title()}"

        md = [
            f"## Plot, `{plot_type}`\n",
            f"> `tuiml_plot(plot_type={repr(plot_type)}, ...)`",
        ]

        if plot_type == 'confusion_matrix':
            code = (
                _data_load_lines(data) + ["\n",
                "from tuiml.evaluation.visualization import plot_confusion_matrix\n",
                f"_preds = {var}.predict(_dataset.X)\n",
                f"plot_confusion_matrix(_dataset.y, _preds, title={repr(title)})\n",
                "plt.show()",
            ])
        elif plot_type == 'roc_curve':
            code = (
                _data_load_lines(data) + ["\n",
                "from tuiml.evaluation.visualization import plot_roc_curve\n",
                f"_probas = {var}.predict_proba(_dataset.X)\n",
                f"plot_roc_curve(_dataset.y, _probas, title={repr(title)})\n",
                "plt.show()",
            ])
        elif plot_type == 'pr_curve':
            code = (
                _data_load_lines(data) + ["\n",
                "from tuiml.evaluation.visualization import plot_pr_curve\n",
                f"_probas = {var}.predict_proba(_dataset.X)\n",
                "# PR curve takes positive-class scores; pick column 1 if 2-D.\n",
                "_score = _probas[:, 1] if _probas.ndim == 2 else _probas\n",
                f"plot_pr_curve(_dataset.y, _score, title={repr(title)})\n",
                "plt.show()",
            ])
        elif plot_type == 'feature_importance':
            code = [
                f"_importances = getattr({model_var}, 'feature_importances_', None)\n",
                "if _importances is None:\n",
                "    raise ValueError('This model does not expose feature importances.')\n",
                "plt.figure(figsize=(10, 5))\n",
                "plt.bar(range(len(_importances)), _importances)\n",
                f"plt.title({repr(title)})\n",
                "plt.xlabel('Feature Index'); plt.ylabel('Importance')\n",
                "plt.tight_layout(); plt.show()",
            ]
        elif plot_type == 'learning_curve':
            # plot_learning_curve takes precomputed (train_sizes, train_scores,
            # test_scores), so compute them here over increasing train subsets.
            code = (
                _data_load_lines(data) + ["\n",
                "import numpy as np\n",
                "from tuiml.evaluation.visualization import plot_learning_curve\n",
                "from tuiml.registry import registry\n",
                "import tuiml.algorithms  # noqa: F401, registers algorithms\n",
                "from tuiml.evaluation.splitting import train_test_split\n",
                "from tuiml.evaluation.metrics import accuracy_score\n",
                f"_cls = registry.get({repr(algo)})\n",
                "_Xtr, _Xte, _ytr, _yte = train_test_split(\n",
                "    _dataset.X, _dataset.y, test_size=0.25, random_state=42)\n",
                "_sizes, _train_sc, _test_sc = [], [], []\n",
                "for _frac in np.linspace(0.2, 1.0, 5):\n",
                "    _n = max(2, int(len(_Xtr) * _frac))\n",
                "    _m = _cls(); _m.fit(_Xtr[:_n], _ytr[:_n])\n",
                "    _sizes.append(_n)\n",
                "    _train_sc.append(accuracy_score(_ytr[:_n], _m.predict(_Xtr[:_n])))\n",
                "    _test_sc.append(accuracy_score(_yte, _m.predict(_Xte)))\n",
                "plot_learning_curve(np.array(_sizes), np.array(_train_sc),\n",
                f"                    np.array(_test_sc), title={repr(title)})\n",
                "plt.show()",
            ])
        elif plot_type in ('cd_diagram', 'boxplot_comparison', 'heatmap', 'ranking_table'):
            exp_results = args.get('benchmark_results', {})
            # cd_diagram maps to plot_critical_difference; all take the scores
            # dict as the first positional argument (not 'benchmark_results=').
            fn = 'plot_critical_difference' if plot_type == 'cd_diagram' else f'plot_{plot_type}'
            code = [
                "import numpy as np\n",
                f"from tuiml.evaluation.visualization import {fn}\n",
                f"_exp = {{k: np.array(v, dtype=float) for k, v in {repr(exp_results)}.items()}}\n",
                f"{fn}(_exp)\n",
                "plt.show()",
            ]
        else:
            return None, None

        return md, code

    # ── tuiml_save_model ─────────────────────────────────────────────────────
    if tool == 'tuiml_save_model':
        model_id = args.get('model_id')
        dest = args.get('destination', './model.joblib')
        var = _resolve_model_var(model_id)
        model_var = var.replace('result_', 'model_')
        md = [
            f"## Save Model → `{dest}`\n",
            f"> `tuiml_save_model(model_id=..., destination={repr(dest)})`",
        ]
        # model.save()/Algorithm.load() are a matched pair; a fitted Workflow
        # round-trips through them too, carrying its steps along.
        code = [
            f"{model_var}.save({repr(dest)})\n",
            f"print('Model saved to {dest}')\n",
            "\n",
            f"# Verify reload\n",
            f"from tuiml.base.algorithms import Algorithm\n",
            f"_reloaded = Algorithm.load({repr(dest)})\n",
            f"print('Reloaded:', _reloaded)",
        ]
        return md, code

    # ── tuiml_generate_data ──────────────────────────────────────────────────
    if tool == 'tuiml_generate_data':
        gen = args.get('generator', '')
        n_samples = args.get('n_samples', 100)
        kw = _call_to_kwargs_str({k: v for k, v in args.items() if k != 'generator'})
        md = [
            f"## Generate Synthetic Data, `{gen}`\n",
            f"> `tuiml_generate_data(generator={repr(gen)}, n_samples={n_samples})`",
        ]
        code = [
            f"from tuiml.datasets.generators import {gen}\n",
            f"_gen = {gen}(\n",
            kw + "\n",
            ")\n",
            "_gen_dataset = _gen.generate()\n",
            "print(f'Generated: {_gen_dataset.X.shape}')",
        ]
        return md, code

    # ── tuiml_preprocess ─────────────────────────────────────────────────────
    if tool == 'tuiml_preprocess':
        data = args.get('data', '')
        steps = args.get('steps', [])
        target = args.get('target')
        # Normalize steps to a list of step names.
        step_list = steps if isinstance(steps, list) else [steps]
        step_list = [s for s in step_list if s]
        md = [
            f"## Preprocess Data, {step_list}\n",
            f"> `tuiml_preprocess(data={repr(data)}, steps={step_list})`",
        ]
        code = _data_load_lines(data) + ["\n", "_X_pre = _dataset.X\n"]
        for step in step_list:
            code += [
                f"from tuiml.preprocessing import {step}\n",
                f"_X_pre = {step}().fit_transform(_X_pre)\n",
            ]
        code.append("print(f'Preprocessed shape: {_X_pre.shape}')")
        return md, code

    # ── tuiml_select_features ────────────────────────────────────────────────
    if tool == 'tuiml_select_features':
        data = args.get('data', '')
        method = args.get('method', '')
        target = args.get('target', '')
        k = args.get('k')
        md = [
            f"## Feature Selection, `{method}`\n",
            f"> `tuiml_select_features(data={repr(data)}, method={repr(method)})`",
        ]
        init_args = {}
        if k:
            init_args['k'] = k
        init_str = ', '.join(f'{kk}={repr(vv)}' for kk, vv in init_args.items())
        code = (
            _data_load_lines(data) + ["\n",
            f"from tuiml.features.selection import {method}\n",
            f"_selector = {method}({init_str})\n",
            "_selector.fit(_dataset.X, _dataset.y)\n",
            "_X_selected = _selector.transform(_dataset.X)\n",
            "print(f'Features: {_dataset.X.shape[1]} → {_X_selected.shape[1]}')\n",
            "if hasattr(_selector, 'selected_indices_'):\n",
            "    print('Selected indices:', _selector.selected_indices_)",
        ])
        return md, code

    # ── tuiml_test_statistics ────────────────────────────────────────────────
    if tool == 'tuiml_test_statistics':
        test = args.get('test', '')
        results = args.get('results', {})
        alpha = args.get('significance_level', 0.05)
        md = [
            f"## Statistical Test, `{test}`\n",
            f"> `tuiml_test_statistics(test={repr(test)}, ...)`",
        ]
        # Statistical tests are functions in tuiml.evaluation.statistics, not
        # classes. Map each test to its function and call shape (mirrors the
        # tuiml_test_statistics executor).
        fn_map = {
            'friedman': 'friedman_test', 'nemenyi': 'nemenyi_post_hoc',
            'wilcoxon': 'wilcoxon_signed_rank_test', 'paired_t': 'paired_t_test',
            'anova': 'one_way_anova', 'friedman_aligned': 'friedman_aligned_ranks_test',
            'quade': 'quade_test',
        }
        fn = fn_map.get(test, 'friedman_test')
        code = [
            "import numpy as np\n",
            f"from tuiml.evaluation.statistics import {fn}\n",
            f"_results = {{k: np.array(v, dtype=float) for k, v in {repr(results)}.items()}}\n",
        ]
        if test in ('friedman', 'friedman_aligned', 'quade'):
            code += [
                f"statistic, p_value, significant = {fn}(_results, significance_level={alpha})\n",
                "print('Statistic:', statistic)\n",
                "print('p-value:  ', p_value)\n",
                "print('Significant:', significant)",
            ]
        elif test == 'anova':
            code += [
                f"f_stat, p_value, significant = {fn}(*_results.values(), significance_level={alpha})\n",
                "print('F-statistic:', f_stat)\n",
                "print('p-value:    ', p_value)\n",
                "print('Significant:', significant)",
            ]
        elif test in ('wilcoxon', 'paired_t'):
            code += [
                "_names = list(_results.keys())\n",
                "_x, _y = _results[_names[0]], _results[_names[1]]\n",
                f"_stats = {fn}(_x, _y, significance_level={alpha})\n",
                "print('Statistic:', _stats.t_statistic)\n",
                "print('p-value:  ', _stats.p_value)\n",
                "print('Significant:', _stats.is_significant())",
            ]
        else:  # nemenyi
            code += [
                f"_pairwise = {fn}(_results, significance_level={alpha})\n",
                "for _pair, _sig in _pairwise.items():\n",
                "    print(_pair, '→ significant:', bool(_sig))",
            ]
        return md, code

    # ── tuiml_upload_data ────────────────────────────────────────────────────
    if tool == 'tuiml_upload_data':
        file_path = args.get('file_path', '')
        name = args.get('name', '')
        md = [
            f"## Load Dataset, `{name or file_path}`\n",
            f"> `tuiml_upload_data(file_path={repr(file_path)})`",
        ]
        if file_path:
            code = [
                "import pandas as pd\n",
                f"_df = pd.read_csv({repr(file_path)})\n",
                "print(_df.shape)\n",
                "_df.head()",
            ]
        else:
            code = [f"# Dataset '{name}' was registered inline, recreate from source"]
        return md, code

    return None, None  # skip unhandled tools


def execute_export_notebook(**kwargs) -> Dict[str, Any]:
    """Export the current MCP session as a reproducible Jupyter notebook.

    Backs the ``tuiml_export_notebook`` tool. Translates every recorded
    successful workflow call into paired markdown + code cells, pinning
    the session's random seed so the notebook reproduces the run.

    Parameters
    ----------
    path : str, default='~/tuiml_session.ipynb'
        Output path for the notebook file (arrives via ``**kwargs``,
        like all parameters below).
    title : str, default='TuiML Session, Exported Notebook'
        Title used in the notebook's header cell.

    Returns
    -------
    result : dict
        On success: ``status`` (``'success'``), ``path`` (absolute),
        ``cells``, ``workflow_calls`` and ``message``. On failure:
        ``status`` (``'error'``) and ``error`` (e.g. when the session has
        no recorded calls or the file cannot be written).
    """
    import json
    import datetime as _dt

    path = os.path.expanduser(kwargs.get('path') or '~/tuiml_session.ipynb')
    title = kwargs.get('title', 'TuiML Session, Exported Notebook')

    with _SESSION_LOCK:
        calls_snapshot = list(_SESSION_CALLS)

    if not calls_snapshot:
        return {
            'status': 'error',
            'error': (
                'No workflow calls have been recorded in this session yet. '
                'Run some tuiml_train / tuiml_benchmark / tuiml_plot calls first.'
            ),
        }

    cells = []

    # ── Header cell ──────────────────────────────────────────────────────────
    cells.append(_nb_markdown([
        f"# {title}\n",
        f"\n",
        f"Exported from MCP session · {_dt.date.today()}  \n",
        "Re-run each cell top-to-bottom to reproduce the full workflow.\n",
        "\n",
        "**Requirements:** `pip install tuiml`",
    ]))

    # ── Install cell ─────────────────────────────────────────────────────────
    # First executable cell installs tuiml (e.g. on Colab / a fresh kernel).
    cells.append(_nb_code([
        "!pip install tuiml",
    ]))

    # ── Imports cell ─────────────────────────────────────────────────────────
    cells.append(_nb_code([
        "import tuiml\n",
        "from tuiml.datasets import load_dataset\n",
        "import matplotlib.pyplot as plt\n",
        "import pandas as pd\n",
        "import numpy as np",
    ]))

    # ── Global seed cell ─────────────────────────────────────────────────────
    # Mirror the MCP session's reproducibility: execute_tool sets a process-wide
    # seed, which the workflow reads as a fallback for any step that doesn't take
    # an explicit seed (data generation, feature selection, CV splits, plots).
    # Pin the same seed here so the notebook reproduces those steps too.
    _session_seed = next(
        (c['args']['random_seed'] for c in calls_snapshot
         if c['args'].get('random_seed') is not None),
        None,
    )
    if _session_seed is not None:
        cells.append(_nb_markdown([
            "## Reproducibility\n",
            f"This session ran with random seed `{_session_seed}`. "
            "Setting it globally pins NumPy/Python RNG so results match the original run.",
        ]))
        cells.append(_nb_code([
            "from tuiml.utils.seed import set_global_seed\n",
            f"set_global_seed({repr(_session_seed)})",
        ]))

    train_counter = [0]
    skipped = 0

    for call in calls_snapshot:
        md_lines, code_lines = _translate_call(call, train_counter)
        if md_lines is None:
            skipped += 1
            continue
        cells.append(_nb_markdown(md_lines))
        cells.append(_nb_code(code_lines))

    # Build notebook JSON
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.11.0"},
        },
        "cells": cells,
    }

    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)) or '.', exist_ok=True)
        with open(path, 'w') as fh:
            json.dump(nb, fh, indent=1)
    except Exception as e:
        return {'status': 'error', 'error': f'Could not write notebook: {e}'}

    abs_path = os.path.abspath(path)
    workflow_count = len(calls_snapshot) - skipped
    return {
        'status': 'success',
        'path': abs_path,
        'cells': len(cells),
        'workflow_calls': workflow_count,
        'message': (
            f'Notebook written to {abs_path} '
            f'({workflow_count} workflow steps → {len(cells)} cells). '
            f'Open with: jupyter notebook {abs_path}'
        ),
    }


# =============================================================================
# Tool Registry
# =============================================================================

TOOL_EXECUTORS = {
    "tuiml_train": execute_train,
    "tuiml_predict": execute_predict,
    "tuiml_evaluate": execute_evaluate,
    "tuiml_benchmark": execute_benchmark,
    "tuiml_upload_data": execute_upload_data,
    "tuiml_save_model": execute_save_model,
    "tuiml_serve_model": execute_serve_model,
    "tuiml_stop_server": execute_stop_server,
    "tuiml_server_status": execute_server_status,
    "tuiml_list": execute_list,
    "tuiml_describe": execute_describe,
    "tuiml_plot": execute_plot,
    "tuiml_profile_data": execute_data_profile,
    "tuiml_generate_data": execute_generate_data,
    "tuiml_preprocess": execute_preprocess,
    "tuiml_select_features": execute_select_features,
    "tuiml_test_statistics": execute_statistical_test,
    "tuiml_tune": execute_tune,
    "tuiml_read_data": execute_read_data,
    "tuiml_system_info": execute_system_info,
    "tuiml_self_update": execute_self_update,
    "tuiml_restart":     execute_restart,
    "tuiml_get_skeleton": execute_algorithm_skeleton,
    "tuiml_create_algorithm": execute_create_algorithm,
    "tuiml_delete_algorithm": execute_delete_user_algorithm,
    "tuiml_read_algorithm": execute_read_algorithm,
    "tuiml_list_files": execute_list_algorithm_files,
    "tuiml_search_source": execute_search_source,
    "tuiml_edit_algorithm": execute_edit_algorithm,
    "tuiml_export_notebook": execute_export_notebook,
}


# Bootstrap: re-register agent-authored algorithms from disk so they survive
# MCP server restarts.
try:
    from tuiml.agent import user_algorithms as _user_algorithms
    _bootstrap_result = _user_algorithms.load_all()
    if _bootstrap_result.get("loaded"):
        import sys as _sys
        print(f"[tuiml] loaded {_bootstrap_result['loaded']} user algorithm(s)",
              file=_sys.stderr)
    if _bootstrap_result.get("errors"):
        import sys as _sys
        for err in _bootstrap_result["errors"]:
            print(f"[tuiml] user algorithm load error: {err}", file=_sys.stderr)
except Exception as _e:  # never block the server on bootstrap failures
    import sys as _sys
    print(f"[tuiml] user-algorithm bootstrap failed: {_e}", file=_sys.stderr)

def get_tool_output_schema(tool_name: str) -> Dict[str, Any]:
    """Get the JSON output schema for a tool.

    Parameters
    ----------
    tool_name : str
        Name of the MCP tool (e.g. ``'tuiml_train'``).

    Returns
    -------
    schema : dict
        The tool's output JSON Schema, or the generic component output
        schema when the tool has no dedicated one.
    """
    return OUTPUT_SCHEMAS.get(tool_name, COMPONENT_OUTPUT_SCHEMA)

def get_tool_annotations(tool_name: str) -> Dict[str, bool]:
    """Get MCP behavior annotations for a tool.

    Parameters
    ----------
    tool_name : str
        Name of the MCP tool.

    Returns
    -------
    annotations : dict
        Annotation flags (``readOnlyHint``, ``destructiveHint``,
        ``idempotentHint``, ``openWorldHint``); component tools fall back
        to read-only defaults.
    """
    # Define annotations for each tool
    TOOL_ANNOTATIONS = {
        "tuiml_train": {
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": False,
            "openWorldHint": True
        },
        "tuiml_predict": {
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False
        },
        "tuiml_evaluate": {
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False
        },
        "tuiml_benchmark": {
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": False,
            "openWorldHint": True
        },
        "tuiml_upload_data": {
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": False,
            "openWorldHint": False
        },
        "tuiml_save_model": {
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False
        },
        "tuiml_serve_model": {
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": False,
            "openWorldHint": True
        },
        "tuiml_stop_server": {
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False
        },
        "tuiml_server_status": {
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False
        },
        "tuiml_list": {
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False
        },
        "tuiml_describe": {
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": True
        },
        "tuiml_search": {
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False
        },
        "tuiml_plot": {
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False
        },
        "tuiml_profile_data": {
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False
        },
        "tuiml_generate_data": {
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": False,
            "openWorldHint": False
        },
        "tuiml_preprocess": {
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": False,
            "openWorldHint": False
        },
        "tuiml_select_features": {
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": False,
            "openWorldHint": False
        },
        "tuiml_test_statistics": {
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False
        },
        "tuiml_tune": {
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": False,
            "openWorldHint": True
        },
        "tuiml_read_data": {
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False
        },
        "tuiml_export_notebook": {
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False
        },
    }

    # Default annotations for component tools
    DEFAULT_COMPONENT_ANNOTATIONS = {
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False
    }

    return TOOL_ANNOTATIONS.get(tool_name, DEFAULT_COMPONENT_ANNOTATIONS)

def get_workflow_tools() -> Dict[str, Dict]:
    """Get all workflow, discovery and code tool schemas.

    Returns
    -------
    tools : dict
        Mapping of tool name to its MCP input schema definition.
    """
    return {**WORKFLOW_TOOLS, **DISCOVERY_TOOLS, **CODE_TOOLS}

def execute_tool(tool_name: str, **kwargs) -> Dict[str, Any]:
    """Execute a tool by name, resolving the random seed first.

    Sets a process-wide seed (explicit ``random_seed`` kwarg, or a fresh
    random one) before dispatching to the workflow executor or, failing
    that, a registered component tool.

    Parameters
    ----------
    tool_name : str
        Name of the tool to execute.
    random_seed : int, default=None
        Random seed for the call; a random seed is generated when
        omitted (arrives via ``**kwargs``, like the tool arguments).
    **kwargs
        Remaining arguments are forwarded to the tool executor.

    Returns
    -------
    result : dict
        The executor's result dict; successful workflow results also get
        the effective ``random_seed`` added. Component tools return
        ``status``, ``result`` (stringified) and ``type``. Unknown tools
        return ``status`` (``'error'``) and ``error``.
    """
    random_seed = kwargs.pop('random_seed', None)
    
    if random_seed is None:
        import random
        random_seed = random.randint(0, 2**31 - 1)
        
    from tuiml.utils.seed import set_global_seed
    set_global_seed(random_seed)
    
    if tool_name in ('tuiml_tune', 'tuiml_generate_data', 'tuiml_train', 'tuiml_benchmark'):
        kwargs['random_seed'] = random_seed

    # Check workflow tools first
    if tool_name in TOOL_EXECUTORS:
        result = TOOL_EXECUTORS[tool_name](**kwargs)
        if isinstance(result, dict) and result.get('status') == 'success':
            result['random_seed'] = random_seed
        return result

    # For any component tool, ensure full registry is loaded
    from tuiml.agent.registry import get_tool
    tool = get_tool(tool_name)
    if tool:
        try:
            result = tool.executor(kwargs)
            return {
                'status': 'success',
                'result': str(result),
                'type': result.__class__.__name__
            }
        except Exception as e:
            return {'status': 'error', 'error': str(e)}

    return {'status': 'error', 'error': f"Unknown tool: {tool_name}"}
