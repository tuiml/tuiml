"""High-level API for TuiML - One-liner functions for common ML tasks."""

from typing import Callable, Union, Optional, List, Dict, Any
import os
import threading
import numpy as np
import pandas as pd

from tuiml.workflow import WorkflowResult

# Module-level state for tracking running servers
_SERVERS: Dict[str, Dict] = {}

# =============================================================================
# API Registry - LLM-friendly function discovery
# =============================================================================

API_REGISTRY = {
    "train": {
        "description": "Train a machine learning model with a complete workflow",
        "parameters": {
            "algorithm": {
                "type": ["string", "object"],
                "required": True,
                "description": "Algorithm name or dict with name and params"
            },
            "data": {
                "type": ["string", "object", "array"],
                "required": True,
                "description": "File path, DataFrame, or numpy array"
            },
            "target": {
                "type": ["string", "array"],
                "required": True,
                "description": "Target column name or array"
            },
            "preprocessing": {
                "type": ["string", "array", "null"],
                "default": None,
                "description": "Preprocessing steps or preset name"
            },
            "feature_selection": {
                "type": ["string", "object", "null"],
                "default": None,
                "description": "Feature selector name or config dict"
            },
            "test_size": {
                "type": "number",
                "default": 0.2,
                "description": "Proportion of data for testing"
            },
            "cv": {
                "type": ["integer", "null"],
                "default": None,
                "description": "Number of cross-validation folds"
            },
            "metrics": {
                "type": ["string", "array"],
                "default": "auto",
                "description": "Metrics to compute"
            },
            "preset": {
                "type": ["string", "null"],
                "default": None,
                "enum": ["minimal", "fast", "standard", "full", "imbalanced"],
                "description": "Preprocessing preset name"
            }
        },
        "returns": "WorkflowResult with model, metrics, predictions"
    },
    "run": {
        "description": "Execute workflow from configuration dict or JSON file",
        "parameters": {
            "config": {
                "type": ["object", "string"],
                "required": True,
                "description": "Config dict or path to JSON file"
            }
        },
        "returns": "WorkflowResult"
    },
    "predict": {
        "description": "Make predictions with a trained model",
        "parameters": {
            "model": {
                "type": "object",
                "required": True,
                "description": "Trained model instance"
            },
            "data": {
                "type": ["object", "array"],
                "required": True,
                "description": "Data to predict on"
            }
        },
        "returns": "numpy array of predictions"
    },
    "evaluate": {
        "description": "Evaluate model performance on test data",
        "parameters": {
            "model": {
                "type": "object",
                "required": True,
                "description": "Trained model instance"
            },
            "X": {
                "type": ["object", "array"],
                "required": True,
                "description": "Test features"
            },
            "y": {
                "type": "array",
                "required": True,
                "description": "True labels"
            },
            "metrics": {
                "type": ["string", "array"],
                "default": "auto",
                "description": "Metrics to compute"
            }
        },
        "returns": "Dict of metric names to values"
    },
    "experiment": {
        "description": "Compare multiple algorithms on multiple datasets",
        "parameters": {
            "algorithms": {
                "type": ["object", "array"],
                "required": True,
                "description": "Dict or list of algorithms to compare"
            },
            "datasets": {
                "type": ["object", "array"],
                "required": True,
                "description": "Dict or list of datasets"
            },
            "cv": {
                "type": "integer",
                "default": 10,
                "description": "Number of CV folds"
            },
            "metrics": {
                "type": ["array", "null"],
                "default": None,
                "description": "Metrics to compute"
            }
        },
        "returns": "Experiment object with results and statistics"
    },
    "list_algorithms": {
        "description": "List available algorithms in registry",
        "parameters": {
            "type": {
                "type": ["string", "null"],
                "default": None,
                "enum": ["classifier", "regressor", "clusterer", None],
                "description": "Filter by algorithm type"
            }
        },
        "returns": "List of algorithm metadata dicts"
    },
    "describe_algorithm": {
        "description": "Get detailed info and parameter schema for an algorithm",
        "parameters": {
            "name": {
                "type": "string",
                "required": True,
                "description": "Algorithm name (e.g., 'RandomForestClassifier')"
            }
        },
        "returns": "Dict with description, parameters schema, type"
    },
    "search_algorithms": {
        "description": "Search algorithms by keyword",
        "parameters": {
            "query": {
                "type": "string",
                "required": True,
                "description": "Search keyword"
            }
        },
        "returns": "List of matching algorithm metadata"
    },
    "save": {
        "description": "Save trained model to disk",
        "parameters": {
            "model": {
                "type": "object",
                "required": True,
                "description": "Model to save"
            },
            "path": {
                "type": "string",
                "required": True,
                "description": "File path"
            }
        },
        "returns": "None"
    },
    "load": {
        "description": "Load trained model from disk",
        "parameters": {
            "path": {
                "type": "string",
                "required": True,
                "description": "File path"
            }
        },
        "returns": "Loaded model instance"
    },
    "serve": {
        "description": "Serve a trained model via REST API",
        "parameters": {
            "model_or_path": {
                "type": ["string", "object"],
                "required": True,
                "description": "File path, WorkflowResult, or model object"
            },
            "host": {
                "type": "string",
                "default": "127.0.0.1",
                "description": "Host to bind the server to"
            },
            "port": {
                "type": "integer",
                "default": 8000,
                "description": "Port to listen on"
            },
            "model_id": {
                "type": "string",
                "default": "default",
                "description": "Identifier for the model"
            },
            "background": {
                "type": "boolean",
                "default": True,
                "description": "Run server in background thread"
            }
        },
        "returns": "Dict with server_id, url, endpoints (background) or None (blocking)"
    },
    "stop_server": {
        "description": "Stop running model server(s)",
        "parameters": {
            "server_id": {
                "type": ["string", "null"],
                "default": None,
                "description": "Server ID to stop, or None to stop all"
            }
        },
        "returns": "None"
    },
    "server_status": {
        "description": "Get status of running model servers",
        "parameters": {},
        "returns": "List of server info dicts"
    }
}

def get_api_info(function_name: str = None) -> dict:
    """Get API function schemas for LLM discovery.

    Parameters
    ----------
    function_name : str, optional
        Specific function name. If None, returns all API schemas.

    Returns
    -------
    dict
        API function schema(s) with parameters, types, and descriptions.

    Examples
    --------
    >>> get_api_info("train")
    >>> get_api_info()  # All functions
    """
    if function_name:
        if function_name not in API_REGISTRY:
            raise ValueError(f"Unknown function: {function_name}. Available: {list(API_REGISTRY.keys())}")
        return API_REGISTRY[function_name]
    return API_REGISTRY

# ===== Main API Functions =====

def train(
    algorithm: Union[str, Dict] = None,
    data: Union[str, pd.DataFrame, np.ndarray] = None,
    target: Union[str, np.ndarray] = None,
    *,
    # Preprocessing (flexible formats)
    preprocessing: Optional[Union[str, List[Union[str, Dict]]]] = None,
    # Feature selection (flexible formats)
    feature_selection: Optional[Union[str, Dict]] = None,
    # Data splitting
    test_size: float = 0.2,
    stratify: bool = True,
    random_state: Optional[int] = 42,
    # Cross-validation
    cv: Optional[int] = None,
    # Metrics
    metrics: Union[List[str], str] = "auto",
    # Output options
    return_model: bool = True,
    return_predictions: bool = False,
    return_probabilities: bool = False,
    # Presets
    preset: Optional[str] = None,
    # Verbosity
    verbose: bool = False,
    # Accept any algorithm parameters as kwargs
    **kwargs
) -> WorkflowResult:
    """Train a machine learning model with a complete workflow.

    This is the **main high-level function** for training models in TuiML.
    It supports multiple flexible input formats for maximum convenience and
    is designed to be LLM-friendly for automated ML workflows.

    Parameters
    ----------
    algorithm : str or dict
        Algorithm specification. Accepts flexible formats:

        - ``"RandomForestClassifier"`` — Class name as string
        - ``{"name": "RandomForestClassifier", "n_trees": 100}`` — Dict with params

    data : str, DataFrame, or ndarray
        Training data source:
        
        - ``"path/to/file.csv"`` — Path to data file
        - ``DataFrame`` — Pandas DataFrame
        - ``ndarray`` — Numpy array

    target : str or ndarray
        Target specification:
        
        - ``"column_name"`` — Column name in DataFrame
        - ``ndarray`` — Target array

    preprocessing : str, list, or None, default=None
        Preprocessing pipeline specification:
        
        - ``"MinMaxScaler"`` — Single preprocessor name
        - ``["SimpleImputer", "MinMaxScaler"]`` — List of names
        - ``[{"name": "SimpleImputer", "strategy": "mean"}]`` — List of dicts
        - ``"standard"`` — Preset name
        - ``None`` — No preprocessing

    feature_selection : str, dict, or None, default=None
        Feature selection specification:
        
        - ``"SelectKBestSelector"`` — Class name
        - ``{"name": "SelectKBestSelector", "k": 10}`` — Dict with params
        - ``None`` — No feature selection

    test_size : float, default=0.2
        Proportion of data reserved for testing.

    stratify : bool, default=True
        Whether to maintain class distribution in train/test split.

    random_state : int or None, default=42
        Random seed for reproducibility.

    cv : int or None, default=None
        Number of cross-validation folds. If set, uses CV instead of holdout.

    metrics : str or list, default="auto"
        Metrics to compute:
        
        - ``"auto"`` — Automatically select based on task type
        - ``["accuracy", "f1", "roc_auc"]`` — List of metric names
        - ``"accuracy"`` — Single metric name

    return_model : bool, default=True
        Whether to include the trained model in the result.

    return_predictions : bool, default=False
        Whether to include test set predictions in the result.

    return_probabilities : bool, default=False
        Whether to include prediction probabilities in the result.

    preset : str or None, default=None
        Preprocessing preset name:
        
        - ``"minimal"`` — No preprocessing
        - ``"fast"`` — Basic imputation only
        - ``"standard"`` — Impute + Normalize + Encode
        - ``"full"`` — All preprocessing + feature selection
        - ``"imbalanced"`` — Standard + SMOTESampler

    verbose : bool, default=False
        Whether to print progress information.

    **kwargs
        Additional algorithm parameters passed directly to the model
        (e.g., ``n_trees=100``, ``max_depth=10``).

    Returns
    -------
    WorkflowResult
        Result object containing:
        
        - ``model`` — Trained model instance
        - ``metrics`` — Dict of computed metrics
        - ``predictions`` — Test predictions (if requested)
        - ``probabilities`` — Prediction probabilities (if requested)
        - ``metadata`` — Additional workflow metadata

    Examples
    --------
    Simplest usage with just algorithm and data:

    >>> result = tuiml.train("RandomForestClassifier", "iris.csv", "class")

    With algorithm parameters as kwargs:

    >>> result = tuiml.train(
    ...     "RandomForestClassifier",
    ...     "iris.csv",
    ...     "class",
    ...     n_trees=100,
    ...     max_depth=10
    ... )

    Using a preprocessing preset:

    >>> result = tuiml.train(
    ...     "RandomForestClassifier",
    ...     "iris.csv",
    ...     "class",
    ...     preset="standard",
    ...     n_trees=100
    ... )

    With explicit preprocessing pipeline:

    >>> result = tuiml.train(
    ...     "RandomForestClassifier",
    ...     "iris.csv",
    ...     "class",
    ...     preprocessing=["SimpleImputer", "MinMaxScaler"],
    ...     feature_selection="SelectKBestSelector",
    ...     n_trees=100
    ... )

    Cross-validation:

    >>> result = tuiml.train(
    ...     "SVM",
    ...     "data.csv",
    ...     "label",
    ...     preprocessing="standard",
    ...     cv=10,
    ...     kernel="rbf",
    ...     C=1.0
    ... )
    """
    # Validate required params
    if algorithm is None:
        raise ValueError("Algorithm must be specified")
    if data is None:
        raise ValueError("Data must be provided")

    # Extract algorithm name and params (only place we need to handle dict format)
    if isinstance(algorithm, dict):
        algo_dict = algorithm.copy()
        algo_name = algo_dict.pop("name", None)
        algo_params = {**algo_dict, **kwargs}  # Merge dict params with kwargs
    else:
        algo_name = algorithm
        algo_params = kwargs

    # Handle preprocessing preset
    if preset and preset in PRESETS:
        preset_config = PRESETS[preset]
        preprocessing = preset_config.get("preprocessing", [])
        if feature_selection is None and preset_config.get("feature_selection"):
            feature_selection = preset_config["feature_selection"]
    elif isinstance(preprocessing, str) and preprocessing in PRESETS:
        preset_config = PRESETS[preprocessing]
        preprocessing = preset_config.get("preprocessing", [])
        if feature_selection is None and preset_config.get("feature_selection"):
            feature_selection = preset_config["feature_selection"]

    # Use Workflow to execute the training
    from tuiml.workflow import Workflow

    workflow = Workflow(data=data, target=target)

    # Add preprocessing steps
    if preprocessing:
        if isinstance(preprocessing, list):
            for step in preprocessing:
                if isinstance(step, dict):
                    workflow.preprocess(**step)
                else:
                    workflow.preprocess(step)
        elif isinstance(preprocessing, str):
            # Assume it's a preprocessor class name
            workflow.preprocess(preprocessing)

    # Add feature selection
    if feature_selection:
        if isinstance(feature_selection, dict):
            workflow._feature_selection = feature_selection
        elif isinstance(feature_selection, str):
            workflow._feature_selection = {"name": feature_selection}

    # Configure data splitting
    if cv is None:
        workflow.split(test_size=test_size, stratify=stratify, random_state=random_state)

    # Set the model
    workflow.train(algo_name, **algo_params)

    # Resolve metrics: pass through user-specified metrics, or None for auto
    if metrics == "auto" or metrics is None:
        resolved_metrics = None  # Workflow._execute() auto-resolves based on algo type
    elif isinstance(metrics, str):
        resolved_metrics = [metrics]
    else:
        resolved_metrics = metrics

    # Configure evaluation
    if cv:
        workflow.cross_validate(cv=cv, metrics=resolved_metrics)
    else:
        workflow.evaluate(metrics=resolved_metrics)

    # Execute and return results
    return workflow.run()

def run(config: Union[Dict, str]) -> WorkflowResult:
    """Run a complete workflow from a configuration dict or JSON file.

    This function enables you to load and execute ML workflows from
    configuration dictionaries or JSON files, making it ideal for
    programmatic workflows and LLM-generated configurations.

    Parameters
    ----------
    config : dict or str
        Configuration specification:
        
        - ``dict`` — Configuration dictionary
        - ``str`` — Path to JSON configuration file
        
        Required keys: ``algorithm``, ``data``, ``target``
        
        Optional keys: ``preprocessing``, ``feature_selection``, ``cv``, ``metrics``, etc.

    Returns
    -------
    WorkflowResult
        Result object with model, metrics, and metadata.

    Examples
    --------
    From a configuration dictionary:

    >>> config = {
    ...     "algorithm": "RandomForestClassifier",
    ...     "data": "iris.csv",
    ...     "target": "class",
    ...     "preprocessing": "standard",
    ...     "n_trees": 100,
    ...     "cv": 10
    ... }
    >>> result = tuiml.run(config)

    From a JSON file:

    >>> result = tuiml.run("config.json")

    LLM-generated configuration:

    >>> llm_config = {
    ...     "algorithm": {"name": "RandomForestClassifier", "n_trees": 100},
    ...     "data": "data.csv",
    ...     "target": "label",
    ...     "preprocessing": ["SimpleImputer", "Normalize"],
    ...     "feature_selection": {"name": "SelectKBestSelector", "k": 10},
    ...     "cv": 5
    ... }
    >>> result = tuiml.run(llm_config)
    """
    if isinstance(config, str):
        # Load from JSON file
        import json
        with open(config, 'r') as f:
            config = json.load(f)

    # Pass entire config to train() as kwargs
    return train(**config)

def predict(model, data: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
    """Make predictions with a trained model.

    Parameters
    ----------
    model : object
        Trained model instance with a ``predict()`` method.
        
    data : DataFrame or ndarray
        Data to make predictions on.

    Returns
    -------
    predictions : ndarray
        Array of predictions.

    Examples
    --------
    >>> predictions = tuiml.predict(model, X_new)
    """
    if hasattr(model, 'predict'):
        return model.predict(data)
    else:
        raise ValueError("Model does not have a predict() method")

def evaluate(
    model,
    X: Union[pd.DataFrame, np.ndarray],
    y: np.ndarray,
    metrics: Union[List[str], str] = "auto"
) -> Dict[str, float]:
    """Evaluate a model on test data.

    Parameters
    ----------
    model : object
        Trained model instance.
        
    X : DataFrame or ndarray
        Test features.
        
    y : ndarray
        True labels.
        
    metrics : str or list, default="auto"
        Metrics to compute:

        - ``"auto"`` — Automatically select based on task type
        - ``list`` — List of metric function names from
          ``tuiml.evaluation.metrics`` (e.g.,
          ``["accuracy_score", "f1_score"]``)

    Returns
    -------
    results : dict
        Dictionary mapping metric names to computed values.

    Examples
    --------
    >>> metrics = tuiml.evaluate(model, X_test, y_test)
    >>> print(metrics["accuracy_score"])
    """
    from tuiml.evaluation import metrics as metrics_module

    # Get predictions
    y_pred = model.predict(X)

    # Determine metrics to compute
    if metrics == "auto":
        # Try to detect task type
        if hasattr(model, '_estimator_type'):
            if model._estimator_type == 'classifier':
                metrics = ['accuracy_score', 'f1_score']
            elif model._estimator_type == 'regressor':
                metrics = ['mean_squared_error', 'r2_score']
            else:
                metrics = ['accuracy_score']
        else:
            # Default to classification
            metrics = ['accuracy_score']
    elif isinstance(metrics, str):
        metrics = [metrics]

    # Compute metrics directly from the metrics module
    results = {}
    for metric_name in metrics:
        metric_func = getattr(metrics_module, metric_name, None)
        if metric_func is None:
            continue
        try:
            results[metric_name] = metric_func(y, y_pred)
        except TypeError:
            # Some metrics may need extra args (e.g., average for f1_score)
            try:
                results[metric_name] = metric_func(y, y_pred, average='macro')
            except TypeError:
                continue

    return results

def experiment(
    algorithms: Union[Dict[str, Any], List[Union[str, tuple, Any]]],
    datasets: Union[Dict[str, tuple], List[Union[str, Dict]]],
    *,
    preprocessing: Optional[Union[List[Dict], str]] = None,
    cv: int = 10,
    metrics: List[str] = None,
    n_jobs: int = 1,
    verbose: int = 0,
    progress_callback: Optional[Callable] = None
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
        - ``["RandomForestClassifier", "SVM"]`` — List of class names (uses defaults)
        - ``[("RF", {"n_trees": 100}), ...]`` — List of (name, params) tuples

    datasets : dict or list
        Dataset specifications:
        
        - ``{"iris": (X, y)}`` — Dict of name to (features, target) tuples
        - ``["iris", "wine"]`` — List of dataset names (loads from registry)
        - ``[{"path": "data.csv", "target": "class"}, ...]`` — List of config dicts

    preprocessing : str, list, or None, default=None
        Preprocessing config or preset name to apply to all datasets.

    cv : int, default=10
        Number of cross-validation folds.

    metrics : list of str, optional
        Metrics to compute for each algorithm/dataset pair. 
        Defaults to ``["accuracy"]`` for classification.

    n_jobs : int, default=1
        Number of parallel jobs to run. Use ``-1`` for all available CPUs.

    verbose : int, default=0
        Verbosity level for progress reporting.

    Returns
    -------
    Experiment
        Experiment object containing results, comparison tables, and 
        statistical tests (e.g., Nemenyi test).

    Examples
    --------
    Compare algorithms using class names:

    >>> exp = tuiml.experiment(
    ...     algorithms=["RandomForestClassifier", "SVM", "NaiveBayesClassifier"],
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
    ...     metrics=["accuracy", "f1_macro"]
    ... )
    >>> print(exp.summary())
    """
    from tuiml.evaluation import run_experiment
    from tuiml.hub import registry
    import tuiml.algorithms  # noqa: F401 - trigger registration

    # Convert algorithms to dict format
    models_dict = {}
    if isinstance(algorithms, dict):
        models_dict = algorithms
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
            elif isinstance(item, tuple):
                if len(item) == 2:
                    name, params_or_model = item
                    if isinstance(params_or_model, dict):
                        try:
                            model_class = registry.get(name)
                        except KeyError:
                            raise ValueError(
                                f"Algorithm '{name}' not found in hub. "
                                f"Use list_algorithms() to see available options."
                            )
                        models_dict[name] = model_class(**params_or_model)
                    else:
                        models_dict[name] = params_or_model
            else:
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
                path = item.get('path') or item.get('data')
                target = item.get('target')
                name = item.get('name', path)
                # load() auto-detects format and returns Dataset
                ds = load_file(path, target_column=target)
                datasets_dict[name] = (ds.X, ds.y)
            elif isinstance(item, tuple) and len(item) == 2:
                datasets_dict[f"dataset_{len(datasets_dict)}"] = item

    # Apply preprocessing to all datasets if specified
    if preprocessing:
        from tuiml.workflow import Workflow

        # Resolve preset name to step list
        if isinstance(preprocessing, str) and preprocessing in PRESETS:
            steps = PRESETS[preprocessing].get("preprocessing", [])
        elif isinstance(preprocessing, str):
            steps = [{"name": preprocessing}]
        else:
            steps = preprocessing

        for ds_name, (X_ds, y_ds) in list(datasets_dict.items()):
            for step in steps:
                if isinstance(step, str):
                    name, params = step, {}
                elif isinstance(step, dict):
                    name = step.get('name')
                    params = {k: v for k, v in step.items() if k != 'name'}
                else:
                    continue
                pp_cls = Workflow._resolve_component(name, 'preprocessing')
                if pp_cls is not None:
                    pp = pp_cls(**params)
                    X_ds = pp.fit_transform(X_ds)
            datasets_dict[ds_name] = (X_ds, y_ds)

    # Run experiment
    exp = run_experiment(
        models=models_dict,
        datasets=datasets_dict,
        n_folds=cv,
        metrics=metrics,
        n_jobs=n_jobs,
        verbose=verbose,
        progress_callback=progress_callback
    )

    return exp

def save(model, path: str, metadata: Optional[Dict] = None) -> None:
    """Save a trained model to disk.

    Parameters
    ----------
    model : object
        Trained model instance to save.
        
    path : str
        Target file path (e.g., ``"model.pkl"``).

    metadata : dict, optional
        Additional metadata to store with the model.

    Examples
    --------
    >>> tuiml.save(model, "my_model.pkl", metadata={"accuracy": 0.95})
    """
    from tuiml.utils.serialization import save_model
    save_model(model, path, metadata=metadata)

def load(path: str):
    """Load a trained model from disk.

    Parameters
    ----------
    path : str
        Path to the saved model file.

    Returns
    -------
    object
        Loaded model instance.

    Examples
    --------
    >>> model = tuiml.load("my_model.pkl")
    """
    from tuiml.utils.serialization import load_model
    return load_model(path)

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

def search_algorithms(query: str) -> List[Dict]:
    """Search for algorithms by keyword in name or tags.

    Parameters
    ----------
    query : str
        Search query (e.g., ``"forest"``, ``"linear"``).

    Returns
    -------
    list of dict
        Metadata for matching algorithms.

    Examples
    --------
    >>> results = tuiml.search_algorithms("forest")
    >>> [algo["name"] for algo in results]
    ['RandomForestClassifier', 'ExtraTrees']
    """
    from tuiml.hub import registry

    return registry.search(query)

# Preset configurations
PRESETS = {
    "minimal": {
        "preprocessing": [],
        "feature_selection": None,
    },
    "fast": {
        "preprocessing": [{"name": "SimpleImputer", "strategy": "most_frequent"}],
        "feature_selection": None,
    },
    "standard": {
        "preprocessing": [
            {"name": "SimpleImputer", "strategy": "mean"},
            {"name": "MinMaxScaler"},
            {"name": "OneHotEncoder"},
        ],
        "feature_selection": None,
    },
    "full": {
        "preprocessing": [
            {"name": "SimpleImputer", "strategy": "median"},
            {"name": "StandardScaler"},
            {"name": "OneHotEncoder"},
        ],
        "feature_selection": {"name": "SelectKBestSelector", "k": 10},
    },
    "imbalanced": {
        "preprocessing": [
            {"name": "SimpleImputer", "strategy": "mean"},
            {"name": "MinMaxScaler"},
            {"name": "SMOTESampler"},
        ],
        "feature_selection": None,
    },
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

    Accepts a file path, a ``WorkflowResult``, or a model object and starts
    a uvicorn server exposing prediction endpoints.

    Parameters
    ----------
    model_or_path : str, WorkflowResult, or model object
        The model to serve:

        - ``str`` — Path to a saved model file
        - ``WorkflowResult`` — Result from ``Workflow.run()`` or ``train()``
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
    Serve from a WorkflowResult:

    >>> result = tuiml.train("NaiveBayesClassifier", "iris", "class")
    >>> info = tuiml.serve(result, port=9999)
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
    elif isinstance(model_or_path, WorkflowResult):
        # WorkflowResult — save model to temp file using TuiML serializer
        if model_or_path.model is None:
            raise RuntimeError("WorkflowResult has no trained model.")
        tmp = tempfile.NamedTemporaryFile(suffix=".pkl", delete=False)
        tmp.close()
        save_model(model_or_path.model, tmp.name)
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

    # Wait briefly for server to start
    time.sleep(1.0)

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
