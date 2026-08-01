"""Dataset profiling."""

from typing import Any, Dict

from .._spec import ToolSpec
from .._shared import _load_data


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


SPEC = ToolSpec(
    name='tuiml_profile_data',
    description="Inspect a dataset before training, shape, dtypes, missing values, "
        "basic statistics, and class distribution. Works with file paths or "
        "built-in dataset names.",
    input_schema={
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
        },
    output_schema={
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
    execute=execute_data_profile,
    group='workflow',
    read_only=True, destructive=False,
    idempotent=True, open_world=False,
)
