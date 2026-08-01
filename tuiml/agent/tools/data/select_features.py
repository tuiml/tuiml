"""Feature selection."""

import os
import tempfile
import uuid
from typing import Any, Dict

from .._spec import ToolSpec
from .._shared import _load_data


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


SPEC = ToolSpec(
    name='tuiml_select_features',
    description="Run feature selection on a dataset and return selected feature names/indices. "
        "Supports filter methods (SelectKBestSelector, SelectPercentileSelector, "
        "VarianceThresholdSelector, SelectFprSelector, SelectThresholdSelector), "
        "correlation-based (CFSSelector), and wrapper methods (WrapperSelector).",
    input_schema={
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
        },
    output_schema={
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
    execute=execute_select_features,
    group='workflow',
    read_only=False, destructive=False,
    idempotent=False, open_world=False,
)
