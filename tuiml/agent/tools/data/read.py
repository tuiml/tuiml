"""Row-level dataset preview."""

from typing import Any, Dict

from .._spec import ToolSpec
from .._shared import _load_data


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


SPEC = ToolSpec(
    name='tuiml_read_data',
    description="Read and preview actual rows from a dataset. Returns sample rows as a list of "
        "dictionaries. Supports head, tail, random sample, or specific row indices.",
    input_schema={
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
        },
    output_schema={
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
    execute=execute_read_data,
    group='workflow',
    read_only=True, destructive=False,
    idempotent=True, open_world=False,
    reproducible=False,
)
