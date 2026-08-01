"""Dataset registration."""

import os
import uuid
from typing import Any, Dict

from .._spec import ToolSpec
from .._state import _UPLOADS_DIR, _DATASET_INDEX
from .._shared import _load_data


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


SPEC = ToolSpec(
    name='tuiml_upload_data',
    description="Register a dataset for use with other TuiML tools. "
        "Provide either a file_path to an existing file on disk (preferred for large datasets), "
        "or content as raw text for small inline datasets. "
        "Supported formats: CSV, TSV, ARFF, Parquet, Excel (xlsx/xls), JSON, JSONL, NumPy (npy/npz). "
        "Returns a validated path for use with tuiml_train, tuiml_preprocess, etc.",
    input_schema={
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
        },
    output_schema={
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
    execute=execute_upload_data,
    group='workflow',
    read_only=False, destructive=False,
    idempotent=False, open_world=False,
)
