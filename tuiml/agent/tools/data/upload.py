"""Dataset registration."""

import os
import re
import uuid
from typing import Any, Dict

from .._spec import ToolSpec
from .._state import _UPLOADS_DIR, _DATASET_INDEX
from .._shared import _load_data

# A dataset name becomes a filename inside the uploads directory, so it must not
# be able to describe a path. Anything outside this set is replaced rather than
# rejected, because the name is often derived from a file the caller did not
# choose and a hard error there is unhelpful.
_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]")

# Content mode writes ``<name>.<format>``; the extension has to come from a
# closed set, or it is a second way to choose the written path.
_CONTENT_FORMATS = {"csv", "tsv", "arff", "json", "jsonl"}


def _safe_dataset_name(raw: str) -> str:
    """Reduce a caller-supplied dataset name to a bare, path-free filename.

    The name arrives from an LLM agent that may be relaying untrusted content,
    and it is joined onto the uploads directory to build a destination path.
    Left alone, ``"../../.claude"`` escapes that directory and overwrites files
    elsewhere in the home directory -- including the MCP client configs that
    ``tuiml setup`` manages.

    Parameters
    ----------
    raw : str
        Caller-supplied name.

    Returns
    -------
    name : str
        A non-empty name of ``[A-Za-z0-9._-]`` only, with no leading dots,
        truncated to a filesystem-safe length.

    Raises
    ------
    ValueError
        If nothing usable survives sanitisation.
    """
    # basename() first: it drops any directory part, including a Windows
    # separator, before the character filter runs.
    name = os.path.basename(str(raw).replace("\\", "/")).strip()
    name = _SAFE_NAME.sub("_", name)
    # A leading dot would create a hidden file, and "." / ".." survive the
    # character filter untouched.
    name = name.lstrip(".")
    name = name[:100]
    if not name:
        raise ValueError(
            f"Dataset name {raw!r} contains no usable characters. "
            "Use letters, digits, dots, dashes or underscores."
        )
    return name


def _dest_within(upload_dir: str, filename: str) -> str:
    """Join ``filename`` onto ``upload_dir`` and verify it stayed inside.

    :func:`_safe_dataset_name` should already make this unreachable. The check
    is kept because containment, not the character filter, is the property that
    actually matters, and it still holds if the filter is later loosened.

    Parameters
    ----------
    upload_dir : str
        Directory uploads are confined to.
    filename : str
        Bare filename to place inside it.

    Returns
    -------
    dest : str
        Absolute destination path.

    Raises
    ------
    ValueError
        If the resolved destination escapes ``upload_dir``.
    """
    root = os.path.realpath(upload_dir)
    dest = os.path.realpath(os.path.join(root, filename))
    if dest != root and not dest.startswith(root + os.sep):
        raise ValueError(f"Refusing to write outside the uploads directory: {filename!r}")
    return dest


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

            name = _safe_dataset_name(
                kwargs.get('name') or os.path.splitext(os.path.basename(src_path))[0]
            )
            dest_path = _dest_within(upload_dir, f'{name}{ext}')
            shutil.copy2(src_path, dest_path)
            file_path = dest_path
        else:
            # --- Content mode: write inline text to file ---
            file_format = str(kwargs.get('format') or 'csv').lower().lstrip('.')
            if file_format not in _CONTENT_FORMATS:
                return {
                    'status': 'error',
                    'error': (
                        f"Unsupported format '{file_format}'. "
                        f"Supported: {sorted(_CONTENT_FORMATS)}"
                    ),
                    'error_type': 'ValueError'
                }
            name = _safe_dataset_name(
                kwargs.get('name') or f'uploaded_{uuid.uuid4().hex[:8]}'
            )
            file_path = _dest_within(upload_dir, f'{name}.{file_format}')
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
            # The path is confined to the uploads directory by _dest_within, so
            # this cannot delete anything the caller chose. Ignore a failed
            # cleanup: reporting why the dataset was invalid is more useful than
            # replacing that with an OSError about the tidy-up.
            try:
                os.remove(file_path)
            except OSError:
                pass
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
                    "pattern": "^[A-Za-z0-9][A-Za-z0-9._-]{0,99}$",
                    "description": (
                        "Optional name for the dataset (without extension). "
                        "Letters, digits, dots, dashes and underscores only -- "
                        "it becomes a filename, so it cannot contain a path."
                    )
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
