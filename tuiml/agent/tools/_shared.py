"""Helpers shared by more than one tool executor.

Model persistence and dataset resolution: every executor that takes a
``model_id`` / ``model_path`` / ``data`` argument routes through here, so the
resolution rules stay identical across tools.
"""

import os
from typing import List

from ._state import _DATASET_INDEX, _MODEL_INDEX, _MODELS_DIR


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
