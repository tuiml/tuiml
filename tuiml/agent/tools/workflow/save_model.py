"""Model export to a user path."""

import os
from typing import Any, Dict

from .._spec import ToolSpec
from .._state import _MODEL_INDEX


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


SPEC = ToolSpec(
    name='tuiml_save_model',
    description="Copy a trained model to a custom path. Use this when the user wants to save or download a model to a specific location.",
    input_schema={
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
        },
    output_schema={
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
    execute=execute_save_model,
    group='workflow',
    read_only=False, destructive=False,
    idempotent=True, open_world=False,
)
