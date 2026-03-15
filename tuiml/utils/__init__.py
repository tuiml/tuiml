"""Utility modules and helper functions.

The ``tuiml.utils`` module provides common infrastructure for model
management, including serialization, checkpointing, and cross-platform
export tools.

Overview
--------
1. **Serialization**: High-level functions for saving and loading models
   with metadata tracking (pickle, joblib, compressed).
2. **Checkpointing**: Automated management of model snapshots during
   long-running training tasks.
3. **Interoperability**: Exporters for industry-standard formats like
   ONNX.
"""

from .serialization import (
    ModelSerializer,
    ModelCheckpoint,
    save_model,
    load_model,
    load_model_info,
    export_to_onnx,
)

__all__ = [
    # Serialization
    "ModelSerializer",
    "ModelCheckpoint",
    "save_model",
    "load_model",
    "load_model_info",
    "export_to_onnx",
]
