"""
Model Manager - In-memory model loading and caching for serving.

Handles loading trained models into memory, caching for performance,
and provides a unified interface for predictions.
"""

import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from collections import OrderedDict
from datetime import datetime
import numpy as np

from tuiml.utils.serialization import load_model, load_model_info


class ModelManager:
    """
    Manage loaded models for serving predictions.

    Provides thread-safe model loading, caching, and prediction interface
    with automatic LRU eviction when cache is full.

    Parameters
    ----------
    max_models : int, default=10
        Maximum number of models to keep in memory.
        Older models are evicted when limit is reached.

    Examples
    --------
    >>> manager = ModelManager(max_models=5)
    >>> manager.load("classifier", "model.pkl")
    >>> predictions = manager.predict("classifier", X)
    >>> manager.list_models()
    ['classifier']
    """

    def __init__(self, max_models: int = 10):
        self.max_models = max_models
        self._models: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._lock = threading.RLock()

    def load(
        self,
        model_id: str,
        path: Union[str, Path],
        metadata: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Load a model into memory.

        Parameters
        ----------
        model_id : str
            Unique identifier for the model.
        path : str or Path
            Path to the saved model file.
        metadata : dict, optional
            Additional metadata to associate with the model.

        Returns
        -------
        dict
            Model info including class name, parameters, and load time.

        Raises
        ------
        FileNotFoundError
            If the model file does not exist.
        """
        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")

        with self._lock:
            # Load model and metadata
            model = load_model(path)
            info = load_model_info(path)

            # Evict oldest model if at capacity
            while len(self._models) >= self.max_models:
                oldest_id = next(iter(self._models))
                del self._models[oldest_id]

            # Store model entry
            self._models[model_id] = {
                "model": model,
                "path": str(path),
                "info": info,
                "metadata": metadata or {},
                "loaded_at": datetime.now().isoformat(),
                "prediction_count": 0,
            }

            return self.get_model_info(model_id)

    def unload(self, model_id: str) -> bool:
        """
        Unload a model from memory.

        Parameters
        ----------
        model_id : str
            Identifier of the model to unload.

        Returns
        -------
        bool
            True if model was unloaded, False if not found.
        """
        with self._lock:
            if model_id in self._models:
                del self._models[model_id]
                return True
            return False

    def get_model(self, model_id: str) -> Any:
        """
        Get a loaded model instance.

        Parameters
        ----------
        model_id : str
            Identifier of the model.

        Returns
        -------
        object
            The loaded model instance.

        Raises
        ------
        KeyError
            If model is not loaded.
        """
        with self._lock:
            if model_id not in self._models:
                raise KeyError(f"Model '{model_id}' not loaded")

            # Move to end (LRU)
            self._models.move_to_end(model_id)
            return self._models[model_id]["model"]

    def get_model_info(self, model_id: str) -> Dict[str, Any]:
        """
        Get metadata for a loaded model.

        Parameters
        ----------
        model_id : str
            Identifier of the model.

        Returns
        -------
        dict
            Model information including class, parameters, and stats.
        """
        with self._lock:
            if model_id not in self._models:
                raise KeyError(f"Model '{model_id}' not loaded")

            entry = self._models[model_id]
            return {
                "model_id": model_id,
                "model_class": entry["info"].get("model_class"),
                "model_module": entry["info"].get("model_module"),
                "params": entry["info"].get("params", {}),
                "path": entry["path"],
                "loaded_at": entry["loaded_at"],
                "prediction_count": entry["prediction_count"],
                "metadata": entry["metadata"],
            }

    def list_models(self) -> List[str]:
        """
        List all loaded model IDs.

        Returns
        -------
        list of str
            List of model identifiers.
        """
        with self._lock:
            return list(self._models.keys())

    def is_loaded(self, model_id: str) -> bool:
        """Check if a model is loaded."""
        with self._lock:
            return model_id in self._models

    def predict(
        self,
        model_id: str,
        X: Union[np.ndarray, List],
    ) -> np.ndarray:
        """
        Make predictions using a loaded model.

        Parameters
        ----------
        model_id : str
            Identifier of the model to use.
        X : array-like
            Input features for prediction.

        Returns
        -------
        ndarray
            Model predictions.

        Raises
        ------
        KeyError
            If model is not loaded.
        ValueError
            If input format is invalid.
        """
        X = np.asarray(X)

        with self._lock:
            if model_id not in self._models:
                raise KeyError(f"Model '{model_id}' not loaded")

            entry = self._models[model_id]
            model = entry["model"]

            # Update stats
            entry["prediction_count"] += len(X)

        # Make prediction outside lock
        return model.predict(X)

    def predict_proba(
        self,
        model_id: str,
        X: Union[np.ndarray, List],
    ) -> np.ndarray:
        """
        Get prediction probabilities using a loaded model.

        Parameters
        ----------
        model_id : str
            Identifier of the model to use.
        X : array-like
            Input features for prediction.

        Returns
        -------
        ndarray
            Class probabilities with shape (n_samples, n_classes).

        Raises
        ------
        KeyError
            If model is not loaded.
        NotImplementedError
            If model doesn't support probability predictions.
        """
        X = np.asarray(X)

        with self._lock:
            if model_id not in self._models:
                raise KeyError(f"Model '{model_id}' not loaded")

            entry = self._models[model_id]
            model = entry["model"]

            if not hasattr(model, "predict_proba"):
                raise NotImplementedError(
                    f"Model '{model_id}' ({type(model).__name__}) "
                    "does not support probability predictions"
                )

            entry["prediction_count"] += len(X)

        return model.predict_proba(X)

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about loaded models.

        Returns
        -------
        dict
            Statistics including model count and total predictions.
        """
        with self._lock:
            total_predictions = sum(
                entry["prediction_count"]
                for entry in self._models.values()
            )

            return {
                "loaded_models": len(self._models),
                "max_models": self.max_models,
                "total_predictions": total_predictions,
                "models": [
                    {
                        "model_id": model_id,
                        "model_class": entry["info"].get("model_class"),
                        "prediction_count": entry["prediction_count"],
                    }
                    for model_id, entry in self._models.items()
                ],
            }

    def clear(self):
        """Unload all models from memory."""
        with self._lock:
            self._models.clear()
