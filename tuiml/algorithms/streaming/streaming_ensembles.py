"""CapyMOA Streaming Ensembles."""

import numpy as np
from typing import Dict, List, Any, Optional

from tuiml.base.algorithms import Algorithm, Classifier, classifier

try:
    from capymoa.classifier import OzaBag, LeveragingBag
    from capymoa.stream import NumpyStream
    CAPYMOA_AVAILABLE = True
except ImportError:
    OzaBag = LeveragingBag = None
    CAPYMOA_AVAILABLE = False


@classifier(tags=["streaming", "ensemble", "capymoa"], version="1.0.0")
class OzaBagClassifier(Classifier):
    """Oza Bagging for data streams."""

    def __init__(self, ensemble_size: int = 10):
        super().__init__()
        if not CAPYMOA_AVAILABLE: raise ImportError("CapyMOA is not installed.")
        self.ensemble_size = ensemble_size
        self.model_ = None
        self.classes_ = None
        self._schema_initialized = False

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        return {"ensemble_size": {"type": "integer", "default": 10}}

    @classmethod
    def get_capabilities(cls) -> List[str]: return ["numeric", "nominal", "binary_class", "multiclass", "streaming"]

    def _initialize_model_if_needed(self, X: np.ndarray, y: np.ndarray):
        if not self._schema_initialized:
            if self.classes_ is None: self.classes_ = np.unique(y)
            dummy_stream = NumpyStream(X, y)
            self.model_ = OzaBag(schema=dummy_stream.get_schema(), ensemble_size=self.ensemble_size)
            self._schema_initialized = True

    def partial_fit(self, X: np.ndarray, y: Optional[np.ndarray] = None, classes: Optional[np.ndarray] = None) -> "Algorithm":
        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=int)
        if classes is not None: self.classes_ = np.asarray(classes)
        self._initialize_model_if_needed(X, y)
        stream = NumpyStream(X, y)
        while stream.has_more_instances():
            self.model_.train(stream.next_instance())
        self._is_fitted = True
        return self

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "Algorithm":
        self._schema_initialized = False
        self.model_ = None
        self.classes_ = np.unique(y)
        return self.partial_fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check_is_fitted()
        X = np.asarray(X, dtype=float)
        stream = NumpyStream(X, np.zeros(len(X), dtype=int))
        preds = []
        while stream.has_more_instances():
            pred = self.model_.predict(stream.next_instance())
            preds.append(pred if pred is not None else 0)
        return np.array(preds)


@classifier(tags=["streaming", "ensemble", "capymoa"], version="1.0.0")
class LeveragingBagClassifier(Classifier):
    """Leveraging Bagging for data streams."""

    def __init__(self, ensemble_size: int = 10):
        super().__init__()
        if not CAPYMOA_AVAILABLE: raise ImportError("CapyMOA is not installed.")
        self.ensemble_size = ensemble_size
        self.model_ = None
        self.classes_ = None
        self._schema_initialized = False

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        return {"ensemble_size": {"type": "integer", "default": 10}}

    @classmethod
    def get_capabilities(cls) -> List[str]: return ["numeric", "nominal", "binary_class", "multiclass", "streaming"]

    def _initialize_model_if_needed(self, X: np.ndarray, y: np.ndarray):
        if not self._schema_initialized:
            if self.classes_ is None: self.classes_ = np.unique(y)
            dummy_stream = NumpyStream(X, y)
            self.model_ = LeveragingBag(schema=dummy_stream.get_schema(), ensemble_size=self.ensemble_size)
            self._schema_initialized = True

    def partial_fit(self, X: np.ndarray, y: Optional[np.ndarray] = None, classes: Optional[np.ndarray] = None) -> "Algorithm":
        X, y = np.asarray(X, dtype=float), np.asarray(y, dtype=int)
        if classes is not None: self.classes_ = np.asarray(classes)
        self._initialize_model_if_needed(X, y)
        stream = NumpyStream(X, y)
        while stream.has_more_instances():
            self.model_.train(stream.next_instance())
        self._is_fitted = True
        return self

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "Algorithm":
        self._schema_initialized = False
        self.model_ = None
        self.classes_ = np.unique(y)
        return self.partial_fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check_is_fitted()
        X = np.asarray(X, dtype=float)
        stream = NumpyStream(X, np.zeros(len(X), dtype=int))
        preds = []
        while stream.has_more_instances():
            pred = self.model_.predict(stream.next_instance())
            preds.append(pred if pred is not None else 0)
        return np.array(preds)
