"""CapyMOA Streaming Classifiers."""

import numpy as np
from typing import Dict, List, Any, Optional

from tuiml.base.algorithms import Classifier, classifier

try:
    from capymoa.classifier import NaiveBayes, SGDClassifier, StreamingRandomPatches
    from capymoa.stream import NumpyStream
    CAPYMOA_AVAILABLE = True
except ImportError:
    NaiveBayes = SGDClassifier = StreamingRandomPatches = NumpyStream = None
    CAPYMOA_AVAILABLE = False


class _BaseCapyMOAClassifier(Classifier):
    """Base class for CapyMOA classifiers."""
    
    def __init__(self):
        super().__init__()
        if not CAPYMOA_AVAILABLE:
            raise ImportError("CapyMOA is not installed. Install it with: pip install capymoa")
        self.model_ = None
        self.classes_ = None
        self._schema_initialized = False

    @classmethod
    def get_capabilities(cls) -> List[str]: return ["numeric", "streaming"]

    def _initialize_model_if_needed(self, X: np.ndarray, y: np.ndarray):
        if not self._schema_initialized:
            if self.classes_ is None:
                self.classes_ = np.unique(y)
            dummy_stream = NumpyStream(X, y, target_type='categorical')
            schema = dummy_stream.get_schema()
            self._init_capymoa_model(schema)
            self._schema_initialized = True

    def _init_capymoa_model(self, schema):
        raise NotImplementedError

    def partial_fit(self, X: np.ndarray, y: Optional[np.ndarray] = None, classes: Optional[np.ndarray] = None) -> "Algorithm":
        if y is None: raise ValueError("y is required")
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        if classes is not None: self.classes_ = np.asarray(classes)

        self._initialize_model_if_needed(X, y)

        stream = NumpyStream(X, y, target_type='categorical')
        while stream.has_more_instances():
            instance = stream.next_instance()
            self.model_.train(instance)
            
        self._is_fitted = True
        return self

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "Algorithm":
        if y is None: raise ValueError("y is required")
        self._schema_initialized = False
        self.model_ = None
        self.classes_ = np.unique(y)
        return self.partial_fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check_is_fitted()
        X = np.asarray(X, dtype=float)
        dummy_y = np.zeros(len(X), dtype=int)
        stream = NumpyStream(X, dummy_y, target_type='categorical')
        preds = []
        while stream.has_more_instances():
            instance = stream.next_instance()
            pred = self.model_.predict(instance)
            if pred is None: pred = 0
            preds.append(pred)
        return np.array(preds)


@classifier(tags=["streaming", "bayesian", "capymoa"], version="1.0.0")
class CapyMOANaiveBayes(_BaseCapyMOAClassifier):
    """CapyMOA Naive Bayes Classifier."""
    
    def _init_capymoa_model(self, schema):
        self.model_ = NaiveBayes(schema=schema)

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Any]: return {}


@classifier(tags=["streaming", "linear", "capymoa"], version="1.0.0")
class CapyMOASGDClassifier(_BaseCapyMOAClassifier):
    """CapyMOA SGD Classifier."""
    
    def __init__(self, learning_rate: float = 0.01, lambda_param: float = 0.0001):
        super().__init__()
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param

    def _init_capymoa_model(self, schema):
        self.model_ = SGDClassifier(schema=schema, learning_rate=self.learning_rate, lambda_param=self.lambda_param)

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Any]:
        return {
            "learning_rate": {"type": "number", "default": 0.01},
            "lambda_param": {"type": "number", "default": 0.0001}
        }


@classifier(tags=["streaming", "ensemble", "capymoa"], version="1.0.0")
class StreamingRandomPatchesClassifier(_BaseCapyMOAClassifier):
    """CapyMOA Streaming Random Patches Classifier."""
    
    def __init__(self, ensemble_size: int = 100):
        super().__init__()
        self.ensemble_size = ensemble_size

    def _init_capymoa_model(self, schema):
        self.model_ = StreamingRandomPatches(schema=schema, ensemble_size=self.ensemble_size)

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Any]:
        return {"ensemble_size": {"type": "integer", "default": 100}}
