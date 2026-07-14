"""CapyMOA Streaming Regressors."""

import numpy as np
from typing import Dict, List, Any, Optional

from tuiml.base.algorithms import Regressor, regressor

try:
    from capymoa.regressor import SGDRegressor, AdaptiveRandomForestRegressor, FIMTDD
    from capymoa.stream import NumpyStream
    CAPYMOA_AVAILABLE = True
except ImportError:
    SGDRegressor = AdaptiveRandomForestRegressor = FIMTDD = NumpyStream = None
    CAPYMOA_AVAILABLE = False


class _BaseCapyMOARegressor(Regressor):
    """Base class for CapyMOA regressors."""
    
    def __init__(self):
        super().__init__()
        if not CAPYMOA_AVAILABLE:
            raise ImportError("CapyMOA is not installed. Install it with: pip install capymoa")
        self.model_ = None
        self._schema_initialized = False

    @classmethod
    def get_capabilities(cls) -> List[str]: return ["numeric", "streaming"]

    def _initialize_model_if_needed(self, X: np.ndarray, y: np.ndarray):
        if not self._schema_initialized:
            dummy_stream = NumpyStream(X, y, target_type='numeric')
            schema = dummy_stream.get_schema()
            self._init_capymoa_model(schema)
            self._schema_initialized = True

    def _init_capymoa_model(self, schema):
        raise NotImplementedError

    def partial_fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "Algorithm":
        if y is None: raise ValueError("y is required")
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        self._initialize_model_if_needed(X, y)

        stream = NumpyStream(X, y, target_type='numeric')
        while stream.has_more_instances():
            instance = stream.next_instance()
            self.model_.train(instance)
            
        self._is_fitted = True
        return self

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "Algorithm":
        if y is None: raise ValueError("y is required")
        self._schema_initialized = False
        self.model_ = None
        return self.partial_fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        self._check_is_fitted()
        X = np.asarray(X, dtype=float)
        dummy_y = np.zeros(len(X), dtype=float)
        stream = NumpyStream(X, dummy_y, target_type='numeric')
        preds = []
        while stream.has_more_instances():
            instance = stream.next_instance()
            pred = self.model_.predict(instance)
            if pred is None: pred = 0.0
            preds.append(pred)
        return np.array(preds)


@regressor(tags=["streaming", "linear", "capymoa"], version="1.0.0")
class CapyMOASGDRegressor(_BaseCapyMOARegressor):
    """CapyMOA SGD Regressor."""
    
    def __init__(self, learning_rate: float = 0.01):
        super().__init__()
        self.learning_rate = learning_rate

    def _init_capymoa_model(self, schema):
        self.model_ = SGDRegressor(schema=schema, learning_rate=self.learning_rate)

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Any]:
        return {"learning_rate": {"type": "number", "default": 0.01}}


@regressor(tags=["streaming", "ensemble", "trees", "capymoa"], version="1.0.0")
class CapyMOAAdaptiveRandomForestRegressor(_BaseCapyMOARegressor):
    """CapyMOA Adaptive Random Forest Regressor."""
    
    def __init__(self, ensemble_size: int = 100):
        super().__init__()
        self.ensemble_size = ensemble_size

    def _init_capymoa_model(self, schema):
        self.model_ = AdaptiveRandomForestRegressor(schema=schema, ensemble_size=self.ensemble_size)

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Any]:
        return {"ensemble_size": {"type": "integer", "default": 100}}


@regressor(tags=["streaming", "tree", "capymoa"], version="1.0.0")
class CapyMOAFIMTDD(_BaseCapyMOARegressor):
    """CapyMOA Fast Incremental Model Tree with Drift Detection (FIMT-DD)."""
    
    def __init__(self, grace_period: int = 200, split_confidence: float = 1e-7):
        super().__init__()
        self.grace_period = grace_period
        self.split_confidence = split_confidence

    def _init_capymoa_model(self, schema):
        self.model_ = FIMTDD(schema=schema, grace_period=self.grace_period, split_confidence=self.split_confidence)

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Any]:
        return {
            "grace_period": {"type": "integer", "default": 200},
            "split_confidence": {"type": "number", "default": 1e-7}
        }
