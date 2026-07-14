"""CapyMOA DDM and EDDM Drift Detectors."""

import numpy as np
from typing import Dict, List, Any, Optional

from tuiml.base.algorithms import Algorithm, Classifier, classifier

try:
    from capymoa.drift.detectors import DDM, EDDM
    CAPYMOA_AVAILABLE = True
except ImportError:
    DDM = EDDM = None
    CAPYMOA_AVAILABLE = False


@classifier(tags=["streaming", "drift-detector", "capymoa"], version="1.0.0")
class DDMDetector(Classifier):
    """
    DDM (Drift Detection Method).
    """
    _algorithm_type = "detector"

    def __init__(self, min_num_instances: int = 30):
        super().__init__()
        if not CAPYMOA_AVAILABLE: raise ImportError("CapyMOA is not installed.")
        self.min_num_instances = min_num_instances
        self.detector_ = DDM()
        self.detected_change_ = False

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        return {"min_num_instances": {"type": "integer", "default": 30}}

    @classmethod
    def get_capabilities(cls) -> List[str]: return ["numeric", "streaming"]

    def update(self, value: float) -> bool:
        self.detector_.add_element(value)
        self.detected_change_ = self.detector_.detected_change()
        self._is_fitted = True
        return self.detected_change_

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "Algorithm":
        X = np.asarray(X).flatten()
        for val in X: self.update(float(val))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X).flatten()
        drift_locations = np.zeros(len(X), dtype=bool)
        for i, val in enumerate(X):
            drift_locations[i] = self.update(float(val))
        return drift_locations


@classifier(tags=["streaming", "drift-detector", "capymoa"], version="1.0.0")
class EDDMDetector(Classifier):
    """
    EDDM (Early Drift Detection Method).
    """
    _algorithm_type = "detector"

    def __init__(self):
        super().__init__()
        if not CAPYMOA_AVAILABLE: raise ImportError("CapyMOA is not installed.")
        self.detector_ = EDDM()
        self.detected_change_ = False

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        return {}

    @classmethod
    def get_capabilities(cls) -> List[str]: return ["numeric", "streaming"]

    def update(self, value: float) -> bool:
        self.detector_.add_element(value)
        self.detected_change_ = self.detector_.detected_change()
        self._is_fitted = True
        return self.detected_change_

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "Algorithm":
        X = np.asarray(X).flatten()
        for val in X: self.update(float(val))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X).flatten()
        drift_locations = np.zeros(len(X), dtype=bool)
        for i, val in enumerate(X):
            drift_locations[i] = self.update(float(val))
        return drift_locations
