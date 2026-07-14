"""CapyMOA Hoeffding Adaptive Tree Classifier."""

import numpy as np
from typing import Dict, List, Any, Optional

from tuiml.base.algorithms import Algorithm, Classifier, classifier

try:
    from capymoa.classifier import HoeffdingAdaptiveTree
    from capymoa.stream import NumpyStream
    CAPYMOA_AVAILABLE = True
except ImportError:
    HoeffdingAdaptiveTree = None
    CAPYMOA_AVAILABLE = False


@classifier(tags=["streaming", "tree", "capymoa", "drift-adaptation"], version="1.0.0")
class HoeffdingAdaptiveTreeClassifier(Classifier):
    """Hoeffding Adaptive Tree for evolving data streams."""

    def __init__(
        self,
        grace_period: int = 200,
        split_criterion: str = "info_gain",
        split_confidence: float = 1e-7,
        tie_threshold: float = 0.05
    ):
        super().__init__()
        if not CAPYMOA_AVAILABLE: raise ImportError("CapyMOA is not installed.")
        self.grace_period = grace_period
        self.split_criterion = split_criterion
        self.split_confidence = split_confidence
        self.tie_threshold = tie_threshold
        self.model_ = None
        self.classes_ = None
        self._schema_initialized = False

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "grace_period": {"type": "integer", "default": 200},
            "split_criterion": {"type": "string", "default": "info_gain", "enum": ["info_gain", "gini"]},
            "split_confidence": {"type": "number", "default": 1e-7},
            "tie_threshold": {"type": "number", "default": 0.05}
        }

    @classmethod
    def get_capabilities(cls) -> List[str]: return ["numeric", "nominal", "binary_class", "multiclass", "streaming"]

    def _initialize_model_if_needed(self, X: np.ndarray, y: np.ndarray):
        if not self._schema_initialized:
            if self.classes_ is None: self.classes_ = np.unique(y)
            split_criterion_str = 'InfoGainSplitCriterion' if self.split_criterion == "info_gain" else 'GiniSplitCriterion'
            dummy_stream = NumpyStream(X, y)
            self.model_ = HoeffdingAdaptiveTree(
                schema=dummy_stream.get_schema(),
                grace_period=self.grace_period,
                split_criterion=split_criterion_str,
                confidence=self.split_confidence,
                tie_threshold=self.tie_threshold
            )
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
