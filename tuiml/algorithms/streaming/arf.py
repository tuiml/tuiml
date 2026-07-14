"""CapyMOA Adaptive Random Forest Classifier."""

import numpy as np
from typing import Dict, List, Any, Optional

from tuiml.base.algorithms import Algorithm, Classifier, classifier

try:
    from capymoa.classifier import AdaptiveRandomForestClassifier as CapyMOA_ARF
    from capymoa.stream import NumpyStream
    CAPYMOA_AVAILABLE = True
except ImportError:
    CapyMOA_ARF = None
    CAPYMOA_AVAILABLE = False

@classifier(tags=["streaming", "ensemble", "random-forest", "capymoa"], version="1.0.0")
class AdaptiveRandomForestClassifier(Classifier):
    """
    Adaptive Random Forest Classifier using CapyMOA.

    An ensemble of Hoeffding Trees for data streams that adapts to concept drift 
    using ADWIN and employs online bagging.

    Parameters
    ----------
    ensemble_size : int, default=100
        The number of trees in the ensemble.
    max_features : int or str, default="sqrt"
        The number of features to consider when looking for the best split.
    disable_drift_detection : bool, default=False
        If True, disables the drift detection mechanism (ADWIN).
    disable_background_learner : bool, default=False
        If True, disables background learners used for drift adaptation.

    Attributes
    ----------
    model_ : capymoa.classifier.AdaptiveRandomForestClassifier
        The underlying CapyMOA ARF model.
    classes_ : np.ndarray
        Unique class labels observed.
    """

    def __init__(
        self,
        ensemble_size: int = 100,
        max_features: Any = "sqrt",
        disable_drift_detection: bool = False,
        disable_background_learner: bool = False
    ):
        super().__init__()
        
        if not CAPYMOA_AVAILABLE:
            raise ImportError(
                "CapyMOA is not installed. Install it with: pip install capymoa"
            )
            
        self.ensemble_size = ensemble_size
        self.max_features = max_features
        self.disable_drift_detection = disable_drift_detection
        self.disable_background_learner = disable_background_learner

        self.model_ = None
        self.classes_ = None
        self._schema_initialized = False

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        """Return parameter schema."""
        return {
            "ensemble_size": {
                "type": "integer",
                "default": 100,
                "minimum": 1,
                "description": "Number of trees in the ensemble"
            },
            "disable_drift_detection": {
                "type": "boolean",
                "default": False,
                "description": "Disable drift detection (ADWIN)"
            },
            "disable_background_learner": {
                "type": "boolean",
                "default": False,
                "description": "Disable background learners"
            }
        }

    @classmethod
    def get_capabilities(cls) -> List[str]:
        """Return capabilities."""
        return ["numeric", "nominal", "binary_class", "multiclass", "streaming"]

    @classmethod
    def get_complexity(cls) -> str:
        """Return time/space complexity."""
        return "O(T) time per instance (T=trees), O(T * leaves * features) space"

    @classmethod
    def get_references(cls) -> List[str]:
        """Return references."""
        return [
            "Gomes, H. M., et al. (2017). Adaptive random forests for evolving data stream classification. Machine Learning."
        ]

    def _initialize_model_if_needed(self, X: np.ndarray, y: np.ndarray):
        """Initialize the CapyMOA model and schema if this is the first batch."""
        if not self._schema_initialized:
            if self.classes_ is None:
                self.classes_ = np.unique(y)
                
            dummy_stream = NumpyStream(X, y)
            schema = dummy_stream.get_schema()
            
            # Map max_features strings if needed, CapyMOA usually accepts string or int
            m_features = 2 if self.max_features == "sqrt" else self.max_features
            if isinstance(m_features, str):
                m_features = 2  # Fallback
            
            self.model_ = CapyMOA_ARF(
                schema=schema,
                ensemble_size=self.ensemble_size,
                # max_features is handled via m_features_per_tree argument in moa
                disable_drift_detection=self.disable_drift_detection,
                disable_background_learner=self.disable_background_learner
            )
            self._schema_initialized = True

    def partial_fit(self, X: np.ndarray, y: Optional[np.ndarray] = None, classes: Optional[np.ndarray] = None) -> "Algorithm":
        """Incrementally train the ARF on a batch of samples."""
        if y is None:
            raise ValueError("y is required for supervised learning")
            
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        
        if classes is not None:
            self.classes_ = np.asarray(classes)

        self._initialize_model_if_needed(X, y)

        stream = NumpyStream(X, y)
        while stream.has_more_instances():
            instance = stream.next_instance()
            self.model_.train(instance)
            
        self._is_fitted = True
        return self

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "Algorithm":
        """Train the model from scratch on the provided data."""
        if y is None:
            raise ValueError("y is required for supervised learning")
        self._schema_initialized = False
        self.model_ = None
        self.classes_ = np.unique(y)
        return self.partial_fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for X."""
        self._check_is_fitted()
        X = np.asarray(X, dtype=float)
        
        dummy_y = np.zeros(len(X), dtype=int)
        stream = NumpyStream(X, dummy_y)
        
        preds = []
        while stream.has_more_instances():
            instance = stream.next_instance()
            pred = self.model_.predict(instance)
            if pred is None:
                pred = 0
            preds.append(pred)
            
        return np.array(preds)

    def __repr__(self) -> str:
        """String representation."""
        return f"AdaptiveRandomForestClassifier(ensemble_size={self.ensemble_size})"
