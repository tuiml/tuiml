"""CapyMOA Hoeffding Tree Classifier."""

import numpy as np
from typing import Dict, List, Any, Optional

from tuiml.base.algorithms import Algorithm, Classifier, classifier

try:
    from capymoa.classifier import HoeffdingTree
    from capymoa.instance import Instance
    from capymoa.stream import NumpyStream
    CAPYMOA_AVAILABLE = True
except ImportError:
    HoeffdingTree = None
    CAPYMOA_AVAILABLE = False

@classifier(tags=["streaming", "tree", "capymoa"], version="1.0.0")
class HoeffdingTreeClassifier(Classifier):
    """
    Hoeffding Tree Classifier using CapyMOA.

    An incremental decision tree for data streams. It uses the Hoeffding bound
    to decide the optimal number of instances needed to split a node.

    Parameters
    ----------
    grace_period : int, default=200
        The number of instances a leaf should observe between split attempts.
    split_criterion : str, default="info_gain"
        Split criterion to use ('info_gain' or 'gini').
    split_confidence : float, default=1e-7
        The allowable error in split decision (1 - alpha).
    tie_threshold : float, default=0.05
        Threshold below which a split will be forced to break ties.

    Attributes
    ----------
    model_ : capymoa.classifier.HoeffdingTree
        The underlying CapyMOA model.
    classes_ : np.ndarray
        Unique class labels observed.
    """

    def __init__(
        self,
        grace_period: int = 200,
        split_criterion: str = "info_gain",
        split_confidence: float = 1e-7,
        tie_threshold: float = 0.05
    ):
        super().__init__()
        
        if not CAPYMOA_AVAILABLE:
            raise ImportError(
                "CapyMOA is not installed. Install it with: pip install capymoa"
            )
            
        self.grace_period = grace_period
        self.split_criterion = split_criterion
        self.split_confidence = split_confidence
        self.tie_threshold = tie_threshold

        self.model_ = None
        self.classes_ = None
        self._schema_initialized = False

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        """Return parameter schema."""
        return {
            "grace_period": {
                "type": "integer",
                "default": 200,
                "minimum": 1,
                "description": "Instances observed between split attempts"
            },
            "split_criterion": {
                "type": "string",
                "default": "info_gain",
                "enum": ["info_gain", "gini"],
                "description": "Criterion for splitting"
            },
            "split_confidence": {
                "type": "number",
                "default": 1e-7,
                "minimum": 0.0,
                "maximum": 1.0,
                "description": "Allowable error in split decision"
            },
            "tie_threshold": {
                "type": "number",
                "default": 0.05,
                "minimum": 0.0,
                "description": "Threshold to break ties"
            }
        }

    @classmethod
    def get_capabilities(cls) -> List[str]:
        """Return capabilities."""
        return ["numeric", "nominal", "binary_class", "multiclass", "streaming"]

    @classmethod
    def get_complexity(cls) -> str:
        """Return time/space complexity."""
        return "O(1) time per instance, O(leaves * classes * features) space"

    @classmethod
    def get_references(cls) -> List[str]:
        """Return references."""
        return [
            "Domingos, P., & Hulten, G. (2000). Mining high-speed data streams. KDD."
        ]

    def _initialize_model_if_needed(self, X: np.ndarray, y: np.ndarray):
        """Initialize the CapyMOA model and schema if this is the first batch."""
        if not self._schema_initialized:
            # We need to create a schema for CapyMOA. The easiest way is to use 
            # stream_from_numpy to generate a temporary stream just to extract the schema.
            # However, for pure partial_fit we might not know all classes upfront.
            # We assume classes are represented as integers 0..n_classes-1.
            
            # Infer number of classes if possible, or just default to 2
            if self.classes_ is None:
                self.classes_ = np.unique(y)
                
            n_classes = max(len(self.classes_), int(np.max(y)) + 1)
            
            # Initialize CapyMOA HoeffdingTree
            # Note: We create a dummy stream just to extract its schema for HT initialization
            # CapyMOA models often require a Schema on initialization.
            # Map string split criterion to CapyMOA string
            split_criterion_str = 'InfoGainSplitCriterion' if self.split_criterion == "info_gain" else 'GiniSplitCriterion'
            dummy_stream = NumpyStream(X, y)
            schema = dummy_stream.get_schema()
            
            self.model_ = HoeffdingTree(
                schema=schema,
                grace_period=self.grace_period,
                split_criterion=split_criterion_str,
                confidence=self.split_confidence,
                tie_threshold=self.tie_threshold
            )
            self._schema_initialized = True

    def partial_fit(self, X: np.ndarray, y: Optional[np.ndarray] = None, classes: Optional[np.ndarray] = None) -> "Algorithm":
        """
        Incrementally train the Hoeffding Tree on a batch of samples.

        Parameters
        ----------
        X : np.ndarray
            Features.
        y : np.ndarray
            Labels.
        classes : np.ndarray, optional
            All possible class labels.

        Returns
        -------
        self
        """
        if y is None:
            raise ValueError("y is required for supervised learning")
            
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        
        if classes is not None:
            self.classes_ = np.asarray(classes)

        self._initialize_model_if_needed(X, y)

        # Convert numpy arrays to CapyMOA stream instances
        stream = NumpyStream(X, y)
        while stream.has_more_instances():
            instance = stream.next_instance()
            self.model_.train(instance)
            
        self._is_fitted = True
        return self

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "Algorithm":
        """
        Train the model from scratch on the provided data.

        For a streaming algorithm, this is equivalent to partial_fit.
        """
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
        
        # We need dummy labels for NumpyStream
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
        return f"HoeffdingTreeClassifier(grace_period={self.grace_period})"
