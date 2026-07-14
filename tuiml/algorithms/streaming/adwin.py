"""CapyMOA ADWIN Drift Detector."""

import numpy as np
from typing import Dict, List, Any, Optional

from tuiml.base.algorithms import Algorithm, Classifier, classifier

try:
    from capymoa.drift.detectors import ADWIN
    CAPYMOA_AVAILABLE = True
except ImportError:
    ADWIN = None
    CAPYMOA_AVAILABLE = False

@classifier(tags=["streaming", "drift-detector", "capymoa"], version="1.0.0")
class ADWINDetector(Classifier):
    """
    ADWIN (ADaptive WINdowing) Drift Detector.

    An adaptive sliding window algorithm for detecting concept drift in data streams.
    It automatically grows the window when there is no apparent change, and shrinks it
    when a change is detected.

    Parameters
    ----------
    delta : float, default=0.002
        The delta parameter for the ADWIN algorithm, which acts as a confidence 
        value. A smaller delta means fewer false positives but slower detection.

    Attributes
    ----------
    detector_ : capymoa.drift.detectors.ADWIN
        The underlying CapyMOA ADWIN object.
    detected_change_ : bool
        Whether a change was detected in the last updated element.
    """
    
    # Using 'detector' type as a custom extension since drift detectors 
    # don't fit perfectly into classifier/regressor/clusterer.
    _algorithm_type = "detector"

    def __init__(self, delta: float = 0.002):
        super().__init__()
        
        if not CAPYMOA_AVAILABLE:
            raise ImportError(
                "CapyMOA is not installed. Install it with: pip install capymoa"
            )
            
        self.delta = delta
        self.detector_ = ADWIN(delta=self.delta)
        self.detected_change_ = False

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        """Return parameter schema."""
        return {
            "delta": {
                "type": "number",
                "default": 0.002,
                "minimum": 0.0,
                "maximum": 1.0,
                "description": "Confidence value for drift detection"
            }
        }

    @classmethod
    def get_capabilities(cls) -> List[str]:
        """Return capabilities."""
        return ["numeric", "streaming"]

    @classmethod
    def get_complexity(cls) -> str:
        """Return time/space complexity."""
        return "O(1) time per element amortized, O(log W) space"

    @classmethod
    def get_references(cls) -> List[str]:
        """Return references."""
        return [
            "Bifet, A., & Gavaldà, R. (2007). Learning from time-changing data with adaptive windowing."
        ]

    def update(self, value: float) -> bool:
        """
        Update the detector with a new value and check for drift.

        Parameters
        ----------
        value : float
            The new value to add to the ADWIN window.

        Returns
        -------
        bool
            True if drift is detected, False otherwise.
        """
        self.detector_.add_element(value)
        self.detected_change_ = self.detector_.detected_change()
        self._is_fitted = True
        return self.detected_change_

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "Algorithm":
        """
        Process a batch of data to find drift points.

        This is provided for API compatibility. It processes the elements sequentially
        and returns self. To see if drift was detected at each step, use `update` iteratively.
        """
        X = np.asarray(X).flatten()
        for val in X:
            self.update(float(val))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Process a batch of data and return a boolean array indicating drift locations.

        Parameters
        ----------
        X : np.ndarray
            The input data stream.

        Returns
        -------
        np.ndarray
            Boolean array where True indicates drift was detected at that index.
        """
        X = np.asarray(X).flatten()
        drift_locations = np.zeros(len(X), dtype=bool)
        
        for i, val in enumerate(X):
            drift_locations[i] = self.update(float(val))
            
        return drift_locations

    def __repr__(self) -> str:
        """String representation."""
        return f"ADWINDetector(delta={self.delta})"
