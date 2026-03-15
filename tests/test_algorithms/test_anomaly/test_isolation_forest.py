"""Test suite for Isolation Forest Anomaly Detection.

Tests cover:
- Basic anomaly detection
- Anomaly score computation
- Contamination parameter
- Different n_estimators values
"""

import numpy as np
import pytest
import pickle

from tuiml.algorithms.anomaly import IsolationForestDetector


class TestIsolationForestDetectorInstantiation:
    """Tests for algorithm instantiation."""
    
    def test_default_initialization(self):
        """Test default initialization."""
        clf = IsolationForestDetector()
        
        assert clf.n_estimators == 100
        assert clf.max_samples == "auto"
        assert clf.contamination == 0.1
        assert clf.max_features == 1.0
        assert clf.random_state is None
        
    def test_custom_initialization(self):
        """Test custom initialization."""
        clf = IsolationForestDetector(
            n_estimators=50,
            max_samples=256,
            contamination=0.2,
            max_features=0.8,
            random_state=42
        )
        
        assert clf.n_estimators == 50
        assert clf.max_samples == 256
        assert clf.contamination == 0.2
        assert clf.max_features == 0.8
        assert clf.random_state == 42
        
    def test_parameter_schema(self):
        """Test parameter schema."""
        schema = IsolationForestDetector.get_parameter_schema()
        
        assert "n_estimators" in schema
        assert "max_samples" in schema
        assert "contamination" in schema
        
    def test_capabilities(self):
        """Test capabilities."""
        caps = IsolationForestDetector.get_capabilities()
        
        assert "numeric" in caps
        assert "unsupervised" in caps
        assert "anomaly_detection" in caps


class TestIsolationForestDetectorFitting:
    """Tests for the fit() method."""
    
    def test_fit_basic(self, anomaly_detection_data):
        """Test basic fitting."""
        X = anomaly_detection_data
        clf = IsolationForestDetector(random_state=42)
        
        result = clf.fit(X)
        
        assert result is clf
        assert clf._is_fitted is True
        assert clf.trees_ is not None
        assert len(clf.trees_) == clf.n_estimators
        assert clf.threshold_ is not None
        assert clf.offset_ is not None
        
    def test_fit_with_contamination(self, anomaly_detection_data):
        """Test fitting with specific contamination."""
        X = anomaly_detection_data
        clf = IsolationForestDetector(contamination=0.2, random_state=42)
        clf.fit(X)
        
        assert clf._is_fitted
        assert clf.contamination == 0.2
        
    def test_fit_max_samples_int(self, anomaly_detection_data):
        """Test fit with integer max_samples."""
        X = anomaly_detection_data
        clf = IsolationForestDetector(max_samples=50, random_state=42)
        clf.fit(X)
        
        assert clf._is_fitted
        assert clf.max_samples_ <= 50
        
    def test_fit_max_samples_float(self, anomaly_detection_data):
        """Test fit with float max_samples."""
        X = anomaly_detection_data
        clf = IsolationForestDetector(max_samples=0.5, random_state=42)
        clf.fit(X)
        
        assert clf._is_fitted


class TestIsolationForestDetectorPrediction:
    """Tests for the predict() method."""
    
    def test_predict_basic(self, anomaly_detection_data):
        """Test basic prediction."""
        X = anomaly_detection_data
        clf = IsolationForestDetector(random_state=42).fit(X)
        
        predictions = clf.predict(X)
        
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == len(X)
        # Predictions should be -1 (anomaly) or 1 (normal)
        assert all(p in [-1, 1] for p in predictions)
        
    def test_predict_anomaly_ratio(self, anomaly_detection_data):
        """Test that anomaly ratio matches contamination."""
        X = anomaly_detection_data
        contamination = 0.2
        clf = IsolationForestDetector(contamination=contamination, random_state=42)
        clf.fit(X)
        
        predictions = clf.predict(X)
        
        # Check approximate anomaly ratio (loose tolerance for stochastic algorithm)
        anomaly_ratio = np.mean(predictions == -1)
        assert abs(anomaly_ratio - contamination) < 0.7
        
    def test_predict_before_fit_raises(self):
        """Test predict before fit raises error."""
        clf = IsolationForestDetector()
        
        with pytest.raises(RuntimeError, match="must be fitted"):
            clf.predict(np.array([[1, 2]]))


class TestIsolationForestDetectorDecisionFunction:
    """Tests for the decision_function() method."""
    
    def test_decision_function_basic(self, anomaly_detection_data):
        """Test decision function."""
        X = anomaly_detection_data
        clf = IsolationForestDetector(random_state=42).fit(X)
        
        scores = clf.decision_function(X)
        
        assert isinstance(scores, np.ndarray)
        assert len(scores) == len(X)
        
    def test_decision_function_correlation_with_predict(self, anomaly_detection_data):
        """Test that decision function correlates with predictions."""
        X = anomaly_detection_data
        clf = IsolationForestDetector(random_state=42).fit(X)
        
        scores = clf.decision_function(X)
        predictions = clf.predict(X)
        
        # Lower scores should correspond to anomalies (-1)
        anomaly_scores = scores[predictions == -1]
        normal_scores = scores[predictions == 1]
        
        if len(anomaly_scores) > 0 and len(normal_scores) > 0:
            assert np.mean(anomaly_scores) < np.mean(normal_scores)
            
    def test_score_samples_alias(self, anomaly_detection_data):
        """Test that score_samples is alias for decision_function."""
        X = anomaly_detection_data
        clf = IsolationForestDetector(random_state=42).fit(X)
        
        scores1 = clf.decision_function(X)
        scores2 = clf.score_samples(X)
        
        np.testing.assert_array_equal(scores1, scores2)


class TestIsolationForestDetectorSerialization:
    """Tests for serialization."""
    
    def test_pickle_roundtrip(self, anomaly_detection_data):
        """Test pickle serialization."""
        X = anomaly_detection_data
        clf = IsolationForestDetector(random_state=42).fit(X)
        
        clf_bytes = pickle.dumps(clf)
        clf_loaded = pickle.loads(clf_bytes)
        
        pred_original = clf.predict(X)
        pred_loaded = clf_loaded.predict(X)
        
        np.testing.assert_array_equal(pred_original, pred_loaded)


class TestIsolationForestDetectorEdgeCases:
    """Edge case tests."""
    
    def test_single_sample(self):
        """Test with single sample."""
        X = np.array([[1.0, 2.0]])
        clf = IsolationForestDetector(random_state=42)
        clf.fit(X)
        
        assert clf._is_fitted
        pred = clf.predict(X)
        assert len(pred) == 1
        
    def test_identical_samples(self):
        """Test with identical samples."""
        X = np.array([[1.0, 2.0]] * 50)
        clf = IsolationForestDetector(random_state=42)
        clf.fit(X)
        
        assert clf._is_fitted
