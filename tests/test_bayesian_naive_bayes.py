"""Test suite for NaiveBayesClassifier.

Auto-generated test stub. Please implement comprehensive tests.
"""

import numpy as np
import pickle

from tuiml.algorithms.bayesian import NaiveBayesClassifier


class TestNaiveBayesClassifierInstantiation:
    """Tests for algorithm instantiation."""
    
    def test_fit_basic(self, binary_cls_data):
        """Test basic fitting."""
        X, y = binary_cls_data
        model = NaiveBayesClassifier()
        model.fit(X, y)
        assert model._is_fitted is True
        
        preds = model.predict(X)
        assert preds.shape == y.shape
        
        probas = model.predict_proba(X)
        assert probas.shape == (len(X), len(model.classes_))
        assert np.allclose(probas.sum(axis=1), 1.0)
        
    def test_partial_fit(self, binary_cls_data):
        """Test partial_fit incremental training."""
        X, y = binary_cls_data
        classes = np.unique(y)
        
        model = NaiveBayesClassifier()
        
        # Split into batches
        n_samples = len(X)
        half = n_samples // 2
        
        # First batch
        model.partial_fit(X[:half], y[:half], classes=classes)
        assert model._is_fitted is True
        assert model.n_samples_seen_ == half
        
        # Second batch
        model.partial_fit(X[half:], y[half:])
        assert model.n_samples_seen_ == n_samples
        
        # Make predictions
        preds = model.predict(X)
        assert preds.shape == y.shape
        
        probas = model.predict_proba(X)
        assert probas.shape == (len(X), len(classes))
        assert np.allclose(probas.sum(axis=1), 1.0)


class TestNaiveBayesClassifierSerialization:
    """Tests for serialization."""
    
    def test_pickle_roundtrip(self, binary_cls_data):
        """Test pickle serialization."""
        X, y = binary_cls_data
        model = NaiveBayesClassifier()
        model.fit(X, y)
        
        data = pickle.dumps(model)
        loaded_model = pickle.loads(data)
        
        assert loaded_model._is_fitted is True
        assert np.array_equal(loaded_model.predict(X), model.predict(X))
