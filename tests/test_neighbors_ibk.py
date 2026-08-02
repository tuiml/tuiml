"""Test suite for KNearestNeighborsClassifier and KNearestNeighborsRegressor.

Auto-generated test suite.
"""

import numpy as np
import pickle

from tuiml.algorithms.neighbors import KNearestNeighborsClassifier, KNearestNeighborsRegressor


class TestKNearestNeighborsClassifierInstantiation:
    """Tests for algorithm instantiation."""
    
    def test_fit_basic(self, binary_cls_data):
        """Test basic fitting."""
        X, y = binary_cls_data
        model = KNearestNeighborsClassifier()
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
        
        model = KNearestNeighborsClassifier()
        
        # Split into batches
        n_samples = len(X)
        half = n_samples // 2
        
        # First batch
        model.partial_fit(X[:half], y[:half], classes=classes)
        assert model._is_fitted is True
        assert len(model.X_train_) == half
        
        # Second batch
        model.partial_fit(X[half:], y[half:])
        assert len(model.X_train_) == n_samples
        
        # Make predictions
        preds = model.predict(X)
        assert preds.shape == y.shape
        
        probas = model.predict_proba(X)
        assert probas.shape == (len(X), len(classes))
        assert np.allclose(probas.sum(axis=1), 1.0)


class TestKNearestNeighborsClassifierSerialization:
    """Tests for serialization."""
    
    def test_pickle_roundtrip(self, binary_cls_data):
        """Test pickle serialization."""
        X, y = binary_cls_data
        model = KNearestNeighborsClassifier()
        model.fit(X, y)
        
        data = pickle.dumps(model)
        loaded_model = pickle.loads(data)
        
        assert loaded_model._is_fitted is True
        assert np.array_equal(loaded_model.predict(X), model.predict(X))


class TestKNearestNeighborsRegressorFitting:
    """Tests for KNearestNeighborsRegressor."""
    
    def test_fit_basic(self, regression_data):
        """Test basic fitting."""
        X, y = regression_data
        model = KNearestNeighborsRegressor()
        model.fit(X, y)
        assert model._is_fitted is True
        
        preds = model.predict(X)
        assert preds.shape == y.shape
        
    def test_partial_fit(self, regression_data):
        """Test partial_fit incremental training."""
        X, y = regression_data
        model = KNearestNeighborsRegressor()
        
        n_samples = len(X)
        half = n_samples // 2
        
        # First batch
        model.partial_fit(X[:half], y[:half])
        assert model._is_fitted is True
        assert len(model.X_train_) == half
        
        # Second batch
        model.partial_fit(X[half:], y[half:])
        assert len(model.X_train_) == n_samples
        
        preds = model.predict(X)
        assert preds.shape == y.shape
