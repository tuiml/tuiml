"""Neural network models.

Merged from: test_neural_perceptron.py, test_neural_multilayer_perceptron.py
"""

import numpy as np
import pickle
from tuiml.algorithms.neural import PerceptronClassifier
from tuiml.algorithms.neural import MultilayerPerceptronClassifier, MultilayerPerceptronRegressor


# --------------------------------------------------------------------------
# Test suite for PerceptronClassifier.
# --------------------------------------------------------------------------

class TestPerceptronClassifierInstantiation:
    """Tests for algorithm instantiation."""
    
    def test_fit_basic(self, binary_cls_data):
        """Test basic fitting."""
        X, y = binary_cls_data
        model = PerceptronClassifier()
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
        
        model = PerceptronClassifier()
        
        # Split into batches
        n_samples = len(X)
        half = n_samples // 2
        
        # First batch
        model.partial_fit(X[:half], y[:half], classes=classes)
        assert model._is_fitted is True
        assert model.n_iter_ == 1
        
        # Second batch
        model.partial_fit(X[half:], y[half:])
        assert model.n_iter_ == 2
        
        # Make predictions
        preds = model.predict(X)
        assert preds.shape == y.shape
        
        probas = model.predict_proba(X)
        assert probas.shape == (len(X), len(classes))
        assert np.allclose(probas.sum(axis=1), 1.0)


class TestPerceptronClassifierSerialization:
    """Tests for serialization."""
    
    def test_pickle_roundtrip(self, binary_cls_data):
        """Test pickle serialization."""
        X, y = binary_cls_data
        model = PerceptronClassifier()
        model.fit(X, y)
        
        data = pickle.dumps(model)
        loaded_model = pickle.loads(data)
        
        assert loaded_model._is_fitted is True
        assert np.array_equal(loaded_model.predict(X), model.predict(X))


# --------------------------------------------------------------------------
# Test suite for MultilayerPerceptronClassifier and MultilayerPerceptronRegressor.
# --------------------------------------------------------------------------

class TestMultilayerPerceptronClassifierInstantiation:
    """Tests for algorithm instantiation."""
    
    def test_fit_basic(self, binary_cls_data):
        """Test basic fitting."""
        X, y = binary_cls_data
        model = MultilayerPerceptronClassifier(max_epochs=10, hidden_layers=[5])
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
        
        model = MultilayerPerceptronClassifier(hidden_layers=[5])
        
        # Split into batches
        n_samples = len(X)
        half = n_samples // 2
        
        # First batch
        model.partial_fit(X[:half], y[:half], classes=classes)
        assert model._is_fitted is True
        assert model._epoch == 1
        
        # Second batch
        model.partial_fit(X[half:], y[half:])
        assert model._epoch == 2
        
        # Make predictions
        preds = model.predict(X)
        assert preds.shape == y.shape
        
        probas = model.predict_proba(X)
        assert probas.shape == (len(X), len(classes))
        assert np.allclose(probas.sum(axis=1), 1.0)


class TestMultilayerPerceptronClassifierSerialization:
    """Tests for serialization."""
    
    def test_pickle_roundtrip(self, binary_cls_data):
        """Test pickle serialization."""
        X, y = binary_cls_data
        model = MultilayerPerceptronClassifier(max_epochs=10, hidden_layers=[5])
        model.fit(X, y)
        
        data = pickle.dumps(model)
        loaded_model = pickle.loads(data)
        
        assert loaded_model._is_fitted is True
        assert np.array_equal(loaded_model.predict(X), model.predict(X))


class TestMultilayerPerceptronRegressorFitting:
    """Tests for the MultilayerPerceptronRegressor."""
    
    def test_fit_basic(self, regression_data):
        """Test basic fitting."""
        X, y = regression_data
        model = MultilayerPerceptronRegressor(max_epochs=10, hidden_layers=[5])
        model.fit(X, y)
        assert model._is_fitted is True
        
        preds = model.predict(X)
        assert preds.shape == y.shape
        
    def test_partial_fit(self, regression_data):
        """Test partial_fit incremental training."""
        X, y = regression_data
        model = MultilayerPerceptronRegressor(hidden_layers=[5])
        
        n_samples = len(X)
        half = n_samples // 2
        
        # First batch
        model.partial_fit(X[:half], y[:half])
        assert model._is_fitted is True
        assert model._epoch == 1
        
        # Second batch
        model.partial_fit(X[half:], y[half:])
        assert model._epoch == 2
        
        preds = model.predict(X)
        assert preds.shape == y.shape
