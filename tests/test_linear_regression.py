"""Test suite for Linear Regression.

Tests cover:
- OLS and ridge regression
- Feature selection methods (M5, greedy)
- Colinearity handling
- Missing value handling
"""

import numpy as np
import pytest
import pickle

from tuiml.algorithms.linear import LinearRegression


class TestLinearRegressionInstantiation:
    """Tests for algorithm instantiation."""
    
    def test_default_initialization(self):
        """Test default initialization."""
        reg = LinearRegression()
        
        assert reg.ridge == 1e-8
        assert reg.attribute_selection == "none"
        assert reg.eliminate_colinear is True
        assert reg.fit_intercept is True
        assert reg._is_fitted is False
        
    def test_custom_initialization(self):
        """Test custom parameter initialization."""
        reg = LinearRegression(
            ridge=0.1,
            attribute_selection="m5",
            eliminate_colinear=False,
            fit_intercept=False
        )
        
        assert reg.ridge == 0.1
        assert reg.attribute_selection == "m5"
        assert reg.eliminate_colinear is False
        assert reg.fit_intercept is False
        
    def test_parameter_schema(self):
        """Test parameter schema."""
        schema = LinearRegression.get_parameter_schema()
        
        assert "ridge" in schema
        assert "attribute_selection" in schema
        assert schema["attribute_selection"]["enum"] == ["none", "m5", "greedy"]
        
    def test_capabilities(self):
        """Test capabilities."""
        caps = LinearRegression.get_capabilities()
        
        assert "numeric" in caps
        assert "missing_values" in caps
        assert "numeric_class" in caps


class TestLinearRegressionFitting:
    """Tests for the fit() method."""
    
    def test_fit_basic(self, regression_data):
        """Test basic fitting."""
        X, y = regression_data
        reg = LinearRegression()
        
        result = reg.fit(X, y)
        
        assert result is reg
        assert reg._is_fitted is True
        assert reg.coefficients_ is not None
        assert reg.intercept_ is not None
        assert reg.n_features_ == X.shape[1]
        
    def test_fit_no_intercept(self, regression_data):
        """Test fitting without intercept."""
        X, y = regression_data
        reg = LinearRegression(fit_intercept=False)
        reg.fit(X, y)
        
        assert reg._is_fitted
        assert reg.intercept_ == 0.0
        
    def test_fit_ridge_regularization(self, regression_data):
        """Test ridge regularization."""
        X, y = regression_data
        reg = LinearRegression(ridge=1.0)
        reg.fit(X, y)
        
        assert reg._is_fitted
        # With strong regularization, coefficients should be smaller
        
    def test_fit_missing_values(self, regression_with_missing):
        """Test fitting with missing values."""
        X, y = regression_with_missing
        reg = LinearRegression()
        reg.fit(X, y)
        
        assert reg._is_fitted
        
    def test_fit_m5_selection(self, regression_data):
        """Test M5 feature selection."""
        X, y = regression_data
        reg = LinearRegression(attribute_selection="m5")
        reg.fit(X, y)
        
        assert reg._is_fitted
        assert reg.selected_features_ is not None
        
    def test_fit_greedy_selection(self, regression_data):
        """Test greedy feature selection."""
        X, y = regression_data
        reg = LinearRegression(attribute_selection="greedy")
        reg.fit(X, y)
        
        assert reg._is_fitted
        assert reg.selected_features_ is not None


class TestLinearRegressionPrediction:
    """Tests for the predict() method."""
    
    def test_predict_basic(self, regression_data):
        """Test basic prediction."""
        X, y = regression_data
        reg = LinearRegression().fit(X, y)
        
        predictions = reg.predict(X)
        
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == len(y)
        assert predictions.dtype == float
        
    def test_predict_single_sample(self, regression_data):
        """Test prediction on single sample."""
        X, y = regression_data
        reg = LinearRegression().fit(X, y)
        
        pred = reg.predict(X[0:1])
        
        assert len(pred) == 1
        assert isinstance(pred[0], (float, np.floating))
        
    def test_predict_before_fit_raises(self):
        """Test predict before fit raises error."""
        reg = LinearRegression()
        
        with pytest.raises(RuntimeError, match="must be fitted"):
            reg.predict(np.array([[1, 2, 3]]))
            
    def test_predict_with_missing(self, regression_with_missing):
        """Test prediction with missing values."""
        X, y = regression_with_missing
        reg = LinearRegression().fit(X, y)
        
        predictions = reg.predict(X)
        assert len(predictions) == len(y)
        assert not np.any(np.isnan(predictions))


class TestLinearRegressionScore:
    """Tests for the score() method (R-squared)."""
    
    def test_score_perfect_fit(self):
        """Test R-squared on perfect fit."""
        X = np.array([[1], [2], [3], [4]])
        y = np.array([2, 4, 6, 8])  # Perfect linear relationship
        
        reg = LinearRegression().fit(X, y)
        score = reg.score(X, y)
        
        assert score > 0.99  # Should be very close to 1.0
        
    def test_score_no_correlation(self):
        """Test R-squared with no correlation."""
        np.random.seed(42)
        X = np.random.randn(100, 3)
        y = np.random.randn(100)
        
        reg = LinearRegression().fit(X, y)
        score = reg.score(X, y)
        
        # R-squared can be negative for poor fits
        assert isinstance(score, float)


class TestLinearRegressionSerialization:
    """Tests for serialization."""
    
    def test_pickle_roundtrip(self, regression_data):
        """Test pickle serialization."""
        X, y = regression_data
        reg = LinearRegression().fit(X, y)
        
        reg_bytes = pickle.dumps(reg)
        reg_loaded = pickle.loads(reg_bytes)
        
        pred_original = reg.predict(X)
        pred_loaded = reg_loaded.predict(X)
        
        np.testing.assert_array_almost_equal(pred_original, pred_loaded)


class TestLinearRegressionEdgeCases:
    """Edge case tests."""
    
    def test_single_feature(self):
        """Test with single feature."""
        X = np.array([[1], [2], [3], [4]])
        y = np.array([2, 4, 6, 8])
        
        reg = LinearRegression()
        reg.fit(X, y)
        
        assert reg._is_fitted
        assert reg.n_features_ == 1
        
    def test_constant_target(self):
        """Test with constant target values."""
        X = np.random.randn(20, 3)
        y = np.ones(20) * 5.0
        
        reg = LinearRegression()
        reg.fit(X, y)
        
        assert reg._is_fitted
        # All predictions should be close to the mean (5.0)
        predictions = reg.predict(X)
        np.testing.assert_array_almost_equal(predictions, y, decimal=5)
