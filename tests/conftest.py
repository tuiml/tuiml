"""Shared fixtures and utilities for TuiML tests.

This module provides common test fixtures for all algorithm tests,
including classification, regression, clustering, and association datasets.
"""

import numpy as np
import pytest
from typing import Tuple, List, Set


# =============================================================================
# Classification Fixtures
# =============================================================================

@pytest.fixture
def binary_cls_data() -> Tuple[np.ndarray, np.ndarray]:
    """Simple binary classification dataset.
    
    Returns
    -------
    X : np.ndarray of shape (100, 4)
        Feature matrix with 4 numeric features.
    y : np.ndarray of shape (100,)
        Binary target labels (0 or 1).
    """
    np.random.seed(42)
    n_samples = 100
    n_features = 4
    
    X = np.random.randn(n_samples, n_features)
    # Create linearly separable-ish data
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    
    return X, y


@pytest.fixture
def multiclass_cls_data() -> Tuple[np.ndarray, np.ndarray]:
    """Multi-class classification dataset with 3 classes.
    
    Returns
    -------
    X : np.ndarray of shape (150, 3)
        Feature matrix with 3 numeric features.
    y : np.ndarray of shape (150,)
        Multi-class target labels (0, 1, or 2).
    """
    np.random.seed(42)
    n_samples = 150
    n_features = 3
    n_classes = 3
    
    X = np.random.randn(n_samples, n_features)
    # Create 3 clusters
    y = np.zeros(n_samples, dtype=int)
    y[50:100] = 1
    y[100:] = 2
    
    # Add class-specific offsets
    X[:50, 0] += 2
    X[50:100, 1] += 2
    X[100:, 2] += 2
    
    return X, y


@pytest.fixture
def cls_data_with_missing() -> Tuple[np.ndarray, np.ndarray]:
    """Binary classification data with missing values.
    
    Returns
    -------
    X : np.ndarray of shape (50, 3)
        Feature matrix with NaN values.
    y : np.ndarray of shape (50,)
        Binary target labels.
    """
    np.random.seed(42)
    X = np.random.randn(50, 3)
    y = (X[:, 0] > 0).astype(int)
    
    # Introduce 10% missing values
    missing_mask = np.random.random(X.shape) < 0.1
    X[missing_mask] = np.nan
    
    return X, y


@pytest.fixture
def cls_single_feature() -> Tuple[np.ndarray, np.ndarray]:
    """Single feature binary classification.
    
    Returns
    -------
    X : np.ndarray of shape (50, 1)
        Single feature column.
    y : np.ndarray of shape (50,)
        Binary target labels.
    """
    np.random.seed(42)
    X = np.random.randn(50, 1)
    y = (X[:, 0] > 0).astype(int)
    return X, y


@pytest.fixture
def imbalanced_cls_data() -> Tuple[np.ndarray, np.ndarray]:
    """Highly imbalanced binary classification data.
    
    Returns
    -------
    X : np.ndarray of shape (100, 3)
        Feature matrix.
    y : np.ndarray of shape (100,)
        Imbalanced binary labels (90% class 0, 10% class 1).
    """
    np.random.seed(42)
    n_samples = 100
    X = np.random.randn(n_samples, 3)
    y = np.zeros(n_samples, dtype=int)
    y[:10] = 1  # Only 10% positive class
    return X, y


# =============================================================================
# Regression Fixtures
# =============================================================================

@pytest.fixture
def regression_data() -> Tuple[np.ndarray, np.ndarray]:
    """Simple regression dataset.
    
    Returns
    -------
    X : np.ndarray of shape (100, 3)
        Feature matrix with 3 numeric features.
    y : np.ndarray of shape (100,)
        Continuous target values.
    """
    np.random.seed(42)
    n_samples = 100
    n_features = 3
    
    X = np.random.randn(n_samples, n_features)
    # Linear relationship with noise
    y = 2 * X[:, 0] + 3 * X[:, 1] - X[:, 2] + np.random.randn(n_samples) * 0.1
    
    return X, y


@pytest.fixture
def regression_data_multivariate() -> Tuple[np.ndarray, np.ndarray]:
    """Multi-output regression dataset.
    
    Returns
    -------
    X : np.ndarray of shape (50, 2)
        Feature matrix.
    y : np.ndarray of shape (50, 2)
        Multi-output target values.
    """
    np.random.seed(42)
    n_samples = 50
    
    X = np.random.randn(n_samples, 2)
    y = np.zeros((n_samples, 2))
    y[:, 0] = 2 * X[:, 0] + np.random.randn(n_samples) * 0.1
    y[:, 1] = -3 * X[:, 1] + np.random.randn(n_samples) * 0.1
    
    return X, y


@pytest.fixture
def regression_with_missing() -> Tuple[np.ndarray, np.ndarray]:
    """Regression data with missing values.
    
    Returns
    -------
    X : np.ndarray of shape (50, 3)
        Feature matrix with NaN values.
    y : np.ndarray of shape (50,)
        Continuous target values.
    """
    np.random.seed(42)
    X = np.random.randn(50, 3)
    y = X[:, 0] + 2 * X[:, 1]
    
    # Introduce missing values
    missing_mask = np.random.random(X.shape) < 0.1
    X[missing_mask] = np.nan
    
    return X, y


# =============================================================================
# Clustering Fixtures
# =============================================================================

@pytest.fixture
def clustering_data() -> np.ndarray:
    """Clustering dataset with 3 well-separated clusters.
    
    Returns
    -------
    X : np.ndarray of shape (150, 2)
        2D data with 3 distinct clusters.
    """
    np.random.seed(42)
    n_samples = 150
    
    # Create 3 clusters
    cluster1 = np.random.randn(50, 2) + np.array([0, 0])
    cluster2 = np.random.randn(50, 2) + np.array([5, 5])
    cluster3 = np.random.randn(50, 2) + np.array([0, 5])
    
    X = np.vstack([cluster1, cluster2, cluster3])
    return X


@pytest.fixture
def clustering_data_high_dim() -> np.ndarray:
    """High-dimensional clustering data.
    
    Returns
    -------
    X : np.ndarray of shape (100, 50)
        50-dimensional data with 3 clusters.
    """
    np.random.seed(42)
    n_samples = 100
    n_features = 50
    
    X = np.random.randn(n_samples, n_features)
    # Add structure
    X[:33, :10] += 2
    X[33:66, 10:20] += 2
    X[66:, 20:30] += 2
    
    return X


@pytest.fixture
def clustering_data_single_cluster() -> np.ndarray:
    """Data that should form a single cluster.
    
    Returns
    -------
    X : np.ndarray of shape (50, 2)
        Single Gaussian blob.
    """
    np.random.seed(42)
    return np.random.randn(50, 2)


# =============================================================================
# Association Rule Fixtures
# =============================================================================

@pytest.fixture
def transaction_data_binary() -> np.ndarray:
    """Binary transaction matrix for association rule mining.
    
    Returns
    -------
    X : np.ndarray of shape (100, 20)
        Binary matrix where X[i,j]=1 means item j is in transaction i.
    """
    np.random.seed(42)
    n_transactions = 100
    n_items = 20
    
    X = np.random.random((n_transactions, n_items)) < 0.1
    X = X.astype(int)
    
    # Add some patterns
    for i in range(0, 30):
        X[i, [0, 1, 2]] = 1  # Pattern: 0,1,2 together
    for i in range(30, 60):
        X[i, [3, 4]] = 1  # Pattern: 3,4 together
        
    return X


@pytest.fixture
def transaction_data_list() -> List[Set[int]]:
    """Transaction data as list of item sets.
    
    Returns
    -------
    transactions : list of sets
        Each set contains item indices.
    """
    np.random.seed(42)
    transactions = []
    
    for i in range(100):
        items = set(np.where(np.random.random(20) < 0.1)[0])
        transactions.append(items)
    
    # Add patterns
    for i in range(0, 30):
        transactions[i].update([0, 1, 2])
    for i in range(30, 60):
        transactions[i].update([3, 4])
        
    return transactions


# =============================================================================
# Edge Case Fixtures
# =============================================================================

@pytest.fixture
def single_sample_data() -> Tuple[np.ndarray, np.ndarray]:
    """Single sample classification data.
    
    Returns
    -------
    X : np.ndarray of shape (1, 3)
        Single sample with 3 features.
    y : np.ndarray of shape (1,)
        Single label.
    """
    X = np.array([[1.0, 2.0, 3.0]])
    y = np.array([0])
    return X, y


@pytest.fixture
def empty_data() -> Tuple[np.ndarray, np.ndarray]:
    """Empty dataset.
    
    Returns
    -------
    X : np.ndarray of shape (0, 3)
        Empty feature matrix.
    y : np.ndarray of shape (0,)
        Empty label array.
    """
    X = np.empty((0, 3))
    y = np.empty((0,))
    return X, y


@pytest.fixture
def constant_features() -> Tuple[np.ndarray, np.ndarray]:
    """Data with constant (zero variance) features.
    
    Returns
    -------
    X : np.ndarray of shape (50, 3)
        Feature matrix where column 1 is constant.
    y : np.ndarray of shape (50,)
        Binary labels.
    """
    np.random.seed(42)
    X = np.random.randn(50, 3)
    X[:, 1] = 5.0  # Constant feature
    y = (X[:, 0] > 0).astype(int)
    return X, y


@pytest.fixture
def high_correlation_data() -> Tuple[np.ndarray, np.ndarray]:
    """Data with highly correlated features.
    
    Returns
    -------
    X : np.ndarray of shape (50, 3)
        Feature matrix where columns 1 and 2 are highly correlated.
    y : np.ndarray of shape (50,)
        Binary labels.
    """
    np.random.seed(42)
    X = np.random.randn(50, 3)
    X[:, 2] = X[:, 1] + np.random.randn(50) * 0.01  # Near-perfect correlation
    y = (X[:, 0] > 0).astype(int)
    return X, y


# =============================================================================
# Large Data Fixtures (for performance testing)
# =============================================================================

@pytest.fixture
def large_classification_data() -> Tuple[np.ndarray, np.ndarray]:
    """Large classification dataset for performance tests.
    
    Returns
    -------
    X : np.ndarray of shape (10000, 50)
        Large feature matrix.
    y : np.ndarray of shape (10000,)
        Binary labels.
    """
    np.random.seed(42)
    n_samples = 10000
    n_features = 50
    
    X = np.random.randn(n_samples, n_features)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    
    return X, y


@pytest.fixture
def large_clustering_data() -> np.ndarray:
    """Large clustering dataset.
    
    Returns
    -------
    X : np.ndarray of shape (5000, 10)
        Large clustering dataset.
    """
    np.random.seed(42)
    return np.random.randn(5000, 10)


# =============================================================================
# Anomaly Detection Fixtures
# =============================================================================

@pytest.fixture
def anomaly_detection_data() -> np.ndarray:
    """Data with outliers for anomaly detection.
    
    Returns
    -------
    X : np.ndarray of shape (100, 2)
        Data with 10 outliers (10% contamination).
    """
    np.random.seed(42)
    
    # Normal data
    X_normal = np.random.randn(90, 2)
    
    # Outliers
    X_outliers = np.random.uniform(-5, 5, (10, 2))
    
    X = np.vstack([X_normal, X_outliers])
    return X


# =============================================================================
# Time Series Fixtures
# =============================================================================

@pytest.fixture
def timeseries_data() -> np.ndarray:
    """Simple time series data.
    
    Returns
    -------
    y : np.ndarray of shape (100,)
        Univariate time series with trend and seasonality.
    """
    np.random.seed(42)
    t = np.arange(100)
    trend = 0.1 * t
    seasonal = 2 * np.sin(2 * np.pi * t / 12)
    noise = np.random.randn(100) * 0.5
    
    y = trend + seasonal + noise
    return y


@pytest.fixture
def timeseries_with_trend() -> Tuple[np.ndarray, np.ndarray]:
    """Time series regression data.
    
    Returns
    -------
    X : np.ndarray of shape (90, 5)
        Lagged features (5 lags).
    y : np.ndarray of shape (90,)
        Target values.
    """
    np.random.seed(42)
    y = np.cumsum(np.random.randn(100))  # Random walk with trend
    
    # Create lagged features
    n_lags = 5
    X = np.zeros((len(y) - n_lags, n_lags))
    for i in range(n_lags):
        X[:, i] = y[i:len(y) - n_lags + i]
    
    y_target = y[n_lags:]
    
    return X, y_target


# =============================================================================
# Utility Functions
# =============================================================================

def assert_fitted_attributes(clf, expected_attrs: List[str]):
    """Assert that all expected fitted attributes exist and are not None.
    
    Parameters
    ----------
    clf : object
        Fitted estimator.
    expected_attrs : list of str
        List of attribute names that should exist (without trailing underscore).
    """
    for attr in expected_attrs:
        full_attr = attr + "_"
        assert hasattr(clf, full_attr), f"Missing attribute: {full_attr}"
        assert getattr(clf, full_attr) is not None, f"Attribute {full_attr} is None"


def assert_predictions_valid(predictions: np.ndarray, expected_len: int, 
                             allowed_values=None):
    """Assert that predictions are valid.
    
    Parameters
    ----------
    predictions : np.ndarray
        Predicted values.
    expected_len : int
        Expected number of predictions.
    allowed_values : array-like, optional
        If provided, assert all predictions are in this set.
    """
    assert len(predictions) == expected_len, \
        f"Expected {expected_len} predictions, got {len(predictions)}"
    
    if allowed_values is not None:
        assert all(p in allowed_values for p in predictions), \
            f"Predictions contain values outside allowed set: {allowed_values}"


def assert_probabilities_valid(proba: np.ndarray, n_samples: int, n_classes: int):
    """Assert that probability predictions are valid.
    
    Parameters
    ----------
    proba : np.ndarray
        Probability matrix of shape (n_samples, n_classes).
    n_samples : int
        Expected number of samples.
    n_classes : int
        Expected number of classes.
    """
    assert proba.shape == (n_samples, n_classes), \
        f"Expected shape ({n_samples}, {n_classes}), got {proba.shape}"
    
    # All probabilities should sum to 1
    np.testing.assert_array_almost_equal(
        proba.sum(axis=1), 
        np.ones(n_samples),
        decimal=5,
        err_msg="Probabilities don't sum to 1"
    )
    
    # All probabilities should be in [0, 1]
    assert np.all(proba >= 0) and np.all(proba <= 1), \
        "Probabilities outside [0, 1] range"
