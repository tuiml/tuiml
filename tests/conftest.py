"""Shared fixtures and utilities for TuiML tests.

This module provides common test fixtures for all algorithm tests,
including classification, regression, clustering, and association datasets.
"""

import numpy as np
import pytest
from typing import Tuple


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


# =============================================================================
# Association Rule Fixtures
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
