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


# ---------------------------------------------------------------------------
# Agent-tool isolation
# ---------------------------------------------------------------------------
# Merged here when the test tree was flattened; these previously lived in
# tests/test_agent/conftest.py.
#
# The tool executors write models, uploads, plots and agent-authored
# algorithms into ``~/.tuiml/`` -- the user's actual working directory. Left
# alone, a test run would add junk there and, worse, read state a previous run
# left behind, so a passing test would depend on machine history.
# :func:`agent_home` redirects every one of those write targets at a tmp dir
# for the duration of a test.
#
# Two patterns matter. Path constants are imported *by value* into several
# modules (``from ._state import _MODELS_DIR``), so rebinding one definition
# is not enough: every import site is patched, which is what ``_PATH_BINDINGS``
# enumerates. The shared *containers* (the model index, the session log) have
# the same problem but cannot be rebound at all -- holders keep the original
# object -- so those are emptied and refilled in place.
#
# Neither fixture is autouse, so tests that do not request them are unaffected.

import matplotlib
import pytest

matplotlib.use("Agg")  # headless: plot tools must not need a display


#: ``(module path, attribute, subdirectory)`` for every binding that points
#: into ``~/.tuiml`` and has to be redirected. A new import site added without
#: being listed here shows up as a test writing to the real home directory.
_PATH_BINDINGS = [
    ("tuiml.agent.tools._state", "_MODELS_DIR", "models"),
    ("tuiml.agent.tools._shared", "_MODELS_DIR", "models"),
    ("tuiml.agent.tools._state", "_UPLOADS_DIR", "uploads"),
    ("tuiml.agent.tools.data.upload", "_UPLOADS_DIR", "uploads"),
    # Six modules re-import USER_ALGS_DIR from _paths. Patching fewer than all
    # of them splits the world: tuiml_create_algorithm writes to the tmp dir
    # while tuiml_edit_algorithm looks in the real one and reports "not found".
    ("tuiml.agent.user_algorithms._paths", "USER_ALGS_DIR", "user_algorithms"),
    ("tuiml.agent.user_algorithms", "USER_ALGS_DIR", "user_algorithms"),
    ("tuiml.agent.user_algorithms.storage", "USER_ALGS_DIR", "user_algorithms"),
    ("tuiml.agent.user_algorithms.sources", "USER_ALGS_DIR", "user_algorithms"),
    ("tuiml.agent.user_algorithms.research_log", "USER_ALGS_DIR", "user_algorithms"),
    ("tuiml.agent.user_algorithms.registration", "USER_ALGS_DIR", "user_algorithms"),
]


def _emptied(containers):
    """Context-manager-free helper: empty containers, return their contents.

    Parameters
    ----------
    containers : iterable of list or dict
        Live shared containers to clear.

    Returns
    -------
    saved : list
        A copy of each container's original contents, in the same order, for
        :func:`_refill`.
    """
    saved = [c.copy() for c in containers]
    for container in containers:
        container.clear()
    return saved


def _refill(containers, saved):
    """Restore contents captured by :func:`_emptied`.

    Parameters
    ----------
    containers : iterable of list or dict
        The same containers, in the same order.
    saved : list
        Contents returned by :func:`_emptied`.

    Returns
    -------
    None
    """
    for container, original in zip(containers, saved):
        container.clear()
        if isinstance(container, dict):
            container.update(original)
        else:
            container.extend(original)


@pytest.fixture
def agent_home(tmp_path, monkeypatch):
    """Point every ``~/.tuiml`` write target at a tmp directory.

    Parameters
    ----------
    tmp_path : pathlib.Path
        pytest's per-test temporary directory.
    monkeypatch : pytest.MonkeyPatch
        Restores the path bindings and the env var when the test ends.

    Yields
    ------
    home : pathlib.Path
        The tmp directory standing in for ``~/.tuiml``.
    """
    import importlib
    from pathlib import Path

    home = tmp_path / "tuiml_home"
    for module_path, attribute, subdir in _PATH_BINDINGS:
        module = importlib.import_module(module_path)
        target = home / subdir
        target.mkdir(parents=True, exist_ok=True)
        # Path-typed bindings must stay Paths and str-typed ones must stay str:
        # consuming code uses os.path.join on some and / on others.
        current = getattr(module, attribute)
        monkeypatch.setattr(
            module, attribute, target if isinstance(current, Path) else str(target)
        )

    plots = home / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("TUIML_PLOT_DIR", str(plots))

    # The in-memory indices map ids to files under the *real* home. Empty them
    # so a model id left there by the user can never resolve during a test.
    from tuiml.agent.tools import _state
    indices = (_state._MODEL_INDEX, _state._DATASET_INDEX)
    saved = _emptied(indices)

    yield home

    _refill(indices, saved)


@pytest.fixture
def clean_session():
    """Empty the notebook-export session log around a test.

    The log is process-global state shared by value across modules, so a test
    that records calls would otherwise leak them into the next test's exported
    notebook.

    Yields
    ------
    state : module
        ``tuiml.agent.tools._state``, with its session containers emptied.
    """
    from tuiml.agent.tools import _state

    containers = (
        _state._SESSION_CALLS,
        _state._MODEL_ID_TO_VAR,
        _state._TRAIN_CALL_SEQ,
    )
    saved = _emptied(containers)

    yield _state

    _refill(containers, saved)
