"""Nearest-neighbour classifiers and their search structures.

Merged from: test_neighbors_ibk.py, test_search_linear_nn.py, test_search_kd_tree.py, test_search_ball_tree.py
"""

import numpy as np
import pickle
from tuiml.algorithms.neighbors import KNearestNeighborsClassifier, KNearestNeighborsRegressor
import pytest
from tuiml.algorithms.neighbors.search import LinearNNSearch
from tuiml.algorithms.neighbors.search import KDTree
from tuiml.algorithms.neighbors.search import BallTree


# --------------------------------------------------------------------------
# Test suite for KNearestNeighborsClassifier and KNearestNeighborsRegressor.
# --------------------------------------------------------------------------

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


# --------------------------------------------------------------------------
# Test suite for LinearNNSearch (brute force) nearest neighbor search.
# --------------------------------------------------------------------------

class TestLinearNNSearchInit:
    """Tests for LinearNNSearch initialization."""

    def test_build_basic(self):
        """Test building from data."""
        np.random.seed(42)
        X = np.random.randn(50, 3)
        search = LinearNNSearch()

        result = search.build(X)

        assert result is search
        assert search._is_built is True
        assert search.n_samples_ == 50
        assert search.n_features_ == 3

    def test_build_small_data(self):
        """Test building with very small dataset."""
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        search = LinearNNSearch()
        search.build(X)

        assert search._is_built is True
        assert search.n_samples_ == 2

    def test_build_single_point(self):
        """Test building with a single data point."""
        X = np.array([[1.0, 2.0, 3.0]])
        search = LinearNNSearch()
        search.build(X)

        assert search._is_built is True
        assert search.n_samples_ == 1

    def test_repr_built(self):
        """Test string representation after building."""
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        search = LinearNNSearch()
        search.build(X)

        repr_str = repr(search)
        assert "LinearNNSearch" in repr_str
        assert "n_samples=3" in repr_str


class TestLinearNNSearchQuery:
    """Tests for the query() method."""

    def test_query_all_distances_correct(self):
        """Test that all returned distances match manual computation."""
        X = np.array([[0.0, 0.0], [3.0, 4.0], [1.0, 0.0]])
        search = LinearNNSearch()
        search.build(X)

        query = np.array([0.0, 0.0])
        dists, indices = search.query(query, k=3)

        # Distances: 0.0, 5.0, 1.0 -> sorted: 0.0, 1.0, 5.0
        np.testing.assert_allclose(dists[0], 0.0, atol=1e-10)
        np.testing.assert_allclose(dists[1], 1.0, atol=1e-10)
        np.testing.assert_allclose(dists[2], 5.0, atol=1e-10)


# --------------------------------------------------------------------------
# Test suite for KDTree nearest neighbor search.
# --------------------------------------------------------------------------

class TestKDTreeInit:
    """Tests for KDTree initialization."""

    def test_default_initialization(self):
        """Test default leaf_size initialization."""
        tree = KDTree()

        assert tree.leaf_size == 10
        assert tree._is_built is False

    def test_custom_leaf_size(self):
        """Test custom leaf_size initialization."""
        tree = KDTree(leaf_size=5)

        assert tree.leaf_size == 5


class TestKDTreeBuild:
    """Tests for the build() method."""

    def test_build_basic(self):
        """Test building a KDTree from data."""
        np.random.seed(42)
        X = np.random.randn(50, 3)
        tree = KDTree(leaf_size=10)

        result = tree.build(X)

        assert result is tree
        assert tree._is_built is True
        assert tree.n_samples_ == 50
        assert tree.n_features_ == 3
        assert tree._root is not None

    def test_build_small_data(self):
        """Test building with very small dataset (fewer points than leaf_size)."""
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        tree = KDTree(leaf_size=10)
        tree.build(X)

        assert tree._is_built is True
        assert tree.n_samples_ == 2

    def test_build_single_point(self):
        """Test building with a single data point."""
        X = np.array([[1.0, 2.0, 3.0]])
        tree = KDTree(leaf_size=10)
        tree.build(X)

        assert tree._is_built is True
        assert tree.n_samples_ == 1

    def test_repr_built(self):
        """Test string representation after building."""
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        tree = KDTree(leaf_size=2)
        tree.build(X)

        repr_str = repr(tree)
        assert "KDTree" in repr_str
        assert "n_samples=4" in repr_str


class TestKDTreeQuery:
    """Tests for the query() method."""

    def test_query_radius_zero_returns_exact_match(self):
        """Test that radius=0 returns only exact matches."""
        X = np.array([[0.0, 0.0], [1.0, 1.0]])
        tree = KDTree(leaf_size=2)
        tree.build(X)

        dists, indices = tree.query_radius(np.array([0.0, 0.0]), radius=0.0)

        assert len(indices) == 1
        assert indices[0] == 0


# --------------------------------------------------------------------------
# Test suite for BallTree nearest neighbor search.
# --------------------------------------------------------------------------

class TestBallTreeInit:
    """Tests for BallTree initialization."""

    def test_default_initialization(self):
        """Test default leaf_size initialization."""
        tree = BallTree()

        assert tree.leaf_size == 10
        assert tree._is_built is False

    def test_custom_leaf_size(self):
        """Test custom leaf_size initialization."""
        tree = BallTree(leaf_size=20)

        assert tree.leaf_size == 20


class TestBallTreeBuild:
    """Tests for the build() method."""

    def test_build_basic(self):
        """Test building a BallTree from data."""
        np.random.seed(42)
        X = np.random.randn(50, 3)
        tree = BallTree(leaf_size=10)

        result = tree.build(X)

        assert result is tree
        assert tree._is_built is True
        assert tree.n_samples_ == 50
        assert tree.n_features_ == 3
        assert tree._root is not None

    def test_build_small_data(self):
        """Test building with fewer points than leaf_size."""
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        tree = BallTree(leaf_size=10)
        tree.build(X)

        assert tree._is_built is True
        assert tree.n_samples_ == 2

    def test_build_single_point(self):
        """Test building with a single data point."""
        X = np.array([[1.0, 2.0, 3.0]])
        tree = BallTree(leaf_size=10)
        tree.build(X)

        assert tree._is_built is True
        assert tree.n_samples_ == 1

    def test_repr_built(self):
        """Test string representation after building."""
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
        tree = BallTree(leaf_size=2)
        tree.build(X)

        repr_str = repr(tree)
        assert "BallTree" in repr_str
        assert "n_samples=4" in repr_str


class TestBallTreeQuery:
    """Tests for the query() method."""

    def test_query_radius_empty_result(self):
        """Test query_radius with no points in range."""
        X = np.array([[100.0, 100.0], [200.0, 200.0]])
        tree = BallTree(leaf_size=2)
        tree.build(X)

        dists, indices = tree.query_radius(np.array([0.0, 0.0]), radius=1.0)

        assert len(indices) == 0
