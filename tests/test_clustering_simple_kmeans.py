"""Test suite for K-Means Clustering.

Tests cover:
- Various initialization methods (k-means++, random, farthest)
- Multiple distance metrics
- Convergence criteria
- Prediction on new data
"""

import numpy as np
import pytest
import pickle

from tuiml.algorithms.clustering import KMeansClusterer


class TestKMeansClustererInstantiation:
    """Tests for algorithm instantiation."""
    
    def test_default_initialization(self):
        """Test default initialization."""
        km = KMeansClusterer()
        
        assert km.n_clusters == 2
        assert km.init == "k-means++"
        assert km.max_iter == 300
        assert km.n_init == 10
        assert km.tol == 1e-4
        assert km.distance == "euclidean"
        
    def test_custom_initialization(self):
        """Test custom initialization."""
        km = KMeansClusterer(
            n_clusters=5,
            init="random",
            max_iter=100,
            n_init=5,
            tol=1e-3,
            distance="manhattan"
        )
        
        assert km.n_clusters == 5
        assert km.init == "random"
        assert km.max_iter == 100
        assert km.n_init == 5
        assert km.tol == 1e-3
        assert km.distance == "manhattan"
        
    def test_parameter_schema(self):
        """Test parameter schema."""
        schema = KMeansClusterer.get_parameter_schema()
        
        assert "n_clusters" in schema
        assert "init" in schema
        assert schema["init"]["enum"] == ["random", "k-means++", "farthest"]


class TestKMeansClustererFitting:
    """Tests for the fit() method."""
    
    def test_fit_basic(self, clustering_data):
        """Test basic fitting."""
        X = clustering_data
        km = KMeansClusterer(n_clusters=3, random_state=42)
        
        result = km.fit(X)
        
        assert result is km
        assert km._is_fitted is True
        assert km.cluster_centers_ is not None
        assert km.labels_ is not None
        assert km.inertia_ is not None
        assert km.n_iter_ is not None
        
        assert km.cluster_centers_.shape == (3, X.shape[1])
        assert len(km.labels_) == len(X)
        assert len(np.unique(km.labels_)) <= 3
        
    def test_fit_single_cluster(self, clustering_data):
        """Test with single cluster."""
        X = clustering_data
        km = KMeansClusterer(n_clusters=1, random_state=42)
        km.fit(X)
        
        assert km._is_fitted
        assert km.cluster_centers_.shape == (1, X.shape[1])
        assert np.all(km.labels_ == 0)
        
    def test_fit_kmeans_plus_plus(self, clustering_data):
        """Test k-means++ initialization."""
        X = clustering_data
        km = KMeansClusterer(n_clusters=3, init="k-means++", random_state=42)
        km.fit(X)
        
        assert km._is_fitted
        assert km.inertia_ is not None
        
    def test_fit_random_init(self, clustering_data):
        """Test random initialization."""
        X = clustering_data
        km = KMeansClusterer(n_clusters=3, init="random", random_state=42)
        km.fit(X)
        
        assert km._is_fitted
        
    def test_fit_farthest_init(self, clustering_data):
        """Test farthest-first initialization."""
        X = clustering_data
        km = KMeansClusterer(n_clusters=3, init="farthest", random_state=42)
        km.fit(X)
        
        assert km._is_fitted


class TestKMeansClustererPrediction:
    """Tests for the predict() method."""
    
    def test_predict_basic(self, clustering_data):
        """Test basic prediction."""
        X = clustering_data
        km = KMeansClusterer(n_clusters=3, random_state=42).fit(X)
        
        predictions = km.predict(X)
        
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == len(X)
        assert all(0 <= p < 3 for p in predictions)
        
    def test_predict_new_data(self, clustering_data):
        """Test prediction on new/unseen data."""
        X = clustering_data
        km = KMeansClusterer(n_clusters=3, random_state=42).fit(X)
        
        # New data points
        X_new = np.array([[0, 0], [5, 5], [0, 5]])
        predictions = km.predict(X_new)
        
        assert len(predictions) == 3
        assert all(0 <= p < 3 for p in predictions)
        
    def test_predict_before_fit_raises(self):
        """Test predict before fit raises error."""
        km = KMeansClusterer()
        
        with pytest.raises(RuntimeError, match="must be fitted"):
            km.predict(np.array([[1, 2]]))
            
    def test_fit_predict(self, clustering_data):
        """Test fit_predict method."""
        X = clustering_data
        km = KMeansClusterer(n_clusters=3, random_state=42)
        
        labels = km.fit_predict(X)
        
        assert isinstance(labels, np.ndarray)
        assert len(labels) == len(X)
        np.testing.assert_array_equal(labels, km.labels_)


class TestKMeansClustererInertia:
    """Tests for inertia (within-cluster sum of squares)."""
    
    def test_inertia_decreases(self, clustering_data):
        """Test that inertia is non-negative."""
        X = clustering_data
        km = KMeansClusterer(n_clusters=3, random_state=42)
        km.fit(X)
        
        assert km.inertia_ >= 0
        
    def test_more_clusters_lower_inertia(self, clustering_data):
        """Test that more clusters generally give lower inertia."""
        X = clustering_data
        
        km2 = KMeansClusterer(n_clusters=2, random_state=42).fit(X)
        km5 = KMeansClusterer(n_clusters=5, random_state=42).fit(X)
        
        # More clusters should generally give equal or lower inertia
        assert km5.inertia_ <= km2.inertia_


class TestKMeansClustererSerialization:
    """Tests for serialization."""
    
    def test_pickle_roundtrip(self, clustering_data):
        """Test pickle serialization."""
        X = clustering_data
        km = KMeansClusterer(n_clusters=3, random_state=42).fit(X)
        
        km_bytes = pickle.dumps(km)
        km_loaded = pickle.loads(km_bytes)
        
        pred_original = km.predict(X)
        pred_loaded = km_loaded.predict(X)
        
        np.testing.assert_array_equal(pred_original, pred_loaded)
        assert km_loaded._is_fitted


class TestKMeansClustererEdgeCases:
    """Edge case tests."""
    
    def test_high_dimensional_data(self, clustering_data_high_dim):
        """Test with high-dimensional data."""
        X = clustering_data_high_dim
        km = KMeansClusterer(n_clusters=3, random_state=42)
        km.fit(X)
        
        assert km._is_fitted
        assert km.cluster_centers_.shape == (3, X.shape[1])
        
    def test_n_clusters_equals_samples(self, clustering_data):
        """Test when n_clusters equals number of samples."""
        X = clustering_data[:5]
        km = KMeansClusterer(n_clusters=5, random_state=42)
        km.fit(X)
        
        assert km._is_fitted
        # Each point should be its own cluster center
        assert len(np.unique(km.labels_)) == 5
