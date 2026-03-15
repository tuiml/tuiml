"""Test suite for C4.5 Decision Tree Classifier (J48).

Tests cover:
- Instantiation with various parameters
- Fitting on different data types
- Prediction and probability estimation
- Tree pruning functionality
- Handling of missing values
- Serialization/deserialization
"""

import numpy as np
import pytest
import pickle

from tuiml.algorithms.trees import C45TreeClassifier


class TestC45TreeClassifierInstantiation:
    """Tests for algorithm instantiation and configuration."""
    
    def test_default_initialization(self):
        """Test algorithm initializes with correct default parameters."""
        clf = C45TreeClassifier()
        
        assert clf.min_samples_leaf == 2
        assert clf.confidence_factor == 0.25
        assert clf.unpruned is False
        assert clf.binary_splits is False
        assert clf.max_depth is None
        assert clf._is_fitted is False
        
    def test_custom_initialization(self):
        """Test algorithm accepts custom parameters."""
        clf = C45TreeClassifier(
            min_samples_leaf=5,
            confidence_factor=0.1,
            unpruned=True,
            binary_splits=True,
            max_depth=10
        )
        
        assert clf.min_samples_leaf == 5
        assert clf.confidence_factor == 0.1
        assert clf.unpruned is True
        assert clf.binary_splits is True
        assert clf.max_depth == 10
        
    def test_parameter_schema(self):
        """Test parameter schema returns valid configuration."""
        schema = C45TreeClassifier.get_parameter_schema()
        
        assert isinstance(schema, dict)
        assert "min_samples_leaf" in schema
        assert "confidence_factor" in schema
        assert "unpruned" in schema
        assert "binary_splits" in schema
        assert "max_depth" in schema
        
    def test_capabilities(self):
        """Test algorithm reports correct capabilities."""
        caps = C45TreeClassifier.get_capabilities()
        
        assert isinstance(caps, list)
        assert "numeric" in caps
        assert "nominal" in caps
        assert "missing_values" in caps
        assert "binary_class" in caps
        assert "multiclass" in caps
        
    def test_complexity(self):
        """Test complexity information is returned."""
        complexity = C45TreeClassifier.get_complexity()
        
        assert isinstance(complexity, str)
        assert len(complexity) > 0
        
    def test_references(self):
        """Test academic references are provided."""
        refs = C45TreeClassifier.get_references()
        
        assert isinstance(refs, list)
        assert len(refs) > 0


class TestC45TreeClassifierFitting:
    """Tests for the fit() method."""
    
    def test_fit_binary_classification(self, binary_cls_data):
        """Test fitting on binary classification data."""
        X, y = binary_cls_data
        clf = C45TreeClassifier()
        
        result = clf.fit(X, y)
        
        assert result is clf
        assert clf._is_fitted is True
        assert clf.tree_ is not None
        assert hasattr(clf, "classes_")
        assert hasattr(clf, "n_features_")
        assert clf.n_features_ == X.shape[1]
        assert len(clf.classes_) == 2
        
    def test_fit_multiclass(self, multiclass_cls_data):
        """Test fitting on multi-class data."""
        X, y = multiclass_cls_data
        clf = C45TreeClassifier()
        clf.fit(X, y)
        
        assert clf._is_fitted
        assert len(clf.classes_) == 3
        
    def test_fit_single_feature(self, cls_single_feature):
        """Test fitting with only one feature."""
        X, y = cls_single_feature
        clf = C45TreeClassifier()
        clf.fit(X, y)
        
        assert clf._is_fitted
        assert clf.n_features_ == 1
        
    def test_fit_missing_values(self, cls_data_with_missing):
        """Test fitting with missing values in data."""
        X, y = cls_data_with_missing
        clf = C45TreeClassifier()
        clf.fit(X, y)
        assert clf._is_fitted
        
    def test_fit_1d_input(self, binary_cls_data):
        """Test fitting with 1D input (single feature)."""
        X, y = binary_cls_data
        X_1d = X[:, 0]
        
        clf = C45TreeClassifier()
        clf.fit(X_1d, y)
        
        assert clf._is_fitted
        assert clf.n_features_ == 1


class TestC45TreeClassifierPrediction:
    """Tests for the predict() and predict_proba() methods."""
    
    def test_predict_basic(self, binary_cls_data):
        """Test basic prediction functionality."""
        X, y = binary_cls_data
        clf = C45TreeClassifier().fit(X, y)
        
        predictions = clf.predict(X)
        
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == len(y)
        assert all(p in clf.classes_ for p in predictions)
        
    def test_predict_single_sample(self, binary_cls_data):
        """Test prediction on a single sample."""
        X, y = binary_cls_data
        clf = C45TreeClassifier().fit(X, y)
        
        pred = clf.predict(X[0:1])
        
        assert len(pred) == 1
        assert pred[0] in clf.classes_
        
    def test_predict_before_fit_raises(self):
        """Test that predict() raises error before fitting."""
        clf = C45TreeClassifier()
        
        with pytest.raises(RuntimeError, match="must be fitted"):
            clf.predict(np.array([[1, 2, 3, 4]]))
            
    def test_predict_proba_basic(self, binary_cls_data):
        """Test probability prediction."""
        X, y = binary_cls_data
        clf = C45TreeClassifier().fit(X, y)
        
        proba = clf.predict_proba(X)
        
        assert proba.shape == (len(X), len(clf.classes_))
        np.testing.assert_array_almost_equal(proba.sum(axis=1), 1.0, decimal=5)
        assert np.all(proba >= 0) and np.all(proba <= 1)
        
    def test_predict_proba_multiclass(self, multiclass_cls_data):
        """Test probability prediction with multiple classes."""
        X, y = multiclass_cls_data
        clf = C45TreeClassifier().fit(X, y)
        
        proba = clf.predict_proba(X)
        
        assert proba.shape == (len(X), 3)
        np.testing.assert_array_almost_equal(proba.sum(axis=1), 1.0, decimal=5)


class TestC45TreeClassifierPruning:
    """Tests for tree pruning functionality."""
    
    def test_unpruned_vs_pruned(self, binary_cls_data):
        """Test that pruning option works."""
        X, y = binary_cls_data
        
        clf_pruned = C45TreeClassifier(unpruned=False).fit(X, y)
        clf_unpruned = C45TreeClassifier(unpruned=True).fit(X, y)
        
        assert clf_pruned._is_fitted
        assert clf_unpruned._is_fitted
        
        pred_pruned = clf_pruned.predict(X)
        pred_unpruned = clf_unpruned.predict(X)
        
        assert len(pred_pruned) == len(y)
        assert len(pred_unpruned) == len(y)


class TestC45TreeClassifierInspection:
    """Tests for model inspection methods."""
    
    def test_tree_description(self, binary_cls_data):
        """Test tree description generation."""
        X, y = binary_cls_data
        clf = C45TreeClassifier().fit(X, y)
        
        desc = clf.get_tree_description()
        
        assert isinstance(desc, str)
        assert len(desc) > 0


class TestC45TreeClassifierSerialization:
    """Tests for model serialization."""
    
    def test_pickle_roundtrip(self, binary_cls_data):
        """Test pickle serialization and deserialization."""
        X, y = binary_cls_data
        clf = C45TreeClassifier().fit(X, y)
        
        clf_bytes = pickle.dumps(clf)
        clf_loaded = pickle.loads(clf_bytes)
        
        pred_original = clf.predict(X)
        pred_loaded = clf_loaded.predict(X)
        
        np.testing.assert_array_equal(pred_original, pred_loaded)
        assert clf_loaded._is_fitted


class TestC45TreeClassifierEdgeCases:
    """Tests for edge cases and error conditions."""
    
    def test_all_same_class(self):
        """Test with all samples having same class label."""
        X = np.random.randn(20, 3)
        y = np.zeros(20, dtype=int)
        
        clf = C45TreeClassifier()
        clf.fit(X, y)
        
        assert clf._is_fitted
        assert len(clf.classes_) == 1
        
        predictions = clf.predict(X)
        assert all(p == 0 for p in predictions)
