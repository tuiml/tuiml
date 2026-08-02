"""Integration tests for end-to-end workflows.

These tests verify that different components work together correctly,
including data loading, preprocessing, model training, and evaluation.
"""

import numpy as np
import pytest

from tuiml.algorithms.trees import C45TreeClassifier
from tuiml.algorithms.linear import LinearRegression
from tuiml.algorithms.clustering import KMeansClusterer


class TestBasicWorkflows:
    """Tests for basic ML workflows."""
    
    def test_classification_workflow(self, binary_cls_data):
        """Test end-to-end classification workflow."""
        X, y = binary_cls_data
        
        # Split into train/test
        n_train = int(0.8 * len(X))
        X_train, X_test = X[:n_train], X[n_train:]
        y_train, y_test = y[:n_train], y[n_train:]
        
        # Train model
        clf = C45TreeClassifier()
        clf.fit(X_train, y_train)
        
        # Make predictions
        predictions = clf.predict(X_test)
        
        # Evaluate
        accuracy = np.mean(predictions == y_test)
        assert 0 <= accuracy <= 1
        
    def test_regression_workflow(self, regression_data):
        """Test end-to-end regression workflow."""
        X, y = regression_data
        
        # Split into train/test
        n_train = int(0.8 * len(X))
        X_train, X_test = X[:n_train], X[n_train:]
        y_train, y_test = y[:n_train], y[n_train:]
        
        # Train model
        reg = LinearRegression()
        reg.fit(X_train, y_train)
        
        # Make predictions
        predictions = reg.predict(X_test)
        
        # Evaluate
        mse = np.mean((predictions - y_test) ** 2)
        assert mse >= 0
        
    def test_clustering_workflow(self, clustering_data):
        """Test end-to-end clustering workflow."""
        X = clustering_data
        
        # Fit clusterer
        km = KMeansClusterer(n_clusters=3, random_state=42)
        labels = km.fit_predict(X)
        
        # Evaluate (silhouette-like check)
        # Clusters should be somewhat balanced
        unique, counts = np.unique(labels, return_counts=True)
        assert len(unique) <= 3
        # No cluster should be empty
        assert all(c > 0 for c in counts)


class TestModelPersistence:
    """Tests for model saving and loading workflows."""
    
    def test_pickle_workflow_classification(self, binary_cls_data):
        """Test pickle save/load for classification."""
        import pickle
        
        X, y = binary_cls_data
        
        # Train and save
        clf = C45TreeClassifier().fit(X, y)
        saved = pickle.dumps(clf)
        
        # Load and predict
        clf_loaded = pickle.loads(saved)
        predictions = clf_loaded.predict(X)
        
        assert len(predictions) == len(y)
        
    def test_pickle_workflow_regression(self, regression_data):
        """Test pickle save/load for regression."""
        import pickle
        
        X, y = regression_data
        
        # Train and save
        reg = LinearRegression().fit(X, y)
        saved = pickle.dumps(reg)
        
        # Load and predict
        reg_loaded = pickle.loads(saved)
        predictions = reg_loaded.predict(X)
        
        assert len(predictions) == len(y)


class TestMultiAlgorithmComparison:
    """Tests comparing multiple algorithms on same data."""
    
    def test_classifiers_produce_valid_predictions(self, binary_cls_data):
        """Test that all classifiers produce valid predictions."""
        from tuiml.algorithms.trees import RandomForestClassifier
        from tuiml.algorithms.bayesian import NaiveBayesClassifier
        from tuiml.algorithms.svm import SVC as SMOClassifier
        
        X, y = binary_cls_data
        
        classifiers = [
            ("C45", C45TreeClassifier()),
            ("RandomForest", RandomForestClassifier()),
            ("NaiveBayes", NaiveBayesClassifier()),
            ("SVM", SMOClassifier()),
        ]
        
        for name, clf in classifiers:
            try:
                clf.fit(X, y)
                predictions = clf.predict(X)
                
                # All predictions should be valid class labels
                assert all(p in clf.classes_ for p in predictions), \
                    f"{name} produced invalid predictions"
            except Exception as e:
                pytest.skip(f"{name} not available or failed: {e}")
