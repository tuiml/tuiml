"""Tests for CFSSelector and WrapperSelector."""

import numpy as np
import pytest

from tuiml.features.selection import CFSSelector, WrapperSelector


class SimpleClassifier:
    """A simple classifier for testing wrapper selectors."""

    def __init__(self):
        self._majority_class = None

    def fit(self, X, y):
        self._majority_class = int(np.mean(y) >= 0.5)
        return self

    def predict(self, X):
        return np.full(X.shape[0], self._majority_class)


@pytest.fixture
def classification_data():
    """Create classification data with informative features."""
    np.random.seed(42)
    n_samples = 50
    X = np.random.randn(n_samples, 6)
    # Make features 0 and 1 informative
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X, y


# ---- CFSSelector ----

class TestCFSSelectorInit:

    def test_default_init(self):
        selector = CFSSelector()
        assert selector.n_bins == 10
        assert selector.search_method == "best_first"
        assert selector.search_termination == 5
        assert selector.locally_predictive is True

    def test_custom_init(self):
        selector = CFSSelector(
            n_bins=5,
            search_method="greedy_forward",
            search_termination=3,
            locally_predictive=False
        )
        assert selector.n_bins == 5
        assert selector.search_method == "greedy_forward"
        assert selector.search_termination == 3
        assert selector.locally_predictive is False


class TestCFSSelectorFit:

    def test_fit_selects_features(self, classification_data):
        X, y = classification_data
        selector = CFSSelector(
            search_method="greedy_forward",
            search_termination=3,
            locally_predictive=False
        )
        selector.fit(X, y)
        assert selector.selected_features_ is not None
        assert len(selector.selected_features_) >= 1

    def test_merit_computed(self, classification_data):
        X, y = classification_data
        selector = CFSSelector(
            search_method="greedy_forward",
            locally_predictive=False
        )
        selector.fit(X, y)
        assert selector.merit_ is not None
        assert isinstance(selector.merit_, float)

    def test_best_first_search(self, classification_data):
        X, y = classification_data
        selector = CFSSelector(
            search_method="best_first",
            search_termination=2,
            locally_predictive=False
        )
        selector.fit(X, y)
        assert selector.selected_features_ is not None

    def test_transform_output_shape(self, classification_data):
        X, y = classification_data
        selector = CFSSelector(
            search_method="greedy_forward",
            locally_predictive=False
        )
        X_new = selector.fit_transform(X, y)
        assert X_new.shape[0] == X.shape[0]
        assert X_new.shape[1] == len(selector.selected_features_)

    def test_requires_y(self, classification_data):
        X, y = classification_data
        selector = CFSSelector()
        with pytest.raises(ValueError, match="requires target values"):
            selector.fit(X)

    def test_transform_before_fit_raises(self, classification_data):
        X, y = classification_data
        selector = CFSSelector()
        with pytest.raises(RuntimeError):
            selector.transform(X)


class TestCFSSelectorSchema:

    def test_get_parameter_schema(self):
        schema = CFSSelector.get_parameter_schema()
        assert "n_bins" in schema
        assert "search_method" in schema
        assert "search_termination" in schema
        assert "locally_predictive" in schema


# ---- WrapperSelector ----

class TestWrapperSelectorInit:

    def test_init(self):
        clf = SimpleClassifier()
        selector = WrapperSelector(estimator=clf)
        assert selector.cv == 5
        assert selector.scoring == "accuracy"
        assert selector.search_method == "greedy_forward"

    def test_custom_init(self):
        clf = SimpleClassifier()
        selector = WrapperSelector(
            estimator=clf,
            cv=3,
            scoring="f1",
            search_method="greedy_backward"
        )
        assert selector.cv == 3
        assert selector.scoring == "f1"
        assert selector.search_method == "greedy_backward"


class TestWrapperSelectorFit:

    def test_greedy_forward(self, classification_data):
        X, y = classification_data
        clf = SimpleClassifier()
        selector = WrapperSelector(
            estimator=clf,
            cv=3,
            search_method="greedy_forward",
            random_state=42
        )
        selector.fit(X, y)
        assert selector.selected_features_ is not None
        assert selector.cv_score_ is not None

    def test_greedy_backward(self, classification_data):
        X, y = classification_data
        clf = SimpleClassifier()
        selector = WrapperSelector(
            estimator=clf,
            cv=3,
            search_method="greedy_backward",
            random_state=42
        )
        selector.fit(X, y)
        assert selector.selected_features_ is not None

    def test_transform_output(self, classification_data):
        X, y = classification_data
        clf = SimpleClassifier()
        selector = WrapperSelector(
            estimator=clf,
            cv=3,
            search_method="greedy_forward",
            random_state=42
        )
        X_new = selector.fit_transform(X, y)
        assert X_new.shape[0] == X.shape[0]
        assert X_new.shape[1] == len(selector.selected_features_)

    def test_requires_y(self, classification_data):
        X, y = classification_data
        clf = SimpleClassifier()
        selector = WrapperSelector(estimator=clf)
        with pytest.raises(ValueError, match="requires target values"):
            selector.fit(X)

    def test_transform_before_fit_raises(self, classification_data):
        X, y = classification_data
        clf = SimpleClassifier()
        selector = WrapperSelector(estimator=clf)
        with pytest.raises(RuntimeError):
            selector.transform(X)


class TestWrapperSelectorSchema:

    def test_get_parameter_schema(self):
        schema = WrapperSelector.get_parameter_schema()
        assert "estimator" in schema
        assert "cv" in schema
        assert "scoring" in schema
        assert "search_method" in schema
        assert "random_state" in schema
