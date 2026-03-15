"""Tests for SequentialFeatureSelector and BestFirstSelector."""

import numpy as np
import pytest

from tuiml.features.selection import SequentialFeatureSelector, BestFirstSelector


class SimpleClassifier:
    """A simple classifier for testing sequential selectors."""

    def __init__(self):
        self._weights = None

    def get_params(self):
        return {}

    def fit(self, X, y):
        # Simple majority class predictor per feature sign
        self._majority_class = int(np.mean(y) >= 0.5)
        return self

    def predict(self, X):
        return np.full(X.shape[0], self._majority_class)


@pytest.fixture
def classification_data():
    """Create classification data with some informative features."""
    np.random.seed(42)
    n_samples = 40
    # Features 0 and 1 are informative, rest are noise
    X = np.random.randn(n_samples, 5)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X, y


@pytest.fixture
def simple_estimator():
    return SimpleClassifier()


# ---- SequentialFeatureSelector ----

class TestSequentialFeatureSelectorInit:

    def test_default_init(self):
        selector = SequentialFeatureSelector()
        assert selector.n_features_to_select == "auto"
        assert selector.direction == "forward"
        assert selector.cv == 5
        assert selector.tol == 0.0

    def test_custom_init(self):
        selector = SequentialFeatureSelector(
            n_features_to_select=3,
            direction="backward",
            cv=3,
            tol=0.01
        )
        assert selector.n_features_to_select == 3
        assert selector.direction == "backward"
        assert selector.cv == 3
        assert selector.tol == 0.01


class TestSequentialFeatureSelectorFit:

    def test_forward_selection(self, classification_data, simple_estimator):
        X, y = classification_data
        selector = SequentialFeatureSelector(
            estimator=simple_estimator,
            n_features_to_select=2,
            direction="forward",
            cv=3,
            random_state=42
        )
        selector.fit(X, y)
        assert selector.n_features_to_select_ == 2
        assert len(selector._selected_indices) == 2

    def test_backward_selection(self, classification_data, simple_estimator):
        X, y = classification_data
        selector = SequentialFeatureSelector(
            estimator=simple_estimator,
            n_features_to_select=2,
            direction="backward",
            cv=3,
            random_state=42
        )
        selector.fit(X, y)
        assert selector.n_features_to_select_ == 2

    def test_transform_output_shape(self, classification_data, simple_estimator):
        X, y = classification_data
        selector = SequentialFeatureSelector(
            estimator=simple_estimator,
            n_features_to_select=3,
            direction="forward",
            cv=3,
            random_state=42
        )
        X_new = selector.fit_transform(X, y)
        assert X_new.shape == (X.shape[0], 3)

    def test_support_attribute(self, classification_data, simple_estimator):
        X, y = classification_data
        selector = SequentialFeatureSelector(
            estimator=simple_estimator,
            n_features_to_select=2,
            direction="forward",
            cv=3,
            random_state=42
        )
        selector.fit(X, y)
        assert selector.support_ is not None
        assert selector.support_.dtype == bool
        assert selector.support_.sum() == 2

    def test_requires_y(self, classification_data, simple_estimator):
        X, y = classification_data
        selector = SequentialFeatureSelector(
            estimator=simple_estimator,
            n_features_to_select=2,
        )
        with pytest.raises(ValueError, match="requires target values"):
            selector.fit(X)

    def test_requires_estimator(self, classification_data):
        X, y = classification_data
        selector = SequentialFeatureSelector(n_features_to_select=2)
        with pytest.raises(ValueError, match="estimator must be provided"):
            selector.fit(X, y)

    def test_fractional_n_features(self, classification_data, simple_estimator):
        X, y = classification_data
        selector = SequentialFeatureSelector(
            estimator=simple_estimator,
            n_features_to_select=0.4,  # 40% of 5 = 2
            direction="forward",
            cv=3,
            random_state=42
        )
        selector.fit(X, y)
        assert selector.n_features_to_select_ == 2


class TestSequentialFeatureSelectorSchema:

    def test_get_parameter_schema(self):
        schema = SequentialFeatureSelector.get_parameter_schema()
        assert "direction" in schema
        assert "cv" in schema
        assert "tol" in schema
        assert "n_features_to_select" in schema


# ---- BestFirstSelector ----

class TestBestFirstSelectorInit:

    def test_default_init(self):
        selector = BestFirstSelector()
        assert selector.direction == "forward"
        assert selector.search_termination == 5
        assert selector.cv == 5

    def test_custom_init(self):
        selector = BestFirstSelector(
            direction="backward",
            search_termination=3,
            cv=3
        )
        assert selector.direction == "backward"
        assert selector.search_termination == 3
        assert selector.cv == 3


class TestBestFirstSelectorFit:

    def test_forward_best_first(self, classification_data, simple_estimator):
        X, y = classification_data
        selector = BestFirstSelector(
            estimator=simple_estimator,
            direction="forward",
            search_termination=2,
            cv=3,
            random_state=42
        )
        selector.fit(X, y)
        assert selector.n_features_selected_ is not None
        assert selector.n_features_selected_ >= 0

    def test_transform_output(self, classification_data, simple_estimator):
        X, y = classification_data
        selector = BestFirstSelector(
            estimator=simple_estimator,
            direction="forward",
            search_termination=2,
            cv=3,
            random_state=42
        )
        X_new = selector.fit_transform(X, y)
        assert X_new.shape[0] == X.shape[0]
        assert X_new.shape[1] == selector.n_features_selected_

    def test_requires_y(self, classification_data, simple_estimator):
        X, y = classification_data
        selector = BestFirstSelector(estimator=simple_estimator)
        with pytest.raises(ValueError, match="requires target values"):
            selector.fit(X)

    def test_requires_estimator(self, classification_data):
        X, y = classification_data
        selector = BestFirstSelector()
        with pytest.raises(ValueError, match="estimator must be provided"):
            selector.fit(X, y)

    def test_transform_before_fit_raises(self, classification_data):
        X, y = classification_data
        selector = BestFirstSelector()
        with pytest.raises(RuntimeError):
            selector.transform(X)


class TestBestFirstSelectorSchema:

    def test_get_parameter_schema(self):
        schema = BestFirstSelector.get_parameter_schema()
        assert "direction" in schema
        assert "search_termination" in schema
        assert "cv" in schema
