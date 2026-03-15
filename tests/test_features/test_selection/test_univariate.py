"""Tests for univariate feature selectors (SelectKBestSelector, SelectPercentileSelector,
SelectThresholdSelector, SelectFprSelector, and GenericUnivariateSelector)."""

import numpy as np
import pytest

from tuiml.features.selection import (
    SelectKBestSelector,
    SelectPercentileSelector,
    SelectThresholdSelector,
    SelectFprSelector,
    GenericUnivariateSelector,
)


def mock_score_func(X, y):
    """A mock score function that returns deterministic scores and p-values."""
    n_features = X.shape[1]
    scores = np.arange(n_features, dtype=float)
    pvalues = 1.0 - (scores / n_features)
    return scores, pvalues


def mock_score_func_no_pvalues(X, y):
    """A mock score function that returns only scores (no p-values)."""
    return np.arange(X.shape[1], dtype=float)


@pytest.fixture
def classification_data():
    """Create simple classification data."""
    np.random.seed(42)
    X = np.random.randn(50, 10)
    y = np.random.randint(0, 2, 50)
    return X, y


# ---- SelectKBestSelector ----

class TestSelectKBestSelector:

    def test_default_init(self):
        selector = SelectKBestSelector()
        assert selector.k == 10

    def test_custom_k(self):
        selector = SelectKBestSelector(k=5)
        assert selector.k == 5

    def test_fit_selects_k_features(self, classification_data):
        X, y = classification_data
        selector = SelectKBestSelector(score_func=mock_score_func, k=3)
        selector.fit(X, y)
        assert len(selector._selected_indices) == 3

    def test_transform_output_shape(self, classification_data):
        X, y = classification_data
        selector = SelectKBestSelector(score_func=mock_score_func, k=4)
        X_new = selector.fit_transform(X, y)
        assert X_new.shape == (50, 4)

    def test_selects_highest_scores(self, classification_data):
        X, y = classification_data
        selector = SelectKBestSelector(score_func=mock_score_func, k=3)
        selector.fit(X, y)
        # mock_score_func returns scores [0,1,...,9], so top 3 should be 7,8,9
        indices = selector.get_support(indices=True)
        assert 9 in indices
        assert 8 in indices
        assert 7 in indices

    def test_k_all(self, classification_data):
        X, y = classification_data
        selector = SelectKBestSelector(score_func=mock_score_func, k="all")
        X_new = selector.fit_transform(X, y)
        assert X_new.shape[1] == X.shape[1]

    def test_requires_y(self, classification_data):
        X, y = classification_data
        selector = SelectKBestSelector(score_func=mock_score_func, k=3)
        with pytest.raises(ValueError, match="requires target values"):
            selector.fit(X)

    def test_transform_before_fit_raises(self, classification_data):
        X, y = classification_data
        selector = SelectKBestSelector(score_func=mock_score_func, k=3)
        with pytest.raises(RuntimeError):
            selector.transform(X)

    def test_get_parameter_schema(self):
        schema = SelectKBestSelector.get_parameter_schema()
        assert "k" in schema
        assert "score_func" in schema


# ---- SelectPercentileSelector ----

class TestSelectPercentileSelector:

    def test_default_init(self):
        selector = SelectPercentileSelector()
        assert selector.percentile == 10

    def test_custom_percentile(self):
        selector = SelectPercentileSelector(percentile=50)
        assert selector.percentile == 50

    def test_fit_transform_shape(self, classification_data):
        X, y = classification_data
        selector = SelectPercentileSelector(score_func=mock_score_func, percentile=50)
        X_new = selector.fit_transform(X, y)
        # 50% of 10 features = 5
        assert X_new.shape[1] == 5

    def test_percentile_100_keeps_all(self, classification_data):
        X, y = classification_data
        selector = SelectPercentileSelector(score_func=mock_score_func, percentile=100)
        X_new = selector.fit_transform(X, y)
        assert X_new.shape[1] == X.shape[1]

    def test_requires_y(self, classification_data):
        X, y = classification_data
        selector = SelectPercentileSelector(score_func=mock_score_func)
        with pytest.raises(ValueError, match="requires target values"):
            selector.fit(X)

    def test_get_parameter_schema(self):
        schema = SelectPercentileSelector.get_parameter_schema()
        assert "percentile" in schema
        assert schema["percentile"]["default"] == 10


# ---- SelectThresholdSelector ----

class TestSelectThresholdSelector:

    def test_default_init(self):
        selector = SelectThresholdSelector()
        assert selector.threshold == 0.0

    def test_fit_selects_above_threshold(self, classification_data):
        X, y = classification_data
        # mock_score_func returns scores [0,1,...,9]
        selector = SelectThresholdSelector(score_func=mock_score_func, threshold=5.0)
        selector.fit(X, y)
        indices = selector.get_support(indices=True)
        # Features with scores >= 5 are indices 5,6,7,8,9
        assert len(indices) == 5
        for idx in [5, 6, 7, 8, 9]:
            assert idx in indices

    def test_ignore_features(self, classification_data):
        X, y = classification_data
        selector = SelectThresholdSelector(
            score_func=mock_score_func,
            threshold=5.0,
            ignore_features=[9]
        )
        selector.fit(X, y)
        indices = selector.get_support(indices=True)
        assert 9 not in indices

    def test_ranking_attribute(self, classification_data):
        X, y = classification_data
        selector = SelectThresholdSelector(score_func=mock_score_func, threshold=0.0)
        selector.fit(X, y)
        assert selector.ranking_ is not None
        assert len(selector.ranking_) == X.shape[1]

    def test_get_ranked_features(self, classification_data):
        X, y = classification_data
        selector = SelectThresholdSelector(score_func=mock_score_func, threshold=0.0)
        selector.fit(X, y)
        ranked = selector.get_ranked_features()
        assert ranked.shape[1] == 2
        # First row should be the highest-scoring feature
        assert ranked[0, 0] == 9

    def test_requires_y(self, classification_data):
        X, y = classification_data
        selector = SelectThresholdSelector(score_func=mock_score_func)
        with pytest.raises(ValueError, match="requires target values"):
            selector.fit(X)

    def test_get_parameter_schema(self):
        schema = SelectThresholdSelector.get_parameter_schema()
        assert "threshold" in schema
        assert "ignore_features" in schema


# ---- SelectFprSelector ----

class TestSelectFprSelector:

    def test_default_init(self):
        selector = SelectFprSelector()
        assert selector.alpha == 0.05

    def test_fit_selects_significant_features(self, classification_data):
        X, y = classification_data
        # mock_score_func returns pvalues = 1 - (i/10) for i in 0..9
        # So pvalues are [1.0, 0.9, 0.8, ..., 0.1]
        # Features with pvalue < 0.15 are index 9 (pvalue=0.1)
        selector = SelectFprSelector(score_func=mock_score_func, alpha=0.15)
        selector.fit(X, y)
        indices = selector.get_support(indices=True)
        assert 9 in indices

    def test_strict_alpha_selects_fewer(self, classification_data):
        X, y = classification_data
        selector_loose = SelectFprSelector(score_func=mock_score_func, alpha=0.5)
        selector_strict = SelectFprSelector(score_func=mock_score_func, alpha=0.05)

        X_loose = selector_loose.fit_transform(X, y)
        X_strict = selector_strict.fit_transform(X, y)

        assert X_strict.shape[1] <= X_loose.shape[1]

    def test_requires_pvalues(self, classification_data):
        X, y = classification_data
        selector = SelectFprSelector(score_func=mock_score_func_no_pvalues, alpha=0.05)
        with pytest.raises(ValueError, match="p-values"):
            selector.fit(X, y)

    def test_requires_y(self, classification_data):
        X, y = classification_data
        selector = SelectFprSelector(score_func=mock_score_func)
        with pytest.raises(ValueError, match="requires target values"):
            selector.fit(X)

    def test_get_parameter_schema(self):
        schema = SelectFprSelector.get_parameter_schema()
        assert "alpha" in schema
        assert schema["alpha"]["default"] == 0.05


# ---- GenericUnivariateSelector ----

class TestGenericUnivariateSelector:

    def test_default_init(self):
        selector = GenericUnivariateSelector(score_func=mock_score_func)
        assert selector.mode == "k_best"
        assert selector.param == 10

    def test_k_best_mode(self, classification_data):
        X, y = classification_data
        selector = GenericUnivariateSelector(
            score_func=mock_score_func, mode="k_best", param=3
        )
        X_new = selector.fit_transform(X, y)
        assert X_new.shape[1] == 3

    def test_percentile_mode(self, classification_data):
        X, y = classification_data
        selector = GenericUnivariateSelector(
            score_func=mock_score_func, mode="percentile", param=50
        )
        X_new = selector.fit_transform(X, y)
        assert X_new.shape[1] == 5

    def test_fpr_mode(self, classification_data):
        X, y = classification_data
        selector = GenericUnivariateSelector(
            score_func=mock_score_func, mode="fpr", param=0.5
        )
        X_new = selector.fit_transform(X, y)
        # pvalues are [1.0, 0.9, ..., 0.1], those < 0.5 are indices 6,7,8,9
        assert X_new.shape[1] >= 1

    def test_invalid_mode_raises(self, classification_data):
        X, y = classification_data
        selector = GenericUnivariateSelector(
            score_func=mock_score_func, mode="invalid_mode"
        )
        with pytest.raises(ValueError, match="mode must be one of"):
            selector.fit(X, y)

    def test_requires_y(self, classification_data):
        X, y = classification_data
        selector = GenericUnivariateSelector(score_func=mock_score_func)
        with pytest.raises(ValueError, match="y is required"):
            selector.fit(X)

    def test_requires_score_func(self, classification_data):
        X, y = classification_data
        selector = GenericUnivariateSelector(score_func=None)
        with pytest.raises(ValueError, match="score_func must be provided"):
            selector.fit(X, y)

    def test_get_parameter_schema(self):
        schema = GenericUnivariateSelector.get_parameter_schema()
        assert "mode" in schema
        assert "param" in schema
        assert "score_func" in schema
