"""Feature selection transformers.

Merged from: test_selection_variance.py, test_selection_univariate.py, test_selection_subset.py, test_selection_random_subset.py, test_selection_sequential.py
"""

import numpy as np
import pytest
from tuiml.features.selection import VarianceThresholdSelector
from tuiml.features.selection import (
    SelectKBestSelector,
    SelectPercentileSelector,
    SelectThresholdSelector,
    SelectFprSelector,
    GenericUnivariateSelector,
)
from tuiml.features.selection import CFSSelector, WrapperSelector
from tuiml.features.selection import RandomSubsetSelector, BootstrapFeaturesSelector
from tuiml.features.selection import SequentialFeatureSelector, BestFirstSelector


# --------------------------------------------------------------------------
# Tests for VarianceThresholdSelector.
# --------------------------------------------------------------------------

@pytest.fixture
def sample_data():
    """Create sample data with known variance characteristics."""
    np.random.seed(42)
    # Column 0: constant (variance = 0)
    # Column 1: low variance
    # Column 2: high variance
    # Column 3: constant (variance = 0)
    n = 50
    col_const1 = np.zeros(n)                         # constant 0
    col_low = np.random.uniform(0, 0.01, n)          # very low variance
    col_high = np.random.randn(n) * 10               # high variance
    col_const2 = np.zeros(n)                          # constant 0
    X = np.column_stack([col_const1, col_low, col_high, col_const2])
    return X


class TestVarianceThresholdSelectorInit:

    def test_default_init(self):
        selector = VarianceThresholdSelector()
        assert selector.threshold == 0.0

    def test_custom_threshold(self):
        selector = VarianceThresholdSelector(threshold=0.5)
        assert selector.threshold == 0.5

    def test_variances_none_before_fit(self):
        selector = VarianceThresholdSelector()
        assert selector.variances_ is None


class TestVarianceThresholdSelectorFit:

    def test_removes_constant_features(self, sample_data):
        selector = VarianceThresholdSelector(threshold=0.0)
        X_new = selector.fit_transform(sample_data)
        # Columns 0 and 3 are constant, should be removed
        assert X_new.shape[1] == 2

    def test_threshold_parameter(self, sample_data):
        # With a higher threshold, more features should be removed
        selector_low = VarianceThresholdSelector(threshold=0.0)
        selector_high = VarianceThresholdSelector(threshold=1.0)

        X_low = selector_low.fit_transform(sample_data)
        X_high = selector_high.fit_transform(sample_data)

        assert X_high.shape[1] <= X_low.shape[1]

    def test_fit_stores_variances(self, sample_data):
        selector = VarianceThresholdSelector()
        selector.fit(sample_data)
        assert selector.variances_ is not None
        assert len(selector.variances_) == sample_data.shape[1]

    def test_negative_threshold_raises(self):
        selector = VarianceThresholdSelector(threshold=-1.0)
        X = np.random.randn(10, 3)
        with pytest.raises(ValueError, match="threshold must be non-negative"):
            selector.fit(X)


class TestVarianceThresholdSelectorTransform:

    def test_fit_transform(self, sample_data):
        selector = VarianceThresholdSelector(threshold=0.0)
        X_new = selector.fit_transform(sample_data)
        assert X_new.shape[0] == sample_data.shape[0]
        assert X_new.shape[1] < sample_data.shape[1]

    def test_all_constant_features(self):
        X = np.ones((20, 5))
        selector = VarianceThresholdSelector()
        X_new = selector.fit_transform(X)
        assert X_new.shape[1] == 0

    def test_no_features_removed_when_all_vary(self):
        np.random.seed(42)
        X = np.random.randn(50, 4) * 5
        selector = VarianceThresholdSelector(threshold=0.0)
        X_new = selector.fit_transform(X)
        assert X_new.shape[1] == 4


class TestVarianceThresholdSelectorSupport:

    def test_get_support_mask(self, sample_data):
        selector = VarianceThresholdSelector(threshold=0.0)
        selector.fit(sample_data)
        mask = selector.get_support(indices=False)
        assert mask.dtype == bool
        assert len(mask) == sample_data.shape[1]
        # Constant columns (0 and 3) should be False
        assert mask[0] is np.bool_(False)
        assert mask[3] is np.bool_(False)

    def test_get_support_indices(self, sample_data):
        selector = VarianceThresholdSelector(threshold=0.0)
        selector.fit(sample_data)
        indices = selector.get_support(indices=True)
        assert indices.dtype in [np.int64, np.int32, np.intp]
        # Should contain indices 1 and 2 (non-constant columns)
        assert 1 in indices
        assert 2 in indices
        assert 0 not in indices
        assert 3 not in indices

    def test_get_support_before_fit_raises(self):
        selector = VarianceThresholdSelector()
        with pytest.raises(RuntimeError):
            selector.get_support()


class TestVarianceThresholdSelectorSchema:

    def test_get_parameter_schema(self):
        schema = VarianceThresholdSelector.get_parameter_schema()
        assert "threshold" in schema
        assert schema["threshold"]["type"] == "number"
        assert schema["threshold"]["default"] == 0.0


# --------------------------------------------------------------------------
# Tests for univariate feature selectors (SelectKBestSelector, SelectPercentileSelector,
# --------------------------------------------------------------------------

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
def univariate_classification_data():
    """Create simple classification data."""
    np.random.seed(42)
    X = np.random.randn(50, 10)
    y = np.random.randint(0, 2, 50)
    return X, y


class TestSelectKBestSelector:

    def test_default_init(self):
        selector = SelectKBestSelector()
        assert selector.k == 10

    def test_custom_k(self):
        selector = SelectKBestSelector(k=5)
        assert selector.k == 5

    def test_fit_selects_k_features(self, univariate_classification_data):
        X, y = univariate_classification_data
        selector = SelectKBestSelector(score_func=mock_score_func, k=3)
        selector.fit(X, y)
        assert len(selector._selected_indices) == 3

    def test_transform_output_shape(self, univariate_classification_data):
        X, y = univariate_classification_data
        selector = SelectKBestSelector(score_func=mock_score_func, k=4)
        X_new = selector.fit_transform(X, y)
        assert X_new.shape == (50, 4)

    def test_selects_highest_scores(self, univariate_classification_data):
        X, y = univariate_classification_data
        selector = SelectKBestSelector(score_func=mock_score_func, k=3)
        selector.fit(X, y)
        # mock_score_func returns scores [0,1,...,9], so top 3 should be 7,8,9
        indices = selector.get_support(indices=True)
        assert 9 in indices
        assert 8 in indices
        assert 7 in indices

    def test_k_all(self, univariate_classification_data):
        X, y = univariate_classification_data
        selector = SelectKBestSelector(score_func=mock_score_func, k="all")
        X_new = selector.fit_transform(X, y)
        assert X_new.shape[1] == X.shape[1]

    def test_requires_y(self, univariate_classification_data):
        X, y = univariate_classification_data
        selector = SelectKBestSelector(score_func=mock_score_func, k=3)
        with pytest.raises(ValueError, match="requires target values"):
            selector.fit(X)

    def test_get_parameter_schema(self):
        schema = SelectKBestSelector.get_parameter_schema()
        assert "k" in schema
        assert "score_func" in schema


class TestSelectPercentileSelector:

    def test_default_init(self):
        selector = SelectPercentileSelector()
        assert selector.percentile == 10

    def test_custom_percentile(self):
        selector = SelectPercentileSelector(percentile=50)
        assert selector.percentile == 50

    def test_fit_transform_shape(self, univariate_classification_data):
        X, y = univariate_classification_data
        selector = SelectPercentileSelector(score_func=mock_score_func, percentile=50)
        X_new = selector.fit_transform(X, y)
        # 50% of 10 features = 5
        assert X_new.shape[1] == 5

    def test_percentile_100_keeps_all(self, univariate_classification_data):
        X, y = univariate_classification_data
        selector = SelectPercentileSelector(score_func=mock_score_func, percentile=100)
        X_new = selector.fit_transform(X, y)
        assert X_new.shape[1] == X.shape[1]

    def test_requires_y(self, univariate_classification_data):
        X, y = univariate_classification_data
        selector = SelectPercentileSelector(score_func=mock_score_func)
        with pytest.raises(ValueError, match="requires target values"):
            selector.fit(X)

    def test_get_parameter_schema(self):
        schema = SelectPercentileSelector.get_parameter_schema()
        assert "percentile" in schema
        assert schema["percentile"]["default"] == 10


class TestSelectThresholdSelector:

    def test_default_init(self):
        selector = SelectThresholdSelector()
        assert selector.threshold == 0.0

    def test_fit_selects_above_threshold(self, univariate_classification_data):
        X, y = univariate_classification_data
        # mock_score_func returns scores [0,1,...,9]
        selector = SelectThresholdSelector(score_func=mock_score_func, threshold=5.0)
        selector.fit(X, y)
        indices = selector.get_support(indices=True)
        # Features with scores >= 5 are indices 5,6,7,8,9
        assert len(indices) == 5
        for idx in [5, 6, 7, 8, 9]:
            assert idx in indices

    def test_ignore_features(self, univariate_classification_data):
        X, y = univariate_classification_data
        selector = SelectThresholdSelector(
            score_func=mock_score_func,
            threshold=5.0,
            ignore_features=[9]
        )
        selector.fit(X, y)
        indices = selector.get_support(indices=True)
        assert 9 not in indices

    def test_ranking_attribute(self, univariate_classification_data):
        X, y = univariate_classification_data
        selector = SelectThresholdSelector(score_func=mock_score_func, threshold=0.0)
        selector.fit(X, y)
        assert selector.ranking_ is not None
        assert len(selector.ranking_) == X.shape[1]

    def test_get_ranked_features(self, univariate_classification_data):
        X, y = univariate_classification_data
        selector = SelectThresholdSelector(score_func=mock_score_func, threshold=0.0)
        selector.fit(X, y)
        ranked = selector.get_ranked_features()
        assert ranked.shape[1] == 2
        # First row should be the highest-scoring feature
        assert ranked[0, 0] == 9

    def test_requires_y(self, univariate_classification_data):
        X, y = univariate_classification_data
        selector = SelectThresholdSelector(score_func=mock_score_func)
        with pytest.raises(ValueError, match="requires target values"):
            selector.fit(X)

    def test_get_parameter_schema(self):
        schema = SelectThresholdSelector.get_parameter_schema()
        assert "threshold" in schema
        assert "ignore_features" in schema


class TestSelectFprSelector:

    def test_default_init(self):
        selector = SelectFprSelector()
        assert selector.alpha == 0.05

    def test_fit_selects_significant_features(self, univariate_classification_data):
        X, y = univariate_classification_data
        # mock_score_func returns pvalues = 1 - (i/10) for i in 0..9
        # So pvalues are [1.0, 0.9, 0.8, ..., 0.1]
        # Features with pvalue < 0.15 are index 9 (pvalue=0.1)
        selector = SelectFprSelector(score_func=mock_score_func, alpha=0.15)
        selector.fit(X, y)
        indices = selector.get_support(indices=True)
        assert 9 in indices

    def test_strict_alpha_selects_fewer(self, univariate_classification_data):
        X, y = univariate_classification_data
        selector_loose = SelectFprSelector(score_func=mock_score_func, alpha=0.5)
        selector_strict = SelectFprSelector(score_func=mock_score_func, alpha=0.05)

        X_loose = selector_loose.fit_transform(X, y)
        X_strict = selector_strict.fit_transform(X, y)

        assert X_strict.shape[1] <= X_loose.shape[1]

    def test_requires_pvalues(self, univariate_classification_data):
        X, y = univariate_classification_data
        selector = SelectFprSelector(score_func=mock_score_func_no_pvalues, alpha=0.05)
        with pytest.raises(ValueError, match="p-values"):
            selector.fit(X, y)

    def test_requires_y(self, univariate_classification_data):
        X, y = univariate_classification_data
        selector = SelectFprSelector(score_func=mock_score_func)
        with pytest.raises(ValueError, match="requires target values"):
            selector.fit(X)

    def test_get_parameter_schema(self):
        schema = SelectFprSelector.get_parameter_schema()
        assert "alpha" in schema
        assert schema["alpha"]["default"] == 0.05


class TestGenericUnivariateSelector:

    def test_default_init(self):
        selector = GenericUnivariateSelector(score_func=mock_score_func)
        assert selector.mode == "k_best"
        assert selector.param == 10

    def test_k_best_mode(self, univariate_classification_data):
        X, y = univariate_classification_data
        selector = GenericUnivariateSelector(
            score_func=mock_score_func, mode="k_best", param=3
        )
        X_new = selector.fit_transform(X, y)
        assert X_new.shape[1] == 3

    def test_percentile_mode(self, univariate_classification_data):
        X, y = univariate_classification_data
        selector = GenericUnivariateSelector(
            score_func=mock_score_func, mode="percentile", param=50
        )
        X_new = selector.fit_transform(X, y)
        assert X_new.shape[1] == 5

    def test_fpr_mode(self, univariate_classification_data):
        X, y = univariate_classification_data
        selector = GenericUnivariateSelector(
            score_func=mock_score_func, mode="fpr", param=0.5
        )
        X_new = selector.fit_transform(X, y)
        # pvalues are [1.0, 0.9, ..., 0.1], those < 0.5 are indices 6,7,8,9
        assert X_new.shape[1] >= 1

    def test_invalid_mode_raises(self, univariate_classification_data):
        X, y = univariate_classification_data
        selector = GenericUnivariateSelector(
            score_func=mock_score_func, mode="invalid_mode"
        )
        with pytest.raises(ValueError, match="mode must be one of"):
            selector.fit(X, y)

    def test_requires_y(self, univariate_classification_data):
        X, y = univariate_classification_data
        selector = GenericUnivariateSelector(score_func=mock_score_func)
        with pytest.raises(ValueError, match="y is required"):
            selector.fit(X)

    def test_requires_score_func(self, univariate_classification_data):
        X, y = univariate_classification_data
        selector = GenericUnivariateSelector(score_func=None)
        with pytest.raises(ValueError, match="score_func must be provided"):
            selector.fit(X, y)

    def test_get_parameter_schema(self):
        schema = GenericUnivariateSelector.get_parameter_schema()
        assert "mode" in schema
        assert "param" in schema
        assert "score_func" in schema


# --------------------------------------------------------------------------
# Tests for CFSSelector and WrapperSelector.
# --------------------------------------------------------------------------

class SubsetSimpleClassifier:
    """A simple classifier for testing wrapper selectors."""

    def __init__(self):
        self._majority_class = None

    def fit(self, X, y):
        self._majority_class = int(np.mean(y) >= 0.5)
        return self

    def predict(self, X):
        return np.full(X.shape[0], self._majority_class)


@pytest.fixture
def subset_classification_data():
    """Create classification data with informative features."""
    np.random.seed(42)
    n_samples = 50
    X = np.random.randn(n_samples, 6)
    # Make features 0 and 1 informative
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X, y


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

    def test_fit_selects_features(self, subset_classification_data):
        X, y = subset_classification_data
        selector = CFSSelector(
            search_method="greedy_forward",
            search_termination=3,
            locally_predictive=False
        )
        selector.fit(X, y)
        assert selector.selected_features_ is not None
        assert len(selector.selected_features_) >= 1

    def test_merit_computed(self, subset_classification_data):
        X, y = subset_classification_data
        selector = CFSSelector(
            search_method="greedy_forward",
            locally_predictive=False
        )
        selector.fit(X, y)
        assert selector.merit_ is not None
        assert isinstance(selector.merit_, float)

    def test_best_first_search(self, subset_classification_data):
        X, y = subset_classification_data
        selector = CFSSelector(
            search_method="best_first",
            search_termination=2,
            locally_predictive=False
        )
        selector.fit(X, y)
        assert selector.selected_features_ is not None

    def test_transform_output_shape(self, subset_classification_data):
        X, y = subset_classification_data
        selector = CFSSelector(
            search_method="greedy_forward",
            locally_predictive=False
        )
        X_new = selector.fit_transform(X, y)
        assert X_new.shape[0] == X.shape[0]
        assert X_new.shape[1] == len(selector.selected_features_)

    def test_requires_y(self, subset_classification_data):
        X, y = subset_classification_data
        selector = CFSSelector()
        with pytest.raises(ValueError, match="requires target values"):
            selector.fit(X)

    def test_transform_before_fit_raises(self, subset_classification_data):
        X, y = subset_classification_data
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


class TestWrapperSelectorInit:

    def test_init(self):
        clf = SubsetSimpleClassifier()
        selector = WrapperSelector(estimator=clf)
        assert selector.cv == 5
        assert selector.scoring == "accuracy"
        assert selector.search_method == "greedy_forward"

    def test_custom_init(self):
        clf = SubsetSimpleClassifier()
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

    def test_greedy_forward(self, subset_classification_data):
        X, y = subset_classification_data
        clf = SubsetSimpleClassifier()
        selector = WrapperSelector(
            estimator=clf,
            cv=3,
            search_method="greedy_forward",
            random_state=42
        )
        selector.fit(X, y)
        assert selector.selected_features_ is not None
        assert selector.cv_score_ is not None

    def test_greedy_backward(self, subset_classification_data):
        X, y = subset_classification_data
        clf = SubsetSimpleClassifier()
        selector = WrapperSelector(
            estimator=clf,
            cv=3,
            search_method="greedy_backward",
            random_state=42
        )
        selector.fit(X, y)
        assert selector.selected_features_ is not None

    def test_transform_output(self, subset_classification_data):
        X, y = subset_classification_data
        clf = SubsetSimpleClassifier()
        selector = WrapperSelector(
            estimator=clf,
            cv=3,
            search_method="greedy_forward",
            random_state=42
        )
        X_new = selector.fit_transform(X, y)
        assert X_new.shape[0] == X.shape[0]
        assert X_new.shape[1] == len(selector.selected_features_)

    def test_requires_y(self, subset_classification_data):
        X, y = subset_classification_data
        clf = SubsetSimpleClassifier()
        selector = WrapperSelector(estimator=clf)
        with pytest.raises(ValueError, match="requires target values"):
            selector.fit(X)

    def test_transform_before_fit_raises(self, subset_classification_data):
        X, y = subset_classification_data
        clf = SubsetSimpleClassifier()
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


# --------------------------------------------------------------------------
# Tests for RandomSubsetSelector and BootstrapFeaturesSelector.
# --------------------------------------------------------------------------

@pytest.fixture
def random_subset_sample_data():
    """Create sample data for testing."""
    np.random.seed(42)
    return np.random.randn(30, 20)


class TestRandomSubsetSelectorInit:

    def test_default_init(self):
        selector = RandomSubsetSelector()
        assert selector.n_features == 0.5
        assert selector.invert is False
        assert selector.random_state is None

    def test_custom_init(self):
        selector = RandomSubsetSelector(n_features=5, invert=True, random_state=42)
        assert selector.n_features == 5
        assert selector.invert is True
        assert selector.random_state == 42


class TestRandomSubsetSelectorFit:

    def test_fit_fraction(self, random_subset_sample_data):
        selector = RandomSubsetSelector(n_features=0.5, random_state=42)
        selector.fit(random_subset_sample_data)
        # 50% of 20 = 10
        assert selector.n_features_selected_ == 10
        assert len(selector.selected_features_) == 10

    def test_fit_absolute(self, random_subset_sample_data):
        selector = RandomSubsetSelector(n_features=7, random_state=42)
        selector.fit(random_subset_sample_data)
        assert selector.n_features_selected_ == 7

    def test_transform_output_shape(self, random_subset_sample_data):
        selector = RandomSubsetSelector(n_features=0.3, random_state=42)
        X_new = selector.fit_transform(random_subset_sample_data)
        expected_n = max(1, int(round(20 * 0.3)))
        assert X_new.shape == (30, expected_n)

    def test_invert_selection(self, random_subset_sample_data):
        selector_normal = RandomSubsetSelector(n_features=5, random_state=42, invert=False)
        selector_invert = RandomSubsetSelector(n_features=5, random_state=42, invert=True)

        selector_normal.fit(random_subset_sample_data)
        selector_invert.fit(random_subset_sample_data)

        # Invert should select the complement
        assert selector_normal.n_features_selected_ + selector_invert.n_features_selected_ == 20

    def test_unsupervised_no_y_needed(self, random_subset_sample_data):
        selector = RandomSubsetSelector(n_features=5, random_state=42)
        # Should not raise even without y
        selector.fit(random_subset_sample_data)
        assert selector.n_features_selected_ == 5


class TestRandomSubsetSelectorReproducibility:

    def test_same_random_state_same_result(self, random_subset_sample_data):
        selector1 = RandomSubsetSelector(n_features=5, random_state=42)
        selector2 = RandomSubsetSelector(n_features=5, random_state=42)

        X1 = selector1.fit_transform(random_subset_sample_data)
        X2 = selector2.fit_transform(random_subset_sample_data)

        np.testing.assert_array_equal(X1, X2)

    def test_different_random_state_different_result(self, random_subset_sample_data):
        selector1 = RandomSubsetSelector(n_features=5, random_state=42)
        selector2 = RandomSubsetSelector(n_features=5, random_state=99)

        selector1.fit(random_subset_sample_data)
        selector2.fit(random_subset_sample_data)

        # Very likely to be different (though not guaranteed for small n)
        assert not np.array_equal(
            selector1.selected_features_, selector2.selected_features_
        )


class TestRandomSubsetSelectorTransform:

    def test_transform_wrong_n_features_raises(self, random_subset_sample_data):
        selector = RandomSubsetSelector(n_features=5, random_state=42)
        selector.fit(random_subset_sample_data)
        X_wrong = np.random.randn(10, 15)  # wrong number of features
        with pytest.raises(ValueError, match="features"):
            selector.transform(X_wrong)


class TestRandomSubsetSelectorSchema:

    def test_get_parameter_schema(self):
        schema = RandomSubsetSelector.get_parameter_schema()
        assert "n_features" in schema
        assert "invert" in schema
        assert "random_state" in schema


class TestBootstrapFeaturesSelectorInit:

    def test_default_init(self):
        selector = BootstrapFeaturesSelector()
        assert selector.n_features == "sqrt"
        assert selector.random_state is None

    def test_custom_init(self):
        selector = BootstrapFeaturesSelector(n_features="log2", random_state=42)
        assert selector.n_features == "log2"
        assert selector.random_state == 42


class TestBootstrapFeaturesSelectorFit:

    def test_sqrt_selection(self, random_subset_sample_data):
        selector = BootstrapFeaturesSelector(n_features="sqrt", random_state=42)
        selector.fit(random_subset_sample_data)
        # sqrt(20) ~ 4, but bootstrap may produce fewer unique indices
        assert selector.n_features_selected_ >= 1
        assert selector.n_features_selected_ <= 20

    def test_log2_selection(self, random_subset_sample_data):
        selector = BootstrapFeaturesSelector(n_features="log2", random_state=42)
        selector.fit(random_subset_sample_data)
        assert selector.n_features_selected_ >= 1

    def test_fraction_selection(self, random_subset_sample_data):
        selector = BootstrapFeaturesSelector(n_features=0.5, random_state=42)
        selector.fit(random_subset_sample_data)
        assert selector.n_features_selected_ >= 1

    def test_absolute_selection(self, random_subset_sample_data):
        selector = BootstrapFeaturesSelector(n_features=8, random_state=42)
        selector.fit(random_subset_sample_data)
        # Bootstrap samples with replacement then takes unique, so may be fewer
        assert selector.n_features_selected_ >= 1
        assert selector.n_features_selected_ <= 8

    def test_transform_output_shape(self, random_subset_sample_data):
        selector = BootstrapFeaturesSelector(n_features="sqrt", random_state=42)
        X_new = selector.fit_transform(random_subset_sample_data)
        assert X_new.shape[0] == random_subset_sample_data.shape[0]
        assert X_new.shape[1] == selector.n_features_selected_

    def test_reproducibility(self, random_subset_sample_data):
        selector1 = BootstrapFeaturesSelector(n_features="sqrt", random_state=42)
        selector2 = BootstrapFeaturesSelector(n_features="sqrt", random_state=42)

        X1 = selector1.fit_transform(random_subset_sample_data)
        X2 = selector2.fit_transform(random_subset_sample_data)

        np.testing.assert_array_equal(X1, X2)

    def test_invalid_n_features_string_raises(self, random_subset_sample_data):
        selector = BootstrapFeaturesSelector(n_features="invalid")
        with pytest.raises(ValueError, match="Unknown n_features string"):
            selector.fit(random_subset_sample_data)


class TestBootstrapFeaturesSelectorSchema:

    def test_get_parameter_schema(self):
        schema = BootstrapFeaturesSelector.get_parameter_schema()
        assert "n_features" in schema
        assert "random_state" in schema


# --------------------------------------------------------------------------
# Tests for SequentialFeatureSelector and BestFirstSelector.
# --------------------------------------------------------------------------

class SequentialSimpleClassifier:
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
def sequential_classification_data():
    """Create classification data with some informative features."""
    np.random.seed(42)
    n_samples = 40
    # Features 0 and 1 are informative, rest are noise
    X = np.random.randn(n_samples, 5)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    return X, y


@pytest.fixture
def simple_estimator():
    return SequentialSimpleClassifier()


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

    def test_forward_selection(self, sequential_classification_data, simple_estimator):
        X, y = sequential_classification_data
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

    def test_backward_selection(self, sequential_classification_data, simple_estimator):
        X, y = sequential_classification_data
        selector = SequentialFeatureSelector(
            estimator=simple_estimator,
            n_features_to_select=2,
            direction="backward",
            cv=3,
            random_state=42
        )
        selector.fit(X, y)
        assert selector.n_features_to_select_ == 2

    def test_transform_output_shape(self, sequential_classification_data, simple_estimator):
        X, y = sequential_classification_data
        selector = SequentialFeatureSelector(
            estimator=simple_estimator,
            n_features_to_select=3,
            direction="forward",
            cv=3,
            random_state=42
        )
        X_new = selector.fit_transform(X, y)
        assert X_new.shape == (X.shape[0], 3)

    def test_support_attribute(self, sequential_classification_data, simple_estimator):
        X, y = sequential_classification_data
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

    def test_requires_y(self, sequential_classification_data, simple_estimator):
        X, y = sequential_classification_data
        selector = SequentialFeatureSelector(
            estimator=simple_estimator,
            n_features_to_select=2,
        )
        with pytest.raises(ValueError, match="requires target values"):
            selector.fit(X)

    def test_requires_estimator(self, sequential_classification_data):
        X, y = sequential_classification_data
        selector = SequentialFeatureSelector(n_features_to_select=2)
        with pytest.raises(ValueError, match="estimator must be provided"):
            selector.fit(X, y)

    def test_fractional_n_features(self, sequential_classification_data, simple_estimator):
        X, y = sequential_classification_data
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

    def test_forward_best_first(self, sequential_classification_data, simple_estimator):
        X, y = sequential_classification_data
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

    def test_transform_output(self, sequential_classification_data, simple_estimator):
        X, y = sequential_classification_data
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

    def test_requires_y(self, sequential_classification_data, simple_estimator):
        X, y = sequential_classification_data
        selector = BestFirstSelector(estimator=simple_estimator)
        with pytest.raises(ValueError, match="requires target values"):
            selector.fit(X)

    def test_requires_estimator(self, sequential_classification_data):
        X, y = sequential_classification_data
        selector = BestFirstSelector()
        with pytest.raises(ValueError, match="estimator must be provided"):
            selector.fit(X, y)

    def test_transform_before_fit_raises(self, sequential_classification_data):
        X, y = sequential_classification_data
        selector = BestFirstSelector()
        with pytest.raises(RuntimeError):
            selector.transform(X)


class TestBestFirstSelectorSchema:

    def test_get_parameter_schema(self):
        schema = BestFirstSelector.get_parameter_schema()
        assert "direction" in schema
        assert "search_termination" in schema
        assert "cv" in schema
