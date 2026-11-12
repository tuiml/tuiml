"""Resampling filters.

Merged from: test_sampling_oversampling.py, test_sampling_undersampling.py, test_sampling_smote.py, test_sampling_class_balancer.py
"""

import numpy as np
import pytest
from tuiml.preprocessing.sampling.oversampling import RandomOverSampler
from tuiml.preprocessing.sampling.undersampling import RandomUnderSampler
from tuiml.preprocessing.sampling.smote import SMOTESampler , SMOTENCSampler
from collections import Counter
from tuiml.preprocessing.sampling.class_balance import ClassBalanceSampler


# --------------------------------------------------------------------------
# Tests for RandomOverSampler.
# --------------------------------------------------------------------------

class TestRandomOverSampler:
    """Tests for RandomOverSampler."""

    @pytest.fixture
    def imbalanced_data(self):
        np.random.seed(42)
        X = np.random.randn(60, 2)
        y = np.array([0] * 50 + [1] * 10)
        return X, y

    def test_init_defaults(self):
        sampler = RandomOverSampler()
        assert sampler.sampling_strategy == "auto"
        assert sampler.random_state is None
        assert sampler.shrinkage is None

    def test_fit_resample_balances_classes(self, imbalanced_data):
        X, y = imbalanced_data
        sampler = RandomOverSampler(random_state=42)
        X_res, y_res = sampler.fit_resample(X, y)
        unique, counts = np.unique(y_res, return_counts=True)
        assert counts[0] == counts[1]

    def test_output_has_more_samples(self, imbalanced_data):
        X, y = imbalanced_data
        sampler = RandomOverSampler(random_state=42)
        X_res, y_res = sampler.fit_resample(X, y)
        assert len(X_res) > len(X)

    def test_shrinkage_adds_noise(self, imbalanced_data):
        X, y = imbalanced_data
        sampler = RandomOverSampler(random_state=42, shrinkage=0.1)
        X_res, y_res = sampler.fit_resample(X, y)
        new_samples = X_res[len(X):]
        X_minority = X[y == 1]
        assert len(new_samples) > 0

        # Without shrinkage every new sample is an exact copy of a minority row;
        # with it, each one is jittered. Asserting only that new samples exist
        # would pass either way and so would not test shrinkage at all.
        exact_copies = sum(
            bool(np.any(np.all(np.isclose(X_minority, row), axis=1)))
            for row in new_samples
        )
        assert exact_copies == 0

    def test_reproducibility(self, imbalanced_data):
        X, y = imbalanced_data
        s1 = RandomOverSampler(random_state=42)
        X1, y1 = s1.fit_resample(X, y)
        s2 = RandomOverSampler(random_state=42)
        X2, y2 = s2.fit_resample(X, y)
        np.testing.assert_allclose(X1, X2)

    def test_transform_raises(self):
        sampler = RandomOverSampler()
        with pytest.raises(NotImplementedError):
            sampler.transform(np.array([[1, 2]]))

    def test_get_parameter_schema(self):
        schema = RandomOverSampler.get_parameter_schema()
        assert "sampling_strategy" in schema
        assert "random_state" in schema
        assert "shrinkage" in schema


# --------------------------------------------------------------------------
# Tests for RandomUnderSampler.
# --------------------------------------------------------------------------

class TestRandomUnderSampler:
    """Tests for RandomUnderSampler."""

    @pytest.fixture
    def imbalanced_data(self):
        np.random.seed(42)
        X = np.random.randn(60, 2)
        y = np.array([0] * 50 + [1] * 10)
        return X, y

    def test_init_defaults(self):
        sampler = RandomUnderSampler()
        assert sampler.sampling_strategy == "auto"
        assert sampler.random_state is None
        assert sampler.replacement is False

    def test_fit_resample_balances_classes(self, imbalanced_data):
        X, y = imbalanced_data
        sampler = RandomUnderSampler(random_state=42)
        X_res, y_res = sampler.fit_resample(X, y)
        unique, counts = np.unique(y_res, return_counts=True)
        # Majority should be reduced to match minority
        assert counts[0] == counts[1]

    def test_output_has_fewer_samples(self, imbalanced_data):
        X, y = imbalanced_data
        sampler = RandomUnderSampler(random_state=42)
        X_res, y_res = sampler.fit_resample(X, y)
        assert len(X_res) < len(X)

    def test_minority_class_preserved(self, imbalanced_data):
        X, y = imbalanced_data
        sampler = RandomUnderSampler(random_state=42)
        X_res, y_res = sampler.fit_resample(X, y)
        # All minority samples should still be present
        minority_count = np.sum(y_res == 1)
        assert minority_count == 10

    def test_reproducibility(self, imbalanced_data):
        X, y = imbalanced_data
        s1 = RandomUnderSampler(random_state=42)
        X1, y1 = s1.fit_resample(X, y)
        s2 = RandomUnderSampler(random_state=42)
        X2, y2 = s2.fit_resample(X, y)
        np.testing.assert_allclose(X1, X2)

    def test_transform_raises(self):
        sampler = RandomUnderSampler()
        with pytest.raises(NotImplementedError):
            sampler.transform(np.array([[1, 2]]))

    def test_get_parameter_schema(self):
        schema = RandomUnderSampler.get_parameter_schema()
        assert "sampling_strategy" in schema
        assert "random_state" in schema
        assert "replacement" in schema


# --------------------------------------------------------------------------
# Tests for SMOTESampler and variants.
# --------------------------------------------------------------------------

class TestSMOTESampler:
    """Tests for SMOTESampler and variants."""

    @pytest.fixture
    def imbalanced_data(self):
        np.random.seed(42)
        X_majority = np.random.randn(50, 2) + 5
        X_minority = np.random.randn(10, 2)
        X = np.vstack([X_majority, X_minority])
        y = np.array([0] * 50 + [1] * 10)
        return X, y

    def test_init_defaults(self):
        sampler = SMOTESampler()
        assert sampler.sampling_strategy == "auto"
        assert sampler.k_neighbors == 5
        assert sampler.random_state is None

    def test_fit_resample_balances_classes(self, imbalanced_data):
        X, y = imbalanced_data
        sampler = SMOTESampler(k_neighbors=3, random_state=42)
        X_res, y_res = sampler.fit_resample(X, y)
        # After SMOTE, minority should match majority
        unique, counts = np.unique(y_res, return_counts=True)
        assert counts[0] == counts[1]

    def test_output_has_more_samples(self, imbalanced_data):
        X, y = imbalanced_data
        sampler = SMOTESampler(k_neighbors=3, random_state=42)
        X_res, y_res = sampler.fit_resample(X, y)
        assert len(X_res) > len(X)
        assert len(y_res) > len(y)

    def test_original_data_preserved(self, imbalanced_data):
        X, y = imbalanced_data
        sampler = SMOTESampler(k_neighbors=3, random_state=42)
        X_res, y_res = sampler.fit_resample(X, y)
        # Original data should be in the beginning
        np.testing.assert_allclose(X_res[:len(X)], X)

    def test_reproducibility(self, imbalanced_data):
        X, y = imbalanced_data
        s1 = SMOTESampler(k_neighbors=3, random_state=42)
        X1, y1 = s1.fit_resample(X, y)
        s2 = SMOTESampler(k_neighbors=3, random_state=42)
        X2, y2 = s2.fit_resample(X, y)
        np.testing.assert_allclose(X1, X2)
        np.testing.assert_allclose(y1, y2)

    def test_transform_raises(self):
        sampler = SMOTESampler()
        X = np.array([[1, 2], [3, 4]])
        with pytest.raises(NotImplementedError):
            sampler.transform(X)

    def test_get_parameter_schema(self):
        schema = SMOTESampler.get_parameter_schema()
        assert "sampling_strategy" in schema
        assert "k_neighbors" in schema
        assert "random_state" in schema

    def test_too_few_minority_samples_raises(self):
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
        y = np.array([0, 0, 0, 0, 1])  # Only 1 minority sample, k=5
        sampler = SMOTESampler(k_neighbors=5)
        with pytest.raises(ValueError):
            sampler.fit_resample(X, y)


# --------------------------------------------------------------------------
# Tests for ClassBalanceSampler.
# --------------------------------------------------------------------------

class TestClassBalanceSampler:
    """Tests for ClassBalanceSampler."""

    @pytest.fixture
    def imbalanced_data(self):
        np.random.seed(42)
        X = np.arange(100).reshape(-1, 1).astype(float)
        y = np.array([0] * 90 + [1] * 10)
        return X, y

    def test_init_defaults(self):
        balancer = ClassBalanceSampler()
        assert balancer.strategy == "oversample"
        assert balancer.target_ratio == 1.0
        assert balancer.random_state is None

    def test_oversample_strategy(self, imbalanced_data):
        X, y = imbalanced_data
        balancer = ClassBalanceSampler(strategy="oversample", random_state=42)
        balancer.fit(X)
        X_bal, y_bal = balancer.transform(X, y)
        counts = Counter(y_bal)
        assert counts[0] == counts[1]

    def test_undersample_strategy(self, imbalanced_data):
        X, y = imbalanced_data
        balancer = ClassBalanceSampler(strategy="undersample", random_state=42)
        balancer.fit(X)
        X_bal, y_bal = balancer.transform(X, y)
        counts = Counter(y_bal)
        assert counts[0] == counts[1]

    def test_both_strategy(self, imbalanced_data):
        X, y = imbalanced_data
        balancer = ClassBalanceSampler(strategy="both", random_state=42)
        balancer.fit(X)
        X_bal, y_bal = balancer.transform(X, y)
        counts = Counter(y_bal)
        assert counts[0] == counts[1]

    def test_invalid_strategy_raises(self):
        with pytest.raises(ValueError):
            ClassBalanceSampler(strategy="invalid")

    def test_invalid_target_ratio_raises(self):
        with pytest.raises(ValueError):
            ClassBalanceSampler(target_ratio=0.0)
        with pytest.raises(ValueError):
            ClassBalanceSampler(target_ratio=2.0)

    def test_transform_before_fit_raises(self):
        balancer = ClassBalanceSampler()
        X = np.array([[1], [2]])
        y = np.array([0, 1])
        with pytest.raises(RuntimeError):
            balancer.transform(X, y)

    def test_get_parameter_schema(self):
        schema = ClassBalanceSampler.get_parameter_schema()
        assert "strategy" in schema
        assert "target_ratio" in schema
        assert "random_state" in schema


class TestSMOTENCSampler:
    @pytest.fixture
    def data(self):
        X = np.array(
            [
                [20.0, 30.0, 0],
                [22.0, 34.0, 0],
                [21.0, 32.0, 1],
                [24.0, 40.0, 1],
                [23.0, 38.0, 0],
                [25.0, 42.0, 1],
                [28.0, 45.0, 0],
            ]
        )

        y = np.array([0, 0, 0, 1, 1, 1, 0])

        return X, y

    def test_categorical_values_remain_valid(self, data):
        X, y = data

        sampler = SMOTENCSampler(
            categorical_features=[2],
            k_neighbors=2,
            random_state=42,
        )

        X_resampled, y_resampled = sampler.fit_resample(X, y)

        original_categories = set(X[:, 2])
        generated_categories = set(X_resampled[len(X):, 2])

        assert generated_categories.issubset(original_categories)

    def test_numeric_values_are_generated(self, data):
        X, y = data

        sampler = SMOTENCSampler(
            categorical_features=[2],
            k_neighbors=2,
            random_state=42,
        )

        X_resampled, y_resampled = sampler.fit_resample(X, y)

        generated = X_resampled[len(X):]

        assert len(generated) > 0
        assert np.all(generated[:, 0] >= X[:, 0].min())
        assert np.all(generated[:, 0] <= X[:, 0].max())

    def test_reproducibility(self, data):
        X, y = data

        sampler1 = SMOTENCSampler(
            categorical_features=[2],
            k_neighbors=2,
            random_state=42,
        )

        sampler2 = SMOTENCSampler(
            categorical_features=[2],
            k_neighbors=2,
            random_state=42,
        )

        X1, y1 = sampler1.fit_resample(X, y)
        X2, y2 = sampler2.fit_resample(X, y)

        np.testing.assert_array_equal(X1, X2)
        np.testing.assert_array_equal(y1, y2)

    def test_invalid_categorical_feature_index(self, data):
        X, y = data

        sampler = SMOTENCSampler(
            categorical_features=[3],
            k_neighbors=2,
            random_state=42,
        )

        with pytest.raises(
            ValueError,
            match="categorical_features contains an invalid column index",
        ):
            sampler.fit(X, y)

    def test_negative_categorical_feature_index(self, data):
        X, y = data

        sampler = SMOTENCSampler(
            categorical_features=[-1],
            k_neighbors=2,
            random_state=42,
        )

        with pytest.raises(
            ValueError,
            match="categorical_features contains an invalid column index",
        ):
            sampler.fit(X, y)

    def test_requires_categorical_feature(self, data):
        X, y = data

        sampler = SMOTENCSampler(
            categorical_features=[],
            k_neighbors=2,
            random_state=42,
        )

        with pytest.raises(
            ValueError,
            match="SMOTENC requires at least one categorical feature",
        ):
            sampler.fit(X, y)

    def test_requires_numerical_feature(self, data):
        X, y = data

        sampler = SMOTENCSampler(
            categorical_features=[0, 1, 2],
            k_neighbors=2,
            random_state=42,
        )

        with pytest.raises(
            ValueError,
            match="SMOTENC requires at least one numerical feature",
        ):
            sampler.fit(X, y)

    def test_sampling_balances_minority_class(self, data):
        X, y = data

        sampler = SMOTENCSampler(
            categorical_features=[2],
            k_neighbors=2,
            random_state=42,
        )

        X_resampled, y_resampled = sampler.fit_resample(X, y)

        unique, counts = np.unique(y_resampled, return_counts=True)
        class_counts = dict(zip(unique, counts))

        assert class_counts[0] == class_counts[1]