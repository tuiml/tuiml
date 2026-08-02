"""Missing-value handling in the shared tree engine.

Regression coverage for a fit that never returned. ``np.argsort`` sorts NaN
last and ``NaN != NaN`` is True, so a boundary at a NaN counted as a valid
split and the threshold came back NaN. Every ``x <= NaN`` comparison is False,
so the split put all rows on one side and the builder recursed on an identical
subproblem until the interpreter hit its recursion limit — surfacing as a
``RecursionError`` with a traceback pointing at impurity arithmetic rather than
at the cause.

These tests cover the trees built by ``_core`` (DecisionTree*, RandomForest*).
j48, lmt and m5p implement their own missing-value handling and are tested
separately.
"""

import numpy as np
import pytest

from tuiml.algorithms.trees import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
    RandomForestClassifier,
)
from tuiml.algorithms.trees._core.predict import predict_single_numpy
from tuiml.datasets import load_vote


@pytest.fixture
def nan_data():
    """A learnable binary problem with 30% of feature values missing."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 5))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    X = X.copy()
    X[rng.random(X.shape) < 0.30] = np.nan
    return X, y


class TestFitTerminates:
    """The tree must build rather than recurse forever."""

    def test_builtin_vote_dataset(self):
        """vote ships with 392 missing values and used to hang the builder."""
        data = load_vote()
        assert np.isnan(data.X).any(), "fixture no longer exercises the bug"

        model = DecisionTreeClassifier(random_state=42).fit(data.X, data.y)

        assert model._is_fitted
        assert len(model.predict(data.X)) == len(data.y)

    def test_forest_on_vote(self):
        data = load_vote()
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(data.X, data.y)
        assert len(model.predict(data.X)) == len(data.y)

    def test_regressor(self, nan_data):
        X, _ = nan_data
        y = np.nan_to_num(X[:, 0], nan=0.0)
        model = DecisionTreeRegressor(random_state=42).fit(X, y)
        assert np.isfinite(model.predict(X)).all()

    @pytest.mark.parametrize(
        "criterion", ["gini", "entropy", "log_loss", "gain_ratio"]
    )
    def test_every_classifier_criterion(self, nan_data, criterion):
        """gini/entropy route via C++ on clean data but fall back here."""
        X, y = nan_data
        model = DecisionTreeClassifier(criterion=criterion, random_state=42)
        model.fit(X, y)
        assert model._is_fitted

    @pytest.mark.parametrize(
        "criterion", ["squared_error", "friedman_mse", "absolute_error"]
    )
    def test_every_regressor_criterion(self, nan_data, criterion):
        X, _ = nan_data
        y = np.nan_to_num(X[:, 0], nan=0.0)
        DecisionTreeRegressor(criterion=criterion, random_state=42).fit(X, y)


class TestDegenerateInput:
    """Shapes of missingness that leave a split with nothing to work with."""

    def _fit_predict(self, X, y):
        model = DecisionTreeClassifier(random_state=42).fit(X, y)
        return model.predict(X)

    def test_column_entirely_missing(self, nan_data):
        X, y = nan_data
        X = X.copy()
        X[:, 2] = np.nan
        assert len(self._fit_predict(X, y)) == len(y)

    def test_every_value_missing(self, nan_data):
        """No feature is usable, so the tree collapses to a single leaf."""
        X, y = nan_data
        X = np.full_like(X, np.nan)

        predictions = self._fit_predict(X, y)

        assert len(np.unique(predictions)) == 1, "expected a majority-class leaf"

    def test_row_entirely_missing(self, nan_data):
        X, y = nan_data
        X = X.copy()
        X[0, :] = np.nan
        assert len(self._fit_predict(X, y)) == len(y)

    def test_infinities(self, nan_data):
        """inf is orderable, so it must not be treated as missing."""
        X, y = nan_data
        X = np.nan_to_num(X, nan=0.0)
        X[:, 1] = np.inf
        assert len(self._fit_predict(X, y)) == len(y)


class TestThresholds:
    """The root cause: a threshold must never be NaN."""

    def test_no_nan_threshold_anywhere(self, nan_data):
        X, y = nan_data
        model = DecisionTreeClassifier(random_state=42).fit(X, y)

        def walk(node):
            if node.is_leaf:
                return
            assert not np.isnan(node.threshold), "NaN threshold splits nothing"
            walk(node.left)
            walk(node.right)

        walk(model.tree_)


class TestRoutingConsistency:
    """Training, batch prediction and single-sample prediction must agree."""

    def test_batch_matches_recursive_walk(self):
        data = load_vote()
        model = DecisionTreeClassifier(random_state=42).fit(data.X, data.y)

        batch = model.predict(data.X)
        probabilities = np.array(
            [predict_single_numpy(model.tree_, row) for row in data.X]
        )
        walked = model.classes_[np.argmax(probabilities, axis=1)]

        assert (batch == walked).all()

    def test_missing_rows_go_right(self, nan_data):
        """The direction itself, pinned: NaN <= t is False, so right."""
        X, y = nan_data
        model = DecisionTreeClassifier(random_state=42).fit(X, y)

        root = model.tree_
        if root.is_leaf:
            pytest.skip("degenerate tree")

        row = np.zeros(X.shape[1])
        row[root.feature_index] = np.nan

        # Walking from the root with a missing split feature lands in the
        # right subtree, matching how _partition assigned it during fitting.
        expected = predict_single_numpy(root.right, row)
        assert np.array_equal(predict_single_numpy(root, row), expected)


class TestCleanDataUnchanged:
    """Missing-value support must not perturb complete data."""

    def test_matches_across_criteria(self):
        rng = np.random.default_rng(1)
        X = rng.normal(size=(150, 4))
        y = (X[:, 0] > 0).astype(int)

        for criterion in ("gini", "entropy", "gain_ratio"):
            first = DecisionTreeClassifier(criterion=criterion, random_state=42)
            second = DecisionTreeClassifier(criterion=criterion, random_state=42)
            assert (
                first.fit(X, y).predict(X) == second.fit(X, y).predict(X)
            ).all()

    def test_learns_a_clean_problem_perfectly(self):
        rng = np.random.default_rng(2)
        X = rng.normal(size=(200, 3))
        y = (X[:, 0] > 0).astype(int)

        model = DecisionTreeClassifier(random_state=42).fit(X, y)
        assert (model.predict(X) == y).mean() == 1.0
