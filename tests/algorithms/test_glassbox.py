"""Tests for the interpretable-by-design glassbox model family."""

import numpy as np
import pytest

from tuiml.algorithms.glassbox import (
    ExplainableBoostingClassifier,
    ExplainableBoostingRegressor,
    RuleFitClassifier,
    RuleFitRegressor,
)
from tuiml.algorithms.trees import DecisionTreeRegressor
from tuiml.evaluation.metrics import accuracy_score, r2_score
from tuiml.registry import registry


# ---------------------------------------------------------------------------
# Explainable Boosting Machine
# ---------------------------------------------------------------------------

def test_ebm_regressor_recovers_additive_structure():
    """EBM must beat a single tree on additive binned data and reconstruct
    its predictions exactly from intercept + shape functions."""
    rng = np.random.RandomState(0)
    n = 800
    X = rng.randn(n, 4)
    # y depends additively (and nonlinearly) on three features; one is noise.
    y = (
        np.sin(2.0 * X[:, 0])
        + 0.5 * X[:, 1] ** 2
        - 0.3 * X[:, 2]
        + rng.randn(n) * 0.1
    )

    ebm = ExplainableBoostingRegressor(
        n_bins=32, max_rounds=200, learning_rate=0.05,
    ).fit(X, y)
    tree = DecisionTreeRegressor(max_depth=3, random_state=0).fit(X, y)

    ebm_r2 = r2_score(y, ebm.predict(X))
    tree_r2 = r2_score(y, tree.predict(X))

    # (a) recover the additive structure better than a single tree
    assert ebm_r2 > 0.8, f"EBM R^2 too low: {ebm_r2:.3f}"
    assert ebm_r2 > tree_r2 + 0.05, (
        f"EBM should beat a depth-3 tree: EBM={ebm_r2:.3f} vs tree={tree_r2:.3f}"
    )

    # (b) predictions reconstruct from intercept_ + sum of shape functions
    pred = ebm.predict(X)
    recon = ebm.intercept_[0] + ebm.explain(X).sum(axis=1)
    assert np.allclose(pred, recon, atol=1e-8)

    # The irrelevant feature (index 3) should carry the least importance.
    imp = ebm.feature_importance_
    assert imp.shape == (4,)
    assert imp[3] < imp[0] and imp[3] < imp[1] and imp[3] < imp[2]

    # Shape functions are readable: one (edges, scores) pair per feature.
    shapes = ebm.get_shape_functions()
    assert len(shapes) == 4
    for edges, scores in shapes:
        assert edges.ndim == 1
        assert scores.shape == (edges.size - 1, 1)


def test_ebm_classifier_binary_and_proba():
    rng = np.random.RandomState(1)
    X = rng.randn(200, 2)
    y = (X[:, 0] + X[:, 1] > 0.0).astype(int)

    clf = ExplainableBoostingClassifier(
        n_bins=16, max_rounds=100, learning_rate=0.1,
    ).fit(X, y)

    pred = clf.predict(X)
    assert pred.shape == (200,)
    assert set(np.unique(pred)).issubset({0, 1})
    assert accuracy_score(y, pred) > 0.9

    proba = clf.predict_proba(X)
    assert proba.shape == (200, 2)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-9)

    # decision_function reconstructs from explain for classification too.
    score = clf.decision_function(X)
    recon = clf.intercept_[0] + clf.explain(X).sum(axis=1)
    assert np.allclose(score, recon, atol=1e-8)


def test_ebm_classifier_multiclass():
    rng = np.random.RandomState(2)
    X = rng.randn(300, 2)
    y = np.digitize(X[:, 0], [-0.5, 0.5])  # three classes

    clf = ExplainableBoostingClassifier(
        n_bins=16, max_rounds=100, learning_rate=0.1,
    ).fit(X, y)

    proba = clf.predict_proba(X)
    assert proba.shape == (300, 3)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-9)
    assert accuracy_score(y, clf.predict(X)) > 0.85


# ---------------------------------------------------------------------------
# RuleFit
# ---------------------------------------------------------------------------

def test_rulefit_classifier_readable_rules_and_reproducibility():
    rng = np.random.RandomState(42)
    X = rng.randn(400, 3)
    # A conjunction the rule extractor should capture.
    y = ((X[:, 0] > 0.0) & (X[:, 1] > 0.0)).astype(int)

    clf = RuleFitClassifier(
        n_estimators=60, tree_size=2, max_rules=30, random_state=0, alpha=0.1,
    ).fit(X, y)

    rules = clf.get_rules()
    assert 0 < len(rules) <= 50, f"expected a small readable rule set, got {len(rules)}"
    for rule_text, coef in rules:
        assert isinstance(rule_text, str)
        assert ("feature_" in rule_text) and (">" in rule_text or "<=" in rule_text)
        assert isinstance(coef, float)

    assert accuracy_score(y, clf.predict(X)) > 0.9

    proba = clf.predict_proba(X)
    assert proba.shape == (400, 2)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-9)

    # Reproducibility: same seed -> identical predictions.
    clf2 = RuleFitClassifier(
        n_estimators=60, tree_size=2, max_rules=30, random_state=0, alpha=0.1,
    ).fit(X, y)
    assert np.array_equal(clf.predict(X), clf2.predict(X))


def test_rulefit_regressor_rules_and_reproducibility():
    rng = np.random.RandomState(7)
    X = rng.randn(400, 3)
    y = 3.0 * X[:, 0] + (X[:, 1] > 0.0) * 5.0 + rng.randn(400) * 0.1

    reg = RuleFitRegressor(
        n_estimators=40, tree_size=2, max_rules=30, random_state=0, alpha=0.1,
    ).fit(X, y)

    rules = reg.get_rules()
    assert 0 < len(rules) <= 50, f"expected a small readable rule set, got {len(rules)}"
    for rule_text, coef in rules:
        assert isinstance(rule_text, str) and "feature_" in rule_text

    assert r2_score(y, reg.predict(X)) > 0.8

    reg2 = RuleFitRegressor(
        n_estimators=40, tree_size=2, max_rules=30, random_state=0, alpha=0.1,
    ).fit(X, y)
    assert np.allclose(reg.predict(X), reg2.predict(X))


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name", [
    "ExplainableBoostingClassifier",
    "ExplainableBoostingRegressor",
    "RuleFitClassifier",
    "RuleFitRegressor",
])
def test_registered_in_hub(name):
    assert registry.get(name) is not None
