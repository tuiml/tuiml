"""Tests for causal / uplift algorithms (meta-learners, uplift tree, metrics)."""

import numpy as np
import pytest

from tuiml.algorithms.causal import (
    SLearner,
    TLearner,
    XLearner,
    UpliftTreeClassifier,
    qini_curve,
    auuc,
    uplift_at_k,
)
from tuiml.algorithms.trees import DecisionTreeRegressor
from tuiml.registry import registry

N_SAMPLES = 2000


def _synthetic_data(n=N_SAMPLES, seed=0):
    """Synthetic dataset with a known heterogeneous treatment effect.

    The outcome is ``y = 1 + 3*x1 + treatment * tau + noise`` with
    ``tau = 2 * x0``, so the true individual treatment effect is a clean
    function of ``x0`` alone.
    """
    rng = np.random.RandomState(seed)
    X = rng.uniform(-1, 1, size=(n, 2))
    treatment = rng.randint(0, 2, size=n)
    tau = 2.0 * X[:, 0]
    y = 1.0 + 3.0 * X[:, 1] + treatment * tau + rng.normal(0, 0.1, size=n)
    return X, treatment, y, tau


X, treatment, y, tau = _synthetic_data()


# --------------------------------------------------------------------------- #
# Meta-learners
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("model_cls", [SLearner, TLearner, XLearner])
def test_meta_learner_uplift_correlates_with_true_tau(model_cls):
    """Each meta-learner recovers the heterogeneous treatment effect."""
    model = model_cls(DecisionTreeRegressor(max_depth=6, min_samples_leaf=20))
    model.fit(X, treatment, y)

    pred = model.predict_uplift(X)
    assert pred.shape == (N_SAMPLES,)
    assert np.isfinite(pred).all()

    corr = np.corrcoef(pred, tau)[0, 1]
    assert corr > 0.5, f"{model_cls.__name__} uplift correlation too low: {corr:.3f}"


def test_t_learner_group_models_differ():
    """The T-learner's two group models actually learned separate surfaces."""
    model = TLearner(DecisionTreeRegressor(max_depth=6, min_samples_leaf=20))
    model.fit(X, treatment, y)

    pred_0 = model.model_0_.predict(X)
    pred_1 = model.model_1_.predict(X)

    assert pred_0.shape == pred_1.shape == (N_SAMPLES,)
    # Control outcome is 1 + 3*x1; treated outcome adds 2*x0, so the two
    # models must produce meaningfully different predictions.
    assert not np.allclose(pred_0, pred_1)
    assert np.mean(np.abs(pred_0 - pred_1)) > 0.5


@pytest.mark.parametrize("model_cls", [SLearner, TLearner, XLearner])
def test_predict_before_fit_raises(model_cls):
    """Predicting before fitting raises a clear error."""
    model = model_cls()
    with pytest.raises(RuntimeError):
        model.predict_uplift(X)


# --------------------------------------------------------------------------- #
# Uplift tree
# --------------------------------------------------------------------------- #

def test_uplift_tree_runs_and_is_finite():
    """The uplift tree fits and returns finite uplifts with an inspectable tree."""
    model = UpliftTreeClassifier(max_depth=5, min_samples_leaf=50, random_state=42)
    model.fit(X, treatment, y)

    pred = model.predict_uplift(X)
    assert pred.shape == (N_SAMPLES,)
    assert np.isfinite(pred).all()

    # Tree structure is exposed for inspection.
    assert model.tree_ is not None
    assert model.tree_["type"] in ("leaf", "internal")
    assert model.n_nodes_ >= 1
    assert model.max_depth_ <= 5

    # A split directly on uplift should still track the true effect somewhat.
    corr = np.corrcoef(pred, tau)[0, 1]
    assert corr > 0.3, f"uplift tree correlation too low: {corr:.3f}"


# --------------------------------------------------------------------------- #
# Empty-group validation
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize(
    "model_cls", [SLearner, TLearner, XLearner, UpliftTreeClassifier]
)
@pytest.mark.parametrize("fill_value", [0, 1])
def test_empty_treatment_group_raises(model_cls, fill_value):
    """A treatment indicator missing one group raises a clear error."""
    bad_treatment = np.full(N_SAMPLES, fill_value)
    model = model_cls()
    with pytest.raises(ValueError, match="both groups"):
        model.fit(X, bad_treatment, y)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #

def test_auuc_true_beats_noise():
    """AUUC of the true uplift exceeds AUUC of random noise."""
    rng = np.random.RandomState(0)
    noise = rng.normal(0, 1, size=N_SAMPLES)
    assert auuc(tau, treatment, y) > auuc(noise, treatment, y)


def test_qini_curve_shape():
    """The Qini curve spans the population and ends at the total gain."""
    x, curve = qini_curve(tau, treatment, y)
    assert x.shape == curve.shape == (N_SAMPLES + 1,)
    assert x[0] == 0.0 and x[-1] == 1.0
    assert curve[0] == 0.0
    # The final value equals n_treated * (mean_treated - mean_control).
    n_t = int(treatment.sum())
    expected_end = n_t * (y[treatment == 1].mean() - y[treatment == 0].mean())
    assert np.isclose(curve[-1], expected_end, rtol=1e-9)


def test_uplift_at_k_matches_ate_on_perfect_ranking():
    """Top-k uplift recovers the true effect when ranking is perfect."""
    treatment_bal = np.tile([0, 1], 200)
    y_bal = treatment_bal * 2.0
    uplift = np.arange(400)
    # Top-100 are a balanced mix of treated (y=2) and control (y=0).
    assert np.isclose(uplift_at_k(uplift, treatment_bal, y_bal, k=100), 2.0)


# --------------------------------------------------------------------------- #
# Registration
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("name", ["SLearner", "TLearner", "XLearner", "UpliftTreeClassifier"])
def test_registered_in_hub(name):
    """Each uplift model is registered under its class name."""
    assert registry.get(name) is not None
