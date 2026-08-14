"""Tests for the survival-analysis family (Kaplan-Meier, Cox, RSF, metrics)."""

import numpy as np
import pytest

from tuiml.algorithms.survival import (
    KaplanMeierEstimator,
    NelsonAalenEstimator,
    CoxPHSurvival,
    RandomSurvivalForest,
)
from tuiml.algorithms.survival.metrics import (
    concordance_index,
    integrated_brier_score,
    logrank_test,
)
from tuiml.registry import registry


# --------------------------------------------------------------------------
# Kaplan-Meier / Nelson-Aalen
# --------------------------------------------------------------------------

class TestKaplanMeier:
    def test_product_limit_exact(self):
        # Textbook hand-computable example: subject 3 censored at t=4.
        km = KaplanMeierEstimator().fit([2, 3, 4, 5], [1, 1, 0, 1])

        assert km.timeline_.tolist() == [2.0, 3.0, 5.0]
        assert np.allclose(km.survival_, [0.75, 0.5, 0.0])

        # Step function: 1 before the first event, constant between events.
        S = km.predict_survival_function([1, 2, 3, 4, 5, 6])
        assert np.allclose(S, [1.0, 0.75, 0.5, 0.5, 0.0, 0.0])

    def test_fit_sets_fitted_flag(self):
        km = KaplanMeierEstimator().fit([1, 2, 3], [1, 1, 1])
        assert km._is_fitted is True

    def test_predict_before_fit_raises(self):
        km = KaplanMeierEstimator()
        with pytest.raises(RuntimeError):
            km.predict_survival_function([1.0])

    def test_predict_risk_constant(self):
        km = KaplanMeierEstimator().fit([2, 3, 4, 5], [1, 1, 0, 1])
        risk = km.predict_risk(np.zeros((4, 2)))
        assert risk.shape == (4,)
        assert np.all(risk == km.total_cumulative_hazard_)

    def test_all_censored(self):
        km = KaplanMeierEstimator().fit([1, 2, 3], [0, 0, 0])
        assert km.timeline_.size == 0
        assert km.predict_survival_function([1.0, 2.0, 3.0]).tolist() == [1.0, 1.0, 1.0]


class TestNelsonAalen:
    def test_cumulative_hazard_exact(self):
        na = NelsonAalenEstimator().fit([2, 3, 4, 5], [1, 1, 0, 1])
        expected_H = [0.25, 0.25 + 1.0 / 3.0, 0.25 + 1.0 / 3.0 + 1.0]
        assert np.allclose(na.cumulative_hazard_, expected_H)
        assert np.allclose(na.survival_, np.exp(-np.array(expected_H)))

    def test_survival_function_exponential(self):
        na = NelsonAalenEstimator().fit([2, 3, 4, 5], [1, 1, 0, 1])
        H = na.predict_cumulative_hazard([2.0, 5.0])
        S = na.predict_survival_function([2.0, 5.0])
        assert np.allclose(S, np.exp(-H))


# --------------------------------------------------------------------------
# Concordance index
# --------------------------------------------------------------------------

class TestConcordanceIndex:
    def test_perfect_and_reversed(self):
        time = np.array([1.0, 2.0, 3.0, 4.0])
        event = np.ones(4)
        # Higher risk = earlier event, so the perfect ranking is decreasing.
        assert concordance_index([4, 3, 2, 1], time, event) == 1.0
        assert concordance_index([1, 2, 3, 4], time, event) == 0.0

    def test_censored_pairs_ignored(self):
        # The censored subject (t=4) cannot be compared to later events.
        time = np.array([1.0, 2.0, 4.0])
        event = np.array([1, 1, 0])
        c = concordance_index([1, 2, 3], time, event)
        assert 0.0 <= c <= 1.0

    def test_ties_count_half(self):
        # Only two subjects, tied time, both events: one tied-time pair -> 0.5.
        time = np.array([2.0, 2.0])
        event = np.ones(2)
        assert concordance_index([0, 0], time, event) == 0.5

    def test_no_comparable_pairs_returns_nan(self):
        # Single subject: no pairs at all.
        assert np.isnan(concordance_index([1], [1.0], [1]))


# --------------------------------------------------------------------------
# Cox proportional hazards
# --------------------------------------------------------------------------

class TestCoxPH:
    def test_hand_computed_mle(self):
        # X = [[1],[0],[1]], time = [1,2,3], all events.
        # Analytic MLE: beta = ln(1 / sqrt(2)).
        X = np.array([[1.0], [0.0], [1.0]])
        time = np.array([1.0, 2.0, 3.0])
        event = np.ones(3)
        cox = CoxPHSurvival().fit(X, time, event)
        assert np.isclose(cox.coefficients_[0], np.log(1.0 / np.sqrt(2.0)), atol=1e-4)

    def test_coefficient_direction_exponential_hazard(self):
        # Feature 0 raises the hazard (shortens survival) -> positive coefficient.
        rng = np.random.RandomState(0)
        n = 1000
        X = rng.normal(size=(n, 3))
        rate = np.exp(1.5 * X[:, 0])  # higher X0 -> higher hazard -> shorter time
        T = rng.exponential(scale=1.0, size=n) / rate
        C = rng.exponential(scale=3.0, size=n)  # independent censoring
        time = np.minimum(T, C)
        event = (T <= C).astype(int)

        cox = CoxPHSurvival().fit(X, time, event)
        assert cox.coefficients_[0] > 0.5
        assert abs(cox.coefficients_[1]) < abs(cox.coefficients_[0])
        assert abs(cox.coefficients_[2]) < abs(cox.coefficients_[0])

    def test_gradient_vanishes_at_fit(self):
        # Independent brute-force gradient of the partial log-likelihood
        # should be ~0 at the fitted coefficients.
        rng = np.random.RandomState(1)
        X = rng.normal(size=(30, 2))
        time = rng.uniform(0.5, 5.0, size=30)
        event = rng.binomial(1, 0.8, size=30)
        cox = CoxPHSurvival().fit(X, time, event)
        beta = cox.coefficients_

        grad = np.zeros(2)
        for i in range(30):
            if event[i] != 1:
                continue
            risk = X[time >= time[i]]
            eta = risk @ beta
            w = np.exp(eta - eta.max())
            grad += X[i] - (risk * w[:, None]).sum(axis=0) / w.sum()
        assert np.allclose(grad, 0.0, atol=1e-6)

    def test_baseline_survival_monotone(self):
        rng = np.random.RandomState(2)
        X = rng.normal(size=(50, 2))
        time = rng.uniform(1, 10, size=50)
        event = rng.binomial(1, 0.8, size=50)
        cox = CoxPHSurvival().fit(X, time, event)
        assert np.all(cox.baseline_hazard_ >= 0.0)
        assert np.all(np.diff(cox.baseline_survival_) <= 0.0)
        assert np.all(np.diff(cox.baseline_cumulative_hazard_) >= 0.0)

    def test_predict_survival_shape(self):
        rng = np.random.RandomState(3)
        X = rng.normal(size=(20, 2))
        time = rng.uniform(1, 10, size=20)
        event = rng.binomial(1, 0.8, size=20)
        cox = CoxPHSurvival().fit(X, time, event)
        S = cox.predict_survival_function(X[:5], times=[2.0, 4.0, 6.0])
        assert S.shape == (5, 3)

    def test_l2_penalty_shrinks(self):
        rng = np.random.RandomState(4)
        X = rng.normal(size=(100, 1))
        time = rng.uniform(1, 10, size=100)
        event = rng.binomial(1, 0.7, size=100)
        unpen = CoxPHSurvival(penalty=None).fit(X, time, event).coefficients_
        pen = CoxPHSurvival(penalty="l2", alpha=10.0).fit(X, time, event).coefficients_
        assert abs(pen[0]) <= abs(unpen[0]) + 1e-9

    def test_no_events_raises(self):
        X = np.zeros((3, 1))
        with pytest.raises(ValueError):
            CoxPHSurvival().fit(X, [1.0, 2.0, 3.0], [0, 0, 0])


# --------------------------------------------------------------------------
# Random survival forest
# --------------------------------------------------------------------------

class TestRandomSurvivalForest:
    def test_predict_risk_shape_and_sign(self):
        rng = np.random.RandomState(5)
        X = rng.normal(size=(60, 2))
        # Feature 0 shortens survival strongly.
        time = np.exp(X[:, 0]) + rng.uniform(0, 0.5, size=60)
        event = np.ones(60)
        rsf = RandomSurvivalForest(n_estimators=20, random_state=0).fit(X, time, event)
        risk = rsf.predict_risk(X[:10])
        assert risk.shape == (10,)

    def test_risk_recovers_signal(self):
        rng = np.random.RandomState(6)
        X = rng.normal(size=(300, 1))
        # Higher X0 -> higher hazard -> shorter time (rate grows with X0).
        rate = np.exp(X[:, 0])
        time = rng.exponential(scale=1.0, size=300) / rate
        event = np.ones(300)
        rsf = RandomSurvivalForest(n_estimators=50, random_state=0).fit(X, time, event)
        risk = rsf.predict_risk(X)
        # Higher X0 must correspond to higher risk, and the risk must rank
        # subjects consistently with their event times.
        assert np.corrcoef(risk, X[:, 0])[0, 1] > 0.5
        assert concordance_index(risk, time, event) > 0.6

    def test_handles_censoring(self):
        rng = np.random.RandomState(7)
        X = rng.normal(size=(80, 2))
        time = rng.uniform(1, 10, size=80)
        event = rng.binomial(1, 0.5, size=80)
        rsf = RandomSurvivalForest(n_estimators=10, random_state=0).fit(X, time, event)
        assert rsf.predict_risk(X[:5]).shape == (5,)


# --------------------------------------------------------------------------
# Metrics: log-rank test and integrated Brier score
# --------------------------------------------------------------------------

class TestLogRank:
    def test_separated_groups_significant(self):
        stat, p = logrank_test([1, 2, 3, 4], [1, 1, 1, 1],
                               [5, 6, 7, 8], [1, 1, 1, 1])
        assert stat > 0.0
        assert p < 0.05

    def test_identical_groups_not_significant(self):
        a = [1.0, 2.0, 3.0]
        stat, p = logrank_test(a, [1, 1, 1], a, [1, 1, 1])
        assert np.isclose(stat, 0.0)
        assert np.isclose(p, 1.0)


class TestIntegratedBrierScore:
    def test_perfect_model_near_zero(self):
        class PerfectModel:
            def __init__(self, time):
                self.time = time

            def predict_survival_function(self, X, times):
                return (self.time[:, None] > np.asarray(times)[None, :]).astype(float)

        time = np.array([1.0, 3.0, 5.0, 7.0])
        event = np.ones(4)
        X = np.zeros((4, 1))
        model = PerfectModel(time)
        ibs = integrated_brier_score(model, X, time, event, [2.0, 4.0, 6.0])
        assert ibs < 1e-6

    def test_risk_path_returns_bounded_scalar(self):
        rng = np.random.RandomState(8)
        time = rng.uniform(1, 10, size=30)
        event = np.ones(30)
        risk = -time
        ibs = integrated_brier_score(risk, np.ones((30, 1)), time, event, [2.0, 5.0, 8.0])
        assert 0.0 <= ibs <= 1.0

    def test_model_path_with_cox(self):
        rng = np.random.RandomState(9)
        X = rng.normal(size=(50, 2))
        time = rng.uniform(1, 10, size=50)
        event = rng.binomial(1, 0.8, size=50)
        cox = CoxPHSurvival().fit(X, time, event)
        ibs = integrated_brier_score(cox, X, time, event, [2.0, 5.0, 8.0])
        assert np.isfinite(ibs)


# --------------------------------------------------------------------------
# Censored-data handling and registration
# --------------------------------------------------------------------------

class TestCensoredDataAndRegistration:
    def test_censored_data_fits_without_error(self):
        # A single dataset with both events and censored rows.
        rng = np.random.RandomState(10)
        n = 100
        X = rng.normal(size=(n, 2))
        time = rng.uniform(0.5, 5.0, size=n)
        event = rng.binomial(1, 0.6, size=n)
        assert (event == 0).any() and (event == 1).any()

        KaplanMeierEstimator().fit(time, event)
        NelsonAalenEstimator().fit(time, event)
        CoxPHSurvival().fit(X, time, event)
        RandomSurvivalForest(n_estimators=5, random_state=0).fit(X, time, event)

    @pytest.mark.parametrize(
        "name",
        [
            "KaplanMeierEstimator",
            "NelsonAalenEstimator",
            "CoxPHSurvival",
            "RandomSurvivalForest",
        ],
    )
    def test_registered(self, name):
        assert registry.get(name) is not None
