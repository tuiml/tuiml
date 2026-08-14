"""Tests for NGBoost (natural gradient boosting for probabilistic prediction).

The interesting tests here are the numerical ones: NGBoost is only worth
anything if its natural gradient really is the metric-preconditioned gradient
and if the predictive distribution it produces is calibrated. Both are checked
against ground truth rather than against the implementation itself.
"""

import pickle

import numpy as np
import pytest

from tuiml.algorithms.gradient_boosting import NGBoostClassifier, NGBoostRegressor
from tuiml.algorithms.gradient_boosting.ngboost import (
    _CategoricalDist,
    _ExponentialDist,
    _LogNormalDist,
    _NormalDist,
    _norm_cdf,
    _norm_ppf,
    _safe_std,
)
from tuiml.registry import registry
from tests.contract._data import KNOWN_CAPABILITIES


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _fd_grad(dist, params, y, h=1e-6):
    """Central-difference gradient of the score, one column per parameter."""
    params = np.asarray(params, dtype=np.float64)
    out = np.empty_like(params)
    for k in range(params.shape[1]):
        up = params.copy()
        dn = params.copy()
        up[:, k] += h
        dn[:, k] -= h
        out[:, k] = (dist.score(up, y) - dist.score(dn, y)) / (2.0 * h)
    return out


def _heteroscedastic(n, rng):
    """Draw from a known heteroscedastic Normal: noise grows with |x_1|."""
    X = rng.uniform(-2.0, 2.0, size=(n, 2))
    sigma = 0.3 + 0.6 * np.abs(X[:, 1])
    y = X[:, 0] + rng.normal(0.0, sigma)
    return X, y, sigma


# --------------------------------------------------------------------------
# Special functions
# --------------------------------------------------------------------------

class TestSpecialFunctions:
    def test_norm_cdf_known_values(self):
        z = np.array([-1.959963984540054, 0.0, 1.0, 1.959963984540054])
        expected = [0.025, 0.5, 0.8413447460685429, 0.975]
        assert np.allclose(_norm_cdf(z), expected, atol=1e-12)

    def test_norm_ppf_inverts_cdf(self):
        for p in (0.001, 0.01, 0.025, 0.25, 0.5, 0.75, 0.95, 0.975, 0.999):
            z = _norm_ppf(p)
            assert abs(float(_norm_cdf(np.array(z))) - p) < 1e-12

    def test_norm_ppf_rejects_out_of_range(self):
        with pytest.raises(ValueError):
            _norm_ppf(0.0)
        with pytest.raises(ValueError):
            _norm_ppf(1.0)

    def test_safe_std_survives_huge_offset(self):
        # E[x^2] - mean^2 would cancel catastrophically here; centring first
        # keeps every digit.
        y = 1e8 + np.array([-1.0, 0.0, 1.0])
        assert abs(_safe_std(y) - np.sqrt(2.0 / 3.0)) < 1e-9

    def test_safe_std_floors_at_one_for_constant_input(self):
        assert _safe_std(np.full(5, 3.0)) == 1.0


# --------------------------------------------------------------------------
# Natural gradient / Fisher information
# --------------------------------------------------------------------------

class TestNormalLogScore:
    def _params_and_y(self):
        rng = np.random.default_rng(0)
        params = np.column_stack([
            rng.normal(0.0, 2.0, size=40),
            rng.uniform(-1.5, 1.5, size=40),
        ])
        y = rng.normal(0.0, 3.0, size=40)
        return params, y

    def test_score_matches_gaussian_nll(self):
        dist = _NormalDist(scoring="log")
        params, y = self._params_and_y()
        mu, log_sigma = params[:, 0], params[:, 1]
        sigma = np.exp(log_sigma)
        expected = (0.5 * ((y - mu) / sigma) ** 2 + np.log(sigma)
                    + 0.5 * np.log(2.0 * np.pi))
        assert np.allclose(dist.score(params, y), expected, atol=1e-12)

    def test_analytic_gradient_matches_finite_differences(self):
        dist = _NormalDist(scoring="log")
        params, y = self._params_and_y()
        assert np.allclose(dist.grad(params, y), _fd_grad(dist, params, y),
                           atol=1e-6, rtol=1e-6)

    def test_fisher_information_matches_monte_carlo(self):
        # I(theta) = E_{y ~ P_theta}[ grad grad^T ]. Estimate it by sampling.
        dist = _NormalDist(scoring="log")
        rng = np.random.default_rng(7)
        for mu, log_sigma in [(0.0, 0.0), (2.5, -0.7), (-1.0, 1.3)]:
            sigma = np.exp(log_sigma)
            n = 400_000
            y = rng.normal(mu, sigma, size=n)
            p = np.tile([mu, log_sigma], (n, 1))
            g = dist.grad(p, y)
            mc = g.T @ g / n
            analytic = dist.metric(p[:1])[0]
            assert np.allclose(mc, analytic, atol=0.05 * max(1.0, analytic.max()))

    def test_natural_gradient_equals_inverse_metric_times_fd_gradient(self):
        """The headline check: nat grad == inv(Fisher) @ (FD gradient)."""
        dist = _NormalDist(scoring="log")
        params, y = self._params_and_y()
        fd = _fd_grad(dist, params, y)
        metric = dist.metric(params)
        expected = np.linalg.solve(metric, fd[:, :, None])[:, :, 0]
        assert np.allclose(dist.natural_grad(params, y), expected,
                           atol=1e-6, rtol=1e-6)

    def test_natural_gradient_closed_form(self):
        # For (mu, log sigma) the natural gradient collapses to
        # (mu - y, (1 - z^2) / 2), which is free of sigma in its first slot.
        dist = _NormalDist(scoring="log")
        params, y = self._params_and_y()
        mu, sigma = params[:, 0], np.exp(params[:, 1])
        z = (y - mu) / sigma
        nat = dist.natural_grad(params, y)
        assert np.allclose(nat[:, 0], mu - y, atol=1e-12)
        assert np.allclose(nat[:, 1], 0.5 * (1.0 - z * z), atol=1e-12)

    def test_natural_gradient_differs_from_ordinary(self):
        dist = _NormalDist(scoring="log")
        params, y = self._params_and_y()
        assert not np.allclose(dist.grad(params, y), dist.natural_grad(params, y))


class TestNormalCRPS:
    def _params_and_y(self):
        rng = np.random.default_rng(1)
        params = np.column_stack([
            rng.normal(0.0, 1.0, size=12),
            rng.uniform(-0.8, 0.8, size=12),
        ])
        y = rng.normal(0.0, 1.5, size=12)
        return params, y

    def test_crps_matches_numerical_integration(self):
        """CRPS = integral of (F(t) - 1{t >= y})^2 dt, done on a fine grid."""
        dist = _NormalDist(scoring="crps")
        params, y = self._params_and_y()
        analytic = dist.score(params, y)

        numeric = np.empty_like(analytic)
        for i in range(len(y)):
            mu, sigma = params[i, 0], np.exp(params[i, 1])
            lo = min(mu - 12.0 * sigma, y[i] - 1.0)
            hi = max(mu + 12.0 * sigma, y[i] + 1.0)
            # Split the grid at y: the indicator is discontinuous there, and
            # straddling it costs the quadrature several digits.
            left = np.linspace(lo, y[i], 200_001)
            right = np.linspace(y[i], hi, 200_001)
            numeric[i] = (
                np.trapezoid(_norm_cdf((left - mu) / sigma) ** 2, left)
                + np.trapezoid((_norm_cdf((right - mu) / sigma) - 1.0) ** 2, right)
            )
        assert np.allclose(analytic, numeric, atol=1e-7, rtol=1e-7)

    def test_crps_gradient_matches_finite_differences(self):
        dist = _NormalDist(scoring="crps")
        params, y = self._params_and_y()
        assert np.allclose(dist.grad(params, y), _fd_grad(dist, params, y),
                           atol=1e-6, rtol=1e-6)

    def test_crps_natural_gradient_matches_metric_solve(self):
        dist = _NormalDist(scoring="crps")
        params, y = self._params_and_y()
        fd = _fd_grad(dist, params, y)
        expected = np.linalg.solve(dist.metric(params), fd[:, :, None])[:, :, 0]
        assert np.allclose(dist.natural_grad(params, y), expected,
                           atol=1e-6, rtol=1e-6)

    def test_crps_is_minimised_at_the_truth(self):
        # A proper scoring rule is minimised in expectation by the true
        # distribution; check against neighbours on a grid.
        dist = _NormalDist(scoring="crps")
        rng = np.random.default_rng(4)
        y = rng.normal(1.0, 2.0, size=200_000)
        best = np.mean(dist.score(np.tile([1.0, np.log(2.0)], (len(y), 1)), y))
        for mu, s in [(1.3, np.log(2.0)), (1.0, np.log(2.4)), (0.7, np.log(1.7))]:
            worse = np.mean(dist.score(np.tile([mu, s], (len(y), 1)), y))
            assert worse > best


class TestOtherDistributions:
    def test_exponential_gradient_and_metric(self):
        dist = _ExponentialDist()
        rng = np.random.default_rng(2)
        params = rng.uniform(-1.0, 1.0, size=(20, 1))
        y = rng.exponential(1.0, size=20)
        assert np.allclose(dist.grad(params, y), _fd_grad(dist, params, y),
                           atol=1e-6, rtol=1e-6)
        # Fisher information of log-scale for the exponential is exactly 1.
        assert np.allclose(dist.metric(params), 1.0)
        assert np.allclose(dist.natural_grad(params, y), dist.grad(params, y))

    def test_exponential_rejects_crps(self):
        with pytest.raises(ValueError):
            _ExponentialDist(scoring="crps")

    def test_lognormal_score_is_normal_on_logs(self):
        dist = _LogNormalDist()
        rng = np.random.default_rng(5)
        params = np.column_stack([rng.normal(size=15), rng.uniform(-1, 1, size=15)])
        y = np.exp(rng.normal(size=15))
        assert np.allclose(dist.score(params, dist.transform_y(y)),
                           _NormalDist().score(params, np.log(y)))
        assert np.allclose(dist.mean(params),
                           np.exp(params[:, 0] + 0.5 * np.exp(params[:, 1]) ** 2))

    def test_categorical_gradient_and_metric(self):
        dist = _CategoricalDist(4)
        rng = np.random.default_rng(6)
        params = rng.normal(size=(25, 3))
        y = rng.integers(0, 4, size=25)
        assert np.allclose(dist.grad(params, y), _fd_grad(dist, params, y),
                           atol=1e-6, rtol=1e-6)
        fd = _fd_grad(dist, params, y)
        expected = np.linalg.solve(dist.metric(params), fd[:, :, None])[:, :, 0]
        assert np.allclose(dist.natural_grad(params, y), expected,
                           atol=1e-6, rtol=1e-6)

    def test_categorical_binary_reduces_to_bernoulli(self):
        dist = _CategoricalDist(2)
        eta = np.array([[-1.0], [0.0], [2.0]])
        y = np.array([0, 1, 1])
        p = 1.0 / (1.0 + np.exp(-eta[:, 0]))
        assert np.allclose(dist.proba(eta)[:, 1], p)
        assert np.allclose(dist.grad(eta, y)[:, 0], p - y)
        assert np.allclose(dist.natural_grad(eta, y)[:, 0], (p - y) / (p * (1 - p)))


# --------------------------------------------------------------------------
# Regressor
# --------------------------------------------------------------------------

class TestNGBoostRegressor:
    def test_fit_predict_shapes(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(80, 3))
        y = X[:, 0] * 2.0 + rng.normal(0, 0.3, size=80)
        model = NGBoostRegressor(n_estimators=30, random_state=0).fit(X, y)
        assert model._is_fitted
        assert model.predict(X).shape == (80,)
        assert model.predict_interval(X).shape == (80, 2)
        assert model.n_features_in_ == 3
        assert 0 < model.n_estimators_ <= 30
        assert len(model.estimators_) == len(model.scalings_) == model.n_estimators_
        assert all(len(stage) == 2 for stage in model.estimators_)

    def test_training_score_decreases_monotonically(self):
        rng = np.random.default_rng(1)
        X = rng.normal(size=(150, 2))
        y = X[:, 0] + rng.normal(0, 0.5, size=150)
        model = NGBoostRegressor(n_estimators=40, random_state=0).fit(X, y)
        scores = np.asarray(model.train_score_)
        # The line search only accepts a strictly improving stage.
        assert np.all(np.diff(scores) < 0)
        assert scores[-1] < scores[0]

    def test_predict_dist_keys_and_positivity(self):
        rng = np.random.default_rng(2)
        X = rng.normal(size=(60, 2))
        y = rng.normal(size=60)
        params = NGBoostRegressor(n_estimators=20, random_state=0).fit(X, y).predict_dist(X)
        assert sorted(params) == ["loc", "scale"]
        assert np.all(params["scale"] > 0)
        assert np.all(np.isfinite(params["loc"]))

    def test_intervals_are_ordered_and_nest(self):
        rng = np.random.default_rng(3)
        X = rng.normal(size=(60, 2))
        y = rng.normal(size=60)
        model = NGBoostRegressor(n_estimators=20, random_state=0).fit(X, y)
        wide = model.predict_interval(X, alpha=0.01)
        narrow = model.predict_interval(X, alpha=0.5)
        assert np.all(wide[:, 0] < wide[:, 1])
        assert np.all(wide[:, 0] < narrow[:, 0])
        assert np.all(wide[:, 1] > narrow[:, 1])

    def test_calibration_on_known_heteroscedastic_normal(self):
        """NGBoost's selling point: sigma tracks the truth and intervals cover."""
        rng = np.random.default_rng(3)
        X_tr, y_tr, _ = _heteroscedastic(3000, rng)
        X_te, y_te, sigma_te = _heteroscedastic(4000, rng)

        model = NGBoostRegressor(
            n_estimators=100, learning_rate=0.05, min_samples_leaf=20,
            random_state=0,
        ).fit(X_tr, y_tr)

        params = model.predict_dist(X_te)
        corr = np.corrcoef(params["scale"], sigma_te)[0, 1]
        assert corr > 0.7, f"sigma does not track the truth (corr={corr:.3f})"
        # And it is on the right scale, not merely correlated with it.
        ratio = float(np.mean(params["scale"] / sigma_te))
        assert 0.85 < ratio < 1.15, f"sigma is off by a factor of {ratio:.3f}"

        lower, upper = model.predict_interval(X_te, alpha=0.10).T
        coverage = float(np.mean((y_te >= lower) & (y_te <= upper)))
        assert 0.85 < coverage < 0.95, f"nominal 90% interval covered {coverage:.3f}"

    def test_mean_is_competitive_with_ordinary_gradient_boosting(self):
        from tuiml.algorithms.gradient_boosting import XGBoostRegressor

        rng = np.random.default_rng(11)
        X_tr, y_tr, _ = _heteroscedastic(1200, rng)
        X_te, y_te, _ = _heteroscedastic(1200, rng)

        ng = NGBoostRegressor(n_estimators=100, learning_rate=0.05,
                              min_samples_leaf=20, random_state=0).fit(X_tr, y_tr)
        xgb = XGBoostRegressor(n_estimators=100, max_depth=3,
                               learning_rate=0.1, random_state=0).fit(X_tr, y_tr)

        rmse_ng = float(np.sqrt(np.mean((ng.predict(X_te) - y_te) ** 2)))
        rmse_xgb = float(np.sqrt(np.mean((xgb.predict(X_te) - y_te) ** 2)))
        assert rmse_ng < 1.25 * rmse_xgb, (rmse_ng, rmse_xgb)

    def test_ordinary_gradient_mode_still_fits(self):
        rng = np.random.default_rng(4)
        X = rng.normal(size=(200, 2))
        y = X[:, 0] + rng.normal(0, 0.4, size=200)
        model = NGBoostRegressor(n_estimators=40, natural_gradient=False,
                                 random_state=0).fit(X, y)
        assert model.train_score_[-1] < model.train_score_[0]
        assert np.all(np.isfinite(model.predict(X)))

    def test_crps_scoring_fits_and_calibrates(self):
        rng = np.random.default_rng(5)
        X_tr, y_tr, _ = _heteroscedastic(1500, rng)
        X_te, y_te, sigma_te = _heteroscedastic(2000, rng)
        model = NGBoostRegressor(scoring="crps", n_estimators=100,
                                 learning_rate=0.05, min_samples_leaf=20,
                                 random_state=0).fit(X_tr, y_tr)
        params = model.predict_dist(X_te)
        assert np.corrcoef(params["scale"], sigma_te)[0, 1] > 0.7

    def test_lognormal_distribution(self):
        rng = np.random.default_rng(6)
        X = rng.normal(size=(300, 2))
        y = np.exp(0.5 * X[:, 0] + rng.normal(0, 0.3, size=300))
        model = NGBoostRegressor(dist="lognormal", n_estimators=40,
                                 random_state=0).fit(X, y)
        pred = model.predict(X)
        assert np.all(pred > 0)
        interval = model.predict_interval(X)
        assert np.all(interval > 0)
        assert np.all(interval[:, 0] < interval[:, 1])
        assert np.corrcoef(pred, y)[0, 1] > 0.5

    def test_lognormal_rejects_non_positive_targets(self):
        X = np.random.default_rng(0).normal(size=(20, 2))
        with pytest.raises(ValueError, match="positive"):
            NGBoostRegressor(dist="lognormal", n_estimators=5).fit(X, np.zeros(20))

    def test_exponential_distribution(self):
        rng = np.random.default_rng(7)
        X = rng.uniform(0, 1, size=(300, 2))
        scale = 1.0 + 3.0 * X[:, 0]
        y = rng.exponential(scale)
        model = NGBoostRegressor(dist="exponential", n_estimators=60,
                                 learning_rate=0.1, min_samples_leaf=10,
                                 random_state=0).fit(X, y)
        params = model.predict_dist(X)
        assert sorted(params) == ["scale"]
        assert np.corrcoef(params["scale"], scale)[0, 1] > 0.5
        assert np.all(model.predict_interval(X) > 0)

    def test_exponential_rejects_negative_targets(self):
        X = np.random.default_rng(0).normal(size=(20, 2))
        with pytest.raises(ValueError, match="non-negative"):
            NGBoostRegressor(dist="exponential", n_estimators=5).fit(X, -np.ones(20))

    def test_score_samples_lower_is_better(self):
        rng = np.random.default_rng(8)
        X = rng.normal(size=(200, 2))
        y = X[:, 0] + rng.normal(0, 0.3, size=200)
        model = NGBoostRegressor(n_estimators=50, random_state=0).fit(X, y)
        good = model.score_samples(X, y).mean()
        bad = model.score_samples(X, y + 5.0).mean()
        assert good < bad

    def test_constant_target_is_stable(self):
        X = np.random.default_rng(0).normal(size=(50, 3))
        y = np.full(50, 7.0)
        model = NGBoostRegressor(n_estimators=20, random_state=0).fit(X, y)
        params = model.predict_dist(X)
        assert np.all(np.isfinite(params["loc"]))
        assert np.all(np.isfinite(params["scale"]))
        assert np.all(params["scale"] > 0)
        assert np.allclose(model.predict(X), 7.0)

    def test_log_scale_is_clamped(self):
        rng = np.random.default_rng(9)
        X = rng.normal(size=(120, 2))
        y = X[:, 0] + rng.normal(0, 0.2, size=120)
        model = NGBoostRegressor(n_estimators=60, random_state=0).fit(X, y)
        params = model.predict_dist(X)
        assert np.all(params["scale"] >= np.exp(model.dist_.log_scale_min))
        assert np.all(params["scale"] <= np.exp(model.dist_.log_scale_max))

    def test_huge_target_offset_does_not_destroy_the_scale(self):
        rng = np.random.default_rng(10)
        X = rng.normal(size=(200, 2))
        y = 1e8 + X[:, 0] + rng.normal(0, 0.5, size=200)
        model = NGBoostRegressor(n_estimators=40, random_state=0).fit(X, y)
        params = model.predict_dist(X)
        assert np.all(np.isfinite(params["scale"]))
        assert params["scale"].mean() < 100.0

    def test_minibatch_fraction(self):
        rng = np.random.default_rng(12)
        X = rng.normal(size=(200, 3))
        y = X[:, 0] + rng.normal(0, 0.4, size=200)
        model = NGBoostRegressor(n_estimators=30, minibatch_frac=0.5,
                                 random_state=0).fit(X, y)
        assert model.n_estimators_ > 0
        assert np.all(np.isfinite(model.predict(X)))

    def test_unfitted_calls_raise_runtime_error(self):
        model = NGBoostRegressor()
        X = np.zeros((3, 2))
        for call in (model.predict, model.predict_dist, model.predict_interval):
            with pytest.raises(RuntimeError, match="must be fitted"):
                call(X)

    def test_feature_count_mismatch_raises(self):
        rng = np.random.default_rng(13)
        X = rng.normal(size=(50, 3))
        model = NGBoostRegressor(n_estimators=10, random_state=0).fit(X, rng.normal(size=50))
        with pytest.raises(ValueError, match="features"):
            model.predict(np.zeros((5, 2)))

    @pytest.mark.parametrize("kwargs", [
        {"n_estimators": 0},
        {"learning_rate": 0.0},
        {"learning_rate": 2.0},
        {"minibatch_frac": 0.0},
        {"minibatch_frac": 1.5},
        {"tol": -1.0},
        {"max_depth": 0},
        {"dist": "poisson"},
        {"scoring": "brier"},
    ])
    def test_invalid_hyperparameters_raise(self, kwargs):
        X = np.random.default_rng(0).normal(size=(20, 2))
        y = np.arange(20, dtype=float)
        with pytest.raises(ValueError):
            NGBoostRegressor(**kwargs).fit(X, y)

    def test_mismatched_lengths_raise(self):
        with pytest.raises(ValueError, match="same number of samples"):
            NGBoostRegressor().fit(np.zeros((10, 2)), np.zeros(9))

    def test_alpha_must_be_a_probability(self):
        rng = np.random.default_rng(14)
        X = rng.normal(size=(40, 2))
        model = NGBoostRegressor(n_estimators=10, random_state=0).fit(X, rng.normal(size=40))
        with pytest.raises(ValueError, match="alpha"):
            model.predict_interval(X, alpha=1.0)

    def test_deterministic_given_random_state(self):
        rng = np.random.default_rng(15)
        X = rng.normal(size=(150, 3))
        y = X[:, 0] + rng.normal(0, 0.5, size=150)
        a = NGBoostRegressor(n_estimators=40, minibatch_frac=0.7, random_state=42).fit(X, y)
        b = NGBoostRegressor(n_estimators=40, minibatch_frac=0.7, random_state=42).fit(X, y)
        assert np.allclose(a.predict(X), b.predict(X))
        assert np.allclose(a.predict_dist(X)["scale"], b.predict_dist(X)["scale"])
        assert a.scalings_ == b.scalings_

    def test_pickle_round_trip(self):
        rng = np.random.default_rng(16)
        X = rng.normal(size=(120, 3))
        y = X[:, 0] + rng.normal(0, 0.4, size=120)
        model = NGBoostRegressor(n_estimators=30, random_state=0).fit(X, y)
        restored = pickle.loads(pickle.dumps(model))
        assert np.allclose(model.predict(X), restored.predict(X))
        assert np.allclose(model.predict_interval(X), restored.predict_interval(X))

    def test_one_dimensional_X_is_accepted(self):
        rng = np.random.default_rng(17)
        X = rng.normal(size=60)
        y = X + rng.normal(0, 0.2, size=60)
        model = NGBoostRegressor(n_estimators=20, random_state=0).fit(X, y)
        assert model.predict(X).shape == (60,)


# --------------------------------------------------------------------------
# Classifier
# --------------------------------------------------------------------------

class TestNGBoostClassifier:
    def test_binary_fit_predict(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(300, 3))
        y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(int)
        model = NGBoostClassifier(n_estimators=40, random_state=0).fit(X, y)
        assert model.classes_.tolist() == [0, 1]
        proba = model.predict_proba(X)
        assert proba.shape == (300, 2)
        assert np.allclose(proba.sum(axis=1), 1.0)
        assert np.all(proba >= 0)
        assert (model.predict(X) == y).mean() > 0.9

    def test_multiclass_fit_predict(self):
        rng = np.random.default_rng(1)
        centres = np.array([[0.0, 0.0], [4.0, 0.0], [0.0, 4.0]])
        y = rng.integers(0, 3, size=300)
        X = centres[y] + rng.normal(0, 0.7, size=(300, 2))
        model = NGBoostClassifier(n_estimators=40, random_state=0).fit(X, y)
        assert model.classes_.tolist() == [0, 1, 2]
        proba = model.predict_proba(X)
        assert proba.shape == (300, 3)
        assert np.allclose(proba.sum(axis=1), 1.0)
        assert (model.predict(X) == y).mean() > 0.9
        assert all(len(stage) == 2 for stage in model.estimators_)

    def test_string_labels_round_trip(self):
        rng = np.random.default_rng(2)
        X = rng.normal(size=(120, 2))
        y = np.where(X[:, 0] > 0, "yes", "no")
        model = NGBoostClassifier(n_estimators=25, random_state=0).fit(X, y)
        assert model.classes_.tolist() == ["no", "yes"]
        assert set(np.unique(model.predict(X))) <= {"no", "yes"}

    def test_probabilities_are_calibrated_on_a_known_logistic_model(self):
        rng = np.random.default_rng(3)
        X = rng.normal(size=(4000, 2))
        true_p = 1.0 / (1.0 + np.exp(-(1.5 * X[:, 0])))
        y = (rng.uniform(size=4000) < true_p).astype(int)
        model = NGBoostClassifier(n_estimators=80, learning_rate=0.1,
                                  min_samples_leaf=30, random_state=0).fit(X, y)

        X_te = rng.normal(size=(4000, 2))
        p_true = 1.0 / (1.0 + np.exp(-(1.5 * X_te[:, 0])))
        p_hat = model.predict_proba(X_te)[:, 1]
        assert np.mean(np.abs(p_hat - p_true)) < 0.06

    def test_predict_dist_returns_probabilities(self):
        rng = np.random.default_rng(4)
        X = rng.normal(size=(80, 2))
        y = (X[:, 0] > 0).astype(int)
        params = NGBoostClassifier(n_estimators=20, random_state=0).fit(X, y).predict_dist(X)
        assert sorted(params) == ["proba"]
        assert params["proba"].shape == (80, 2)

    def test_credible_set_covers_the_truth(self):
        rng = np.random.default_rng(5)
        centres = np.array([[0.0, 0.0], [1.5, 0.0], [0.0, 1.5]])
        y = rng.integers(0, 3, size=600)
        X = centres[y] + rng.normal(0, 1.0, size=(600, 2))
        model = NGBoostClassifier(n_estimators=40, random_state=0).fit(X, y)
        mask = model.predict_interval(X, alpha=0.10)
        assert mask.dtype == bool
        assert mask.shape == (600, 3)
        assert np.all(mask.sum(axis=1) >= 1)
        covered = mask[np.arange(600), y].mean()
        assert covered > 0.85
        # A tighter set never exceeds a looser one.
        loose = model.predict_interval(X, alpha=0.01)
        assert np.all(loose.sum(axis=1) >= mask.sum(axis=1))

    def test_score_samples_lower_is_better(self):
        rng = np.random.default_rng(6)
        X = rng.normal(size=(200, 2))
        y = (X[:, 0] > 0).astype(int)
        model = NGBoostClassifier(n_estimators=40, random_state=0).fit(X, y)
        assert model.score_samples(X, y).mean() < model.score_samples(X, 1 - y).mean()

    def test_score_samples_rejects_unseen_labels(self):
        rng = np.random.default_rng(7)
        X = rng.normal(size=(60, 2))
        y = (X[:, 0] > 0).astype(int)
        model = NGBoostClassifier(n_estimators=10, random_state=0).fit(X, y)
        with pytest.raises(ValueError, match="unseen"):
            model.score_samples(X, np.full(60, 5))

    def test_single_class_raises(self):
        X = np.random.default_rng(0).normal(size=(20, 2))
        with pytest.raises(ValueError, match="at least 2 classes"):
            NGBoostClassifier(n_estimators=5).fit(X, np.zeros(20, dtype=int))

    def test_unfitted_calls_raise_runtime_error(self):
        model = NGBoostClassifier()
        X = np.zeros((3, 2))
        for call in (model.predict, model.predict_proba, model.predict_dist,
                     model.predict_interval):
            with pytest.raises(RuntimeError, match="must be fitted"):
                call(X)

    def test_deterministic_and_picklable(self):
        rng = np.random.default_rng(8)
        X = rng.normal(size=(150, 3))
        y = (X[:, 0] + X[:, 2] > 0).astype(int)
        a = NGBoostClassifier(n_estimators=30, minibatch_frac=0.8, random_state=1).fit(X, y)
        b = NGBoostClassifier(n_estimators=30, minibatch_frac=0.8, random_state=1).fit(X, y)
        assert np.allclose(a.predict_proba(X), b.predict_proba(X))
        restored = pickle.loads(pickle.dumps(a))
        assert np.allclose(a.predict_proba(X), restored.predict_proba(X))


# --------------------------------------------------------------------------
# Contract-facing metadata
# --------------------------------------------------------------------------

@pytest.mark.parametrize("cls", [NGBoostRegressor, NGBoostClassifier])
class TestMetadata:
    def test_schema_lists_every_constructor_parameter(self, cls):
        import inspect

        expected = set(inspect.signature(cls.__init__).parameters) - {"self"}
        assert set(cls.get_parameter_schema()) == expected

    def test_schema_defaults_match_the_signature(self, cls):
        import inspect

        signature = inspect.signature(cls.__init__)
        for name, spec in cls.get_parameter_schema().items():
            assert spec["default"] == signature.parameters[name].default, name

    def test_capabilities_are_known(self, cls):
        caps = cls.get_capabilities()
        assert caps
        unknown = set(caps) - set(KNOWN_CAPABILITIES)
        assert not unknown, f"unknown capabilities: {sorted(unknown)}"

    def test_complexity_and_references(self, cls):
        assert isinstance(cls.get_complexity(), str)
        assert any("Duan" in r for r in cls.get_references())

    def test_repr_is_informative(self, cls):
        assert cls.__name__ in repr(cls())


class TestRegistry:
    def test_registered_under_their_class_names(self):
        assert "NGBoostRegressor" in registry
        assert "NGBoostClassifier" in registry
        assert registry.get("NGBoostRegressor") is NGBoostRegressor
        assert registry.get("NGBoostClassifier") is NGBoostClassifier
