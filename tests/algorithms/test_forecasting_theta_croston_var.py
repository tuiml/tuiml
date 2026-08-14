"""Tests for the Theta, Croston and VAR forecasters.

Each of the three models carries a closed-form anchor that pins the
implementation to the literature: the Theta method must reproduce
Hyndman and Billah's SES-with-drift identity, Croston's ratio must land on
the analytically known value for a perfectly regular demand series, and
VAR's OLS must recover a simulated coefficient matrix and collapse onto a
univariate AR fit when given a single series.
"""

import pickle

import numpy as np
import pytest

from tuiml.algorithms.timeseries.croston import CrostonForecaster
from tuiml.algorithms.timeseries.theta import ThetaForecaster
from tuiml.algorithms.timeseries.var import VAR


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ses_final_level(y, alpha):
    """Return the SES level after the last observation, initialised at ``y[0]``.

    Parameters
    ----------
    y : np.ndarray of shape (n_samples,)
        Series values.
    alpha : float
        Smoothing parameter.

    Returns
    -------
    level : float
        Final level.
    """
    level = float(y[0])
    for value in y:
        level = alpha * float(value) + (1.0 - alpha) * level
    return level


def _ols_slope(y):
    """Return the OLS slope of ``y`` on ``t = 1..n``.

    Parameters
    ----------
    y : np.ndarray of shape (n_samples,)
        Series values.

    Returns
    -------
    slope : float
        The fitted slope.
    """
    t = np.arange(1, len(y) + 1, dtype=float)
    return float(np.polyfit(t, y, 1)[0])


def _sample_series(seed=0, n=60):
    """Return a reproducible trended, noisy series.

    Parameters
    ----------
    seed : int, default=0
        Seed for the generator.
    n : int, default=60
        Series length.

    Returns
    -------
    y : np.ndarray of shape (n,)
        The series.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n, dtype=float)
    return 10.0 + 0.4 * t + rng.normal(scale=1.5, size=n)


# ---------------------------------------------------------------------------
# ThetaForecaster
# ---------------------------------------------------------------------------

class TestThetaForecaster:
    """Behaviour of the Theta method."""

    @pytest.mark.parametrize("alpha", [0.1, 0.3, 0.5, 0.85])
    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_equivalent_to_ses_with_drift(self, alpha, seed):
        """Theta(2) equals SES with drift b/2 (Hyndman & Billah, 2003)."""
        y = _sample_series(seed=seed)
        n = len(y)
        model = ThetaForecaster(theta=2.0, alpha=alpha).fit(y)

        steps = 12
        got = model.predict(steps=steps)

        b = _ols_slope(y)
        level = _ses_final_level(y, alpha)
        h = np.arange(1, steps + 1, dtype=float)
        expected = level + (b / 2.0) * (
            h - 1.0 + 1.0 / alpha - (1.0 - alpha) ** n / alpha
        )

        assert np.max(np.abs(got - expected)) < 1e-8

    def test_drift_attribute_is_half_the_slope(self):
        """For theta=2 the forecast slope is exactly b/2."""
        y = _sample_series()
        model = ThetaForecaster(theta=2.0, alpha=0.3).fit(y)
        assert model.drift_ == pytest.approx(_ols_slope(y) / 2.0, abs=1e-10)
        forecast = model.predict(steps=5)
        assert np.allclose(np.diff(forecast), model.drift_, atol=1e-10)

    def test_theta_zero_line_is_the_ols_line(self):
        """theta=1 keeps the data unchanged; the fit stays finite and sane."""
        y = _sample_series()
        model = ThetaForecaster(theta=1.0, alpha=0.3).fit(y)
        # theta=1 puts all weight on the SES of the raw series: a flat forecast.
        forecast = model.predict(steps=4)
        assert np.allclose(forecast, forecast[0], atol=1e-12)
        assert forecast[0] == pytest.approx(_ses_final_level(y, 0.3), abs=1e-10)

    def test_alpha_optimisation_is_deterministic(self):
        """Two fits without an explicit alpha pick the same value."""
        y = _sample_series(seed=3)
        first = ThetaForecaster().fit(y)
        second = ThetaForecaster().fit(y)
        assert first.alpha_ == second.alpha_
        assert 0.0 < first.alpha_ <= 1.0
        assert np.array_equal(first.predict(6), second.predict(6))

    def test_seasonal_adjustment_recovers_the_pattern(self):
        """A strongly seasonal series is forecast with its seasonality intact."""
        m = 12
        season = np.array([0.7, 0.8, 0.9, 1.0, 1.2, 1.5, 1.6, 1.4, 1.1, 0.9, 0.8, 0.7])
        t = np.arange(120)
        y = (50.0 + 0.5 * t) * season[t % m]

        model = ThetaForecaster(theta=2.0, alpha=0.3, season_length=m).fit(y)
        assert model.is_seasonal_
        assert model.seasonal_mode_ == "mul"
        assert model.seasonal_indices_.shape == (m,)
        assert np.mean(model.seasonal_indices_) == pytest.approx(1.0, abs=1e-10)

        forecast = model.predict(steps=m)
        # Peak and trough land on the right months of the forecast horizon.
        assert int(np.argmax(forecast)) == 6
        assert int(np.argmin(forecast)) in (0, 11)
        # A deterministic seasonal ramp is forecast well.
        truth = (50.0 + 0.5 * np.arange(120, 132)) * season[np.arange(120, 132) % m]
        assert np.mean(np.abs(forecast - truth) / truth) < 0.05

    def test_seasonality_test_rejects_white_noise(self):
        """No seasonal adjustment when the lag-m autocorrelation is not there."""
        rng = np.random.default_rng(7)
        y = 100.0 + rng.normal(scale=5.0, size=120)
        model = ThetaForecaster(season_length=12, alpha=0.3).fit(y)
        assert not model.is_seasonal_
        assert model.seasonal_indices_ is None

    def test_seasonality_test_can_be_disabled(self):
        """``seasonality_test=False`` forces the decomposition."""
        rng = np.random.default_rng(7)
        y = 100.0 + rng.normal(scale=5.0, size=120)
        model = ThetaForecaster(
            season_length=12, alpha=0.3, seasonality_test=False
        ).fit(y)
        assert model.is_seasonal_

    def test_additive_fallback_for_non_positive_series(self):
        """Multiplicative decomposition falls back to additive on zeros."""
        m = 4
        season = np.array([-5.0, 0.0, 5.0, 0.0])
        t = np.arange(80)
        y = 0.1 * t + season[t % m]
        model = ThetaForecaster(
            season_length=m, alpha=0.3, seasonal="mul", seasonality_test=False
        ).fit(y)
        assert model.seasonal_mode_ == "add"
        assert np.mean(model.seasonal_indices_) == pytest.approx(0.0, abs=1e-10)

    def test_fitted_values_and_residuals_align(self):
        """In-sample containers have one entry per observation."""
        y = _sample_series()
        model = ThetaForecaster(alpha=0.3).fit(y)
        assert model.fitted_values_.shape == y.shape
        assert np.allclose(model.resid_, y - model.fitted_values_)

    def test_predict_before_fit_raises(self):
        """Predicting an unfitted model raises rather than crashing on None."""
        with pytest.raises(RuntimeError):
            ThetaForecaster().predict(steps=2)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"theta": 0.0},
            {"theta": -1.0},
            {"alpha": 0.0},
            {"alpha": 1.5},
            {"seasonal": "additive"},
        ],
    )
    def test_invalid_parameters_raise(self, kwargs):
        """Bad constructor parameters are rejected at fit time."""
        with pytest.raises(ValueError):
            ThetaForecaster(**kwargs).fit(_sample_series())

    def test_too_short_series_raises(self):
        """Fewer than three points cannot support a line plus a level."""
        with pytest.raises(ValueError):
            ThetaForecaster().fit(np.array([1.0, 2.0]))


# ---------------------------------------------------------------------------
# CrostonForecaster
# ---------------------------------------------------------------------------

def _regular_intermittent(size=10.0, interval=4, cycles=12):
    """Return a demand of ``size`` occurring every ``interval`` periods.

    Parameters
    ----------
    size : float, default=10.0
        Demand size at each arrival.
    interval : int, default=4
        Periods between arrivals.
    cycles : int, default=12
        Number of arrivals.

    Returns
    -------
    y : np.ndarray
        The demand series, ending on an arrival.
    """
    y = np.zeros(interval * cycles, dtype=float)
    y[interval - 1::interval] = size
    return y


class TestCrostonForecaster:
    """Behaviour of Croston's method and its variants."""

    @pytest.mark.parametrize("alpha", [0.05, 0.1, 0.3, 0.7, 1.0])
    def test_classic_converges_to_size_over_interval(self, alpha):
        """A demand of 10 every 4 periods forecasts exactly 2.5 per period."""
        y = _regular_intermittent()
        model = CrostonForecaster(alpha=alpha, variant="classic").fit(y)
        assert model.demand_ == pytest.approx(10.0, abs=1e-12)
        assert model.interval_ == pytest.approx(4.0, abs=1e-12)
        assert model.predict(steps=1)[0] == pytest.approx(2.5, abs=1e-12)

    @pytest.mark.parametrize("alpha", [0.05, 0.1, 0.3, 0.7, 1.0])
    def test_sba_applies_the_syntetos_boylan_factor(self, alpha):
        """SBA is exactly ``(1 - alpha/2)`` times the classic forecast."""
        y = _regular_intermittent()
        model = CrostonForecaster(alpha=alpha, variant="sba").fit(y)
        assert model.correction_ == pytest.approx(1.0 - alpha / 2.0, abs=1e-15)
        assert model.predict(steps=1)[0] == pytest.approx(
            2.5 * (1.0 - alpha / 2.0), abs=1e-12
        )

    @pytest.mark.parametrize("alpha", [0.05, 0.1, 0.3, 0.7])
    def test_sbj_applies_the_shale_boylan_johnston_factor(self, alpha):
        """SBJ is exactly ``(1 - alpha/(2 - alpha))`` times the classic value."""
        y = _regular_intermittent()
        model = CrostonForecaster(alpha=alpha, variant="sbj").fit(y)
        factor = 1.0 - alpha / (2.0 - alpha)
        assert model.correction_ == pytest.approx(factor, abs=1e-15)
        assert model.predict(steps=1)[0] == pytest.approx(2.5 * factor, abs=1e-12)

    def test_corrections_shrink_the_classic_forecast(self):
        """Both bias corrections pull the ratio estimate downward."""
        y = _regular_intermittent()
        classic = CrostonForecaster(alpha=0.3, variant="classic").fit(y).forecast_
        sba = CrostonForecaster(alpha=0.3, variant="sba").fit(y).forecast_
        sbj = CrostonForecaster(alpha=0.3, variant="sbj").fit(y).forecast_
        assert sbj < sba < classic

    def test_tsb_forecast_is_probability_times_size(self):
        """TSB multiplies the smoothed size by the smoothed demand probability."""
        y = _regular_intermittent()
        model = CrostonForecaster(alpha=0.2, variant="tsb").fit(y)
        assert model.forecast_ == pytest.approx(
            model.probability_ * model.demand_, abs=1e-12
        )
        assert 0.0 < model.probability_ < 1.0
        assert np.isnan(model.interval_)
        # A regular 1-in-4 series has a demand probability near 0.25.
        assert model.probability_ == pytest.approx(0.25, abs=0.1)

    def test_tsb_decays_after_demand_stops(self):
        """Unlike Croston, TSB lets the forecast decay towards zero."""
        active = _regular_intermittent(cycles=12)
        dead = np.concatenate([active, np.zeros(60)])

        croston = CrostonForecaster(alpha=0.2, variant="classic")
        assert croston.fit(dead).forecast_ == pytest.approx(
            croston.fit(active).forecast_, abs=1e-12
        )

        tsb_active = CrostonForecaster(alpha=0.2, variant="tsb").fit(active).forecast_
        tsb_dead = CrostonForecaster(alpha=0.2, variant="tsb").fit(dead).forecast_
        assert tsb_dead < 0.05 * tsb_active

    @pytest.mark.parametrize("variant", ["classic", "sba", "sbj", "tsb"])
    def test_all_zero_series_forecasts_zero_without_dividing_by_zero(self, variant):
        """An empty demand history is handled, not divided by."""
        y = np.zeros(40)
        model = CrostonForecaster(alpha=0.2, variant=variant).fit(y)
        forecast = model.predict(steps=5)
        assert np.all(forecast == 0.0)
        assert np.isfinite(forecast).all()
        assert model.n_nonzero_ == 0

    @pytest.mark.parametrize("variant", ["classic", "sba", "sbj", "tsb"])
    def test_never_zero_series_is_plain_smoothing(self, variant):
        """With demand every period the interval is 1 and TSB's probability is 1."""
        y = np.full(30, 7.0)
        model = CrostonForecaster(alpha=0.3, variant=variant).fit(y)
        forecast = model.predict(steps=3)
        assert np.isfinite(forecast).all()
        if variant == "tsb":
            assert model.probability_ == pytest.approx(1.0, abs=1e-12)
            assert forecast[0] == pytest.approx(7.0, abs=1e-12)
        else:
            assert model.interval_ == pytest.approx(1.0, abs=1e-12)
            assert forecast[0] == pytest.approx(7.0 * model.correction_, abs=1e-12)

    def test_single_observation_series(self):
        """A one-period history still produces a finite forecast."""
        model = CrostonForecaster(alpha=0.2).fit(np.array([5.0]))
        assert model.predict(steps=2).tolist() == [5.0, 5.0]

    def test_forecast_is_flat(self):
        """Croston forecasts are constant over the horizon."""
        y = _regular_intermittent()
        forecast = CrostonForecaster(alpha=0.2).fit_predict(y, steps=7)
        assert forecast.shape == (7,)
        assert np.allclose(forecast, forecast[0])

    def test_predict_before_fit_raises(self):
        """Predicting an unfitted model raises."""
        with pytest.raises(RuntimeError):
            CrostonForecaster().predict()

    @pytest.mark.parametrize(
        "kwargs",
        [{"alpha": 0.0}, {"alpha": 1.2}, {"variant": "croston"}, {"alpha_prob": 0.0}],
    )
    def test_invalid_parameters_raise(self, kwargs):
        """Bad constructor parameters are rejected at fit time."""
        with pytest.raises(ValueError):
            CrostonForecaster(**kwargs).fit(_regular_intermittent())


# ---------------------------------------------------------------------------
# VAR
# ---------------------------------------------------------------------------

def _simulate_var1(A, n, seed=0, scale=1.0):
    """Simulate a stable VAR(1) with unit-variance Gaussian shocks.

    Parameters
    ----------
    A : np.ndarray of shape (k, k)
        Coefficient matrix.
    n : int
        Number of time points.
    seed : int, default=0
        Seed for the generator.
    scale : float, default=1.0
        Innovation standard deviation.

    Returns
    -------
    y : np.ndarray of shape (n, k)
        The simulated panel.
    """
    rng = np.random.default_rng(seed)
    k = A.shape[0]
    y = np.zeros((n + 200, k), dtype=float)
    shocks = rng.normal(scale=scale, size=(n + 200, k))
    for t in range(1, n + 200):
        y[t] = A @ y[t - 1] + shocks[t]
    return y[200:]


class TestVAR:
    """Behaviour of the vector autoregression."""

    def test_recovers_known_coefficients(self):
        """OLS converges to the true VAR(1) matrix on a long sample."""
        A = np.array([[0.5, 0.1], [-0.2, 0.3]])
        y = _simulate_var1(A, n=5000, seed=11)
        model = VAR(lags=1).fit(y)

        assert model.coefs_.shape == (1, 2, 2)
        assert np.max(np.abs(model.coefs_[0] - A)) < 0.05
        assert np.max(np.abs(model.intercept_)) < 0.1

    def test_matches_univariate_ar1_ols(self):
        """A single-series VAR(1) reproduces the univariate AR(1) OLS fit."""
        rng = np.random.default_rng(5)
        n = 400
        y = np.zeros(n)
        for t in range(1, n):
            y[t] = 1.0 + 0.6 * y[t - 1] + rng.normal()

        model = VAR(lags=1).fit(y)

        design = np.column_stack([np.ones(n - 1), y[:-1]])
        beta = np.linalg.lstsq(design, y[1:], rcond=None)[0]

        assert abs(model.intercept_[0] - beta[0]) < 1e-10
        assert abs(model.coefs_[0, 0, 0] - beta[1]) < 1e-10

        # And the one-step forecast matches the closed-form OLS prediction.
        expected = beta[0] + beta[1] * y[-1]
        assert abs(float(model.predict(steps=1)[0]) - expected) < 1e-10

    def test_accepts_a_one_dimensional_series(self):
        """A 1-D series is a single-series panel and forecasts back in 1-D."""
        y = _sample_series(n=60)
        model = VAR(lags=1).fit(y)
        assert model.n_series_ == 1
        assert model.input_was_1d_
        forecast = model.predict(steps=3)
        assert forecast.shape == (3,)
        assert np.isfinite(forecast).all()

    def test_multi_step_forecast_iterates_the_companion_form(self):
        """Recursive forecasts equal a hand-rolled iteration of the VAR."""
        A = np.array([[0.5, 0.1], [-0.2, 0.3]])
        y = _simulate_var1(A, n=300, seed=2)
        model = VAR(lags=2).fit(y)

        steps = 6
        got = model.predict(steps=steps)

        history = [y[-2], y[-1]]
        expected = []
        for _ in range(steps):
            pred = model.intercept_ + model.coefs_[0] @ history[-1] + model.coefs_[1] @ history[-2]
            expected.append(pred)
            history.append(pred)
        assert np.max(np.abs(got - np.array(expected))) < 1e-12

    def test_forecast_of_a_stable_var_decays_to_the_mean(self):
        """A zero-mean stable VAR forecasts back towards zero."""
        A = np.array([[0.5, 0.1], [-0.2, 0.3]])
        y = _simulate_var1(A, n=1000, seed=4)
        forecast = VAR(lags=1).fit(y).predict(steps=60)
        assert np.max(np.abs(forecast[-1])) < 0.2

    @pytest.mark.parametrize("ic", ["aic", "bic"])
    def test_auto_lag_selection(self, ic):
        """Order selection recovers a low order for VAR(1) data."""
        A = np.array([[0.5, 0.1], [-0.2, 0.3]])
        y = _simulate_var1(A, n=1500, seed=6)
        model = VAR(lags="auto", maxlags=6, ic=ic).fit(y)
        assert model.lags_ in (1, 2)
        assert set(model.ic_values_) == set(range(1, 7))
        assert model.ic_values_[model.lags_] == min(model.ic_values_.values())

    def test_bic_is_no_more_generous_than_aic(self):
        """BIC's heavier penalty never selects a larger order here."""
        A = np.array([[0.5, 0.1], [-0.2, 0.3]])
        y = _simulate_var1(A, n=1500, seed=8)
        aic = VAR(lags="auto", maxlags=6, ic="aic").fit(y).lags_
        bic = VAR(lags="auto", maxlags=6, ic="bic").fit(y).lags_
        assert bic <= aic

    def test_trend_none_omits_the_intercept(self):
        """``trend="n"`` leaves the intercept at exactly zero."""
        A = np.array([[0.5, 0.1], [-0.2, 0.3]])
        y = _simulate_var1(A, n=500, seed=9)
        model = VAR(lags=1, trend="n").fit(y)
        assert np.array_equal(model.intercept_, np.zeros(2))

    def test_residuals_are_orthogonal_to_the_design(self):
        """OLS residuals have zero mean when an intercept is fitted."""
        A = np.array([[0.5, 0.1], [-0.2, 0.3]])
        y = _simulate_var1(A, n=800, seed=10)
        model = VAR(lags=2).fit(y)
        assert np.max(np.abs(model.resid_.mean(axis=0))) < 1e-10
        assert model.fitted_values_.shape == (len(y) - 2, 2)
        assert model.sigma_.shape == (2, 2)

    def test_singular_design_raises_a_clear_error(self):
        """A duplicated series is rejected instead of yielding NaN estimates."""
        A = np.array([[0.5, 0.1], [-0.2, 0.3]])
        base = _simulate_var1(A, n=300, seed=12)
        y = np.column_stack([base[:, 0], base[:, 0]])
        with pytest.raises(ValueError, match="rank deficient"):
            VAR(lags=1).fit(y)

    def test_constant_series_raises(self):
        """A constant series is collinear with the intercept."""
        y = np.ones((50, 2))
        with pytest.raises(ValueError, match="rank deficient"):
            VAR(lags=1).fit(y)

    def test_series_too_short_for_the_lag_order_raises(self):
        """Fewer rows than regressors is an error, not a pinv fit."""
        A = np.array([[0.5, 0.1], [-0.2, 0.3]])
        y = _simulate_var1(A, n=8, seed=13)
        with pytest.raises(ValueError):
            VAR(lags=5).fit(y)

    def test_predict_before_fit_raises(self):
        """Predicting an unfitted model raises."""
        with pytest.raises(RuntimeError):
            VAR().predict(steps=2)

    @pytest.mark.parametrize(
        "kwargs", [{"lags": 0}, {"lags": "best"}, {"ic": "hqic"}, {"trend": "ct"}]
    )
    def test_invalid_parameters_raise(self, kwargs):
        """Bad constructor parameters are rejected at fit time."""
        A = np.array([[0.5, 0.1], [-0.2, 0.3]])
        y = _simulate_var1(A, n=200, seed=14)
        if kwargs.get("ic") == "hqic":
            kwargs["lags"] = "auto"
        with pytest.raises(ValueError):
            VAR(**kwargs).fit(y)


# ---------------------------------------------------------------------------
# Shared contract-style checks
# ---------------------------------------------------------------------------

def _fitted_models():
    """Return one fitted instance of each new forecaster.

    Returns
    -------
    models : list
        Fitted estimators.
    """
    y = _sample_series()
    return [
        ThetaForecaster(alpha=0.3).fit(y),
        CrostonForecaster(alpha=0.2, variant="sba").fit(_regular_intermittent()),
        VAR(lags=1).fit(y),
    ]


@pytest.mark.parametrize("model", _fitted_models())
def test_pickle_roundtrip_preserves_forecasts(model):
    """Every model survives a pickle roundtrip with identical forecasts."""
    before = model.predict(steps=5)
    restored = pickle.loads(pickle.dumps(model))
    assert np.array_equal(before, restored.predict(steps=5))


@pytest.mark.parametrize("cls", [ThetaForecaster, CrostonForecaster, VAR])
def test_parameter_schema_covers_every_constructor_argument(cls):
    """The schema lists exactly the constructor parameters."""
    import inspect

    expected = set(inspect.signature(cls.__init__).parameters) - {"self"}
    assert set(cls.get_parameter_schema()) == expected


@pytest.mark.parametrize("cls", [ThetaForecaster, CrostonForecaster, VAR])
def test_capabilities_are_known_and_include_forecasting(cls):
    """Declared capabilities are spelled the way the contract suite expects."""
    from tests.contract._data import KNOWN_CAPABILITIES

    caps = cls.get_capabilities()
    assert "forecasting" in caps
    assert "timeseries" in caps
    unknown = set(caps) - set(KNOWN_CAPABILITIES)
    assert not unknown, f"{cls.__name__} declares unknown capabilities: {unknown}"


@pytest.mark.parametrize("cls", [ThetaForecaster, CrostonForecaster, VAR])
def test_fit_is_deterministic(cls):
    """Two fits on the same data give bit-identical forecasts."""
    y = _sample_series(seed=21)
    first = cls().fit(y).predict(steps=4)
    second = cls().fit(y).predict(steps=4)
    assert np.array_equal(first, second)


@pytest.mark.parametrize("cls", [ThetaForecaster, CrostonForecaster, VAR])
def test_fit_does_not_mutate_the_input(cls):
    """``fit`` leaves the caller's array untouched."""
    y = _sample_series(seed=22)
    original = y.copy()
    cls().fit(y)
    assert np.array_equal(y, original)


@pytest.mark.parametrize("cls", [ThetaForecaster, CrostonForecaster, VAR])
def test_fit_returns_self_and_fit_predict_matches(cls):
    """``fit`` returns the estimator and ``fit_predict`` is the composition."""
    y = _sample_series(seed=23)
    model = cls()
    assert model.fit(y) is model
    assert np.array_equal(model.predict(steps=3), cls().fit_predict(y, steps=3))
