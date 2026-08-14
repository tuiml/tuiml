"""Numerical anchors for the SARIMAX and TBATS forecasters.

Every check here pins a claim the docstrings make: that SARIMAX maximises the
*exact* Gaussian likelihood, that it recovers exogenous coefficients ARIMA
cannot see, that its seasonal differencing is real rather than accepted-and-
ignored, and that TBATS' trigonometric seasonality handles non-integer periods
and multiple simultaneous cycles.
"""

import pickle
import time

import numpy as np
import pytest

from tuiml.algorithms.timeseries.sarimax import SARIMAX, _pacf_to_ar, _psi_weights
from tuiml.algorithms.timeseries.tbats import TBATS, _box_cox, _inv_box_cox

SEED = 0


# ---------------------------------------------------------------------------
# Synthetic series
# ---------------------------------------------------------------------------

def ar1_series(phi=0.7, n=2000, scale=0.5, seed=SEED):
    """Simulate a zero-mean AR(1) process."""
    rng = np.random.default_rng(seed)
    eps = rng.normal(scale=scale, size=n)
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = phi * y[t - 1] + eps[t]
    return y


def ols_ar1(y):
    """Fit AR(1) with an intercept by ordinary least squares."""
    design = np.column_stack([np.ones(len(y) - 1), y[:-1]])
    beta, *_ = np.linalg.lstsq(design, y[1:], rcond=None)
    return beta[1]


def seasonal_series(n_cycles=20, period=12, amp=5.0, noise=0.1, seed=SEED):
    """Return a purely seasonal series plus its next-cycle continuation."""
    rng = np.random.default_rng(seed)
    base = amp * np.sin(2 * np.pi * np.arange(period) / period)
    y = np.tile(base, n_cycles) + rng.normal(scale=noise, size=n_cycles * period)
    return y, np.tile(base, 2)


# ---------------------------------------------------------------------------
# SARIMAX -- helper level
# ---------------------------------------------------------------------------

def test_pacf_to_ar_is_stationary():
    """Every partial autocorrelation vector maps inside the unit circle."""
    rng = np.random.default_rng(SEED)
    for _ in range(50):
        pacf = rng.uniform(-0.99, 0.99, size=4)
        phi = _pacf_to_ar(pacf)
        roots = np.roots(np.concatenate([[1.0], -phi]))
        assert np.all(np.abs(roots) < 1.0), "companion roots must be stable"


def test_psi_weights_match_ar1_powers():
    """The MA(inf) weights of an AR(1) are powers of phi."""
    psi = _psi_weights(np.array([0.6]), np.zeros(0), 6)
    np.testing.assert_allclose(psi, 0.6 ** np.arange(6), atol=1e-12)


# ---------------------------------------------------------------------------
# SARIMAX -- correctness anchors
# ---------------------------------------------------------------------------

def test_ar1_coefficient_matches_ols():
    """Anchor: the ML AR(1) coefficient matches a direct OLS fit to ~0.02."""
    y = ar1_series()
    model = SARIMAX(order=(1, 0, 0), trend="c").fit(y)
    assert abs(model.ar_params_[0] - ols_ar1(y)) < 0.02
    assert abs(model.ar_params_[0] - 0.7) < 0.05


def test_exog_coefficient_recovered():
    """Anchor: y = 3x + tiny noise with order=(0,0,0) recovers beta = 3."""
    rng = np.random.default_rng(SEED)
    x = rng.normal(size=(400, 1))
    y = 3.0 * x[:, 0] + 0.01 * rng.normal(size=400)
    model = SARIMAX(order=(0, 0, 0)).fit(y, x)
    assert model.exog_params_.shape == (1,)
    assert abs(model.exog_params_[0] - 3.0) < 0.01


def test_exog_multi_column_and_forecast():
    """Two exogenous columns are recovered and drive the forecast."""
    rng = np.random.default_rng(SEED)
    X = rng.normal(size=(400, 2))
    y = 2.0 * X[:, 0] - 1.5 * X[:, 1] + 0.02 * rng.normal(size=400)
    model = SARIMAX(order=(0, 0, 0)).fit(y, X)
    np.testing.assert_allclose(model.exog_params_, [2.0, -1.5], atol=0.02)

    X_future = np.array([[1.0, 0.0], [0.0, 1.0]])
    forecast = model.predict(steps=2, X=X_future)
    np.testing.assert_allclose(forecast, [2.0, -1.5], atol=0.05)


def test_loglik_maximised_at_truth():
    """Anchor: the Kalman log-likelihood peaks at the true AR coefficient."""
    y = ar1_series(phi=0.7)
    model = SARIMAX(order=(1, 0, 0))
    model.n_obs_ = len(y)
    empty = np.zeros((len(y), 0))

    def loglik(phi):
        return -model._neg_loglik(np.array([np.arctanh(phi)]), y, empty)

    at_truth = loglik(0.7)
    assert at_truth > loglik(0.3)
    assert at_truth > loglik(0.0)
    assert at_truth > loglik(0.95)
    assert at_truth > loglik(-0.5)
    # The fitted optimum is at least as good as the truth.
    assert model.fit(y).loglik_ >= at_truth - 1e-6


def test_seasonal_differencing_removes_planted_pattern():
    """Anchor: seasonal differencing genuinely eliminates the seasonal cycle."""
    y, _ = seasonal_series()
    fitted = SARIMAX(order=(0, 0, 0), seasonal_order=(0, 1, 0, 12)).fit(y)
    # After (1 - L^12) the seasonal signal is gone: the residual scale should
    # collapse to the noise level rather than the amplitude of the cycle.
    assert np.std(y) > 3.0
    assert np.sqrt(fitted.sigma2_) < 0.5


def test_seasonal_model_forecasts_the_cycle():
    """A seasonal specification reproduces the next two cycles."""
    y, next_cycles = seasonal_series()
    model = SARIMAX(order=(1, 0, 0), seasonal_order=(0, 1, 0, 12)).fit(y)
    forecast = model.predict(steps=24)
    assert np.mean(np.abs(forecast - next_cycles)) < 0.3

    flat = SARIMAX(order=(1, 0, 0), trend="c").fit(y).predict(steps=24)
    assert (np.mean(np.abs(forecast - next_cycles))
            < 0.25 * np.mean(np.abs(flat - next_cycles)))


def test_differencing_round_trip_on_a_trend():
    """d=1 differencing is inverted correctly, with and without drift."""
    rng = np.random.default_rng(SEED)
    y = 3.0 + 0.5 * np.arange(100.0) + 0.01 * rng.normal(size=100)
    truth = 3.0 + 0.5 * np.arange(100.0, 105.0)

    # trend="c" puts a drift on the differenced scale, so integrating it back
    # must reproduce the linear trend.
    drift = SARIMAX(order=(0, 1, 0), trend="c").fit(y)
    assert abs(drift.trend_params_[0] - 0.5) < 0.01
    np.testing.assert_allclose(drift.predict(steps=5), truth, atol=0.05)

    # Without a constant an ARIMA(0,1,0) is a driftless random walk: the
    # forecast is flat at the last observation.
    flat = SARIMAX(order=(0, 1, 0)).fit(y).predict(steps=5)
    np.testing.assert_allclose(flat, y[-1], atol=0.05)


def test_intervals_widen_and_cover():
    """Prediction intervals widen with the horizon and bracket the forecast."""
    y = ar1_series(n=500)
    model = SARIMAX(order=(1, 0, 0)).fit(y)
    forecast, lower, upper = model.predict_interval(steps=10, alpha=0.05)
    assert np.all(lower < forecast) and np.all(forecast < upper)
    widths = upper - lower
    assert np.all(np.diff(widths) > 0), "interval must widen with the horizon"
    # One step ahead the interval is 2 * 1.96 * sigma wide.
    assert abs(widths[0] - 2 * 1.959964 * np.sqrt(model.sigma2_)) < 1e-3
    narrow = model.predict_interval(steps=10, alpha=0.32)[2] - \
        model.predict_interval(steps=10, alpha=0.32)[1]
    assert np.all(narrow < widths)


def test_random_walk_interval_grows_like_sqrt_h():
    """With d=1 the forecast variance accumulates, as an ARIMA(0,1,0) should."""
    rng = np.random.default_rng(SEED)
    y = np.cumsum(rng.normal(size=400))
    model = SARIMAX(order=(0, 1, 0)).fit(y)
    var = model.forecast_variance(steps=9)
    np.testing.assert_allclose(var / var[0], np.arange(1, 10), rtol=1e-8)


def test_ma_component_is_estimated():
    """The MA block is actually fitted, not left at zero as in ARIMA."""
    rng = np.random.default_rng(SEED)
    eps = rng.normal(size=1500)
    y = eps[1:] + 0.6 * eps[:-1]
    model = SARIMAX(order=(0, 0, 1)).fit(y)
    assert abs(model.ma_params_[0] - 0.6) < 0.1


# ---------------------------------------------------------------------------
# SARIMAX -- API contract
# ---------------------------------------------------------------------------

def test_predict_raises_before_fit():
    """predict() reports the not-fitted state before touching any attribute."""
    with pytest.raises(RuntimeError, match="must be fitted"):
        SARIMAX().predict(steps=1)
    with pytest.raises(RuntimeError, match="must be fitted"):
        SARIMAX().predict_interval(steps=1)
    with pytest.raises(RuntimeError, match="must be fitted"):
        SARIMAX().forecast_variance(steps=1)


def test_exog_mismatch_errors():
    """Exogenous presence must agree between fit() and predict()."""
    rng = np.random.default_rng(SEED)
    x = rng.normal(size=(120, 1))
    y = 3.0 * x[:, 0] + 0.1 * rng.normal(size=120)

    with_exog = SARIMAX(order=(0, 0, 0)).fit(y, x)
    with pytest.raises(ValueError, match="requires X"):
        with_exog.predict(steps=3)
    with pytest.raises(ValueError, match="must have shape"):
        with_exog.predict(steps=3, X=np.zeros((2, 1)))

    without = SARIMAX(order=(0, 0, 0)).fit(y)
    with pytest.raises(ValueError, match="without exogenous"):
        without.predict(steps=3, X=np.zeros((3, 1)))

    with pytest.raises(ValueError, match="aligned with the series"):
        SARIMAX(order=(0, 0, 0)).fit(y, x[:50])


def test_seasonal_order_needs_a_period():
    """A seasonal order without a usable period is rejected loudly."""
    y = ar1_series(n=100)
    with pytest.raises(ValueError, match="period m > 1"):
        SARIMAX(order=(1, 0, 0), seasonal_order=(1, 0, 0, 0)).fit(y)


def test_trend_terms():
    """A constant and a linear trend are both estimated."""
    y = 5.0 + np.zeros(200)
    rng = np.random.default_rng(SEED)
    y = y + 0.05 * rng.normal(size=200)
    model = SARIMAX(order=(0, 0, 0), trend="c").fit(y)
    assert abs(model.trend_params_[0] - 5.0) < 0.05
    np.testing.assert_allclose(model.predict(steps=3), 5.0, atol=0.05)

    ct = SARIMAX(order=(0, 0, 0), trend="ct").fit(y)
    assert ct.trend_params_.shape == (2,)


def test_sarimax_fit_predict_and_pickle():
    """fit_predict works and a fitted model survives a pickle round-trip."""
    y = ar1_series(n=300)
    model = SARIMAX(order=(1, 0, 0))
    forecast = model.fit_predict(y, steps=4)
    assert forecast.shape == (4,)

    restored = pickle.loads(pickle.dumps(model))
    np.testing.assert_allclose(restored.predict(steps=4), forecast)


def test_sarimax_is_deterministic():
    """Two identical fits produce identical numbers."""
    y = ar1_series(n=300)
    a = SARIMAX(order=(1, 0, 1)).fit(y).predict(steps=5)
    b = SARIMAX(order=(1, 0, 1)).fit(y).predict(steps=5)
    np.testing.assert_array_equal(a, b)


def test_sarimax_contract_shape_and_speed():
    """The generic contract case -- 60 points, default params -- is fast."""
    rng = np.random.default_rng(SEED)
    y = rng.normal(size=60)
    start = time.perf_counter()
    model = SARIMAX().fit(y)
    out = model.predict()
    elapsed = time.perf_counter() - start
    assert out.shape == (1,)
    assert np.all(np.isfinite(out))
    assert elapsed < 1.0, f"default fit+predict took {elapsed:.3f}s"


# ---------------------------------------------------------------------------
# TBATS -- helper level
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("lam", [0.0, 0.25, 0.5, 1.0, -0.5])
def test_box_cox_round_trip(lam):
    """The Box-Cox transform and its inverse compose to the identity."""
    y = np.linspace(0.5, 20.0, 50)
    np.testing.assert_allclose(_inv_box_cox(_box_cox(y, lam), lam), y, rtol=1e-9)


# ---------------------------------------------------------------------------
# TBATS -- correctness anchors
# ---------------------------------------------------------------------------

def two_sinusoids(n=240, start=0.0):
    """Return a series that is a linear trend plus two sinusoids."""
    t = np.arange(start, start + n, dtype=float)
    return (20.0 + 0.08 * t
            + 3.0 * np.sin(2 * np.pi * t / 12.0)
            + 2.0 * np.sin(2 * np.pi * t / 7.0 + 0.4))


def test_two_seasonalities_beat_a_no_seasonality_baseline():
    """Anchor: two simultaneous cycles are forecast far better than without."""
    y = two_sinusoids()
    truth = two_sinusoids(n=24, start=240.0)

    model = TBATS(seasonal_periods=[12, 7]).fit(y)
    seasonal_mae = np.mean(np.abs(model.predict(steps=24) - truth))

    baseline = TBATS(seasonal_periods=None).fit(y)
    baseline_mae = np.mean(np.abs(baseline.predict(steps=24) - truth))

    assert seasonal_mae < 0.05
    assert seasonal_mae / baseline_mae < 0.05, (
        f"seasonal MAE {seasonal_mae:.4g} vs baseline {baseline_mae:.4g}"
    )


def test_box_cox_lambda_one_is_the_identity():
    """Anchor: lambda = 1 reproduces the untransformed fit exactly."""
    t = np.arange(240.0)
    y = 50.0 + 0.05 * t + 4.0 * np.sin(2 * np.pi * t / 12.0)

    transformed = TBATS(seasonal_periods=[12], box_cox=True,
                        box_cox_lambda=1.0).fit(y)
    plain = TBATS(seasonal_periods=[12], box_cox=False).fit(y)

    assert transformed.lambda_ == 1.0
    assert plain.lambda_ is None
    np.testing.assert_allclose(transformed.predict(steps=12),
                               plain.predict(steps=12), atol=1e-6)


def test_box_cox_selects_a_lambda_and_handles_multiplicative_data():
    """A grid search picks a lambda and the log scale is fitted sensibly."""
    t = np.arange(200.0)
    y = np.exp(2.0 + 0.005 * t + 0.3 * np.sin(2 * np.pi * t / 12.0))
    model = TBATS(seasonal_periods=[12], box_cox=True).fit(y)
    assert model.lambda_ in (0.0, 0.25, 0.5, 0.75, 1.0)
    truth = np.exp(2.0 + 0.005 * np.arange(200.0, 212.0)
                   + 0.3 * np.sin(2 * np.pi * np.arange(200.0, 212.0) / 12.0))
    forecast = model.predict(steps=12)
    assert np.mean(np.abs(forecast - truth) / truth) < 0.05

    with pytest.raises(ValueError, match="strictly positive"):
        TBATS(box_cox=True).fit(np.array([1.0, -2.0, 3.0, 4.0]))


def test_non_integer_seasonal_period():
    """Anchor: a non-integer period works and crushes a naive baseline."""
    period = 365.25 / 7.0
    t = np.arange(300.0)
    y = 10.0 + 2.0 * np.sin(2 * np.pi * t / period)
    future = np.arange(300.0, 320.0)
    truth = 10.0 + 2.0 * np.sin(2 * np.pi * future / period)

    model = TBATS(seasonal_periods=[period]).fit(y)
    forecast = model.predict(steps=20)
    naive = np.full(20, y[-1])

    tbats_mae = np.mean(np.abs(forecast - truth))
    naive_mae = np.mean(np.abs(naive - truth))
    assert np.all(np.isfinite(forecast))
    assert tbats_mae < 0.05
    assert tbats_mae < 0.1 * naive_mae


def test_non_integer_period_scalar_argument():
    """A bare float is accepted wherever a list of periods is."""
    t = np.arange(200.0)
    y = 5.0 + 2.0 * np.sin(2 * np.pi * t / 52.18)
    model = TBATS(seasonal_periods=52.18).fit(y)
    assert model.harmonics_ == [5]
    assert model.predict(steps=4).shape == (4,)


def test_high_frequency_period_keeps_the_state_small():
    """A 365.25 period costs harmonics, not one state per seasonal index."""
    rng = np.random.default_rng(SEED)
    t = np.arange(800.0)
    y = 10.0 + 3.0 * np.sin(2 * np.pi * t / 365.25) + rng.normal(scale=0.1,
                                                                 size=800)
    model = TBATS(seasonal_periods=[365.25], n_harmonics=2,
                  use_arma_errors=False).fit(y)
    assert model.harmonics_ == [2]
    assert model.seasonal_.shape == (4,)  # 2 harmonic pairs -> 4 states
    future = np.arange(800.0, 830.0)
    truth = 10.0 + 3.0 * np.sin(2 * np.pi * future / 365.25)
    assert np.mean(np.abs(model.predict(steps=30) - truth)) < 0.5


def test_trend_extrapolation_and_damping_parameter():
    """An undamped trend extrapolates linearly; damping stays in its box."""
    t = np.arange(120.0)
    y = 1.0 + 0.5 * t
    truth = 1.0 + 0.5 * np.arange(120.0, 150.0)

    undamped = TBATS(seasonal_periods=None, damped_trend=False,
                     use_arma_errors=False).fit(y)
    np.testing.assert_allclose(undamped.predict(steps=30), truth, atol=0.5)
    assert undamped.params_["phi"] == 1.0

    damped = TBATS(seasonal_periods=None, damped_trend=True,
                   use_arma_errors=False).fit(y)
    assert 0.8 <= damped.params_["phi"] <= 1.0
    # On an exactly linear series the optimum damping is at the top of the
    # box, so the damped forecast should not fall short of the trend.
    assert np.mean(np.abs(damped.predict(steps=30) - truth)) < 1.0


def test_no_trend_forecast_is_flat():
    """With use_trend=False and no seasonality the forecast is a level."""
    rng = np.random.default_rng(SEED)
    y = 7.0 + rng.normal(scale=0.1, size=200)
    model = TBATS(seasonal_periods=None, use_trend=False,
                  use_arma_errors=False).fit(y)
    forecast = model.predict(steps=5)
    np.testing.assert_allclose(forecast, forecast[0], atol=1e-9)
    assert abs(forecast[0] - 7.0) < 0.2


def test_arma_errors_are_fitted():
    """The residual ARMA stage recovers autocorrelation left by smoothing."""
    y = ar1_series(phi=0.8, n=600, scale=0.3) + 30.0
    model = TBATS(seasonal_periods=None, use_arma_errors=True,
                  arma_order=(1, 0)).fit(y)
    assert model.ar_params_.shape == (1,)
    off = TBATS(seasonal_periods=None, use_arma_errors=False).fit(y)
    assert off.ar_params_.shape == (0,)
    assert np.all(np.isfinite(model.predict(steps=5)))


def test_tbats_validation_errors():
    """Bad specifications are rejected with actionable messages."""
    y = np.linspace(1.0, 10.0, 60)
    with pytest.raises(ValueError, match="must all be > 1"):
        TBATS(seasonal_periods=[1.0]).fit(y)
    with pytest.raises(ValueError, match="one entry per seasonal period"):
        TBATS(seasonal_periods=[7, 12], n_harmonics=[2]).fit(y)
    with pytest.raises(ValueError, match="non-finite"):
        TBATS().fit(np.array([1.0, np.nan, 3.0, 4.0]))
    with pytest.raises(ValueError, match="steps must be"):
        TBATS().fit(y).predict(steps=0)


# ---------------------------------------------------------------------------
# TBATS -- API contract
# ---------------------------------------------------------------------------

def test_tbats_predict_raises_before_fit():
    """predict() reports the not-fitted state first."""
    with pytest.raises(RuntimeError, match="must be fitted"):
        TBATS().predict(steps=1)


def test_tbats_fit_predict_and_pickle():
    """fit_predict works and a fitted model survives a pickle round-trip."""
    y = two_sinusoids()
    model = TBATS(seasonal_periods=[12, 7])
    forecast = model.fit_predict(y, steps=6)
    assert forecast.shape == (6,)

    restored = pickle.loads(pickle.dumps(model))
    np.testing.assert_allclose(restored.predict(steps=6), forecast)


def test_tbats_is_deterministic():
    """Two identical fits produce identical numbers."""
    y = two_sinusoids()
    a = TBATS(seasonal_periods=[12]).fit(y).predict(steps=5)
    b = TBATS(seasonal_periods=[12]).fit(y).predict(steps=5)
    np.testing.assert_array_equal(a, b)


def test_tbats_contract_shape_and_speed():
    """The generic contract case -- 60 points, default params -- is fast."""
    rng = np.random.default_rng(SEED)
    y = rng.normal(size=60)
    start = time.perf_counter()
    out = TBATS().fit(y).predict()
    elapsed = time.perf_counter() - start
    assert out.shape == (1,)
    assert np.all(np.isfinite(out))
    assert elapsed < 1.0, f"default fit+predict took {elapsed:.3f}s"


# ---------------------------------------------------------------------------
# Registry / metadata conventions
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cls", [SARIMAX, TBATS])
def test_schema_lists_every_constructor_parameter(cls):
    """get_parameter_schema() must cover the full constructor signature."""
    import inspect

    params = set(inspect.signature(cls.__init__).parameters) - {"self"}
    schema = cls.get_parameter_schema()
    assert params == set(schema), (
        f"schema mismatch for {cls.__name__}: "
        f"missing={params - set(schema)}, extra={set(schema) - params}"
    )
    for name, spec in schema.items():
        assert "type" in spec and "description" in spec
        assert "default" in spec, f"{name} has no documented default"


@pytest.mark.parametrize("cls", [SARIMAX, TBATS])
def test_capabilities_are_known_strings(cls):
    """Capabilities must come from the shared vocabulary and include forecasting."""
    from tests.contract._data import KNOWN_CAPABILITIES

    caps = cls.get_capabilities()
    assert "forecasting" in caps
    assert "timeseries" in caps
    unknown = set(caps) - set(KNOWN_CAPABILITIES)
    assert not unknown, f"{cls.__name__} declares unknown capabilities: {unknown}"


@pytest.mark.parametrize("cls", [SARIMAX, TBATS])
def test_get_params_round_trip(cls):
    """Constructor defaults survive get_params/set_params."""
    model = cls()
    params = model.get_params()
    for key in cls.get_parameter_schema():
        assert key in params
    clone = cls(**params)
    assert clone.get_params() == params
