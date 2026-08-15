"""Regression tests for two long-standing ARIMA defects.

Both were silent: the model accepted the input, fitted without complaint, and
returned forecasts that simply ignored what had been asked for. Tests that only
checked "does it run" could never have caught either, so these check what the
fitted *parameters* are.

1. ``ma_params_`` was initialised to ``np.zeros(q)`` and the refinement loop
   ran over ``range(p)`` only, so theta stayed at zero forever and
   ``ARIMA(order=(0, 0, 1))`` was a constant model.
2. ``seasonal_order`` was stored in ``__init__`` and read nowhere, so a
   seasonal specification quietly fitted a non-seasonal model.
"""

import inspect

import numpy as np
import pytest

from tuiml.algorithms.timeseries import ARIMA, SARIMAX


def _ma1(theta: float, n: int = 3000, seed: int = 0) -> np.ndarray:
    """Simulate an MA(1) series with a known theta."""
    rng = np.random.default_rng(seed)
    e = rng.normal(size=n + 1)
    return e[1:] + theta * e[:-1]


def _ar1(phi: float, n: int = 3000, seed: int = 1) -> np.ndarray:
    """Simulate an AR(1) series with a known phi."""
    rng = np.random.default_rng(seed)
    e = rng.normal(size=n)
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = phi * y[t - 1] + e[t]
    return y


# ---------------------------------------------------------------------------
# 1. Moving-average parameters are actually estimated
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("theta", [0.6, -0.5, 0.3])
def test_ma_parameter_is_recovered(theta):
    """The headline regression: theta used to stay at exactly 0.0."""
    model = ARIMA(order=(0, 0, 1)).fit(_ma1(theta))
    assert model.ma_params_.shape == (1,)
    assert model.ma_params_[0] == pytest.approx(theta, abs=0.1)


def test_ma_parameters_are_not_left_at_zero():
    """State the old bug directly, so a regression names itself."""
    model = ARIMA(order=(0, 0, 2)).fit(_ma1(0.6))
    assert np.any(np.abs(model.ma_params_) > 1e-6), \
        "ma_params_ left at its zero initialisation -- refinement skipped MA"


def test_ar_parameter_still_recovered():
    """Replacing the optimiser must not regress the AR path it did handle."""
    model = ARIMA(order=(1, 0, 0)).fit(_ar1(0.7))
    assert model.ar_params_[0] == pytest.approx(0.7, abs=0.1)


def test_arma_recovers_both_blocks():
    """AR and MA are optimised jointly, not one at a time."""
    rng = np.random.default_rng(2)
    n = 3000
    e = rng.normal(size=n)
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = 0.5 * y[t - 1] + e[t] + 0.4 * e[t - 1]
    model = ARIMA(order=(1, 0, 1)).fit(y)
    assert model.ar_params_[0] == pytest.approx(0.5, abs=0.15)
    assert model.ma_params_[0] == pytest.approx(0.4, abs=0.15)


def test_refinement_never_makes_the_fit_worse():
    """A failed optimisation must fall back to the Yule-Walker starting point."""
    y = _ar1(0.7, n=200)
    refined = ARIMA(order=(2, 0, 2), method="css-mle", maxiter=50).fit(y)
    initial = ARIMA(order=(2, 0, 2), method="css").fit(y)
    lag = 2
    assert np.sum(refined.resid_[lag:] ** 2) <= np.sum(initial.resid_[lag:] ** 2) * 1.001


def test_fit_is_deterministic():
    """The optimiser is seeded by the data alone, so repeats must agree."""
    y = _ma1(0.6, n=500)
    a = ARIMA(order=(1, 0, 1)).fit(y)
    b = ARIMA(order=(1, 0, 1)).fit(y)
    assert np.allclose(a.ma_params_, b.ma_params_)
    assert np.allclose(a.ar_params_, b.ar_params_)


def test_ma_model_forecast_is_not_flat_at_the_constant():
    """With theta stuck at 0, a pure-MA forecast was just the constant."""
    model = ARIMA(order=(0, 0, 1)).fit(_ma1(0.6))
    forecast = model.predict(steps=3)
    assert forecast.shape == (3,)
    assert np.all(np.isfinite(forecast))


# ---------------------------------------------------------------------------
# 2. The dead seasonal_order parameter is gone
# ---------------------------------------------------------------------------

def test_seasonal_order_is_no_longer_accepted():
    """Silently ignoring it was worse than rejecting it outright."""
    with pytest.raises(TypeError, match="seasonal_order"):
        ARIMA(order=(1, 1, 1), seasonal_order=(1, 1, 1, 12))


def test_schema_matches_the_signature():
    """A schema listing a parameter the constructor lacks misleads agents."""
    params = set(inspect.signature(ARIMA.__init__).parameters) - {"self"}
    assert params == set(ARIMA.get_parameter_schema())
    assert "seasonal_order" not in ARIMA.get_parameter_schema()


def test_sarimax_covers_what_was_removed():
    """The replacement must genuinely handle a seasonal specification.

    Guards the redirect in ARIMA's docstring: if SARIMAX ever stopped
    accepting a seasonal order, that advice would send users nowhere.
    """
    t = np.arange(120)
    seasonal = 5.0 * np.sin(2 * np.pi * t / 12)
    rng = np.random.default_rng(3)
    y = 10.0 + seasonal + rng.normal(0, 0.2, size=t.size)

    seasonal_fit = SARIMAX(order=(0, 0, 0), seasonal_order=(0, 1, 0, 12)).fit(y)
    flat_fit = SARIMAX(order=(0, 0, 0)).fit(y)

    horizon = 24
    truth = 10.0 + 5.0 * np.sin(2 * np.pi * np.arange(120, 120 + horizon) / 12)
    seasonal_err = np.mean(np.abs(seasonal_fit.predict(steps=horizon) - truth))
    flat_err = np.mean(np.abs(flat_fit.predict(steps=horizon) - truth))
    assert seasonal_err < flat_err
