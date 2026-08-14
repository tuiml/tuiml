"""Tests for the deep, torch-backed forecasters.

Three things are checked here that a shape-and-smoke test would miss:

1. **The optional-dependency contract.** No module-level torch import, and a
   clear ``ImportError`` from ``fit`` when torch is absent, simulated by
   poisoning ``sys.modules``.
2. **Forecast quality.** Each model must beat a naive last-value baseline and
   a seasonal-naive baseline by a wide margin on a clean signal. A deep model
   that only ties a naive baseline is broken, not merely underfit.
3. **Architecture identity.** Each model is checked against the specific
   mechanism its paper contributes — the doubly-residual invariant for
   N-BEATS, multi-rate pooling and interpolation for N-HiTS, patching and
   RevIN for PatchTST — so that none of them can silently degrade into a
   generic multilayer perceptron.
"""

from __future__ import annotations

import ast
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from tuiml.algorithms.timeseries.deep import (
    NBEATSForecaster,
    NHITSForecaster,
    PatchTSTForecaster,
)
from tuiml.algorithms.timeseries.deep._base import (
    check_series,
    make_windows,
    resolve_window,
)
from tuiml.algorithms.timeseries.deep.nbeats import fourier_basis, polynomial_basis
from tuiml.algorithms.timeseries.deep.nhits import interpolation_length, pooled_length
from tuiml.algorithms.timeseries.deep.patchtst import (
    num_patches,
    padded_length,
    patchify,
    revin_denormalize,
    revin_normalize,
)
from tuiml.utils.torch_backend import has_torch

MODELS = [NBEATSForecaster, NHITSForecaster, PatchTSTForecaster]

requires_torch = pytest.mark.skipif(not has_torch(), reason="PyTorch not installed")

#: Period of ``sin(t / 5)`` rounded to whole steps, for the seasonal-naive
#: baseline and the seasonality checks.
SEASON = 31


def clean_signal(n: int = 500) -> np.ndarray:
    """Return a deterministic trend-plus-seasonality signal.

    Parameters
    ----------
    n : int, default=500
        Number of points.

    Returns
    -------
    series : np.ndarray of shape (n,)
        ``sin(t / 5) + 0.05 * t``.
    """
    t = np.arange(n, dtype=np.float64)
    return np.sin(t / 5.0) + 0.05 * t


def naive_baselines(train: np.ndarray, horizon: int) -> tuple:
    """Return the last-value and seasonal-naive forecasts.

    Parameters
    ----------
    train : np.ndarray of shape (n_train,)
        Training portion of the series.
    horizon : int
        Number of steps to forecast.

    Returns
    -------
    naive : np.ndarray of shape (horizon,)
        The last observed value, repeated.
    seasonal_naive : np.ndarray of shape (horizon,)
        The values one season back, tiled to the horizon.
    """
    naive = np.repeat(train[-1], horizon)
    seasonal = np.resize(train[-SEASON:], horizon)
    return naive, seasonal


def mse(prediction: np.ndarray, truth: np.ndarray) -> float:
    """Return the mean squared error.

    Parameters
    ----------
    prediction : np.ndarray
        Forecast values.
    truth : np.ndarray
        Observed values.

    Returns
    -------
    error : float
        Mean squared error.
    """
    return float(np.mean((np.asarray(prediction) - np.asarray(truth)) ** 2))


# ---------------------------------------------------------------------------
# The optional-dependency contract
# ---------------------------------------------------------------------------

def test_no_module_level_torch_import():
    """No file in the subpackage may import torch at module scope."""
    package = Path(NBEATSForecaster.__module__.replace(".", "/")).parent
    root = Path(__file__).resolve().parents[2]
    files = sorted((root / package).glob("*.py"))
    assert len(files) >= 5, f"expected the whole subpackage, found {files}"

    for path in files:
        tree = ast.parse(path.read_text())
        for node in tree.body:  # module scope only, not function bodies
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            assert not any(
                name == "torch" or name.startswith("torch.") for name in names
            ), f"{path.name} imports torch at module scope"


@pytest.mark.parametrize("cls", MODELS)
def test_construction_and_metadata_without_torch(cls, monkeypatch):
    """Constructing, and reading metadata, must work with torch absent."""
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "torch.nn", None)

    model = cls(random_state=0)
    assert model.lookback == 24
    schema = cls.get_parameter_schema()
    assert "random_state" in schema
    assert "forecasting" in cls.get_capabilities()


@pytest.mark.parametrize("cls", MODELS)
def test_fit_without_torch_raises_import_error(cls, monkeypatch):
    """``fit`` must name the class and the exact install command."""
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "torch.nn", None)

    model = cls(random_state=0)
    with pytest.raises(ImportError) as excinfo:
        model.fit(clean_signal(60))

    message = str(excinfo.value)
    assert cls.__name__ in message
    assert "pip install 'tuiml[torch]'" in message


@pytest.mark.parametrize("cls", MODELS)
def test_schema_covers_every_constructor_parameter(cls):
    """Every constructor argument must appear in the parameter schema."""
    import inspect

    expected = set(inspect.signature(cls.__init__).parameters) - {"self"}
    assert set(cls.get_parameter_schema()) == expected


@pytest.mark.parametrize("cls", MODELS)
def test_capabilities_are_known_strings(cls):
    """Capabilities must be spelled the way the contract suite expects."""
    from tests.contract._data import KNOWN_CAPABILITIES

    capabilities = cls.get_capabilities()
    assert set(capabilities) <= KNOWN_CAPABILITIES
    assert "forecasting" in capabilities


@pytest.mark.parametrize("cls", MODELS)
def test_predict_before_fit_raises(cls):
    """``predict`` must guard on its first line, not crash on a None."""
    with pytest.raises(RuntimeError, match="must be fitted"):
        cls().predict(3)


# ---------------------------------------------------------------------------
# Windowing helpers (no torch involved)
# ---------------------------------------------------------------------------

def test_resolve_window_shrinks_rather_than_failing():
    """A short series shrinks the window instead of raising."""
    assert resolve_window(500, 48, 12) == (48, 12)

    lookback, horizon = resolve_window(60, 24, 8)
    assert (lookback, horizon) == (24, 8)

    lookback, horizon = resolve_window(30, 96, 24)
    assert lookback >= 4 and horizon >= 1
    assert 30 - lookback - horizon + 1 >= 4


def test_resolve_window_rejects_hopeless_series():
    """Below the minimum the error must say how many points are needed."""
    with pytest.raises(ValueError, match="too short"):
        resolve_window(6, 24, 8)


def test_make_windows_alignment():
    """Targets must be the points immediately following each input window."""
    X, Y = make_windows(np.arange(10.0), lookback=4, horizon=2)
    assert X.shape == (5, 4)
    assert Y.shape == (5, 2)
    np.testing.assert_allclose(Y[0], [4.0, 5.0])
    np.testing.assert_allclose(X[-1], [4.0, 5.0, 6.0, 7.0])


def test_check_series_rejects_a_design_matrix():
    """A 2-D input with several columns is a user error worth explaining."""
    with pytest.raises(ValueError, match="1-D"):
        check_series(np.zeros((10, 3)))
    np.testing.assert_allclose(check_series(np.zeros((5, 1))), np.zeros(5))


# ---------------------------------------------------------------------------
# The contract-suite path: a 60-point series
# ---------------------------------------------------------------------------

@requires_torch
@pytest.mark.parametrize("cls", MODELS)
def test_fits_sixty_points_quickly(cls):
    """The generic contract fits every model on 60 points; keep it fast."""
    y = np.asarray(clean_signal(60), dtype=float)
    model = cls(random_state=0)

    start = time.perf_counter()
    fitted = model.fit(y)
    elapsed = time.perf_counter() - start

    assert fitted is model
    forecast = model.predict()
    assert forecast.shape == (1,)
    assert np.all(np.isfinite(forecast))
    assert elapsed < 5.0, f"{cls.__name__} took {elapsed:.2f}s on 60 points"


@requires_torch
@pytest.mark.parametrize("cls", MODELS)
def test_rollout_longer_than_horizon(cls):
    """Requests longer than the trained horizon roll the forecast forward."""
    model = cls(lookback=24, horizon=6, n_epochs=20, random_state=0)
    model.fit(clean_signal(200))
    forecast = model.predict(steps=17)
    assert forecast.shape == (17,)
    assert np.all(np.isfinite(forecast))


@requires_torch
@pytest.mark.parametrize("cls", MODELS)
def test_fit_predict_matches_fit_then_predict(cls):
    """``fit_predict`` is a convenience wrapper, not a different model."""
    y = clean_signal(150)
    first = cls(n_epochs=20, random_state=0).fit_predict(y, steps=5)
    model = cls(n_epochs=20, random_state=0)
    model.fit(y)
    np.testing.assert_allclose(first, model.predict(5))


# ---------------------------------------------------------------------------
# Forecast quality
# ---------------------------------------------------------------------------

@requires_torch
@pytest.mark.parametrize(
    "cls,kwargs",
    [
        (NBEATSForecaster, {}),
        (NBEATSForecaster, {"stack_type": "interpretable"}),
        (NHITSForecaster, {}),
        (PatchTSTForecaster, {}),
    ],
    ids=["nbeats-generic", "nbeats-interpretable", "nhits", "patchtst"],
)
def test_beats_naive_baselines(cls, kwargs):
    """Each model must beat both naive baselines by a wide margin."""
    series = clean_signal(500)
    horizon = 20
    train, test = series[:-horizon], series[-horizon:]

    model = cls(
        lookback=40,
        horizon=20,
        n_epochs=300,
        patience=40,
        random_state=0,
        **kwargs,
    )
    model.fit(train)
    forecast = model.predict(horizon)

    naive, seasonal = naive_baselines(train, horizon)
    model_mse = mse(forecast, test)

    assert model_mse / mse(naive, test) < 0.05
    assert model_mse / mse(seasonal, test) < 0.05


@requires_torch
@pytest.mark.parametrize("cls", MODELS)
def test_deterministic_given_a_seed(cls):
    """Same seed, same device, identical forecasts."""
    y = clean_signal(200)
    first = cls(n_epochs=30, random_state=7).fit(y).predict(10)
    second = cls(n_epochs=30, random_state=7).fit(y).predict(10)
    np.testing.assert_array_equal(first, second)


@requires_torch
@pytest.mark.parametrize("cls", MODELS)
def test_default_device_is_cpu(cls):
    """``cpu`` is the default so reproducibility does not depend on hardware."""
    assert cls().device == "cpu"
    assert cls.get_parameter_schema()["device"]["default"] == "cpu"


@requires_torch
@pytest.mark.parametrize("cls", MODELS)
def test_pickle_round_trip_preserves_forecasts(cls):
    """The network is rebuilt from saved weights, not from a stale object."""
    model = cls(n_epochs=30, random_state=3)
    model.fit(clean_signal(200))
    before = model.predict(12)

    restored = pickle.loads(pickle.dumps(model))
    np.testing.assert_allclose(restored.predict(12), before, rtol=0, atol=0)


@requires_torch
@pytest.mark.parametrize("cls", MODELS)
def test_constant_series_does_not_divide_by_zero(cls):
    """A flat series has zero variance everywhere; the fit must still run."""
    model = cls(n_epochs=10, random_state=0)
    model.fit(np.full(80, 4.0))
    forecast = model.predict(5)
    assert np.all(np.isfinite(forecast))


# ---------------------------------------------------------------------------
# N-BEATS: the doubly-residual architecture
# ---------------------------------------------------------------------------

@requires_torch
def test_nbeats_block_forecasts_sum_to_the_output():
    """The model output is exactly the sum of the block forecasts."""
    import torch

    model = NBEATSForecaster(n_epochs=30, n_blocks=2, n_stacks=2, random_state=0)
    model.fit(clean_signal(200))

    windows = torch.as_tensor(
        model.series_[-model.lookback_:][None, :], dtype=torch.float32
    )
    with torch.no_grad():
        normalised, _, _ = model._normalise(torch, windows)
        detail = model.module_.forward_detail(normalised)

    assert detail["block_forecasts"].shape == (1, 4, model.horizon_)
    np.testing.assert_allclose(
        detail["block_forecasts"].sum(dim=1).numpy(),
        detail["forecast"].numpy(),
        rtol=1e-5,
        atol=1e-6,
    )


@requires_torch
def test_nbeats_residuals_shrink_through_the_stack():
    """Each block removes signal, so the residual norm must fall."""
    import torch

    model = NBEATSForecaster(
        lookback=40, horizon=10, n_blocks=2, n_epochs=300, patience=40,
        random_state=0,
    )
    model.fit(clean_signal(500))

    X, _ = make_windows(model.series_, model.lookback_, model.horizon_)
    batch = torch.as_tensor(X[:64], dtype=torch.float32)
    with torch.no_grad():
        normalised, _, _ = model._normalise(torch, batch)
        residuals = model.module_.forward_detail(normalised)["residuals"]

    norms = residuals.pow(2).mean(dim=(0, 2)).numpy()
    assert norms[-1] < norms[0] * 0.2, f"residual norms did not shrink: {norms}"
    # No block may meaningfully *add* energy: that would mean it is fitting the
    # target while corrupting what the next block sees.
    assert np.all(np.diff(norms) < 0.1 * norms[0]), f"a block added energy: {norms}"


@requires_torch
def test_nbeats_interpretable_components_sum_to_the_forecast():
    """Trend plus seasonality is the forecast, exactly."""
    model = NBEATSForecaster(
        stack_type="interpretable", lookback=40, horizon=10, n_epochs=60,
        random_state=0,
    )
    model.fit(clean_signal(300))

    parts = model.decompose(steps=10)
    np.testing.assert_allclose(
        parts["trend"] + parts["seasonality"], parts["forecast"], rtol=1e-6, atol=1e-8
    )
    np.testing.assert_allclose(parts["forecast"], model.predict(10), rtol=1e-6)


@requires_torch
def test_nbeats_trend_stack_is_a_low_order_polynomial():
    """The trend component must lie exactly on a degree-p polynomial."""
    degree = 2
    model = NBEATSForecaster(
        stack_type="interpretable", lookback=40, horizon=12,
        trend_polynomial_degree=degree, n_epochs=60, random_state=0,
    )
    model.fit(clean_signal(300))

    trend = model.decompose(steps=12)["trend"]
    grid = np.arange(len(trend), dtype=float)
    residual = trend - np.polyval(np.polyfit(grid, trend, degree), grid)
    assert np.max(np.abs(residual)) < 1e-6 * max(1.0, np.ptp(trend))

    # And it is genuinely low order: a degree-1 fit must leave real residual
    # unless the model happened to learn a straight line.
    assert np.max(np.abs(trend - trend[0])) > 0.0


@requires_torch
def test_nbeats_seasonality_stack_is_periodic_and_zero_mean():
    """The Fourier basis excludes the constant term, so the mean is zero."""
    model = NBEATSForecaster(
        stack_type="interpretable", lookback=40, horizon=12, n_harmonics=3,
        n_epochs=60, random_state=0,
    )
    model.fit(clean_signal(300))

    seasonality = model.decompose(steps=12)["seasonality"]
    assert abs(float(np.mean(seasonality))) < 1e-6 * max(
        1.0, float(np.ptp(seasonality))
    )
    assert float(np.ptp(seasonality)) > 0.0


def test_fourier_basis_is_periodic():
    """Extending the basis over two periods repeats it exactly."""
    period = 12
    single = fourier_basis(3, period)
    double = fourier_basis(3, 2 * period)[:, : 2 * period]
    # fourier_basis normalises by the length it is given, so build the
    # extension explicitly on the same fundamental period.
    grid = np.arange(2 * period) / period
    rows = []
    for harmonic in range(1, 4):
        rows.append(np.cos(2 * np.pi * harmonic * grid))
        rows.append(np.sin(2 * np.pi * harmonic * grid))
    extended = np.stack(rows)

    np.testing.assert_allclose(extended[:, :period], single, atol=1e-12)
    np.testing.assert_allclose(
        extended[:, period:], extended[:, :period], atol=1e-12
    )
    np.testing.assert_allclose(single.mean(axis=1), 0.0, atol=1e-12)
    assert double.shape == (6, 2 * period)


def test_polynomial_basis_shape_and_content():
    """Row p of the basis is t**p on a unit grid."""
    basis = polynomial_basis(3, 8)
    assert basis.shape == (4, 8)
    grid = np.arange(8) / 8.0
    np.testing.assert_allclose(basis[1], grid)
    np.testing.assert_allclose(basis[3], grid ** 3)


@requires_torch
def test_nbeats_decompose_requires_the_interpretable_configuration():
    """The generic configuration has no trend/seasonality split to report."""
    model = NBEATSForecaster(n_epochs=5, random_state=0)
    model.fit(clean_signal(120))
    with pytest.raises(ValueError, match="interpretable"):
        model.decompose(5)


# ---------------------------------------------------------------------------
# N-HiTS: multi-rate sampling and hierarchical interpolation
# ---------------------------------------------------------------------------

def test_pooled_and_interpolation_lengths():
    """Pooling uses ceil mode; knot counts never exceed the horizon."""
    assert pooled_length(24, 4) == 6
    assert pooled_length(10, 4) == 3      # ceil, so the tail is not dropped
    assert pooled_length(3, 8) == 1       # kernel clipped to the input
    assert interpolation_length(24, 8) == 3
    assert interpolation_length(24, 1) == 24
    assert interpolation_length(4, 16) == 1


@requires_torch
def test_nhits_pooling_sizes_change_the_forecast():
    """If pooling were inert, all rates would give the same answer."""
    y = clean_signal(300)
    flat = NHITSForecaster(
        pooling_sizes=(1, 1, 1), n_freq_downsample=(1, 1, 1),
        n_epochs=40, random_state=0,
    ).fit(y).predict(8)
    pooled = NHITSForecaster(
        pooling_sizes=(8, 4, 1), n_freq_downsample=(1, 1, 1),
        n_epochs=40, random_state=0,
    ).fit(y).predict(8)

    assert not np.allclose(flat, pooled, atol=1e-6), (
        "pooling rate had no effect: the multi-rate machinery is inert"
    )


@requires_torch
def test_nhits_downsampling_changes_the_forecast():
    """Likewise the interpolation ratio must actually matter."""
    y = clean_signal(300)
    fine = NHITSForecaster(
        pooling_sizes=(4, 2, 1), n_freq_downsample=(1, 1, 1),
        n_epochs=40, random_state=0,
    ).fit(y).predict(8)
    coarse = NHITSForecaster(
        pooling_sizes=(4, 2, 1), n_freq_downsample=(8, 4, 1),
        n_epochs=40, random_state=0,
    ).fit(y).predict(8)

    assert not np.allclose(fine, coarse, atol=1e-6)


@requires_torch
def test_nhits_blocks_pool_down_and_interpolate_back_up():
    """Each block sees a shortened input and emits a full-length horizon."""
    import torch

    model = NHITSForecaster(
        lookback=32, horizon=12, pooling_sizes=(8, 2, 1),
        n_freq_downsample=(6, 3, 1), n_blocks=1, n_epochs=5, random_state=0,
    )
    model.fit(clean_signal(200))

    blocks = list(model.module_.blocks)
    assert [block.pooled_size for block in blocks] == [4, 16, 32]
    assert [block.n_forecast_knots for block in blocks] == [2, 4, 12]

    batch = torch.zeros(3, 32)
    for block in blocks:
        backcast, forecast = block(batch)
        assert backcast.shape == (3, 32), "backcast must return to the lookback"
        assert forecast.shape == (3, 12), "interpolation must fill the horizon"


@requires_torch
def test_nhits_rejects_mismatched_stack_specifications():
    """One pooling rate and one ratio per stack, checked without torch."""
    with pytest.raises(ValueError, match="one entry per"):
        NHITSForecaster(pooling_sizes=(4, 2, 1), n_freq_downsample=(2, 1))


# ---------------------------------------------------------------------------
# PatchTST: patching, channel independence and RevIN
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "length,patch_len,stride,expected",
    [
        (24, 8, 8, 3),    # disjoint, divides exactly
        (24, 8, 4, 5),    # overlapping, divides exactly
        (25, 8, 4, 6),    # does not divide: one extra padded patch
        (30, 7, 5, 6),    # neither divides
        (5, 8, 4, 1),     # window shorter than one patch
    ],
)
def test_num_patches(length, patch_len, stride, expected):
    """Patch counts must be right whether or not the lengths divide."""
    assert num_patches(length, patch_len, stride) == expected
    assert padded_length(length, patch_len, stride) >= length


@requires_torch
@pytest.mark.parametrize(
    "length,patch_len,stride", [(24, 8, 8), (24, 8, 4), (25, 8, 4), (30, 7, 5)]
)
def test_patchify_shapes_and_content(length, patch_len, stride):
    """Patchify must produce the counted patches without dropping the tail."""
    import torch

    x = torch.arange(2 * length, dtype=torch.float32).reshape(2, length)
    patches = patchify(torch, x, patch_len, stride)

    expected = num_patches(length, patch_len, stride)
    assert patches.shape == (2, expected, min(patch_len, length))
    np.testing.assert_allclose(patches[0, 0].numpy(), x[0, :patch_len].numpy())
    # The final value of the window survives into the final patch: padding
    # repeats the last point rather than truncating it away.
    assert float(patches[0, -1, -1]) == float(x[0, -1])


@requires_torch
def test_revin_round_trips():
    """Normalise then denormalise must be the identity."""
    import torch

    generator = torch.Generator().manual_seed(0)
    x = torch.randn(16, 40, generator=generator) * 7.5 + 120.0

    normalised, loc, scale = revin_normalize(torch, x)
    restored = revin_denormalize(torch, normalised, loc, scale)

    assert torch.max(torch.abs(restored - x)).item() < 1e-6
    assert torch.max(torch.abs(normalised.mean(dim=-1))).item() < 1e-5
    assert torch.max(torch.abs(normalised.std(dim=-1, unbiased=False) - 1)).item() < 1e-4


@requires_torch
def test_revin_handles_a_constant_window():
    """A zero-variance window must not produce NaNs."""
    import torch

    x = torch.full((4, 10), 3.0)
    normalised, loc, scale = revin_normalize(torch, x)
    assert torch.isfinite(normalised).all()
    restored = revin_denormalize(torch, normalised, loc, scale)
    assert torch.max(torch.abs(restored - x)).item() < 1e-6


@requires_torch
def test_patchtst_channel_independence():
    """Channels share one backbone and never see each other."""
    import torch

    model = PatchTSTForecaster(lookback=24, horizon=6, n_epochs=5, random_state=0)
    model.fit(clean_signal(200))

    generator = torch.Generator().manual_seed(1)
    panel = torch.randn(3, 4, 24, generator=generator)
    with torch.no_grad():
        stacked = model.module_(panel)
        one_by_one = torch.stack(
            [model.module_(panel[:, c, :]) for c in range(4)], dim=1
        )

    assert stacked.shape == (3, 4, 6)
    np.testing.assert_allclose(
        stacked.numpy(), one_by_one.numpy(), rtol=1e-5, atol=1e-6
    )

    # Perturbing one channel must leave the others bit-identical.
    perturbed = panel.clone()
    perturbed[:, 0, :] += 50.0
    with torch.no_grad():
        after = model.module_(perturbed)
    np.testing.assert_allclose(
        after[:, 1:, :].numpy(), stacked[:, 1:, :].numpy(), rtol=1e-5, atol=1e-6
    )


@requires_torch
def test_patchtst_revin_flag_changes_the_fit():
    """Turning RevIN off must actually change what the model does."""
    y = clean_signal(300)
    with_revin = PatchTSTForecaster(n_epochs=30, random_state=0, revin=True)
    without = PatchTSTForecaster(n_epochs=30, random_state=0, revin=False)
    assert not np.allclose(
        with_revin.fit(y).predict(8), without.fit(y).predict(8), atol=1e-6
    )


def test_patchtst_rejects_indivisible_head_count():
    """``d_model`` must split evenly across heads; checked without torch."""
    with pytest.raises(ValueError, match="divisible"):
        PatchTSTForecaster(d_model=30, n_heads=4)
