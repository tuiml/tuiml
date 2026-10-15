"""Contract and behaviour tests for the zero-shot forecaster wrappers.

Mirrors ``test_tabicl.py``. Everything that must hold on *any* install -- the
schema, the capabilities, the namespaced registration, the forecaster API
shape, the error raised when the extra is missing -- runs unconditionally.

The tests that actually run a model are gated on the upstream package being
importable, and they download a checkpoint on first use.
"""

import inspect
import pickle
import sys
from importlib.abc import MetaPathFinder

import numpy as np
import pytest

import tuiml
from tuiml.registry import registry
from tuiml.foundation import ChronosForecaster, TimesFMForecaster
from ..contract._data import KNOWN_CAPABILITIES

WRAPPERS = [ChronosForecaster, TimesFMForecaster]
UPSTREAM = {ChronosForecaster: "chronos", TimesFMForecaster: "timesfm"}


def _importable(name: str) -> bool:
    try:  # pragma: no cover - depends on whether the extra is installed
        __import__(name)
        return True
    except ImportError:  # pragma: no cover
        return False


needs_chronos = pytest.mark.skipif(
    not _importable("chronos"),
    reason="needs pip install 'tuiml[foundation]' and downloads a checkpoint",
)
needs_timesfm = pytest.mark.skipif(
    not _importable("timesfm"),
    reason="needs pip install 'tuiml[foundation]' and downloads a checkpoint",
)


def _series(n: int = 200) -> np.ndarray:
    t = np.arange(n)
    return 10 + 0.05 * t + np.sin(2 * np.pi * t / 12)


# ---------------------------------------------------------------------------
# Promises that hold on every install, with or without the extra
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cls", WRAPPERS)
def test_constructs_without_the_extra(cls):
    """Construction records hyperparameters and must never need the backend."""
    model = cls()
    assert model.model_ is None
    assert model.series_ is None


@pytest.mark.parametrize("cls", WRAPPERS)
def test_schema_covers_every_constructor_parameter(cls):
    """A parameter missing from the schema is invisible to agents and the CLI."""
    params = set(inspect.signature(cls.__init__).parameters) - {"self"}
    assert params == set(cls.get_parameter_schema())


@pytest.mark.parametrize("cls", WRAPPERS)
def test_schema_defaults_match_the_signature(cls):
    """A schema default that disagrees with the signature misinforms callers."""
    sig = inspect.signature(cls.__init__).parameters
    for name, spec in cls.get_parameter_schema().items():
        assert spec["default"] == sig[name].default, name


@pytest.mark.parametrize("cls", WRAPPERS)
def test_get_params_round_trips_through_pickle(cls):
    """Fitted-state slots carry a trailing underscore so they stay out of params."""
    model = cls()
    params = set(inspect.signature(cls.__init__).parameters) - {"self"}
    assert set(model.get_params()) == params
    clone = pickle.loads(pickle.dumps(model))
    assert clone.get_params() == model.get_params()


@pytest.mark.parametrize("cls", WRAPPERS)
def test_capabilities_are_known_strings(cls):
    """A typo'd capability silently exempts a model from checks keyed on it."""
    caps = set(cls.get_capabilities())
    assert caps <= KNOWN_CAPABILITIES
    assert {"timeseries", "forecasting", "univariate"} <= caps


@pytest.mark.parametrize("cls, key", [
    (ChronosForecaster, "foundation.ChronosForecaster"),
    (TimesFMForecaster, "foundation.TimesFMForecaster"),
])
def test_registered_under_the_foundation_namespace(cls, key):
    """The namespace marks these as delegating to an upstream artifact.

    It also keeps them out of the generic contract sweep, which skips any
    dotted name -- important here, because that sweep fits every algorithm it
    finds and these would each pull a checkpoint over the network.
    """
    assert registry.get(key) is cls
    names = {info["name"] for info in tuiml.list_algorithms()}
    assert key in names
    assert cls.__name__ not in names, "bare name would collide and be swept"


@pytest.mark.parametrize("cls", WRAPPERS)
def test_fit_without_the_extra_explains_how_to_install(cls, monkeypatch):
    """The missing-dependency error must name the package and the command.

    Blocking happens on ``sys.meta_path`` because the wrapper reaches the
    package through :func:`importlib.import_module`; cached submodules are
    evicted so the import cannot succeed from ``sys.modules``.
    """
    upstream = UPSTREAM[cls]

    class Blocker(MetaPathFinder):
        def find_spec(self, name, path=None, target=None):
            if name == upstream or name.startswith(upstream + "."):
                raise ImportError(f"No module named {name!r}")
            return None

    for cached in [m for m in sys.modules
                   if m == upstream or m.startswith(upstream + ".")]:
        monkeypatch.delitem(sys.modules, cached, raising=False)
    monkeypatch.setattr(sys, "meta_path", [Blocker(), *sys.meta_path])

    with pytest.raises(ImportError) as excinfo:
        cls().fit(_series())
    message = str(excinfo.value)
    assert upstream in message
    assert "pip install 'tuiml[foundation]'" in message
    assert cls.__name__ in message


@pytest.mark.parametrize("cls", WRAPPERS)
def test_predict_before_fit_raises(cls):
    """``predict`` must check fitted state before touching ``model_``."""
    with pytest.raises(Exception):
        cls().predict(3)
    with pytest.raises(Exception):
        cls().predict_quantiles(3)


@pytest.mark.parametrize("cls", WRAPPERS)
@pytest.mark.parametrize("bad", [np.array([1.0]), np.array([1.0, np.nan, 2.0])])
def test_fit_validates_the_series_before_loading_anything(cls, bad, monkeypatch):
    """A bad series must fail on its own merits, never by reaching the backend."""
    def boom(self):
        raise AssertionError("backend was loaded")

    monkeypatch.setattr(cls, "_load_model", boom)
    with pytest.raises(ValueError):
        cls().fit(bad)


def test_chronos_rejects_quantile_levels_outside_the_unit_interval(monkeypatch):
    """Levels are validated at fit, which is the first place they are needed."""
    monkeypatch.setattr(ChronosForecaster, "_load_model", lambda self: object())
    with pytest.raises(ValueError):
        ChronosForecaster(quantile_levels=(0.5, 1.5)).fit(_series())


@pytest.mark.parametrize("cls", WRAPPERS)
def test_median_and_sorted_quantiles_come_from_the_hook(cls, monkeypatch):
    """The shared base must sort quantiles and report the one nearest 0.5.

    The upstream forward pass is replaced by a hook returning deliberately
    crossing quantiles, so this exercises only TuiML's own code.
    """
    monkeypatch.setattr(cls, "_load_model", lambda self: object())

    def fake(self, context, steps):
        rng = np.random.default_rng(0)
        return rng.normal(size=(steps, len(self.quantile_levels_)))

    monkeypatch.setattr(cls, "_forecast_quantiles", fake)
    model = cls().fit(_series())
    q = model.predict_quantiles(steps=4)
    assert q.shape == (4, len(model.quantile_levels_))
    assert (np.diff(q, axis=1) >= 0).all()
    j = int(np.argmin(np.abs(model.quantile_levels_ - 0.5)))
    np.testing.assert_array_equal(model.predict(steps=4), q[:, j])
    assert model.n_obs_ == 200


@pytest.mark.parametrize("cls", WRAPPERS)
def test_pickle_drops_the_checkpoint_and_reloads_it_on_demand(cls, monkeypatch):
    """A saved model must hold the context, never the upstream weights.

    The loader is counted: once at fit, not at pickle, and once more on the
    first forecast after unpickling. The stand-in model is a local closure,
    which is exactly what breaks pickling in the real TimesFM.
    """
    loads = []

    def loader(self):
        def unpicklable():  # a local function, like TimesFM's compiled decode
            return None
        loads.append(1)
        if isinstance(self, TimesFMForecaster):
            self.compiled_horizon_ = 128
        return unpicklable

    monkeypatch.setattr(cls, "_load_model", loader)
    monkeypatch.setattr(cls, "_forecast_quantiles",
                        lambda self, context, steps: np.zeros((steps, 9)))
    model = cls().fit(_series())
    assert len(loads) == 1

    blob = pickle.dumps(model)
    assert len(blob) < 50_000
    clone = pickle.loads(blob)
    assert clone.model_ is None
    assert model.model_ is not None, "pickling must not unload the live model"
    np.testing.assert_array_equal(clone.series_, model.series_)
    assert clone.predict(steps=3).shape == (3,)
    assert len(loads) == 2
    assert clone.model_ is not None


def test_context_length_keeps_only_the_tail(monkeypatch):
    """``context_length`` is the window the model sees, not a validation knob."""
    monkeypatch.setattr(ChronosForecaster, "_load_model", lambda self: object())
    model = ChronosForecaster(context_length=50).fit(_series(200))
    assert model.series_.shape == (50,)
    assert model.series_.dtype == np.float32
    np.testing.assert_allclose(model.series_, _series(200)[-50:].astype(np.float32))


# ---------------------------------------------------------------------------
# Behaviour, only where the extra is installed
# ---------------------------------------------------------------------------

@needs_chronos
def test_chronos_forecasts_a_trend_zero_shot():
    """A clean linear-plus-seasonal series must be continued sensibly."""
    y = _series()
    model = ChronosForecaster().fit(y)
    pred = model.predict(steps=12)
    assert pred.shape == (12,)
    assert np.isfinite(pred).all()
    assert abs(pred.mean() - y[-24:].mean()) < 3.0
    q = model.predict_quantiles(steps=12)
    assert q.shape == (12, 9)
    assert (q[:, 0] <= q[:, 4]).all() and (q[:, 4] <= q[:, 8]).all()


@needs_timesfm
def test_timesfm_forecasts_a_trend_zero_shot_and_recompiles_for_long_horizons():
    """Same promise for TimesFM, plus the recompile path beyond max_horizon."""
    y = _series()
    model = TimesFMForecaster(context_length=128, max_horizon=128).fit(y)
    pred = model.predict(steps=12)
    assert pred.shape == (12,)
    assert np.isfinite(pred).all()
    assert abs(pred.mean() - y[-24:].mean()) < 3.0
    assert model.compiled_horizon_ == 128
    assert model.predict(steps=200).shape == (200,)
    assert model.compiled_horizon_ == 256
