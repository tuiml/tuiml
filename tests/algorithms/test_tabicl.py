"""Contract and behaviour tests for the TabICL foundation-model wrappers.

Split deliberately in two. Everything that must hold on *any* install -- the
schema, the capabilities, the namespaced registration, the error raised when
the extra is missing -- runs unconditionally, because those are exactly the
promises that would otherwise only be checked on a developer's machine.

The tests that actually run the model are gated on ``tabicl`` being importable,
and they download a ~150 MB checkpoint on first use.
"""

import inspect
import pickle
import sys
from importlib.abc import MetaPathFinder

import numpy as np
import pytest

import tuiml
from tuiml.registry import registry
from tuiml.algorithms.tabular_foundation import TabICLClassifier, TabICLRegressor
from ..contract._data import KNOWN_CAPABILITIES

WRAPPERS = [TabICLClassifier, TabICLRegressor]

try:  # pragma: no cover - depends on whether the extra is installed
    import tabicl as _tabicl  # noqa: F401

    HAS_TABICL = True
except ImportError:  # pragma: no cover
    HAS_TABICL = False

needs_tabicl = pytest.mark.skipif(
    not HAS_TABICL,
    reason="needs pip install 'tuiml[foundation]' and downloads a checkpoint",
)


# ---------------------------------------------------------------------------
# Promises that hold on every install, with or without the extra
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cls", WRAPPERS)
def test_constructs_without_the_extra(cls):
    """Construction records hyperparameters and must never need the backend."""
    model = cls(n_estimators=3)
    assert model.n_estimators == 3
    assert model.model_ is None


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
def test_capabilities_are_known_strings(cls):
    """A typo'd capability silently exempts a model from checks keyed on it."""
    assert set(cls.get_capabilities()) <= KNOWN_CAPABILITIES


def test_classifier_and_regressor_declare_their_own_task():
    """The two wrappers must not claim each other's task."""
    clf, reg = set(TabICLClassifier.get_capabilities()), set(TabICLRegressor.get_capabilities())
    assert {"binary_class", "multiclass"} <= clf and "regression" not in clf
    assert "regression" in reg and not {"binary_class", "multiclass"} & reg


@pytest.mark.parametrize("cls, key", [
    (TabICLClassifier, "foundation.TabICLClassifier"),
    (TabICLRegressor, "foundation.TabICLRegressor"),
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

    Blocking has to happen on ``sys.meta_path``: the wrapper reaches the
    package through :func:`importlib.import_module`, which resolves against the
    finders directly and never consults ``builtins.__import__``. Cached
    submodules must be evicted too, or the import succeeds from ``sys.modules``
    without a finder ever running.
    """
    class Blocker(MetaPathFinder):
        def find_spec(self, name, path=None, target=None):
            if name == "tabicl" or name.startswith("tabicl."):
                raise ImportError(f"No module named {name!r}")
            return None

    for cached in [m for m in sys.modules if m == "tabicl" or m.startswith("tabicl.")]:
        monkeypatch.delitem(sys.modules, cached, raising=False)
    monkeypatch.setattr(sys, "meta_path", [Blocker(), *sys.meta_path])

    X = np.zeros((6, 2))
    y = np.array([0, 1, 0, 1, 0, 1])
    with pytest.raises(ImportError) as excinfo:
        cls().fit(X, y)
    message = str(excinfo.value)
    assert "tabicl" in message
    assert "pip install 'tuiml[foundation]'" in message
    assert cls.__name__ in message


@pytest.mark.parametrize("cls", WRAPPERS)
def test_predict_before_fit_raises(cls):
    """``predict`` must check fitted state before touching ``model_``."""
    with pytest.raises(Exception):
        cls().predict(np.zeros((3, 2)))


def test_tuiml_ships_no_weights():
    """TuiML must never vendor a checkpoint -- the license story depends on it.

    Weights come from the upstream package's own download. If a ``.ckpt`` or
    ``.safetensors`` ever lands inside the installed package, the distribution
    question changes completely and this test should fail loudly.
    """
    root = __import__("pathlib").Path(tuiml.__file__).parent
    weights = [
        p for pattern in ("*.ckpt", "*.safetensors", "*.pt", "*.pth")
        for p in root.rglob(pattern)
    ]
    assert weights == []


# ---------------------------------------------------------------------------
# Behaviour, gated on the extra being installed
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def xor_data():
    """A target no linear model can fit, so accuracy is evidence of learning."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 4))
    y = (X[:, 0] * X[:, 1] > 0).astype(int)
    Xte = rng.normal(size=(60, 4))
    yte = (Xte[:, 0] * Xte[:, 1] > 0).astype(int)
    return X, y, Xte, yte


@needs_tabicl
def test_classifier_learns_without_any_gradient_step(xor_data):
    """In-context learning must beat chance on a genuinely non-linear target."""
    X, y, Xte, yte = xor_data
    model = TabICLClassifier(n_estimators=2, device="cpu").fit(X, y)
    assert (model.predict(Xte) == yte).mean() > 0.85


@needs_tabicl
def test_classifier_probabilities_are_a_distribution(xor_data):
    """Rows must sum to 1 and be aligned with ``classes_``."""
    X, y, Xte, _ = xor_data
    model = TabICLClassifier(n_estimators=2, device="cpu").fit(X, y)
    proba = model.predict_proba(Xte)
    assert proba.shape == (len(Xte), len(model.classes_))
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert (proba >= 0).all()
    # argmax of the probabilities must agree with the hard prediction
    assert (model.classes_[proba.argmax(axis=1)] == model.predict(Xte)).all()


@needs_tabicl
def test_regressor_recovers_a_linear_signal(xor_data):
    """A clean signal should come back with high R^2."""
    X, _, Xte, _ = xor_data
    y = X[:, 0] * 2.0 - X[:, 1]
    yte = Xte[:, 0] * 2.0 - Xte[:, 1]
    pred = TabICLRegressor(n_estimators=2, device="cpu").fit(X, y).predict(Xte)
    r2 = 1 - ((pred - yte) ** 2).sum() / ((yte - yte.mean()) ** 2).sum()
    assert r2 > 0.9


@needs_tabicl
def test_fit_sets_the_documented_attributes(xor_data):
    """``classes_`` and ``n_features_in_`` are part of the public surface."""
    X, y, _, _ = xor_data
    model = TabICLClassifier(n_estimators=2, device="cpu").fit(X, y)
    assert model.n_features_in_ == X.shape[1]
    assert list(model.classes_) == [0, 1]
    assert TabICLRegressor(n_estimators=2, device="cpu").fit(
        X, y.astype(float)).n_features_in_ == X.shape[1]


@needs_tabicl
def test_same_random_state_gives_identical_predictions(xor_data):
    """The ensemble permutations are seeded, so repeated runs must agree."""
    X, y, Xte, _ = xor_data
    kw = dict(n_estimators=2, device="cpu", random_state=7)
    a = TabICLClassifier(**kw).fit(X, y).predict(Xte)
    b = TabICLClassifier(**kw).fit(X, y).predict(Xte)
    assert np.array_equal(a, b)


@needs_tabicl
def test_pickle_roundtrip_preserves_predictions(xor_data):
    """Serialising a fitted wrapper must not change what it predicts."""
    X, y, Xte, _ = xor_data
    model = TabICLClassifier(n_estimators=2, device="cpu").fit(X, y)
    restored = pickle.loads(pickle.dumps(model))
    assert np.array_equal(restored.predict(Xte), model.predict(Xte))


@needs_tabicl
def test_fit_does_not_mutate_its_input(xor_data):
    """Preprocessing must copy rather than scale the caller's array in place."""
    X, y, _, _ = xor_data
    X_before = X.copy()
    TabICLClassifier(n_estimators=2, device="cpu").fit(X, y)
    assert np.array_equal(X, X_before)
