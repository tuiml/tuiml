"""Contract checks every TuiML algorithm is expected to satisfy.

Each ``check_*`` function asserts one invariant against one algorithm. They are
run over the whole registry by ``test_common.py``, so coverage follows the
catalog rather than author patience: registering an algorithm subscribes it to
the whole battery, and a check added here applies to every algorithm at once.

This replaces the six invariants that used to be copy-pasted into ~80 per-
algorithm modules -- 418 of 709 algorithm tests, ~2,950 lines -- while leaving
22 registered algorithms untested. Algorithm-*specific* behaviour still belongs
in the per-algorithm modules; this file only covers what every algorithm owes
its callers.

Checks take ``(name, algorithm)`` where ``algorithm`` is a freshly constructed
instance, mirroring scikit-learn's ``check_estimator`` shape. They raise
``AssertionError`` on failure and return ``None`` otherwise, so they also work
outside pytest via :func:`check_algorithm`.
"""

from __future__ import annotations

import inspect
import pickle
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ._data import KNOWN_CAPABILITIES, algorithm_kind, make_data

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _capabilities(algorithm) -> List[str]:
    """Return declared capabilities, tolerating an algorithm that has none.

    Parameters
    ----------
    algorithm : object
        Algorithm instance or class.

    Returns
    -------
    capabilities : list of str
        Declared capabilities; empty when unavailable.
    """
    try:
        caps = type(algorithm).get_capabilities()
    except Exception:
        return []
    return list(caps) if caps else []


def _kind_of(algorithm) -> str:
    """Return the data contract an algorithm obeys.

    Parameters
    ----------
    algorithm : object
        Algorithm instance.

    Returns
    -------
    kind : str
        Result of :func:`algorithm_kind`.
    """
    from tuiml.base.algorithms import Classifier, Clusterer, Regressor

    if isinstance(algorithm, Classifier):
        base = "classifier"
    elif isinstance(algorithm, Regressor):
        base = "regressor"
    elif isinstance(algorithm, Clusterer):
        base = "clusterer"
    else:
        base = "classifier"
    return algorithm_kind(base, _capabilities(algorithm))


def _fit(algorithm, X, y):
    """Fit an algorithm, passing ``y`` only when its kind uses one.

    Parameters
    ----------
    algorithm : object
        Instance to fit.
    X : np.ndarray
        Feature matrix or series.
    y : np.ndarray or None
        Targets, or None for unsupervised kinds.

    Returns
    -------
    fitted : object
        Whatever ``fit`` returned.
    """
    return algorithm.fit(X) if y is None else algorithm.fit(X, y)


# ---------------------------------------------------------------------------
# Declaration
# ---------------------------------------------------------------------------

def check_schema_matches_signature(name: str, algorithm) -> None:
    """Every constructor parameter is declared in ``get_parameter_schema``.

    The schema is not documentation: it is what ``tuiml_describe`` shows an
    agent and what the MCP layer validates against. A parameter missing from
    it is a parameter no agent can discover -- which is how a number of
    algorithms ended up with an undiscoverable ``random_state``, putting
    reproducibility out of reach through the agent path.

    Parameters
    ----------
    name : str
        Registry name, used in failure messages.
    algorithm : object
        A constructed instance.

    Returns
    -------
    None
    """
    cls = type(algorithm)
    schema = cls.get_parameter_schema()
    assert isinstance(schema, dict), f"{name}: get_parameter_schema must return a dict"

    accepted = {
        p.name for p in inspect.signature(cls.__init__).parameters.values()
        if p.name != "self" and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
    }
    undeclared = accepted - set(schema)
    assert not undeclared, (
        f"{name}: __init__ accepts {sorted(undeclared)} but get_parameter_schema "
        f"does not declare them, so agents cannot discover or set them"
    )
    phantom = set(schema) - accepted
    assert not phantom, (
        f"{name}: get_parameter_schema declares {sorted(phantom)} which __init__ "
        f"does not accept"
    )


def check_capabilities_are_known(name: str, algorithm) -> None:
    """Declared capabilities are a non-empty subset of the known vocabulary.

    Parameters
    ----------
    name : str
        Registry name, used in failure messages.
    algorithm : object
        A constructed instance.

    Returns
    -------
    None
    """
    caps = type(algorithm).get_capabilities()
    assert isinstance(caps, list), f"{name}: get_capabilities must return a list"
    assert caps, f"{name}: declares no capabilities"
    assert all(isinstance(c, str) for c in caps), f"{name}: capabilities must be strings"
    unknown = set(caps) - KNOWN_CAPABILITIES
    assert not unknown, (
        f"{name}: unrecognised capabilities {sorted(unknown)}. A typo here "
        f"silently exempts the algorithm from capability-driven checks"
    )


def check_params_roundtrip(name: str, algorithm) -> None:
    """``set_params(**get_params())`` leaves the algorithm unchanged.

    Parameters
    ----------
    name : str
        Registry name, used in failure messages.
    algorithm : object
        A constructed instance.

    Returns
    -------
    None
    """
    before = algorithm.get_params()
    assert isinstance(before, dict), f"{name}: get_params must return a dict"
    algorithm.set_params(**before)
    after = algorithm.get_params()

    assert set(before) == set(after), (
        f"{name}: set_params changed which parameters exist "
        f"({sorted(set(before) ^ set(after))})"
    )
    for key, value in before.items():
        if isinstance(value, np.ndarray):
            continue
        assert after[key] is value or after[key] == value, (
            f"{name}: parameter {key!r} changed across a get/set roundtrip "
            f"({value!r} -> {after[key]!r})"
        )


# ---------------------------------------------------------------------------
# Fitting
# ---------------------------------------------------------------------------

def check_predict_before_fit_raises(name: str, algorithm) -> None:
    """Predicting on an unfitted algorithm raises rather than returning junk.

    Parameters
    ----------
    name : str
        Registry name, used in failure messages.
    algorithm : object
        A constructed instance.

    Returns
    -------
    None
    """
    X, _ = make_data(_kind_of(algorithm), _capabilities(algorithm))
    try:
        algorithm.predict(X)
    except Exception as exc:  # noqa: BLE001 - any refusal is acceptable, but see below
        # It must be a deliberate refusal, not an incidental crash deep inside
        # the predict path. The per-algorithm tests this check replaces
        # asserted RuntimeError(match="must be fitted"); keep that intent
        # without pinning every algorithm to one exception type.
        message = str(exc).lower()
        assert "fit" in message, (
            f"{name}: predict() before fit raised "
            f"{type(exc).__name__}: {exc} -- which reads as an incidental "
            f"crash rather than a deliberate 'not fitted' error"
        )
        return
    raise AssertionError(
        f"{name}: predict() on an unfitted algorithm returned instead of raising"
    )


def check_fit_returns_self(name: str, algorithm) -> None:
    """``fit`` returns the instance, so ``fit(...).predict(...)`` chains.

    Parameters
    ----------
    name : str
        Registry name, used in failure messages.
    algorithm : object
        A constructed instance.

    Returns
    -------
    None
    """
    X, y = make_data(_kind_of(algorithm), _capabilities(algorithm))
    returned = _fit(algorithm, X, y)
    assert returned is algorithm, (
        f"{name}: fit() returned {type(returned).__name__} rather than self, so "
        f"fit(...).predict(...) does not chain"
    )


def check_fit_does_not_mutate_input(name: str, algorithm) -> None:
    """Fitting leaves the caller's arrays untouched.

    Parameters
    ----------
    name : str
        Registry name, used in failure messages.
    algorithm : object
        A constructed instance.

    Returns
    -------
    None
    """
    X, y = make_data(_kind_of(algorithm), _capabilities(algorithm))
    X_before = X.copy()
    y_before = None if y is None else y.copy()

    _fit(algorithm, X, y)

    assert np.array_equal(X, X_before, equal_nan=True), f"{name}: fit() modified X in place"
    if y is not None:
        assert np.array_equal(y, y_before, equal_nan=True), f"{name}: fit() modified y in place"


def check_predict_output_shape(name: str, algorithm) -> None:
    """``predict`` returns one finite value per input row.

    Parameters
    ----------
    name : str
        Registry name, used in failure messages.
    algorithm : object
        A constructed instance.

    Returns
    -------
    None
    """
    kind = _kind_of(algorithm)
    if kind == "timeseries":
        return  # forecast horizons are not row-aligned
    X, y = make_data(kind, _capabilities(algorithm))
    _fit(algorithm, X, y)
    pred = np.asarray(algorithm.predict(X))

    assert pred.shape[0] == X.shape[0], (
        f"{name}: predict() returned {pred.shape[0]} values for {X.shape[0]} rows"
    )
    if pred.dtype.kind == "f":
        assert np.isfinite(pred).all(), f"{name}: predict() returned non-finite values"


def check_predict_proba_is_a_distribution(name: str, algorithm) -> None:
    """Classifier probabilities are non-negative and sum to one per row.

    Parameters
    ----------
    name : str
        Registry name, used in failure messages.
    algorithm : object
        A constructed instance.

    Returns
    -------
    None
    """
    if _kind_of(algorithm) != "classifier" or not hasattr(algorithm, "predict_proba"):
        return
    X, y = make_data("classifier", _capabilities(algorithm))
    _fit(algorithm, X, y)
    try:
        proba = np.asarray(algorithm.predict_proba(X), dtype=float)
    except NotImplementedError:
        return

    assert proba.shape[0] == X.shape[0], (
        f"{name}: predict_proba returned {proba.shape[0]} rows for {X.shape[0]} inputs"
    )
    assert (proba >= -1e-9).all(), f"{name}: predict_proba returned negative probabilities"
    sums = proba.sum(axis=1)
    assert np.allclose(sums, 1.0, atol=1e-6), (
        f"{name}: predict_proba rows sum to {sums.min():.4f}..{sums.max():.4f}, not 1"
    )


# ---------------------------------------------------------------------------
# Reproducibility and persistence
# ---------------------------------------------------------------------------

def check_seeded_fit_is_reproducible(name: str, algorithm) -> None:
    """Two fits at the same ``random_state`` predict identically.

    Applies only to algorithms that accept a seed. Without it a stochastic
    model can drift between runs while every other check still passes, and a
    benchmark table ends up measuring the seed rather than the change.

    Parameters
    ----------
    name : str
        Registry name, used in failure messages.
    algorithm : object
        A constructed instance.

    Returns
    -------
    None
    """
    cls = type(algorithm)
    if "random_state" not in inspect.signature(cls.__init__).parameters:
        return
    kind = _kind_of(algorithm)
    if kind == "timeseries":
        return
    X, y = make_data(kind, _capabilities(algorithm))

    first = np.asarray(_fit(cls(random_state=0), X, y).predict(X))
    second = np.asarray(_fit(cls(random_state=0), X, y).predict(X))
    assert np.array_equal(first, second, equal_nan=True), (
        f"{name}: two fits at random_state=0 produced different predictions"
    )


def check_pickle_roundtrip(name: str, algorithm) -> None:
    """A fitted algorithm survives pickling with its predictions intact.

    Parameters
    ----------
    name : str
        Registry name, used in failure messages.
    algorithm : object
        A constructed instance.

    Returns
    -------
    None
    """
    kind = _kind_of(algorithm)
    X, y = make_data(kind, _capabilities(algorithm))
    _fit(algorithm, X, y)

    restored = pickle.loads(pickle.dumps(algorithm))
    if kind == "timeseries":
        return
    before = np.asarray(algorithm.predict(X))
    after = np.asarray(restored.predict(X))
    assert np.array_equal(before, after, equal_nan=True), (
        f"{name}: predictions changed after a pickle roundtrip"
    )


# ---------------------------------------------------------------------------
# Capability honesty
# ---------------------------------------------------------------------------

def check_missing_value_support_is_honest(name: str, algorithm) -> None:
    """An algorithm declaring ``missing_values`` actually fits data with NaN.

    Declared support that does not hold is worse than none: the workflow layer
    routes on this capability, so a false declaration sends incomplete data to
    an algorithm that cannot take it. The tree ``RecursionError`` fixed in
    0.1.7 was exactly this shape -- the trees advertised ``missing_values``
    and hung on it.

    Parameters
    ----------
    name : str
        Registry name, used in failure messages.
    algorithm : object
        A constructed instance.

    Returns
    -------
    None
    """
    caps = _capabilities(algorithm)
    if "missing_values" not in caps:
        return
    kind = _kind_of(algorithm)
    if kind == "timeseries":
        return

    X, y = make_data(kind, caps, missing=True)
    assert np.isnan(X).any(), "fixture no longer contains missing values"

    try:
        _fit(algorithm, X, y)
        pred = np.asarray(algorithm.predict(X))
    except Exception as exc:  # noqa: BLE001 - the failure is the finding
        raise AssertionError(
            f"{name}: declares 'missing_values' but fitting on data with NaN "
            f"raised {type(exc).__name__}: {exc}"
        ) from exc

    assert pred.shape[0] == X.shape[0], (
        f"{name}: declares 'missing_values' but predicted {pred.shape[0]} of "
        f"{X.shape[0]} rows"
    )


#: Every check, in run order. Registration point: a check added here applies to
#: every algorithm in the registry on the next test run.
ALL_CHECKS: Tuple[Callable[[str, Any], None], ...] = (
    check_schema_matches_signature,
    check_capabilities_are_known,
    check_params_roundtrip,
    check_predict_before_fit_raises,
    check_fit_returns_self,
    check_fit_does_not_mutate_input,
    check_predict_output_shape,
    check_predict_proba_is_a_distribution,
    check_seeded_fit_is_reproducible,
    check_pickle_roundtrip,
    check_missing_value_support_is_honest,
)


def check_algorithm(algorithm, name: Optional[str] = None) -> Dict[str, Optional[str]]:
    """Run every contract check against one algorithm, without pytest.

    Useful for triaging the whole catalog at once, and for validating an
    algorithm that is not in the registry yet.

    Parameters
    ----------
    algorithm : type or object
        The algorithm class, or an already-constructed instance.
    name : str, default=None
        Name to use in messages; defaults to the class name.

    Returns
    -------
    results : dict
        Mapping of check name to ``None`` when it passed, or the failure
        message when it did not.
    """
    factory = algorithm if inspect.isclass(algorithm) else type(algorithm)
    label = name or factory.__name__

    results: Dict[str, Optional[str]] = {}
    for check in ALL_CHECKS:
        try:
            check(label, factory())
        except AssertionError as exc:
            results[check.__name__] = str(exc)
        except Exception as exc:  # noqa: BLE001 - report, do not abort the sweep
            results[check.__name__] = f"{type(exc).__name__}: {exc}"
        else:
            results[check.__name__] = None
    return results
