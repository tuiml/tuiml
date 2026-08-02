"""Fixture data shaped to what each component contract expects.

A single ``(X, y)`` pair cannot drive the whole catalog: clusterers take no
labels, timeseries models want one series rather than a design matrix,
samplers need a skewed target to resample, and vectorizers need text. Routing
the data here keeps every ``check_*`` free of per-component special cases.

Everything is generated from a fixed seed, because the checks compare runs
against each other.
"""

from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

#: Capability strings the library uses. A capability outside this set is almost
#: always a typo, and a typo silently disables the checks that key off it --
#: ``missing_values`` spelled ``missing-values`` would exempt a component from
#: the missing-data check with nobody noticing.
KNOWN_CAPABILITIES = frozenset({
    # data types accepted
    "numeric", "nominal", "categorical", "sparse", "text", "date", "string",
    "missing_values", "empty_nominal", "high_dimensional",
    # target / task
    "binary_class", "multiclass", "numeric_class", "regression", "clustering",
    "unsupervised", "anomaly_detection", "novelty_detection",
    "noise_detection", "uncertainty", "probabilistic",
    # timeseries
    "timeseries", "forecasting", "univariate", "multivariate", "stationary",
    "seasonal", "seasonality", "trend", "holidays",
    # learning style / assumptions
    "online", "incremental", "ensemble", "weighted_instances",
    "gaussian_assumption", "kernel_methods",
})

SEED = 0
N_SAMPLES = 60
N_FEATURES = 4

#: Capability strings the library uses. A capability outside this set is almost
#: always a typo, and a typo silently disables the checks that key off it --
#: ``missing_values`` spelled ``missing-values`` would exempt an algorithm from
#: the missing-data check with nobody noticing.
KNOWN_CAPABILITIES = frozenset({
    # data types accepted
    "numeric", "nominal", "categorical", "sparse", "text", "date", "string",
    "missing_values", "empty_nominal", "high_dimensional",
    # target / task
    "binary_class", "multiclass", "numeric_class", "regression", "clustering",
    "unsupervised", "anomaly_detection", "novelty_detection",
    "noise_detection", "uncertainty", "probabilistic",
    # timeseries
    "timeseries", "forecasting", "univariate", "multivariate", "stationary",
    "seasonal", "seasonality", "trend", "holidays",
    # learning style / assumptions
    "online", "incremental", "ensemble", "weighted_instances",
    "gaussian_assumption", "kernel_methods",
})


# ---------------------------------------------------------------------------
# Fixture data
# ---------------------------------------------------------------------------

def algorithm_kind(kind: str, capabilities: Sequence[str]) -> str:
    """Classify an algorithm by the data contract it obeys.

    The registry type alone is insufficient: timeseries models register as
    regressors but reject a 2-D design matrix, so they need separating before
    any data is generated.

    Parameters
    ----------
    kind : str
        The registry component type, e.g. ``"classifier"``.
    capabilities : sequence of str
        Declared capabilities from ``get_capabilities()``.

    Returns
    -------
    kind : str
        One of ``"classifier"``, ``"regressor"``, ``"clusterer"``,
        ``"timeseries"``.
    """
    if "timeseries" in capabilities or "forecasting" in capabilities:
        return "timeseries"
    return kind


def make_data(kind: str, capabilities: Sequence[str] = (),
              missing: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Build a small fitting problem for one algorithm kind.

    Parameters
    ----------
    kind : str
        Result of :func:`algorithm_kind`.
    capabilities : sequence of str, default=()
        Declared capabilities; keeps targets inside what the algorithm claims
        to support, so a binary-only model is not handed three classes.
    missing : bool, default=False
        Punch ``NaN`` into roughly 20% of ``X``, for the check that holds an
        algorithm to its declared ``missing_values`` support.

    Returns
    -------
    X : np.ndarray
        Feature matrix, or a 1-D series when ``kind`` is ``"timeseries"``.
    y : np.ndarray or None
        Targets; ``None`` for clusterers and timeseries, which fit without.
    """
    rng = np.random.default_rng(SEED)

    if kind == "timeseries":
        t = np.arange(N_SAMPLES, dtype=float)
        return 10.0 + 0.3 * t + 2.0 * np.sin(t / 3.0) + rng.normal(0, 0.2, N_SAMPLES), None

    if "nominal" in capabilities and "numeric" not in capabilities:
        # Nominal-only models (categorical Naive Bayes and friends) index into
        # per-category tables, so a standard normal is not merely unhelpful --
        # it raises. Give them small non-negative integer codes instead.
        X = rng.integers(0, 3, size=(N_SAMPLES, N_FEATURES)).astype(float)
    else:
        X = rng.normal(size=(N_SAMPLES, N_FEATURES))

    if missing:
        # Leave column 0 intact so every row keeps a usable value and an
        # algorithm that drops incomplete rows still has something to fit.
        holes = rng.random(X.shape) < 0.20
        holes[:, 0] = False
        X = X.copy()
        X[holes] = np.nan

    if kind == "clusterer":
        return X, None

    if kind == "regressor":
        return X, X[:, 0] * 2.0 - X[:, 1] + rng.normal(0, 0.1, N_SAMPLES)

    n_classes = 3 if "multiclass" in capabilities else 2
    y = np.zeros(N_SAMPLES, dtype=int)
    for c in range(1, n_classes):
        y[c::n_classes] = c
    return X, y


# ---------------------------------------------------------------------------
# Transformers, samplers, text and splitters
# ---------------------------------------------------------------------------

#: Text transformers that consume a count matrix rather than raw documents.
#: They live in the text package but sit downstream of a vectorizer.
_MATRIX_TEXT_TRANSFORMERS = {"TfidfTransformer"}


def transformer_kind(cls) -> str:
    """Classify a transformer by the input it accepts.

    Parameters
    ----------
    cls : type
        The transformer class.

    Returns
    -------
    kind : str
        ``"sampler"``, ``"supervised"``, ``"text"``, ``"timeseries"`` or
        ``"numeric"``.
    """
    import inspect

    module = getattr(cls, "__module__", "")
    if hasattr(cls, "fit_resample"):
        return "sampler"
    if cls.__name__ in _MATRIX_TEXT_TRANSFORMERS:
        return "numeric"
    if ".text" in module:
        return "text"
    if "timeseries" in module:
        return "timeseries"
    # Supervised transformers (MDL discretisation and friends) declare y as a
    # required positional, so they need labels like a classifier does.
    try:
        params = list(inspect.signature(cls.fit).parameters.values())[1:]
    except (TypeError, ValueError):
        return "numeric"
    if any(p.name == "y" and p.default is inspect.Parameter.empty
           and p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) for p in params):
        return "supervised"
    return "numeric"


def make_transformer_data(kind: str):
    """Build an input suited to one transformer kind.

    Parameters
    ----------
    kind : str
        Result of :func:`transformer_kind`.

    Returns
    -------
    X : np.ndarray or list of str
        Input matrix, or a list of documents for text transformers.
    y : np.ndarray or None
        Targets; only samplers need them, and theirs is deliberately
        imbalanced so there is something to resample.
    """
    rng = np.random.default_rng(SEED)

    if kind == "text":
        vocab = ["alpha beta gamma", "beta delta", "gamma gamma epsilon",
                 "delta epsilon zeta", "alpha zeta", "beta gamma delta"]
        return [vocab[i % len(vocab)] for i in range(N_SAMPLES)], None

    if kind == "timeseries":
        t = np.arange(N_SAMPLES, dtype=float)
        return (10.0 + 0.3 * t + rng.normal(0, 0.2, N_SAMPLES)).reshape(-1, 1), None

    X = rng.normal(size=(N_SAMPLES, N_FEATURES))

    if kind == "sampler":
        # 80/20 split: enough minority rows for a k-neighbour sampler to
        # interpolate between, skewed enough that resampling does something.
        y = (rng.random(N_SAMPLES) < 0.2).astype(int)
        if y.sum() < 6:
            y[:6] = 1
        return X, y

    if kind == "supervised":
        return X, np.tile([0, 1], N_SAMPLES // 2)

    return X, None


def make_split_data():
    """Build a labelled, grouped problem for cross-validation splitters.

    Returns
    -------
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Balanced binary targets, so stratified splitters have both classes
        available in every fold.
    groups : np.ndarray
        Group labels for the group-aware splitters, which raise without them.
        Six groups, so a default 5-fold split has something to work with.
    """
    rng = np.random.default_rng(SEED)
    X = rng.normal(size=(N_SAMPLES, N_FEATURES))
    y = np.tile([0, 1], N_SAMPLES // 2)
    groups = np.arange(N_SAMPLES) % 6
    return X, y, groups


def make_transactions():
    """Build a transaction matrix for association-rule miners.

    Returns
    -------
    X : np.ndarray
        Binary item-presence matrix with a planted frequent pattern, so
        support and confidence have something to find.
    """
    rng = np.random.default_rng(SEED)
    X = (rng.random((N_SAMPLES, 6)) < 0.3).astype(int)
    # Plant {0,1} -> {2} in most rows so any correct miner reports a rule.
    X[:40, 0] = 1
    X[:40, 1] = 1
    X[:36, 2] = 1
    return X
