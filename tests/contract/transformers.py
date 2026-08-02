"""Contract checks for preprocessing and feature transformers.

Transformers obey a different contract from algorithms: they produce a matrix
rather than a prediction, they are composed inside workflows where a shape or
row-count surprise silently corrupts everything downstream, and several
families (samplers, vectorizers) accept inputs the others reject. The kind
router in :mod:`tests.contract._data` handles that, so the checks below stay
free of per-transformer special cases.
"""

from __future__ import annotations

import inspect
import pickle
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

from ._data import make_transformer_data, transformer_kind

#: Transformers that are stateless by design: ``transform`` derives everything
#: from its input, so calling it before ``fit`` is legal rather than a bug.
#: HashingVectorizer is the canonical case -- the hashing trick exists
#: precisely so no vocabulary has to be learned.
STATELESS = {
    "HashingVectorizer", "Stemmer", "StopWordRemover", "TextCleaner",
}

#: Transformers that deliberately change the row count inside ``transform``
#: rather than through ``fit_resample``. Row preservation and output-equality
#: checks do not apply to them; that they resize is the point.
RESIZING = {
    "ClassBalanceSampler", "ReservoirSampler",
}


def _fit(transformer, X, y):
    """Fit a transformer, passing ``y`` only when its kind uses one.

    Parameters
    ----------
    transformer : object
        Instance to fit.
    X : array-like
        Input matrix or documents.
    y : np.ndarray or None
        Targets, or None.

    Returns
    -------
    fitted : object
        Whatever ``fit`` returned.
    """
    return transformer.fit(X) if y is None else transformer.fit(X, y)


def check_schema_matches_signature(name: str, transformer) -> None:
    """Every constructor parameter is declared in ``get_parameter_schema``.

    Same reasoning as the algorithm check: the schema is what an agent sees,
    so an omission is a parameter nobody can discover.

    Parameters
    ----------
    name : str
        Registry name, used in failure messages.
    transformer : object
        A constructed instance.

    Returns
    -------
    None
    """
    cls = type(transformer)
    if not hasattr(cls, "get_parameter_schema"):
        return
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


def check_transform_before_fit_raises(name: str, transformer) -> None:
    """Transforming before fitting raises rather than returning junk.

    Parameters
    ----------
    name : str
        Registry name, used in failure messages.
    transformer : object
        A constructed instance.

    Returns
    -------
    None
    """
    if name in STATELESS:
        return
    kind = transformer_kind(type(transformer))
    X, _ = make_transformer_data(kind)
    try:
        transformer.transform(X)
    except Exception:
        return
    raise AssertionError(
        f"{name}: transform() before fit() returned instead of raising"
    )


def check_fit_returns_self(name: str, transformer) -> None:
    """``fit`` returns the instance, so ``fit(...).transform(...)`` chains.

    Parameters
    ----------
    name : str
        Registry name, used in failure messages.
    transformer : object
        A constructed instance.

    Returns
    -------
    None
    """
    kind = transformer_kind(type(transformer))
    X, y = make_transformer_data(kind)
    returned = _fit(transformer, X, y)
    assert returned is transformer, (
        f"{name}: fit() returned {type(returned).__name__} rather than self"
    )


def check_fit_does_not_mutate_input(name: str, transformer) -> None:
    """Fitting leaves the caller's array untouched.

    Parameters
    ----------
    name : str
        Registry name, used in failure messages.
    transformer : object
        A constructed instance.

    Returns
    -------
    None
    """
    kind = transformer_kind(type(transformer))
    X, y = make_transformer_data(kind)
    if not isinstance(X, np.ndarray):
        return
    before = X.copy()
    _fit(transformer, X, y)
    assert np.array_equal(X, before, equal_nan=True), f"{name}: fit() modified X in place"


def check_transform_preserves_row_count(name: str, transformer) -> None:
    """``transform`` returns one row per input row.

    Samplers are exempt by construction -- changing the row count is their
    whole purpose -- and are covered by
    :func:`check_resample_keeps_X_and_y_aligned` instead. So are the
    :data:`RESIZING` transformers, which subsample inside ``transform``.

    Parameters
    ----------
    name : str
        Registry name, used in failure messages.
    transformer : object
        A constructed instance.

    Returns
    -------
    None
    """
    kind = transformer_kind(type(transformer))
    if kind == "sampler" or name in RESIZING:
        return
    X, y = make_transformer_data(kind)
    _fit(transformer, X, y)
    out = transformer.transform(X)

    n_in = len(X)
    n_out = out.shape[0] if hasattr(out, "shape") else len(out)
    assert n_out == n_in, (
        f"{name}: transform() returned {n_out} rows for {n_in} inputs; a "
        f"row-count change silently misaligns X from y downstream"
    )


def check_fit_transform_matches_fit_then_transform(name: str, transformer) -> None:
    """``fit_transform`` agrees with ``fit`` followed by ``transform``.

    A divergence here is how a workflow ends up training on one
    representation and predicting on another.

    Parameters
    ----------
    name : str
        Registry name, used in failure messages.
    transformer : object
        A constructed instance.

    Returns
    -------
    None
    """
    cls = type(transformer)
    kind = transformer_kind(cls)
    if kind == "sampler" or name in RESIZING or not hasattr(transformer, "fit_transform"):
        return

    X, y = make_transformer_data(kind)
    combined = cls().fit_transform(X) if y is None else cls().fit_transform(X, y)
    stepwise = _fit(cls(), X, y).transform(X)

    combined = np.asarray(combined.todense() if hasattr(combined, "todense") else combined)
    stepwise = np.asarray(stepwise.todense() if hasattr(stepwise, "todense") else stepwise)

    assert combined.shape == stepwise.shape, (
        f"{name}: fit_transform gave shape {combined.shape}, fit+transform "
        f"gave {stepwise.shape}"
    )
    if combined.dtype.kind in "fiu":
        assert np.allclose(combined, stepwise, equal_nan=True), (
            f"{name}: fit_transform and fit+transform produced different values"
        )


def check_resample_keeps_X_and_y_aligned(name: str, transformer) -> None:
    """A sampler returns X and y of equal length.

    Parameters
    ----------
    name : str
        Registry name, used in failure messages.
    transformer : object
        A constructed instance.

    Returns
    -------
    None
    """
    if not hasattr(transformer, "fit_resample"):
        return
    X, y = make_transformer_data("sampler")
    X_out, y_out = transformer.fit_resample(X, y)
    assert len(X_out) == len(y_out), (
        f"{name}: fit_resample returned {len(X_out)} rows of X but "
        f"{len(y_out)} of y"
    )
    assert len(X_out) > 0, f"{name}: fit_resample returned an empty dataset"


def check_pickle_roundtrip(name: str, transformer) -> None:
    """A fitted transformer survives pickling with its output intact.

    Parameters
    ----------
    name : str
        Registry name, used in failure messages.
    transformer : object
        A constructed instance.

    Returns
    -------
    None
    """
    kind = transformer_kind(type(transformer))
    X, y = make_transformer_data(kind)
    _fit(transformer, X, y)
    restored = pickle.loads(pickle.dumps(transformer))

    if kind == "sampler" or name in RESIZING:
        return
    before = transformer.transform(X)
    after = restored.transform(X)
    before = np.asarray(before.todense() if hasattr(before, "todense") else before)
    after = np.asarray(after.todense() if hasattr(after, "todense") else after)
    assert before.shape == after.shape, f"{name}: shape changed after pickling"
    if before.dtype.kind in "fiu":
        assert np.allclose(before, after, equal_nan=True), (
            f"{name}: transform output changed after a pickle roundtrip"
        )


#: Every transformer check, in run order.
ALL_CHECKS: Tuple[Callable[[str, Any], None], ...] = (
    check_schema_matches_signature,
    check_transform_before_fit_raises,
    check_fit_returns_self,
    check_fit_does_not_mutate_input,
    check_transform_preserves_row_count,
    check_fit_transform_matches_fit_then_transform,
    check_resample_keeps_X_and_y_aligned,
    check_pickle_roundtrip,
)
