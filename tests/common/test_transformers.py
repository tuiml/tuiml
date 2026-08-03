"""Contract tests over every preprocessing and feature transformer.

Discovered by introspecting the public API rather than the component registry:
transformers are exposed as classes on ``tuiml.preprocessing`` and
``tuiml.features``, the same way scikit-learn's ``all_estimators()`` walks its
own modules. Anything exporting ``fit`` and ``transform`` is swept in.
"""

import inspect
import warnings

import pytest

import tuiml.features as features_module
import tuiml.preprocessing as preprocessing_module

from ..contract.transformers import ALL_CHECKS

# Base classes and abstract scaffolding, not transformers in their own right.
SKIP = {
    "Preprocessor", "Filter", "FeatureMethod", "FeatureSelector",
    "FeatureExtractor", "FeatureConstructor", "BaseSplitter",
}

XFAIL_CHECKS: dict = {}


def _discover():
    """Find every concrete transformer on the public API.

    Returns
    -------
    transformers : list of tuple
        ``(name, cls)`` pairs, sorted and de-duplicated.
    """
    found = {}
    for module in (preprocessing_module, features_module):
        for name in dir(module):
            if name.startswith("_") or name in SKIP:
                continue
            obj = getattr(module, name)
            if not inspect.isclass(obj):
                continue
            if not (hasattr(obj, "fit") and hasattr(obj, "transform")):
                continue
            if inspect.isabstract(obj):
                continue
            found.setdefault(name, obj)
    return sorted(found.items())


TRANSFORMERS = _discover()


def _cases():
    """Build the (transformer, check) grid, marking known failures xfail.

    Returns
    -------
    params : list of pytest.param
        One param per (transformer, check) pair.
    """
    params = []
    for name, cls in TRANSFORMERS:
        for check in ALL_CHECKS:
            reason = XFAIL_CHECKS.get(name, {}).get(check.__name__)
            marks = [pytest.mark.xfail(reason=reason, strict=False)] if reason else []
            params.append(
                pytest.param(name, cls, check, id=f"{name}-{check.__name__}", marks=marks)
            )
    return params


def test_transformers_were_discovered():
    """Guard against the sweep silently covering nothing."""
    assert len(TRANSFORMERS) > 30, (
        f"only {len(TRANSFORMERS)} transformers discovered; the API sweep is "
        f"probably wrong"
    )


@pytest.mark.parametrize("name, cls, check", _cases())
def test_transformer_contract(name, cls, check):
    """Every transformer satisfies every contract check."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        check(name, cls())
