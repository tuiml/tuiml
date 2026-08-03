"""Contract tests over every association-rule miner.

Associators are absent from ``list_algorithms()`` because their contract has
no ``predict`` -- ``fit`` yields itemsets and rules -- so the algorithm sweep
cannot reach them. Before this module they were the one component family with
no real coverage at all.
"""

import warnings

import pytest

from tuiml.registry import ComponentType, registry

from ..contract.associators import ALL_CHECKS

XFAIL_CHECKS: dict = {}


def _discover():
    """Find every associator in the registry.

    Returns
    -------
    associators : list of tuple
        ``(name, cls)`` pairs, sorted.
    """
    found = []
    for name in registry.list_names(ComponentType.ASSOCIATOR):
        if "." in name:
            continue
        try:
            found.append((name, registry.get(name)))
        except Exception:
            continue
    return sorted(found)


ASSOCIATORS = _discover()


def _cases():
    """Build the (associator, check) grid.

    Returns
    -------
    params : list of pytest.param
        One param per (associator, check) pair.
    """
    params = []
    for name, cls in ASSOCIATORS:
        for check in ALL_CHECKS:
            reason = XFAIL_CHECKS.get(name, {}).get(check.__name__)
            marks = [pytest.mark.xfail(reason=reason, strict=False)] if reason else []
            params.append(
                pytest.param(name, cls, check, id=f"{name}-{check.__name__}", marks=marks)
            )
    return params


def test_associators_were_discovered():
    """Guard against the sweep silently covering nothing."""
    assert len(ASSOCIATORS) >= 3, f"only {len(ASSOCIATORS)} associators discovered"


@pytest.mark.parametrize("name, cls, check", _cases())
def test_associator_contract(name, cls, check):
    """Every associator satisfies every contract check."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        check(name, cls())
