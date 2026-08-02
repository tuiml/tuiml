"""Contract checks for association-rule miners.

Associators sit outside the algorithm sweep because their contract has no
``predict``: ``fit`` yields frequent itemsets and rules. They were previously
the one component family with no real coverage at all.
"""

from __future__ import annotations

import pickle
from typing import Any, Callable, Tuple

from ._data import make_transactions


def _rules(associator):
    """Return the fitted rules, whatever the attribute is called.

    Parameters
    ----------
    associator : object
        A fitted associator.

    Returns
    -------
    rules : list
        The discovered rules, or an empty list if none are exposed.
    """
    for attr in ("rules_", "association_rules_", "rules"):
        value = getattr(associator, attr, None)
        if value is not None and not callable(value):
            return list(value)
    return []


def check_fit_returns_self(name: str, associator) -> None:
    """``fit`` returns the instance.

    Parameters
    ----------
    name : str
        Registry name, used in failure messages.
    associator : object
        A constructed instance.

    Returns
    -------
    None
    """
    returned = associator.fit(make_transactions())
    assert returned is associator, (
        f"{name}: fit() returned {type(returned).__name__} rather than self"
    )


def check_finds_the_planted_pattern(name: str, associator) -> None:
    """Mining a matrix with a planted frequent pattern yields itemsets.

    The fixture plants ``{0, 1} -> {2}`` in most rows, so a miner that returns
    nothing is not merely conservative -- it is not working.

    Parameters
    ----------
    name : str
        Registry name, used in failure messages.
    associator : object
        A constructed instance.

    Returns
    -------
    None
    """
    associator.fit(make_transactions())
    itemsets = getattr(associator, "frequent_itemsets_", None)
    found = len(itemsets) if itemsets is not None else len(_rules(associator))
    assert found > 0, (
        f"{name}: found no itemsets or rules in data containing a pattern "
        f"present in 60% of transactions"
    )


def check_rule_metrics_are_probabilities(name: str, associator) -> None:
    """Support and confidence on every rule lie within [0, 1].

    Parameters
    ----------
    name : str
        Registry name, used in failure messages.
    associator : object
        A constructed instance.

    Returns
    -------
    None
    """
    associator.fit(make_transactions())
    for rule in _rules(associator):
        for metric in ("support", "confidence"):
            value = getattr(rule, metric, None)
            if value is None and isinstance(rule, dict):
                value = rule.get(metric)
            if value is None:
                continue
            assert 0.0 - 1e-9 <= float(value) <= 1.0 + 1e-9, (
                f"{name}: rule {metric} is {value}, outside [0, 1]"
            )


def check_pickle_roundtrip(name: str, associator) -> None:
    """A fitted associator survives pickling with its rule count intact.

    Parameters
    ----------
    name : str
        Registry name, used in failure messages.
    associator : object
        A constructed instance.

    Returns
    -------
    None
    """
    associator.fit(make_transactions())
    restored = pickle.loads(pickle.dumps(associator))
    assert len(_rules(restored)) == len(_rules(associator)), (
        f"{name}: rule count changed after a pickle roundtrip"
    )


#: Every associator check, in run order.
ALL_CHECKS: Tuple[Callable[[str, Any], None], ...] = (
    check_fit_returns_self,
    check_finds_the_planted_pattern,
    check_rule_metrics_are_probabilities,
    check_pickle_roundtrip,
)
