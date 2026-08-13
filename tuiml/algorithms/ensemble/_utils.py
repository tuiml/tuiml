"""Shared helpers for the ensemble meta-learners."""

import inspect
from typing import Any, Optional, Type


def make_base_estimator(base_class: Type, seed: Optional[int] = None) -> Any:
    """Instantiate a base learner, seeding it when it accepts a seed.

    Parameters
    ----------
    base_class : type
        The base learner class to construct.
    seed : int or None, default=None
        Seed to pass as ``random_state``. When None, or when ``base_class``
        does not accept ``random_state``, the class is constructed with its
        own defaults.

    Returns
    -------
    estimator : object
        A fresh, unfitted base learner.

    Notes
    -----
    A meta-learner that seeds only its own sampling is still not reproducible
    if the base learner carries its own RNG: an unseeded base breaks ties
    between equally good splits differently on each fit, so two runs at the
    same ``random_state`` can return different predictions. Threading a
    per-estimator seed through closes that gap.

    Examples
    --------
    >>> from tuiml.algorithms.trees import DecisionTreeClassifier
    >>> from tuiml.algorithms.ensemble._utils import make_base_estimator
    >>> est = make_base_estimator(DecisionTreeClassifier, seed=7)
    >>> est.random_state
    7
    """
    if seed is None:
        return base_class()
    try:
        params = inspect.signature(base_class.__init__).parameters
    except (TypeError, ValueError):  # pragma: no cover - builtins without a signature
        return base_class()
    if "random_state" in params:
        return base_class(random_state=seed)
    return base_class()
