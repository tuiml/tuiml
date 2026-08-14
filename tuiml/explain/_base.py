"""Shared result type for explanations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class Explanation:
    """A model explanation, in a form that survives being printed or plotted.

    Every explainer in :mod:`tuiml.explain` returns one of these rather than a
    bare array, so the numbers always arrive with the feature names and the
    method that produced them attached.

    Attributes
    ----------
    values : np.ndarray
        The explanation itself. Shape depends on the method: ``(n_features,)``
        for a global importance, ``(n_samples, n_features)`` for a local
        attribution, ``(n_samples, n_features, n_outputs)`` when the model is
        multi-output.
    feature_names : list of str
        One name per feature, defaulted to ``feature_0``... when the caller
        supplies none.
    method : str
        Which explainer produced this.
    base_value : float or np.ndarray, optional
        The model's expected output over the background data. Present for
        additive attributions, where ``values.sum() + base_value`` reconstructs
        the prediction.
    metadata : dict
        Method-specific extras — standard deviations, grids, per-repeat scores.

    See Also
    --------
    :func:`~tuiml.explain.permutation_importance` : Produces a global explanation.
    :class:`~tuiml.explain.TreeExplainer` : Produces a local additive one.

    Examples
    --------
    >>> import numpy as np
    >>> from tuiml.explain import Explanation
    >>> e = Explanation(values=np.array([0.4, 0.1]), method='demo')
    >>> e.feature_names
    ['feature_0', 'feature_1']
    >>> e.top(1)
    [('feature_0', 0.4)]
    """

    values: np.ndarray
    feature_names: Optional[List[str]] = None
    method: str = ""
    base_value: Any = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Fill in default feature names when none were supplied."""
        self.values = np.asarray(self.values)
        if self.feature_names is None:
            n_features = (
                self.values.shape[-1]
                if self.values.ndim == 1
                else self.values.shape[1]
            )
            self.feature_names = [f"feature_{i}" for i in range(n_features)]

    def top(self, k: int = 10) -> List[tuple]:
        """Return the ``k`` most important features, largest magnitude first.

        Parameters
        ----------
        k : int, default=10
            How many to return.

        Returns
        -------
        ranked : list of tuple
            ``(feature_name, value)`` pairs. For local attributions the value
            is the mean absolute contribution across samples, which is the
            standard way to read a local method globally.
        """
        if self.values.ndim == 1:
            magnitude = self.values
        else:
            axes = tuple(i for i in range(self.values.ndim) if i != 1)
            magnitude = np.abs(self.values).mean(axis=axes)

        order = np.argsort(-np.abs(magnitude))[:k]
        return [(self.feature_names[i], float(magnitude[i])) for i in order]

    def __repr__(self) -> str:
        """Return a readable summary naming the top few features."""
        ranked = self.top(3)
        head = ", ".join(f"{name}={value:.4g}" for name, value in ranked)
        return f"Explanation(method={self.method!r}, top: {head})"
