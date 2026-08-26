"""Model-guided sampling over one candidate's search space.

Where :mod:`~tuiml.automl.search_space` decides *what* may be searched, this
module decides *where to look next* inside that space. It is the piece that
turns AutoML's inner loop from uniform random sampling into Bayesian
optimization, and it does so with TuiML's own surrogate machinery:
:class:`~tuiml.evaluation.tuning.bayesian_search.GaussianProcess` and
:class:`~tuiml.evaluation.tuning.bayesian_search.AcquisitionFunction` are used
directly, rather than a second implementation being written here.

Those two classes are deliberately independent of
:class:`~tuiml.evaluation.tuning.BayesianSearchCV`, which is what makes the
reuse possible: ``BayesianSearchCV`` owns its own cross-validation loop and
refits, whereas AutoML must keep control of evaluation so that every trial
shares one set of splits and contributes its fitted model to the ensemble
pool. Borrowing the surrogate without the estimator gets both.

Overview
--------
1. :class:`ParameterEncoder` maps a parameter dict to a fixed-length numeric
   vector, because a GP needs points in :math:`\\mathbb{R}^d` and a search
   space is a mixture of numeric ranges and unordered categories.
2. :class:`SurrogateSampler` records every ``(params, score)`` observed, fits
   the GP to them, scores a pool of random draws with the acquisition
   function, and returns the most promising draw.
3. Until a few observations exist there is nothing to condition on, so the
   sampler returns plain random draws and behaves exactly as before.

Notes
-----
The encoding is the part that carries the modelling assumptions. Numeric
ranges are scaled to :math:`[0, 1]` so that one RBF length scale is
meaningful across parameters of wildly different magnitudes, and log-scaled
ranges are encoded in log space so that the surrogate sees the geometry the
space was defined with. Categories are one-hot encoded rather than given an
integer index, since an index would tell the GP that ``'linear'`` is nearer
``'poly'`` than ``'rbf'``, which is not true of an unordered set.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from tuiml.base.tuning import ParameterDistribution
from tuiml.evaluation.tuning.bayesian_search import (
    AcquisitionFunction,
    GaussianProcess,
)

#: Observations needed before the surrogate is trusted. Below this the GP has
#: too little to condition on for its posterior to beat a random draw, and
#: fitting it is wasted time.
MIN_OBSERVATIONS = 6

#: Random draws scored by the acquisition function per suggestion. The
#: acquisition surface is cheap to evaluate and the fits it is choosing
#: between are not, so a generous pool is worth its cost.
N_ACQUISITION_CANDIDATES = 96


class ParameterEncoder:
    """Encode parameter dicts as fixed-length numeric vectors for a GP.

    Parameters
    ----------
    param_distributions : dict
        The mapping held by a
        :class:`~tuiml.base.tuning.ParameterDistribution`: each value is a
        list of choices, or a ``(low, high)`` / ``(low, high, 'log')`` /
        ``(low, high, 'int')`` range tuple.

    Attributes
    ----------
    columns_ : list of tuple
        One ``(name, kind, payload)`` entry per encoded column group, in
        encoding order. ``kind`` is ``"numeric"`` or ``"choice"``.
    n_features_ : int
        Width of the encoded vector.

    Examples
    --------
    >>> from tuiml.automl.surrogate import ParameterEncoder
    >>> encoder = ParameterEncoder({'C': (0.01, 100.0, 'log'),
    ...                             'kernel': ['linear', 'rbf']})
    >>> encoder.n_features_
    3
    >>> encoder.encode({'C': 1.0, 'kernel': 'rbf'}).round(3).tolist()
    [0.5, 0.0, 1.0]
    """

    def __init__(self, param_distributions: Dict[str, Any]):
        """Build the column layout from a parameter distribution mapping."""
        self.columns_: List[Tuple[str, str, Any]] = []
        for name, spec in param_distributions.items():
            if isinstance(spec, tuple) and len(spec) in (2, 3):
                low, high = float(spec[0]), float(spec[1])
                scale = spec[2] if len(spec) == 3 else None
                logarithmic = scale == "log" and low > 0 and high > 0
                if logarithmic:
                    low, high = math.log(low), math.log(high)
                span = high - low
                self.columns_.append(
                    (name, "numeric", (low, span if span else 1.0, logarithmic))
                )
            elif isinstance(spec, (list, tuple)):
                self.columns_.append((name, "choice", list(spec)))
            # Anything else (a bare callable) carries no usable geometry and
            # is left out of the encoding rather than guessed at.

        self.n_features_ = sum(
            1 if kind == "numeric" else len(payload)
            for _, kind, payload in self.columns_
        )

    def encode(self, params: Dict[str, Any]) -> np.ndarray:
        """Encode one parameter dict as a numeric vector.

        Parameters
        ----------
        params : dict
            A sampled configuration.

        Returns
        -------
        vector : np.ndarray of shape (n_features_,)
            Numeric ranges scaled to :math:`[0, 1]` (in log space where the
            range is logarithmic), categories one-hot encoded. An absent or
            unrecognised value encodes as all-zero for its column group.
        """
        vector = np.zeros(self.n_features_, dtype=float)
        offset = 0
        for name, kind, payload in self.columns_:
            if kind == "numeric":
                low, span, logarithmic = payload
                value = params.get(name)
                if value is not None:
                    try:
                        numeric = float(value)
                        if logarithmic:
                            numeric = math.log(numeric) if numeric > 0 else low
                        vector[offset] = (numeric - low) / span
                    except (TypeError, ValueError):
                        pass
                offset += 1
            else:
                value = params.get(name)
                for index, choice in enumerate(payload):
                    if _same(choice, value):
                        vector[offset + index] = 1.0
                        break
                offset += len(payload)
        return vector

    def encode_many(self, rows: Sequence[Dict[str, Any]]) -> np.ndarray:
        """Encode a sequence of parameter dicts as a matrix.

        Parameters
        ----------
        rows : sequence of dict
            Configurations to encode.

        Returns
        -------
        matrix : np.ndarray of shape (n_rows, n_features_)
            One encoded vector per row.
        """
        if not rows:
            return np.zeros((0, self.n_features_), dtype=float)
        return np.vstack([self.encode(row) for row in rows])


def _same(choice: Any, value: Any) -> bool:
    """Compare a declared choice with a sampled value tolerantly.

    ``ParameterDistribution.sample`` returns NumPy scalars, so a sampled
    ``'rbf'`` is a ``np.str_`` and a sampled ``True`` is a ``np.bool_``. Plain
    ``==`` on those is fine, but ``None`` and NumPy values need care, and a
    comparison that raises must not take the search down.

    Parameters
    ----------
    choice : Any
        A value from the declared choice list.
    value : Any
        The sampled value to test against it.

    Returns
    -------
    match : bool
        Whether the two denote the same choice.
    """
    if choice is None or value is None:
        return choice is None and value is None
    try:
        return bool(choice == value)
    except Exception:  # pragma: no cover - exotic comparison types
        return False


class SurrogateSampler:
    """Suggest the next configuration for one candidate, GP-guided.

    Holds the observation history for a single algorithm and proposes the
    draw the acquisition function rates highest. The first
    :data:`MIN_OBSERVATIONS` suggestions are plain random draws, so early
    behaviour is identical to random search and the surrogate only takes over
    once it has something to learn from.

    Parameters
    ----------
    space : ParameterDistribution
        The candidate's search space, from
        :func:`~tuiml.automl.search_space.search_space_for`.
    acquisition : str, default='ei'
        Acquisition policy, passed to
        :class:`~tuiml.evaluation.tuning.bayesian_search.AcquisitionFunction`:
        ``'ei'``, ``'pi'`` or ``'ucb'``.
    n_candidates : int, default=N_ACQUISITION_CANDIDATES
        Random draws scored per suggestion.

    Attributes
    ----------
    observations_ : list of tuple
        The ``(params, score)`` pairs recorded so far.
    n_guided_ : int
        How many suggestions were chosen by the surrogate rather than drawn
        at random. Useful for confirming the surrogate actually engaged.

    Examples
    --------
    >>> from tuiml.automl.search_space import search_space_for
    >>> from tuiml.automl.surrogate import SurrogateSampler
    >>> from tuiml.registry import registry
    >>> import numpy as np
    >>> sampler = SurrogateSampler(search_space_for(registry.get('SVC')))
    >>> rng = np.random.RandomState(0)
    >>> params = sampler.suggest(rng)
    >>> sorted(params)
    ['C', 'degree', 'kernel', 'tol']
    >>> sampler.observe(params, 0.9)
    >>> len(sampler.observations_)
    1
    """

    def __init__(
        self,
        space: ParameterDistribution,
        acquisition: str = "ei",
        n_candidates: int = N_ACQUISITION_CANDIDATES,
    ):
        """Store the space and prepare an empty observation history."""
        self.space = space
        self.acquisition = acquisition
        self.n_candidates = int(n_candidates)
        self.observations_: List[Tuple[Dict[str, Any], float]] = []
        self.n_guided_ = 0
        self._encoder = ParameterEncoder(space.param_distributions)
        self._acquire = AcquisitionFunction(kind=acquisition)

    def observe(self, params: Dict[str, Any], score: Optional[float]) -> None:
        """Record the score a configuration achieved.

        Parameters
        ----------
        params : dict
            The configuration that was evaluated.
        score : float or None
            Its score, higher being better. ``None`` (a failed trial) is
            ignored rather than treated as a bad score, since a crash says
            nothing about where the optimum lies.
        """
        if score is None or not np.isfinite(score):
            return
        self.observations_.append((dict(params), float(score)))

    def suggest(self, rng: np.random.RandomState) -> Dict[str, Any]:
        """Propose the next configuration to evaluate.

        Parameters
        ----------
        rng : np.random.RandomState
            Source of randomness, so a whole search stays reproducible from
            one seed.

        Returns
        -------
        params : dict
            A random draw while the history is too short or the space has no
            encodable geometry, otherwise the draw maximising the acquisition
            function over the GP posterior.
        """
        draw = lambda: self.space.sample(random_state=int(rng.randint(1 << 30)))

        if (
            len(self.observations_) < MIN_OBSERVATIONS
            or self._encoder.n_features_ == 0
        ):
            return draw()

        pool = [draw() for _ in range(self.n_candidates)]
        try:
            best = self._rank(pool)
        except Exception:
            # A singular kernel matrix or a degenerate posterior must cost a
            # suggestion's quality, never the run. Falling back to the first
            # random draw keeps the search going at random-search quality.
            return pool[0]
        self.n_guided_ += 1
        return best

    def _rank(self, pool: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Fit the GP to the history and return the pool's best draw.

        Parameters
        ----------
        pool : list of dict
            Random draws to choose between.

        Returns
        -------
        params : dict
            The draw with the highest acquisition value.
        """
        observed = self._encoder.encode_many([p for p, _ in self.observations_])
        scores = np.array([s for _, s in self.observations_], dtype=float)

        # The GP prior has zero mean, so the observations are centred and
        # scaled before conditioning. Without this a metric living around
        # 0.97 looks to the prior like a large constant offset to explain.
        spread = scores.std()
        centred = (scores - scores.mean()) / (spread if spread > 1e-12 else 1.0)

        gp = GaussianProcess(length_scale=1.0, noise=1e-6)
        gp.fit(observed, centred)

        pool_encoded = self._encoder.encode_many(pool)
        values = self._acquire(pool_encoded, gp, float(centred.max()))
        return pool[int(np.argmax(values))]
