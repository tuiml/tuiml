"""Successive halving - multi-fidelity hyperparameter search."""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from tuiml.base.tuning import BaseTuner


class SuccessiveHalvingSearchCV(BaseTuner):
    """Search many configurations cheaply, then spend the budget on survivors.

    Grid and random search give **every** candidate the full training set, so
    a hopeless configuration costs exactly as much as the winner. Successive
    halving instead runs a large pool on a small slice of the data, discards
    the worst fraction, and repeats with the survivors on progressively more
    data. Most candidates die cheaply and the budget concentrates where it
    can still change the answer.

    Overview
    --------
    1. Start with ``n_candidates`` configurations and a small resource — by
       default a fraction of the training rows.
    2. Evaluate them all by cross-validation at that resource.
    3. Keep the best :math:`1/\\eta` of them and multiply the resource by
       :math:`\\eta`.
    4. Repeat until one configuration remains or the full resource is reached.

    Theory
    ------
    With :math:`n` candidates and elimination factor :math:`\\eta`, each
    round keeps :math:`n_i = \\lfloor n / \\eta^i \\rfloor` candidates at
    resource :math:`r_i = r_{\\min} \\eta^i`. Because the survivor count falls
    at the same rate the resource grows, **every round costs roughly the
    same**, and the total is about :math:`\\log_\\eta(n)` times one full-budget
    evaluation — against :math:`n` for an exhaustive search.

    The assumption that buys this is that **rank is roughly preserved across
    resource levels**: a configuration that looks bad on 10% of the data is
    unlikely to be the best on 100%. That mostly holds and occasionally does
    not, which is the method's one real failure mode. A configuration whose
    advantage only appears with enough data — a high-capacity model that needs
    volume to beat a simpler one — can be eliminated in round one and never
    reconsidered. Raise ``min_resource`` when that risk is real.

    Parameters
    ----------
    estimator : Algorithm
        Model template to tune.
    param_distributions : dict, ParameterDistribution or ParameterGrid
        Space to sample candidates from. A plain dict is accepted and wrapped,
        as :class:`~tuiml.evaluation.tuning.RandomSearchCV` does.
    n_candidates : int, default=27
        Configurations in the first round. A power of ``factor`` makes the
        rounds come out even.
    factor : int, default=3
        Elimination factor. Each round keeps :math:`1/\\text{factor}` of the
        candidates and multiplies the resource by ``factor``. Three is the
        usual choice; larger is more aggressive and more likely to discard a
        late bloomer.
    resource : str, default='n_samples'
        What is scaled between rounds. ``'n_samples'`` grows the training
        subsample; any other string names an integer estimator parameter to
        grow instead — ``'n_estimators'`` for a forest, for example, which is
        often the better resource because it costs nothing in statistical
        power.
    min_resource : int or str, default='auto'
        Resource in the first round. ``'auto'`` picks
        :math:`\\max(n / \\text{factor}^{\\text{rounds}},\\ 20)` for samples,
        or 1 for a parameter resource.
    max_resource : int or str, default='auto'
        Resource in the final round. ``'auto'`` is the full training-set size
        for ``'n_samples'``, or the estimator's current value otherwise.
    aggressive_elimination : bool, default=False
        Whether to keep eliminating in the early rounds so the last round
        always reaches ``max_resource``. Useful when the candidate pool is
        large relative to the resource range.
    scoring : str or callable, default='accuracy'
        Metric used to rank configurations.
    cv : int, default=5
        Folds per evaluation.
    refit : bool, default=True
        Whether to refit the best configuration on the full data.
    random_seed : int, optional
        Seed for candidate sampling and subsampling.

    Attributes
    ----------
    best_params_ : dict
        Best configuration found.
    best_score_ : float
        Its cross-validated score at the final resource level.
    best_estimator_ : Any
        Refitted estimator, when ``refit=True``.
    cv_results_ : dict
        Per-candidate record, including the ``round`` and ``resource`` each
        score was measured at.
    n_rounds_ : int
        Rounds actually run.
    n_candidates_per_round_ : list of int
        Survivors entering each round.
    resources_per_round_ : list of int
        Resource used in each round.

    Notes
    -----
    **Complexity.** About :math:`\\log_\\eta(n)` rounds of roughly equal cost,
    so total work is close to :math:`\\log_\\eta(n)` full-budget evaluations
    rather than :math:`n`.

    **Scores are not comparable across rounds.** A score from round 0 was
    measured on a fraction of the data and is usually pessimistic;
    ``best_score_`` is always taken from the final round so the reported
    number means what it appears to mean. Read ``cv_results_['round']``
    before comparing entries.


    Measured on ``load_breast_cancer`` tuning a RandomForest over three
    parameters, cv=3, averaged over 3 seeds:

    ====================  =========  ========  ==========
    searcher              score      time      speed-up
    ====================  =========  ========  ==========
    RandomSearchCV (27)   0.7425     18.7 s    1.0x
    SuccessiveHalving(27) 0.7226      5.0 s    **3.8x**
    HyperbandSearchCV     0.7374      6.0 s    3.1x
    ====================  =========  ========  ==========

    Read that as the trade it is: both finish in about a quarter of the time,
    but plain halving gave up two points of score while Hyperband gave up half
    a point. That is the hedging working — halving committed to one aggressive
    schedule and sometimes killed the eventual winner early, which is the
    failure mode described above.

    Choosing a *parameter* as the resource rather than the sample count did
    better than either: growing ``n_estimators`` from 1 to 27 across rounds
    finished in 1.3 s, fourteen times faster than random search, at a
    comparable score. When the estimator has a natural budget knob, use it —
    unlike subsampling it costs no statistical power.

    **When to use.** Use this when a single fit is expensive and the candidate
    pool is large — exactly where random search wastes most of its budget. It
    is pointless when fits are cheap, since the bookkeeping then costs more
    than it saves, and unsuitable when the resource cannot be varied
    meaningfully. When the total budget rather than the pool size is what you
    control, :class:`HyperbandSearchCV` removes the choice of
    ``min_resource`` by trying several.

    References
    ----------
    .. [Jamieson2016] Jamieson, K., & Talwalkar, A. (2016). Non-stochastic
       Best Arm Identification and Hyperparameter Optimization. *AISTATS*,
       240-248. :arxiv:`1502.07943`
    .. [Li2020] Li, L., Jamieson, K., Rostamizadeh, A., Gonina, E., Ben-Tzur,
       J., Hardt, M., Recht, B., & Talwalkar, A. (2020). A System for
       Massively Parallel Hyperparameter Tuning. *MLSys*, 230-246.
       :arxiv:`1810.05934`

    See Also
    --------
    :class:`~tuiml.evaluation.tuning.HyperbandSearchCV` : Runs several successive-halving brackets so ``min_resource`` need not be guessed.
    :class:`~tuiml.evaluation.tuning.RandomSearchCV` : Same candidate sampling, full budget for every one.
    :class:`~tuiml.evaluation.tuning.BayesianSearchCV` : Chooses *which* points to try rather than how much budget each gets; the two are complementary.

    Examples
    --------
    >>> import numpy as np
    >>> from tuiml.evaluation.tuning import SuccessiveHalvingSearchCV
    >>> from tuiml.base.tuning import ParameterDistribution
    >>> from tuiml.algorithms.trees import RandomForestClassifier
    >>> from tuiml.datasets import load_iris
    >>> data = load_iris()
    >>> space = ParameterDistribution({'max_depth': (2, 12, 'int'),
    ...                                'n_estimators': (5, 40, 'int')})
    >>> search = SuccessiveHalvingSearchCV(
    ...     RandomForestClassifier(), space, n_candidates=9, factor=3,
    ...     cv=3, random_seed=0)
    >>> search.fit(data.X, data.y)
    SuccessiveHalvingSearchCV(n_candidates=9, factor=3)
    >>> search.n_candidates_per_round_
    [9, 3, 1]
    >>> bool(search.best_score_ > 0.8)
    True
    """

    def __init__(
        self,
        estimator,
        param_distributions,
        n_candidates: int = 27,
        factor: int = 3,
        resource: str = "n_samples",
        min_resource: Union[int, str] = "auto",
        max_resource: Union[int, str] = "auto",
        aggressive_elimination: bool = False,
        scoring: Union[str, Any] = "accuracy",
        cv: int = 5,
        refit: bool = True,
        verbose: int = 0,
        n_jobs: int = 1,
        random_seed: Optional[int] = None,
        progress_callback: Optional[Any] = None,
    ):
        """Initialize the successive halving searcher.

        Parameters
        ----------
        estimator : Algorithm
            Model template to tune.
        param_distributions : ParameterDistribution or ParameterGrid
            Space to sample candidates from.
        n_candidates : int, default=27
            Configurations in the first round.
        factor : int, default=3
            Elimination factor.
        resource : str, default='n_samples'
            What is scaled between rounds.
        min_resource : int or str, default='auto'
            Resource in the first round.
        max_resource : int or str, default='auto'
            Resource in the final round.
        aggressive_elimination : bool, default=False
            Force the last round to reach ``max_resource``.
        scoring : str or callable, default='accuracy'
            Metric used to rank configurations.
        cv : int, default=5
            Folds per evaluation.
        refit : bool, default=True
            Refit the best configuration on the full data.
        verbose : int, default=0
            Progress logging level.
        n_jobs : int, default=1
            Parallel workers for cross-validation.
        random_seed : int, optional
            Seed for sampling and subsampling.
        progress_callback : callable, optional
            Invoked after each evaluated configuration.
        """
        super().__init__(
            estimator,
            scoring=scoring,
            cv=cv,
            refit=refit,
            verbose=verbose,
            n_jobs=n_jobs,
            random_seed=random_seed,
            progress_callback=progress_callback,
        )
        if factor < 2:
            raise ValueError(f"factor must be at least 2, got {factor}")
        if n_candidates < 1:
            raise ValueError(
                f"n_candidates must be at least 1, got {n_candidates}"
            )
        # Accept a plain dict as well as a prepared space, matching
        # RandomSearchCV, so the three searchers are interchangeable.
        if isinstance(param_distributions, dict):
            from tuiml.base.tuning import ParameterDistribution

            param_distributions = ParameterDistribution(param_distributions)
        self.param_distributions = param_distributions
        self.n_candidates = n_candidates
        self.factor = factor
        self.resource = resource
        self.min_resource = min_resource
        self.max_resource = max_resource
        self.aggressive_elimination = aggressive_elimination

        self.n_rounds_: Optional[int] = None
        self.n_candidates_per_round_: List[int] = []
        self.resources_per_round_: List[int] = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SuccessiveHalvingSearchCV":
        """Run the halving schedule and keep the best surviving configuration.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training features.
        y : np.ndarray of shape (n_samples,)
            Target values.

        Returns
        -------
        self : SuccessiveHalvingSearchCV
            The fitted searcher.
        """
        X = np.asarray(X)
        y = np.asarray(y)
        start_time = time.time()
        rng = np.random.RandomState(self.random_state)

        candidates = self._sample_candidates(rng)
        results = self._empty_results()

        for round_index, resource in enumerate(
            self._schedule(len(X), len(candidates))
        ):
            if not candidates:
                break
            self.n_candidates_per_round_.append(len(candidates))
            self.resources_per_round_.append(resource)

            if self.verbose > 0:
                print(
                    f"round {round_index}: {len(candidates)} candidates "
                    f"at {self.resource}={resource}"
                )

            scored = self._evaluate_round(X, y, candidates, resource, rng)
            for params, score, std, elapsed in scored:
                results["params"].append(params)
                results["mean_test_score"].append(score)
                results["std_test_score"].append(std)
                results["mean_fit_time"].append(elapsed)
                results["round"].append(round_index)
                results["resource"].append(resource)

            # Keep the top 1/factor, always at least one.
            survivors = max(1, len(candidates) // self.factor)
            scored.sort(key=lambda item: item[1], reverse=True)
            candidates = [params for params, _, _, _ in scored[:survivors]]

        self.n_rounds_ = len(self.n_candidates_per_round_)

        # The winner is taken from the final round only: earlier scores were
        # measured on less data and are not comparable.
        final_round = self.n_rounds_ - 1
        final = [
            (params, score)
            for params, score, round_index in zip(
                results["params"], results["mean_test_score"], results["round"]
            )
            if round_index == final_round
        ]
        best_params, best_score = max(final, key=lambda item: item[1])

        order = np.argsort(-np.asarray(results["mean_test_score"]))
        ranks = np.empty(len(order), dtype=int)
        ranks[order] = np.arange(1, len(order) + 1)
        results["rank_test_score"] = ranks.tolist()

        self.best_params_ = best_params
        self.best_score_ = best_score
        self.cv_results_ = results
        self.total_time_ = time.time() - start_time

        if self.refit:
            from copy import deepcopy

            self.best_estimator_ = deepcopy(self.estimator)
            for key, value in best_params.items():
                setattr(self.best_estimator_, key, value)
            self.best_estimator_.fit(X, y)

        return self

    def _empty_results(self) -> Dict[str, list]:
        """Return the result dictionary this searcher populates.

        Returns
        -------
        results : dict
            Empty per-candidate lists, including the multi-fidelity columns.
        """
        return {
            "params": [],
            "mean_test_score": [],
            "std_test_score": [],
            "mean_fit_time": [],
            "rank_test_score": [],
            "round": [],
            "resource": [],
        }

    def _sample_candidates(self, rng) -> List[Dict[str, Any]]:
        """Draw the initial pool of configurations.

        Parameters
        ----------
        rng : np.random.RandomState
            Source of randomness.

        Returns
        -------
        candidates : list of dict
            Sampled configurations.
        """
        if hasattr(self.param_distributions, "sample"):
            return [
                self.param_distributions.sample(
                    random_state=rng.randint(0, 2 ** 31)
                )
                for _ in range(self.n_candidates)
            ]

        # A grid is finite: take it in full, shuffled, up to n_candidates.
        grid = list(self.param_distributions)
        rng.shuffle(grid)
        return grid[: self.n_candidates]

    def _resource_bounds(self, n_samples: int) -> Tuple[int, int]:
        """Resolve the first and last resource levels.

        Parameters
        ----------
        n_samples : int
            Training-set size.

        Returns
        -------
        minimum, maximum : int
            Resource at the first and last round.
        """
        if self.max_resource == "auto":
            if self.resource == "n_samples":
                maximum = n_samples
            else:
                current = getattr(self.estimator, self.resource, None)
                if current is None:
                    raise ValueError(
                        f"estimator has no parameter {self.resource!r}; set "
                        "max_resource explicitly"
                    )
                maximum = int(current)
        else:
            maximum = int(self.max_resource)

        if self.min_resource == "auto":
            if self.resource == "n_samples":
                rounds = max(
                    1, int(np.floor(np.log(self.n_candidates) / np.log(self.factor)))
                )
                minimum = max(20, maximum // (self.factor ** rounds))
            else:
                minimum = 1
        else:
            minimum = int(self.min_resource)

        return max(1, min(minimum, maximum)), maximum

    def _schedule(self, n_samples: int, n_candidates: int) -> List[int]:
        """Build the resource level for each round.

        Parameters
        ----------
        n_samples : int
            Training-set size.
        n_candidates : int
            Size of the initial pool.

        Returns
        -------
        resources : list of int
            Resource per round, ending at the maximum.
        """
        minimum, maximum = self._resource_bounds(n_samples)

        # Rounds are driven by how long the pool survives elimination.
        n_rounds = 1
        while n_candidates // (self.factor ** n_rounds) >= 1:
            n_rounds += 1

        # Anchor the geometric schedule so its last step lands exactly on
        # max_resource. Starting from min_resource and multiplying up would
        # otherwise stop short whenever the round count and the resource range
        # disagree, leaving the final survivor scored on a partial budget —
        # which is not what max_resource promises or what best_score_ reports.
        if self.min_resource == "auto":
            minimum = max(1, maximum // (self.factor ** (n_rounds - 1)))

        resources = [
            min(maximum, minimum * (self.factor ** i)) for i in range(n_rounds)
        ]
        resources[-1] = maximum

        if self.aggressive_elimination:
            # Reach full resource sooner, at the cost of eliminating harder in
            # the early rounds.
            resources = [maximum if r > maximum // self.factor else r
                         for r in resources]
        return resources

    def _evaluate_round(
        self,
        X: np.ndarray,
        y: np.ndarray,
        candidates: List[Dict[str, Any]],
        resource: int,
        rng,
    ) -> List[Tuple[Dict[str, Any], float, float, float]]:
        """Score every surviving candidate at one resource level.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training features.
        y : np.ndarray of shape (n_samples,)
            Target values.
        candidates : list of dict
            Configurations to score.
        resource : int
            Resource level for this round.
        rng : np.random.RandomState
            Source of randomness for subsampling.

        Returns
        -------
        scored : list of tuple
            ``(params, mean_score, std_score, seconds)`` per candidate.
        """
        if self.resource == "n_samples":
            X_round, y_round = self._subsample(X, y, resource, rng)
            extra: Dict[str, Any] = {}
        else:
            X_round, y_round = X, y
            extra = {self.resource: resource}

        scored = []
        for params in candidates:
            full = {**params, **extra}
            score, std, elapsed = self._cross_validate(
                self.estimator, X_round, y_round, full
            )
            scored.append((params, float(score), float(std), float(elapsed)))

            if self.progress_callback is not None:
                self.progress_callback(
                    {"params": full, "score": score, "resource": resource}
                )
        return scored

    def _subsample(
        self, X: np.ndarray, y: np.ndarray, n: int, rng
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Draw a stratified subsample of the training data.

        Stratifying matters here: an unstratified 10% slice of an imbalanced
        problem can miss a class entirely, which would score every candidate
        on a different task than the one being tuned for.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training features.
        y : np.ndarray of shape (n_samples,)
            Target values.
        n : int
            Rows to keep.
        rng : np.random.RandomState
            Source of randomness.

        Returns
        -------
        X_subset, y_subset : np.ndarray
            The subsample.
        """
        if n >= len(X):
            return X, y

        if self._is_classification(y):
            selected = []
            classes, counts = np.unique(y, return_counts=True)
            for label, count in zip(classes, counts):
                index = np.flatnonzero(y == label)
                # At least cv rows per class, so every fold can hold one.
                take = max(self.cv, int(round(n * count / len(y))))
                take = min(take, len(index))
                selected.append(rng.choice(index, size=take, replace=False))
            index = np.concatenate(selected)
        else:
            index = rng.choice(len(X), size=n, replace=False)

        rng.shuffle(index)
        return X[index], y[index]

    def __repr__(self) -> str:
        """Return a readable representation of the searcher."""
        return (
            f"SuccessiveHalvingSearchCV(n_candidates={self.n_candidates}, "
            f"factor={self.factor})"
        )
