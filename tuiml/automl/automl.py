"""Automatic model selection and tuning under a wall-clock budget.

:class:`AutoMLClassifier` and :class:`AutoMLRegressor` take a dataset and a
number of seconds, and give back a fitted model, a leaderboard of everything
they tried, and -- the part that makes the result portable -- ``best_spec_``,
the winning configuration written as a :func:`tuiml.train` spec. The spec is
plain JSON-compatible data, so the outcome of a search can be committed to a
repository, diffed, reviewed, and replayed without re-running the search.
"""

import math
import time
import warnings
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import numpy as np

from tuiml.automl.ensembling import GreedyEnsemble, align_proba, greedy_selection
from tuiml.automl.portfolio import build_portfolio
from tuiml.automl.surrogate import SurrogateSampler
from tuiml.automl.search_space import BUDGET_PARAMETERS, search_space_for
from tuiml.base.algorithms import Classifier, Regressor, classifier, regressor

#: Safety factor applied to a candidate's worst observed trial time before
#: starting another one. A sampled configuration is routinely slower than the
#: defaults, so a candidate is skipped unless its worst run so far fits the
#: remaining budget with this much room to spare.
_COST_SAFETY = 3.0

#: Metrics where a **smaller** value is better. Scores are negated internally
#: so that every comparison in the search is "higher is better".
_LOWER_IS_BETTER = frozenset({
    "mean_squared_error",
    "root_mean_squared_error",
    "mean_absolute_error",
    "log_loss",
    "hamming_loss",
    "zero_one_loss",
    "relative_absolute_error",
    "root_relative_squared_error",
})


class _Scorer:
    """A metric wrapped so that larger is always better.

    A named class rather than a closure so that a fitted searcher stays
    picklable, which is what :meth:`~tuiml.base.algorithms.Algorithm.save`
    relies on.

    Parameters
    ----------
    function : callable
        The underlying metric ``(y_true, y_pred) -> float``.
    negate : bool
        Whether the metric is a loss, and must be negated.
    """

    def __init__(self, function: Callable, negate: bool):
        """Store the metric and whether to flip its sign."""
        self.function = function
        self.negate = negate

    def __call__(self, y_true, y_pred) -> float:
        """Return the score, negated when the metric is a loss."""
        value = float(self.function(y_true, y_pred))
        return -value if self.negate else value


def _resolve_metric(metric: Union[str, Callable, None], default: str):
    """Resolve a metric name to a higher-is-better scorer.

    Parameters
    ----------
    metric : str or callable or None
        A function name from :mod:`tuiml.evaluation.metrics`, a callable
        ``(y_true, y_pred) -> float``, or None to use ``default``.
    default : str
        Metric name used when ``metric`` is None.

    Returns
    -------
    name : str or None
        The metric's registry name, or None when a raw callable was given.
    scorer : callable
        A scorer where larger values are better.

    Raises
    ------
    ValueError
        If the named metric does not exist in :mod:`tuiml.evaluation.metrics`.
    """
    if callable(metric):
        return None, metric

    from tuiml.evaluation import metrics as metrics_module

    name = metric or default
    function = getattr(metrics_module, name, None)
    if function is None:
        raise ValueError(
            f"Unknown metric {name!r}. Use a function name from "
            f"tuiml.evaluation.metrics, or pass a callable."
        )
    return name, _Scorer(function, negate=name in _LOWER_IS_BETTER)


def _to_native(value: Any) -> Any:
    """Convert NumPy scalars and containers to plain Python equivalents.

    Sampled hyperparameters arrive as ``np.str_`` / ``np.int64`` / ``np.bool_``.
    They behave like their Python counterparts but do not serialise to JSON,
    which would make ``best_spec_`` unwritable.

    Parameters
    ----------
    value : Any
        The value to convert.

    Returns
    -------
    native : Any
        The same value using built-in Python types.
    """
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_to_native(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {key: _to_native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_native(item) for item in value]
    return value


class _AutoMLBase:
    """Shared search machinery for :class:`AutoMLClassifier` / :class:`AutoMLRegressor`.

    Not used directly; see the two public classes for the documented API.
    """

    #: Set by the subclasses to ``"classification"`` or ``"regression"``.
    _task = "classification"
    #: Metric used when the caller does not name one.
    _default_metric = "accuracy_score"

    def __init__(
        self,
        time_budget: float = 60.0,
        metric: Union[str, Callable, None] = None,
        cv: Optional[int] = 3,
        random_state: Optional[int] = None,
        ensemble: bool = True,
        candidates: Optional[Sequence[Any]] = None,
        max_candidates: Optional[int] = None,
        n_ensemble_rounds: int = 25,
        halving_factor: int = 3,
        acquisition: str = "ei",
        verbose: int = 0,
    ):
        """Store the search configuration; see the class docstring."""
        self.time_budget = time_budget
        self.metric = metric
        self.cv = cv
        self.random_state = random_state
        self.ensemble = ensemble
        self.candidates = list(candidates) if candidates is not None else None
        self.max_candidates = max_candidates
        self.n_ensemble_rounds = n_ensemble_rounds
        self.halving_factor = halving_factor
        self.acquisition = acquisition
        self.verbose = verbose
        self._is_fitted = False

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        """Return JSON Schema for constructor parameters."""
        return {
            "time_budget": {
                "type": "number",
                "default": 60.0,
                "minimum": 0,
                "description": "Wall-clock seconds the search may spend",
            },
            "metric": {
                "type": ["string", "null"],
                "default": None,
                "description": "Metric name from tuiml.evaluation.metrics",
            },
            "cv": {
                "type": ["integer", "null"],
                "default": 3,
                "minimum": 2,
                "description": "Cross-validation folds; None for a holdout split",
            },
            "random_state": {
                "type": ["integer", "null"],
                "default": None,
                "description": "Random seed for reproducibility",
            },
            "ensemble": {
                "type": "boolean",
                "default": True,
                "description": "Build a greedy ensemble from the trial pool",
            },
            "candidates": {
                "type": ["array", "null"],
                "default": None,
                "description": "Explicit candidate algorithm names to search",
            },
            "max_candidates": {
                "type": ["integer", "null"],
                "default": None,
                "minimum": 1,
                "description": "Truncate the discovered portfolio to this many",
            },
            "halving_factor": {
                "type": "integer",
                "default": 3,
                "minimum": 1,
                "description": (
                    "Candidate elimination rate per rung; 1 disables halving"
                ),
            },
            "acquisition": {
                "type": "string",
                "default": "ei",
                "enum": ["ei", "pi", "ucb"],
                "description": "Acquisition policy for the GP surrogate",
            },
            "n_ensemble_rounds": {
                "type": "integer",
                "default": 25,
                "minimum": 1,
                "description": "Greedy ensemble selection rounds",
            },
            "verbose": {
                "type": "integer",
                "default": 0,
                "description": "Print per-trial progress when greater than 0",
            },
        }

    # -- search -----------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_AutoMLBase":
        """Search for the best model within the time budget.

        Overview
        --------
        1. Build the candidate portfolio (cheap and strong models first).
        2. **Pass one**: evaluate every candidate with its default
           parameters, so even a tiny budget returns a full comparison.
        3. **Pass two**: successive halving over the candidates. Each rung
           gives every survivor the same number of draws, drops the weakest
           fraction, and gives the next rung's survivors more draws each, so
           budget concentrates on what is winning. Draws come from a
           :class:`~tuiml.automl.surrogate.SurrogateSampler` -- a Gaussian
           process fitted to that candidate's own history -- falling back to
           random until it has enough observations to beat one.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training features.
        y : np.ndarray of shape (n_samples,)
            Target values.

        Returns
        -------
        self : _AutoMLBase
            The fitted searcher, with ``best_estimator_``, ``best_spec_``,
            and ``leaderboard_`` populated.

        Raises
        ------
        RuntimeError
            If every candidate failed, leaving nothing to return.
        """
        started = time.time()
        X = np.asarray(X)
        y = np.asarray(y)
        seed = 42 if self.random_state is None else int(self.random_state)
        rng = np.random.RandomState(seed)

        self.metric_name_, self._scorer = _resolve_metric(
            self.metric, self._default_metric
        )
        if self._task == "classification":
            self.classes_ = np.unique(y)

        portfolio = build_portfolio(
            self._task,
            candidates=self.candidates,
            max_candidates=self.max_candidates,
        )
        self.portfolio_ = [candidate.name for candidate in portfolio]

        self.trials_: List[Dict[str, Any]] = []
        self.eliminated_: List[str] = []
        self.rungs_ = 0
        self.survivors_: List[str] = []
        self._pool: List[Dict[str, Any]] = []
        splits = self._make_splits(X, y, seed)

        # Pass one: defaults for every candidate.
        #
        # A candidate that has never run has no cost history, so
        # ``_expected_seconds`` reports 0 for it and cannot gate anything here.
        # The stand-in is the worst duration any trial has taken so far: it is
        # the only evidence available about how expensive an unseen learner on
        # this dataset might be. Checking merely that the budget has not
        # *already* expired is not enough -- that admits a fresh candidate at
        # the last moment, and one slow learner entered just under the wire
        # runs to completion and overshoots the budget several times over.
        # Admitting only while a worst-case trial still fits bounds the
        # overshoot to roughly one trial, which is the honest guarantee: fits
        # are not interruptible, so no check between trials can do better.
        for candidate in portfolio:
            remaining = self.time_budget - (time.time() - started)
            if remaining <= 0:
                break
            if self._worst_observed() > remaining:
                continue
            self._run_trial(candidate, {}, X, y, splits, started)

        # Pass two: successive halving over candidates, Bayesian within each.
        #
        # Two decisions are separated here. *Which* candidate to spend the
        # next fit on is an allocation problem, solved by halving: every
        # survivor gets the same number of draws in a rung, the weakest
        # fraction is then dropped, and the next rung gives the survivors
        # more draws each. The field narrows geometrically, so budget flows
        # to the algorithms that are actually winning instead of being split
        # evenly across a field that already contains obvious losers.
        #
        # *Where* to sample inside a surviving candidate's space is a
        # modelling problem, handed to SurrogateSampler -- TuiML's own
        # Gaussian process and acquisition function, conditioned on that
        # candidate's own history. Until it has enough observations it draws
        # at random, so a candidate eliminated in an early rung costs no more
        # than random search would have.
        # A factor below 1 would divide by zero when sizing the next rung,
        # and the schema's ``minimum`` is advisory: it documents the contract
        # for a spec, it does not enforce it on a direct constructor call.
        halving = max(1, int(self.halving_factor))
        by_name = {candidate.name: candidate for candidate in portfolio}
        scored = self._candidate_best()
        survivors = [
            name
            for name, _ in sorted(scored.items(), key=lambda item: -item[1])
        ]
        spaces: Dict[str, Any] = {}
        samplers: Dict[str, SurrogateSampler] = {}
        rung = 0

        while survivors and time.time() - started < self.time_budget:
            # Draws per survivor this rung. It grows as the field shrinks, so
            # the last candidate standing inherits the whole remaining budget.
            per_candidate = max(1, halving ** rung)
            progressed = False

            for name in list(survivors):
                candidate = by_name[name]
                if name not in spaces:
                    spaces[name] = search_space_for(
                        candidate.cls, exclude=BUDGET_PARAMETERS
                    )
                space = spaces[name]
                if not space.param_distributions:
                    # Nothing tunable: its default score from pass one is the
                    # best it will ever do, so it stays on the leaderboard but
                    # takes no further budget.
                    survivors.remove(name)
                    continue
                if name not in samplers:
                    samplers[name] = SurrogateSampler(
                        space, acquisition=self.acquisition
                    )
                sampler = samplers[name]

                for _ in range(per_candidate):
                    remaining = self.time_budget - (time.time() - started)
                    if remaining <= 0:
                        break
                    if self._expected_seconds(name) * _COST_SAFETY > remaining:
                        break
                    params = _to_native(sampler.suggest(rng))
                    score = self._run_trial(
                        candidate, params, X, y, splits, started
                    )
                    sampler.observe(params, score)
                    progressed = True

            if not progressed:
                # Every survivor is now too expensive for the time left;
                # another rung would only spin.
                break

            if len(survivors) > 1:
                scored = self._candidate_best()
                survivors.sort(key=lambda n: -scored.get(n, -np.inf))
                keep = max(1, math.ceil(len(survivors) / halving))
                self.eliminated_.extend(survivors[keep:])
                survivors = survivors[:keep]
            rung += 1

        self.rungs_ = rung
        self.survivors_ = list(survivors)

        self._finalize(X, y, started)
        return self

    def _make_splits(self, X, y, seed):
        """Build the evaluation splits used for every trial.

        ``cv >= 2`` gives out-of-fold predictions over the whole training set;
        otherwise a single stratified 75/25 holdout is used, which costs one
        fit per trial instead of ``cv``.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training features.
        y : np.ndarray of shape (n_samples,)
            Target values.
        seed : int
            Seed for the splitter.

        Returns
        -------
        splits : list of tuple
            ``(train_index, validation_index)`` pairs.
        """
        from tuiml.evaluation.splitting import KFold, StratifiedKFold

        stratify = self._task == "classification"
        if self.cv and int(self.cv) >= 2:
            splitter_cls = StratifiedKFold if stratify else KFold
            splitter = splitter_cls(
                n_splits=int(self.cv), shuffle=True, random_state=seed
            )
            return list(splitter.split(X, y))

        # A single fold of a 4-way split is a 75/25 holdout, and reusing the
        # splitter keeps stratification behaviour identical to the CV path.
        splitter_cls = StratifiedKFold if stratify else KFold
        splitter = splitter_cls(n_splits=4, shuffle=True, random_state=seed)
        return [next(iter(splitter.split(X, y)))]

    def _run_trial(self, candidate, params, X, y, splits, started) -> Optional[float]:
        """Evaluate one (algorithm, parameters) pair and record the result.

        Parameters
        ----------
        candidate : Candidate
            The algorithm to try.
        params : dict
            Constructor parameters for this trial.
        X : np.ndarray of shape (n_samples, n_features)
            Training features.
        y : np.ndarray of shape (n_samples,)
            Target values.
        splits : list of tuple
            Evaluation splits from :meth:`_make_splits`.
        started : float
            Wall-clock time the search began, for the elapsed column.

        Returns
        -------
        score : float or None
            The score achieved, or None if the trial raised. Returned so the
            caller can feed it back to the surrogate that proposed it.
        """
        trial_started = time.time()
        record: Dict[str, Any] = {
            "name": candidate.name,
            "params": _to_native(params),
            "score": None,
            "seconds": 0.0,
            "elapsed": round(trial_started - started, 3),
            "error": None,
        }
        try:
            score, validation, model = self._evaluate(candidate.cls, params, X, y, splits)
            record["score"] = score
            self._pool.append(
                {
                    "name": candidate.name,
                    "params": record["params"],
                    "score": score,
                    "validation": validation,
                    "model": model,
                }
            )
        except Exception as error:  # one broken candidate must not end the run
            record["error"] = f"{type(error).__name__}: {error}"
        record["seconds"] = round(time.time() - trial_started, 3)
        self.trials_.append(record)
        if self.verbose:
            print(
                f"[automl] {record['name']:<32} "
                f"score={record['score']!s:<10} {record['seconds']}s "
                f"{record['error'] or ''}"
            )
        return record["score"]

    def _evaluate(self, cls, params, X, y, splits):
        """Score one configuration and refit it on all the data.

        Parameters
        ----------
        cls : type
            The algorithm class.
        params : dict
            Constructor parameters.
        X : np.ndarray of shape (n_samples, n_features)
            Training features.
        y : np.ndarray of shape (n_samples,)
            Target values.
        splits : list of tuple
            Evaluation splits.

        Returns
        -------
        score : float
            Validation score, higher is better.
        validation : dict
            ``{"index": ..., "prediction": ...}`` -- the validation rows and
            the model's predictions on them, kept for ensemble selection.
        model : Algorithm
            The configuration refitted on the full training set.
        """
        index_parts, prediction_parts = [], []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for train_index, validation_index in splits:
                fold_model = cls(**params)
                fold_model.fit(X[train_index], y[train_index])
                index_parts.append(validation_index)
                prediction_parts.append(
                    self._predictions(fold_model, X[validation_index])
                )
            model = cls(**params)
            model.fit(X, y)

        index = np.concatenate(index_parts)
        prediction = np.concatenate(prediction_parts, axis=0)
        score = self._scorer(y[index], self._decode(prediction))
        if not np.isfinite(score):
            raise ValueError("scorer returned a non-finite value")
        return float(score), {"index": index, "prediction": prediction}, model

    def _finalize(self, X, y, started) -> None:
        """Rank the trials, pick the winner, and build ``best_spec_``.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training features (unused; kept for symmetry with ``fit``).
        y : np.ndarray of shape (n_samples,)
            Target values, used as the ensemble-selection ground truth.
        started : float
            Wall-clock time the search began.
        """
        if not self._pool:
            errors = {trial["name"]: trial["error"] for trial in self.trials_}
            raise RuntimeError(
                f"Every candidate failed, so there is no model to return. "
                f"Errors per candidate: {errors}"
            )

        self._pool.sort(key=lambda entry: -entry["score"])
        best = self._pool[0]
        self.best_name_ = best["name"]
        self.best_params_ = best["params"]
        self.best_score_ = best["score"]
        self.best_estimator_ = best["model"]
        self.leaderboard_ = [
            {
                "rank": position + 1,
                "name": entry["name"],
                "score": entry["score"],
                "params": entry["params"],
            }
            for position, entry in enumerate(self._pool)
        ]
        self.failed_ = [
            {"name": trial["name"], "error": trial["error"]}
            for trial in self.trials_
            if trial["error"]
        ]

        self.ensemble_ = None
        self.ensemble_score_ = None
        if self.ensemble and len(self._pool) > 1:
            self._build_ensemble(y)

        self.best_spec_ = self._make_spec()
        self.search_time_ = round(time.time() - started, 3)
        self._is_fitted = True

    def _build_ensemble(self, y) -> None:
        """Run greedy selection over the trial pool and keep it if it wins.

        Only trials evaluated on the *same* validation rows can be compared
        and averaged, so the pool is restricted to the dominant index layout
        before selection.

        Parameters
        ----------
        y : np.ndarray of shape (n_samples,)
            Target values.
        """
        reference = self._pool[0]["validation"]["index"]
        usable = [
            entry
            for entry in self._pool
            if entry["validation"]["index"].shape == reference.shape
            and np.array_equal(entry["validation"]["index"], reference)
        ]
        if len(usable) < 2:
            return

        order = np.argsort(reference)
        y_validation = y[reference][order]
        predictions = [
            entry["validation"]["prediction"][order] for entry in usable
        ]
        try:
            weights, score, _ = greedy_selection(
                predictions,
                y_validation,
                self._scorer,
                n_rounds=self.n_ensemble_rounds,
                decode=self._decode,
            )
        except ValueError:
            return

        if score <= self.best_score_:
            return

        self.ensemble_ = GreedyEnsemble(
            [entry["model"] for entry in usable],
            weights,
            task=self._task,
            classes=getattr(self, "classes_", None),
        )
        self.ensemble_score_ = float(score)

    def _make_spec(self) -> Dict[str, Any]:
        """Return the winning single model as a :func:`tuiml.train` spec.

        The spec always describes the best **single** model, even when an
        ensemble is doing the predicting: a spec names one model, and an
        ensemble of trial-fitted members is not expressible as one. Use
        ``ensemble_`` for that; use ``best_spec_`` for the portable result.

        Returns
        -------
        spec : dict
            ``{"model": ..., "pipeline": [], "evaluation": ...}``, all
            JSON-compatible.
        """
        evaluation: Dict[str, Any] = (
            {"cv": int(self.cv)} if self.cv and int(self.cv) >= 2
            else {"test_size": 0.25, "stratify": self._task == "classification"}
        )
        if self.metric_name_:
            evaluation["metrics"] = [self.metric_name_]
        return {
            "model": {"name": self.best_name_, "params": dict(self.best_params_)},
            "pipeline": [],
            "evaluation": evaluation,
        }

    # -- helpers ----------------------------------------------------------

    def _candidate_best(self) -> Dict[str, float]:
        """Return each candidate's best score so far.

        Returns
        -------
        scores : dict
            Registry name to the highest score that candidate has reached.
        """
        best: Dict[str, float] = {}
        for entry in self._pool:
            name = entry["name"]
            if entry["score"] > best.get(name, -np.inf):
                best[name] = entry["score"]
        return best

    def _expected_seconds(self, name: str) -> float:
        """Return the longest a candidate has taken so far, in seconds.

        The maximum, not the mean: it is the pessimistic estimate that keeps
        the budget from being blown by one unusually slow configuration.

        Parameters
        ----------
        name : str
            Registry name of the candidate.

        Returns
        -------
        seconds : float
            Worst duration observed for this candidate, falling back to the
            worst observed for any candidate when it has never run, and to 0
            when nothing has run at all.
        """
        durations = [
            trial["seconds"] for trial in self.trials_ if trial["name"] == name
        ]
        if durations:
            return max(durations)
        # No history for this candidate: fall back to the worst any trial has
        # taken. Returning 0 here would make every caller's guard trivially
        # true, so a never-run candidate would be admitted no matter how
        # little budget was left -- which is exactly how an expensive learner
        # ends up starting in the last fraction of a second and running to
        # completion long after the budget expired.
        return self._worst_observed()

    def _worst_observed(self) -> float:
        """Return the longest any trial has taken so far, in seconds.

        The cost estimate for a candidate that has never run, used to decide
        whether starting it can still fit inside the remaining budget. It is
        deliberately the maximum over every trial rather than the mean, since
        the risk being guarded against is one unusually slow learner rather
        than the typical case.

        Returns
        -------
        seconds : float
            Worst observed trial duration, or 0 if nothing has run yet.
        """
        return max((trial["seconds"] for trial in self.trials_), default=0.0)

    def _check_fitted(self) -> None:
        """Raise ``RuntimeError`` if the search has not been run yet."""
        if not self._is_fitted:
            raise RuntimeError(
                f"{type(self).__name__} must be fitted before predicting."
            )

    @property
    def final_estimator_(self):
        """The predictor actually used: the ensemble if it won, else the model."""
        self._check_fitted()
        return self.ensemble_ if self.ensemble_ is not None else self.best_estimator_

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict with the best model (or the ensemble, when it won).

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        y_pred : np.ndarray of shape (n_samples,)
            Predictions.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return self.final_estimator_.predict(np.asarray(X))


@classifier(tags=["automl", "meta", "search"], version="1.0.0")
class AutoMLClassifier(_AutoMLBase, Classifier):
    """Automatic classifier selection and tuning under a time budget.

    Searches TuiML's own classifiers -- no external AutoML backend -- and
    returns the best one it found, plus everything needed to reproduce it.

    Overview
    --------
    1. Rank the library's classifiers cheap-and-strong first
       (:func:`~tuiml.automl.portfolio.build_portfolio`).
    2. Evaluate every candidate at its defaults, then narrow the field by
       successive halving while a Gaussian process picks where to sample
       configurations derived from each algorithm's own parameter schema
       (:func:`~tuiml.automl.search_space.search_space_for`) until the
       wall-clock budget is spent.
    3. Optionally combine the fitted trials with greedy ensemble selection
       (:func:`~tuiml.automl.ensembling.greedy_selection`).
    4. Publish the winner as ``best_spec_``, a :func:`tuiml.train` spec.

    Parameters
    ----------
    time_budget : float, default=60.0
        Wall-clock seconds the search may spend. A model fit cannot be
        interrupted once started, so the budget bounds when a *new* trial may
        begin, not when the last one ends: a candidate is admitted only while
        a worst-case trial still fits in the time left. Overshoot is therefore
        bounded by roughly one trial, and is largest when the first expensive
        candidate is admitted on the evidence of cheap ones.
    metric : str or callable, default=None
        Metric to optimise: a function name from
        :mod:`tuiml.evaluation.metrics`, or a callable
        ``(y_true, y_pred) -> float``. Loss-style names are negated
        automatically. Defaults to ``"accuracy_score"``.
    cv : int or None, default=3
        Cross-validation folds used to score each trial. ``None`` (or 1)
        switches to a single stratified 75/25 holdout, which is roughly
        ``cv`` times cheaper per trial.
    random_state : int, optional
        Seed for the splits and for hyperparameter sampling.
    ensemble : bool, default=True
        Run greedy ensemble selection over the trial pool and predict with
        the ensemble if it beats the best single model.
    candidates : sequence of str or type, optional
        Explicit candidates to search, overriding portfolio discovery. Order
        is respected.
    max_candidates : int, optional
        Truncate the discovered portfolio to its top ``max_candidates``.
    n_ensemble_rounds : int, default=25
        Greedy selection rounds (the ensemble size counted with multiplicity).
    halving_factor : int, default=3
        Aggressiveness of the candidate elimination in pass two. Each rung
        keeps the top ``1 / halving_factor`` of the field and multiplies the
        draws per survivor by the same amount. ``1`` disables elimination and
        gives every candidate an equal share of the budget, which is what
        plain round-robin random search does.
    acquisition : {'ei', 'pi', 'ucb'}, default='ei'
        Acquisition policy the surrogate maximises when proposing the next
        configuration. See
        :class:`~tuiml.evaluation.tuning.bayesian_search.AcquisitionFunction`.
    verbose : int, default=0
        Print one line per trial when greater than 0.

    Attributes
    ----------
    best_spec_ : dict
        The winning single model as a :func:`tuiml.train` spec:
        ``{"model": {"name": ..., "params": {...}}, "pipeline": [],
        "evaluation": {...}}``. JSON-serialisable, so a search result can be
        committed and replayed.
    best_estimator_ : Classifier
        The best single model, refitted on all the training data.
    best_name_, best_params_, best_score_ : str, dict, float
        The winner's registry name, parameters, and validation score.
    leaderboard_ : list of dict
        Every successful trial as ``{"rank", "name", "score", "params"}``,
        best first.
    failed_ : list of dict
        Candidates that raised, with the error text.
    ensemble_ : GreedyEnsemble or None
        The selected ensemble, when ``ensemble=True`` and it beat the best
        single model.
    portfolio_ : list of str
        The candidates the search considered, in the order it tried them.
    eliminated_ : list of str
        Candidates dropped by halving, in the order they were cut. Reading it
        against ``leaderboard_`` shows where the budget stopped going.
    survivors_ : list of str
        Candidates still alive when the budget ran out.
    rungs_ : int
        Halving rungs completed.
    search_time_ : float
        Wall-clock seconds the search actually used.

    Notes
    -----
    **Complexity.** One fit per trial (``cv + 1`` fits when ``cv >= 2``); the
    trial count is whatever the budget allows. Ensemble selection is
    :math:`O(R \\cdot M \\cdot n)` for :math:`R` rounds, :math:`M` pool
    members and :math:`n` validation rows, with no model fitting at all.

    **When to use.** As a strong first pass on a new tabular dataset, or to
    generate a starting ``train()`` spec to hand-tune afterwards. When the
    algorithm is already known, tune it directly with
    :class:`~tuiml.evaluation.tuning.RandomSearchCV` and skip the portfolio.

    References
    ----------
    .. [Caruana2004] Caruana, R., Niculescu-Mizil, A., Crew, G., & Ksikes, A.
       (2004). Ensemble selection from libraries of models. *Proceedings of
       the 21st International Conference on Machine Learning (ICML)*, 18.
       :doi:`10.1145/1015330.1015432`
    .. [Feurer2015] Feurer, M., Klein, A., Eggensperger, K., Springenberg, J.,
       Blum, M., & Hutter, F. (2015). Efficient and robust automated machine
       learning. *Advances in Neural Information Processing Systems*, 28.

    See Also
    --------
    :class:`~tuiml.automl.automl.AutoMLRegressor` : The regression counterpart.
    :func:`~tuiml.automl.search_space.search_space_for` : Derives the
        per-algorithm search space.
    :func:`~tuiml.automl.portfolio.build_portfolio` : Chooses and ranks the
        candidates.

    Examples
    --------
    >>> from tuiml.automl import AutoMLClassifier
    >>> from tuiml.datasets import load_iris
    >>> data = load_iris()
    >>> automl = AutoMLClassifier(time_budget=5, cv=3, random_state=0).fit(
    ...     data.X, data.y)
    >>> automl.best_spec_["model"]["name"]                  # doctest: +SKIP
    'LogisticRegression'
    >>> automl.leaderboard_[0]["score"] > 0.9
    True

    The spec is the deliverable -- replay it with :func:`tuiml.train`:

    >>> import tuiml
    >>> spec = dict(automl.best_spec_, data={"X": data.X, "y": data.y})
    >>> model = tuiml.train(spec)                            # doctest: +SKIP
    """

    _task = "classification"
    _default_metric = "accuracy_score"

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities with the winning model or ensemble.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        proba : np.ndarray of shape (n_samples, n_classes)
            Class probabilities, columns ordered as ``classes_``.
        """
        self._check_fitted()
        estimator = self.final_estimator_
        if isinstance(estimator, GreedyEnsemble):
            return estimator.predict_proba(np.asarray(X))
        return align_proba(estimator, np.asarray(X), self.classes_)

    def _predictions(self, model, X) -> np.ndarray:
        """Return a model's validation predictions as a probability matrix."""
        return align_proba(model, X, self.classes_)

    def _decode(self, prediction: np.ndarray) -> np.ndarray:
        """Convert a probability matrix to hard class labels."""
        return self.classes_[np.argmax(prediction, axis=1)]

    @classmethod
    def get_capabilities(cls) -> List[str]:
        """Return classifier capabilities.

        A search is only as capable as the portfolio it draws from, so this is
        the intersection of what every candidate classifier handles rather than
        the union: ``missing_values`` is absent because a portfolio member that
        cannot impute would fail on data this claimed to accept.
        """
        return [
            "numeric",
            "nominal",
            "binary_class",
            "multiclass",
            "probabilistic",
            "ensemble",
        ]


@regressor(tags=["automl", "meta", "search"], version="1.0.0")
class AutoMLRegressor(_AutoMLBase, Regressor):
    """Automatic regressor selection and tuning under a time budget.

    The regression counterpart of
    :class:`~tuiml.automl.automl.AutoMLClassifier`: same three-stage search
    (portfolio, schema-derived sampling, greedy ensembling) and the same
    ``best_spec_`` deliverable, scoring with :math:`R^2` by default and
    averaging predicted values rather than class probabilities.

    Parameters
    ----------
    time_budget : float, default=60.0
        Wall-clock seconds the search may spend. A model fit cannot be
        interrupted once started, so the budget bounds when a *new* trial may
        begin, not when the last one ends: a candidate is admitted only while
        a worst-case trial still fits in the time left. Overshoot is therefore
        bounded by roughly one trial, and is largest when the first expensive
        candidate is admitted on the evidence of cheap ones.
    metric : str or callable, default=None
        Metric to optimise, from :mod:`tuiml.evaluation.metrics` or a
        callable. Error metrics such as ``"mean_squared_error"`` are negated
        automatically so that higher always means better. Defaults to
        ``"r2_score"``.
    cv : int or None, default=3
        Cross-validation folds per trial; ``None`` uses a 75/25 holdout.
    random_state : int, optional
        Seed for the splits and for hyperparameter sampling.
    ensemble : bool, default=True
        Average the trial pool by greedy selection when that scores better.
    candidates : sequence of str or type, optional
        Explicit candidates, overriding portfolio discovery.
    max_candidates : int, optional
        Truncate the discovered portfolio.
    n_ensemble_rounds : int, default=25
        Greedy selection rounds.
    verbose : int, default=0
        Print one line per trial when greater than 0.

    Attributes
    ----------
    best_spec_ : dict
        The winning model as a :func:`tuiml.train` spec.
    best_estimator_ : Regressor
        The best single model, refitted on all the training data.
    best_name_, best_params_, best_score_ : str, dict, float
        The winner's registry name, parameters, and validation score.
    leaderboard_ : list of dict
        Every successful trial, best first.
    failed_ : list of dict
        Candidates that raised, with the error text.
    ensemble_ : GreedyEnsemble or None
        The selected ensemble, when it beat the best single model.
    portfolio_ : list of str
        Candidates considered, in the order they were tried.
    search_time_ : float
        Wall-clock seconds actually used.

    Notes
    -----
    **Complexity.** ``cv + 1`` fits per trial; the trial count is set by the
    budget. Ensemble selection fits nothing.

    **When to use.** As a first pass on a new tabular regression problem.
    For a known algorithm, tune it directly with
    :class:`~tuiml.evaluation.tuning.RandomSearchCV`.

    References
    ----------
    .. [Caruana2004] Caruana, R., Niculescu-Mizil, A., Crew, G., & Ksikes, A.
       (2004). Ensemble selection from libraries of models. *Proceedings of
       the 21st International Conference on Machine Learning (ICML)*, 18.
       :doi:`10.1145/1015330.1015432`

    See Also
    --------
    :class:`~tuiml.automl.automl.AutoMLClassifier` : The classification
        counterpart.
    :func:`~tuiml.automl.portfolio.build_portfolio` : Chooses and ranks the
        candidates.

    Examples
    --------
    >>> from tuiml.automl import AutoMLRegressor
    >>> from tuiml.datasets import load_cpu
    >>> data = load_cpu()
    >>> automl = AutoMLRegressor(time_budget=5, cv=3, random_state=0).fit(
    ...     data.X, data.y)
    >>> automl.best_score_ > 0.5
    True
    >>> sorted(automl.best_spec_)
    ['evaluation', 'model', 'pipeline']
    """

    _task = "regression"
    _default_metric = "r2_score"

    def _predictions(self, model, X) -> np.ndarray:
        """Return a model's validation predictions as a float vector."""
        return np.asarray(model.predict(X), dtype=float)

    def _decode(self, prediction: np.ndarray) -> np.ndarray:
        """Return the predicted values unchanged (no decoding needed)."""
        return prediction

    @classmethod
    def get_capabilities(cls) -> List[str]:
        """Return regressor capabilities.

        As with the classifier, this is the intersection over the portfolio
        rather than the union: a capability is only claimed when every
        candidate the search may select can honour it.
        """
        return [
            "numeric",
            "nominal",
            "numeric_class",
            "regression",
            "ensemble",
        ]


__all__ = ["AutoMLClassifier", "AutoMLRegressor"]
