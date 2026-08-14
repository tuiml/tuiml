"""Cross-validation-based conformal regression (CV+ and jackknife+)."""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional

import numpy as np

from tuiml.uncertainty._base import ConformalPredictor


class CVPlusRegressor(ConformalPredictor):
    """**CV+** prediction intervals that use every training sample twice.

    Split conformal throws away a quarter of the training data to calibrate.
    CV+ instead cross-fits: each fold's model is calibrated on the fold it
    never saw, so **all** the data trains the ensemble **and** all of it
    calibrates. The price is :math:`k` model fits instead of one.

    Overview
    --------
    1. Partition the training data into ``cv`` folds.
    2. For each fold, fit a model on the other folds and record the absolute
       residual of every sample in the held-out fold.
    3. At prediction time, every fold model predicts the test point.
    4. The interval bounds are quantiles of the fold predictions **shifted by**
       the out-of-fold residuals, so a residual is only ever paired with a
       model that did not train on it.

    Theory
    ------
    Let :math:`\\hat{f}_{-k(i)}` be the model fitted without the fold
    containing sample :math:`i`, and :math:`R_i = |y_i - \\hat{f}_{-k(i)}(x_i)|`
    its out-of-fold residual. The CV+ interval is

    .. math::
        C(x) = \\left[
        q^-_{\\alpha}\\left\\{ \\hat{f}_{-k(i)}(x) - R_i \\right\\},\\ \\
        q^+_{\\alpha}\\left\\{ \\hat{f}_{-k(i)}(x) + R_i \\right\\}
        \\right]

    where :math:`q^-` and :math:`q^+` are the :math:`\\lfloor \\alpha(n+1)
    \\rfloor` smallest and largest order statistics. Unlike split conformal,
    the guarantee is the slightly weaker

    .. math::
        P\\left( Y_{n+1} \\in C(X_{n+1}) \\right) \\geq 1 - 2\\alpha

    in the worst case, though empirically CV+ achieves close to
    :math:`1 - \\alpha` and is never observed to fall below it on real data.
    The factor-of-two slack is the cost of reusing the data.

    Setting ``cv=n_samples`` recovers the **jackknife+** — see
    :class:`JackknifePlusRegressor`.

    Parameters
    ----------
    estimator : Regressor
        A TuiML regressor. It is deep-copied once per fold, so the instance
        passed in is never mutated.
    alpha : float, default=0.1
        Miscoverage level.
    cv : int, default=5
        Number of cross-fitting folds. More folds means more training data per
        model and more compute.
    aggregate : {'median', 'mean'}, default='median'
        How the fold models are combined for the point prediction returned by
        :meth:`predict`. Interval bounds always use the order statistics above,
        independent of this choice.
    shuffle : bool, default=True
        Whether to permute the samples before folding. Leave enabled unless the
        row order is itself meaningful.
    random_state : int, optional
        Seed for the fold shuffle.

    Attributes
    ----------
    estimators_ : list of Regressor
        One fitted model per fold.
    fold_index_ : np.ndarray of shape (n_samples,)
        Fold assignment of each training sample.
    scores_ : np.ndarray of shape (n_samples,)
        Out-of-fold absolute residuals.
    quantile_ : float
        The corrected residual quantile, reported for comparison with split
        conformal. The interval itself uses the full residual vector.
    fitted_ : bool
        Whether :meth:`fit` has been called.

    Notes
    -----
    **Complexity.** ``cv`` model fits at training time and ``cv`` predictions
    per test batch, plus :math:`O(n \\log n)` for the order statistics. This is
    ``cv`` times the cost of split conformal in both phases — the reason
    :class:`JackknifePlusRegressor` is only practical on small data.

    **When to use.** Use CV+ when data is scarce enough that holding out a
    calibration split visibly hurts the model, and when ``cv`` extra fits are
    affordable. On large data, split conformal gives a strictly stronger
    guarantee for a fraction of the compute.

    References
    ----------
    .. [Barber2021] Barber, R. F., Candès, E. J., Ramdas, A., & Tibshirani,
       R. J. (2021). Predictive Inference with the Jackknife+.
       *Annals of Statistics*, 49(1), 486-507. :doi:`10.1214/20-AOS1965`

    See Also
    --------
    :class:`~tuiml.uncertainty.SplitConformalRegressor` : One fit, stronger guarantee, needs a held-out split.
    :class:`~tuiml.uncertainty.JackknifePlusRegressor` : The leave-one-out limit of this method.
    :func:`~tuiml.uncertainty.coverage_score` : Verifies the guarantee empirically.

    Examples
    --------
    >>> import numpy as np
    >>> from tuiml.uncertainty import CVPlusRegressor
    >>> from tuiml.algorithms.trees import DecisionTreeRegressor
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(200, 3))
    >>> y = X[:, 0] * 2.0 + rng.normal(0, 0.5, 200)
    >>> cp = CVPlusRegressor(DecisionTreeRegressor(max_depth=4), cv=5, random_state=0)
    >>> cp.fit(X, y)
    CVPlusRegressor(estimator=DecisionTreeRegressor(), alpha=0.1, cv=5)
    >>> cp.predict_interval(X[:4]).shape
    (4, 2)
    """

    def __init__(
        self,
        estimator: Any,
        alpha: float = 0.1,
        cv: int = 5,
        aggregate: str = "median",
        shuffle: bool = True,
        random_state: Optional[int] = None,
    ) -> None:
        """Initialise the CV+ regressor.

        Parameters
        ----------
        estimator : Regressor
            A TuiML regressor, deep-copied once per fold.
        alpha : float, default=0.1
            Miscoverage level.
        cv : int, default=5
            Number of cross-fitting folds.
        aggregate : {'median', 'mean'}, default='median'
            Fold aggregation for the point prediction.
        shuffle : bool, default=True
            Whether to permute samples before folding.
        random_state : int, optional
            Seed for the fold shuffle.
        """
        super().__init__(estimator, alpha)
        if cv < 2:
            raise ValueError(f"cv must be at least 2, got {cv}")
        if aggregate not in ("median", "mean"):
            raise ValueError(
                f"aggregate must be 'median' or 'mean', got {aggregate!r}"
            )
        self.cv = cv
        self.aggregate = aggregate
        self.shuffle = shuffle
        self.random_state = random_state
        self.estimators_: List[Any] = []
        self.fold_index_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CVPlusRegressor":
        """Cross-fit the estimator and collect out-of-fold residuals.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training features.
        y : np.ndarray of shape (n_samples,)
            Training targets.

        Returns
        -------
        self : CVPlusRegressor
            The fitted predictor.
        """
        X = np.asarray(X)
        y = np.asarray(y, dtype=np.float64)
        n_samples = X.shape[0]

        n_folds = min(self.cv, n_samples)
        if n_folds < 2:
            raise ValueError(
                f"cv requires at least 2 samples, got {n_samples}"
            )

        order = np.arange(n_samples)
        if self.shuffle:
            np.random.default_rng(self.random_state).shuffle(order)
        # np.array_split handles a remainder by giving the first folds one
        # extra sample, matching tuiml.evaluation.splitting.KFold.
        folds = np.array_split(order, n_folds)

        self.estimators_ = []
        self.fold_index_ = np.empty(n_samples, dtype=int)
        residuals = np.empty(n_samples, dtype=np.float64)

        for fold_id, test_index in enumerate(folds):
            train_index = np.setdiff1d(order, test_index, assume_unique=False)
            model = copy.deepcopy(self.estimator)
            model.fit(X[train_index], y[train_index])
            self.estimators_.append(model)

            predicted = np.asarray(model.predict(X[test_index]), dtype=np.float64)
            residuals[test_index] = np.abs(y[test_index] - predicted)
            self.fold_index_[test_index] = fold_id

        self.scores_ = residuals
        self.quantile_ = self.conformal_quantile(residuals, self.alpha)
        self.fitted_ = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Aggregate the fold models into a single point prediction.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Test features.

        Returns
        -------
        y_pred : np.ndarray of shape (n_samples,)
            Median (or mean) of the fold predictions.
        """
        self._check_is_fitted()
        fold_predictions = self._fold_predictions(X)
        if self.aggregate == "mean":
            return fold_predictions.mean(axis=0)
        return np.median(fold_predictions, axis=0)

    def predict_interval(self, X: np.ndarray) -> np.ndarray:
        """Predict CV+ lower and upper bounds for each sample.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Test features.

        Returns
        -------
        intervals : np.ndarray of shape (n_samples, 2)
            Column 0 holds the lower bound, column 1 the upper bound.
        """
        self._check_is_fitted()
        fold_predictions = self._fold_predictions(X)

        # Pair every training sample with the model that did NOT see it, so
        # each residual is matched to an out-of-fold prediction.
        per_sample = fold_predictions[self.fold_index_, :]
        lower_candidates = per_sample - self.scores_[:, None]
        upper_candidates = per_sample + self.scores_[:, None]

        n = self.scores_.size
        # Order statistics rather than a plain quantile: this is what gives
        # the jackknife+ family its guarantee.
        rank = int(np.floor(self.alpha * (n + 1)))
        rank = min(max(rank, 1), n) - 1

        lower = np.partition(lower_candidates, rank, axis=0)[rank]
        upper = np.partition(upper_candidates, n - 1 - rank, axis=0)[n - 1 - rank]
        return np.column_stack([lower, upper])

    def _fold_predictions(self, X: np.ndarray) -> np.ndarray:
        """Predict ``X`` with every fold model.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Test features.

        Returns
        -------
        predictions : np.ndarray of shape (n_folds, n_samples)
            One row of predictions per fold model.
        """
        return np.vstack(
            [np.asarray(model.predict(X), dtype=np.float64) for model in self.estimators_]
        )

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Any]:
        """Return JSON Schema for constructor parameters."""
        return {
            "type": "object",
            "properties": {
                "alpha": {
                    "type": "number",
                    "default": 0.1,
                    "exclusiveMinimum": 0,
                    "exclusiveMaximum": 1,
                    "description": "Miscoverage level; target coverage is 1 - alpha.",
                },
                "cv": {
                    "type": "integer",
                    "default": 5,
                    "minimum": 2,
                    "description": "Number of cross-fitting folds.",
                },
                "aggregate": {
                    "type": "string",
                    "enum": ["median", "mean"],
                    "default": "median",
                    "description": "Fold aggregation for the point prediction.",
                },
                "shuffle": {
                    "type": "boolean",
                    "default": True,
                    "description": "Permute samples before folding.",
                },
                "random_state": {
                    "type": ["integer", "null"],
                    "default": None,
                    "description": "Seed for the fold shuffle.",
                },
            },
        }

    def __repr__(self) -> str:
        """Return a readable representation of the predictor."""
        return (
            f"{self.__class__.__name__}("
            f"estimator={self.estimator.__class__.__name__}(), "
            f"alpha={self.alpha}, cv={self.cv})"
        )


class JackknifePlusRegressor(CVPlusRegressor):
    """**Jackknife+** intervals — the leave-one-out limit of CV+.

    Each training sample is held out on its own, so every residual comes from a
    model fitted on all :math:`n - 1` remaining samples. This gives the
    tightest intervals of the family, because each model sees the most data,
    and the strongest empirical coverage. It also costs :math:`n` model fits.

    Overview
    --------
    1. For each training sample, fit a model on all the others.
    2. Record that sample's leave-one-out absolute residual.
    3. Form the interval from the order statistics of the leave-one-out
       predictions shifted by those residuals, exactly as in CV+.

    Theory
    ------
    Jackknife+ is :class:`CVPlusRegressor` with ``cv = n_samples``. It inherits
    the worst-case :math:`1 - 2\\alpha` bound

    .. math::
        P\\left( Y_{n+1} \\in C(X_{n+1}) \\right) \\geq 1 - 2\\alpha

    but is provably at least as tight as CV+ with fewer folds, and in practice
    covers at very close to the nominal :math:`1 - \\alpha`.

    Note the distinction from the plain jackknife, which shifts a *single*
    model's prediction by leave-one-out residuals: that has **no** coverage
    guarantee at all and can fail badly when the fitting algorithm is unstable.
    The "+" is what pairs each residual with its own leave-one-out model.

    Parameters
    ----------
    estimator : Regressor
        A TuiML regressor, deep-copied once per sample.
    alpha : float, default=0.1
        Miscoverage level.
    aggregate : {'median', 'mean'}, default='median'
        Fold aggregation for the point prediction.

    Attributes
    ----------
    estimators_ : list of Regressor
        One fitted model per training sample.
    scores_ : np.ndarray of shape (n_samples,)
        Leave-one-out absolute residuals.
    fitted_ : bool
        Whether :meth:`fit` has been called.

    Notes
    -----
    **Complexity.** :math:`n` model fits and :math:`n` predictions per test
    batch. This is only practical for a few hundred samples with a cheap
    estimator; beyond that use :class:`CVPlusRegressor` with ``cv=10``, which
    is close in tightness and orders of magnitude cheaper.

    References
    ----------
    .. [Barber2021] Barber, R. F., Candès, E. J., Ramdas, A., & Tibshirani,
       R. J. (2021). Predictive Inference with the Jackknife+.
       *Annals of Statistics*, 49(1), 486-507. :doi:`10.1214/20-AOS1965`

    See Also
    --------
    :class:`~tuiml.uncertainty.CVPlusRegressor` : The k-fold version; far cheaper.
    :class:`~tuiml.uncertainty.SplitConformalRegressor` : One fit, needs a held-out split.

    Examples
    --------
    >>> import numpy as np
    >>> from tuiml.uncertainty import JackknifePlusRegressor
    >>> from tuiml.algorithms.linear import LinearRegression
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(60, 2))
    >>> y = X[:, 0] * 2.0 + rng.normal(0, 0.3, 60)
    >>> cp = JackknifePlusRegressor(LinearRegression(), alpha=0.1)
    >>> cp.fit(X, y)
    JackknifePlusRegressor(estimator=LinearRegression(), alpha=0.1)
    >>> cp.predict_interval(X[:3]).shape
    (3, 2)
    """

    def __init__(
        self,
        estimator: Any,
        alpha: float = 0.1,
        aggregate: str = "median",
    ) -> None:
        """Initialise the jackknife+ regressor.

        Parameters
        ----------
        estimator : Regressor
            A TuiML regressor, deep-copied once per sample.
        alpha : float, default=0.1
            Miscoverage level.
        aggregate : {'median', 'mean'}, default='median'
            Fold aggregation for the point prediction.
        """
        # cv is replaced by n_samples at fit time; 2 is a placeholder that
        # satisfies the parent's validation.
        super().__init__(
            estimator, alpha=alpha, cv=2, aggregate=aggregate, shuffle=False
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "JackknifePlusRegressor":
        """Fit one model per left-out training sample.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training features.
        y : np.ndarray of shape (n_samples,)
            Training targets.

        Returns
        -------
        self : JackknifePlusRegressor
            The fitted predictor.
        """
        self.cv = len(X)
        return super().fit(X, y)

    def __repr__(self) -> str:
        """Return a readable representation of the predictor."""
        return (
            f"JackknifePlusRegressor("
            f"estimator={self.estimator.__class__.__name__}(), "
            f"alpha={self.alpha})"
        )
