"""Split (inductive) conformal prediction for classifiers and regressors."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from tuiml.uncertainty._base import ConformalPredictor, SetPredictorMixin


class SplitConformalClassifier(ConformalPredictor, SetPredictorMixin):
    """**Split conformal** prediction sets with a finite-sample coverage guarantee.

    Wraps any TuiML classifier and returns, instead of a single label, a **set
    of labels** that contains the truth with probability at least
    :math:`1 - \\alpha`. The guarantee is **distribution-free**: it needs no
    assumption about the model or the data beyond exchangeability, and holds at
    finite sample size rather than asymptotically.

    Overview
    --------
    1. Split the training data into a proper training part and a calibration
       part.
    2. Fit the wrapped estimator on the proper training part only.
    3. Score each calibration sample by its **nonconformity** — how poorly the
       model predicted its true label.
    4. Take the :math:`\\lceil (n+1)(1-\\alpha) \\rceil / n` empirical quantile
       of those scores as the threshold.
    5. A test label joins the prediction set when its nonconformity falls below
       that threshold.

    Theory
    ------
    With the least-ambiguous-set score :math:`s(x, y) = 1 - \\hat{p}_y(x)` and
    the corrected quantile :math:`\\hat{q}` of the calibration scores, the set

    .. math::
        C(x) = \\{ y : 1 - \\hat{p}_y(x) \\leq \\hat{q} \\}

    satisfies

    .. math::
        P\\left( Y_{n+1} \\in C(X_{n+1}) \\right) \\geq 1 - \\alpha

    The finite-sample correction is what makes this exact: using the plain
    :math:`1-\\alpha` quantile would undercover by roughly :math:`1/n`.

    Coverage is **marginal**, averaged over the data draw. It says nothing
    about coverage for a particular subgroup — see
    :class:`~tuiml.uncertainty.MondrianConformalClassifier` for class-conditional
    validity.

    Parameters
    ----------
    estimator : Classifier
        A TuiML classifier exposing ``predict_proba``.
    alpha : float, default=0.1
        Miscoverage level; the target coverage is ``1 - alpha``.
    score : {'lac', 'margin'}, default='lac'
        Nonconformity score. ``'lac'`` (least ambiguous set-valued classifier)
        uses :math:`1 - \\hat{p}_y` and gives the smallest average set size;
        ``'margin'`` uses the gap to the best competing class and adapts better
        to hard samples.
    calibration_size : float, default=0.25
        Fraction of the training data held out for calibration.
    random_state : int, optional
        Seed for the train/calibration split.

    Attributes
    ----------
    classes_ : np.ndarray of shape (n_classes,)
        Class labels seen during :meth:`fit`.
    scores_ : np.ndarray of shape (n_calibration,)
        Nonconformity scores on the calibration set.
    quantile_ : float
        The conformal threshold derived from ``scores_``.
    fitted_ : bool
        Whether :meth:`fit` has been called.

    Notes
    -----
    **Complexity.** One estimator fit plus :math:`O(n \\log n)` for the
    quantile. Prediction costs one ``predict_proba`` call plus :math:`O(mc)`.

    **When to use.** Use split conformal whenever a calibrated *set* is more
    useful than a point label — triage, selective prediction, or any setting
    where abstention is allowed. It is the cheapest conformal method: one model
    fit. When data is scarce and holding out 25% hurts, use
    :class:`~tuiml.uncertainty.CVPlusRegressor` or its classification analogue
    instead. Calibration needs at least :math:`\\lceil 1/\\alpha \\rceil - 1`
    samples; below that no finite threshold can certify the level and the
    predictor returns the full label set.

    References
    ----------
    .. [Vovk2005] Vovk, V., Gammerman, A., & Shafer, G. (2005).
       *Algorithmic Learning in a Random World*. Springer.
       :doi:`10.1007/b106715`
    .. [Sadinle2019] Sadinle, M., Lei, J., & Wasserman, L. (2019). Least
       Ambiguous Set-Valued Classifiers with Bounded Error Levels.
       *Journal of the American Statistical Association*, 114(525), 223-234.
       :doi:`10.1080/01621459.2017.1395341`

    See Also
    --------
    :class:`~tuiml.uncertainty.APSConformalClassifier` : Adaptive sets with better conditional coverage.
    :class:`~tuiml.uncertainty.MondrianConformalClassifier` : Class-conditional coverage.
    :class:`~tuiml.uncertainty.SplitConformalRegressor` : Prediction intervals.
    :func:`~tuiml.uncertainty.coverage_score` : Verifies the guarantee empirically.

    Examples
    --------
    >>> import numpy as np
    >>> from tuiml.uncertainty import SplitConformalClassifier
    >>> from tuiml.algorithms.trees import DecisionTreeClassifier
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(400, 4))
    >>> y = (X[:, 0] + X[:, 1] > 0).astype(int)
    >>> cp = SplitConformalClassifier(DecisionTreeClassifier(max_depth=4),
    ...                               alpha=0.1, random_state=0)
    >>> cp.fit(X, y)
    SplitConformalClassifier(estimator=DecisionTreeClassifier(), alpha=0.1)
    >>> sets = cp.predict_set(X[:5])
    >>> sets.shape
    (5, 2)
    """

    def __init__(
        self,
        estimator: Any,
        alpha: float = 0.1,
        score: str = "lac",
        calibration_size: float = 0.25,
        random_state: Optional[int] = None,
    ) -> None:
        """Initialise the split conformal classifier.

        Parameters
        ----------
        estimator : Classifier
            A TuiML classifier exposing ``predict_proba``.
        alpha : float, default=0.1
            Miscoverage level.
        score : {'lac', 'margin'}, default='lac'
            Nonconformity score.
        calibration_size : float, default=0.25
            Fraction of the training data held out for calibration.
        random_state : int, optional
            Seed for the train/calibration split.
        """
        super().__init__(estimator, alpha)
        if score not in ("lac", "margin"):
            raise ValueError(f"score must be 'lac' or 'margin', got {score!r}")
        if not 0.0 < calibration_size < 1.0:
            raise ValueError(
                f"calibration_size must be in (0, 1), got {calibration_size}"
            )
        self.score = score
        self.calibration_size = calibration_size
        self.random_state = random_state
        self.classes_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SplitConformalClassifier":
        """Fit the estimator on a training split and calibrate on the rest.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training features.
        y : np.ndarray of shape (n_samples,)
            Training labels.

        Returns
        -------
        self : SplitConformalClassifier
            The fitted predictor.
        """
        from tuiml.evaluation.splitting import train_test_split

        X = np.asarray(X)
        y = np.asarray(y)

        X_train, X_cal, y_train, y_cal = train_test_split(
            X, y, test_size=self.calibration_size, random_state=self.random_state
        )
        self.estimator.fit(X_train, y_train)
        self.classes_ = np.asarray(
            getattr(self.estimator, "classes_", np.unique(y))
        )

        proba = np.asarray(self.estimator.predict_proba(X_cal), dtype=np.float64)
        true_index = np.searchsorted(self.classes_, y_cal)
        self.scores_ = self._nonconformity(proba, true_index)
        self.quantile_ = self.conformal_quantile(self.scores_, self.alpha)
        self.fitted_ = True
        return self

    def fit_calibrated(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_cal: np.ndarray,
        y_cal: np.ndarray,
    ) -> "SplitConformalClassifier":
        """Fit with an explicit, caller-supplied calibration set.

        Use this when the calibration data must respect a grouping or temporal
        structure that a random split would break.

        Parameters
        ----------
        X_train : np.ndarray of shape (n_train, n_features)
            Proper training features.
        y_train : np.ndarray of shape (n_train,)
            Proper training labels.
        X_cal : np.ndarray of shape (n_calibration, n_features)
            Calibration features, disjoint from the training set.
        y_cal : np.ndarray of shape (n_calibration,)
            Calibration labels.

        Returns
        -------
        self : SplitConformalClassifier
            The fitted predictor.
        """
        self.estimator.fit(X_train, y_train)
        self.classes_ = np.asarray(
            getattr(self.estimator, "classes_", np.unique(y_train))
        )
        proba = np.asarray(self.estimator.predict_proba(X_cal), dtype=np.float64)
        true_index = np.searchsorted(self.classes_, np.asarray(y_cal))
        self.scores_ = self._nonconformity(proba, true_index)
        self.quantile_ = self.conformal_quantile(self.scores_, self.alpha)
        self.fitted_ = True
        return self

    def predict_set(self, X: np.ndarray) -> np.ndarray:
        """Predict a boolean class-membership mask.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Test features.

        Returns
        -------
        include : np.ndarray of shape (n_samples, n_classes) of bool
            ``include[i, k]`` is True when class ``k`` is in the prediction set
            of sample ``i``.
        """
        self._check_is_fitted()
        proba = np.asarray(self.estimator.predict_proba(X), dtype=np.float64)
        candidate_scores = self._candidate_scores(proba)
        return candidate_scores <= self.quantile_

    def _nonconformity(self, proba: np.ndarray, true_index: np.ndarray) -> np.ndarray:
        """Score how poorly the model predicted each calibration label.

        Parameters
        ----------
        proba : np.ndarray of shape (n_samples, n_classes)
            Predicted probabilities on the calibration set.
        true_index : np.ndarray of shape (n_samples,)
            Column index of the true class for each sample.

        Returns
        -------
        scores : np.ndarray of shape (n_samples,)
            Nonconformity scores; larger means a worse fit.
        """
        rows = np.arange(proba.shape[0])
        return self._candidate_scores(proba)[rows, true_index]

    def _candidate_scores(self, proba: np.ndarray) -> np.ndarray:
        """Score every class of every sample as a prediction-set candidate.

        Parameters
        ----------
        proba : np.ndarray of shape (n_samples, n_classes)
            Predicted probabilities.

        Returns
        -------
        scores : np.ndarray of shape (n_samples, n_classes)
            Nonconformity of each candidate label.
        """
        if self.score == "lac":
            return 1.0 - proba

        # 'margin': the gap to the strongest competing class. Computed by
        # blanking each column in turn so a class never competes with itself.
        n_samples, n_classes = proba.shape
        if n_classes == 1:
            return np.zeros_like(proba)
        masked = np.repeat(proba[:, None, :], n_classes, axis=1)
        masked[:, np.arange(n_classes), np.arange(n_classes)] = -np.inf
        best_other = masked.max(axis=2)
        return best_other - proba

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
                "score": {
                    "type": "string",
                    "enum": ["lac", "margin"],
                    "default": "lac",
                    "description": "Nonconformity score.",
                },
                "calibration_size": {
                    "type": "number",
                    "default": 0.25,
                    "description": "Fraction of training data held out for calibration.",
                },
                "random_state": {
                    "type": ["integer", "null"],
                    "default": None,
                    "description": "Seed for the train/calibration split.",
                },
            },
        }


class SplitConformalRegressor(ConformalPredictor):
    """**Split conformal** prediction intervals with guaranteed coverage.

    Turns any TuiML regressor into an interval predictor whose intervals
    contain the truth with probability at least :math:`1 - \\alpha`, without
    any distributional assumption. The interval is the point prediction plus
    and minus a single calibrated radius.

    Overview
    --------
    1. Split the training data into a proper training part and a calibration
       part.
    2. Fit the wrapped regressor on the proper training part only.
    3. Take the absolute residual :math:`|y - \\hat{y}|` on each calibration
       sample as its nonconformity score.
    4. The corrected empirical quantile of those residuals is the interval
       radius.

    Theory
    ------
    With :math:`\\hat{q}` the corrected quantile of the calibration residuals,

    .. math::
        C(x) = \\left[ \\hat{f}(x) - \\hat{q}, \\ \\hat{f}(x) + \\hat{q} \\right]

    satisfies :math:`P(Y_{n+1} \\in C(X_{n+1})) \\geq 1 - \\alpha`.

    The width is **constant** across the input space, which is exactly its
    weakness: a homoscedastic interval over-covers where the model is
    confident and under-covers where it is not. Setting
    ``normalize=True`` divides residuals by a fitted difficulty estimate to
    restore local adaptivity, and
    :class:`~tuiml.uncertainty.ConformalizedQuantileRegressor` does so
    directly by conformalising quantile predictions.

    Parameters
    ----------
    estimator : Regressor
        A TuiML regressor.
    alpha : float, default=0.1
        Miscoverage level; the target coverage is ``1 - alpha``.
    calibration_size : float, default=0.25
        Fraction of the training data held out for calibration.
    normalize : bool, default=False
        Whether to scale residuals by a fitted difficulty model, producing
        locally adaptive interval widths.
    random_state : int, optional
        Seed for the train/calibration split.

    Attributes
    ----------
    scores_ : np.ndarray of shape (n_calibration,)
        Absolute (optionally normalised) calibration residuals.
    quantile_ : float
        Interval radius derived from ``scores_``.
    difficulty_estimator_ : Regressor or None
        Model of the log absolute residual, fitted only when ``normalize``.
    fitted_ : bool
        Whether :meth:`fit` has been called.

    Notes
    -----
    **Complexity.** One estimator fit (two when ``normalize=True``) plus
    :math:`O(n \\log n)` for the quantile.

    **When to use.** This is the default interval method: cheapest to fit and
    exactly valid. Prefer :class:`~tuiml.uncertainty.CVPlusRegressor` when
    data is too scarce to hold out a calibration split, and
    :class:`~tuiml.uncertainty.ConformalizedQuantileRegressor` when the noise
    is strongly heteroscedastic.

    References
    ----------
    .. [Lei2018] Lei, J., G'Sell, M., Rinaldo, A., Tibshirani, R. J., &
       Wasserman, L. (2018). Distribution-Free Predictive Inference for
       Regression. *Journal of the American Statistical Association*,
       113(523), 1094-1111. :doi:`10.1080/01621459.2017.1307116`
    .. [Papadopoulos2002] Papadopoulos, H., Proedrou, K., Vovk, V., &
       Gammerman, A. (2002). Inductive Confidence Machines for Regression.
       *ECML*, 345-356. :doi:`10.1007/3-540-36755-1_29`

    See Also
    --------
    :class:`~tuiml.uncertainty.CVPlusRegressor` : Uses all data, no held-out split.
    :class:`~tuiml.uncertainty.ConformalizedQuantileRegressor` : Heteroscedastic intervals.
    :func:`~tuiml.uncertainty.interval_width` : Compares interval sharpness.

    Examples
    --------
    >>> import numpy as np
    >>> from tuiml.uncertainty import SplitConformalRegressor
    >>> from tuiml.algorithms.trees import DecisionTreeRegressor
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(400, 3))
    >>> y = X[:, 0] * 2.0 + rng.normal(0, 0.5, 400)
    >>> cp = SplitConformalRegressor(DecisionTreeRegressor(max_depth=5),
    ...                              alpha=0.1, random_state=0)
    >>> cp.fit(X, y)
    SplitConformalRegressor(estimator=DecisionTreeRegressor(), alpha=0.1)
    >>> intervals = cp.predict_interval(X[:5])
    >>> intervals.shape
    (5, 2)
    """

    def __init__(
        self,
        estimator: Any,
        alpha: float = 0.1,
        calibration_size: float = 0.25,
        normalize: bool = False,
        random_state: Optional[int] = None,
    ) -> None:
        """Initialise the split conformal regressor.

        Parameters
        ----------
        estimator : Regressor
            A TuiML regressor.
        alpha : float, default=0.1
            Miscoverage level.
        calibration_size : float, default=0.25
            Fraction of the training data held out for calibration.
        normalize : bool, default=False
            Whether to scale residuals by a fitted difficulty model.
        random_state : int, optional
            Seed for the train/calibration split.
        """
        super().__init__(estimator, alpha)
        if not 0.0 < calibration_size < 1.0:
            raise ValueError(
                f"calibration_size must be in (0, 1), got {calibration_size}"
            )
        self.calibration_size = calibration_size
        self.normalize = normalize
        self.random_state = random_state
        self.difficulty_estimator_: Optional[Any] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SplitConformalRegressor":
        """Fit the regressor on a training split and calibrate on the rest.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training features.
        y : np.ndarray of shape (n_samples,)
            Training targets.

        Returns
        -------
        self : SplitConformalRegressor
            The fitted predictor.
        """
        from tuiml.evaluation.splitting import train_test_split

        X = np.asarray(X)
        y = np.asarray(y, dtype=np.float64)

        X_train, X_cal, y_train, y_cal = train_test_split(
            X, y, test_size=self.calibration_size, random_state=self.random_state
        )
        self.estimator.fit(X_train, y_train)

        if self.normalize:
            self._fit_difficulty(X_train, y_train)

        residual = np.abs(y_cal - np.asarray(self.estimator.predict(X_cal)))
        self.scores_ = residual / self._difficulty(X_cal)
        self.quantile_ = self.conformal_quantile(self.scores_, self.alpha)
        self.fitted_ = True
        return self

    def predict_interval(self, X: np.ndarray) -> np.ndarray:
        """Predict lower and upper bounds for each sample.

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
        center = np.asarray(self.estimator.predict(X), dtype=np.float64)
        radius = self.quantile_ * self._difficulty(X)
        return np.column_stack([center - radius, center + radius])

    def _fit_difficulty(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit a model of the log absolute training residual.

        Parameters
        ----------
        X : np.ndarray of shape (n_train, n_features)
            Proper training features.
        y : np.ndarray of shape (n_train,)
            Proper training targets.

        Returns
        -------
        None
        """
        import copy

        residual = np.abs(y - np.asarray(self.estimator.predict(X)))
        # Fit in log space so the exponentiated prediction is always positive.
        self.difficulty_estimator_ = copy.deepcopy(self.estimator)
        self.difficulty_estimator_.fit(X, np.log(residual + 1e-9))

    def _difficulty(self, X: np.ndarray) -> np.ndarray:
        """Return the per-sample scale used to normalise residuals.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Features.

        Returns
        -------
        scale : np.ndarray of shape (n_samples,)
            All ones unless ``normalize`` is enabled.
        """
        if not self.normalize or self.difficulty_estimator_ is None:
            return np.ones(len(X))
        predicted = np.asarray(self.difficulty_estimator_.predict(X), dtype=np.float64)
        return np.exp(predicted) + 1e-9

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
                "calibration_size": {
                    "type": "number",
                    "default": 0.25,
                    "description": "Fraction of training data held out for calibration.",
                },
                "normalize": {
                    "type": "boolean",
                    "default": False,
                    "description": "Scale residuals by a fitted difficulty model.",
                },
                "random_state": {
                    "type": ["integer", "null"],
                    "default": None,
                    "description": "Seed for the train/calibration split.",
                },
            },
        }
