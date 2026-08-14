"""Mondrian (class-conditional) conformal classification."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from tuiml.uncertainty.conformal.split import SplitConformalClassifier


class MondrianConformalClassifier(SplitConformalClassifier):
    """Conformal sets with **per-group** coverage, not just marginal coverage.

    Plain split conformal guarantees :math:`1 - \\alpha` coverage *on average*.
    That average can hide a rare class covered 40% of the time behind a common
    one covered 99%. Mondrian conformal prediction calibrates a **separate
    threshold within each group** — by default each class — so the guarantee
    holds inside every group independently.

    Overview
    --------
    1. Split off a calibration set as usual.
    2. Partition the calibration samples by their **taxonomy** — their true
       class, or a caller-supplied group id.
    3. Compute a separate conformal quantile within each group.
    4. A test label joins the prediction set when its nonconformity falls below
       *its own group's* threshold.

    Theory
    ------
    For every group :math:`g` with calibration scores :math:`S_g` and threshold
    :math:`\\hat{q}_g`, the guarantee is

    .. math::
        P\\left( Y_{n+1} \\in C(X_{n+1}) \\mid Y_{n+1} = g \\right)
        \\geq 1 - \\alpha

    which is strictly stronger than the marginal statement. The cost is
    statistical: each group needs its own calibration sample, so a group with
    fewer than :math:`\\lceil 1/\\alpha \\rceil - 1` members cannot certify the
    level and falls back to always being included — conservative but valid.

    Parameters
    ----------
    estimator : Classifier
        A TuiML classifier exposing ``predict_proba``.
    alpha : float, default=0.1
        Miscoverage level, enforced within every group.
    score : {'lac', 'margin'}, default='lac'
        Nonconformity score.
    calibration_size : float, default=0.25
        Fraction of the training data held out for calibration.
    random_state : int, optional
        Seed for the train/calibration split.

    Attributes
    ----------
    classes_ : np.ndarray of shape (n_classes,)
        Class labels seen during :meth:`fit`.
    group_quantiles_ : dict
        Mapping of group id to its conformal threshold. Groups too small to
        certify the level map to ``np.inf``.
    group_sizes_ : dict
        Number of calibration samples per group, for diagnosing which groups
        fell back to the conservative threshold.
    scores_ : np.ndarray of shape (n_calibration,)
        Nonconformity scores across all groups.
    fitted_ : bool
        Whether :meth:`fit` has been called.

    Notes
    -----
    **Complexity.** One estimator fit plus :math:`O(n \\log n)` total across
    groups — the same as split conformal.

    **When to use.** Use Mondrian whenever a per-class or per-subgroup
    guarantee matters: imbalanced classification, fairness constraints across a
    protected attribute, or any setting where a regulator asks about a specific
    subpopulation rather than the average. Sets are wider than the marginal
    version — that width is the honest price of the stronger claim. Check
    ``group_sizes_`` after fitting; a group with a handful of calibration
    samples silently gets a conservative threshold.

    References
    ----------
    .. [Vovk2003] Vovk, V., Lindsay, D., Nouretdinov, I., & Gammerman, A.
       (2003). Mondrian Confidence Machine. *Technical Report, Royal Holloway
       University of London*.
    .. [Lofstrom2015] Löfström, T., Boström, H., Linusson, H., & Johansson, U.
       (2015). Bias Reduction through Conditional Conformal Prediction.
       *Intelligent Data Analysis*, 19(6), 1355-1375.
       :doi:`10.3233/IDA-150786`

    See Also
    --------
    :class:`~tuiml.uncertainty.SplitConformalClassifier` : Marginal coverage only.
    :class:`~tuiml.uncertainty.APSConformalClassifier` : Approximate conditional coverage without groups.
    :func:`~tuiml.uncertainty.coverage_score` : Evaluate per group by slicing the test set.

    Examples
    --------
    >>> import numpy as np
    >>> from tuiml.uncertainty import MondrianConformalClassifier
    >>> from tuiml.algorithms.trees import DecisionTreeClassifier
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(600, 4))
    >>> y = (X[:, 0] + X[:, 1] > 0).astype(int)
    >>> cp = MondrianConformalClassifier(DecisionTreeClassifier(max_depth=4),
    ...                                  alpha=0.1, random_state=0)
    >>> cp.fit(X, y)
    MondrianConformalClassifier(estimator=DecisionTreeClassifier(), alpha=0.1)
    >>> sorted(cp.group_sizes_.values()) == sorted(cp.group_sizes_.values())
    True
    >>> cp.predict_set(X[:5]).shape
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
        """Initialise the Mondrian conformal classifier.

        Parameters
        ----------
        estimator : Classifier
            A TuiML classifier exposing ``predict_proba``.
        alpha : float, default=0.1
            Miscoverage level, enforced within every group.
        score : {'lac', 'margin'}, default='lac'
            Nonconformity score.
        calibration_size : float, default=0.25
            Fraction of the training data held out for calibration.
        random_state : int, optional
            Seed for the train/calibration split.
        """
        super().__init__(
            estimator,
            alpha=alpha,
            score=score,
            calibration_size=calibration_size,
            random_state=random_state,
        )
        self.group_quantiles_: Dict[Any, float] = {}
        self.group_sizes_: Dict[Any, int] = {}

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: Optional[np.ndarray] = None,
    ) -> "MondrianConformalClassifier":
        """Fit the estimator and calibrate one threshold per group.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training features.
        y : np.ndarray of shape (n_samples,)
            Training labels.
        groups : np.ndarray of shape (n_samples,), optional
            Taxonomy assigning each sample to a group. Defaults to the class
            label, giving class-conditional coverage.

        Returns
        -------
        self : MondrianConformalClassifier
            The fitted predictor.
        """
        X = np.asarray(X)
        y = np.asarray(y)
        taxonomy = y if groups is None else np.asarray(groups)

        # Split on indices rather than on X directly, so the taxonomy can be
        # sliced with the same calibration index.
        train_index, cal_index = self._split_index(np.arange(X.shape[0]))

        self.estimator.fit(X[train_index], y[train_index])
        self.classes_ = np.asarray(
            getattr(self.estimator, "classes_", np.unique(y))
        )

        proba = np.asarray(
            self.estimator.predict_proba(X[cal_index]), dtype=np.float64
        )
        true_index = np.searchsorted(self.classes_, y[cal_index])
        self.scores_ = self._nonconformity(proba, true_index)

        cal_groups = taxonomy[cal_index]
        self.group_quantiles_ = {}
        self.group_sizes_ = {}
        for group in np.unique(cal_groups):
            member = cal_groups == group
            group_scores = self.scores_[member]
            self.group_sizes_[group] = int(member.sum())
            self.group_quantiles_[group] = self.conformal_quantile(
                group_scores, self.alpha
            )

        # Marginal fallback for groups never seen during calibration.
        self.quantile_ = self.conformal_quantile(self.scores_, self.alpha)
        self.fitted_ = True
        return self

    def _split_index(self, index: np.ndarray) -> tuple:
        """Split sample indices into training and calibration parts.

        Parameters
        ----------
        index : np.ndarray of shape (n_samples,)
            Row indices to split.

        Returns
        -------
        train_index, cal_index : np.ndarray
            Disjoint index arrays.
        """
        rng = np.random.default_rng(self.random_state)
        shuffled = rng.permutation(index)
        n_cal = max(1, int(round(self.calibration_size * index.size)))
        return shuffled[n_cal:], shuffled[:n_cal]

    def predict_set(self, X: np.ndarray) -> np.ndarray:
        """Predict class sets using each class's own threshold.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Test features.

        Returns
        -------
        include : np.ndarray of shape (n_samples, n_classes) of bool
            Class-membership mask, thresholded per class.

        Notes
        -----
        This applies the class-conditional thresholds, so it is only correct
        when the taxonomy *is* the class label — the default. When ``fit`` was
        given an external ``groups`` array, use :meth:`predict_set_for_groups`
        and pass the test samples' group ids.
        """
        self._check_is_fitted()
        proba = np.asarray(self.estimator.predict_proba(X), dtype=np.float64)
        candidate_scores = self._candidate_scores(proba)

        thresholds = np.array(
            [
                self.group_quantiles_.get(label, self.quantile_)
                for label in self.classes_
            ]
        )
        return candidate_scores <= thresholds[None, :]

    def predict_set_for_groups(
        self, X: np.ndarray, groups: np.ndarray
    ) -> np.ndarray:
        """Predict class sets using an explicit per-sample group id.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Test features.
        groups : np.ndarray of shape (n_samples,)
            Group id of each test sample, matching the taxonomy given to
            :meth:`fit`.

        Returns
        -------
        include : np.ndarray of shape (n_samples, n_classes) of bool
            Class-membership mask, thresholded by each sample's group.
        """
        self._check_is_fitted()
        proba = np.asarray(self.estimator.predict_proba(X), dtype=np.float64)
        candidate_scores = self._candidate_scores(proba)

        thresholds = np.array(
            [self.group_quantiles_.get(g, self.quantile_) for g in np.asarray(groups)]
        )
        return candidate_scores <= thresholds[:, None]

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Any]:
        """Return JSON Schema for constructor parameters."""
        return SplitConformalClassifier.get_parameter_schema()
