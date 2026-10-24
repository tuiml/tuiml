"""Recursive feature elimination selectors."""

from typing import Any

import numpy as np

from tuiml.base.algorithms import Regressor
from tuiml.base.features import FeatureSelector, feature_selector
from tuiml.evaluation.metrics import (
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from tuiml.evaluation.splitting import KFold

from ._base import SelectorMixin, _ensure_numpy


def _get_feature_importances(estimator: Any) -> np.ndarray:
    """Extract one importance value for each feature.

    Parameters
    ----------
    estimator : object
        Fitted estimator exposing ``feature_importances_`` or ``coef_``.

    Returns
    -------
    importances : ndarray of shape (n_features,)
        Non-negative feature importance values.

    Raises
    ------
    ValueError
        If the estimator does not expose a supported importance attribute.
    """
    if hasattr(estimator, "feature_importances_"):
        importances = np.asarray(estimator.feature_importances_)
    elif hasattr(estimator, "coef_"):
        importances = np.abs(np.asarray(estimator.coef_))
        if importances.ndim > 1:
            importances = np.max(importances, axis=0)
    else:
        raise ValueError(
            "Estimator must expose either 'feature_importances_' or "
            "'coef_' after fitting."
        )

    if importances.ndim != 1:
        raise ValueError(
            "Feature importance must be one-dimensional after extraction."
        )

    return np.abs(importances)


def _validate_step(step: float) -> None:
    """Validate an RFE step value."""
    if isinstance(step, bool):
        raise TypeError("step must be an int or float.")

    if isinstance(step, float):
        if not 0 < step < 1:
            raise ValueError("step as a float must be between 0 and 1.")
    elif isinstance(step, int):
        if step < 1:
            raise ValueError("step must be at least 1.")
    else:
        raise TypeError("step must be an int or float.")


def _score_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    scoring: str | Any,
) -> float:
    """Score predictions using an RFECV scoring strategy."""
    if callable(scoring):
        return float(scoring(y_true, y_pred))

    scoring_lower = scoring.lower()

    if scoring_lower == "accuracy":
        return float(np.mean(y_true == y_pred))
    if scoring_lower == "f1":
        return float(f1_score(y_true, y_pred, average="macro"))
    if scoring_lower == "precision":
        return float(precision_score(y_true, y_pred, average="macro"))
    if scoring_lower == "recall":
        return float(recall_score(y_true, y_pred, average="macro"))
    if scoring_lower == "r2":
        return float(r2_score(y_true, y_pred))
    if scoring_lower in ("mse", "neg_mean_squared_error"):
        return float(-mean_squared_error(y_true, y_pred))
    if scoring_lower in ("mae", "neg_mean_absolute_error"):
        return float(-mean_absolute_error(y_true, y_pred))

    raise ValueError(
        f"Unknown scoring '{scoring}'. Valid options: "
        "'accuracy', 'f1', 'precision', 'recall', 'r2', 'mse', 'mae'"
    )


@feature_selector(
    tags=["wrapper", "recursive", "feature-selection"],
    version="1.0.0",
)
class RFE(FeatureSelector, SelectorMixin):
    """Recursive feature elimination.

    Parameters
    ----------
    estimator : object
        A supervised estimator exposing ``fit`` and either
        ``feature_importances_`` or ``coef_`` after fitting.
    n_features_to_select : int, optional
        Number of features to retain. If ``None``, half of the
        input features are retained.
    step : int or float, default=1
        Number of features to remove at each iteration, or the
        fraction of features to remove when a float is supplied.

    Attributes
    ----------
    n_features_to_select_ : int
        Number of features selected after fitting.
    support_ : ndarray of bool
        Boolean mask indicating selected features.
    ranking_ : ndarray of int
        Feature ranking. Selected features have rank 1.

    Examples
    --------
    >>> import numpy as np
    >>> from tuiml.features.selection import RFE
    >>> from tuiml.algorithms.trees.random_forest import RandomForestClassifier
    >>> X = np.random.RandomState(0).randn(30, 5)
    >>> y = np.random.RandomState(0).randint(0, 2, 30)
    >>> selector = RFE(
    ...     estimator=RandomForestClassifier(n_estimators=5, random_state=0),
    ...     n_features_to_select=2,
    ... )
    >>> _ = selector.fit(X, y)
    >>> int(selector.support_.sum())
    2
    """

    def __init__(
        self,
        estimator: Any = None,
        n_features_to_select: int | None = None,
        step: float = 1,
    ):
        super().__init__()
        self.estimator = estimator
        self.n_features_to_select = n_features_to_select
        self.step = step
        self.n_features_to_select_: int | None = None
        self.support_: np.ndarray | None = None
        self.ranking_: np.ndarray | None = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
    ) -> "RFE":
        """Fit RFE and recursively eliminate features.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training feature matrix.
        y : ndarray of shape (n_samples,)
            Target values.

        Returns
        -------
        self : RFE
            The fitted selector.
        """
        if y is None:
            raise ValueError("RFE requires target values (y)")
        if self.estimator is None:
            raise ValueError("estimator must be provided")

        X = _ensure_numpy(X)
        y = _ensure_numpy(y)
        n_features = X.shape[1]
        self._n_features_in = n_features

        if self.n_features_to_select is None:
            target = max(1, n_features // 2)
        else:
            target = int(self.n_features_to_select)

        if target < 1 or target > n_features:
            raise ValueError(
                "n_features_to_select must be between 1 and "
                "the number of features."
            )

        _validate_step(self.step)

        support = np.ones(n_features, dtype=bool)
        ranking = np.ones(n_features, dtype=int)
        current_rank = 1

        while support.sum() > target:
            active = np.flatnonzero(support)
            estimator = self._clone_estimator()
            estimator.fit(X[:, active], y)

            importances = _get_feature_importances(estimator)

            if importances.shape[0] != active.size:
                raise ValueError(
                    "Estimator feature importance does not match "
                    "the number of input features."
                )

            if isinstance(self.step, float):
                n_remove = max(1, int(active.size * self.step))
            else:
                n_remove = self.step

            n_remove = min(n_remove, active.size - target)
            order = np.argsort(importances, kind="stable")
            remove_global = active[order[:n_remove]]

            ranking[remove_global] = current_rank + 1
            support[remove_global] = False
            current_rank += 1

        self._selected_indices = np.flatnonzero(support)
        self.n_features_to_select_ = int(support.sum())
        self.support_ = support
        self.ranking_ = ranking
        self._feature_scores = support.astype(float)
        self._is_fitted = True

        return self

    def _clone_estimator(self) -> Any:
        """Create a fresh estimator with the same parameters."""
        if hasattr(self.estimator, "get_params"):
            return self.estimator.__class__(**self.estimator.get_params())
        return self.estimator.__class__()

    def _compute_scores(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> np.ndarray:
        """Return binary scores for selected features."""
        return self._feature_scores

    @classmethod
    def get_parameter_schema(cls) -> dict:
        """Return the parameter schema for RFE."""
        return {
            "estimator": {
                "type": "object",
                "default": None,
                "description": "Estimator used for recursive elimination.",
            },
            "n_features_to_select": {
                "type": "integer",
                "default": None,
                "description": "Number of features to retain.",
            },
            "step": {
                "type": ["integer", "number"],
                "default": 1,
                "description": (
                    "Number or fraction of features removed per iteration."
                ),
            },
        }


@feature_selector(
    tags=["wrapper", "recursive", "cross-validation"],
    version="1.0.0",
)
class RFECV(RFE):
    """Recursive feature elimination with cross-validation.

    Parameters
    ----------
    estimator : object
        A supervised estimator exposing ``fit``, ``predict`` and either
        ``feature_importances_`` or ``coef_`` after fitting.
    step : int or float, default=1
        Number of features to remove at each iteration, or the
        fraction of features to remove when a float is supplied.
    cv : int or splitter, default=5
        Number of folds when an integer is supplied, or a splitter
        exposing ``split(X, y)``.
    scoring : str or callable, optional
        Scoring strategy. If ``None``, classification uses accuracy
        and regression uses R².
    random_state : int, optional
        Random seed used when ``cv`` is an integer.

    Attributes
    ----------
    n_features_to_select_ : int
        Number of features selected by cross-validation.
    support_ : ndarray of bool
        Boolean mask indicating selected features.
    ranking_ : ndarray of int
        Feature ranking.
    cv_results_ : dict
        Cross-validation results for each tested feature count.

    Examples
    --------
    >>> import numpy as np
    >>> from tuiml.features.selection import RFECV
    >>> from tuiml.algorithms.trees.random_forest import RandomForestClassifier
    >>> X = np.random.RandomState(0).randn(40, 5)
    >>> y = np.random.RandomState(1).randint(0, 2, 40)
    >>> selector = RFECV(
    ...     estimator=RandomForestClassifier(n_estimators=5, random_state=0),
    ...     cv=3,
    ... )
    >>> _ = selector.fit(X, y)
    >>> selector.n_features_to_select_ >= 1
    True

    A splitter instance can also be supplied:

    >>> from tuiml.evaluation.splitting import KFold
    >>> cv = KFold(n_splits=3, shuffle=True, random_state=0)
    >>> selector = RFECV(
    ...     estimator=RandomForestClassifier(n_estimators=5, random_state=0),
    ...     cv=cv,
    ... )
    >>> _ = selector.fit(X, y)
    """

    def __init__(
        self,
        estimator: Any = None,
        step: float = 1,
        cv: Any = 5,
        scoring: Any | None = None,
        random_state: int | None = None,
    ):
        super().__init__(
            estimator=estimator,
            n_features_to_select=None,
            step=step,
        )
        self.cv = cv
        self.scoring = scoring
        self.random_state = random_state
        self.cv_results_: dict | None = None

    def _feature_counts(self, n_features: int) -> list[int]:
        """Build the feature-count path used by RFECV."""
        feature_counts = [n_features]
        current = n_features

        while current > 1:
            if isinstance(self.step, float):
                n_remove = max(1, int(current * self.step))
            else:
                n_remove = self.step

            current = max(1, current - n_remove)

            if current not in feature_counts:
                feature_counts.append(current)

        return feature_counts

    def _elimination_path(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_counts: list[int],
    ) -> list[np.ndarray]:
        """Compute one complete RFE feature path for a training fold."""
        active = np.arange(X.shape[1])
        path = [active.copy()]

        for count in feature_counts[1:]:
            estimator = self._clone_estimator()
            estimator.fit(X[:, active], y)

            importances = _get_feature_importances(estimator)

            if importances.shape[0] != active.size:
                raise ValueError(
                    "Estimator feature importance does not match "
                    "the number of input features."
                )

            n_remove = active.size - count
            remove_local = np.argsort(importances, kind="stable")[:n_remove]
            active = np.delete(active, remove_local)
            path.append(active.copy())

        return path

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray | None = None,
    ) -> "RFECV":
        """Fit RFECV and select the feature count with the best CV score.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training feature matrix.
        y : ndarray of shape (n_samples,)
            Target values.

        Returns
        -------
        self : RFECV
            The fitted selector.
        """
        if y is None:
            raise ValueError("RFECV requires target values (y)")
        if self.estimator is None:
            raise ValueError("estimator must be provided")

        X = _ensure_numpy(X)
        y = _ensure_numpy(y)
        n_features = X.shape[1]
        self._n_features_in = n_features

        if isinstance(self.cv, bool):
            raise TypeError("cv must be an integer or splitter.")

        if isinstance(self.cv, int):
            if self.cv < 2:
                raise ValueError("cv must be at least 2.")
            splitter = KFold(
                n_splits=self.cv,
                shuffle=True,
                random_state=self.random_state,
            )
        elif hasattr(self.cv, "split"):
            splitter = self.cv
        else:
            raise TypeError("cv must be an integer or splitter.")

        _validate_step(self.step)

        if self.scoring is None:
            scoring = (
                "r2"
                if isinstance(self.estimator, Regressor)
                else "accuracy"
            )
        else:
            scoring = self.scoring

        feature_counts = self._feature_counts(n_features)
        fold_scores = [[] for _ in feature_counts]

        for train_idx, test_idx in splitter.split(X, y):
            X_train = X[train_idx]
            X_test = X[test_idx]
            y_train = y[train_idx]
            y_test = y[test_idx]

            path = self._elimination_path(
                X_train,
                y_train,
                feature_counts,
            )

            for index, active in enumerate(path):
                estimator = self._clone_estimator()
                estimator.fit(X_train[:, active], y_train)
                predictions = estimator.predict(X_test[:, active])

                score = _score_predictions(
                    y_test,
                    predictions,
                    scoring,
                )
                fold_scores[index].append(score)

        scores_array = np.asarray(
            [float(np.mean(scores)) for scores in fold_scores]
        )

        best_score = np.max(scores_array)
        best_candidates = [
            count
            for count, score in zip(feature_counts, scores_array)
            if np.isclose(score, best_score)
        ]
        best_count = min(best_candidates)

        final_selector = RFE(
            estimator=self.estimator,
            n_features_to_select=best_count,
            step=self.step,
        )
        final_selector.fit(X, y)

        self._selected_indices = final_selector._selected_indices
        self.n_features_to_select_ = final_selector.n_features_to_select_
        self.support_ = final_selector.support_
        self.ranking_ = final_selector.ranking_
        self._feature_scores = final_selector._feature_scores
        self._is_fitted = True

        self.cv_results_ = {
            "n_features": np.asarray(feature_counts),
            "mean_test_score": scores_array,
        }

        return self

    @classmethod
    def get_parameter_schema(cls) -> dict:
        """Return the parameter schema for RFECV."""
        return {
            "estimator": {
                "type": "object",
                "default": None,
                "description": "Estimator used for recursive elimination.",
            },
            "step": {
                "type": ["integer", "number"],
                "default": 1,
                "description": (
                    "Number or fraction of features removed per iteration."
                ),
            },
            "cv": {
                "type": ["integer", "object"],
                "default": 5,
                "description": "Number of folds or a CV splitter.",
            },
            "scoring": {
                "type": ["string", "object"],
                "default": None,
                "description": "Scoring strategy for cross-validation.",
            },
            "random_state": {
                "type": ["integer", "null"],
                "default": None,
                "description": "Random seed for integer CV.",
            },
        }
