"""Base classes for uncertainty quantification."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np


class Calibrator(ABC):
    """Base class for probability calibrators.

    A calibrator is fitted on held-out scores and their true labels, then maps
    raw classifier scores onto calibrated probabilities. It is a post-processor
    around an already-fitted model, not an algorithm, so it is not entered in
    the TuiML algorithm hub.

    Attributes
    ----------
    fitted_ : bool
        Whether :meth:`fit` has been called.

    See Also
    --------
    :class:`~tuiml.uncertainty.PlattCalibrator` : Sigmoid calibration.
    :class:`~tuiml.uncertainty.IsotonicCalibrator` : Non-parametric calibration.
    """

    def __init__(self) -> None:
        """Initialise the calibrator in an unfitted state."""
        self.fitted_ = False

    @abstractmethod
    def fit(self, scores: np.ndarray, y: np.ndarray) -> "Calibrator":
        """Fit the calibration map on held-out scores.

        Parameters
        ----------
        scores : np.ndarray of shape (n_samples,) or (n_samples, n_classes)
            Uncalibrated scores or probabilities.
        y : np.ndarray of shape (n_samples,)
            True labels.

        Returns
        -------
        self : Calibrator
            The fitted calibrator.
        """

    @abstractmethod
    def transform(self, scores: np.ndarray) -> np.ndarray:
        """Map raw scores onto calibrated probabilities.

        Parameters
        ----------
        scores : np.ndarray of shape (n_samples,) or (n_samples, n_classes)
            Uncalibrated scores or probabilities.

        Returns
        -------
        proba : np.ndarray
            Calibrated probabilities, same shape as ``scores``.
        """

    def fit_transform(self, scores: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Fit on held-out scores and return the calibrated probabilities.

        Parameters
        ----------
        scores : np.ndarray of shape (n_samples,) or (n_samples, n_classes)
            Uncalibrated scores or probabilities.
        y : np.ndarray of shape (n_samples,)
            True labels.

        Returns
        -------
        proba : np.ndarray
            Calibrated probabilities, same shape as ``scores``.
        """
        return self.fit(scores, y).transform(scores)

    def _check_is_fitted(self) -> None:
        """Raise if the calibrator has not been fitted.

        Raises
        ------
        RuntimeError
            If :meth:`fit` has not been called.
        """
        if not self.fitted_:
            raise RuntimeError(
                f"{self.__class__.__name__} is not fitted. Call fit() first."
            )


class ConformalPredictor(ABC):
    """Base class for conformal predictors.

    A conformal predictor wraps a point predictor and converts it into a set
    (classification) or interval (regression) predictor with a finite-sample,
    distribution-free coverage guarantee of at least :math:`1 - \\alpha` under
    exchangeability.

    Parameters
    ----------
    estimator : Algorithm
        A TuiML classifier or regressor.
    alpha : float, default=0.1
        Miscoverage level. The guarantee is :math:`1 - \\alpha` marginal coverage.

    Attributes
    ----------
    scores_ : np.ndarray of shape (n_calibration,)
        Nonconformity scores computed on the calibration set.
    quantile_ : float
        The conformal quantile of ``scores_`` used to form predictions.
    fitted_ : bool
        Whether :meth:`fit` has been called.

    Notes
    -----
    The guarantee is **marginal**, averaged over calibration and test draws;
    it is not conditional on a particular feature vector. Use
    :class:`~tuiml.uncertainty.MondrianConformalClassifier` when per-class
    (conditional) coverage is required.

    References
    ----------
    .. [Vovk2005] Vovk, V., Gammerman, A., & Shafer, G. (2005).
       *Algorithmic Learning in a Random World*. Springer.
       :doi:`10.1007/b106715`
    .. [Angelopoulos2023] Angelopoulos, A. N., & Bates, S. (2023).
       Conformal Prediction: A Gentle Introduction.
       *Foundations and Trends in Machine Learning*, 16(4), 494-591.
       :doi:`10.1561/2200000101`

    See Also
    --------
    :class:`~tuiml.uncertainty.SplitConformalClassifier` : Prediction sets.
    :class:`~tuiml.uncertainty.SplitConformalRegressor` : Prediction intervals.
    """

    def __init__(self, estimator: Any, alpha: float = 0.1) -> None:
        """Initialise the conformal predictor.

        Parameters
        ----------
        estimator : Algorithm
            A TuiML classifier or regressor.
        alpha : float, default=0.1
            Miscoverage level in ``(0, 1)``.
        """
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        self.estimator = estimator
        self.alpha = alpha
        self.scores_: Optional[np.ndarray] = None
        self.quantile_: Optional[float] = None
        self.fitted_ = False

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "ConformalPredictor":
        """Fit the underlying estimator and calibrate the nonconformity scores.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training features.
        y : np.ndarray of shape (n_samples,)
            Training targets.

        Returns
        -------
        self : ConformalPredictor
            The fitted predictor.
        """

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return the underlying estimator's point predictions.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Test features.

        Returns
        -------
        y_pred : np.ndarray of shape (n_samples,)
            Point predictions from the wrapped estimator.
        """
        self._check_is_fitted()
        return self.estimator.predict(X)

    @staticmethod
    def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
        """Compute the finite-sample corrected conformal quantile.

        The correction uses :math:`\\lceil (n+1)(1-\\alpha) \\rceil / n` rather
        than the plain :math:`1-\\alpha` empirical quantile; this is what makes
        the coverage guarantee hold at finite ``n`` instead of only
        asymptotically.

        Parameters
        ----------
        scores : np.ndarray of shape (n_calibration,)
            Nonconformity scores from the calibration set.
        alpha : float
            Miscoverage level in ``(0, 1)``.

        Returns
        -------
        q : float
            The corrected quantile. ``np.inf`` when the calibration set is too
            small to certify the requested level.
        """
        scores = np.asarray(scores, dtype=np.float64)
        n = scores.size
        if n == 0:
            return float("inf")
        level = np.ceil((n + 1) * (1.0 - alpha)) / n
        if level > 1.0:
            # n is too small for this alpha: no finite threshold can certify it.
            return float("inf")
        return float(np.quantile(scores, level, method="higher"))

    def _check_is_fitted(self) -> None:
        """Raise if the predictor has not been fitted.

        Raises
        ------
        RuntimeError
            If :meth:`fit` has not been called.
        """
        if not self.fitted_:
            raise RuntimeError(
                f"{self.__class__.__name__} is not fitted. Call fit() first."
            )

    def get_params(self) -> Dict[str, Any]:
        """Return the constructor parameters of this predictor.

        Returns
        -------
        params : dict
            Mapping of parameter name to value.
        """
        return {"estimator": self.estimator, "alpha": self.alpha}

    def __repr__(self) -> str:
        """Return a readable representation of the predictor."""
        return (
            f"{self.__class__.__name__}("
            f"estimator={self.estimator.__class__.__name__}(), "
            f"alpha={self.alpha})"
        )


class SetPredictorMixin:
    """Mixin providing set-valued prediction helpers for classifiers."""

    def predict_set(self, X: np.ndarray) -> np.ndarray:
        """Predict a boolean membership mask over the classes.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Test features.

        Returns
        -------
        include : np.ndarray of shape (n_samples, n_classes) of bool
            ``include[i, k]`` is True when class ``k`` belongs to the
            prediction set of sample ``i``.
        """
        raise NotImplementedError

    def predict_sets_as_labels(self, X: np.ndarray) -> List[np.ndarray]:
        """Predict prediction sets as explicit lists of class labels.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Test features.

        Returns
        -------
        sets : list of np.ndarray
            One array of class labels per sample.
        """
        include = self.predict_set(X)
        classes = self.classes_
        return [classes[row] for row in include]
