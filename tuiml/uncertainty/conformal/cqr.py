"""Conformalized quantile regression."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from tuiml.uncertainty._base import ConformalPredictor


class ConformalizedQuantileRegressor(ConformalPredictor):
    """**Conformalized quantile regression** — valid *and* heteroscedastic intervals.

    :class:`~tuiml.uncertainty.SplitConformalRegressor` adds a constant radius
    everywhere, which over-covers where the noise is small and under-covers
    where it is large. CQR starts from a **quantile regressor**, which already
    models how the spread varies with :math:`x`, and conformalises it — keeping
    the shape while restoring the exact coverage guarantee the quantile fit
    lacks on its own.

    Overview
    --------
    1. Fit a lower-quantile model at :math:`\\alpha/2` and an upper-quantile
       model at :math:`1 - \\alpha/2` on the proper training set.
    2. On the calibration set, score each sample by how far outside the
       predicted band it falls — negative when comfortably inside.
    3. Take the corrected quantile of those scores.
    4. Widen (or, when the quantile fit was conservative, **narrow**) the band
       by that amount.

    Theory
    ------
    With :math:`\\hat{q}_{lo}` and :math:`\\hat{q}_{hi}` the fitted quantile
    functions, the conformity score is the signed distance outside the band:

    .. math::
        E_i = \\max\\left( \\hat{q}_{lo}(x_i) - y_i,\\ \\
        y_i - \\hat{q}_{hi}(x_i) \\right)

    and with :math:`\\hat{q}` its corrected quantile the interval is

    .. math::
        C(x) = \\left[ \\hat{q}_{lo}(x) - \\hat{q},\\ \\
        \\hat{q}_{hi}(x) + \\hat{q} \\right]

    A **negative** :math:`\\hat{q}` is not an error: it means the quantile
    models over-covered, and CQR correctly shrinks the band. Coverage remains
    at least :math:`1 - \\alpha` either way.

    Parameters
    ----------
    lower_estimator : Regressor
        Model fitted to predict the :math:`\\alpha/2` quantile. Any regressor
        trained with a pinball loss at that level — for example
        ``tuiml.sklearn.linear.QuantileRegressor(quantile=0.05)`` or
        ``tuiml.sklearn.ensemble.GradientBoostingRegressor(loss='quantile',
        alpha=0.05)``.
    upper_estimator : Regressor
        Model fitted to predict the :math:`1 - \\alpha/2` quantile.
    alpha : float, default=0.1
        Miscoverage level. Must match the quantile levels the two estimators
        were configured for, otherwise the band is valid but needlessly wide.
    calibration_size : float, default=0.25
        Fraction of the training data held out for calibration.
    random_state : int, optional
        Seed for the train/calibration split.

    Attributes
    ----------
    scores_ : np.ndarray of shape (n_calibration,)
        Signed distances outside the predicted band.
    quantile_ : float
        The additive correction applied to both edges; may be negative.
    fitted_ : bool
        Whether :meth:`fit` has been called.

    Notes
    -----
    **Complexity.** Two estimator fits plus :math:`O(n \\log n)`.

    **When to use.** Use CQR whenever the noise level plainly depends on the
    input — sensor readings that degrade with range, demand that fluctuates
    more at high volume, any funnel-shaped residual plot. On homoscedastic
    data it costs a second model fit for intervals that are no tighter than
    split conformal's.

    Because TuiML has no native quantile regressor yet, this class takes two
    externally-fitted quantile models rather than cloning one estimator. Pass
    the ``tuiml.sklearn`` wrappers named above, or any object with ``fit`` and
    ``predict``.

    References
    ----------
    .. [Romano2019] Romano, Y., Patterson, E., & Candès, E. J. (2019).
       Conformalized Quantile Regression. *NeurIPS*, 3543-3553.
       :arxiv:`1905.03222`

    See Also
    --------
    :class:`~tuiml.uncertainty.SplitConformalRegressor` : Constant-width intervals; ``normalize=True`` is a cheaper adaptive alternative.
    :class:`~tuiml.uncertainty.CVPlusRegressor` : Uses all the data, still constant width.
    :func:`~tuiml.uncertainty.interval_width` : Compares sharpness against the alternatives.

    Examples
    --------
    Requires ``pip install tuiml[sklearn]`` for a quantile regressor. Passing
    two mean regressors instead silently degenerates this class into
    :class:`~tuiml.uncertainty.SplitConformalRegressor`: both models predict
    the same thing, the band has zero width, and only the constant correction
    survives.

    >>> import numpy as np
    >>> from tuiml.uncertainty import ConformalizedQuantileRegressor
    >>> from tuiml.sklearn.ensemble import GradientBoostingRegressor
    >>> rng = np.random.default_rng(0)
    >>> X = rng.uniform(0, 4, size=(400, 1))
    >>> y = X[:, 0] + rng.normal(0, 0.2 + 0.5 * X[:, 0], 400)
    >>> cqr = ConformalizedQuantileRegressor(
    ...     GradientBoostingRegressor(loss='quantile', alpha=0.05, n_estimators=50),
    ...     GradientBoostingRegressor(loss='quantile', alpha=0.95, n_estimators=50),
    ...     alpha=0.1, random_state=0)
    >>> cqr.fit(X, y)
    ConformalizedQuantileRegressor(alpha=0.1)
    >>> widths = np.diff(cqr.predict_interval(X), axis=1).ravel()
    >>> bool(widths.std() > 0.0)  # the band adapts to the local noise
    True
    """

    def __init__(
        self,
        lower_estimator: Any,
        upper_estimator: Any,
        alpha: float = 0.1,
        calibration_size: float = 0.25,
        random_state: Optional[int] = None,
    ) -> None:
        """Initialise the conformalized quantile regressor.

        Parameters
        ----------
        lower_estimator : Regressor
            Model for the ``alpha / 2`` quantile.
        upper_estimator : Regressor
            Model for the ``1 - alpha / 2`` quantile.
        alpha : float, default=0.1
            Miscoverage level.
        calibration_size : float, default=0.25
            Fraction of the training data held out for calibration.
        random_state : int, optional
            Seed for the train/calibration split.
        """
        # The parent stores a single estimator; the lower model stands in as
        # the point predictor so predict() and repr() keep working.
        super().__init__(lower_estimator, alpha)
        if not 0.0 < calibration_size < 1.0:
            raise ValueError(
                f"calibration_size must be in (0, 1), got {calibration_size}"
            )
        self.lower_estimator = lower_estimator
        self.upper_estimator = upper_estimator
        self.calibration_size = calibration_size
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ConformalizedQuantileRegressor":
        """Fit both quantile models and calibrate the band correction.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training features.
        y : np.ndarray of shape (n_samples,)
            Training targets.

        Returns
        -------
        self : ConformalizedQuantileRegressor
            The fitted predictor.
        """
        from tuiml.evaluation.splitting import train_test_split

        X = np.asarray(X)
        y = np.asarray(y, dtype=np.float64)

        X_train, X_cal, y_train, y_cal = train_test_split(
            X, y, test_size=self.calibration_size, random_state=self.random_state
        )
        self.lower_estimator.fit(X_train, y_train)
        self.upper_estimator.fit(X_train, y_train)

        lower = np.asarray(self.lower_estimator.predict(X_cal), dtype=np.float64)
        upper = np.asarray(self.upper_estimator.predict(X_cal), dtype=np.float64)

        # Signed distance outside the band; negative when the point sits
        # comfortably inside, which lets the correction shrink the interval.
        self.scores_ = np.maximum(lower - y_cal, y_cal - upper)
        self.quantile_ = self.conformal_quantile(self.scores_, self.alpha)
        self.fitted_ = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return the midpoint of the calibrated band as a point prediction.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Test features.

        Returns
        -------
        y_pred : np.ndarray of shape (n_samples,)
            Midpoint of the lower and upper quantile predictions.
        """
        self._check_is_fitted()
        lower = np.asarray(self.lower_estimator.predict(X), dtype=np.float64)
        upper = np.asarray(self.upper_estimator.predict(X), dtype=np.float64)
        return (lower + upper) / 2.0

    def predict_interval(self, X: np.ndarray) -> np.ndarray:
        """Predict calibrated, input-dependent lower and upper bounds.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Test features.

        Returns
        -------
        intervals : np.ndarray of shape (n_samples, 2)
            Column 0 holds the lower bound, column 1 the upper bound. Widths
            vary with ``X`` wherever the quantile models say they should.
        """
        self._check_is_fitted()
        lower = np.asarray(self.lower_estimator.predict(X), dtype=np.float64)
        upper = np.asarray(self.upper_estimator.predict(X), dtype=np.float64)

        low = lower - self.quantile_
        high = upper + self.quantile_
        # A negative correction can cross the edges when the quantile models
        # were badly ordered; keep the interval well-formed.
        return np.column_stack([np.minimum(low, high), np.maximum(low, high)])

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
                "random_state": {
                    "type": ["integer", "null"],
                    "default": None,
                    "description": "Seed for the train/calibration split.",
                },
            },
        }

    def __repr__(self) -> str:
        """Return a readable representation of the predictor."""
        return f"ConformalizedQuantileRegressor(alpha={self.alpha})"
