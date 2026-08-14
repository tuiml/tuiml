"""Meta-learners for uplift / heterogeneous treatment effect estimation.

The S-, T- and X-learners wrap an arbitrary TuiML regressor (for example
:class:`~tuiml.algorithms.trees.DecisionTreeRegressor`) and re-arrange the
``(X, treatment, y)`` data so that an ordinary supervised learner can estimate
the conditional average treatment effect

.. math::
    \\tau(x) = E[Y(1) - Y(0) \\mid X = x],

where :math:`Y(1)` and :math:`Y(0)` are the potential outcomes under treatment
and control.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional

import numpy as np

from tuiml.base.algorithms import UpliftModel, uplift


# =============================================================================
# Shared validation / cloning helpers
# =============================================================================

def _check_arrays(X, treatment, y):
    """Validate and coerce ``(X, treatment, y)`` for uplift fitting.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Covariates.
    treatment : array-like of shape (n_samples,)
        Binary treatment indicator (``0`` = control, ``1`` = treated).
    y : array-like of shape (n_samples,)
        Numeric outcome.

    Returns
    -------
    X : np.ndarray of shape (n_samples, n_features)
        Coerced float covariates.
    treatment : np.ndarray of shape (n_samples,) of int
        Coerced binary treatment indicator.
    y : np.ndarray of shape (n_samples,)
        Coerced float outcome.

    Raises
    ------
    ValueError
        If ``treatment`` is not binary or is missing one of the two groups.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.ndim != 2:
        raise ValueError(
            "X must be a 2D array of shape (n_samples, n_features)"
        )
    n_samples = X.shape[0]

    treatment = np.asarray(treatment, dtype=float)
    if treatment.ndim != 1:
        treatment = np.ravel(treatment)
    y = np.asarray(y, dtype=float)
    if y.ndim != 1:
        y = np.ravel(y)

    if treatment.shape[0] != n_samples or y.shape[0] != n_samples:
        raise ValueError(
            "X, treatment and y must all have the same number of samples "
            f"(got {n_samples}, {treatment.shape[0]} and {y.shape[0]})"
        )

    unique = np.unique(treatment)
    if not np.all(np.isin(unique, [0.0, 1.0])):
        raise ValueError(
            "treatment must be binary (0 = control, 1 = treated); "
            f"got values {unique.tolist()}"
        )

    treatment = treatment.astype(int)
    n_treated = int(np.sum(treatment))
    n_control = int(treatment.size - n_treated)
    if n_treated == 0 or n_control == 0:
        raise ValueError(
            "treatment must contain both groups; got "
            f"{n_treated} treated and {n_control} control samples"
        )

    return X, treatment, y


def _clone_estimator(estimator):
    """Return a fresh, unfitted copy of ``estimator``.

    Accepts either an instance (deep-copied) or a class (instantiated with
    no arguments), so callers may pass ``DecisionTreeRegressor`` or
    ``DecisionTreeRegressor(max_depth=3)`` interchangeably.

    Parameters
    ----------
    estimator : type or object
        The estimator (or its class) to clone.

    Returns
    -------
    clone : object
        A new estimator instance with the same configuration.
    """
    if isinstance(estimator, type):
        return estimator()
    return copy.deepcopy(estimator)


# =============================================================================
# S-Learner
# =============================================================================

@uplift(tags=["causal", "meta-learner", "uplift"], version="1.0.0")
class SLearner(UpliftModel):
    """S-learner: a single model on ``[X, treatment]``.

    Summary
    -------
    The **S-learner** stacks the treatment indicator onto the covariates and
    fits **one** model :math:`f(X, t)`. The uplift is the difference between
    the two counterfactual predictions:

    .. math::
        \\hat{\\tau}(x) = f(x, 1) - f(x, 0).

    Overview
    --------
    1. Append the treatment column to ``X``.
    2. Fit a single regressor on the augmented ``[X, treatment]``.
    3. Predict the uplift as ``f(X, 1) - f(X, 0)``.

    Theory
    ------
    The S-learner lets a single model learn both response surfaces jointly,
    regularizing them toward each other. This is efficient when the two
    surfaces are similar, but the treatment indicator can be ignored by
    flexible learners (its signal diluted among the other features), which
    shrinks the estimated uplift toward zero.

    Parameters
    ----------
    estimator : object, default=DecisionTreeRegressor()
        A TuiML regressor (instance or class) with ``fit``/``predict``.

    Attributes
    ----------
    model_ : object
        The fitted regressor on ``[X, treatment]``.
    n_features_in_ : int
        Number of features in ``X`` (without the treatment column).
    n_treated_, n_control_ : int
        Number of samples in each treatment group.

    Notes
    -----
    **Complexity:** one supervised fit plus ``2`` predictions per sample.

    **When to use:** a strong default; the T- and X-learners beat it when the
    treatment groups are imbalanced or their response surfaces differ a lot.

    References
    ----------
    .. [Kunzel2019] Kunzel, S.R., Sekhon, J.S., Bickel, P.J. and Yu, B. (2019).
       **Metalearners for estimating heterogeneous treatment effects using
       machine learning.** *Proceedings of the National Academy of Sciences*,
       116(10), 4156-4165. DOI: `10.1073/pnas.1804597116
       <https://doi.org/10.1073/pnas.1804597116>`_

    See Also
    --------
    :class:`~tuiml.algorithms.causal.TLearner` : Two separate group models.
    :class:`~tuiml.algorithms.causal.XLearner` : T-learner plus imputed effects.

    Examples
    --------
    >>> from tuiml.algorithms.causal import SLearner
    >>> from tuiml.algorithms.trees import DecisionTreeRegressor
    >>> import numpy as np
    >>> rng = np.random.RandomState(0)
    >>> X = rng.uniform(-1, 1, size=(300, 2))
    >>> t = rng.randint(0, 2, size=300)
    >>> y = 1.0 + X[:, 1] + t * (2.0 * X[:, 0]) + rng.normal(0, 0.1, size=300)
    >>> model = SLearner(DecisionTreeRegressor(max_depth=4)).fit(X, t, y)
    >>> model.predict_uplift(X).shape
    (300,)
    """

    def __init__(self, estimator: Optional[object] = None):
        """Initialize the S-learner.

        Parameters
        ----------
        estimator : object, optional
            A TuiML regressor (instance or class). Defaults to a
            :class:`~tuiml.algorithms.trees.DecisionTreeRegressor`.
        """
        super().__init__()
        if estimator is None:
            from tuiml.algorithms.trees import DecisionTreeRegressor

            estimator = DecisionTreeRegressor()
        self.estimator = estimator

        # Fitted attributes
        self.model_ = None
        self.n_features_in_ = None
        self.n_treated_ = None
        self.n_control_ = None

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        """Return JSON Schema for constructor parameters."""
        return {
            "estimator": {
                "type": "object",
                "default": None,
                "description": "TuiML regressor (instance or class) used as the base model",
            },
        }

    @classmethod
    def get_capabilities(cls) -> List[str]:
        """Return supported capabilities."""
        return ["numeric", "uplift", "binary_treatment", "continuous_outcome"]

    @classmethod
    def get_complexity(cls) -> str:
        """Return complexity analysis."""
        return "One supervised fit; prediction is 2 forward passes per sample"

    @classmethod
    def get_references(cls) -> List[str]:
        """Return academic citations."""
        return [
            "Kunzel, S.R., Sekhon, J.S., Bickel, P.J. and Yu, B. (2019). "
            "Metalearners for estimating heterogeneous treatment effects using "
            "machine learning. PNAS, 116(10), 4156-4165.",
        ]

    def fit(self, X, treatment, y) -> "SLearner":
        """Fit a single model on the augmented ``[X, treatment]``.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Covariates.
        treatment : np.ndarray of shape (n_samples,)
            Binary treatment indicator.
        y : np.ndarray of shape (n_samples,)
            Numeric outcome.

        Returns
        -------
        self : SLearner
            Fitted estimator.
        """
        X, treatment, y = _check_arrays(X, treatment, y)
        self.n_features_in_ = X.shape[1]
        self.n_treated_ = int(np.sum(treatment))
        self.n_control_ = int(treatment.size - self.n_treated_)

        X_aug = np.column_stack([X, treatment.astype(float)])
        self.model_ = _clone_estimator(self.estimator).fit(X_aug, y)

        self._is_fitted = True
        return self

    def predict_uplift(self, X: np.ndarray) -> np.ndarray:
        """Return the predicted uplift ``f(X, 1) - f(X, 0)``.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Covariates.

        Returns
        -------
        uplift : np.ndarray of shape (n_samples,)
            Predicted individual treatment effect.
        """
        self._check_is_fitted()
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        n = X.shape[0]
        treated = np.column_stack([X, np.ones(n)])
        control = np.column_stack([X, np.zeros(n)])
        return np.asarray(self.model_.predict(treated)) - np.asarray(
            self.model_.predict(control)
        )


# =============================================================================
# T-Learner
# =============================================================================

@uplift(tags=["causal", "meta-learner", "uplift"], version="1.0.0")
class TLearner(UpliftModel):
    """T-learner: two models, one per treatment group.

    Summary
    -------
    The **T-learner** fits **two** regressors :math:`f_0` and :math:`f_1` on
    the control and treated samples separately. The uplift is their
    difference:

    .. math::
        \\hat{\\tau}(x) = f_1(x) - f_0(x).

    Overview
    --------
    1. Split ``(X, y)`` by the treatment indicator.
    2. Fit :math:`f_0` on the control group and :math:`f_1` on the treated.
    3. Predict the uplift as :math:`f_1(X) - f_0(X)`.

    Theory
    ------
    Each group gets its own response surface, so a strong treatment signal in
    one group cannot be diluted by the other. The trade-off is data
    efficiency: each model sees only its own group, which can hurt when one
    group is small or the surfaces share a lot of structure.

    Parameters
    ----------
    estimator : object, default=DecisionTreeRegressor()
        A TuiML regressor (instance or class) with ``fit``/``predict``.

    Attributes
    ----------
    model_0_ : object
        Fitted regressor on the control group.
    model_1_ : object
        Fitted regressor on the treated group.
    n_features_in_ : int
        Number of features in ``X``.
    n_treated_, n_control_ : int
        Number of samples in each treatment group.

    Notes
    -----
    **Complexity:** two supervised fits plus two predictions per sample.

    **When to use:** the two group models are genuinely independent, which
    makes the T-learner the cleanest baseline when treatment groups are
    balanced and large.

    References
    ----------
    .. [Kunzel2019] Kunzel, S.R., Sekhon, J.S., Bickel, P.J. and Yu, B. (2019).
       **Metalearners for estimating heterogeneous treatment effects using
       machine learning.** *Proceedings of the National Academy of Sciences*,
       116(10), 4156-4165. DOI: `10.1073/pnas.1804597116
       <https://doi.org/10.1073/pnas.1804597116>`_

    See Also
    --------
    :class:`~tuiml.algorithms.causal.SLearner` : A single shared model.
    :class:`~tuiml.algorithms.causal.XLearner` : T-learner plus imputed effects.

    Examples
    --------
    >>> from tuiml.algorithms.causal import TLearner
    >>> from tuiml.algorithms.trees import DecisionTreeRegressor
    >>> import numpy as np
    >>> rng = np.random.RandomState(0)
    >>> X = rng.uniform(-1, 1, size=(300, 2))
    >>> t = rng.randint(0, 2, size=300)
    >>> y = 1.0 + X[:, 1] + t * (2.0 * X[:, 0]) + rng.normal(0, 0.1, size=300)
    >>> model = TLearner(DecisionTreeRegressor(max_depth=4)).fit(X, t, y)
    >>> model.predict_uplift(X).shape
    (300,)
    """

    def __init__(self, estimator: Optional[object] = None):
        """Initialize the T-learner.

        Parameters
        ----------
        estimator : object, optional
            A TuiML regressor (instance or class). Defaults to a
            :class:`~tuiml.algorithms.trees.DecisionTreeRegressor`.
        """
        super().__init__()
        if estimator is None:
            from tuiml.algorithms.trees import DecisionTreeRegressor

            estimator = DecisionTreeRegressor()
        self.estimator = estimator

        # Fitted attributes
        self.model_0_ = None
        self.model_1_ = None
        self.n_features_in_ = None
        self.n_treated_ = None
        self.n_control_ = None

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        """Return JSON Schema for constructor parameters."""
        return {
            "estimator": {
                "type": "object",
                "default": None,
                "description": "TuiML regressor (instance or class) used for each group model",
            },
        }

    @classmethod
    def get_capabilities(cls) -> List[str]:
        """Return supported capabilities."""
        return ["numeric", "uplift", "binary_treatment", "continuous_outcome"]

    @classmethod
    def get_complexity(cls) -> str:
        """Return complexity analysis."""
        return "Two supervised fits; prediction is 2 forward passes per sample"

    @classmethod
    def get_references(cls) -> List[str]:
        """Return academic citations."""
        return [
            "Kunzel, S.R., Sekhon, J.S., Bickel, P.J. and Yu, B. (2019). "
            "Metalearners for estimating heterogeneous treatment effects using "
            "machine learning. PNAS, 116(10), 4156-4165.",
        ]

    def fit(self, X, treatment, y) -> "TLearner":
        """Fit separate models for the treated and control groups.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Covariates.
        treatment : np.ndarray of shape (n_samples,)
            Binary treatment indicator.
        y : np.ndarray of shape (n_samples,)
            Numeric outcome.

        Returns
        -------
        self : TLearner
            Fitted estimator.
        """
        X, treatment, y = _check_arrays(X, treatment, y)
        self.n_features_in_ = X.shape[1]
        self.n_treated_ = int(np.sum(treatment))
        self.n_control_ = int(treatment.size - self.n_treated_)

        mask_1 = treatment == 1
        mask_0 = ~mask_1

        self.model_0_ = _clone_estimator(self.estimator).fit(X[mask_0], y[mask_0])
        self.model_1_ = _clone_estimator(self.estimator).fit(X[mask_1], y[mask_1])

        self._is_fitted = True
        return self

    def predict_uplift(self, X: np.ndarray) -> np.ndarray:
        """Return the predicted uplift ``f_1(X) - f_0(X)``.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Covariates.

        Returns
        -------
        uplift : np.ndarray of shape (n_samples,)
            Predicted individual treatment effect.
        """
        self._check_is_fitted()
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return np.asarray(self.model_1_.predict(X)) - np.asarray(
            self.model_0_.predict(X)
        )


# =============================================================================
# X-Learner
# =============================================================================

@uplift(tags=["causal", "meta-learner", "uplift"], version="1.0.0")
class XLearner(UpliftModel):
    """X-learner: T-learner plus cross-group imputed-effect models.

    Summary
    -------
    The **X-learner** starts from a T-learner, then fits two further models on
    the *imputed* treatment effects — the residuals each group leaves behind
    under the other group's model — and combines them with a propensity
    weight.

    Overview
    --------
    1. Fit the T-learner response models :math:`f_0` (control) and
       :math:`f_1` (treated).
    2. Impute the effect for each treated unit
       :math:`D_1 = y_1 - f_0(X_1)` and each control unit
       :math:`D_0 = f_1(X_0) - y_0`.
    3. Fit :math:`\\tau_1` on ``(X_1, D_1)`` and :math:`\\tau_0` on
       ``(X_0, D_0)``.
    4. Predict :math:`\\hat{\\tau}(x) = p(x)\\,\\tau_0(x) +
       (1 - p(x))\\,\\tau_1(x)`, where :math:`p(x)` is the propensity score.

    Theory
    ------
    The imputed effect :math:`D_i` is a noisy, per-unit estimate of the
    individual treatment effect: for a treated unit it is the outcome above
    what the control model would have predicted; for a control unit it is the
    outcome below what the treated model would have predicted. Modeling these
    imputed effects directly recovers :math:`\\tau(x)` even when one group is
    much smaller than the other, and the propensity-weighted combination
    regularizes the two estimates toward the model with more local data.

    Parameters
    ----------
    estimator : object, default=DecisionTreeRegressor()
        A TuiML regressor (instance or class) used for all four sub-models.
    propensity_model : object or None, default=None
        A TuiML classifier (instance or class) used to estimate
        :math:`P(\\text{treatment} = 1 \\mid X)`. If ``None``, a constant
        propensity equal to the overall treatment rate is used.

    Attributes
    ----------
    model_0_ : object
        Fitted control response model :math:`f_0`.
    model_1_ : object
        Fitted treated response model :math:`f_1`.
    tau_0_ : object
        Fitted imputed-effect model on control units.
    tau_1_ : object
        Fitted imputed-effect model on treated units.
    propensity_model_ : object or None
        Fitted propensity model (``None`` when a constant propensity is used).
    propensity_ : float
        Constant propensity (overall treatment rate) when no model is given.
    n_features_in_ : int
        Number of features in ``X``.
    n_treated_, n_control_ : int
        Number of samples in each treatment group.

    Notes
    -----
    **Complexity:** four supervised fits plus four predictions per sample.

    **When to use:** imbalanced treatment groups, or when the base learners
    for the two groups are not equally accurate.

    References
    ----------
    .. [Kunzel2019] Kunzel, S.R., Sekhon, J.S., Bickel, P.J. and Yu, B. (2019).
       **Metalearners for estimating heterogeneous treatment effects using
       machine learning.** *Proceedings of the National Academy of Sciences*,
       116(10), 4156-4165. DOI: `10.1073/pnas.1804597116
       <https://doi.org/10.1073/pnas.1804597116>`_

    See Also
    --------
    :class:`~tuiml.algorithms.causal.SLearner` : A single shared model.
    :class:`~tuiml.algorithms.causal.TLearner` : Two separate group models.

    Examples
    --------
    >>> from tuiml.algorithms.causal import XLearner
    >>> from tuiml.algorithms.trees import DecisionTreeRegressor
    >>> import numpy as np
    >>> rng = np.random.RandomState(0)
    >>> X = rng.uniform(-1, 1, size=(300, 2))
    >>> t = rng.randint(0, 2, size=300)
    >>> y = 1.0 + X[:, 1] + t * (2.0 * X[:, 0]) + rng.normal(0, 0.1, size=300)
    >>> model = XLearner(DecisionTreeRegressor(max_depth=4)).fit(X, t, y)
    >>> model.predict_uplift(X).shape
    (300,)
    """

    def __init__(
        self,
        estimator: Optional[object] = None,
        propensity_model: Optional[object] = None,
    ):
        """Initialize the X-learner.

        Parameters
        ----------
        estimator : object, optional
            A TuiML regressor (instance or class). Defaults to a
            :class:`~tuiml.algorithms.trees.DecisionTreeRegressor`.
        propensity_model : object or None, default=None
            A TuiML classifier used to estimate the propensity score. When
            ``None``, a constant propensity equal to the treatment rate is
            used.
        """
        super().__init__()
        if estimator is None:
            from tuiml.algorithms.trees import DecisionTreeRegressor

            estimator = DecisionTreeRegressor()
        self.estimator = estimator
        self.propensity_model = propensity_model

        # Fitted attributes
        self.model_0_ = None
        self.model_1_ = None
        self.tau_0_ = None
        self.tau_1_ = None
        self.propensity_model_ = None
        self.propensity_ = None
        self.n_features_in_ = None
        self.n_treated_ = None
        self.n_control_ = None

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        """Return JSON Schema for constructor parameters."""
        return {
            "estimator": {
                "type": "object",
                "default": None,
                "description": "TuiML regressor (instance or class) used for all sub-models",
            },
            "propensity_model": {
                "type": ["object", "null"],
                "default": None,
                "description": "TuiML classifier for the propensity score; None uses a constant",
            },
        }

    @classmethod
    def get_capabilities(cls) -> List[str]:
        """Return supported capabilities."""
        return ["numeric", "uplift", "binary_treatment", "continuous_outcome"]

    @classmethod
    def get_complexity(cls) -> str:
        """Return complexity analysis."""
        return "Four supervised fits; prediction is up to 4 forward passes per sample"

    @classmethod
    def get_references(cls) -> List[str]:
        """Return academic citations."""
        return [
            "Kunzel, S.R., Sekhon, J.S., Bickel, P.J. and Yu, B. (2019). "
            "Metalearners for estimating heterogeneous treatment effects using "
            "machine learning. PNAS, 116(10), 4156-4165.",
        ]

    def fit(self, X, treatment, y) -> "XLearner":
        """Fit the T-learner, the imputed-effect models, and the propensity.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Covariates.
        treatment : np.ndarray of shape (n_samples,)
            Binary treatment indicator.
        y : np.ndarray of shape (n_samples,)
            Numeric outcome.

        Returns
        -------
        self : XLearner
            Fitted estimator.
        """
        X, treatment, y = _check_arrays(X, treatment, y)
        self.n_features_in_ = X.shape[1]
        self.n_treated_ = int(np.sum(treatment))
        self.n_control_ = int(treatment.size - self.n_treated_)

        mask_1 = treatment == 1
        mask_0 = ~mask_1
        X_0, y_0 = X[mask_0], y[mask_0]
        X_1, y_1 = X[mask_1], y[mask_1]

        # Step 1: T-learner response models.
        self.model_0_ = _clone_estimator(self.estimator).fit(X_0, y_0)
        self.model_1_ = _clone_estimator(self.estimator).fit(X_1, y_1)

        # Step 2: imputed effects (cross-group residuals).
        d_1 = y_1 - np.asarray(self.model_0_.predict(X_1))  # treated units
        d_0 = np.asarray(self.model_1_.predict(X_0)) - y_0  # control units

        # Step 3: imputed-effect models.
        self.tau_1_ = _clone_estimator(self.estimator).fit(X_1, d_1)
        self.tau_0_ = _clone_estimator(self.estimator).fit(X_0, d_0)

        # Step 4: propensity model (or a constant fallback).
        if self.propensity_model is None:
            self.propensity_model_ = None
            self.propensity_ = float(self.n_treated_ / treatment.size)
        else:
            pm = _clone_estimator(self.propensity_model)
            self.propensity_model_ = pm.fit(X, treatment)
            self.propensity_ = float(self.n_treated_ / treatment.size)

        self._is_fitted = True
        return self

    def _propensity(self, X: np.ndarray) -> np.ndarray:
        """Return the propensity :math:`P(\\text{treatment}=1 \\mid X)`.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Covariates.

        Returns
        -------
        propensity : np.ndarray of shape (n_samples,)
            Estimated treatment probability for each sample.
        """
        if self.propensity_model_ is None:
            return np.full(X.shape[0], self.propensity_)

        pm = self.propensity_model_
        if hasattr(pm, "predict_proba"):
            proba = np.asarray(pm.predict_proba(X))
            if proba.ndim == 2 and proba.shape[1] >= 2:
                return np.clip(proba[:, 1], 0.0, 1.0)
        return np.clip(np.asarray(pm.predict(X)).ravel(), 0.0, 1.0)

    def predict_uplift(self, X: np.ndarray) -> np.ndarray:
        """Return the propensity-weighted imputed-effect prediction.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Covariates.

        Returns
        -------
        uplift : np.ndarray of shape (n_samples,)
            Predicted individual treatment effect.
        """
        self._check_is_fitted()
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        tau_0 = np.asarray(self.tau_0_.predict(X))
        tau_1 = np.asarray(self.tau_1_.predict(X))
        p = self._propensity(X)
        return p * tau_0 + (1.0 - p) * tau_1
