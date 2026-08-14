"""Platt (sigmoid) probability calibration."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from tuiml.uncertainty._base import Calibrator


class PlattCalibrator(Calibrator):
    """Probability calibration by fitting a **sigmoid** to held-out scores.

    Platt scaling maps a raw score :math:`s` onto a probability through a
    one-dimensional **logistic link** whose two parameters are fitted by
    maximum likelihood on a calibration set. It is the standard remedy for the
    margin-like, uncalibrated outputs of SVMs and boosted ensembles.

    Overview
    --------
    1. Hold out a calibration set that the model was **not** trained on.
    2. Replace the hard labels with Platt's regularised targets, which pull the
       fit away from 0 and 1 and prevent overfitting on small samples.
    3. Fit :math:`A` and :math:`B` by Newton descent on the log-likelihood.
    4. At transform time, apply the fitted sigmoid to new scores.

    Theory
    ------
    The calibration map is

    .. math::
        P(y = 1 \\mid s) = \\frac{1}{1 + \\exp(A s + B)}

    fitted by minimising the regularised cross-entropy

    .. math::
        -\\sum_i t_i \\log p_i + (1 - t_i) \\log (1 - p_i)

    where the targets follow Platt's correction for :math:`N_+` positive and
    :math:`N_-` negative calibration samples:

    .. math::
        t_i = \\frac{N_+ + 1}{N_+ + 2} \\ \\text{if } y_i = 1, \\quad
        t_i = \\frac{1}{N_- + 2} \\ \\text{otherwise}

    A negative :math:`A` gives the usual increasing map; the sign is learned,
    so the calibrator also handles scores oriented the other way round.

    Parameters
    ----------
    max_iter : int, default=100
        Maximum number of Newton iterations.
    tol : float, default=1e-10
        Convergence tolerance on the gradient norm.
    regularize_targets : bool, default=True
        Whether to use Platt's regularised targets instead of the raw 0/1
        labels. Strongly recommended on small calibration sets.

    Attributes
    ----------
    a_ : float
        Fitted slope of the sigmoid.
    b_ : float
        Fitted intercept of the sigmoid.
    n_iter_ : int
        Newton iterations actually performed.
    classes_ : np.ndarray of shape (n_classes,)
        Class labels seen during :meth:`fit`.
    fitted_ : bool
        Whether :meth:`fit` has been called.

    Notes
    -----
    **Complexity.** :math:`O(n)` per Newton iteration, :math:`O(1)` memory
    beyond the score vector. Transform is :math:`O(m)`.

    **When to use.** Platt scaling is the right default when the calibration
    set is small (a few hundred samples) or the miscalibration is a smooth
    monotone squashing — the typical SVM or AdaBoost case. When the
    calibration set is large or the distortion is not sigmoidal, prefer
    :class:`~tuiml.uncertainty.IsotonicCalibrator`.

    References
    ----------
    .. [Platt1999] Platt, J. (1999). Probabilistic Outputs for Support Vector
       Machines and Comparisons to Regularized Likelihood Methods.
       *Advances in Large Margin Classifiers*, 61-74.
    .. [Lin2007] Lin, H.-T., Lin, C.-J., & Weng, R. C. (2007). A Note on
       Platt's Probabilistic Outputs for Support Vector Machines.
       *Machine Learning*, 68(3), 267-276. :doi:`10.1007/s10994-007-5018-6`

    See Also
    --------
    :class:`~tuiml.uncertainty.IsotonicCalibrator` : Non-parametric calibration.
    :class:`~tuiml.uncertainty.TemperatureScaler` : Multiclass single-parameter calibration.

    Examples
    --------
    >>> import numpy as np
    >>> from tuiml.uncertainty import PlattCalibrator
    >>> rng = np.random.default_rng(0)
    >>> scores = np.concatenate([rng.normal(-1, 1, 200), rng.normal(1, 1, 200)])
    >>> y = np.concatenate([np.zeros(200), np.ones(200)])
    >>> cal = PlattCalibrator().fit(scores, y)
    >>> proba = cal.transform(np.array([-2.0, 0.0, 2.0]))
    >>> bool(proba[0] < proba[1] < proba[2])
    True
    """

    def __init__(
        self,
        max_iter: int = 100,
        tol: float = 1e-10,
        regularize_targets: bool = True,
    ) -> None:
        """Initialise the Platt calibrator.

        Parameters
        ----------
        max_iter : int, default=100
            Maximum number of Newton iterations.
        tol : float, default=1e-10
            Convergence tolerance on the gradient norm.
        regularize_targets : bool, default=True
            Whether to use Platt's regularised targets.
        """
        super().__init__()
        self.max_iter = max_iter
        self.tol = tol
        self.regularize_targets = regularize_targets
        self.a_: float = 0.0
        self.b_: float = 0.0
        self.n_iter_: int = 0
        self.classes_: Optional[np.ndarray] = None

    def fit(self, scores: np.ndarray, y: np.ndarray) -> "PlattCalibrator":
        """Fit the sigmoid on held-out scores.

        Parameters
        ----------
        scores : np.ndarray of shape (n_samples,) or (n_samples, 2)
            Uncalibrated scores or decision values.
        y : np.ndarray of shape (n_samples,)
            True binary labels.

        Returns
        -------
        self : PlattCalibrator
            The fitted calibrator.
        """
        s = self._as_positive_scores(scores)
        y = np.asarray(y)

        self.classes_ = np.unique(y)
        if self.classes_.size > 2:
            raise ValueError(
                "PlattCalibrator handles binary problems; got "
                f"{self.classes_.size} classes. Calibrate one-vs-rest or use "
                "TemperatureScaler."
            )
        positive = y == self.classes_[-1]

        n_pos = int(positive.sum())
        n_neg = int(positive.size - n_pos)
        if self.regularize_targets:
            hi = (n_pos + 1.0) / (n_pos + 2.0)
            lo = 1.0 / (n_neg + 2.0)
        else:
            hi, lo = 1.0, 0.0
        t = np.where(positive, hi, lo)

        # Platt's own initialisation: zero slope, log-odds intercept.
        a = 0.0
        b = float(np.log((n_neg + 1.0) / (n_pos + 1.0)))

        for iteration in range(self.max_iter):
            p = self._sigmoid(a * s + b)
            # Gradient of the cross-entropy w.r.t. (a, b). With
            # p = 1/(1+exp(z)) the derivative of the loss in z is (t - p).
            diff = t - p
            grad_a = float(np.dot(diff, s))
            grad_b = float(diff.sum())
            if abs(grad_a) + abs(grad_b) < self.tol:
                break

            # Hessian of the logistic loss; always positive semi-definite.
            v = p * (1.0 - p)
            h11 = float(np.dot(v, s * s))
            h12 = float(np.dot(v, s))
            h22 = float(v.sum())

            # Levenberg-style ridge keeps the Newton step defined when the
            # calibration scores are near-separable and v collapses to zero.
            ridge = 1e-12
            det = (h11 + ridge) * (h22 + ridge) - h12 * h12
            if det <= 0.0:
                break
            step_a = -((h22 + ridge) * grad_a - h12 * grad_b) / det
            step_b = -(-h12 * grad_a + (h11 + ridge) * grad_b) / det

            a, b = self._line_search(s, t, a, b, step_a, step_b)
            self.n_iter_ = iteration + 1

        self.a_, self.b_ = a, b
        self.fitted_ = True
        return self

    def transform(self, scores: np.ndarray) -> np.ndarray:
        """Map raw scores onto calibrated probabilities.

        Parameters
        ----------
        scores : np.ndarray of shape (n_samples,) or (n_samples, 2)
            Uncalibrated scores.

        Returns
        -------
        proba : np.ndarray of shape (n_samples,)
            Calibrated probability of the positive class.
        """
        self._check_is_fitted()
        s = self._as_positive_scores(scores)
        return self._sigmoid(self.a_ * s + self.b_)

    def predict_proba(self, scores: np.ndarray) -> np.ndarray:
        """Return two-column calibrated probabilities.

        Parameters
        ----------
        scores : np.ndarray of shape (n_samples,) or (n_samples, 2)
            Uncalibrated scores.

        Returns
        -------
        proba : np.ndarray of shape (n_samples, 2)
            Calibrated probabilities for the negative and positive class.
        """
        positive = self.transform(scores)
        return np.column_stack([1.0 - positive, positive])

    def _line_search(
        self,
        s: np.ndarray,
        t: np.ndarray,
        a: float,
        b: float,
        step_a: float,
        step_b: float,
    ) -> tuple:
        """Backtrack along the Newton direction until the loss decreases.

        Parameters
        ----------
        s : np.ndarray of shape (n_samples,)
            Calibration scores.
        t : np.ndarray of shape (n_samples,)
            Regularised targets.
        a, b : float
            Current sigmoid parameters.
        step_a, step_b : float
            Newton step for each parameter.

        Returns
        -------
        a, b : float
            Updated parameters. Unchanged when no step reduces the loss.
        """
        current = self._loss(s, t, a, b)
        scale = 1.0
        for _ in range(30):
            trial_a = a + scale * step_a
            trial_b = b + scale * step_b
            if self._loss(s, t, trial_a, trial_b) < current:
                return trial_a, trial_b
            scale *= 0.5
        return a, b

    def _loss(self, s: np.ndarray, t: np.ndarray, a: float, b: float) -> float:
        """Compute the regularised cross-entropy of a parameter pair.

        Parameters
        ----------
        s : np.ndarray of shape (n_samples,)
            Calibration scores.
        t : np.ndarray of shape (n_samples,)
            Regularised targets.
        a, b : float
            Sigmoid parameters.

        Returns
        -------
        loss : float
            Mean cross-entropy.
        """
        z = a * s + b
        # -[t log p + (1-t) log(1-p)] with p = 1/(1+exp(z)) collapses to
        # log(1+exp(z)) - (1-t) z; logaddexp keeps the first term stable.
        return float(np.mean(np.logaddexp(0.0, z) - (1.0 - t) * z))

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        """Evaluate ``1 / (1 + exp(z))`` without overflow.

        Parameters
        ----------
        z : np.ndarray
            Linear predictor values.

        Returns
        -------
        p : np.ndarray
            Sigmoid output in ``(0, 1)``.
        """
        out = np.empty_like(z, dtype=np.float64)
        positive = z >= 0
        # Two branches keep exp() away from overflow in either tail.
        exp_neg = np.exp(-z[positive])
        out[positive] = exp_neg / (1.0 + exp_neg)
        exp_pos = np.exp(z[~positive])
        out[~positive] = 1.0 / (1.0 + exp_pos)
        return out

    @staticmethod
    def _as_positive_scores(scores: np.ndarray) -> np.ndarray:
        """Reduce a score array to a 1-D positive-class score vector.

        Parameters
        ----------
        scores : np.ndarray of shape (n_samples,) or (n_samples, 2)
            Uncalibrated scores or two-column probabilities.

        Returns
        -------
        s : np.ndarray of shape (n_samples,)
            Positive-class scores as float64.
        """
        s = np.asarray(scores, dtype=np.float64)
        if s.ndim == 2:
            if s.shape[1] != 2:
                raise ValueError(
                    "2-D scores must have exactly 2 columns for binary "
                    f"calibration, got shape {s.shape}"
                )
            s = s[:, 1]
        elif s.ndim != 1:
            raise ValueError(f"scores must be 1-D or 2-D, got {s.ndim}-D")
        return np.ascontiguousarray(s)

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Any]:
        """Return JSON Schema for constructor parameters."""
        return {
            "type": "object",
            "properties": {
                "max_iter": {
                    "type": "integer",
                    "default": 100,
                    "minimum": 1,
                    "description": "Maximum number of Newton iterations.",
                },
                "tol": {
                    "type": "number",
                    "default": 1e-10,
                    "description": "Convergence tolerance on the gradient norm.",
                },
                "regularize_targets": {
                    "type": "boolean",
                    "default": True,
                    "description": "Use Platt's regularised targets.",
                },
            },
        }

    def __repr__(self) -> str:
        """Return a readable representation of the calibrator."""
        if self.fitted_:
            return f"PlattCalibrator(a={self.a_:.4f}, b={self.b_:.4f})"
        return f"PlattCalibrator(max_iter={self.max_iter})"
