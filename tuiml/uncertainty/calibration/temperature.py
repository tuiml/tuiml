"""Temperature and vector scaling for multiclass probability calibration."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from tuiml.uncertainty._base import Calibrator


def _log_softmax(logits: np.ndarray) -> np.ndarray:
    """Compute the log-softmax of a logit matrix without overflow.

    Parameters
    ----------
    logits : np.ndarray of shape (n_samples, n_classes)
        Unnormalised scores.

    Returns
    -------
    log_proba : np.ndarray of shape (n_samples, n_classes)
        Row-wise log-softmax.
    """
    shifted = logits - logits.max(axis=1, keepdims=True)
    return shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))


class TemperatureScaler(Calibrator):
    """Multiclass calibration by dividing logits by a single **temperature**.

    Temperature scaling rescales every logit by one learned scalar
    :math:`T > 0`. Because a positive scalar cannot change the ranking of the
    logits, the calibrated model has **exactly the same accuracy** as the
    original — only its confidence changes. This is the standard fix for the
    over-confidence of modern neural networks.

    Overview
    --------
    1. Collect logits on a held-out calibration set.
    2. Fit a single temperature :math:`T` by minimising the negative
       log-likelihood with a bounded 1-D search.
    3. At transform time, apply :math:`\\text{softmax}(z / T)`.

    A fitted :math:`T > 1` softens an over-confident model; :math:`T < 1`
    sharpens an under-confident one.

    Theory
    ------
    The calibrated probability of class :math:`k` is

    .. math::
        p_k = \\frac{\\exp(z_k / T)}{\\sum_j \\exp(z_j / T)}

    and :math:`T` minimises the calibration-set cross-entropy

    .. math::
        \\mathcal{L}(T) = -\\frac{1}{n} \\sum_{i=1}^{n}
        \\log p_{y_i}(z_i / T)

    which is convex in :math:`1/T`, so a golden-section search on
    :math:`\\log T` finds the global optimum without gradients.

    Parameters
    ----------
    max_iter : int, default=200
        Maximum number of golden-section iterations.
    tol : float, default=1e-6
        Convergence tolerance on the bracketed interval in ``log(T)``.
    log_t_bounds : tuple of float, default=(-4.0, 4.0)
        Search bracket for :math:`\\log T`, i.e. roughly ``T`` in
        ``[0.018, 54.6]``.

    Attributes
    ----------
    temperature_ : float
        The fitted temperature.
    n_iter_ : int
        Golden-section iterations performed.
    classes_ : np.ndarray of shape (n_classes,)
        Class labels seen during :meth:`fit`.
    fitted_ : bool
        Whether :meth:`fit` has been called.

    Notes
    -----
    **Complexity.** :math:`O(n \\cdot c)` per iteration for ``n`` samples and
    ``c`` classes; the number of iterations is fixed by ``tol``, not by ``n``.

    **When to use.** Use temperature scaling for any multiclass model whose
    ranking must not change — the accuracy-preserving property is the reason it
    is preferred over per-class isotonic calibration for deep networks. It
    cannot fix class-dependent bias; reach for
    :class:`~tuiml.uncertainty.VectorScaler` when different classes are
    miscalibrated in different directions.

    References
    ----------
    .. [Guo2017] Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On
       Calibration of Modern Neural Networks. *ICML*, 1321-1330.
       :arxiv:`1706.04599`

    See Also
    --------
    :class:`~tuiml.uncertainty.VectorScaler` : Per-class scale and bias.
    :class:`~tuiml.uncertainty.PlattCalibrator` : Binary sigmoid calibration.
    :func:`~tuiml.uncertainty.expected_calibration_error` : Measures the improvement.

    Examples
    --------
    >>> import numpy as np
    >>> from tuiml.uncertainty import TemperatureScaler
    >>> from tuiml.uncertainty import expected_calibration_error
    >>> rng = np.random.default_rng(0)
    >>> y = rng.integers(0, 3, 300)
    >>> noisy = np.where(rng.random(300) < 0.3, (y + 1) % 3, y)
    >>> logits = np.eye(3)[noisy] * 6.0 + rng.normal(0, 1.0, (300, 3))
    >>> scaler = TemperatureScaler().fit(logits, y)
    >>> bool(scaler.temperature_ > 1.0)  # the model was over-confident
    True
    >>> proba = scaler.transform(logits)
    >>> bool(expected_calibration_error(y, proba) < 0.1)
    True
    """

    def __init__(
        self,
        max_iter: int = 200,
        tol: float = 1e-6,
        log_t_bounds: tuple = (-4.0, 4.0),
    ) -> None:
        """Initialise the temperature scaler.

        Parameters
        ----------
        max_iter : int, default=200
            Maximum number of golden-section iterations.
        tol : float, default=1e-6
            Convergence tolerance on the bracketed interval in ``log(T)``.
        log_t_bounds : tuple of float, default=(-4.0, 4.0)
            Search bracket for :math:`\\log T`.
        """
        super().__init__()
        self.max_iter = max_iter
        self.tol = tol
        self.log_t_bounds = log_t_bounds
        self.temperature_: float = 1.0
        self.n_iter_: int = 0
        self.classes_: Optional[np.ndarray] = None

    def fit(self, scores: np.ndarray, y: np.ndarray) -> "TemperatureScaler":
        """Fit the temperature on held-out logits.

        Parameters
        ----------
        scores : np.ndarray of shape (n_samples, n_classes)
            Uncalibrated logits. Probabilities are accepted and converted to
            logits internally via ``log``.
        y : np.ndarray of shape (n_samples,)
            True labels.

        Returns
        -------
        self : TemperatureScaler
            The fitted scaler.
        """
        logits = self._as_logits(scores)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        if self.classes_.size != logits.shape[1]:
            raise ValueError(
                f"scores has {logits.shape[1]} columns but y has "
                f"{self.classes_.size} distinct classes"
            )
        y_index = np.searchsorted(self.classes_, y)
        rows = np.arange(logits.shape[0])

        def nll(log_t: float) -> float:
            """Negative log-likelihood at a given log-temperature."""
            return -float(
                _log_softmax(logits / np.exp(log_t))[rows, y_index].mean()
            )

        # Golden-section search: the NLL is unimodal in log(T), so this
        # converges globally without needing a gradient.
        invphi = (np.sqrt(5.0) - 1.0) / 2.0
        lo, hi = self.log_t_bounds
        c = hi - invphi * (hi - lo)
        d = lo + invphi * (hi - lo)
        fc, fd = nll(c), nll(d)

        for iteration in range(self.max_iter):
            if abs(hi - lo) < self.tol:
                break
            if fc < fd:
                hi, d, fd = d, c, fc
                c = hi - invphi * (hi - lo)
                fc = nll(c)
            else:
                lo, c, fc = c, d, fd
                d = lo + invphi * (hi - lo)
                fd = nll(d)
            self.n_iter_ = iteration + 1

        self.temperature_ = float(np.exp((lo + hi) / 2.0))
        self.fitted_ = True
        return self

    def transform(self, scores: np.ndarray) -> np.ndarray:
        """Apply the fitted temperature and return calibrated probabilities.

        Parameters
        ----------
        scores : np.ndarray of shape (n_samples, n_classes)
            Uncalibrated logits or probabilities.

        Returns
        -------
        proba : np.ndarray of shape (n_samples, n_classes)
            Calibrated probabilities, rows summing to one.
        """
        self._check_is_fitted()
        logits = self._as_logits(scores)
        return np.exp(_log_softmax(logits / self.temperature_))

    def predict_proba(self, scores: np.ndarray) -> np.ndarray:
        """Return calibrated probabilities; alias of :meth:`transform`.

        Parameters
        ----------
        scores : np.ndarray of shape (n_samples, n_classes)
            Uncalibrated logits or probabilities.

        Returns
        -------
        proba : np.ndarray of shape (n_samples, n_classes)
            Calibrated probabilities.
        """
        return self.transform(scores)

    @staticmethod
    def _as_logits(scores: np.ndarray) -> np.ndarray:
        """Coerce a score matrix to logits.

        Parameters
        ----------
        scores : np.ndarray of shape (n_samples, n_classes)
            Logits, or probabilities whose rows sum to one.

        Returns
        -------
        logits : np.ndarray of shape (n_samples, n_classes)
            Logit-scale scores as float64.
        """
        s = np.asarray(scores, dtype=np.float64)
        if s.ndim != 2:
            raise ValueError(f"scores must be 2-D (n_samples, n_classes), got {s.ndim}-D")
        # Rows summing to one and bounded in [0, 1] are probabilities; log
        # turns them back into logits up to an irrelevant per-row constant.
        looks_like_proba = (
            np.all(s >= 0.0) and np.all(s <= 1.0)
            and np.allclose(s.sum(axis=1), 1.0, atol=1e-6)
        )
        if looks_like_proba:
            return np.log(np.clip(s, 1e-12, None))
        return s

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Any]:
        """Return JSON Schema for constructor parameters."""
        return {
            "type": "object",
            "properties": {
                "max_iter": {
                    "type": "integer",
                    "default": 200,
                    "minimum": 1,
                    "description": "Maximum golden-section iterations.",
                },
                "tol": {
                    "type": "number",
                    "default": 1e-6,
                    "description": "Tolerance on the log-temperature bracket.",
                },
            },
        }

    def __repr__(self) -> str:
        """Return a readable representation of the scaler."""
        if self.fitted_:
            return f"TemperatureScaler(temperature={self.temperature_:.4f})"
        return f"TemperatureScaler(max_iter={self.max_iter})"


class VectorScaler(Calibrator):
    """Multiclass calibration with a **per-class scale and bias**.

    Vector scaling generalises :class:`TemperatureScaler` by learning one
    weight and one bias per class, :math:`z_k \\mapsto w_k z_k + b_k`. It can
    correct class-dependent miscalibration that a single temperature cannot,
    at the cost of :math:`2c` parameters and the loss of the
    accuracy-preserving guarantee.

    Overview
    --------
    1. Collect logits on a held-out calibration set.
    2. Fit :math:`w` and :math:`b` by gradient descent on the cross-entropy.
    3. At transform time, apply :math:`\\text{softmax}(w \\odot z + b)`.

    Theory
    ------
    The calibration map is

    .. math::
        p_k = \\frac{\\exp(w_k z_k + b_k)}{\\sum_j \\exp(w_j z_j + b_j)}

    with the objective convex in :math:`(w, b)`, so plain gradient descent with
    a decaying step reaches the optimum. The gradient of the mean
    cross-entropy is

    .. math::
        \\nabla_{w_k} \\mathcal{L} = \\frac{1}{n} \\sum_i (p_{ik} - y_{ik}) z_{ik},
        \\quad
        \\nabla_{b_k} \\mathcal{L} = \\frac{1}{n} \\sum_i (p_{ik} - y_{ik})

    Parameters
    ----------
    max_iter : int, default=500
        Number of gradient-descent iterations.
    learning_rate : float, default=0.05
        Initial step size; decayed as :math:`1/\\sqrt{t}`.
    tol : float, default=1e-7
        Stop when the loss improves by less than this between iterations.

    Attributes
    ----------
    weights_ : np.ndarray of shape (n_classes,)
        Per-class logit scale.
    bias_ : np.ndarray of shape (n_classes,)
        Per-class logit offset.
    n_iter_ : int
        Iterations performed.
    classes_ : np.ndarray of shape (n_classes,)
        Class labels seen during :meth:`fit`.
    fitted_ : bool
        Whether :meth:`fit` has been called.

    Notes
    -----
    **Complexity.** :math:`O(n \\cdot c)` per iteration.

    **When to use.** Use vector scaling when different classes are
    miscalibrated in different directions — typically under class imbalance.
    On small calibration sets it overfits where a single temperature would not,
    so compare the two with
    :func:`~tuiml.uncertainty.expected_calibration_error` on a third split.

    References
    ----------
    .. [Guo2017] Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On
       Calibration of Modern Neural Networks. *ICML*, 1321-1330.
       :arxiv:`1706.04599`

    See Also
    --------
    :class:`~tuiml.uncertainty.TemperatureScaler` : Single-parameter, accuracy preserving.

    Examples
    --------
    >>> import numpy as np
    >>> from tuiml.uncertainty import VectorScaler
    >>> rng = np.random.default_rng(0)
    >>> y = rng.integers(0, 3, 300)
    >>> logits = np.eye(3)[y] * 5.0 + rng.normal(0, 1.0, (300, 3))
    >>> scaler = VectorScaler(max_iter=200).fit(logits, y)
    >>> proba = scaler.transform(logits)
    >>> bool(np.allclose(proba.sum(axis=1), 1.0))
    True
    """

    def __init__(
        self,
        max_iter: int = 500,
        learning_rate: float = 0.05,
        tol: float = 1e-7,
    ) -> None:
        """Initialise the vector scaler.

        Parameters
        ----------
        max_iter : int, default=500
            Number of gradient-descent iterations.
        learning_rate : float, default=0.05
            Initial step size.
        tol : float, default=1e-7
            Minimum loss improvement before stopping.
        """
        super().__init__()
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.tol = tol
        self.weights_: Optional[np.ndarray] = None
        self.bias_: Optional[np.ndarray] = None
        self.n_iter_: int = 0
        self.classes_: Optional[np.ndarray] = None

    def fit(self, scores: np.ndarray, y: np.ndarray) -> "VectorScaler":
        """Fit per-class scale and bias on held-out logits.

        Parameters
        ----------
        scores : np.ndarray of shape (n_samples, n_classes)
            Uncalibrated logits or probabilities.
        y : np.ndarray of shape (n_samples,)
            True labels.

        Returns
        -------
        self : VectorScaler
            The fitted scaler.
        """
        logits = TemperatureScaler._as_logits(scores)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        n_samples, n_classes = logits.shape
        if self.classes_.size != n_classes:
            raise ValueError(
                f"scores has {n_classes} columns but y has "
                f"{self.classes_.size} distinct classes"
            )

        y_index = np.searchsorted(self.classes_, y)
        one_hot = np.zeros((n_samples, n_classes))
        one_hot[np.arange(n_samples), y_index] = 1.0

        w = np.ones(n_classes)
        b = np.zeros(n_classes)
        previous = np.inf

        for iteration in range(self.max_iter):
            log_proba = _log_softmax(logits * w + b)
            loss = -float((log_proba * one_hot).sum() / n_samples)
            if abs(previous - loss) < self.tol:
                break
            previous = loss

            residual = np.exp(log_proba) - one_hot
            grad_w = (residual * logits).sum(axis=0) / n_samples
            grad_b = residual.sum(axis=0) / n_samples

            # 1/sqrt(t) decay keeps the convex objective from oscillating
            # once the iterates approach the optimum.
            step = self.learning_rate / np.sqrt(iteration + 1.0)
            w -= step * grad_w
            b -= step * grad_b
            self.n_iter_ = iteration + 1

        self.weights_, self.bias_ = w, b
        self.fitted_ = True
        return self

    def transform(self, scores: np.ndarray) -> np.ndarray:
        """Apply the fitted scale and bias and return calibrated probabilities.

        Parameters
        ----------
        scores : np.ndarray of shape (n_samples, n_classes)
            Uncalibrated logits or probabilities.

        Returns
        -------
        proba : np.ndarray of shape (n_samples, n_classes)
            Calibrated probabilities, rows summing to one.
        """
        self._check_is_fitted()
        logits = TemperatureScaler._as_logits(scores)
        return np.exp(_log_softmax(logits * self.weights_ + self.bias_))

    def predict_proba(self, scores: np.ndarray) -> np.ndarray:
        """Return calibrated probabilities; alias of :meth:`transform`.

        Parameters
        ----------
        scores : np.ndarray of shape (n_samples, n_classes)
            Uncalibrated logits or probabilities.

        Returns
        -------
        proba : np.ndarray of shape (n_samples, n_classes)
            Calibrated probabilities.
        """
        return self.transform(scores)

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Any]:
        """Return JSON Schema for constructor parameters."""
        return {
            "type": "object",
            "properties": {
                "max_iter": {
                    "type": "integer",
                    "default": 500,
                    "minimum": 1,
                    "description": "Gradient-descent iterations.",
                },
                "learning_rate": {
                    "type": "number",
                    "default": 0.05,
                    "description": "Initial step size.",
                },
                "tol": {
                    "type": "number",
                    "default": 1e-7,
                    "description": "Minimum loss improvement before stopping.",
                },
            },
        }

    def __repr__(self) -> str:
        """Return a readable representation of the scaler."""
        return f"VectorScaler(max_iter={self.max_iter}, learning_rate={self.learning_rate})"
