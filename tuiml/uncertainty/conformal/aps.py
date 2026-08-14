"""Adaptive prediction sets (APS) and their regularised variant (RAPS)."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from tuiml.uncertainty.conformal.split import SplitConformalClassifier


class APSConformalClassifier(SplitConformalClassifier):
    """**Adaptive prediction sets** that grow where the model is uncertain.

    The LAC score behind :class:`~tuiml.uncertainty.SplitConformalClassifier`
    minimises average set size, but it does so by leaving hard samples
    under-covered and easy ones over-covered. APS instead accumulates
    probability mass down the sorted class ranking, producing **small sets on
    easy inputs and large sets on ambiguous ones** — much better conditional
    coverage at a modest cost in average size.

    Overview
    --------
    1. Sort each sample's class probabilities in decreasing order.
    2. The nonconformity of the true label is the total probability mass down
       to and including it, minus a uniform random fraction of its own mass.
    3. Calibrate the corrected quantile of those scores as usual.
    4. A test set includes classes in rank order until the accumulated mass
       exceeds the threshold.

    Theory
    ------
    For sorted probabilities :math:`\\hat{p}_{(1)} \\geq \\dots \\geq
    \\hat{p}_{(c)}` and the true label at rank :math:`r`, the score is

    .. math::
        s = \\sum_{j=1}^{r} \\hat{p}_{(j)} - u \\cdot \\hat{p}_{(r)},
        \\quad u \\sim \\mathrm{Uniform}(0, 1)

    The randomised term :math:`u` is what makes coverage **exact** rather than
    merely conservative: without it the discrete jumps between ranks force the
    set to over-cover. Set ``randomized=False`` for deterministic, reproducible
    sets at the cost of slight over-coverage.

    Parameters
    ----------
    estimator : Classifier
        A TuiML classifier exposing ``predict_proba``.
    alpha : float, default=0.1
        Miscoverage level.
    randomized : bool, default=True
        Whether to apply the uniform randomisation that makes coverage exact.
    calibration_size : float, default=0.25
        Fraction of the training data held out for calibration.
    random_state : int, optional
        Seed for the split and for the randomisation term.

    Attributes
    ----------
    classes_ : np.ndarray of shape (n_classes,)
        Class labels seen during :meth:`fit`.
    scores_ : np.ndarray of shape (n_calibration,)
        Cumulative-mass nonconformity scores.
    quantile_ : float
        The conformal threshold on accumulated probability mass.
    fitted_ : bool
        Whether :meth:`fit` has been called.

    Notes
    -----
    **Complexity.** :math:`O(n c \\log c)` for the per-sample sort, on top of
    one estimator fit.

    **When to use.** Prefer APS over LAC whenever coverage must hold across
    *subgroups*, not just on average — the LAC set is smaller overall but
    systematically fails the hard tail. Its known weakness is a long tail of
    very large sets when the probability estimates are noisy; :class:`RAPS`
    fixes exactly that.

    References
    ----------
    .. [Romano2020] Romano, Y., Sesia, M., & Candès, E. J. (2020).
       Classification with Valid and Adaptive Coverage. *NeurIPS*, 3581-3591.
       :arxiv:`2006.02544`

    See Also
    --------
    :class:`~tuiml.uncertainty.RAPSConformalClassifier` : Penalises the long tail of large sets.
    :class:`~tuiml.uncertainty.SplitConformalClassifier` : Smaller sets, worse conditional coverage.
    :class:`~tuiml.uncertainty.MondrianConformalClassifier` : Exact per-class coverage.

    Examples
    --------
    >>> import numpy as np
    >>> from tuiml.uncertainty import APSConformalClassifier
    >>> from tuiml.algorithms.trees import DecisionTreeClassifier
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(400, 4))
    >>> y = (X[:, 0] + X[:, 1] > 0).astype(int)
    >>> cp = APSConformalClassifier(DecisionTreeClassifier(max_depth=4),
    ...                             alpha=0.1, random_state=0)
    >>> cp.fit(X, y)
    APSConformalClassifier(estimator=DecisionTreeClassifier(), alpha=0.1)
    >>> cp.predict_set(X[:5]).shape
    (5, 2)
    """

    def __init__(
        self,
        estimator: Any,
        alpha: float = 0.1,
        randomized: bool = True,
        calibration_size: float = 0.25,
        random_state: Optional[int] = None,
    ) -> None:
        """Initialise the APS classifier.

        Parameters
        ----------
        estimator : Classifier
            A TuiML classifier exposing ``predict_proba``.
        alpha : float, default=0.1
            Miscoverage level.
        randomized : bool, default=True
            Whether to apply the uniform randomisation term.
        calibration_size : float, default=0.25
            Fraction of the training data held out for calibration.
        random_state : int, optional
            Seed for the split and the randomisation.
        """
        super().__init__(
            estimator,
            alpha=alpha,
            score="lac",
            calibration_size=calibration_size,
            random_state=random_state,
        )
        self.randomized = randomized
        self._rng = np.random.default_rng(random_state)

    def _nonconformity(self, proba: np.ndarray, true_index: np.ndarray) -> np.ndarray:
        """Score each calibration sample by the mass accumulated to its label.

        Parameters
        ----------
        proba : np.ndarray of shape (n_samples, n_classes)
            Predicted probabilities on the calibration set.
        true_index : np.ndarray of shape (n_samples,)
            Column index of the true class for each sample.

        Returns
        -------
        scores : np.ndarray of shape (n_samples,)
            Cumulative probability mass down to the true label.
        """
        order = np.argsort(-proba, axis=1)
        sorted_proba = np.take_along_axis(proba, order, axis=1)
        cumulative = np.cumsum(sorted_proba, axis=1)

        # Rank of the true label within each sample's sorted ranking.
        rank = np.argmax(order == true_index[:, None], axis=1)
        rows = np.arange(proba.shape[0])
        scores = cumulative[rows, rank]

        if self.randomized:
            mass = sorted_proba[rows, rank]
            scores = scores - self._rng.uniform(size=rows.size) * mass
        return scores

    def predict_set(self, X: np.ndarray) -> np.ndarray:
        """Predict adaptive prediction sets.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Test features.

        Returns
        -------
        include : np.ndarray of shape (n_samples, n_classes) of bool
            Class-membership mask; sets grow on ambiguous samples.
        """
        self._check_is_fitted()
        proba = np.asarray(self.estimator.predict_proba(X), dtype=np.float64)
        order = np.argsort(-proba, axis=1)
        sorted_proba = np.take_along_axis(proba, order, axis=1)
        cumulative = np.cumsum(sorted_proba, axis=1)

        penalty = self._penalty(sorted_proba)
        # Include classes until the accumulated mass crosses the threshold.
        # The class that crosses it is kept, so the set is never empty.
        include_sorted = (cumulative - sorted_proba + penalty) <= self.quantile_
        include_sorted[:, 0] = True

        include = np.zeros_like(include_sorted)
        np.put_along_axis(include, order, include_sorted, axis=1)
        return include

    def _penalty(self, sorted_proba: np.ndarray) -> np.ndarray:
        """Return the per-rank penalty added to the cumulative mass.

        Parameters
        ----------
        sorted_proba : np.ndarray of shape (n_samples, n_classes)
            Probabilities sorted in decreasing order.

        Returns
        -------
        penalty : np.ndarray
            Zero for plain APS; :class:`RAPSConformalClassifier` overrides it.
        """
        return np.zeros_like(sorted_proba)

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Any]:
        """Return JSON Schema for constructor parameters."""
        schema = SplitConformalClassifier.get_parameter_schema()
        schema["properties"].pop("score", None)
        schema["properties"]["randomized"] = {
            "type": "boolean",
            "default": True,
            "description": "Apply the randomisation that makes coverage exact.",
        }
        return schema


class RAPSConformalClassifier(APSConformalClassifier):
    """**Regularised adaptive prediction sets** — APS without the long tail.

    APS produces well-adapted sets but occasionally enormous ones, because
    noisy tail probabilities let the cumulative mass creep past the threshold
    over many low-ranked classes. RAPS adds a **penalty that grows with rank**,
    so admitting a badly-ranked class becomes progressively more expensive.
    The result keeps APS's conditional coverage while sharply bounding the
    worst-case set size.

    Overview
    --------
    1. Compute the APS cumulative-mass score.
    2. Add ``lambda_penalty`` for every class ranked beyond ``k_reg``.
    3. Calibrate and predict exactly as APS does, with the penalty applied on
       both sides so the guarantee is preserved.

    Theory
    ------
    With the true label at rank :math:`r`, the RAPS score is

    .. math::
        s = \\sum_{j=1}^{r} \\hat{p}_{(j)} - u \\cdot \\hat{p}_{(r)}
            + \\lambda \\cdot \\max(0,\\ r - k_{\\text{reg}})

    Because the penalty is a deterministic function of rank and is applied
    identically at calibration and prediction time, the exchangeability
    argument is untouched — the :math:`1 - \\alpha` guarantee still holds. The
    penalty only reshapes *which* sets achieve it.

    Parameters
    ----------
    estimator : Classifier
        A TuiML classifier exposing ``predict_proba``.
    alpha : float, default=0.1
        Miscoverage level.
    lambda_penalty : float, default=0.01
        Penalty added per rank beyond ``k_reg``. Larger values shrink the tail
        harder; too large and every set collapses to ``k_reg`` classes.
    k_reg : int, default=1
        Rank beyond which the penalty applies. A good default is the typical
        number of plausible classes.
    randomized : bool, default=True
        Whether to apply the uniform randomisation term.
    calibration_size : float, default=0.25
        Fraction of the training data held out for calibration.
    random_state : int, optional
        Seed for the split and the randomisation.

    Attributes
    ----------
    classes_ : np.ndarray of shape (n_classes,)
        Class labels seen during :meth:`fit`.
    scores_ : np.ndarray of shape (n_calibration,)
        Penalised cumulative-mass scores.
    quantile_ : float
        The conformal threshold.
    fitted_ : bool
        Whether :meth:`fit` has been called.

    Notes
    -----
    **Complexity.** Identical to APS: :math:`O(n c \\log c)`.

    **When to use.** Use RAPS on problems with many classes, where APS's tail
    of huge sets makes the output unusable. On binary or few-class problems the
    penalty has little to bite on and plain APS is simpler. Tune
    ``lambda_penalty`` and ``k_reg`` on a validation split against
    :func:`~tuiml.uncertainty.average_set_size` at fixed coverage.

    References
    ----------
    .. [Angelopoulos2021] Angelopoulos, A. N., Bates, S., Malik, J., & Jordan,
       M. I. (2021). Uncertainty Sets for Image Classifiers using Conformal
       Prediction. *ICLR*. :arxiv:`2009.14193`

    See Also
    --------
    :class:`~tuiml.uncertainty.APSConformalClassifier` : The unregularised version.
    :func:`~tuiml.uncertainty.average_set_size` : The quantity this class minimises.

    Examples
    --------
    >>> import numpy as np
    >>> from tuiml.uncertainty import RAPSConformalClassifier
    >>> from tuiml.algorithms.trees import DecisionTreeClassifier
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(600, 4))
    >>> y = rng.integers(0, 4, 600)
    >>> cp = RAPSConformalClassifier(DecisionTreeClassifier(max_depth=3),
    ...                              alpha=0.2, lambda_penalty=0.05, random_state=0)
    >>> cp.fit(X, y)
    RAPSConformalClassifier(estimator=DecisionTreeClassifier(), alpha=0.2)
    >>> cp.predict_set(X[:5]).shape
    (5, 4)
    """

    def __init__(
        self,
        estimator: Any,
        alpha: float = 0.1,
        lambda_penalty: float = 0.01,
        k_reg: int = 1,
        randomized: bool = True,
        calibration_size: float = 0.25,
        random_state: Optional[int] = None,
    ) -> None:
        """Initialise the RAPS classifier.

        Parameters
        ----------
        estimator : Classifier
            A TuiML classifier exposing ``predict_proba``.
        alpha : float, default=0.1
            Miscoverage level.
        lambda_penalty : float, default=0.01
            Penalty per rank beyond ``k_reg``.
        k_reg : int, default=1
            Rank beyond which the penalty applies.
        randomized : bool, default=True
            Whether to apply the uniform randomisation term.
        calibration_size : float, default=0.25
            Fraction of the training data held out for calibration.
        random_state : int, optional
            Seed for the split and the randomisation.
        """
        super().__init__(
            estimator,
            alpha=alpha,
            randomized=randomized,
            calibration_size=calibration_size,
            random_state=random_state,
        )
        if lambda_penalty < 0.0:
            raise ValueError(
                f"lambda_penalty must be non-negative, got {lambda_penalty}"
            )
        if k_reg < 1:
            raise ValueError(f"k_reg must be at least 1, got {k_reg}")
        self.lambda_penalty = lambda_penalty
        self.k_reg = k_reg

    def _nonconformity(self, proba: np.ndarray, true_index: np.ndarray) -> np.ndarray:
        """Score each calibration sample with the rank-penalised mass.

        Parameters
        ----------
        proba : np.ndarray of shape (n_samples, n_classes)
            Predicted probabilities on the calibration set.
        true_index : np.ndarray of shape (n_samples,)
            Column index of the true class for each sample.

        Returns
        -------
        scores : np.ndarray of shape (n_samples,)
            APS score plus the rank penalty of the true label.
        """
        base = super()._nonconformity(proba, true_index)

        order = np.argsort(-proba, axis=1)
        rank = np.argmax(order == true_index[:, None], axis=1)
        # rank is 0-based; k_reg counts classes, so compare against rank + 1.
        excess = np.maximum(0, (rank + 1) - self.k_reg)
        return base + self.lambda_penalty * excess

    def _penalty(self, sorted_proba: np.ndarray) -> np.ndarray:
        """Return the rank penalty applied to each candidate class.

        Parameters
        ----------
        sorted_proba : np.ndarray of shape (n_samples, n_classes)
            Probabilities sorted in decreasing order.

        Returns
        -------
        penalty : np.ndarray of shape (n_samples, n_classes)
            ``lambda_penalty`` times the rank excess over ``k_reg``.
        """
        ranks = np.arange(1, sorted_proba.shape[1] + 1)
        excess = np.maximum(0, ranks - self.k_reg)
        return np.broadcast_to(
            self.lambda_penalty * excess, sorted_proba.shape
        ).copy()

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Any]:
        """Return JSON Schema for constructor parameters."""
        schema = APSConformalClassifier.get_parameter_schema()
        schema["properties"]["lambda_penalty"] = {
            "type": "number",
            "default": 0.01,
            "minimum": 0,
            "description": "Penalty added per rank beyond k_reg.",
        }
        schema["properties"]["k_reg"] = {
            "type": "integer",
            "default": 1,
            "minimum": 1,
            "description": "Rank beyond which the penalty applies.",
        }
        return schema
