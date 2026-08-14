"""Explainable Boosting Machine (EBM / GA2M): additive glassbox models.

An Explainable Boosting Machine learns an **additive model** of per-feature
*shape functions*. Each feature is binned, and a boosting procedure learns a
score for every bin, so the whole model is a lookup table that a human can
read directly:

.. math::
    \\hat{y} = g\\Big(\\beta_0 + f_1(x_1) + f_2(x_2) + \\cdots + f_m(x_m)\\Big)

where :math:`f_j` is a step function over the quantile bins of feature
:math:`j` and :math:`g` is the identity (regression), the sigmoid (binary
classification), or the softmax (multiclass). Because the model is additive,
its predictions can be decomposed exactly into one contribution per feature
via :meth:`explain`.

This module implements the **GA1M** (generalised additive model) form.
Pairwise-interaction terms (the "GA2M" extension) are not yet included.
"""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Any, Optional, Tuple

from tuiml.base.algorithms import Classifier, classifier, Regressor, regressor


# ---------------------------------------------------------------------------
# Binning helpers
# ---------------------------------------------------------------------------

def _as_2d(X) -> np.ndarray:
    """Return ``X`` as a 2-D float array, treating 1-D input as one column.

    Parameters
    ----------
    X : array-like
        Feature data, either 2-D or a single 1-D feature.

    Returns
    -------
    X2d : np.ndarray of shape (n_samples, n_features)
        Two-dimensional copy of the input.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        return X.reshape(-1, 1)
    return np.array(X, dtype=float)


def _quantile_edges(col: np.ndarray, n_bins: int) -> np.ndarray:
    """Compute quantile bin edges for a single feature column.

    Parameters
    ----------
    col : np.ndarray of shape (n_samples,)
        One feature column.
    n_bins : int
        Requested number of bins.

    Returns
    -------
    edges : np.ndarray
        Sorted bin edges. The number of bins is ``edges.size - 1`` and may be
        smaller than ``n_bins`` when quantiles tie (low-cardinality or
        constant features collapse to a single bin).
    """
    col = np.asarray(col, dtype=float).ravel()
    quantiles = np.quantile(col, np.linspace(0.0, 1.0, n_bins + 1))
    edges = np.unique(quantiles)
    if edges.size < 2:
        # Constant feature: one synthetic bin covering the observed value.
        center = float(col.min())
        edges = np.array([center - 0.5, center + 0.5], dtype=float)
    return edges


def _digitize(col: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Map feature values to bin indices in ``[0, n_bins - 1]``.

    Parameters
    ----------
    col : np.ndarray of shape (n_samples,)
        One feature column.
    edges : np.ndarray
        Sorted bin edges from :func:`_quantile_edges`.

    Returns
    -------
    indices : np.ndarray of shape (n_samples,)
        Integer bin index per sample. Values outside the training range are
        clipped to the first/last bin.
    """
    col = np.asarray(col, dtype=float).ravel()
    n_bins = edges.size - 1
    indices = np.digitize(col, edges) - 1
    return np.clip(indices, 0, n_bins - 1)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable logistic function."""
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500.0, 500.0)))


def _softmax(Z: np.ndarray) -> np.ndarray:
    """Row-wise softmax."""
    Z = Z - np.max(Z, axis=1, keepdims=True)
    exp = np.exp(Z)
    return exp / np.sum(exp, axis=1, keepdims=True)


class _BaseEBM:
    """Shared additive fitting for the EBM classifier and regressor.

    This is an internal mixin: it is not registered and not meant to be
    instantiated directly. It bins the features, runs the per-feature
    boosting updates, and exposes the learned shape functions.
    """

    def _bin_data(self, X: np.ndarray) -> None:
        """Quantile-bin every feature and store edges plus bin indices.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training features.
        """
        X = _as_2d(X)
        self.n_features_ = X.shape[1]
        self.bin_edges_: List[np.ndarray] = []
        self.n_bins_per_feature_: List[int] = []
        bin_cols: List[np.ndarray] = []
        for j in range(self.n_features_):
            edges = _quantile_edges(X[:, j], self.n_bins)
            self.bin_edges_.append(edges)
            self.n_bins_per_feature_.append(edges.size - 1)
            bin_cols.append(_digitize(X[:, j], edges))
        self._bin_indices_ = np.column_stack(bin_cols).astype(np.intp)

    def _initialize_scores(self, n_outputs: int) -> None:
        """Allocate intercept and per-feature shape functions.

        Parameters
        ----------
        n_outputs : int
            1 for regression/binary, ``n_classes`` for multiclass.
        """
        self.n_outputs_ = n_outputs
        self.shape_functions_ = [
            np.zeros((self.n_bins_per_feature_[j], n_outputs), dtype=float)
            for j in range(self.n_features_)
        ]
        self.intercept_ = np.zeros(n_outputs, dtype=float)

    def _bin_updates(self, residual: np.ndarray, j: int) -> np.ndarray:
        """Return the per-bin mean of ``residual`` scaled by learning rate.

        Parameters
        ----------
        residual : np.ndarray of shape (n_samples,)
            Negative gradient (regression: ``y - pred``; classification:
            ``target - p``).
        j : int
            Feature index whose bins the residual is averaged over.

        Returns
        -------
        update : np.ndarray of shape (n_bins,)
            Learning-rate-scaled per-bin mean residual.
        """
        bins = self._bin_indices_[:, j]
        n_bins = self.n_bins_per_feature_[j]
        sums = np.bincount(bins, weights=residual, minlength=n_bins)
        counts = np.bincount(bins, minlength=n_bins)
        return self.learning_rate * sums / np.maximum(counts, 1)

    def _fit_regression(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit additive shape functions for a continuous target."""
        y = np.asarray(y, dtype=float).ravel()
        self._initialize_scores(1)
        self.intercept_[0] = float(np.mean(y))
        pred = np.full(X.shape[0], self.intercept_[0])
        for _ in range(self.max_rounds):
            for j in range(self.n_features_):
                update = self._bin_updates(y - pred, j)
                self.shape_functions_[j][:, 0] += update
                pred += update[self._bin_indices_[:, j]]
        self._finalize()

    def _fit_binary(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit additive shape functions for a binary target (log loss)."""
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        if self.classes_.size != 2:
            raise ValueError(
                "ExplainableBoostingClassifier requires exactly two classes "
                "for binary fit; use more than two classes for multiclass."
            )
        y01 = np.where(y == self.classes_[1], 1.0, 0.0)
        self._initialize_scores(1)
        p0 = float(np.clip(y01.mean(), 1e-6, 1 - 1e-6))
        self.intercept_[0] = np.log(p0 / (1.0 - p0))
        pred = np.full(X.shape[0], self.intercept_[0])
        for _ in range(self.max_rounds):
            for j in range(self.n_features_):
                p = _sigmoid(pred)
                update = self._bin_updates(y01 - p, j)
                self.shape_functions_[j][:, 0] += update
                pred += update[self._bin_indices_[:, j]]
        self._finalize()

    def _fit_multiclass(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit additive shape functions for a multiclass target (softmax)."""
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        n_classes = self.classes_.size
        Y = np.zeros((X.shape[0], n_classes), dtype=float)
        for k, c in enumerate(self.classes_):
            Y[:, k] = (y == c).astype(float)
        self._initialize_scores(n_classes)
        pred = np.zeros((X.shape[0], n_classes), dtype=float)
        for _ in range(self.max_rounds):
            for j in range(self.n_features_):
                P = _softmax(pred)
                for k in range(n_classes):
                    update = self._bin_updates(Y[:, k] - P[:, k], j)
                    self.shape_functions_[j][:, k] += update
                    pred[:, k] += update[self._bin_indices_[:, j]]
        self._finalize()

    def _finalize(self) -> None:
        """Center shape functions, fold offsets into the intercept, score importances."""
        for j in range(self.n_features_):
            mean = self.shape_functions_[j].mean(axis=0)
            self.shape_functions_[j] = self.shape_functions_[j] - mean
            self.intercept_ = self.intercept_ + mean
        importance = np.zeros(self.n_features_, dtype=float)
        for j in range(self.n_features_):
            importance[j] = float(np.mean(np.abs(self.shape_functions_[j])))
        self.feature_importance_ = importance
        self._is_fitted = True

    def _score(self, X: np.ndarray) -> np.ndarray:
        """Return the raw additive score (pre-link) for each sample.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        score : np.ndarray of shape (n_samples, n_outputs)
            ``intercept + sum_j shape_j(x_j)``.
        """
        X = _as_2d(X)
        n = X.shape[0]
        out = np.zeros((n, self.n_outputs_), dtype=float) + self.intercept_
        for j in range(self.n_features_):
            idx = _digitize(X[:, j], self.bin_edges_[j])
            out += self.shape_functions_[j][idx]
        return out

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Return the raw additive score before the link function.

        For regression this equals :meth:`predict`; for classification it is
        the log-odds (binary) or logits (multiclass).

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        score : np.ndarray of shape (n_samples,) or (n_samples, n_classes)
            Additive score ``intercept + sum of shape functions``.
        """
        self._check_is_fitted()
        score = self._score(X)
        if self.n_outputs_ == 1:
            return score[:, 0]
        return score

    def explain(self, X: np.ndarray) -> np.ndarray:
        """Decompose each prediction into one additive contribution per feature.

        The sum over features plus ``intercept_`` reconstructs
        :meth:`decision_function` exactly::

            decision_function(X) == intercept_ + explain(X).sum(axis=1)

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        contributions : np.ndarray
            Shape ``(n_samples, n_features)`` for single-output models, or
            ``(n_samples, n_features, n_classes)`` for multiclass.
        """
        self._check_is_fitted()
        X = _as_2d(X)
        n = X.shape[0]
        if self.n_outputs_ == 1:
            contribs = np.zeros((n, self.n_features_), dtype=float)
            for j in range(self.n_features_):
                idx = _digitize(X[:, j], self.bin_edges_[j])
                contribs[:, j] = self.shape_functions_[j][idx, 0]
            return contribs
        contribs = np.zeros((n, self.n_features_, self.n_outputs_), dtype=float)
        for j in range(self.n_features_):
            idx = _digitize(X[:, j], self.bin_edges_[j])
            contribs[:, j, :] = self.shape_functions_[j][idx]
        return contribs

    def get_shape_functions(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Return every feature's bin edges and learned bin scores.

        Returns
        -------
        shapes : list of tuple (edges, scores)
            For each feature, ``edges`` is the bin boundaries and ``scores``
            the learned score per bin (shape ``(n_bins, n_outputs)``).
        """
        self._check_is_fitted()
        return list(zip(self.bin_edges_, self.shape_functions_))

    def get_shape_function(self, feature: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return the bin edges and learned scores for a single feature.

        Parameters
        ----------
        feature : int
            Feature index.

        Returns
        -------
        edges : np.ndarray
            Bin boundaries.
        scores : np.ndarray
            Learned score per bin.
        """
        self._check_is_fitted()
        return self.bin_edges_[feature], self.shape_functions_[feature]


@regressor(tags=["glassbox", "interpretable", "additive", "boosting"], version="1.0.0")
class ExplainableBoostingRegressor(Regressor, _BaseEBM):
    """Explainable Boosting Machine for regression (additive shape functions).

    An **interpretable** additive model that learns one **shape function**
    per feature by boosting per-bin mean residuals. Each feature is quantile
    binned and each bin is assigned a score, so the fitted model is a set of
    lookup tables whose sum -- plus an intercept -- is the prediction.

    Overview
    --------
    1. Quantile-bin each feature into (up to) ``n_bins`` bins
    2. Initialize the intercept to the mean target and every bin score to zero
    3. For each boosting round, cycle over features and add the learning-rate
       scaled mean residual of each bin to that bin's score
    4. Center each shape function and fold the offsets into the intercept
    5. Predict as ``intercept_ + sum_j shape_j(x_j)``

    Theory
    ------
    The model is the additive expansion

    .. math::
        \\hat{y}(x) = \\beta_0 + \\sum_{j=1}^{m} f_j(x_j)

    where :math:`f_j` is constant over each quantile bin of feature :math:`j`.
    Training minimises squared error by gradient boosting: at each step the
    negative gradient :math:`y - \\hat{y}` is averaged per bin and added to
    the bin score, exactly the optimal leaf value for a squared-error stump.

    Parameters
    ----------
    n_bins : int, default=32
        Number of quantile bins per feature (fewer when quantiles tie).
    max_rounds : int, default=100
        Number of boosting rounds. Each round updates every feature once.
    learning_rate : float, default=0.01
        Shrinkage applied to each per-bin update.
    feature_names : list of str, optional
        Names used when reporting shape functions. Defaults to
        ``feature_0, feature_1, ...``.

    Attributes
    ----------
    intercept_ : np.ndarray of shape (1,)
        Additive intercept (the mean target plus the centered bin offsets).
    shape_functions_ : list of np.ndarray
        Per-feature bin scores, each of shape ``(n_bins, 1)``.
    bin_edges_ : list of np.ndarray
        Per-feature quantile bin boundaries.
    n_bins_per_feature_ : list of int
        Actual number of bins per feature after deduplicating tied quantiles.
    feature_importance_ : np.ndarray of shape (n_features,)
        Mean absolute bin score per feature (interpretable magnitude).
    n_features_ : int
        Number of features seen during fit.

    Notes
    -----
    **Complexity:**

    - Training: :math:`O(R \\cdot m \\cdot n)` where :math:`R` = max_rounds,
      :math:`m` = features, :math:`n` = samples.
    - Prediction: :math:`O(m)` per sample (one bin lookup per feature).

    **When to use ExplainableBoostingRegressor:**

    - When you need a model a human can audit feature-by-feature
    - When the signal is roughly additive (no strong interactions)
    - When you want exact, per-feature prediction decompositions via
      :meth:`explain`

    References
    ----------
    .. [Nori2019] Nori, H., Jenkins, S., Koch, P., and Caruana, R. (2019).
           **InterpretML: A Unified Framework for Machine Learning Interpretability.**
           *arXiv preprint* arXiv:1909.09223.

    .. [Lou2012] Lou, Y., Caruana, R., Gehrke, J., and Hooker, G. (2012).
           **Accurate Intelligible Models with Pairwise Interactions.**
           *KDD 2012*, pp. 623-631.
           DOI: `10.1145/2339530.2339657 <https://doi.org/10.1145/2339530.2339657>`_

    See Also
    --------
    :class:`~tuiml.algorithms.glassbox.ExplainableBoostingClassifier` : Classification counterpart.
    :class:`~tuiml.algorithms.linear.LinearRegression` : Non-additive linear baseline.

    Examples
    --------
    >>> from tuiml.algorithms.glassbox import ExplainableBoostingRegressor
    >>> import numpy as np
    >>> X = np.array([[0.], [1.], [2.], [3.], [4.], [5.], [6.], [7.]])
    >>> y = 2.0 * X.ravel() + 1.0
    >>> reg = ExplainableBoostingRegressor(n_bins=8, max_rounds=200, learning_rate=0.1)
    >>> _ = reg.fit(X, y)
    >>> np.allclose(reg.predict(np.array([[2.0], [6.0]])), [5.0, 13.0], atol=1e-3)
    True
    >>> np.allclose(reg.predict(X), reg.intercept_[0] + reg.explain(X).sum(axis=1), atol=1e-12)
    True
    """

    def __init__(self, n_bins: int = 32, max_rounds: int = 100,
                 learning_rate: float = 0.01,
                 feature_names: Optional[List[str]] = None):
        """Initialize ExplainableBoostingRegressor.

        Parameters
        ----------
        n_bins : int, default=32
            Number of quantile bins per feature.
        max_rounds : int, default=100
            Number of boosting rounds.
        learning_rate : float, default=0.01
            Shrinkage for each per-bin update.
        feature_names : list of str, optional
            Names used when reporting shape functions.
        """
        super().__init__()
        self.n_bins = n_bins
        self.max_rounds = max_rounds
        self.learning_rate = learning_rate
        self.feature_names = feature_names
        self.intercept_ = None
        self.shape_functions_ = None
        self.bin_edges_ = None
        self.n_bins_per_feature_ = None
        self.feature_importance_ = None
        self.n_features_ = None

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        """Return JSON Schema for constructor parameters."""
        return {
            "n_bins": {"type": "integer", "default": 32, "minimum": 2,
                       "description": "Number of quantile bins per feature"},
            "max_rounds": {"type": "integer", "default": 100, "minimum": 1,
                           "description": "Number of boosting rounds"},
            "learning_rate": {"type": "number", "default": 0.01, "minimum": 1e-6,
                              "description": "Shrinkage per bin update"},
            "feature_names": {"type": ["array", "null"], "default": None,
                              "description": "Optional feature names"},
        }

    @classmethod
    def get_capabilities(cls) -> List[str]:
        """Return regressor capabilities."""
        return ["numeric", "regression"]

    @classmethod
    def get_complexity(cls) -> str:
        """Return time/space complexity."""
        return "O(max_rounds * n_features * n_samples) training, O(n_features) prediction"

    @classmethod
    def get_references(cls) -> List[str]:
        """Return academic references."""
        return [
            "Nori, H., Jenkins, S., Koch, P., & Caruana, R. (2019). "
            "InterpretML: A Unified Framework for Machine Learning Interpretability. arXiv:1909.09223.",
            "Lou, Y., Caruana, R., Gehrke, J., & Hooker, G. (2012). Accurate "
            "Intelligible Models with Pairwise Interactions. KDD 2012.",
        ]

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ExplainableBoostingRegressor":
        """Fit the additive model.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training features.
        y : np.ndarray of shape (n_samples,)
            Target values.

        Returns
        -------
        self : ExplainableBoostingRegressor
            Fitted regressor.
        """
        X = _as_2d(X)
        self._bin_data(X)
        self._fit_regression(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict target values.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        y_pred : np.ndarray of shape (n_samples,)
            Predicted values.
        """
        self._check_is_fitted()
        return self.decision_function(X)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return the R-squared score on the given data.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Test samples.
        y : np.ndarray of shape (n_samples,)
            True target values.

        Returns
        -------
        r2 : float
            R-squared score.
        """
        y = np.asarray(y, dtype=float).ravel()
        y_pred = self.predict(X)
        ss_res = float(np.sum((y - y_pred) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        if ss_tot == 0.0:
            return 0.0
        return 1.0 - ss_res / ss_tot

    def __repr__(self) -> str:
        if self._is_fitted:
            return (f"ExplainableBoostingRegressor(n_bins={self.n_bins}, "
                    f"max_rounds={self.max_rounds}, n_features={self.n_features_})")
        return f"ExplainableBoostingRegressor(n_bins={self.n_bins}, max_rounds={self.max_rounds})"


@classifier(tags=["glassbox", "interpretable", "additive", "boosting"], version="1.0.0")
class ExplainableBoostingClassifier(Classifier, _BaseEBM):
    """Explainable Boosting Machine for classification (additive shape functions).

    An **interpretable** additive classifier that learns one **shape
    function** per feature by boosting per-bin mean residuals. It supports
    binary (sigmoid link) and multiclass (softmax link) targets, and every
    prediction decomposes exactly into per-feature contributions.

    Overview
    --------
    1. Quantile-bin each feature into (up to) ``n_bins`` bins
    2. Initialize an intercept and zero bin scores
    3. For each boosting round, cycle over features and add the learning-rate
       scaled mean residual (log-loss gradient) of each bin to its score
    4. Center each shape function and fold offsets into the intercept
    5. Map the additive score through the sigmoid (binary) or softmax
       (multiclass) link to obtain class probabilities

    Theory
    ------
    The additive score is

    .. math::
        s(x) = \\beta_0 + \\sum_{j=1}^{m} f_j(x_j)

    For binary classification the probability is :math:`p = \\sigma(s)` with
    the logistic function, and the negative gradient used for boosting is
    :math:`y - p`. For multiclass, a score vector per class is used and the
    negative gradient is the one-hot target minus the softmax output.

    Parameters
    ----------
    n_bins : int, default=32
        Number of quantile bins per feature (fewer when quantiles tie).
    max_rounds : int, default=100
        Number of boosting rounds. Each round updates every feature once.
    learning_rate : float, default=0.01
        Shrinkage applied to each per-bin update.
    feature_names : list of str, optional
        Names used when reporting shape functions.

    Attributes
    ----------
    intercept_ : np.ndarray
        Additive intercept (log-odds / logits).
    shape_functions_ : list of np.ndarray
        Per-feature bin scores of shape ``(n_bins, 1)`` (binary) or
        ``(n_bins, n_classes)`` (multiclass).
    bin_edges_ : list of np.ndarray
        Per-feature quantile bin boundaries.
    classes_ : np.ndarray
        Unique class labels.
    feature_importance_ : np.ndarray of shape (n_features,)
        Mean absolute bin score per feature.
    n_features_ : int
        Number of features seen during fit.

    Notes
    -----
    **Complexity:**

    - Training: :math:`O(R \\cdot m \\cdot n \\cdot K)` where :math:`R` =
      max_rounds, :math:`m` = features, :math:`n` = samples, :math:`K` =
      classes (1 for binary).
    - Prediction: :math:`O(m \\cdot K)` per sample.

    **When to use ExplainableBoostingClassifier:**

    - When a human must be able to audit how each feature drives the score
    - When the signal is roughly additive
    - When you want exact per-feature log-odds contributions via
      :meth:`explain`

    References
    ----------
    .. [Nori2019] Nori, H., Jenkins, S., Koch, P., and Caruana, R. (2019).
           **InterpretML: A Unified Framework for Machine Learning Interpretability.**
           *arXiv preprint* arXiv:1909.09223.

    .. [Lou2012] Lou, Y., Caruana, R., Gehrke, J., and Hooker, G. (2012).
           **Accurate Intelligible Models with Pairwise Interactions.**
           *KDD 2012*, pp. 623-631.
           DOI: `10.1145/2339530.2339657 <https://doi.org/10.1145/2339530.2339657>`_

    See Also
    --------
    :class:`~tuiml.algorithms.glassbox.ExplainableBoostingRegressor` : Regression counterpart.
    :class:`~tuiml.algorithms.linear.LogisticRegression` : Non-additive logistic baseline.

    Examples
    --------
    >>> from tuiml.algorithms.glassbox import ExplainableBoostingClassifier
    >>> import numpy as np
    >>> X = np.array([[0.], [1.], [2.], [3.], [4.], [5.], [6.], [7.]])
    >>> y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    >>> clf = ExplainableBoostingClassifier(n_bins=8, max_rounds=200, learning_rate=0.5)
    >>> _ = clf.fit(X, y)
    >>> clf.predict(np.array([[2.0], [6.0]])).tolist()
    [0, 1]
    >>> bool(clf.predict_proba(np.array([[6.0]]))[0, 1] > 0.5)
    True
    """

    def __init__(self, n_bins: int = 32, max_rounds: int = 100,
                 learning_rate: float = 0.01,
                 feature_names: Optional[List[str]] = None):
        """Initialize ExplainableBoostingClassifier.

        Parameters
        ----------
        n_bins : int, default=32
            Number of quantile bins per feature.
        max_rounds : int, default=100
            Number of boosting rounds.
        learning_rate : float, default=0.01
            Shrinkage for each per-bin update.
        feature_names : list of str, optional
            Names used when reporting shape functions.
        """
        super().__init__()
        self.n_bins = n_bins
        self.max_rounds = max_rounds
        self.learning_rate = learning_rate
        self.feature_names = feature_names
        self.intercept_ = None
        self.shape_functions_ = None
        self.bin_edges_ = None
        self.n_bins_per_feature_ = None
        self.classes_ = None
        self.feature_importance_ = None
        self.n_features_ = None

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        """Return JSON Schema for constructor parameters."""
        return {
            "n_bins": {"type": "integer", "default": 32, "minimum": 2,
                       "description": "Number of quantile bins per feature"},
            "max_rounds": {"type": "integer", "default": 100, "minimum": 1,
                           "description": "Number of boosting rounds"},
            "learning_rate": {"type": "number", "default": 0.01, "minimum": 1e-6,
                              "description": "Shrinkage per bin update"},
            "feature_names": {"type": ["array", "null"], "default": None,
                              "description": "Optional feature names"},
        }

    @classmethod
    def get_capabilities(cls) -> List[str]:
        """Return classifier capabilities."""
        return ["numeric", "binary_class", "multiclass"]

    @classmethod
    def get_complexity(cls) -> str:
        """Return time/space complexity."""
        return "O(max_rounds * n_features * n_samples * n_classes) training, O(n_features * n_classes) prediction"

    @classmethod
    def get_references(cls) -> List[str]:
        """Return academic references."""
        return [
            "Nori, H., Jenkins, S., Koch, P., & Caruana, R. (2019). "
            "InterpretML: A Unified Framework for Machine Learning Interpretability. arXiv:1909.09223.",
            "Lou, Y., Caruana, R., Gehrke, J., & Hooker, G. (2012). Accurate "
            "Intelligible Models with Pairwise Interactions. KDD 2012.",
        ]

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ExplainableBoostingClassifier":
        """Fit the additive classifier.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training features.
        y : np.ndarray of shape (n_samples,)
            Target class labels.

        Returns
        -------
        self : ExplainableBoostingClassifier
            Fitted classifier.
        """
        X = _as_2d(X)
        y = np.asarray(y)
        self._bin_data(X)
        n_classes = np.unique(y).size
        if n_classes == 2:
            self._fit_binary(X, y)
        elif n_classes > 2:
            self._fit_multiclass(X, y)
        else:
            raise ValueError("ExplainableBoostingClassifier requires at least two classes.")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        y_pred : np.ndarray of shape (n_samples,)
            Predicted class labels.
        """
        self._check_is_fitted()
        score = self._score(X)
        if self.n_outputs_ == 1:
            indices = (score[:, 0] >= 0).astype(int)
        else:
            indices = np.argmax(score, axis=1)
        return self.classes_[indices]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        proba : np.ndarray of shape (n_samples, n_classes)
            Class probabilities (sigmoid for binary, softmax for multiclass).
        """
        self._check_is_fitted()
        score = self._score(X)
        if self.n_outputs_ == 1:
            p = _sigmoid(score[:, 0])
            return np.column_stack([1.0 - p, p])
        return _softmax(score)

    def __repr__(self) -> str:
        if self._is_fitted:
            return (f"ExplainableBoostingClassifier(n_bins={self.n_bins}, "
                    f"max_rounds={self.max_rounds}, n_features={self.n_features_})")
        return f"ExplainableBoostingClassifier(n_bins={self.n_bins}, max_rounds={self.max_rounds})"
