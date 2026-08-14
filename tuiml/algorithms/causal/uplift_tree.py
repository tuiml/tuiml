"""A decision tree that splits directly on uplift gain.

Ordinary regression trees split to reduce outcome variance; an uplift tree
instead splits to *separate* high-treatment-effect regions from
low-treatment-effect regions, so the leaves are groups of individuals for whom
the treatment works differently.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from tuiml.base.algorithms import UpliftModel, uplift
from tuiml.algorithms.causal.meta_learners import _check_arrays


@uplift(tags=["causal", "tree", "uplift"], version="1.0.0")
class UpliftTreeClassifier(UpliftModel):
    """Single uplift tree that splits on the difference in treatment effects.

    Summary
    -------
    A greedy binary tree whose split criterion is the **uplift gain** — the
    squared difference in treatment effect between the two child nodes,
    weighted by their size. Each leaf stores the observed treatment effect
    :math:`\\bar{y}_{t=1} - \\bar{y}_{t=0}` of the samples that land there.

    Overview
    --------
    1. For each candidate feature and threshold, split the node and estimate
       the uplift of each child as its treated-mean minus control-mean.
    2. Choose the split that maximizes
       :math:`\\frac{n_L n_R}{n}\\,(\\hat{\\tau}_L - \\hat{\\tau}_R)^2`.
    3. Recurse until a stopping rule fires, then store the leaf uplift.

    Theory
    ------
    Let a node contain :math:`n` samples, :math:`n_t` treated and
    :math:`n_c` control, with outcomes :math:`y`. Its estimated uplift is

    .. math::
        \\hat{\\tau} = \\frac{1}{n_t}\\sum_{i: t_i=1} y_i
        - \\frac{1}{n_c}\\sum_{i: t_i=0} y_i.

    A split sends the node's samples to a left child :math:`L` and a right
    child :math:`R`. The split is chosen to maximize the between-child uplift
    variance,

    .. math::
        \\text{gain} = \\frac{n_L n_R}{n}
        \\left(\\hat{\\tau}_L - \\hat{\\tau}_R\\right)^2,

    which is large exactly when the two children have very different treatment
    effects. This targets heterogeneity directly rather than the outcome level.

    Parameters
    ----------
    max_depth : int or None, default=None
        Maximum tree depth (``None`` for no limit).
    min_samples_split : int, default=2
        Minimum samples required to split an internal node.
    min_samples_leaf : int, default=20
        Minimum samples required in a child for a split to be accepted. Each
        child must also contain at least one treated and one control sample so
        its uplift is defined.
    max_features : int or None, default=None
        Number of features to consider at each split (``None`` uses all).
    random_state : int or None, default=None
        Random seed for feature sub-sampling.

    Attributes
    ----------
    tree_ : dict
        The root node of the fitted tree. Internal nodes have ``feature``,
        ``threshold``, ``left`` and ``right`` keys; leaves have ``uplift``,
        ``n_treated`` and ``n_control``.
    n_features_in_ : int
        Number of features in ``X``.
    n_nodes_ : int
        Total number of nodes in the fitted tree.
    max_depth_ : int
        Depth of the fitted tree.

    Notes
    -----
    **Complexity:** each split sorts the node by each candidate feature, so
    training is roughly :math:`O(d \\, n \\log n)` per level and prediction is
    :math:`O(\\text{depth})` per sample.

    **When to use:** when you want a single, inspectable tree (rather than a
    black-box meta-learner) and the treatment effect is piecewise-constant in
    the features.

    References
    ----------
    .. [Rzepakowski2012] Rzepakowski, P. and Jaroszewicz, S. (2012).
       **Decision trees for uplift modeling with single and multiple
       treatments.** *Knowledge and Information Systems*, 32(2), 303-327.
       DOI: `10.1007/s10115-011-0434-0 <https://doi.org/10.1007/s10115-011-0434-0>`_

    .. [Athey2016] Athey, S. and Imbens, G. (2016).
       **Recursive partitioning for heterogeneous causal effects.**
       *Proceedings of the National Academy of Sciences*, 113(27), 7353-7360.
       DOI: `10.1073/pnas.1510489113 <https://doi.org/10.1073/pnas.1510489113>`_

    See Also
    --------
    :class:`~tuiml.algorithms.causal.TLearner` : Two group models rather than a
        single uplift-splitting tree.

    Examples
    --------
    >>> from tuiml.algorithms.causal import UpliftTreeClassifier
    >>> import numpy as np
    >>> rng = np.random.RandomState(0)
    >>> X = rng.uniform(-1, 1, size=(500, 2))
    >>> t = rng.randint(0, 2, size=500)
    >>> y = 1.0 + X[:, 1] + t * (2.0 * X[:, 0]) + rng.normal(0, 0.1, size=500)
    >>> model = UpliftTreeClassifier(max_depth=4, min_samples_leaf=20).fit(X, t, y)
    >>> model.predict_uplift(X).shape
    (500,)
    """

    def __init__(
        self,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 20,
        max_features: Optional[int] = None,
        random_state: Optional[int] = None,
    ):
        """Initialize the uplift tree.

        Parameters
        ----------
        max_depth : int or None, default=None
            Maximum tree depth.
        min_samples_split : int, default=2
            Minimum samples to split a node.
        min_samples_leaf : int, default=20
            Minimum samples per child for a valid split.
        max_features : int or None, default=None
            Number of features to consider per split.
        random_state : int or None, default=None
            Random seed.
        """
        super().__init__()
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.random_state = random_state

        # Fitted attributes
        self.tree_ = None
        self.n_features_in_ = None
        self.n_nodes_ = None
        self.max_depth_ = None

        self._rng = np.random.RandomState(random_state)

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        """Return JSON Schema for constructor parameters."""
        return {
            "max_depth": {
                "type": ["integer", "null"],
                "default": None,
                "minimum": 1,
                "description": "Maximum depth of the tree (None = unlimited)",
            },
            "min_samples_split": {
                "type": "integer",
                "default": 2,
                "minimum": 2,
                "description": "Minimum samples required to split a node",
            },
            "min_samples_leaf": {
                "type": "integer",
                "default": 20,
                "minimum": 1,
                "description": "Minimum samples required in a child node",
            },
            "max_features": {
                "type": ["integer", "null"],
                "default": None,
                "minimum": 1,
                "description": "Number of features considered per split (None = all)",
            },
            "random_state": {
                "type": ["integer", "null"],
                "default": None,
                "description": "Random seed for reproducibility",
            },
        }

    @classmethod
    def get_capabilities(cls) -> List[str]:
        """Return supported capabilities."""
        return ["numeric", "uplift", "binary_treatment", "continuous_outcome", "interpretable"]

    @classmethod
    def get_complexity(cls) -> str:
        """Return complexity analysis."""
        return "Training: O(d * n * log(n)) per level, Prediction: O(depth) per sample"

    @classmethod
    def get_references(cls) -> List[str]:
        """Return academic citations."""
        return [
            "Rzepakowski, P. and Jaroszewicz, S. (2012). Decision trees for "
            "uplift modeling with single and multiple treatments. KAIS, 32(2), 303-327.",
            "Athey, S. and Imbens, G. (2016). Recursive partitioning for "
            "heterogeneous causal effects. PNAS, 113(27), 7353-7360.",
        ]

    def fit(self, X, treatment, y) -> "UpliftTreeClassifier":
        """Build the uplift tree.

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
        self : UpliftTreeClassifier
            Fitted estimator.
        """
        X, treatment, y = _check_arrays(X, treatment, y)
        self.n_features_in_ = X.shape[1]

        self.tree_ = self._build(X, treatment, y, depth=0)
        self.n_nodes_ = self._count_nodes(self.tree_)
        self.max_depth_ = self._tree_depth(self.tree_)

        self._is_fitted = True
        return self

    # ------------------------------------------------------------------ #
    # Splitting
    # ------------------------------------------------------------------ #

    def _leaf(self, tn: np.ndarray, yn: np.ndarray) -> Dict[str, Any]:
        """Build a leaf node from the local treatment/outcome arrays.

        Parameters
        ----------
        tn : np.ndarray of shape (n,)
            Binary treatment indicator for the node's samples.
        yn : np.ndarray of shape (n,)
            Outcome for the node's samples.

        Returns
        -------
        node : dict
            A leaf node holding the estimated uplift and group counts.
        """
        n_treated = int(np.sum(tn))
        n_control = int(tn.size - n_treated)
        if n_treated > 0 and n_control > 0:
            uplift = float(np.mean(yn[tn == 1])) - float(np.mean(yn[tn == 0]))
        else:
            uplift = 0.0
        return {
            "type": "leaf",
            "uplift": uplift,
            "n": int(tn.size),
            "n_treated": n_treated,
            "n_control": n_control,
        }

    def _best_split(self, Xn, tn, yn):
        """Find the split maximizing uplift gain for a node.

        Parameters
        ----------
        Xn : np.ndarray of shape (n, d)
            Local feature matrix.
        tn : np.ndarray of shape (n,)
            Binary treatment indicator.
        yn : np.ndarray of shape (n,)
            Outcome.

        Returns
        -------
        feature : int or None
            Feature index of the best split (``None`` if no split improves).
        threshold : float or None
            Split threshold.
        left_mask : np.ndarray of bool or None
            Boolean mask selecting the left child's samples.
        gain : float
            The uplift gain of the chosen split.
        """
        n = Xn.shape[0]
        d = Xn.shape[1]
        min_leaf = self.min_samples_leaf

        total_t = float(np.sum(tn))
        total_c = float(n - total_t)
        total_yt = float(np.sum(yn * tn))
        total_yc = float(np.sum(yn * (1.0 - tn)))

        if self.max_features is not None and self.max_features < d:
            features = self._rng.choice(d, size=self.max_features, replace=False)
        else:
            features = np.arange(d)

        best_feature = None
        best_threshold = None
        best_left = None
        best_gain = -np.inf

        for j in features:
            col = Xn[:, j]
            order = np.argsort(col, kind="mergesort")
            xs = col[order]
            ts = tn[order].astype(float)
            ys = yn[order].astype(float)

            cum_t = np.cumsum(ts)
            cum_yt = np.cumsum(ys * ts)
            cum_yc = np.cumsum(ys * (1.0 - ts))

            # Left sizes k = 1 .. n-1, indexed by pos = k - 1.
            pos = np.arange(n - 1)          # prefix index for left size pos+1
            n_left = pos + 1
            n_right = n - n_left

            t_left = cum_t[pos]
            c_left = n_left - t_left
            t_right = total_t - t_left
            c_right = total_c - c_left

            yt_left = cum_yt[pos]
            yc_left = cum_yc[pos]
            yt_right = total_yt - yt_left
            yc_right = total_yc - yc_left

            with np.errstate(divide="ignore", invalid="ignore"):
                u_left = np.where(
                    (t_left > 0) & (c_left > 0),
                    yt_left / t_left - yc_left / c_left,
                    0.0,
                )
                u_right = np.where(
                    (t_right > 0) & (c_right > 0),
                    yt_right / t_right - yc_right / c_right,
                    0.0,
                )

            gain = (n_left * n_right / n) * (u_left - u_right) ** 2

            valid = (
                (t_left >= 1)
                & (c_left >= 1)
                & (t_right >= 1)
                & (c_right >= 1)
                & (n_left >= min_leaf)
                & (n_right >= min_leaf)
                & (xs[1:] > xs[:-1])
            )
            gain = np.where(valid, gain, -np.inf)

            best_pos = int(np.argmax(gain))
            if gain[best_pos] > best_gain:
                best_gain = float(gain[best_pos])
                best_feature = int(j)
                best_threshold = float((xs[best_pos] + xs[best_pos + 1]) / 2.0)
                best_left = order[: best_pos + 1]

        if best_feature is None or best_gain <= 0.0:
            return None, None, None, 0.0

        left_mask = np.zeros(n, dtype=bool)
        left_mask[best_left] = True
        return best_feature, best_threshold, left_mask, best_gain

    def _build(self, Xn, tn, yn, depth):
        """Recursively build the tree.

        Parameters
        ----------
        Xn : np.ndarray of shape (n, d)
            Local feature matrix.
        tn : np.ndarray of shape (n,)
            Binary treatment indicator.
        yn : np.ndarray of shape (n,)
            Outcome.
        depth : int
            Current depth.

        Returns
        -------
        node : dict
            Internal or leaf node.
        """
        n = Xn.shape[0]
        n_treated = int(np.sum(tn))
        n_control = n - n_treated

        stop = (
            (self.max_depth is not None and depth >= self.max_depth)
            or n < self.min_samples_split
            or n_treated == 0
            or n_control == 0
            or np.all(yn == yn[0])
        )
        if stop:
            return self._leaf(tn, yn)

        feature, threshold, left_mask, gain = self._best_split(Xn, tn, yn)
        if feature is None:
            return self._leaf(tn, yn)

        right_mask = ~left_mask
        return {
            "type": "internal",
            "feature": feature,
            "threshold": threshold,
            "gain": gain,
            "n": n,
            "left": self._build(Xn[left_mask], tn[left_mask], yn[left_mask], depth + 1),
            "right": self._build(Xn[right_mask], tn[right_mask], yn[right_mask], depth + 1),
        }

    # ------------------------------------------------------------------ #
    # Prediction / inspection
    # ------------------------------------------------------------------ #

    def predict_uplift(self, X: np.ndarray) -> np.ndarray:
        """Return the leaf uplift for each sample.

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

        out = np.empty(X.shape[0], dtype=float)
        for i, x in enumerate(X):
            out[i] = self._traverse(x, self.tree_)
        return out

    @staticmethod
    def _traverse(x: np.ndarray, node: Dict[str, Any]) -> float:
        """Descend the tree for a single sample and return the leaf uplift.

        Parameters
        ----------
        x : np.ndarray of shape (n_features,)
            A single sample.
        node : dict
            The current node.

        Returns
        -------
        uplift : float
            The uplift stored in the reached leaf.
        """
        while node["type"] != "leaf":
            if x[node["feature"]] < node["threshold"]:
                node = node["left"]
            else:
                node = node["right"]
        return node["uplift"]

    @staticmethod
    def _count_nodes(node: Dict[str, Any]) -> int:
        """Count the total number of nodes in a subtree.

        Parameters
        ----------
        node : dict
            Root of the subtree.

        Returns
        -------
        count : int
            Number of nodes in the subtree.
        """
        if node["type"] == "leaf":
            return 1
        return 1 + UpliftTreeClassifier._count_nodes(node["left"]) + UpliftTreeClassifier._count_nodes(node["right"])

    @staticmethod
    def _tree_depth(node: Dict[str, Any]) -> int:
        """Return the maximum depth of a subtree.

        Parameters
        ----------
        node : dict
            Root of the subtree.

        Returns
        -------
        depth : int
            Maximum depth (leaves have depth 0).
        """
        if node["type"] == "leaf":
            return 0
        return 1 + max(
            UpliftTreeClassifier._tree_depth(node["left"]),
            UpliftTreeClassifier._tree_depth(node["right"]),
        )
