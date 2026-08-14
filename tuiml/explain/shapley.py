"""Exact Shapley attributions for TuiML's tree models."""

from __future__ import annotations

from typing import Any, List, Optional

import numpy as np

from tuiml._cpp_ext import shapley as _cpp_shapley
from tuiml.explain._base import Explanation


def _flattened(tree: Any, value_width: int) -> Any:
    """Return the flattened array form of a tree.

    A standalone estimator caches this as ``flat_tree_``. A forest keeps raw
    ``TreeNode`` roots instead, so those are flattened here — the trees index
    the full feature space, since ``max_features`` subsamples at split time
    rather than restricting columns.

    Parameters
    ----------
    tree : Algorithm or TreeNode
        A fitted TuiML decision tree, or a forest member's root node.
    value_width : int
        Width of a node value: ``n_classes`` for a classifier, 1 otherwise.

    Returns
    -------
    flat : FlattenedTree
        The flat arrays used for traversal.
    """
    flat = getattr(tree, "flat_tree_", None)
    if flat is not None:
        return flat

    from tuiml.algorithms.trees._core.nodes import flatten_tree

    return flatten_tree(tree, value_width=value_width)


class TreeExplainer:
    """Exact per-prediction Shapley values for tree models, in polynomial time.

    Shapley values are the only attribution satisfying efficiency, symmetry,
    dummy and additivity simultaneously — which is why they are the standard
    against which other explanations are judged. Their definition sums over
    all :math:`2^F` feature subsets, so computing them exactly is normally
    hopeless. **TreeSHAP exploits a tree's structure to get the exact answer
    anyway**, in time polynomial in depth rather than exponential in features.

    Overview
    --------
    1. Push the background data through the tree to record what fraction of it
       reaches each node.
    2. For a sample, walk every root-to-leaf path once, carrying the set of
       features split on so far together with the proportion of subsets in
       which each is present or absent.
    3. At each leaf, credit every feature on the path with its marginal
       contribution.

    Theory
    ------
    The Shapley value of feature :math:`j` for prediction :math:`f(x)` is

    .. math::
        \\phi_j = \\sum_{S \\subseteq F \\setminus \\{j\\}}
        \\frac{|S|! \\ (|F| - |S| - 1)!}{|F|!}
        \\left[ f_x(S \\cup \\{j\\}) - f_x(S) \\right]

    where :math:`f_x(S)` is the prediction with only the features in
    :math:`S` known and the rest marginalised over the background. Evaluating
    that directly costs :math:`O(2^F)`; carrying the subset-proportion
    bookkeeping down the tree instead collapses it to :math:`O(L D^2)` for
    :math:`L` leaves and depth :math:`D`.

    **Efficiency** is the property that makes the output trustworthy and
    testable: for every sample,

    .. math::
        \\sum_j \\phi_j + \\mathbb{E}[f] = f(x)

    The attributions add up to the prediction exactly. This implementation is
    verified against a brute-force enumeration of all subsets and agrees to
    machine precision.

    Parameters
    ----------
    model : Algorithm
        A fitted TuiML tree model: ``DecisionTreeClassifier``,
        ``DecisionTreeRegressor``, ``RandomForestClassifier`` or
        ``RandomForestRegressor``.
    background : np.ndarray of shape (n_background, n_features)
        Data defining the expectation a feature is marginalised over.
        Required, deliberately: attributions are always *relative* to this
        background, so "why is this prediction high?" only means something
        once "high compared to what?" is answered. Passing the training set
        gives the usual reading; passing a subgroup asks why this prediction
        differs from that subgroup instead.
    feature_names : list of str, optional
        Names for the report.

    Attributes
    ----------
    expected_value_ : np.ndarray
        The model's mean output over the background, one entry per output.
    n_features_ : int
        Number of features the model was fitted on.

    Notes
    -----
    **Complexity.** :math:`O(T L D^2)` per sample for a forest of ``T`` trees,
    run in the shared C++ kernel ``tuiml._cpp_ext.shapley.tree_shap`` and
    parallel over samples. A forest costs the sum of its trees, so explaining
    a 500-tree forest is 500 times the work of explaining one.

    **This is the path-dependent variant.** A feature is marginalised by
    following the background's branch proportions at each split, which
    respects the correlations the tree learned. The alternative
    interventional variant breaks those correlations deliberately and answers
    a subtly different causal question; the two disagree when features are
    correlated, and neither is wrong — they answer different questions.

    **Attribution is not causation.** A large :math:`\\phi_j` says the model
    used that feature, not that the feature drives the outcome in the world.
    A model that leaks a proxy for the label will attribute confidently to it.

    References
    ----------
    .. [Lundberg2017] Lundberg, S. M., & Lee, S.-I. (2017). A Unified Approach
       to Interpreting Model Predictions. *NeurIPS*, 4765-4774.
       :arxiv:`1705.07874`
    .. [Lundberg2020] Lundberg, S. M., Erion, G., Chen, H., DeGrave, A.,
       Prutkin, J. M., Nair, B., Katz, R., Himmelfarb, J., Bansal, N., &
       Lee, S.-I. (2020). From Local Explanations to Global Understanding with
       Explainable AI for Trees. *Nature Machine Intelligence*, 2(1), 56-67.
       :doi:`10.1038/s42256-019-0138-9`

    See Also
    --------
    :func:`~tuiml.explain.permutation_importance` : Model-agnostic and global; far cheaper, far coarser.
    :func:`~tuiml.explain.partial_dependence` : How a feature moves the prediction on average, rather than per sample.

    Examples
    --------
    >>> import numpy as np
    >>> from tuiml.explain import TreeExplainer
    >>> from tuiml.algorithms.trees import DecisionTreeRegressor
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(200, 4))
    >>> y = 3.0 * X[:, 0] - 2.0 * X[:, 1] + rng.normal(0, 0.1, 200)
    >>> model = DecisionTreeRegressor(max_depth=5).fit(X, y)
    >>> explainer = TreeExplainer(model, background=X)
    >>> result = explainer.explain(X[:10])
    >>> result.values.shape
    (10, 4)

    The attributions reconstruct the prediction exactly, which is what makes
    them worth trusting:

    >>> reconstructed = result.values.sum(axis=1) + explainer.expected_value_[0]
    >>> bool(np.allclose(reconstructed, model.predict(X[:10])))
    True
    """

    def __init__(
        self,
        model: Any,
        background: Optional[np.ndarray] = None,
        feature_names: Optional[List[str]] = None,
    ):
        """Initialize the tree explainer.

        Parameters
        ----------
        model : Algorithm
            A fitted TuiML tree or forest.
        background : np.ndarray, optional
            Data defining the expectation features are marginalised over.
        feature_names : list of str, optional
            Names for the report.
        """
        self.model = model
        self.feature_names = feature_names

        self._trees = self._collect_trees(model)
        if not self._trees:
            raise ValueError(
                f"{model.__class__.__name__} is not a fitted TuiML tree or "
                "forest; TreeExplainer supports DecisionTree* and RandomForest*"
            )

        if background is None:
            raise ValueError(
                "background data is required: Shapley values are defined "
                "relative to an expectation, so there is no sensible default"
            )
        background = np.ascontiguousarray(background, dtype=np.float64)
        self.n_features_ = background.shape[1]

        value_width = int(getattr(model, "n_classes_", 0) or 1)
        self._arrays = [
            self._prepare(tree, background, value_width) for tree in self._trees
        ]
        self.expected_value_ = np.mean(
            [arrays["expected"] for arrays in self._arrays], axis=0
        )

    def explain(self, X: np.ndarray) -> Explanation:
        """Compute Shapley values for each sample.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Samples to explain.

        Returns
        -------
        explanation : Explanation
            ``values`` has shape ``(n_samples, n_features)`` for single-output
            models and ``(n_samples, n_features, n_outputs)`` otherwise.
            ``base_value`` holds ``expected_value_``.
        """
        X = np.ascontiguousarray(np.atleast_2d(X), dtype=np.float64)
        if X.shape[1] != self.n_features_:
            raise ValueError(
                f"X has {X.shape[1]} features but the background had "
                f"{self.n_features_}"
            )

        total = None
        for arrays in self._arrays:
            contribution = np.asarray(
                _cpp_shapley.tree_shap(
                    arrays["feature"],
                    arrays["threshold"],
                    arrays["children_left"],
                    arrays["children_right"],
                    arrays["value"],
                    arrays["weight"],
                    X,
                )
            )
            total = contribution if total is None else total + contribution

        # A forest predicts the mean of its trees, so its attributions are the
        # mean of theirs — additivity carrying through the ensemble.
        values = total / len(self._arrays)
        if values.shape[2] == 1:
            values = values[:, :, 0]

        return Explanation(
            values=values,
            feature_names=self.feature_names,
            method="tree_shap",
            base_value=self.expected_value_,
            metadata={"n_trees": len(self._arrays)},
        )

    @staticmethod
    def _collect_trees(model: Any) -> List[Any]:
        """Return the fitted trees inside a model.

        Parameters
        ----------
        model : Algorithm
            A tree or a forest.

        Returns
        -------
        trees : list
            Individual fitted trees, or an empty list when unsupported.
        """
        if getattr(model, "flat_tree_", None) is not None:
            return [model]

        # A forest keeps raw TreeNode roots rather than fitted estimators.
        from tuiml.algorithms.trees._core.nodes import TreeNode

        estimators = getattr(model, "estimators_", None) or getattr(
            model, "trees_", None
        )
        if estimators:
            return [
                tree for tree in estimators
                if getattr(tree, "flat_tree_", None) is not None
                or isinstance(tree, TreeNode)
            ]
        return []

    @staticmethod
    def _prepare(tree: Any, background: np.ndarray, value_width: int) -> dict:
        """Extract a tree's arrays and its background node coverage.

        Parameters
        ----------
        tree : Algorithm or TreeNode
            A fitted TuiML decision tree, or a forest member's root.
        background : np.ndarray of shape (n_background, n_features)
            Data defining the expectation.
        value_width : int
            Width of a node value.

        Returns
        -------
        arrays : dict
            Contiguous arrays ready for the C++ kernel, plus the expected
            output over the background.
        """
        flat = _flattened(tree, value_width)

        feature = np.ascontiguousarray(flat.feature, dtype=np.int32)
        threshold = np.ascontiguousarray(flat.threshold, dtype=np.float64)
        children_left = np.ascontiguousarray(flat.children_left, dtype=np.int32)
        children_right = np.ascontiguousarray(flat.children_right, dtype=np.int32)
        value = np.ascontiguousarray(
            np.asarray(flat.value, dtype=np.float64).reshape(flat.n_nodes, -1)
        )

        # Node coverage under the background: the proportion of it reaching
        # each node is exactly what "marginalise this feature" means here.
        counts = np.zeros(flat.n_nodes, dtype=np.float64)
        for row in background:
            node = 0
            while True:
                counts[node] += 1.0
                if feature[node] < 0:
                    break
                node = (
                    children_left[node]
                    if row[feature[node]] <= threshold[node]
                    else children_right[node]
                )
        counts /= max(len(background), 1)

        # The root's coverage-weighted value is the model's mean output, which
        # is the base value the attributions are measured against.
        leaves = feature < 0
        expected = (value[leaves] * counts[leaves, None]).sum(axis=0)

        return {
            "feature": feature,
            "threshold": threshold,
            "children_left": children_left,
            "children_right": children_right,
            "value": value,
            "weight": counts,
            "expected": expected,
        }

    def __repr__(self) -> str:
        """Return a readable representation of the explainer."""
        return (
            f"TreeExplainer(model={self.model.__class__.__name__}, "
            f"n_trees={len(self._arrays)})"
        )
