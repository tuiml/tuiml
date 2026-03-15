"""Shared tree engine: nodes, criteria, splitters, builders, pruning, prediction."""

from .nodes import TreeNode, FlattenedTree, flatten_tree, count_nodes, max_depth_of
from .utils import get_tree_description
from .criteria import (
    gini_impurity,
    entropy,
    entropy_from_counts,
    gain_ratio_score,
    classifier_node_impurity,
    squared_error,
    friedman_mse,
    absolute_error,
    sdr,
    regressor_node_impurity,
)
from .splitters import (
    compute_max_features,
    best_split_classifier,
    best_split_regressor,
    best_split_stump,
)
from .builders import TreeConfig, build_classifier_tree, build_regressor_tree
from .pruning import (
    cost_complexity_prune,
    reduced_error_prune_classifier,
    reduced_error_prune_regressor,
    pessimistic_prune,
)
from .predict import (
    build_jit_functions,
    predict_batch,
    predict_proba_batch,
    predict_single_numpy,
    predict_proba_single_numpy,
)

__all__ = [
    # Nodes
    "TreeNode",
    "FlattenedTree",
    "flatten_tree",
    "count_nodes",
    "max_depth_of",
    # Utils
    "get_tree_description",
    # Criteria
    "gini_impurity",
    "entropy",
    "entropy_from_counts",
    "gain_ratio_score",
    "classifier_node_impurity",
    "squared_error",
    "friedman_mse",
    "absolute_error",
    "sdr",
    "regressor_node_impurity",
    # Splitters
    "compute_max_features",
    "best_split_classifier",
    "best_split_regressor",
    "best_split_stump",
    # Builders
    "TreeConfig",
    "build_classifier_tree",
    "build_regressor_tree",
    # Pruning
    "cost_complexity_prune",
    "reduced_error_prune_classifier",
    "reduced_error_prune_regressor",
    "pessimistic_prune",
    # Predict
    "build_jit_functions",
    "predict_batch",
    "predict_proba_batch",
    "predict_single_numpy",
    "predict_proba_single_numpy",
]
