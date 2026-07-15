"""Tree-based algorithms.

This module provides decision trees and tree ensembles for classification
and regression tasks.

Algorithms
----------
- **DecisionTreeClassifier:** CART decision tree classifier.
- **DecisionTreeRegressor:** CART decision tree regressor.
- **C45TreeClassifier:** C4.5 decision tree with pruning.
- **C45TreeRegressor:** C4.5 regression tree with pruning.
- **DecisionStumpClassifier:** One-level decision tree (weak learner).
- **RandomTreeClassifier:** Randomized decision tree (Random Forest component).
- **RandomTreeRegressor:** Randomized regression tree.
- **RandomForestClassifier:** Ensemble of random trees for classification.
- **RandomForestRegressor:** Ensemble of random trees for regression.
- **ReducedErrorPruningTreeClassifier:** Reduced Error Pruning tree for classification.
- **ReducedErrorPruningTreeRegressor:** Reduced Error Pruning tree for regression.
- **HoeffdingTreeClassifier:** Streaming decision tree (VFDT).
- **LogisticModelTreeClassifier:** Logistic Model Trees.
- **M5ModelTreeRegressor:** Model trees for regression.
"""

from tuiml.algorithms.trees.decision_stump import DecisionStumpClassifier
from tuiml.algorithms.trees.j48 import C45TreeClassifier, C45TreeRegressor, C45DecisionTreeClassifier, C45DecisionTreeRegressor
from tuiml.algorithms.trees.random_tree import RandomTreeClassifier, RandomTreeRegressor
from tuiml.algorithms.trees.random_forest import RandomForestClassifier, RandomForestRegressor
from tuiml.algorithms.trees.rep_tree import ReducedErrorPruningTreeClassifier, ReducedErrorPruningTreeRegressor
from tuiml.algorithms.trees.hoeffding_tree import HoeffdingTreeClassifier
from tuiml.algorithms.trees.m5p import M5ModelTreeRegressor
from tuiml.algorithms.trees.lmt import LogisticModelTreeClassifier
from tuiml.algorithms.trees.decision_tree import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
)

__all__ = [
    "DecisionTreeClassifier",
    "DecisionTreeRegressor",
    "DecisionStumpClassifier",
    "C45TreeClassifier",
    "C45TreeRegressor",
    "RandomTreeClassifier",
    "RandomTreeRegressor",
    "RandomForestClassifier",
    "RandomForestRegressor",
    "ReducedErrorPruningTreeClassifier",
    "ReducedErrorPruningTreeRegressor",
    "HoeffdingTreeClassifier",
    "M5ModelTreeRegressor",
    "LogisticModelTreeClassifier",
    # Backward compat
    "C45DecisionTreeClassifier",
    "C45DecisionTreeRegressor",
]
