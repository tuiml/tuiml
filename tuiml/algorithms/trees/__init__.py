"""Tree-based algorithms.

This module provides decision trees and tree ensembles for classification
and regression tasks.

Algorithms
----------
- **DecisionTreeClassifier:** CART decision tree classifier.
- **DecisionTreeRegressor:** CART decision tree regressor.
- **DecisionStumpClassifier:** One-level decision tree (weak learner).
- **RandomForestClassifier:** Ensemble of random trees for classification.
- **RandomForestRegressor:** Ensemble of random trees for regression.
- **BalancedRandomForestClassifier:** Class-balanced random forest.
"""

from tuiml.algorithms.trees.decision_stump import DecisionStumpClassifier
from tuiml.algorithms.trees.random_forest import (
    BalancedRandomForestClassifier,
    RandomForestClassifier,
    RandomForestRegressor,
)
from tuiml.algorithms.trees.decision_tree import (
    DecisionTreeClassifier,
    DecisionTreeRegressor,
)

__all__ = [
    "DecisionTreeClassifier",
    "DecisionTreeRegressor",
    "DecisionStumpClassifier",
    "RandomForestClassifier",
    "RandomForestRegressor",
    "BalancedRandomForestClassifier",
    # Backward compat
]