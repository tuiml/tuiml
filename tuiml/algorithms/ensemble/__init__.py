"""Ensemble (meta-learning) algorithms.

Meta-learners that combine multiple base estimators to improve performance,
stability, and flexibility.

Available algorithms
--------------------
- **AdaBoostClassifier:** Adaptive boosting for multiclass classification.
- **GradientBoostingRegressor:** Gradient boosting for regression.
- **BaggingClassifier:** Bootstrap aggregating for classification.
- **BaggingRegressor:** Bootstrap aggregating for regression.
- **OneVsRestClassifier:** Handles multi-class problems via binary decomposition.
- **StackingClassifier:** Combines classifiers using a meta-learner.
- **StackingRegressor:** Combines regressors using a meta-learner.
- **VotingClassifier:** Combines classifiers using various voting rules.
- **VotingRegressor:** Combines regressors using various aggregation rules.
"""

from tuiml.algorithms.ensemble.bagging import BaggingClassifier, BaggingRegressor
from tuiml.algorithms.ensemble.adaboost import AdaBoostClassifier, AdaBoostRegressor
from tuiml.algorithms.ensemble.voting import VotingClassifier, VotingRegressor
from tuiml.algorithms.ensemble.stacking import StackingClassifier, StackingRegressor
from tuiml.algorithms.ensemble.gradient_boosting_regressor import GradientBoostingRegressor
from tuiml.algorithms.ensemble.one_vs_rest import OneVsRestClassifier

__all__ = [
    "BaggingClassifier",
    "BaggingRegressor",
    "AdaBoostClassifier",
    "AdaBoostRegressor",
    "VotingClassifier",
    "VotingRegressor",
    "StackingClassifier",
    "StackingRegressor",
    "GradientBoostingRegressor",
    "OneVsRestClassifier",
]
