"""Ensemble (meta-learning) algorithms.

Meta-learners that combine multiple base estimators to improve performance,
stability, and flexibility.

Available algorithms
--------------------
- **AdaBoostClassifier:** Adaptive boosting for multiclass classification.
- **AdditiveRegression:** Gradient boosting for regression.
- **BaggingClassifier:** Bootstrap aggregating for classification.
- **BaggingRegressor:** Bootstrap aggregating for regression.
- **FilteredClassifier:** Applies preprocessing filters before classification.
- **LogitBoostClassifier:** Additive logistic regression using boosting.
- **MultiClassClassifier:** Handles multi-class problems via binary decomposition.
- **RandomCommitteeClassifier:** Ensemble of randomizable base classifiers.
- **RandomCommitteeRegressor:** Ensemble of randomizable base regressors.
- **RandomSubspaceClassifier:** Ensemble based on random feature subsets for classification.
- **RandomSubspaceRegressor:** Ensemble based on random feature subsets for regression.
- **RegressionByDiscretization:** Regression via target discretization.
- **StackingClassifier:** Combines classifiers using a meta-learner.
- **StackingRegressor:** Combines regressors using a meta-learner.
- **VotingClassifier:** Combines classifiers using various voting rules.
- **VotingRegressor:** Combines regressors using various aggregation rules.
"""

from tuiml.algorithms.ensemble.bagging import BaggingClassifier, BaggingRegressor
from tuiml.algorithms.ensemble.adaboost_m1 import AdaBoostClassifier, AdaBoostRegressor
from tuiml.algorithms.ensemble.vote import VotingClassifier, VotingRegressor
from tuiml.algorithms.ensemble.stacking import StackingClassifier, StackingRegressor
from tuiml.algorithms.ensemble.additive_regression import AdditiveRegression
from tuiml.algorithms.ensemble.regression_by_discretization import RegressionByDiscretization
from tuiml.algorithms.ensemble.logit_boost import LogitBoostClassifier
from tuiml.algorithms.ensemble.random_committee import RandomCommitteeClassifier, RandomCommitteeRegressor
from tuiml.algorithms.ensemble.random_subspace import RandomSubspaceClassifier, RandomSubspaceRegressor
from tuiml.algorithms.ensemble.multi_class_classifier import MultiClassClassifier
from tuiml.algorithms.ensemble.filtered_classifier import FilteredClassifier

__all__ = [
    "BaggingClassifier",
    "BaggingRegressor",
    "AdaBoostClassifier",
    "AdaBoostRegressor",
    "VotingClassifier",
    "VotingRegressor",
    "StackingClassifier",
    "StackingRegressor",
    "AdditiveRegression",
    "RegressionByDiscretization",
    "LogitBoostClassifier",
    "RandomCommitteeClassifier",
    "RandomCommitteeRegressor",
    "RandomSubspaceClassifier",
    "RandomSubspaceRegressor",
    "MultiClassClassifier",
    "FilteredClassifier",
]
