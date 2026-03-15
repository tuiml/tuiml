"""Bayesian and probabilistic learning algorithms.

This module provides classifiers and regressors based on Bayes' theorem,
graphical models, and kernel-based probabilistic methods. It includes
standard implementations of Naive Bayes, Bayesian Networks, and Gaussian
Processes.

Algorithms
----------
- **NaiveBayesClassifier:** Gaussian and kernel-indexed probabilistic classifier.
- **NaiveBayesMultinomialClassifier:** Specialized for discrete/text classification.
- **BayesianNetworkClassifier:** Graphical models with structure learning (TAN, K2).
- **GaussianProcessesRegressor:** Bayesian non-parametric regression with
  uncertainty estimation.

Estimators
----------
The module also provides pluggable probability estimators used by the
above models for modeling continuous and discrete feature distributions.
"""

from tuiml.algorithms.bayesian.naive_bayes import NaiveBayesClassifier
from tuiml.algorithms.bayesian.naive_bayes_multinomial import NaiveBayesMultinomialClassifier
from tuiml.algorithms.bayesian.bayes_net import BayesianNetworkClassifier
from tuiml.algorithms.bayesian.gaussian_processes import GaussianProcessesRegressor
from tuiml.algorithms.bayesian.bayesian_linear_regression import BayesianLinearRegressor

from tuiml.algorithms.bayesian import estimators
from tuiml.algorithms.bayesian.estimators import (
    Estimator,
    NormalEstimator,
    DiscreteEstimator,
    KernelEstimator
)

__all__ = [
    "NaiveBayesClassifier",
    "NaiveBayesMultinomialClassifier",
    "BayesianNetworkClassifier",
    "GaussianProcessesRegressor",
    "BayesianLinearRegressor",
    "estimators",
    "Estimator",
    "NormalEstimator",
    "DiscreteEstimator",
    "KernelEstimator",
]
