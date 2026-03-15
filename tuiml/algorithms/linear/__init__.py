"""Linear algorithms for classification and regression."""

from tuiml.algorithms.linear.logistic import LogisticRegression
from tuiml.algorithms.linear.linear_regression import LinearRegression
from tuiml.algorithms.linear.simple_linear_regression import SimpleLinearRegression
from tuiml.algorithms.linear.sgd import SGDClassifier, SGDRegressor
from tuiml.algorithms.linear.simple_logistic import SimpleLogisticRegression

__all__ = [
    "LogisticRegression",
    "LinearRegression",
    "SimpleLinearRegression",
    "SGDClassifier",
    "SGDRegressor",
    "SimpleLogisticRegression",
]
