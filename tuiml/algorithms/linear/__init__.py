"""Linear algorithms for classification and regression.

- **SGDRegressor:** Stochastic Gradient Descent for regression.
- **SklearnLogisticRegression:** Scikit-Learn Logistic Regression.
"""

from tuiml.algorithms.linear.logistic import LogisticRegression
from tuiml.algorithms.linear.linear_regression import LinearRegression
from tuiml.algorithms.linear.simple_linear_regression import SimpleLinearRegression
from tuiml.algorithms.linear.sgd import SGDClassifier, SGDRegressor
from tuiml.algorithms.linear.simple_logistic import SimpleLogisticRegression

try:
    import sklearn
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

if SKLEARN_AVAILABLE:
    from tuiml.algorithms.linear.sklearn_logistic import SklearnLogisticRegression
    from tuiml.algorithms.linear.sklearn_lin_reg import SklearnLinearRegression
    from tuiml.algorithms.linear.sklearn_regularized_linear import SklearnRidge, SklearnRidgeClassifier, SklearnLasso, SklearnElasticNet

__all__ = [
    "LogisticRegression",
    "LinearRegression",
    "SimpleLinearRegression",
    "SGDClassifier",
    "SGDRegressor",
    "SimpleLogisticRegression",
]

if SKLEARN_AVAILABLE:
    __all__.extend([
        "SklearnLogisticRegression",
        "SklearnLinearRegression",
        "SklearnRidge",
        "SklearnRidgeClassifier",
        "SklearnLasso",
        "SklearnElasticNet",
    ])

