"""Linear algorithms for classification and regression.

Models whose prediction is a weighted sum of the inputs. Fast to fit, cheap
to predict, and directly interpretable — the coefficients say what the model
learned — which makes them the sensible baseline before anything heavier.

Algorithms
----------
- **LinearRegression:** Ordinary least squares over several features.
- **SimpleLinearRegression:** Least squares on the single best feature.
- **LogisticRegression:** Linear classification with a logistic link.
- **SimpleLogisticRegression:** LogitBoost-fitted additive logistic model.
- **SGDClassifier:** Linear classifier fitted by stochastic gradient descent.
- **SGDRegressor:** Linear regressor fitted by stochastic gradient descent.

Notes
-----
The SGD variants fit incrementally, so they suit data too large to hold in
memory at once and support ``partial_fit``. Scale your features first
(:class:`~tuiml.preprocessing.scaling.StandardScaler`): gradient descent
converges poorly when columns differ wildly in magnitude.

Examples
--------
>>> from tuiml.algorithms.linear import LogisticRegression
>>> from tuiml.datasets import load_iris
>>> data = load_iris()
>>> model = LogisticRegression().fit(data.X, data.y)
>>> model.predict(data.X[:5]).shape
(5,)
"""

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
