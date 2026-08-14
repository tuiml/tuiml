"""Interpretable-by-design ("glassbox") model family.

Glassbox models are accurate enough to use in production yet simple enough
for a human to read the entire model from top to bottom. Unlike post-hoc
explanations (SHAP, LIME) that approximate a black box, every prediction
these models make is the exact sum of the parts you can inspect.

Algorithms
----------
- **ExplainableBoostingClassifier / ExplainableBoostingRegressor:** Additive
  models of per-feature *shape functions* (EBM / GA2M). Each feature's
  bin-to-score mapping is learned and exposed, so the whole model reads as
  ``intercept + f_0(x_0) + f_1(x_1) + ...``.
- **RuleFitClassifier / RuleFitRegressor:** A forest is distilled into a
  small set of human-readable ``feature <= t`` / ``feature > t`` rules, then
  a sparse linear model is fit over the rules plus the original features.

See Also
--------
:mod:`tuiml.algorithms.trees` : The tree learners RuleFit distils.
:mod:`tuiml.algorithms.linear` : Plain (non-additive) linear baselines.
"""

from tuiml.algorithms.glassbox.ebm import (
    ExplainableBoostingClassifier,
    ExplainableBoostingRegressor,
)
from tuiml.algorithms.glassbox.rulefit import (
    RuleFitClassifier,
    RuleFitRegressor,
)

__all__ = [
    "ExplainableBoostingClassifier",
    "ExplainableBoostingRegressor",
    "RuleFitClassifier",
    "RuleFitRegressor",
]
