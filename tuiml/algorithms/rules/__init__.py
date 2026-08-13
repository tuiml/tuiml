"""Rule-based algorithms.

Rule-based classifiers and regressors that learn interpretable models in the
form of "IF...THEN..." statements.

Available algorithms
--------------------
- **ZeroRuleClassifier:** Simple baseline predicting the majority class.
"""

from tuiml.algorithms.rules.zero_rule import ZeroRuleClassifier

__all__ = [
    "ZeroRuleClassifier",
]
