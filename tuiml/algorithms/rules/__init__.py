"""Rule-based algorithms.

Rule-based classifiers and regressors that learn interpretable models in the
form of "IF...THEN..." statements.

Available algorithms
--------------------
- **ZeroRuleClassifier:** Simple baseline predicting the majority class.
- **OneRuleClassifier:** Learns a single-attribute rule.
- **RIPPERClassifier:** Implementation of the RIPPER rule learner.
- **PARTClassifier:** Generates rules via partial decision trees.
- **M5ModelRulesRegressor:** Learns regression rules from M5 model trees.
- **DecisionTableClassifier:** Simple decision table majority classifier.
"""

from tuiml.algorithms.rules.zeror import ZeroRuleClassifier
from tuiml.algorithms.rules.oner import OneRuleClassifier
from tuiml.algorithms.rules.jrip import RIPPERClassifier
from tuiml.algorithms.rules.part import PARTClassifier
from tuiml.algorithms.rules.m5rules import M5ModelRulesRegressor
from tuiml.algorithms.rules.decision_table import DecisionTableClassifier

__all__ = [
    "ZeroRuleClassifier",
    "OneRuleClassifier",
    "RIPPERClassifier",
    "PARTClassifier",
    "M5ModelRulesRegressor",
    "DecisionTableClassifier",
]
