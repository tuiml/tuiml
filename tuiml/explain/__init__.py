"""Model explanation.

Answering *why* a model predicted what it did, and *what* it relies on. These
wrap an already-fitted model rather than being algorithms, so none of them is
registered in the algorithm hub.

Global — what the model relies on overall
-----------------------------------------
- **permutation_importance:** Shuffle a feature, watch the score fall. Works
  on any model; measures what the fitted model uses.
- **drop_column_importance:** Remove a feature and refit. Answers the
  different question of whether the feature is worth collecting at all.

Effects — how a feature moves the prediction
--------------------------------------------
- **partial_dependence:** The average effect across the data.
- **individual_conditional_expectation:** One curve per sample. Look here
  before trusting a flat partial-dependence curve, which can hide curves that
  rise for half the population and fall for the other half.
- **accumulated_local_effects:** The same question answered without evaluating
  the model on feature combinations the data never contains — the right choice
  when features are correlated.

Local — why *this* prediction
-----------------------------
- **TreeExplainer:** Exact Shapley values for tree models in polynomial time.
  The attributions sum to the prediction exactly.

Notes
-----
Compute importance on **held-out** data. On training data these measure what
the model memorised rather than what it generalises with.

None of this establishes causation. A large attribution says the model used a
feature, not that the feature drives the outcome in the world.
"""

from tuiml.explain._base import Explanation
from tuiml.explain.dependence import (
    accumulated_local_effects,
    individual_conditional_expectation,
    partial_dependence,
)
from tuiml.explain.importance import (
    drop_column_importance,
    permutation_importance,
)
from tuiml.explain.shapley import TreeExplainer

__all__ = [
    "Explanation",
    "permutation_importance",
    "drop_column_importance",
    "partial_dependence",
    "individual_conditional_expectation",
    "accumulated_local_effects",
    "TreeExplainer",
]
