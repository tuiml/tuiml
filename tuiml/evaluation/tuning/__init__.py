"""Hyperparameter search.

Searching a parameter space for the settings that score best under
cross-validation. The three strategies trade breadth against cost.

Searchers
---------
- **GridSearchCV:** Every combination in the grid. Exhaustive and
  reproducible, but the cost multiplies with each parameter added.
- **RandomSearchCV:** Samples a fixed number of combinations. Usually finds a
  comparable result far sooner when only a few parameters actually matter,
  since it spends its budget on distinct values rather than a full lattice.
- **BayesianSearchCV:** Models the score surface with a Gaussian process and
  picks each next trial from it. Fewest evaluations, best when a single fit
  is expensive.

Supporting types
----------------
- **ParameterGrid / ParameterDistribution:** The spaces to search.
- **TuningResult:** Best parameters, best score and the per-trial record.
- **BaseTuner:** The base class to subclass for a new strategy.

Notes
-----
Tuning and evaluating on the same split reports an optimistic score: the
parameters were chosen using that data. :class:`tuiml.Benchmark` avoids this
by running tuning inside each outer fold's training half, so the reported
score stays honest.
"""

from .grid_search import GridSearchCV
from .random_search import RandomSearchCV
from .bayesian_search import BayesianSearchCV
from tuiml.base.tuning import (
    BaseTuner,
    TuningResult,
    ParameterGrid,
    ParameterDistribution,
)

__all__ = [
    "BaseTuner",
    "TuningResult",
    "ParameterGrid",
    "ParameterDistribution",
    "GridSearchCV",
    "RandomSearchCV",
    "BayesianSearchCV",
]
