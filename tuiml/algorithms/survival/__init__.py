"""Survival analysis algorithms.

Time-to-event modelling where some subjects are right-censored: the event of
interest (failure, churn, relapse) had not happened when observation stopped,
so their recorded time is a lower bound on the true event time. Input is
therefore ``(X, time, event)`` rather than ``(X, y)``, with ``event`` marking
which ``time`` values are observed events and which are censoring times.

Algorithms
----------
- **KaplanMeierEstimator:** The non-parametric product-limit estimate of the
  survival curve. No covariates; the population-level baseline.
- **NelsonAalenEstimator:** The non-parametric cumulative-hazard estimate.
  Useful as the hazard analogue of Kaplan-Meier and as a building block for
  leaf hazards in :class:`RandomSurvivalForest`.
- **CoxPHSurvival:** The Cox proportional-hazards model. Semiparametric:
  estimates covariate effects (``coefficients_``) without specifying the
  baseline hazard, with optional L2 regularisation.
- **RandomSurvivalForest:** An ensemble of censoring-aware survival trees.

Metrics
-------
- ``concordance_index``: Harrell's C-index, the share of comparable pairs the
  model ranks correctly.
- ``integrated_brier_score``: IPCW-weighted Brier score integrated over a grid
  of times.
- ``logrank_test``: two-sample log-rank test of whether two survival curves
  differ.

Notes
-----
Risk scores are oriented so that a **larger** value means an **earlier**
expected event. That is the convention :class:`~tuiml.base.algorithms.Survival`
documents, and every model here follows it.

See Also
--------
:mod:`tuiml.base.algorithms.Survival` : The base class these models subclass.
"""

from tuiml.base.algorithms import Survival, survival

from tuiml.algorithms.survival.kaplan_meier import (
    KaplanMeierEstimator,
    NelsonAalenEstimator,
)
from tuiml.algorithms.survival.cox_ph import CoxPHSurvival
from tuiml.algorithms.survival.random_survival_forest import RandomSurvivalForest
from tuiml.algorithms.survival.metrics import (
    concordance_index,
    integrated_brier_score,
    logrank_test,
)

__all__ = [
    "Survival",
    "survival",
    "KaplanMeierEstimator",
    "NelsonAalenEstimator",
    "CoxPHSurvival",
    "RandomSurvivalForest",
    "concordance_index",
    "integrated_brier_score",
    "logrank_test",
]
