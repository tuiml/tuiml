"""Causal inference — uplift and heterogeneous treatment effects.

Estimating *who* a treatment helps: the difference between the outcome with
and without treatment, per individual. Unlike a plain regression, these models
take ``(X, treatment, y)`` and predict the individual treatment effect, or
uplift, which is what decides who to treat.

Algorithms
----------
- **SLearner:** One model on ``[X, treatment]``; uplift is the difference
  between the treated and control predictions. Simplest, but the treatment
  signal can get diluted among the covariates.
- **TLearner:** Two models, one per treatment group; uplift is their
  difference. Lets each group have its own response surface.
- **XLearner:** T-learner plus two "imputed effect" models fitted on the
  cross-group residuals, combined with a propensity weight. Often the best of
  the meta-learners when treatment groups are imbalanced.
- **UpliftTreeClassifier:** A single tree that splits directly on uplift gain,
  so the leaves are groups with genuinely different treatment effects.

Utilities
---------
- :func:`~tuiml.algorithms.causal.metrics.qini_curve` — cumulative incremental
  gain of treating the top-ranked individuals.
- :func:`~tuiml.algorithms.causal.metrics.auuc` — area under the uplift curve.
- :func:`~tuiml.algorithms.causal.metrics.uplift_at_k` — mean uplift of the
  top-k predicted group.

Notes
-----
``treatment`` is binary (``0`` = control, ``1`` = treated) and must contain
both groups. ``y`` is a numeric outcome; uplift is ``E[y | t=1] - E[y | t=0]``.

See Also
--------
:mod:`tuiml.algorithms.trees` : The regressors these meta-learners wrap.
"""

from tuiml.base.algorithms import UpliftModel, uplift

from tuiml.algorithms.causal.meta_learners import SLearner, TLearner, XLearner
from tuiml.algorithms.causal.uplift_tree import UpliftTreeClassifier
from tuiml.algorithms.causal.metrics import qini_curve, auuc, uplift_at_k

__all__ = [
    "UpliftModel",
    "uplift",
    "SLearner",
    "TLearner",
    "XLearner",
    "UpliftTreeClassifier",
    "qini_curve",
    "auuc",
    "uplift_at_k",
]
