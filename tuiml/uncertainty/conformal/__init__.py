"""Conformal prediction.

Wrappers that convert a fitted point predictor into a set or interval
predictor with a distribution-free, finite-sample coverage guarantee.

- **Split conformal:** one model fit, needs a held-out calibration split.
- **CV+ / jackknife+:** cross-fitting instead of a held-out split; uses all the
  data at the cost of ``k`` fits.
- **APS / RAPS:** adaptive sets that grow on ambiguous inputs.
- **Mondrian:** a separate threshold per class or group, for conditional rather
  than merely marginal coverage.
- **CQR:** conformalised quantile regression, for heteroscedastic intervals.
- **Venn-Abers:** probability intervals that report their own calibration
  uncertainty.
"""

from tuiml.uncertainty.conformal.aps import (
    APSConformalClassifier,
    RAPSConformalClassifier,
)
from tuiml.uncertainty.conformal.cqr import ConformalizedQuantileRegressor
from tuiml.uncertainty.conformal.cv_plus import (
    CVPlusRegressor,
    JackknifePlusRegressor,
)
from tuiml.uncertainty.conformal.mondrian import MondrianConformalClassifier
from tuiml.uncertainty.conformal.split import (
    SplitConformalClassifier,
    SplitConformalRegressor,
)
from tuiml.uncertainty.conformal.venn_abers import VennAbersCalibrator

__all__ = [
    "SplitConformalClassifier",
    "SplitConformalRegressor",
    "CVPlusRegressor",
    "JackknifePlusRegressor",
    "APSConformalClassifier",
    "RAPSConformalClassifier",
    "MondrianConformalClassifier",
    "ConformalizedQuantileRegressor",
    "VennAbersCalibrator",
]
