"""Uncertainty quantification.

Distribution-free prediction sets and intervals, probability calibration, and
the metrics that verify both.

- **Conformal prediction:** :class:`SplitConformalClassifier`,
  :class:`SplitConformalRegressor`, :class:`CVPlusRegressor`,
  :class:`JackknifePlusRegressor`, :class:`APSConformalClassifier`,
  :class:`RAPSConformalClassifier`, :class:`MondrianConformalClassifier`,
  :class:`ConformalizedQuantileRegressor` — prediction sets and intervals with
  a finite-sample coverage guarantee that needs no distributional assumption.
- **Venn-Abers:** :class:`VennAbersCalibrator` — probability *intervals* that
  report their own calibration uncertainty.
- **Calibration:** :class:`PlattCalibrator`, :class:`IsotonicCalibrator`,
  :class:`TemperatureScaler`, :class:`VectorScaler` — post-processors that turn
  raw scores into probabilities you can act on.
- **Metrics:** :func:`coverage_score`, :func:`average_set_size`,
  :func:`interval_width`, :func:`brier_score`,
  :func:`expected_calibration_error`, :func:`maximum_calibration_error`,
  :func:`reliability_curve`.

These are post-processors around an already-fitted model rather than
algorithms, so they are not registered in the TuiML algorithm hub.
"""

from tuiml.uncertainty._base import Calibrator, ConformalPredictor
from tuiml.uncertainty.calibration import (
    IsotonicCalibrator,
    PlattCalibrator,
    TemperatureScaler,
    VectorScaler,
)
from tuiml.uncertainty.conformal import (
    APSConformalClassifier,
    ConformalizedQuantileRegressor,
    CVPlusRegressor,
    JackknifePlusRegressor,
    MondrianConformalClassifier,
    RAPSConformalClassifier,
    SplitConformalClassifier,
    SplitConformalRegressor,
    VennAbersCalibrator,
)
from tuiml.uncertainty.metrics import (
    average_set_size,
    brier_score,
    coverage_score,
    expected_calibration_error,
    interval_width,
    maximum_calibration_error,
    reliability_curve,
)

__all__ = [
    "ConformalPredictor",
    "Calibrator",
    "SplitConformalClassifier",
    "SplitConformalRegressor",
    "CVPlusRegressor",
    "JackknifePlusRegressor",
    "APSConformalClassifier",
    "RAPSConformalClassifier",
    "MondrianConformalClassifier",
    "ConformalizedQuantileRegressor",
    "VennAbersCalibrator",
    "PlattCalibrator",
    "IsotonicCalibrator",
    "TemperatureScaler",
    "VectorScaler",
    "coverage_score",
    "average_set_size",
    "interval_width",
    "brier_score",
    "expected_calibration_error",
    "maximum_calibration_error",
    "reliability_curve",
]
