"""Probability calibration.

Post-processors that turn raw classifier scores into calibrated probabilities:

- **PlattCalibrator:** parametric sigmoid fit; the default for small
  calibration sets and margin-like scores.
- **IsotonicCalibrator:** non-parametric monotone step function via PAVA;
  more expressive, needs more calibration data.
- **TemperatureScaler:** single-parameter multiclass scaling that preserves
  accuracy exactly.
- **VectorScaler:** per-class scale and bias for class-dependent
  miscalibration.
"""

from tuiml.uncertainty.calibration.isotonic import IsotonicCalibrator
from tuiml.uncertainty.calibration.platt import PlattCalibrator
from tuiml.uncertainty.calibration.temperature import TemperatureScaler, VectorScaler

__all__ = [
    "PlattCalibrator",
    "IsotonicCalibrator",
    "TemperatureScaler",
    "VectorScaler",
]
