"""Handling extreme values before they distort a model.

A few extreme rows can move a mean, stretch a scaler's range, and dominate
any squared-error objective. These transformers deal with them as a
preprocessing step.

Transformers
------------
- **IQROutlierDetector:** Flags values beyond a multiple of the interquartile
  range. Quartiles are themselves resistant to outliers, so the rule does not
  get dragged around by the points it is trying to find.
- **ValueClipper:** Caps values at explicit bounds, keeping the row and its
  label while limiting how far the value can pull a fit.

Notes
-----
Clipping preserves your sample size; dropping rows does not, and it discards
labels along with them. Prefer clipping unless the extremes are known to be
recording errors.

An outlier is not automatically noise. In fraud, fault and intrusion
detection the rare rows *are* the signal — there, model them with
:mod:`tuiml.algorithms.anomaly` rather than removing them here.

See Also
--------
:mod:`tuiml.algorithms.anomaly` : When anomalies are the thing to predict.
"""

from tuiml.preprocessing.outliers.iqr import IQROutlierDetector
from tuiml.preprocessing.outliers.clip import ValueClipper

__all__ = [
    "IQROutlierDetector",
    "ValueClipper",
]
