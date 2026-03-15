"""
Outlier handling transformers.

Available:
    - IQROutlierDetector: IQR-based outlier detection (WEKA: InterquartileRange)
    - ValueClipper: Clip values to specified range
"""

from tuiml.preprocessing.outliers.interquartile_range import IQROutlierDetector
from tuiml.preprocessing.outliers.clip import ValueClipper

__all__ = [
    "IQROutlierDetector",
    "ValueClipper",
]
