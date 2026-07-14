"""Support Vector Machine algorithms.

This module provides SVM classifiers and regressors with various kernel functions.

Algorithms
----------
- **SVC:** SVM classifier using Sequential Minimal Optimization.
- **SVR:** SVM regressor for regression tasks.
- **SklearnSVC:** Scikit-Learn SVC wrapper.

Kernels
-------
- **LinearKernel:** Linear kernel.
- **PolynomialKernel:** Polynomial kernel.
- **RBFKernel:** Radial Basis Function (Gaussian) kernel.
- **SigmoidKernel:** Sigmoid (tanh) kernel.
- **PearsonUniversalKernel:** Pearson VII Universal Kernel.
- **StringKernel:** Kernel for string data.
- **PrecomputedKernel:** Precomputed kernel matrix.
"""

from tuiml.algorithms.svm.smo import SVC
from tuiml.algorithms.svm.smoreg import SVR

# Import all kernels for convenience
from tuiml.algorithms.svm.kernels import (
    Kernel,
    CachedKernel,
    kernel,
    LinearKernel,
    PolynomialKernel,
    NormalizedPolynomialKernel,
    RBFKernel,
    SigmoidKernel,
    PearsonUniversalKernel,
    StringKernel,
    PrecomputedKernel,
)

try:
    import sklearn
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

if SKLEARN_AVAILABLE:
    from tuiml.algorithms.svm.sklearn_svc import SklearnSVC
    from tuiml.algorithms.svm.sklearn_svr import SklearnSVR

__all__ = [
    "SVC",
    "SVR",
    "Kernel",
    "CachedKernel",
    "kernel",
    "LinearKernel",
    "PolynomialKernel",
    "NormalizedPolynomialKernel",
    "RBFKernel",
    "SigmoidKernel",
    "PearsonUniversalKernel",
    "StringKernel",
    "PrecomputedKernel",
]

if SKLEARN_AVAILABLE:
    __all__.extend([
        "SklearnSVC",
        "SklearnSVR",
    ])

