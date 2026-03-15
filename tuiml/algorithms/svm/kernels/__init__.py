"""SVM Kernel Functions.

Kernel functions for Support Vector Machines that compute similarity
in (possibly infinite) feature spaces via the kernel trick.

Available kernels
-----------------
- **LinearKernel:** Standard dot product.
- **PolynomialKernel:** Polynomial transformation of dot product.
- **RBFKernel:** Radial Basis Function (Gaussian) kernel.
- **SigmoidKernel:** Hyperbolic tangent kernel.
- **PearsonUniversalKernel:** Pearson VII Universal Kernel.
- **StringKernel:** Subsequence matching for text data.
- **PrecomputedKernel:** Uses a user-provided kernel matrix.
"""

# Base classes
from tuiml.base.kernels import (
    Kernel,
    CachedKernel,
    kernel,
)

# Kernels
from tuiml.algorithms.svm.kernels.linear import LinearKernel
from tuiml.algorithms.svm.kernels.polynomial import PolynomialKernel, NormalizedPolynomialKernel
from tuiml.algorithms.svm.kernels.rbf import RBFKernel
from tuiml.algorithms.svm.kernels.sigmoid import SigmoidKernel
from tuiml.algorithms.svm.kernels.puk import PearsonUniversalKernel
from tuiml.algorithms.svm.kernels.string import StringKernel
from tuiml.algorithms.svm.kernels.precomputed import PrecomputedKernel

__all__ = [
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
