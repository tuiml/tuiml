"""C++ acceleration backend for TuiML.

This package provides optimized C++ implementations of computational hot
paths. The compiled extension module (``_cpp_ext``) is built via pybind11 and
scikit-build-core during ``pip install`` / ``uv sync``.

A C++ compiler is **required** to install TuiML.
"""

from tuiml._cpp_ext import tree, distance, neighbors, svm  # noqa: F401

__all__ = ["tree", "distance", "neighbors", "svm"]
