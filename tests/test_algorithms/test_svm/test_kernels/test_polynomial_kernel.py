"""Test suite for PolynomialKernel.

Tests cover:
- Evaluate with known values
- Degree parameter behavior
- Parameter schema
- Kernel matrix properties
"""

import numpy as np
import pytest

from tuiml.algorithms.svm.kernels import PolynomialKernel


class TestPolynomialKernelInit:
    """Tests for PolynomialKernel initialization."""

    def test_default_initialization(self):
        """Test default initialization."""
        k = PolynomialKernel()

        assert k.degree == 3
        assert k.gamma == 1.0
        assert k.coef0 == 1.0  # lower_order=True by default
        assert k.lower_order is True

    def test_custom_initialization(self):
        """Test custom initialization."""
        k = PolynomialKernel(degree=2, gamma=0.5, coef0=2.0, lower_order=False)

        assert k.degree == 2
        assert k.gamma == 0.5
        assert k.coef0 == 2.0

    def test_lower_order_false_default_coef0(self):
        """Test that lower_order=False sets coef0=0."""
        k = PolynomialKernel(lower_order=False)

        assert k.coef0 == 0.0

    def test_parameter_schema(self):
        """Test parameter schema."""
        schema = PolynomialKernel.get_parameter_schema()

        assert "degree" in schema
        assert "gamma" in schema
        assert "coef0" in schema
        assert "lower_order" in schema


class TestPolynomialKernelEvaluate:
    """Tests for the evaluate() method."""

    def test_evaluate_known_values(self):
        """Test evaluate with known values."""
        # K(x, y) = (gamma * <x, y> + coef0)^degree
        k = PolynomialKernel(degree=2, gamma=1.0, coef0=1.0)
        x1 = np.array([1.0, 2.0])
        x2 = np.array([3.0, 4.0])

        # dot = 1*3 + 2*4 = 11
        # K = (1.0 * 11 + 1.0)^2 = 12^2 = 144
        result = k.evaluate(x1, x2)
        np.testing.assert_allclose(result, 144.0)

    def test_evaluate_degree_one_linear(self):
        """Test that degree=1 with coef0=0 is equivalent to linear kernel."""
        k = PolynomialKernel(degree=1, gamma=1.0, coef0=0.0)
        x1 = np.array([1.0, 2.0, 3.0])
        x2 = np.array([4.0, 5.0, 6.0])

        result = k.evaluate(x1, x2)
        expected = np.dot(x1, x2)

        np.testing.assert_allclose(result, expected)

    def test_evaluate_symmetry(self):
        """Test that K(x, y) == K(y, x)."""
        k = PolynomialKernel(degree=3)
        np.random.seed(42)
        x1 = np.random.randn(5)
        x2 = np.random.randn(5)

        np.testing.assert_allclose(k.evaluate(x1, x2), k.evaluate(x2, x1))

    def test_evaluate_homogeneous(self):
        """Test homogeneous polynomial kernel (coef0=0)."""
        k = PolynomialKernel(degree=2, gamma=1.0, coef0=0.0)
        x1 = np.array([1.0, 0.0])
        x2 = np.array([0.0, 1.0])

        # Orthogonal vectors: dot = 0, K = 0^2 = 0
        np.testing.assert_allclose(k.evaluate(x1, x2), 0.0)

    def test_evaluate_with_gamma_scaling(self):
        """Test that gamma scales the dot product correctly."""
        x1 = np.array([1.0, 2.0])
        x2 = np.array([3.0, 4.0])

        k1 = PolynomialKernel(degree=2, gamma=1.0, coef0=0.0)
        k2 = PolynomialKernel(degree=2, gamma=2.0, coef0=0.0)

        # dot = 11
        # k1: (1.0 * 11)^2 = 121
        # k2: (2.0 * 11)^2 = 484
        np.testing.assert_allclose(k1.evaluate(x1, x2), 121.0)
        np.testing.assert_allclose(k2.evaluate(x1, x2), 484.0)


class TestPolynomialKernelBuildAndMatrix:
    """Tests for build and compute_matrix."""

    def test_build_and_compute_matrix(self):
        """Test building and computing kernel matrix."""
        np.random.seed(42)
        X = np.random.randn(5, 3)
        k = PolynomialKernel(degree=2)
        k.build(X)

        K = k.compute_matrix()

        assert K.shape == (5, 5)
        np.testing.assert_allclose(K, K.T, atol=1e-10)

    def test_repr(self):
        """Test string representation."""
        k = PolynomialKernel(degree=3, gamma=1.0, coef0=1.0)

        repr_str = repr(k)
        assert "PolynomialKernel" in repr_str
        assert "degree=3" in repr_str
