"""Test suite for SigmoidKernel (Hyperbolic Tangent kernel).

Tests cover:
- Evaluate returns values in (-1, 1)
- Parameter schema
- Symmetry
- Known values
"""

import numpy as np
import pytest

from tuiml.algorithms.svm.kernels import SigmoidKernel


class TestSigmoidKernelInit:
    """Tests for SigmoidKernel initialization."""

    def test_default_initialization(self):
        """Test default initialization."""
        k = SigmoidKernel()

        assert k.gamma == 0.01
        assert k.coef0 == 0.0

    def test_custom_initialization(self):
        """Test custom initialization."""
        k = SigmoidKernel(gamma=0.1, coef0=-1.0)

        assert k.gamma == 0.1
        assert k.coef0 == -1.0

    def test_parameter_schema(self):
        """Test parameter schema contains gamma and coef0."""
        schema = SigmoidKernel.get_parameter_schema()

        assert "gamma" in schema
        assert "coef0" in schema
        assert schema["gamma"]["type"] == "number"
        assert schema["coef0"]["type"] == "number"


class TestSigmoidKernelEvaluate:
    """Tests for the evaluate() method."""

    def test_evaluate_known_values(self):
        """Test evaluate with known values."""
        k = SigmoidKernel(gamma=1.0, coef0=0.0)
        x1 = np.array([0.0])
        x2 = np.array([0.0])

        # tanh(1.0 * 0 + 0) = tanh(0) = 0
        np.testing.assert_allclose(k.evaluate(x1, x2), 0.0, atol=1e-10)

    def test_evaluate_in_minus1_to_1(self):
        """Test that evaluate returns values in (-1, 1)."""
        k = SigmoidKernel(gamma=0.01, coef0=0.0)
        np.random.seed(42)

        for _ in range(20):
            x1 = np.random.randn(5)
            x2 = np.random.randn(5)
            val = k.evaluate(x1, x2)
            assert -1.0 <= val <= 1.0

    def test_evaluate_symmetry(self):
        """Test that K(x, y) == K(y, x)."""
        k = SigmoidKernel(gamma=0.1, coef0=-0.5)
        np.random.seed(42)
        x1 = np.random.randn(5)
        x2 = np.random.randn(5)

        np.testing.assert_allclose(k.evaluate(x1, x2), k.evaluate(x2, x1))

    def test_evaluate_positive_for_large_positive_dot(self):
        """Test that the kernel is positive for large positive dot products."""
        k = SigmoidKernel(gamma=1.0, coef0=0.0)
        x1 = np.array([10.0, 10.0])
        x2 = np.array([10.0, 10.0])

        # dot = 200, tanh(200) ~ 1.0
        val = k.evaluate(x1, x2)
        assert val > 0.9

    def test_evaluate_negative_coef0(self):
        """Test with negative coef0 (common for valid Mercer conditions)."""
        k = SigmoidKernel(gamma=0.01, coef0=-1.0)
        x1 = np.array([1.0, 2.0])
        x2 = np.array([3.0, 4.0])

        # tanh(0.01 * 11 - 1) = tanh(-0.89)
        result = k.evaluate(x1, x2)
        expected = np.tanh(0.01 * 11.0 - 1.0)
        np.testing.assert_allclose(result, expected)


class TestSigmoidKernelBuild:
    """Tests for build and kernel matrix."""

    def test_build_and_compute_matrix(self):
        """Test building and computing kernel matrix."""
        np.random.seed(42)
        X = np.random.randn(5, 3)
        k = SigmoidKernel(gamma=0.1, coef0=0.0)
        k.build(X)

        K = k.compute_matrix()

        assert K.shape == (5, 5)
        np.testing.assert_allclose(K, K.T, atol=1e-10)

    def test_repr(self):
        """Test string representation."""
        k = SigmoidKernel(gamma=0.01, coef0=0.0)

        repr_str = repr(k)
        assert "SigmoidKernel" in repr_str
        assert "gamma=0.01" in repr_str
