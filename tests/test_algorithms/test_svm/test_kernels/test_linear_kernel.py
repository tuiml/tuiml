"""Test suite for LinearKernel.

Tests cover:
- evaluate equals dot product
- compute_matrix is symmetric
- Self-kernel equals squared norm
- Parameter schema
"""

import numpy as np
import pytest

from tuiml.algorithms.svm.kernels import LinearKernel


class TestLinearKernelInit:
    """Tests for LinearKernel initialization."""

    def test_default_initialization(self):
        """Test default initialization."""
        k = LinearKernel()

        assert k._is_built is False

    def test_parameter_schema(self):
        """Test that parameter schema is empty (no parameters)."""
        schema = LinearKernel.get_parameter_schema()

        assert isinstance(schema, dict)
        assert len(schema) == 0

    def test_repr(self):
        """Test string representation."""
        k = LinearKernel()

        assert repr(k) == "LinearKernel()"


class TestLinearKernelBuild:
    """Tests for the build() method."""

    def test_build_basic(self):
        """Test building the kernel with training data."""
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        k = LinearKernel()

        k.build(X)

        assert k._is_built is True
        assert k.n_samples_ == 3
        assert k.n_features_ == 2


class TestLinearKernelEvaluate:
    """Tests for the evaluate() method."""

    def test_evaluate_equals_dot_product(self):
        """Test that evaluate returns the dot product."""
        k = LinearKernel()
        x1 = np.array([1.0, 2.0, 3.0])
        x2 = np.array([4.0, 5.0, 6.0])

        result = k.evaluate(x1, x2)
        expected = np.dot(x1, x2)  # 1*4 + 2*5 + 3*6 = 32

        np.testing.assert_allclose(result, expected)
        np.testing.assert_allclose(result, 32.0)

    def test_self_kernel_equals_squared_norm(self):
        """Test that K(x, x) = ||x||^2."""
        k = LinearKernel()
        x = np.array([1.0, 2.0, 3.0])

        result = k.evaluate(x, x)
        expected = np.sum(x ** 2)  # 1 + 4 + 9 = 14

        np.testing.assert_allclose(result, expected)

    def test_evaluate_symmetry(self):
        """Test that K(x, y) == K(y, x)."""
        k = LinearKernel()
        np.random.seed(42)
        x1 = np.random.randn(5)
        x2 = np.random.randn(5)

        np.testing.assert_allclose(k.evaluate(x1, x2), k.evaluate(x2, x1))

    def test_evaluate_orthogonal_vectors(self):
        """Test that the kernel of orthogonal vectors is zero."""
        k = LinearKernel()
        x1 = np.array([1.0, 0.0])
        x2 = np.array([0.0, 1.0])

        np.testing.assert_allclose(k.evaluate(x1, x2), 0.0)


class TestLinearKernelComputeMatrix:
    """Tests for the compute_matrix() method."""

    def test_compute_matrix_shape(self):
        """Test that compute_matrix returns correct shape."""
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        k = LinearKernel()
        k.build(X)

        K = k.compute_matrix()

        assert K.shape == (3, 3)

    def test_compute_matrix_symmetric(self):
        """Test that the kernel matrix is symmetric."""
        np.random.seed(42)
        X = np.random.randn(10, 3)
        k = LinearKernel()
        k.build(X)

        K = k.compute_matrix()

        np.testing.assert_allclose(K, K.T, atol=1e-10)

    def test_compute_matrix_equals_X_XT(self):
        """Test that compute_matrix equals X @ X.T."""
        np.random.seed(42)
        X = np.random.randn(5, 3)
        k = LinearKernel()
        k.build(X)

        K = k.compute_matrix()
        expected = X @ X.T

        np.testing.assert_allclose(K, expected, atol=1e-10)

    def test_compute_matrix_diagonal_is_squared_norms(self):
        """Test that diagonal of kernel matrix equals squared norms."""
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        k = LinearKernel()
        k.build(X)

        K = k.compute_matrix()

        np.testing.assert_allclose(K[0, 0], 5.0)   # 1^2 + 2^2
        np.testing.assert_allclose(K[1, 1], 25.0)  # 3^2 + 4^2

    def test_compute_matrix_before_build_raises(self):
        """Test that compute_matrix before build raises an error."""
        k = LinearKernel()

        with pytest.raises(Exception):
            k.compute_matrix()
