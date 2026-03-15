"""Test suite for PrecomputedKernel.

Tests cover:
- Build with precomputed matrix
- compute returns correct values
- compute_matrix returns a copy
- evaluate raises NotImplementedError
- Parameter schema
"""

import numpy as np
import pytest

from tuiml.algorithms.svm.kernels import PrecomputedKernel


class TestPrecomputedKernelInit:
    """Tests for PrecomputedKernel initialization."""

    def test_default_initialization(self):
        """Test default initialization without a kernel matrix."""
        k = PrecomputedKernel()

        assert k._is_built is False
        assert k._kernel_matrix is None

    def test_initialization_with_matrix(self):
        """Test initialization with a precomputed kernel matrix."""
        K = np.array([[1.0, 0.5], [0.5, 1.0]])
        k = PrecomputedKernel(kernel_matrix=K)

        assert k._kernel_matrix is not None

    def test_parameter_schema(self):
        """Test parameter schema."""
        schema = PrecomputedKernel.get_parameter_schema()

        assert "kernel_matrix" in schema


class TestPrecomputedKernelBuild:
    """Tests for the build() method."""

    def test_build_with_provided_matrix(self):
        """Test building with a precomputed kernel matrix."""
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        K = X @ X.T
        k = PrecomputedKernel(kernel_matrix=K)
        k.build(X)

        assert k._is_built is True
        assert k.n_samples_ == 3

    def test_build_without_matrix_uses_X_as_kernel(self):
        """Test that if no kernel_matrix is provided, X is treated as the kernel matrix."""
        K = np.array([[1.0, 0.5, 0.3],
                      [0.5, 1.0, 0.4],
                      [0.3, 0.4, 1.0]])
        k = PrecomputedKernel()
        k.build(K)

        assert k._is_built is True
        assert k.n_samples_ == 3

    def test_build_non_square_without_matrix_raises(self):
        """Test that a non-square X without kernel_matrix raises ValueError."""
        X = np.array([[1.0, 2.0], [3.0, 4.0]])  # 2x2 is square
        X_nonsquare = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])  # 2x3

        k = PrecomputedKernel()
        with pytest.raises(ValueError):
            k.build(X_nonsquare)

    def test_build_matrix_size_mismatch_raises(self):
        """Test that a mismatch between matrix size and X raises ValueError."""
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        K = np.array([[1.0, 0.5], [0.5, 1.0]])  # 2x2 but X has 3 samples

        k = PrecomputedKernel(kernel_matrix=K)
        with pytest.raises(ValueError):
            k.build(X)


class TestPrecomputedKernelCompute:
    """Tests for the compute() method."""

    def test_compute_returns_correct_values(self):
        """Test that compute returns the correct matrix entries."""
        K = np.array([[1.0, 0.5, 0.3],
                      [0.5, 1.0, 0.7],
                      [0.3, 0.7, 1.0]])
        k = PrecomputedKernel()
        k.build(K)

        np.testing.assert_allclose(k.compute(0, 0), 1.0)
        np.testing.assert_allclose(k.compute(0, 1), 0.5)
        np.testing.assert_allclose(k.compute(1, 2), 0.7)
        np.testing.assert_allclose(k.compute(2, 0), 0.3)

    def test_compute_before_build_raises(self):
        """Test that compute before build raises an error."""
        k = PrecomputedKernel()

        with pytest.raises(Exception):
            k.compute(0, 0)


class TestPrecomputedKernelComputeMatrix:
    """Tests for the compute_matrix() method."""

    def test_compute_matrix_returns_copy(self):
        """Test that compute_matrix returns a copy of the kernel matrix."""
        K = np.array([[1.0, 0.5], [0.5, 1.0]])
        k = PrecomputedKernel()
        k.build(K)

        result = k.compute_matrix()

        np.testing.assert_array_equal(result, K)
        # Verify it is a copy (not the same object)
        result[0, 0] = 999.0
        assert k._kernel_matrix[0, 0] == 1.0

    def test_compute_matrix_shape(self):
        """Test that compute_matrix returns correct shape."""
        K = np.eye(5)
        k = PrecomputedKernel()
        k.build(K)

        result = k.compute_matrix()
        assert result.shape == (5, 5)


class TestPrecomputedKernelEvaluate:
    """Tests for the evaluate() method."""

    def test_evaluate_raises_not_implemented(self):
        """Test that evaluate raises NotImplementedError."""
        k = PrecomputedKernel()

        with pytest.raises(NotImplementedError):
            k.evaluate(np.array([1.0]), np.array([2.0]))


class TestPrecomputedKernelSetMatrix:
    """Tests for the set_kernel_matrix() method."""

    def test_set_kernel_matrix(self):
        """Test setting a new kernel matrix."""
        k = PrecomputedKernel()
        K = np.array([[1.0, 0.5], [0.5, 1.0]])
        result = k.set_kernel_matrix(K)

        assert result is k
        assert k._kernel_matrix is not None

    def test_set_kernel_matrix_non_square_raises(self):
        """Test that a non-square matrix raises ValueError."""
        k = PrecomputedKernel()
        K = np.array([[1.0, 0.5, 0.3], [0.5, 1.0, 0.7]])

        with pytest.raises(ValueError):
            k.set_kernel_matrix(K)


class TestPrecomputedKernelRepr:
    """Tests for string representation."""

    def test_repr_built(self):
        """Test repr after building."""
        K = np.eye(3)
        k = PrecomputedKernel()
        k.build(K)

        repr_str = repr(k)
        assert "PrecomputedKernel" in repr_str
        assert "n_samples=3" in repr_str

    def test_repr_not_built(self):
        """Test repr before building."""
        k = PrecomputedKernel()

        repr_str = repr(k)
        assert "not built" in repr_str
