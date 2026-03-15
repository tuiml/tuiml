"""Test suite for RBFKernel (Radial Basis Function / Gaussian kernel).

Tests cover:
- Self-kernel equals 1.0
- Evaluate is in (0, 1]
- Gamma parameter behavior
- compute_matrix shape and symmetry
- Parameter schema
"""

import numpy as np
import pytest

from tuiml.algorithms.svm.kernels import RBFKernel


class TestRBFKernelInit:
    """Tests for RBFKernel initialization."""

    def test_default_initialization(self):
        """Test default initialization."""
        k = RBFKernel()

        assert k.gamma == 0.01
        assert k._is_built is False

    def test_custom_gamma(self):
        """Test initialization with custom gamma."""
        k = RBFKernel(gamma=0.5)

        assert k.gamma == 0.5

    def test_parameter_schema(self):
        """Test parameter schema contains gamma."""
        schema = RBFKernel.get_parameter_schema()

        assert "gamma" in schema
        assert schema["gamma"]["type"] == "number"


class TestRBFKernelBuild:
    """Tests for the build() method."""

    def test_build_basic(self):
        """Test building the kernel with training data."""
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        k = RBFKernel(gamma=0.1)
        k.build(X)

        assert k._is_built is True
        assert k.gamma_ == 0.1
        assert k.n_samples_ == 3

    def test_build_gamma_scale(self):
        """Test build with gamma='scale'."""
        np.random.seed(42)
        X = np.random.randn(50, 3)
        k = RBFKernel(gamma="scale")
        k.build(X)

        assert k.gamma_ > 0
        assert np.isfinite(k.gamma_)

    def test_build_gamma_auto(self):
        """Test build with gamma='auto'."""
        np.random.seed(42)
        X = np.random.randn(50, 3)
        k = RBFKernel(gamma="auto")
        k.build(X)

        np.testing.assert_allclose(k.gamma_, 1.0 / 3.0, atol=1e-10)


class TestRBFKernelEvaluate:
    """Tests for the evaluate() method."""

    def test_self_kernel_is_one(self):
        """Test that K(x, x) = 1.0."""
        k = RBFKernel(gamma=0.1)
        x = np.array([1.0, 2.0, 3.0])

        np.testing.assert_allclose(k.evaluate(x, x), 1.0)

    def test_evaluate_in_0_1_range(self):
        """Test that evaluate returns values in (0, 1]."""
        k = RBFKernel(gamma=0.1)
        np.random.seed(42)

        for _ in range(20):
            x1 = np.random.randn(3)
            x2 = np.random.randn(3)
            val = k.evaluate(x1, x2)
            assert 0.0 < val <= 1.0 + 1e-10

    def test_evaluate_decreases_with_distance(self):
        """Test that kernel value decreases as distance increases."""
        k = RBFKernel(gamma=0.1)
        x = np.array([0.0, 0.0])
        near = np.array([0.1, 0.1])
        far = np.array([10.0, 10.0])

        val_near = k.evaluate(x, near)
        val_far = k.evaluate(x, far)

        assert val_near > val_far

    def test_evaluate_symmetry(self):
        """Test that K(x, y) == K(y, x)."""
        k = RBFKernel(gamma=0.1)
        np.random.seed(42)
        x1 = np.random.randn(5)
        x2 = np.random.randn(5)

        np.testing.assert_allclose(k.evaluate(x1, x2), k.evaluate(x2, x1))

    def test_higher_gamma_narrower_peak(self):
        """Test that higher gamma produces a narrower kernel response."""
        x = np.array([0.0, 0.0])
        y = np.array([1.0, 1.0])

        k_small = RBFKernel(gamma=0.01)
        k_large = RBFKernel(gamma=10.0)

        val_small = k_small.evaluate(x, y)
        val_large = k_large.evaluate(x, y)

        # Higher gamma => faster decay => smaller value at same distance
        assert val_small > val_large


class TestRBFKernelComputeMatrix:
    """Tests for the compute_matrix() method."""

    def test_compute_matrix_shape(self):
        """Test that compute_matrix returns correct shape."""
        np.random.seed(42)
        X = np.random.randn(10, 3)
        k = RBFKernel(gamma=0.1)
        k.build(X)

        K = k.compute_matrix()

        assert K.shape == (10, 10)

    def test_compute_matrix_symmetric(self):
        """Test that the kernel matrix is symmetric."""
        np.random.seed(42)
        X = np.random.randn(10, 3)
        k = RBFKernel(gamma=0.1)
        k.build(X)

        K = k.compute_matrix()

        np.testing.assert_allclose(K, K.T, atol=1e-10)

    def test_compute_matrix_diagonal_is_one(self):
        """Test that diagonal of the kernel matrix is 1.0."""
        np.random.seed(42)
        X = np.random.randn(10, 3)
        k = RBFKernel(gamma=0.1)
        k.build(X)

        K = k.compute_matrix()

        np.testing.assert_allclose(np.diag(K), 1.0, atol=1e-10)

    def test_compute_matrix_values_in_range(self):
        """Test that all matrix values are in (0, 1]."""
        np.random.seed(42)
        X = np.random.randn(10, 3)
        k = RBFKernel(gamma=0.1)
        k.build(X)

        K = k.compute_matrix()

        assert np.all(K > -1e-10)
        assert np.all(K <= 1.0 + 1e-10)
