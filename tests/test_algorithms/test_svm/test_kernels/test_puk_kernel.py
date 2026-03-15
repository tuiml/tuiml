"""Test suite for PearsonUniversalKernel (PUK).

Tests cover:
- Evaluate returns values in (0, 1]
- Self-kernel equals 1.0
- Parameter schema
- Omega and sigma parameter behavior
- Kernel matrix properties
"""

import numpy as np
import pytest

from tuiml.algorithms.svm.kernels import PearsonUniversalKernel


class TestPUKInit:
    """Tests for PearsonUniversalKernel initialization."""

    def test_default_initialization(self):
        """Test default initialization."""
        k = PearsonUniversalKernel()

        assert k.omega == 1.0
        assert k.sigma == 1.0

    def test_custom_initialization(self):
        """Test custom initialization."""
        k = PearsonUniversalKernel(omega=2.0, sigma=0.5)

        assert k.omega == 2.0
        assert k.sigma == 0.5

    def test_parameter_schema(self):
        """Test parameter schema contains omega and sigma."""
        schema = PearsonUniversalKernel.get_parameter_schema()

        assert "omega" in schema
        assert "sigma" in schema
        assert schema["omega"]["type"] == "number"
        assert schema["sigma"]["type"] == "number"


class TestPUKBuild:
    """Tests for the build() method."""

    def test_build_basic(self):
        """Test building the kernel with training data."""
        np.random.seed(42)
        X = np.random.randn(10, 3)
        k = PearsonUniversalKernel(omega=1.0, sigma=1.0)
        k.build(X)

        assert k._is_built is True
        assert k.n_samples_ == 10
        assert k._factor is not None
        assert k._dot_precalc is not None

    def test_build_precomputes_factor(self):
        """Test that build precomputes the scaling factor."""
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        k = PearsonUniversalKernel(omega=1.0, sigma=1.0)
        k.build(X)

        expected_factor = 2.0 * np.sqrt(np.power(2.0, 1.0 / 1.0) - 1) / 1.0
        np.testing.assert_allclose(k._factor, expected_factor)


class TestPUKEvaluate:
    """Tests for the evaluate() method."""

    def test_self_kernel_is_one(self):
        """Test that K(x, x) = 1.0."""
        k = PearsonUniversalKernel(omega=1.0, sigma=1.0)
        x = np.array([1.0, 2.0, 3.0])

        np.testing.assert_allclose(k.evaluate(x, x), 1.0, atol=1e-10)

    def test_evaluate_in_0_1_range(self):
        """Test that evaluate returns values in (0, 1]."""
        k = PearsonUniversalKernel(omega=1.0, sigma=1.0)
        # Need to build so _factor is set
        X = np.array([[0.0, 0.0]])
        k.build(X)

        np.random.seed(42)
        for _ in range(20):
            x1 = np.random.randn(2)
            x2 = np.random.randn(2)
            val = k.evaluate(x1, x2)
            assert 0.0 < val <= 1.0 + 1e-10

    def test_evaluate_decreases_with_distance(self):
        """Test that kernel value decreases as distance increases."""
        k = PearsonUniversalKernel(omega=1.0, sigma=1.0)
        X = np.array([[0.0, 0.0]])
        k.build(X)

        x = np.array([0.0, 0.0])
        near = np.array([0.1, 0.1])
        far = np.array([10.0, 10.0])

        val_near = k.evaluate(x, near)
        val_far = k.evaluate(x, far)

        assert val_near > val_far

    def test_evaluate_symmetry(self):
        """Test that K(x, y) == K(y, x)."""
        k = PearsonUniversalKernel(omega=1.0, sigma=1.0)
        X = np.array([[0.0, 0.0]])
        k.build(X)

        np.random.seed(42)
        x1 = np.random.randn(2)
        x2 = np.random.randn(2)

        np.testing.assert_allclose(k.evaluate(x1, x2), k.evaluate(x2, x1), atol=1e-10)


class TestPUKCompute:
    """Tests for the compute() method."""

    def test_compute_self_is_one(self):
        """Test that compute(i, i) = 1.0."""
        np.random.seed(42)
        X = np.random.randn(5, 3)
        k = PearsonUniversalKernel(omega=1.0, sigma=1.0)
        k.build(X)

        for i in range(5):
            np.testing.assert_allclose(k.compute(i, i), 1.0, atol=1e-10)

    def test_compute_symmetric(self):
        """Test that compute(i, j) == compute(j, i)."""
        np.random.seed(42)
        X = np.random.randn(5, 3)
        k = PearsonUniversalKernel(omega=1.0, sigma=1.0)
        k.build(X)

        for i in range(5):
            for j in range(i + 1, 5):
                np.testing.assert_allclose(
                    k.compute(i, j), k.compute(j, i), atol=1e-10
                )


class TestPUKParameterEffects:
    """Tests for omega and sigma parameter effects."""

    def test_larger_sigma_wider_kernel(self):
        """Test that larger sigma produces a wider kernel response."""
        x = np.array([0.0, 0.0])
        y = np.array([2.0, 2.0])

        k_narrow = PearsonUniversalKernel(omega=1.0, sigma=0.5)
        k_wide = PearsonUniversalKernel(omega=1.0, sigma=5.0)

        # Build with dummy data to initialize factor
        dummy = np.array([[0.0, 0.0]])
        k_narrow.build(dummy)
        k_wide.build(dummy)

        val_narrow = k_narrow.evaluate(x, y)
        val_wide = k_wide.evaluate(x, y)

        # Wider sigma should give higher value at same distance
        assert val_wide > val_narrow

    def test_repr(self):
        """Test string representation."""
        k = PearsonUniversalKernel(omega=1.0, sigma=1.0)

        repr_str = repr(k)
        assert "PearsonUniversalKernel" in repr_str
        assert "omega=1.0" in repr_str
        assert "sigma=1.0" in repr_str
