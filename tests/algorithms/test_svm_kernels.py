"""SVM kernel functions.

Merged from: test_kernels_linear.py, test_kernels_polynomial.py, test_kernels_rbf.py, test_kernels_sigmoid.py, test_kernels_puk.py, test_kernels_string.py, test_kernels_precomputed.py
"""

import numpy as np
import pytest
from tuiml.algorithms.svm.kernels import LinearKernel
from tuiml.algorithms.svm.kernels import PolynomialKernel
from tuiml.algorithms.svm.kernels import RBFKernel
from tuiml.algorithms.svm.kernels import SigmoidKernel
from tuiml.algorithms.svm.kernels import PearsonUniversalKernel
from tuiml.algorithms.svm.kernels import StringKernel
from tuiml.algorithms.svm.kernels import PrecomputedKernel


# --------------------------------------------------------------------------
# Test suite for LinearKernel.
# --------------------------------------------------------------------------

class TestLinearKernelInit:
    """Tests for LinearKernel initialization."""

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


# --------------------------------------------------------------------------
# Test suite for PolynomialKernel.
# --------------------------------------------------------------------------

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


# --------------------------------------------------------------------------
# Test suite for RBFKernel (Radial Basis Function / Gaussian kernel).
# --------------------------------------------------------------------------

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


# --------------------------------------------------------------------------
# Test suite for SigmoidKernel (Hyperbolic Tangent kernel).
# --------------------------------------------------------------------------

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


# --------------------------------------------------------------------------
# Test suite for PearsonUniversalKernel (PUK).
# --------------------------------------------------------------------------

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


# --------------------------------------------------------------------------
# Test suite for StringKernel (String Subsequence Kernel).
# --------------------------------------------------------------------------

class TestStringKernelInit:
    """Tests for StringKernel initialization."""

    def test_default_initialization(self):
        """Test default initialization."""
        k = StringKernel()

        assert k.subsequence_length == 3
        assert k.lambda_decay == 0.5
        assert k.normalize is True

    def test_custom_initialization(self):
        """Test custom initialization."""
        k = StringKernel(subsequence_length=5, lambda_decay=0.8, normalize=False)

        assert k.subsequence_length == 5
        assert k.lambda_decay == 0.8
        assert k.normalize is False

    def test_parameter_schema(self):
        """Test parameter schema."""
        schema = StringKernel.get_parameter_schema()

        assert "subsequence_length" in schema
        assert "lambda_decay" in schema
        assert "normalize" in schema
        assert schema["subsequence_length"]["type"] == "integer"
        assert schema["lambda_decay"]["type"] == "number"


class TestStringKernelBuild:
    """Tests for the build() method."""

    def test_build_with_string_list(self):
        """Test building from a list of strings."""
        k = StringKernel(subsequence_length=2)
        k.build(["hello", "world", "test"])

        assert k._is_built is True
        assert k.n_samples_ == 3

    def test_build_with_numpy_array(self):
        """Test building from a numpy array of strings."""
        k = StringKernel(subsequence_length=2)
        X = np.array(["hello", "world", "test"])
        k.build(X)

        assert k._is_built is True
        assert k.n_samples_ == 3


class TestStringKernelCompute:
    """Tests for the compute() method."""

    def test_self_similarity_normalized(self):
        """Test that normalized self-similarity is 1.0."""
        k = StringKernel(subsequence_length=2, normalize=True)
        k.build(["hello world", "foo bar"])

        val = k.compute(0, 0)
        np.testing.assert_allclose(val, 1.0, atol=1e-10)

    def test_compute_symmetric(self):
        """Test that K(i, j) == K(j, i)."""
        k = StringKernel(subsequence_length=2)
        k.build(["hello", "world", "test string"])

        val_01 = k.compute(0, 1)
        val_10 = k.compute(1, 0)

        np.testing.assert_allclose(val_01, val_10, atol=1e-10)

    def test_similar_strings_higher_kernel(self):
        """Test that similar strings have higher kernel value than dissimilar ones."""
        k = StringKernel(subsequence_length=2, normalize=True)
        k.build(["hello world", "hello there", "xyz abc"])

        val_similar = k.compute(0, 1)       # "hello world" vs "hello there"
        val_dissimilar = k.compute(0, 2)     # "hello world" vs "xyz abc"

        assert val_similar >= val_dissimilar

    def test_compute_non_negative_normalized(self):
        """Test that normalized kernel values are non-negative."""
        k = StringKernel(subsequence_length=2, normalize=True)
        k.build(["abc", "def", "ghi"])

        for i in range(3):
            for j in range(3):
                assert k.compute(i, j) >= -1e-10


class TestStringKernelEvaluate:
    """Tests for the evaluate() method."""

    def test_evaluate_with_string_inputs(self):
        """Test evaluate with direct string inputs."""
        k = StringKernel(subsequence_length=2, normalize=True)
        k.build(["dummy"])

        val = k.evaluate("hello", "hello")
        np.testing.assert_allclose(val, 1.0, atol=1e-10)

    def test_evaluate_empty_strings(self):
        """Test evaluate with empty strings."""
        k = StringKernel(subsequence_length=2, normalize=False)
        k.build(["dummy"])

        val = k.evaluate("", "hello")
        assert val == 0.0


class TestStringKernelRepr:
    """Tests for string representation."""

    def test_repr(self):
        """Test string representation."""
        k = StringKernel(subsequence_length=3, lambda_decay=0.5)

        repr_str = repr(k)
        assert "StringKernel" in repr_str
        assert "subsequence_length=3" in repr_str


# --------------------------------------------------------------------------
# Test suite for PrecomputedKernel.
# --------------------------------------------------------------------------

class TestPrecomputedKernelInit:
    """Tests for PrecomputedKernel initialization."""

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
