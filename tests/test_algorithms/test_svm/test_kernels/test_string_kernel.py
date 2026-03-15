"""Test suite for StringKernel (String Subsequence Kernel).

Tests cover:
- Initialization with default and custom parameters
- Parameter schema
- Build with string data
- Compute between training strings
- Self-similarity behavior
"""

import numpy as np
import pytest

from tuiml.algorithms.svm.kernels import StringKernel


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
