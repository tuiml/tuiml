"""Tests for the ARFF file loader (tuiml.datasets.loaders.arff)."""

import numpy as np
import pytest
from pathlib import Path

from tuiml.datasets.loaders.arff import load_arff, save_arff, Dataset, _parse_arff


# ---------------------------------------------------------------------------
# Path to builtin ARFF datasets used for integration-style tests
# ---------------------------------------------------------------------------
_BUILTIN_DIR = (
    Path(__file__).resolve().parents[3]
    / "tuiml"
    / "datasets"
    / "builtin"
    / "classification"
)


# ===================================================================
# Dataset dataclass tests
# ===================================================================

class TestDataset:
    """Tests for the Dataset dataclass container."""

    def test_basic_properties(self):
        """Dataset should expose n_samples, n_features, and shape."""
        X = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
        y = np.array([0, 1, 0])
        ds = Dataset(X=X, y=y, feature_names=["a", "b"])

        assert ds.n_samples == 3
        assert ds.n_features == 2
        assert ds.shape == (3, 2)

    def test_unpacking(self):
        """Dataset should support `X, y = dataset` unpacking."""
        X = np.ones((5, 3))
        y = np.zeros(5)
        ds = Dataset(X=X, y=y)

        X_out, y_out = ds
        np.testing.assert_array_equal(X_out, X)
        np.testing.assert_array_equal(y_out, y)

    def test_repr(self):
        """Dataset repr should contain name, n_samples, n_features."""
        ds = Dataset(X=np.zeros((10, 4)), name="test_ds")
        r = repr(ds)
        assert "test_ds" in r
        assert "10" in r
        assert "4" in r

    def test_default_values(self):
        """Dataset defaults should be sensible."""
        ds = Dataset(X=np.zeros((2, 2)))
        assert ds.y is None
        assert ds.feature_names == []
        assert ds.target_names is None
        assert ds.name == "dataset"
        assert ds.description == ""


# ===================================================================
# load_arff tests
# ===================================================================

class TestLoadArff:
    """Tests for load_arff on builtin ARFF files."""

    def test_load_iris(self):
        """Load iris.arff and verify shape and metadata."""
        path = _BUILTIN_DIR / "iris.arff"
        if not path.exists():
            pytest.skip("iris.arff not found in builtin datasets")

        ds = load_arff(path)
        assert ds.n_samples == 150
        assert ds.n_features == 4
        assert ds.y is not None
        assert len(ds.feature_names) == 4
        assert ds.target_names is not None
        assert len(ds.target_names) == 3

    def test_load_weather_numeric(self):
        """Load weather.numeric.arff and verify shape."""
        path = _BUILTIN_DIR / "weather.numeric.arff"
        if not path.exists():
            pytest.skip("weather.numeric.arff not found")

        ds = load_arff(path)
        assert ds.n_samples == 14
        assert ds.n_features == 4
        assert ds.y is not None

    def test_load_no_target(self):
        """Load ARFF with target_column=None should have y=None."""
        path = _BUILTIN_DIR / "iris.arff"
        if not path.exists():
            pytest.skip("iris.arff not found")

        ds = load_arff(path, target_column=None)
        assert ds.y is None
        # All columns become features
        assert ds.n_features == 5

    def test_load_target_column_zero(self):
        """Specifying target_column=0 should use the first column as target."""
        path = _BUILTIN_DIR / "iris.arff"
        if not path.exists():
            pytest.skip("iris.arff not found")

        ds = load_arff(path, target_column=0)
        assert ds.y is not None
        assert ds.n_features == 4  # Still 4 features (one removed, one added from class)

    def test_dataset_name_from_filename(self):
        """Dataset name should be derived from the ARFF @relation or filename stem."""
        path = _BUILTIN_DIR / "iris.arff"
        if not path.exists():
            pytest.skip("iris.arff not found")

        ds = load_arff(path)
        # The relation name in iris.arff is typically "iris"
        assert ds.name != ""


# ===================================================================
# save_arff tests
# ===================================================================

class TestSaveArff:
    """Tests for save_arff round-trip."""

    def test_save_and_reload_numeric(self, tmp_path):
        """Saving and reloading numeric-only data should preserve values."""
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        filepath = tmp_path / "test.arff"

        save_arff(filepath, X, feature_names=["f1", "f2"], relation="test_rel")
        ds = load_arff(filepath, target_column=None)

        np.testing.assert_array_almost_equal(ds.X, X)
        assert ds.name == "test_rel"

    def test_save_with_target(self, tmp_path):
        """Saving with a target array should produce correct round-trip."""
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        y = np.array([0, 1])
        filepath = tmp_path / "with_target.arff"

        save_arff(filepath, X, feature_names=["a", "b"],
                  target=y, target_names=["cat", "dog"], relation="animals")

        ds = load_arff(filepath, target_column=-1)
        assert ds.n_samples == 2
        assert ds.n_features == 2
        assert ds.y is not None
        assert ds.target_names == ["cat", "dog"]

    def test_save_default_feature_names(self, tmp_path):
        """When feature_names is None, generic names should be generated."""
        X = np.random.rand(5, 3)
        filepath = tmp_path / "defaults.arff"

        save_arff(filepath, X)
        ds = load_arff(filepath, target_column=None)

        assert ds.n_features == 3
        assert "attr0" in ds.feature_names[0]


# ===================================================================
# _parse_arff internal tests
# ===================================================================

class TestParseArff:
    """Tests for the internal _parse_arff function."""

    def test_parse_simple_arff(self):
        """Parse a minimal ARFF string."""
        content = """\
@relation simple

@attribute x1 numeric
@attribute x2 numeric
@attribute class {a,b}

@data
1.0,2.0,a
3.0,4.0,b
"""
        ds = _parse_arff(content, target_column=-1, name="fallback")
        assert ds.n_samples == 2
        assert ds.n_features == 2
        assert ds.name == "simple"
        assert ds.target_names == ["a", "b"]
        np.testing.assert_array_equal(ds.y, [0, 1])

    def test_parse_with_comments(self):
        """Comments (% lines) should be collected in description."""
        content = """\
% This is a comment
% Second comment
@relation commented

@attribute x numeric

@data
1.0
2.0
"""
        ds = _parse_arff(content, target_column=None, name="fallback")
        assert "This is a comment" in ds.description
        assert "Second comment" in ds.description

    def test_parse_missing_values(self):
        """Missing values (?) should become NaN."""
        content = """\
@relation missing

@attribute x1 numeric
@attribute x2 numeric

@data
1.0,2.0
?,4.0
3.0,?
"""
        ds = _parse_arff(content, target_column=None, name="test")
        assert np.isnan(ds.X[1, 0])
        assert np.isnan(ds.X[2, 1])
        assert ds.X[0, 0] == 1.0

    def test_parse_sparse_format(self):
        """Sparse data format should be handled correctly."""
        content = """\
@relation sparse

@attribute x1 numeric
@attribute x2 numeric
@attribute x3 numeric

@data
{0 1.0, 2 3.0}
{1 5.0}
"""
        ds = _parse_arff(content, target_column=None, name="sparse")
        assert ds.n_samples == 2
        assert ds.X[0, 0] == 1.0
        assert ds.X[0, 1] == 0.0
        assert ds.X[0, 2] == 3.0
        assert ds.X[1, 0] == 0.0
        assert ds.X[1, 1] == 5.0
        assert ds.X[1, 2] == 0.0
