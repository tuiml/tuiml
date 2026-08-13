"""Dataset loaders for the supported file formats.

Merged from: test_loaders_arff.py, test_loaders_csv.py, test_loaders_numpy.py, test_loaders_auto.py
"""

import numpy as np
import pytest
from pathlib import Path
import tuiml
from tuiml.datasets.loaders.arff import load_arff, save_arff, Dataset, _parse_arff
from tuiml.datasets.loaders.csv import load_csv, save_csv
from tuiml.datasets.loaders.arff import Dataset
from tuiml.datasets.loaders.numpy import load_numpy, save_numpy
from tuiml.datasets.loaders.auto import load, save, LOADERS, SAVERS
from tuiml.datasets.loaders.arff import Dataset, save_arff


# --------------------------------------------------------------------------
# Tests for the ARFF file loader (tuiml.datasets.loaders.arff).
# --------------------------------------------------------------------------

_BUILTIN_DIR = (
    Path(tuiml.__file__).resolve().parent
    / "datasets" / "builtin" / "data" / "classification"
)


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

    def test_load_glass_arff(self):
        """Load glass.arff and verify shape."""
        path = _BUILTIN_DIR / "glass.arff"
        if not path.exists():
            pytest.skip("glass.arff not found")

        ds = load_arff(path)
        assert ds.n_samples == 214
        assert ds.n_features == 9
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


# --------------------------------------------------------------------------
# Tests for the CSV file loader (tuiml.datasets.loaders.csv).
# --------------------------------------------------------------------------

class TestLoadCsv:
    """Tests for load_csv with temporary CSV files."""

    def test_load_basic_csv(self, tmp_path):
        """Load a simple CSV with header and numeric data."""
        filepath = tmp_path / "basic.csv"
        filepath.write_text(
            "f1,f2,target\n"
            "1.0,2.0,0\n"
            "3.0,4.0,1\n"
            "5.0,6.0,0\n"
        )
        ds = load_csv(filepath)
        assert isinstance(ds, Dataset)
        assert ds.n_samples == 3
        assert ds.n_features == 2
        assert ds.y is not None
        assert len(ds.feature_names) == 2

    def test_load_csv_returns_correct_values(self, tmp_path):
        """Loaded values should match what was written."""
        filepath = tmp_path / "values.csv"
        filepath.write_text(
            "a,b,c\n"
            "1.5,2.5,3.5\n"
            "4.5,5.5,6.5\n"
        )
        ds = load_csv(filepath)
        np.testing.assert_array_almost_equal(ds.X, [[1.5, 2.5], [4.5, 5.5]])
        np.testing.assert_array_almost_equal(ds.y, [3.5, 6.5])

    def test_load_csv_no_target(self, tmp_path):
        """target_column=None should keep all columns as features."""
        filepath = tmp_path / "notarget.csv"
        filepath.write_text(
            "x,y,z\n"
            "1,2,3\n"
            "4,5,6\n"
        )
        ds = load_csv(filepath, target_column=None)
        assert ds.y is None
        assert ds.n_features == 3

    def test_load_csv_target_by_name(self, tmp_path):
        """Specify target column by string name."""
        filepath = tmp_path / "named_target.csv"
        filepath.write_text(
            "age,income,label\n"
            "25,50000,A\n"
            "30,60000,B\n"
            "35,70000,A\n"
        )
        ds = load_csv(filepath, target_column="label")
        assert ds.n_features == 2
        assert ds.y is not None
        assert "age" in ds.feature_names
        assert "income" in ds.feature_names

    def test_load_csv_categorical_target(self, tmp_path):
        """Categorical target should be encoded to integers."""
        filepath = tmp_path / "cat.csv"
        filepath.write_text(
            "x1,x2,species\n"
            "1.0,2.0,cat\n"
            "3.0,4.0,dog\n"
            "5.0,6.0,cat\n"
        )
        ds = load_csv(filepath)
        assert ds.y is not None
        # cat=0, dog=1 (sorted)
        np.testing.assert_array_equal(ds.y, [0, 1, 0])

    def test_load_csv_missing_values(self, tmp_path):
        """Missing values should become NaN."""
        filepath = tmp_path / "missing.csv"
        filepath.write_text(
            "x1,x2,y\n"
            "1.0,,0\n"
            "?,4.0,1\n"
        )
        ds = load_csv(filepath)
        assert np.isnan(ds.X[0, 1])
        assert np.isnan(ds.X[1, 0])

    def test_load_csv_custom_delimiter(self, tmp_path):
        """Custom delimiter (tab) should work."""
        filepath = tmp_path / "tab.csv"
        filepath.write_text(
            "x1\tx2\ty\n"
            "1.0\t2.0\t0\n"
            "3.0\t4.0\t1\n"
        )
        ds = load_csv(filepath, delimiter='\t')
        assert ds.n_samples == 2
        assert ds.n_features == 2

    def test_load_csv_no_header(self, tmp_path):
        """CSV without header should use generic column names."""
        filepath = tmp_path / "noheader.csv"
        filepath.write_text(
            "1.0,2.0,0\n"
            "3.0,4.0,1\n"
        )
        ds = load_csv(filepath, header=False)
        assert ds.n_samples == 2
        assert ds.n_features == 2
        assert ds.feature_names[0].startswith("col")

    def test_load_csv_empty_file_raises(self, tmp_path):
        """Loading an empty CSV should raise ValueError."""
        filepath = tmp_path / "empty.csv"
        filepath.write_text("")
        with pytest.raises(ValueError, match="Empty file"):
            load_csv(filepath)

    def test_load_csv_invalid_target_name_raises(self, tmp_path):
        """Non-existent target column name should raise ValueError."""
        filepath = tmp_path / "data.csv"
        filepath.write_text(
            "a,b,c\n"
            "1,2,3\n"
        )
        with pytest.raises(ValueError, match="not found"):
            load_csv(filepath, target_column="nonexistent")

    def test_dataset_name_from_stem(self, tmp_path):
        """Dataset name should be derived from the CSV filename stem."""
        filepath = tmp_path / "my_data.csv"
        filepath.write_text("x,y\n1,2\n")
        ds = load_csv(filepath, target_column=None)
        assert ds.name == "my_data"


class TestSaveCsv:
    """Tests for save_csv round-trip."""

    def test_save_and_reload(self, tmp_path):
        """Saving and reloading CSV should preserve numeric data."""
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        filepath = tmp_path / "roundtrip.csv"

        save_csv(filepath, X, feature_names=["a", "b"])
        ds = load_csv(filepath, target_column=None)

        np.testing.assert_array_almost_equal(ds.X, X)
        assert ds.feature_names == ["a", "b"]

    def test_save_with_target(self, tmp_path):
        """Saving with a target should round-trip correctly."""
        X = np.array([[10.0, 20.0], [30.0, 40.0]])
        y = np.array([0.0, 1.0])
        filepath = tmp_path / "with_y.csv"

        save_csv(filepath, X, feature_names=["x1", "x2"],
                 target=y, target_name="label")
        ds = load_csv(filepath, target_column="label")

        np.testing.assert_array_almost_equal(ds.X, X)
        np.testing.assert_array_almost_equal(ds.y, y)

    def test_save_default_feature_names(self, tmp_path):
        """Default feature names should be col0, col1, etc."""
        X = np.random.rand(3, 2)
        filepath = tmp_path / "defaults.csv"

        save_csv(filepath, X)
        ds = load_csv(filepath, target_column=None)

        assert ds.feature_names[0] == "col0"
        assert ds.feature_names[1] == "col1"


# --------------------------------------------------------------------------
# Tests for the NumPy format loader (tuiml.datasets.loaders.numpy).
# --------------------------------------------------------------------------

class TestLoadNumpy:
    """Tests for load_numpy with .npy and .npz files."""

    def test_load_npy(self, tmp_path):
        """Load a .npy file containing a 2-D feature array."""
        X = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        filepath = tmp_path / "data.npy"
        np.save(filepath, X)

        ds = load_numpy(filepath)
        assert isinstance(ds, Dataset)
        assert ds.n_samples == 2
        assert ds.n_features == 3
        assert ds.y is None
        np.testing.assert_array_almost_equal(ds.X, X)

    def test_load_npz_with_X_and_y(self, tmp_path):
        """Load a .npz file with both X and y arrays."""
        X = np.array([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]])
        y = np.array([0, 1, 2])
        filepath = tmp_path / "data.npz"
        np.savez(filepath, X=X, y=y)

        ds = load_numpy(filepath)
        assert ds.n_samples == 3
        assert ds.n_features == 2
        np.testing.assert_array_equal(ds.y, y)
        np.testing.assert_array_almost_equal(ds.X, X)

    def test_load_npz_X_only(self, tmp_path):
        """Load a .npz file with only X (no y key)."""
        X = np.random.rand(5, 4)
        filepath = tmp_path / "xonly.npz"
        np.savez(filepath, X=X)

        ds = load_numpy(filepath)
        assert ds.n_samples == 5
        assert ds.n_features == 4
        assert ds.y is None

    def test_load_npz_custom_keys(self, tmp_path):
        """Load .npz with custom key names for data and target."""
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        y = np.array([10, 20])
        filepath = tmp_path / "custom.npz"
        np.savez(filepath, features=X, labels=y)

        ds = load_numpy(filepath, data_key="features", target_key="labels")
        np.testing.assert_array_almost_equal(ds.X, X)
        np.testing.assert_array_equal(ds.y, y)

    def test_feature_names_auto_generated(self, tmp_path):
        """Feature names should be auto-generated as feat0, feat1, etc."""
        X = np.random.rand(3, 5)
        filepath = tmp_path / "data.npy"
        np.save(filepath, X)

        ds = load_numpy(filepath)
        assert len(ds.feature_names) == 5
        assert ds.feature_names[0] == "feat0"
        assert ds.feature_names[4] == "feat4"

    def test_dataset_name_from_filename(self, tmp_path):
        """Dataset name should be derived from the file stem."""
        X = np.ones((2, 2))
        filepath = tmp_path / "my_experiment.npy"
        np.save(filepath, X)

        ds = load_numpy(filepath)
        assert ds.name == "my_experiment"


class TestSaveNumpy:
    """Tests for save_numpy round-trip."""

    def test_save_npy_data_only(self, tmp_path):
        """Saving X-only should produce a .npy that loads correctly."""
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        filepath = tmp_path / "out.npy"
        save_numpy(filepath, X)

        loaded = np.load(filepath)
        np.testing.assert_array_almost_equal(loaded, X)

    def test_save_npz_with_target(self, tmp_path):
        """Saving with target should produce .npz with X and y keys."""
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        y = np.array([0, 1])
        filepath = tmp_path / "out.npz"
        save_numpy(filepath, X, target=y)

        archive = np.load(filepath)
        np.testing.assert_array_almost_equal(archive["X"], X)
        np.testing.assert_array_equal(archive["y"], y)

    def test_save_with_feature_names(self, tmp_path):
        """Saving with feature_names should store them in .npz."""
        X = np.random.rand(3, 2)
        filepath = tmp_path / "named.npz"
        save_numpy(filepath, X, feature_names=["alpha", "beta"])

        archive = np.load(filepath)
        assert "feature_names" in archive
        np.testing.assert_array_equal(archive["feature_names"], ["alpha", "beta"])

    def test_roundtrip_npz(self, tmp_path):
        """Full save/load roundtrip for .npz format."""
        X = np.random.rand(20, 5)
        y = np.random.randint(0, 3, 20)
        filepath = tmp_path / "roundtrip.npz"

        save_numpy(filepath, X, target=y, feature_names=["f0", "f1", "f2", "f3", "f4"])
        ds = load_numpy(filepath)

        np.testing.assert_array_almost_equal(ds.X, X)
        np.testing.assert_array_equal(ds.y, y)
        assert ds.n_features == 5

    def test_save_compressed(self, tmp_path):
        """Compressed .npz should be smaller or equal in size."""
        X = np.ones((100, 10))  # Highly compressible
        fp_compressed = tmp_path / "compressed.npz"
        fp_uncompressed = tmp_path / "uncompressed.npz"

        save_numpy(fp_compressed, X, target=np.zeros(100), compressed=True)
        save_numpy(fp_uncompressed, X, target=np.zeros(100), compressed=False)

        assert fp_compressed.stat().st_size <= fp_uncompressed.stat().st_size


# --------------------------------------------------------------------------
# Tests for the auto-detect loader (tuiml.datasets.loaders.auto).
# --------------------------------------------------------------------------

class TestAutoLoad:
    """Tests for auto-format detection via load()."""

    def test_load_csv_auto(self, tmp_path):
        """Auto-loader should detect .csv extension and load correctly."""
        filepath = tmp_path / "data.csv"
        filepath.write_text(
            "x1,x2,y\n"
            "1.0,2.0,0\n"
            "3.0,4.0,1\n"
        )
        ds = load(filepath)
        assert isinstance(ds, Dataset)
        assert ds.n_samples == 2
        assert ds.n_features == 2

    def test_load_arff_auto(self, tmp_path):
        """Auto-loader should detect .arff extension and load correctly."""
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        filepath = tmp_path / "data.arff"
        save_arff(filepath, X, feature_names=["a", "b"], relation="test")

        ds = load(filepath, target_column=None)
        assert isinstance(ds, Dataset)
        assert ds.n_samples == 2

    def test_load_npy_auto(self, tmp_path):
        """Auto-loader should detect .npy extension and load correctly."""
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        filepath = tmp_path / "data.npy"
        np.save(filepath, X)

        ds = load(filepath)
        assert ds.n_samples == 3
        assert ds.n_features == 2

    def test_load_npz_auto(self, tmp_path):
        """Auto-loader should detect .npz extension and load correctly."""
        X = np.array([[10.0, 20.0], [30.0, 40.0]])
        y = np.array([0, 1])
        filepath = tmp_path / "data.npz"
        np.savez(filepath, X=X, y=y)

        ds = load(filepath)
        assert ds.n_samples == 2
        assert ds.y is not None
        np.testing.assert_array_equal(ds.y, y)

    def test_load_unsupported_extension_raises(self, tmp_path):
        """Unsupported file extension should raise ValueError."""
        filepath = tmp_path / "data.xyz"
        filepath.write_text("dummy")
        with pytest.raises(ValueError, match="Unsupported file format"):
            load(filepath)

    def test_loaders_dict_has_expected_keys(self):
        """LOADERS dict should contain at least .arff, .csv, .npy, .npz."""
        assert ".arff" in LOADERS
        assert ".csv" in LOADERS
        assert ".npy" in LOADERS
        assert ".npz" in LOADERS
        assert ".json" in LOADERS

    def test_savers_dict_has_expected_keys(self):
        """SAVERS dict should contain at least .arff, .csv, .npy."""
        assert ".arff" in SAVERS
        assert ".csv" in SAVERS
        assert ".npy" in SAVERS


class TestAutoSave:
    """Tests for auto-format detection via save()."""

    def test_save_csv_auto(self, tmp_path):
        """save() should detect .csv and write a valid CSV file."""
        X = np.array([[1.0, 2.0], [3.0, 4.0]])
        filepath = tmp_path / "out.csv"
        save(filepath, X, feature_names=["a", "b"])

        ds = load(filepath, target_column=None)
        np.testing.assert_array_almost_equal(ds.X, X)

    def test_save_arff_auto(self, tmp_path):
        """save() should detect .arff and write a valid ARFF file."""
        X = np.array([[5.0, 6.0], [7.0, 8.0]])
        filepath = tmp_path / "out.arff"
        save(filepath, X, feature_names=["x1", "x2"])

        ds = load(filepath, target_column=None)
        np.testing.assert_array_almost_equal(ds.X, X)

    def test_save_unsupported_extension_raises(self, tmp_path):
        """Unsupported save extension should raise ValueError."""
        X = np.ones((2, 2))
        filepath = tmp_path / "data.unknown"
        with pytest.raises(ValueError, match="Unsupported file format"):
            save(filepath, X)

    def test_roundtrip_csv(self, tmp_path):
        """Full save/load roundtrip for CSV."""
        X = np.random.rand(10, 3)
        y = np.random.randint(0, 2, 10).astype(float)
        filepath = tmp_path / "roundtrip.csv"

        save(filepath, X, target=y, feature_names=["f1", "f2", "f3"])
        ds = load(filepath)

        assert ds.n_samples == 10
        assert ds.n_features == 3
        assert ds.y is not None
