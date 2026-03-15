"""Tests for the CSV file loader (tuiml.datasets.loaders.csv)."""

import numpy as np
import pytest

from tuiml.datasets.loaders.csv import load_csv, save_csv
from tuiml.datasets.loaders.arff import Dataset


# ===================================================================
# load_csv tests
# ===================================================================

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


# ===================================================================
# save_csv tests
# ===================================================================

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
