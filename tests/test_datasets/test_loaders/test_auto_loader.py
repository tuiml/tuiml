"""Tests for the auto-detect loader (tuiml.datasets.loaders.auto)."""

import numpy as np
import pytest
from pathlib import Path

from tuiml.datasets.loaders.auto import load, save, LOADERS, SAVERS
from tuiml.datasets.loaders.arff import Dataset, save_arff
from tuiml.datasets.loaders.csv import save_csv


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
