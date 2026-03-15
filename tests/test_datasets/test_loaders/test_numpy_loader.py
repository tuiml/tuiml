"""Tests for the NumPy format loader (tuiml.datasets.loaders.numpy)."""

import numpy as np
import pytest

from tuiml.datasets.loaders.numpy import load_numpy, save_numpy
from tuiml.datasets.loaders.arff import Dataset


# ===================================================================
# load_numpy tests
# ===================================================================

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


# ===================================================================
# save_numpy tests
# ===================================================================

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
