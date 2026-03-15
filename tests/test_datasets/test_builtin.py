"""Tests for built-in datasets (tuiml.datasets.builtin)."""

import numpy as np
import pytest

from tuiml.datasets.builtin import (
    list_datasets,
    load_dataset,
    load_iris,
    load_diabetes,
    load_cpu,
    load_weather,
    load_glass,
    load_ionosphere,
    get_dataset_info,
    get_datasets_by_task,
    DATASET_REGISTRY,
)
from tuiml.datasets.loaders.arff import Dataset


# ===================================================================
# list_datasets tests
# ===================================================================

class TestListDatasets:
    """Tests for the list_datasets utility."""

    def test_list_all_datasets(self):
        """list_datasets() should return a non-empty sorted list."""
        datasets = list_datasets()
        assert isinstance(datasets, list)
        assert len(datasets) > 0
        assert datasets == sorted(datasets)

    def test_list_classification_datasets(self):
        """Listing classification datasets should include iris."""
        datasets = list_datasets("classification")
        assert "iris" in datasets

    def test_list_regression_datasets(self):
        """Listing regression datasets should include cpu."""
        datasets = list_datasets("regression")
        assert "cpu" in datasets

    def test_list_other_datasets(self):
        """Listing 'other' datasets should include supermarket."""
        datasets = list_datasets("other")
        assert "supermarket" in datasets

    def test_invalid_category_raises(self):
        """Invalid category should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown category"):
            list_datasets("nonexistent")


# ===================================================================
# load_dataset tests
# ===================================================================

class TestLoadDataset:
    """Tests for the load_dataset function."""

    def test_load_iris_by_name(self):
        """load_dataset('iris') should load the Iris dataset."""
        ds = load_dataset("iris")
        assert isinstance(ds, Dataset)
        assert ds.n_samples == 150
        assert ds.n_features == 4

    def test_load_cpu_by_name(self):
        """load_dataset('cpu') should load the CPU regression dataset."""
        ds = load_dataset("cpu")
        assert isinstance(ds, Dataset)
        assert ds.n_samples == 209

    def test_load_nonexistent_raises(self):
        """Loading a non-existent dataset should raise ValueError."""
        with pytest.raises(ValueError, match="not found"):
            load_dataset("totally_fake_dataset_xyz")


# ===================================================================
# Specific loader function tests
# ===================================================================

class TestSpecificLoaders:
    """Tests for individual load_* functions."""

    def test_load_iris(self):
        """load_iris should return correct shape and metadata."""
        ds = load_iris()
        assert ds.n_samples == 150
        assert ds.n_features == 4
        assert ds.y is not None
        assert ds.target_names is not None
        assert len(ds.target_names) == 3

    def test_load_iris_unpacking(self):
        """load_iris should support X, y = load_iris() unpacking."""
        X, y = load_iris()
        assert X.shape == (150, 4)
        assert y.shape == (150,)

    def test_load_diabetes(self):
        """load_diabetes should return 768 samples, 8 features."""
        ds = load_diabetes()
        assert ds.n_samples == 768
        assert ds.n_features == 8

    def test_load_glass(self):
        """load_glass should return 214 samples, 9 features."""
        ds = load_glass()
        assert ds.n_samples == 214
        assert ds.n_features == 9

    def test_load_ionosphere(self):
        """load_ionosphere should return 351 samples, 34 features."""
        ds = load_ionosphere()
        assert ds.n_samples == 351
        assert ds.n_features == 34

    def test_load_weather(self):
        """load_weather should return 14 samples."""
        ds = load_weather()
        assert ds.n_samples == 14
        assert ds.n_features == 4

    def test_load_cpu(self):
        """load_cpu should return 209 samples for regression."""
        ds = load_cpu()
        assert ds.n_samples == 209
        assert ds.n_features == 6


# ===================================================================
# Dataset registry tests
# ===================================================================

class TestDatasetRegistry:
    """Tests for dataset registry and metadata access."""

    def test_registry_is_dict(self):
        """DATASET_REGISTRY should be a dict."""
        assert isinstance(DATASET_REGISTRY, dict)

    def test_registry_contains_iris(self):
        """Registry should contain iris entry with correct metadata."""
        assert "iris" in DATASET_REGISTRY
        info = DATASET_REGISTRY["iris"]
        assert info["task"] == "classification"
        assert info["samples"] == 150
        assert info["features"] == 4
        assert info["classes"] == 3

    def test_get_dataset_info_single(self):
        """get_dataset_info('iris') should return iris metadata dict."""
        info = get_dataset_info("iris")
        assert isinstance(info, dict)
        assert info["task"] == "classification"

    def test_get_dataset_info_all(self):
        """get_dataset_info(None) should return full registry."""
        all_info = get_dataset_info()
        assert isinstance(all_info, dict)
        assert "iris" in all_info
        assert "cpu" in all_info

    def test_get_dataset_info_unknown_raises(self):
        """get_dataset_info with unknown name should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown dataset"):
            get_dataset_info("nonexistent_dataset")

    def test_get_datasets_by_task_classification(self):
        """Filtering by 'classification' should include iris and exclude cpu."""
        result = get_datasets_by_task("classification")
        assert "iris" in result
        assert "cpu" not in result

    def test_get_datasets_by_task_regression(self):
        """Filtering by 'regression' should include cpu and exclude iris."""
        result = get_datasets_by_task("regression")
        assert "cpu" in result
        assert "iris" not in result

    def test_registry_loader_field(self):
        """Each registry entry should have a 'loader' field."""
        for name, info in DATASET_REGISTRY.items():
            assert "loader" in info, f"'{name}' missing 'loader' field"
            assert info["loader"].startswith("load_")
