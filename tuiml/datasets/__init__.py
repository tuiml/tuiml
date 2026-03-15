"""Datasets module for TuiML - Loading and generation of ML data.

The datasets module provides a unified interface for accessing classic 
machine learning datasets, loading data from various file formats, and 
generating synthetic data for benchmarking and experimentation.

Overview
--------
This module is organized into three primary areas:

1. **Built-in Datasets:** Easy access to standard datasets like Iris, 
   Diabetes, and CPU performance.
2. **Loaders:** High-performance loaders for ARFF, CSV, Excel, Parquet, 
   and JSON formats.
3. **Generators:** Synthetic data generation for classification, 
   regression, and clustering.

Examples
--------
Loading a built-in dataset:

>>> from tuiml.datasets import load_iris
>>> data = load_iris()
>>> X, y = data.X, data.y

Loading from a file with auto-detection:

>>> from tuiml.datasets import load
>>> data = load('experiment_results.csv')

Generating synthetic data:

>>> from tuiml.datasets import Blobs
>>> data = Blobs(n_samples=1000).generate()
"""

# =============================================================================
# File Format Loaders
# =============================================================================
from tuiml.datasets.loaders import (
    # Auto-detect
    load,
    save,
    # ARFF
    load_arff,
    save_arff,
    # CSV
    load_csv,
    save_csv,
    # NumPy
    load_numpy,
    save_numpy,
    # Pandas
    load_pandas,
    to_pandas,
    from_pandas,
    # Excel
    load_excel,
    save_excel,
    load_excel_sheets,
    # Parquet
    load_parquet,
    save_parquet,
    load_parquet_partitioned,
    # JSON
    load_json,
    save_json,
    load_jsonl,
    save_jsonl,
    load_json_nested,
    # Dataset class
    Dataset,
)

# =============================================================================
# Built-in Datasets
# =============================================================================
from tuiml.datasets.builtin import (
    # LLM-friendly metadata
    DATASET_REGISTRY,
    get_dataset_info,
    get_datasets_by_task,
    # Utilities
    list_datasets,
    load_dataset,
    # Classification datasets
    load_iris,
    load_iris_2d,
    load_diabetes,
    load_breast_cancer,
    load_glass,
    load_ionosphere,
    load_vote,
    load_credit,
    load_weather,
    load_weather_nominal,
    load_soybean,
    load_labor,
    load_contact_lenses,
    load_hypothyroid,
    load_segment,
    load_segment_test,
    load_unbalanced,
    # Regression datasets
    load_cpu,
    load_cpu_with_vendor,
    load_airline,
    # Other datasets
    load_supermarket,
    load_reuters_corn,
    load_reuters_grain,
)

# =============================================================================
# Synthetic Data Generators
# =============================================================================
from tuiml.datasets.generators import (
    # Base classes
    DataGenerator,
    ClassificationGenerator,
    RegressionGenerator,
    ClusteringGenerator,
    GeneratedData,
    # Classification generators
    RandomRBF,
    Agrawal,
    LED,
    Hyperplane,
    # Regression generators
    Friedman,
    MexicanHat,
    Sine,
    # Clustering generators
    Blobs,
    Moons,
    Circles,
    SwissRoll,
)

__all__ = [
    # ==========================================================================
    # File Loaders
    # ==========================================================================
    "load",
    "save",
    "load_arff",
    "save_arff",
    "load_csv",
    "save_csv",
    "load_numpy",
    "save_numpy",
    "load_pandas",
    "to_pandas",
    "from_pandas",
    "load_excel",
    "save_excel",
    "load_excel_sheets",
    "load_parquet",
    "save_parquet",
    "load_parquet_partitioned",
    "load_json",
    "save_json",
    "load_jsonl",
    "save_jsonl",
    "load_json_nested",
    "Dataset",
    # ==========================================================================
    # Built-in Datasets
    # ==========================================================================
    # LLM-friendly metadata
    "DATASET_REGISTRY",
    "get_dataset_info",
    "get_datasets_by_task",
    # Utilities
    "list_datasets",
    "load_dataset",
    # Classification
    "load_iris",
    "load_iris_2d",
    "load_diabetes",
    "load_breast_cancer",
    "load_glass",
    "load_ionosphere",
    "load_vote",
    "load_credit",
    "load_weather",
    "load_weather_nominal",
    "load_soybean",
    "load_labor",
    "load_contact_lenses",
    "load_hypothyroid",
    "load_segment",
    "load_segment_test",
    "load_unbalanced",
    # Regression
    "load_cpu",
    "load_cpu_with_vendor",
    "load_airline",
    # Other
    "load_supermarket",
    "load_reuters_corn",
    "load_reuters_grain",
    # ==========================================================================
    # Generators
    # ==========================================================================
    "DataGenerator",
    "ClassificationGenerator",
    "RegressionGenerator",
    "ClusteringGenerator",
    "GeneratedData",
    # Classification
    "RandomRBF",
    "Agrawal",
    "LED",
    "Hyperplane",
    # Regression
    "Friedman",
    "MexicanHat",
    "Sine",
    # Clustering
    "Blobs",
    "Moons",
    "Circles",
    "SwissRoll",
]
