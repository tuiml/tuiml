"""Base abstractions and foundation classes.

The ``tuiml.base`` module provides the object-oriented foundation for the 
entire library. It defines the core interfaces for algorithms, preprocessors, 
metrics, and experiment workflows.

Overview
--------
This module is organized into several key abstraction layers:

1. **Algorithms**: Base classes for supervised (Classifier, Regressor) 
   and unsupervised (Clusterer, Associator) models.
2. **Preprocessing**: Interfaces for data filters and feature transformers 
   (Preprocessor, Transformer).
3. **Features**: classes for feature selection and extraction.
4. **Metrics**: Performance assessment tools and evaluation metrics.
5. **Hyper-parameter Tuning**: Abstractions for parameter search and optimization.
6. **Experiments**: Framework for systematic model validation and benchmarking.
"""

# Algorithm base classes
from tuiml.base.algorithms import (
    Algorithm,
    Classifier,
    Clusterer,
    DensityBasedClusterer,
    UpdateableClusterer,
    Regressor,
    Associator,
    FrequentItemset,
    AssociationRule,
    AlgorithmRegistry,
    algorithm,
    classifier,
    clusterer,
    regressor,
    associator,
    get_algorithm,
    list_algorithms,
    search_algorithms,
)

# Preprocessing base classes
from tuiml.base.preprocessing import (
    Preprocessor,
    Filter,
    Transformer,
    SupervisedTransformer,
    InstanceTransformer,
    preprocessor,
    filter_method,
    transformer,
)

# Feature engineering base classes
from tuiml.base.features import (
    FeatureMethod,
    FeatureSelector,
    FeatureExtractor,
    FeatureConstructor,
    feature_selector,
    feature_extractor,
    feature_constructor,
)

# Evaluation base classes
from tuiml.base.metrics import (
    Metric,
    MetricType,
    AverageType,
    check_consistent_length,
    check_classification_targets,
    get_num_classes,
    get_class_labels,
    is_binary,
    weighted_sum,
    safe_divide,
)

from tuiml.base.splitting import BaseSplitter

from tuiml.base.tuning import (
    TuningResult,
    ParameterGrid,
    ParameterDistribution,
    BaseTuner,
)

from tuiml.base.experiments import (
    ExperimentType,
    ValidationMethod,
    ExperimentConfig,
    FoldResult,
    ModelResult,
    DatasetResult,
    ExperimentResults,
    BaseValidator,
    BaseExperiment,
)

# Dataset base classes
from tuiml.base.generators import (
    GeneratedData,
    DataGenerator,
    ClassificationGenerator,
    RegressionGenerator,
    ClusteringGenerator,
)

# Algorithm-specific base classes
from tuiml.base.kernels import (
    Kernel,
    CachedKernel,
    kernel,
)

from tuiml.base.neighbors import NearestNeighborSearch

from tuiml.base.estimators import Estimator

__all__ = [
    # Algorithms
    "Algorithm",
    "Classifier",
    "Clusterer",
    "DensityBasedClusterer",
    "UpdateableClusterer",
    "Regressor",
    "Associator",
    "FrequentItemset",
    "AssociationRule",
    "AlgorithmRegistry",
    "algorithm",
    "classifier",
    "clusterer",
    "regressor",
    "associator",
    "get_algorithm",
    "list_algorithms",
    "search_algorithms",
    # Preprocessing
    "Preprocessor",
    "Filter",
    "Transformer",
    "SupervisedTransformer",
    "InstanceTransformer",
    "preprocessor",
    "filter_method",
    "transformer",
    # Features
    "FeatureMethod",
    "FeatureSelector",
    "FeatureExtractor",
    "FeatureConstructor",
    "feature_selector",
    "feature_extractor",
    "feature_constructor",
    # Metrics
    "Metric",
    "MetricType",
    "AverageType",
    "check_consistent_length",
    "check_classification_targets",
    "get_num_classes",
    "get_class_labels",
    "is_binary",
    "weighted_sum",
    "safe_divide",
    # Splitting
    "BaseSplitter",
    # Tuning
    "TuningResult",
    "ParameterGrid",
    "ParameterDistribution",
    "BaseTuner",
    # Experiments
    "ExperimentType",
    "ValidationMethod",
    "ExperimentConfig",
    "FoldResult",
    "ModelResult",
    "DatasetResult",
    "ExperimentResults",
    "BaseValidator",
    "BaseExperiment",
    # Generators
    "GeneratedData",
    "DataGenerator",
    "ClassificationGenerator",
    "RegressionGenerator",
    "ClusteringGenerator",
    # Kernels
    "Kernel",
    "CachedKernel",
    "kernel",
    # Neighbors
    "NearestNeighborSearch",
    # Estimators
    "Estimator",
]
