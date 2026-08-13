"""Base abstractions and foundation classes.

The ``tuiml.base`` package provides the object-oriented foundation for the
entire library. It defines the core interfaces that concrete components
implement, along with the decorators (``@classifier``, ``@regressor``,
``@clusterer``, ``@associator``, ``@kernel``, ...) that register them with
the component registry.

Overview
--------
This package is organized into several key abstraction layers:

1. **Algorithms**: Base classes for supervised (``Classifier``, ``Regressor``)
   and unsupervised (``Clusterer``, ``Associator``) models, plus their
   registration decorators and registry lookup helpers (``get_algorithm``,
   ``list_algorithms``, ``search_algorithms``).
2. **Preprocessing**: Interfaces for data filters and feature transformers
   (``Preprocessor``, ``Transformer``).
3. **Features**: Base classes for feature selection, extraction, and
   construction.
4. **Metrics**: The ``Metric`` base class and shared validation/aggregation
   helpers used by ``tuiml.evaluation.metrics``.
5. **Splitting**: ``BaseSplitter``, the interface behind the train/test
   splitters in ``tuiml.evaluation.splitting``.
6. **Hyper-parameter Tuning**: Abstractions for parameter search
   (``BaseTuner``, ``ParameterGrid``, ``ParameterDistribution``).
7. **Generators**: Base classes for synthetic data generators
   (``DataGenerator`` and its task-specific subclasses).
8. **Kernels**: The ``Kernel`` base class and ``@kernel`` decorator for
   SVM-style kernel functions, including ``CachedKernel``.
9. **Neighbors and Estimators**: ``NearestNeighborSearch`` and probability
   ``ProbabilityEstimator`` primitives shared across algorithms.
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
    Transformer,
    SupervisedTransformer,
    ResamplingTransformer,
    preprocessor,
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

from tuiml.base.estimators import ProbabilityEstimator

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
    "Transformer",
    "SupervisedTransformer",
    "ResamplingTransformer",
    "preprocessor",
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
    "ProbabilityEstimator",
]
