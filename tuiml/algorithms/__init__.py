"""
Machine Learning Algorithms.

Organized by algorithm family:
- bayesian: Probabilistic models (NaiveBayesClassifier, BayesianNetworkClassifier, GaussianProcessesRegressor)
- trees: Decision trees and forests (C45TreeClassifier, RandomForestClassifier, M5ModelTreeRegressor)
- neighbors: Instance-based learning (KNearestNeighborsClassifier, KStarClassifier, LocallyWeightedLearningRegressor)
- linear: Linear models (LogisticRegression, LinearRegression, SGDClassifier)
- svm: Support Vector Machines (SMO, SMOreg)
- neural: Neural networks (MultilayerPerceptronClassifier)
- rules: Rule-based classifiers/regressors (ZeroRuleClassifier, OneRuleClassifier, RIPPERClassifier, PARTClassifier, M5ModelRulesRegressor)
- ensemble: WEKA-style meta-learners (BaggingClassifier, AdaBoostClassifier, StackingClassifier, VotingClassifier)
- gradient_boosting: External frameworks (XGBoostClassifier, CatBoostClassifier, LightGBMClassifier)
- clustering: Clustering algorithms (KMeansClusterer, DBSCANClusterer, AgglomerativeClusterer, GaussianMixtureClusterer)
- associations: Association rule mining (AprioriAssociator, FPGrowthAssociator)
- anomaly: Anomaly detection (IsolationForestDetector, LocalOutlierFactorDetector, EllipticEnvelopeDetector)
- timeseries: Time series analysis (ARIMA, ExponentialSmoothing, STLDecomposition)
"""

# Base classes (single source of truth)
from tuiml.base.algorithms import (
    # Core base classes
    Algorithm,
    Classifier,
    Regressor,
    Clusterer,
    DensityBasedClusterer,
    UpdateableClusterer,
    Associator,
    # Data structures
    FrequentItemset,
    AssociationRule,
    # Decorators (with hub registration)
    classifier,
    regressor,
    clusterer,
    associator,
)

# Bayesian algorithms
from tuiml.algorithms.bayesian import (
    NaiveBayesClassifier,
    NaiveBayesMultinomialClassifier,
    BayesianNetworkClassifier,
    GaussianProcessesRegressor,
)

# Tree-based algorithms
from tuiml.algorithms.trees import (
    DecisionStumpClassifier,
    C45TreeClassifier,
    RandomTreeClassifier,
    RandomForestClassifier,
    RandomForestRegressor,
    ReducedErrorPruningTreeClassifier,
    HoeffdingTreeClassifier,
    M5ModelTreeRegressor,
    LogisticModelTreeClassifier,
)

# Neighbor-based algorithms
from tuiml.algorithms.neighbors import (
    KNearestNeighborsClassifier,
    KNearestNeighborsRegressor,
    KStarClassifier,
    LocallyWeightedLearningRegressor,
)

# Linear algorithms
from tuiml.algorithms.linear import (
    LogisticRegression,
    LinearRegression,
    SimpleLinearRegression,
    SGDClassifier,
    SGDRegressor,
    SimpleLogisticRegression,
)

# SVM algorithms
from tuiml.algorithms.svm import (
    SVC,
    SVR,
)

# Neural networks
from tuiml.algorithms.neural import (
    MultilayerPerceptronClassifier,
    VotedPerceptronClassifier,
)

# Rule-based algorithms
from tuiml.algorithms.rules import (
    ZeroRuleClassifier,
    OneRuleClassifier,
    RIPPERClassifier,
    PARTClassifier,
    M5ModelRulesRegressor,
    DecisionTableClassifier,
)

# Ensemble (WEKA-style meta-learners)
from tuiml.algorithms.ensemble import (
    BaggingClassifier,
    AdaBoostClassifier,
    VotingClassifier,
    StackingClassifier,
    AdditiveRegression,
    RegressionByDiscretization,
    LogitBoostClassifier,
    RandomCommitteeClassifier,
    RandomSubspaceClassifier,
    MultiClassClassifier,
    FilteredClassifier,
)

# Gradient Boosting (external frameworks)
from tuiml.algorithms.gradient_boosting import (
    XGBoostClassifier,
    XGBoostRegressor,
    CatBoostClassifier,
    CatBoostRegressor,
    LightGBMClassifier,
    LightGBMRegressor,
)

# Clustering algorithms
from tuiml.algorithms.clustering import (
    KMeansClusterer,
    FarthestFirstClusterer,
    AgglomerativeClusterer,
    DBSCANClusterer,
    GaussianMixtureClusterer,
    CanopyClusterer,
    CobwebClusterer,
    FilteredClusterer,
)

# Distance functions (from clustering/distance)
from tuiml.algorithms.clustering.distance import (
    euclidean_distance,
    manhattan_distance,
    cosine_distance,
    pairwise_distances,
)

# Association rule mining
from tuiml.algorithms.associations import (
    AprioriAssociator,
    FPGrowthAssociator,
)

# Anomaly detection
from tuiml.algorithms.anomaly import (
    IsolationForestDetector,
    LocalOutlierFactorDetector,
    EllipticEnvelopeDetector,
    OneClassSVMDetector,
    ABODDetector,
)

# Time series analysis
from tuiml.algorithms.timeseries import (
    AR,
    MA,
    ARMA,
    ARIMA,
    ExponentialSmoothing,
    STLDecomposition,
    Prophet,
)

__all__ = [
    # Base classes
    "Algorithm",
    "Classifier",
    "Regressor",
    "Clusterer",
    "DensityBasedClusterer",
    "UpdateableClusterer",
    "Associator",
    # Data structures
    "FrequentItemset",
    "AssociationRule",
    # Decorators
    "classifier",
    "regressor",
    "clusterer",
    "associator",
    # Bayesian
    "NaiveBayesClassifier",
    "NaiveBayesMultinomialClassifier",
    "BayesianNetworkClassifier",
    "GaussianProcessesRegressor",
    # Trees
    "DecisionStumpClassifier",
    "C45TreeClassifier",
    "RandomTreeClassifier",
    "RandomForestClassifier",
    "RandomForestRegressor",
    "ReducedErrorPruningTreeClassifier",
    "HoeffdingTreeClassifier",
    "M5ModelTreeRegressor",
    "LogisticModelTreeClassifier",
    # Neighbors
    "KNearestNeighborsClassifier",
    "KNearestNeighborsRegressor",
    "KStarClassifier",
    "LocallyWeightedLearningRegressor",
    # Linear
    "LogisticRegression",
    "LinearRegression",
    "SimpleLinearRegression",
    "SGDClassifier",
    "SGDRegressor",
    "SimpleLogisticRegression",
    # SVM
    "SVC",
    "SVR",
    # Neural
    "MultilayerPerceptronClassifier",
    "VotedPerceptronClassifier",
    # Rules
    "ZeroRuleClassifier",
    "OneRuleClassifier",
    "RIPPERClassifier",
    "PARTClassifier",
    "M5ModelRulesRegressor",
    "DecisionTableClassifier",
    # Ensemble (WEKA-style meta-learners)
    "BaggingClassifier",
    "AdaBoostClassifier",
    "VotingClassifier",
    "StackingClassifier",
    "AdditiveRegression",
    "RegressionByDiscretization",
    "LogitBoostClassifier",
    "RandomCommitteeClassifier",
    "RandomSubspaceClassifier",
    "MultiClassClassifier",
    "FilteredClassifier",
    # Gradient Boosting (external frameworks)
    "XGBoostClassifier",
    "XGBoostRegressor",
    "CatBoostClassifier",
    "CatBoostRegressor",
    "LightGBMClassifier",
    "LightGBMRegressor",
    # Clustering
    "KMeansClusterer",
    "FarthestFirstClusterer",
    "AgglomerativeClusterer",
    "DBSCANClusterer",
    "GaussianMixtureClusterer",
    "CanopyClusterer",
    "CobwebClusterer",
    "FilteredClusterer",
    # Distance utilities
    "euclidean_distance",
    "manhattan_distance",
    "cosine_distance",
    "pairwise_distances",
    # Associations
    "AprioriAssociator",
    "FPGrowthAssociator",
    # Anomaly detection
    "IsolationForestDetector",
    "LocalOutlierFactorDetector",
    "EllipticEnvelopeDetector",
    "OneClassSVMDetector",
    "ABODDetector",
    # Time series
    "AR",
    "MA",
    "ARMA",
    "ARIMA",
    "ExponentialSmoothing",
    "STLDecomposition",
    "Prophet",
]
