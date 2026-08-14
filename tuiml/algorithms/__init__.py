"""Every learning algorithm TuiML implements.

Grouped into thirteen families by what the algorithm *does*, so a wrapped
scikit-learn or CapyMOA estimator sits in the same place as its native
counterpart. Every class here is registered in the hub under its own name,
which is how :func:`tuiml.train` and the MCP tools reach it — you name the
algorithm, you do not import it.

Supervised learning
-------------------
- :mod:`~tuiml.algorithms.trees` — decision trees and forests. The strongest
  default on tabular data: ``RandomForestClassifier``,
  ``DecisionTreeClassifier``, ``DecisionStumpClassifier``.
- :mod:`~tuiml.algorithms.gradient_boosting` — boosted ensembles wrapping
  XGBoost, LightGBM and CatBoost. Usually the accuracy ceiling, at the cost
  of tuning.
- :mod:`~tuiml.algorithms.ensemble` — meta-learners that combine other
  algorithms: ``BaggingClassifier``, ``AdaBoostClassifier``,
  ``StackingClassifier``, ``VotingClassifier``.
- :mod:`~tuiml.algorithms.linear` — the interpretable baseline:
  ``LinearRegression``, ``LogisticRegression``, ``SGDClassifier``.
- :mod:`~tuiml.algorithms.svm` — maximum-margin classifiers with pluggable
  kernels: ``SVC``, ``SVR``.
- :mod:`~tuiml.algorithms.bayesian` — probabilistic models:
  ``NaiveBayesClassifier``, ``NaiveBayesMultinomialClassifier``,
  ``GaussianProcessesRegressor``.
- :mod:`~tuiml.algorithms.neighbors` — instance-based learning, which stores
  the training data instead of building a model:
  ``KNearestNeighborsClassifier``, ``KNearestNeighborsRegressor``.
- :mod:`~tuiml.algorithms.rules` — ``ZeroRuleClassifier``, the majority-class
  baseline every other model has to beat.
- :mod:`~tuiml.algorithms.neural` — ``MultilayerPerceptronClassifier``.

Unsupervised learning
---------------------
- :mod:`~tuiml.algorithms.clustering` — grouping without labels:
  ``KMeansClusterer``, ``DBSCANClusterer``, ``AgglomerativeClusterer``,
  ``GaussianMixtureClusterer``.
- :mod:`~tuiml.algorithms.anomaly` — finding the rare rows:
  ``IsolationForestDetector``, ``LocalOutlierFactorDetector``.
- :mod:`~tuiml.algorithms.associations` — co-occurrence rules:
  ``AprioriAssociator``, ``FPGrowthAssociator``.

Sequential data
---------------
- :mod:`~tuiml.algorithms.timeseries` — forecasting, where order matters and
  ordinary cross-validation leaks the future into the past: ``ARIMA``,
  ``ExponentialSmoothing``, ``STLDecomposition``.

Notes
-----
Names are addressed through the hub, and the namespace is what disambiguates
them: ``"RandomForestClassifier"`` is the native implementation,
``"sklearn.RandomForestClassifier"`` the wrapped one. Both can appear in the
same benchmark.

Examples
--------
>>> import tuiml
>>> model = tuiml.train({
...     "model": {"name": "RandomForestClassifier",
...               "params": {"n_estimators": 50}},
...     "data": {"source": "iris", "target": "class"},
...     "evaluation": {"cv": 5, "metrics": ["accuracy_score"]},
... })
>>> round(model.metrics_["cv_accuracy_score_mean"], 1)
1.0

See Also
--------
:mod:`tuiml.sklearn` : scikit-learn estimators, under ``sklearn.*`` names.
:mod:`tuiml.capymoa` : CapyMOA streaming learners, under ``capymoa.*`` names.
:mod:`tuiml.weka` : Weka learners, under ``weka.*`` names.
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
    CategoricalNBClassifier,
    BayesianLinearRegressor,
    NaiveBayesMultinomialClassifier,
    GaussianProcessesRegressor,
)

# Tree-based algorithms
from tuiml.algorithms.trees import (
    DecisionStumpClassifier,
    DecisionTreeClassifier,
    DecisionTreeRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)

# Neighbor-based algorithms
from tuiml.algorithms.neighbors import (
    KNearestNeighborsClassifier,
    KNearestNeighborsRegressor,
)

# Linear algorithms
from tuiml.algorithms.linear import (
    LogisticRegression,
    LinearRegression,
    SGDClassifier,
    SGDRegressor,
)

# SVM algorithms
from tuiml.algorithms.svm import (
    SVC,
    SVR,
)

# Neural networks
from tuiml.algorithms.neural import (
    MultilayerPerceptronClassifier,
    MultilayerPerceptronRegressor,
    PerceptronClassifier,
)

# Rule-based algorithms
from tuiml.algorithms.rules import (
    ZeroRuleClassifier,
)

# Ensemble (meta-learners)
from tuiml.algorithms.ensemble import (
    BaggingClassifier,
    BaggingRegressor,
    AdaBoostClassifier,
    AdaBoostRegressor,
    VotingClassifier,
    StackingClassifier,
    StackingRegressor,
    VotingRegressor,
    GradientBoostingRegressor,
    OneVsRestClassifier,
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
    AgglomerativeClusterer,
    DBSCANClusterer,
    GaussianMixtureClusterer,
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
    ECLATAssociator,
)

# Anomaly detection
from tuiml.algorithms.anomaly import (
    IsolationForestDetector,
    LocalOutlierFactorDetector,
    EllipticEnvelopeDetector,
    OneClassSVMDetector,
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

# Interpretable-by-design models
from tuiml.algorithms.glassbox import (
    ExplainableBoostingClassifier,
    ExplainableBoostingRegressor,
    RuleFitClassifier,
    RuleFitRegressor,
)

# Survival analysis
from tuiml.algorithms.survival import (
    CoxPHSurvival,
    KaplanMeierEstimator,
    NelsonAalenEstimator,
    RandomSurvivalForest,
)

# Uplift / heterogeneous treatment effects
from tuiml.algorithms.causal import (
    SLearner,
    TLearner,
    UpliftTreeClassifier,
    XLearner,
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
    "CategoricalNBClassifier",
    "BayesianLinearRegressor",
    "NaiveBayesMultinomialClassifier",
    "GaussianProcessesRegressor",
    # Trees
    "DecisionStumpClassifier",
    "DecisionTreeClassifier",
    "DecisionTreeRegressor",
    "RandomForestClassifier",
    "RandomForestRegressor",
    # Neighbors
    "KNearestNeighborsClassifier",
    "KNearestNeighborsRegressor",
    # Linear
    "LogisticRegression",
    "LinearRegression",
    "SGDClassifier",
    "SGDRegressor",
    # SVM
    "SVC",
    "SVR",
    # Neural
    "MultilayerPerceptronClassifier",
    "MultilayerPerceptronRegressor",
    "PerceptronClassifier",
    # Rules
    "ZeroRuleClassifier",
    # Ensemble (meta-learners)
    "BaggingClassifier",
    "BaggingRegressor",
    "AdaBoostClassifier",
    "AdaBoostRegressor",
    "VotingClassifier",
    "StackingClassifier",
    "StackingRegressor",
    "VotingRegressor",
    "GradientBoostingRegressor",
    "OneVsRestClassifier",
    # Gradient Boosting (external frameworks)
    "XGBoostClassifier",
    "XGBoostRegressor",
    "CatBoostClassifier",
    "CatBoostRegressor",
    "LightGBMClassifier",
    "LightGBMRegressor",
    # Clustering
    "KMeansClusterer",
    "AgglomerativeClusterer",
    "DBSCANClusterer",
    "GaussianMixtureClusterer",
    # Distance utilities
    "euclidean_distance",
    "manhattan_distance",
    "cosine_distance",
    "pairwise_distances",
    # Associations
    "AprioriAssociator",
    "FPGrowthAssociator",
    "ECLATAssociator",
    # Anomaly detection
    "IsolationForestDetector",
    "LocalOutlierFactorDetector",
    "EllipticEnvelopeDetector",
    "OneClassSVMDetector",
    # Time series
    "AR",
    "MA",
    "ARMA",
    "ARIMA",
    "ExponentialSmoothing",
    "STLDecomposition",
    "Prophet",

    "ExplainableBoostingClassifier",
    "ExplainableBoostingRegressor",
    "RuleFitClassifier",
    "RuleFitRegressor",

    "KaplanMeierEstimator",
    "NelsonAalenEstimator",
    "CoxPHSurvival",
    "RandomSurvivalForest",

    "SLearner",
    "TLearner",
    "XLearner",
    "UpliftTreeClassifier",
]
