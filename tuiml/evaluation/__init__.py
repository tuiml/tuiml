"""
Model evaluation: metrics, resampling, significance tests, and visualization.

Everything needed to measure how well a model performs and to decide whether a
difference between two models is real. The subpackages are:

- ``metrics``: classification, regression, clustering, and information-theoretic
  scores.
- ``splitting``: resampling strategies (:class:`~tuiml.evaluation.splitting.KFold`,
  :func:`~tuiml.evaluation.splitting.train_test_split`, and friends).
- ``statistics``: significance tests for comparing models across datasets, plus
  multiple-comparison corrections.
- ``tuning``: hyperparameter search (grid, random, and Bayesian).
- ``visualization``: plots and critical-difference diagrams.

The most common metrics and splitters are re-exported here, so
``from tuiml.evaluation import accuracy_score`` works without naming the
subpackage.

Examples
--------
Score predictions:

>>> import numpy as np
>>> from tuiml.evaluation import accuracy_score
>>> y_true = np.array([0, 1, 1, 0])
>>> y_pred = np.array([0, 1, 0, 0])
>>> accuracy_score(y_true, y_pred)
0.75

Hold out a test set:

>>> from tuiml.datasets import load_iris
>>> from tuiml.evaluation import train_test_split
>>> X, y = load_iris()
>>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
>>> len(X_train), len(X_test)
(120, 30)

Cross-validate:

>>> from tuiml.evaluation import KFold
>>> cv = KFold(n_splits=5, shuffle=True, random_state=0)
>>> sum(1 for _ in cv.split(X))
5

Search hyperparameters:

>>> from tuiml.evaluation import GridSearchCV
>>> from tuiml.algorithms.bayesian import NaiveBayesClassifier
>>> search = GridSearchCV(NaiveBayesClassifier(),
...                       param_grid={'use_kernel_estimator': [True, False]})
>>> search = search.fit(X, y)                              # doctest: +SKIP
>>> search.best_params_                                    # doctest: +SKIP
{'use_kernel_estimator': False}
"""

# Import commonly used metrics at top level
from .metrics import (
    # Base
    Metric,
    MetricType,
    AverageType,
    # Classification
    confusion_matrix,
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    precision_recall_fscore_support,
    matthews_corrcoef,
    cohen_kappa_score,
    roc_auc_score,
    roc_curve,
    auc,
    log_loss,
    classification_report,
    # Regression
    mean_absolute_error,
    mean_squared_error,
    root_mean_squared_error,
    r2_score,
    # Clustering
    silhouette_score,
    adjusted_rand_score,
    # Information-theoretic
    entropy,
    mutual_information,
    information_gain,
)

# Import splitting utilities
from .splitting import (
    BaseSplitter,
    cross_val_score,
    KFold,
    StratifiedKFold,
    RepeatedKFold,
    RepeatedStratifiedKFold,
    train_test_split,
    HoldoutSplit,
    StratifiedHoldoutSplit,
    LeaveOneOut,
    LeavePOut,
    BootstrapSplit,
    TimeSeriesSplit,
    GroupKFold,
    StratifiedGroupKFold,
    ShuffleSplit,
    StratifiedShuffleSplit,
)

# Import statistical tests
from .statistics import (
    paired_t_test,
    corrected_paired_t_test,
    one_way_anova,
    wilcoxon_signed_rank_test,
    friedman_test,
    nemenyi_post_hoc,
    bonferroni_correction,
    holm_correction,
    benjamini_hochberg,
    SignificanceLevel,
    PairedStats,
)


# Import visualization (plots)
from .visualization import (
    plot_critical_difference,
    plot_ranking_table,
    plot_boxplot_comparison,
    plot_heatmap,
    plot_roc_curve,
    plot_pr_curve,
    plot_learning_curve,
    plot_confusion_matrix,
    compute_ranks,
    critical_difference,
)

# Import tuning
from .tuning import (
    BaseTuner,
    TuningResult,
    ParameterGrid,
    ParameterDistribution,
    GridSearchCV,
    RandomSearchCV,
)


__all__ = [
    # Submodules
    "metrics",
    "splitting",
    "statistics",
    "visualization",
    "tuning",
    # Common metrics
    "Metric",
    "MetricType",
    "AverageType",
    "confusion_matrix",
    "accuracy_score",
    "balanced_accuracy_score",
    "precision_score",
    "recall_score",
    "f1_score",
    "fbeta_score",
    "precision_recall_fscore_support",
    "matthews_corrcoef",
    "cohen_kappa_score",
    "roc_auc_score",
    "roc_curve",
    "auc",
    "log_loss",
    "classification_report",
    "mean_absolute_error",
    "mean_squared_error",
    "root_mean_squared_error",
    "r2_score",
    "silhouette_score",
    "adjusted_rand_score",
    "entropy",
    "mutual_information",
    "information_gain",
    # Splitting
    "BaseSplitter",
    "cross_val_score",
    "KFold",
    "StratifiedKFold",
    "RepeatedKFold",
    "RepeatedStratifiedKFold",
    "train_test_split",
    "HoldoutSplit",
    "StratifiedHoldoutSplit",
    "LeaveOneOut",
    "LeavePOut",
    "BootstrapSplit",
    "TimeSeriesSplit",
    "GroupKFold",
    "StratifiedGroupKFold",
    "ShuffleSplit",
    "StratifiedShuffleSplit",
    # Statistics
    "paired_t_test",
    "corrected_paired_t_test",
    "one_way_anova",
    "wilcoxon_signed_rank_test",
    "friedman_test",
    "nemenyi_post_hoc",
    "bonferroni_correction",
    "holm_correction",
    "benjamini_hochberg",
    "SignificanceLevel",
    "PairedStats",
    # Reporting
    # Visualization
    "plot_critical_difference",
    "plot_ranking_table",
    "plot_boxplot_comparison",
    "plot_heatmap",
    "plot_roc_curve",
    "plot_pr_curve",
    "plot_learning_curve",
    "plot_confusion_matrix",
    "compute_ranks",
    "critical_difference",
    # Tuning
    "BaseTuner",
    "TuningResult",
    "ParameterGrid",
    "ParameterDistribution",
    "GridSearchCV",
    "RandomSearchCV",
]
