"""
Evaluation module for TuiML.

This module provides:
- metrics: Classification, regression, clustering, and information-theoretic metrics
- splitting: Data splitting strategies (KFold, train_test_split, etc.)
- statistics: Statistical tests for model comparison
- visualization: Plots and diagrams for result visualization
- reporting: Output formatters (LaTeX, HTML, Markdown)
- tuning: Hyperparameter tuning (GridSearchCV, RandomSearchCV)
- experiments: Controlled experiments with statistical significance testing

Examples
--------
>>> from tuiml.evaluation import accuracy_score, f1_score
>>> from tuiml.evaluation.experiments import Experiment, run_experiment
>>> from tuiml.evaluation.splitting import KFold, train_test_split
>>> from tuiml.evaluation.tuning import GridSearchCV, RandomSearchCV
>>>
>>> # Quick metric computation
>>> accuracy = accuracy_score(y_true, y_pred)
>>>
>>> # Data splitting
>>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
>>>
>>> # Cross-validation
>>> kfold = KFold(n_splits=5, shuffle=True)
>>> for train_idx, test_idx in kfold.split(X):
...     pass
>>>
>>> # Hyperparameter tuning
>>> grid = GridSearchCV(estimator, param_grid={'C': [0.1, 1, 10]})
>>> grid.fit(X, y)
>>>
>>> # Full experiment
>>> exp = run_experiment(
...     models={'RF': RandomForestClassifier(), 'SVM': SVC()},
...     datasets={'data': (X, y)},
...     n_folds=10,
...     metrics=['accuracy_score', 'f1_score']
... )
>>> print(exp.to_latex())
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

# Import reporting (formatters)
from .reporting import (
    ResultMatrix,
    format_results,
    to_latex_table,
    to_html_table,
    to_markdown_table,
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

# Import experiment classes
from .experiments import (
    Experiment,
    run_experiment,
    ExperimentConfig,
    ExperimentResults,
    ExperimentType,
    ValidationMethod,
)

__all__ = [
    # Submodules
    "metrics",
    "splitting",
    "statistics",
    "visualization",
    "reporting",
    "tuning",
    "experiments",
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
    "ResultMatrix",
    "format_results",
    "to_latex_table",
    "to_html_table",
    "to_markdown_table",
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
    # Experiment classes
    "Experiment",
    "run_experiment",
    "ExperimentConfig",
    "ExperimentResults",
    "ExperimentType",
    "ValidationMethod",
]
