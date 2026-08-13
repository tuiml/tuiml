"""Scoring functions for every task type.

Around seventy metrics, all as plain functions taking ``(y_true, y_pred)``.
Anywhere TuiML accepts a ``metrics`` list — :func:`tuiml.train`,
:class:`tuiml.Benchmark`, the MCP tools — the names come from here.

Classification
--------------
**accuracy_score**, **balanced_accuracy_score**, **precision_score**,
**recall_score**, **f1_score**, **fbeta_score**, **roc_auc_score**,
**average_precision_score**, **log_loss**, **matthews_corrcoef**,
**cohen_kappa_score**, **confusion_matrix**, **classification_report**, plus
the rate family (**true_positive_rate**, **false_positive_rate**,
**specificity_score**, ...) and the curves (**roc_curve**,
**precision_recall_curve**).

Regression
----------
**mean_squared_error**, **root_mean_squared_error**,
**mean_absolute_error**, **r2_score**, **correlation_coefficient**, and the
relative errors (**relative_absolute_error**,
**root_relative_squared_error**).

Clustering
----------
**silhouette_score**, **davies_bouldin_score**, **calinski_harabasz_score**
for unlabelled data; **adjusted_rand_score**, **normalized_mutual_info_score**,
**homogeneity_score**, **completeness_score**, **v_measure_score** when true
labels are known.

Information theory
------------------
**entropy**, **information_gain**, **gain_ratio**, **mutual_information**,
**symmetrical_uncertainty**, **kullback_leibler_divergence** and friends,
used for splitting criteria and feature ranking as well as evaluation.

Notes
-----
On imbalanced data accuracy is misleading: predicting the majority class for
everything already scores well. Prefer **balanced_accuracy_score**,
**f1_score** or **matthews_corrcoef** there.

Multi-class variants take an ``average`` argument (``"macro"``, ``"micro"``,
``"weighted"``); ``"macro"`` weights every class equally, ``"weighted"`` by
class frequency.

Examples
--------
>>> from tuiml.evaluation.metrics import accuracy_score, f1_score
>>> y_true = [0, 1, 1, 0, 1]
>>> y_pred = [0, 1, 0, 0, 1]
>>> float(accuracy_score(y_true, y_pred))
0.8
"""

from tuiml.base.metrics import Metric, MetricType, AverageType

# Classification metrics
from .classification import (
    confusion_matrix,
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    precision_recall_fscore_support,
    true_positive_rate,
    false_positive_rate,
    true_negative_rate,
    false_negative_rate,
    sensitivity_score,
    specificity_score,
    num_true_positives,
    num_true_negatives,
    num_false_positives,
    num_false_negatives,
    matthews_corrcoef,
    cohen_kappa_score,
    roc_auc_score,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    log_loss,
    hamming_loss,
    zero_one_loss,
    classification_report
)

# Regression metrics
from .regression import (
    mean_absolute_error,
    mean_squared_error,
    root_mean_squared_error,
    r2_score,
    relative_absolute_error,
    root_relative_squared_error,
    correlation_coefficient
)

# Clustering metrics
from .clustering import (
    adjusted_rand_score,
    rand_score,
    silhouette_score,
    silhouette_samples,
    davies_bouldin_score,
    calinski_harabasz_score,
    mutual_info_score,
    normalized_mutual_info_score,
    v_measure_score,
    homogeneity_score,
    completeness_score,
    fowlkes_mallows_score
)

# Information-theoretic metrics
from .information_theoretic import (
    entropy,
    conditional_entropy,
    mutual_information,
    information_gain,
    gain_ratio,
    kullback_leibler_divergence,
    jensen_shannon_divergence,
    cross_entropy,
    symmetrical_uncertainty,
    prior_entropy,
    prediction_entropy,
    entropy_gain,
    kb_information,
)

# Feature scoring metrics
from .feature_scoring import (
    chi2,
    f_classif,
    f_regression,
    correlation,
    single_rule_score,
    relief_f
)

__all__ = [
    # Base
    'Metric', 'MetricType', 'AverageType',
    
    # Classification
    'confusion_matrix',
    'accuracy_score',
    'balanced_accuracy_score',
    'precision_score',
    'recall_score',
    'f1_score',
    'fbeta_score',
    'precision_recall_fscore_support',
    'true_positive_rate',
    'false_positive_rate', 
    'true_negative_rate',
    'false_negative_rate',
    'sensitivity_score',
    'specificity_score',
    'num_true_positives',
    'num_true_negatives',
    'num_false_positives',
    'num_false_negatives',
    'matthews_corrcoef',
    'cohen_kappa_score',
    'roc_auc_score',
    'roc_curve',
    'auc',
    'precision_recall_curve',
    'average_precision_score',
    'log_loss',
    'hamming_loss',
    'zero_one_loss',
    'classification_report',
    
    # Regression
    'mean_absolute_error',
    'mean_squared_error',
    'root_mean_squared_error',
    'r2_score',
    'relative_absolute_error',
    'root_relative_squared_error',
    'correlation_coefficient',
    
    # Clustering
    'adjusted_rand_score',
    'rand_score',
    'silhouette_score',
    'silhouette_samples',
    'davies_bouldin_score',
    'calinski_harabasz_score',
    'mutual_info_score',
    'normalized_mutual_info_score',
    'v_measure_score',
    'homogeneity_score',
    'completeness_score',
    'fowlkes_mallows_score',
    
    # Information-theoretic
    'entropy',
    'conditional_entropy',
    'mutual_information',
    'information_gain',
    'gain_ratio',
    'kullback_leibler_divergence',
    'jensen_shannon_divergence',
    'cross_entropy',
    'symmetrical_uncertainty',
    'prior_entropy',
    'prediction_entropy',
    'entropy_gain',
    'kb_information',

    # Feature scoring
    'chi2',
    'f_classif',
    'f_regression',
    'correlation',
    'single_rule_score',
    'relief_f'
]

