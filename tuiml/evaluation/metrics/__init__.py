"""
Evaluation metrics for TuiML.

Complete implementation of all Weka evaluation metrics plus sklearn-style APIs.
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
    scheme_entropy,
    entropy_gain,
    kb_information,
    mean_kb_information
)

# Feature scoring metrics
from .feature_scoring import (
    chi2,
    f_classif,
    f_regression,
    correlation,
    oner_score,
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
    'scheme_entropy',
    'entropy_gain',
    'kb_information',
    'mean_kb_information',

    # Feature scoring
    'chi2',
    'f_classif',
    'f_regression',
    'correlation',
    'oner_score',
    'relief_f'
]

