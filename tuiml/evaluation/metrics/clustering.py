"""
Clustering evaluation metrics.

Complete implementations of clustering quality metrics.
"""

from typing import Optional, Union
import numpy as np
from tuiml.base.metrics import safe_divide, check_consistent_length

def _contingency_matrix(labels_true: np.ndarray, labels_pred: np.ndarray) -> np.ndarray:
    """Build a contingency matrix (confusion matrix for clustering)."""
    classes_true = np.unique(labels_true)
    classes_pred = np.unique(labels_pred)
    
    contingency = np.zeros((len(classes_true), len(classes_pred)), dtype=np.int64)
    
    for i, c_true in enumerate(classes_true):
        for j, c_pred in enumerate(classes_pred):
            contingency[i, j] = np.sum((labels_true == c_true) & (labels_pred == c_pred))
    
    return contingency

def adjusted_rand_score(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    """
    Compute Adjusted Rand Index (ARI).
    
    The ARI measures the similarity between two clusterings, adjusted for chance.
    
    Args:
        labels_true: Ground truth cluster labels
        labels_pred: Predicted cluster labels
        
    Returns:
        ARI score in range [-1, 1], where 1 is perfect agreement
        
    Example:
        >>> adjusted_rand_score([0, 0, 1, 1], [0, 0, 1, 1])
        1.0
        >>> adjusted_rand_score([0, 0, 1, 1], [0, 1, 0, 1])
        0.0
    """
    labels_true = np.asarray(labels_true)
    labels_pred = np.asarray(labels_pred)
    check_consistent_length(labels_true, labels_pred)
    
    # Get contingency matrix
    contingency = _contingency_matrix(labels_true, labels_pred)
    
    # Sum over rows and columns
    sum_comb_c = np.sum(_comb2(np.sum(contingency, axis=1)))
    sum_comb_k = np.sum(_comb2(np.sum(contingency, axis=0)))
    sum_comb = np.sum(_comb2(contingency.ravel()))
    
    n_samples = len(labels_true)
    prod_comb = (sum_comb_c * sum_comb_k) / _comb2(n_samples) if n_samples > 1 else 0
    mean_comb = (sum_comb_c + sum_comb_k) / 2.0
    
    if mean_comb == prod_comb:
        return 1.0 if sum_comb == mean_comb else 0.0
    
    return float((sum_comb - prod_comb) / (mean_comb - prod_comb))

def rand_score(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    """
    Compute Rand Index (RI).
    
    Args:
        labels_true: Ground truth cluster labels
        labels_pred: Predicted cluster labels
        
    Returns:
        RI score in range [0, 1]
    """
    labels_true = np.asarray(labels_true)
    labels_pred = np.asarray(labels_pred)
    
    n_samples = len(labels_true)
    contingency = _contingency_matrix(labels_true, labels_pred)
    
    # Compute TP + TN
    sum_comb = np.sum(_comb2(contingency.ravel()))
    sum_comb_c = np.sum(_comb2(np.sum(contingency, axis=1)))
    sum_comb_k = np.sum(_comb2(np.sum(contingency, axis=0)))
    
    # TP = sum of combinations in each cell
    # TN = total_comb - TP - FP - FN
    total_comb = _comb2(n_samples)
    
    return float(sum_comb / total_comb) if total_comb > 0 else 1.0

def _comb2(n: Union[int, np.ndarray]) -> Union[int, np.ndarray]:
    """Compute binomial coefficient n choose 2."""
    if isinstance(n, np.ndarray):
        return n * (n - 1) / 2
    return int(n * (n - 1) / 2)

def silhouette_score(
    X: np.ndarray,
    labels: np.ndarray,
    metric: str = 'euclidean'
) -> float:
    """
    Compute mean Silhouette Coefficient.
    
    The Silhouette Coefficient is calculated using the mean intra-cluster 
    distance (a) and the mean nearest-cluster distance (b) for each sample.
    
    Args:
        X: Feature matrix, shape (n_samples, n_features)
        labels: Cluster labels for each sample
        metric: Distance metric ('euclidean', 'manhattan', 'cosine')
        
    Returns:
        Mean silhouette score in range [-1, 1]
        
    Example:
        >>> X = np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]])
        >>> labels = np.array([0, 0, 0, 1, 1, 1])
        >>> silhouette_score(X, labels)  # doctest: +SKIP
        0.55...
    """
    X = np.asarray(X)
    labels = np.asarray(labels)
    
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    
    if n_clusters == 1 or n_clusters == len(labels):
        return 0.0
    
    # Compute pairwise distances
    distances = _pairwise_distances(X, metric)
    
    silhouette_vals = []
    
    for i, label in enumerate(labels):
        # Same cluster mask
        same_cluster_mask = labels == label
        n_same = np.sum(same_cluster_mask)
        
        if n_same == 1:
            # Singleton cluster
            silhouette_vals.append(0.0)
            continue
        
        # a: mean distance to points in same cluster
        a = np.sum(distances[i, same_cluster_mask]) / (n_same - 1)
        
        # b: mean distance to points in nearest other cluster
        b = np.inf
        for other_label in unique_labels:
            if other_label == label:
                continue
            
            other_cluster_mask = labels == other_label
            if np.sum(other_cluster_mask) > 0:
                mean_dist = np.mean(distances[i, other_cluster_mask])
                b = min(b, mean_dist)
        
        # Silhouette coefficient
        s = (b - a) / max(a, b) if max(a, b) > 0 else 0.0
        silhouette_vals.append(s)
    
    return float(np.mean(silhouette_vals))

def silhouette_samples(
    X: np.ndarray,
    labels: np.ndarray,
    metric: str = 'euclidean'
) -> np.ndarray:
    """
    Compute Silhouette Coefficient for each sample.
    
    Args:
        X: Feature matrix
        labels: Cluster labels
        metric: Distance metric
        
    Returns:
        Silhouette scores for each sample
    """
    X = np.asarray(X)
    labels = np.asarray(labels)
    
    unique_labels = np.unique(labels)
    distances = _pairwise_distances(X, metric)
    
    silhouette_vals = np.zeros(len(labels))
    
    for i, label in enumerate(labels):
        same_cluster_mask = labels == label
        n_same = np.sum(same_cluster_mask)
        
        if n_same == 1:
            silhouette_vals[i] = 0.0
            continue
        
        a = np.sum(distances[i, same_cluster_mask]) / (n_same - 1)
        
        b = np.inf
        for other_label in unique_labels:
            if other_label == label:
                continue
            other_cluster_mask = labels == other_label
            if np.sum(other_cluster_mask) > 0:
                mean_dist = np.mean(distances[i, other_cluster_mask])
                b = min(b, mean_dist)
        
        silhouette_vals[i] = (b - a) / max(a, b) if max(a, b) > 0 else 0.0
    
    return silhouette_vals

def davies_bouldin_score(X: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute Davies-Bouldin Index.
    
    Lower values indicate better clustering (minimum is 0).
    
    Args:
        X: Feature matrix
        labels: Cluster labels
        
    Returns:
        Davies-Bouldin score (lower is better)
        
    Example:
        >>> X = np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]])
        >>> labels = np.array([0, 0, 0, 1, 1, 1])
        >>> davies_bouldin_score(X, labels)  # doctest: +SKIP
        0.6...
    """
    X = np.asarray(X)
    labels = np.asarray(labels)
    
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    
    if n_clusters == 1:
        return 0.0
    
    # Compute cluster centers
    centers = np.array([np.mean(X[labels == label], axis=0) for label in unique_labels])
    
    # Compute average within-cluster distances
    avg_within = np.zeros(n_clusters)
    for i, label in enumerate(unique_labels):
        cluster_points = X[labels == label]
        avg_within[i] = np.mean(np.linalg.norm(cluster_points - centers[i], axis=1))
    
    # Compute Davies-Bouldin index
    db_values = []
    
    for i in range(n_clusters):
        max_ratio = 0.0
        for j in range(n_clusters):
            if i != j:
                between_dist = np.linalg.norm(centers[i] - centers[j])
                if between_dist > 0:
                    ratio = (avg_within[i] + avg_within[j]) / between_dist
                    max_ratio = max(max_ratio, ratio)
        db_values.append(max_ratio)
    
    return float(np.mean(db_values))

def calinski_harabasz_score(X: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute Calinski-Harabasz Index (Variance Ratio Criterion).
    
    Higher values indicate better clustering.
    
    Args:
        X: Feature matrix
        labels: Cluster labels
        
    Returns:
        Calinski-Harabasz score (higher is better)
        
    Example:
        >>> X = np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]])
        >>> labels = np.array([0, 0, 0, 1, 1, 1])
        >>> calinski_harabasz_score(X, labels)  # doctest: +SKIP
        5.0
    """
    X = np.asarray(X)
    labels = np.asarray(labels)
    
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels)
    n_samples = len(X)
    
    if n_clusters == 1 or n_clusters == n_samples:
        return 0.0
    
    # Overall mean
    mean_overall = np.mean(X, axis=0)
    
    # Between-cluster dispersion
    ssb = 0.0
    for label in unique_labels:
        cluster_points = X[labels == label]
        n_cluster = len(cluster_points)
        mean_cluster = np.mean(cluster_points, axis=0)
        ssb += n_cluster * np.sum((mean_cluster - mean_overall) ** 2)
    
    # Within-cluster dispersion
    ssw = 0.0
    for label in unique_labels:
        cluster_points = X[labels == label]
        mean_cluster = np.mean(cluster_points, axis=0)
        ssw += np.sum((cluster_points - mean_cluster) ** 2)
    
    # Calinski-Harabasz score
    if ssw == 0:
        return 0.0
    
    ch_score = (ssb / (n_clusters - 1)) / (ssw / (n_samples - n_clusters))
    return float(ch_score)

def mutual_info_score(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    """
    Compute Mutual Information between two clusterings.
    
    Args:
        labels_true: Ground truth cluster labels
        labels_pred: Predicted cluster labels
        
    Returns:
        Mutual information score
    """
    labels_true = np.asarray(labels_true)
    labels_pred = np.asarray(labels_pred)
    
    contingency = _contingency_matrix(labels_true, labels_pred)
    
    # Normalize
    n_samples = len(labels_true)
    contingency = contingency / n_samples
    
    # Marginals
    pi = np.sum(contingency, axis=1)
    pj = np.sum(contingency, axis=0)
    
    # Mutual information
    mi = 0.0
    for i in range(len(pi)):
        for j in range(len(pj)):
            if contingency[i, j] > 0:
                mi += contingency[i, j] * np.log(contingency[i, j] / (pi[i] * pj[j]))
    
    return float(mi)

def normalized_mutual_info_score(
    labels_true: np.ndarray,
    labels_pred: np.ndarray,
    average_method: str = 'arithmetic'
) -> float:
    """
    Compute Normalized Mutual Information (NMI).
    
    Args:
        labels_true: Ground truth cluster labels
        labels_pred: Predicted cluster labels
        average_method: 'arithmetic', 'geometric', 'min', 'max'
        
    Returns:
        NMI score in range [0, 1]
    """
    mi = mutual_info_score(labels_true, labels_pred)
    
    # Compute entropies
    h_true = _entropy(labels_true)
    h_pred = _entropy(labels_pred)
    
    if average_method == 'arithmetic':
        normalizer = (h_true + h_pred) / 2.0
    elif average_method == 'geometric':
        normalizer = np.sqrt(h_true * h_pred)
    elif average_method == 'min':
        normalizer = min(h_true, h_pred)
    elif average_method == 'max':
        normalizer = max(h_true, h_pred)
    else:
        raise ValueError(f"Unknown average_method: {average_method}")
    
    if normalizer == 0:
        return 0.0
    
    return float(mi / normalizer)

def _entropy(labels: np.ndarray) -> float:
    """Compute entropy of a label distribution."""
    _, counts = np.unique(labels, return_counts=True)
    probs = counts / len(labels)
    return float(-np.sum(probs * np.log(probs + 1e-10)))

def _pairwise_distances(X: np.ndarray, metric: str = 'euclidean') -> np.ndarray:
    """
    Compute pairwise distances between samples.
    
    Args:
        X: Feature matrix
        metric: Distance metric
        
    Returns:
        Distance matrix
    """
    n_samples = len(X)
    distances = np.zeros((n_samples, n_samples))
    
    if metric == 'euclidean':
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                dist = np.linalg.norm(X[i] - X[j])
                distances[i, j] = dist
                distances[j, i] = dist
    
    elif metric == 'manhattan':
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                dist = np.sum(np.abs(X[i] - X[j]))
                distances[i, j] = dist
                distances[j, i] = dist
    
    elif metric == 'cosine':
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                norm_i = np.linalg.norm(X[i])
                norm_j = np.linalg.norm(X[j])
                if norm_i > 0 and norm_j > 0:
                    cos_sim = np.dot(X[i], X[j]) / (norm_i * norm_j)
                    dist = 1 - cos_sim
                else:
                    dist = 1.0
                distances[i, j] = dist
                distances[j, i] = dist
    
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    return distances

def v_measure_score(
    labels_true: np.ndarray,
    labels_pred: np.ndarray,
    beta: float = 1.0
) -> float:
    """
    Compute V-measure (harmonic mean of homogeneity and completeness).
    
    Args:
        labels_true: Ground truth cluster labels
        labels_pred: Predicted cluster labels
        beta: Weight of homogeneity vs completeness
        
    Returns:
        V-measure score in range [0, 1]
    """
    homogeneity = homogeneity_score(labels_true, labels_pred)
    completeness = completeness_score(labels_true, labels_pred)
    
    if homogeneity + completeness == 0.0:
        return 0.0
    
    return float((1 + beta) * homogeneity * completeness / (beta * homogeneity + completeness))

def homogeneity_score(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    """
    Compute homogeneity metric (each cluster contains only members of a single class).
    
    Args:
        labels_true: Ground truth class labels
        labels_pred: Predicted cluster labels
        
    Returns:
        Homogeneity score in range [0, 1]
    """
    h_true = _entropy(labels_true)
    
    if h_true == 0:
        return 1.0
    
    mi = mutual_info_score(labels_true, labels_pred)
    return float(1.0 - (h_true - mi) / h_true)

def completeness_score(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    """
    Compute completeness metric (all members of a class are in the same cluster).
    
    Args:
        labels_true: Ground truth class labels
        labels_pred: Predicted cluster labels
        
    Returns:
        Completeness score in range [0, 1]
    """
    h_pred = _entropy(labels_pred)
    
    if h_pred == 0:
        return 1.0
    
    mi = mutual_info_score(labels_true, labels_pred)
    return float(1.0 - (h_pred - mi) / h_pred)

def fowlkes_mallows_score(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    """
    Compute Fowlkes-Mallows Index.
    
    Args:
        labels_true: Ground truth cluster labels
        labels_pred: Predicted cluster labels
        
    Returns:
        FM index in range [0, 1]
    """
    n_samples = len(labels_true)
    contingency = _contingency_matrix(labels_true, labels_pred)
    
    # TP
    tp = np.sum(_comb2(contingency.ravel()))
    
    # FP + TP
    fp_tp = np.sum(_comb2(np.sum(contingency, axis=0)))
    
    # FN + TP
    fn_tp = np.sum(_comb2(np.sum(contingency, axis=1)))
    
    if fp_tp == 0 or fn_tp == 0:
        return 0.0
    
    return float(np.sqrt((tp / fp_tp) * (tp / fn_tp)))
