"""
Clustering evaluation metrics.

Two kinds of measure live here, and mixing them up is the usual mistake.

**External** metrics compare a clustering against known ground-truth labels:
:func:`adjusted_rand_score`, :func:`rand_score`, :func:`mutual_info_score`,
:func:`normalized_mutual_info_score`, :func:`homogeneity_score`,
:func:`completeness_score`, :func:`v_measure_score` and
:func:`fowlkes_mallows_score`. They take ``(labels_true, labels_pred)`` and are
invariant to how the clusters are named, so permuting labels changes nothing.

**Internal** metrics judge the clustering from the data geometry alone, with no
ground truth: :func:`silhouette_score`, :func:`silhouette_samples`,
:func:`davies_bouldin_score` and :func:`calinski_harabasz_score`. They take
``(X, labels)`` and are what you use to pick a number of clusters.

Two directions to watch. :func:`davies_bouldin_score` is LOWER-is-better, unlike
every other metric here. And :func:`rand_score` and :func:`mutual_info_score`
are uncorrected, so they drift upward as the cluster count grows -- prefer
:func:`adjusted_rand_score` or :func:`normalized_mutual_info_score` when
comparing clusterings of different sizes.

Examples
--------
>>> from tuiml.evaluation.metrics import adjusted_rand_score, v_measure_score
>>> true = [0, 0, 1, 1, 2, 2]
>>> pred = [1, 1, 0, 0, 2, 2]          # same partition, different names
>>> adjusted_rand_score(true, pred)
1.0
>>> v_measure_score(true, pred)
1.0
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
    
    The Rand index corrected for chance, using the contingency table
    :math:`n_{ij}` with row sums :math:`a_i` and column sums :math:`b_j`:

    .. math::
        \\text{ARI} = \\frac{\\sum_{ij} \\binom{n_{ij}}{2} -
        \\left[\\sum_i \\binom{a_i}{2} \\sum_j \\binom{b_j}{2}\\right] \\Big/ \\binom{n}{2}}
        {\\tfrac{1}{2}\\left[\\sum_i \\binom{a_i}{2} + \\sum_j \\binom{b_j}{2}\\right] -
        \\left[\\sum_i \\binom{a_i}{2} \\sum_j \\binom{b_j}{2}\\right] \\Big/ \\binom{n}{2}}

    Subtracting the expected index makes 0.0 the score of random labelling,
    so unlike :func:`rand_score` the value does not drift upward with the
    number of clusters.

    Parameters
    ----------
    labels_true : array-like of shape (n_samples,)
        Ground-truth cluster labels.
    labels_pred : array-like of shape (n_samples,)
        Predicted cluster labels.

    Returns
    -------
    score : float
        ARI in the range [-1, 1]. 1.0 is perfect agreement, 0.0 is the value
        expected from random labelling, and negative values mean the agreement
        is worse than chance.

    Examples
    --------
    >>> from tuiml.evaluation.metrics import adjusted_rand_score
    >>> adjusted_rand_score([0, 0, 1, 1], [0, 0, 1, 1])
    1.0
    >>> round(adjusted_rand_score([0, 0, 1, 1], [0, 1, 0, 1]), 2)
    -0.5
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

    The fraction of sample pairs that both labellings agree about, where
    :math:`a` counts pairs together in both and :math:`b` counts pairs apart
    in both:

    .. math::
        \\text{RI} = \\frac{a + b}{\\binom{n}{2}}

    Not corrected for chance: random labellings score well above 0. Use
    :func:`adjusted_rand_score` when comparing across different cluster counts.

    Parameters
    ----------
    labels_true : array-like of shape (n_samples,)
        Ground-truth cluster labels.
    labels_pred : array-like of shape (n_samples,)
        Predicted cluster labels.

    Returns
    -------
    score : float
        RI score in the range [0, 1].

    Examples
    --------
    >>> from tuiml.evaluation.metrics import rand_score
    >>> round(rand_score([0, 0, 1, 1], [0, 0, 1, 1]), 3)
    0.333
    >>> rand_score([0, 0, 1, 1], [0, 1, 0, 1])
    0.0
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

    The mean silhouette over all samples. For one sample, with :math:`a(i)` its
    mean distance to its own cluster and :math:`b(i)` the mean distance to the
    nearest other cluster:

    .. math::
        s(i) = \\frac{b(i) - a(i)}{\\max\\{a(i),\\, b(i)\\}}

    +1 means the sample sits well inside its cluster, 0 means it lies on a
    boundary, and negative means it is closer to a different cluster.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Feature matrix.
    labels : array-like of shape (n_samples,)
        Cluster label for each sample.
    metric : str, default='euclidean'
        Distance metric to use. One of ``'euclidean'``, ``'manhattan'``,
        or ``'cosine'``.

    Returns
    -------
    score : float
        Mean silhouette score in the range [-1, 1].

    Examples
    --------
    >>> import numpy as np
    >>> from tuiml.evaluation.metrics import silhouette_score
    >>> X = np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]])
    >>> labels = np.array([0, 0, 0, 1, 1, 1])
    >>> round(silhouette_score(X, labels), 3)
    0.287
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

    Per-sample version of :func:`silhouette_score`:

    .. math::
        s(i) = \\frac{b(i) - a(i)}{\\max\\{a(i),\\, b(i)\\}}

    Useful for finding which individual points are badly clustered rather
    than only the overall average.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Feature matrix.
    labels : array-like of shape (n_samples,)
        Cluster label for each sample.
    metric : str, default='euclidean'
        Distance metric to use. One of ``'euclidean'``, ``'manhattan'``,
        or ``'cosine'``.

    Returns
    -------
    scores : np.ndarray of shape (n_samples,)
        Silhouette coefficient for each sample, in the range [-1, 1].

    Examples
    --------
    >>> import numpy as np
    >>> from tuiml.evaluation.metrics import silhouette_samples
    >>> X = np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]])
    >>> labels = np.array([0, 0, 0, 1, 1, 1])
    >>> np.round(silhouette_samples(X, labels), 3)
    array([0.412, 0.225, 0.225, 0.412, 0.225, 0.225])
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

    Average over clusters of the worst-case similarity to any other cluster,
    where :math:`s_i` is the mean distance from cluster :math:`i` to its own
    centroid and :math:`d_{ij}` the distance between centroids:

    .. math::
        \\text{DB} = \\frac{1}{k} \\sum_{i=1}^{k}
        \\max_{j \\neq i} \\frac{s_i + s_j}{d_{ij}}

    LOWER is better, and 0 is the best possible value: the opposite direction
    to most metrics in this module.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Feature matrix.
    labels : array-like of shape (n_samples,)
        Cluster label for each sample.

    Returns
    -------
    score : float
        Davies-Bouldin score (lower is better).

    Examples
    --------
    >>> import numpy as np
    >>> from tuiml.evaluation.metrics import davies_bouldin_score
    >>> X = np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]])
    >>> labels = np.array([0, 0, 0, 1, 1, 1])
    >>> round(davies_bouldin_score(X, labels), 3)
    0.889
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

    Ratio of between-cluster to within-cluster dispersion, each corrected for
    its degrees of freedom:

    .. math::
        \\text{CH} = \\frac{\\operatorname{tr}(B_k) \\big/ (k - 1)}
        {\\operatorname{tr}(W_k) \\big/ (n - k)}

    Higher is better. The score is unbounded above, so it is meaningful for
    ranking candidate cluster counts on one dataset, not across datasets.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Feature matrix.
    labels : array-like of shape (n_samples,)
        Cluster label for each sample.

    Returns
    -------
    score : float
        Calinski-Harabasz score (higher is better).

    Examples
    --------
    >>> import numpy as np
    >>> from tuiml.evaluation.metrics import calinski_harabasz_score
    >>> X = np.array([[1, 2], [1, 4], [1, 0], [4, 2], [4, 4], [4, 0]])
    >>> labels = np.array([0, 0, 0, 1, 1, 1])
    >>> round(calinski_harabasz_score(X, labels), 3)
    3.375
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

    How much knowing the cluster tells you about the true class:

    .. math::
        \\text{MI}(U, V) = \\sum_{i}\\sum_{j} \\frac{n_{ij}}{n}
        \\log \\frac{n \\, n_{ij}}{a_i b_j}

    Measured in nats and unbounded above, which makes raw MI hard to compare;
    :func:`normalized_mutual_info_score` rescales it to [0, 1].

    Parameters
    ----------
    labels_true : array-like of shape (n_samples,)
        Ground-truth cluster labels.
    labels_pred : array-like of shape (n_samples,)
        Predicted cluster labels.

    Returns
    -------
    score : float
        Mutual information score (non-negative, unbounded above).

    Examples
    --------
    >>> from tuiml.evaluation.metrics import mutual_info_score
    >>> round(mutual_info_score([0, 0, 1, 1], [0, 0, 1, 1]), 3)
    0.693
    >>> mutual_info_score([0, 0, 1, 1], [0, 1, 0, 1])
    0.0
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

    Mutual information rescaled by the entropies of the two labellings:

    .. math::
        \\text{NMI}(U, V) = \\frac{\\text{MI}(U, V)}
        {\\tfrac{1}{2}\\left[H(U) + H(V)\\right]}

    Bounded in [0, 1], reaching 1.0 exactly when the two labellings agree up
    to relabelling.

    Parameters
    ----------
    labels_true : array-like of shape (n_samples,)
        Ground-truth cluster labels.
    labels_pred : array-like of shape (n_samples,)
        Predicted cluster labels.
    average_method : str, default='arithmetic'
        Normalizer to divide the mutual information by. One of
        ``'arithmetic'``, ``'geometric'``, ``'min'``, or ``'max'``,
        computed from the entropies of ``labels_true`` and ``labels_pred``.

    Returns
    -------
    score : float
        NMI score in the range [0, 1].

    Examples
    --------
    >>> from tuiml.evaluation.metrics import normalized_mutual_info_score
    >>> round(normalized_mutual_info_score([0, 0, 1, 1], [0, 0, 1, 1]), 3)
    1.0
    >>> normalized_mutual_info_score([0, 0, 1, 1], [0, 1, 0, 1])
    0.0
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
    # np.unique returns only observed labels, so every count is >= 1 and every
    # probability is strictly positive. An epsilon inside the log would guard a
    # case that cannot occur while inflating the entropy, which leaked into the
    # normalized scores and pushed them just above their maximum of 1.0.
    return float(-np.sum(probs * np.log(probs)))

def _pairwise_distances(X: np.ndarray, metric: str = 'euclidean') -> np.ndarray:
    """
    Compute pairwise distances between samples.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Feature matrix.
    metric : str, default='euclidean'
        Distance metric to use. One of ``'euclidean'``, ``'manhattan'``,
        or ``'cosine'``.

    Returns
    -------
    distances : np.ndarray of shape (n_samples, n_samples)
        Symmetric matrix of pairwise distances.
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

    The harmonic mean of homogeneity :math:`h` and completeness :math:`c`:

    .. math::
        v = \\frac{2 h c}{h + c}

    Symmetric in the two labellings, and equal to
    :func:`normalized_mutual_info_score` under arithmetic-mean normalization.

    Parameters
    ----------
    labels_true : array-like of shape (n_samples,)
        Ground-truth class labels.
    labels_pred : array-like of shape (n_samples,)
        Predicted cluster labels.
    beta : float, default=1.0
        Weight of homogeneity relative to completeness. Values greater
        than 1.0 favor completeness; values less than 1.0 favor
        homogeneity.

    Returns
    -------
    score : float
        V-measure score in the range [0, 1].

    Examples
    --------
    >>> from tuiml.evaluation.metrics import v_measure_score
    >>> round(v_measure_score([0, 0, 1, 1], [0, 0, 1, 1]), 3)
    1.0
    >>> v_measure_score([0, 0, 1, 1], [0, 1, 0, 1])
    0.0
    """
    homogeneity = homogeneity_score(labels_true, labels_pred)
    completeness = completeness_score(labels_true, labels_pred)
    
    if homogeneity + completeness == 0.0:
        return 0.0
    
    return float((1 + beta) * homogeneity * completeness / (beta * homogeneity + completeness))

def homogeneity_score(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    """
    Compute homogeneity metric (each cluster contains only members of a single class).

    Whether each cluster contains only members of a single class:

    .. math::
        h = 1 - \\frac{H(C \\mid K)}{H(C)}

    Splitting one true class across many clusters does not hurt this score --
    that is what :func:`completeness_score` measures.

    Parameters
    ----------
    labels_true : array-like of shape (n_samples,)
        Ground-truth class labels.
    labels_pred : array-like of shape (n_samples,)
        Predicted cluster labels.

    Returns
    -------
    score : float
        Homogeneity score in the range [0, 1].

    Examples
    --------
    >>> from tuiml.evaluation.metrics import homogeneity_score
    >>> round(homogeneity_score([0, 0, 1, 1], [0, 0, 1, 1]), 3)
    1.0
    >>> homogeneity_score([0, 0, 1, 1], [0, 1, 0, 1])
    0.0
    """
    h_true = _entropy(labels_true)
    
    if h_true == 0:
        return 1.0
    
    mi = mutual_info_score(labels_true, labels_pred)
    return float(1.0 - (h_true - mi) / h_true)

def completeness_score(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    """
    Compute completeness metric (all members of a class are in the same cluster).

    Whether all members of a class land in the same cluster:

    .. math::
        c = 1 - \\frac{H(K \\mid C)}{H(K)}

    The mirror image of :func:`homogeneity_score`; putting everything in one
    cluster scores 1.0 here and poorly there.

    Parameters
    ----------
    labels_true : array-like of shape (n_samples,)
        Ground-truth class labels.
    labels_pred : array-like of shape (n_samples,)
        Predicted cluster labels.

    Returns
    -------
    score : float
        Completeness score in the range [0, 1].

    Examples
    --------
    >>> from tuiml.evaluation.metrics import completeness_score
    >>> round(completeness_score([0, 0, 1, 1], [0, 0, 1, 1]), 3)
    1.0
    >>> completeness_score([0, 0, 1, 1], [0, 1, 0, 1])
    0.0
    """
    h_pred = _entropy(labels_pred)
    
    if h_pred == 0:
        return 1.0
    
    mi = mutual_info_score(labels_true, labels_pred)
    return float(1.0 - (h_pred - mi) / h_pred)

def fowlkes_mallows_score(labels_true: np.ndarray, labels_pred: np.ndarray) -> float:
    """
    Compute Fowlkes-Mallows Index.

    The geometric mean of the pairwise precision and recall, counting pairs of
    samples placed in the same cluster:

    .. math::
        \\text{FMI} = \\frac{\\text{TP}}
        {\\sqrt{(\\text{TP} + \\text{FP})(\\text{TP} + \\text{FN})}}

    Bounded in [0, 1]; unlike :func:`rand_score` it ignores the true-negative
    pairs that dominate when there are many clusters.

    Parameters
    ----------
    labels_true : array-like of shape (n_samples,)
        Ground-truth cluster labels.
    labels_pred : array-like of shape (n_samples,)
        Predicted cluster labels.

    Returns
    -------
    score : float
        Fowlkes-Mallows index in the range [0, 1].

    Examples
    --------
    >>> from tuiml.evaluation.metrics import fowlkes_mallows_score
    >>> fowlkes_mallows_score([0, 0, 1, 1], [0, 0, 1, 1])
    1.0
    >>> fowlkes_mallows_score([0, 0, 1, 1], [0, 1, 0, 1])
    0.0
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
