"""
SMOTE family of oversampling algorithms.

This module contains all SMOTE variants:
- SMOTE: Original Synthetic Minority Over-sampling Technique
- BorderlineSMOTESampler: SMOTE for borderline instances
- ADASYN: Adaptive Synthetic Sampling
- SVMSMOTE: SMOTE using SVM to find support vectors
- KMeansSMOTE: SMOTE with K-Means clustering
"""

import numpy as np
from typing import Optional, Tuple, Union, Dict, List
from tuiml.base.preprocessing import Transformer

class SMOTESampler(Transformer):
    """Synthetic Minority Over-sampling Technique (SMOTE).

    Generates synthetic samples for the minority class(es) by interpolating 
    between existing instances and their nearest neighbors.

    Overview
    --------
    SMOTE addresses class imbalance by creating "plausible" synthetic examples 
    rather than simply duplicating existing ones. This helps the model generalize 
    better by expanding the minority class decision regions.

    Theory
    ------
    For a minority sample :math:`x_i`, a neighbor :math:`\\hat{x}_i` is randomly
    chosen from its :math:`k` nearest minority neighbors. A new sample :math:`x_{new}`
    is generated as:

    .. math::
        x_{new} = x_i + \\lambda \\cdot (\\hat{x}_i - x_i)

    where :math:`\\lambda` is a random number in :math:`[0, 1]`.

    Parameters
    ----------
    sampling_strategy : float, str or dict, default="auto"
        Determines which classes to resample and by how much:
        - ``"auto"`` / ``"not majority"``: Resample all classes except the majority.
        - ``"minority"``: Resample only the minority class.
        - ``"all"``: Resample all classes to match the majority.
        - ``Dict``: ``{class_label: n_samples}`` specifying exact counts.

    k_neighbors : int, default=5
        Number of nearest neighbors to use for interpolation.

    random_state : int, optional
        Seed for the random number generator to ensure reproducibility.

    Attributes
    ----------
    sampling_strategy_ : dict
        The resolved mapping of class labels to the number of samples to generate.

    See Also
    --------
    :class:`~tuiml.preprocessing.sampling.ADASYNSampler` : Adaptive synthetic sampling.
    :class:`~tuiml.preprocessing.sampling.BorderlineSMOTESampler` : Borderline oversampling.

    Examples
    --------
    Oversample the minority class:

    >>> from tuiml.preprocessing.sampling import SMOTESampler
    >>> import numpy as np
    >>> X = np.array([[1, 2], [2, 1], [8, 9], [7, 8], [8, 8]])
    >>> y = np.array([0, 0, 1, 1, 1])  # 0 is minority
    >>> smote = SMOTESampler(k_neighbors=1)
    >>> X_res, y_res = smote.fit_resample(X, y)
    """

    def __init__(
        self,
        sampling_strategy: Union[float, str, dict] = 'auto',
        k_neighbors: int = 5,
        random_state: Optional[int] = None
    ):
        super().__init__()
        self.sampling_strategy = sampling_strategy
        self.k_neighbors = k_neighbors
        self.random_state = random_state
        self.sampling_strategy_: Dict = {}

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict]:
        """Return JSON Schema for parameters."""
        return {
            "sampling_strategy": {
                "type": ["string", "number", "object"],
                "default": "auto",
                "description": "Sampling strategy: 'auto', 'minority', 'not majority', float ratio, or dict {class: count}"
            },
            "k_neighbors": {
                "type": "integer",
                "default": 5,
                "minimum": 1,
                "description": "Number of nearest neighbors for synthetic sample generation"
            },
            "random_state": {
                "type": ["integer", "null"],
                "default": None,
                "description": "Random seed for reproducibility"
            }
        }

    def fit(self, X: np.ndarray, y: np.ndarray) -> "SMOTE":
        """Fit and compute sampling strategy."""
        X, y = np.asarray(X), np.asarray(y)
        self._validate_input(X, y)
        self.sampling_strategy_ = self._compute_sampling_strategy(y)
        self._is_fitted = True
        return self

    def _validate_input(self, X: np.ndarray, y: np.ndarray):
        """Validate input data."""
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have same number of samples")
        classes, counts = np.unique(y, return_counts=True)
        if counts.min() < self.k_neighbors + 1:
            raise ValueError(
                f"Minority class has fewer samples than k_neighbors+1. "
                f"Reduce k_neighbors or collect more samples."
            )

    def _compute_sampling_strategy(self, y: np.ndarray) -> Dict:
        """Compute number of samples to generate per class."""
        classes, counts = np.unique(y, return_counts=True)
        class_counts = dict(zip(classes, counts))
        max_count = counts.max()

        if isinstance(self.sampling_strategy, dict):
            return self.sampling_strategy
        elif self.sampling_strategy in ['auto', 'not majority']:
            majority = classes[np.argmax(counts)]
            return {c: max_count - n for c, n in class_counts.items()
                    if c != majority and n < max_count}
        elif self.sampling_strategy == 'minority':
            minority = classes[np.argmin(counts)]
            return {minority: max_count - class_counts[minority]}
        elif self.sampling_strategy == 'all':
            return {c: max_count - n for c, n in class_counts.items() if n < max_count}
        else:
            raise ValueError(f"Unknown sampling_strategy: {self.sampling_strategy}")

    def fit_resample(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Fit and resample the dataset."""
        self.fit(X, y)
        return self._resample(X, y)

    def _resample(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic samples."""
        rng = np.random.RandomState(self.random_state)
        X_res, y_res = [X.copy()], [y.copy()]

        for target_class, n_samples in self.sampling_strategy_.items():
            if n_samples <= 0:
                continue
            X_minority = X[y == target_class]
            X_syn = self._generate_samples(X_minority, n_samples, rng)
            X_res.append(X_syn)
            y_res.append(np.full(n_samples, target_class))

        return np.vstack(X_res), np.hstack(y_res)

    def _generate_samples(self, X_minority: np.ndarray, n_samples: int,
                          rng: np.random.RandomState) -> np.ndarray:
        """Generate synthetic samples using SMOTE algorithm."""
        n_minority, n_features = X_minority.shape
        neighbors = self._find_neighbors(X_minority)
        synthetic = np.zeros((n_samples, n_features))

        for i in range(n_samples):
            idx = rng.randint(0, n_minority)
            nn_idx = rng.choice(neighbors[idx])
            alpha = rng.random()
            synthetic[i] = X_minority[idx] + alpha * (X_minority[nn_idx] - X_minority[idx])

        return synthetic

    def _find_neighbors(self, X: np.ndarray) -> np.ndarray:
        """Find k nearest neighbors for each sample."""
        n = X.shape[0]
        neighbors = np.zeros((n, self.k_neighbors), dtype=int)
        for i in range(n):
            dists = np.sqrt(np.sum((X - X[i]) ** 2, axis=1))
            dists[i] = np.inf
            neighbors[i] = np.argsort(dists)[:self.k_neighbors]
        return neighbors

    def transform(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Use fit_resample() instead")

    def __repr__(self) -> str:
        return f"SMOTE(k_neighbors={self.k_neighbors})"

class BorderlineSMOTESampler(SMOTESampler):
    """Borderline-SMOTE for oversampling near decision boundaries.

    A variant of SMOTE that only generates synthetic samples for minority 
    instances that are "at risk" of being misclassified (i.e., near the 
    boundary with the majority class).

    Overview
    --------
    Borderline-SMOTE identifies minority instances whose neighbors are mostly 
    from the majority class and prioritizes them for oversampling. This 
    focuses the reinforcement where the classification task is hardest.

    Parameters
    ----------
    sampling_strategy : float, str or dict, default="auto"
        Sampling strategy (see :class:`SMOTESampler`).

    k_neighbors : int, default=5
        Number of nearest neighbors used for SMOTE interpolation.

    m_neighbors : int, default=10
        Number of nearest neighbors used to determine if a sample is 
        on the borderline.

    kind : {"borderline-1", "borderline-2"}, default="borderline-1"
        The type of Borderline-SMOTE:
        - ``"borderline-1"``: Interpolates between borderline samples and 
          their minority neighbors.
        - ``"borderline-2"``: Interpolates between borderline samples and 
          any of their nearest neighbors (minority or majority).

    random_state : int, optional
        Seed for reproducibility.

    See Also
    --------
    :class:`~tuiml.preprocessing.sampling.SMOTESampler` : The base SMOTE algorithm.
    :class:`~tuiml.preprocessing.sampling.SVMSMOTESampler` : SVM-based boundary detection.

    Examples
    --------
    Oversample near the decision boundary:

    >>> from tuiml.preprocessing.sampling import BorderlineSMOTESampler
    >>> import numpy as np
    >>> X = np.random.rand(100, 2)
    >>> y = (X[:, 0] + X[:, 1] > 1).astype(int)
    >>> sampler = BorderlineSMOTESampler(m_neighbors=5)
    >>> X_res, y_res = sampler.fit_resample(X, y)
    """

    def __init__(
        self,
        sampling_strategy: Union[float, str, dict] = 'auto',
        k_neighbors: int = 5,
        m_neighbors: int = 10,
        kind: str = 'borderline-1',
        random_state: Optional[int] = None
    ):
        super().__init__(sampling_strategy, k_neighbors, random_state)
        self.m_neighbors = m_neighbors
        self.kind = kind

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict]:
        """Return JSON Schema for parameters."""
        return {
            "sampling_strategy": {
                "type": ["string", "number", "object"],
                "default": "auto",
                "description": "Sampling strategy"
            },
            "k_neighbors": {
                "type": "integer",
                "default": 5,
                "minimum": 1,
                "description": "Number of neighbors for SMOTE generation"
            },
            "m_neighbors": {
                "type": "integer",
                "default": 10,
                "minimum": 1,
                "description": "Number of neighbors for borderline detection"
            },
            "kind": {
                "type": "string",
                "default": "borderline-1",
                "enum": ["borderline-1", "borderline-2"],
                "description": "Type of borderline SMOTE"
            },
            "random_state": {
                "type": ["integer", "null"],
                "default": None,
                "description": "Random seed for reproducibility"
            }
        }

    def _resample(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Generate samples from borderline instances."""
        rng = np.random.RandomState(self.random_state)
        X_res, y_res = [X.copy()], [y.copy()]

        for target_class, n_samples in self.sampling_strategy_.items():
            if n_samples <= 0:
                continue

            X_minority = X[y == target_class]
            borderline = self._find_borderline(X_minority, X, y, target_class)
            X_border = X_minority[borderline] if borderline.any() else X_minority

            X_syn = self._generate_from_borderline(X_border, X_minority, n_samples, rng)
            X_res.append(X_syn)
            y_res.append(np.full(len(X_syn), target_class))

        return np.vstack(X_res), np.hstack(y_res)

    def _find_borderline(self, X_min: np.ndarray, X: np.ndarray,
                         y: np.ndarray, target) -> np.ndarray:
        """Find borderline instances."""
        borderline = np.zeros(len(X_min), dtype=bool)
        for i, sample in enumerate(X_min):
            dists = np.sqrt(np.sum((X - sample) ** 2, axis=1))
            nn_idx = np.argsort(dists)[1:self.m_neighbors + 1]
            n_majority = np.sum(y[nn_idx] != target)
            if self.m_neighbors / 2 <= n_majority < self.m_neighbors:
                borderline[i] = True
        return borderline

    def _generate_from_borderline(self, X_border: np.ndarray, X_min: np.ndarray,
                                   n_samples: int, rng: np.random.RandomState) -> np.ndarray:
        """Generate from borderline instances."""
        n_border, n_feat = X_border.shape
        synthetic = np.zeros((n_samples, n_feat))

        # Find neighbors in minority set
        for i in range(n_samples):
            idx = rng.randint(0, n_border)
            sample = X_border[idx]
            dists = np.sqrt(np.sum((X_min - sample) ** 2, axis=1))
            nn_idx = np.argsort(dists)[:self.k_neighbors]
            neighbor = X_min[rng.choice(nn_idx)]
            alpha = rng.random()
            synthetic[i] = sample + alpha * (neighbor - sample)

        return synthetic

    def __repr__(self) -> str:
        return f"BorderlineSMOTESampler(k={self.k_neighbors}, m={self.m_neighbors})"

class ADASYNSampler(SMOTESampler):
    """Adaptive Synthetic Sampling (ADASYN).

    Generates synthetic samples by focusing on minority instances that are 
    "harder" to learn, based on the density of majority class neighbors.

    Overview
    --------
    Unlike SMOTE, ADASYN uses a weighted distribution for minority classes 
    according to their level of difficulty in learning. More synthetic data 
    is generated for minority class samples that are harder to learn compared 
    to those that are easier to learn.

    Theory
    ------
    The number of samples to generate for a minority instance :math:`x_i` is
    proportional to its difficulty ratio :math:`r_i`:

    .. math::
        r_i = \\frac{\\Delta_i}{k}

    where :math:`\\Delta_i` is the number of majority class neighbors among
    the :math:`k` nearest neighbors of :math:`x_i`.

    Parameters
    ----------
    sampling_strategy : float, str or dict, default="auto"
        Sampling strategy (see :class:`SMOTESampler`).

    k_neighbors : int, default=5
        Number of nearest neighbors used to compute the difficulty ratio 
        and for interpolation.

    random_state : int, optional
        Seed for reproducibility.

    See Also
    --------
    :class:`~tuiml.preprocessing.sampling.SMOTESampler` : Standard SMOTE.

    Examples
    --------
    Adaptive oversampling:

    >>> from tuiml.preprocessing.sampling import ADASYNSampler
    >>> import numpy as np
    >>> X = np.random.rand(100, 2)
    >>> y = (X[:, 0] > 0.8).astype(int) # Highly imbalanced
    >>> sampler = ADASYNSampler(k_neighbors=5)
    >>> X_res, y_res = sampler.fit_resample(X, y)
    """

    def __init__(
        self,
        sampling_strategy: Union[float, str, dict] = 'auto',
        k_neighbors: int = 5,
        random_state: Optional[int] = None
    ):
        super().__init__(sampling_strategy, k_neighbors, random_state)

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict]:
        """Return JSON Schema for parameters."""
        return {
            "sampling_strategy": {
                "type": ["string", "number", "object"],
                "default": "auto",
                "description": "Sampling strategy"
            },
            "k_neighbors": {
                "type": "integer",
                "default": 5,
                "minimum": 1,
                "description": "Number of nearest neighbors"
            },
            "random_state": {
                "type": ["integer", "null"],
                "default": None,
                "description": "Random seed for reproducibility"
            }
        }

    def _resample(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Generate adaptive synthetic samples."""
        rng = np.random.RandomState(self.random_state)
        X_res, y_res = [X.copy()], [y.copy()]

        for target_class, n_samples in self.sampling_strategy_.items():
            if n_samples <= 0:
                continue

            X_minority = X[y == target_class]
            ratios = self._compute_ratios(X_minority, X, y, target_class)

            if ratios.sum() > 0:
                ratios = ratios / ratios.sum()
            else:
                ratios = np.ones(len(X_minority)) / len(X_minority)

            X_syn = self._generate_adaptive(X_minority, n_samples, ratios, rng)
            X_res.append(X_syn)
            y_res.append(np.full(len(X_syn), target_class))

        return np.vstack(X_res), np.hstack(y_res)

    def _compute_ratios(self, X_min: np.ndarray, X: np.ndarray,
                        y: np.ndarray, target) -> np.ndarray:
        """Compute difficulty ratio for each minority instance."""
        ratios = np.zeros(len(X_min))
        for i, sample in enumerate(X_min):
            dists = np.sqrt(np.sum((X - sample) ** 2, axis=1))
            nn_idx = np.argsort(dists)[1:self.k_neighbors + 1]
            n_majority = np.sum(y[nn_idx] != target)
            ratios[i] = n_majority / self.k_neighbors
        return ratios

    def _generate_adaptive(self, X_min: np.ndarray, n_samples: int,
                           ratios: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
        """Generate samples with adaptive weighting."""
        n_per_instance = (ratios * n_samples).astype(int)
        diff = n_samples - n_per_instance.sum()
        if diff > 0:
            top_idx = np.argsort(ratios)[-diff:]
            n_per_instance[top_idx] += 1

        neighbors = self._find_neighbors(X_min)
        synthetic_list = []

        for i, n in enumerate(n_per_instance):
            for _ in range(n):
                nn_idx = rng.choice(neighbors[i])
                alpha = rng.random()
                syn = X_min[i] + alpha * (X_min[nn_idx] - X_min[i])
                synthetic_list.append(syn)

        return np.array(synthetic_list) if synthetic_list else np.zeros((0, X_min.shape[1]))

    def __repr__(self) -> str:
        return f"ADASYN(k_neighbors={self.k_neighbors})"

class SVMSMOTESampler(SMOTESampler):
    """SVM-SMOTE for oversampling using SVM support vectors.

    Uses an SVM classifier to identify the decision boundary and generates 
    synthetic samples around the minority class support vectors.

    Overview
    --------
    By focusing on support vectors, SVMSMOTE concentrates oversampling in the 
    region where the minority and majority classes are most likely to overlap, 
    effectively strengthening the decision boundary.

    Parameters
    ----------
    sampling_strategy : float, str or dict, default="auto"
        Sampling strategy (see :class:`SMOTESampler`).

    k_neighbors : int, default=5
        Number of nearest neighbors used for interpolation.

    svm_estimator : object, optional
        The SVM estimator used to find support vectors. If ``None``, a
        default :class:`~tuiml.algorithms.svm.SVC` is used.

    random_state : int, optional
        Seed for reproducibility.

    Examples
    --------
    Oversample using SVM support vectors:

    >>> from tuiml.preprocessing.sampling import SVMSMOTESampler
    >>> import numpy as np
    >>> X = np.random.rand(100, 2)
    >>> y = (X[:, 1] > 0.7).astype(int)
    >>> sampler = SVMSMOTESampler()
    >>> X_res, y_res = sampler.fit_resample(X, y)
    """

    def __init__(
        self,
        sampling_strategy: Union[float, str, dict] = 'auto',
        k_neighbors: int = 5,
        svm_estimator=None,
        random_state: Optional[int] = None
    ):
        super().__init__(sampling_strategy, k_neighbors, random_state)
        self.svm_estimator = svm_estimator

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict]:
        """Return JSON Schema for parameters."""
        return {
            "sampling_strategy": {
                "type": ["string", "number", "object"],
                "default": "auto",
                "description": "Sampling strategy"
            },
            "k_neighbors": {
                "type": "integer",
                "default": 5,
                "minimum": 1,
                "description": "Number of nearest neighbors"
            },
            "random_state": {
                "type": ["integer", "null"],
                "default": None,
                "description": "Random seed for reproducibility"
            }
        }

    def _resample(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Generate samples around SVM support vectors."""
        rng = np.random.RandomState(self.random_state)

        # Get SVM support vectors
        from tuiml.algorithms.svm import SVC
        svm = self.svm_estimator or SVC()
        svm.fit(X, y)
        support_indices = svm.support_

        X_res, y_res = [X.copy()], [y.copy()]

        for target_class, n_samples in self.sampling_strategy_.items():
            if n_samples <= 0:
                continue

            # Get minority support vectors
            minority_mask = y == target_class
            sv_mask = np.zeros(len(X), dtype=bool)
            sv_mask[support_indices] = True
            minority_sv = minority_mask & sv_mask

            X_sv = X[minority_sv] if minority_sv.any() else X[minority_mask]
            X_minority = X[minority_mask]

            X_syn = self._generate_around_sv(X_sv, X_minority, n_samples, rng)
            X_res.append(X_syn)
            y_res.append(np.full(len(X_syn), target_class))

        return np.vstack(X_res), np.hstack(y_res)

    def _generate_around_sv(self, X_sv: np.ndarray, X_min: np.ndarray,
                            n_samples: int, rng: np.random.RandomState) -> np.ndarray:
        """Generate samples around support vectors."""
        n_sv, n_feat = X_sv.shape
        synthetic = np.zeros((n_samples, n_feat))

        for i in range(n_samples):
            idx = rng.randint(0, n_sv)
            sample = X_sv[idx]
            dists = np.sqrt(np.sum((X_min - sample) ** 2, axis=1))
            nn_idx = np.argsort(dists)[:self.k_neighbors]
            neighbor = X_min[rng.choice(nn_idx)]
            alpha = rng.random()
            synthetic[i] = sample + alpha * (neighbor - sample)

        return synthetic

    def __repr__(self) -> str:
        return f"SVMSMOTE(k_neighbors={self.k_neighbors})"

class KMeansSMOTESampler(SMOTESampler):
    """K-Means SMOTE for oversampling in "safe" clusters.

    Combines K-Means clustering with SMOTE to avoid generating noise 
    and to focus oversampling on dense minority regions.

    Overview
    --------
    The algorithm performs three steps:
    1. Cluster the entire dataset using K-Means.
    2. Filter clusters, keeping only those with a high proportion of minority 
       samples ("safe" clusters).
    3. Apply SMOTE within each safe cluster.

    Parameters
    ----------
    sampling_strategy : float, str or dict, default="auto"
        Sampling strategy (see :class:`SMOTESampler`).

    k_neighbors : int, default=5
        Number of nearest neighbors used for SMOTE.

    n_clusters : int, optional
        Number of clusters for K-Means. If ``None``, defaults to 
        :math:`\\sqrt{n_{minority}}`.

    cluster_balance_threshold : float, default=0.5
        The minimum ratio of minority samples in a cluster for it to be 
        considered "safe" for oversampling.

    random_state : int, optional
        Seed for reproducibility.

    Examples
    --------
    Cluster-based oversampling:

    >>> from tuiml.preprocessing.sampling import KMeansSMOTESampler
    >>> import numpy as np
    >>> X = np.random.rand(200, 2)
    >>> y = (np.linalg.norm(X - 0.5, axis=1) < 0.2).astype(int)
    >>> sampler = KMeansSMOTESampler(n_clusters=10)
    >>> X_res, y_res = sampler.fit_resample(X, y)
    """

    def __init__(
        self,
        sampling_strategy: Union[float, str, dict] = 'auto',
        k_neighbors: int = 5,
        n_clusters: int = None,
        cluster_balance_threshold: float = 0.5,
        random_state: Optional[int] = None
    ):
        super().__init__(sampling_strategy, k_neighbors, random_state)
        self.n_clusters = n_clusters
        self.cluster_balance_threshold = cluster_balance_threshold

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict]:
        """Return JSON Schema for parameters."""
        return {
            "sampling_strategy": {
                "type": ["string", "number", "object"],
                "default": "auto",
                "description": "Sampling strategy"
            },
            "k_neighbors": {
                "type": "integer",
                "default": 5,
                "minimum": 1,
                "description": "Number of nearest neighbors"
            },
            "n_clusters": {
                "type": ["integer", "null"],
                "default": None,
                "description": "Number of clusters (None for sqrt(n_minority))"
            },
            "cluster_balance_threshold": {
                "type": "number",
                "default": 0.5,
                "minimum": 0.0,
                "maximum": 1.0,
                "description": "Minimum ratio of minority in cluster to be safe"
            },
            "random_state": {
                "type": ["integer", "null"],
                "default": None,
                "description": "Random seed for reproducibility"
            }
        }

    def _resample(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Generate samples within safe clusters."""
        rng = np.random.RandomState(self.random_state)
        X_res, y_res = [X.copy()], [y.copy()]

        for target_class, n_samples in self.sampling_strategy_.items():
            if n_samples <= 0:
                continue

            X_minority = X[y == target_class]
            n_clusters = self.n_clusters or max(1, int(np.sqrt(len(X_minority))))

            # Simple k-means clustering
            clusters, centroids = self._kmeans(X_minority, n_clusters, rng)

            # Find safe clusters
            safe_clusters = self._find_safe_clusters(
                X_minority, clusters, X, y, target_class
            )

            if not safe_clusters:
                safe_clusters = list(range(n_clusters))

            X_syn = self._generate_in_clusters(
                X_minority, clusters, safe_clusters, n_samples, rng
            )
            X_res.append(X_syn)
            y_res.append(np.full(len(X_syn), target_class))

        return np.vstack(X_res), np.hstack(y_res)

    def _kmeans(self, X: np.ndarray, k: int, rng: np.random.RandomState,
                max_iter: int = 100) -> Tuple[np.ndarray, np.ndarray]:
        """Simple k-means clustering."""
        n = X.shape[0]
        centroids = X[rng.choice(n, k, replace=False)]

        for _ in range(max_iter):
            dists = np.array([[np.sqrt(np.sum((x - c) ** 2)) for c in centroids] for x in X])
            clusters = np.argmin(dists, axis=1)
            new_centroids = np.array([X[clusters == i].mean(axis=0) if (clusters == i).any()
                                      else centroids[i] for i in range(k)])
            if np.allclose(centroids, new_centroids):
                break
            centroids = new_centroids

        return clusters, centroids

    def _find_safe_clusters(self, X_min: np.ndarray, clusters: np.ndarray,
                            X: np.ndarray, y: np.ndarray, target) -> List[int]:
        """Find clusters that are safe for oversampling."""
        safe = []
        for c in np.unique(clusters):
            cluster_samples = X_min[clusters == c]
            # Check density of minority in this region
            total_nearby = 0
            minority_nearby = 0
            for sample in cluster_samples:
                dists = np.sqrt(np.sum((X - sample) ** 2, axis=1))
                nearby = dists < np.median(dists)
                total_nearby += nearby.sum()
                minority_nearby += (nearby & (y == target)).sum()

            if total_nearby > 0 and minority_nearby / total_nearby >= self.cluster_balance_threshold:
                safe.append(c)
        return safe

    def _generate_in_clusters(self, X_min: np.ndarray, clusters: np.ndarray,
                              safe_clusters: List[int], n_samples: int,
                              rng: np.random.RandomState) -> np.ndarray:
        """Generate samples within safe clusters."""
        safe_mask = np.isin(clusters, safe_clusters)
        X_safe = X_min[safe_mask]

        if len(X_safe) < self.k_neighbors + 1:
            X_safe = X_min

        return self._generate_samples(X_safe, n_samples, rng)

    def __repr__(self) -> str:
        return f"KMeansSMOTE(k={self.k_neighbors}, n_clusters={self.n_clusters})"
