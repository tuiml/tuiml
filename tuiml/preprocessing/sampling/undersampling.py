"""
Undersampling methods for imbalanced learning.

Methods to reduce majority class samples to balance datasets.
"""

import numpy as np
from typing import Optional, Tuple, Union, Dict, List
from tuiml.base.preprocessing import Transformer

class RandomUnderSampler(Transformer):
    """
    Random under-sampling by removing majority samples.

    Randomly removes samples from majority class to balance classes.

    Parameters
    ----------
    sampling_strategy : float or str or dict, default='auto'
        Sampling strategy:
        - 'auto': reduce all classes to match minority
        - 'majority': only undersample majority class
        - dict: {class_label: target_count}
    random_state : int, optional
        Random seed.
    replacement : bool, default=False
        Whether to sample with replacement.

    Examples
    --------
    >>> from tuiml.preprocessing.sampling import RandomUnderSampler
    >>> rus = RandomUnderSampler(sampling_strategy='auto')
    >>> X_res, y_res = rus.fit_resample(X, y)
    """

    def __init__(
        self,
        sampling_strategy: Union[float, str, dict] = 'auto',
        random_state: Optional[int] = None,
        replacement: bool = False
    ):
        super().__init__()
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state
        self.replacement = replacement
        self.sampling_strategy_: Dict = {}

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict]:
        """Return JSON Schema for parameters."""
        return {
            "sampling_strategy": {
                "type": ["string", "number", "object"],
                "default": "auto",
                "description": "Sampling strategy: 'auto', 'majority', or dict {class: target_count}"
            },
            "random_state": {
                "type": ["integer", "null"],
                "default": None,
                "description": "Random seed for reproducibility"
            },
            "replacement": {
                "type": "boolean",
                "default": False,
                "description": "Whether to sample with replacement"
            }
        }

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomUnderSampler":
        """Fit the sampler."""
        X, y = np.asarray(X), np.asarray(y)
        classes, counts = np.unique(y, return_counts=True)
        class_counts = dict(zip(classes, counts))
        min_count = counts.min()

        if isinstance(self.sampling_strategy, dict):
            self.sampling_strategy_ = self.sampling_strategy
        elif self.sampling_strategy in ['auto', 'not minority']:
            minority = classes[np.argmin(counts)]
            self.sampling_strategy_ = {
                c: min_count for c in classes if c != minority
            }
        elif self.sampling_strategy == 'majority':
            majority = classes[np.argmax(counts)]
            self.sampling_strategy_ = {majority: min_count}
        else:
            self.sampling_strategy_ = {c: n for c, n in class_counts.items()}

        self._is_fitted = True
        return self

    def fit_resample(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Fit and resample."""
        self.fit(X, y)
        rng = np.random.RandomState(self.random_state)

        X_res, y_res = [], []

        for cls in np.unique(y):
            mask = y == cls
            X_class = X[mask]

            if cls in self.sampling_strategy_:
                n_samples = self.sampling_strategy_[cls]
                if n_samples < len(X_class):
                    idx = rng.choice(len(X_class), n_samples, replace=self.replacement)
                    X_res.append(X_class[idx])
                    y_res.append(np.full(n_samples, cls))
                else:
                    X_res.append(X_class)
                    y_res.append(np.full(len(X_class), cls))
            else:
                X_res.append(X_class)
                y_res.append(np.full(len(X_class), cls))

        return np.vstack(X_res), np.hstack(y_res)

    def transform(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Use fit_resample() instead")

    def __repr__(self) -> str:
        return f"RandomUnderSampler(sampling_strategy={self.sampling_strategy!r})"

class TomekLinksSampler(Transformer):
    """
    Tomek Links under-sampling.

    Removes majority class samples that form Tomek links with minority
    samples. A Tomek link exists if two samples of different classes
    are each other's nearest neighbors.

    Parameters
    ----------
    sampling_strategy : str, default='auto'
        Which samples to remove:
        - 'auto' or 'majority': Remove majority samples
        - 'all': Remove both samples from Tomek links

    Examples
    --------
    >>> from tuiml.preprocessing.sampling import TomekLinksSampler
    >>> tl = TomekLinksSampler()
    >>> X_res, y_res = tl.fit_resample(X, y)

    References
    ----------
    Tomek, I. (1976). Two Modifications of CNN. IEEE Transactions on
    Systems, Man, and Cybernetics.
    """

    def __init__(self, sampling_strategy: str = 'auto'):
        super().__init__()
        self.sampling_strategy = sampling_strategy

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict]:
        """Return JSON Schema for parameters."""
        return {
            "sampling_strategy": {
                "type": "string",
                "default": "auto",
                "enum": ["auto", "majority", "all"],
                "description": "Which samples to remove from Tomek links"
            }
        }

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TomekLinksSampler":
        """Fit (no-op)."""
        self._is_fitted = True
        return self

    def fit_resample(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Find and remove Tomek links."""
        X, y = np.asarray(X), np.asarray(y)

        tomek_links = self._find_tomek_links(X, y)

        classes, counts = np.unique(y, return_counts=True)
        majority_class = classes[np.argmax(counts)]

        remove_mask = np.zeros(len(X), dtype=bool)

        for i, j in tomek_links:
            if self.sampling_strategy in ['auto', 'majority']:
                if y[i] == majority_class:
                    remove_mask[i] = True
                else:
                    remove_mask[j] = True
            else:
                remove_mask[i] = True
                remove_mask[j] = True

        keep_mask = ~remove_mask
        return X[keep_mask], y[keep_mask]

    def _find_tomek_links(self, X: np.ndarray, y: np.ndarray) -> List[Tuple[int, int]]:
        """Find all Tomek links."""
        n = len(X)
        nn = np.zeros(n, dtype=int)

        for i in range(n):
            dists = np.sqrt(np.sum((X - X[i]) ** 2, axis=1))
            dists[i] = np.inf
            nn[i] = np.argmin(dists)

        tomek_links = []
        for i in range(n):
            j = nn[i]
            if nn[j] == i and y[i] != y[j] and i < j:
                tomek_links.append((i, j))

        return tomek_links

    def transform(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Use fit_resample() instead")

    def __repr__(self) -> str:
        return f"TomekLinksSampler(sampling_strategy='{self.sampling_strategy}')"

class ENNSampler(Transformer):
    """
    Edited Nearest Neighbours (ENN) under-sampling.

    Removes samples whose class differs from the majority of their
    k nearest neighbors.

    Parameters
    ----------
    n_neighbors : int, default=3
        Number of nearest neighbors.
    kind_sel : {'all', 'mode'}, default='all'
        - 'all': All neighbors must agree
        - 'mode': Majority of neighbors must agree
    sampling_strategy : str, default='auto'
        Which classes to clean.

    Examples
    --------
    >>> from tuiml.preprocessing.sampling import ENNSampler
    >>> enn = ENNSampler(n_neighbors=3)
    >>> X_res, y_res = enn.fit_resample(X, y)

    References
    ----------
    Wilson, D. L. (1972). Asymptotic Properties of Nearest Neighbor Rules
    Using Edited Data. IEEE Transactions on Systems, Man, and Cybernetics.
    """

    def __init__(
        self,
        n_neighbors: int = 3,
        kind_sel: str = 'all',
        sampling_strategy: str = 'auto'
    ):
        super().__init__()
        self.n_neighbors = n_neighbors
        self.kind_sel = kind_sel
        self.sampling_strategy = sampling_strategy

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict]:
        """Return JSON Schema for parameters."""
        return {
            "n_neighbors": {
                "type": "integer",
                "default": 3,
                "minimum": 1,
                "description": "Number of nearest neighbors"
            },
            "kind_sel": {
                "type": "string",
                "default": "all",
                "enum": ["all", "mode"],
                "description": "'all' = all neighbors must agree, 'mode' = majority"
            },
            "sampling_strategy": {
                "type": "string",
                "default": "auto",
                "description": "Which classes to clean"
            }
        }

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ENNSampler":
        """Fit (no-op)."""
        self._is_fitted = True
        return self

    def fit_resample(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Remove noisy samples."""
        X, y = np.asarray(X), np.asarray(y)

        classes, counts = np.unique(y, return_counts=True)
        majority_class = classes[np.argmax(counts)]

        keep_mask = np.ones(len(X), dtype=bool)

        for i in range(len(X)):
            if self.sampling_strategy == 'auto' and y[i] == majority_class:
                # Only clean majority class
                pass
            elif self.sampling_strategy != 'auto':
                pass
            else:
                continue

            # Find k nearest neighbors
            dists = np.sqrt(np.sum((X - X[i]) ** 2, axis=1))
            dists[i] = np.inf
            nn_idx = np.argsort(dists)[:self.n_neighbors]
            nn_labels = y[nn_idx]

            if self.kind_sel == 'all':
                # All neighbors must have same class
                if not np.all(nn_labels == y[i]):
                    keep_mask[i] = False
            else:
                # Majority of neighbors
                if np.sum(nn_labels == y[i]) < self.n_neighbors / 2:
                    keep_mask[i] = False

        return X[keep_mask], y[keep_mask]

    def transform(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Use fit_resample() instead")

    def __repr__(self) -> str:
        return f"ENNSampler(n_neighbors={self.n_neighbors})"

class CNNSampler(Transformer):
    """
    Condensed Nearest Neighbour (CNN) under-sampling.

    Iteratively removes samples that do not affect the nearest
    neighbor classification rule.

    Parameters
    ----------
    n_neighbors : int, default=1
        Number of nearest neighbors.
    random_state : int, optional
        Random seed.

    Examples
    --------
    >>> from tuiml.preprocessing.sampling import CNNSampler
    >>> cnn = CNNSampler()
    >>> X_res, y_res = cnn.fit_resample(X, y)

    References
    ----------
    Hart, P. (1968). The Condensed Nearest Neighbor Rule. IEEE Transactions
    on Information Theory.
    """

    def __init__(
        self,
        n_neighbors: int = 1,
        random_state: Optional[int] = None
    ):
        super().__init__()
        self.n_neighbors = n_neighbors
        self.random_state = random_state

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict]:
        """Return JSON Schema for parameters."""
        return {
            "n_neighbors": {
                "type": "integer",
                "default": 1,
                "minimum": 1,
                "description": "Number of nearest neighbors"
            },
            "random_state": {
                "type": ["integer", "null"],
                "default": None,
                "description": "Random seed for reproducibility"
            }
        }

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CNNSampler":
        """Fit (no-op)."""
        self._is_fitted = True
        return self

    def fit_resample(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Condense the dataset."""
        X, y = np.asarray(X), np.asarray(y)
        rng = np.random.RandomState(self.random_state)

        classes = np.unique(y)

        # Initialize with one sample from each class
        store_idx = []
        for cls in classes:
            cls_idx = np.where(y == cls)[0]
            store_idx.append(rng.choice(cls_idx))

        # Iteratively add misclassified samples
        changed = True
        while changed:
            changed = False
            for i in range(len(X)):
                if i in store_idx:
                    continue

                X_store = X[store_idx]
                y_store = y[store_idx]

                # Find nearest neighbor in store
                dists = np.sqrt(np.sum((X_store - X[i]) ** 2, axis=1))
                nn_idx = np.argmin(dists)

                # If misclassified, add to store
                if y_store[nn_idx] != y[i]:
                    store_idx.append(i)
                    changed = True

        return X[store_idx], y[store_idx]

    def transform(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Use fit_resample() instead")

    def __repr__(self) -> str:
        return f"CNNSampler(n_neighbors={self.n_neighbors})"

class NearMissSampler(Transformer):
    """
    NearMissSampler under-sampling.

    Selects majority samples based on their distance to minority samples.

    Parameters
    ----------
    version : int, default=1
        Version of NearMissSampler:
        - 1: Select samples with smallest average distance to k nearest minority
        - 2: Select samples with smallest average distance to k farthest minority
        - 3: Select samples with largest average distance to k nearest minority
    n_neighbors : int, default=3
        Number of nearest neighbors.
    sampling_strategy : str or dict, default='auto'
        Sampling strategy.

    Examples
    --------
    >>> from tuiml.preprocessing.sampling import NearMissSampler
    >>> nm = NearMissSampler(version=1)
    >>> X_res, y_res = nm.fit_resample(X, y)

    References
    ----------
    Mani, I., & Zhang, I. (2003). kNN approach to unbalanced data
    distributions: a case study involving information extraction.
    """

    def __init__(
        self,
        version: int = 1,
        n_neighbors: int = 3,
        sampling_strategy: Union[str, dict] = 'auto'
    ):
        super().__init__()
        self.version = version
        self.n_neighbors = n_neighbors
        self.sampling_strategy = sampling_strategy
        self.sampling_strategy_: Dict = {}

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict]:
        """Return JSON Schema for parameters."""
        return {
            "version": {
                "type": "integer",
                "default": 1,
                "enum": [1, 2, 3],
                "description": "NearMissSampler version (1, 2, or 3)"
            },
            "n_neighbors": {
                "type": "integer",
                "default": 3,
                "minimum": 1,
                "description": "Number of nearest neighbors"
            },
            "sampling_strategy": {
                "type": ["string", "object"],
                "default": "auto",
                "description": "Sampling strategy"
            }
        }

    def fit(self, X: np.ndarray, y: np.ndarray) -> "NearMissSampler":
        """Fit the sampler."""
        X, y = np.asarray(X), np.asarray(y)
        classes, counts = np.unique(y, return_counts=True)
        class_counts = dict(zip(classes, counts))
        min_count = counts.min()

        if isinstance(self.sampling_strategy, dict):
            self.sampling_strategy_ = self.sampling_strategy
        else:
            minority = classes[np.argmin(counts)]
            self.sampling_strategy_ = {
                c: min_count for c in classes if c != minority
            }

        self._is_fitted = True
        return self

    def fit_resample(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply NearMissSampler under-sampling."""
        self.fit(X, y)

        classes, counts = np.unique(y, return_counts=True)
        minority_class = classes[np.argmin(counts)]
        X_minority = X[y == minority_class]

        X_res, y_res = [], []

        for cls in classes:
            mask = y == cls
            X_class = X[mask]

            if cls in self.sampling_strategy_:
                n_samples = self.sampling_strategy_[cls]
                if n_samples < len(X_class):
                    selected_idx = self._select_samples(X_class, X_minority, n_samples)
                    X_res.append(X_class[selected_idx])
                    y_res.append(np.full(n_samples, cls))
                else:
                    X_res.append(X_class)
                    y_res.append(np.full(len(X_class), cls))
            else:
                X_res.append(X_class)
                y_res.append(np.full(len(X_class), cls))

        return np.vstack(X_res), np.hstack(y_res)

    def _select_samples(self, X_majority: np.ndarray, X_minority: np.ndarray,
                        n_samples: int) -> np.ndarray:
        """Select samples based on NearMissSampler version."""
        n_majority = len(X_majority)
        scores = np.zeros(n_majority)

        for i, sample in enumerate(X_majority):
            dists = np.sqrt(np.sum((X_minority - sample) ** 2, axis=1))
            sorted_dists = np.sort(dists)

            if self.version == 1:
                # Average distance to k nearest minority
                scores[i] = sorted_dists[:self.n_neighbors].mean()
            elif self.version == 2:
                # Average distance to k farthest minority
                scores[i] = sorted_dists[-self.n_neighbors:].mean()
            else:  # version == 3
                # Average distance to k nearest minority (select largest)
                scores[i] = -sorted_dists[:self.n_neighbors].mean()

        # Select samples with smallest scores
        selected_idx = np.argsort(scores)[:n_samples]
        return selected_idx

    def transform(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Use fit_resample() instead")

    def __repr__(self) -> str:
        return f"NearMissSampler(version={self.version}, n_neighbors={self.n_neighbors})"

class HardnessThresholdSampler(Transformer):
    """
    Instance Hardness Threshold under-sampling.

    Removes samples that are hard to classify based on classifier
    probability estimates.

    Parameters
    ----------
    estimator : object, optional
        Classifier to use. If None, uses RandomForest.
    cv : int, default=5
        Cross-validation folds for probability estimation.
    sampling_strategy : str or dict, default='auto'
        Sampling strategy.
    random_state : int, optional
        Random seed.

    Examples
    --------
    >>> from tuiml.preprocessing.sampling import HardnessThresholdSampler
    >>> iht = HardnessThresholdSampler()
    >>> X_res, y_res = iht.fit_resample(X, y)

    References
    ----------
    Smith, M. R., Martinez, T., & Giraud-Carrier, C. (2014). An instance
    level analysis of data complexity. Machine learning.
    """

    def __init__(
        self,
        estimator=None,
        cv: int = 5,
        sampling_strategy: Union[str, dict] = 'auto',
        random_state: Optional[int] = None
    ):
        super().__init__()
        self.estimator = estimator
        self.cv = cv
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state
        self.sampling_strategy_: Dict = {}

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict]:
        """Return JSON Schema for parameters."""
        return {
            "cv": {
                "type": "integer",
                "default": 5,
                "minimum": 2,
                "description": "Cross-validation folds for probability estimation"
            },
            "sampling_strategy": {
                "type": ["string", "object"],
                "default": "auto",
                "description": "Sampling strategy"
            },
            "random_state": {
                "type": ["integer", "null"],
                "default": None,
                "description": "Random seed for reproducibility"
            }
        }

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HardnessThresholdSampler":
        """Fit the sampler."""
        X, y = np.asarray(X), np.asarray(y)
        classes, counts = np.unique(y, return_counts=True)
        min_count = counts.min()

        if isinstance(self.sampling_strategy, dict):
            self.sampling_strategy_ = self.sampling_strategy
        else:
            minority = classes[np.argmin(counts)]
            self.sampling_strategy_ = {
                c: min_count for c in classes if c != minority
            }

        self._is_fitted = True
        return self

    def fit_resample(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Remove hard instances."""
        self.fit(X, y)

        # Get cross-validated probability estimates
        probas = self._get_probas(X, y)

        classes = np.unique(y)
        X_res, y_res = [], []

        for cls in classes:
            mask = y == cls
            X_class = X[mask]
            probas_class = probas[mask]

            if cls in self.sampling_strategy_:
                n_samples = self.sampling_strategy_[cls]
                if n_samples < len(X_class):
                    # Keep samples with highest probability of correct class
                    cls_idx = np.where(classes == cls)[0][0]
                    scores = probas_class[:, cls_idx]
                    selected_idx = np.argsort(scores)[-n_samples:]
                    X_res.append(X_class[selected_idx])
                    y_res.append(np.full(n_samples, cls))
                else:
                    X_res.append(X_class)
                    y_res.append(np.full(len(X_class), cls))
            else:
                X_res.append(X_class)
                y_res.append(np.full(len(X_class), cls))

        return np.vstack(X_res), np.hstack(y_res)

    def _get_probas(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Get cross-validated probability estimates."""
        n_samples = len(X)
        n_classes = len(np.unique(y))
        probas = np.zeros((n_samples, n_classes))

        rng = np.random.RandomState(self.random_state)
        indices = np.arange(n_samples)
        rng.shuffle(indices)
        folds = np.array_split(indices, self.cv)

        from tuiml.algorithms.trees import RandomForestClassifier
        estimator = self.estimator or RandomForestClassifier(
            n_estimators=50, random_state=self.random_state
        )

        for i, test_idx in enumerate(folds):
            train_idx = np.concatenate([folds[j] for j in range(self.cv) if j != i])
            # Clone: filter out fitted attributes (trailing _) from params
            params = {k: v for k, v in estimator.get_params().items()
                      if not k.endswith('_')}
            est = type(estimator)(**params)
            est.fit(X[train_idx], y[train_idx])
            probas[test_idx] = est.predict_proba(X[test_idx])

        return probas

    def transform(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Use fit_resample() instead")

    def __repr__(self) -> str:
        return f"HardnessThresholdSampler(cv={self.cv})"
