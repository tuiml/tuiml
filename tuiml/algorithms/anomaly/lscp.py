"""LSCP - Locally Selective Combination in Parallel outlier ensembles."""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional

import numpy as np

from tuiml.base.algorithms import Classifier, anomaly_detector


@anomaly_detector(
    tags=["anomaly-detection", "ensemble", "unsupervised", "local"],
    version="1.0.0",
)
class LSCPDetector(Classifier):
    """LSCP picks the **best detector for each point's own neighbourhood**.

    Combining detectors by averaging assumes one of them is right everywhere.
    LSCP does not: it defines a **local region** around each test point and
    chooses, within that region alone, whichever base detector best agrees with
    the ensemble's consensus. A detector that shines in a dense region and
    fails in a sparse one is used only where it works.

    Overview
    --------
    1. Fit a pool of base detectors on the training data.
    2. Standardise their training scores and take a **pseudo ground truth** —
       the per-point maximum across the pool, which is the ensemble's best
       unsupervised guess at which points are anomalies.
    3. For a test point, find its local region: the training points that are
       repeatedly its nearest neighbours across an ensemble of **random
       feature subspaces**.
    4. Within that region, rank the detectors by Pearson correlation with the
       pseudo ground truth, and combine the winners.

    Theory
    ------
    Let :math:`s_c` be detector :math:`c`'s standardised training scores and

    .. math::
        t_i = \\max_c s_{c,i}

    the pseudo ground truth. For a test point with local region
    :math:`\\mathcal{R}`, detector competence is

    .. math::
        \\rho_c = \\mathrm{corr}\\left( s_c[\\mathcal{R}],\\ t[\\mathcal{R}] \\right)

    and the final score is either the single most competent detector
    (``method='maximum'``) or the mean of the top half of the pool
    (``method='average'``).

    The local region is deliberately built from **random subspaces** rather
    than one nearest-neighbour list in the full space. In high dimension a
    single full-space neighbourhood is unstable and nearly meaningless;
    requiring a training point to appear in many independently drawn subspaces
    before it joins the region makes the region far more robust.

    LSCP is **unsupervised throughout** — the pseudo ground truth is derived
    from the detectors themselves, never from labels. That is also its main
    weakness: if the whole pool agrees on something wrong, the consensus
    inherits the error and local selection cannot rescue it. Diversity in the
    pool is what makes the method work.

    Parameters
    ----------
    detectors : list of Classifier, optional
        Pool of unfitted base detectors, each deep-copied before fitting.
        Defaults to four :class:`~tuiml.algorithms.anomaly.KNNDetector`
        instances with ``n_neighbors`` of 5, 10, 20 and 35, matching the
        varying-``k`` pool of the original paper.
    local_region_size : int, default=30
        Number of nearest neighbours drawn per subspace. Larger regions give
        steadier correlations and less locality.
    n_subspaces : int, default=10
        Number of random feature subspaces used to build each local region.
    method : {'average', 'maximum'}, default='average'
        ``'average'`` (LSCP_A) averages the top half of the pool by local
        competence; ``'maximum'`` (LSCP_M) uses the single best detector.
        Averaging is steadier and the better default; maximum is sharper when
        the pool genuinely contains one specialist per region.
    contamination : float, default=0.1
        Expected proportion of outliers. Sets the decision threshold.
    random_state : int, optional
        Seed for the subspace sampling.

    Attributes
    ----------
    detectors_ : list of Classifier
        The fitted base detectors.
    X_train_ : np.ndarray of shape (n_samples, n_features)
        Training data retained for local-region search.
    train_scores_ : np.ndarray of shape (n_samples, n_detectors)
        Standardised training scores, higher meaning more anomalous.
    pseudo_target_ : np.ndarray of shape (n_samples,)
        The per-point maximum across detectors.
    threshold_ : float
        Decision-function value separating inliers from outliers.
    n_features_in_ : int
        Number of features seen during :meth:`fit`.

    Notes
    -----
    **Complexity.** Fitting costs the sum of the pool's fits. Scoring is the
    expensive part: :math:`O(m \\cdot p \\cdot n \\cdot d')` for ``p``
    subspaces of width ``d'``, plus every base detector's own scoring cost.
    Expect LSCP to be roughly an order of magnitude slower than its slowest
    member — it buys accuracy with compute, and there is no way around that.

    **When to use.** LSCP pays off when the data has **regions of genuinely
    different character** — mixed density, several clusters with different
    shapes — and no single detector wins everywhere. On homogeneous data a
    plain average of the same pool performs just as well for a fraction of the
    cost, so benchmark against that baseline before adopting it. Give it a
    diverse pool; a pool of near-identical detectors leaves nothing to select
    between.

    References
    ----------
    .. [Zhao2019] Zhao, Y., Nasrullah, Z., Hryniewicki, M. K., & Li, Z.
       (2019). LSCP: Locally Selective Combination in Parallel Outlier
       Ensembles. *SIAM International Conference on Data Mining (SDM)*,
       585-593. :doi:`10.1137/1.9781611975673.66`

    See Also
    --------
    :class:`~tuiml.algorithms.anomaly.KNNDetector` : The default pool member.
    :class:`~tuiml.algorithms.anomaly.LocalOutlierFactorDetector` : Also local, but a single model.
    :class:`~tuiml.algorithms.anomaly.ECODDetector` : A fast, diverse addition to a custom pool.

    Examples
    --------
    >>> import numpy as np
    >>> from tuiml.algorithms.anomaly import LSCPDetector
    >>> rng = np.random.default_rng(0)
    >>> X = np.vstack([rng.normal(0, 1, (200, 4)), rng.normal(7, 1, (10, 4))])
    >>> detector = LSCPDetector(contamination=0.05, random_state=0).fit(X)
    >>> int((detector.predict(X)[-10:] == -1).sum())
    10

    A custom, deliberately diverse pool:

    >>> from tuiml.algorithms.anomaly import ECODDetector, KNNDetector
    >>> pool = [ECODDetector(), KNNDetector(n_neighbors=10),
    ...         KNNDetector(n_neighbors=30)]
    >>> detector = LSCPDetector(detectors=pool, random_state=0).fit(X)
    >>> len(detector.detectors_)
    3
    """

    def __init__(
        self,
        detectors: Optional[List[Any]] = None,
        local_region_size: int = 30,
        n_subspaces: int = 10,
        method: str = "average",
        contamination: float = 0.1,
        random_state: Optional[int] = None,
    ):
        """Initialize the LSCP ensemble detector.

        Parameters
        ----------
        detectors : list of Classifier, optional
            Pool of unfitted base detectors.
        local_region_size : int, default=30
            Nearest neighbours drawn per subspace.
        n_subspaces : int, default=10
            Number of random feature subspaces.
        method : {'average', 'maximum'}, default='average'
            Local combination rule.
        contamination : float, default=0.1
            Expected proportion of outliers.
        random_state : int, optional
            Seed for the subspace sampling.
        """
        super().__init__()
        if method not in ("average", "maximum"):
            raise ValueError(
                f"method must be 'average' or 'maximum', got {method!r}"
            )
        if detectors is not None and len(detectors) < 2:
            raise ValueError(
                f"LSCP needs at least 2 detectors to select between, got "
                f"{len(detectors)}"
            )
        self.detectors = detectors
        self.local_region_size = local_region_size
        self.n_subspaces = n_subspaces
        self.method = method
        self.contamination = contamination
        self.random_state = random_state

        # Fitted attributes
        self.detectors_ = None
        self.X_train_ = None
        self.train_scores_ = None
        self.pseudo_target_ = None
        self.threshold_ = None
        self.n_features_in_ = None
        self._raw_mean_ = None
        self._raw_std_ = None

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        """Return JSON Schema for algorithm parameters."""
        return {
            "local_region_size": {
                "type": "integer",
                "default": 30,
                "minimum": 2,
                "description": "Nearest neighbors drawn per random subspace"
            },
            "n_subspaces": {
                "type": "integer",
                "default": 10,
                "minimum": 1,
                "description": "Number of random feature subspaces per local region"
            },
            "method": {
                "type": "string",
                "enum": ["average", "maximum"],
                "default": "average",
                "description": "Local combination rule: LSCP_A or LSCP_M"
            },
            "contamination": {
                "type": "number",
                "default": 0.1,
                "minimum": 0.0,
                "maximum": 0.5,
                "description": "Expected proportion of outliers in the dataset"
            },
            "random_state": {
                "type": ["integer", "null"],
                "default": None,
                "description": "Random seed for subspace sampling"
            }
        }

    @classmethod
    def get_capabilities(cls) -> List[str]:
        """Return supported capabilities."""
        return ["numeric", "binary_class", "unsupervised", "anomaly_detection", "ensemble"]

    @classmethod
    def get_complexity(cls) -> str:
        """Return complexity analysis."""
        return "Training: sum of base detectors, Prediction: O(m*p*n*d') plus base scoring"

    @classmethod
    def get_references(cls) -> List[str]:
        """Return academic citations."""
        return [
            "Zhao, Y., Nasrullah, Z., Hryniewicki, M.K. and Li, Z., 2019. "
            "LSCP: Locally selective combination in parallel outlier ensembles. SDM."
        ]

    def fit(self, X: np.ndarray, _y: Optional[np.ndarray] = None) -> "LSCPDetector":
        """Fit the base detectors and build the pseudo ground truth.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data. Labels are ignored; the method is unsupervised.
        _y : np.ndarray, optional
            Ignored, present for API consistency.

        Returns
        -------
        self : LSCPDetector
            The fitted detector.
        """
        X = np.ascontiguousarray(np.atleast_2d(X), dtype=np.float64)
        self.X_train_ = X
        self.n_features_in_ = X.shape[1]

        pool = self._resolve_pool()
        self.detectors_ = [copy.deepcopy(detector).fit(X) for detector in pool]

        raw = np.column_stack(
            [-np.asarray(d.decision_function(X)) for d in self.detectors_]
        )
        self._raw_mean_ = raw.mean(axis=0, keepdims=True)
        self._raw_std_ = raw.std(axis=0, keepdims=True)
        self.train_scores_ = np.divide(
            raw - self._raw_mean_,
            self._raw_std_,
            out=np.zeros_like(raw),
            where=self._raw_std_ > 0,
        )
        # The ensemble's own best guess at which points are anomalies. Taking
        # the maximum rather than the mean keeps a point that any one detector
        # finds extreme, which is the behaviour an outlier target wants.
        self.pseudo_target_ = self.train_scores_.max(axis=1)

        scores = self._outlier_scores(X)
        self.threshold_ = float(
            np.percentile(-scores, 100.0 * self.contamination)
        )
        self._is_fitted = True
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Compute anomaly scores for samples.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        scores : np.ndarray of shape (n_samples,)
            Anomaly scores. Lower scores indicate anomalies.
        """
        self._check_is_fitted()
        X = np.ascontiguousarray(np.atleast_2d(X), dtype=np.float64)
        return -self._outlier_scores(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict if samples are anomalies or not.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        predictions : np.ndarray of shape (n_samples,)
            -1 for anomalies, 1 for normal instances.
        """
        self._check_is_fitted()
        return np.where(self.decision_function(X) >= self.threshold_, 1, -1)

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """Alias for decision_function for compatibility.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        scores : np.ndarray of shape (n_samples,)
            Anomaly scores. Lower scores indicate anomalies.
        """
        return self.decision_function(X)

    def local_competence(self, X: np.ndarray) -> np.ndarray:
        """Return each detector's local competence for each sample.

        Exposes the selection LSCP makes internally, which is the useful
        diagnostic: it shows which pool member is trusted where, and whether
        the pool is diverse enough for the selection to mean anything.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input samples.

        Returns
        -------
        competence : np.ndarray of shape (n_samples, n_detectors)
            Pearson correlation with the pseudo ground truth inside each
            sample's local region. Detectors whose local scores are constant
            score 0.
        """
        self._check_is_fitted()
        X = np.ascontiguousarray(np.atleast_2d(X), dtype=np.float64)
        regions = self._local_regions(X)
        return np.vstack(
            [self._competence(region) for region in regions]
        )

    def _resolve_pool(self) -> List[Any]:
        """Return the pool of base detectors to fit.

        Returns
        -------
        pool : list of Classifier
            Either the caller's detectors, or the default varying-k kNN pool.
        """
        if self.detectors is not None:
            return list(self.detectors)

        from tuiml.algorithms.anomaly.knn_detector import KNNDetector

        # Varying k gives locality diversity: small k reacts to isolated
        # points, large k to sparse regions. That spread is what LSCP selects
        # between.
        return [KNNDetector(n_neighbors=k) for k in (5, 10, 20, 35)]

    def _local_regions(self, X: np.ndarray) -> List[np.ndarray]:
        """Build each sample's local region from random feature subspaces.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Samples whose neighbourhoods are needed.

        Returns
        -------
        regions : list of np.ndarray
            Training-set indices forming each sample's local region.
        """
        from tuiml._cpp_ext import distance as _cpp_distance

        rng = np.random.default_rng(self.random_state)
        n_train, n_features = self.X_train_.shape
        region_size = min(self.local_region_size, n_train)

        # Count how often each training point is a neighbour across subspaces.
        counts = np.zeros((len(X), n_train), dtype=np.int32)
        for _ in range(self.n_subspaces):
            width = rng.integers(max(1, n_features // 2), n_features + 1)
            columns = rng.choice(n_features, size=int(width), replace=False)

            distances = np.asarray(
                _cpp_distance.euclidean(
                    np.ascontiguousarray(X[:, columns]),
                    np.ascontiguousarray(self.X_train_[:, columns]),
                )
            )
            nearest = np.argpartition(distances, region_size - 1, axis=1)[
                :, :region_size
            ]
            np.put_along_axis(
                counts, nearest, np.take_along_axis(counts, nearest, axis=1) + 1,
                axis=1,
            )

        # A training point joins the region only if it survived repeated
        # independent subspace draws — that stability is the point of the
        # subspace ensemble.
        regions = []
        for row in counts:
            keep = row >= max(1, self.n_subspaces // 2)
            if keep.sum() < 2:
                # Too unstable to correlate: fall back to the plain top-k.
                keep = np.zeros_like(keep)
                keep[np.argsort(-row)[:region_size]] = True
            regions.append(np.flatnonzero(keep))
        return regions

    def _competence(self, region: np.ndarray) -> np.ndarray:
        """Correlate each detector with the pseudo target inside one region.

        Parameters
        ----------
        region : np.ndarray of shape (region_size,)
            Training indices forming the local region.

        Returns
        -------
        competence : np.ndarray of shape (n_detectors,)
            Pearson correlation per detector; 0 where undefined.
        """
        local_scores = self.train_scores_[region]
        local_target = self.pseudo_target_[region]

        target_centred = local_target - local_target.mean()
        scores_centred = local_scores - local_scores.mean(axis=0, keepdims=True)

        numerator = scores_centred.T @ target_centred
        denominator = (
            np.linalg.norm(scores_centred, axis=0) * np.linalg.norm(target_centred)
        )
        # A constant detector or a constant target leaves correlation
        # undefined; treating it as zero competence keeps it out of the
        # selection without special-casing downstream.
        return np.divide(
            numerator,
            denominator,
            out=np.zeros(local_scores.shape[1]),
            where=denominator > 0,
        )

    def _outlier_scores(self, X: np.ndarray) -> np.ndarray:
        """Compute the raw LSCP score, where higher means more anomalous.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Samples to score.

        Returns
        -------
        scores : np.ndarray of shape (n_samples,)
            Locally selected combination of the base detectors' scores.
        """
        raw = np.column_stack(
            [-np.asarray(d.decision_function(X)) for d in self.detectors_]
        )
        # Standardise against the training distribution, not this batch, so a
        # test batch's own composition cannot shift the scores.
        scores = np.divide(
            raw - self._raw_mean_,
            self._raw_std_,
            out=np.zeros_like(raw),
            where=self._raw_std_ > 0,
        )

        regions = self._local_regions(X)
        n_detectors = scores.shape[1]
        n_selected = max(1, n_detectors // 2)

        out = np.empty(len(X))
        for i, region in enumerate(regions):
            competence = self._competence(region)
            if self.method == "maximum":
                out[i] = scores[i, int(np.argmax(competence))]
            else:
                best = np.argsort(-competence)[:n_selected]
                out[i] = scores[i, best].mean()
        return out

    def __repr__(self) -> str:
        """Return a readable representation of the detector."""
        size = len(self.detectors_) if self.detectors_ is not None else "unfitted"
        return (
            f"LSCPDetector(n_detectors={size}, method={self.method!r}, "
            f"local_region_size={self.local_region_size})"
        )
