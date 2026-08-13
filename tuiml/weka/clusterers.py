"""Weka clusterer wrappers.

Unsupervised learners from ``weka.clusterers``, registered under
``weka.<ClassName>`` hub keys.

Notes
-----
Requires the optional Weka extra: ``pip install 'tuiml[weka]'`` plus a Java
runtime (11+) on ``PATH``.
"""

from typing import Any, Dict, List, Optional, Sequence

from tuiml.base.algorithms import Clusterer
from tuiml.weka._base import _WekaClustererMixin, fmt_num, weka_clusterer

__all__ = [
    "SimpleKMeans",
    "EM",
    "Canopy",
    "FarthestFirst",
    "Cobweb",
    "HierarchicalClusterer",
]


@weka_clusterer(tags=["clustering", "kmeans", "partitional"])
class SimpleKMeans(_WekaClustererMixin, Clusterer):
    """**SimpleKMeans** — k-means clustering (hub key ``weka.SimpleKMeans``).

    Wraps ``weka.clusterers.SimpleKMeans``.

    Parameters
    ----------
    n_clusters : int, default=2
        Number of clusters (Weka ``-N``).
    max_iterations : int, default=500
        Maximum iterations (Weka ``-I``).
    seed : int, default=10
        Random seed (Weka ``-S``).
    nominal_features : sequence of int or None, default=None
        0-based indices of columns to treat as categorical.
    options : sequence of str or None, default=None
        Extra raw Weka options.

    Attributes
    ----------
    labels_ : np.ndarray of shape (n_samples,)
        Cluster assignment for each training row.
    n_clusters_ : int
        Number of clusters produced.

    See Also
    --------
    :class:`~tuiml.algorithms.clustering.KMeansClusterer` : TuiML's native k-means.

    Examples
    --------
    >>> from tuiml.weka import SimpleKMeans
    >>> from tuiml.datasets import load_iris
    >>> data = load_iris()
    >>> km = SimpleKMeans(n_clusters=3, seed=42).fit(data.X)
    >>> km.labels_.shape
    (150,)
    """

    _weka_classname = "weka.clusterers.SimpleKMeans"

    def __init__(
        self,
        n_clusters: int = 2,
        max_iterations: int = 500,
        seed: int = 10,
        nominal_features: Optional[Sequence[int]] = None,
        options: Optional[Sequence[str]] = None,
    ):
        super().__init__()
        self.n_clusters = n_clusters
        self.max_iterations = max_iterations
        self.seed = seed
        self.nominal_features = nominal_features
        self.options = options

    def _options(self) -> List[str]:
        """Return the Weka option tokens for this configuration."""
        return ["-N", fmt_num(self.n_clusters), "-I", fmt_num(self.max_iterations),
                "-S", fmt_num(self.seed)]

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Any]:
        """Return JSON Schema for constructor parameters."""
        return {
            "n_clusters": {"type": "integer", "default": 2, "minimum": 1},
            "max_iterations": {"type": "integer", "default": 500, "minimum": 1},
            "seed": {"type": "integer", "default": 10},
        }

    @classmethod
    def get_capabilities(cls) -> List[str]:
        """Return the capability names this wrapper supports."""
        return ["numeric", "nominal", "clustering"]


@weka_clusterer(tags=["clustering", "em", "probabilistic"])
class EM(_WekaClustererMixin, Clusterer):
    """**EM** — expectation-maximization clustering (hub key ``weka.EM``).

    Wraps ``weka.clusterers.EM``. Fits a mixture of Gaussians and can choose the
    number of clusters itself by cross-validation.

    Parameters
    ----------
    n_clusters : int, default=-1
        Number of clusters (Weka ``-N``); ``-1`` selects it by cross-validation.
    max_iterations : int, default=100
        Maximum EM iterations (Weka ``-I``).
    seed : int, default=100
        Random seed (Weka ``-S``).
    nominal_features : sequence of int or None, default=None
        0-based indices of columns to treat as categorical.
    options : sequence of str or None, default=None
        Extra raw Weka options.

    See Also
    --------
    :class:`~tuiml.algorithms.clustering.GaussianMixtureClusterer` : TuiML's native version.

    Examples
    --------
    >>> from tuiml.weka import EM
    >>> from tuiml.datasets import load_iris
    >>> data = load_iris()
    >>> em = EM(n_clusters=3, seed=42).fit(data.X)
    >>> em.n_clusters_
    3
    """

    _weka_classname = "weka.clusterers.EM"

    def __init__(
        self,
        n_clusters: int = -1,
        max_iterations: int = 100,
        seed: int = 100,
        nominal_features: Optional[Sequence[int]] = None,
        options: Optional[Sequence[str]] = None,
    ):
        super().__init__()
        self.n_clusters = n_clusters
        self.max_iterations = max_iterations
        self.seed = seed
        self.nominal_features = nominal_features
        self.options = options

    def _options(self) -> List[str]:
        """Return the Weka option tokens for this configuration."""
        return ["-N", fmt_num(self.n_clusters), "-I", fmt_num(self.max_iterations),
                "-S", fmt_num(self.seed)]

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Any]:
        """Return JSON Schema for constructor parameters."""
        return {
            "n_clusters": {"type": "integer", "default": -1},
            "max_iterations": {"type": "integer", "default": 100, "minimum": 1},
            "seed": {"type": "integer", "default": 100},
        }

    @classmethod
    def get_capabilities(cls) -> List[str]:
        """Return the capability names this wrapper supports."""
        return ["numeric", "nominal", "clustering"]


@weka_clusterer(tags=["clustering", "approximate", "preclustering"])
class Canopy(_WekaClustererMixin, Clusterer):
    """**Canopy** — cheap approximate pre-clustering (hub key ``weka.Canopy``).

    Wraps ``weka.clusterers.Canopy``. Uses two distance thresholds to form
    overlapping "canopies" in a single cheap pass, normally as a seeding step
    for a more expensive algorithm.

    Parameters
    ----------
    n_clusters : int, default=-1
        Requested number of clusters (Weka ``-N``); ``-1`` lets the thresholds
        decide.
    t2 : float, default=-1.0
        Loose distance threshold (Weka ``-t2``); a negative value asks Weka to
        derive it from the data.
    t1 : float, default=-1.25
        Tight distance threshold (Weka ``-t1``); a negative value is
        interpreted as a multiple of ``t2``.
    seed : int, default=1
        Random seed (Weka ``-S``).
    nominal_features : sequence of int or None, default=None
        0-based indices of columns to treat as categorical.
    options : sequence of str or None, default=None
        Extra raw Weka options.

    References
    ----------
    .. [McCallum2000] McCallum, A., Nigam, K. and Ungar, L.H. (2000).
           **Efficient Clustering of High-Dimensional Data Sets with Application
           to Reference Matching.** *KDD '00*, 169-178.

    Examples
    --------
    >>> from tuiml.weka import Canopy
    >>> from tuiml.datasets import load_iris
    >>> data = load_iris()
    >>> cp = Canopy(n_clusters=3, seed=42).fit(data.X)
    >>> cp.labels_.shape
    (150,)
    """

    _weka_classname = "weka.clusterers.Canopy"

    def __init__(
        self,
        n_clusters: int = -1,
        t2: float = -1.0,
        t1: float = -1.25,
        seed: int = 1,
        nominal_features: Optional[Sequence[int]] = None,
        options: Optional[Sequence[str]] = None,
    ):
        super().__init__()
        self.n_clusters = n_clusters
        self.t2 = t2
        self.t1 = t1
        self.seed = seed
        self.nominal_features = nominal_features
        self.options = options

    def _options(self) -> List[str]:
        """Return the Weka option tokens for this configuration."""
        return ["-N", fmt_num(self.n_clusters), "-t2", str(self.t2),
                "-t1", str(self.t1), "-S", fmt_num(self.seed)]

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Any]:
        """Return JSON Schema for constructor parameters."""
        return {
            "n_clusters": {"type": "integer", "default": -1},
            "t2": {"type": "number", "default": -1.0},
            "t1": {"type": "number", "default": -1.25},
            "seed": {"type": "integer", "default": 1},
        }

    @classmethod
    def get_capabilities(cls) -> List[str]:
        """Return the capability names this wrapper supports."""
        return ["numeric", "nominal", "clustering"]


@weka_clusterer(tags=["clustering", "seeding", "partitional"])
class FarthestFirst(_WekaClustererMixin, Clusterer):
    """**FarthestFirst** — farthest-first traversal clustering (hub key ``weka.FarthestFirst``).

    Wraps ``weka.clusterers.FarthestFirst``. Picks each new centre as the point
    farthest from the existing ones, which is fast and gives well-separated
    seeds for k-means.

    Parameters
    ----------
    n_clusters : int, default=2
        Number of clusters (Weka ``-N``).
    seed : int, default=1
        Random seed (Weka ``-S``).
    nominal_features : sequence of int or None, default=None
        0-based indices of columns to treat as categorical.
    options : sequence of str or None, default=None
        Extra raw Weka options.

    See Also
    --------
    :class:`~tuiml.algorithms.clustering.KMeansClusterer` : TuiML's native centroid-based clusterer.

    Examples
    --------
    >>> from tuiml.weka import FarthestFirst
    >>> from tuiml.datasets import load_iris
    >>> data = load_iris()
    >>> ff = FarthestFirst(n_clusters=3, seed=42).fit(data.X)
    >>> ff.labels_.shape
    (150,)
    """

    _weka_classname = "weka.clusterers.FarthestFirst"

    def __init__(
        self,
        n_clusters: int = 2,
        seed: int = 1,
        nominal_features: Optional[Sequence[int]] = None,
        options: Optional[Sequence[str]] = None,
    ):
        super().__init__()
        self.n_clusters = n_clusters
        self.seed = seed
        self.nominal_features = nominal_features
        self.options = options

    def _options(self) -> List[str]:
        """Return the Weka option tokens for this configuration."""
        return ["-N", fmt_num(self.n_clusters), "-S", fmt_num(self.seed)]

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Any]:
        """Return JSON Schema for constructor parameters."""
        return {
            "n_clusters": {"type": "integer", "default": 2, "minimum": 1},
            "seed": {"type": "integer", "default": 1},
        }

    @classmethod
    def get_capabilities(cls) -> List[str]:
        """Return the capability names this wrapper supports."""
        return ["numeric", "clustering"]


@weka_clusterer(tags=["clustering", "incremental", "conceptual"])
class Cobweb(_WekaClustererMixin, Clusterer):
    """**Cobweb** — incremental conceptual clustering (hub key ``weka.Cobweb``).

    Wraps ``weka.clusterers.Cobweb``. Builds a hierarchy one instance at a time,
    guided by category utility. The number of clusters emerges from the data
    rather than being requested.

    Parameters
    ----------
    acuity : float, default=1.0
        Minimum standard deviation allowed for a numeric attribute
        (Weka ``-A``).
    cutoff : float, default=0.002
        Category utility threshold below which a node is pruned (Weka ``-C``).
    seed : int, default=42
        Random seed (Weka ``-S``).
    nominal_features : sequence of int or None, default=None
        0-based indices of columns to treat as categorical.
    options : sequence of str or None, default=None
        Extra raw Weka options.

    See Also
    --------
    :class:`~tuiml.algorithms.clustering.AgglomerativeClusterer` : TuiML's native hierarchical clusterer.

    Examples
    --------
    >>> from tuiml.weka import Cobweb
    >>> from tuiml.datasets import load_iris
    >>> data = load_iris()
    >>> cw = Cobweb().fit(data.X)
    >>> cw.labels_.shape
    (150,)
    """

    _weka_classname = "weka.clusterers.Cobweb"

    def __init__(
        self,
        acuity: float = 1.0,
        cutoff: float = 0.002,
        seed: int = 42,
        nominal_features: Optional[Sequence[int]] = None,
        options: Optional[Sequence[str]] = None,
    ):
        super().__init__()
        self.acuity = acuity
        self.cutoff = cutoff
        self.seed = seed
        self.nominal_features = nominal_features
        self.options = options

    def _options(self) -> List[str]:
        """Return the Weka option tokens for this configuration."""
        return ["-A", str(self.acuity), "-C", str(self.cutoff),
                "-S", fmt_num(self.seed)]

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Any]:
        """Return JSON Schema for constructor parameters."""
        return {
            "acuity": {"type": "number", "default": 1.0, "minimum": 0.0},
            "cutoff": {"type": "number", "default": 0.002, "minimum": 0.0},
            "seed": {"type": "integer", "default": 42},
        }

    @classmethod
    def get_capabilities(cls) -> List[str]:
        """Return the capability names this wrapper supports."""
        return ["numeric", "nominal", "clustering"]


@weka_clusterer(tags=["clustering", "hierarchical", "agglomerative"])
class HierarchicalClusterer(_WekaClustererMixin, Clusterer):
    """**HierarchicalClusterer** — agglomerative clustering (hub key ``weka.HierarchicalClusterer``).

    Wraps ``weka.clusterers.HierarchicalClusterer``. Merges the closest pair of
    clusters repeatedly under the chosen linkage until the requested number of
    clusters remains.

    Parameters
    ----------
    n_clusters : int, default=2
        Number of clusters (Weka ``-N``).
    link_type : str, default="SINGLE"
        Linkage criterion (Weka ``-L``): one of ``SINGLE``, ``COMPLETE``,
        ``AVERAGE``, ``MEAN``, ``CENTROID``, ``WARD``, ``ADJCOMPLETE``,
        ``NEIGHBOR_JOINING``.
    nominal_features : sequence of int or None, default=None
        0-based indices of columns to treat as categorical.
    options : sequence of str or None, default=None
        Extra raw Weka options.

    See Also
    --------
    :class:`~tuiml.algorithms.clustering.AgglomerativeClusterer` : TuiML's native version.

    Examples
    --------
    >>> from tuiml.weka import HierarchicalClusterer
    >>> from tuiml.datasets import load_iris
    >>> data = load_iris()
    >>> hc = HierarchicalClusterer(n_clusters=3, link_type="COMPLETE").fit(data.X)
    >>> hc.labels_.shape
    (150,)
    """

    _weka_classname = "weka.clusterers.HierarchicalClusterer"

    def __init__(
        self,
        n_clusters: int = 2,
        link_type: str = "SINGLE",
        nominal_features: Optional[Sequence[int]] = None,
        options: Optional[Sequence[str]] = None,
    ):
        super().__init__()
        self.n_clusters = n_clusters
        self.link_type = link_type
        self.nominal_features = nominal_features
        self.options = options

    def _options(self) -> List[str]:
        """Return the Weka option tokens for this configuration."""
        return ["-N", fmt_num(self.n_clusters), "-L", str(self.link_type)]

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Any]:
        """Return JSON Schema for constructor parameters."""
        return {
            "n_clusters": {"type": "integer", "default": 2, "minimum": 1},
            "link_type": {"type": "string", "default": "SINGLE",
                          "enum": ["SINGLE", "COMPLETE", "AVERAGE", "MEAN",
                                   "CENTROID", "WARD", "ADJCOMPLETE",
                                   "NEIGHBOR_JOINING"]},
        }

    @classmethod
    def get_capabilities(cls) -> List[str]:
        """Return the capability names this wrapper supports."""
        return ["numeric", "clustering"]
