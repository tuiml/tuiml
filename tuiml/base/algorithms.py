"""
Base classes and registration decorators for machine learning algorithms.

This module provides the foundation for the plugin-based algorithm system.
All base classes (``Classifier``, ``Clusterer``, ``Regressor``, ``Associator``)
are defined here and integrate with the component registry (``tuiml.registry``).
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Type, FrozenSet, Union
from dataclasses import dataclass
import numpy as np
import threading
import time
import asyncio

from tuiml.registry import registry, ComponentType, Registrable

def call_metric(metric_func, y_true, y_pred):
    """Call a metric function, choosing macro averaging for multiclass data.

    TuiML's ``f1_score``/``precision_score``/``recall_score`` default to
    binary averaging, which on multiclass labels silently reports only the
    score of class 1. Whenever the metric accepts an ``average`` argument and
    the data has more than two classes, macro averaging is used instead.

    Parameters
    ----------
    metric_func : callable
        A metric function from ``tuiml.evaluation.metrics``.
    y_true : array-like of shape (n_samples,)
        True labels or target values.
    y_pred : array-like of shape (n_samples,)
        Predicted labels or values.

    Returns
    -------
    float or np.ndarray
        The metric value.
    """
    import inspect

    try:
        parameters = inspect.signature(metric_func).parameters.values()
        # Metrics declare ``average`` either as a named parameter or behind
        # ``**kwargs`` (as TuiML's f1_score does).
        may_take_average = any(
            p.name == "average" or p.kind is inspect.Parameter.VAR_KEYWORD
            for p in parameters
        )
    except (TypeError, ValueError):
        may_take_average = False

    # Multiclass is decided by the union of true and predicted labels: a
    # fold whose truths span two classes is still multiclass when the model
    # predicts a third.
    if may_take_average and len(np.union1d(np.unique(y_true), np.unique(y_pred))) > 2:
        try:
            return metric_func(y_true, y_pred, average="macro")
        except TypeError:
            pass  # took **kwargs but not ``average``: use the plain call
    return metric_func(y_true, y_pred)


# =============================================================================
# Algorithm Base Class
# =============================================================================

class Algorithm(Registrable, ABC):
    """Abstract base class for all machine learning algorithms.

    Provides a unified interface for model lifecycle management, including
    training, prediction, evaluation, persistence, REST serving, and
    registration with the component registry.

    Overview
    --------
    This class serves as the foundation for the specialized base classes
    such as :class:`Classifier` and :class:`Regressor`. It defines standard
    methods for metadata retrieval, parameter handling, and REST serving.

    Attributes
    ----------
    _is_fitted : bool
        Internal flag indicating whether the model has been trained.

    Notes
    -----
    Subclasses MUST implement :meth:`fit` and :meth:`predict`.

    See Also
    --------
    :class:`~tuiml.base.algorithms.Classifier` : Base class for classification.
    :class:`~tuiml.base.algorithms.Regressor` : Base class for regression.
    :class:`~tuiml.base.algorithms.Clusterer` : Base class for clustering.
    :class:`~tuiml.base.algorithms.Associator` : Base class for association rule mining.
    """

    _component_type = ComponentType.ALGORITHM

    def __init__(self):
        """Initialize algorithm."""
        self._is_fitted = False

    @classmethod
    def get_metadata(cls) -> Dict[str, Any]:
        """Return algorithm metadata for registration.

        Returns
        -------
        metadata : dict
            Dictionary with the algorithm's name, type, description,
            parameter schema, capabilities, complexity, and references.
        """
        return {
            "name": cls.__name__,
            "type": getattr(cls, "_algorithm_type", "unknown"),
            "description": cls.__doc__ or "No description available",
            "parameters": cls.get_parameter_schema(),
            "capabilities": cls.get_capabilities(),
            "complexity": cls.get_complexity(),
            "references": cls.get_references(),
        }

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Dict[str, Any]]:
        """Return JSON Schema for algorithm parameters.

        Returns
        -------
        schema : dict
            Dictionary mapping parameter names to their JSON Schemas,
            for example::

                {
                    "n_trees": {
                        "type": "integer",
                        "default": 100,
                        "minimum": 1,
                        "maximum": 1000,
                        "description": "Number of trees in the forest"
                    }
                }
        """
        return {}

    @classmethod
    def get_capabilities(cls) -> List[str]:
        """Return the list of algorithm capabilities.

        Returns
        -------
        capabilities : list of str
            Capability strings, for example::

                ["numeric", "nominal", "missing_values", "binary_class", "multiclass"]
        """
        return []

    @classmethod
    def get_complexity(cls) -> str:
        """Return the algorithm's time/space complexity.

        Returns
        -------
        complexity : str
            String describing complexity, e.g. ``"O(n * m * log(n))"``.
        """
        return "Not specified"

    @classmethod
    def get_references(cls) -> List[str]:
        """Return the list of academic references.

        Returns
        -------
        references : list of str
            Citation strings, for example::

                ["Breiman, L. (2001). Random Forests. Machine Learning, 45(1), 5-32."]
        """
        return []

    @abstractmethod
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "Algorithm":
        """Train the algorithm on data.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training features.
        y : np.ndarray of shape (n_samples,), optional
            Training labels or target values. Optional for unsupervised
            algorithms.

        Returns
        -------
        self : Algorithm
            The fitted estimator (for method chaining).
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions on new data.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Test features.

        Returns
        -------
        predictions : np.ndarray of shape (n_samples,)
            Predicted labels or values.
        """
        pass

    def fit_predict(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """Fit and predict in one step (useful for clustering).

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training features.
        y : np.ndarray of shape (n_samples,), optional
            Training labels or target values.

        Returns
        -------
        predictions : np.ndarray of shape (n_samples,)
            Predictions on the training data.
        """
        self.fit(X, y)
        return self.predict(X)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return the default score of the model on the given data.

        Classifiers return **accuracy**, regressors return **R²**. Other
        component types (clusterers, associators) have no single default
        score and must use :meth:`evaluate` with explicit metrics.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Test features.
        y : np.ndarray of shape (n_samples,)
            True labels or target values.

        Returns
        -------
        score : float
            Accuracy for classifiers, R² for regressors.
        """
        from tuiml.evaluation import metrics as metrics_module

        y_pred = self.predict(X)
        # Anomaly detectors predict a discrete inlier/outlier label, so the
        # classification score applies to them unchanged.
        if self._component_type in (ComponentType.CLASSIFIER, ComponentType.ANOMALY):
            return float(metrics_module.accuracy_score(y, y_pred))
        if self._component_type == ComponentType.REGRESSOR:
            return float(metrics_module.r2_score(y, y_pred))
        raise NotImplementedError(
            f"{self.__class__.__name__} has no default score; use "
            f"evaluate(X, y, metrics=[...]) with explicit metric names."
        )

    def evaluate(
        self, X: np.ndarray, y: np.ndarray,
        metrics: Union[str, List[str]] = "auto",
    ) -> Dict[str, float]:
        """Evaluate the model on test data with one or more metrics.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Test features.
        y : np.ndarray of shape (n_samples,)
            True labels or target values.
        metrics : str or list of str, default="auto"
            Metric function names from ``tuiml.evaluation.metrics``
            (e.g., ``["accuracy_score", "f1_score"]``). ``"auto"`` selects
            accuracy/F1 for classifiers and MSE/R² for regressors.

        Returns
        -------
        results : dict
            Mapping of metric names to computed values.
        """
        from tuiml.evaluation import metrics as metrics_module

        y_pred = self.predict(X)

        if metrics == "auto":
            if self._component_type in (ComponentType.CLASSIFIER, ComponentType.ANOMALY):
                metrics = ["accuracy_score", "f1_score"]
            elif self._component_type == ComponentType.REGRESSOR:
                metrics = ["mean_squared_error", "r2_score"]
            else:
                raise NotImplementedError(
                    f"{self.__class__.__name__} has no auto metrics; pass "
                    f"explicit metric names."
                )
        elif isinstance(metrics, str):
            metrics = [metrics]

        results = {}
        for metric_name in metrics:
            metric_func = getattr(metrics_module, metric_name, None)
            if metric_func is None:
                raise ValueError(
                    f"Unknown metric '{metric_name}'. Use a function name "
                    f"from tuiml.evaluation.metrics."
                )
            results[metric_name] = call_metric(metric_func, y, y_pred)
        return results

    def save(self, path: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Save the model to disk.

        Parameters
        ----------
        path : str
            Target file path (e.g., ``"model.pkl"``).
        metadata : dict, optional
            Additional metadata stored alongside the model.
        """
        from tuiml.utils.serialization import save_model
        save_model(self, path, metadata=metadata)

    @classmethod
    def load(cls, path: str) -> "Algorithm":
        """Load a saved model from disk.

        Parameters
        ----------
        path : str
            Path to a file written by :meth:`save`.

        Returns
        -------
        model : Algorithm
            The loaded model instance.

        Raises
        ------
        TypeError
            If called on a subclass and the file contains a different type
            (``Algorithm.load(path)`` accepts any saved model).
        """
        from tuiml.utils.serialization import load_model
        model = load_model(path)
        if cls is not Algorithm and not isinstance(model, cls):
            raise TypeError(
                f"{path!r} contains a {type(model).__name__}, not a "
                f"{cls.__name__}. Use {type(model).__name__}.load() or "
                f"Algorithm.load()."
            )
        return model

    def partial_fit(self, X: np.ndarray, y: Optional[np.ndarray] = None, classes: Optional[np.ndarray] = None) -> "Algorithm":
        """Incrementally fit the model on a batch of samples.

        For algorithms that do not support online learning natively, this
        fallback accumulates the samples in memory and calls :meth:`fit` on
        the entire accumulated dataset.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Batch of training features.
        y : np.ndarray of shape (n_samples,), optional
            Batch of training labels or target values.
        classes : np.ndarray, optional
            Full array of possible class labels (classifiers only). Used to
            populate ``classes_`` before all classes have been seen.

        Returns
        -------
        self : Algorithm
            The fitted estimator (for method chaining).
        """
        import warnings
        warnings.warn(
            f"{self.__class__.__name__} does not support incremental learning natively. "
            "Accumulating data in memory and retraining on all accumulated data.",
            UserWarning,
            stacklevel=2
        )
        
        # Initialize history if not present
        if not hasattr(self, "_history_X"):
            self._history_X = X
            self._history_y = y
        else:
            self._history_X = np.vstack([self._history_X, X])
            if y is not None:
                self._history_y = np.concatenate([self._history_y, y])
                
        # Update classes_ if classifier
        if hasattr(self, "classes_") and y is not None:
            if classes is not None:
                self.classes_ = np.asarray(classes)
            elif self.classes_ is None:
                self.classes_ = np.unique(self._history_y)
                
        self.fit(self._history_X, self._history_y)
        return self

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Return the algorithm's constructor parameters.

        Fitted attributes are excluded: by convention they carry a trailing
        underscore (``classes_``, ``estimators_``), marking them as learned
        state rather than configuration. Keeping them out is what lets a
        fitted estimator be cloned back to an unfitted copy of itself.

        Parameters
        ----------
        deep : bool, default=True
            Accepted for API symmetry with composite estimators, which use it
            to include their children's parameters. Plain algorithms have no
            nested components, so it makes no difference here.

        Returns
        -------
        params : dict
            Parameter names mapped to their current values.
        """
        return {
            key: value
            for key, value in self.__dict__.items()
            if not key.startswith("_") and not key.endswith("_")
        }

    def set_params(self, **params) -> "Algorithm":
        """Set algorithm parameters.

        Parameters
        ----------
        **params : dict
            Parameter names mapped to their new values. Each name must be
            an existing attribute of the estimator.

        Returns
        -------
        self : Algorithm
            The estimator (for method chaining).

        Raises
        ------
        ValueError
            If a parameter name is not a valid attribute.
        """
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid parameter: {key}")
        return self

    def _check_is_fitted(self):
        """Raise ``RuntimeError`` if the algorithm has not been fitted yet."""
        if not self._is_fitted:
            raise RuntimeError(
                f"{self.__class__.__name__} must be fitted before calling predict"
            )

    def serve(
        self,
        port: int = 8000,
        host: str = "127.0.0.1",
        model_id: Optional[str] = None,
        background: Optional[bool] = None,
    ):
        """Serve this fitted model behind a REST API.

        Delegates to :func:`tuiml.serve`, so the server is tracked and can be
        shut down with :func:`tuiml.stop_server`.

        Parameters
        ----------
        port : int, default=8000
            Port to listen on.
        host : str, default="127.0.0.1"
            Host to bind to.
        model_id : str, optional
            Identifier for the model in the server. Defaults to the class name.
        background : bool, optional
            Whether to run the server on a background thread. The default
            adapts to the caller: ``True`` inside a notebook, where blocking
            would freeze the cell, and ``False`` in a plain script, where the
            process should stay up to serve requests.

        Returns
        -------
        dict or None
            Server details (``server_id``, ``url``, ``endpoints``) when
            running in the background; ``None`` when blocking.

        Raises
        ------
        RuntimeError
            If the model is not fitted, or the server cannot start (most often
            because the port is already in use).

        Examples
        --------
        >>> from tuiml.algorithms.bayesian import NaiveBayesClassifier
        >>> from tuiml.datasets import load_iris
        >>> from tuiml.evaluation.splitting import train_test_split
        >>> X_train, X_test, y_train, y_test = train_test_split(*load_iris())
        >>> model = NaiveBayesClassifier().fit(X_train, y_train)   # doctest: +SKIP
        >>> info = model.serve(port=8000)                          # doctest: +SKIP
        >>> info["endpoints"]["predict"]                           # doctest: +SKIP
        'http://127.0.0.1:8000/models/NaiveBayesClassifier/predict'
        """
        self._check_is_fitted()

        if background is None:
            try:
                asyncio.get_running_loop()
                background = True
            except RuntimeError:
                background = False

        from tuiml.serving import serve as _serve

        return _serve(
            self,
            host=host,
            port=port,
            model_id=model_id or type(self).__name__,
            background=background,
        )

# =============================================================================
# Classifier Base Class
# =============================================================================

class Classifier(Algorithm):
    """Base class for supervised classification algorithms.

    Classifiers learn to assign categorical labels (classes) to instances
    based on training data. The default :meth:`~Algorithm.score` is
    accuracy.

    Attributes
    ----------
    classes_ : np.ndarray
        Class labels known to the classifier, when set by the subclass
        during :meth:`~Algorithm.fit`.

    See Also
    --------
    :class:`~tuiml.base.algorithms.Regressor` : For continuous output.
    :class:`~tuiml.base.algorithms.Algorithm` : Common algorithm interface.
    """

    _component_type = ComponentType.CLASSIFIER

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities.

        If the specific classifier does not support probability prediction
        natively, this method returns a one-hot probability representation
        based on the predicted class labels.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Test features.

        Returns
        -------
        proba : np.ndarray of shape (n_samples, n_classes)
            Class membership probabilities.
        """
        import warnings
        warnings.warn(
            f"{self.__class__.__name__} does not support probability prediction natively. "
            "Using a one-hot fallback based on hard predictions.",
            UserWarning,
            stacklevel=2
        )
        
        self._check_is_fitted()
        
        # Predict hard labels
        preds = self.predict(X)
        
        # Get classes_ if available, otherwise get unique elements from predictions
        classes = getattr(self, "classes_", None)
        if classes is None:
            classes = np.unique(preds)
            
        n_samples = len(X)
        n_classes = len(classes)
        
        # Build mapping from class value to index
        class_to_idx = {c: i for i, c in enumerate(classes)}
        
        proba = np.zeros((n_samples, n_classes))
        for i, pred in enumerate(preds):
            if pred in class_to_idx:
                proba[i, class_to_idx[pred]] = 1.0
                
        return proba

# =============================================================================
# Clusterer Base Classes
# =============================================================================

class Clusterer(Algorithm):
    """Base class for unsupervised clustering algorithms.

    Clustering groups similar instances together without the need for
    pre-defined labels, discovering the underlying structure of the data.

    Overview
    --------
    Algorithms in this category typically handle partition-based (e.g., K-Means)
    or hierarchical grouping.

    Attributes
    ----------
    n_clusters_ : int, optional
        The number of resulting clusters (if applicable).
    labels_ : np.ndarray
        The labels assigned to each instance in the training set.
    cluster_centers_ : np.ndarray, optional
        Coordinates of the cluster centroids.

    See Also
    --------
    :class:`~tuiml.base.algorithms.DensityBasedClusterer` : Probabilistic cluster membership.
    :class:`~tuiml.base.algorithms.UpdateableClusterer` : Incremental clustering.
    """

    _component_type = ComponentType.CLUSTERER

    def __init__(self):
        """Initialize the clusterer."""
        super().__init__()
        self.n_clusters_ = None
        self.labels_ = None
        self.cluster_centers_ = None

    # Restated as abstract. This override exists only to document that y is
    # ignored, and a plain body silently cancelled Algorithm.fit's
    # @abstractmethod: a clusterer that forgot to implement fit instantiated
    # happily and returned None from fit(), failing later and elsewhere.
    @abstractmethod
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "Clusterer":
        """Build the clustering model from training data.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data.
        y : None
            Ignored (unsupervised learning).

        Returns
        -------
        self : Clusterer
            The fitted clusterer (for method chaining).
        """
        pass

    def fit_predict(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """Fit the model and return cluster labels.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Training data.
        y : None
            Ignored.

        Returns
        -------
        labels : np.ndarray of shape (n_samples,)
            Cluster label of each training instance.
        """
        self.fit(X)
        return self.labels_

    def __repr__(self) -> str:
        """Return a string representation showing fit status and cluster count."""
        name = self.__class__.__name__
        if self._is_fitted:
            return f"{name}(n_clusters={self.n_clusters_})"
        return f"{name}(not fitted)"

class DensityBasedClusterer(Clusterer):
    """Base class for density-based clusterers.

    Density-based clusterers can estimate the probability of cluster
    membership for each instance.

    See Also
    --------
    :class:`~tuiml.base.algorithms.Clusterer` : General clustering base class.
    """

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict cluster membership probabilities.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Data to score.

        Returns
        -------
        proba : np.ndarray of shape (n_samples, n_clusters)
            Cluster membership probability matrix.
        """
        pass

    def log_likelihood(self, X: np.ndarray) -> float:
        """Compute the log-likelihood of data under the model.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Data to score.

        Returns
        -------
        log_likelihood : float
            Total log-likelihood of the data.
        """
        self._check_is_fitted()
        proba = self.predict_proba(X)
        return np.sum(np.log(np.sum(proba, axis=1) + 1e-10))

class UpdateableClusterer(Clusterer):
    """Base class for clusterers that support incremental updates.

    See Also
    --------
    :class:`~tuiml.base.algorithms.Clusterer` : General clustering base class.
    """

    @abstractmethod
    def update(self, X: np.ndarray) -> "UpdateableClusterer":
        """Update the model with new instances.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            New data.

        Returns
        -------
        self : UpdateableClusterer
            The updated clusterer (for method chaining).
        """
        pass

    def partial_fit(self, X: np.ndarray, y: Optional[np.ndarray] = None, classes: Optional[np.ndarray] = None) -> "UpdateableClusterer":
        """Incrementally fit the model on a batch of samples.

        Delegates to the clusterer's native :meth:`update` method.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Batch of training data.
        y : None
            Ignored (unsupervised learning).
        classes : None
            Ignored; accepted for API symmetry with classifiers.

        Returns
        -------
        self : UpdateableClusterer
            The updated clusterer (for method chaining).
        """
        self.update(X)
        self._is_fitted = True
        return self

# =============================================================================
# Regressor Base Class
# =============================================================================

class Regressor(Algorithm):
    """Base class for supervised regression algorithms.

    Regressors learn to predict continuous target values from training
    data. The default :meth:`~Algorithm.score` is R². Timeseries models
    also use this base class.

    See Also
    --------
    :class:`~tuiml.base.algorithms.Classifier` : For categorical output.
    :class:`~tuiml.base.algorithms.Algorithm` : Common algorithm interface.
    """

    _component_type = ComponentType.REGRESSOR


class Survival(Algorithm):
    """Base class for survival analysis models.

    Survival models estimate the time to an event of interest — failure,
    churn, relapse — where some subjects are right-censored: the event had
    not happened when observation stopped. Input is therefore
    ``(X, time, event)`` rather than ``(X, y)``, with ``event`` marking
    which ``time`` values are observed event times and which are censoring
    times.

    A model predicts a risk score, where a larger value means an earlier
    expected event, and optionally a survival function
    :math:`S(t) = P(T > t)`.

    See Also
    --------
    :class:`~tuiml.base.algorithms.Classifier` : For categorical output.
    :class:`~tuiml.base.algorithms.Regressor` : For uncensored continuous output.
    """

    _component_type = ComponentType.SURVIVAL

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return a risk score for each sample.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Covariates.

        Returns
        -------
        risk : np.ndarray of shape (n_samples,)
            Predicted risk. Higher values mean an earlier expected event.
        """
        return self.predict_risk(X)

    @abstractmethod
    def predict_risk(self, X: np.ndarray) -> np.ndarray:
        """Return a risk score for each sample.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Covariates.

        Returns
        -------
        risk : np.ndarray of shape (n_samples,)
            Predicted risk. Higher values mean an earlier expected event.
        """


class UpliftModel(Algorithm):
    """Base class for uplift / heterogeneous-treatment-effect models.

    Uplift models estimate the causal effect of a treatment on an individual:
    the difference between the outcome *with* treatment and *without* it.
    Input is therefore ``(X, treatment, y)`` rather than ``(X, y)``, with
    ``treatment`` a binary indicator of which group each sample belonged to.

    The headline output is the individual treatment effect, or uplift, which
    is what decides who to treat.

    See Also
    --------
    :class:`~tuiml.base.algorithms.Classifier` : For plain classification.
    """

    _component_type = ComponentType.UPLIFT

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return the predicted uplift for each sample.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Covariates.

        Returns
        -------
        uplift : np.ndarray of shape (n_samples,)
            Predicted individual treatment effect.
        """
        return self.predict_uplift(X)

    @abstractmethod
    def predict_uplift(self, X: np.ndarray) -> np.ndarray:
        """Return the predicted uplift for each sample.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Covariates.

        Returns
        -------
        uplift : np.ndarray of shape (n_samples,)
            Predicted individual treatment effect.
        """


# =============================================================================
# Associator Base Class and Data Structures
# =============================================================================

@dataclass
class FrequentItemset:
    """Represents a set of items that appear together frequently.

    Parameters
    ----------
    items : frozenset of int
        The set of item identifiers included in the itemset.
    support : float
        The proportion of transactions containing this itemset:
        :math:`P(items)`.
    count : int, default=0
        The absolute frequency count of the itemset.
    """
    items: FrozenSet[int]
    support: float
    count: int = 0

    def __repr__(self) -> str:
        """Return a string representation showing items and support."""
        return f"Itemset({set(self.items)}, sup={self.support:.3f})"

    def __len__(self) -> int:
        """Return the number of items in the itemset."""
        return len(self.items)

    def __hash__(self) -> int:
        """Return a hash based on the item set."""
        return hash(self.items)

    def __eq__(self, other) -> bool:
        """Compare itemsets by their item sets."""
        if isinstance(other, FrequentItemset):
            return self.items == other.items
        return False

@dataclass
class AssociationRule:
    """Represents a discovered relationship in the form $A \\Rightarrow C$.

    Overview
    --------
    Association rules quantify how likely the consequent ($C$) is to appear
    given the presence of the antecedent ($A$).

    Parameters
    ----------
    antecedent : frozenset of int
        The conditional part of the rule ($A$).
    consequent : frozenset of int
        The predicted part of the rule ($C$).
    support : float
        The joint probability: :math:`P(A \\cup C)`.
    confidence : float
        The conditional probability: :math:`P(C|A) = \\frac{P(A \\cup C)}{P(A)}`.
    lift : float, default=1.0
        The ratio of observed support to expected support if independent:
        :math:`\\frac{P(C|A)}{P(C)}`.
    leverage : float, default=0.0
        The difference from independence: :math:`P(A \\cup C) - P(A)P(C)`.
    conviction : float, default=1.0
        Implication strength: :math:`\\frac{1 - P(C)}{1 - \\text{confidence}}`.
    jaccard : float, default=0.0
        Jaccard coefficient:
        :math:`\\frac{P(A \\cup C)}{P(A) + P(C) - P(A \\cup C)}`.
    kulczynski : float, default=0.0
        Average of the two conditional confidences :math:`P(C|A)` and
        :math:`P(A|C)`.
    all_confidence : float, default=0.0
        Minimum of the two conditional confidences.
    """
    antecedent: FrozenSet[int]
    consequent: FrozenSet[int]
    support: float
    confidence: float
    lift: float = 1.0
    leverage: float = 0.0
    conviction: float = 1.0
    jaccard: float = 0.0
    kulczynski: float = 0.0
    all_confidence: float = 0.0

    def __repr__(self) -> str:
        """Return a string representation showing the rule and its metrics."""
        return (f"{set(self.antecedent)} -> {set(self.consequent)} "
                f"(conf={self.confidence:.3f}, lift={self.lift:.3f})")

    def __hash__(self) -> int:
        """Return a hash based on antecedent and consequent."""
        return hash((self.antecedent, self.consequent))

    def __eq__(self, other) -> bool:
        """Compare rules by antecedent and consequent."""
        if isinstance(other, AssociationRule):
            return (self.antecedent == other.antecedent and
                    self.consequent == other.consequent)
        return False

class Associator(Algorithm):
    """Base class for Association Rule Mining.

    Discovers interesting patterns and relationships between items in
    large datasets (e.g., market basket analysis).

    Theory
    ------
    Mining typically involves two steps:
    1. Finding all **Frequent Itemsets** that satisfy a minimum support.
    2. Generating **Association Rules** that satisfy a minimum confidence.

    Attributes
    ----------
    frequent_itemsets_ : list of FrequentItemset
        The collection of all itemsets found above the minimum support threshold.
    rules_ : list of AssociationRule
        The collection of all generated rules.
    """

    _component_type = ComponentType.ASSOCIATOR

    def __init__(self):
        """Initialize the associator."""
        super().__init__()
        self.frequent_itemsets_: List[FrequentItemset] = []
        self.rules_: List[AssociationRule] = []
        self.n_transactions_ = 0
        self.n_items_ = 0

    @abstractmethod
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "Associator":
        """Find frequent itemsets and generate association rules.

        Parameters
        ----------
        X : np.ndarray or list of list of int
            Transaction data, either a binary matrix of shape
            (n_transactions, n_items) or a list of lists of item indices.
        y : None
            Ignored.

        Returns
        -------
        self : Associator
            The fitted associator (for method chaining).
        """
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Not applicable for association mining.

        Raises
        ------
        NotImplementedError
            Always; use :meth:`get_frequent_itemsets` or :meth:`get_rules`
            instead.
        """
        raise NotImplementedError(
            "Association rule miners don't support predict(). "
            "Use get_frequent_itemsets() or get_rules() instead."
        )

    def get_frequent_itemsets(self, min_size: int = 1,
                              max_size: Optional[int] = None) -> List[FrequentItemset]:
        """Return the discovered frequent itemsets.

        Parameters
        ----------
        min_size : int, default=1
            Minimum itemset size.
        max_size : int, optional
            Maximum itemset size (``None`` for no limit).

        Returns
        -------
        itemsets : list of FrequentItemset
            Frequent itemsets within the requested size range.
        """
        self._check_is_fitted()
        result = [fs for fs in self.frequent_itemsets_ if len(fs) >= min_size]
        if max_size is not None:
            result = [fs for fs in result if len(fs) <= max_size]
        return result

    def get_rules(self, min_confidence: Optional[float] = None,
                  min_lift: Optional[float] = None) -> List[AssociationRule]:
        """Return the discovered association rules.

        Parameters
        ----------
        min_confidence : float, optional
            Minimum confidence threshold.
        min_lift : float, optional
            Minimum lift threshold.

        Returns
        -------
        rules : list of AssociationRule
            Rules passing all requested thresholds.
        """
        self._check_is_fitted()
        result = self.rules_
        if min_confidence is not None:
            result = [r for r in result if r.confidence >= min_confidence]
        if min_lift is not None:
            result = [r for r in result if r.lift >= min_lift]
        return result

    def _preprocess_transactions(self, X) -> List[FrozenSet[int]]:
        """Convert input to a list of transaction sets.

        Handles both binary matrix and list-of-lists formats; empty
        transactions are dropped.

        Parameters
        ----------
        X : np.ndarray or list of list of int
            Transaction data.

        Returns
        -------
        transactions : list of frozenset of int
            One frozenset of item indices per non-empty transaction.
        """
        if isinstance(X, np.ndarray):
            # Binary matrix format
            transactions = []
            for row in X:
                items = frozenset(np.where(row > 0)[0])
                if items:
                    transactions.append(items)
            return transactions
        else:
            # List of lists format
            return [frozenset(t) for t in X if t]

    def __repr__(self) -> str:
        """Return a string representation showing itemset and rule counts."""
        name = self.__class__.__name__
        if self._is_fitted:
            return (f"{name}(n_itemsets={len(self.frequent_itemsets_)}, "
                    f"n_rules={len(self.rules_)})")
        return f"{name}(not fitted)"

# =============================================================================
# Legacy AlgorithmRegistry (backward compatibility)
# =============================================================================

class AlgorithmRegistry:
    """Central registry for all algorithms (legacy wrapper).

    Provides discovery, registration, and instantiation of algorithms.

    Notes
    -----
    This class now wraps the unified component registry
    (``tuiml.registry``) for backward compatibility. New code should use
    the registry directly.
    """

    @classmethod
    def register(cls, algorithm_class: Type[Algorithm]) -> Type[Algorithm]:
        """Register an algorithm class with the component registry.

        Parameters
        ----------
        algorithm_class : type
            Algorithm class to register.

        Returns
        -------
        algorithm_class : type
            The same class (for decorator usage).
        """
        # Determine component type
        algorithm_type = getattr(algorithm_class, "_algorithm_type", "algorithm")
        type_map = {
            "classifier": ComponentType.CLASSIFIER,
            "clusterer": ComponentType.CLUSTERER,
            "regressor": ComponentType.REGRESSOR,
            "anomaly": ComponentType.ANOMALY,
            "associator": ComponentType.ASSOCIATOR,
            "survival": ComponentType.SURVIVAL,
            "uplift": ComponentType.UPLIFT,
        }
        component_type = type_map.get(algorithm_type, ComponentType.ALGORITHM)

        # Register with the component registry
        registry.register_class(algorithm_class, component_type)
        return algorithm_class

    @classmethod
    def get(cls, name: str) -> Type[Algorithm]:
        """Get an algorithm class by name.

        Parameters
        ----------
        name : str
            Registered algorithm name.

        Returns
        -------
        algorithm_class : type
            The algorithm class.

        Raises
        ------
        ValueError
            If no algorithm with that name is registered.
        """
        try:
            return registry.get(name)
        except KeyError as e:
            raise ValueError(str(e))

    @classmethod
    def list(cls, type: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all algorithms, optionally filtered by type.

        Parameters
        ----------
        type : str, optional
            Filter by type (``'classifier'``, ``'clusterer'``,
            ``'regressor'``, ``'associator'``).

        Returns
        -------
        algorithms : list of dict
            Algorithm metadata dictionaries.
        """
        type_map = {
            "classifier": ComponentType.CLASSIFIER,
            "clusterer": ComponentType.CLUSTERER,
            "regressor": ComponentType.REGRESSOR,
            "anomaly": ComponentType.ANOMALY,
            "associator": ComponentType.ASSOCIATOR,
            "survival": ComponentType.SURVIVAL,
            "uplift": ComponentType.UPLIFT,
        }

        if type:
            component_type = type_map.get(type)
            return registry.list(component_type)
        else:
            # Return all algorithm types
            results = []
            for ct in [ComponentType.ALGORITHM, ComponentType.CLASSIFIER,
                       ComponentType.CLUSTERER, ComponentType.REGRESSOR,
                       ComponentType.ASSOCIATOR]:
                results.extend(registry.list(ct))
            return results

    @classmethod
    def search(cls, query: str) -> List[Dict[str, Any]]:
        """Search algorithms by keyword.

        Parameters
        ----------
        query : str
            Search query.

        Returns
        -------
        matches : list of dict
            Metadata of matching algorithms.
        """
        return registry.search(query)

    @classmethod
    def get_by_type(cls, algorithm_type: str) -> List[str]:
        """Get all algorithm names of a specific type.

        Parameters
        ----------
        algorithm_type : str
            Type to filter by (``'classifier'``, ``'clusterer'``,
            ``'regressor'``, ``'associator'``).

        Returns
        -------
        names : list of str
            Registered algorithm names of that type.
        """
        type_map = {
            "classifier": ComponentType.CLASSIFIER,
            "clusterer": ComponentType.CLUSTERER,
            "regressor": ComponentType.REGRESSOR,
            "anomaly": ComponentType.ANOMALY,
            "associator": ComponentType.ASSOCIATOR,
            "survival": ComponentType.SURVIVAL,
            "uplift": ComponentType.UPLIFT,
        }
        component_type = type_map.get(algorithm_type, ComponentType.ALGORITHM)
        return registry.list_names(component_type)

    @classmethod
    def clear(cls):
        """Clear all registered algorithms (mainly for testing)."""
        registry.clear()

# =============================================================================
# Decorators (with registry registration)
# =============================================================================

def algorithm(type: str = "classifier"):
    """Decorator to register an algorithm with the component registry.

    Parameters
    ----------
    type : str, default="classifier"
        Algorithm type (``'classifier'``, ``'clusterer'``, ``'regressor'``,
        ``'associator'``).

    Returns
    -------
    decorator : callable
        Class decorator that registers the algorithm.

    Examples
    --------
    >>> from tuiml.base.algorithms import algorithm, Classifier
    >>> @algorithm(type="classifier")
    ... class RandomForest(Classifier):
    ...     pass
    """
    type_map = {
        "classifier": ComponentType.CLASSIFIER,
        "clusterer": ComponentType.CLUSTERER,
        "regressor": ComponentType.REGRESSOR,
        "anomaly": ComponentType.ANOMALY,
        "associator": ComponentType.ASSOCIATOR,
    }
    component_type = type_map.get(type, ComponentType.ALGORITHM)

    def decorator(cls: Type[Algorithm]) -> Type[Algorithm]:
        cls._algorithm_type = type
        return registry.register(component_type)(cls)

    return decorator

def classifier(
    name: Optional[str] = None,
    tags: Optional[List[str]] = None,
    version: str = "1.0.0",
    author: Optional[str] = None,
):
    """Decorator to register a classifier with the component registry.

    Parameters
    ----------
    name : str, optional
        Registry name. Defaults to the class name.
    tags : list of str, optional
        Tags for discovery and search.
    version : str, default="1.0.0"
        Component version string.
    author : str, optional
        Component author.

    Returns
    -------
    decorator : callable
        Class decorator that registers the classifier.

    Examples
    --------
    >>> from tuiml.base.algorithms import classifier, Classifier
    >>> @classifier(tags=["ensemble", "tree"])
    ... class RandomForest(Classifier):
    ...     pass
    """
    return registry.register(
        ComponentType.CLASSIFIER,
        name=name,
        tags=tags,
        version=version,
        author=author,
    )

def clusterer(
    name: Optional[str] = None,
    tags: Optional[List[str]] = None,
    version: str = "1.0.0",
    author: Optional[str] = None,
):
    """Decorator to register a clusterer with the component registry.

    Parameters
    ----------
    name : str, optional
        Registry name. Defaults to the class name.
    tags : list of str, optional
        Tags for discovery and search.
    version : str, default="1.0.0"
        Component version string.
    author : str, optional
        Component author.

    Returns
    -------
    decorator : callable
        Class decorator that registers the clusterer.

    Examples
    --------
    >>> from tuiml.base.algorithms import clusterer, Clusterer
    >>> @clusterer(tags=["partitioning"])
    ... class KMeans(Clusterer):
    ...     pass
    """
    return registry.register(
        ComponentType.CLUSTERER,
        name=name,
        tags=tags,
        version=version,
        author=author,
    )

def regressor(
    name: Optional[str] = None,
    tags: Optional[List[str]] = None,
    version: str = "1.0.0",
    author: Optional[str] = None,
):
    """Decorator to register a regressor with the component registry.

    Parameters
    ----------
    name : str, optional
        Registry name. Defaults to the class name.
    tags : list of str, optional
        Tags for discovery and search.
    version : str, default="1.0.0"
        Component version string.
    author : str, optional
        Component author.

    Returns
    -------
    decorator : callable
        Class decorator that registers the regressor.

    Examples
    --------
    >>> from tuiml.base.algorithms import regressor, Regressor
    >>> @regressor(tags=["linear"])
    ... class LinearRegression(Regressor):
    ...     pass
    """
    return registry.register(
        ComponentType.REGRESSOR,
        name=name,
        tags=tags,
        version=version,
        author=author,
    )

def anomaly_detector(
    name: Optional[str] = None,
    tags: Optional[List[str]] = None,
    version: str = "1.0.0",
    author: Optional[str] = None,
):
    """Decorator to register an anomaly detector with the component registry.

    Detectors subclass :class:`Classifier` and predict a discrete
    inlier/outlier label, so they score like a classifier. They register under
    their own component type instead, so :func:`tuiml.list_algorithms` can
    report them as anomaly detectors rather than burying them among the
    classifiers.

    Parameters
    ----------
    name : str, optional
        Registry name. Defaults to the class name.
    tags : list of str, optional
        Tags for discovery and search.
    version : str, default="1.0.0"
        Component version string.
    author : str, optional
        Component author.

    Returns
    -------
    decorator : callable
        Class decorator that registers the detector.

    Examples
    --------
    >>> from tuiml.base.algorithms import anomaly_detector, Classifier
    >>> @anomaly_detector(tags=["anomaly-detection", "tree-based"])
    ... class MyForestDetector(Classifier):
    ...     pass
    """
    return registry.register(
        ComponentType.ANOMALY,
        name=name,
        tags=tags,
        version=version,
        author=author,
    )

def associator(
    name: Optional[str] = None,
    tags: Optional[List[str]] = None,
    version: str = "1.0.0",
    author: Optional[str] = None,
):
    """Decorator to register an associator with the component registry.

    Parameters
    ----------
    name : str, optional
        Registry name. Defaults to the class name.
    tags : list of str, optional
        Tags for discovery and search.
    version : str, default="1.0.0"
        Component version string.
    author : str, optional
        Component author.

    Returns
    -------
    decorator : callable
        Class decorator that registers the associator.

    Examples
    --------
    >>> from tuiml.base.algorithms import associator, Associator
    >>> @associator(tags=["itemset", "frequent"])
    ... class Apriori(Associator):
    ...     pass
    """
    return registry.register(
        ComponentType.ASSOCIATOR,
        name=name,
        tags=tags,
        version=version,
        author=author,
    )


def survival(
    name: Optional[str] = None,
    tags: Optional[List[str]] = None,
    version: str = "1.0.0",
    author: Optional[str] = None,
):
    """Decorator to register a survival model with the component registry.

    Survival models subclass :class:`Survival` and estimate time-to-event
    with right-censored data.

    Parameters
    ----------
    name : str, optional
        Registry name. Defaults to the class name.
    tags : list of str, optional
        Tags for discovery and search.
    version : str, default="1.0.0"
        Component version string.
    author : str, optional
        Component author.

    Returns
    -------
    decorator : callable
        Class decorator that registers the survival model.
    """
    return registry.register(
        ComponentType.SURVIVAL,
        name=name,
        tags=tags,
        version=version,
        author=author,
    )


def uplift(
    name: Optional[str] = None,
    tags: Optional[List[str]] = None,
    version: str = "1.0.0",
    author: Optional[str] = None,
):
    """Decorator to register an uplift model with the component registry.

    Uplift models subclass :class:`UpliftModel` and estimate heterogeneous
    treatment effects.

    Parameters
    ----------
    name : str, optional
        Registry name. Defaults to the class name.
    tags : list of str, optional
        Tags for discovery and search.
    version : str, default="1.0.0"
        Component version string.
    author : str, optional
        Component author.

    Returns
    -------
    decorator : callable
        Class decorator that registers the uplift model.
    """
    return registry.register(
        ComponentType.UPLIFT,
        name=name,
        tags=tags,
        version=version,
        author=author,
    )

# =============================================================================
# Convenience functions
# =============================================================================

def get_algorithm(name: str) -> Type[Algorithm]:
    """Get an algorithm class by name from the component registry.

    Parameters
    ----------
    name : str
        Registered algorithm name.

    Returns
    -------
    algorithm_class : type
        The algorithm class.
    """
    return registry.get(name)

def list_algorithms(type: Optional[str] = None) -> List[Dict[str, Any]]:
    """List available algorithms.

    Parameters
    ----------
    type : str, optional
        Filter by type (``'classifier'``, ``'clusterer'``, ``'regressor'``,
        ``'associator'``).

    Returns
    -------
    algorithms : list of dict
        Algorithm metadata dictionaries.
    """
    return AlgorithmRegistry.list(type)

def search_algorithms(query: str) -> List[Dict[str, Any]]:
    """Search algorithms by keyword.

    Parameters
    ----------
    query : str
        Search query.

    Returns
    -------
    matches : list of dict
        Metadata of matching algorithms.
    """
    return registry.search(query)
