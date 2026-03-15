"""
Main experiment runner.
"""

import numpy as np
import time
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from copy import deepcopy
import warnings

try:
    from joblib import Parallel, delayed
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

from tuiml.base.experiments import (
    BaseExperiment, ExperimentConfig, ExperimentResults,
    ExperimentType, ValidationMethod,
    FoldResult, ModelResult, DatasetResult, _ensure_numpy
)

# Import from new module structure
from ..splitting import (
    KFold, StratifiedKFold, RepeatedKFold, RepeatedStratifiedKFold,
    HoldoutSplit, StratifiedHoldoutSplit,
    LeaveOneOut, BootstrapSplit, TimeSeriesSplit
)
from ..statistics import paired_t_test, SignificanceLevel
from ..reporting import ResultMatrix, format_results
from ..visualization import (
    plot_critical_difference as _plot_cd,
    plot_ranking_table as _plot_ranking,
    plot_boxplot_comparison as _plot_boxplot,
    compute_ranks
)

class Experiment(BaseExperiment):
    """
    Main experiment runner for model comparison.

    Runs multiple models on multiple datasets with cross-validation
    and computes statistical comparisons.

    Parameters
    ----------
    config : ExperimentConfig, optional
        Experiment configuration.
    metrics : list of str or dict, optional
        Metrics to compute. Can be:
        - List of metric names: ['accuracy', 'f1']
        - Dict of {name: function}: {'custom': my_metric_func}
    experiment_type : ExperimentType
        Type of experiment (classification, regression, clustering).

    Examples
    --------
    >>> from tuiml.evaluation.experiments import Experiment, ExperimentConfig
    >>> from tuiml.algorithms.trees import RandomForestClassifier
    >>> from tuiml.algorithms.svm import SVC
    >>>
    >>> # Define models
    >>> models = {
    ...     'RF': RandomForestClassifier(n_estimators=100),
    ...     'SVM': SVC()
    ... }
    >>>
    >>> # Define datasets
    >>> datasets = {
    ...     'iris': (X_iris, y_iris),
    ...     'wine': (X_wine, y_wine)
    ... }
    >>>
    >>> # Run experiment
    >>> config = ExperimentConfig(n_folds=10, n_repeats=1)
    >>> exp = Experiment(config, metrics=['accuracy', 'f1_macro'])
    >>> results = exp.run(models, datasets)
    >>>
    >>> # Print results
    >>> print(exp.summary())
    >>> print(exp.to_latex())
    """

    @classmethod
    def get_parameter_schema(cls) -> Dict[str, Any]:
        """
        Get JSON Schema for the Experiment class constructor parameters.

        Returns
        -------
        schema : dict
            JSON Schema describing the __init__ parameters.

        Examples
        --------
        >>> schema = Experiment.get_parameter_schema()
        >>> print(schema['properties'].keys())
        """
        return {
            "$schema": "https://json-schema.org/draft/2020-12/schema",
            "title": "Experiment",
            "description": "Main experiment runner for model comparison. Runs multiple models on multiple datasets with cross-validation and computes statistical comparisons.",
            "type": "object",
            "properties": {
                "config": {
                    "description": "Experiment configuration object containing settings for validation method, number of folds, repeats, etc.",
                    "oneOf": [
                        {"type": "null"},
                        {
                            "type": "object",
                            "description": "ExperimentConfig object with experiment settings"
                        }
                    ],
                    "default": None
                },
                "metrics": {
                    "description": "Metrics to compute using exact function names from tuiml.evaluation.metrics (e.g., ['accuracy_score', 'f1_score']). If None, defaults are chosen based on experiment_type.",
                    "oneOf": [
                        {"type": "null"},
                        {
                            "type": "array",
                            "items": {
                                "type": "string",
                                "enum": [
                                    "accuracy_score", "f1_score", "precision_score",
                                    "recall_score", "roc_auc_score",
                                    "mean_squared_error", "root_mean_squared_error",
                                    "mean_absolute_error", "r2_score",
                                    "silhouette_score", "calinski_harabasz_score",
                                    "davies_bouldin_score", "adjusted_rand_score"
                                ]
                            },
                            "description": "List of metric function names to compute"
                        },
                        {
                            "type": "object",
                            "additionalProperties": {
                                "type": "string",
                                "description": "Callable metric function"
                            },
                            "description": "Dict mapping metric names to callable functions"
                        }
                    ],
                    "default": None
                },
                "experiment_type": {
                    "type": "string",
                    "enum": ["classification", "regression", "clustering"],
                    "description": "Type of experiment. Determines default metrics if not specified.",
                    "default": "classification"
                }
            },
            "additionalProperties": False
        }

    def __init__(
        self,
        config: ExperimentConfig = None,
        metrics: Union[List[str], Dict[str, Callable]] = None,
        experiment_type: ExperimentType = ExperimentType.CLASSIFICATION
    ):
        super().__init__(config)
        self.experiment_type = experiment_type

        # Set default metrics based on experiment type
        # Use full function names matching tuiml.evaluation.metrics
        if metrics is None:
            if experiment_type == ExperimentType.CLASSIFICATION:
                self.metrics = ['accuracy_score', 'f1_score']
            elif experiment_type == ExperimentType.REGRESSION:
                self.metrics = ['r2_score', 'root_mean_squared_error']
            else:
                self.metrics = ['silhouette_score']
        else:
            self.metrics = metrics

        self._metric_funcs = self._setup_metrics()

    def _setup_metrics(self) -> Dict[str, Callable]:
        """Setup metric functions."""
        funcs = {}

        if isinstance(self.metrics, dict):
            return self.metrics

        # Import metric functions
        for metric in self.metrics:
            func = self._get_metric_func(metric)
            if func:
                funcs[metric] = func

        return funcs

    def _get_metric_func(self, name: str) -> Callable:
        """Get metric function by exact name from ``tuiml.evaluation.metrics``.

        Uses ``getattr`` on the metrics module directly — the same approach
        as ``Workflow._execute`` and ``api.evaluate``.

        Parameters
        ----------
        name : str
            Exact function name as it appears in
            ``tuiml.evaluation.metrics`` (e.g. ``'accuracy_score'``,
            ``'f1_score'``, ``'mean_squared_error'``).

        Returns
        -------
        func : callable
            The metric function.

        Raises
        ------
        ValueError
            If ``name`` does not match any function in the metrics module.
        """
        from .. import metrics as metrics_module

        func = getattr(metrics_module, name, None)
        if func is not None:
            return func

        raise ValueError(
            f"Metric '{name}' not available in tuiml.evaluation.metrics. "
            f"Use the exact function name (e.g. 'accuracy_score', 'f1_score', "
            f"'mean_squared_error', 'r2_score', 'silhouette_score')."
        )

    def _create_validator(self):
        """Create validator based on config."""
        method = self.config.validation_method

        if method == ValidationMethod.CROSS_VALIDATION:
            if self.config.stratify:
                return StratifiedKFold(
                    n_splits=self.config.n_folds,
                    shuffle=self.config.shuffle,
                    random_state=self.config.random_state
                )
            else:
                return KFold(
                    n_splits=self.config.n_folds,
                    shuffle=self.config.shuffle,
                    random_state=self.config.random_state
                )
        elif method == ValidationMethod.REPEATED_CV:
            if self.config.stratify:
                return RepeatedStratifiedKFold(
                    n_splits=self.config.n_folds,
                    n_repeats=self.config.n_repeats,
                    random_state=self.config.random_state
                )
            else:
                return RepeatedKFold(
                    n_splits=self.config.n_folds,
                    n_repeats=self.config.n_repeats,
                    random_state=self.config.random_state
                )
        elif method == ValidationMethod.HOLDOUT:
            if self.config.stratify:
                return StratifiedHoldoutSplit(
                    test_size=self.config.test_size,
                    shuffle=self.config.shuffle,
                    random_state=self.config.random_state
                )
            else:
                return HoldoutSplit(
                    test_size=self.config.test_size,
                    shuffle=self.config.shuffle,
                    random_state=self.config.random_state
                )
        elif method == ValidationMethod.LEAVE_ONE_OUT:
            return LeaveOneOut()
        elif method == ValidationMethod.BOOTSTRAP:
            return BootstrapSplit(
                n_iterations=self.config.n_repeats,
                random_state=self.config.random_state
            )
        else:
            raise ValueError(f"Unknown validation method: {method}")

    def run(
        self,
        models: Dict[str, Any],
        datasets: Dict[str, Union[Tuple[np.ndarray, np.ndarray], np.ndarray]],
        metrics: List[str] = None,
        progress_callback: Optional[Callable] = None
    ) -> ExperimentResults:
        """
        Run the experiment.

        Parameters
        ----------
        models : dict
            Dictionary of {name: model} pairs.
        datasets : dict
            Dictionary of {name: (X, y)} pairs for supervised learning,
            or {name: X} for unsupervised learning (clustering, anomaly detection).
        metrics : list of str, optional
            Override default metrics.

        Returns
        -------
        results : ExperimentResults
        """
        if metrics:
            self.metrics = metrics
            self._metric_funcs = self._setup_metrics()

        self.results = ExperimentResults(
            config=self.config,
            experiment_type=self.experiment_type,
            start_time=datetime.now()
        )

        # Detect if models are supervised or unsupervised
        from tuiml.hub import registry, ComponentType
        model_types = {}
        for model_name, model in models.items():
            try:
                # Try to get component info from registry
                model_class_name = model.__class__.__name__
                if model_class_name in registry:
                    info = registry.get_info(model_class_name)
                    model_types[model_name] = info.get('type', 'unknown')
                else:
                    # Fallback: check if model has fit(X, y) or fit(X) signature
                    import inspect
                    sig = inspect.signature(model.fit)
                    params = sig.parameters
                    # Check if y is a required parameter (no default)
                    has_required_y = any(
                        p.default is inspect.Parameter.empty
                        for name, p in list(params.items())[1:]  # skip X
                        if name in ('y', 'Y', '_y', 'target')
                    )
                    model_types[model_name] = 'classifier' if has_required_y else 'clusterer'
            except Exception:
                model_types[model_name] = 'unknown'

        # Check if all models are of the same type
        unique_types = set(model_types.values())
        is_unsupervised = all(t in ['clusterer', 'clustering'] for t in unique_types)

        validator = self._create_validator() if not is_unsupervised else None

        # Run on each dataset
        for dataset_name, dataset_data in datasets.items():
            # Handle both (X, y) tuples and X-only for clustering
            if isinstance(dataset_data, tuple) and len(dataset_data) == 2:
                X, y = dataset_data
                X = _ensure_numpy(X)
                y = _ensure_numpy(y)
            else:
                X = _ensure_numpy(dataset_data)
                y = None

            if self.config.verbose > 0:
                print(f"Processing dataset: {dataset_name}")

            dataset_result = DatasetResult(
                dataset_name=dataset_name,
                n_samples=len(X),
                n_features=X.shape[1] if X.ndim > 1 else 1
            )

            # Get splits for supervised learning only
            splits = None
            if not is_unsupervised and y is not None:
                splits = list(validator.split(X, y))

            # Run each model
            for model_name, model in models.items():
                if self.config.verbose > 0:
                    print(f"  Running model: {model_name}")

                if is_unsupervised or model_types[model_name] in ['clusterer', 'clustering']:
                    # Unsupervised learning - no CV, fit on all data
                    model_result = self._run_unsupervised_model(
                        model_name, model, X, y
                    )
                else:
                    # Supervised learning - use CV
                    model_result = self._run_model(
                        model_name, model, X, y, splits
                    )
                dataset_result.add_model_result(model_result)

                if progress_callback is not None:
                    # Report per-model progress
                    mean_scores = {}
                    if model_result.fold_results:
                        # Try _metric_funcs first, then fallback to accuracy
                        metric_sources = self._metric_funcs if self._metric_funcs else {}
                        if metric_sources:
                            for metric_name in metric_sources:
                                scores = [
                                    fr.metrics.get(metric_name, 0.0)
                                    for fr in model_result.fold_results
                                    if fr.metrics
                                ]
                                if scores:
                                    mean_scores[metric_name] = float(np.mean(scores))
                        # Fallback: compute accuracy from y_true/y_pred if no metrics found
                        if not mean_scores:
                            acc_scores = []
                            for fr in model_result.fold_results:
                                if fr.y_true is not None and fr.y_pred is not None:
                                    acc_scores.append(float(np.mean(fr.y_true == fr.y_pred)))
                            if acc_scores:
                                mean_scores['accuracy'] = float(np.mean(acc_scores))
                    progress_callback({
                        'type': 'experiment_progress',
                        'dataset': dataset_name,
                        'model': model_name,
                        'dataset_index': list(datasets.keys()).index(dataset_name) + 1,
                        'total_datasets': len(datasets),
                        'model_index': list(models.keys()).index(model_name) + 1,
                        'total_models': len(models),
                        'mean_scores': mean_scores,
                    })

            self.results.add_dataset_result(dataset_result)

        self.results.end_time = datetime.now()
        return self.results

    def _run_model(
        self,
        model_name: str,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        splits: List[Tuple[np.ndarray, np.ndarray]]
    ) -> ModelResult:
        """Run a single model on all folds."""
        model_result = ModelResult(model_name=model_name, model=model)

        if self.config.n_jobs != 1 and JOBLIB_AVAILABLE:
            fold_results = Parallel(n_jobs=self.config.n_jobs)(
                delayed(_run_single_fold)(
                    fold_idx, train_idx, test_idx, model, X, y,
                    self._metric_funcs, self.config.verbose
                )
                for fold_idx, (train_idx, test_idx) in enumerate(splits)
            )
            model_result.fold_results = fold_results
        else:
            if self.config.n_jobs != 1 and not JOBLIB_AVAILABLE:
                warnings.warn("joblib not installed, falling back to sequential execution.")

            for fold_idx, (train_idx, test_idx) in enumerate(splits):
                fold_result = _run_single_fold(
                    fold_idx, train_idx, test_idx, model, X, y,
                    self._metric_funcs, self.config.verbose
                )
                model_result.fold_results.append(fold_result)

        return model_result

    def _run_unsupervised_model(
        self,
        model_name: str,
        model: Any,
        X: np.ndarray,
        y: Optional[np.ndarray] = None
    ) -> ModelResult:
        """Run unsupervised model (clustering, anomaly detection).

        For unsupervised learning, we fit on all data and compute clustering metrics.
        """
        from copy import deepcopy

        model_result = ModelResult(model_name=model_name, model=model)

        # Clone model and fit on all data
        model_clone = deepcopy(model)

        try:
            # Fit model
            train_start = time.time()
            if hasattr(model_clone, 'fit_predict'):
                cluster_labels = model_clone.fit_predict(X)
            else:
                model_clone.fit(X)
                cluster_labels = model_clone.predict(X) if hasattr(model_clone, 'predict') else model_clone.labels_
            train_time = time.time() - train_start

            # External clustering metrics require (y_true, y_pred)
            _EXTERNAL_CLUSTERING = {
                'adjusted_rand_score', 'adjusted_rand',
                'normalized_mutual_info_score', 'nmi',
                'v_measure_score', 'v_measure',
                'homogeneity_score', 'homogeneity',
                'completeness_score', 'completeness',
            }

            # Compute clustering metrics
            metrics_dict = {}
            for metric_name, metric_func in self._metric_funcs.items():
                try:
                    if metric_name.lower() in _EXTERNAL_CLUSTERING and y is not None:
                        # External validation metric (needs ground truth)
                        score = metric_func(y, cluster_labels)
                    else:
                        # Internal validation metric (takes X, labels)
                        score = metric_func(X, cluster_labels)
                    metrics_dict[metric_name] = float(score)
                except Exception as e:
                    if self.config.verbose > 1:
                        warnings.warn(f"Failed to compute {metric_name}: {e}")

            # Create a single fold result representing the full dataset
            fold_result = FoldResult(
                fold_idx=0,
                train_time=train_time,
                test_time=0.0,
                metrics=metrics_dict,
                y_true=y if y is not None else cluster_labels,
                y_pred=cluster_labels,
                y_prob=None
            )
            model_result.fold_results.append(fold_result)

        except Exception as e:
            if self.config.verbose > 0:
                print(f"    Error running {model_name}: {e}")
            # Create empty fold result with error
            fold_result = FoldResult(
                fold_idx=0,
                train_time=0.0,
                test_time=0.0,
                metrics={},
                y_true=np.array([]),
                y_pred=np.array([]),
                y_prob=None
            )
            model_result.fold_results.append(fold_result)

        return model_result

    def summary(self, metric: str = None) -> str:
        """
        Get summary of results.

        Parameters
        ----------
        metric : str, optional
            Specific metric to summarize. If None, uses first metric.

        Returns
        -------
        summary : str
        """
        if not self.results:
            return "No results available. Run experiment first."

        if metric is None:
            metric = list(self._metric_funcs.keys())[0]

        lines = []
        lines.append(f"Experiment: {self.config.name}")
        lines.append(f"Validation: {self.config.validation_method.value}")
        lines.append(f"Metric: {metric}")
        lines.append("")

        for dataset_name, dataset_result in self.results.dataset_results.items():
            lines.append(f"Dataset: {dataset_name}")
            lines.append("-" * 50)

            for model_name, model_result in dataset_result.model_results.items():
                stats = model_result.get_metric_stats(metric)
                lines.append(
                    f"  {model_name}: {stats['mean']:.4f} ± {stats['std']:.4f}"
                )

            lines.append("")

        return "\n".join(lines)

    def to_result_matrix(
        self,
        metric: str = None,
        **kwargs
    ) -> ResultMatrix:
        """
        Convert results to ResultMatrix for formatted output.

        Parameters
        ----------
        metric : str, optional
            Metric to use.
        **kwargs
            Additional arguments for ResultMatrix.

        Returns
        -------
        matrix : ResultMatrix
        """
        if not self.results:
            raise ValueError("No results available. Run experiment first.")

        if metric is None:
            metric = list(self._metric_funcs.keys())[0]

        model_names = self.results.get_model_names()
        dataset_names = self.results.get_dataset_names()

        matrix = ResultMatrix(
            model_names=model_names,
            dataset_names=dataset_names,
            metric_name=metric,
            **kwargs
        )

        for dataset_name, dataset_result in self.results.dataset_results.items():
            for model_name, model_result in dataset_result.model_results.items():
                values = model_result.get_metric_values(metric)
                matrix.add_result(dataset_name, model_name, values)

        return matrix

    def to_string(self, metric: str = None, **kwargs) -> str:
        """Get plain text formatted results."""
        return self.to_result_matrix(metric, **kwargs).to_string()

    def to_csv(self, metric: str = None, **kwargs) -> str:
        """Get CSV formatted results."""
        return self.to_result_matrix(metric, **kwargs).to_csv()

    def to_latex(self, metric: str = None, **kwargs) -> str:
        """Get LaTeX formatted results."""
        return self.to_result_matrix(metric, **kwargs).to_latex()

    def to_html(self, metric: str = None, **kwargs) -> str:
        """Get HTML formatted results."""
        return self.to_result_matrix(metric, **kwargs).to_html()

    def to_markdown(self, metric: str = None, **kwargs) -> str:
        """Get Markdown formatted results."""
        return self.to_result_matrix(metric, **kwargs).to_markdown()

    def compare_models(
        self,
        baseline: str = None,
        metric: str = None,
        significance_level: float = 0.05
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compare all models against a baseline.

        Parameters
        ----------
        baseline : str, optional
            Baseline model name. If None, uses first model.
        metric : str, optional
            Metric to compare.
        significance_level : float
            Significance level for t-tests.

        Returns
        -------
        comparisons : dict
            Pairwise comparison results.
        """
        if not self.results:
            raise ValueError("No results available. Run experiment first.")

        if metric is None:
            metric = list(self._metric_funcs.keys())[0]

        model_names = self.results.get_model_names()
        if baseline is None:
            baseline = model_names[0]

        comparisons = {}

        for dataset_name, dataset_result in self.results.dataset_results.items():
            baseline_result = dataset_result.model_results.get(baseline)
            if not baseline_result:
                continue

            baseline_values = baseline_result.get_metric_values(metric)

            for model_name, model_result in dataset_result.model_results.items():
                if model_name == baseline:
                    continue

                model_values = model_result.get_metric_values(metric)

                try:
                    stats = paired_t_test(
                        model_values, baseline_values,
                        significance_level=significance_level
                    )

                    key = f"{dataset_name}_{model_name}_vs_{baseline}"
                    comparisons[key] = {
                        "dataset": dataset_name,
                        "model": model_name,
                        "baseline": baseline,
                        "model_mean": stats.x_mean,
                        "baseline_mean": stats.y_mean,
                        "p_value": stats.p_value,
                        "significant": stats.is_significant(),
                        "winner": model_name if stats.x_better() else (baseline if stats.y_better() else "tie")
                    }
                except Exception:
                    pass

        return comparisons

    def get_scores_matrix(
        self,
        metric: str = None
    ) -> Tuple[np.ndarray, List[str], List[str]]:
        """
        Get scores matrix for visualization.

        Returns
        -------
        scores : ndarray of shape (n_datasets, n_models)
        model_names : list of str
        dataset_names : list of str
        """
        if not self.results:
            raise ValueError("No results available. Run experiment first.")

        if metric is None:
            metric = list(self._metric_funcs.keys())[0]

        model_names = self.results.get_model_names()
        dataset_names = self.results.get_dataset_names()

        scores = np.zeros((len(dataset_names), len(model_names)))

        for i, dataset_name in enumerate(dataset_names):
            dataset_result = self.results.dataset_results[dataset_name]
            for j, model_name in enumerate(model_names):
                model_result = dataset_result.model_results[model_name]
                scores[i, j] = model_result.get_metric_mean(metric)

        return scores, model_names, dataset_names

    def plot_critical_difference(
        self,
        metric: str = None,
        lower_better: bool = False,
        alpha: float = 0.05,
        test: str = 'nemenyi',
        correction: str = 'holm',
        title: str = None,
        save_path: str = None,
        **kwargs
    ):
        """
        Plot Critical Difference diagram from experiment results.

        Parameters
        ----------
        metric : str, optional
            Metric to use (default: first metric).
        lower_better : bool, default=False
            Set True for error metrics.
        alpha : float, default=0.05
            Significance level.
        test : {'nemenyi', 'wilcoxon'}, default='nemenyi'
            Statistical test.
        correction : {'holm', 'bonferroni', 'none'}, default='holm'
            Multiple comparison correction.
        title : str, optional
            Plot title.
        save_path : str, optional
            Path to save the figure.

        Examples
        --------
        >>> exp = run_experiment(models, datasets)
        >>> exp.plot_critical_difference(metric='accuracy')
        >>> exp.plot_critical_difference(metric='error', lower_better=True)
        """
        scores, model_names, dataset_names = self.get_scores_matrix(metric)

        if title is None:
            title = f"Critical Difference Diagram ({metric or 'metric'})"

        return _plot_cd(
            scores=scores,
            names=model_names,
            lower_better=lower_better,
            alpha=alpha,
            test=test,
            correction=correction,
            title=title,
            save_path=save_path,
            **kwargs
        )

    def plot_ranking_table(
        self,
        metric: str = None,
        lower_better: bool = False,
        save_path: str = None,
        **kwargs
    ):
        """
        Plot ranking table from experiment results.

        Parameters
        ----------
        metric : str, optional
            Metric to use.
        lower_better : bool, default=False
            Set True for error metrics.
        save_path : str, optional
            Path to save the figure.
        """
        scores, model_names, dataset_names = self.get_scores_matrix(metric)

        return _plot_ranking(
            scores=scores,
            names=model_names,
            dataset_names=dataset_names,
            lower_better=lower_better,
            metric_name=metric or 'metric',
            save_path=save_path,
            **kwargs
        )

    def plot_boxplot(
        self,
        metric: str = None,
        save_path: str = None,
        **kwargs
    ):
        """
        Plot boxplot comparison from experiment results.

        Parameters
        ----------
        metric : str, optional
            Metric to use.
        save_path : str, optional
            Path to save the figure.
        """
        scores, model_names, dataset_names = self.get_scores_matrix(metric)

        return _plot_boxplot(
            scores=scores,
            names=model_names,
            metric_name=metric or 'metric',
            save_path=save_path,
            **kwargs
        )

def run_experiment(
    models: Dict[str, Any],
    datasets: Dict[str, Tuple[np.ndarray, np.ndarray]],
    n_folds: int = 10,
    metrics: List[str] = None,
    random_state: int = 42,
    n_jobs: int = 1,
    verbose: int = 0,
    progress_callback: Optional[Callable] = None
) -> Experiment:
    """
    Convenience function to run a quick experiment.

    Parameters
    ----------
    models : dict
        Models to compare.
    datasets : dict
        Datasets to evaluate on.
    n_folds : int, default=10
        Number of CV folds.
    metrics : list, optional
        Metrics to compute.
    random_state : int, default=42
        Random seed for reproducibility.
    n_jobs : int, default=1
        Number of parallel jobs (-1 for all CPUs).
    verbose : int, default=0
        Verbosity level.

    Returns
    -------
    experiment : Experiment
        Completed experiment with results.

    Examples
    --------
    >>> from tuiml.evaluation.experiments import run_experiment
    >>> exp = run_experiment(
    ...     models={'RF': RandomForestClassifier(), 'SVM': SVC()},
    ...     datasets={'data': (X, y)},
    ...     n_folds=5,
    ...     metrics=['accuracy_score', 'f1_score']
    ... )
    >>> print(exp.summary())
    """
    config = ExperimentConfig(
        n_folds=n_folds,
        random_state=random_state,
        n_jobs=n_jobs,
        verbose=verbose
    )

    exp = Experiment(config, metrics=metrics)
    exp.run(models, datasets, progress_callback=progress_callback)

    return exp

def _run_single_fold(
    fold_idx: int,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    metric_funcs: Dict[str, Callable],
    verbose: int = 0
) -> FoldResult:
    """Run a single fold for parallel execution."""
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Clone model
    model_clone = deepcopy(model)

    # Train
    train_start = time.time()
    model_clone.fit(X_train, y_train)
    train_time = time.time() - train_start

    # Predict
    test_start = time.time()
    y_pred = model_clone.predict(X_test)
    test_time = time.time() - test_start

    # Get probabilities if available
    y_prob = None
    if hasattr(model_clone, 'predict_proba'):
        try:
            y_prob = model_clone.predict_proba(X_test)
        except Exception:
            pass

    # Compute metrics — retry with average='macro' on TypeError
    # (same pattern as api.evaluate for multi-class metrics like f1_score)
    metrics = {}
    for metric_name, metric_func in metric_funcs.items():
        try:
            metrics[metric_name] = metric_func(y_test, y_pred)
        except TypeError:
            try:
                metrics[metric_name] = metric_func(y_test, y_pred, average='macro')
            except Exception as e:
                if verbose > 1:
                    print(f"    Warning: Could not compute {metric_name}: {e}")
                metrics[metric_name] = np.nan
        except Exception as e:
            if verbose > 1:
                print(f"    Warning: Could not compute {metric_name}: {e}")
            metrics[metric_name] = np.nan

    return FoldResult(
        fold_idx=fold_idx,
        train_indices=train_idx,
        test_indices=test_idx,
        y_true=y_test,
        y_pred=y_pred,
        y_prob=y_prob,
        train_time=train_time,
        test_time=test_time,
        metrics=metrics
    )
