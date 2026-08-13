"""Visualizations."""

import os
import uuid
from typing import Any, Dict

from .._spec import ToolSpec
from .._shared import _load_model_from_disk, _load_data


def execute_plot(**kwargs) -> Dict[str, Any]:
    """Generate a visualization and return it as a saved PNG plus base64.

    Backs the ``tuiml_plot`` tool. Supported plot types: confusion_matrix,
    roc_curve, pr_curve, learning_curve, tree, feature_importance,
    cd_diagram, boxplot_comparison, heatmap, ranking_table.

    Parameters
    ----------
    plot_type : str
        Which plot to produce (arrives via ``**kwargs``, like all
        parameters below).
    title : str, default=None
        Custom plot title; each plot type has a sensible default.
    model_id : str, default=None
        Trained model to plot from (model-based plot types).
    model_path : str, default=None
        Explicit path to a serialized model file.
    data : str, default=None
        Dataset to evaluate/plot against (dataset_id, path, or built-in
        name); required by data-driven plot types.
    normalize : bool, default=False
        Normalize counts in the confusion matrix.
    algorithm : str, default=None
        Algorithm name; required for ``learning_curve``.
    benchmark_results : dict, default=None
        ``{algorithm_name: [scores...]}`` mapping; required for the
        comparison plots (cd_diagram, boxplot_comparison, heatmap,
        ranking_table).

    Returns
    -------
    result : dict
        On success: ``status`` (``'success'``), ``plot_type``,
        ``description``, ``path`` (saved PNG on disk), ``_image_base64``
        and ``_image_mime``. On failure: ``status`` (``'error'``),
        ``error`` and optionally ``error_type`` / ``suggestion``.
    """
    import base64
    import numpy as np

    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt

        plot_type = kwargs['plot_type']
        title = kwargs.get('title')

        # Save plot to a persistent, discoverable directory so the AI can
        # reference the file path in markdown reports (in addition to seeing
        # the inline image). Override via $TUIML_PLOT_DIR.
        from pathlib import Path
        plot_dir = Path(os.environ.get('TUIML_PLOT_DIR',
                                       str(Path.home() / '.tuiml' / 'plots')))
        plot_dir.mkdir(parents=True, exist_ok=True)
        plot_path = str(plot_dir / f'{plot_type}_{uuid.uuid4().hex[:8]}.png')

        if plot_type == 'confusion_matrix':
            from tuiml.evaluation.visualization import plot_confusion_matrix
            import tuiml

            model = _load_model_from_disk(kwargs.get('model_id'), kwargs.get('model_path'))
            if model is None:
                return {'status': 'error', 'error': 'Model not found. Provide model_id or model_path.'}

            dataset = _load_data(kwargs['data'])
            predictions = model.predict(dataset.X)
            plot_confusion_matrix(
                dataset.y, predictions,
                title=title or 'Confusion Matrix',
                save_path=plot_path,
                normalize=kwargs.get('normalize', False),
            )
            description = 'Confusion matrix showing predicted vs actual class labels.'

        elif plot_type == 'roc_curve':
            from tuiml.evaluation.visualization import plot_roc_curve
            import tuiml

            model = _load_model_from_disk(kwargs.get('model_id'), kwargs.get('model_path'))
            if model is None:
                return {'status': 'error', 'error': 'Model not found. Provide model_id or model_path.'}

            dataset = _load_data(kwargs['data'])

            if not hasattr(model, 'predict_proba'):
                return {
                    'status': 'error',
                    'error': 'Model does not support predict_proba, required for ROC curve.',
                    'suggestion': 'Use a classifier that supports probability estimates (e.g., RandomForestClassifier, NaiveBayesClassifier).'
                }
            probas = model.predict_proba(dataset.X)
            # Multiclass (>2 columns): pass the full proba matrix + class
            # labels so plot_roc_curve draws per-class OvR curves + macro avg.
            # Binary (1- or 2-D): pass positive-class probabilities.
            classes = getattr(model, 'classes_', None)
            if probas.ndim == 2 and probas.shape[1] > 2:
                y_score = probas
                desc_suffix = ' (one-vs-rest per class, plus macro-average)'
            elif probas.ndim == 2:
                y_score = probas[:, 1]
                desc_suffix = ''
            else:
                y_score = probas
                desc_suffix = ''

            plot_roc_curve(
                dataset.y, y_score,
                title=title or 'ROC Curve',
                save_path=plot_path,
                classes=list(classes) if classes is not None else None,
            )
            description = f'ROC curve with AUC score.{desc_suffix}'

        elif plot_type == 'pr_curve':
            from tuiml.evaluation.visualization import plot_pr_curve
            import tuiml

            model = _load_model_from_disk(kwargs.get('model_id'), kwargs.get('model_path'))
            if model is None:
                return {'status': 'error', 'error': 'Model not found. Provide model_id or model_path.'}

            dataset = _load_data(kwargs['data'])

            if not hasattr(model, 'predict_proba'):
                return {
                    'status': 'error',
                    'error': 'Model does not support predict_proba, required for PR curve.',
                    'suggestion': 'Use a classifier that supports probability estimates.'
                }
            probas = model.predict_proba(dataset.X)
            if probas.ndim == 2:
                y_score = probas[:, 1]
            else:
                y_score = probas

            plot_pr_curve(
                dataset.y, y_score,
                title=title or 'Precision-Recall Curve',
                save_path=plot_path,
            )
            description = 'Precision-Recall curve with Average Precision score.'

        elif plot_type == 'learning_curve':
            from tuiml.evaluation.visualization import plot_learning_curve
            from tuiml.registry import registry
            from tuiml.evaluation.metrics import accuracy_score
            import tuiml.algorithms  # noqa: F401 - trigger registration

            algorithm_name = kwargs.get('algorithm')
            if not algorithm_name:
                return {'status': 'error', 'error': 'algorithm parameter is required for learning_curve.'}

            dataset = _load_data(kwargs['data'])

            algo_cls = registry.get(algorithm_name)
            if algo_cls is None:
                return {'status': 'error', 'error': f"Algorithm '{algorithm_name}' not found."}

            # Compute learning curve manually with k-fold CV
            from tuiml.evaluation.splitting import KFold
            n_splits = 5
            train_fractions = np.linspace(0.1, 1.0, 10)
            n_samples = len(dataset.y)
            from tuiml.utils.seed import get_global_seed
            seed = get_global_seed()
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed if seed is not None else 42)

            all_train_sizes = []
            all_train_scores = []  # shape: (n_sizes, n_splits)
            all_test_scores = []

            for frac in train_fractions:
                fold_train_scores = []
                fold_test_scores = []
                for train_idx, test_idx in kf.split(dataset.X):
                    X_train_full, y_train_full = dataset.X[train_idx], dataset.y[train_idx]
                    X_test, y_test = dataset.X[test_idx], dataset.y[test_idx]

                    # Subsample training set
                    subset_size = max(2, int(len(X_train_full) * frac))
                    X_train = X_train_full[:subset_size]
                    y_train = y_train_full[:subset_size]

                    model_lc = algo_cls()
                    model_lc.fit(X_train, y_train)

                    train_pred = model_lc.predict(X_train)
                    test_pred = model_lc.predict(X_test)
                    fold_train_scores.append(accuracy_score(y_train, train_pred))
                    fold_test_scores.append(accuracy_score(y_test, test_pred))

                # Use the actual training size for the first fold as representative
                all_train_sizes.append(max(2, int(len(dataset.X) * (n_splits - 1) / n_splits * frac)))
                all_train_scores.append(fold_train_scores)
                all_test_scores.append(fold_test_scores)

            train_sizes_arr = np.array(all_train_sizes)
            train_scores_arr = np.array(all_train_scores)
            test_scores_arr = np.array(all_test_scores)

            plot_learning_curve(
                train_sizes_arr,
                train_scores_arr,
                test_scores_arr,
                title=title or 'Learning Curve',
                save_path=plot_path,
                metric_name='Accuracy',
            )
            description = f'Learning curve for {algorithm_name} showing training vs validation accuracy.'

        elif plot_type == 'tree':
            from tuiml.evaluation.visualization import plot_tree

            model = _load_model_from_disk(kwargs.get('model_id'), kwargs.get('model_path'))
            if model is None:
                return {'status': 'error', 'error': 'Model not found. Provide model_id or model_path.'}

            # tuiml_train saves a Workflow, so a model_id always resolves to
            # one; the tree lives in its final estimator. Handing the wrapper
            # to plot_tree fails its fitted-check ("The tree model is not
            # fitted yet"), because a Workflow carries none of the markers it
            # looks for. A model_path may point at a bare estimator, so fall
            # back to the object itself.
            estimator = getattr(model, 'model_', None)
            if estimator is None:
                estimator = model

            # Only trees, tree ensembles and stumps can be drawn. Say so
            # plainly rather than letting plot_tree report a fitted model as
            # unfitted.
            #
            # `estimators_` alone is not evidence of an ensemble of trees:
            # NaiveBayesClassifier stores a list-of-lists of *probability*
            # estimators under the same name, and plot_tree would happily
            # draw nonsense from it. Require the members to be trees.
            def _is_tree_like(obj):
                return getattr(obj, 'tree_', None) is not None or hasattr(obj, 'is_leaf')

            members = getattr(estimator, 'estimators_', None) or []
            if not (
                _is_tree_like(estimator)
                or getattr(estimator, 'feature_index_', None) is not None
                or (len(members) > 0 and _is_tree_like(members[0]))
            ):
                return {
                    'status': 'error',
                    'error': (
                        f"'{type(estimator).__name__}' is not a tree-based model, so "
                        f"plot_type='tree' does not apply. Use it with a decision "
                        f"tree, a tree ensemble (e.g. RandomForestClassifier) or a "
                        f"decision stump; for other models try "
                        f"plot_type='feature_importance'."
                    ),
                }

            # Get feature names from the model if available
            feature_names = None
            for source in (estimator, model):
                if getattr(source, 'feature_names_', None) is not None:
                    feature_names = source.feature_names_
                    break

            plot_tree(
                estimator,
                feature_names=feature_names,
                filled=True,
                rounded=True,
                title=title or 'Decision Tree',
                save_path=plot_path,
            )
            description = 'Decision tree structure visualization.'

        elif plot_type == 'feature_importance':
            model = _load_model_from_disk(kwargs.get('model_id'), kwargs.get('model_path'))
            if model is None:
                return {'status': 'error', 'error': 'Model not found. Provide model_id or model_path.'}

            importances = None

            # Try direct attribute first
            if hasattr(model, 'feature_importances_') and model.feature_importances_ is not None:
                importances = np.array(model.feature_importances_)

            # Try wrapped inner model (e.g., XGBoost, GradientBoosting store
            # the sklearn-compatible model in model.model_)
            if importances is None and hasattr(model, 'model_'):
                inner = model.model_
                if hasattr(inner, 'feature_importances_') and inner.feature_importances_ is not None:
                    importances = np.array(inner.feature_importances_)

            # Try coef_ for linear models (Logistic Regression, SVM, etc.)
            if importances is None:
                coef = getattr(model, 'coef_', None)
                if coef is None and hasattr(model, 'model_'):
                    coef = getattr(model.model_, 'coef_', None)
                if coef is not None:
                    coef = np.array(coef)
                    if coef.ndim > 1:
                        importances = np.mean(np.abs(coef), axis=0)
                    else:
                        importances = np.abs(coef)

            def _count_feature_usage(node, counts):
                """Recursively count feature usage in a tree node."""
                if node is None or getattr(node, 'is_leaf', True):
                    return
                feat = getattr(node, 'feature_index', None)
                if feat is not None and feat >= 0:
                    samples = getattr(node, 'n_samples', 1)
                    counts[feat] += samples
                _count_feature_usage(getattr(node, 'left', None), counts)
                _count_feature_usage(getattr(node, 'right', None), counts)

            # For ensemble models, compute from estimators' trees
            if importances is None and hasattr(model, 'estimators_') and model.estimators_:
                n_features = getattr(model, 'n_features_', None)
                if n_features and all(hasattr(est, 'tree_') for est in model.estimators_):
                    total = np.zeros(n_features)
                    for est in model.estimators_:
                        _count_feature_usage(est.tree_, total)
                    if total.sum() > 0:
                        importances = total / total.sum()

            # For single tree models
            if importances is None and hasattr(model, 'tree_'):
                n_features = getattr(model, 'n_features_', None)
                if n_features:
                    total = np.zeros(n_features)
                    _count_feature_usage(model.tree_, total)
                    if total.sum() > 0:
                        importances = total / total.sum()

            if importances is None:
                return {
                    'status': 'error',
                    'error': 'Cannot compute feature importances from this model.',
                    'suggestion': 'Use a tree-based model (e.g., RandomForestClassifier, XGBoostClassifier, DecisionTreeClassifier).'
                }

            importances = np.array(importances)
            feature_names = None
            if hasattr(model, 'feature_names_'):
                feature_names = model.feature_names_
            if feature_names is None and hasattr(model, 'model_'):
                inner = model.model_
                if hasattr(inner, 'feature_names_in_'):
                    feature_names = list(inner.feature_names_in_)
            if feature_names is None:
                feature_names = [f'Feature {i}' for i in range(len(importances))]

            # Sort by importance
            indices = np.argsort(importances)[::-1]
            sorted_names = [feature_names[i] for i in indices]
            sorted_importances = importances[indices]

            from tuiml.evaluation.visualization import setup_figure, style_axis, get_colors
            fig, ax = setup_figure(figsize=(10, max(6, len(sorted_names) * 0.35)))
            colors = get_colors(len(sorted_names))

            ax.barh(range(len(sorted_names)), sorted_importances[::-1], color=colors[0])
            ax.set_yticks(range(len(sorted_names)))
            ax.set_yticklabels(sorted_names[::-1])
            style_axis(
                ax,
                title=title or 'Feature Importance',
                xlabel='Importance',
                ylabel=None,
                legend=False,
            )
            fig.tight_layout()
            fig.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            description = 'Feature importance bar chart from model.'

        elif plot_type in ('cd_diagram', 'boxplot_comparison', 'heatmap', 'ranking_table'):
            benchmark_results = kwargs.get('benchmark_results')
            if not benchmark_results:
                return {
                    'status': 'error',
                    'error': f"'{plot_type}' requires benchmark_results parameter with algorithm CV scores.",
                    'suggestion': "Provide benchmark_results: { 'AlgoName': [score1, score2, ...], ... }"
                }

            scores_dict = {
                name: np.array(scores) for name, scores in benchmark_results.items()
            }

            if plot_type == 'cd_diagram':
                from tuiml.evaluation.visualization import plot_critical_difference
                plot_critical_difference(
                    scores=scores_dict,
                    title=title or 'Critical Difference Diagram',
                    save_path=plot_path,
                )
                description = 'Critical difference diagram showing statistically significant differences between algorithms.'

            elif plot_type == 'boxplot_comparison':
                from tuiml.evaluation.visualization import plot_boxplot_comparison
                plot_boxplot_comparison(
                    scores=scores_dict,
                    save_path=plot_path,
                )
                description = 'Box plot comparison of algorithm cross-validation scores.'

            elif plot_type == 'heatmap':
                from tuiml.evaluation.visualization import plot_heatmap
                plot_heatmap(
                    scores=scores_dict,
                    save_path=plot_path,
                )
                description = 'Heatmap of algorithm scores across datasets.'

            elif plot_type == 'ranking_table':
                from tuiml.evaluation.visualization import plot_ranking_table
                plot_ranking_table(
                    scores=scores_dict,
                    title=title or 'Algorithm Ranking',
                    save_path=plot_path,
                )
                description = 'Ranking table of algorithm performance.'

        else:
            return {'status': 'error', 'error': f"Unknown plot_type: '{plot_type}'"}

        # Close any remaining figures to free memory
        plt.close('all')

        # Read the saved plot and base64 encode it (so the AI can see it
        # inline). We keep the file on disk so the AI can also embed it
        # in markdown reports via the returned `path`.
        with open(plot_path, 'rb') as f:
            image_bytes = f.read()
        image_b64 = base64.b64encode(image_bytes).decode('utf-8')

        return {
            'status': 'success',
            'plot_type': plot_type,
            'description': description,
            'path': plot_path,
            '_image_base64': image_b64,
            '_image_mime': 'image/png',
        }

    except Exception as e:
        plt.close('all')
        return {
            'status': 'error',
            'error': str(e),
            'error_type': type(e).__name__
        }


SPEC = ToolSpec(
    name='tuiml_plot',
    description="Generate a visualization/plot for model analysis. Returns the plot as an "
        "inline image. Supported plot types: confusion_matrix, roc_curve, pr_curve, "
        "learning_curve, tree, feature_importance.",
    input_schema={
            "type": "object",
            "properties": {
                "plot_type": {
                    "type": "string",
                    "enum": [
                        "confusion_matrix",
                        "roc_curve",
                        "pr_curve",
                        "learning_curve",
                        "tree",
                        "feature_importance",
                        "cd_diagram",
                        "boxplot_comparison",
                        "heatmap",
                        "ranking_table"
                    ],
                    "description": (
                        "Type of plot to generate:\n"
                        "- confusion_matrix: Heatmap of predicted vs actual classes (requires model_id + data + target)\n"
                        "- roc_curve: ROC curve with AUC for binary or multiclass classifiers (requires model_id + data + target)\n"
                        "- pr_curve: Precision-Recall curve with AP for binary classifiers (requires model_id + data + target)\n"
                        "- learning_curve: Training vs validation score over dataset sizes (requires algorithm + data + target)\n"
                        "- tree: Decision tree structure visualization (requires model_id)\n"
                        "- feature_importance: Bar chart of feature importances (requires model_id)\n"
                        "- cd_diagram: Critical difference diagram for algorithm comparison (requires benchmark_results)\n"
                        "- boxplot_comparison: Box plot comparing algorithm scores (requires benchmark_results)\n"
                        "- heatmap: Heatmap of algorithm scores across datasets (requires benchmark_results)\n"
                        "- ranking_table: Ranking table of algorithms (requires benchmark_results)"
                    )
                },
                "model_id": {
                    "type": "string",
                    "description": "Model ID from tuiml_train (required for most plot types)"
                },
                "model_path": {
                    "type": "string",
                    "description": "Path to saved model file (alternative to model_id)"
                },
                "data": {
                    "type": "string",
                    "description": "Data file path or built-in dataset name (required for confusion_matrix, roc_curve, pr_curve, learning_curve)"
                },
                "target": {
                    "type": "string",
                    "description": "Target column name (required for confusion_matrix, roc_curve, pr_curve, learning_curve)"
                },
                "algorithm": {
                    "type": "string",
                    "description": "Algorithm class name (required for learning_curve)"
                },
                "title": {
                    "type": "string",
                    "description": "Custom plot title (optional)"
                },
                "normalize": {
                    "type": "boolean",
                    "default": False,
                    "description": "Normalize confusion matrix to show percentages (confusion_matrix only)"
                },
                "benchmark_results": {
                    "type": "object",
                    "description": "Algorithm CV scores for comparison plots: { 'AlgoName': [score1, score2, ...], ... }"
                }
            },
            "required": ["plot_type"]
        },
    output_schema={
            "type": "object",
            "properties": {
                "status": {"type": "string", "enum": ["success", "error"]},
                "plot_type": {"type": "string"},
                "description": {"type": "string"},
                "error": {"type": "string"}
            },
            "required": ["status"]
        },
    execute=execute_plot,
    group='workflow',
    read_only=True, destructive=False,
    idempotent=True, open_world=False,
)
