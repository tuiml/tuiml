"""Successive-halving search over an exhaustive parameter grid."""

from .successive_halving import SuccessiveHalvingSearchCV
from tuiml.base.tuning import ParameterGrid


class HalvingGridSearchCV(SuccessiveHalvingSearchCV):
    """Successive-halving search over every combination in a parameter grid.

    Parameters
    ----------
    estimator : Algorithm
        The model template to optimize.
    param_grid : dict or list of dict
        Parameter grid defining the combinations to evaluate.
    factor : int, default=3
        Reduction factor between successive rounds.
    resource : str, default="n_samples"
        Resource increased between successive rounds.
    min_resource : int or str, default="auto"
        Minimum resource used in the first round.
    max_resource : int or str, default="auto"
        Maximum resource used in the final round.
    aggressive_elimination : bool, default=False
        Whether to use aggressive candidate elimination.
    scoring : str or callable, default="accuracy"
        Metric used to rank candidates.
    cv : int, default=5
        Number of cross-validation folds.
    refit : bool, default=True
        Whether to refit the best estimator on the complete dataset.
    verbose : int, default=0
        Verbosity level.
    n_jobs : int, default=1
        Number of parallel jobs used during cross-validation.
    random_seed : int, optional
        Seed used for reproducible candidate ordering and CV splits.
    progress_callback : callable, optional
        Callback invoked after each evaluated parameter combination.

    Examples
    --------
    >>> from tuiml.evaluation.tuning import HalvingGridSearchCV
    >>> from tuiml.algorithms.trees import RandomForestClassifier
    >>> from tuiml.datasets import load_iris
    >>> data = load_iris()
    >>> search = HalvingGridSearchCV(
    ...     RandomForestClassifier(),
    ...     {"max_depth": [2, 4], "n_estimators": [5, 10]},
    ...     factor=2,
    ...     cv=3,
    ...     random_seed=0,
    ... )
    >>> _ = search.fit(data.X, data.y)
    >>> type(search).__name__
    'HalvingGridSearchCV'
    >>> len(search.cv_results_["params"]) > 0
    True
    """

    def __init__(
        self,
        estimator,
        param_grid,
        factor: int = 3,
        resource: str = "n_samples",
        min_resource="auto",
        max_resource="auto",
        aggressive_elimination: bool = False,
        scoring="accuracy",
        cv: int = 5,
        refit: bool = True,
        verbose: int = 0,
        n_jobs: int = 1,
        random_seed=None,
        progress_callback=None,
    ):
        """Initialize the exhaustive successive-halving search."""
        grid = ParameterGrid(param_grid)

        super().__init__(
            estimator=estimator,
            param_distributions=grid,
            n_candidates=len(grid),
            factor=factor,
            resource=resource,
            min_resource=min_resource,
            max_resource=max_resource,
            aggressive_elimination=aggressive_elimination,
            scoring=scoring,
            cv=cv,
            refit=refit,
            verbose=verbose,
            n_jobs=n_jobs,
            random_seed=random_seed,
            progress_callback=progress_callback,
        )

        self.param_grid = param_grid
