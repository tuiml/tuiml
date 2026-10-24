"""Successive-halving search over randomly sampled parameter configurations."""

from .successive_halving import SuccessiveHalvingSearchCV


class HalvingRandomSearchCV(SuccessiveHalvingSearchCV):
    """Successive-halving search over randomly sampled configurations.

    Parameters
    ----------
    estimator : Algorithm
        The model template to optimize.
    param_distributions : dict or ParameterDistribution
        Parameter distributions from which candidate configurations are
        sampled.
    n_candidates : int, default=27
        Number of configurations sampled for the initial round.
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
        Seed used for reproducible candidate sampling and CV splits.
    progress_callback : callable, optional
        Callback invoked after each evaluated parameter combination.

    Examples
    --------
    >>> from tuiml.evaluation.tuning import HalvingRandomSearchCV
    >>> from tuiml.algorithms.trees import RandomForestClassifier
    >>> from tuiml.base.tuning import ParameterDistribution
    >>> from tuiml.datasets import load_iris
    >>> data = load_iris()
    >>> space = ParameterDistribution({
    ...     "max_depth": (2, 12, "int"),
    ...     "n_estimators": (5, 40, "int"),
    ... })
    >>> search = HalvingRandomSearchCV(
    ...     RandomForestClassifier(),
    ...     space,
    ...     n_candidates=9,
    ...     factor=3,
    ...     cv=3,
    ...     random_seed=0,
    ... )
    >>> _ = search.fit(data.X, data.y)
    >>> type(search).__name__
    'HalvingRandomSearchCV'
    >>> search.n_candidates_per_round_
    [9, 3, 1]
    """

    def __init__(
        self,
        estimator,
        param_distributions,
        n_candidates: int = 27,
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
        """Initialize the random successive-halving search."""
        super().__init__(
            estimator=estimator,
            param_distributions=param_distributions,
            n_candidates=n_candidates,
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
