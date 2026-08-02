"""Contract tests run over every algorithm in the registry.

One parametrised test replaces the six invariants that used to be copy-pasted
into every algorithm test module. Coverage now follows the registry rather
than author patience: registering an algorithm subscribes it to the whole
battery, and adding a check in :mod:`tests._contract` applies it to every
algorithm at once.

Algorithm-*specific* behaviour still belongs in the per-algorithm modules
under ``test_algorithms/``. This file only asserts what every algorithm owes
its callers.
"""

import warnings

import pytest

import tuiml
from tuiml.registry import registry

from ._contract import ALL_CHECKS

# Algorithms excluded from the sweep entirely, with the reason. Prefer
# XFAIL_CHECKS below: skipping an algorithm drops it from every check, whereas
# an xfail entry keeps the remaining checks honest.
SKIP_ALGORITHMS = {
    "STLDecomposition": "__init__ requires a `period` argument with no default",
}

# Known contract violations, as {algorithm: {check: reason}}. Every entry is a
# bug to fix, not a permanent exemption -- an empty table is the goal. Listing
# them keeps the suite green while making the debt explicit and greppable, and
# `strict=False` means a fix turns the entry into an XPASS rather than a
# failure, so the table degrades safely as things are repaired.
#
# Generated from an actual run; regenerate rather than hand-editing after
# fixing an algorithm.
XFAIL_CHECKS = {
    'AdditiveRegression': {
        'check_schema_matches_signature':
            "get_parameter_schema omits ['base_regressor']",
    },
    'AgglomerativeClusterer': {
        **{c: 'predict() after fit() requires store_data=True' for c in (
            'check_pickle_roundtrip',
            'check_predict_output_shape',
        )},
    },
    'AveragedPerceptronClassifier': {
        'check_schema_matches_signature':
            "get_parameter_schema omits ['random_state']",
    },
    'BayesianNetworkClassifier': {
        'check_schema_matches_signature':
            "get_parameter_schema omits ['random_state']",
    },
    'CanopyClusterer': {
        'check_schema_matches_signature':
            "get_parameter_schema omits ['random_state']",
    },
    'CatBoostClassifier': {
        'check_schema_matches_signature':
            "get_parameter_schema omits ['cat_features', 'random_state', 'verbose']",
    },
    'CatBoostRegressor': {
        'check_missing_value_support_is_honest':
            'declares missing_values but raises on NaN',
        'check_schema_matches_signature':
            "get_parameter_schema omits ['cat_features', 'random_state', 'verbose']",
    },
    'DBSCANClusterer': {
        **{c: 'IndexError when no core samples are found (empty float index array)' for c in (
            'check_fit_does_not_mutate_input',
            'check_fit_returns_self',
            'check_pickle_roundtrip',
            'check_predict_output_shape',
        )},
    },
    'DecisionTableClassifier': {
        'check_schema_matches_signature':
            "get_parameter_schema omits ['random_state']",
    },
    'FarthestFirstClusterer': {
        'check_schema_matches_signature':
            "get_parameter_schema omits ['random_state']",
    },
    'GaussianMixtureClusterer': {
        'check_schema_matches_signature':
            "get_parameter_schema omits ['random_state']",
    },
    'HoeffdingTreeClassifier': {
        'check_pickle_roundtrip':
            'fitted model is not picklable, so it cannot be saved or served',
    },
    'KMeansClusterer': {
        'check_missing_value_support_is_honest':
            'declares missing_values but raises on NaN',
        'check_schema_matches_signature':
            "get_parameter_schema omits ['random_state']",
    },
    'LightGBMClassifier': {
        'check_schema_matches_signature':
            "get_parameter_schema omits ['random_state', 'verbose']",
    },
    'LightGBMRegressor': {
        'check_schema_matches_signature':
            "get_parameter_schema omits ['random_state', 'verbose']",
    },
    'MultiClassClassifier': {
        **{c: "references unregistered component 'SMO'" for c in (
            'check_fit_does_not_mutate_input',
            'check_fit_returns_self',
            'check_missing_value_support_is_honest',
            'check_pickle_roundtrip',
            'check_predict_output_shape',
            'check_predict_proba_is_a_distribution',
            'check_seeded_fit_is_reproducible',
        )},
    },
    'MultilayerPerceptronClassifier': {
        'check_schema_matches_signature':
            "get_parameter_schema omits ['random_state']",
    },
    'MultilayerPerceptronRegressor': {
        'check_schema_matches_signature':
            "get_parameter_schema omits ['random_state']",
    },
    'NaiveBayesClassifier': {
        'check_schema_matches_signature':
            "get_parameter_schema omits ['var_smoothing']",
    },
    'NaiveBayesMultinomialClassifier': {
        **{c: "declares 'numeric' but rejects negative features" for c in (
            'check_fit_does_not_mutate_input',
            'check_fit_returns_self',
            'check_pickle_roundtrip',
            'check_predict_output_shape',
            'check_predict_proba_is_a_distribution',
        )},
    },
    'PARTClassifier': {
        'check_schema_matches_signature':
            "get_parameter_schema omits ['random_state']",
    },
    'PerceptronClassifier': {
        'check_schema_matches_signature':
            "get_parameter_schema omits ['random_state']",
    },
    'Prophet': {
        'check_schema_matches_signature':
            "get_parameter_schema omits ['changepoints', 'holidays_prior_scale', 'interval_width', 'mcmc_samples', 'uncertainty_samples']",
    },
    'RIPPERClassifier': {
        'check_schema_matches_signature':
            "get_parameter_schema omits ['random_state']",
    },
    'ReducedErrorPruningTreeClassifier': {
        'check_schema_matches_signature':
            "get_parameter_schema omits ['random_state']",
    },
    'ReducedErrorPruningTreeRegressor': {
        'check_schema_matches_signature':
            "get_parameter_schema omits ['random_state']",
    },
    'RegressionByDiscretization': {
        'check_schema_matches_signature':
            "get_parameter_schema omits ['base_classifier']",
    },
    'SVR': {
        'check_pickle_roundtrip':
            'fitted model is not picklable, so it cannot be saved or served',
    },
    'StackingClassifier': {
        **{c: "references unregistered component 'Logistic'" for c in (
            'check_fit_does_not_mutate_input',
            'check_fit_returns_self',
            'check_pickle_roundtrip',
            'check_predict_output_shape',
            'check_predict_proba_is_a_distribution',
            'check_seeded_fit_is_reproducible',
        )},
    },
    'VotedPerceptronClassifier': {
        'check_schema_matches_signature':
            "get_parameter_schema omits ['random_state']",
    },
    'XGBoostClassifier': {
        'check_schema_matches_signature':
            "get_parameter_schema omits ['objective', 'random_state']",
    },
    'XGBoostRegressor': {
        'check_missing_value_support_is_honest':
            'declares missing_values but raises on NaN',
        'check_schema_matches_signature':
            "get_parameter_schema omits ['objective', 'random_state']",
    },
}


def _registered_algorithms():
    """Yield ``(name, cls)`` for every native algorithm in the registry.

    Third-party wrappers (``sklearn.*``, ``capymoa.*``) are excluded: they obey
    their upstream library's contract, not TuiML's. Versioned aliases are
    excluded so each algorithm is checked once, and anything defined outside
    the ``tuiml`` package is excluded so the suite does not depend on which
    user-authored algorithms happen to sit in ``~/.tuiml/user_algorithms``.

    Returns
    -------
    algorithms : list of tuple
        ``(name, cls)`` pairs, sorted by name.
    """
    out, seen = [], set()
    for info in tuiml.list_algorithms():
        name = info["name"]
        if "." in name or "_v" in name or name in SKIP_ALGORITHMS or name in seen:
            continue
        try:
            cls = registry.get(name)
        except Exception:
            continue
        if not getattr(cls, "__module__", "").startswith("tuiml."):
            continue
        seen.add(name)
        out.append((name, cls))
    return sorted(out, key=lambda pair: pair[0])


ALGORITHMS = _registered_algorithms()


def _cases():
    """Build the (algorithm, check) grid, marking known failures xfail.

    Returns
    -------
    params : list of pytest.param
        One param per (algorithm, check) pair.
    """
    params = []
    for name, cls in ALGORITHMS:
        for check in ALL_CHECKS:
            reason = XFAIL_CHECKS.get(name, {}).get(check.__name__)
            marks = [pytest.mark.xfail(reason=reason, strict=False)] if reason else []
            params.append(
                pytest.param(name, cls, check, id=f"{name}-{check.__name__}", marks=marks)
            )
    return params


def test_registry_is_not_empty():
    """Guard against the sweep silently covering nothing."""
    assert len(ALGORITHMS) > 50, (
        f"only {len(ALGORITHMS)} algorithms discovered; the registry filter is "
        f"probably wrong and the contract suite is testing almost nothing"
    )


@pytest.mark.parametrize("name, cls, check", _cases())
def test_algorithm_contract(name, cls, check):
    """Every registered algorithm satisfies every contract check."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        check(name, cls())
