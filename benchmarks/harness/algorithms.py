"""The matching-algorithm registry: only algorithms available in ALL three
libraries (tuiml, scikit-learn, python-weka-wrapper3).

Two configurations are defined for every algorithm, selected with ``--config``:

``defaults``
    Each library exactly as it ships. This is what a user gets out of the box,
    and it is a legitimate thing to report — but it is **not** a like-for-like
    comparison, because the libraries disagree about defaults in ways that
    change both cost and quality. Weka's ``LinearRegression``, for instance,
    runs M5 attribute selection by default; scikit-learn's is plain OLS.

``matched``
    Hyperparameters aligned across the three libraries so the same model is
    being fitted wherever that is possible. This is the only defensible basis
    for runtime and accuracy claims. Where the underlying algorithms are simply
    not the same (C4.5 vs CART, online backprop vs Adam), ``matched`` aligns
    what it can and the residual mismatch is recorded in ``note`` — which is
    written into every result JSON rather than left in a comment here.

Data-dependent options are written as placeholders and filled in by
:func:`resolve`:

``$GAMMA``
    RBF kernel width, using scikit-learn's ``gamma="scale"`` definition
    ``1 / (n_features * X.var())``, computed once on the one-hot representation
    and given explicitly to all three libraries.
``$SQRT_P``
    ``floor(sqrt(n_attributes))`` for random-forest feature subsampling,
    evaluated against the attribute count each library actually sees (Weka
    keeps nominal attributes intact where the others one-hot expand them).
"""

import math

# task:     "classification" or "regression"
# prep:     "standard" (default) or "discretized"
# defaults: sklearn -> (module, class_name, kwargs)
#           tuiml   -> (registry_name, kwargs)
#           weka    -> (classname, options_list)
# matched:  per-framework kwargs / options replacing the defaults' last element.
ALGORITHMS = {
    # ---------------------------- classifiers ----------------------------
    "random_forest": {
        "task": "classification",
        "defaults": {
            "sklearn": ("sklearn.ensemble", "RandomForestClassifier",
                        {"n_estimators": 100, "random_state": 42}),
            "tuiml": ("RandomForestClassifier", {"n_estimators": 100}),
            "weka": ("weka.classifiers.trees.RandomForest", ["-I", "100"]),
        },
        # Weka picks log2(p)+1 attributes per split, scikit-learn sqrt(p);
        # matched mode puts all three on sqrt(p), unlimited depth, one instance
        # per leaf, 100% bootstrap bags.
        "matched": {
            "sklearn": {"n_estimators": 100, "max_features": "sqrt",
                        "min_samples_leaf": 1, "bootstrap": True, "random_state": 42},
            "tuiml": {"n_estimators": 100, "max_features": "sqrt",
                      "min_samples_leaf": 1, "bootstrap": True, "random_state": 42},
            "weka": ["-I", "100", "-K", "$SQRT_P", "-depth", "0", "-M", "1",
                     "-P", "100", "-S", "42"],
        },
        "note": ("sqrt(p) is evaluated on each library's own representation: "
                 "Weka splits on intact nominal attributes, scikit-learn/TuiML "
                 "on one-hot columns."),
    },
    "decision_tree": {
        "task": "classification",
        "defaults": {
            "sklearn": ("sklearn.tree", "DecisionTreeClassifier", {"random_state": 42}),
            "tuiml": ("DecisionTreeClassifier", {}),
            "weka": ("weka.classifiers.trees.J48", []),
        },
        # J48 prunes by default (confidence 0.25) and splits nominal attributes
        # multi-way; CART does neither. Matched mode unprunes J48, forces binary
        # splits, and switches the CART implementations to information gain.
        "matched": {
            "sklearn": {"criterion": "entropy", "min_samples_leaf": 1,
                        "random_state": 42},
            "tuiml": {"criterion": "entropy", "min_samples_leaf": 1,
                      "random_state": 42},
            "weka": ["-U", "-B", "-M", "1"],
        },
        "note": ("C4.5 vs CART: even matched, J48 splits on gain ratio while "
                 "scikit-learn/TuiML use information gain. Related algorithms, "
                 "not identical ones."),
    },
    "naive_bayes": {
        "task": "classification",
        # Categorical-aware NB: features are ordinal-encoded / quantile-binned to
        # a discrete matrix (see Prepared.discretized). All three use their native
        # categorical NB on that representation, rather than Gaussian NB over
        # standardized one-hot columns. Laplace alpha=1 is the default everywhere,
        # so defaults and matched coincide.
        "prep": "discretized",
        "defaults": {
            "sklearn": ("sklearn.naive_bayes", "CategoricalNB", {"alpha": 1.0}),
            "tuiml": ("CategoricalNBClassifier", {"alpha": 1.0}),
            "weka": ("weka.classifiers.bayes.NaiveBayes", []),
        },
        "matched": {
            "sklearn": {"alpha": 1.0},
            "tuiml": {"alpha": 1.0},
            "weka": [],
        },
        "note": "All three use add-one (Laplace) smoothing on discrete attributes.",
    },
    "logistic": {
        "task": "classification",
        "defaults": {
            "sklearn": ("sklearn.linear_model", "LogisticRegression", {"max_iter": 1000}),
            "tuiml": ("LogisticRegression", {}),
            "weka": ("weka.classifiers.functions.Logistic", []),
        },
        # scikit-learn minimizes 0.5||w||^2 + C*sum(loss); Weka/TuiML minimize
        # sum(loss) + ridge*||w||^2. So ridge = 1/(2C): C=1.0 <-> ridge=0.5.
        # Weka's default ridge of 1e-8 is effectively unregularized, which is a
        # different model, not just a different constant.
        "matched": {
            "sklearn": {"C": 1.0, "max_iter": 1000},
            "tuiml": {"ridge": 0.5, "max_iter": 1000},
            "weka": ["-R", "0.5", "-M", "1000"],
        },
        "note": "Ridge 0.5 in Weka/TuiML corresponds to scikit-learn's C=1.0.",
    },
    "knn": {
        "task": "classification",
        "defaults": {
            "sklearn": ("sklearn.neighbors", "KNeighborsClassifier", {"n_neighbors": 5}),
            "tuiml": ("KNearestNeighborsClassifier", {"k": 5}),
            "weka": ("weka.classifiers.lazy.IBk", ["-K", "5"]),
        },
        # IBk min-max normalizes inside EuclideanDistance by default, on top of
        # the standardization the harness already applied — a different metric
        # from the other two. -D switches that off.
        "matched": {
            "sklearn": {"n_neighbors": 5, "weights": "uniform"},
            "tuiml": {"k": 5, "distance_weighting": "uniform"},
            "weka": ["-K", "5", "-A",
                     'weka.core.neighboursearch.LinearNNSearch -A '
                     '"weka.core.EuclideanDistance -R first-last -D"'],
        },
        "note": ("Nominal attributes contribute a distance of 1 per mismatch in "
                 "Weka but sqrt(2) across the corresponding one-hot columns. "
                 "Neighbour-search structure is left to each library."),
    },
    "svm": {
        "task": "classification",
        "defaults": {
            "sklearn": ("sklearn.svm", "SVC", {}),
            "tuiml": ("SVC", {}),
            "weka": ("weka.classifiers.functions.SMO", []),
        },
        # SMO defaults to a linear PolyKernel and normalizes internally; SVC
        # defaults to RBF. Matched mode puts both on RBF with the *same*
        # explicit gamma, C=1, and no extra normalization (-N 2).
        "matched": {
            "sklearn": {"C": 1.0, "kernel": "rbf", "gamma": "$GAMMA"},
            "tuiml": {"C": 1.0, "kernel": "rbf", "gamma": "$GAMMA"},
            "weka": ["-N", "2", "-C", "1.0", "-K",
                     "weka.classifiers.functions.supportVector.RBFKernel -G $GAMMA"],
        },
        "note": ("Both use one-vs-one for multiclass. Optimizer tolerances "
                 "differ (SMO -T 1e-3 / -P 1e-12 vs libsvm tol 1e-3)."),
    },
    "mlp": {
        "task": "classification",
        "defaults": {
            "sklearn": ("sklearn.neural_network", "MLPClassifier",
                        {"max_iter": 300, "random_state": 42}),
            "tuiml": ("MultilayerPerceptronClassifier", {}),
            "weka": ("weka.classifiers.functions.MultilayerPerceptron", []),
        },
        # Architecture and epochs aligned (one hidden layer of 100 units, 300
        # epochs, sigmoid units, lr 0.3, momentum 0.2). -I/-C switch off Weka's
        # internal attribute/class normalization, which the harness has already
        # applied and the other two libraries do not do.
        "matched": {
            # lr 0.05, not Weka's default 0.3: at 0.3 (and at 0.1)
            # scikit-learn's mini-batch SGD diverges to non-finite weights on
            # standardized inputs. Weka survives 0.3 only because it silently
            # halves the rate and restarts the network on divergence — itself a
            # difference worth not hiding. Each library's own decay/stability
            # mechanism is left enabled.
            "sklearn": {"hidden_layer_sizes": (100,), "max_iter": 300,
                        "solver": "sgd", "learning_rate": "adaptive",
                        "learning_rate_init": 0.05, "momentum": 0.2,
                        "activation": "logistic", "random_state": 42},
            "tuiml": {"hidden_layers": [100], "max_epochs": 300,
                      "learning_rate": 0.05, "momentum": 0.2, "decay": True,
                      "activation": "sigmoid", "random_state": 42},
            "weka": ["-H", "100", "-N", "300", "-L", "0.05", "-M", "0.2",
                     "-D", "-I", "-C", "-S", "42"],
        },
        "note": ("Weka updates weights per instance (online backprop); "
                 "scikit-learn's SGD uses mini-batches. Same architecture and "
                 "learning rate, different optimizer schedule and different "
                 "divergence handling — the least matchable row in the table."),
    },
    # ---------------------------- regressors -----------------------------
    "linear_regression": {
        "task": "regression",
        "defaults": {
            "sklearn": ("sklearn.linear_model", "LinearRegression", {}),
            "tuiml": ("LinearRegression", {}),
            "weka": ("weka.classifiers.functions.LinearRegression", []),
        },
        # Weka's default (-S 0) runs M5 attribute selection and collinearity
        # elimination, which dominates its runtime; scikit-learn solves one
        # least-squares problem. -S 1 disables selection, -C disables
        # collinearity elimination.
        "matched": {
            "sklearn": {},
            "tuiml": {"ridge": 1e-8, "attribute_selection": "none",
                      "eliminate_colinear": False},
            "weka": ["-S", "1", "-C", "-R", "1.0E-8"],
        },
        "note": ("Weka/TuiML keep a 1e-8 ridge term for conditioning; "
                 "scikit-learn solves plain OLS by least squares."),
    },
    "random_forest_reg": {
        "task": "regression",
        "defaults": {
            "sklearn": ("sklearn.ensemble", "RandomForestRegressor",
                        {"n_estimators": 100, "random_state": 42}),
            "tuiml": ("RandomForestRegressor", {"n_estimators": 100}),
            "weka": ("weka.classifiers.trees.RandomForest", ["-I", "100"]),
        },
        # scikit-learn's regression default is max_features=1.0 (every
        # attribute at every split) — far more work than Weka's log2(p)+1.
        # Matched mode puts all three on sqrt(p).
        "matched": {
            "sklearn": {"n_estimators": 100, "max_features": "sqrt",
                        "min_samples_leaf": 1, "bootstrap": True, "random_state": 42},
            "tuiml": {"n_estimators": 100, "max_features": "sqrt",
                      "min_samples_leaf": 1, "bootstrap": True, "random_state": 42},
            "weka": ["-I", "100", "-K", "$SQRT_P", "-depth", "0", "-M", "1",
                     "-V", "1.0E-8", "-P", "100", "-S", "42"],
        },
        "note": ("scikit-learn's RandomForestRegressor defaults to all "
                 "attributes per split; that default alone explains much of the "
                 "untuned runtime gap."),
    },
    "decision_tree_reg": {
        "task": "regression",
        "defaults": {
            "sklearn": ("sklearn.tree", "DecisionTreeRegressor", {"random_state": 42}),
            "tuiml": ("DecisionTreeRegressor", {}),
            "weka": ("weka.classifiers.trees.REPTree", []),
        },
        # REPTree does reduced-error pruning on a held-out third by default;
        # the CART regressors grow unpruned. -P disables pruning.
        "matched": {
            "sklearn": {"min_samples_leaf": 1, "random_state": 42},
            "tuiml": {"min_samples_leaf": 1, "random_state": 42},
            "weka": ["-P", "-M", "1", "-V", "1.0E-8", "-L", "-1", "-S", "42"],
        },
        "note": "All three then reduce variance on binary splits; close match.",
    },
    "knn_reg": {
        "task": "regression",
        "defaults": {
            "sklearn": ("sklearn.neighbors", "KNeighborsRegressor", {"n_neighbors": 5}),
            "tuiml": ("KNearestNeighborsRegressor", {"k": 5}),
            "weka": ("weka.classifiers.lazy.IBk", ["-K", "5"]),
        },
        "matched": {
            "sklearn": {"n_neighbors": 5, "weights": "uniform"},
            "tuiml": {"k": 5, "distance_weighting": "uniform"},
            "weka": ["-K", "5", "-A",
                     'weka.core.neighboursearch.LinearNNSearch -A '
                     '"weka.core.EuclideanDistance -R first-last -D"'],
        },
        "note": "See knn: nominal distance semantics differ between encodings.",
    },
    "svm_reg": {
        "task": "regression",
        "defaults": {
            "sklearn": ("sklearn.svm", "SVR", {}),
            "tuiml": ("SVR", {}),
            "weka": ("weka.classifiers.functions.SMOreg", []),
        },
        # RBF with a shared explicit gamma, C=1, and the epsilon-insensitive
        # loss set to 0.1 on both sides (SMOreg's RegSMOImproved defaults to
        # 1e-3, SVR to 0.1).
        "matched": {
            "sklearn": {"C": 1.0, "epsilon": 0.1, "kernel": "rbf", "gamma": "$GAMMA"},
            "tuiml": {"C": 1.0, "epsilon": 0.1, "kernel": "rbf", "gamma": "$GAMMA"},
            "weka": ["-N", "2", "-C", "1.0",
                     "-I", "weka.classifiers.functions.supportVector.RegSMOImproved -L 0.1",
                     "-K", "weka.classifiers.functions.supportVector.RBFKernel -G $GAMMA"],
        },
        "note": "Optimizer tolerances still differ between SMOreg and libsvm.",
    },
    "mlp_reg": {
        "task": "regression",
        "defaults": {
            "sklearn": ("sklearn.neural_network", "MLPRegressor",
                        {"max_iter": 300, "random_state": 42}),
            "tuiml": ("MultilayerPerceptronRegressor", {}),
            "weka": ("weka.classifiers.functions.MultilayerPerceptron", []),
        },
        # By default Weka normalizes the numeric class and the others do not,
        # which is the documented reason scikit-learn's MLPRegressor scored
        # negative R^2 in the previous run. -C switches Weka's internal version
        # off; the harness then standardizes the target for all three (see
        # run_experiment), so the step is explicit and identical rather than
        # hidden inside one library. Without it, SGD at lr 0.3 diverges on raw
        # targets for every framework.
        "matched": {
            # lr 0.05, not Weka's default 0.3: at 0.3 (and at 0.1)
            # scikit-learn's mini-batch SGD diverges to non-finite weights on
            # standardized inputs. Weka survives 0.3 only because it silently
            # halves the rate and restarts the network on divergence — itself a
            # difference worth not hiding. Each library's own decay/stability
            # mechanism is left enabled.
            "sklearn": {"hidden_layer_sizes": (100,), "max_iter": 300,
                        "solver": "sgd", "learning_rate": "adaptive",
                        "learning_rate_init": 0.05, "momentum": 0.2,
                        "activation": "logistic", "random_state": 42},
            "tuiml": {"hidden_layers": [100], "max_epochs": 300,
                      "learning_rate": 0.05, "momentum": 0.2, "decay": True,
                      "activation": "sigmoid", "random_state": 42},
            "weka": ["-H", "100", "-N", "300", "-L", "0.05", "-M", "0.2",
                     "-D", "-I", "-C", "-S", "42"],
        },
        "note": ("Target is left unscaled for all three, so none of them get "
                 "Weka's default class normalization. See mlp for the optimizer "
                 "difference."),
    },
}

CONFIGS = ("matched", "defaults")


def keys_for_task(task: str):
    """Return algorithm keys that apply to the given task."""
    return [k for k, v in ALGORITHMS.items() if v["task"] == task]


def prep_for(algo_key: str) -> str:
    """Return the data preparation this algorithm needs."""
    return ALGORITHMS[algo_key].get("prep", "standard")


def spec_for(algo_key: str, framework: str):
    """Return the ``defaults`` tuple (class/registry identifiers + kwargs)."""
    return ALGORITHMS[algo_key]["defaults"][framework]


def gamma_scale(X) -> float:
    """scikit-learn's ``gamma="scale"``: ``1 / (n_features * X.var())``.

    Computed once on the one-hot representation and passed explicitly to every
    library, so the RBF kernels really are the same kernel.

    Parameters
    ----------
    X : np.ndarray of shape (n_samples, n_features)
        Training matrix.

    Returns
    -------
    gamma : float
        Kernel width; falls back to ``1 / n_features`` on a zero-variance matrix.
    """
    var = float(X.var()) if X.size else 0.0
    n_features = max(1, X.shape[1])
    if var <= 0.0:
        return 1.0 / n_features
    return 1.0 / (n_features * var)


def needs_gamma(algo_key: str, config: str) -> bool:
    """Whether this (algorithm, config) uses the ``$GAMMA`` placeholder.

    Callers use this to decide whether to materialize the one-hot matrix purely
    to compute the shared kernel width — worth avoiding on wide datasets when no
    kernel is involved.

    Parameters
    ----------
    algo_key : str
        Key into :data:`ALGORITHMS`.
    config : str
        ``"matched"`` or ``"defaults"``.

    Returns
    -------
    bool
        True if any framework's parameters reference ``$GAMMA``.
    """
    spec = ALGORITHMS[algo_key]
    for fw in ("sklearn", "tuiml", "weka"):
        if config == "matched" and fw in spec.get("matched", {}):
            params = spec["matched"][fw]
        else:
            params = spec["defaults"][fw][-1]
        values = params.values() if isinstance(params, dict) else params
        if any(isinstance(v, str) and "$GAMMA" in v for v in values):
            return True
    return False


def _fmt(value) -> str:
    """Format a substituted number for a Weka option string."""
    if isinstance(value, float):
        return f"{value:.10g}"
    return str(value)


def _substitute(value, ctx):
    """Replace ``$GAMMA`` / ``$SQRT_P`` placeholders in one option value."""
    if not isinstance(value, str):
        return value
    if value == "$GAMMA":
        return ctx["gamma"]
    if value == "$SQRT_P":
        return ctx["sqrt_p"]
    for key in ("$GAMMA", "$SQRT_P"):
        if key in value:
            value = value.replace(key, _fmt(ctx[key.lower().lstrip("$")]))
    return value


def resolve(algo_key: str, framework: str, config: str, ctx: dict):
    """Return the parameters to use for one (algorithm, framework, config).

    Parameters
    ----------
    algo_key : str
        Key into :data:`ALGORITHMS`.
    framework : str
        ``"sklearn"``, ``"tuiml"``, or ``"weka"``.
    config : str
        ``"matched"`` or ``"defaults"``.
    ctx : dict
        Data-dependent values: ``gamma`` (float) and ``sqrt_p`` (int).

    Returns
    -------
    params : dict or list
        Constructor kwargs (scikit-learn / TuiML) or a Weka option list, with
        placeholders substituted. Weka option lists are all strings.
    """
    if config not in CONFIGS:
        raise ValueError(f"config must be one of {CONFIGS}, got {config!r}")
    spec = ALGORITHMS[algo_key]
    if config == "matched" and framework in spec.get("matched", {}):
        params = spec["matched"][framework]
    else:
        params = spec["defaults"][framework][-1]

    if isinstance(params, dict):
        return {k: _substitute(v, ctx) for k, v in params.items()}
    return [_fmt(_substitute(v, ctx)) for v in params]


def context(X_train, gamma_source=None) -> dict:
    """Build the substitution context for one framework's representation.

    Parameters
    ----------
    X_train : np.ndarray
        The matrix this framework will actually be given; its column count
        drives ``$SQRT_P``.
    gamma_source : np.ndarray or None, default=None
        Matrix to compute ``$GAMMA`` from. Pass the one-hot representation so
        every library gets the same kernel width even when Weka is handed
        un-expanded nominal attributes. Defaults to ``X_train``.

    Returns
    -------
    ctx : dict
        ``{"gamma": float, "sqrt_p": int}``.
    """
    source = X_train if gamma_source is None else gamma_source
    return {
        "gamma": gamma_scale(source),
        "sqrt_p": max(1, int(math.sqrt(max(1, X_train.shape[1])))),
    }
