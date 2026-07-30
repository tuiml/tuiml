#!/usr/bin/env python3
"""Generate the scikit-learn wrapper modules from the declarative spec table.

The wrappers under ``tuiml/sklearn/`` are generated code. The source of truth is
``tuiml/sklearn/specs.py``; this script renders one module per TuiML algorithm
family from those rows, emitting real ``class`` statements so that the AST-based
documentation generator and pickling of saved models both keep working.

Usage
-----
    uv run scripts/generate_sklearn_wrappers.py            # regenerate modules
    uv run scripts/generate_sklearn_wrappers.py --bootstrap  # also rewrite specs.py
    uv run scripts/generate_sklearn_wrappers.py --check     # fail if stale

``--bootstrap`` re-derives ``specs.py`` by enumerating the installed
scikit-learn. Use it when upgrading scikit-learn to pick up new estimators; the
resulting diff is the review surface for what changed upstream.
"""

import argparse
import inspect
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_ROOT = REPO_ROOT / "tuiml" / "sklearn"
sys.path.insert(0, str(REPO_ROOT))

from tuiml.sklearn._spec import KIND_IMPORTS  # noqa: E402

# ---------------------------------------------------------------------------
# Family layout: sklearn module -> generated module path, mirroring the native
# TuiML taxonomy in tuiml/algorithms, tuiml/preprocessing and tuiml/features.
# ---------------------------------------------------------------------------

MODULE_TO_FAMILY = {
    "linear_model": "linear",
    "discriminant_analysis": "linear",
    "cross_decomposition": "linear",
    "semi_supervised": "linear",
    "kernel_ridge": "linear",
    "isotonic": "linear",
    "dummy": "linear",
    "svm": "svm",
    "tree": "trees",
    "ensemble": "ensemble",
    "multiclass": "ensemble",
    "naive_bayes": "bayesian",
    "gaussian_process": "bayesian",
    "neighbors": "neighbors",
    "neural_network": "neural",
    "cluster": "clustering",
    "mixture": "clustering",
    "covariance": "anomaly",
    "preprocessing": "preprocessing/scaling",
    "impute": "preprocessing/imputation",
    "feature_extraction": "preprocessing/text",
    "feature_selection": "features/selection",
    "decomposition": "features/extraction",
    "manifold": "features/extraction",
    "random_projection": "features/extraction",
    "kernel_approximation": "features/extraction",
}

#: Estimator names that belong to the anomaly family regardless of their module.
ANOMALY_ESTIMATORS = {
    "IsolationForest",
    "LocalOutlierFactor",
    "EllipticEnvelope",
    "OneClassSVM",
    "SGDOneClassSVM",
}

#: Wrapper names that must not change: they are already published hub keys.
NAME_OVERRIDES = {"PCA": "PCAExtractor"}

#: Tier 1 = curated and hand-reviewed; surfaced by default in search and MCP.
TIER1 = {
    # the originally curated 26
    "RandomForestClassifier", "HistGradientBoostingClassifier", "SVC",
    "LogisticRegression", "KNeighborsClassifier", "MLPClassifier", "GaussianNB",
    "RandomForestRegressor", "HistGradientBoostingRegressor", "SVR", "Ridge",
    "Lasso", "KMeans", "DBSCAN", "GaussianMixture", "StandardScaler",
    "MinMaxScaler", "RobustScaler", "SimpleImputer", "IterativeImputer",
    "KBinsDiscretizer", "PolynomialFeatures", "SelectKBest", "VarianceThreshold",
    "PCAExtractor", "TruncatedSVD",
    # rounding out the commonly used surface
    "ExtraTreesClassifier", "ExtraTreesRegressor", "GradientBoostingClassifier",
    "GradientBoostingRegressor", "AdaBoostClassifier", "AdaBoostRegressor",
    "BaggingClassifier", "BaggingRegressor", "DecisionTreeClassifier",
    "DecisionTreeRegressor", "LinearRegression", "ElasticNet", "SGDClassifier",
    "SGDRegressor", "LinearSVC", "LinearSVR", "MLPRegressor",
    "KNeighborsRegressor", "MultinomialNB", "BernoulliNB",
    "AgglomerativeClustering", "SpectralClustering", "Birch", "MeanShift",
    "IsolationForest", "LocalOutlierFactor", "OneClassSVM", "EllipticEnvelope",
    "OneHotEncoder", "OrdinalEncoder", "MaxAbsScaler",
    "Normalizer", "QuantileTransformer", "PowerTransformer", "KNNImputer",
    "SelectPercentile", "SelectFromModel", "RFE", "FastICA", "NMF", "KernelPCA",
    "TSNE", "CountVectorizer", "TfidfVectorizer",
}

#: Estimators that do not fit TuiML's ``fit(X, y)`` contract, where ``X`` is a
#: 2-D feature matrix and ``y`` a 1-D target. Excluded with the reason recorded
#: in ``specs.NOT_WRAPPED`` so the gap stays visible.
INCOMPATIBLE = {
    "IsotonicRegression": "requires 1-D X",
    "MultiTaskElasticNet": "requires 2-D (multi-output) y",
    "MultiTaskElasticNetCV": "requires 2-D (multi-output) y",
    "MultiTaskLasso": "requires 2-D (multi-output) y",
    "MultiTaskLassoCV": "requires 2-D (multi-output) y",
    "CCA": "cross-decomposition, requires multi-column y",
    "PLSCanonical": "cross-decomposition, requires multi-column y",
    "PLSSVD": "cross-decomposition, requires multi-column y",
    "KernelCenterer": "operates on a kernel matrix, not a feature matrix",
    "LabelBinarizer": "transforms labels, signature is fit(y) not fit(X, y)",
    "LabelEncoder": "transforms labels, signature is fit(y) not fit(X, y)",
    "MultiLabelBinarizer": "transforms labels, signature is fit(y) not fit(X, y)",
    "SelfTrainingClassifier": "meta-estimator, requires an inner estimator",
    "FeatureAgglomeration": "n_clusters is bounded by n_features, not n_samples",
}

#: Estimators whose spec needs a hand-written builder.
BUILDERS = {
    "SVC": "svc_with_optional_calibration",
    "IterativeImputer": "iterative_imputer",
}

#: Extra per-estimator spec detail: opinionated defaults and schema exclusions.
DEFAULTS = {
    "SVC": {"probability": True},
    # predict() only exists when novelty=True; TuiML classifiers need it.
    "LocalOutlierFactor": {"novelty": True},
}
EXCLUDES = {"SVC": ("kernel_precomputed",)}

#: Highlighted parameters, surfaced first in the derived schema.
HIGHLIGHT = {
    "RandomForestClassifier": ("n_estimators", "max_depth", "random_state"),
    "RandomForestRegressor": ("n_estimators", "max_depth", "random_state"),
    "ExtraTreesClassifier": ("n_estimators", "max_depth", "random_state"),
    "ExtraTreesRegressor": ("n_estimators", "max_depth", "random_state"),
    "HistGradientBoostingClassifier": ("learning_rate", "max_iter", "max_depth"),
    "HistGradientBoostingRegressor": ("learning_rate", "max_iter", "max_depth"),
    "GradientBoostingClassifier": ("n_estimators", "learning_rate", "max_depth"),
    "GradientBoostingRegressor": ("n_estimators", "learning_rate", "max_depth"),
    "SVC": ("C", "kernel", "gamma", "probability"),
    "SVR": ("C", "kernel", "epsilon"),
    "LinearSVC": ("C", "loss", "max_iter"),
    "LogisticRegression": ("C", "penalty", "solver", "max_iter"),
    "Ridge": ("alpha", "solver"),
    "Lasso": ("alpha", "max_iter"),
    "ElasticNet": ("alpha", "l1_ratio"),
    "KMeans": ("n_clusters", "init", "n_init", "random_state"),
    "DBSCAN": ("eps", "min_samples", "metric"),
    "AgglomerativeClustering": ("n_clusters", "linkage", "metric"),
    "GaussianMixture": ("n_components", "covariance_type", "random_state"),
    "KNeighborsClassifier": ("n_neighbors", "weights", "metric"),
    "KNeighborsRegressor": ("n_neighbors", "weights", "metric"),
    "MLPClassifier": ("hidden_layer_sizes", "activation", "max_iter"),
    "MLPRegressor": ("hidden_layer_sizes", "activation", "max_iter"),
    "DecisionTreeClassifier": ("criterion", "max_depth", "random_state"),
    "DecisionTreeRegressor": ("criterion", "max_depth", "random_state"),
    "IsolationForest": ("n_estimators", "contamination", "random_state"),
    "LocalOutlierFactor": ("n_neighbors", "contamination"),
    "OneClassSVM": ("kernel", "nu", "gamma"),
    "SimpleImputer": ("strategy", "fill_value"),
    "IterativeImputer": ("max_iter", "random_state"),
    "KNNImputer": ("n_neighbors", "weights"),
    "StandardScaler": ("with_mean", "with_std"),
    "MinMaxScaler": ("feature_range",),
    "RobustScaler": ("with_centering", "with_scaling", "quantile_range"),
    "KBinsDiscretizer": ("n_bins", "encode", "strategy"),
    "PolynomialFeatures": ("degree", "interaction_only", "include_bias"),
    "OneHotEncoder": ("handle_unknown", "drop", "sparse_output"),
    "SelectKBest": ("k", "score_func"),
    "SelectPercentile": ("percentile", "score_func"),
    "VarianceThreshold": ("threshold",),
    "PCAExtractor": ("n_components", "whiten", "random_state"),
    "TruncatedSVD": ("n_components", "algorithm", "random_state"),
    "KernelPCA": ("n_components", "kernel", "gamma"),
    "FastICA": ("n_components", "random_state"),
    "NMF": ("n_components", "init", "random_state"),
    "TSNE": ("n_components", "perplexity", "random_state"),
    "CountVectorizer": ("max_features", "ngram_range", "stop_words"),
    "TfidfVectorizer": ("max_features", "ngram_range", "stop_words"),
}


def classify(cls) -> str:
    """Return the TuiML component kind for a scikit-learn estimator class.

    Parameters
    ----------
    cls : type
        A scikit-learn estimator class.

    Returns
    -------
    kind : str
        A key of :data:`~tuiml.sklearn._spec.KIND_IMPORTS`, or ``""`` when the
        estimator does not map onto a TuiML component kind.
    """
    mro = {c.__name__ for c in cls.__mro__}
    module = cls.__module__.split(".")[1]
    if "ClassifierMixin" in mro:
        return "classifier"
    if "RegressorMixin" in mro:
        return "regressor"
    if "ClusterMixin" in mro:
        return "clusterer"
    # Outlier detectors follow the native TuiML convention of using Classifier.
    if "OutlierMixin" in mro:
        return "classifier"
    # A usable transformer needs ``transform``, not merely ``fit_transform``:
    # the wrapper mixins call ``transform`` on held-out data. This excludes the
    # transductive manifold learners (MDS, SpectralEmbedding) by design, and
    # includes the text vectorizers, which predate TransformerMixin.
    if "TransformerMixin" in mro or hasattr(cls, "transform"):
        if not hasattr(cls, "transform"):
            return ""
        if module == "feature_selection":
            return "feature_selector"
        if module in ("decomposition", "manifold", "random_projection",
                      "kernel_approximation"):
            return "feature_extractor"
        return "transformer"
    # Density/mixture models expose fit_predict and behave as clusterers.
    if hasattr(cls, "fit_predict"):
        return "clusterer"
    return ""


def derive_tags(name: str, cls, family: str) -> tuple:
    """Build the hub search tags for one estimator.

    Parameters
    ----------
    name : str
        Wrapper class name.
    cls : type
        The backing scikit-learn class.
    family : str
        Generated module path, e.g. ``"linear"`` or ``"features/extraction"``.

    Returns
    -------
    tags : tuple of str
        Search tags, most specific first.
    """
    tags = [family.split("/")[-1]]
    module = cls.__module__.split(".")[1]
    if module not in tags:
        tags.append(module.replace("_", "-"))
    if name in ANOMALY_ESTIMATORS:
        tags.append("anomaly-detection")
    for keyword, tag in (
        ("Forest", "ensemble"), ("Boosting", "gradient-boosting"),
        ("SV", "svm"), ("Tree", "tree"), ("NB", "naive-bayes"),
        ("Neighbors", "instance-based"), ("MLP", "neural"),
        ("Scaler", "scaling"), ("Imputer", "imputation"),
        ("Vectorizer", "text"), ("Encoder", "encoding"),
    ):
        if keyword in name and tag not in tags:
            tags.append(tag)
    return tuple(tags)


def derive_capabilities(cls, kind: str) -> tuple:
    """Determine the capability strings for a wrapper.

    Parameters
    ----------
    cls : type
        The backing scikit-learn class.
    kind : str
        The TuiML component kind.

    Returns
    -------
    capabilities : tuple of str
        Capability names.
    """
    caps = ["numeric"]
    if kind == "classifier":
        caps.append("multiclass")
        if hasattr(cls, "predict_proba"):
            caps.append("probabilities")
    return tuple(caps)


def collect_specs():
    """Enumerate installed scikit-learn estimators into spec tuples.

    Returns
    -------
    families : dict
        Mapping of generated module path to a list of spec keyword dicts.
    skipped : list of tuple
        ``(name, reason)`` for every estimator deliberately not wrapped.
    """
    # IterativeImputer is gated behind an experimental flag and is invisible to
    # all_estimators() until the enabler is imported.
    from sklearn.experimental import enable_iterative_imputer  # noqa: F401
    from sklearn.utils import all_estimators

    families = defaultdict(list)
    skipped = []

    for sk_name, cls in all_estimators():
        signature = inspect.signature(cls.__init__)
        required = [
            n for n, p in signature.parameters.items()
            if n != "self"
            and p.default is inspect.Parameter.empty
            and p.kind not in (p.VAR_KEYWORD, p.VAR_POSITIONAL)
        ]
        if required:
            skipped.append((sk_name, f"needs required arg(s) {required}"))
            continue

        if sk_name in INCOMPATIBLE:
            skipped.append((sk_name, INCOMPATIBLE[sk_name]))
            continue

        module = cls.__module__.split(".")[1]
        family = "anomaly" if sk_name in ANOMALY_ESTIMATORS else MODULE_TO_FAMILY.get(module)
        if family is None:
            skipped.append((sk_name, f"module '{module}' not mapped to a family"))
            continue

        kind = classify(cls)
        if not kind:
            skipped.append((sk_name, "no TuiML component kind"))
            continue

        name = NAME_OVERRIDES.get(sk_name, sk_name)
        families[family].append({
            "name": name,
            "target": f"{cls.__module__}:{sk_name}",
            "kind": kind,
            "tags": derive_tags(name, cls, family),
            "capabilities": derive_capabilities(cls, kind),
            "defaults": DEFAULTS.get(name, {}),
            "highlight": HIGHLIGHT.get(name, ()),
            "exclude": EXCLUDES.get(name, ()),
            "tier": 1 if name in TIER1 else 2,
            "builder": BUILDERS.get(name),
        })

    for rows in families.values():
        rows.sort(key=lambda r: (r["kind"], r["name"]))
    return families, skipped


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

BANNER = "# Generated by scripts/generate_sklearn_wrappers.py -- do not edit by hand.\n"


def render_specs(families, skipped) -> str:
    """Render ``specs.py``, the declarative source of truth.

    Parameters
    ----------
    families : dict
        Mapping of module path to spec keyword dicts.
    skipped : list of tuple
        Estimators not wrapped, recorded as a comment block.

    Returns
    -------
    source : str
        Complete Python source for ``specs.py``.
    """
    total = sum(len(v) for v in families.values())
    out = [
        '"""Declarative table of every wrapped scikit-learn estimator.\n',
        "This module is the source of truth for the generated wrapper modules in",
        "this package. To add an estimator, add a row here and re-run",
        "``uv run scripts/generate_sklearn_wrappers.py``.\n",
        f"Currently {total} estimators across {len(families)} families. Rows with",
        "``tier=1`` are curated and surfaced by default; ``tier=2`` rows are also",
        'tagged ``sklearn-extended`` so search and listings can filter them out.',
        '"""',
        "",
        BANNER.rstrip(),
        "",
        "from tuiml.sklearn._spec import SklearnSpec as S",
        "",
    ]

    for family in sorted(families):
        rows = families[family]
        const = family.replace("/", "_").upper()
        out.append(f"#: {family}.py -- {len(rows)} estimators")
        out.append(f"{const} = [")
        for row in rows:
            out.append("    S(")
            out.append(f"        name={row['name']!r},")
            out.append(f"        target={row['target']!r},")
            out.append(f"        kind={row['kind']!r},")
            out.append(f"        tags={row['tags']!r},")
            out.append(f"        capabilities={row['capabilities']!r},")
            if row["defaults"]:
                out.append(f"        defaults={row['defaults']!r},")
            if row["highlight"]:
                out.append(f"        highlight={row['highlight']!r},")
            if row["exclude"]:
                out.append(f"        exclude={row['exclude']!r},")
            out.append(f"        tier={row['tier']},")
            if row["builder"]:
                out.append(f"        builder={row['builder']!r},")
            out.append("    ),")
        out.append("]")
        out.append("")

    out.append("#: Every family table, keyed by generated module path.")
    out.append("FAMILIES = {")
    for family in sorted(families):
        out.append(f"    {family!r}: {family.replace('/', '_').upper()},")
    out.append("}")
    out.append("")
    out.append("#: Estimators deliberately not wrapped, with the reason. Recorded so the")
    out.append("#: gap is visible rather than silent; see the module docstring.")
    out.append("NOT_WRAPPED = {")
    for name, reason in sorted(skipped):
        out.append(f"    {name!r}: {reason!r},")
    out.append("}")
    out.append("")
    return "\n".join(out)


def render_module(family: str, rows) -> str:
    """Render one generated wrapper module.

    Parameters
    ----------
    family : str
        Module path, e.g. ``"linear"`` or ``"features/extraction"``.
    rows : list of dict
        Spec keyword dicts for this module.

    Returns
    -------
    source : str
        Complete Python source for the module.
    """
    kinds = sorted({r["kind"] for r in rows})
    decorators = sorted({KIND_IMPORTS[k][0] for k in kinds})
    mixins = sorted({KIND_IMPORTS[k][1] for k in kinds})
    bases = sorted({KIND_IMPORTS[k][2] for k in kinds})

    algo_bases = [b for b in bases if b in ("Classifier", "Regressor", "Clusterer")]
    prep_bases = [b for b in bases if b == "Transformer"]
    feat_bases = [b for b in bases if b in ("FeatureSelector", "FeatureExtractor")]

    label = family.split("/")[-1].replace("_", " ")
    depth = family.count("/")
    spec_const = family.replace("/", "_").upper()

    out = [
        f'"""scikit-learn {label} wrappers.',
        "",
        f"Generated from the ``{spec_const}`` table in :mod:`tuiml.sklearn.specs`.",
        "Registered under ``sklearn.<ClassName>`` hub keys, mirroring the native",
        f"TuiML ``{family}`` family.",
        '"""',
        "",
        BANNER.rstrip(),
        "",
        "from typing import Any, Dict, List",
        "",
    ]
    if algo_bases:
        out.append(f"from tuiml.base.algorithms import {', '.join(algo_bases)}")
    if prep_bases:
        out.append(f"from tuiml.base.preprocessing import {', '.join(prep_bases)}")
    if feat_bases:
        out.append(f"from tuiml.base.features import {', '.join(feat_bases)}")
    out.append("from tuiml.sklearn._base import (")
    for item in mixins + decorators:
        out.append(f"    {item},")
    out.append(")")
    out.append("from tuiml.sklearn._spec import build_estimator, derive_schema")
    if any(r["builder"] for r in rows):
        out.append("from tuiml.sklearn import _overrides")
    out.append("")
    # Relative depth back up to the package root for the spec import.
    out.append(f"from tuiml.sklearn.specs import {spec_const} as _SPECS")
    out.append("")
    out.append("_BY_NAME = {s.name: s for s in _SPECS}")
    out.append("")
    out.append("")

    for row in rows:
        decorator, mixin, base = KIND_IMPORTS[row["kind"]]
        name = row["name"]
        sk_class = row["target"].split(":")[1]
        sk_module = row["target"].split(":")[0]
        tags = list(row["tags"])
        if row["tier"] == 2:
            tags.append("sklearn-extended")

        out.append(f"@{decorator}(tags={tags!r})")
        out.append(f"class {name}({mixin}, {base}):")
        out.append(f'    """scikit-learn {sk_class} (hub key ``sklearn.{name}``).')
        out.append("")
        out.append(f"    Wraps :class:`{sk_module}.{sk_class}`. Accepts that")
        out.append("    estimator's constructor parameters as keyword arguments; call")
        out.append("    :meth:`get_parameter_schema` for the full list with types and")
        out.append("    defaults derived from the installed scikit-learn.")
        if row["highlight"]:
            out.append("")
            out.append(f"    Commonly set: {', '.join(row['highlight'])}.")
        out.append('    """')
        out.append("")
        out.append(f"    _SPEC = _BY_NAME[{name!r}]")
        out.append("")
        out.append("    def __init__(self, **params: Any):")
        out.append("        super().__init__()")
        out.append("        self._params = {**self._SPEC.defaults, **params}")
        out.append("        for key, value in self._params.items():")
        out.append("            setattr(self, key, value)")
        out.append("")
        out.append("    def _build_estimator(self):")
        out.append('        """Construct the backing scikit-learn estimator."""')
        if row["builder"]:
            out.append(
                f"        return _overrides.{row['builder']}("
            )
            out.append(
                f"            self._SPEC.target, self._params, {name!r}"
            )
            out.append("        )")
        else:
            out.append(
                f"        return build_estimator(self._SPEC.target, self._params, {name!r})"
            )
        out.append("")
        out.append("    @classmethod")
        out.append("    def get_parameter_schema(cls) -> Dict[str, Any]:")
        out.append('        """Return JSON Schema for constructor parameters."""')
        out.append("        return derive_schema(")
        out.append("            cls._SPEC.target, cls._SPEC.highlight, cls._SPEC.exclude")
        out.append("        )")
        out.append("")
        out.append("    @classmethod")
        out.append("    def get_capabilities(cls) -> List[str]:")
        out.append('        """Return the capability names this wrapper supports."""')
        out.append("        return list(cls._SPEC.capabilities)")
        out.append("")
        out.append("")

    out.append(f"__all__ = {[r['name'] for r in rows]!r}")
    out.append("")
    return "\n".join(out)


def render_subpackage_init(family_dir: str, modules) -> str:
    """Render the ``__init__.py`` for a generated subpackage.

    Parameters
    ----------
    family_dir : str
        Subpackage name, ``"preprocessing"`` or ``"features"``.
    modules : list of str
        Module names inside the subpackage.

    Returns
    -------
    source : str
        Complete Python source.
    """
    out = [
        f'"""scikit-learn {family_dir} wrappers, mirroring TuiML\'s native layout."""',
        "",
        BANNER.rstrip(),
        "",
    ]
    for module in sorted(modules):
        out.append(f"from tuiml.sklearn.{family_dir} import {module}  # noqa: F401")
    for module in sorted(modules):
        out.append(f"from tuiml.sklearn.{family_dir}.{module} import *  # noqa: F401,F403")
    out.append("")
    out.append("__all__ = [")
    for module in sorted(modules):
        out.append(f"    *{module}.__all__,")
    out.append("]")
    out.append("")
    return "\n".join(out)


def main() -> int:
    """Generate the wrapper modules. Returns a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap", action="store_true",
                        help="re-derive specs.py from the installed scikit-learn")
    parser.add_argument("--check", action="store_true",
                        help="exit non-zero if generated output is stale")
    args = parser.parse_args()

    if args.bootstrap:
        # Render from the in-memory table: specs.py cannot be imported yet on a
        # first run, since importing it would pull in wrapper modules that this
        # invocation has not written.
        by_family, skipped = collect_specs()
        specs_source = render_specs(by_family, skipped)
        target = PACKAGE_ROOT / "specs.py"
        if args.check and target.exists() and target.read_text() != specs_source:
            print(f"STALE: {target}")
            return 1
        target.write_text(specs_source)
        print(f"wrote {target.relative_to(REPO_ROOT)} "
              f"({sum(len(v) for v in by_family.values())} specs, "
              f"{len(skipped)} not wrapped)")
    else:
        # Steady state: specs.py on disk is the source of truth.
        for module in [m for m in list(sys.modules) if m.startswith("tuiml.sklearn")]:
            del sys.modules[module]
        from tuiml.sklearn.specs import FAMILIES

        by_family = {
            family: [
                {
                    "name": s.name, "target": s.target, "kind": s.kind,
                    "tags": s.tags, "highlight": s.highlight, "tier": s.tier,
                    "builder": s.builder, "defaults": s.defaults,
                    "capabilities": s.capabilities, "exclude": s.exclude,
                }
                for s in specs
            ]
            for family, specs in FAMILIES.items()
        }

    stale = []
    subpackages = defaultdict(list)
    for family, rows in sorted(by_family.items()):
        source = render_module(family, rows)
        path = PACKAGE_ROOT / f"{family}.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        if "/" in family:
            subpackages[family.split("/")[0]].append(family.split("/")[1])
        if args.check:
            if not path.exists() or path.read_text() != source:
                stale.append(path)
        else:
            path.write_text(source)
            print(f"  {family + '.py':38} {len(rows):>3} wrappers")

    for subpackage, modules in subpackages.items():
        source = render_subpackage_init(subpackage, modules)
        path = PACKAGE_ROOT / subpackage / "__init__.py"
        if args.check:
            if not path.exists() or path.read_text() != source:
                stale.append(path)
        else:
            path.write_text(source)

    if args.check:
        for path in stale:
            print(f"STALE: {path}")
        return 1 if stale else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
