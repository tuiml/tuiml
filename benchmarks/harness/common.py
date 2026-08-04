"""Shared benchmark harness: identical data prep, metrics, and resource capture.

All three framework runners (sklearn / tuiml / weka) import this so they are
compared on *exactly* the same train/test split and the same information — we
benchmark the algorithms, not each library's data handling. Each experiment runs
in its own OS process (launched by run_all.sh), so peak RSS via ``resource`` is
per-experiment.

Attribute types
---------------
Column types come from ``schema.json`` (written by ``fetch_schema.py`` from the
OpenML attribute declarations), **not** from pandas dtypes. Inferring from dtype
silently treats integer-coded nominal attributes as continuous, which misleads
every framework — and denies Weka the nominal handling that is native to its
tree and instance-based learners. When no schema is present the loader falls
back to dtype inference and records ``schema_source="dtype-fallback"`` in the
result so those rows can be excluded or flagged.

Representations
---------------
Preparation produces a *representation-neutral* split (a standardized numeric
block plus an integer-coded nominal block), and each runner materializes the
form its library actually consumes:

* :meth:`Prepared.onehot`      — numeric block + one-hot nominal (scikit-learn, TuiML)
* :meth:`Prepared.mixed`       — numeric block + nominal codes, with the nominal
  column indices so Weka can mark them nominal in the ARFF header
* :meth:`Prepared.discretized` — everything as discrete codes (the categorical
  Naive Bayes rows)

The information content is the same in all three; only the encoding differs, and
each library gets the encoding it is designed for. Materialization happens
outside the timed region.

All encoders (imputers, scalers, category vocabularies, bin edges) are fit on
the **training split only**.
"""

from __future__ import annotations

import json
import os
import resource
import time
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
TEST_SIZE = 0.2
N_FOLDS = 10
N_BINS = 10

# High-cardinality nominal attributes are folded to the MAX_LEVELS most frequent
# training categories plus a single "other" level. Without a cap, one-hot
# encoding a 7,500-level attribute over 32k rows materializes a ~2 GB dense
# matrix and the run OOMs under parallelism. The fold is applied to the *codes*,
# before any representation is built, so all three frameworks see exactly the
# same information — the cap is a property of the benchmark, not of one library.
MAX_LEVELS = int(os.environ.get("BENCH_MAX_LEVELS", "100"))


def _split_indices(y, is_clf, seed, fold, n):
    """Train/test row indices: fold k of stratified 10-fold CV, or None for holdout.

    Parameters
    ----------
    y : array-like of shape (n_samples,)
        Target values (used for stratification).
    is_clf : bool
        Whether the task is classification (stratified folds).
    seed : int
        Shuffle seed (same seed => identical folds across frameworks).
    fold : int
        Fold index in [0, N_FOLDS).
    n : int
        Number of samples.

    Returns
    -------
    (train_idx, test_idx) : tuple of np.ndarray
        Row indices for the requested fold.
    """
    from sklearn.model_selection import StratifiedKFold, KFold
    if not 0 <= fold < N_FOLDS:
        raise ValueError(f"fold must be in [0, {N_FOLDS}), got {fold}")
    cls = StratifiedKFold if is_clf else KFold
    kf = cls(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    splits = list(kf.split(np.zeros(n), y if is_clf else None))
    return splits[fold]


def _read_schema(dataset_csv: str, columns) -> tuple[set, str]:
    """Return the set of nominal column names and where that knowledge came from.

    Parameters
    ----------
    dataset_csv : str
        Path to the dataset CSV; ``schema.json`` is looked up alongside it.
    columns : list of str
        Feature column names (target excluded).

    Returns
    -------
    (nominal, source) : tuple of (set of str, str)
        ``source`` is ``"openml-schema"`` or ``"dtype-fallback"``.
    """
    path = Path(dataset_csv).with_name("schema.json")
    if not path.exists():
        return set(), "dtype-fallback"
    try:
        schema = json.loads(path.read_text())
        cols = schema.get("columns", {})
    except (ValueError, OSError):
        return set(), "dtype-fallback"
    known = [c for c in columns if c in cols]
    if not known:
        return set(), "dtype-fallback"
    nominal = {c for c in known if cols[c].get("type") == "nominal"}
    # Columns absent from the schema (rare: renamed on export) fall back to
    # dtype inference in the caller; flag the mix so it is visible in results.
    source = "openml-schema" if len(known) == len(columns) else "openml-schema-partial"
    return nominal, source


class Prepared:
    """One prepared train/test split, in a framework-neutral form.

    Attributes
    ----------
    num_tr, num_te : np.ndarray
        Standardized numeric block (may have zero columns).
    nom_tr, nom_te : np.ndarray of int
        Nominal block as integer codes in ``[0, n_categories[j])``.
    y_tr, y_te : np.ndarray
        Targets (label-encoded for classification).
    n_categories : list of int
        Number of levels per nominal column, including the "unseen" level.
    meta : dict
        Shapes, class counts, and provenance recorded into the result JSON.
    """

    def __init__(self, num_tr, num_te, nom_tr, nom_te, y_tr, y_te,
                 n_categories, meta):
        self.num_tr, self.num_te = num_tr, num_te
        self.nom_tr, self.nom_te = nom_tr, nom_te
        self.y_tr, self.y_te = y_tr, y_te
        self.n_categories = n_categories
        self.meta = meta

    def onehot(self):
        """Materialize the dense float representation (scikit-learn, TuiML).

        Returns
        -------
        (X_train, X_test) : tuple of np.ndarray
            Numeric block followed by one-hot blocks, dense float64.
        """
        def build(num, nom):
            blocks = [num] if num.shape[1] else []
            for j, k in enumerate(self.n_categories):
                col = nom[:, j]
                oh = np.zeros((col.shape[0], k), dtype=np.float64)
                oh[np.arange(col.shape[0]), col] = 1.0
                blocks.append(oh)
            if not blocks:
                return np.zeros((num.shape[0], 0), dtype=np.float64)
            return np.hstack(blocks).astype(np.float64, copy=False)
        return build(self.num_tr, self.nom_tr), build(self.num_te, self.nom_te)

    def mixed(self):
        """Materialize the mixed representation for Weka.

        Numeric columns first, then nominal columns as integer codes. The
        returned indices tell the Weka runner which attributes to declare
        nominal, so Weka's tree and instance-based learners get the multi-way
        nominal handling they are designed around (and its function-based
        learners apply their own NominalToBinary internally, which reproduces
        the one-hot the other frameworks receive).

        Returns
        -------
        (X_train, X_test, nominal_idx) : tuple
            Arrays are float64 (Weka's matrix constructor takes doubles);
            ``nominal_idx`` is the 0-based column indices of the nominal block.
        """
        n_num = self.num_tr.shape[1]
        nominal_idx = list(range(n_num, n_num + self.nom_tr.shape[1]))

        def build(num, nom):
            blocks = [b for b in (num, nom.astype(np.float64)) if b.shape[1]]
            if not blocks:
                return np.zeros((num.shape[0], 0), dtype=np.float64)
            return np.hstack(blocks).astype(np.float64, copy=False)
        return build(self.num_tr, self.nom_tr), build(self.num_te, self.nom_te), nominal_idx

    def discretized(self):
        """Materialize an all-discrete representation (categorical Naive Bayes).

        Numeric columns are quantile-binned into ``N_BINS`` ordinal codes with
        **bin edges fit on the training split only**; nominal columns keep their
        codes. This is the representation a categorical NB expects, and matches
        what Weka's NaiveBayes does with an all-nominal ARFF.

        Returns
        -------
        (X_train, X_test, n_categories) : tuple
            Integer code matrices and the per-column level counts.
        """
        parts_tr, parts_te, n_cats = [], [], []
        if self.num_tr.shape[1]:
            from sklearn.preprocessing import KBinsDiscretizer
            disc = KBinsDiscretizer(n_bins=N_BINS, encode="ordinal",
                                    strategy="quantile")
            # subsample=None keeps binning deterministic across sklearn versions.
            try:
                disc.set_params(subsample=None)
            except ValueError:
                pass
            parts_tr.append(disc.fit_transform(self.num_tr).astype(int))
            # Test values outside the training range clip to the edge bins.
            parts_te.append(disc.transform(self.num_te).astype(int))
            n_cats.extend(int(b) for b in disc.n_bins_)
        if self.nom_tr.shape[1]:
            parts_tr.append(self.nom_tr)
            parts_te.append(self.nom_te)
            n_cats.extend(self.n_categories)
        if not parts_tr:
            zero = np.zeros((self.num_tr.shape[0], 0), dtype=int)
            return zero, np.zeros((self.num_te.shape[0], 0), dtype=int), []
        return np.hstack(parts_tr), np.hstack(parts_te), n_cats


def load_and_prepare(dataset_csv: str, task: str, seed: int = SEED,
                     fold=None) -> Prepared:
    """Load a CSV (target = last column), split, and preprocess identically.

    Column types come from ``schema.json`` when available (see module docstring).
    Numeric features: median impute + standardize. Nominal features:
    most-frequent impute + integer coding, with high-cardinality attributes
    folded to ``MAX_LEVELS`` levels + "other". Classification targets are
    label-encoded. All encoders are fit on the training split only; categories
    unseen in training map to a dedicated final level.

    Parameters
    ----------
    dataset_csv : str
        Path to the dataset CSV.
    task : str
        ``"classification"`` or ``"regression"``.
    seed : int, default=SEED
        Split seed (enables repeated-holdout runs).
    fold : int or None, default=None
        0-based fold of stratified ``N_FOLDS``-fold CV; ``None`` = 80/20 holdout.

    Returns
    -------
    prep : Prepared
        Framework-neutral split; call ``.onehot()`` / ``.mixed()`` /
        ``.discretized()`` to materialize a representation.
    """
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder

    header = pd.read_csv(dataset_csv, nrows=0)
    feature_cols = list(header.columns[:-1])
    target = header.columns[-1]

    nominal_declared, schema_source = _read_schema(dataset_csv, feature_cols)
    # Force declared-nominal columns to load as strings so integer-coded
    # categories are not silently taken for numbers.
    dtypes = {c: str for c in nominal_declared}
    df = pd.read_csv(dataset_csv, dtype=dtypes)

    X = df.drop(columns=[target])
    y = df[target]

    if schema_source == "dtype-fallback":
        num_cols = X.select_dtypes(include="number").columns.tolist()
        nom_cols = [c for c in X.columns if c not in num_cols]
    else:
        nom_cols = [c for c in X.columns if c in nominal_declared]
        rest = [c for c in X.columns if c not in nominal_declared]
        # Any column the schema didn't cover: fall back to its dtype.
        num_cols = [c for c in rest if pd.api.types.is_numeric_dtype(X[c])]
        nom_cols += [c for c in rest if c not in num_cols]

    is_clf = task == "classification"
    if is_clf:
        y = pd.Series(LabelEncoder().fit_transform(y.astype(str)), index=y.index)
    else:
        y = pd.to_numeric(y, errors="coerce")

    if fold is not None:
        tr_idx, te_idx = _split_indices(y, is_clf, seed, fold, len(X))
        X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
        y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]
    else:
        strat = y if is_clf else None
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=seed, stratify=strat
        )

    num_tr, num_te = _prepare_numeric(X_tr, X_te, num_cols)
    nom_tr, nom_te, n_categories = _prepare_nominal(X_tr, X_te, nom_cols)

    meta = {
        "n_train": int(num_tr.shape[0]),
        "n_test": int(num_te.shape[0]),
        "n_features_raw": int(X.shape[1]),
        "n_numeric": len(num_cols),
        "n_nominal": len(nom_cols),
        "n_features_onehot": int(num_tr.shape[1] + sum(n_categories)),
        "n_classes": int(len(np.unique(np.asarray(y_tr)))) if is_clf else None,
        "schema_source": schema_source,
        "max_levels": MAX_LEVELS,
        "n_capped_nominal": int(sum(1 for k in n_categories if k >= MAX_LEVELS + 1)),
    }
    return Prepared(num_tr, num_te, nom_tr, nom_te,
                    np.asarray(y_tr), np.asarray(y_te), n_categories, meta)


def _prepare_numeric(X_tr, X_te, num_cols):
    """Median-impute and standardize the numeric block (fit on train only).

    Parameters
    ----------
    X_tr, X_te : pd.DataFrame
        Raw train/test feature frames.
    num_cols : list of str
        Numeric column names.

    Returns
    -------
    (num_tr, num_te) : tuple of np.ndarray
        Standardized float64 blocks, possibly with zero columns.
    """
    if not num_cols:
        return (np.zeros((len(X_tr), 0)), np.zeros((len(X_te), 0)))
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    pipe = Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("sc", StandardScaler())])
    tr = X_tr[num_cols].apply(pd.to_numeric, errors="coerce")
    te = X_te[num_cols].apply(pd.to_numeric, errors="coerce")
    # An all-missing column leaves NaN after imputation; zero it out so the
    # frameworks are not handed NaNs they each treat differently.
    return (np.nan_to_num(pipe.fit_transform(tr)).astype(np.float64),
            np.nan_to_num(pipe.transform(te)).astype(np.float64))


def _prepare_nominal(X_tr, X_te, nom_cols):
    """Impute and integer-code the nominal block (vocabulary from train only).

    Levels beyond the ``MAX_LEVELS`` most frequent training levels, plus any
    level unseen in training, collapse into one final "other/unseen" code.

    Parameters
    ----------
    X_tr, X_te : pd.DataFrame
        Raw train/test feature frames.
    nom_cols : list of str
        Nominal column names.

    Returns
    -------
    (nom_tr, nom_te, n_categories) : tuple
        Integer code matrices and the per-column level count (kept levels + 1).
    """
    if not nom_cols:
        return (np.zeros((len(X_tr), 0), dtype=int),
                np.zeros((len(X_te), 0), dtype=int), [])

    tr_codes, te_codes, n_categories = [], [], []
    for col in nom_cols:
        s_tr = X_tr[col].astype(str)
        s_te = X_te[col].astype(str)
        # Impute with the training mode before deciding the vocabulary.
        counts = s_tr[s_tr != "nan"].value_counts()
        fill = counts.index[0] if len(counts) else "missing"
        s_tr = s_tr.replace("nan", fill)
        s_te = s_te.replace("nan", fill)

        keep = list(counts.index[:MAX_LEVELS])
        lookup = {v: i for i, v in enumerate(keep)}
        other = len(keep)  # single trailing level for rare + unseen values
        c_tr = s_tr.map(lookup).fillna(other).astype(int).to_numpy()
        c_te = s_te.map(lookup).fillna(other).astype(int).to_numpy()
        tr_codes.append(c_tr)
        te_codes.append(c_te)
        # Only declare the "other" level when something actually lands in it.
        # Carrying an always-empty level would add a constant-zero one-hot
        # column and inflate the attribute count that drives sqrt(p) for the
        # random forests — a 50% inflation on all-binary datasets.
        used_other = bool((c_tr == other).any() or (c_te == other).any())
        n_categories.append(max(1, other + 1 if used_other else other))

    return (np.column_stack(tr_codes), np.column_stack(te_codes), n_categories)


def compute_metrics(task: str, y_true, y_pred) -> dict:
    """Return the quality metrics appropriate to the task."""
    if task == "classification":
        from sklearn.metrics import (accuracy_score, f1_score,
                                      balanced_accuracy_score, precision_score,
                                      recall_score)
        return {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
            "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        }
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    return {
        "rmse": float(mean_squared_error(y_true, y_pred) ** 0.5),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def peak_rss_mb() -> float:
    """Peak resident set size of this process, in MB (Linux ru_maxrss is KB)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def write_result(out_dir: str, record: dict) -> str:
    """Write one experiment's result as a standalone JSON file (parallel-safe)."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    stem = (f"{record['framework']}__{record['algorithm']}__{record['dataset']}"
            f"__{record.get('config', 'matched')}")
    if record.get("seed", SEED) != SEED:
        stem += f"__s{record['seed']}"
    if record.get("fold") is not None:
        stem += f"__f{record['fold']}"
    path = os.path.join(out_dir, stem + ".json")
    with open(path, "w") as fh:
        json.dump(record, fh, indent=2, default=str)
    return path


def run_experiment(framework, algo_key, dataset_csv, task, bucket, out_dir,
                   build_and_run, config="matched", seed=SEED, fold=None):
    """Shared driver: prepare data, time the model, capture metrics + resources.

    ``build_and_run(prep, task, config)`` must return
    ``(y_pred, fit_s, predict_s, info)`` and is supplied by each framework
    runner; ``info`` is a dict of resolved options recorded for transparency.
    Materializing a representation from ``prep`` happens inside the runner but
    **before** its timers start.

    Parameters
    ----------
    framework : str
        ``"sklearn"``, ``"tuiml"``, or ``"weka"``.
    algo_key : str
        Key into ``algorithms.ALGORITHMS``.
    dataset_csv : str
        Path to the dataset CSV.
    task : str
        ``"classification"`` or ``"regression"``.
    bucket : str
        Dataset bucket folder name (``binary`` / ``multiclass`` / ``regression``).
    out_dir : str
        Directory for the per-experiment result JSON.
    build_and_run : callable
        Framework-specific fit/predict closure.
    config : str, default="matched"
        ``"matched"`` (hyperparameters aligned across libraries) or
        ``"defaults"`` (each library's out-of-the-box settings).
    seed : int, default=SEED
        Split seed.
    fold : int or None, default=None
        CV fold index, or ``None`` for the holdout split.

    Returns
    -------
    path : str
        Path to the written result JSON.
    """
    dataset = Path(dataset_csv).parent.name
    cpu0 = time.process_time()
    wall0 = time.perf_counter()
    record = {
        "framework": framework, "algorithm": algo_key, "dataset": dataset,
        "bucket": bucket, "task": task, "config": config, "seed": seed,
        "fold": fold, "status": "ok",
    }
    try:
        prep = load_and_prepare(dataset_csv, task, seed=seed, fold=fold)
        record.update(prep.meta)

        # Matched regression standardizes the target for every framework, then
        # inverts the predictions before scoring. Weka's MultilayerPerceptron
        # normalizes the numeric class internally and the other two libraries do
        # not — a preprocessing step hidden inside one algorithm. Leaving it to
        # each library either hands Weka a silent advantage or (with -C) makes
        # all three diverge on raw targets. Doing it in the harness makes the
        # step explicit and identical. It is a no-op for the scale-invariant
        # learners (trees, k-NN, linear), and it gives SVR's epsilon the same
        # meaning everywhere. `defaults` deliberately leaves this alone.
        y_mu, y_sd = 0.0, 1.0
        scaled = task == "regression" and config == "matched"
        if scaled:
            y_mu = float(np.mean(prep.y_tr))
            y_sd = float(np.std(prep.y_tr)) or 1.0
            prep.y_tr = (prep.y_tr - y_mu) / y_sd
        record["target_standardized"] = scaled

        y_pred, fit_s, predict_s, info = build_and_run(prep, task, config)
        if scaled:
            y_pred = np.asarray(y_pred, dtype=float) * y_sd + y_mu
        # y_te is never scaled, so metrics stay in the original units.
        record["metrics"] = compute_metrics(task, prep.y_te, y_pred)
        record["fit_s"] = round(fit_s, 4)
        record["predict_s"] = round(predict_s, 4)
        record.update(info)
    except Exception as e:  # noqa: BLE001
        record["status"] = "error"
        record["error"] = f"{type(e).__name__}: {str(e)[:300]}"

    record["wall_total_s"] = round(time.perf_counter() - wall0, 4)
    record["cpu_total_s"] = round(time.process_time() - cpu0, 4)
    record["peak_rss_mb"] = round(peak_rss_mb(), 1)
    path = write_result(out_dir, record)
    status = record["status"]
    extra = record.get("metrics") or record.get("error")
    print(f"[{framework}/{config}] {algo_key} / {dataset}: {status} "
          f"wall={record['wall_total_s']}s rss={record['peak_rss_mb']}MB :: {extra}")
    return path
