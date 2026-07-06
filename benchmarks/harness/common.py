"""Shared benchmark harness: identical data prep, metrics, and resource capture.

All three framework runners (sklearn / tuiml / weka) import this so they are
compared on *exactly* the same train/test split and preprocessing — we benchmark
the algorithms, not each library's data handling. Each experiment runs in its own
OS process (launched by run_all.sh), so peak RSS via ``resource`` is per-experiment.
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


def load_and_prepare(dataset_csv: str, task: str, seed: int = SEED, fold=None):
    """Load a CSV (target = last column), split, and preprocess identically.

    Numeric features: median impute + standardize. Categorical: most-frequent
    impute + one-hot (capped categories). Classification targets are label-encoded.
    ``seed`` controls the train/test split, enabling repeated-holdout runs.
    ``fold`` (0-based) selects one fold of stratified ``N_FOLDS``-fold CV
    instead of the 80/20 holdout.

    Returns
    -------
    (X_train, X_test, y_train, y_test, meta) : tuple
        Arrays are dense float64; meta has shapes / class info.
    """
    from sklearn.model_selection import train_test_split
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder

    df = pd.read_csv(dataset_csv)
    target = df.columns[-1]
    X = df.drop(columns=[target])
    y = df[target]

    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

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

    num_pipe = Pipeline([("imp", SimpleImputer(strategy="median")),
                         ("sc", StandardScaler())])
    cat_pipe = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                         ("oh", OneHotEncoder(handle_unknown="ignore",
                                              max_categories=20, sparse_output=False))])
    pre = ColumnTransformer([("num", num_pipe, num_cols),
                             ("cat", cat_pipe, cat_cols)], remainder="drop")

    X_tr_t = pre.fit_transform(X_tr).astype(np.float64)
    X_te_t = pre.transform(X_te).astype(np.float64)
    y_tr_a = np.asarray(y_tr)
    y_te_a = np.asarray(y_te)

    meta = {
        "n_train": int(X_tr_t.shape[0]),
        "n_test": int(X_te_t.shape[0]),
        "n_features_raw": int(X.shape[1]),
        "n_features_prepared": int(X_tr_t.shape[1]),
        "n_classes": int(len(np.unique(y_tr_a))) if is_clf else None,
    }
    return X_tr_t, X_te_t, y_tr_a, y_te_a, meta


def load_and_prepare_categorical(dataset_csv: str, task: str, seed: int = SEED,
                                 fold=None):
    """Load a CSV and prepare an **all-categorical**, integer-coded matrix.

    Categorical-aware variant used by the Naive Bayes rows: nominal features
    are ordinal-encoded and continuous features are quantile-binned, so every
    column becomes a discrete category (the representation a categorical NB
    expects). The category *vocabulary* (encoder categories / bin edges) is fit
    on the full feature matrix — exactly the information Weka also derives from
    the ARFF header — so train and test share an identical, consistent coding
    and no framework sees an out-of-range category. Conditional probabilities
    are still estimated on the training split only.

    Returns
    -------
    (X_train, X_test, y_train, y_test, meta) : tuple
        Feature arrays are dense int matrices; ``meta['categorical']`` is True
        and ``meta['n_categories']`` lists the category count per column.
    """
    from sklearn.model_selection import train_test_split
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OrdinalEncoder, KBinsDiscretizer, LabelEncoder

    df = pd.read_csv(dataset_csv)
    target = df.columns[-1]
    X = df.drop(columns=[target])
    y = df[target]

    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]

    is_clf = task == "classification"
    if is_clf:
        y = pd.Series(LabelEncoder().fit_transform(y.astype(str)), index=y.index)
    else:
        y = pd.to_numeric(y, errors="coerce")

    n_bins = 10
    coded_parts = []      # list of (n_features, ) int columns, column-aligned
    n_categories = []

    # Continuous features: median impute then quantile-bin into ordinal codes.
    if num_cols:
        num_imp = SimpleImputer(strategy="median")
        Xnum = num_imp.fit_transform(X[num_cols])
        disc = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="quantile")
        # subsample=None keeps binning deterministic across sklearn versions.
        try:
            disc.set_params(subsample=None)
        except ValueError:
            pass
        Xnum_coded = disc.fit_transform(Xnum).astype(int)
        coded_parts.append(Xnum_coded)
        n_categories.extend(int(b) for b in disc.n_bins_)

    # Categorical features: most-frequent impute then ordinal-encode.
    if cat_cols:
        cat_imp = SimpleImputer(strategy="most_frequent")
        Xcat = cat_imp.fit_transform(X[cat_cols].astype(str))
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        Xcat_coded = enc.fit_transform(Xcat).astype(int)
        coded_parts.append(Xcat_coded)
        n_categories.extend(len(cats) for cats in enc.categories_)

    Xc = np.hstack(coded_parts).astype(int)
    # Any unknown (-1) -> 0; vocabulary was fit on full data so this is rare.
    Xc[Xc < 0] = 0

    if fold is not None:
        tr_idx, te_idx = _split_indices(y, is_clf, seed, fold, len(Xc))
        X_tr, X_te = Xc[tr_idx], Xc[te_idx]
        y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]
    else:
        strat = y if is_clf else None
        X_tr, X_te, y_tr, y_te = train_test_split(
            Xc, y, test_size=TEST_SIZE, random_state=seed, stratify=strat
        )

    meta = {
        "n_train": int(X_tr.shape[0]),
        "n_test": int(X_te.shape[0]),
        "n_features_raw": int(X.shape[1]),
        "n_features_prepared": int(X_tr.shape[1]),
        "n_classes": int(len(np.unique(y))) if is_clf else None,
        "categorical": True,
        "n_categories": [int(k) for k in n_categories],
    }
    return X_tr, X_te, np.asarray(y_tr), np.asarray(y_te), meta


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
    fname = f"{record['framework']}__{record['algorithm']}__{record['dataset']}.json"
    if record.get("seed", SEED) != SEED:
        fname = fname[:-5] + f"__s{record['seed']}.json"
    if record.get("fold") is not None:
        fname = fname[:-5] + f"__f{record['fold']}.json"
    path = os.path.join(out_dir, fname)
    with open(path, "w") as fh:
        json.dump(record, fh, indent=2, default=str)
    return path


def run_experiment(framework, algo_key, dataset_csv, task, bucket, out_dir,
                   build_and_run, prep="standard", seed=SEED, fold=None):
    """Shared driver: prepare data, time the model, capture metrics + resources.

    ``build_and_run(X_tr, X_te, y_tr, task, meta)`` must return
    ``(y_pred, fit_s, predict_s)`` and is supplied by each framework runner.
    ``prep`` selects the data preparation: ``"standard"`` (impute + scale +
    one-hot) or ``"categorical"`` (ordinal-encode + quantile-bin to an
    all-categorical matrix, used by the Naive Bayes rows). ``seed`` controls
    the train/test split for repeated-holdout runs.
    """
    dataset = Path(dataset_csv).parent.name
    cpu0 = time.process_time()
    wall0 = time.perf_counter()
    record = {
        "framework": framework, "algorithm": algo_key, "dataset": dataset,
        "bucket": bucket, "task": task, "seed": seed, "fold": fold, "status": "ok",
    }
    try:
        loader = load_and_prepare_categorical if prep == "categorical" else load_and_prepare
        X_tr, X_te, y_tr, y_te, meta = loader(dataset_csv, task, seed=seed, fold=fold)
        record.update(meta)
        y_pred, fit_s, predict_s = build_and_run(X_tr, X_te, y_tr, task, meta)
        record["metrics"] = compute_metrics(task, y_te, y_pred)
        record["fit_s"] = round(fit_s, 4)
        record["predict_s"] = round(predict_s, 4)
    except Exception as e:  # noqa: BLE001
        record["status"] = "error"
        record["error"] = f"{type(e).__name__}: {str(e)[:300]}"

    record["wall_total_s"] = round(time.perf_counter() - wall0, 4)
    record["cpu_total_s"] = round(time.process_time() - cpu0, 4)
    record["peak_rss_mb"] = round(peak_rss_mb(), 1)
    path = write_result(out_dir, record)
    status = record["status"]
    extra = record.get("metrics") or record.get("error")
    print(f"[{framework}] {algo_key} / {dataset}: {status} "
          f"wall={record['wall_total_s']}s rss={record['peak_rss_mb']}MB :: {extra}")
    return path
