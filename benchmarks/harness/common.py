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


def load_and_prepare(dataset_csv: str, task: str):
    """Load a CSV (target = last column), split, and preprocess identically.

    Numeric features: median impute + standardize. Categorical: most-frequent
    impute + one-hot (capped categories). Classification targets are label-encoded.

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

    strat = y if is_clf else None
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=strat
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
    path = os.path.join(out_dir, fname)
    with open(path, "w") as fh:
        json.dump(record, fh, indent=2, default=str)
    return path


def run_experiment(framework, algo_key, dataset_csv, task, bucket, out_dir, build_and_run):
    """Shared driver: prepare data, time the model, capture metrics + resources.

    ``build_and_run(X_tr, X_te, y_tr, task, meta)`` must return
    ``(y_pred, fit_s, predict_s)`` and is supplied by each framework runner.
    """
    dataset = Path(dataset_csv).parent.name
    cpu0 = time.process_time()
    wall0 = time.perf_counter()
    record = {
        "framework": framework, "algorithm": algo_key, "dataset": dataset,
        "bucket": bucket, "task": task, "status": "ok",
    }
    try:
        X_tr, X_te, y_tr, y_te, meta = load_and_prepare(dataset_csv, task)
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
