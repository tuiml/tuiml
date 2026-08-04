#!/usr/bin/env python3
"""Benchmark one (algorithm, dataset) with scikit-learn.

Usage:
    python3 bench_sklearn.py --algo random_forest --dataset <path/to.csv> \
        --task classification --bucket binary --config matched --out results/
"""
import os
# Pin BLAS/OpenMP to one thread per process: we parallelize at the process level
# (xargs -P), so per-process threading only causes oversubscription — and on
# high-core machines (e.g. 128 cores) it overruns OpenBLAS's NUM_THREADS limit
# and segfaults KNeighbors. Must be set before numpy/sklearn import.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import importlib
import time

import common
import algorithms as algos


def _lib_version() -> str:
    """Return the scikit-learn version, recorded with every result."""
    import sklearn
    return sklearn.__version__


def build_and_run(prep, task, config):
    """Fit and evaluate one scikit-learn estimator on the prepared split.

    Parameters
    ----------
    prep : common.Prepared
        Framework-neutral prepared split.
    task : str
        ``"classification"`` or ``"regression"``.
    config : str
        ``"matched"`` or ``"defaults"``.

    Returns
    -------
    (y_pred, fit_s, predict_s, info) : tuple
        Predictions, fit/predict seconds, and the resolved constructor kwargs.
    """
    algo_key = ARGS.algo
    module, cls_name, _ = algos.spec_for(algo_key, "sklearn")

    # Materialize the representation *before* any timer starts.
    if algos.prep_for(algo_key) == "discretized":
        X_tr, X_te, n_categories = prep.discretized()
    else:
        X_tr, X_te = prep.onehot()
        n_categories = None

    ctx = algos.context(X_tr)
    kwargs = algos.resolve(algo_key, "sklearn", config, ctx)
    # CategoricalNB needs the full per-feature category count so test-set
    # categories never index past the fitted tables.
    if n_categories is not None and cls_name == "CategoricalNB":
        kwargs["min_categories"] = n_categories

    cls = getattr(importlib.import_module(module), cls_name)
    model = cls(**kwargs)

    t0 = time.perf_counter()
    model.fit(X_tr, prep.y_tr)
    fit_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    y_pred = model.predict(X_te)
    predict_s = time.perf_counter() - t1

    shown = {k: v for k, v in kwargs.items() if k != "min_categories"}
    info = {"options": repr(shown), "lib_version": _lib_version(),
            "note": algos.ALGORITHMS[algo_key].get("note", "")}
    return y_pred, fit_s, predict_s, info


def main():
    global ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--task", required=True, choices=["classification", "regression"])
    ap.add_argument("--bucket", required=True)
    ap.add_argument("--config", default="matched", choices=list(algos.CONFIGS))
    ap.add_argument("--out", default="results")
    ap.add_argument("--seed", type=int, default=common.SEED)
    ap.add_argument("--fold", type=int, default=None)
    ARGS = ap.parse_args()
    common.run_experiment("sklearn", ARGS.algo, ARGS.dataset, ARGS.task,
                          ARGS.bucket, ARGS.out, build_and_run,
                          config=ARGS.config, seed=ARGS.seed, fold=ARGS.fold)


if __name__ == "__main__":
    main()
