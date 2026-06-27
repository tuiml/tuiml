#!/usr/bin/env python3
"""Benchmark one (algorithm, dataset) with scikit-learn.

Usage:
    python3 bench_sklearn.py --algo random_forest --dataset <path/to.csv> \
        --task classification --bucket binary --out results/
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
from algorithms import ALGORITHMS


def build_and_run(X_tr, X_te, y_tr, task, meta):
    spec = ALGORITHMS[ARGS.algo]["sklearn"]
    module, cls_name, kwargs = spec
    cls = getattr(importlib.import_module(module), cls_name)
    model = cls(**kwargs)

    t0 = time.perf_counter()
    model.fit(X_tr, y_tr)
    fit_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    y_pred = model.predict(X_te)
    predict_s = time.perf_counter() - t1
    return y_pred, fit_s, predict_s


def main():
    global ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--task", required=True, choices=["classification", "regression"])
    ap.add_argument("--bucket", required=True)
    ap.add_argument("--out", default="results")
    ARGS = ap.parse_args()
    common.run_experiment("sklearn", ARGS.algo, ARGS.dataset, ARGS.task,
                          ARGS.bucket, ARGS.out, build_and_run)


if __name__ == "__main__":
    main()
