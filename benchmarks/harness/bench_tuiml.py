#!/usr/bin/env python3
"""Benchmark one (algorithm, dataset) with TuiML.

Usage:
    python3 bench_tuiml.py --algo random_forest --dataset <path/to.csv> \
        --task classification --bucket binary --out results/
"""
import os
# Pin BLAS/OpenMP to one thread per process (we parallelize at the process
# level); avoids oversubscription/segfaults on high-core machines. Set before
# numpy import.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import time

import common
from algorithms import ALGORITHMS

import tuiml  # noqa: F401
import tuiml.algorithms  # noqa: F401  triggers registration
from tuiml.hub import registry


def build_and_run(X_tr, X_te, y_tr, task, meta):
    name, kwargs = ALGORITHMS[ARGS.algo]["tuiml"]
    model = registry.create(name, **kwargs)

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
    common.run_experiment("tuiml", ARGS.algo, ARGS.dataset, ARGS.task,
                          ARGS.bucket, ARGS.out, build_and_run)


if __name__ == "__main__":
    main()
