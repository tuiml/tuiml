#!/usr/bin/env python3
"""Benchmark one (algorithm, dataset) with python-weka-wrapper3.

Feeds the SAME preprocessed numeric arrays as the other runners (via common),
converts them to Weka Instances, and trains/evaluates the mapped Weka classifier.
Note: peak RSS includes the in-process JVM, so weka's memory baseline is higher
than the pure-Python frameworks — this is inherent to running Weka.

Usage:
    python3 bench_weka.py --algo random_forest --dataset <path/to.csv> \
        --task classification --bucket binary --out results/
"""
import os
# Pin BLAS/OpenMP to one thread per process (we parallelize at the process
# level); avoids oversubscription on high-core machines. Set before numpy import.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
# Cap the JVM to a single processor so each Weka process doesn't size its
# GC/JIT/ForkJoinPool threads to the full core count (128 here) — that
# oversubscribes the box under parallel runs and inflates memory/wall-time.
os.environ.setdefault("JAVA_TOOL_OPTIONS", "-XX:ActiveProcessorCount=1")

import argparse
import time

import numpy as np

import common
from algorithms import ALGORITHMS

import weka.core.jvm as jvm
from weka.core.dataset import create_instances_from_matrices, Instances
from weka.filters import Filter
from weka.classifiers import Classifier


def build_and_run(X_tr, X_te, y_tr, task, meta):
    classname, options = ALGORITHMS[ARGS.algo]["weka"]
    n_train = X_tr.shape[0]

    # Build one combined Instances so train/test share an identical header
    # (essential for a nominal class to carry the same label set).
    X_all = np.vstack([X_tr, X_te])
    # y for test rows is unknown to the model; fill with train's first label.
    y_all = np.concatenate([np.asarray(y_tr, dtype=float),
                            np.full(X_te.shape[0], float(np.asarray(y_tr)[0]))])
    data = create_instances_from_matrices(X_all, y_all, name="bench")

    if task == "classification":
        n2n = Filter(classname="weka.filters.unsupervised.attribute.NumericToNominal",
                     options=["-R", "last"])
        n2n.inputformat(data)
        data = n2n.filter(data)
    data.class_is_last()

    train = Instances.copy_instances(data, 0, n_train)
    test = Instances.copy_instances(data, n_train, data.num_instances - n_train)

    model = Classifier(classname=classname, options=options)
    t0 = time.perf_counter()
    model.build_classifier(train)
    fit_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    preds = []
    for i in range(test.num_instances):
        p = model.classify_instance(test.get_instance(i))
        if task == "classification":
            # p is an index into the nominal class values; map back to the
            # original encoded integer label.
            preds.append(int(test.class_attribute.value(int(p))))
        else:
            preds.append(float(p))
    predict_s = time.perf_counter() - t1
    return np.asarray(preds), fit_s, predict_s


def main():
    global ARGS
    ap = argparse.ArgumentParser()
    ap.add_argument("--algo", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--task", required=True, choices=["classification", "regression"])
    ap.add_argument("--bucket", required=True)
    ap.add_argument("--out", default="results")
    ap.add_argument("--heap", default="2048m", help="JVM max heap")
    ARGS = ap.parse_args()

    jvm.start(packages=False, max_heap_size=ARGS.heap)
    try:
        common.run_experiment("weka", ARGS.algo, ARGS.dataset, ARGS.task,
                              ARGS.bucket, ARGS.out, build_and_run)
    finally:
        jvm.stop()


if __name__ == "__main__":
    main()
