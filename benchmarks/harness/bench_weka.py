#!/usr/bin/env python3
"""Benchmark one (algorithm, dataset) with python-weka-wrapper3.

Feeds the SAME prepared split as the other runners (via common), materialized in
the form Weka is designed for: numeric attributes plus **genuinely nominal**
attributes, declared as such in the ARFF header. Weka's tree and instance-based
learners then use their native nominal handling, and its function-based learners
(SMO, Logistic, MultilayerPerceptron) apply their own internal NominalToBinary,
which reproduces the one-hot the other frameworks are given.

Inference is measured as a **single batch call** (``distributionsForInstances``),
matching the vectorized ``predict`` the other two runners use. The previous
per-instance ``classifyInstance`` loop crossed the Python/JVM boundary once per
test row, so it mostly measured JPype call overhead rather than Weka.

Note: peak RSS includes the in-process JVM, so weka's memory baseline is higher
than the pure-Python frameworks — this is inherent to running Weka.

Usage:
    python3 bench_weka.py --algo random_forest --dataset <path/to.csv> \
        --task classification --bucket binary --config matched --out results/
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
import algorithms as algos

import weka.core.jvm as jvm
from weka.core.dataset import create_instances_from_matrices, Instances
from weka.filters import Filter
from weka.classifiers import Classifier


def _range_spec(idx0) -> str:
    """Convert 0-based column indices to a Weka 1-based attribute range string.

    Consecutive indices are compressed into ``a-b`` runs so the option string
    stays short on wide datasets.

    Parameters
    ----------
    idx0 : list of int
        0-based column indices.

    Returns
    -------
    spec : str
        Range specification such as ``"3,5-9,12"``; empty if no indices.
    """
    if not idx0:
        return ""
    idx = sorted(i + 1 for i in idx0)
    parts, start, prev = [], idx[0], idx[0]
    for i in idx[1:]:
        if i == prev + 1:
            prev = i
            continue
        parts.append(f"{start}-{prev}" if prev > start else f"{start}")
        start = prev = i
    parts.append(f"{start}-{prev}" if prev > start else f"{start}")
    return ",".join(parts)


def _to_instances(X_tr, X_te, y_tr, task, nominal_idx):
    """Build train/test Weka Instances sharing one header.

    Train and test must share an identical header so a nominal class carries the
    same label set on both sides, so they are converted together and split
    afterwards.

    Parameters
    ----------
    X_tr, X_te : np.ndarray
        Feature matrices (float64; nominal columns hold integer codes).
    y_tr : np.ndarray
        Training targets.
    task : str
        ``"classification"`` or ``"regression"``.
    nominal_idx : list of int
        0-based indices of columns to declare nominal.

    Returns
    -------
    (train, test) : tuple of Instances
        Class attribute set to the last column.
    """
    n_train = X_tr.shape[0]
    X_all = np.vstack([X_tr, X_te]).astype(np.float64)
    # y for test rows is unknown to the model; fill with train's first label.
    y_all = np.concatenate([np.asarray(y_tr, dtype=float),
                            np.full(X_te.shape[0], float(np.asarray(y_tr)[0]))])
    data = create_instances_from_matrices(X_all, y_all, name="bench")

    # Declare the nominal feature columns (and, for classification, the class)
    # nominal. Everything else stays numeric.
    to_convert = list(nominal_idx)
    if task == "classification":
        to_convert.append(X_all.shape[1])  # class column sits last
    spec = _range_spec(to_convert)
    if spec:
        n2n = Filter(classname="weka.filters.unsupervised.attribute.NumericToNominal",
                     options=["-R", spec])
        n2n.inputformat(data)
        data = n2n.filter(data)
    data.class_is_last()

    train = Instances.copy_instances(data, 0, n_train)
    test = Instances.copy_instances(data, n_train, data.num_instances - n_train)
    return train, test


def _decode(dists, test, task):
    """Turn a batch distribution matrix into the harness's label space.

    Parameters
    ----------
    dists : np.ndarray of shape (n_test, n_outputs)
        Output of ``distributionsForInstances``.
    test : Instances
        Test set, used to map class indices back to encoded integer labels.
    task : str
        ``"classification"`` or ``"regression"``.

    Returns
    -------
    y_pred : np.ndarray
        Predicted labels (int) or values (float).
    """
    dists = np.asarray(dists)
    if task != "classification":
        return dists[:, 0].astype(float)
    idx = dists.argmax(axis=1)
    # Nominal class values are the string forms of the encoded integer labels.
    values = [int(float(test.class_attribute.value(i)))
              for i in range(test.class_attribute.num_values)]
    lookup = np.asarray(values)
    return lookup[idx]


def build_and_run(prep, task, config):
    """Fit and evaluate one Weka classifier on the prepared split.

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
        Predictions, fit/predict seconds, and the resolved options.
    """
    algo_key = ARGS.algo
    classname = algos.spec_for(algo_key, "weka")[0]

    # Materialize Weka's representation *before* any timer starts.
    if algos.prep_for(algo_key) == "discretized":
        X_tr, X_te, _ = prep.discretized()
        nominal_idx = list(range(X_tr.shape[1]))
    else:
        X_tr, X_te, nominal_idx = prep.mixed()

    # $GAMMA comes from the one-hot representation so the RBF kernel matches the
    # other libraries; $SQRT_P from the attribute count Weka actually splits on.
    gamma_src = prep.onehot()[0] if algos.needs_gamma(algo_key, config) else X_tr
    ctx = algos.context(X_tr, gamma_source=gamma_src)
    options = algos.resolve(algo_key, "weka", config, ctx)

    train, test = _to_instances(X_tr, X_te, prep.y_tr, task, nominal_idx)

    model = Classifier(classname=classname, options=options)
    t0 = time.perf_counter()
    model.build_classifier(train)
    fit_s = time.perf_counter() - t0

    # Batch inference: one JVM call for the whole test set.
    t1 = time.perf_counter()
    dists = model.distributions_for_instances(test)
    predict_s = time.perf_counter() - t1

    batched = dists is not None
    if not batched:
        # Not a BatchPredictor (should not happen: AbstractClassifier implements
        # the interface) — fall back, but record that this row is not comparable.
        t1 = time.perf_counter()
        dists = np.asarray([model.distribution_for_instance(test.get_instance(i))
                            for i in range(test.num_instances)])
        predict_s = time.perf_counter() - t1

    y_pred = _decode(dists, test, task)
    from weka.core.version import weka_version
    info = {"options": " ".join(options), "batch_predict": batched,
            "lib_version": weka_version(),
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
    ap.add_argument("--heap", default="2048m", help="JVM max heap")
    ap.add_argument("--seed", type=int, default=common.SEED)
    ap.add_argument("--fold", type=int, default=None)
    ARGS = ap.parse_args()

    jvm.start(packages=False, max_heap_size=ARGS.heap)
    try:
        common.run_experiment("weka", ARGS.algo, ARGS.dataset, ARGS.task,
                              ARGS.bucket, ARGS.out, build_and_run,
                              config=ARGS.config, seed=ARGS.seed, fold=ARGS.fold)
    finally:
        jvm.stop()


if __name__ == "__main__":
    main()
