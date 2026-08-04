#!/usr/bin/env python3
"""Benchmark one (algorithm, dataset) with TuiML.

Usage:
    python3 bench_tuiml.py --algo random_forest --dataset <path/to.csv> \
        --task classification --bucket binary --config matched --out results/
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
import algorithms as algos

import tuiml  # noqa: F401
import tuiml.algorithms  # noqa: F401  triggers registration
try:
    # Current layout.
    from tuiml.registry import registry
except ImportError:  # pragma: no cover - depends on installed version
    # 0.1.6 and earlier kept the registry under tuiml.hub. The benchmark has to
    # run against whatever release is installed on the machine, so accept both.
    from tuiml.hub import registry


def _lib_version() -> str:
    """Return the TuiML version, recorded with every result."""
    return getattr(tuiml, "__version__", "unknown")


def build_and_run(prep, task, config):
    """Fit and evaluate one TuiML estimator on the prepared split.

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
    name = algos.spec_for(algo_key, "tuiml")[0]

    # Materialize the representation *before* any timer starts.
    if algos.prep_for(algo_key) == "discretized":
        X_tr, X_te, n_categories = prep.discretized()
    else:
        X_tr, X_te = prep.onehot()
        n_categories = None

    ctx = algos.context(X_tr)
    kwargs = algos.resolve(algo_key, "tuiml", config, ctx)
    if n_categories is not None and "min_categories" not in kwargs:
        kwargs["min_categories"] = n_categories

    model = registry.create(name, **kwargs)

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
    common.run_experiment("tuiml", ARGS.algo, ARGS.dataset, ARGS.task,
                          ARGS.bucket, ARGS.out, build_and_run,
                          config=ARGS.config, seed=ARGS.seed, fold=ARGS.fold)


if __name__ == "__main__":
    main()
