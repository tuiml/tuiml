#!/usr/bin/env python3
"""Build the algorithm-first benchmark JSON for the website from summary.csv.

Emits one JSON file per config (``matched`` / ``defaults``) so the site can
show either a like-for-like comparison or the out-of-the-box numbers. Output
goes into ``website/static/benchmarks/``, which ``build.py`` serves verbatim.

Usage:
    python3 gen_web_json.py                          # both configs
    python3 gen_web_json.py --config matched         # matched only
    python3 gen_web_json.py --src ../summary_v2.csv

The JSON shape is unchanged from the previous version so the existing benchmark
page JS continues to work:

  meta         - dataset/framework counts, labels, protocol description
  algorithms   - ordered selector entries {key, label, task, score_label}
  data[algo]   - per-dataset cells
  timing       - per-algorithm median time per 1K samples (train + inference)
  datasets_meta - per-dataset row/feature counts and task type
"""
import argparse
import csv
import json
import statistics as st
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent  # benchmarks/tools/

LABELS = {
    "random_forest": "Random Forest", "decision_tree": "Decision Tree",
    "naive_bayes": "Naive Bayes", "logistic": "Logistic Regression",
    "knn": "K-Nearest Neighbors", "svm": "SVM", "mlp": "Neural Network (MLP)",
    "linear_regression": "Linear Regression", "random_forest_reg": "Random Forest",
    "decision_tree_reg": "Decision Tree", "knn_reg": "K-Nearest Neighbors",
    "svm_reg": "SVR", "mlp_reg": "Neural Network (MLP)",
}
CLF_ORDER = ["random_forest", "svm", "mlp", "logistic", "knn", "decision_tree", "naive_bayes"]
REG_ORDER = ["random_forest_reg", "svm_reg", "mlp_reg", "linear_regression", "knn_reg", "decision_tree_reg"]
FRAMEWORKS = ["tuiml", "sklearn", "weka"]
CONFIG_LABELS = {
    "matched": "Matched hyperparameters (aligned across libraries)",
    "defaults": "Library defaults (out-of-the-box settings)",
}


def fnum(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def build(src_csv, out_path, config_filter=None):
    """Read a summary CSV and write the benchmark JSON.

    Parameters
    ----------
    src_csv : Path
        Path to the summary.csv.
    out_path : Path
        Path to write the JSON to.
    config_filter : str or None
        ``"matched"``, ``"defaults"``, or ``None`` (write separate files for both).
    """
    rows_all = list(csv.DictReader(open(src_csv)))

    # Exclude numerically diverged runs before any aggregation.
    rows = [r for r in rows_all if r.get("diverged", "False") != "True"]

    # Accumulate per (algo, dataset, framework, config) cell, then reduce to
    # mean +/- std over the ok seeds (repeated-holdout aggregation). A cell with
    # zero ok seeds keeps its timeout/error status.
    raw = {}
    datasets_meta = {}
    n_seeds_seen = set()
    configs_present = set()
    for r in rows:
        cfg = r.get("config", "defaults")
        if config_filter and cfg != config_filter:
            continue
        configs_present.add(cfg)
        algo, ds, fw, task = r["algorithm"], r["dataset"], r["framework"], r["task"]
        score = fnum(r["metric_accuracy"]) if task == "classification" else fnum(r["metric_r2"])
        n_train, n_test = fnum(r["n_train"]), fnum(r["n_test"])
        n_seeds_seen.add((r.get("seed") or "42", r.get("fold") or ""))
        rec = {
            "score": score,
            "fit_s": fnum(r["fit_s"]),
            "predict_s": fnum(r["predict_s"]),
            "mem_mb": fnum(r["peak_rss_mb"]),
            "status": r["status"],
        }
        cell = raw.setdefault(algo, {}).setdefault(ds, {})
        # Gather all seeds for this framework+config combination for the
        # cell-level reduce; kept under the framework key so the reducer finds
        # them automatically.
        cell_key = fw if config_filter else f"{fw}:{cfg}"
        cell.setdefault(cell_key, []).append(rec)
        if n_train:
            cell["_n_train"] = n_train
            cell["_n_test"] = n_test
        if ds not in datasets_meta or datasets_meta[ds]["rows"] is None:
            datasets_meta[ds] = {
                "rows": int((n_train or 0) + (n_test or 0)) or None,
                "features": int(fnum(r["n_features_raw"])) if fnum(r["n_features_raw"]) else None,
                "task": task,
                "bucket": r.get("bucket", ""),
            }

    if not rows:
        print("No rows; check source or --config filter.")
        return

    def reduce_cell(recs):
        """Mean +/- std over ok seeds; keep failure status if no seed succeeded."""
        ok = [r for r in recs if r["status"] == "ok" and r["score"] is not None]
        if not ok:
            return {"score": None, "fit_s": None, "predict_s": None, "mem_mb": None,
                    "status": recs[0]["status"], "n_seeds": 0}
        mean = lambda k: st.mean(r[k] for r in ok if r[k] is not None)
        return {
            "score": mean("score"),
            "score_std": st.stdev(r["score"] for r in ok) if len(ok) > 1 else 0.0,
            "fit_s": round(mean("fit_s"), 4),
            "predict_s": round(mean("predict_s"), 4),
            "mem_mb": round(mean("mem_mb"), 1),
            "status": "ok",
            "n_seeds": len(ok),
        }

    def framework_keys_from_cell(cell):
        """Return the framework-anchored keys present in this cell."""
        return [k for k in cell if k in FRAMEWORKS
                or (not config_filter and any(k.startswith(fw + ":") for fw in FRAMEWORKS))]

    # If multi-config and no filter, split per config to write separate files.
    configs = [config_filter] if config_filter else sorted(configs_present)
    for cfg in configs:
        data = {}
        for algo, dss in raw.items():
            for ds, cell in dss.items():
                red = {k: v for k, v in cell.items() if k.startswith("_")}
                for fw in FRAMEWORKS:
                    key = fw if config_filter else f"{fw}:{cfg}"
                    if key in cell:
                        red[fw] = reduce_cell(cell[key])
                data.setdefault(algo, {})[ds] = red

        # If single-config, strip the config suffix from the output path so the
        # site knows where to find a default file.
        out = out_path
        if not config_filter:
            stem = out_path.stem
            out = out_path.with_stem(f"{stem}_{cfg}")

        algos_present = [a for a in CLF_ORDER + REG_ORDER if a in data]

        out_data = {}
        for algo in algos_present:
            lst = []
            for ds in sorted(data[algo]):
                cell = data[algo][ds]
                n_train, n_test = cell.get("_n_train"), cell.get("_n_test")
                row = {"dataset": ds, "rows": datasets_meta[ds]["rows"],
                       "features": datasets_meta[ds]["features"],
                       "n_train": int(n_train) if n_train else None,
                       "n_test": int(n_test) if n_test else None}
                for fw in FRAMEWORKS:
                    row[fw] = cell.get(fw)
                lst.append(row)
            out_data[algo] = lst

        # Per-algorithm median time per 1K samples (train + inference), per framework.
        timing = []
        for algo in algos_present:
            task = "classification" if algo in CLF_ORDER else "regression"
            entry = {"key": algo, "label": LABELS[algo], "task": task}
            for fw in FRAMEWORKS:
                tr, inf = [], []
                for ds in data[algo]:
                    cell = data[algo][ds]
                    rec = cell.get(fw)
                    nt, nte = cell.get("_n_train"), cell.get("_n_test")
                    if not rec or rec["status"] != "ok":
                        continue
                    if rec["fit_s"] is not None and nt:
                        tr.append(rec["fit_s"] / nt * 1000.0)
                    if rec["predict_s"] is not None and nte:
                        inf.append(rec["predict_s"] / nte * 1000.0)
                entry[fw] = {
                    "train_per1k": round(st.median(tr), 5) if tr else None,
                    "infer_per1k": round(st.median(inf), 5) if inf else None,
                }
            timing.append(entry)

        algorithms = [{
            "key": a, "label": LABELS[a],
            "task": "classification" if a in CLF_ORDER else "regression",
            "score_label": "Accuracy" if a in CLF_ORDER else "R²",
        } for a in algos_present]

        n_clf = sum(1 for m in datasets_meta.values() if m["task"] == "classification")
        n_reg = sum(1 for m in datasets_meta.values() if m["task"] == "regression")
        n_seeds = len(n_seeds_seen)
        protocol = (
            f"{n_seeds}-seed holdout (80/20 split, seed 42) with shared preprocessing"
            if n_seeds <= 1 else
            f"{n_seeds}-seed repeated holdout (80/20 split) with shared preprocessing"
        )
        meta = {
            "frameworks": FRAMEWORKS,
            "framework_labels": {"tuiml": "TuiML", "sklearn": "scikit-learn", "weka": "Weka"},
            "framework_colors": {"tuiml": "#f97316", "sklearn": "#3b82f6", "weka": "#a855f7"},
            "n_datasets": len(datasets_meta), "n_classification": n_clf, "n_regression": n_reg,
            "n_experiments": len(rows), "n_seeds": n_seeds,
            "config": cfg,
            "config_label": CONFIG_LABELS.get(cfg, cfg),
            "protocol": protocol,
            "suite": "TabArena-v0.1 (OpenML study 457)",
        }

        out.parent.mkdir(parents=True, exist_ok=True)
        json.dump({"meta": meta, "algorithms": algorithms, "datasets_meta": datasets_meta,
                   "data": out_data, "timing": timing}, open(out, "w"), indent=1)
        n_timeout = sum(1 for r in rows if r.get("config", "defaults") == cfg
                        and r["status"] in ("timeout", "error"))
        print(f"[{cfg}] wrote {out}")
        print(f"  datasets={len(datasets_meta)} algos={len(algorithms)} timing_rows={len(timing)}")
        print(f"  experiments={meta['n_experiments']}  seeds={n_seeds}  "
              f"timeout/error={n_timeout}  diverged_excluded={len(rows_all) - len(rows)}")
        if timing:
            print(f"  sample timing[0]: {json.dumps(timing[0], indent=0)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=str(SCRIPT.parent / "summary_v2.csv"),
                    help="path to summary.csv (default: ../summary_v2.csv)")
    ap.add_argument("--out", default=str(SCRIPT.parent.parent
                                         / "website" / "static" / "benchmarks"
                                         / "tabarena_results.json"),
                    help="path to write the JSON (default: ../../website/static/benchmarks/tabarena_results.json)")
    ap.add_argument("--config", choices=list(CONFIG_LABELS), default=None,
                    help="filter to one config (default: write separate files for both)")
    args = ap.parse_args()
    build(Path(args.src), Path(args.out), args.config)


if __name__ == "__main__":
    main()
