#!/usr/bin/env python3
"""Build the algorithm-first benchmark JSON for the website from summary.csv.

Emits:
  meta         - dataset/framework counts + labels
  algorithms   - ordered selector entries {key,label,task,score_label}
  data[algo]   - per-dataset cells: {dataset, rows, features, n_train, n_test,
                 <fw>:{score, fit_s, predict_s, mem_mb, status}}
  timing       - per-algorithm median TIME PER 1K SAMPLES (train + inference),
                 per framework -> drives the two-panel dot plot
  score = accuracy (classification) or r2 (regression).
"""
import csv
import json
import statistics as st
from pathlib import Path

SRC = Path("/Users/nileshverma/Documents/GitHub/Work/tuiml-project/benchmark-results/summary.csv")
OUT = Path("/Users/nileshverma/Documents/GitHub/Work/tuiml-project/tuiml-website/static/benchmarks/tabarena_results.json")

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


def fnum(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


rows = list(csv.DictReader(open(SRC)))

# Accumulate every seed row per (algo, dataset, framework) cell, then reduce
# to mean +/- std over the ok seeds (repeated-holdout aggregation). A cell
# with zero ok seeds keeps its timeout/error status.
raw = {}
datasets_meta = {}
n_seeds_seen = set()
for r in rows:
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
    cell.setdefault(fw, []).append(rec)
    if n_train:
        cell["_n_train"] = n_train
        cell["_n_test"] = n_test
    if ds not in datasets_meta or datasets_meta[ds]["rows"] is None:
        datasets_meta[ds] = {
            "rows": int((n_train or 0) + (n_test or 0)) or None,
            "features": int(fnum(r["n_features_raw"])) if fnum(r["n_features_raw"]) else None,
            "task": task,
        }


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


data = {}
for algo, dss in raw.items():
    for ds, cell in dss.items():
        red = {k: v for k, v in cell.items() if k.startswith("_")}
        for fw in FRAMEWORKS:
            if fw in cell:
                red[fw] = reduce_cell(cell[fw])
        data.setdefault(algo, {})[ds] = red

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
meta = {
    "frameworks": FRAMEWORKS,
    "framework_labels": {"tuiml": "TuiML", "sklearn": "scikit-learn", "weka": "Weka"},
    "framework_colors": {"tuiml": "#f97316", "sklearn": "#3b82f6", "weka": "#a855f7"},
    "n_datasets": len(datasets_meta), "n_classification": n_clf, "n_regression": n_reg,
    "n_experiments": len(rows), "n_seeds": len(n_seeds_seen),
    "protocol": "stratified 10-fold cross-validation (shuffled, seed 42); scores are mean over folds",
    "suite": "TabArena-v0.1 (OpenML study 457)",
}

OUT.parent.mkdir(parents=True, exist_ok=True)
json.dump({"meta": meta, "algorithms": algorithms, "datasets_meta": datasets_meta,
           "data": out_data, "timing": timing}, open(OUT, "w"), indent=1)
print(f"wrote {OUT}")
print(f"datasets={len(datasets_meta)} algos={len(algorithms)} timing_rows={len(timing)}")
print("sample timing[0]:", json.dumps(timing[0], indent=0))
