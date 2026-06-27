#!/usr/bin/env python3
"""Show fit time vs row count for the expensive algorithms (svm, mlp, knn)."""
import glob
import json

rows = []
for fp in glob.glob("results/*.json"):
    r = json.load(open(fp))
    if r["algorithm"] not in ("svm", "svm_reg", "mlp", "mlp_reg", "knn", "knn_reg"):
        continue
    rows.append((r.get("n_train", 0), r["dataset"], r["algorithm"],
                 r["framework"], r.get("status"),
                 r.get("fit_s"), r.get("wall_total_s")))

rows.sort(reverse=True)
print(f"{'n_train':>8} {'algo':<10} {'framework':<8} {'status':<8} {'fit_s':>9} {'wall_s':>9}  dataset")
for n_train, ds, algo, fw, status, fit_s, wall in rows[:45]:
    fit_str = f"{fit_s:.1f}" if isinstance(fit_s, (int, float)) else "-"
    wall_str = f"{wall:.1f}" if isinstance(wall, (int, float)) else "-"
    print(f"{n_train:>8} {algo:<10} {fw:<8} {str(status):<8} {fit_str:>9} {wall_str:>9}  {ds}")
