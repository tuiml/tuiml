#!/usr/bin/env python3
"""Print human-readable comparison tables from the benchmark result JSONs."""
import glob
import json

import pandas as pd

pd.set_option("display.width", 160)
pd.set_option("display.max_rows", 200)

rows = []
for fp in glob.glob("results/*.json"):
    r = json.load(open(fp))
    m = r.pop("metrics", {}) or {}
    base = {k: v for k, v in r.items() if not isinstance(v, dict)}
    base.update({f"m_{k}": v for k, v in m.items()})
    rows.append(base)

df = pd.DataFrame(rows)
ok = df[df.status == "ok"].copy()
print(f"TOTAL result files: {len(df)}   ok: {len(ok)}   error: {(df.status!='ok').sum()}")

print("\n================ COVERAGE (count by framework x status) ================")
print(df.pivot_table(index="framework", columns="status", values="dataset",
                     aggfunc="count", fill_value=0))

clf = ok[ok.task == "classification"]
reg = ok[ok.task == "regression"]

print("\n================ CLASSIFICATION  (means over all clf experiments) ================")
g = clf.groupby("framework").agg(
    n=("dataset", "count"),
    accuracy=("m_accuracy", "mean"),
    f1_macro=("m_f1_macro", "mean"),
    fit_s=("fit_s", "mean"),
    wall_s=("wall_total_s", "mean"),
    peak_rss_mb=("peak_rss_mb", "mean"),
).round(4)
print(g)

print("\n  -- mean accuracy by algorithm x framework --")
print(clf.pivot_table(index="algorithm", columns="framework",
                      values="m_accuracy", aggfunc="mean").round(4))

print("\n================ REGRESSION  (means over all reg experiments) ================")
g2 = reg.groupby("framework").agg(
    n=("dataset", "count"),
    r2=("m_r2", "mean"),
    fit_s=("fit_s", "mean"),
    wall_s=("wall_total_s", "mean"),
    peak_rss_mb=("peak_rss_mb", "mean"),
).round(4)
print(g2)

print("\n  -- mean R2 by algorithm x framework --")
print(reg.pivot_table(index="algorithm", columns="framework",
                      values="m_r2", aggfunc="mean").round(4))

if (df.status != "ok").any():
    print("\n================ FAILURES ================")
    print(df[df.status != "ok"].groupby(["framework", "algorithm"]).size())
