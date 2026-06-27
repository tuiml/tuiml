#!/usr/bin/env python3
"""Dataset-wise comparison: best result each framework achieves per dataset."""
import glob
import json

import pandas as pd

pd.set_option("display.width", 200)
pd.set_option("display.max_rows", 300)
pd.set_option("display.max_columns", 30)

rows = []
for fp in glob.glob("results/*.json"):
    r = json.load(open(fp))
    m = r.pop("metrics", {}) or {}
    base = {k: v for k, v in r.items() if not isinstance(v, dict)}
    base.update({f"m_{k}": v for k, v in m.items()})
    rows.append(base)
df = pd.DataFrame(rows)
ok = df[df.status == "ok"].copy()


def show(task, metric, higher_better=True):
    sub = ok[ok.task == task]
    if sub.empty:
        return
    agg = "max" if higher_better else "min"
    piv = sub.pivot_table(index="dataset", columns="framework",
                          values=f"m_{metric}", aggfunc=agg).round(4)
    fws = [c for c in ["sklearn", "tuiml", "weka"] if c in piv.columns]
    piv["best_framework"] = piv[fws].idxmax(axis=1) if higher_better else piv[fws].idxmin(axis=1)
    label = "accuracy" if metric == "accuracy" else metric
    print(f"\n================ {task.upper()} — best {label} per dataset (across algorithms) ================")
    print(piv.to_string())
    # win tally
    print("\n  wins per framework:")
    print(piv["best_framework"].value_counts().to_string())


show("classification", "accuracy", higher_better=True)
show("regression", "r2", higher_better=True)
