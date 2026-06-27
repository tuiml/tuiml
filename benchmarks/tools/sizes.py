#!/usr/bin/env python3
"""Show dataset sizes, flag which ones had any timeout, sorted by row count."""
import glob
import json
from pathlib import Path

# datasets that produced a timeout stub or are missing a result
timed_out = set()
for fp in glob.glob("results/*.json"):
    r = json.load(open(fp))
    if r.get("status") == "timeout":
        timed_out.add(r["dataset"])

rows = []
for mp in glob.glob(str(Path.home() / "TuiML/datasets/*/*/metadata.json")):
    m = json.load(open(mp))
    rows.append((m["num_instances"], m["num_features"], m["bucket"], m["dataset"]))

rows.sort(reverse=True)
print(f"{'rows':>8} {'feats':>6}  {'bucket':<11} {'flag':<8} dataset")
for inst, feat, bucket, name in rows:
    flag = "TIMEOUT" if name in timed_out else ""
    print(f"{inst:>8} {feat:>6}  {bucket:<11} {flag:<8} {name}")
