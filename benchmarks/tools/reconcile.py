#!/usr/bin/env python3
"""Compare expected experiments (jobs.txt) against produced result files.

Prints which (framework, algo, dataset) combos lack a result JSON (e.g. killed
by timeout) and writes the missing commands to jobs_missing.txt for re-running.
"""
import re
from pathlib import Path

jobs = Path("jobs.txt").read_text().splitlines()
results = {p.name for p in Path("results").glob("*.json")}

missing_cmds = []
missing = []
for line in jobs:
    if not line.strip():
        continue
    fw = re.search(r"bench_(\w+)\.py", line).group(1)
    algo = re.search(r"--algo (\S+)", line).group(1)
    ds_path = re.search(r"--dataset (\S+)", line).group(1)
    dataset = Path(ds_path).parent.name
    fname = f"{fw}__{algo}__{dataset}.json"
    if fname not in results:
        missing.append((fw, algo, dataset))
        missing_cmds.append(line)

print(f"expected={len([l for l in jobs if l.strip()])}  produced={len(results)}  missing={len(missing)}")
if missing:
    Path("jobs_missing.txt").write_text("\n".join(missing_cmds) + "\n")
    by_fw = {}
    for fw, algo, ds in missing:
        by_fw.setdefault(fw, []).append(f"{algo}/{ds}")
    for fw, items in sorted(by_fw.items()):
        print(f"\n{fw}: {len(items)} missing")
        for it in items:
            print(f"   {it}")
    print("\nwrote jobs_missing.txt")
