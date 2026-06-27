#!/usr/bin/env python3
"""Write a status=timeout stub for any (fw,algo,dataset) still lacking a result,
so every framework ends with the identical, complete set of result files."""
import json
import re
from pathlib import Path

jobs = Path("jobs.txt").read_text().splitlines()
results = {p.name for p in Path("results").glob("*.json")}

n = 0
for line in jobs:
    if not line.strip():
        continue
    fw = re.search(r"bench_(\w+)\.py", line).group(1)
    algo = re.search(r"--algo (\S+)", line).group(1)
    dataset = Path(re.search(r"--dataset (\S+)", line).group(1)).parent.name
    bucket = re.search(r"--bucket (\S+)", line).group(1)
    task = re.search(r"--task (\S+)", line).group(1)
    fname = f"{fw}__{algo}__{dataset}.json"
    if fname in results:
        continue
    rec = {"framework": fw, "algorithm": algo, "dataset": dataset,
           "bucket": bucket, "task": task, "status": "timeout",
           "error": "killed by per-job timeout (computation too slow to finish)"}
    Path("results", fname).write_text(json.dumps(rec, indent=2))
    n += 1
print(f"stubbed {n} still-missing combos as timeout")
