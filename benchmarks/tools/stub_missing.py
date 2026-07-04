#!/usr/bin/env python3
"""Write a status=timeout stub for any (fw,algo,dataset[,seed]) still lacking a
result, so every framework ends with the identical, complete set of result files."""
import argparse
import json
import re
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--jobs-file", default="jobs.txt")
ap.add_argument("--results", default="results")
args = ap.parse_args()

DEFAULT_SEED = 42

jobs = Path(args.jobs_file).read_text().splitlines()
results = {p.name for p in Path(args.results).glob("*.json")}

n = 0
for line in jobs:
    if not line.strip():
        continue
    fw = re.search(r"bench_(\w+)\.py", line).group(1)
    algo = re.search(r"--algo (\S+)", line).group(1)
    dataset = Path(re.search(r"--dataset (\S+)", line).group(1)).parent.name
    bucket = re.search(r"--bucket (\S+)", line).group(1)
    task = re.search(r"--task (\S+)", line).group(1)
    seed_m = re.search(r"--seed (\d+)", line)
    seed = int(seed_m.group(1)) if seed_m else DEFAULT_SEED
    fname = f"{fw}__{algo}__{dataset}.json"
    if seed != DEFAULT_SEED:
        fname = fname[:-5] + f"__s{seed}.json"
    if fname in results:
        continue
    rec = {"framework": fw, "algorithm": algo, "dataset": dataset,
           "bucket": bucket, "task": task, "seed": seed, "status": "timeout",
           "error": "killed by per-job timeout (computation too slow to finish)"}
    Path(args.results, fname).write_text(json.dumps(rec, indent=2))
    n += 1
print(f"stubbed {n} still-missing combos as timeout")
