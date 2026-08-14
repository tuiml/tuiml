#!/usr/bin/env python3
"""Write an incomplete stub for every scheduled job lacking a result file.

The generated filename mirrors :func:`benchmarks.harness.common.write_result`,
including configuration, seed, and fold suffixes.  This makes aggregation
complete without pretending that a killed or otherwise lost job produced a
metric.
"""
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
    config_m = re.search(r"--config (\S+)", line)
    config = config_m.group(1) if config_m else "matched"
    seed_m = re.search(r"--seed (\d+)", line)
    seed = int(seed_m.group(1)) if seed_m else DEFAULT_SEED
    fold_m = re.search(r"--fold (\d+)", line)
    fold = int(fold_m.group(1)) if fold_m else None
    fname = f"{fw}__{algo}__{dataset}__{config}.json"
    if seed != DEFAULT_SEED:
        fname = fname[:-5] + f"__s{seed}.json"
    if fold is not None:
        fname = fname[:-5] + f"__f{fold}.json"
    if fname in results:
        continue
    rec = {"framework": fw, "algorithm": algo, "dataset": dataset,
           "bucket": bucket, "task": task, "config": config, "seed": seed,
           "fold": fold, "status": "incomplete",
           "error": "scheduled job produced no result file"}
    Path(args.results, fname).write_text(json.dumps(rec, indent=2))
    n += 1
print(f"stubbed {n} scheduled jobs with no result")
