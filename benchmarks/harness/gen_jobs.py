#!/usr/bin/env python3
"""Generate the flat job list: one line = one (framework, algorithm, dataset)
experiment, each runnable as its own process. Written to jobs.txt for run_all.sh.

Usage:
    python3 gen_jobs.py --datasets ~/TuiML/datasets --out results \
        [--frameworks sklearn tuiml weka] [--only-bucket regression]
"""
import argparse
import os
from pathlib import Path

from algorithms import ALGORITHMS, keys_for_task

BUCKET_TASK = {"regression": "regression", "binary": "classification",
               "multiclass": "classification"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", required=True, help="root with regression/binary/multiclass")
    ap.add_argument("--out", default="results")
    ap.add_argument("--frameworks", nargs="+", default=["sklearn", "tuiml", "weka"])
    ap.add_argument("--only-bucket", choices=list(BUCKET_TASK))
    ap.add_argument("--jobs-file", default="jobs.txt")
    ap.add_argument("--python", default="python3")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42],
                    help="split seeds; one job per seed (repeated holdout)")
    args = ap.parse_args()

    here = Path(__file__).resolve().parent
    root = Path(os.path.expanduser(args.datasets))
    buckets = [args.only_bucket] if args.only_bucket else list(BUCKET_TASK)

    lines = []
    for bucket in buckets:
        task = BUCKET_TASK[bucket]
        bdir = root / bucket
        if not bdir.is_dir():
            continue
        for ds_dir in sorted(p for p in bdir.iterdir() if p.is_dir()):
            csv = ds_dir / f"{ds_dir.name}.csv"
            if not csv.exists():
                continue
            for algo in keys_for_task(task):
                for fw in args.frameworks:
                    runner = here / f"bench_{fw}.py"
                    for seed in args.seeds:
                        lines.append(
                            f"{args.python} {runner} --algo {algo} "
                            f"--dataset {csv} --task {task} --bucket {bucket} "
                            f"--out {args.out} --seed {seed}"
                        )

    Path(args.jobs_file).write_text("\n".join(lines) + "\n")
    print(f"Wrote {len(lines)} jobs to {args.jobs_file} "
          f"({len(args.frameworks)} frameworks x algorithms x datasets "
          f"x {len(args.seeds)} seeds)")


if __name__ == "__main__":
    main()
