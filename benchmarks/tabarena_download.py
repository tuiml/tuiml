#!/usr/bin/env python3
"""Download all TabArena-v0.1 datasets from OpenML into a task-bucketed tree.

Reads the TabArena curation metadata CSV (which carries the OpenML ``dataset_id``
and task type for each of the 51 datasets) and downloads each dataset via the
``openml`` package into:

    ~/TuiML/datasets/<bucket>/<dataset_name>/
        <dataset_name>.csv     # features + target, target column last
        metadata.json          # task type, target, sizes, OpenML ids, source

where <bucket> is one of: regression, binary, multiclass.

Usage:
    python3 tabarena_download.py [--only regression|binary|multiclass] [--limit N]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd

HOME = Path.home()
REPO = HOME / "TuiML" / "benchmark" / "tabarena_dataset_curation"
META_CSV = REPO / "dataset_creation_scripts" / "metadata" / "tabarena_dataset_metadata.csv"
OUT_ROOT = HOME / "TuiML" / "datasets"


def bucket_for(row) -> str:
    """Map a metadata row to a task bucket folder name."""
    problem = str(row.get("problem_type", "")).lower()
    if "regress" in problem:
        return "regression"
    # classification: split binary vs multiclass on num_classes
    try:
        n = int(row.get("num_classes", 0) or 0)
    except (ValueError, TypeError):
        n = 0
    return "binary" if n <= 2 else "multiclass"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["regression", "binary", "multiclass"],
                    help="restrict to one task bucket")
    ap.add_argument("--limit", type=int, default=0, help="cap number of datasets (0 = all)")
    args = ap.parse_args()

    import openml  # imported here so --help works without it

    if not META_CSV.exists():
        print(f"ERROR: metadata CSV not found at {META_CSV}", file=sys.stderr)
        return 1

    meta = pd.read_csv(META_CSV)
    meta["bucket"] = meta.apply(bucket_for, axis=1)
    if args.only:
        meta = meta[meta["bucket"] == args.only]
    if args.limit:
        meta = meta.head(args.limit)

    counts = meta["bucket"].value_counts().to_dict()
    print(f"Planning to download {len(meta)} datasets: {counts}")
    print("-" * 70)

    ok, failed, skipped = [], [], []
    for i, row in enumerate(meta.itertuples(index=False), 1):
        d = row._asdict()
        name = str(d["dataset_name"])
        ds_id = int(d["dataset_id"])
        bucket = d["bucket"]
        target = d.get("target_feature")
        out_dir = OUT_ROOT / bucket / name
        csv_path = out_dir / f"{name}.csv"

        tag = f"[{i}/{len(meta)}] {bucket}/{name} (oml#{ds_id})"
        if csv_path.exists():
            print(f"  SKIP  {tag} -- already present")
            skipped.append(name)
            continue

        try:
            t0 = time.time()
            ds = openml.datasets.get_dataset(ds_id, download_data=True,
                                             download_qualities=False,
                                             download_features_meta_data=False)
            X, y, _, _ = ds.get_data(target=ds.default_target_attribute)
            frame = X.copy()
            tgt_name = ds.default_target_attribute or (target if isinstance(target, str) else "target")
            if y is not None:
                frame[tgt_name] = y
            out_dir.mkdir(parents=True, exist_ok=True)
            frame.to_csv(csv_path, index=False)

            metadata = {
                "dataset_name": name,
                "openml_dataset_id": ds_id,
                "openml_task_id": (int(d["task_id"]) if pd.notna(d.get("task_id")) else None),
                "problem_type": d.get("problem_type"),
                "bucket": bucket,
                "target_feature": tgt_name,
                "num_instances": int(d.get("num_instances") or frame.shape[0]),
                "num_features": int(d.get("num_features") or (frame.shape[1] - 1)),
                "num_classes": (int(d["num_classes"]) if pd.notna(d.get("num_classes")) else None),
                "data_source": d.get("data_source"),
                "original_data_url": d.get("original_data_url"),
                "licence": d.get("licence"),
                "tabarena_suite": "tabarena-v0.1 (OpenML study 457)",
            }
            (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str))
            print(f"  OK    {tag} -> {frame.shape[0]}x{frame.shape[1]}  ({time.time()-t0:.1f}s)")
            ok.append(name)
        except Exception as e:  # noqa: BLE001 - keep going on any single failure
            print(f"  FAIL  {tag} -- {type(e).__name__}: {str(e)[:120]}")
            failed.append((name, f"{type(e).__name__}: {e}"))

    print("-" * 70)
    print(f"Done. ok={len(ok)} skipped={len(skipped)} failed={len(failed)}")
    if failed:
        print("Failures:")
        for n, err in failed:
            print(f"  - {n}: {err[:140]}")
    return 0 if not failed else 2


if __name__ == "__main__":
    raise SystemExit(main())
