#!/usr/bin/env python3
"""Fetch each dataset's *declared* attribute types from OpenML.

Why this exists
---------------
``tabarena_download.py`` writes each dataset as a plain CSV, which throws away
the ARFF/OpenML attribute-type declarations. The harness used to recover them by
inspecting pandas dtypes, which silently misclassifies **integer-coded nominal
attributes as numeric** — they then get standardized and fed to every framework
as continuous features. Weka additionally lost the ability to treat them as
nominal at all.

This script re-fetches the per-attribute metadata (a small JSON call per
dataset, no data download) and writes a ``schema.json`` next to each CSV::

    ~/TuiML/datasets/<bucket>/<name>/schema.json

    {
      "dataset_name": "...",
      "openml_dataset_id": 46905,
      "target": "ResourceApproved",
      "source": "openml-features-api",
      "columns": {"RESOURCE": {"type": "nominal", "n_categories": 7518},
                  "age":      {"type": "numeric"}}
    }

Only the *types* are stored, not the category vocabularies: the harness fits its
encoders on the training split alone, so it must not be handed the full level
set (that would leak test information into the encoding).

Usage:
    python3 fetch_schema.py                     # all datasets under the root
    python3 fetch_schema.py --only regression
    python3 fetch_schema.py --force             # re-fetch even if present
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HOME = Path.home()
OUT_ROOT = HOME / "TuiML" / "datasets"
BUCKETS = ("regression", "binary", "multiclass")

# OpenML data_type values that mean "discrete set of labels".
NOMINAL_TYPES = {"nominal", "string", "categorical"}


def schema_for(ds) -> dict:
    """Build the ``columns`` mapping for one OpenML dataset object.

    Parameters
    ----------
    ds : openml.datasets.OpenMLDataset
        Dataset whose feature metadata has been downloaded.

    Returns
    -------
    columns : dict
        Maps attribute name -> ``{"type": "nominal"|"numeric", ...}``.
    """
    columns = {}
    for feat in ds.features.values():
        is_nominal = str(feat.data_type).lower() in NOMINAL_TYPES
        entry = {"type": "nominal" if is_nominal else "numeric"}
        if is_nominal and feat.nominal_values is not None:
            entry["n_categories"] = len(feat.nominal_values)
        columns[feat.name] = entry
    return columns


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default=str(OUT_ROOT), help="dataset root")
    ap.add_argument("--only", choices=list(BUCKETS), help="restrict to one bucket")
    ap.add_argument("--force", action="store_true", help="re-fetch existing schema.json")
    args = ap.parse_args()

    import openml  # imported here so --help works without it

    root = Path(args.datasets).expanduser()
    buckets = [args.only] if args.only else list(BUCKETS)

    ok, skipped, failed = [], [], []
    for bucket in buckets:
        bdir = root / bucket
        if not bdir.is_dir():
            continue
        for ds_dir in sorted(p for p in bdir.iterdir() if p.is_dir()):
            name = ds_dir.name
            meta_path = ds_dir / "metadata.json"
            schema_path = ds_dir / "schema.json"
            if not meta_path.exists():
                continue
            if schema_path.exists() and not args.force:
                print(f"  SKIP  {bucket}/{name} -- schema.json present")
                skipped.append(name)
                continue

            meta = json.loads(meta_path.read_text())
            ds_id = meta.get("openml_dataset_id")
            try:
                t0 = time.time()
                ds = openml.datasets.get_dataset(
                    ds_id, download_data=False, download_qualities=False,
                    download_features_meta_data=True)
                columns = schema_for(ds)
                target = meta.get("target_feature")
                # The target is described by the same metadata; keep it out of
                # the feature schema so the harness can't treat it as an input.
                columns.pop(target, None)
                schema = {
                    "dataset_name": name,
                    "openml_dataset_id": ds_id,
                    "target": target,
                    "source": "openml-features-api",
                    "columns": columns,
                }
                schema_path.write_text(json.dumps(schema, indent=2))
                n_nom = sum(1 for c in columns.values() if c["type"] == "nominal")
                print(f"  OK    {bucket}/{name} -- {len(columns)} attrs "
                      f"({n_nom} nominal)  ({time.time() - t0:.1f}s)")
                ok.append(name)
            except Exception as e:  # noqa: BLE001 - keep going on any single failure
                print(f"  FAIL  {bucket}/{name} -- {type(e).__name__}: {str(e)[:120]}")
                failed.append((name, f"{type(e).__name__}: {e}"))

    print("-" * 70)
    print(f"Done. ok={len(ok)} skipped={len(skipped)} failed={len(failed)}")
    if failed:
        print("Failures (harness will fall back to dtype inference for these):",
              file=sys.stderr)
        for n, err in failed:
            print(f"  - {n}: {err[:140]}", file=sys.stderr)
    return 0 if not failed else 2


if __name__ == "__main__":
    raise SystemExit(main())
