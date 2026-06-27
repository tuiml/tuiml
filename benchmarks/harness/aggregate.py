#!/usr/bin/env python3
"""Aggregate per-experiment result JSON files into a single tidy CSV.

Usage:
    python3 aggregate.py --results results --out summary.csv
"""
import argparse
import json
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="summary.csv")
    args = ap.parse_args()

    rows = []
    for fp in sorted(Path(args.results).glob("*.json")):
        rec = json.loads(fp.read_text())
        metrics = rec.pop("metrics", {}) or {}
        row = {k: v for k, v in rec.items() if not isinstance(v, dict)}
        row.update({f"metric_{k}": v for k, v in metrics.items()})
        rows.append(row)

    if not rows:
        print(f"No result JSON files found in {args.results}/")
        return
    df = pd.DataFrame(rows).sort_values(["bucket", "dataset", "algorithm", "framework"])
    df.to_csv(args.out, index=False)

    n_ok = int((df["status"] == "ok").sum())
    print(f"Aggregated {len(df)} experiments ({n_ok} ok) -> {args.out}")
    print("\nBy framework (ok / total, mean wall_s, mean peak_rss_mb):")
    for fw, g in df.groupby("framework"):
        ok = g[g.status == "ok"]
        print(f"  {fw:10s} {len(ok)}/{len(g)}  "
              f"wall={ok['wall_total_s'].mean():.2f}s  rss={ok['peak_rss_mb'].mean():.0f}MB")


if __name__ == "__main__":
    main()
