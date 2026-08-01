"""Per-algorithm experiment history recorded from benchmark runs."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

from ._paths import USER_ALGS_DIR, _VERSION_RE, _read_metadata
from .registration import _versioned_alias_name

_ALIAS_RE = re.compile(r"^(?P<class>[A-Za-z_][A-Za-z0-9_]*)_v(?P<v>\d+_\d+_\d+)$")


def _resolve_user_algorithm(name: str) -> Optional[Dict[str, Any]]:
    """Map an experiment algorithm name back to a user-algorithm directory.

    Accepts both the bare class name (``WeightedTreeBag``: resolves to the
    newest on-disk version) and a pinned alias (``WeightedTreeBag_v1_0_1``).

    Parameters
    ----------
    name : str
        Algorithm name as used in an experiment: bare class name or
        versioned alias.

    Returns
    -------
    match : Optional[Dict[str, Any]]
        Dict with keys ``name`` (bare name), ``version`` (semver string),
        and ``dir`` (Path to the version directory), or None if this name
        doesn't correspond to a user algorithm on disk.
    """
    # Pinned alias first.
    m = _ALIAS_RE.match(name)
    if m:
        base = m.group("class")
        ver = m.group("v").replace("_", ".")
        p = USER_ALGS_DIR / base / ver
        if p.is_dir():
            return {"name": base, "version": ver, "dir": p}

    # Bare class name: return the highest semver present on disk.
    base_dir = USER_ALGS_DIR / name
    if base_dir.is_dir():
        versions = sorted(
            (v for v in base_dir.iterdir() if v.is_dir() and _VERSION_RE.match(v.name)),
            key=lambda p: tuple(int(x) for x in p.name.split(".")),
            reverse=True,
        )
        if versions:
            v = versions[0]
            return {"name": name, "version": v.name, "dir": v}
    return None


def _best_metric(scores_block: Dict[str, Any]) -> Optional[Tuple[str, float]]:
    """Pick a representative metric: prefer accuracy_score, then first available.

    Parameters
    ----------
    scores_block : Dict[str, Any]
        Per-algorithm scores from an experiment result: metric name mapped to
        a dict containing at least a ``mean`` value.

    Returns
    -------
    best : Optional[Tuple[str, float]]
        ``(metric_name, mean_score)``, or None if no usable metric was found.
    """
    if not isinstance(scores_block, dict):
        return None
    if "accuracy_score" in scores_block and isinstance(scores_block["accuracy_score"], dict):
        mean = scores_block["accuracy_score"].get("mean")
        if mean is not None:
            return "accuracy_score", float(mean)
    for metric, block in scores_block.items():
        if isinstance(block, dict) and block.get("mean") is not None:
            return metric, float(block["mean"])
    return None


def record_experiment_runs(experiment_result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Append matching experiment results to user algorithms' ``runs.jsonl`` files.

    Scans a ``tuiml_benchmark`` result for algorithms that resolve to user
    algorithms on disk and logs one entry per (dataset, algorithm) pair.
    Never raises: logging failures are swallowed so the caller's experiment
    result is not affected.

    Parameters
    ----------
    experiment_result : Dict[str, Any]
        Full result dict returned by ``tuiml_benchmark``; must have
        ``status='success'`` and a ``results`` mapping of dataset name to
        per-algorithm scores.

    Returns
    -------
    appended : List[Dict[str, Any]]
        One dict per logged run with keys ``name``, ``version``, ``dataset``,
        and ``path`` (the ``runs.jsonl`` file). Empty if nothing matched.
    """
    import datetime as _dt, json as _json

    if not isinstance(experiment_result, dict) or experiment_result.get("status") != "success":
        return []
    results = experiment_result.get("results")
    if not isinstance(results, dict):
        return []

    appended: List[Dict[str, Any]] = []
    ts = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
    cv = experiment_result.get("cv") or experiment_result.get("n_folds")

    for dataset_name, algo_scores in results.items():
        if not isinstance(algo_scores, dict):
            continue
        for algo_name, scores_block in algo_scores.items():
            match = _resolve_user_algorithm(algo_name)
            if not match:
                continue
            entry: Dict[str, Any] = {
                "timestamp": ts,
                "dataset": dataset_name,
                "cv": cv,
                "algorithm_name": algo_name,
                "resolved_version": match["version"],
                "metrics": scores_block,
            }
            best = _best_metric(scores_block)
            if best:
                entry["primary_metric"] = best[0]
                entry["primary_score"] = best[1]
            log_path = match["dir"] / "runs.jsonl"
            try:
                with log_path.open("a", encoding="utf-8") as fh:
                    fh.write(_json.dumps(entry) + "\n")
                appended.append({"name": match["name"], "version": match["version"],
                                 "dataset": dataset_name, "path": str(log_path)})
            except Exception:
                pass
    return appended


def research_log(name: Optional[str] = None) -> Dict[str, Any]:
    """Return the aggregated research view for one (or all) user algorithms.

    Combines ``metadata.json`` (static: class, kind, version, source hash)
    with ``runs.jsonl`` (dynamic: every recorded experiment run) and computes:
    best primary score per version, run count per version, and the most recent
    timestamp. Agents use this as the equivalent of the landing-page "research
    log" panel.

    Parameters
    ----------
    name : str, optional
        Restrict the view to one user algorithm. When None, all algorithms
        under ``USER_ALGS_DIR`` are included.

    Returns
    -------
    result : Dict[str, Any]
        On success: keys ``status``, ``count``, ``root``, and ``algorithms``,
        a list of dicts with ``name``, ``overall_best_version``, and
        ``versions`` (per-version dicts: ``version``, ``class_name``,
        ``kind``, ``description``, ``source_hash``, ``pinned_alias``,
        ``run_count``, ``best_score``, ``best_on_dataset``, ``best_metric``,
        ``last_run``, ``path``). On failure: keys ``status``, ``error_type``,
        ``error``.
    """
    import json as _json

    if not USER_ALGS_DIR.exists():
        return {"status": "success", "algorithms": [], "root": str(USER_ALGS_DIR)}

    roots: List[Path]
    if name:
        base = USER_ALGS_DIR / name
        if not base.is_dir():
            return {"status": "error", "error_type": "NotFound",
                    "error": f"no user algorithm named {name!r}"}
        roots = [base]
    else:
        roots = [p for p in USER_ALGS_DIR.iterdir() if p.is_dir()]

    algorithms: List[Dict[str, Any]] = []
    for name_dir in sorted(roots):
        versions: List[Dict[str, Any]] = []
        for ver_dir in sorted(
            (p for p in name_dir.iterdir() if p.is_dir() and _VERSION_RE.match(p.name)),
            key=lambda p: tuple(int(x) for x in p.name.split(".")),
        ):
            meta = _read_metadata(ver_dir)
            runs_path = ver_dir / "runs.jsonl"
            runs: List[Dict[str, Any]] = []
            if runs_path.exists():
                try:
                    for line in runs_path.read_text(encoding="utf-8").splitlines():
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            runs.append(_json.loads(line))
                        except Exception:
                            continue
                except Exception:
                    runs = []

            best_score: Optional[float] = None
            best_dataset: Optional[str] = None
            best_metric: Optional[str] = None
            for r in runs:
                score = r.get("primary_score")
                if score is None:
                    continue
                if best_score is None or score > best_score:
                    best_score = float(score)
                    best_dataset = r.get("dataset")
                    best_metric = r.get("primary_metric")

            last_run_ts = max((r.get("timestamp", "") for r in runs), default=None) or None

            versions.append({
                "version": ver_dir.name,
                "class_name": meta.get("class_name", name_dir.name),
                "kind": meta.get("kind"),
                "description": meta.get("description"),
                "source_hash": meta.get("source_hash"),
                "pinned_alias": _versioned_alias_name(
                    meta.get("class_name", name_dir.name), ver_dir.name),
                "run_count": len(runs),
                "best_score": best_score,
                "best_on_dataset": best_dataset,
                "best_metric": best_metric,
                "last_run": last_run_ts,
                "path": str(ver_dir),
            })

        # Which version is the current best across all recorded runs?
        scored = [v for v in versions if v["best_score"] is not None]
        overall_best = max(scored, key=lambda v: v["best_score"])["version"] if scored else None

        algorithms.append({
            "name": name_dir.name,
            "versions": versions,
            "overall_best_version": overall_best,
        })

    return {
        "status": "success",
        "count": len(algorithms),
        "algorithms": algorithms,
        "root": str(USER_ALGS_DIR),
    }
