#!/usr/bin/env python3
"""Parallel 10-fold CV launcher: fast tier first, then slow tier.

Splits jobs into two tiers so Weka SVM/MLP (30-60 min per fold) cannot clog
all worker slots while thousands of sub-second sklearn/tuiml jobs wait.

Tier 1 (fast):  sklearn + tuiml + weka-except-svm-mlp  -- drains in ~15 min
Tier 2 (slow):  weka svm / svm_reg / mlp / mlp_reg     -- runs after tier 1

Usage:
    python3 launch_cv.py [--max-jobs 100] [--timeout 3600] [jobs_cv10.txt]
"""
import argparse, os, subprocess, sys, time
from pathlib import Path

HARN = Path(__file__).resolve().parent

SLOW_ALGOS = {"svm", "svm_reg", "mlp", "mlp_reg"}


def tag_for(cmd: str) -> str:
    parts = []
    for tok in cmd.split():
        if tok in ("--algo", "--dataset", "--config", "--seed", "--fold"):
            parts.append("__")
        elif parts and parts[-1] == "__":
            parts.append(tok.rsplit("/", 1)[-1].replace(".csv", ""))
    tag = "".join(parts).lstrip("_").replace(" ", "_")[:200]
    return tag or f"job_{int(time.time())}"


def run_tier(jobs, label, max_jobs, timeout, out_dir, logs_dir):
    total = len(jobs)
    print(f"\n=== TIER {label}: {total} jobs ===", flush=True)
    t_start = time.time()

    env = dict(os.environ)
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS"):
        env[v] = "1"

    running = {}    # pid -> (tag, start_time)
    done = ok = err = 0
    job_iter = iter(jobs)
    exhausted = False

    while not exhausted or running:
        while len(running) < max_jobs:
            try:
                cmd = next(job_iter)
            except StopIteration:
                exhausted = True
                break
            tag = tag_for(cmd)
            log_path = str(logs_dir / f"{tag}.log")
            wrapped = f"timeout {timeout} {cmd}"
            p = subprocess.Popen(
                wrapped, shell=True, env=env,
                stdout=open(log_path, "w"), stderr=subprocess.STDOUT,
                preexec_fn=os.setsid,
            )
            running[p.pid] = (tag, time.time())

        if not running:
            break

        try:
            pid, status = os.waitpid(-1, 0)
        except ChildProcessError:
            break
        if pid not in running:
            continue
        tag, t0 = running.pop(pid)
        elapsed = time.time() - t0
        code = os.WEXITSTATUS(status) if os.WIFEXITED(status) else -1
        done += 1
        if code == 0:
            ok += 1
        else:
            err += 1
            reason = "TIMEOUT" if code == 124 else f"exit={code}"
            print(f"[{done:5d}/{total}] {reason:8s}  {elapsed:6.0f}s  {tag}",
                  flush=True)
        # Progress heartbeat every 2 min or every 200 completions.
        if done % 200 == 0 or (done > 0 and elapsed > 120 and done % 50 == 0):
            elapsed_total = time.time() - t_start
            rate = done / max(elapsed_total, 1)
            eta = (total - done) / max(rate, 0.001)
            print(f"  ... {done}/{total} ok={ok} err={err}  "
                  f"rate={rate:.1f}/min  eta={eta/60:.0f}min  "
                  f"live={len(running)}", flush=True)

    elapsed_total = time.time() - t_start
    print(f"TIER {label} DONE  ok={ok} err={err}  "
          f"{elapsed_total/60:.1f}min  {time.strftime('%H:%M:%S')}", flush=True)
    return ok, err


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs_file", nargs="?", default="jobs_cv10.txt")
    ap.add_argument("--max-jobs", type=int, default=100)
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--out", default="results_cv10")
    ap.add_argument("--logs", default="logs_cv10")
    args = ap.parse_args()

    all_jobs = [ln.strip() for ln in open(HARN / args.jobs_file) if ln.strip()]
    fast, slow = [], []
    for cmd in all_jobs:
        # Classify by framework + algo: weka svm/mlp -> slow, everything else -> fast
        if "--algo" in cmd and "bench_weka.py" in cmd:
            algo = cmd.split("--algo ")[1].split()[0]
            if algo in SLOW_ALGOS:
                slow.append(cmd)
                continue
        fast.append(cmd)

    out = HARN / args.out
    logs = HARN / args.logs
    out.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)

    total_ok = total_err = 0

    # Tier 1: fast everything
    ok, err = run_tier(fast, "1-fast", args.max_jobs, args.timeout, out, logs)
    total_ok += ok; total_err += err

    # Tier 2: slow Weka SVM/MLP
    if slow:
        ok, err = run_tier(slow, "2-slow", args.max_jobs, args.timeout, out, logs)
        total_ok += ok; total_err += err

    n_results = len(list(out.glob("*.json")))
    print(f"\nALL DONE  results={n_results}  ok={total_ok}  err={total_err}  "
          f"{time.strftime('%H:%M:%S')}", flush=True)


if __name__ == "__main__":
    main()
