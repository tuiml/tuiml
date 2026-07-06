#!/usr/bin/env bash
# Parallel benchmark launcher — pure bash style.
#
# Each experiment is its own independent OS process (no Python multiprocessing
# or threadpool). We generate a flat job list and run it through `xargs -P`,
# which keeps up to MAX_JOBS processes alive at once, launching the next as each
# finishes. Per-job wall-clock is bounded by `timeout`.
#
# Usage:
#   ./run_all.sh                         # all frameworks, all datasets
#   MAX_JOBS=16 ./run_all.sh             # cap parallelism (default: nproc-2)
#   ./run_all.sh --only-bucket regression
#   FRAMEWORKS="tuiml sklearn" ./run_all.sh
#
set -uo pipefail
cd "$(dirname "$0")"

# Pin numerical libs to one thread per process. We parallelize at the process
# level (xargs -P), so per-process threading only oversubscribes the CPUs; on
# high-core machines that also overruns OpenBLAS's NUM_THREADS limit and
# segfaults sklearn KNeighbors. One thread each = stable + fairer timing.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

DATASETS="${DATASETS:-$HOME/TuiML/datasets}"
OUT="${OUT:-results}"
JOBS_FILE="${JOBS_FILE:-jobs.txt}"
MAX_JOBS="${MAX_JOBS:-$(( $(nproc) - 2 ))}"
[ "$MAX_JOBS" -lt 1 ] && MAX_JOBS=1
PER_JOB_TIMEOUT="${PER_JOB_TIMEOUT:-1800}"   # seconds per experiment
FRAMEWORKS="${FRAMEWORKS:-sklearn tuiml weka}"
PYTHON="${PYTHON:-python3}"

mkdir -p "$OUT" logs

# 1. Generate the job list (one self-contained command per line).
$PYTHON gen_jobs.py --datasets "$DATASETS" --out "$OUT" \
    --frameworks $FRAMEWORKS --jobs-file "$JOBS_FILE" --python "$PYTHON" "$@"

echo "Launching with MAX_JOBS=$MAX_JOBS, per-job timeout=${PER_JOB_TIMEOUT}s"
echo "Logs -> logs/  |  Results -> $OUT/"

# 2. Fan out: each line runs in its own process; xargs maintains MAX_JOBS in flight.
#    A per-line log captures stdout/stderr; `timeout` caps runaway experiments.
< "$JOBS_FILE" xargs -P "$MAX_JOBS" -I {} \
    bash -c 'cmd="{}"; tag=$(echo "$cmd" | grep -oE -- "--algo [^ ]+|--dataset [^ ]+|--seed [^ ]+|--fold [^ ]+" | tr -d "\n" | tr "/ " "__"); \
             timeout '"$PER_JOB_TIMEOUT"' $cmd > "logs/${tag}.log" 2>&1; \
             echo "done: $cmd (exit $?)"'

echo "All jobs finished. Aggregate with: $PYTHON aggregate.py --results $OUT"
