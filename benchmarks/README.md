# TuiML Benchmarks — TuiML vs scikit-learn vs Weka

A reproducible benchmark comparing **TuiML**, **scikit-learn**, and **Weka**
(via `python-weka-wrapper3`) on **51 real-world tabular datasets** from the
[TabArena v0.1](https://github.com/autogluon/tabarena) suite — measuring
**accuracy / R²**, **training time**, and **peak memory** for every algorithm
that exists in all three libraries.

The headline results live in [`summary_cv10.csv`](summary_cv10.csv) (one row per
`configuration × framework × algorithm × dataset × fold`, 20,640 rows) and are
rendered on the website benchmarks page. A score is published only when all ten
folds for that cell completed successfully, and every published aggregate is
restricted to cells where **all three** frameworks finished — otherwise a
framework that crashed on the hard datasets would score better than one that
completed them.

---

## What we benchmark

### Datasets — 51 from TabArena v0.1 (OpenML study 457)

- **38 classification** (30 binary, 8 multiclass) + **13 regression**
- Real-world tabular data, ~700 to ~150,000 rows, up to ~1,800 features
- Curated and standardized by the
  [`tabarena_dataset_curation`](https://github.com/TabArena/tabarena_dataset_curation)
  project and published on [OpenML](https://www.openml.org/search?type=study&study_type=task&id=457)

### Algorithms — the intersection of all three libraries

Only algorithms available in **TuiML, scikit-learn, and Weka** are compared, so
every cell is apples-to-apples.

| Task | Algorithms |
|------|------------|
| Classification | Random Forest, SVM, Neural Network (MLP), Logistic Regression, k-NN, Decision Tree, Naive Bayes |
| Regression | Random Forest, SVR, Neural Network (MLP), Linear Regression, k-NN, Decision Tree |

The exact per-framework class/option mapping lives in
[`harness/algorithms.py`](harness/algorithms.py).

### Two configurations

Every experiment runs under one of two configurations, selected with `--config`
and recorded in each result row:

| Config | Meaning | Use it for |
|--------|---------|-----------|
| `matched` | Hyperparameters aligned across the three libraries so the same model is being fitted wherever that is possible | **runtime and accuracy claims** |
| `defaults` | Each library exactly as it ships | "what you get out of the box" |

This split exists because the libraries disagree about defaults in ways that
change both cost and quality, so comparing defaults is not a like-for-like
comparison. Concrete examples the `matched` config removes:

- Weka's `LinearRegression` runs **M5 attribute selection and collinearity
  elimination** by default; scikit-learn's solves one least-squares problem.
- Weka's `RandomForest` samples `log2(p)+1` attributes per split, scikit-learn
  `sqrt(p)` for classification and **all `p`** for regression.
- `SVC` uses a linear kernel by default and normalizes internally; sklearn `SVC`
  defaults to RBF. Matched mode puts both on RBF with the *same* explicit gamma.
- `Logistic` defaults to ridge `1e-8` (effectively unregularized) while
  scikit-learn defaults to `C=1.0`, i.e. ridge `0.5`.
- `IBk` min-max normalizes inside its distance function on top of the
  standardization the harness already applied.
- Weka's `MultilayerPerceptron` normalizes the numeric class internally and the
  others do not — the documented reason scikit-learn's `MLPRegressor` scored
  negative R² in the previous run.

Where the underlying algorithms are simply not the same (C4.5 vs CART, online
backprop vs mini-batch SGD), `matched` aligns what it can and the residual
mismatch is recorded in the `note` field of every result row rather than left
implicit. The full mapping, with rationale, is in
[`harness/algorithms.py`](harness/algorithms.py).

### Protocol

- **Split:** shuffled 10-fold cross-validation with fixed seed 42.
  Classification folds are stratified; regression uses ordinary shuffled
  `KFold`. Fold assignments are *identical* across all three frameworks.
- **Attribute types** come from `schema.json` (written by
  [`fetch_schema.py`](fetch_schema.py) from the OpenML attribute declarations),
  **not** from pandas dtypes. Inferring from dtype silently treats
  integer-coded nominal attributes as continuous — on this suite that
  misclassified whole datasets (`hiva_agnostic` is 1617/1617 nominal, `splice`
  60/60, `MIC` 94/111). Rows prepared without a schema are flagged
  `schema_source="dtype-fallback"`.
- **Preprocessing:** numeric → median impute + standardize; nominal →
  most-frequent impute + integer coding, all fit on the **training split only**.
  High-cardinality attributes are folded to the 100 most frequent training
  levels plus one "other" level (`BENCH_MAX_LEVELS`); without a cap, one-hot
  encoding a 7,500-level attribute materializes a ~2 GB dense matrix. The fold
  is applied to the codes, so all three frameworks see the same information.
- **Representation:** each library is then given the encoding it is designed
  for — one-hot for scikit-learn and TuiML, **genuinely nominal attributes** for
  Weka, so its tree and instance-based learners use their native nominal
  handling. Weka's function-based learners (SMO, Logistic, MLP) apply their own
  internal `NominalToBinary`, which reproduces the one-hot the others receive.
  Materialization happens outside the timed region.
- **Regression targets** are standardized on the training split in `matched`
  mode and the predictions inverted before scoring, so Weka's internal class
  normalization is neither a hidden advantage nor something the others have to
  do without. Metrics stay in the original units.
- **Inference** is measured as a **single batch call** for all three frameworks:
  `distributionsForInstances` for Weka, `predict` for the others. The previous
  per-instance `classifyInstance` loop crossed the Python/JVM boundary once per
  test row and overstated Weka's inference time by ~4.5× on cheap models
  (identical predictions; k-NN was unaffected, since real compute dominates
  there).
- **Metrics:** classification → accuracy, F1-macro, balanced accuracy, precision,
  recall; regression → RMSE, MAE, R²; plus `fit_s`, `predict_s`, `wall_total_s`,
  `cpu_total_s`, `peak_rss_mb`, and the resolved `options` and `lib_version` for
  every run.
- **Isolation:** each experiment runs in its **own OS process** (no Python
  multiprocessing/threadpool), parallelized with `xargs -P`. Each process is
  pinned to a **single thread** (`OMP_NUM_THREADS=1`, and `-XX:ActiveProcessorCount=1`
  for Weka's JVM) to avoid oversubscription on many-core machines and to keep
  timing fair.

---

## Folder layout

```
benchmarks/
├── tabarena_download.py     # step 1 — download the 51 datasets from OpenML
├── fetch_schema.py          # step 1b — write schema.json (OpenML attribute types)
├── harness/                 # step 2 — run the benchmark
│   ├── run_all.sh               entry point (generates jobs, runs them in parallel)
│   ├── gen_jobs.py              builds the framework × algo × dataset job list
│   ├── common.py                shared data prep, metrics, resource capture
│   ├── algorithms.py            matching-algorithm registry
│   ├── bench_sklearn.py         single-experiment runner (scikit-learn)
│   ├── bench_tuiml.py           single-experiment runner (TuiML)
│   ├── bench_weka.py            single-experiment runner (Weka / JVM)
│   └── aggregate.py             results_cv10/*.json → summary_cv10.csv
├── tools/                   # step 3 — analyze / report
│   ├── present.py               per-framework summary tables
│   ├── present_bydataset.py     per-dataset "who wins" tables
│   ├── reconcile.py             list any missing framework × algo × dataset cells
│   ├── stub_missing.py          record scheduled folds lacking a result
│   ├── sizes.py                 dataset size table
│   ├── timing.py                fit-time-vs-rows analysis
│   ├── gen_web_json.py          summary_cv10.csv → website JSON exports
│   ├── headline.py              recompute every published number from the CSV
│   └── figure.py                regenerate assets/benchmark_summary_*.svg
├── summary_cv10.csv         # current run: two configs × 10-fold (20,640 rows)
└── summary_cv10_2026-07.csv # superseded July run, kept for provenance
```

### Which snapshot is authoritative

`summary_cv10.csv` (2026-08-14) is the one to use. It covers the same 51
datasets and 13 algorithms as the earlier run, but adds the `matched`
configuration arm and thirteen provenance columns — `options`, `lib_version`,
`note`, `diverged` and others — that record how each cell was actually
configured.

`summary_cv10_2026-07.csv` is the superseded 2026-07-06 run. It is kept only
because the figures published before 2026-08 were computed from it; nothing new
should cite it. Both runs agree on the headline: under library defaults TuiML
leads on accuracy. They are not interchangeable, though — the older file has no
`config` column, so pooling the two silently mixes the default and matched arms
and produces a number that matches neither.

### Reproducing the published numbers

```bash
uv run python tools/headline.py   # every figure quoted in README.md / the site
uv run python tools/figure.py     # regenerate the four-panel SVG
```

`headline.py` prints both configuration arms. The published chart reports
**defaults** — each library as it ships, which is what a user gets on install.
The **matched** arm forces hyperparameters to agree and is the one to read when
asking whether an *implementation* is correct: it is where TuiML's logistic
regression (−2.6 pt) and MLP (−1.7 pt) deficits against scikit-learn show up,
while SVM, naive Bayes and k-NN match to the decimal and Random Forest leads.

---

## Prerequisites

```bash
# Core
python3 -m pip install numpy pandas scikit-learn openml psutil

# TuiML (requires a C++ compiler; builds native extensions)
python3 -m pip install tuiml

# Weka backend (requires a JDK — Java 11+ / 21 recommended)
python3 -m pip install python-weka-wrapper3
```

On a many-core / large-RAM machine (the reference run used a 128-core, 1 TB box),
`pip` on an externally-managed Python may need `--user --break-system-packages`.

**Versions used for the reference run:** Python 3.12 · TuiML 0.1.5 ·
scikit-learn 1.8.0 · python-weka-wrapper3 0.3.2 (Java 21).

---

## How to run

### Step 1 — Get the datasets

`tabarena_download.py` reads the TabArena curation metadata (which carries each
dataset's OpenML id + task type) and downloads all 51 datasets into a
task-bucketed tree.

```bash
# It expects the curation repo's metadata here:
#   ~/TuiML/benchmark/tabarena_dataset_curation/
git clone https://github.com/TabArena/tabarena_dataset_curation.git \
    ~/TuiML/benchmark/tabarena_dataset_curation

python3 tabarena_download.py                 # all 51 datasets
# python3 tabarena_download.py --only regression   # one bucket
# python3 tabarena_download.py --limit 5           # quick smoke test
```

Result tree (default root `~/TuiML/datasets`):

```
~/TuiML/datasets/
├── regression/<name>/<name>.csv  (+ metadata.json)
├── binary/<name>/...
└── multiclass/<name>/...
```

> Paths (`~/TuiML/...`) are defined at the top of `tabarena_download.py` — edit
> them if you keep data elsewhere, and pass the same root to the harness via the
> `DATASETS` env var below.

### Step 1b — Fetch the attribute schemas

The CSV export loses the ARFF/OpenML attribute-type declarations, so recover
them before running the harness (one small metadata call per dataset, no data
download):

```bash
python3 fetch_schema.py            # writes schema.json next to each CSV
```

Without this the harness falls back to pandas dtype inference and flags every
affected row `schema_source="dtype-fallback"`.

### Step 2 — Run the benchmark

```bash
cd harness
CONFIGS="matched defaults" OUT=results_cv10 JOBS_FILE=jobs_cv10.txt \
    MAX_JOBS=32 ./run_all.sh --folds 10
```

`run_all.sh` generates one job per
`configuration × framework × algorithm × dataset × fold`, then runs them with
`xargs -P` (each experiment is its own process). Per-fold results are written to
`results_cv10/`; per-job logs go to `logs/`.

Environment knobs (all optional):

| Var | Default | Meaning |
|-----|---------|---------|
| `MAX_JOBS` | `nproc - 2` | parallel processes |
| `PER_JOB_TIMEOUT` | `1800` | seconds before a single experiment is killed |
| `FRAMEWORKS` | `sklearn tuiml weka` | subset to run |
| `CONFIGS` | `matched` | `matched`, `defaults`, or both |
| `DATASETS` | `~/TuiML/datasets` | dataset root |
| `OUT` | `results` | output dir for per-experiment JSON |
| `BENCH_MAX_LEVELS` | `100` | levels kept per high-cardinality nominal attribute |

Examples:

```bash
# only regression, only two frameworks
FRAMEWORKS="tuiml sklearn" ./run_all.sh --only-bucket regression

# give the slow Weka SVM/MLP jobs more time
PER_JOB_TIMEOUT=3600 MAX_JOBS=16 ./run_all.sh
```

### Step 3 — Aggregate, reconcile, report

```bash
# from harness/ (where results_cv10/ was created)
uv run python ../tools/stub_missing.py \
    --jobs-file jobs_cv10.txt --results results_cv10
uv run python aggregate.py --results results_cv10 --out ../summary_cv10.csv

# regenerate explicit matched/default exports and the canonical matched export
uv run python ../tools/gen_web_json.py
uv run python ../tools/gen_web_json.py --config matched
```

---

## Results snapshot

The matched export contains 1,013 complete ten-fold cells, 2 partial cells,
16 cells with no successful folds, and 1 numerically diverged cell. The defaults
export contains 1,011 complete cells, 2 partial cells, and 19 cells with no
successful folds. Partial and failed cells are not averaged into a displayed
score.

---

## Gotchas learned the hard way

- **Thread oversubscription on many-core machines.** Without pinning threads,
  scikit-learn's k-NN can *segfault* (OpenBLAS exceeds its compiled
  `NUM_THREADS` ceiling) and Weka spawns ~60+ JVM threads per process. The
  harness pins every process to one thread (`OMP_NUM_THREADS=1`,
  `-XX:ActiveProcessorCount=1`); since we parallelize at the process level this
  is also faster and fairer.
- **TuiML's k-NN parameter is `k`, not `n_neighbors`** — reflected in
  `algorithms.py`.
- **MLP regression target scaling.** TuiML standardizes the target internally
  (good); scikit-learn's `MLPRegressor` does not, so it can score negative R² on
  unscaled targets in this protocol.
- **Memory is not directly comparable for Weka** — `peak_rss_mb` includes the
  in-process JVM baseline.
