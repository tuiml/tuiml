# TuiML Benchmarks — TuiML vs scikit-learn vs Weka

A reproducible benchmark comparing **TuiML**, **scikit-learn**, and **Weka**
(via `python-weka-wrapper3`) on **51 real-world tabular datasets** from the
[TabArena v0.1](https://github.com/autogluon/tabarena) suite — measuring
**accuracy / R²**, **training time**, and **peak memory** for every algorithm
that exists in all three libraries.

The headline results live in [`summary.csv`](summary.csv) (one row per
`framework × algorithm × dataset`, 1,032 rows) and are rendered on the website
benchmarks page.

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
- `SMO` defaults to a linear PolyKernel and normalizes internally; `SVC`
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

- **Split:** stratified 80/20 holdout, fixed seed (42) — *identical* across all
  three frameworks.
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
│   └── aggregate.py             results/*.json → summary.csv
├── tools/                   # step 3 — analyze / report
│   ├── present.py               per-framework summary tables
│   ├── present_bydataset.py     per-dataset "who wins" tables
│   ├── reconcile.py             list any missing framework × algo × dataset cells
│   ├── stub_missing.py          record still-missing cells as status=timeout
│   ├── sizes.py                 dataset size table
│   ├── timing.py                fit-time-vs-rows analysis
│   └── gen_web_json.py          summary.csv → website JSON (tabarena_results.json)
└── summary.csv              # result snapshot (1,032 rows)
```

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
MAX_JOBS=32 ./run_all.sh
```

`run_all.sh` generates one job per `framework × algorithm × dataset`, then runs
them with `xargs -P` (each experiment is its own process). Per-experiment results
are written as `results/<framework>__<algo>__<dataset>.json`; per-job logs go to
`logs/`.

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
# from harness/ (where results/ was created)
python3 aggregate.py --results results --out ../summary.csv

# completeness check; record genuinely-too-slow cells as status=timeout
python3 ../tools/reconcile.py
python3 ../tools/stub_missing.py

# tables
python3 ../tools/present.py            # per-framework means
python3 ../tools/present_bydataset.py  # who-wins-per-dataset
python3 ../tools/timing.py             # fit time vs row count

# regenerate the website JSON from summary.csv
python3 ../tools/gen_web_json.py
```

---

## Results snapshot

From the reference run (`summary.csv`, 1,032 experiments, **1,023 ok / 9 timeouts**):

| | Classification (mean acc) | Regression (mean R²) | Notes |
|---|---|---|---|
| **Weka** | **0.843** | **0.649** | best on average, but slowest; 9 SVM/MLP jobs timed out on the biggest data |
| scikit-learn | 0.819 | 0.406 | lightest memory (~250–280 MB) |
| TuiML | 0.803 | 0.478 | fastest on classification |

- On the workhorse **Random Forest**, all three tie (~0.875 accuracy / ~0.78 R²) —
  a good sanity check that the harness is fair.
- The **9 timeouts** are all Weka `SMO`/`MultilayerPerceptron` on the
  largest/widest datasets (e.g. `customer_satisfaction` 130k rows,
  `hiva_agnostic` 1,618 features) — single-threaded `O(n²)` compute, a genuine
  scaling signal rather than a bug.

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
