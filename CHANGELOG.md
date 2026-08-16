# Changelog

All notable changes to TuiML will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **New `tuiml.uncertainty` package — conformal prediction and probability
  calibration.** TuiML could rank and predict, but it could not say how sure it
  was, and nothing in the library measured whether a probability meant
  anything. Neither scikit-learn, Weka nor CapyMOA offers conformal
  prediction, so this is native:

  - `SplitConformalClassifier` / `SplitConformalRegressor` — prediction sets
    and intervals with a **distribution-free, finite-sample** guarantee of at
    least `1 - alpha` coverage. LAC and margin nonconformity scores; optional
    difficulty normalisation gives locally adaptive interval widths.
  - `CVPlusRegressor` / `JackknifePlusRegressor` — cross-fitting instead of a
    held-out calibration split, so all the data trains *and* all of it
    calibrates, at the cost of `k` model fits.
  - `APSConformalClassifier` / `RAPSConformalClassifier` — adaptive sets that
    grow on ambiguous inputs. Measured on a noisy 4-class problem: APS lifts
    worst-class coverage from 0.866 to 0.893 for a set size of 3.08 vs 2.64.
  - `MondrianConformalClassifier` — a separate threshold per class or group,
    giving **conditional** rather than merely marginal coverage. On an
    imbalanced problem where plain split conformal covered the rare class only
    3.6% of the time, Mondrian covers it fully.
  - `ConformalizedQuantileRegressor` — heteroscedastic intervals from a pair of
    quantile models; width correlates 0.97 with the input on funnel-shaped
    noise, where split conformal's is constant by construction.
  - `VennAbersCalibrator` — probability *intervals* that report their own
    calibration uncertainty; the width shrinks as the calibration set grows.
  - `PlattCalibrator`, `IsotonicCalibrator`, `TemperatureScaler`,
    `VectorScaler` — post-processors mapping raw scores to probabilities you
    can act on. Temperature scaling provably preserves accuracy.
  - `coverage_score`, `average_set_size`, `interval_width`, `brier_score`,
    `expected_calibration_error`, `maximum_calibration_error`,
    `reliability_curve` — the metrics that verify the above.

  These wrap an already-fitted model rather than being algorithms, so they are
  not registered in the algorithm hub.
- **New `tuiml.explain` package — answering why a model predicted what it did.**
  Nothing in scikit-learn, Weka or CapyMOA covers the local-attribution half
  of this, and the global half only partially.

  - `TreeExplainer` — **exact** Shapley values for TuiML trees and forests, in
    time polynomial in depth rather than exponential in features. Verified
    against a brute-force enumeration of all 2^F subsets, agreeing to 4e-16,
    and against the efficiency property (attributions plus base value
    reconstruct the prediction) for single trees, 30-tree forests and
    per-class on multiclass problems.
  - `permutation_importance`, `drop_column_importance` — model-agnostic global
    importance. The docstrings are explicit that these answer *different*
    questions on correlated features: permutation says neither of a duplicated
    pair matters *given the other*; drop-column refits and so says the column
    is genuinely droppable.
  - `partial_dependence`, `individual_conditional_expectation`,
    `accumulated_local_effects` — how a feature moves the prediction. ALE costs
    two prediction passes regardless of resolution, against one per grid point
    for partial dependence, and never evaluates the model on feature
    combinations the data does not contain.
  - `lime_explain` — a local linear surrogate around one sample. Works on any
    model, including ones TreeExplainer cannot touch; the docstring says to
    check the returned `local_r2`, since the coefficients describe the
    surrogate and say nothing if it fits badly.
  - `counterfactual` — the smallest change that flips a prediction, anchored
    on a real background sample so the answer stays feasible rather than
    inventing an impossible row. Returns the per-feature delta, which is the
    form an explanation has to take when someone is entitled to act on it.
  - `surrogate_tree` — a shallow, readable decision tree trained on the
    model's own predictions, with `fidelity` reporting how well it reproduces
    them. The surrogate is a genuine TuiML tree, so TreeExplainer and the
    dependence tools work on it.
  - `friedman_h_statistic` — interaction strength between two features. Uses
    centred partial dependences; without centring, a flat surface reports a
    spurious interaction of exactly 1.0, which a test pins.
  - `Explanation` — the shared result type, carrying feature names, method and
    base value so numbers never travel bare.

  These wrap a fitted model rather than being algorithms, so none is
  registered in the hub.
- **New shared C++ kernel `tuiml._cpp_ext.shapley`.** `tree_shap` implements
  the path-dependent TreeSHAP recursion over the flattened tree layout TuiML's
  trees already produce, parallel over samples.
- **Three new native algorithm families** — glassbox models, survival
  analysis, and uplift/causal estimation — built in parallel and now passing
  the library's full algorithm-contract suite.

  **`tuiml.algorithms.glassbox/`** — interpretable-by-design models:
  `ExplainableBoostingClassifier` / `ExplainableBoostingRegressor` (additive
  GA1M shape functions with a readable per-feature bin→score map) and
  `RuleFitClassifier` / `RuleFitRegressor` (sparse linear model over
  tree-extracted rules plus the raw features, exposing human-readable rules).

  **`tuiml.algorithms.survival/`** — a new task type with a new
  `Survival` base class, `@survival` decorator and `ComponentType.SURVIVAL`:
  `KaplanMeierEstimator`, `NelsonAalenEstimator`, `CoxPHSurvival` (partial
  likelihood via Newton–Raphson with l2 penalty and Breslow baseline),
  `RandomSurvivalForest`, plus hand-rolled `concordance_index`,
  `integrated_brier_score` and `logrank_test`.

  **`tuiml.algorithms.causal/`** — a new task type with a new `UpliftModel`
  base class, `@uplift` decorator and `ComponentType.UPLIFT`: `SLearner`,
  `TLearner`, `XLearner` (with cross-group residual models) and
  `UpliftTreeClassifier` (greedy uplift-gain splitting), plus `qini_curve`,
  `auuc` and `uplift_at_k`.

  All three are verified against hand-computable references: Kaplan-Meier
  matches the product-limit formula exactly, CoxPH coefficients match a
  hand-solved MLE, concordance of perfect/reversed rankings is 1.0/0.0, and
  the meta-learners' uplift correlates 0.95-0.99 with a known ground-truth
  heterogeneous effect.

  **The contract suite now understands the new task types.** `tests/contract`
  previously assumed every algorithm takes `fit(X, y)`; it now dispatches
  `fit(X, time, event)`, `fit(X, treatment, y)` and the two-argument
  `fit(time, event)` of the marginal estimators, and generates the matching
  data. This surfaced and fixed a set of real gaps: `multi_class` was a typo
  for `multiclass`, several hyphenated capability strings were normalised to
  the underscore vocabulary, five `get_parameter_schema` methods omitted an
  `estimator`/`detectors`/`components` parameter, three timeseries
  classifiers raised `AttributeError: NoneType` instead of "not fitted" before
  fitting, and `LSCPDetector` was non-deterministic with `random_state=None`.
- **Multi-fidelity hyperparameter search** in `tuiml.evaluation.tuning`.
  `SuccessiveHalvingSearchCV` runs a large candidate pool on a small slice of
  the data, discards the worst fraction and repeats with more, so most
  candidates die cheaply. `HyperbandSearchCV` runs several such schedules at
  different aggression levels, removing the need to guess one. Either can
  scale the training subsample or a named estimator parameter — growing a
  forest's `n_estimators` from 1 to 27 across rounds costs no statistical
  power at all, unlike subsampling.

  These allocate *budget*; the existing grid, random and Bayesian searchers
  choose *which points to try*. They are complementary, so no TPE sampler was
  added — `BayesianSearchCV` already covers Bayesian point selection, and a
  second one would be a redundant path.

  Measured on `load_breast_cancer` tuning a RandomForest, cv=3, mean of 3
  seeds: random search 0.7425 in 18.7s; successive halving 0.7226 in 5.0s
  (3.8x); Hyperband 0.7374 in 6.0s (3.1x). Both docstrings carry that table,
  including the part where plain halving gives up two points of score by
  committing to one aggressive schedule.
- **`BayesianSearchCV` is now exported from `tuiml.evaluation`.** It existed
  in `tuiml.evaluation.tuning` but was never surfaced alongside `GridSearchCV`
  and `RandomSearchCV`, so the documented "three strategies" were two.
- **Six new native anomaly detectors** in `tuiml.algorithms.anomaly`, chosen
  because scikit-learn, Weka and CapyMOA offer none of them. They split into
  three groups, and which group to reach for is the whole decision:

  - **Per-feature, very fast** — `ECODDetector`, `COPODDetector`,
    `HBOSDetector`. Parameter-free (ECOD/COPOD), invariant to monotone
    rescaling, and scale to high dimension. On 1500 points in 20 dimensions
    they reach AUC 1.00 in **1-6 ms** against IsolationForest's 958 ms.
    `ECODDetector.feature_contributions()` names the features that caused each
    flag; `HBOSDetector` keeps no training data, so the fitted model stays
    small however large the training set.
  - **Joint-structure** — `KNNDetector`, `ABODDetector`. Slower, but they see
    what the first group cannot. On anomalies that are ordinary in every
    individual feature and only strange in combination, `KNNDetector` scores
    AUC 0.955 where `ECODDetector` scores 0.243 — worse than chance. In 120
    dimensions the reverse holds for LocalOutlierFactor, which falls to 0.56
    while kNN, ABOD and ECOD all reach 1.00.

  - **Ensemble** — `LSCPDetector`. Picks the best base detector separately for
    each point's own neighbourhood instead of assuming one wins everywhere. On
    mixed-density data where kNN scores 0.47 at `k=5` and 0.99 at `k=35`, a
    plain average of that pool manages only 0.69 while LSCP reaches 0.98 —
    without being told which `k` to trust. `local_competence()` exposes which
    detector was selected where. It costs roughly an order of magnitude more
    than its slowest member, so the docstring says plainly to benchmark it
    against averaging the same pool before adopting it.

  Both `KNNDetector` and `ABODDetector` are verified against brute-force
  implementations of their published formulas. `ABODDetector` carries an
  explicit warning with measurements: a tight group of anomalies masks itself
  and **inverts** the ranking (AUC 0.00 at cluster spread 0.05), so bursty
  anomalies belong to kNN or ECOD instead. `KNNDetector` has a milder form of
  the same effect and documents the fix — set `n_neighbors` above the largest
  anomaly group you expect.
- **New `tuiml.algorithms.timeseries.classification` subpackage — a task type
  the library could not serve at all.** The existing `timeseries` package only
  forecasts; classifying a *whole series* by its shape is a different problem,
  and flattening a series into columns for an ordinary classifier throws away
  the time ordering that carries the signal. Nothing in scikit-learn, Weka or
  CapyMOA covers it.

  - `MiniRocketClassifier` / `MiniRocketTransformer` — 84 fixed dilated
    kernels summarised by proportion of positive values, then a linear head.
    On a synthetic problem where the class is the *frequency* of a burst
    hidden at a random position (amplitude, phase, sign and position
    randomised, series z-normalised, so no energy cue survives): **0.983**
    against Euclidean 1NN's 0.895, DTW's 0.772 and RandomForest's 0.590 — and
    it predicted in 168 ms where DTW took 12.9 seconds. Prediction cost is
    flat in training-set size, which is the reason to reach for it first.
  - `HIVECOTEClassifier` — meta-ensemble weighting each representation by its
    own cross-validated accuracy raised to a power. The weighting demonstrably
    works (the dictionary component was down-weighted to 0.10 where it was
    weak) and the ensemble tracks the best member without being told which it
    is — but MINIROCKET alone matched or beat it on every problem measured, at
    a quarter of the cost, and the docstring says so with the table.
  - `TimeSeriesForestClassifier` — mean, standard deviation and slope of
    random intervals, in a forest. The temporally localised view: a split on
    "the slope between t=40 and t=90" says *where* the difference lives.
  - `BOSSClassifier` — bag of symbolic words built from low-frequency Fourier
    coefficients of sliding windows, classified by the asymmetric BOSS
    distance. A genuinely different view: what patterns a series contains and
    how often, discarding where. It beats a Euclidean neighbour decisively
    when position varies (0.844 against 0.588 on a motif-count problem) but
    `MiniRocketClassifier` beat it on every problem measured, so the docstring
    positions it as a **diverse ensemble component** rather than a standalone
    winner — which is why every strong meta-ensemble includes a dictionary
    member.
  - `ShapeletTransformClassifier` — finds the short subsequences that separate
    the classes and represents each series by its distance to them. The
    **interpretable** member of the family: `shapelets_` holds real
    subsequences you can plot, and `shapelet_info_` records which training
    series, channel and position each came from. Verified against ground truth
    on a planted-motif problem — the top five shapelets all came from series
    containing the motif, at windows overlapping where it was planted.
  - `DTWNeighborsClassifier` — nearest neighbour under Dynamic Time Warping,
    the field's standard baseline. Univariate and multivariate panels,
    unequal lengths, uniform or distance weighting, Sakoe-Chiba band.
  - `dtw_distance`, `dtw_pairwise`, `lb_keogh`, `lb_keogh_envelope`,
    `as_panel` as public building blocks.

  The docstring documents, with measurements, the caveat that decides whether
  to use it at all: DTW is deliberately blind to *when* things happen, so when
  the classes differ by **shape** it scores 1.000 against Euclidean 1NN's 0.969
  and RandomForest's 0.925 — but when the classes differ by **timing** the same
  invariance destroys the only signal there is and DTW drops to 0.812 while
  Euclidean gets 1.000.
- **New shared C++ kernel `tuiml._cpp_ext.timeseries`.** `interval_features`
  returns the mean, standard deviation and slope of arbitrary intervals from
  prefix sums, so an interval costs O(1) whatever its width; intervals of 32
  points or fewer take a direct path instead, because differencing prefix sums
  left a ~1e-14 variance residue that `sqrt` turned into a ~1e-7 error in the
  standard deviation of a width-1 interval. Also `sfa_transform`
  computes the low-frequency DFT of every sliding window, advancing the window
  with the momentary Fourier transform so each step costs one complex multiply
  per coefficient instead of a fresh transform. Verified against a direct DFT
  across five window/word/normalisation configurations. Also `shapelet_distances`
  computes the minimum z-normalised distance from each series to each shapelet,
  folding the window normalisation into the algebra so only one dot product per
  window is needed. It centres each series first: the running variance is
  `E[x^2] - mean^2`, which cancels catastrophically far from zero, and without
  centring a series offset by 1e6 drifted ~4e-2 from the same series at zero.
  Also `minirocket_transform`
  and `minirocket_biases` exploit the algebraic shortcut that makes MINIROCKET
  fast: a kernel is -1 everywhere except three positions holding +2, so the
  all -1 convolution is computed once per dilation and the nine corrections
  cached, after which each of the 84 kernels costs three vector additions
  instead of a fresh convolution. Verified against a direct dilated
  convolution across 16 (kernel, dilation) combinations. Also `dtw_distance` (with
  Sakoe-Chiba banding and early abandoning), `lb_keogh` / `lb_keogh_envelope`
  (O(n) envelope via a monotonic deque) and `dtw_pairwise` / `dtw_knn`. The
  kNN path orders candidates by their lower bound and skips any that cannot
  beat the running k-th best: **12.6x** faster than building the full distance
  matrix on 60 queries against 400 series of length 100, returning bit-identical
  neighbours. DTW is verified against a direct implementation of the recurrence,
  and LB_Keogh is tested to never exceed the true distance — if it did, the
  pruning would silently return wrong answers.
- **New shared C++ kernels `tuiml._cpp_ext.stats`.** Beyond the PAVA kernel:
  `tail_probabilities` (per-dimension empirical CDF by sorted binary search,
  floored so `-log` stays finite), `skewness` (adjusted Fisher-Pearson,
  matching `scipy.stats.skew(bias=False)`), `equal_width_histogram`,
  `equal_frequency_histogram` and `histogram_density`. All are OpenMP-parallel
  over dimensions and verified against numpy/scipy references. The histogram
  pair is reusable by `tuiml.preprocessing.discretization`.
  `pool_adjacent_violators` and `isotonic_fit` implement PAVA in O(n), used by
  `IsotonicCalibrator` and `VennAbersCalibrator`.

- **`NGBoostRegressor` / `NGBoostClassifier` — probabilistic gradient
  boosting.** Boosts against the *natural* gradient of a proper scoring rule
  (the ordinary gradient premultiplied by the inverse Fisher information), so
  the update is invariant to how the predicted distribution is parameterised.
  Predicts a full distribution rather than a point: `predict_dist`,
  `predict_interval` and `score_samples`. Normal, log-normal and exponential
  distributions; log score and CRPS. Pure NumPy on TuiML's own C++ tree
  learner — no scipy, no sklearn (`erf` via libm, the normal quantile by
  Acklam plus a Halley step, inverting the CDF to 1e-12).

  Calibration is the point of the method and was measured, not assumed: on a
  known heteroscedastic Normal the fitted sigma correlates **0.959** with the
  true noise scale (mean ratio 1.008), and a nominal 90% interval achieves
  **89.2%** empirical coverage on held-out data. The natural gradient agrees
  with `solve(FisherInfo, finite_difference_gradient)` to 1.0e-08, and the CRPS
  closed form matches numerical quadrature of its definition to 1e-07. Held-out
  RMSE is competitive with ordinary boosting (1.019 vs XGBoost's 1.036).

  Note that `predict_interval` on the *classifier* returns a boolean
  highest-probability credible set, not a numeric interval — a nominal target
  has no ordering.

- **Five classical forecasters: `SARIMAX`, `VAR`, `ThetaForecaster`, `TBATS`,
  `CrostonForecaster`.** No new dependencies.

  - `SARIMAX` — seasonal ARIMA with exogenous regressors, estimated by exact
    Gaussian maximum likelihood through a Kalman filter, with stationarity
    enforced by the Monahan/Jones partial-autocorrelation transform so the
    optimiser cannot wander into an explosive region. Adds forecast intervals
    from the Kalman variance. Recovers AR(1) phi=0.7 as 0.69662 (OLS: 0.69697),
    a pure-exog coefficient of 3 as 3.00029, and MA(1) theta=0.6 as 0.60.
  - `VAR` — vector autoregression; several series predicted jointly from the
    lagged history of all of them, with AIC/BIC lag selection. Recovers a known
    coefficient matrix to 0.021 at n=5000, and reduces to univariate AR(1) OLS
    bit-identically on one series. Accepts a 1-D series as a single-series
    panel.
  - `ThetaForecaster` — verified against Hyndman & Billah's equivalence result:
    the standard method is simple exponential smoothing with drift b/2, matched
    to **1.4e-14** across six alpha values, three seeds and horizons 1-24.
    Optional deseasonalisation gated on an ACF seasonality test.
  - `TBATS` — trigonometric seasonality, which is what lets it take
    high-frequency and **non-integer** seasonal periods (365.25) that seasonal
    ARIMA cannot represent. Multiple simultaneous periods, Box-Cox, damped
    trend. On a two-sinusoid-plus-trend series it forecasts to MAE 0.0004
    against a 1.918 no-seasonality baseline; a 52.18-period series gives MAE
    3.1e-14.
  - `CrostonForecaster` — intermittent demand, smoothing demand sizes and
    inter-arrival intervals separately. `classic`, `sba`, `sbj` and `tsb`
    variants; on demand 10 every 4 periods the classic forecast is exactly 2.5
    and SBA exactly 2.5(1 - alpha/2).

- **New optional `tuiml[torch]` extra, and six neural algorithms behind it.**
  `tuiml/algorithms/tabular_foundation/` adds `FTTransformerClassifier` /
  `Regressor` (per-feature tokenisation plus a CLS token through pre-norm
  Transformer blocks), `SAINTClassifier` / `Regressor` (attention across *rows*
  as well as features) and `NODEClassifier` / `Regressor` (differentiable
  oblivious decision trees over a self-implemented `entmax15`).
  `tuiml/algorithms/timeseries/deep/` adds `NBEATSForecaster` (doubly-residual
  stacking, generic and interpretable bases), `NHITSForecaster` (multi-rate
  pooling and hierarchical interpolation) and `PatchTSTForecaster` (patch
  tokens, channel independence, RevIN instance normalisation).

  All six learn: 0.98-0.99 accuracy on an XOR-style target where a linear model
  gets 0.5, and R^2 0.98-0.99 on a non-linear regression. The forecasters beat
  a naive baseline by four to six orders of magnitude on a clean signal.

  **torch stays genuinely optional, at three levels.** Importing TuiML never
  imports torch, so the catalog is byte-identical on either install — all 243
  algorithms are listed and their schemas readable with torch absent.
  Constructing a model never needs torch, so parameter grids and pickling work
  everywhere. Only `fit` requires it, and raises an `ImportError` naming the
  class and the exact command, `pip install 'tuiml[torch]'`. Enforced centrally
  by `tuiml.utils.torch_backend` and pinned by an AST test asserting no
  module-scope torch import anywhere in either package.

- **`foundation.TabICLClassifier` / `foundation.TabICLRegressor` — the first
  pretrained tabular foundation model, behind a new `tuiml[foundation]`
  extra.** TabICL predicts *without training*: your rows are fed to a frozen
  transformer as in-context examples and the answer comes out of one forward
  pass. There is no gradient step, no tuning, and no fitted coefficient — `fit`
  only memorises the training set, and essentially all the compute lands in
  `predict`, which is the reverse of every other algorithm in TuiML. On a
  non-linear target where a linear model scores 0.5, it reaches 1.0 held-out
  accuracy with zero training.

  Registered under the `foundation.` namespace, like `sklearn.SVC` and
  `weka.J48`, because TuiML delegates to the upstream `tabicl` package rather
  than running its own implementation. The namespace also keeps these out of
  the generic contract sweep, which fits every algorithm it finds — and these
  would each pull a checkpoint over the network.

  **TuiML ships no model weights, and that is a deliberate constraint rather
  than an implementation detail.** The upstream package downloads its own
  ~150 MB checkpoint on first use, so that transfer is between the user and
  the publisher, under the publisher's license. TabICL is the only tabular
  foundation model integrated because its **weights** are BSD-3-Clause, the
  same license as TuiML: nothing for a user to accept, no commercial-use
  restriction. Others — TabPFN, Google's TabFM — publish weights restricted to
  non-commercial use, and a BSD-3 wrapper cannot relicense what it wraps, so
  integrating those would need a consent gate that deliberately does not exist
  yet. A test asserts no checkpoint file ever lands inside the installed
  package.

- **The installer now detects your GPU and offers the neural extras.**
  `install.sh` probes for CUDA (via `nvidia-smi`, including VRAM), ROCm and
  Apple Silicon's Metal backend, then offers `tuiml[torch]` and
  `tuiml[foundation]` — defaulting to yes when an accelerator is present and
  to no when it is not, since both work on CPU but slowly. Warns below 8 GB of
  VRAM. `TUIML_GPU=cuda|rocm|mps|cpu` skips the probe and
  `TUIML_EXTRAS="torch,foundation"` skips the questions. An `nvidia-smi` that
  is present but failing — a common driver-mismatch state — is treated as no
  GPU rather than trusted.

### Changed
- **`curl … | install.sh | bash` now installs a release, and asks first.** The
  installer only ever installed from `git+https://github.com/tuiml/tuiml.git`,
  so the documented one-liner handed out whatever happened to be on `main` —
  unreleased code between releases — and compiled the C++ extensions locally,
  bypassing the published wheels and requiring a compiler. It also disagreed
  with `tuiml update`, which resolves against PyPI, so installing with the
  script and then updating silently switched channels. It now offers the
  choice:

  ```
    1) Stable    latest release from PyPI, prebuilt — recommended
    2) Developer newest code from GitHub main, unreleased, built from source
  ```

  `TUIML_CHANNEL=stable|git` skips the question, and a run with no terminal
  takes stable rather than blocking, so CI and Docker are unaffected. `git`
  and a C++ compiler are now required only on the developer channel.
- **`import tuiml` no longer imports the library.** It imported the registry,
  every algorithm, training, benchmarking, serving, workflow, the agent
  package and both optional bridges at module scope — about 2.3s, pulling in
  matplotlib and seaborn, paid by every import including the one behind
  `tuiml --version`. Public names now resolve on first use (PEP 562), so
  nothing is imported until something is actually used:

  | | before | after |
  |---|---|---|
  | `import tuiml` | 2.25s | 0.03s |
  | `tuiml --version` | 2.0s | 0.05s |
  | `tuiml --help` | 2.0s | 0.49s |

  Those imports were also what populated the component registry, which now
  fills itself on first read instead (`Registry._ensure_populated`). This is
  stricter than before, not looser: `from tuiml.registry import registry`
  previously saw a populated registry only because importing the parent
  package had imported everything, and now populates however it is reached.
  The catalogue is unchanged at 252 components with the same per-type
  breakdown. `tuiml.agent`, `tuiml.algorithms` and the other submodules
  remain reachable as attributes, and `import tuiml.agent as x` still yields
  the module rather than the `agent()` function.
- **User algorithms are no longer registered by importing
  `tuiml.agent.tools`.** The CLI, the MCP server and `execute_tool` all load
  them at the point they read the registry, so nothing changes for those
  paths. Code that imported the module directly and then expected the
  registry to contain agent-authored algorithms should call
  `tuiml.agent.user_algorithms.ensure_loaded()`.
- **`tuiml_system_info` reports `install_source`**, and on a VCS install
  `installed_commit`, `tracking_ref` and `latest_commit`. `latest_version` is
  still reported on every channel.

- **XGBoost, LightGBM and CatBoost are imported lazily.** They remain part of
  the default install, but `tuiml.algorithms` no longer imports them at module
  load. Each ships its own OpenMP runtime, and importing all three
  unconditionally left every session one `import torch` away from a segfault
  (see Fixed). Being *installed* was never the problem; being imported eagerly
  was.

  The availability check also moved out of `__init__` and into `fit`, matching
  every other backend: constructing a wrapper records hyperparameters and reads
  its schema without loading the library. Nothing else changes for callers.

  A `tuiml[boosting]` extra exists as a no-op alias so an existing Dockerfile
  or CI line carrying it keeps resolving.

- **New `tuiml[all]` extra** — every wrapper backend, the neural models and the
  pretrained foundation model in one command. On Linux, pin CPU wheels unless
  you want the CUDA build: `uv pip install --torch-backend=cpu 'tuiml[all]'`.

- **The installer picks the right PyTorch build.** PyPI serves the CUDA wheel
  by default, so `pip install torch` on Linux pulls roughly twenty NVIDIA
  runtime packages — several gigabytes — *even with no NVIDIA GPU present*.
  `install.sh` and `install.ps1` now pass `--torch-backend=cpu` when they find
  no accelerator and `auto` when they do, turning a multi-gigabyte download
  into a few hundred megabytes on an ordinary laptop. The neural extras
  therefore default to yes on every machine rather than only on GPU boxes.

- **Getting-started and contributing pages rewritten for the new extras.**
  Both had drifted. Getting started claimed every optional backend registers
  under a namespaced hub key — true for `sklearn.SVC`, `weka.J48` and
  `foundation.TabICLClassifier`, but the boosting and neural models keep bare
  names, so a contributor copying that would have got it wrong. Contributing
  listed 6 of the 17 directories under `algorithms/`, and did not mention that
  `uv sync` now yields a core-only environment in which backend tests *skip*
  rather than fail — so a change to `tabular_foundation/` could show a fully
  green run having tested none of it. It now documents `--all-extras`,
  per-extra sync and `pytest -rs`, plus the three-level rule any new optional
  dependency has to follow.

### Changed (breaking)
- **`ARIMA` no longer accepts `seasonal_order`.** The argument was stored in
  `__init__` and read nowhere else in the file, so `seasonal_order=(1,1,1,12)`
  silently fitted a **non-seasonal** model and returned forecasts with no
  seasonal structure. Rather than leave a parameter that lies, it is removed;
  `SARIMAX` implements seasonal terms properly, along with exogenous
  regressors, exact maximum likelihood and forecast intervals. Passing it now
  raises `TypeError` instead of being ignored.

### Added
- **The installers offer to install `git` instead of just refusing.** Building
  from source (`TUIML_CHANNEL=git`) needs git, and both installers previously
  aborted with instructions and left you to start over. They now offer to
  install it: Homebrew or the Xcode Command Line Tools on macOS,
  apt/dnf/yum/pacman/zypper/apk on Linux, winget/scoop/choco on Windows.

  Consent is required, because this installs software system-wide and may use
  `sudo`. `TUIML_INSTALL_GIT=1` answers yes for automation and `0` answers no;
  with no terminal and no variable set the answer is **no**, so an unattended
  run never escalates on its own. Declining still prints the manual command,
  and now also points at the stable channel, which needs no git at all.

  Two details that would otherwise bite:

  - On macOS, `/usr/bin/git` exists as a stub even with no Command Line Tools
    installed, so `command -v git` succeeds while every invocation fails. The
    check runs `git --version` and tests the exit status instead.
  - On Windows, winget writes PATH to the registry but the running session
    keeps the PATH it started with, so git would still look missing right
    after a successful install. `$env:Path` is rebuilt from the machine and
    user scopes before re-checking.

  The Xcode Command Line Tools installer is a separate GUI process that runs
  on after `xcode-select` returns, so that path polls for git to appear rather
  than assuming the command exiting means it finished.

### Fixed
- **The installers failed on any machine whose default Python is older than
  3.10.** Neither passed `--python` to `uv tool install`, so uv built the tool
  environment against whatever interpreter it found first. On a Windows box
  defaulting to 3.9.19 — and equally on macOS, which still ships 3.9 — the
  install died after every question had been answered, with an error that
  blamed the extras rather than the Python version:

  ```
  Because the current Python version (3.9.19) does not satisfy Python>=3.10
  and you require tuiml[sklearn], your requirements are unsatisfiable.
  ```

  Both installers now pass `--python ">=3.10"`, mirroring `requires-python`,
  so uv selects a suitable interpreter or downloads a managed one. Verified on
  a host whose default is 3.9.6: the tool environment is built on 3.13.11.
  Override with `TUIML_PYTHON`.

- **The installers now survive `--torch-backend` being changed or removed.**
  uv prints "the `--torch-backend` option is experimental and may change
  without warning" when it is used, so a future uv release could break the
  install outright. Both installers now retry once without the flag, warning
  that this may pull the CUDA build. A failed install is worse than a large
  one.

- **`install.ps1` printed the low-VRAM warning before the section it belongs
  to**, so "Only 6144 MB of VRAM" appeared above the "Neural models" heading
  that explains what it refers to. The probe now returns the figure and the
  caller prints it in place.

- **`ARIMA` never estimated its moving-average parameters.** `ma_params_` was
  initialised to `np.zeros(q)` and `_refine_parameters` looped over `range(p)`
  only, so theta stayed at exactly zero for the life of the model. Any `q > 0`
  specification was silently a pure-AR fit, and `ARIMA(order=(0, 0, 1))` was a
  constant.

  The fixed-step AR-only descent is replaced by an L-BFGS-B minimisation of the
  conditional sum of squares over the constant, the AR block and the MA block
  together, bounded to keep the recursion from exploding, and accepted only
  when it improves on the Yule-Walker starting point. Measured on simulated
  series: MA(1) theta=0.6 recovers as **0.6001**, AR(1) phi=0.7 as 0.7111, and
  ARMA(1,1) with (0.5, 0.4) as (0.521, 0.389).

- **Interpreter segfault when a neural model was fitted after any boosting
  import (macOS).** torch bundles its own `libomp.dylib`, while xgboost,
  LightGBM and CatBoost each resolve `@rpath/libomp.dylib` to a different copy.
  Because `tuiml.algorithms` imports all three eagerly, a lazily-imported torch
  landed in a process holding two OpenMP runtimes and the first `LayerNorm`
  killed the interpreter with SIGSEGV and no traceback.

  **Fixed at the root** by importing the three lazily (above). They are still
  installed by default — that was never the problem. A process that never
  boosts now never loads a second OpenMP runtime, so torch keeps all its
  threads: measured 8, against 1 under the previous thread-clamping
  mitigation.

  The clash is **symmetric**, and a guard remains for the case where one
  process genuinely uses both. Whichever runtime initialises second can crash,
  and the two directions do not share a fix — measured:

  | Situation | `KMP_DUPLICATE_LIB_OK=TRUE` | `torch.set_num_threads(1)` | `OMP_NUM_THREADS=1` in env |
  |---|---|---|---|
  | boosting loaded first, then torch | still segfaults | **works** | works |
  | torch loaded first, then boosting | still segfaults | still segfaults | **works** |

  So `tuiml.utils.torch_backend.guard_duplicate_openmp` clamps torch after
  importing it, and `tuiml.algorithms.gradient_boosting._backend` sets
  `OMP_NUM_THREADS` before importing a boosting library — each only on darwin,
  and only when the other library is already loaded. Note that the widely
  recommended `KMP_DUPLICATE_LIB_OK=TRUE` fixes neither direction.

- **Exported notebooks failed on user algorithms authored in an earlier
  session.** `tuiml_export_notebook` inlines a user algorithm's source only
  when the session recorded the `tuiml_create_algorithm` call that wrote it.
  But an algorithm outlives the session that authored it — it is stored in
  `~/.tuiml/user_algorithms/` and re-registered at server startup — so a later
  session that merely trained on it exported a notebook naming an algorithm
  nothing defined:

  ```
  ValueError: Algorithm 'WeightedSoftVoteEnsemble_v1_1_0' not found in hub.
  ```

  The source is now read off disk at export time for any user algorithm the
  session referenced, and inlined ahead of the cells that use it. Every
  definition cell also names where the algorithm is stored
  (`~/.tuiml/user_algorithms/<Name>/<version>/algorithm.py`), so a reader can
  tell the notebook's copy from the original the MCP server loads, and knows
  which one to edit.
- **Inlined user algorithms registered the wrong name.** Executing the source
  fires its `@classifier`/`@regressor` decorator, which registers the bare
  class name and nothing else — but the MCP server also registers a versioned
  alias (`MyGBM_v1_0_0`), and that alias is the name every recorded call
  carries. So even a notebook that *did* inline the source raised "not found
  in hub" on the training cell. The definition cell now registers the alias
  the same way the loader does, keyed off the class the source actually
  defines rather than the name it is stored under (`create` records those
  separately, and they can differ).
- **Every `tuiml` command loaded your user algorithms and announced it.**

  ```
  $ tuiml --version
  [tuiml] loaded 2 user algorithm(s)
  tuiml, version 0.1.9
  ```

  `tuiml/agent/tools/__init__.py` called `load_all()` at module scope, and 27
  of the 30 CLI subcommand modules import it — so asking for the version
  scanned `~/.tuiml/user_algorithms` and executed whatever Python it found
  there. `load_all()`'s own docstring says "Called once at MCP server
  startup", which was the intent; module scope made it universal. Loading is
  now on demand, and `--version` and `--help` never trigger it.
- **`tuiml update` reported `✓ TuiML upgraded: ? → ?`** on every successful
  upgrade. The summary read `previous_version` and `version`, but the tool
  returns `version_before` and `version_after`, so both lookups missed and
  fell to a placeholder. The upgrade itself always worked; only the report
  was wrong, which made a working command look broken.
- **`tuiml update` moved a git install onto PyPI.** It always ran
  `uv tool install --reinstall --force tuiml`, naming the released package
  whatever the install actually was, so a developer-channel install was
  quietly switched to the release — and because `main` and the last release
  share a version string, the report read as a no-op (`0.1.9 → 0.1.9`) while
  the channel changed underneath. It now updates along the channel it finds,
  and prints the commit, which is the only thing that visibly moves on a
  branch install. `--target` on a git install is a clear error rather than a
  silent switch.
- **Update checks only ever consulted PyPI**, so a git install always looked
  current: `main`'s version string normally equals the last release, however
  far behind the checkout was. The channel is detected from PEP 610
  `direct_url.json` and compared appropriately — released version for PyPI,
  tracked branch head via `git ls-remote` for git (no API token, no rate
  limit), nothing for an editable checkout, which `git pull` updates.
- **The installer appeared to hang after "Installed 2 executables".** The next
  thing it runs is `tuiml --version`, which on a fresh install compiles
  bytecode for the whole dependency tree while printing nothing. Installs now
  pass `--compile-bytecode`, so that work happens during installation where
  uv shows progress, and the verification step announces itself.
- **`tuiml update` printed nothing at all until it finished.** The upgrade
  blocks inside a `pip` / `uv` subprocess whose output it captures, so the
  terminal stayed blank for as long as the install took — up to the 300s
  timeout — with no way to tell a slow download from a hung command. It now
  reports each phase as it starts, on a live status line that shows elapsed
  seconds:

  ```
  | Installing tuiml via uv-tool, this can take a minute (12s)
  ```

  Progress goes to stderr, so `tuiml update --json` still pipes cleanly; it
  animates only on a TTY and prints one plain line per phase when redirected.
  MCP clients get the same phases as progress notifications — `tuiml_self_update`
  was silent there for the same reason.
- **`tuiml update --dry-run` crashed in a development checkout** with
  `TypeError: can only join an iterable`. The tool deliberately returns
  `command: None` there — an editable install has no upgrade command to
  predict, and it reports the refusal in `note` instead — but the CLI joined
  the command unconditionally. It now prints the refusal, as the dry run
  intended.
- **`tuiml_export_notebook` dropped agent-authored algorithms.** A session that
  created an algorithm and trained it exported a notebook whose
  `tuiml.train({"model": {"name": "MyAlgo"}})` cell referenced a name nothing
  defined: `tuiml_create_algorithm` was recorded, but the translator had no
  branch for it, so it was skipped without a warning and the count of exported
  steps was quietly reduced to match. Since a user algorithm lives in
  `~/.tuiml/user_algorithms/` rather than the installed package, the exported
  notebook could not run anywhere else — which is the one thing an exported
  notebook is for. Its source is now inlined ahead of the cells that use it, so
  the `@classifier` / `@regressor` decorator registers the class on execution.
  `tuiml_edit_algorithm` exports the full post-edit source, repeated edits
  collapse to one definition, and `tuiml_get_skeleton` / `tuiml_delete_algorithm`
  are no longer recorded — a template is superseded by the create call, and an
  exported delete would remove real files from `~/.tuiml/` on re-run.

## [0.1.9] - 2026-08-03

### Fixed
- **30 components omitted constructor parameters from `get_parameter_schema`**,
  18 of them `random_state`. The schema is not documentation: it is what
  `tuiml_describe` shows an agent, so an omitted parameter is one no agent can
  discover — meaning reproducibility was unreachable through the agent path for
  every algorithm that hid its seed. All 25 affected algorithms and 5
  transformers now declare their full signature, and a contract check fails the
  build if they drift apart again.
- **Repeating a tool call in one session gave different numbers.**
  `execute_tool` drew a fresh `random.randint()` for every call that omitted
  `random_seed`, so two identical `tuiml_benchmark` runs disagreed and
  comparing two runs measured the seed rather than the change. The only way to
  reproduce anything was to copy a seed out of an earlier response and pass it
  back. Calls without an explicit seed now share one **session seed**, fixed
  for the life of the server process, so a repeated call reproduces its
  numbers. An explicit `random_seed` still wins per call. `tuiml_system_info`
  reports the session seed, and `TUIML_SEED` pins it up front for CI and bug
  reports. The seed was always genuinely applied — `set_global_seed` was
  called with it — so this is a change of scope, not of whether seeding worked.
- **Benchmark ranking tables ranked backwards for lower-is-better metrics.**
  `Benchmark.plot_ranking` handed per-dataset means to `plot_ranking_table`
  without saying which direction was better, and the default is
  higher-is-better — so an RMSE or log-loss table ranked the worst model
  first. The same call dropped the dataset names, leaving rows identified only
  by position. Direction now comes from `_higher_is_better` and the row labels
  from the score table's index. Tied ranks were separately truncated by
  `int()`, so a two-way tie at rank 1.5 printed as `1` for both models; ties
  now render as the fraction they are.
- **Critical-difference diagrams crowded long model names.** Figure width was
  derived from the number of algorithms alone, so long names ran into the rank
  axis, and widening only the data limits compressed the axis instead of the
  figure. Width now also accounts for the longest label on each side. The
  clique bars additionally sat below every label, where they crossed none of
  the connectors whose grouping they encode; they now sit directly under the
  rank axis so each bar passes through the connector of every method in its
  group.

### Changed
- **Tutorial chapter 8 no longer claims tree algorithms crash on `NaN`.** The
  benchmark's `SimpleImputer` was justified by a `RecursionError` that 0.1.7
  fixed. The imputer is still not optional, but for a better reason:
  `RandomForestClassifier` now handles missing values natively (`vote`, 10-fold
  CV: 0.961 ± 0.025 raw vs 0.966 ± 0.027 imputed) while `LogisticRegression`
  silently degrades to 0.614 ± 0.004 from 0.963 ± 0.023 without complaint.
  Silent degradation is the sharper lesson, and the chapter now teaches that.

## [0.1.8] - 2026-08-02

### Changed (breaking)
- **The MCP server now targets the 2.x MCP SDK** (`mcp>=2.0.0`, was
  `mcp>=1.0.0`). The 2.0 SDK replaced the low-level decorator API the server
  was built on — `@server.list_tools()`, `@server.call_tool()` and the
  resource decorators — with handlers registered on the `Server(...)`
  constructor, so the two are not source-compatible and there is no single
  code path that serves both. Because the old pin was unbounded, a fresh
  `pip install tuiml` / `uv tool install tuiml` resolved mcp 2.0.0 and
  `tuiml-mcp` died at startup with `AttributeError: 'Server' object has no
  attribute 'list_tools'`, which surfaced in Claude Desktop and Cursor as
  `connection closed: calling "initialize"`. If you pin `mcp` yourself, move
  to 2.x; an SDK that is too old is now reported at startup with an
  actionable message instead of an `AttributeError`.

### Fixed
- **Tuning and benchmark progress reaches the client again.** Progress was
  streamed as `notifications/message` log entries, which 2.x gates behind a
  per-request opt-in that no client sends by default. It now goes out as
  spec-correct `notifications/progress` against the client's `progressToken`,
  so `tuiml_tune` and `tuiml_benchmark` report live progress in clients that
  ask for it, with the iteration total attached for tuning runs.
- **Tool arguments are validated again.** Input validation against each
  tool's `inputSchema` was performed by the 1.x decorator; on 2.x it has to
  be done in the handler, without which a malformed argument reached the tool
  as a confusing `TypeError` instead of a clear validation error.
- **`tuiml-mcp --info` reported version `1.0.0`** regardless of the installed
  version. It now reports the package version, which is also what the server
  advertises in the initialize handshake.
- **`tuiml_plot(plot_type='tree')` failed for every agent-trained model** with
  "The tree model is not fitted yet." `tuiml_train` saves a `Workflow`, and
  the `tree` branch handed that wrapper to `plot_tree` instead of unwrapping
  `workflow.model_` the way `feature_importance` already did — so a `model_id`
  could never plot, however tree-based the model was. Asking for a tree plot
  of a model that genuinely has no tree now returns a clear message naming the
  model and pointing at `feature_importance`, instead of claiming it is
  unfitted. `NaiveBayesClassifier` is excluded correctly: it stores a
  list-of-lists of *probability* estimators under `estimators_`, which a
  looser check mistook for a tree ensemble and drew nonsense from.

## [0.1.7] - 2026-08-02

### Removed (breaking)
- **Flat agent adapter modules.** The five framework adapters moved under
  `tuiml.agent.adapters`. The old module paths are gone, with no compatibility
  shims — pre-1.0, one blessed path per concept:

  | Old | New |
  |-----|-----|
  | `from tuiml.agent import openai` | `from tuiml.agent.adapters import openai` |
  | `from tuiml.agent import anthropic` | `from tuiml.agent.adapters import anthropic` |
  | `from tuiml.agent.langchain import get_tools` | `from tuiml.agent.adapters.langchain import get_tools` |
  | `from tuiml.agent.crewai import get_tools` | `from tuiml.agent.adapters.crewai import get_tools` |
  | `from tuiml.agent.pydantic_ai import get_tools` | `from tuiml.agent.adapters.pydantic_ai import get_tools` |

  `tuiml.agent.registry` moved to `tuiml.agent.tools._components`, and
  `tuiml.agent.restart_util` to `tuiml.agent.tools.system.restart`.
- **`tuiml.agent()` as a top-level function.** Binding it at the root shadowed
  the `tuiml.agent` *package*, so `import tuiml.agent as x` handed back the
  function instead of the module. Use `from tuiml.agent import agent` (or
  `tuiml.agent.agent()`). The returned Pydantic-AI agent is unchanged.
- **`tuiml_search` MCP tool.** It was never registered in any tool table, so
  calling it always returned `Unknown tool`. Keyword search over components is
  `tuiml_list(search=...)`, which has always worked.

### Fixed
- **Fitting a tree on data with missing values never returned.**
  `DecisionTreeClassifier`, `DecisionTreeRegressor` and both random forests
  hung and then died with a `RecursionError` on any `X` containing `NaN` —
  including the built-in `vote` dataset — with a traceback pointing at
  impurity arithmetic several frames from the cause.

  In the split search, `np.argsort` sorts `NaN` last and the candidate mask
  `sorted_col[1:] != sorted_col[:-1]` counts a boundary at a `NaN` as valid,
  because `NaN != NaN` is True. The chosen threshold was then the midpoint of
  two values one of which was `NaN`, so the threshold itself was `NaN`; every
  `x <= NaN` is False, so the builder put all rows on one side and recursed on
  a subproblem identical to its parent forever. `build_classifier_tree`
  already routed `NaN` away from the C++ builder and into the Python one, so
  the Python path was the designated missing-value path — it just could not
  handle them.

  Candidate thresholds now come from observed values only, so a threshold is
  always finite. Impurities are normalised over that subset and the gain is
  scaled by the fraction observed — C4.5's correction, which stops a
  mostly-missing feature outranking a complete one by being scored on an
  easier subset; `j48` already dropped `NaN` the same way. Missing rows route
  right, matching `NaN <= t`, and a split that fails to partition falls back
  to a leaf. Verified a no-op on complete data: predictions are unchanged
  across `gini`/`entropy`/`gain_ratio`,
  `squared_error`/`friedman_mse`/`absolute_error` and `RandomForest` on
  diabetes, iris, glass and cpu. `vote` now trains at 0.961 ± 0.025 under
  10-fold CV.
- **A batch prediction and a single-sample prediction could disagree on the
  same row.** `predict_single_numpy` sent missing values to whichever child
  held more training samples, which matched neither the structure the tree was
  actually fitted with nor the flattened batch predictor that `predict()`
  uses. Fitting, batch prediction and single-sample prediction now share one
  rule. Previously unreachable, because fitting on such data crashed first.
- **`list_algorithms()` hid every native algorithm after a bare `import
  tuiml`.** `tuiml/__init__.py` never imported `tuiml.algorithms`, and the
  `@classifier` / `@regressor` decorators only register on module import — so
  discovery reported 103 of 189 components and `RandomForestClassifier` was
  absent from the catalog entirely until something else happened to import it.
  Machines with entries in `~/.tuiml/user_algorithms` never saw this, because
  loading those pulls the imports in as a side effect. `train()` was
  unaffected (it resolves lazily), as was the MCP `tuiml_list` tool.
- **Every plot logged a matplotlib warning.** The stylesheet set
  `font.weight: 'medium'`, which no bundled font advertises as a face, so each
  figure emitted `findfont: Failed to find font weight medium, now using 400`.
  Same rendering, without the noise. The comparison-table renderer set it too.
- **A quarter of the See Also links in the API docs were broken, and the site
  hid it.** 147 of 544 pointed at pages that do not exist. The resolver only
  indexed *classes*, so every `:func:` reference fell through to a fallback
  that guessed a path — producing URLs like
  `/docs/utils/serialization/load_model.html` for a function documented on
  its module's page — while `:mod:` references produced `/docs//train.html`
  or a dead `#`. Modules, packages and functions are now indexed too, with
  anchors, and a reference that resolves to nothing renders as plain text
  rather than a link that only fails once clicked.

  The reason this went unnoticed: `404.html` was a byte-for-byte copy of
  `getting_started.html`, so every broken link looked like a successful
  navigation to the Getting Started page. It is now a real not-found page.
- **The next release would have shipped two files on the old version.**
  `bump_version.py` still pointed at `tuiml/agent/SKILL.md`, which moved to
  `tuiml/agent/prompts/` in the agent reorg. The script printed `SKIP (not
  found)` and carried on, so the skill file's advertised version would have
  silently lagged. A pre-flight check now verifies every registered file
  exists and still matches its pattern, and aborts before writing anything —
  a rejected bump can no longer leave the tree half-updated. Verified by
  running a real bump against a throwaway clone.
- **Every version bump broke `test_schemas.py`.** It hardcoded the release
  version twice, but the registered regex matched only the constructor
  argument (`version="0.1.6"`), not the assertion (`== "0.1.6"`), so a bump
  rewrote one and left the other. The test now uses an arbitrary version
  string — the real endpoint passes `tuiml.__version__` in, so the test only
  ever checked that the field round-trips — and is no longer version-tracked.
- **The published MCP tutorial showed a stale version.** A stored output cell
  in `tutorials/llm_friendly/02_mcp_server.ipynb` prints `TuiML version:` and
  was never bumped, so readers on tuiml.ai saw whatever it last said. Now
  registered.
- **The Pages workflow ignored two of its own build inputs.** It fired only on
  `website/**`, but `build.py` also reads `tutorials/` and `CHANGELOG.md` from
  the repo root. A tutorial-only edit changed what the site would publish
  without triggering the build that publishes it, leaving the 19 `/tutorials/`
  pages behind the repo until an unrelated `website/` change fired a deploy.
- **`tuiml_self_update(dry_run=True)` failed in a dev checkout.** The
  editable-install guard ran before the dry-run branch, so a call that promises
  to change nothing and report what would happen returned an error instead. A
  dry run now always succeeds, reporting the refusal as its prediction.
- **The notebook-export smoke test had been dead since the `tuiml/agent/`
  reorg.** It imported `_SESSION_CALLS` and `execute_export_notebook` from
  `tuiml.agent.tools`, which moved to `tools/_state.py` and behind
  `execute_tool`, so the test errored on import rather than checking anything —
  leaving every generated notebook cell unverified. Fixed, and it again
  executes all 21 generated cells.
- **Eight `tuiml.sklearn` wrappers were unusable inside a pipeline.** The
  wrappers read `get_params()` off `__dict__`, which also picked up the
  attributes TuiML's own base classes set (`FeatureSelector` sets `k` and
  `threshold`, `FeatureExtractor` sets `n_components`). Cloning a step for each
  CV fold fed those back through the constructor, and scikit-learn rejected
  them: `sklearn.SelectKBest: unknown parameter(s) ['threshold']`. Affected
  `SelectKBest`, `SelectPercentile`, `SelectFdr`, `SelectFpr`, `SelectFwe`,
  `GenericUnivariateSelect`, `VarianceThreshold` and `AdditiveChi2Sampler`.
  `get_params()` now reports exactly the parameters the wrapper was built with.
- **`random_seed` never reached the `tuiml.sklearn` wrappers.** Seed injection
  looked for `random_state` in the constructor signature, and every wrapper
  takes `**params`, so `tuiml.train(..., "random_seed": 42)` left them
  unseeded and runs were not reproducible. The seed now also consults
  `get_parameter_schema()`, so it reaches wrappers whose backing estimator
  accepts a seed and still skips those that do not.
- **`tuiml uninstall` could not unwire OpenClaw or OpenCode.** Both kinds fell
  through the dispatcher to `unknown client kind`, so `tuiml setup` could add
  an entry that `tuiml uninstall` then refused to remove. OpenClaw is removed
  via `openclaw mcp remove` (falling back to a direct config edit) and OpenCode
  through its `mcp` key.
- **Docstring examples that could not run.** `tuiml/sklearn/__init__.py`
  documented `tuiml.train("sklearn.RandomForestClassifier", {...}, cv=5)` and
  `tuiml.benchmark(...)`, neither of which exists; the CapyMOA `partial_fit`
  examples showed no output for a call that returns `self`. Every example in
  both bridge packages is now a passing doctest.
- **Staged training raised `ImportError`.** `tuiml_train` with `stage="init"`,
  `stage="fit"` (new model), or `stage="partial_fit"` imported
  `_inject_seed_to_algorithm` from `tuiml.workflow`, which is named
  `_inject_seed`. All four staged paths work again.
- **Framework adapters were missing the discovery tools.** `get_tools()` for
  LangChain / OpenAI / Anthropic / CrewAI / Pydantic-AI exposed only the 24
  workflow tools, so agents had no `tuiml_list` or `tuiml_describe` and could
  only train an algorithm whose exact class name they already knew. All 30
  tools are now exposed, matching what an MCP client sees.
- **Agent-started and library-started model servers could not see each other.**
  The `tuiml_serve_model` / `tuiml_stop_server` / `tuiml_server_status` tools
  kept their own server registry, separate from `tuiml.serve()` /
  `tuiml.stop_server()` / `tuiml.server_status()`. They are now wrappers over
  the root serving API: one registry, one `"host:port"` server-id scheme, and
  the real readiness wait instead of a fixed one-second sleep (so a busy port
  now reports an error rather than handing back a URL for a dead server).
- **`model_id`s no longer survive only in memory.** The model index is
  rehydrated from `~/.tuiml/models/` at import, matching the dataset index, so
  a `model_id` still resolves after an MCP server restart.
- **MCP annotations understated several tools.** `tuiml_delete_algorithm`,
  `tuiml_edit_algorithm`, `tuiml_self_update` and `tuiml_restart` had no
  annotation entry and were advertised to clients as read-only and
  non-destructive. They are now marked destructive; `tuiml_create_algorithm` is
  no longer read-only and `tuiml_system_info` is marked open-world.

### Added
- **"View source on GitHub" links throughout the API docs.** Every module,
  class, function and method now carries a GitHub icon linking to its own
  implementation, anchored to the line it starts on and opening in a new tab.
  2501 links, each verified to point at a file that exists at a line that
  does.
- **Package overviews for every package.** All 65 package index pages now
  carry real documentation — what the package is for, what is in it, and when
  to reach for which piece. 31 were previously under 400 characters and
  `tuiml/agent/mcp/` had no docstring at all, so its page rendered blank.
- **Smoke test covering all 30 MCP tools** (`tests/test_agent/`). Each tool is
  dispatched through `execute_tool` — the entry point `server.py` uses — in one
  coherent session, so ids flow from the tool that produces them to the tools
  that consume them, and each result is recorded exactly as the server records
  it. A coverage test compares the registry against the plan, so a tool added
  without being smoke-tested fails the suite instead of shipping untested. One
  tool is deliberately not executed, with the reason stated in code:
  `tuiml_restart` kills every running `tuiml-mcp` process, so only its
  read-only discovery half runs. Fixtures redirect every `~/.tuiml` write
  target at a tmp dir, so a test run neither pollutes the user's home nor reads
  state left by a previous one.

### Removed
- **The `/docs/` prefix on hand-written pages.** `/docs/` is where the 350
  generated API pages live; a privacy policy is not documentation. The eight
  hand-written pages moved to the site root — `/getting_started.html`,
  `/api-reference.html`, `/benchmarks.html`, `/changelog.html`,
  `/contributing.html`, `/privacy.html`, `/terms.html`, `/about.html` — and
  every old URL redirects to its new home, so existing links and search
  results keep working. The generated API tree is unchanged. Redirects
  previously ran the other way (`/privacy.html` into `/docs/privacy.html`);
  those are now reversed.
- **`remote-mcp.html`.** Nothing linked to it from the nav, the footer, or any
  page, and it was absent from the sitemap. The MCP server printed its URL on
  startup, which is now the getting-started page.
- **Dead test code.** 102 unused imports across 91 test files (mostly an
  `import pickle` copied into ~50 algorithm test modules), 6 unused locals, and
  from `tests/conftest.py` 10 fixtures and 3 assertion helpers no test has ever
  referenced — shrinking conftest from 557 lines to 299. Test count is
  unchanged at 1647.
- **A test that asserted nothing.** `test_shrinkage_adds_noise` set up the
  minority rows, then asserted only that resampling produced *some* new rows,
  which is true with or without shrinkage — so it passed whether or not the
  feature worked. It now asserts what its comment always claimed: that no new
  sample is an exact copy of a minority row.

### Changed
- **The tutorials are now a book.** The ten notebooks are replaced by fifteen
  numbered chapters that run in reading order, each assuming the ones before
  it: chapters 0–8 build the Python arc (data, evaluation, pipelines,
  features, imbalance, tuning, benchmarking), 9 covers experiments-as-data,
  10–12 cover driving TuiML from an agent, and 13–14 serve and apply it. Every
  notebook ships executed, so the published pages carry real tables, plots and
  CD diagrams.

  The old set mirrored the API's own table of contents; this one is organised
  around decisions, and reports what the library actually does rather than
  what a tutorial would prefer it did — imputation barely moves a random
  forest on the diabetes data, every feature-engineering technique loses on
  eight curated columns, logistic regression beats a 200-tree forest, and
  tuning's reported gain is smaller than its own optimism. Two leakage
  demonstrations carry the fold-discipline argument: feature selection outside
  the fold scores 82% on pure noise, and SMOTE before splitting inflates by
  six points.

  Every previous tutorial URL redirects to the chapter that now covers it.
- **`tuiml setup` client registry brought up to date.** Three entries pointed
  at config files the vendors no longer read:
  - **ChatGPT Desktop** has no MCP config of its own. OpenAI put the desktop
    app, the Codex CLI and the IDE extension on one file, `~/.codex/config.toml`
    (`[mcp_servers.tuiml]`), so they are now a single client. `--client
    chatgpt-desktop` still resolves, as an alias.
  - **VS Code** moved MCP out of `settings.json` (`mcp.servers`) into a
    dedicated user-profile `mcp.json`, whose root key is `servers` and whose
    stdio entries need an explicit `"type": "stdio"`.
  - **Antigravity** now has a native global MCP config shared by the IDE, the
    `agy` CLI and the SDK at `~/.gemini/config/mcp_config.json`; wiring it no
    longer depends on which VS Code extension happens to be installed. Gemini
    CLI is kept as a legacy entry (retired for consumer tiers, still live for
    enterprise and API-key users) and is detected on its `settings.json` rather
    than on `~/.gemini/`, which Antigravity now also owns.

  An existing server entry is merged rather than replaced, so extra fields the
  user added (`env`, `args`) survive a re-run.
- **Package overviews now appear in the generated API docs.** `generate_docs.py`
  dropped every `__init__.py`, so a package's docstring — its overview, install
  notes and usage guide — was never rendered anywhere. Package index pages now
  lead with it. `tuiml.sklearn` and `tuiml.capymoa` gained full usage guides in
  the process: how to install the extra, the two equivalent ways to reach a
  wrapper (import the class, or name it `"sklearn.<ClassName>"` in a spec),
  mixing wrapped and native components in one pipeline, benchmarking a wrapper
  against its native counterpart, and discovering parameters via
  `get_parameter_schema()`.
- **`install.sh` warns when CapyMOA is selected without Java.** The wheel
  installs fine without a JVM, then every learner fails at fit time. The
  installer's client list also matched an older registry; it now names the
  clients `tuiml setup` actually detects and points at `tuiml setup --list`.
- **`tuiml/agent/` reorganized into packages.** The 6741-line
  `tuiml/agent/tools.py` is now a `tools/` package with one module per tool,
  grouped by domain (`workflow/`, `data/`, `analysis/`, `discovery/`,
  `authoring/`, `system/`, `notebook/`). Each tool declares a single `ToolSpec`
  next to its executor; the schema tables, dispatch table, MCP annotations and
  notebook-export skip list are all derived from those specs rather than
  maintained as parallel dicts. `user_algorithms.py` was likewise split into a
  package, and `SKILL.md` moved to `tuiml/agent/prompts/`. Public entry points
  (`tuiml.agent.execute_tool`, `get_workflow_tools`, and the whole `tuiml.cli`
  surface) are unchanged.
- **`train()` / `run()`: spec-based, consistent component API.** Every ML
  component is now described the same way — a spec dict `{"name": ..., **params}`
  — for the model, each preprocessing step, and the feature selector. The data
  is its own spec, `{"source": ..., "target": ..., "features": ...}` (or
  `{"X": ..., "y": ...}` for in-memory arrays), so the file/label pairing lives
  in one place. `run(spec)` takes one declarative object with `model` / `data` /
  `preprocessing` / `feature_selection` / run-option keys. The previous forms —
  string algorithm name, positional `data`/`target`, loose hyperparameter kwargs,
  configured instances, and the legacy `run()` config schema — all still work,
  so this is fully backward compatible.

### Fixed
- **`train()` ignored the target column for file paths.** When `data` was a file
  path, `Workflow` loaded it without forwarding `target`, silently falling back
  to the last column — training on the wrong label if the class column wasn't
  last. The target is now passed through to the loader (ARFF still uses its
  declared class attribute), and a warning is emitted when a tabular file falls
  back to the last column.

### Added
- **`features=` / data-spec `"features"`** on `train()` — restrict the feature
  matrix to a named subset of columns.
- **`CategoricalNBClassifier`** — Naive Bayes for nominal / integer-coded
  categorical features (per-feature categorical distributions with Laplace
  smoothing), the discrete-data analogue of the Gaussian `NaiveBayesClassifier`.
  Matches scikit-learn's `CategoricalNB`.

### Fixed
- **MCP server: dataset resources unreadable.** The MCP SDK passes resource
  URIs as `AnyUrl` objects, not strings, so `read_resource` crashed with
  `AttributeError` on any `tuiml://dataset/...` resource; the URI is now
  coerced to `str` first.
- **Decision trees / Random Forests: NaN predictions from degenerate splits.**
  When the two feature values flanking the best split are adjacent doubles,
  the midpoint threshold `(a+b)/2` can round up to `b`; the `<=` partition
  then sends every sample left, and the empty right child's mean / class
  distribution is `0/0 = NaN`, which poisons any prediction routed to that
  leaf (seen as rare `Input contains NaN` failures on high-precision
  features). The C++ splitters now clamp the threshold to the lower value
  (as scikit-learn does), and the builders keep a node a leaf if a partition
  comes back empty.
- **`LogisticRegression` under-convergence:** the fixed-step batch gradient
  descent solver (`learning_rate=1.0`, `max_iter=100`, loss-delta stopping)
  stopped far short of the optimum on large or high-dimensional datasets,
  and the default `ridge=1e-8` was effectively unregularized (overfitting
  when p >> n). The default solver is now **L-BFGS** (SciPy) with
  `max_iter=1000` and `ridge="auto"` (= `1/n_samples`, equivalent to
  scikit-learn's `C=1.0`). The legacy solver remains available via
  `solver="gd"`.
- **`RandomForestClassifier` / `RandomForestRegressor` memory:** parallel tree
  builds each materialize a full bootstrap copy of `X`, so with `n_jobs=-1` on
  a many-core machine peak memory reached `n_workers * X.nbytes` (8-16 GB on
  the largest datasets). The number of *concurrent* builds is now bounded to a
  memory budget (default 1 GB, override with `TUIML_RF_MEM_BUDGET_MB`), cutting
  peak RSS to ~2 GB with no change to accuracy or fit time. Bootstrap indices
  are also no longer retained unless `oob_score=True`.

## [0.1.6] - 2026-06-29

### Added
- **Optional scikit-learn backend** (`tuiml.sklearn`): a curated set of
  scikit-learn estimators registered into the hub under namespaced keys
  (`sklearn.RandomForestClassifier`, …) plus a generic `SklearnAdapter` /
  `wrap_sklearn()` that wraps *any* scikit-learn-compatible estimator
  (pipelines, `GridSearchCV`, third-party). `train()` / `experiment()`
  auto-wrap a passed estimator. Install with `pip install tuiml[sklearn]`.
- **Optional CapyMOA backend** (`tuiml.capymoa`): streaming/online learners
  under `capymoa.*` keys. Install with `pip install tuiml[capymoa]`.
- Both backends are optional and lazily imported — native algorithms keep
  working with no extra dependencies; a missing backend errors only at
  instantiation, never at `import tuiml`.
- Reproducible cross-library benchmark suite (`benchmarks/`): TuiML vs
  scikit-learn vs Weka across 51 TabArena datasets.

### Changed
- `NaiveBayesClassifier` gains a `var_smoothing` parameter (scale-relative
  variance floor, scikit-learn compatible).
- `SVC` / `SVR` default `max_iter` is now `-1` (auto-scales the SMO iteration
  cap to `max(10000, n_samples)`).

### Fixed
- **Decision trees and Random Forests (classification & regression):** feature
  values and split thresholds were truncated to `float32` in prediction and
  flattening, which rounded values across split boundaries and misrouted
  samples on high-precision features — producing catastrophic, seed-dependent
  trees. Now `float64` end-to-end; results match scikit-learn.
- **`NaiveBayesClassifier`:** a fixed absolute `1e-6` variance floor plus a
  `log(prob)` underflow clamp caused collapses on standardized / one-hot data.
  Replaced with a scale-relative floor and direct log-density.
- **`SVC` / `SVR`:** the old fixed `max_iter=1000` left the SMO solver
  under-converged (degenerate but fast) on datasets larger than a few thousand
  rows; the auto-scaling default fixes this.
- **`MultilayerPerceptronRegressor`:** the gradient was averaged by `n_samples`
  three times (near-zero updates → ~0 / negative R²); corrected the averaging
  and added global-norm gradient clipping.
- **`tuiml_export_notebook`:** exported predict / evaluate / plot cells now
  re-apply the fitted preprocessing / feature-selection pipeline (via the
  `WorkflowResult`) instead of running the bare model on raw data, so the
  notebook reproduces correct results.

## [0.1.5] - 2026-06-17

### Added
- `tuiml_export_notebook` MCP tool — export the current MCP session as a
  reproducible Jupyter notebook (`.ipynb`). Training, experiment, tuning,
  plotting, and data-prep steps are translated to equivalent Python API
  calls so the workflow can be re-run without the AI client.

### Fixed
- `tuiml_export_notebook` now embeds the effective random seed. The seed is
  resolved into the call *result* (not the args), so auto-resolved seeds were
  previously dropped; exported notebooks now fold it into train/experiment/
  tune cells and emit a `set_global_seed()` cell for full reproducibility.
- Silenced spurious "Component '...' is already registered. Overwriting."
  warnings on intentional user-algorithm re-registration (new version,
  restart bootstrap). Genuine name clashes still warn, now on stderr so the
  message can no longer corrupt the MCP stdio JSON-RPC stream.

## [0.1.4] - 2026-05-20

### Added
- C++ pybind11 kernels for SGD (classifier + regressor),
  agglomerative hierarchical clustering, and Gaussian Mixture EM —
  closing large performance gaps vs sklearn (see Performance).
- MCP setup support for 6 new clients: Gemini CLI, Cline, Roo Code,
  Kilo Code, OpenCode (custom `mcp.<name>` schema), and Antigravity.
- `path` field in `tuiml_plot` MCP response — plots are now persisted
  to `~/.tuiml/plots/` (override with `$TUIML_PLOT_DIR`) so agents
  can embed them in markdown reports via `![](/path/to/plot.png)`.

### Changed
- `plot_roc_curve` now handles multiclass input: draws per-class
  one-vs-rest curves with class labels in the legend, plus a
  dotted macro-average overlay. Previously silently took
  `probas[:, 1]` and plotted a single curve, giving misleading
  AUC=1.000 on 3-class problems like iris.
- AdaBoost: vectorised the prediction aggregation loop
  (241× → 14× slower vs sklearn on 10k×20 5-class data).
- RandomForest: default `n_jobs=-1` so all CPU cores are used.

### Fixed
- `np.trapz` → `np.trapezoid` in `evaluation.metrics.classification`;
  `roc_auc_score` was crashing on NumPy >= 2.0.

### Performance
- SGDRegressor: 278× → 2.4× slower vs sklearn (C++ kernel; was
  pure-Python mini-batch loop with per-sample allocations).
- Gaussian Mixture (EM, full covariance): 292× → 1.4× slower vs
  sklearn (C++ kernel with pre-allocated scratch + manual Cholesky).
- Hierarchical (ward, 1k samples): 96× → 7× slower vs sklearn
  (C++ kernel with min-heap + condensed distance matrix +
  Lance-Williams updates).

## [0.1.3] - 2026-05-20

### Added
- `tuiml uninstall` CLI command and Auto/Manual mode menu in `tuiml setup`.
- `tuiml_system_info` and `tuiml_self_update` MCP tools.
- Agent-authored algorithm MCP tools with versioned aliases.
- Lineage store and `tuiml_research_log` for agent-authored algorithm history.
- NemoClaw (NVIDIA) client and generic `instructions` kind for MCP registration.
- OpenClaw kind for CLI-based MCP registration in `tuiml setup`.
- Native Linux aarch64 wheel builds via ARM runners.

### Changed
- Renamed `tuiml.llm` → `tuiml.agent` with framework adapters.
- Clarified example prompts with "using TuiML" suffix.
- Agent module docs no longer hardcode tool count.

## [0.1.2] - 2026-04-22

### Added
- `tuiml setup` CLI wizard for configuring MCP clients (Claude Code, Claude Desktop, Cursor, OpenClaw, Perplexity Desktop, and more) with auto-detection of installed clients.
- Star History chart and download badge in README.

### Changed
- Repositioned tuiml as an agent-native ML runtime; slimmed README to focus on agent workflows over Python/CLI examples.
- Improved algorithm defaults and extended benchmarks to 10k samples.
- Updated SVM kernel handling and neighbor search internals.
- Enhanced workflow preprocessing and switched Python badge to show ≥3.10.

### Fixed
- XGBoost and LightGBM wrappers no longer require scikit-learn to be installed: both now use their native training APIs (`xgb.train` + `DMatrix`, `lgb.train` + `Dataset`) instead of the sklearn-compatible estimator wrappers.
- MCP server upload/train/serve path resolution issues.
- Author display on the PyPI sidebar.
- Logo image source in README.

## [0.1.1] - 2026-03-15

### Added
- 100+ machine learning algorithms across 13 categories: trees, linear, bayesian, clustering, ensemble, gradient boosting, SVM, neural, neighbors, rules, associations, time series, and anomaly detection.
- High-level API with one-liner `train()`, `predict()`, `experiment()`, and `run()` functions.
- Fluent `Workflow` builder for chainable ML pipelines with preprocessing, feature engineering, and evaluation.
- Comprehensive preprocessing module: scaling, encoding, imputation, discretization, outlier handling, SMOTE, and text vectorizers.
- Feature engineering: PCA, random projection, univariate selection, sequential selection, variance threshold, polynomial and mathematical feature generation.
- Built-in datasets: iris, diabetes, wine, breast cancer, glass, ionosphere, soybean, and segment challenge.
- Dataset generators for classification (Agrawal, Hyperplane, LED, Random RBF), clustering (blobs, circles, moons, swiss roll), and regression (Friedman, Mexican Hat, Sine).
- Data loaders for ARFF, CSV, JSON, Excel, Parquet, NumPy, and pandas formats.
- LLM integration via Model Context Protocol (MCP) server with 200+ tools for agentic ML workflows.
- CLI for training, prediction, evaluation, experiments, model serving, and hub operations.
- WekaHub community platform for algorithm discovery, publishing, and benchmarking.
- Dataset Hub for browsing, downloading, and sharing datasets with the community.
- 21 interactive Jupyter tutorials across 6 learning tracks: Quickstart, ML Simplified, LLM Friendly, Community, Deploy, and Case Studies.
- Full API documentation generated from NumPy-style docstrings with KaTeX math rendering.
- Base classes (`Classifier`, `Regressor`) with scikit-learn compatible fit/predict interface and `@classifier`/`@regressor` decorators.
- Model serialization via joblib with save/load utilities.
- Cross-validation, grid search, and hyperparameter tuning support.

[0.1.9]: https://github.com/tuiml/tuiml/releases/tag/v0.1.9
[0.1.8]: https://github.com/tuiml/tuiml/releases/tag/v0.1.8
[0.1.7]: https://github.com/tuiml/tuiml/releases/tag/v0.1.7
[0.1.6]: https://github.com/tuiml/tuiml/releases/tag/v0.1.6
[0.1.5]: https://github.com/tuiml/tuiml/releases/tag/v0.1.5
[0.1.4]: https://github.com/tuiml/tuiml/releases/tag/v0.1.4
[0.1.3]: https://github.com/tuiml/tuiml/releases/tag/v0.1.3
[0.1.2]: https://github.com/tuiml/tuiml/releases/tag/v0.1.2
[0.1.1]: https://github.com/tuiml/tuiml/releases/tag/v0.1.1
