# TuiML Python API Reference

Read this when writing Python against TuiML (scripts, notebooks, custom
components). For MCP tool calls see `mcp-tools.md`; for the shell see `cli.md`.

## Contents
- Three API levels and the two dialects (Python spec vs MCP args)
- High-level API: train(), Benchmark, serve(), PRESETS, discovery
- Workflow, the pipeline (fit, fitted attributes, On per-column steps)
- Algorithms (13 families, exact class names)
- Datasets (built-in, file loaders, generators, Dataset object)
- Preprocessing, Feature engineering
- Evaluation (metrics, splitting, tuning, statistics, plots)
- Building custom components
- Local registry and optional backends (sklearn, capymoa, weka)
- Common patterns

### Three API Levels

All three levels drive the same engine, a pipeline whose last step is the model.

```python
# High-level: one declarative spec (strings + dicts, fully serializable)
from tuiml import train
model = train({
    "model": {"name": "RandomForestClassifier"},
    "data": {"source": "iris"},
    "evaluation": {"cv": 10},
})

# Mid-level: a pipeline of objects (imports give autocomplete + type checking)
from tuiml import Workflow
from tuiml.preprocessing import SimpleImputer, StandardScaler
from tuiml.algorithms.trees import RandomForestClassifier
wf = Workflow([SimpleImputer(), StandardScaler(),
               RandomForestClassifier(n_estimators=100)])
wf.fit("iris", cv=10, metrics=["accuracy_score", "f1_score"])
print(wf.metrics_)

# Low-level: full OOP control
from tuiml.algorithms.trees import RandomForestClassifier
from tuiml.datasets import load_iris
from tuiml.evaluation import accuracy_score, train_test_split

data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(data.X, data.y, test_size=0.2)
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)
print(accuracy_score(y_test, clf.predict(X_test)))
```

### Two dialects: Python specs vs MCP tool arguments

TuiML is driven two ways, and **they take different shapes for the same
information**. Mixing them is the most common source of errors, so fix which
one you are in before writing a call.

| | **Python spec** (`train`, `Benchmark`) | **MCP tool args** (`tuiml_train`, …) |
|---|---|---|
| Shape | nested, strict | flat, permissive |
| Component | `{"name": "X", "params": {...}}` only | `"X"` **or** `{"name": "X", <params inline>}` |
| Hyperparameters | inside `"params"` | inline, or `algorithm_params` |
| Evaluation | `{"evaluation": {"cv": 10}}` | `cv=10` as a top-level argument |
| Bare strings | rejected | accepted |
| Loose keys | rejected | accepted |

```python
# Python: strict and nested. Every component is {"name", "params"}.
train({
    "model": {"name": "SVC", "params": {"C": 2.0}},
    "data": {"source": "data.csv", "target": "label"},
    "pipeline": [{"name": "SimpleImputer", "params": {"strategy": "median"}}],
    "evaluation": {"cv": 10},
})
```

```python
# MCP: flat. The same run, as tool arguments.
execute_tool("tuiml_train",
             algorithm="SVC", algorithm_params={"C": 2.0},
             data="data.csv", target="label",
             preprocessing=[{"name": "SimpleImputer", "strategy": "median"}],
             cv=10)
```

Both are correct **in their own dialect**. The flat form exists because models
emit flat schemas more reliably; the strict form exists so an agent-authored
plan is serializable and replayable. Common mistakes:

```python
train(spec, cv=5)                                    # ✗ train() takes ONE argument
train({..., "evaluation": {"cv": 5}})                # ✓

train({"model": "SVC", ...})                         # ✗ bare string in a Python spec
train({"model": {"name": "SVC"}, ...})               # ✓

train({"model": {"name": "SVC", "C": 2.0}, ...})     # ✗ loose key
train({"model": {"name": "SVC", "params": {"C": 2.0}}, ...})   # ✓

from sklearn.svm import SVC
train({"model": SVC(C=2.0), ...})                    # ✗ raw estimator objects are rejected
train({"model": {"name": "sklearn.SVC", "params": {"C": 2.0}}, ...})   # ✓ namespaced wrapper
Workflow([SVC(C=2.0)])                               # ✓ instances belong to Workflow
```

Note also two similarly-named parameters that are **not** interchangeable:
`random_seed` is the run-level seed (spec, `Workflow.fit`, CLI `--random-seed`);
`random_state` is a constructor parameter on individual algorithms
(`RandomForestClassifier(random_state=42)`).

---

## 2. High-Level API

All top-level functions are importable from `tuiml` directly.

```python
from tuiml import (
    train, Benchmark,                    # Benchmark is a CLASS; there is no `benchmark` function
    list_algorithms, describe_algorithm, search_algorithms,
    serve, stop_server, server_status,
    PRESETS,
    Workflow, On,
    agent,
    registry, ComponentType,
)
```

`predict` / `evaluate` / `save` / `load` are **methods on the model**, not
top-level functions, see §3. `run()` is gone: pass the spec dict straight to
`train()`.

### train()

`train()` takes exactly **one argument**: the spec dict (or a path to a
`.json` file holding it). Every component is `{"name": ..., "params": {...}}`.
Hyperparameters go inside `"params"`: loose keys are rejected, and so are
unknown top-level keys.

```python
model = train({
    "model": {"name": "RandomForestClassifier", "params": {"n_estimators": 100}},
    "data": {"source": "sales.csv", "target": "label"},
    #   data also accepts {"X": X_array, "y": y_array} for in-memory arrays,
    #   and "features": [...] restricts the input columns.
    "pipeline": [                            # ordered steps, all feature engineering
        {"name": "SimpleImputer", "params": {"strategy": "mean"}},
        {"name": "StandardScaler"},
        {"name": "PCAExtractor", "params": {"n_components": 5}},
        {"name": "SelectKBestSelector", "params": {"k": 10}},
    ],
    "evaluation": {                          # cv OR test_size/stratify, plus metrics
        "cv": 10,
        "metrics": ["accuracy_score", "f1_score"],
    },
    "random_seed": 42,
})
model = train("spec.json")                   # same spec from a file
# Returns a FITTED Workflow: model.metrics_, model.predict(X), model.save(path)
```

Spec keys: `model` (required), `data` (required), `target`, `features`,
`pipeline`, `evaluation`, `random_seed`. The model and every pipeline step
use one shape only: `{"name": ..., "params": {...}}` (`params` optional). To work with imported classes and
configured instances instead of strings, use `Workflow` (section 3).

### Benchmark

Compare models across datasets with per-fold scoring. Configure the class,
then execute with `.run()`. Model specs are strict `{"name", "params"}` dicts
and dataset specs are always dicts (bare strings are rejected).

```python
from tuiml import Benchmark

bench = Benchmark(
    models=[                                   # strict {"name","params"} specs
        {"name": "NaiveBayesClassifier"},
        {"name": "RandomForestClassifier", "params": {"n_estimators": 100}},
        {"name": "DecisionTreeClassifier",          # optional per-model tuning (nested,
         "tune": {"method": "grid",            #  runs on training folds only;
                  "space": {"max_depth": [3, 5, 8]}}},   # also "random" + "iterations")
        {"name": "SVC",                        # optional per-model pipeline
         "pipeline": [{"name": "StandardScaler"}],
         "label": "SVC-scaled"},               # optional display name
    ],
    datasets=[                                 # ALWAYS spec dicts
        {"source": "iris"},                    # builtin (defines its own target)
        {"source": "glass", "features": ["RI", "Na"]},
        {"source": "data.csv", "target": "label", "features": [...], "name": "sales"},
        {"name": "custom", "X": X, "y": y},    # in-memory arrays
    ],
    pipeline=[{"name": "SimpleImputer"}],      # shared steps or preset name
    evaluation={"cv": 10,                      # OR {"test_size": 0.2}
                "repeats": 3,                  # repeated scheme -> spread + stats
                "metrics": ["accuracy_score", "f1_score"]},
    random_seed=42,                            # fully reproducible
    task=None,                                 # optional override; inferred from models
)
bench.run()                                    # executes; returns self
```

Results live on the instance (all pandas DataFrames where tabular):

```python
bench.scores_                    # tidy per-fold df: dataset|model|metric|fold|value
bench.table(metric=None, formatted=True)     # datasets x models, "mean ± std"
bench.best(metric=None)          # winner per dataset + overall mean rank
bench.best_params_               # tuning choices per dataset/model/fold
bench.compare(baseline="...", metric=None)   # paired t-test table (readable df)
bench.summary()                  # text report, best marked
bench.to_markdown() / bench.to_latex() / bench.to_csv() / bench.to_html()
bench.plot_critical_difference() / bench.plot_boxplot() / bench.plot_ranking()
bench.task, bench.metrics, bench.random_seed
```

### serve()

```python
# Returns a dict (not a URL string) when background=True; None when background=False
info = serve(
    model_or_path,           # file path, fitted Workflow, or any object with .predict()
    host="127.0.0.1",
    port=8000,
    model_id="default",
    background=True,         # False blocks the process
)
# info = {"server_id": "127.0.0.1:8000", "url": "http://...", "endpoints": {...}}

stop_server(server_id=None)              # server_id format: "host:port" e.g. "127.0.0.1:8000"
                                         # None stops all running servers
```

### PRESETS

```python
PRESETS = {
    "minimal":    [],                                               # no steps
    "fast":       SimpleImputer(strategy="most_frequent"),
    "standard":   SimpleImputer(mean) → MinMaxScaler → OneHotEncoder,
    "full":       SimpleImputer(median) → StandardScaler → OneHotEncoder → SelectKBestSelector(k=10),
    "imbalanced": SimpleImputer(mean) → MinMaxScaler → SMOTESampler,
}
# Pass the name as the pipeline: train(..., pipeline="standard")
# Each preset is a plain step list, so PRESETS["standard"] + [...] composes.
```

### Discovery

```python
algorithms = list_algorithms(type=None)  # type: "classifier", "regressor", etc.
info = describe_algorithm("RandomForestClassifier")
results = search_algorithms("ensemble")
```

---

## 3. Workflow, the pipeline

A `Workflow` is an ordered list of steps whose **last step is the model**.
Everything before it transforms. A `Workflow` *is* a model: same
`fit`/`predict`/`score`/`save` interface as any single algorithm, so it works
anywhere one does, including nested inside another `Workflow`.

Steps are **configured instances**. Strings and spec dicts belong to
`train()`; passing them here raises a `TypeError` pointing you there.

```python
from tuiml import Workflow
from tuiml.preprocessing import StandardScaler
from tuiml.algorithms.trees import RandomForestClassifier
from tuiml.algorithms.svm import SVC

Workflow([StandardScaler(), RandomForestClassifier(n_estimators=100)])
Workflow([("scale", StandardScaler()), ("clf", SVC())])   # explicit step names
```

### fit(): the only execution method

`fit()` always leaves the pipeline fitted on **all** the data given, and
returns `self`. Passing `cv` or `test_size` *additionally* measures held-out
performance into `metrics_` first.

```python
from tuiml.preprocessing import SimpleImputer, StandardScaler
from tuiml.algorithms.trees import RandomForestClassifier

wf = Workflow([SimpleImputer(), StandardScaler(), RandomForestClassifier()])

wf.fit(X, y)                                      # plain fit, no evaluation
wf.fit("sales.csv", target="label", cv=10)        # k-fold, then refit on all
wf.fit("iris", test_size=0.2, stratify=True)      # holdout, then refit on all
wf.fit("iris", features=["sepallength"], metrics=["accuracy_score"], random_seed=42)
```

### Results, fitted attributes (trailing underscore)

```python
wf.metrics_            # held-out scores, or None if no cv/test_size was given
wf.cv_results_         # per-fold scores
wf.predictions_        # predictions on the held-out split
wf.model_              # the fitted final model
wf.steps_              # the fitted transformation steps
wf.feature_names_in_   # training column names, when known
wf.metadata_           # algorithm, step names, evaluation method, n_samples
```

### Use it like a model

```python
wf.predict(X)          # transforms through the fitted steps, then predicts
wf.predict_proba(X)
wf.score(X, y)         # accuracy (classifier) / R² (regressor)
wf.evaluate(X, y, metrics=["accuracy_score", "f1_score"])
wf.save("pipeline.pkl")            # the whole pipeline, one file
wf = Workflow.load("pipeline.pkl")
wf.serve(port=8000)
```

### Inspect and tune

Step names are the lowercased class name; duplicates get `-2`, `-3` suffixes.
The `step__param` path is what lets a whole pipeline be tuned, not just a model.

```python
wf.named_steps         # {"simpleimputer": ..., "randomforestclassifier": ...}
wf[0], wf[-1], wf["standardscaler"]
wf.model               # the final step (unfitted prototype)
wf.transformers        # [(name, component), ...] before the model

wf.set_params(randomforestclassifier__n_estimators=200, pcaextractor__n_components=3)
wf.get_params()        # includes nested "<step>__<param>" keys

wf.to_config()         # export back to a train() spec dict
wf._repr_html_()       # pipeline diagram (renders automatically in notebooks)
```

### On, per-column steps

Mixed-type tables need different treatment per column. `On` wraps a
transformer so it sees only the columns you name, and is itself a normal step.

```python
from tuiml import Workflow, On
from tuiml.preprocessing import (
    MinMaxScaler, OneHotEncoder, SimpleImputer, StandardScaler,
)
from tuiml.algorithms.trees import RandomForestClassifier

Workflow([
    On("number", SimpleImputer()),              # every numeric column
    On("category", OneHotEncoder()),            # every non-numeric column
    On(["age", "fare"], StandardScaler()),      # these named columns
    On([0, 3], MinMaxScaler()),                 # these positions
    On("number", StandardScaler(), remainder="drop"),   # discard the rest
    RandomForestClassifier(),
])
```

Transformed columns come first in the output, then the untouched ones
(`remainder="passthrough"`, the default). Selecting by **name** needs the
original column names, so put name-based `On` steps first, before anything
that changes the column layout. Type- and position-based selection works
anywhere.

---

## 4. Algorithms

13 algorithm families with exact class names as imports.

```python
from tuiml.algorithms import (
    # Bayesian
    BayesianLinearRegressor, CategoricalNBClassifier,
    GaussianProcessesRegressor, NaiveBayesClassifier,
    NaiveBayesMultinomialClassifier,
    # Trees
    DecisionStumpClassifier, DecisionTreeClassifier, DecisionTreeRegressor,
    RandomForestClassifier, RandomForestRegressor,
    # Neighbors
    KNearestNeighborsClassifier, KNearestNeighborsRegressor,
    # Linear
    LinearRegression, LogisticRegression, SGDClassifier, SGDRegressor,
    # SVM
    SVC, SVR,
    # Neural
    MultilayerPerceptronClassifier, MultilayerPerceptronRegressor,
    PerceptronClassifier,
    # Rules
    ZeroRuleClassifier,
    # Ensemble
    AdaBoostClassifier, AdaBoostRegressor, BaggingClassifier,
    BaggingRegressor, GradientBoostingRegressor, OneVsRestClassifier,
    StackingClassifier, StackingRegressor, VotingClassifier,
    VotingRegressor,
    # Gradient Boosting
    CatBoostClassifier, CatBoostRegressor, LightGBMClassifier,
    LightGBMRegressor, XGBoostClassifier, XGBoostRegressor,
    # Clustering
    AgglomerativeClusterer, DBSCANClusterer, GaussianMixtureClusterer,
    KMeansClusterer,
    # Associations
    AprioriAssociator, ECLATAssociator, FPGrowthAssociator,
    # Anomaly
    EllipticEnvelopeDetector, IsolationForestDetector,
    LocalOutlierFactorDetector, OneClassSVMDetector,
    # Time Series
    AR, ARIMA, ARMA, ExponentialSmoothing, MA, Prophet, STLDecomposition,
)
```

### Usage Pattern

All algorithms follow the same fit/predict interface:

```python
clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)     # classifiers only

# Fitted attributes (trailing underscore convention)
clf.classes_
clf.n_features_
clf.oob_score_       # algorithm-specific

# Static metadata
schema = RandomForestClassifier.get_parameter_schema()
caps = RandomForestClassifier.get_capabilities()
meta = RandomForestClassifier.get_metadata()
```

### Clustering (unsupervised)

```python
km = KMeansClusterer(n_clusters=3)
km.fit(X)
labels = km.predict(X)
```

### Association Rules

```python
assoc = AprioriAssociator(min_support=0.3, min_confidence=0.7)
assoc.fit(transactions)
rules = assoc.association_rules_   # list of AssociationRule objects
```

### Time Series

```python
model = ARIMA(order=(1, 1, 1))
model.fit(y_time_series)              # historical values in time order
forecast = model.predict(steps=10)    # next ten observations
```

`Benchmark` recognises forecasters (models tagged `timeseries`) and never
shuffles them: `{"cv": 5}` gives five expanding-window folds
(`TimeSeriesSplit`), `{"test_size": 0.2}` a single tail holdout, and the
default metrics are `mean_absolute_error` and `root_mean_squared_error`.
The series is the dataset's target column. `tune` blocks are rejected for
forecasters.

```python
from tuiml import Benchmark
bench = Benchmark(
    models=[{"name": "ThetaForecaster"}, {"name": "ARIMA", "params": {"order": (1, 1, 1)}}],
    datasets=[{"source": "sales.csv", "target": "units"}],
    evaluation={"cv": 5},
).run()
```

### Pretrained foundation models (`tuiml.foundation`)

Namespaced `foundation.*` keys wrap pretrained checkpoints: nothing is
trained on your data. They need `pip install 'tuiml[foundation]'` (pulls
PyTorch); construction and `get_parameter_schema()` work without it, `fit`
raises an `ImportError` naming the install command. Weights download on
first use into `~/.cache/huggingface`; TuiML ships none, and only models
with BSD/Apache-licensed weights are included.

```python
from tuiml.foundation import (
    TabICLClassifier, TabICLRegressor,       # tabular, in-context learning
    ChronosForecaster, TimesFMForecaster,    # zero-shot time series
)

clf = TabICLClassifier(n_estimators=4).fit(X_train, y_train)
proba = clf.predict_proba(X_test)

fm = ChronosForecaster(checkpoint="amazon/chronos-bolt-small")   # or chronos-bolt-base, chronos-2
fm.fit(y_time_series)                 # stores the context, loads the checkpoint
point = fm.predict(steps=12)          # median forecast
q = fm.predict_quantiles(steps=12)    # (12, 9): deciles 0.1 … 0.9, see fm.quantile_levels_

tfm = TimesFMForecaster(context_length=512).fit(y_time_series)
tfm.predict(steps=12)
```

Pickling a fitted forecaster stores only the context, never the
checkpoint, so saved models stay a few kilobytes.

---

## 5. Datasets

### Built-in Datasets

```python
from tuiml.datasets import (
    load_iris, load_diabetes, load_breast_cancer,
    load_glass, load_ionosphere, load_vote, load_credit,
    load_soybean, load_labor, load_hypothyroid, load_contact_lenses,
    load_cpu,
    list_datasets, load_dataset, get_dataset_info, get_datasets_by_task,
    DATASET_REGISTRY,
)

data = load_iris()
X, y = data.X, data.y
print(data.feature_names, data.n_samples, data.n_features)

# Load by name string
data = load_dataset("iris")

# Browse
all_datasets = list_datasets()
classification_sets = get_datasets_by_task("classification")
info = get_dataset_info("iris")
```

### File Loaders

```python
from tuiml.datasets import (
    load, save,                                # auto-detect by extension
    load_csv, save_csv,
    load_arff, save_arff,
    load_excel, save_excel, load_excel_sheets,
    load_parquet, save_parquet, load_parquet_partitioned,
    load_json, save_json, load_jsonl, save_jsonl, load_json_nested,
    load_numpy, save_numpy,
    load_pandas, from_pandas, to_pandas,
    Dataset,
)

data = load("data.csv")            # auto-detect
data = load_csv("data.csv")        # explicit
data = load_arff("data.arff")
data = load_parquet("data.parquet")

# Pandas interop
df = to_pandas(data)
data = from_pandas(df, target_column="class")

# Save
save(data, "output.csv")
save_parquet(data, "output.parquet")
```

### Synthetic Data Generators

```python
from tuiml.datasets import (
    RandomRBF, Agrawal, LED, Hyperplane,       # classification
    Friedman, MexicanHat, Sine,                # regression
    Blobs, Moons, Circles, SwissRoll,          # clustering
)

data = Blobs(n_samples=1000, n_clusters=5, random_state=42).generate()
X, y = data.X, data.y
```

### Dataset Object

All loaders return a `Dataset` with:

```python
data.X              # np.ndarray (n_samples, n_features)
data.y              # np.ndarray (n_samples,) or None
data.feature_names  # list of column names
data.target_names   # list of class labels (classification)
data.name           # dataset name
data.description    # metadata string
data.n_samples      # row count
data.n_features     # column count
data.shape          # (n_samples, n_features)
```

---

## 6. Preprocessing

All preprocessors follow the fit/transform pattern. Import from `tuiml.preprocessing`.

### Scaling

```python
from tuiml.preprocessing import StandardScaler, MinMaxScaler, CenterScaler

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)
```

### Imputation

```python
from tuiml.preprocessing import SimpleImputer, KNNImputer

imputer = SimpleImputer(strategy="mean")   # mean, median, most_frequent
X_clean = imputer.fit_transform(X_train)
```

### Encoding

```python
from tuiml.preprocessing import (
    OneHotEncoder, OrdinalEncoder, LabelEncoder,
    RareCategoryEncoder,
)

encoder = OneHotEncoder()
X_enc = encoder.fit_transform(X_train)
```

### Discretization

```python
from tuiml.preprocessing import (
    EqualWidthDiscretizer, QuantileDiscretizer, MDLDiscretizer,
)

disc = QuantileDiscretizer(n_bins=10)
X_disc = disc.fit_transform(X_train)
```

### Sampling & Class Balancing

```python
from tuiml.preprocessing import (
    # SMOTE family
    SMOTESampler, BorderlineSMOTESampler, ADASYNSampler,
    SVMSMOTESampler, KMeansSMOTESampler,
    # Oversampling
    RandomOverSampler, ClusterOverSampler,
    # Undersampling
    RandomUnderSampler, TomekLinksSampler, ENNSampler, CNNSampler,
    NearMissSampler, HardnessThresholdSampler,
    # Utility
    ClassBalanceSampler, ReservoirSampler,
)

smote = SMOTESampler(k_neighbors=5)
X_bal, y_bal = smote.fit_resample(X_train, y_train)
```

### Outliers

```python
from tuiml.preprocessing import IQROutlierDetector, ValueClipper

detector = IQROutlierDetector(multiplier=1.5)
X_clean = detector.fit_transform(X_train)
```

### Text

```python
from tuiml.preprocessing import (
    WordTokenizer, NGramTokenizer, RegexTokenizer, SentenceTokenizer,
    CountVectorizer, TfidfVectorizer, TfidfTransformer, HashingVectorizer,
    TextCleaner, StopWordRemover, Stemmer,
)

tfidf = TfidfVectorizer(max_features=1000)
X_tfidf = tfidf.fit_transform(documents)
```

### Time Series

```python
from tuiml.preprocessing import LagTransformer, DifferenceTransformer

lag = LagTransformer(n_lags=5)
X_lagged = lag.fit_transform(X_ts)
```

---

## 7. Feature Engineering

### Selection

```python
from tuiml.features.selection import (
    SelectKBestSelector, SelectPercentileSelector,
    SelectThresholdSelector, SelectFprSelector,
    CFSSelector, SequentialFeatureSelector,
    VarianceThresholdSelector, RandomSubsetSelector,
    BootstrapFeaturesSelector,
)

selector = SelectKBestSelector(score_func=information_gain, k=10)
X_sel = selector.fit_transform(X_train, y_train)
indices = selector.get_support(indices=True)
```

### Extraction

```python
from tuiml.features.extraction import PCAExtractor, RandomProjectionExtractor

pca = PCAExtractor(n_components=0.95)    # 95% variance retained
X_pca = pca.fit_transform(X_train)
```

### Generation

```python
from tuiml.features.generation import (
    PolynomialFeaturesGenerator, MathematicalFeaturesGenerator,
)

poly = PolynomialFeaturesGenerator(degree=2)
X_poly = poly.fit_transform(X_train)
```

---

## 8. Evaluation

Everything is importable from `tuiml.evaluation`.

### Metrics

```python
from tuiml.evaluation import (
    # Classification
    accuracy_score, balanced_accuracy_score,
    precision_score, recall_score, f1_score, fbeta_score,
    precision_recall_fscore_support, matthews_corrcoef, cohen_kappa_score,
    roc_auc_score, roc_curve, auc, log_loss,
    confusion_matrix, classification_report,
    # Regression
    mean_absolute_error, mean_squared_error, root_mean_squared_error, r2_score,
    # Clustering
    silhouette_score, adjusted_rand_score,
    # Information-theoretic
    entropy, mutual_information, information_gain,
    # Base
    Metric, MetricType, AverageType,
)

acc = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred, average="weighted")
cm = confusion_matrix(y_true, y_pred)
```

### Splitting

```python
from tuiml.evaluation import (
    train_test_split, cross_val_score,
    KFold, StratifiedKFold, RepeatedKFold, RepeatedStratifiedKFold,
    HoldoutSplit, StratifiedHoldoutSplit,
    LeaveOneOut, LeavePOut,
    BootstrapSplit, TimeSeriesSplit,
    GroupKFold, StratifiedGroupKFold,
    ShuffleSplit, StratifiedShuffleSplit,
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
scores = cross_val_score(model, X, y, cv=10, scoring="f1_weighted")
```

### Tuning

```python
from tuiml.evaluation import (
    GridSearchCV, RandomSearchCV,
    ParameterGrid, ParameterDistribution,
)

grid = GridSearchCV(
    estimator=RandomForestClassifier(),
    param_grid={"n_estimators": [50, 100, 200], "max_depth": [5, 10, None]},
    cv=5,
    scoring="f1_weighted",
)
grid.fit(X_train, y_train)
print(grid.best_params_, grid.best_score_)
```

### Benchmarking

Multi-model comparison lives in the `tuiml.Benchmark` class (section 2). The old
`Experiment` / `run_experiment` / `ExperimentConfig` names are gone from
`tuiml.evaluation`.

```python
import tuiml

bench = tuiml.Benchmark(
    models=[
        {"name": "RandomForestClassifier"},
        {"name": "SVC"},
        {"name": "NaiveBayesClassifier"},
    ],
    datasets=[{"name": "iris", "X": X, "y": y}],
    evaluation={"cv": 10,
                "metrics": ["accuracy_score", "f1_score"]},  # exact function names
)                                                            # from tuiml.evaluation.metrics
bench.run()
print(bench.to_latex())
print(bench.to_markdown())
```

### Statistics

```python
from tuiml.evaluation import (
    paired_t_test, corrected_paired_t_test, one_way_anova,
    wilcoxon_signed_rank_test, friedman_test,
    nemenyi_post_hoc, bonferroni_correction,
    holm_correction, benjamini_hochberg,
    SignificanceLevel,
)

# paired_t_test returns a PairedStats OBJECT, not a tuple - do not unpack it
stats = paired_t_test(scores_a, scores_b)
print(stats.t_statistic, stats.p_value, stats.significance)

# The multi-model tests take a DICT of {model_name: fold_scores}, not a list,
# and friedman_test returns THREE values.
results = {"rf": scores_a, "svc": scores_b, "nb": scores_c}
chi2, p_value, significant = friedman_test(results)
pairwise = nemenyi_post_hoc(results, significance_level=0.05)   # {(m1, m2): is_significant}
```

### Visualization

```python
from tuiml.evaluation import (
    plot_roc_curve, plot_pr_curve, plot_confusion_matrix,
    plot_learning_curve, plot_critical_difference,
    plot_ranking_table, plot_boxplot_comparison, plot_heatmap,
)

plot_confusion_matrix(y_true, y_pred, filename="cm.png")
plot_roc_curve(y_true, y_proba, filename="roc.png")
plot_critical_difference(results_matrix, filename="cd.png")
```

### Reporting

Result tables come straight off a finished ``Benchmark``:

```python
b.to_markdown()   # also: b.to_latex(), b.to_html()
```

---

## 9. Building Custom Components

### Custom Algorithm

```python
from tuiml.base.algorithms import Classifier, classifier
import numpy as np

@classifier(tags=["custom"], version="1.0.0")
class MyClassifier(Classifier):
    """My custom classifier.

    Parameters
    ----------
    k : int, default=5
        Number of neighbors.
    """

    def __init__(self, k=5):
        super().__init__()
        self.k = k

    def fit(self, X, y):
        """Fit the model."""
        self.classes_ = np.unique(y)
        self.X_train_ = np.asarray(X)
        self.y_train_ = np.asarray(y)
        self._is_fitted = True
        return self

    def predict(self, X):
        """Predict class labels."""
        self._check_is_fitted()
        # prediction logic
        return predictions

# Automatically registered and discoverable via tuiml_list, tuiml_train, tuiml_benchmark
```

### Custom Preprocessor

```python
from tuiml.base.preprocessing import Transformer, transformer
import numpy as np

@transformer(tags=["custom", "scaling"], version="1.0.0")
class MyScaler(Transformer):
    """Custom scaler."""

    def fit(self, X, y=None):
        """Learn parameters."""
        self.mean_ = np.mean(X, axis=0)
        self._is_fitted = True
        return self

    def transform(self, X):
        """Apply transformation."""
        self._check_is_fitted()
        return X - self.mean_
```

### Custom Feature Selector

```python
from tuiml.base.features import FeatureSelector, feature_selector
import numpy as np

@feature_selector(tags=["custom"], version="1.0.0")
class MySelector(FeatureSelector):
    """Custom feature selector."""

    def __init__(self, threshold=0.01):
        self.threshold = threshold

    def fit(self, X, y=None):
        """Identify features to keep."""
        self.variances_ = np.var(X, axis=0)
        self.support_ = self.variances_ > self.threshold
        self._is_fitted = True
        return self

    def transform(self, X):
        """Return selected features."""
        self._check_is_fitted()
        return np.asarray(X)[:, self.support_]
```

### Custom Metric

```python
from tuiml.base.metrics import Metric, MetricType

class MyMetric(Metric):
    """Custom evaluation metric."""

    def __init__(self):
        super().__init__("my_metric", MetricType.CLASSIFICATION)

    def compute(self, y_true, y_pred, **kwargs):
        """Compute the metric value."""
        correct = sum(a == b for a, b in zip(y_true, y_pred))
        return correct / len(y_true)
```

### Base Classes Reference

| Component | Base Class | Decorator | Required Methods |
|-----------|-----------|-----------|-----------------|
| Classifier | `Classifier` | `@classifier` | `fit()`, `predict()` |
| Regressor | `Regressor` | `@regressor` | `fit()`, `predict()` |
| Clusterer | `Clusterer` | `@clusterer` | `fit()`, `predict()` |
| Associator | `Associator` | `@associator` | `fit()` |
| Feature Selector | `FeatureSelector` | `@feature_selector` | `fit()`, `transform()` |
| Feature Extractor | `FeatureExtractor` | `@feature_extractor` | `fit()`, `transform()` |
| Feature Constructor | `FeatureConstructor` | `@feature_constructor` | `fit()`, `transform()` |
| Preprocessor | `Preprocessor` | `@preprocessor` | `fit()`, `transform()` |
| Transformer | `Transformer` | `@transformer` | `fit()`, `transform()` |
| Resampling Transformer | `ResamplingTransformer` | `@transformer` | `fit()`, `transform()` |
| Metric | `Metric` | (none) | `compute()` |

---

## 11. Local Registry

```python
from tuiml.registry import registry, ComponentType

# Local, in-process registry (no network calls)
classifiers = registry.list(ComponentType.CLASSIFIER)          # metadata dicts
model = registry.create("RandomForestClassifier", n_estimators=100)
cls = registry.get("RandomForestClassifier")                   # the class itself
names = registry.list_names(ComponentType.CLASSIFIER)
```

The remote community hub is currently decommissioned, use agent-authored algorithms (section 12 + 14) to add algorithms to the registry at runtime instead.

### Optional algorithm backends (scikit-learn, CapyMOA, Weka)

Native TuiML algorithms always work with no extra dependencies. Three
**optional** backends add the external ecosystems, each as a separate package
that registers into the same hub under a **namespaced key** (so they never
collide with native names):

```bash
pip install tuiml[sklearn]    # scikit-learn estimators
pip install tuiml[capymoa]    # CapyMOA streaming learners (needs Java)
pip install tuiml[weka]       # Weka learners via python-weka-wrapper3 (needs Java)
```

```python
# Curated, namespaced wrappers, discoverable via tuiml_list / train / Benchmark
from tuiml.sklearn import RandomForestClassifier      # sklearn-backed
from tuiml.algorithms import RandomForestClassifier   # native (no clash)

# In a spec, reach them by their namespaced NAME - train() takes one spec dict:
train({
    "model": {"name": "sklearn.RandomForestClassifier", "params": {"n_estimators": 100}},
    "data": {"source": "iris", "target": "class"},
    "evaluation": {"cv": 5},
})
train({
    "model": {"name": "capymoa.HoeffdingTree"},
    "data": {"source": "electricity", "target": "class"},
})
train({
    "model": {"name": "weka.J48"},                     # Weka's C4.5
    "data": {"source": "iris", "target": "class"},
})
```

Weka learners distinguish **nominal** from numeric attributes, so declare which
columns hold category codes — otherwise Weka treats them as a continuous scale:

```python
from tuiml.weka import J48
clf = J48(nominal_features=[0, 3]).fit(X, y)
print(clf.to_weka_string())     # Weka's own model dump (the tree, rules, ...)
```

Raw third-party estimator **instances** are rejected in specs, with an error
naming the wrapper to use instead. Use the namespaced name, or drop the
instance into a `Workflow`, which takes configured objects by design:

```python
from sklearn.svm import SVC

train({"model": SVC(C=2.0), "data": {"source": "iris"}})       # ✗ ValueError
train({"model": {"name": "sklearn.SVC", "params": {"C": 2.0}},  # ✓
       "data": {"source": "iris", "target": "class"}})
```

A missing backend only errors at instantiation (with a `pip install tuiml[...]`
hint), never at `import tuiml`. Hub keys are namespaced `sklearn.<Name>` /
`capymoa.<Name>` / `weka.<Name>`; native algorithms keep their bare names.

---

## 13. Common Patterns

### Full Pipeline with Cross-Validation

```python
from tuiml.algorithms.trees import RandomForestClassifier
from tuiml.preprocessing import SimpleImputer, StandardScaler
from tuiml.features.selection import CFSSelector
from tuiml.evaluation import KFold, accuracy_score
import numpy as np

kf = KFold(n_splits=5, shuffle=True)
scores = []

for train_idx, test_idx in kf.split(X):
    X_tr, X_te = X[train_idx], X[test_idx]
    y_tr, y_te = y[train_idx], y[test_idx]

    # Preprocessing (fit on train only)
    imp = SimpleImputer(strategy="mean")
    scl = StandardScaler()
    sel = CFSSelector()

    X_tr = sel.fit_transform(scl.fit_transform(imp.fit_transform(X_tr)), y_tr)
    X_te = sel.transform(scl.transform(imp.transform(X_te)))

    # Train and evaluate
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X_tr, y_tr)
    scores.append(accuracy_score(y_te, clf.predict(X_te)))

print(f"CV Accuracy: {np.mean(scores):.3f} +/- {np.std(scores):.3f}")
```

### Imbalanced Classification

```python
from tuiml.preprocessing import SMOTESampler

unique, counts = np.unique(y_train, return_counts=True)
print(f"Class distribution: {dict(zip(unique, counts))}")

smote = SMOTESampler(k_neighbors=5)
X_bal, y_bal = smote.fit_resample(X_train, y_train)

clf = RandomForestClassifier()
clf.fit(X_bal, y_bal)
```

### Hyperparameter Tuning

```python
from tuiml.evaluation import GridSearchCV

grid = GridSearchCV(
    estimator=RandomForestClassifier(),
    param_grid={"n_estimators": [50, 100, 200], "max_depth": [5, 10, None]},
    cv=5,
    scoring="f1_weighted",
)
grid.fit(X_train, y_train)
best = grid.best_estimator_
```

### Model Comparison with Statistical Testing

```python
import tuiml
from tuiml.evaluation import friedman_test, nemenyi_post_hoc

bench = tuiml.Benchmark(
    models=[{"name": "RandomForestClassifier"}, {"name": "SVC"},
            {"name": "NaiveBayesClassifier"}],
    datasets=[{"name": "iris", "X": X, "y": y}],
    evaluation={"cv": 10},
    random_seed=42,
)
bench.run()

# Built-in paired t-tests against a baseline (readable DataFrame)
print(bench.compare(baseline="RandomForestClassifier"))

# Friedman + Nemenyi on the per-fold scores (dict of {model: fold_scores})
acc = bench.scores_[bench.scores_.metric == "accuracy_score"]
per_model = {m: g["value"].to_numpy() for m, g in acc.groupby("model")}
chi2, p_val, significant = friedman_test(per_model)
pairwise = nemenyi_post_hoc(per_model, significance_level=0.05)

# Publication-ready output
print(bench.to_latex())
print(bench.to_markdown())
```

### Preprocessing Order

Recommended pipeline order: Imputation → Scaling → Encoding → Sampling → Feature Selection

### Naming

- Always use exact class names: `RandomForestClassifier`, not `random_forest`
- Case-sensitive: `NaiveBayesClassifier`, not `naivebayes`

### Metric Compatibility

- Classification: `accuracy_score`, `f1_score`, `precision_score`, `recall_score`, `roc_auc_score`
- Regression: `r2_score`, `mean_squared_error`, `mean_absolute_error`, `root_mean_squared_error`
- Clustering: `silhouette_score`, `adjusted_rand_score`
- If `metrics` is omitted in high-level API, defaults are selected automatically

---
