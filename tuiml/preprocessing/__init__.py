"""Data Preprocessing and Transformation.

The ``tuiml.preprocessing`` module provides a wide array of tools for
preparing dataset for machine learning. It follows a consistent API where
most components are ``Transformers`` that can be used in pipelines.

Overview
--------
This module is organized into several functional categories:

1. **Scaling**: Standardize or normalize numerical features (e.g., Z-score, Min-Max).
2. **Encoding**: Convert categorical/nominal data into numerical formats (e.g., One-Hot, Ordinal).
3. **Imputation**: Handle missing data using statistical or distance-based strategies.
4. **Discretization**: Partition continuous features into discrete bins (e.g., Equal-Width, MDL).
5. **Outliers**: Detect and manage extreme values (e.g., IQR clipping).
6. **Sampling**: Balance classes (SMOTE, ADASYN) or reduce dataset size.
7. **Text**: Convert raw text into numerical vectors (TF-IDF, Hashing).
8. **Time Series**: Extract temporal features through lags and differences.

Basic Usage
-----------
Impute missing values and scale features in sequence:

>>> from tuiml.preprocessing import SimpleImputer, StandardScaler
>>> import numpy as np
>>> X = np.array([[1, 2], [np.nan, 3], [7, 6]])
>>> imputer = SimpleImputer(strategy="mean")
>>> scaler = StandardScaler()
>>> X_clean = scaler.fit_transform(imputer.fit_transform(X))

Imbalanced Learning
-------------------
Use SMOTE to generate synthetic samples for a minority class:

>>> from tuiml.preprocessing.sampling import SMOTESampler
>>> X = np.random.rand(10, 2)
>>> y = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1])
>>> sampler = SMOTESampler(k_neighbors=2)
>>> X_res, y_res = sampler.fit_resample(X, y)
"""

# Base classes
from tuiml.base.preprocessing import (
    Preprocessor,
    Filter,
    Transformer,
    SupervisedTransformer,
    InstanceTransformer,
    preprocessor,
    filter_method,
    transformer,
)

# Scaling
from tuiml.preprocessing.scaling import (
    MinMaxScaler,
    StandardScaler,
    CenterScaler,
)

# Encoding
from tuiml.preprocessing.encoding import (
    OneHotEncoder,
    OrdinalEncoder,
    LabelEncoder,
    RareCategoryEncoder,
)

# Imputation
from tuiml.preprocessing.imputation import (
    SimpleImputer,
    KNNImputer,
)

# Discretization
from tuiml.preprocessing.discretization import (
    EqualWidthDiscretizer,
    QuantileDiscretizer,
    MDLDiscretizer,
)

# Outliers
from tuiml.preprocessing.outliers import (
    IQROutlierDetector,
    ValueClipper,
)

# Sampling
from tuiml.preprocessing.sampling import (
    ReservoirSampler,
    ClassBalanceSampler,
    # SMOTE family
    SMOTESampler,
    BorderlineSMOTESampler,
    ADASYNSampler,
    SVMSMOTESampler,
    KMeansSMOTESampler,
    # Oversampling
    RandomOverSampler,
    ClusterOverSampler,
    # Undersampling
    RandomUnderSampler,
    TomekLinksSampler,
    ENNSampler,
    CNNSampler,
    NearMissSampler,
    HardnessThresholdSampler,
)

# Text preprocessing
from tuiml.preprocessing.text import (
    # Tokenizers
    WordTokenizer,
    NGramTokenizer,
    RegexTokenizer,
    SentenceTokenizer,
    # Vectorizers
    CountVectorizer,
    TfidfVectorizer,
    TfidfTransformer,
    HashingVectorizer,
    # Cleaners
    TextCleaner,
    StopWordRemover,
    Stemmer,
)

# Time series preprocessing
from tuiml.preprocessing.timeseries import (
    LagTransformer,
    DifferenceTransformer,
)

__all__ = [
    # Base
    "Preprocessor",
    "Filter",
    "Transformer",
    "SupervisedTransformer",
    "InstanceTransformer",
    "preprocessor",
    "filter_method",
    "transformer",
    # Scaling
    "MinMaxScaler",
    "StandardScaler",
    "CenterScaler",
    # Encoding
    "OneHotEncoder",
    "OrdinalEncoder",
    "LabelEncoder",
    "RareCategoryEncoder",
    # Imputation
    "SimpleImputer",
    "KNNImputer",
    # Discretization
    "EqualWidthDiscretizer",
    "QuantileDiscretizer",
    "MDLDiscretizer",
    # Outliers
    "IQROutlierDetector",
    "ValueClipper",
    # Sampling
    "ReservoirSampler",
    "ClassBalanceSampler",
    # Imbalanced Learning - SMOTE family
    "SMOTESampler",
    "BorderlineSMOTESampler",
    "ADASYNSampler",
    "SVMSMOTESampler",
    "KMeansSMOTESampler",
    # Imbalanced Learning - Oversampling
    "RandomOverSampler",
    "ClusterOverSampler",
    # Imbalanced Learning - Undersampling
    "RandomUnderSampler",
    "TomekLinksSampler",
    "ENNSampler",
    "CNNSampler",
    "NearMissSampler",
    "HardnessThresholdSampler",
    # Text Preprocessing
    "WordTokenizer",
    "NGramTokenizer",
    "RegexTokenizer",
    "SentenceTokenizer",
    "CountVectorizer",
    "TfidfVectorizer",
    "TfidfTransformer",
    "HashingVectorizer",
    "TextCleaner",
    "StopWordRemover",
    "Stemmer",
    # Time Series
    "LagTransformer",
    "DifferenceTransformer",
]
