"""
Built-in datasets for TuiML.

Provides easy access to classic ML datasets organized by task type:
- Classification: iris, diabetes, breast_cancer, glass, etc.
- Regression: cpu, airline
- Other: supermarket (association), reuters (text)

Usage:
    >>> from tuiml.datasets import load_iris
    >>> data = load_iris()
    >>> X, y = load_iris()  # Can unpack

    >>> from tuiml.datasets import list_datasets
    >>> list_datasets()  # All datasets
    >>> list_datasets("classification")  # Only classification
"""

from pathlib import Path
from typing import List, Optional

from tuiml.datasets.loaders import load_arff, Dataset

# Paths to built-in datasets by category
_BUILTIN_DIR = Path(__file__).parent
_CLASSIFICATION_DIR = _BUILTIN_DIR / "classification"
_REGRESSION_DIR = _BUILTIN_DIR / "regression"
_OTHER_DIR = _BUILTIN_DIR / "other"

# =============================================================================
# Dataset Registry - LLM-friendly metadata for all built-in datasets
# =============================================================================

DATASET_REGISTRY = {
    # Classification datasets
    "iris": {
        "task": "classification",
        "samples": 150,
        "features": 4,
        "classes": 3,
        "description": "Classic flower classification: setosa, versicolor, virginica",
        "loader": "load_iris"
    },
    "iris_2d": {
        "task": "classification",
        "samples": 150,
        "features": 2,
        "classes": 3,
        "description": "Iris with reduced features (2D version)",
        "loader": "load_iris_2d"
    },
    "diabetes": {
        "task": "classification",
        "samples": 768,
        "features": 8,
        "classes": 2,
        "description": "Pima Indians diabetes prediction",
        "loader": "load_diabetes"
    },
    "breast_cancer": {
        "task": "classification",
        "samples": 286,
        "features": 9,
        "classes": 2,
        "description": "Breast cancer recurrence prediction",
        "loader": "load_breast_cancer"
    },
    "glass": {
        "task": "classification",
        "samples": 214,
        "features": 9,
        "classes": 7,
        "description": "Glass type identification from chemical properties",
        "loader": "load_glass"
    },
    "ionosphere": {
        "task": "classification",
        "samples": 351,
        "features": 34,
        "classes": 2,
        "description": "Radar signal classification (good/bad)",
        "loader": "load_ionosphere"
    },
    "vote": {
        "task": "classification",
        "samples": 435,
        "features": 16,
        "classes": 2,
        "description": "Congressional voting records (democrat/republican)",
        "loader": "load_vote"
    },
    "credit": {
        "task": "classification",
        "samples": 1000,
        "features": 20,
        "classes": 2,
        "description": "German credit risk assessment",
        "loader": "load_credit"
    },
    "weather": {
        "task": "classification",
        "samples": 14,
        "features": 4,
        "classes": 2,
        "description": "Weather conditions for playing tennis (numeric)",
        "loader": "load_weather"
    },
    "weather_nominal": {
        "task": "classification",
        "samples": 14,
        "features": 4,
        "classes": 2,
        "description": "Weather conditions for playing tennis (nominal)",
        "loader": "load_weather_nominal"
    },
    "soybean": {
        "task": "classification",
        "samples": 683,
        "features": 35,
        "classes": 19,
        "description": "Soybean disease classification",
        "loader": "load_soybean"
    },
    "labor": {
        "task": "classification",
        "samples": 57,
        "features": 16,
        "classes": 2,
        "description": "Labor relations negotiation outcomes",
        "loader": "load_labor"
    },
    "contact_lenses": {
        "task": "classification",
        "samples": 24,
        "features": 4,
        "classes": 3,
        "description": "Contact lens prescription recommendation",
        "loader": "load_contact_lenses"
    },
    "hypothyroid": {
        "task": "classification",
        "samples": 3772,
        "features": 29,
        "classes": 4,
        "description": "Hypothyroid disease diagnosis",
        "loader": "load_hypothyroid"
    },
    "segment": {
        "task": "classification",
        "samples": 1500,
        "features": 19,
        "classes": 7,
        "description": "Image segmentation (challenge set)",
        "loader": "load_segment"
    },
    "segment_test": {
        "task": "classification",
        "samples": 810,
        "features": 19,
        "classes": 7,
        "description": "Image segmentation (test set)",
        "loader": "load_segment_test"
    },
    "unbalanced": {
        "task": "classification",
        "samples": None,
        "features": None,
        "classes": 2,
        "description": "Dataset with imbalanced class distribution",
        "loader": "load_unbalanced"
    },
    # Regression datasets
    "cpu": {
        "task": "regression",
        "samples": 209,
        "features": 6,
        "classes": None,
        "description": "CPU performance prediction",
        "loader": "load_cpu"
    },
    "cpu_with_vendor": {
        "task": "regression",
        "samples": 209,
        "features": 7,
        "classes": None,
        "description": "CPU performance with vendor information",
        "loader": "load_cpu_with_vendor"
    },
    "airline": {
        "task": "regression",
        "samples": None,
        "features": None,
        "classes": None,
        "description": "Airline scheduling/time series data",
        "loader": "load_airline"
    },
    # Other datasets
    "supermarket": {
        "task": "association",
        "samples": 4627,
        "features": 217,
        "classes": None,
        "description": "Supermarket transactions for association rule mining",
        "loader": "load_supermarket"
    },
    "reuters_corn": {
        "task": "text_classification",
        "samples": None,
        "features": None,
        "classes": 2,
        "description": "Reuters news about corn (train/test splits available)",
        "loader": "load_reuters_corn"
    },
    "reuters_grain": {
        "task": "text_classification",
        "samples": None,
        "features": None,
        "classes": 2,
        "description": "Reuters news about grain (train/test splits available)",
        "loader": "load_reuters_grain"
    },
}

def get_dataset_info(name: str = None) -> dict:
    """Get metadata about built-in datasets in an LLM-friendly format.

    Parameters
    ----------
    name : str or None, default=None
        The name of a specific dataset (e.g., "iris"). If None, metadata 
        for all registered datasets will be returned.

    Returns
    -------
    dict
        A dictionary containing metadata such as task type, sample count, 
        feature count, class count, and description.

    Examples
    --------
    >>> info = get_dataset_info("diabetes")
    >>> print(info["samples"])
    768
    """
    if name:
        if name not in DATASET_REGISTRY:
            raise ValueError(f"Unknown dataset: {name}. Available: {list(DATASET_REGISTRY.keys())}")
        return DATASET_REGISTRY[name]
    return DATASET_REGISTRY

def get_datasets_by_task(task: str) -> dict:
    """
    Get datasets filtered by task type (LLM-friendly).

    Parameters
    ----------
    task : str
        Task type: "classification", "regression", "association", "text_classification"

    Returns
    -------
    dict
        Filtered dataset registry.

    Examples
    --------
    >>> get_datasets_by_task("classification")
    >>> get_datasets_by_task("regression")
    """
    return {
        name: info for name, info in DATASET_REGISTRY.items()
        if info["task"] == task
    }

def _get_path(category: str, filename: str) -> Path:
    """Get path to a built-in dataset file.

    Parameters
    ----------
    category : str
        Dataset category: ``"classification"``, ``"regression"``, or ``"other"``.
    filename : str
        Name of the ARFF file (e.g., ``"iris.arff"``).

    Returns
    -------
    Path
        Resolved path to the dataset file.
    """
    dirs = {
        "classification": _CLASSIFICATION_DIR,
        "regression": _REGRESSION_DIR,
        "other": _OTHER_DIR,
    }
    path = dirs[category] / filename
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {filename}")
    return path

def list_datasets(category: Optional[str] = None) -> List[str]:
    """List names of all available built-in datasets.

    Parameters
    ----------
    category : str or None, default=None
        Optional filter to restrict results to a specific category:
        
        - ``"classification"``
        - ``"regression"``
        - ``"other"`` (Association, Text, etc.)

    Returns
    -------
    List[str]
        Alphabetical list of dataset names.

    Examples
    --------
    >>> available = list_datasets("regression")
    >>> print(available)
    ['airline', 'cpu', 'cpu_with_vendor']
    """
    dirs = {
        "classification": _CLASSIFICATION_DIR,
        "regression": _REGRESSION_DIR,
        "other": _OTHER_DIR,
    }

    if category:
        if category not in dirs:
            raise ValueError(f"Unknown category: {category}. Use: {list(dirs.keys())}")
        return sorted([f.stem for f in dirs[category].glob("*.arff")])

    # Return all
    all_datasets = []
    for d in dirs.values():
        all_datasets.extend([f.stem for f in d.glob("*.arff")])
    return sorted(set(all_datasets))

def load_dataset(name: str) -> Dataset:
    """Load a built-in dataset by its registry name.

    Automatically identifies the correct file path and uses the ARFF loader 
    to return a standardized Dataset object.

    Parameters
    ----------
    name : str
        The name of the dataset to load (e.g., ``'iris'``, ``'diabetes'``, 
        ``'cpu'``).

    Returns
    -------
    Dataset
        Standardized dataset object containing the data and metadata.

    Examples
    --------
    >>> from tuiml.datasets import load_dataset
    >>> iris = load_dataset('iris')
    >>> X, y = iris
    """
    # Search in all directories
    for category_dir in [_CLASSIFICATION_DIR, _REGRESSION_DIR, _OTHER_DIR]:
        path = category_dir / f"{name}.arff"
        if path.exists():
            return load_arff(path)

    # Try exact filename
    for category_dir in [_CLASSIFICATION_DIR, _REGRESSION_DIR, _OTHER_DIR]:
        path = category_dir / name
        if path.exists():
            return load_arff(path)

    available = list_datasets()
    raise ValueError(f"Dataset '{name}' not found. Available: {available}")

# =============================================================================
# Classification Datasets
# =============================================================================

def load_iris() -> Dataset:
    """Load the classic Iris flower dataset.

    150 samples, 4 features, and 3 classes (setosa, versicolor, virginica).

    Returns
    -------
    Dataset
        Standardized dataset object containing the data and metadata.

    Examples
    --------
    >>> from tuiml.datasets import load_iris
    >>> X, y = load_iris()
    """
    return load_arff(_get_path("classification", "iris.arff"))

def load_iris_2d() -> Dataset:
    """Load the Iris 2D dataset (reduced features).

    150 samples, 2 features, 3 classes.

    Returns
    -------
    dataset : Dataset
        Iris dataset with only petal length and petal width features.
    """
    return load_arff(_get_path("classification", "iris.2D.arff"))

def load_diabetes() -> Dataset:
    """Load the Pima Indians Diabetes dataset.

    768 samples, 8 features, 2 classes.

    Returns
    -------
    dataset : Dataset
        Diabetes classification dataset.
    """
    return load_arff(_get_path("classification", "diabetes.arff"))

def load_breast_cancer() -> Dataset:
    """Load the Breast Cancer Wisconsin dataset.

    286 samples, 9 features, 2 classes (recurrence, no-recurrence).

    Returns
    -------
    dataset : Dataset
        Breast cancer recurrence classification dataset.
    """
    return load_arff(_get_path("classification", "breast-cancer.arff"))

def load_glass() -> Dataset:
    """Load the Glass Identification dataset.

    214 samples, 9 features, 7 classes.

    Returns
    -------
    dataset : Dataset
        Glass type classification dataset.
    """
    return load_arff(_get_path("classification", "glass.arff"))

def load_ionosphere() -> Dataset:
    """Load the Ionosphere dataset.

    351 samples, 34 features, 2 classes.

    Returns
    -------
    dataset : Dataset
        Radar signal classification dataset.
    """
    return load_arff(_get_path("classification", "ionosphere.arff"))

def load_vote() -> Dataset:
    """Load the Congressional Voting Records dataset.

    435 samples, 16 features, 2 classes.

    Returns
    -------
    dataset : Dataset
        Congressional voting classification dataset.
    """
    return load_arff(_get_path("classification", "vote.arff"))

def load_credit() -> Dataset:
    """Load the German Credit dataset.

    1000 samples, 20 features, 2 classes.

    Returns
    -------
    dataset : Dataset
        German credit risk classification dataset.
    """
    return load_arff(_get_path("classification", "credit-g.arff"))

def load_weather() -> Dataset:
    """Load the Weather dataset (numeric version).

    14 samples, 4 features, 2 classes.

    Returns
    -------
    dataset : Dataset
        Weather classification dataset with numeric features.
    """
    return load_arff(_get_path("classification", "weather.numeric.arff"))

def load_weather_nominal() -> Dataset:
    """Load the Weather dataset (nominal version).

    14 samples, 4 features, 2 classes.

    Returns
    -------
    dataset : Dataset
        Weather classification dataset with nominal features.
    """
    return load_arff(_get_path("classification", "weather.nominal.arff"))

def load_soybean() -> Dataset:
    """Load the Soybean dataset.

    683 samples, 35 features, 19 classes.

    Returns
    -------
    dataset : Dataset
        Soybean disease classification dataset.
    """
    return load_arff(_get_path("classification", "soybean.arff"))

def load_labor() -> Dataset:
    """Load the Labor Relations dataset.

    57 samples, 16 features, 2 classes.

    Returns
    -------
    dataset : Dataset
        Labor negotiations outcome classification dataset.
    """
    return load_arff(_get_path("classification", "labor.arff"))

def load_contact_lenses() -> Dataset:
    """Load the Contact Lenses dataset.

    24 samples, 4 features, 3 classes.

    Returns
    -------
    dataset : Dataset
        Contact lens prescription classification dataset.
    """
    return load_arff(_get_path("classification", "contact-lenses.arff"))

def load_hypothyroid() -> Dataset:
    """Load the Hypothyroid dataset.

    3772 samples, 29 features, 4 classes.

    Returns
    -------
    dataset : Dataset
        Hypothyroid disease classification dataset.
    """
    return load_arff(_get_path("classification", "hypothyroid.arff"))

def load_segment() -> Dataset:
    """Load the Image Segmentation dataset (challenge set).

    1500 samples, 19 features, 7 classes.

    Returns
    -------
    dataset : Dataset
        Image segmentation challenge dataset.
    """
    return load_arff(_get_path("classification", "segment-challenge.arff"))

def load_segment_test() -> Dataset:
    """Load the Image Segmentation test dataset.

    810 samples, 19 features, 7 classes.

    Returns
    -------
    dataset : Dataset
        Image segmentation test dataset.
    """
    return load_arff(_get_path("classification", "segment-test.arff"))

def load_unbalanced() -> Dataset:
    """Load the Unbalanced dataset.

    Dataset with imbalanced class distribution.

    Returns
    -------
    dataset : Dataset
        Classification dataset with imbalanced classes.
    """
    return load_arff(_get_path("classification", "unbalanced.arff"))

# =============================================================================
# Regression Datasets
# =============================================================================

def load_cpu() -> Dataset:
    """Load the Computer Hardware (CPU) Performance dataset.

    209 instances with 6 continuous features for regression tasks.

    Returns
    -------
    Dataset
        Standardized dataset object containing the data and metadata.

    Examples
    --------
    >>> from tuiml.datasets import load_cpu
    >>> data = load_cpu()
    >>> print(data.X.shape)
    (209, 6)
    """
    return load_arff(_get_path("regression", "cpu.arff"))

def load_cpu_with_vendor() -> Dataset:
    """Load the CPU Performance dataset with vendor information.

    209 samples, 7 features (regression task).

    Returns
    -------
    dataset : Dataset
        CPU performance regression dataset including vendor feature.
    """
    return load_arff(_get_path("regression", "cpu.with.vendor.arff"))

def load_airline() -> Dataset:
    """Load the Airline dataset.

    Small dataset for time series / scheduling examples.

    Returns
    -------
    dataset : Dataset
        Airline scheduling regression dataset.
    """
    return load_arff(_get_path("regression", "airline.arff"))

# =============================================================================
# Other Datasets (Association, Text)
# =============================================================================

def load_supermarket() -> Dataset:
    """Load the Supermarket dataset.

    4627 samples, 217 features (for association rule mining).

    Returns
    -------
    dataset : Dataset
        Supermarket transaction dataset for association rules.
    """
    return load_arff(_get_path("other", "supermarket.arff"))

def load_reuters_corn(split: str = 'train') -> Dataset:
    """Load the Reuters Corn dataset.

    Parameters
    ----------
    split : str, default='train'
        Which split to load: ``'train'`` or ``'test'``.

    Returns
    -------
    dataset : Dataset
        Reuters corn text classification dataset.
    """
    return load_arff(_get_path("other", f"ReutersCorn-{split}.arff"))

def load_reuters_grain(split: str = 'train') -> Dataset:
    """Load the Reuters Grain dataset.

    Parameters
    ----------
    split : str, default='train'
        Which split to load: ``'train'`` or ``'test'``.

    Returns
    -------
    dataset : Dataset
        Reuters grain text classification dataset.
    """
    return load_arff(_get_path("other", f"ReutersGrain-{split}.arff"))

__all__ = [
    # LLM-friendly metadata
    "DATASET_REGISTRY",
    "get_dataset_info",
    "get_datasets_by_task",
    # Utilities
    "list_datasets",
    "load_dataset",
    # Classification
    "load_iris",
    "load_iris_2d",
    "load_diabetes",
    "load_breast_cancer",
    "load_glass",
    "load_ionosphere",
    "load_vote",
    "load_credit",
    "load_weather",
    "load_weather_nominal",
    "load_soybean",
    "load_labor",
    "load_contact_lenses",
    "load_hypothyroid",
    "load_segment",
    "load_segment_test",
    "load_unbalanced",
    # Regression
    "load_cpu",
    "load_cpu_with_vendor",
    "load_airline",
    # Other
    "load_supermarket",
    "load_reuters_corn",
    "load_reuters_grain",
]
