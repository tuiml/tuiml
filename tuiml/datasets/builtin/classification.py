"""Convenience loaders for the bundled classification datasets.

Each function reads one ARFF file that ships with TuiML and returns a
:class:`~tuiml.datasets.loaders.arff.Dataset`. They take no arguments and need
no download, so they are the quickest way to get a labelled dataset for an
example, a test, or a benchmark.

Every loader here is also reachable by name through
:func:`~tuiml.datasets.builtin.catalog.load_dataset`, and its shape and class
count are recorded in
:data:`~tuiml.datasets.builtin.catalog.DATASET_REGISTRY`.

Examples
--------
>>> from tuiml.datasets import load_iris
>>> X, y = load_iris()
>>> X.shape
(150, 4)
"""

from tuiml.datasets.builtin._paths import _get_path
from tuiml.datasets.loaders import Dataset, load_arff


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


__all__ = [
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
]
