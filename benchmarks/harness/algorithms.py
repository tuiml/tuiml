"""The matching-algorithm registry: only algorithms available in ALL three
libraries (tuiml, scikit-learn, python-weka-wrapper3).

Each entry maps a neutral key -> per-framework spec + which task it applies to.
Hyperparameters are aligned where trivial (RF=100 trees, kNN k=5) and left at
defaults otherwise.
"""

# task: "classification" or "regression"
# sklearn: (module, class_name, kwargs)
# tuiml:   (registry_name, kwargs)
# weka:    (classname, options_list)
ALGORITHMS = {
    # ---- classifiers ----
    "random_forest": {
        "task": "classification",
        "sklearn": ("sklearn.ensemble", "RandomForestClassifier", {"n_estimators": 100, "random_state": 42}),
        "tuiml": ("RandomForestClassifier", {"n_estimators": 100}),
        "weka": ("weka.classifiers.trees.RandomForest", ["-I", "100"]),
    },
    "decision_tree": {
        "task": "classification",
        "sklearn": ("sklearn.tree", "DecisionTreeClassifier", {"random_state": 42}),
        "tuiml": ("DecisionTreeClassifier", {}),
        "weka": ("weka.classifiers.trees.J48", []),
    },
    "naive_bayes": {
        "task": "classification",
        "sklearn": ("sklearn.naive_bayes", "GaussianNB", {}),
        "tuiml": ("NaiveBayesClassifier", {}),
        "weka": ("weka.classifiers.bayes.NaiveBayes", []),
    },
    "logistic": {
        "task": "classification",
        "sklearn": ("sklearn.linear_model", "LogisticRegression", {"max_iter": 1000}),
        "tuiml": ("LogisticRegression", {}),
        "weka": ("weka.classifiers.functions.Logistic", []),
    },
    "knn": {
        "task": "classification",
        "sklearn": ("sklearn.neighbors", "KNeighborsClassifier", {"n_neighbors": 5}),
        "tuiml": ("KNearestNeighborsClassifier", {"k": 5}),
        "weka": ("weka.classifiers.lazy.IBk", ["-K", "5"]),
    },
    "svm": {
        "task": "classification",
        "sklearn": ("sklearn.svm", "SVC", {}),
        "tuiml": ("SVC", {}),
        "weka": ("weka.classifiers.functions.SMO", []),
    },
    "mlp": {
        "task": "classification",
        "sklearn": ("sklearn.neural_network", "MLPClassifier", {"max_iter": 300, "random_state": 42}),
        "tuiml": ("MultilayerPerceptronClassifier", {}),
        "weka": ("weka.classifiers.functions.MultilayerPerceptron", []),
    },
    # ---- regressors ----
    "linear_regression": {
        "task": "regression",
        "sklearn": ("sklearn.linear_model", "LinearRegression", {}),
        "tuiml": ("LinearRegression", {}),
        "weka": ("weka.classifiers.functions.LinearRegression", []),
    },
    "random_forest_reg": {
        "task": "regression",
        "sklearn": ("sklearn.ensemble", "RandomForestRegressor", {"n_estimators": 100, "random_state": 42}),
        "tuiml": ("RandomForestRegressor", {"n_estimators": 100}),
        "weka": ("weka.classifiers.trees.RandomForest", ["-I", "100"]),
    },
    "decision_tree_reg": {
        "task": "regression",
        "sklearn": ("sklearn.tree", "DecisionTreeRegressor", {"random_state": 42}),
        "tuiml": ("DecisionTreeRegressor", {}),
        "weka": ("weka.classifiers.trees.REPTree", []),
    },
    "knn_reg": {
        "task": "regression",
        "sklearn": ("sklearn.neighbors", "KNeighborsRegressor", {"n_neighbors": 5}),
        "tuiml": ("KNearestNeighborsRegressor", {"k": 5}),
        "weka": ("weka.classifiers.lazy.IBk", ["-K", "5"]),
    },
    "svm_reg": {
        "task": "regression",
        "sklearn": ("sklearn.svm", "SVR", {}),
        "tuiml": ("SVR", {}),
        "weka": ("weka.classifiers.functions.SMOreg", []),
    },
    "mlp_reg": {
        "task": "regression",
        "sklearn": ("sklearn.neural_network", "MLPRegressor", {"max_iter": 300, "random_state": 42}),
        "tuiml": ("MultilayerPerceptronRegressor", {}),
        "weka": ("weka.classifiers.functions.MultilayerPerceptron", []),
    },
}


def keys_for_task(task: str):
    """Return algorithm keys that apply to the given task."""
    return [k for k, v in ALGORITHMS.items() if v["task"] == task]
