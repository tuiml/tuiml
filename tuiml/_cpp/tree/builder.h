#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <tuple>
#include <string>

namespace py = pybind11;

namespace tuiml {
namespace tree {

// Build a complete classification tree, returning flattened parallel arrays
// Returns: (feature, threshold, children_left, children_right, value, n_nodes)
std::tuple<
    py::array_t<int>,     // feature
    py::array_t<double>,  // threshold
    py::array_t<int>,     // children_left
    py::array_t<int>,     // children_right
    py::array_t<double>,  // value (n_nodes, n_classes)
    int                   // n_nodes
> build_classifier_tree(
    py::array_t<double> X,
    py::array_t<int> y,
    const std::string& criterion,
    int n_classes,
    int max_depth,
    int min_samples_split,
    int min_samples_leaf,
    double min_impurity_decrease,
    int random_seed,
    int max_features
);

// Build a complete regression tree, returning flattened parallel arrays
std::tuple<
    py::array_t<int>,     // feature
    py::array_t<double>,  // threshold
    py::array_t<int>,     // children_left
    py::array_t<int>,     // children_right
    py::array_t<double>,  // value (n_nodes, 1)
    int                   // n_nodes
> build_regressor_tree(
    py::array_t<double> X,
    py::array_t<double> y,
    const std::string& criterion,
    int max_depth,
    int min_samples_split,
    int min_samples_leaf,
    double min_impurity_decrease,
    int random_seed,
    int max_features
);

}  // namespace tree
}  // namespace tuiml
