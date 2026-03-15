#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <tuple>
#include <string>

namespace py = pybind11;

namespace tuiml {
namespace tree {

// Returns (best_feature, best_threshold, best_gain)
std::tuple<int, double, double> best_split_classifier(
    py::array_t<double> X,
    py::array_t<int> y,
    const std::string& criterion,
    int n_classes,
    int min_samples_leaf,
    int random_seed,
    int max_features
);

// Returns (best_feature, best_threshold, best_gain)
std::tuple<int, double, double> best_split_regressor(
    py::array_t<double> X,
    py::array_t<double> y,
    const std::string& criterion,
    int min_samples_leaf,
    int random_seed,
    int max_features
);

}  // namespace tree
}  // namespace tuiml
