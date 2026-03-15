#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <tuple>
#include <string>

namespace py = pybind11;

namespace tuiml {
namespace neighbors {

/// Batch k-nearest neighbor query using brute force.
/// Returns (distances[n_query, k], indices[n_query, k]).
std::tuple<py::array_t<double>, py::array_t<int>>
brute_knn_query(py::array_t<double> X_train,
                py::array_t<double> X_query, int k);

/// Full KNN classification in C++: returns predicted integer labels.
py::array_t<int>
brute_knn_classify(py::array_t<double> X_train,
                   py::array_t<int> y_train,
                   py::array_t<double> X_query,
                   int k, int n_classes,
                   const std::string& weight_mode);

/// Full KNN regression in C++: returns predicted values.
py::array_t<double>
brute_knn_regress(py::array_t<double> X_train,
                  py::array_t<double> y_train,
                  py::array_t<double> X_query,
                  int k,
                  const std::string& weight_mode);

}  // namespace neighbors
}  // namespace tuiml
