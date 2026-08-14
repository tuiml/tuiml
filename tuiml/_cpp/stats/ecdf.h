#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <utility>

namespace py = pybind11;

namespace tuiml {
namespace stats {

// Per-dimension empirical tail probabilities.
//
// For every column j the training values are sorted once, then each query
// value is located by binary search. Returns the pair
//   left[i, j]  = P(X_j <= q[i, j])   estimated as (#train <= q) / n
//   right[i, j] = P(X_j >= q[i, j])   estimated as (#train >= q) / n
// Both are clamped away from zero so a caller may take their logarithm.
//
// Shared by ECOD, COPOD and any future ECDF-based scorer.
std::pair<py::array_t<double>, py::array_t<double>> tail_probabilities(
    py::array_t<double> X_train, py::array_t<double> X_query);

// Adjusted Fisher-Pearson skewness of each column, matching scipy's
// ``skew(bias=False)``. Used to pick which tail a dimension is scored on.
py::array_t<double> skewness(py::array_t<double> X);

}  // namespace stats
}  // namespace tuiml
