#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

namespace tuiml {
namespace stats {

// Pool Adjacent Violators Algorithm.
//
// Fits the isotonic (non-decreasing) least-squares regression of ``y`` with
// per-sample weights ``w``. Inputs must already be sorted by the covariate.
// Runs in O(n) with a single block stack and is shared by isotonic
// calibration, Venn-Abers predictors and conformalized quantile regression.
py::array_t<double> pool_adjacent_violators(py::array_t<double> y,
                                            py::array_t<double> w,
                                            bool increasing);

// PAVA over an unsorted covariate.
//
// Sorts ``x`` internally, runs PAVA on the sorted response, and returns the
// (x_sorted, fitted) pair that defines the calibration step function.
std::pair<py::array_t<double>, py::array_t<double>> isotonic_fit(
    py::array_t<double> x, py::array_t<double> y, py::array_t<double> w,
    bool increasing);

}  // namespace stats
}  // namespace tuiml
