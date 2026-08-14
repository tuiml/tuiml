#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

namespace tuiml {
namespace timeseries {

// Summary statistics of arbitrary intervals of every series.
//
// For each series and each half-open interval [start, end) returns the mean,
// standard deviation and least-squares slope against time — the three
// features of the classical time-series forest.
//
// Prefix sums of x, x^2 and t*x are built once per series, after which every
// interval costs O(1) regardless of its width. A direct implementation would
// be O(interval width) each, which dominates when hundreds of wide intervals
// are drawn per tree.
//
// Returns (n_series, n_intervals * 3), with the three statistics adjacent for
// each interval.
py::array_t<double> interval_features(py::array_t<double> X,
                                      py::array_t<int> starts,
                                      py::array_t<int> ends);

}  // namespace timeseries
}  // namespace tuiml
