#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <utility>

namespace py = pybind11;

namespace tuiml {
namespace stats {

// Per-dimension equal-width histogram.
//
// Returns (edges, density) with shapes (d, n_bins + 1) and (d, n_bins).
// Density is normalised so that sum(density * bin_width) == 1 per dimension.
std::pair<py::array_t<double>, py::array_t<double>> equal_width_histogram(
    py::array_t<double> X, int n_bins);

// Per-dimension equal-frequency (quantile) histogram.
//
// Bin edges are the empirical quantiles, so every bin holds roughly the same
// number of samples and the widths vary instead. Same return shapes as
// equal_width_histogram; degenerate zero-width bins are merged away, so a
// dimension may come back with fewer than n_bins usable bins (the surplus
// edges are repeated and their density is zero).
std::pair<py::array_t<double>, py::array_t<double>> equal_frequency_histogram(
    py::array_t<double> X, int n_bins);

// Look up the density of each query value in a fitted histogram.
//
// Values outside the outermost edges take the density of the nearest bin,
// which keeps the score finite for unseen extremes.
py::array_t<double> histogram_density(py::array_t<double> edges,
                                      py::array_t<double> density,
                                      py::array_t<double> X_query);

}  // namespace stats
}  // namespace tuiml
