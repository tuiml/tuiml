#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <array>
#include <tuple>
#include <vector>

namespace py = pybind11;

namespace tuiml {
namespace timeseries {

// The 84 fixed MINIROCKET kernels.
//
// Every kernel has length 9 with weights drawn from {-1, 2}: exactly three of
// the nine positions carry the weight 2, giving C(9,3) = 84 combinations and
// a mean weight of zero. Returns an (84, 3) array of the "gamma" positions.
py::array_t<int> minirocket_kernel_indices();

// Fit the per-dilation bias quantiles of a MINIROCKET transform.
//
// Biases are quantiles of the actual convolution output, so the PPV features
// land where the data has resolution rather than at arbitrary thresholds.
// Returns the bias vector, laid out dilation-major then kernel-major.
py::array_t<double> minirocket_biases(py::array_t<double> X,
                                      py::array_t<int> dilations,
                                      py::array_t<int> features_per_dilation,
                                      py::array_t<double> quantiles,
                                      unsigned int seed);

// Apply a fitted MINIROCKET transform.
//
// Produces one PPV (proportion of positive values) feature per
// (kernel, dilation, bias) triple. The outer loop over series runs in
// parallel.
py::array_t<double> minirocket_transform(py::array_t<double> X,
                                         py::array_t<int> dilations,
                                         py::array_t<int> features_per_dilation,
                                         py::array_t<double> biases);

}  // namespace timeseries
}  // namespace tuiml
