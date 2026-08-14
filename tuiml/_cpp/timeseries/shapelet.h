#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

namespace tuiml {
namespace timeseries {

// Shapelet distance: the smallest z-normalised Euclidean distance between a
// shapelet and any equal-length window of a series.
//
// Shapelets arrive already z-normalised and packed end to end in
// ``shapelets``, with ``offsets[i]`` and ``lengths[i]`` locating shapelet i.
// Returns an (n_series, n_shapelets) matrix of distances, normalised by
// sqrt(length) so shapelets of different lengths stay comparable.
//
// The z-normalisation of each window is folded into the algebra rather than
// materialised: because a z-normalised shapelet has zero mean and unit
// variance, the squared distance collapses to 2m - 2*dot/sigma, so only one
// dot product per window is needed.
py::array_t<double> shapelet_distances(py::array_t<double> X,
                                       py::array_t<double> shapelets,
                                       py::array_t<int> offsets,
                                       py::array_t<int> lengths);

}  // namespace timeseries
}  // namespace tuiml
