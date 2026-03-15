#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

namespace tuiml {
namespace distance {

// Pairwise distance matrix: (n, d) x (m, d) -> (n, m)
py::array_t<double> euclidean_distance(py::array_t<double> X, py::array_t<double> Y);
py::array_t<double> manhattan_distance(py::array_t<double> X, py::array_t<double> Y);
py::array_t<double> cosine_distance(py::array_t<double> X, py::array_t<double> Y);

}  // namespace distance
}  // namespace tuiml
