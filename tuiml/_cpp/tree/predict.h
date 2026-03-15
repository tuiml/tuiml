#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

namespace tuiml {
namespace tree {

// Batch prediction over a flattened tree (parallel arrays)
py::array_t<double> predict_batch(
    py::array_t<int> feature,         // (n_nodes,)
    py::array_t<double> threshold,    // (n_nodes,)
    py::array_t<int> children_left,   // (n_nodes,)
    py::array_t<int> children_right,  // (n_nodes,)
    py::array_t<double> value,        // (n_nodes, value_width)
    py::array_t<double> X             // (n_samples, n_features)
);

}  // namespace tree
}  // namespace tuiml
