#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

namespace tuiml {
namespace shapley {

// Exact Shapley values for a decision tree, in polynomial time.
//
// Consumes the flattened layout TuiML's trees already produce: per-node
// feature, threshold, children and value, plus node_weight, the fraction of
// background samples reaching each node. A leaf is marked by feature < 0.
//
// The naive definition sums over all 2^F feature subsets. TreeSHAP instead
// walks each root-to-leaf path once, carrying the set of features already
// split on together with the proportion of subsets in which each is present
// or absent, so the exponential sum collapses to O(L * D^2) per sample for L
// leaves and depth D.
//
// Returns (n_samples, n_features, output_dim). Summing over features and
// adding the background mean recovers the model's prediction exactly, which
// is the efficiency property and the thing worth testing.
py::array_t<double> tree_shap(py::array_t<int> feature,
                              py::array_t<double> threshold,
                              py::array_t<int> children_left,
                              py::array_t<int> children_right,
                              py::array_t<double> value,
                              py::array_t<double> node_weight,
                              py::array_t<double> X);

}  // namespace shapley
}  // namespace tuiml
