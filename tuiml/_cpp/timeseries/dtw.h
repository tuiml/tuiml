#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

namespace tuiml {
namespace timeseries {

// Dynamic Time Warping distance between two univariate series.
//
// ``window`` is the Sakoe-Chiba band half-width in time steps; a negative
// value means no constraint. ``cutoff`` enables early abandoning: as soon as
// every cell of the current row exceeds it the computation stops and returns
// infinity, which is what makes the kNN search affordable.
double dtw_distance(py::array_t<double> a, py::array_t<double> b, int window,
                    double cutoff);

// LB_Keogh lower bound of the DTW distance.
//
// Cheap O(n) bound used to skip full O(n*m) DTW computations during search.
// Requires the envelope of the reference series, from ``lb_keogh_envelope``.
double lb_keogh(py::array_t<double> query, py::array_t<double> lower,
                py::array_t<double> upper);

// Running min/max envelope of a series under a Sakoe-Chiba band.
std::pair<py::array_t<double>, py::array_t<double>> lb_keogh_envelope(
    py::array_t<double> series, int window);

// Pairwise DTW between every row of A and every row of B.
//
// Multivariate input is accepted as (n, channels, length); the channel
// dimension is folded into the local cost so one warping path is shared
// across channels, which is the dependent-DTW formulation. The outer loop
// runs in parallel.
//
// No lower bound is applied here: every cell of the matrix is requested, so
// there is nothing to prune. Use dtw_knn when only the nearest neighbours are
// needed — that is where pruning pays.
py::array_t<double> dtw_pairwise(py::array_t<double> A, py::array_t<double> B,
                                 int window);

// k nearest neighbours of every row of A within B, under DTW.
//
// This is the path that makes DTW affordable. Candidates are ordered by their
// LB_Keogh lower bound, and a candidate whose bound already exceeds the
// current k-th best is skipped without any DTW computation at all; those that
// survive are computed with early abandoning against the same threshold.
//
// Returns (distances, indices), both (n_a, k), sorted nearest first.
std::pair<py::array_t<double>, py::array_t<int>> dtw_knn(py::array_t<double> A,
                                                         py::array_t<double> B,
                                                         int k, int window);

}  // namespace timeseries
}  // namespace tuiml
