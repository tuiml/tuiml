#include "shapelet.h"
#include "../common/parallel.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

namespace tuiml {
namespace timeseries {

namespace {

// Smallest z-normalised distance between one shapelet and any window of one
// series. ``shapelet`` is already z-normalised, so sum(s) == 0 and
// sum(s*s) == m; that is what lets the window normalisation cancel.
double min_window_distance(const double* series, int n, const double* shapelet,
                           int m) {
    if (m > n) return std::numeric_limits<double>::infinity();

    double best = std::numeric_limits<double>::infinity();

    // Running sums give each window's mean and variance in O(1).
    double sum = 0.0, sum_squares = 0.0;
    for (int t = 0; t < m; t++) {
        sum += series[t];
        sum_squares += series[t] * series[t];
    }

    for (int start = 0; start + m <= n; start++) {
        if (start > 0) {
            const double leaving = series[start - 1];
            const double entering = series[start + m - 1];
            sum += entering - leaving;
            sum_squares += entering * entering - leaving * leaving;
        }

        const double mean = sum / m;
        const double variance = sum_squares / m - mean * mean;

        double squared;
        if (variance <= 1e-12) {
            // A flat window has no shape to compare: its z-score is
            // undefined, so treat it as maximally unlike a unit-variance
            // shapelet rather than dividing by zero.
            squared = static_cast<double>(m);
        } else {
            double dot = 0.0;
            for (int t = 0; t < m; t++) dot += series[start + t] * shapelet[t];
            // 2m - 2*dot/sigma, using sum(shapelet) == 0 to drop the mean term.
            squared = 2.0 * m - 2.0 * dot / std::sqrt(variance);
        }

        best = std::min(best, squared);
        // Zero is the floor; nothing later can beat an exact match.
        if (best <= 0.0) { best = 0.0; break; }
    }

    // Normalise by length so shapelets of different sizes are comparable.
    return std::sqrt(std::max(best, 0.0) / m);
}

}  // namespace

py::array_t<double> shapelet_distances(py::array_t<double> X,
                                       py::array_t<double> shapelets,
                                       py::array_t<int> offsets,
                                       py::array_t<int> lengths) {
    auto x_buf = X.unchecked<2>();
    auto shapelet_buf = shapelets.unchecked<1>();
    auto offset_buf = offsets.unchecked<1>();
    auto length_buf = lengths.unchecked<1>();

    const int n_series = static_cast<int>(x_buf.shape(0));
    const int n = static_cast<int>(x_buf.shape(1));
    const int n_shapelets = static_cast<int>(offset_buf.shape(0));

    py::array_t<double> result({n_series, n_shapelets});
    auto out = result.mutable_unchecked<2>();

    WK_PARALLEL_FOR
    for (int i = 0; i < n_series; i++) {
        std::vector<double> series(n);

        // Centre the series on its own mean before sliding. Window
        // z-normalisation is invariant to a global shift, so this changes
        // nothing mathematically — but the running variance below is computed
        // as E[x^2] - mean^2, which loses precision catastrophically when the
        // values sit far from zero. Without this, a series offset by 1e6 drifts
        // by ~4e-2 against the same series offset by 0.
        double series_mean = 0.0;
        for (int t = 0; t < n; t++) series_mean += x_buf(i, t);
        series_mean /= n;
        for (int t = 0; t < n; t++) series[t] = x_buf(i, t) - series_mean;

        for (int s = 0; s < n_shapelets; s++) {
            out(i, s) = min_window_distance(
                series.data(), n, shapelet_buf.data(offset_buf(s)),
                length_buf(s));
        }
    }

    return result;
}

}  // namespace timeseries
}  // namespace tuiml
