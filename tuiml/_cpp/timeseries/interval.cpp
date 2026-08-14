#include "interval.h"
#include "../common/parallel.h"

#include <cmath>
#include <vector>

namespace tuiml {
namespace timeseries {

namespace {

// Intervals at or below this width are computed directly rather than by
// differencing prefix sums, which is where the cancellation bites.
constexpr int kDirectWidth = 32;

}  // namespace

py::array_t<double> interval_features(py::array_t<double> X,
                                      py::array_t<int> starts,
                                      py::array_t<int> ends) {
    auto x_buf = X.unchecked<2>();
    auto start_buf = starts.unchecked<1>();
    auto end_buf = ends.unchecked<1>();

    const int n_series = static_cast<int>(x_buf.shape(0));
    const int n = static_cast<int>(x_buf.shape(1));
    const int n_intervals = static_cast<int>(start_buf.shape(0));

    py::array_t<double> result({n_series, n_intervals * 3});
    auto out = result.mutable_unchecked<2>();

    WK_PARALLEL_FOR
    for (int i = 0; i < n_series; i++) {
        // Centre the series before accumulating squares and cross-products:
        // the variance below is E[x^2] - mean^2, which cancels badly far from
        // zero. Mean, standard deviation and slope are all either unaffected
        // by a shift or shifted by a known constant, restored afterwards.
        double series_mean = 0.0;
        for (int t = 0; t < n; t++) series_mean += x_buf(i, t);
        series_mean /= n;

        // Prefix sums, one element longer so an empty prefix is representable.
        std::vector<double> sum_x(n + 1, 0.0);
        std::vector<double> sum_xx(n + 1, 0.0);
        std::vector<double> sum_tx(n + 1, 0.0);

        for (int t = 0; t < n; t++) {
            const double value = x_buf(i, t) - series_mean;
            sum_x[t + 1] = sum_x[t] + value;
            sum_xx[t + 1] = sum_xx[t] + value * value;
            sum_tx[t + 1] = sum_tx[t] + t * value;
        }

        for (int k = 0; k < n_intervals; k++) {
            const int from = start_buf(k);
            const int to = end_buf(k);
            const int width = to - from;

            // Narrow intervals are computed directly. Differencing two large
            // prefix sums to recover a small quantity loses precision, and
            // sqrt then amplifies what is left: for a width-1 interval the
            // variance is exactly zero, but the residue came back as ~1e-14
            // and its square root as ~1e-7. A direct pass is exact and, at
            // this width, no more expensive.
            if (width <= kDirectWidth) {
                double mean = 0.0;
                for (int t = from; t < to; t++) mean += x_buf(i, t);
                mean /= width;

                double variance = 0.0, covariance = 0.0;
                const double t_mean = (from + to - 1) / 2.0;
                for (int t = from; t < to; t++) {
                    const double centred_x = x_buf(i, t) - mean;
                    variance += centred_x * centred_x;
                    covariance += (t - t_mean) * centred_x;
                }
                variance /= width;

                double narrow_slope = 0.0;
                if (width > 1) {
                    const double t_variance =
                        (static_cast<double>(width) * width - 1.0) / 12.0;
                    narrow_slope = (covariance / width) / t_variance;
                }

                out(i, k * 3 + 0) = mean;
                out(i, k * 3 + 1) = std::sqrt(std::max(variance, 0.0));
                out(i, k * 3 + 2) = narrow_slope;
                continue;
            }

            const double total = sum_x[to] - sum_x[from];
            const double total_squares = sum_xx[to] - sum_xx[from];
            const double total_cross = sum_tx[to] - sum_tx[from];

            const double mean = total / width;
            const double variance = total_squares / width - mean * mean;

            double slope = 0.0;
            if (width > 1) {
                // Time is a contiguous run of integers, so its own moments are
                // closed-form and never need a prefix sum.
                const double t_mean = (from + to - 1) / 2.0;
                // Var(t) over w consecutive integers is (w^2 - 1) / 12.
                const double t_variance =
                    (static_cast<double>(width) * width - 1.0) / 12.0;
                const double covariance = total_cross / width - t_mean * mean;
                slope = covariance / t_variance;
            }

            out(i, k * 3 + 0) = mean + series_mean;  // undo the centring
            out(i, k * 3 + 1) = std::sqrt(std::max(variance, 0.0));
            out(i, k * 3 + 2) = slope;
        }
    }

    return result;
}

}  // namespace timeseries
}  // namespace tuiml
