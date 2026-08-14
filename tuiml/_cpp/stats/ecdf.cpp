#include "ecdf.h"
#include "../common/parallel.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace tuiml {
namespace stats {

std::pair<py::array_t<double>, py::array_t<double>> tail_probabilities(
    py::array_t<double> X_train, py::array_t<double> X_query) {
    auto train = X_train.unchecked<2>();
    auto query = X_query.unchecked<2>();

    const int n_train = static_cast<int>(train.shape(0));
    const int n_query = static_cast<int>(query.shape(0));
    const int n_dim = static_cast<int>(train.shape(1));

    py::array_t<double> left({n_query, n_dim});
    py::array_t<double> right({n_query, n_dim});
    auto left_buf = left.mutable_unchecked<2>();
    auto right_buf = right.mutable_unchecked<2>();

    if (n_train == 0 || n_dim == 0) return {left, right};

    // A probability of exactly zero would make -log() infinite, so the
    // estimates are floored at 1/n — the smallest non-vacuous frequency the
    // training set can express.
    const double floor_prob = 1.0 / static_cast<double>(n_train);

    WK_PARALLEL_FOR
    for (int j = 0; j < n_dim; j++) {
        std::vector<double> column(n_train);
        for (int i = 0; i < n_train; i++) column[i] = train(i, j);
        std::sort(column.begin(), column.end());

        for (int i = 0; i < n_query; i++) {
            const double value = query(i, j);

            // upper_bound counts strictly-less-or-equal, lower_bound counts
            // strictly-less, so the two together handle ties correctly.
            const auto upper =
                std::upper_bound(column.begin(), column.end(), value);
            const auto lower =
                std::lower_bound(column.begin(), column.end(), value);

            const double n_le = static_cast<double>(upper - column.begin());
            const double n_lt = static_cast<double>(lower - column.begin());
            const double n_ge = static_cast<double>(n_train) - n_lt;

            left_buf(i, j) = std::max(n_le / n_train, floor_prob);
            right_buf(i, j) = std::max(n_ge / n_train, floor_prob);
        }
    }

    return {left, right};
}

py::array_t<double> skewness(py::array_t<double> X) {
    auto buf = X.unchecked<2>();
    const int n = static_cast<int>(buf.shape(0));
    const int n_dim = static_cast<int>(buf.shape(1));

    py::array_t<double> result(n_dim);
    auto res = result.mutable_unchecked<1>();

    WK_PARALLEL_FOR
    for (int j = 0; j < n_dim; j++) {
        if (n < 3) {
            res(j) = 0.0;
            continue;
        }

        double mean = 0.0;
        for (int i = 0; i < n; i++) mean += buf(i, j);
        mean /= n;

        double m2 = 0.0, m3 = 0.0;
        for (int i = 0; i < n; i++) {
            const double d = buf(i, j) - mean;
            m2 += d * d;
            m3 += d * d * d;
        }
        m2 /= n;
        m3 /= n;

        if (m2 <= 0.0) {
            // A constant column has no skew to speak of.
            res(j) = 0.0;
            continue;
        }

        const double g1 = m3 / std::pow(m2, 1.5);
        // Adjusted Fisher-Pearson correction, as used by scipy.stats.skew
        // with bias=False.
        res(j) = g1 * std::sqrt(static_cast<double>(n) * (n - 1)) / (n - 2);
    }

    return result;
}

}  // namespace stats
}  // namespace tuiml
