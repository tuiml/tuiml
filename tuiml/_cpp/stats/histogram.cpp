#include "histogram.h"
#include "../common/parallel.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace tuiml {
namespace stats {

namespace {

// Turn per-bin counts and edges into a normalised density.
void counts_to_density(const std::vector<double>& counts,
                       const std::vector<double>& edges, int n_bins,
                       int n_samples, double* out) {
    for (int b = 0; b < n_bins; b++) {
        const double width = edges[b + 1] - edges[b];
        // A zero-width bin cannot hold probability mass; leaving it at zero
        // lets the caller detect and skip it.
        out[b] = width > 0.0 ? counts[b] / (n_samples * width) : 0.0;
    }
}

}  // namespace

std::pair<py::array_t<double>, py::array_t<double>> equal_width_histogram(
    py::array_t<double> X, int n_bins) {
    auto buf = X.unchecked<2>();
    const int n = static_cast<int>(buf.shape(0));
    const int n_dim = static_cast<int>(buf.shape(1));

    py::array_t<double> edges({n_dim, n_bins + 1});
    py::array_t<double> density({n_dim, n_bins});
    auto edges_buf = edges.mutable_unchecked<2>();
    auto density_buf = density.mutable_unchecked<2>();

    WK_PARALLEL_FOR
    for (int j = 0; j < n_dim; j++) {
        double lo = buf(0, j), hi = buf(0, j);
        for (int i = 1; i < n; i++) {
            lo = std::min(lo, buf(i, j));
            hi = std::max(hi, buf(i, j));
        }
        // Widen a constant column so it still has a well-defined bin.
        if (hi <= lo) {
            hi = lo + 1.0;
        }

        const double width = (hi - lo) / n_bins;
        std::vector<double> local_edges(n_bins + 1);
        for (int b = 0; b <= n_bins; b++) local_edges[b] = lo + b * width;
        local_edges[n_bins] = hi;

        std::vector<double> counts(n_bins, 0.0);
        for (int i = 0; i < n; i++) {
            int b = static_cast<int>((buf(i, j) - lo) / width);
            b = std::min(std::max(b, 0), n_bins - 1);
            counts[b] += 1.0;
        }

        std::vector<double> local_density(n_bins);
        counts_to_density(counts, local_edges, n_bins, n, local_density.data());

        for (int b = 0; b <= n_bins; b++) edges_buf(j, b) = local_edges[b];
        for (int b = 0; b < n_bins; b++) density_buf(j, b) = local_density[b];
    }

    return {edges, density};
}

std::pair<py::array_t<double>, py::array_t<double>> equal_frequency_histogram(
    py::array_t<double> X, int n_bins) {
    auto buf = X.unchecked<2>();
    const int n = static_cast<int>(buf.shape(0));
    const int n_dim = static_cast<int>(buf.shape(1));

    py::array_t<double> edges({n_dim, n_bins + 1});
    py::array_t<double> density({n_dim, n_bins});
    auto edges_buf = edges.mutable_unchecked<2>();
    auto density_buf = density.mutable_unchecked<2>();

    WK_PARALLEL_FOR
    for (int j = 0; j < n_dim; j++) {
        std::vector<double> column(n);
        for (int i = 0; i < n; i++) column[i] = buf(i, j);
        std::sort(column.begin(), column.end());

        // Quantile edges: bin b spans the samples in [b*n/n_bins,
        // (b+1)*n/n_bins), so counts are equal and widths adapt to density.
        std::vector<double> local_edges(n_bins + 1);
        std::vector<double> counts(n_bins, 0.0);
        for (int b = 0; b <= n_bins; b++) {
            const int index = std::min(
                n - 1, static_cast<int>(static_cast<double>(b) * n / n_bins));
            local_edges[b] = column[index];
        }
        local_edges[n_bins] = column[n - 1];
        // Guard the top edge so the largest sample falls inside the last bin.
        if (local_edges[n_bins] <= local_edges[n_bins - 1]) {
            local_edges[n_bins] = local_edges[n_bins - 1] +
                                  std::max(1e-12, std::abs(local_edges[n_bins - 1]) * 1e-9);
        }

        for (int i = 0; i < n; i++) {
            const auto it = std::upper_bound(local_edges.begin(),
                                             local_edges.end(), column[i]);
            int b = static_cast<int>(it - local_edges.begin()) - 1;
            b = std::min(std::max(b, 0), n_bins - 1);
            counts[b] += 1.0;
        }

        std::vector<double> local_density(n_bins);
        counts_to_density(counts, local_edges, n_bins, n, local_density.data());

        for (int b = 0; b <= n_bins; b++) edges_buf(j, b) = local_edges[b];
        for (int b = 0; b < n_bins; b++) density_buf(j, b) = local_density[b];
    }

    return {edges, density};
}

py::array_t<double> histogram_density(py::array_t<double> edges,
                                      py::array_t<double> density,
                                      py::array_t<double> X_query) {
    auto edges_buf = edges.unchecked<2>();
    auto density_buf = density.unchecked<2>();
    auto query = X_query.unchecked<2>();

    const int n_query = static_cast<int>(query.shape(0));
    const int n_dim = static_cast<int>(query.shape(1));
    const int n_bins = static_cast<int>(density_buf.shape(1));

    py::array_t<double> result({n_query, n_dim});
    auto res = result.mutable_unchecked<2>();

    WK_PARALLEL_FOR
    for (int j = 0; j < n_dim; j++) {
        std::vector<double> local_edges(n_bins + 1);
        for (int b = 0; b <= n_bins; b++) local_edges[b] = edges_buf(j, b);

        for (int i = 0; i < n_query; i++) {
            const double value = query(i, j);
            const auto it = std::upper_bound(local_edges.begin(),
                                             local_edges.end(), value);
            int b = static_cast<int>(it - local_edges.begin()) - 1;
            // Clamping rather than returning zero keeps an unseen extreme
            // scored by its nearest bin instead of an infinite surprise.
            b = std::min(std::max(b, 0), n_bins - 1);
            res(i, j) = density_buf(j, b);
        }
    }

    return result;
}

}  // namespace stats
}  // namespace tuiml
