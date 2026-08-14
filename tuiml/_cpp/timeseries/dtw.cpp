#include "dtw.h"
#include "../common/parallel.h"

#include <algorithm>
#include <cmath>
#include <deque>
#include <limits>
#include <vector>

namespace tuiml {
namespace timeseries {

namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

// Core DTW over raw pointers, so both the univariate and multivariate paths
// share one implementation. ``channels`` folds the per-channel squared
// differences into a single local cost, giving dependent DTW.
double dtw_core(const double* a, int n, const double* b, int m, int channels,
                int stride_a, int stride_b, int window, double cutoff,
                std::vector<double>& previous, std::vector<double>& current) {
    if (n == 0 || m == 0) return (n == m) ? 0.0 : kInf;

    // An unconstrained band still has to reach the far corner, so it must be
    // at least as wide as the length difference.
    if (window < 0) window = std::max(n, m);
    window = std::max(window, std::abs(n - m));

    previous.assign(m + 1, kInf);
    current.assign(m + 1, kInf);
    previous[0] = 0.0;

    for (int i = 1; i <= n; i++) {
        const int lo = std::max(1, i - window);
        const int hi = std::min(m, i + window);

        std::fill(current.begin(), current.end(), kInf);
        double row_best = kInf;

        for (int j = lo; j <= hi; j++) {
            double cost = 0.0;
            for (int c = 0; c < channels; c++) {
                const double d = a[(i - 1) * stride_a + c] - b[(j - 1) * stride_b + c];
                cost += d * d;
            }

            const double best = std::min({previous[j], current[j - 1], previous[j - 1]});
            current[j] = (best == kInf) ? kInf : cost + best;
            row_best = std::min(row_best, current[j]);
        }

        // Every path through this row already exceeds the cutoff, so no
        // continuation can come in under it.
        if (cutoff > 0.0 && row_best > cutoff) return kInf;

        previous.swap(current);
    }

    const double total = previous[m];
    return (total == kInf) ? kInf : std::sqrt(total);
}

// Streaming min/max over a sliding window, using the standard monotonic
// deque so the envelope costs O(n) rather than O(n * window).
void envelope_core(const double* series, int n, int window, double* lower,
                   double* upper) {
    std::deque<int> min_deque, max_deque;
    for (int i = 0; i < n + window; i++) {
        if (i < n) {
            while (!min_deque.empty() && series[min_deque.back()] >= series[i])
                min_deque.pop_back();
            min_deque.push_back(i);
            while (!max_deque.empty() && series[max_deque.back()] <= series[i])
                max_deque.pop_back();
            max_deque.push_back(i);
        }

        const int centre = i - window;
        if (centre >= 0) {
            while (!min_deque.empty() && min_deque.front() < centre - window)
                min_deque.pop_front();
            while (!max_deque.empty() && max_deque.front() < centre - window)
                max_deque.pop_front();
            lower[centre] = series[min_deque.front()];
            upper[centre] = series[max_deque.front()];
        }
    }
}

double lb_keogh_core(const double* query, const double* lower,
                     const double* upper, int n) {
    double total = 0.0;
    for (int i = 0; i < n; i++) {
        if (query[i] > upper[i]) {
            const double d = query[i] - upper[i];
            total += d * d;
        } else if (query[i] < lower[i]) {
            const double d = lower[i] - query[i];
            total += d * d;
        }
    }
    return std::sqrt(total);
}

}  // namespace

double dtw_distance(py::array_t<double> a, py::array_t<double> b, int window,
                    double cutoff) {
    auto a_buf = a.unchecked<1>();
    auto b_buf = b.unchecked<1>();
    std::vector<double> previous, current;
    return dtw_core(a_buf.data(0), static_cast<int>(a_buf.shape(0)),
                    b_buf.data(0), static_cast<int>(b_buf.shape(0)), 1, 1, 1,
                    window, cutoff, previous, current);
}

std::pair<py::array_t<double>, py::array_t<double>> lb_keogh_envelope(
    py::array_t<double> series, int window) {
    auto buf = series.unchecked<1>();
    const int n = static_cast<int>(buf.shape(0));
    if (window < 0) window = n;

    py::array_t<double> lower(n), upper(n);
    envelope_core(buf.data(0), n, window, lower.mutable_data(0),
                  upper.mutable_data(0));
    return {lower, upper};
}

double lb_keogh(py::array_t<double> query, py::array_t<double> lower,
                py::array_t<double> upper) {
    auto q = query.unchecked<1>();
    return lb_keogh_core(q.data(0), lower.unchecked<1>().data(0),
                         upper.unchecked<1>().data(0),
                         static_cast<int>(q.shape(0)));
}

namespace {

// Copy one (channels, length) series into time-major order, so the DTW inner
// loop reads all channels of a timestep contiguously.
template <typename Accessor>
void gather_series(const Accessor& buf, int row, int channels, int length,
                   std::vector<double>& out) {
    out.resize(static_cast<size_t>(length) * channels);
    for (int t = 0; t < length; t++)
        for (int c = 0; c < channels; c++)
            out[static_cast<size_t>(t) * channels + c] = buf(row, c, t);
}

}  // namespace

py::array_t<double> dtw_pairwise(py::array_t<double> A, py::array_t<double> B,
                                 int window) {
    // Both inputs arrive as (n, channels, length); the univariate case is
    // simply channels == 1, so there is one code path.
    auto a_buf = A.unchecked<3>();
    auto b_buf = B.unchecked<3>();

    const int n_a = static_cast<int>(a_buf.shape(0));
    const int n_b = static_cast<int>(b_buf.shape(0));
    const int channels = static_cast<int>(a_buf.shape(1));
    const int len_a = static_cast<int>(a_buf.shape(2));
    const int len_b = static_cast<int>(b_buf.shape(2));

    py::array_t<double> result({n_a, n_b});
    auto res = result.mutable_unchecked<2>();

    WK_PARALLEL_FOR
    for (int i = 0; i < n_a; i++) {
        std::vector<double> query, reference, previous, current;
        gather_series(a_buf, i, channels, len_a, query);

        for (int j = 0; j < n_b; j++) {
            gather_series(b_buf, j, channels, len_b, reference);
            res(i, j) = dtw_core(query.data(), len_a, reference.data(), len_b,
                                 channels, channels, channels, window, 0.0,
                                 previous, current);
        }
    }

    return result;
}

std::pair<py::array_t<double>, py::array_t<int>> dtw_knn(py::array_t<double> A,
                                                         py::array_t<double> B,
                                                         int k, int window) {
    auto a_buf = A.unchecked<3>();
    auto b_buf = B.unchecked<3>();

    const int n_a = static_cast<int>(a_buf.shape(0));
    const int n_b = static_cast<int>(b_buf.shape(0));
    const int channels = static_cast<int>(a_buf.shape(1));
    const int len_a = static_cast<int>(a_buf.shape(2));
    const int len_b = static_cast<int>(b_buf.shape(2));

    k = std::min(k, n_b);
    py::array_t<double> distances({n_a, k});
    py::array_t<int> indices({n_a, k});
    auto dist_buf = distances.mutable_unchecked<2>();
    auto index_buf = indices.mutable_unchecked<2>();

    // LB_Keogh needs a single channel and equal lengths; outside that the
    // search still runs, just without the bound.
    const bool bound_applies = (channels == 1) && (len_a == len_b);
    const int effective_window = (window < 0) ? len_b : window;

    // Envelopes of the reference set, computed once and shared by all queries.
    std::vector<double> lower_all, upper_all;
    if (bound_applies) {
        lower_all.resize(static_cast<size_t>(n_b) * len_b);
        upper_all.resize(static_cast<size_t>(n_b) * len_b);
        std::vector<double> series(len_b);
        for (int j = 0; j < n_b; j++) {
            for (int t = 0; t < len_b; t++) series[t] = b_buf(j, 0, t);
            envelope_core(series.data(), len_b, effective_window,
                          lower_all.data() + static_cast<size_t>(j) * len_b,
                          upper_all.data() + static_cast<size_t>(j) * len_b);
        }
    }

    WK_PARALLEL_FOR
    for (int i = 0; i < n_a; i++) {
        std::vector<double> query, reference, previous, current;
        gather_series(a_buf, i, channels, len_a, query);

        // Order candidates by their lower bound, so the k-th best tightens as
        // early as possible and prunes as much of the tail as possible.
        std::vector<std::pair<double, int>> candidates(n_b);
        for (int j = 0; j < n_b; j++) {
            const double bound =
                bound_applies
                    ? lb_keogh_core(
                          query.data(),
                          lower_all.data() + static_cast<size_t>(j) * len_b,
                          upper_all.data() + static_cast<size_t>(j) * len_b,
                          len_a)
                    : 0.0;
            candidates[j] = {bound, j};
        }
        if (bound_applies) std::sort(candidates.begin(), candidates.end());

        // Max-heap of the k best found so far, keyed by distance.
        std::vector<std::pair<double, int>> best;
        best.reserve(k + 1);
        double worst = kInf;

        for (const auto& candidate : candidates) {
            // The bound never exceeds the true distance, so a candidate whose
            // bound already loses cannot win. Because candidates are sorted by
            // bound, no later one can either.
            if (static_cast<int>(best.size()) == k && candidate.first >= worst) break;

            gather_series(b_buf, candidate.second, channels, len_b, reference);
            // Squared cutoff: dtw_core accumulates squared cost and only takes
            // the root at the end.
            const double cutoff =
                (static_cast<int>(best.size()) == k) ? worst * worst : 0.0;
            const double d =
                dtw_core(query.data(), len_a, reference.data(), len_b, channels,
                         channels, channels, window, cutoff, previous, current);
            if (d == kInf) continue;  // early abandoned: cannot make the top k

            best.emplace_back(d, candidate.second);
            std::push_heap(best.begin(), best.end());
            if (static_cast<int>(best.size()) > k) {
                std::pop_heap(best.begin(), best.end());
                best.pop_back();
            }
            if (static_cast<int>(best.size()) == k) worst = best.front().first;
        }

        std::sort_heap(best.begin(), best.end());
        for (int slot = 0; slot < k; slot++) {
            if (slot < static_cast<int>(best.size())) {
                dist_buf(i, slot) = best[slot].first;
                index_buf(i, slot) = best[slot].second;
            } else {
                dist_buf(i, slot) = kInf;
                index_buf(i, slot) = -1;
            }
        }
    }

    return {distances, indices};
}

}  // namespace timeseries
}  // namespace tuiml
