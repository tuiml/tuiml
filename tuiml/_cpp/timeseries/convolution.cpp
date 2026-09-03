#include "convolution.h"
#include "../common/parallel.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <random>
#include <utility>
#include <vector>

namespace tuiml {
namespace timeseries {

namespace {

constexpr int kKernelLength = 9;
constexpr int kNumGamma = 3;
constexpr int kNumKernels = 84;  // C(9, 3)

// The 84 index triples, generated once in a fixed order.
const std::vector<std::array<int, kNumGamma>>& kernel_table() {
    static const std::vector<std::array<int, kNumGamma>> table = [] {
        std::vector<std::array<int, kNumGamma>> out;
        for (int i = 0; i < kKernelLength; i++)
            for (int j = i + 1; j < kKernelLength; j++)
                for (int k = j + 1; k < kKernelLength; k++)
                    out.push_back({i, j, k});
        return out;
    }();
    return table;
}

// Convolution scratch for one (series, dilation) pair.
//
// A kernel is -1 everywhere except at three gamma positions holding +2. Rather
// than convolving 84 times, the all -1 convolution is computed once as
// ``alpha`` and the per-position +3 corrections are cached in ``gamma``; a
// kernel's output is then alpha plus three cached rows. That is what makes
// MINIROCKET orders of magnitude cheaper than random-kernel ROCKET.
struct DilationWorkspace {
    std::vector<double> alpha;  // length n
    std::vector<double> gamma;  // kKernelLength * n
};

void convolve_dilation(const double* series, int n, int dilation,
                       DilationWorkspace& work) {
    work.alpha.assign(n, 0.0);
    work.gamma.assign(static_cast<size_t>(kKernelLength) * n, 0.0);

    // alpha starts as -X; gamma's centre row is 3X. The remaining rows are
    // shifted copies, one per kernel position either side of centre.
    for (int t = 0; t < n; t++) {
        work.alpha[t] = -series[t];
        work.gamma[static_cast<size_t>(4) * n + t] = 3.0 * series[t];
    }

    // Positions 0..3 look backwards, 5..8 forwards, by multiples of dilation.
    for (int position = 0; position < 4; position++) {
        const int shift = (4 - position) * dilation;
        double* row = work.gamma.data() + static_cast<size_t>(position) * n;
        for (int t = shift; t < n; t++) {
            work.alpha[t] += -series[t - shift];
            row[t] = 3.0 * series[t - shift];
        }
    }
    for (int position = 5; position < kKernelLength; position++) {
        const int shift = (position - 4) * dilation;
        double* row = work.gamma.data() + static_cast<size_t>(position) * n;
        for (int t = 0; t + shift < n; t++) {
            work.alpha[t] += -series[t + shift];
            row[t] = 3.0 * series[t + shift];
        }
    }
}

// Output of one kernel at one dilation, written into ``out``.
void kernel_output(const DilationWorkspace& work, int n,
                   const std::array<int, kNumGamma>& gamma_index,
                   std::vector<double>& out) {
    out.assign(work.alpha.begin(), work.alpha.end());
    for (int g = 0; g < kNumGamma; g++) {
        const double* row =
            work.gamma.data() + static_cast<size_t>(gamma_index[g]) * n;
        for (int t = 0; t < n; t++) out[t] += row[t];
    }
}

// Proportion of values above a threshold, over a half-open index range.
double ppv(const std::vector<double>& values, int from, int to, double bias) {
    if (to <= from) return 0.0;
    int count = 0;
    for (int t = from; t < to; t++)
        if (values[t] > bias) count++;
    return static_cast<double>(count) / static_cast<double>(to - from);
}

// Half of the (kernel, dilation) combinations are scored over the whole
// padded convolution and half over the valid centre only. Mixing the two
// gives features sensitive to events near the ends as well as the middle.
//
// A dilation wide enough to leave no valid centre falls back to the full
// range. Callers should cap dilations at (n-1)/(kernel_length-1) so this
// never fires, but bias fitting and transform must agree either way — scoring
// a bias over one range and the PPV over another would be silently wrong.
std::pair<int, int> feature_range(int n, int dilation, int dilation_index,
                                  int kernel_index) {
    const bool pad = ((dilation_index + kernel_index) % 2) == 0;
    if (pad) return {0, n};

    const int padding = ((kKernelLength - 1) * dilation) / 2;
    const int from = std::min(padding, n);
    const int to = n - padding;
    if (to <= from) return {0, n};
    return {from, to};
}

}  // namespace

py::array_t<int> minirocket_kernel_indices() {
    const auto& table = kernel_table();
    py::array_t<int> result({static_cast<int>(table.size()), kNumGamma});
    auto buf = result.mutable_unchecked<2>();
    for (size_t i = 0; i < table.size(); i++)
        for (int g = 0; g < kNumGamma; g++)
            buf(static_cast<py::ssize_t>(i), g) = table[i][g];
    return result;
}

py::array_t<double> minirocket_biases(py::array_t<double> X,
                                      py::array_t<int> dilations,
                                      py::array_t<int> features_per_dilation,
                                      py::array_t<double> quantiles,
                                      unsigned int seed) {
    auto x_buf = X.unchecked<2>();
    auto dil = dilations.unchecked<1>();
    auto per_dil = features_per_dilation.unchecked<1>();
    auto quant = quantiles.unchecked<1>();

    const int n_series = static_cast<int>(x_buf.shape(0));
    const int n = static_cast<int>(x_buf.shape(1));
    const int n_dilations = static_cast<int>(dil.shape(0));
    const auto& table = kernel_table();

    int total = 0;
    for (int d = 0; d < n_dilations; d++) total += per_dil(d) * kNumKernels;

    py::array_t<double> biases(total);
    auto bias_buf = biases.mutable_unchecked<1>();

    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> pick(0, n_series - 1);

    DilationWorkspace work;
    std::vector<double> output(n), sorted;
    int cursor = 0, quantile_cursor = 0;

    for (int d = 0; d < n_dilations; d++) {
        const int count = per_dil(d);
        if (count == 0) continue;

        for (int k = 0; k < kNumKernels; k++) {
            // One randomly chosen training series sets this kernel's biases,
            // as in the reference implementation: sampling rather than using
            // the whole set is what keeps fitting near-instant.
            const int row = pick(rng);
            std::vector<double> series(n);
            for (int t = 0; t < n; t++) series[t] = x_buf(row, t);
            convolve_dilation(series.data(), n, dil(d), work);
            kernel_output(work, n, table[k], output);

            const auto range = feature_range(n, dil(d), d, k);
            sorted.assign(output.begin() + range.first,
                          output.begin() + range.second);
            std::sort(sorted.begin(), sorted.end());

            for (int f = 0; f < count; f++) {
                const double q = quant(quantile_cursor % quant.shape(0));
                quantile_cursor++;
                const double position = q * (sorted.size() - 1);
                const size_t lower = static_cast<size_t>(position);
                const size_t upper = std::min(lower + 1, sorted.size() - 1);
                const double weight = position - lower;
                bias_buf(cursor++) =
                    sorted[lower] * (1.0 - weight) + sorted[upper] * weight;
            }
        }
    }

    return biases;
}

py::array_t<double> minirocket_transform(py::array_t<double> X,
                                         py::array_t<int> dilations,
                                         py::array_t<int> features_per_dilation,
                                         py::array_t<double> biases) {
    auto x_buf = X.unchecked<2>();
    auto dil = dilations.unchecked<1>();
    auto per_dil = features_per_dilation.unchecked<1>();
    auto bias_buf = biases.unchecked<1>();

    const int n_series = static_cast<int>(x_buf.shape(0));
    const int n = static_cast<int>(x_buf.shape(1));
    const int n_dilations = static_cast<int>(dil.shape(0));
    const auto& table = kernel_table();

    int n_features = 0;
    for (int d = 0; d < n_dilations; d++) n_features += per_dil(d) * kNumKernels;

    py::array_t<double> result({n_series, n_features});
    auto out_buf = result.mutable_unchecked<2>();

    WK_PARALLEL_FOR
    for (int i = 0; i < n_series; i++) {
        DilationWorkspace work;
        std::vector<double> series(n), output(n);
        for (int t = 0; t < n; t++) series[t] = x_buf(i, t);

        int cursor = 0;
        for (int d = 0; d < n_dilations; d++) {
            const int count = per_dil(d);
            if (count == 0) continue;
            convolve_dilation(series.data(), n, dil(d), work);

            for (int k = 0; k < kNumKernels; k++) {
                kernel_output(work, n, table[k], output);
                const auto range = feature_range(n, dil(d), d, k);

                for (int f = 0; f < count; f++) {
                    out_buf(i, cursor) =
                        ppv(output, range.first, range.second, bias_buf(cursor));
                    cursor++;
                }
            }
        }
    }

    return result;
}

}  // namespace timeseries
}  // namespace tuiml
