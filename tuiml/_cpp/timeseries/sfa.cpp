#include "sfa.h"
#include "../common/parallel.h"

#include <algorithm>
#include <cmath>
#include <complex>
#include <vector>

namespace tuiml {
namespace timeseries {

namespace {

constexpr double kTwoPi = 6.283185307179586476925286766559;

}  // namespace

py::array_t<double> sfa_transform(py::array_t<double> X, int window_size,
                                  int word_length, bool norm_mean) {
    auto x_buf = X.unchecked<2>();
    const int n_series = static_cast<int>(x_buf.shape(0));
    const int n = static_cast<int>(x_buf.shape(1));
    const int n_windows = n - window_size + 1;

    // Each retained coefficient contributes a real and an imaginary part, and
    // the DC term is skipped when the windows are mean-normalised.
    const int n_coefficients = (word_length + 1) / 2 + (norm_mean ? 1 : 0);

    py::array_t<double> result({n_series, n_windows, word_length});
    auto out = result.mutable_unchecked<3>();

    // Twiddle factors for advancing the window by one step.
    std::vector<std::complex<double>> twiddle(n_coefficients);
    for (int k = 0; k < n_coefficients; k++) {
        const double angle = kTwoPi * k / window_size;
        twiddle[k] = std::complex<double>(std::cos(angle), std::sin(angle));
    }

    WK_PARALLEL_FOR
    for (int i = 0; i < n_series; i++) {
        std::vector<std::complex<double>> coefficients(n_coefficients);

        // Centre the series before accumulating squares: the running variance
        // below is E[x^2] - mean^2, which cancels badly far from zero.
        //
        // Only valid when norm_mean drops the DC coefficient, since that is
        // the one term a constant shift changes. With the DC term retained the
        // series must be left exactly as given.
        double series_mean = 0.0;
        if (norm_mean) {
            for (int t = 0; t < n; t++) series_mean += x_buf(i, t);
            series_mean /= n;
        }

        std::vector<double> centred(n);
        for (int t = 0; t < n; t++) centred[t] = x_buf(i, t) - series_mean;

        // Running window sums, kept in step with the sliding DFT.
        double sum = 0.0, sum_squares = 0.0;
        for (int t = 0; t < window_size; t++) {
            sum += centred[t];
            sum_squares += centred[t] * centred[t];
        }

        // Direct DFT of the first window; every later window is derived from
        // it by the momentary Fourier update below.
        for (int k = 0; k < n_coefficients; k++) {
            std::complex<double> total(0.0, 0.0);
            for (int j = 0; j < window_size; j++) {
                const double angle = -kTwoPi * k * j / window_size;
                total += centred[j] *
                         std::complex<double>(std::cos(angle), std::sin(angle));
            }
            coefficients[k] = total;
        }

        for (int w = 0; w < n_windows; w++) {
            if (w > 0) {
                // Sliding by one: drop the departing sample, add the arriving
                // one, then rotate. One complex multiply per coefficient.
                const double leaving = centred[w - 1];
                const double entering = centred[w - 1 + window_size];
                for (int k = 0; k < n_coefficients; k++) {
                    coefficients[k] =
                        (coefficients[k] - leaving + entering) * twiddle[k];
                }
                sum += entering - leaving;
                sum_squares += entering * entering - leaving * leaving;
            }

            // The window's standard deviation must come from the running sums,
            // not from Parseval over the retained coefficients: only the lowest
            // few are kept, so their energy accounts for a small and varying
            // fraction of the window's variance.
            double scale = 1.0;
            if (norm_mean) {
                const double mean = sum / window_size;
                const double variance = sum_squares / window_size - mean * mean;
                scale = variance > 1e-12 ? 1.0 / std::sqrt(variance) : 1.0;
            }

            const int first = norm_mean ? 1 : 0;
            for (int f = 0; f < word_length; f++) {
                const int k = first + f / 2;
                const double value = (f % 2 == 0)
                                         ? coefficients[k].real()
                                         : coefficients[k].imag();
                out(i, w, f) = value * scale / window_size;
            }
        }
    }

    return result;
}

}  // namespace timeseries
}  // namespace tuiml
