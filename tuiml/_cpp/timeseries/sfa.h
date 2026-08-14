#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

namespace tuiml {
namespace timeseries {

// Symbolic Fourier Approximation: the low-frequency DFT coefficients of every
// sliding window of every series.
//
// Returns (n_series, n_windows, word_length) real values, laid out as
// alternating real and imaginary parts of the lowest Fourier coefficients.
// When ``norm_mean`` is true the DC term is dropped and each window is scaled
// by its own standard deviation, making the representation invariant to that
// window's offset and amplitude.
//
// Windows are advanced with the momentary Fourier transform: sliding by one
// step updates every coefficient with a single complex multiply, so the whole
// pass costs O(L * word_length) per series instead of O(L * window * word).
py::array_t<double> sfa_transform(py::array_t<double> X, int window_size,
                                  int word_length, bool norm_mean);

}  // namespace timeseries
}  // namespace tuiml
