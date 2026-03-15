#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <stdexcept>

namespace py = pybind11;

namespace tuiml {

// Zero-copy read access to numpy arrays
template <typename T, int ndim>
inline auto unchecked(const py::array_t<T>& arr) {
    return arr.template unchecked<ndim>();
}

// Get shape of a numpy array
template <typename T>
inline ssize_t shape(const py::array_t<T>& arr, int dim) {
    return arr.shape(dim);
}

// Create a 1D numpy array from a vector
template <typename T>
inline py::array_t<T> to_numpy(const std::vector<T>& vec) {
    py::array_t<T> result(vec.size());
    auto buf = result.mutable_unchecked<1>();
    for (size_t i = 0; i < vec.size(); i++) {
        buf(i) = vec[i];
    }
    return result;
}

// Create a 2D numpy array from nested vectors
template <typename T>
inline py::array_t<T> to_numpy_2d(const std::vector<std::vector<T>>& data) {
    if (data.empty()) return py::array_t<T>({0, 0});
    ssize_t rows = data.size();
    ssize_t cols = data[0].size();
    py::array_t<T> result({rows, cols});
    auto buf = result.mutable_unchecked<2>();
    for (ssize_t i = 0; i < rows; i++) {
        for (ssize_t j = 0; j < cols; j++) {
            buf(i, j) = data[i][j];
        }
    }
    return result;
}

}  // namespace tuiml
