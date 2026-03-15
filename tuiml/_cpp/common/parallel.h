#pragma once

#ifdef TUIML_USE_OPENMP
#include <omp.h>
#define WK_PARALLEL_FOR _Pragma("omp parallel for schedule(dynamic)")
#define WK_NUM_THREADS(n) omp_set_num_threads(n)
#define WK_GET_NUM_THREADS() omp_get_max_threads()
#else
#define WK_PARALLEL_FOR
#define WK_NUM_THREADS(n) ((void)(n))
#define WK_GET_NUM_THREADS() 1
#endif

namespace tuiml {

inline int get_num_threads() {
    return WK_GET_NUM_THREADS();
}

inline void set_num_threads(int n) {
    WK_NUM_THREADS(n);
}

}  // namespace tuiml
