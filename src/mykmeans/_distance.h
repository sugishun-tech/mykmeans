#ifndef MYKMEANS_DISTANCE_H
#define MYKMEANS_DISTANCE_H

#include <stddef.h>
#include <math.h>
#if defined(_MSC_VER)
#define MYK_RESTRICT __restrict
#else
#define MYK_RESTRICT __restrict__
#endif
#ifdef _OPENMP
#include <omp.h>
#endif

/* No fast-math: inputs are validated by the Python estimator.  Accumulating
 * float32 differences in double avoids overflow and improves reproducibility.
 * OpenMP SIMD permits vector reduction without changing the scalar interface.
 */
static inline double myk_sqdist_f32(const float *a, const float *b, ptrdiff_t d) {
    double total = 0.0;
    ptrdiff_t j;
    if (d <= 4) {
        double v0 = (double)a[0] - (double)b[0];
        total = v0*v0;
        if (d > 1) { double v1 = (double)a[1] - (double)b[1]; total += v1*v1; }
        if (d > 2) { double v2 = (double)a[2] - (double)b[2]; total += v2*v2; }
        if (d > 3) { double v3 = (double)a[3] - (double)b[3]; total += v3*v3; }
        return total;
    }
    #if defined(_OPENMP) && _OPENMP >= 201307
    #pragma omp simd reduction(+:total)
    #endif
    for (j = 0; j < d; ++j) {
        double delta = (double)a[j] - (double)b[j];
        total += delta * delta;
    }
    return total;
}

static inline double myk_sqdist_f64(const double *a, const double *b, ptrdiff_t d) {
    double total = 0.0;
    ptrdiff_t j;
    if (d <= 4) {
        double v0 = a[0] - b[0];
        total = v0*v0;
        if (d > 1) { double v1 = a[1] - b[1]; total += v1*v1; }
        if (d > 2) { double v2 = a[2] - b[2]; total += v2*v2; }
        if (d > 3) { double v3 = a[3] - b[3]; total += v3*v3; }
        return total;
    }
    #if defined(_OPENMP) && _OPENMP >= 201307
    #pragma omp simd reduction(+:total)
    #endif
    for (j = 0; j < d; ++j) {
        double delta = a[j] - b[j];
        total += delta * delta;
    }
    return total;
}

static inline void myk_add_f32(double * MYK_RESTRICT out,
                               const float * MYK_RESTRICT x,
                               ptrdiff_t d, double w) {
    ptrdiff_t j;
    #if defined(_OPENMP) && _OPENMP >= 201307
    #pragma omp simd
    #endif
    for (j=0; j<d; ++j) out[j] += w*(double)x[j];
}

static inline void myk_add_f64(double * MYK_RESTRICT out,
                               const double * MYK_RESTRICT x,
                               ptrdiff_t d, double w) {
    ptrdiff_t j;
    #if defined(_OPENMP) && _OPENMP >= 201307
    #pragma omp simd
    #endif
    for (j=0; j<d; ++j) out[j] += w*x[j];
}

static inline int myk_openmp_enabled(void) {
    #ifdef _OPENMP
    return 1;
    #else
    return 0;
    #endif
}

static inline int myk_max_threads(void) {
    #ifdef _OPENMP
    return omp_get_max_threads();
    #else
    return 1;
    #endif
}

static inline int myk_native_build(void) {
    #ifdef MYKMEANS_NATIVE_BUILD
    return MYKMEANS_NATIVE_BUILD;
    #else
    return 0;
    #endif
}

#endif
