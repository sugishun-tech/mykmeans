# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False
# cython: nonecheck=False, cdivision=True
"""Dense, weighted k-means kernels, written independently in Cython.

Hamerly bounds follow G. Hamerly, *Making k-means even faster*, SDM 2010.
Elkan uses per-center lower bounds (ICML 2003). Exponion uses the candidate
ball of Newling and Fleuret (ICML 2016), with full neighbor sorting rather
than the paper's partial-sort buckets. All kernels are independently written.
Elkan needs O(n*k + k*k) bounds; Exponion needs O(n + k*k). Lloyd/Hamerly
avoid an n-by-k work array. The Python layer owns validation.
"""

import numpy as np
cimport numpy as cnp
from cython.parallel cimport prange
from libc.math cimport sqrt, INFINITY, fmax
from libc.math cimport isfinite
from libc.float cimport DBL_EPSILON, FLT_EPSILON, DBL_MIN, FLT_MIN, DBL_MAX, FLT_MAX
from libc.stdint cimport int32_t, int64_t
from scipy.linalg.cython_blas cimport dgemm, sgemm

cnp.import_array()

ctypedef fused floating:
    float
    double

cdef extern from "_distance.h" nogil:
    double myk_sqdist_f32(const float*, const float*, Py_ssize_t) noexcept
    double myk_sqdist_f64(const double*, const double*, Py_ssize_t) noexcept
    void myk_add_f32(double*, const float*, Py_ssize_t, double) noexcept
    void myk_add_f64(double*, const double*, Py_ssize_t, double) noexcept
    int myk_openmp_enabled() noexcept
    int myk_max_threads() noexcept
    int myk_native_build() noexcept


def openmp_enabled():
    """Whether this extension was compiled with OpenMP."""
    return bool(myk_openmp_enabled())


def max_threads():
    """Current OpenMP limit (respects threadpoolctl and OMP_NUM_THREADS)."""
    return myk_max_threads()


def build_info():
    """Compiler choices relevant to reproducing a benchmark."""
    return {'openmp': bool(myk_openmp_enabled()),
            'native': bool(myk_native_build()),
            'max_threads': myk_max_threads(), 'fast_math': False,
            'distance_accumulation': 'float64'}


cdef inline double _sqdist(const floating* x, const floating* y,
                           Py_ssize_t d) noexcept nogil:
    if floating is float:
        return myk_sqdist_f32(x, y, d)
    else:
        return myk_sqdist_f64(x, y, d)


cdef inline double _norm(const floating* x, Py_ssize_t d) noexcept nogil:
    cdef Py_ssize_t j
    cdef double value, result = 0.0
    for j in range(d):
        value = <double>x[j]
        result += value*value
    return result


cdef inline void _add(double* result, const floating* x, Py_ssize_t d,
                     double weight) noexcept nogil:
    if floating is float:
        myk_add_f32(result,x,d,weight)
    else:
        myk_add_f64(result,x,d,weight)


cdef inline int _nearest_plain(const floating* x, const floating* c,
                               Py_ssize_t k, Py_ssize_t d,
                               double* best) noexcept nogil:
    cdef Py_ssize_t j
    cdef int label = 0
    cdef double dd, v0,v1,v2,v3, best_d = INFINITY
    # Dispatch once per observation, keeping branches out of the center loop.
    if d == 1:
        for j in range(k):
            v0 = <double>x[0]-<double>c[j]
            dd = v0*v0
            if dd < best_d:
                best_d = dd
                label = <int>j
    elif d == 2:
        for j in range(k):
            v0 = <double>x[0]-<double>c[2*j]
            v1 = <double>x[1]-<double>c[2*j+1]
            dd = v0*v0+v1*v1
            if dd < best_d:
                best_d = dd
                label = <int>j
    elif d == 3:
        for j in range(k):
            v0 = <double>x[0]-<double>c[3*j]
            v1 = <double>x[1]-<double>c[3*j+1]
            v2 = <double>x[2]-<double>c[3*j+2]
            dd = v0*v0+v1*v1+v2*v2
            if dd < best_d:
                best_d = dd
                label = <int>j
    elif d == 4:
        for j in range(k):
            v0 = <double>x[0]-<double>c[4*j]
            v1 = <double>x[1]-<double>c[4*j+1]
            v2 = <double>x[2]-<double>c[4*j+2]
            v3 = <double>x[3]-<double>c[4*j+3]
            dd = v0*v0+v1*v1+v2*v2+v3*v3
            if dd < best_d:
                best_d = dd
                label = <int>j
    else:
        for j in range(k):
            dd = _sqdist(x,c+j*d,d)
            if dd < best_d:
                best_d = dd
                label = <int>j
    best[0] = best_d
    return label


cdef inline int _nearest(const floating* x, const floating* c,
                         Py_ssize_t k, Py_ssize_t d,
                         double* best, double* second) noexcept nogil:
    cdef Py_ssize_t j
    cdef int label = 0
    cdef double dd, best_d = INFINITY, next_d = INFINITY
    for j in range(k):
        dd = _sqdist(x, c + j*d, d)
        if dd < best_d:
            next_d = best_d
            best_d = dd
            label = <int>j
        elif dd < next_d:
            next_d = dd
    best[0] = best_d
    second[0] = next_d
    return label


cdef int _threads(int requested, Py_ssize_t n, Py_ssize_t work):
    cdef int nt
    if requested < 1:
        raise ValueError("n_threads must be a positive integer")
    nt = min(requested, max(1, <int>((n + 1023) // 1024)))
    if work < 32768 or not myk_openmp_enabled():
        nt = 1
    return nt


cdef void _plain_block(const floating* x, const floating* centers,
                       const double* weights, Py_ssize_t first,
                       Py_ssize_t last, Py_ssize_t k, Py_ssize_t d,
                       int32_t* labels, double* distances,
                       double* sums, double* masses, int64_t* counts,
                       int64_t* changes, int64_t* evaluations,
                       bint accumulate) noexcept nogil:
    cdef Py_ssize_t i, j
    cdef int label
    cdef double best, second, w
    cdef int64_t changed = 0
    for i in range(first, last):
        label = _nearest_plain(x + i*d, centers, k, d, &best)
        if labels[i] != label:
            changed += 1
        labels[i] = label
        distances[i] = best
        if accumulate:
            w = weights[i]
            if w > 0:
                masses[label] += w
                counts[label] += 1
                _add(sums+label*d,x+i*d,d,w)
    changes[0] = changed
    evaluations[0] = (last-first)*k


cdef void _restored_block(const floating* x, const floating* offset,
                          const floating* centers, Py_ssize_t first,
                          Py_ssize_t last, Py_ssize_t k, Py_ssize_t d,
                          int32_t* labels, double* distances,
                          floating* scratch) noexcept nogil:
    """Assign exactly the values the in-place caller restores with X+=offset.

    One d-vector per thread avoids another n*d input copy. The cast deliberately
    rounds each reconstructed coordinate to the public input precision.
    """
    cdef Py_ssize_t i,j
    cdef double best
    for i in range(first,last):
        for j in range(d):
            scratch[j] = <floating>(x[i*d+j] + offset[j])
        labels[i] = _nearest_plain(scratch,centers,k,d,&best)
        distances[i] = best


cdef void _gemm_block(const floating* x, const floating* centers,
                      const double* xnorm, const double* cnorm,
                      const double* weights, Py_ssize_t first,
                      Py_ssize_t last, int k, int d,
                      int32_t* labels, double* distances,
                      double* sums, double* masses, int64_t* counts,
                      int64_t* changes, int64_t* evaluations,
                      floating* buffer, bint accumulate,
                      double* upper, double* lower_bounds) noexcept nogil:
    """Blocked dot products screen candidates; direct distances settle ties.

    For the dot-product formula the conservative error interval is bounded by
    8*d*eps*(||x||^2+||c||^2), using Cauchy--Schwarz. Any candidate whose lower
    endpoint can beat the exact winner is recomputed with direct differences.
    Thus cancellation affects performance, not the nearest-center decision.
    """
    cdef Py_ssize_t begin, row, i, j, f
    cdef int rows, label, old, initial_label
    cdef char trans = 'T', normal = 'N'
    cdef float alpha_f = -2.0, beta_f = 0.0
    cdef double alpha_d = -2.0, beta_d = 0.0
    cdef double gamma, underflow, estimate, min_estimate, next_estimate
    cdef double exact, best, lower, w, max_cnorm = 0.0, max_norm
    cdef double bound_eps = 32*DBL_EPSILON*(d+1)
    cdef double slack = 8*sqrt(d*DBL_MIN)
    cdef int64_t changed = 0, evaluated = 0
    if floating is float:
        gamma = 8.0*d*FLT_EPSILON
        underflow = 8.0*d*FLT_MIN
        max_norm = 0.25*FLT_MAX
    else:
        gamma = 8.0*d*DBL_EPSILON
        underflow = 8.0*d*DBL_MIN
        max_norm = 0.25*DBL_MAX
    for j in range(k):
        if cnorm[j] > max_cnorm:
            max_cnorm = cnorm[j]
    begin = first
    while begin < last:
        rows = <int>min(256, last-begin)
        if floating is float:
            sgemm(&trans, &normal, &k, &rows, &d, &alpha_f,
                  <float*>centers, &d, <float*>(x+begin*d), &d,
                  &beta_f, <float*>buffer, &k)
        else:
            dgemm(&trans, &normal, &k, &rows, &d, &alpha_d,
                  <double*>centers, &d, <double*>(x+begin*d), &d,
                  &beta_d, <double*>buffer, &k)
        for row in range(rows):
            i = begin+row
            label = 0
            min_estimate = INFINITY
            next_estimate = INFINITY
            for j in range(k):
                estimate = <double>buffer[row*k+j] + cnorm[j]
                if estimate < min_estimate:
                    next_estimate = min_estimate
                    min_estimate = estimate
                    label = <int>j
                elif estimate < next_estimate:
                    next_estimate = estimate
            best = _sqdist(x+i*d, centers+label*d, d)
            initial_label = label
            evaluated += 1
            # Scan every possible competitor. Including the winner's index
            # in tie comparison enforces smallest-index deterministic ties.
            lower = next_estimate+xnorm[i]-gamma*(xnorm[i]+max_cnorm)-underflow
            if not (xnorm[i]+max_cnorm < max_norm and lower > best):
                for j in range(k):
                    if j == label:
                        continue
                    estimate = <double>buffer[row*k+j] + cnorm[j] + xnorm[i]
                    lower = estimate - gamma*(xnorm[i]+cnorm[j]) - underflow
                    if not isfinite(estimate) or not lower > best:
                        exact = _sqdist(x+i*d, centers+j*d, d)
                        evaluated += 1
                        if exact < best or (exact == best and j < label):
                            best = exact
                            label = <int>j
            if upper != NULL:
                upper[i] = sqrt(best)*(1.0+bound_eps)+slack
                estimate = next_estimate if label == initial_label else min_estimate
                lower = estimate+xnorm[i]-gamma*(xnorm[i]+max_cnorm)-underflow
                if not isfinite(lower) or not xnorm[i]+max_cnorm < max_norm:
                    lower = 0.0
                lower_bounds[i] = fmax(0.0,sqrt(fmax(0.0,lower))*(1.0-bound_eps)-slack)
            old = labels[i]
            if old != label:
                changed += 1
            labels[i] = label
            distances[i] = best
            if accumulate:
                w = weights[i]
                if w > 0:
                    masses[label] += w
                    counts[label] += 1
                    _add(sums+label*d,x+i*d,d,w)
        begin += rows
    changes[0] = changed
    # GEMM evaluates every point-center dot product, plus exact candidates.
    evaluations[0] = (last-first)*k + evaluated


cdef void _bound_block(const floating* x, const floating* centers,
                       const double* weights, Py_ssize_t first,
                       Py_ssize_t last, Py_ssize_t k, Py_ssize_t d,
                       int32_t* labels, double* distances,
                       double* upper, double* lower, const double* separation,
                       double eps, bint initialized, bint incremental,
                       double* sums, double* masses, int64_t* counts,
                       int64_t* changes, int64_t* evaluations) noexcept nogil:
    cdef Py_ssize_t i, j
    cdef int label, old
    cdef double best, second, w, cutoff, distance
    cdef double slack = 8*sqrt(d*DBL_MIN)
    cdef int64_t changed = 0, evaluated = 0
    for i in range(first, last):
        old = labels[i]
        label = old
        if initialized:
            cutoff = fmax(lower[i], separation[old])
            if not upper[i] < cutoff:
                best = _sqdist(x+i*d, centers+old*d, d)
                evaluated += 1
                upper[i] = sqrt(best) * (1.0 + eps) + slack
                distances[i] = best
                if not upper[i] < cutoff:
                    label = _nearest(x+i*d, centers, k, d, &best, &second)
                    evaluated += k
                    upper[i] = sqrt(best) * (1.0 + eps) + slack
                    lower[i] = fmax(0.0, sqrt(second) * (1.0 - eps) - slack)
                    distances[i] = best
        else:
            label = _nearest(x+i*d, centers, k, d, &best, &second)
            evaluated += k
            upper[i] = sqrt(best) * (1.0 + eps) + slack
            lower[i] = fmax(0.0, sqrt(second) * (1.0 - eps) - slack)
            distances[i] = best
        if old != label:
            changed += 1
        labels[i] = label
        w = weights[i]
        if w > 0:
            if incremental:
                if old != label:
                    masses[old] -= w
                    masses[label] += w
                    counts[old] -= 1
                    counts[label] += 1
                    _add(sums+old*d,x+i*d,d,-w)
                    _add(sums+label*d,x+i*d,d,w)
            else:
                masses[label] += w
                counts[label] += 1
                _add(sums+label*d,x+i*d,d,w)
    changes[0] = changed
    evaluations[0] = evaluated


cdef void _center_geometry_row(const floating* centers, Py_ssize_t row,
                                Py_ssize_t k, Py_ssize_t d, double eps,
                                double* center_lower, double* separation,
                                double* nearest_upper) noexcept nogil:
    """Outward-slack center distances used by Elkan and Exponion."""
    cdef Py_ssize_t j
    cdef double distance, nearest = INFINITY
    cdef double slack = 8*sqrt(d*DBL_MIN)
    for j in range(k):
        if j == row:
            center_lower[row*k+j] = 0.0
        else:
            distance = sqrt(_sqdist(centers+row*d, centers+j*d, d))
            center_lower[row*k+j] = fmax(0.0, distance*(1.0-eps)-slack)
            if distance < nearest:
                nearest = distance
    separation[row] = fmax(0.0, 0.5*nearest*(1.0-eps)-slack)
    nearest_upper[row] = nearest*(1.0+eps)+slack


cdef void _elkan_block(const floating* x, const floating* centers,
                       const double* weights, Py_ssize_t first,
                       Py_ssize_t last, Py_ssize_t k, Py_ssize_t d,
                       int32_t* labels, double* distances,
                       double* upper, double* lower,
                       const double* center_lower, const double* separation,
                       double eps, bint initialized, bint incremental,
                       double* sums, double* masses, int64_t* counts,
                       int64_t* changes, int64_t* evaluations) noexcept nogil:
    """Elkan's per-point/per-center bounds and two triangle filters."""
    cdef Py_ssize_t i,j
    cdef int old, label
    cdef double best, dd, distance, w
    cdef double slack = 8*sqrt(d*DBL_MIN)
    cdef bint tight
    cdef int64_t changed=0, evaluated=0
    for i in range(first,last):
        old = labels[i]
        label = old
        tight = False
        if not initialized:
            best = INFINITY
            label = 0
            for j in range(k):
                dd = _sqdist(x+i*d,centers+j*d,d)
                evaluated += 1
                distance = sqrt(dd)
                lower[i*k+j] = fmax(0.0,distance*(1.0-eps)-slack)
                if dd < best:
                    best = dd
                    label = <int>j
            upper[i] = sqrt(best)*(1.0+eps)+slack
            distances[i] = best
        elif not upper[i] < separation[old]:
            for j in range(k):
                if j == label:
                    continue
                if upper[i] < lower[i*k+j] or upper[i] < 0.5*center_lower[label*k+j]:
                    continue
                if not tight:
                    best = _sqdist(x+i*d,centers+label*d,d)
                    evaluated += 1
                    distance = sqrt(best)
                    upper[i] = distance*(1.0+eps)+slack
                    lower[i*k+label] = fmax(0.0,distance*(1.0-eps)-slack)
                    distances[i] = best
                    tight = True
                    if upper[i] < lower[i*k+j] or upper[i] < 0.5*center_lower[label*k+j]:
                        continue
                dd = _sqdist(x+i*d,centers+j*d,d)
                evaluated += 1
                distance = sqrt(dd)
                lower[i*k+j] = fmax(0.0,distance*(1.0-eps)-slack)
                if dd < best or (dd == best and j < label):
                    best = dd
                    label = <int>j
                    upper[i] = distance*(1.0+eps)+slack
                    distances[i] = best
        if old != label:
            changed += 1
        labels[i] = label
        w = weights[i]
        if w > 0:
            if incremental:
                if old != label:
                    masses[old] -= w
                    masses[label] += w
                    counts[old] -= 1
                    counts[label] += 1
                    _add(sums+old*d,x+i*d,d,-w)
                    _add(sums+label*d,x+i*d,d,w)
            else:
                masses[label] += w
                counts[label] += 1
                _add(sums+label*d,x+i*d,d,w)
    changes[0] = changed
    evaluations[0] = evaluated


cdef void _exponion_block(const floating* x, const floating* centers,
                          const double* weights, Py_ssize_t first,
                          Py_ssize_t last, Py_ssize_t k, Py_ssize_t d,
                          int32_t* labels, double* distances,
                          double* upper, double* lower,
                          const double* separation, const double* center_lower,
                          const double* nearest_upper,
                          const cnp.intp_t* order, double eps,
                          bint initialized, bint incremental,
                          double* sums, double* masses, int64_t* counts,
                          int64_t* changes, int64_t* evaluations) noexcept nogil:
    """Hamerly bounds plus Exponion's ball containing both nearest centers.

    The ball is anchored at the previous label, with outward radius
    2*u + nearest-center-separation. Full sorted neighbor rows select its
    candidates; deterministic ties use center index, not neighbor order.
    """
    cdef Py_ssize_t i,j,p
    cdef int old,label
    cdef double best,second,dd,w,cutoff,radius
    cdef double slack = 8*sqrt(d*DBL_MIN)
    cdef int64_t changed=0,evaluated=0
    for i in range(first,last):
        old = labels[i]
        label = old
        if initialized:
            cutoff = fmax(lower[i],separation[old])
            if not upper[i] < cutoff:
                best = _sqdist(x+i*d,centers+old*d,d)
                evaluated += 1
                upper[i] = sqrt(best)*(1.0+eps)+slack
                distances[i] = best
                if not upper[i] < cutoff:
                    # nearest_upper is the full inter-center distance, whereas
                    # separation above is half that distance (rounded down).
                    radius = (2.0*upper[i]+nearest_upper[old])*(1.0+eps)+slack
                    second = INFINITY
                    for p in range(k):
                        j = order[old*k+p]
                        if center_lower[old*k+j] > radius:
                            break
                        if j == old:
                            continue
                        dd = _sqdist(x+i*d,centers+j*d,d)
                        evaluated += 1
                        if dd < best or (dd == best and j < label):
                            second = best
                            best = dd
                            label = <int>j
                        elif dd < second:
                            second = dd
                    upper[i] = sqrt(best)*(1.0+eps)+slack
                    lower[i] = fmax(0.0,sqrt(second)*(1.0-eps)-slack)
                    distances[i] = best
        else:
            label = _nearest(x+i*d,centers,k,d,&best,&second)
            evaluated += k
            upper[i] = sqrt(best)*(1.0+eps)+slack
            lower[i] = fmax(0.0,sqrt(second)*(1.0-eps)-slack)
            distances[i] = best
        if old != label:
            changed += 1
        labels[i] = label
        w = weights[i]
        if w > 0:
            if incremental:
                if old != label:
                    masses[old] -= w
                    masses[label] += w
                    counts[old] -= 1
                    counts[label] += 1
                    _add(sums+old*d,x+i*d,d,-w)
                    _add(sums+label*d,x+i*d,d,w)
            else:
                masses[label] += w
                counts[label] += 1
                _add(sums+label*d,x+i*d,d,w)
    changes[0] = changed
    evaluations[0] = evaluated


cdef void _accumulate_block(const floating* x, const double* weights,
                            const int32_t* labels, Py_ssize_t first,
                            Py_ssize_t last, Py_ssize_t d,
                            double* sums, double* masses,
                            int64_t* counts) noexcept nogil:
    cdef Py_ssize_t i, j
    cdef int label
    cdef double w
    for i in range(first,last):
        w = weights[i]
        if w > 0:
            label = labels[i]
            masses[label] += w
            counts[label] += 1
            _add(sums+label*d,x+i*d,d,w)


cdef bint _reduce_totals(const double* sums, const double* masses,
                         const int64_t* counts, Py_ssize_t nt, Py_ssize_t k,
                         Py_ssize_t d, double* total_sums,
                         double* total_masses, int64_t* total_counts) noexcept nogil:
    cdef Py_ssize_t t,i,j
    cdef bint invalid = False
    for i in range(k):
        total_masses[i] = 0
        total_counts[i] = 0
        for j in range(d):
            total_sums[i*d+j] = 0
        for t in range(nt):
            total_masses[i] += masses[t*k+i]
            total_counts[i] += counts[t*k+i]
            for j in range(d):
                total_sums[i*d+j] += sums[(t*k+i)*d+j]
        if total_counts[i] == 0:
            total_masses[i] = 0
            for j in range(d):
                total_sums[i*d+j] = 0
        elif total_masses[i] <= 0:
            invalid = True
    return invalid


cdef double _separation_one(const floating* centers, Py_ssize_t c,
                            Py_ssize_t k, Py_ssize_t d,
                            double eps) noexcept nogil:
    cdef Py_ssize_t j
    cdef double value, best = INFINITY
    for j in range(k):
        if c != j:
            value = _sqdist(centers+c*d, centers+j*d, d)
            if value < best:
                best = value
    return fmax(0.0,0.5*sqrt(best)*(1.0-eps)-8*sqrt(d*DBL_MIN))


cdef double _assigned_inertia(const floating* x, const floating* centers,
                              const double* weights, const int32_t* labels,
                              Py_ssize_t n, Py_ssize_t d) noexcept nogil:
    cdef Py_ssize_t i
    cdef double result = 0
    for i in range(n):
        if weights[i] > 0:
            result += weights[i] * _sqdist(x+i*d, centers+labels[i]*d, d)
    return result


cdef int _relocate(const floating* x, const floating* centers,
                   const double* weights, int32_t* labels,
                   double* sums, double* masses, int64_t* counts,
                   Py_ssize_t n, Py_ssize_t k, Py_ssize_t d) noexcept nogil:
    cdef Py_ssize_t e, i, j, chosen
    cdef int donor, moved = 0
    cdef double score, largest_score, w
    for e in range(k):
        if masses[e] > 0:
            continue
        chosen = -1
        largest_score = 0.0
        for i in range(n):
            donor = labels[i]
            w = weights[i]
            if w > 0 and counts[donor] > 1 and masses[donor] > w:
                score = w * _sqdist(x+i*d, centers+donor*d, d)
                if score > largest_score:
                    largest_score = score
                    chosen = i
        if chosen < 0:
            continue
        donor = labels[chosen]
        w = weights[chosen]
        for j in range(d):
            sums[donor*d+j] -= w * <double>x[chosen*d+j]
            sums[e*d+j] = w * <double>x[chosen*d+j]
        masses[donor] -= w
        masses[e] = w
        counts[donor] -= 1
        counts[e] = 1
        labels[chosen] = <int32_t>e
        moved += 1
    return moved


def assign(const floating[:, ::1] X, const floating[:, ::1] centers,
           int n_threads=1):
    """Return nearest-center labels and float64 squared distances."""
    cdef Py_ssize_t n = X.shape[0], d = X.shape[1], k = centers.shape[0]
    if d < 1 or k < 1 or centers.shape[1] != d:
        raise ValueError("centers and X must have matching nonempty features")
    cdef int nt = _threads(n_threads, n, n*k*d)
    cdef object lab = np.full(n, -1, dtype=np.int32)
    cdef object dis = np.empty(n, dtype=np.float64)
    cdef int32_t[::1] labels = lab
    cdef double[::1] ds = dis
    cdef object stats_np = np.zeros((nt, 2), dtype=np.int64)
    cdef int64_t[:, ::1] stats = stats_np
    cdef bint gemm = d >= 8 and k >= 8
    cdef object xn_np = np.empty(n if gemm else 1, dtype=np.float64)
    cdef object cn_np = np.empty(k if gemm else 1, dtype=np.float64)
    cdef double[::1] xn = xn_np, cn = cn_np
    cdef object buf_np = np.empty((nt if gemm else 1,256 if gemm else 1,k if gemm else 1),
                                  dtype=np.float32 if floating is float else np.float64)
    cdef floating[:,:,::1] buf = buf_np
    cdef Py_ssize_t i
    cdef int t
    if n == 0:
        return lab, dis
    if gemm:
        for i in prange(n, nogil=True, num_threads=nt, schedule='static'):
            xn[i] = _norm(&X[i,0], d)
        for i in prange(k, nogil=True, num_threads=nt, schedule='static'):
            cn[i] = _norm(&centers[i,0], d)
        for t in prange(nt, nogil=True, num_threads=nt, schedule='static'):
            _gemm_block(&X[0,0], &centers[0,0], &xn[0], &cn[0], NULL,
                        t*n//nt,(t+1)*n//nt,<int>k,<int>d,&labels[0],&ds[0],
                        NULL,NULL,NULL,&stats[t,0],&stats[t,1],&buf[t,0,0],False,NULL,NULL)
    else:
        for t in prange(nt, nogil=True, num_threads=nt, schedule='static'):
            _plain_block(&X[0,0], &centers[0,0], NULL, t*n//nt, (t+1)*n//nt,
                         k, d, &labels[0], &ds[0], NULL, NULL, NULL,
                         &stats[t,0], &stats[t,1], False)
    return lab, dis


def point_distances(const floating[:, ::1] X, const floating[::1] center,
                    int n_threads=1):
    """Squared distances to one center, accumulated and returned in float64."""
    cdef Py_ssize_t n = X.shape[0], d = X.shape[1], i
    if d < 1 or center.shape[0] != d:
        raise ValueError("center must have n_features elements")
    cdef int nt = _threads(n_threads, n, n*d)
    cdef object out_np = np.empty(n, dtype=np.float64)
    cdef double[::1] out = out_np
    for i in prange(n, nogil=True, num_threads=nt, schedule='static'):
        out[i] = _sqdist(&X[i,0], &center[0], d)
    return out_np


def point_distances_clipped(const floating[:,::1] X,
                            const floating[::1] candidate,
                            const double[::1] closest,
                            const int32_t[::1] nearest_labels,
                            const double[::1] center_distances_squared,
                            int n_threads=1):
    """Return min(closest, squared distance to candidate), with exact pruning.

    ``closest[i]`` is the squared distance to the existing center indexed by
    ``nearest_labels[i]``. ``center_distances_squared[j]`` is the squared
    distance from existing center j to the candidate. The triangle inequality
    rules out an improvement when the latter distance exceeds twice the former
    (four times for squared distances). Strict comparisons with outward error
    margins keep ambiguous boundaries on the direct-distance path.

    No sampling decisions happen here: the caller retains its D-squared or
    greedy local-trial rule. Missing label indices conservatively disable
    pruning for their observation. All input buffers may be read-only.
    """
    cdef Py_ssize_t n = X.shape[0], d = X.shape[1], m = center_distances_squared.shape[0], i
    cdef int label, nt
    cdef double value, current, bound
    cdef double eps = 64*DBL_EPSILON*(d+1)
    cdef double floor = 64*d*DBL_MIN
    if d < 1 or candidate.shape[0] != d:
        raise ValueError("candidate must have n_features elements")
    if closest.shape[0] != n or nearest_labels.shape[0] != n:
        raise ValueError("closest and nearest_labels must have n_samples elements")
    nt = _threads(n_threads,n,n*d)
    cdef object out_np = np.empty(n,dtype=np.float64)
    cdef double[::1] out = out_np
    for i in prange(n,nogil=True,num_threads=nt,schedule='static'):
        current = closest[i]
        label = nearest_labels[i]
        if 0 <= label < m:
            bound = center_distances_squared[label]*(1.0-eps)-floor
            # Overflow on the right disables pruning, conservatively.
            if bound > 4.0*(current*(1.0+eps)+floor):
                out[i] = current
                continue
        value = _sqdist(&X[i,0],&candidate[0],d)
        out[i] = value if value < current else current
    return out_np


cdef void _distances_row(const floating* x, const floating* centers,
                         floating* result, Py_ssize_t k, Py_ssize_t d,
                         bint squared) noexcept nogil:
    cdef Py_ssize_t j
    cdef double value
    for j in range(k):
        value = _sqdist(x, centers+j*d, d)
        result[j] = <floating>(value if squared else sqrt(value))


def distances(const floating[:, ::1] X, const floating[:, ::1] centers,
              bint squared=False, int n_threads=1):
    """Dense n-by-k Euclidean distance matrix (input precision)."""
    cdef Py_ssize_t n = X.shape[0], d = X.shape[1], k = centers.shape[0], i
    if d < 1 or k < 1 or centers.shape[1] != d:
        raise ValueError("centers and X must have matching nonempty features")
    cdef int nt = _threads(n_threads, n, n*k*d)
    cdef object out_np = np.empty((n,k), dtype=np.float32 if floating is float else np.float64)
    cdef floating[:, ::1] out = out_np
    for i in prange(n, nogil=True, num_threads=nt, schedule='static'):
        _distances_row(&X[i,0], &centers[0,0], &out[i,0], k, d, squared)
    return out_np


def squared_distances(const floating[:, ::1] X, const floating[:, ::1] centers,
                      int n_threads=1):
    """Dense n-by-k squared Euclidean distances, always in float64."""
    cdef Py_ssize_t n = X.shape[0], d = X.shape[1], k = centers.shape[0], i, j
    if d < 1 or k < 1 or centers.shape[1] != d:
        raise ValueError("centers and X must have matching nonempty features")
    cdef int nt = _threads(n_threads, n, n*k*d)
    cdef object out_np = np.empty((n,k),dtype=np.float64)
    cdef double[:,::1] out = out_np
    for i in prange(n, nogil=True, num_threads=nt, schedule='static'):
        for j in range(k):
            out[i,j] = _sqdist(&X[i,0],&centers[j,0],d)
    return out_np


def run(const floating[:, ::1] X, const double[::1] weights,
        const floating[:, ::1] centers, int max_iter=300, double tol=1e-4,
        int n_threads=1, algorithm='lloyd', bint record_history=False,
        final_X=None, center_offset=None):
    """Run one initialized Lloyd/Hamerly/Elkan/Exponion fit.

    ``tol`` is the absolute squared Frobenius shift threshold. History's
    inertia is the objective after each update with that iteration's labels.
    The return value always includes a final nearest-center assignment.
    When ``center_offset`` is supplied, that final assignment uses rounded
    public centers. ``final_X`` supplies unchanged original observations; when
    omitted, observations are reconstructed exactly as an in-place restore.
    """
    cdef Py_ssize_t n = X.shape[0], d = X.shape[1], k = centers.shape[0]
    if n < 1 or d < 1 or k < 1 or centers.shape[1] != d:
        raise ValueError("nonempty compatible X and centers are required")
    if weights.shape[0] != n:
        raise ValueError("weights must have n_samples elements")
    if max_iter < 1 or tol < 0:
        raise ValueError("max_iter must be positive and tol nonnegative")
    if algorithm not in ('lloyd', 'hamerly', 'elkan', 'exponion'):
        raise ValueError("native algorithm must be 'lloyd', 'hamerly', 'elkan' or 'exponion'")
    cdef bint use_bounds = algorithm != 'lloyd', initialized = False
    cdef bint use_elkan = algorithm == 'elkan'
    cdef bint use_exponion = algorithm == 'exponion'
    cdef bint use_center_table = use_elkan or use_exponion
    cdef bint gemm = d >= 8 and k >= 8
    cdef int nt = _threads(n_threads, n, n*k*d)
    cdef object centers_np = np.array(centers, order='C', copy=True)
    cdef floating[:, ::1] c = centers_np
    cdef object labels_np = np.full(n, -1, dtype=np.int32)
    cdef int32_t[::1] labels = labels_np
    cdef object dist_np = np.empty(n, dtype=np.float64)
    cdef double[::1] dist = dist_np
    cdef object upper_np = np.empty(n if use_bounds else 1, dtype=np.float64)
    cdef object lower_np = np.empty(n if use_bounds else 1, dtype=np.float64)
    cdef double[::1] upper = upper_np, lower = lower_np
    cdef object full_lower_np = np.empty((n if use_elkan else 1, k if use_elkan else 1), dtype=np.float64)
    cdef double[:,::1] full_lower = full_lower_np
    cdef object center_lower_np = np.empty((k if use_center_table else 1, k if use_center_table else 1), dtype=np.float64)
    cdef double[:,::1] center_lower = center_lower_np
    cdef object nearest_upper_np = np.empty(k if use_center_table else 1, dtype=np.float64)
    cdef double[::1] nearest_upper = nearest_upper_np
    cdef object neighbor_order_np = np.empty((1,1),dtype=np.intp)
    cdef cnp.intp_t[:,::1] neighbor_order = neighbor_order_np
    cdef object separation_np = np.empty(k, dtype=np.float64)
    cdef double[::1] separation = separation_np
    cdef object shifts_np = np.zeros(k, dtype=np.float64)
    cdef double[::1] shifts = shifts_np
    cdef object sums_np = np.zeros((nt,k,d), dtype=np.float64)
    cdef double[:, :, ::1] sums = sums_np
    cdef object masses_np = np.zeros((nt,k), dtype=np.float64)
    cdef double[:, ::1] masses = masses_np
    cdef object counts_np = np.zeros((nt,k), dtype=np.int64)
    cdef int64_t[:, ::1] counts = counts_np
    cdef object total_sums_np = np.zeros((k,d), dtype=np.float64)
    cdef object total_mass_np = np.zeros(k, dtype=np.float64)
    cdef object total_counts_np = np.zeros(k, dtype=np.int64)
    cdef double[:,::1] total_sums = total_sums_np
    cdef double[::1] total_mass = total_mass_np
    cdef int64_t[::1] total_counts = total_counts_np
    cdef object stats_np = np.zeros((nt,2), dtype=np.int64)
    cdef int64_t[:, ::1] stats = stats_np
    cdef object xn_np = np.empty(n if gemm else 1, dtype=np.float64)
    cdef object cn_np = np.empty(k if gemm else 1, dtype=np.float64)
    cdef double[::1] xn = xn_np, cn = cn_np
    cdef object buf_np = np.empty((nt if gemm else 1,256 if gemm else 1,k if gemm else 1),
                                  dtype=np.float32 if floating is float else np.float64)
    cdef floating[:,:,::1] buf = buf_np
    cdef double eps = 32 * DBL_EPSILON * (d+1)
    cdef double slack = 8*sqrt(d*DBL_MIN)
    cdef double total_shift, shift, value, delta, max1, max2, inertia
    cdef Py_ssize_t i, j, t, it, largest, moving_most
    cdef int relocated
    cdef int64_t changed, evaluated = 0, iteration_evaluated
    cdef bint converged = False, incremental, invalid_mass, centers_changed
    cdef object reason = 'max_iter'
    cdef list history = []
    cdef const floating[:,::1] final_x
    cdef const floating[::1] offset_view
    cdef floating[:,::1] final_centers, restore_buffer
    cdef object final_centers_np = None, restore_buffer_np = None
    cdef bint public_final = center_offset is not None or final_X is not None
    if final_X is not None:
        final_x = final_X
        if final_x.shape[0] != n or final_x.shape[1] != d:
            raise ValueError("final_X must have the same shape and dtype as X")
    else:
        final_x = X
    if center_offset is not None:
        offset_view = center_offset
        if offset_view.shape[0] != d:
            raise ValueError("center_offset must have n_features elements")
    if gemm:
        for i in prange(n, nogil=True, num_threads=nt, schedule='static'):
            xn[i] = _norm(&X[i,0], d)
    for it in range(max_iter):
        incremental = use_bounds and initialized and it % 32 != 0
        if not incremental:
            sums_np.fill(0)
            masses_np.fill(0)
            counts_np.fill(0)
        if use_center_table:
            for i in prange(k,nogil=True,num_threads=nt,schedule='static'):
                _center_geometry_row(&c[0,0],i,k,d,eps,&center_lower[0,0],
                                      &separation[0],&nearest_upper[0])
            if use_exponion and initialized:
                neighbor_order_np = np.asarray(np.argsort(center_lower_np,axis=1),
                                                dtype=np.intp,order='C')
                neighbor_order = neighbor_order_np
        if use_elkan:
            for t in prange(nt,nogil=True,num_threads=nt,schedule='static'):
                _elkan_block(&X[0,0],&c[0,0],&weights[0],t*n//nt,(t+1)*n//nt,
                              k,d,&labels[0],&dist[0],&upper[0],&full_lower[0,0],
                              &center_lower[0,0],&separation[0],eps,initialized,
                              incremental,&sums[t,0,0],&masses[t,0],&counts[t,0],
                              &stats[t,0],&stats[t,1])
        elif use_bounds and gemm and not initialized:
            for i in prange(k, nogil=True, num_threads=nt, schedule='static'):
                cn[i] = _norm(&c[i,0], d)
            for t in prange(nt, nogil=True, num_threads=nt, schedule='static'):
                _gemm_block(&X[0,0],&c[0,0],&xn[0],&cn[0],&weights[0],
                            t*n//nt,(t+1)*n//nt,<int>k,<int>d,&labels[0],&dist[0],
                            &sums[t,0,0],&masses[t,0],&counts[t,0],
                            &stats[t,0],&stats[t,1],&buf[t,0,0],True,&upper[0],&lower[0])
        elif use_exponion and initialized:
            for t in prange(nt,nogil=True,num_threads=nt,schedule='static'):
                _exponion_block(&X[0,0],&c[0,0],&weights[0],t*n//nt,(t+1)*n//nt,
                                 k,d,&labels[0],&dist[0],&upper[0],&lower[0],
                                 &separation[0],&center_lower[0,0],&nearest_upper[0],
                                 &neighbor_order[0,0],eps,initialized,incremental,
                                 &sums[t,0,0],&masses[t,0],&counts[t,0],
                                 &stats[t,0],&stats[t,1])
        elif use_bounds:
            for i in prange(k, nogil=True, num_threads=nt, schedule='static'):
                separation[i] = _separation_one(&c[0,0], i, k, d, eps)
            for t in prange(nt, nogil=True, num_threads=nt, schedule='static'):
                _bound_block(&X[0,0], &c[0,0], &weights[0], t*n//nt, (t+1)*n//nt,
                             k, d, &labels[0], &dist[0], &upper[0], &lower[0],
                             &separation[0], eps, initialized, incremental, &sums[t,0,0],
                             &masses[t,0], &counts[t,0], &stats[t,0], &stats[t,1])
        elif gemm:
            for i in prange(k, nogil=True, num_threads=nt, schedule='static'):
                cn[i] = _norm(&c[i,0], d)
            for t in prange(nt, nogil=True, num_threads=nt, schedule='static'):
                _gemm_block(&X[0,0],&c[0,0],&xn[0],&cn[0],&weights[0],
                            t*n//nt,(t+1)*n//nt,<int>k,<int>d,&labels[0],&dist[0],
                            &sums[t,0,0],&masses[t,0],&counts[t,0],
                            &stats[t,0],&stats[t,1],&buf[t,0,0],True,NULL,NULL)
        else:
            for t in prange(nt, nogil=True, num_threads=nt, schedule='static'):
                _plain_block(&X[0,0], &c[0,0], &weights[0], t*n//nt, (t+1)*n//nt,
                             k, d, &labels[0], &dist[0], &sums[t,0,0],
                             &masses[t,0], &counts[t,0], &stats[t,0], &stats[t,1], True)
        changed = 0
        iteration_evaluated = 0
        with nogil:
            for t in range(nt):
                changed += stats[t,0]
                iteration_evaluated += stats[t,1]
            invalid_mass = _reduce_totals(&sums[0,0,0],&masses[0,0],&counts[0,0],
                                           nt,k,d,&total_sums[0,0],&total_mass[0],
                                           &total_counts[0])
        # Refresh after strict convergence or dangerous cancellation. Periodic
        # refresh above also prevents long incremental chains accumulating drift.
        if incremental and (changed == 0 or invalid_mass):
            sums_np.fill(0)
            masses_np.fill(0)
            counts_np.fill(0)
            for t in prange(nt, nogil=True, num_threads=nt, schedule='static'):
                _accumulate_block(&X[0,0],&weights[0],&labels[0],t*n//nt,
                                   (t+1)*n//nt,d,&sums[t,0,0],&masses[t,0],&counts[t,0])
            with nogil:
                _reduce_totals(&sums[0,0,0],&masses[0,0],&counts[0,0],nt,k,d,
                                &total_sums[0,0],&total_mass[0],&total_counts[0])
        with nogil:
            relocated = _relocate(&X[0,0], &c[0,0], &weights[0], &labels[0],
                                  &total_sums[0,0], &total_mass[0], &total_counts[0], n,k,d)
            largest = 0
            for i in range(1,k):
                if total_mass[i] > total_mass[largest]:
                    largest = i
            total_shift = 0.0
            centers_changed = False
            max1 = 0.0
            max2 = 0.0
            moving_most = 0
            for i in range(k):
                shift = 0.0
                for j in range(d):
                    if total_mass[i] > 0:
                        value = total_sums[i,j] / total_mass[i]
                    else:
                        value = total_sums[largest,j] / total_mass[largest]
                    # Measure the movement of the actually stored float center.
                    delta = <double>(<floating>value) - <double>c[i,j]
                    if <floating>value != c[i,j]:
                        centers_changed = True
                    shift += delta*delta
                    c[i,j] = <floating>value
                total_shift += shift
                shifts[i] = sqrt(shift) * (1.0+eps) + slack
                if shifts[i] > max1:
                    max2 = max1
                    max1 = shifts[i]
                    moving_most = i
                elif shifts[i] > max2:
                    max2 = shifts[i]
        evaluated += iteration_evaluated
        if record_history:
            with nogil:
                inertia = _assigned_inertia(&X[0,0], &c[0,0], &weights[0],
                                            &labels[0], n,d)
            history.append({'iteration': it+1, 'objective_fixed_assignment': inertia,
                            'center_shift': sqrt(total_shift),
                            'changed_labels': int(changed + relocated),
                            'distance_evaluations': int(iteration_evaluated)})
        if changed == 0 and relocated == 0:
            converged = True
            reason = 'labels'
            break
        if total_shift <= tol:
            converged = True
            reason = 'tolerance'
            break
        if use_bounds and relocated == 0:
            if use_elkan:
                for i in prange(n,nogil=True,num_threads=nt,schedule='static'):
                    upper[i] = (upper[i]+shifts[labels[i]])*(1.0+eps)+slack
                    for j in range(k):
                        full_lower[i,j] = fmax(0.0,(full_lower[i,j]-shifts[j])*(1.0-eps)-slack)
            else:
                for i in prange(n, nogil=True, num_threads=nt, schedule='static'):
                    upper[i] = (upper[i] + shifts[labels[i]]) * (1.0+eps)
                    if labels[i] == moving_most:
                        lower[i] = fmax(0.0, (lower[i] - max2) * (1.0-eps))
                    else:
                        lower[i] = fmax(0.0, (lower[i] - max1) * (1.0-eps))
            initialized = True
        else:
            initialized = False

    # Public-coordinate finalization is unconditional. Centering and restoring
    # centers can change even tiny boundary ties in floating-point arithmetic.
    # Each restart is ranked by this actual public objective.
    if public_final:
        if center_offset is not None:
            final_centers_np = np.ascontiguousarray(centers_np + np.asarray(center_offset))
        else:
            final_centers_np = centers_np
        final_centers = final_centers_np
        if final_X is None and center_offset is not None:
            restore_buffer_np = np.empty((nt,d),dtype=centers_np.dtype)
            restore_buffer = restore_buffer_np
            for t in prange(nt,nogil=True,num_threads=nt,schedule='static'):
                _restored_block(&X[0,0],&offset_view[0],&final_centers[0,0],
                                 t*n//nt,(t+1)*n//nt,k,d,&labels[0],&dist[0],
                                 &restore_buffer[t,0])
            evaluated += n*k
        elif gemm:
            for i in prange(n,nogil=True,num_threads=nt,schedule='static'):
                xn[i] = _norm(&final_x[i,0],d)
            for i in prange(k,nogil=True,num_threads=nt,schedule='static'):
                cn[i] = _norm(&final_centers[i,0],d)
            for t in prange(nt,nogil=True,num_threads=nt,schedule='static'):
                _gemm_block(&final_x[0,0],&final_centers[0,0],&xn[0],&cn[0],NULL,
                            t*n//nt,(t+1)*n//nt,<int>k,<int>d,&labels[0],&dist[0],
                            NULL,NULL,NULL,&stats[t,0],&stats[t,1],&buf[t,0,0],False,NULL,NULL)
            for t in range(nt):
                evaluated += stats[t,1]
        else:
            for t in prange(nt,nogil=True,num_threads=nt,schedule='static'):
                _plain_block(&final_x[0,0],&final_centers[0,0],NULL,
                             t*n//nt,(t+1)*n//nt,k,d,&labels[0],&dist[0],
                             NULL,NULL,NULL,&stats[t,0],&stats[t,1],False)
            evaluated += n*k
    # Reassignment is essential when the last update moved any center. If no
    # center moved and no point was relocated, the existing assignment already
    # is nearest, so avoid another full n*k pass. Hamerly's cached distances may
    # be stale even though its labels are valid; tighten those n distances.
    elif centers_changed or relocated > 0:
        if gemm:
            for i in prange(k, nogil=True, num_threads=nt, schedule='static'):
                cn[i] = _norm(&c[i,0], d)
            for t in prange(nt, nogil=True, num_threads=nt, schedule='static'):
                _gemm_block(&X[0,0],&c[0,0],&xn[0],&cn[0],NULL,
                            t*n//nt,(t+1)*n//nt,<int>k,<int>d,&labels[0],&dist[0],
                            NULL,NULL,NULL,&stats[t,0],&stats[t,1],&buf[t,0,0],False,NULL,NULL)
        else:
            for t in prange(nt, nogil=True, num_threads=nt, schedule='static'):
                _plain_block(&X[0,0], &c[0,0], NULL, t*n//nt, (t+1)*n//nt, k,d,
                             &labels[0], &dist[0], NULL,NULL,NULL,
                             &stats[t,0], &stats[t,1], False)
        for t in range(nt):
            evaluated += stats[t,1]
    elif use_bounds:
        for i in prange(n, nogil=True, num_threads=nt, schedule='static'):
            dist[i] = _sqdist(&X[i,0], &c[labels[i],0], d)
        evaluated += n
    masses_np.fill(0)
    inertia = 0.0
    with nogil:
        for i in range(n):
            masses[0,labels[i]] += weights[i]
            if weights[i] > 0:
                inertia += weights[i] * dist[i]
    return {'centers': centers_np, 'labels': labels_np,
            'squared_distances': dist_np, 'inertia': inertia,
            'n_iter': int(it+1), 'history': history,
            'distance_evaluations': int(evaluated), 'converged': bool(converged),
            'convergence_reason': reason, 'cluster_weight': masses_np[0].copy(),
            'n_threads': nt,
            'assignment_engine': algorithm if use_bounds else ('gemm' if gemm else 'direct')}
