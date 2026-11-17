# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False
# cython: nonecheck=False, cdivision=True
"""Independent weighted CSR Lloyd and Elkan kernels.

CSR indices must be sorted, unique within rows, and validated by the facade.
The n-by-d sample matrix is never materialized. Elkan follows the geometric
bounds in C. Elkan, ICML 2003; no external clustering source is used.
"""

import numpy as np
cimport numpy as cnp
from cython.parallel cimport prange
from libc.math cimport sqrt, INFINITY, fmax, isfinite
from libc.float cimport DBL_EPSILON, DBL_MIN
from libc.stdint cimport int32_t, int64_t

cnp.import_array()

ctypedef fused floating:
    float
    double

cdef extern from "_distance.h" nogil:
    double myk_sqdist_f32(const float*, const float*, Py_ssize_t) noexcept
    double myk_sqdist_f64(const double*, const double*, Py_ssize_t) noexcept
    int myk_openmp_enabled() noexcept


cdef int _threads(int requested, Py_ssize_t n, Py_ssize_t work):
    if requested < 1:
        raise ValueError("n_threads must be positive")
    if not myk_openmp_enabled() or work < 32768:
        return 1
    return min(requested, max(1, <int>((n + 1023) // 1024)))


cdef inline double _center_distance(const floating* a, const floating* b,
                                    Py_ssize_t d) noexcept nogil:
    if floating is float:
        return myk_sqdist_f32(a, b, d)
    else:
        return myk_sqdist_f64(a, b, d)


cdef inline double _center_norm(const floating* center, Py_ssize_t d) noexcept nogil:
    cdef Py_ssize_t j
    cdef double value, total = 0
    for j in range(d):
        value = center[j]
        total += value * value
    return total


cdef double _direct(const floating* data, const int64_t* indices,
                    Py_ssize_t first, Py_ssize_t last,
                    const floating* center, Py_ssize_t d) noexcept nogil:
    """Stable direct coordinate merge, used only for ambiguous arithmetic."""
    cdef Py_ssize_t j, p = first
    cdef double delta, total = 0.0
    for j in range(d):
        if p < last and indices[p] == j:
            delta = <double>data[p] - <double>center[j]
            p += 1
        else:
            delta = -<double>center[j]
        total += delta * delta
    return total


cdef inline double _value(const floating* data, const int64_t* indices,
                          Py_ssize_t first, Py_ssize_t last,
                          const floating* center, Py_ssize_t d,
                          double center_norm, double* error) noexcept nogil:
    """Sparse residual norm plus the missing-coordinate center energy.

    The error screen avoids catastrophic cancellation without forcing a dense
    coordinate scan for ordinary sparse observations.
    """
    cdef Py_ssize_t p, col
    cdef double delta, value, active = 0.0, residual = 0.0
    cdef double gamma = 32.0 * DBL_EPSILON * (d + last-first + 2)
    for p in range(first, last):
        col = indices[p]
        value = <double>center[col]
        active += value * value
        delta = <double>data[p] - value
        residual += delta * delta
    value = (center_norm - active) + residual
    error[0] = gamma * (center_norm + active + residual) + 16.0 * d * DBL_MIN
    if not isfinite(value) or not isfinite(error[0]) or value <= 16.0 * error[0]:
        value = _direct(data, indices, first, last, center, d)
        error[0] = 8.0 * DBL_EPSILON * (d+1) * value + 16.0 * d * DBL_MIN
    return value


cdef int _nearest(const floating* data, const int64_t* indices,
                  Py_ssize_t first, Py_ssize_t last,
                  const floating* centers, const double* norms,
                  Py_ssize_t k, Py_ssize_t d,
                  double* values, double* errors, double* best,
                  int64_t* evaluations) noexcept nogil:
    cdef Py_ssize_t j
    cdef int label = 0
    cdef double value, current = INFINITY, guard
    cdef bint ambiguous = False
    for j in range(k):
        value = _value(data, indices, first, last, centers+j*d, d, norms[j], errors+j)
        values[j] = value
        if value < current:
            current = value
            label = <int>j
    evaluations[0] += k
    guard = current + errors[label]
    for j in range(k):
        if j != label and values[j] - errors[j] <= guard:
            ambiguous = True
            break
    if ambiguous:
        current = _direct(data, indices, first, last, centers+label*d, d)
        evaluations[0] += 1
        values[label] = current
        errors[label] = 8.0 * DBL_EPSILON * (d+1) * current + 16.0*d*DBL_MIN
        for j in range(k):
            if j != label and values[j]-errors[j] <= current + errors[label]:
                value = _direct(data, indices, first, last, centers+j*d, d)
                evaluations[0] += 1
                values[j] = value
                errors[j] = 8.0 * DBL_EPSILON * (d+1) * value + 16.0*d*DBL_MIN
                if value < current or (value == current and j < label):
                    current = value
                    label = <int>j
    best[0] = current
    return label


cdef void _assign_block(const floating* data, const int64_t* indices,
                       const int64_t* indptr, const floating* centers,
                       const double* norms, const double* weights,
                       Py_ssize_t first, Py_ssize_t last, Py_ssize_t k,
                       Py_ssize_t d, int32_t* labels, double* distances,
                       double* values, double* errors, double* sums,
                       double* masses, int64_t* counts,
                       int64_t* changed, int64_t* evaluations,
                       bint accumulate, double* upper, double* lower) noexcept nogil:
    cdef Py_ssize_t i, j, p
    cdef int label
    cdef double best, weight, slack = 8.0 * sqrt(d*DBL_MIN)
    cdef double eps = 32.0 * DBL_EPSILON * (d+1)
    cdef int64_t delta = 0, evals = 0
    for i in range(first, last):
        label = _nearest(data, indices, indptr[i], indptr[i+1], centers,
                         norms, k,d, values, errors, &best, &evals)
        if labels[i] != label:
            delta += 1
        labels[i] = label
        distances[i] = best
        if lower != NULL:
            for j in range(k):
                lower[i*k+j] = fmax(0.0, sqrt(fmax(0.0, values[j]-errors[j])) * (1.0-eps)-slack)
            upper[i] = sqrt(fmax(0.0, best+errors[label])) * (1.0+eps)+slack
        if accumulate:
            weight = weights[i]
            if weight > 0:
                masses[label] += weight
                counts[label] += 1
                for p in range(indptr[i], indptr[i+1]):
                    sums[label*d+indices[p]] += weight * <double>data[p]
    changed[0] = delta
    evaluations[0] = evals


cdef void _elkan_block(const floating* data, const int64_t* indices,
                      const int64_t* indptr, const floating* centers,
                      const double* norms, const double* weights,
                      Py_ssize_t first, Py_ssize_t last, Py_ssize_t k,
                      Py_ssize_t d, int32_t* labels, double* distances,
                      double* upper, double* lower, const double* half,
                      const double* separation, double* sums, double* masses,
                      int64_t* counts, int64_t* changed,
                      int64_t* evaluations) noexcept nogil:
    cdef Py_ssize_t i,j,p
    cdef int old, label
    cdef double value, err, best, best_err, u, weight
    cdef double eps = 32.0*DBL_EPSILON*(d+1), slack = 8.0*sqrt(d*DBL_MIN)
    cdef bint tightened, better
    cdef int64_t delta=0, evals=0
    for i in range(first,last):
        old = labels[i]
        label = old
        u = upper[i]
        best = distances[i]
        best_err = 0.0
        tightened = False
        if not u < separation[label]:
            for j in range(k):
                if j == label or u < lower[i*k+j] or u < half[label*k+j]:
                    continue
                if not tightened:
                    best = _value(data,indices,indptr[i],indptr[i+1],
                                  centers+label*d,d,norms[label],&best_err)
                    evals += 1
                    u = sqrt(fmax(0.0,best+best_err))*(1.0+eps)+slack
                    lower[i*k+label] = fmax(0.0,sqrt(fmax(0.0,best-best_err))*(1.0-eps)-slack)
                    tightened = True
                    if u < lower[i*k+j] or u < half[label*k+j]:
                        continue
                value = _value(data,indices,indptr[i],indptr[i+1],
                               centers+j*d,d,norms[j],&err)
                evals += 1
                if value-err <= best+best_err and best-best_err <= value+err:
                    best = _direct(data,indices,indptr[i],indptr[i+1],centers+label*d,d)
                    value = _direct(data,indices,indptr[i],indptr[i+1],centers+j*d,d)
                    evals += 2
                    best_err = 8.0*DBL_EPSILON*(d+1)*best+16.0*d*DBL_MIN
                    err = 8.0*DBL_EPSILON*(d+1)*value+16.0*d*DBL_MIN
                lower[i*k+j] = fmax(0.0,sqrt(fmax(0.0,value-err))*(1.0-eps)-slack)
                better = value < best or (value == best and j < label)
                if better:
                    label = <int>j
                    best = value
                    best_err = err
                u = sqrt(fmax(0.0,best+best_err))*(1.0+eps)+slack
            upper[i] = u
            distances[i] = best
        if label != old:
            delta += 1
        labels[i] = label
        weight = weights[i]
        if weight > 0:
            masses[label] += weight
            counts[label] += 1
            for p in range(indptr[i],indptr[i+1]):
                sums[label*d+indices[p]] += weight*<double>data[p]
    changed[0] = delta
    evaluations[0] = evals


cdef void _half_row(const floating* centers, Py_ssize_t row, Py_ssize_t k,
                   Py_ssize_t d, double* half, double* separation) noexcept nogil:
    cdef Py_ssize_t j
    cdef double value, minimum=INFINITY
    cdef double eps=32.0*DBL_EPSILON*(d+1), slack=8.0*sqrt(d*DBL_MIN)
    for j in range(k):
        if j == row:
            half[j] = 0.0
        else:
            value = fmax(0.0,0.5*sqrt(_center_distance(centers+row*d,centers+j*d,d))*(1.0-eps)-slack)
            half[j] = value
            if value < minimum:
                minimum = value
    separation[0] = minimum


cdef int _relocate(const floating* data, const int64_t* indices,
                   const int64_t* indptr, const floating* centers,
                   const double* norms, const double* weights,
                   int32_t* labels, double* sums, double* masses,
                   int64_t* counts, Py_ssize_t n, Py_ssize_t k,
                   Py_ssize_t d) noexcept nogil:
    cdef Py_ssize_t e,i,p,chosen
    cdef int donor,moved=0
    cdef double largest,score,weight,err
    for e in range(k):
        if masses[e] > 0:
            continue
        chosen=-1
        largest=0.0
        for i in range(n):
            donor=labels[i]
            weight=weights[i]
            if weight > 0 and counts[donor] > 1 and masses[donor] > weight:
                score=weight*_value(data,indices,indptr[i],indptr[i+1],
                                    centers+donor*d,d,norms[donor],&err)
                if score > largest:
                    largest=score
                    chosen=i
        if chosen < 0:
            continue
        donor=labels[chosen]
        weight=weights[chosen]
        for p in range(indptr[chosen],indptr[chosen+1]):
            sums[donor*d+indices[p]] -= weight*<double>data[p]
            sums[e*d+indices[p]] = weight*<double>data[p]
        masses[donor] -= weight
        masses[e] = weight
        counts[donor] -= 1
        counts[e] = 1
        labels[chosen] = <int32_t>e
        moved += 1
    return moved


cdef double _assigned_inertia(const floating* data,const int64_t* indices,
                              const int64_t* indptr,const floating* centers,
                              const double* norms,const double* weights,
                              const int32_t* labels,Py_ssize_t n,
                              Py_ssize_t d) noexcept nogil:
    cdef Py_ssize_t i
    cdef double total=0.0,err
    for i in range(n):
        if weights[i] > 0:
            total += weights[i]*_value(data,indices,indptr[i],indptr[i+1],
                                       centers+labels[i]*d,d,norms[labels[i]],&err)
    return total


def assign(const floating[::1] data,const int64_t[::1] indices,
           const int64_t[::1] indptr,Py_ssize_t n_features,
           const floating[:,::1] centers,int n_threads=1):
    """Return nearest labels and float64 squared distances from canonical CSR."""
    cdef Py_ssize_t n=indptr.shape[0]-1,d=n_features,k=centers.shape[0],i
    if n < 0 or d < 1 or k < 1 or centers.shape[1] != d:
        raise ValueError("CSR and centers must have nonempty matching features")
    cdef int nt=_threads(n_threads,n,max(1,data.shape[0])*k),t
    cdef object labels_np=np.full(n,-1,dtype=np.int32)
    cdef object distance_np=np.empty(n,dtype=np.float64)
    cdef object norms_np=np.empty(k,dtype=np.float64)
    cdef object values_np=np.empty((nt,k),dtype=np.float64)
    cdef object errors_np=np.empty((nt,k),dtype=np.float64)
    cdef object stats_np=np.zeros((nt,2),dtype=np.int64)
    cdef int32_t[::1] labels=labels_np
    cdef double[::1] distance=distance_np,norms=norms_np
    cdef double[:,::1] values=values_np,errors=errors_np
    cdef int64_t[:,::1] stats=stats_np
    if n == 0:
        return labels_np,distance_np
    # &data[0] is permitted for a zero-length memoryview only as a never-read
    # pointer; use NumPy's data pointer explicitly for all-zero CSR inputs.
    cdef const floating* dp=<const floating*>cnp.PyArray_DATA(np.asarray(data))
    cdef const int64_t* ip=<const int64_t*>cnp.PyArray_DATA(np.asarray(indices))
    for i in prange(k,nogil=True,num_threads=nt,schedule='static'):
        norms[i]=_center_norm(&centers[i,0],d)
    for t in prange(nt,nogil=True,num_threads=nt,schedule='static'):
        _assign_block(dp,ip,&indptr[0],&centers[0,0],&norms[0],NULL,
                       t*n//nt,(t+1)*n//nt,k,d,&labels[0],&distance[0],
                       &values[t,0],&errors[t,0],NULL,NULL,NULL,
                       &stats[t,0],&stats[t,1],False,NULL,NULL)
    return labels_np,distance_np


cdef double _one_distance(const floating* data,const int64_t* indices,
                          const int64_t* indptr,Py_ssize_t row,
                          const floating* center,Py_ssize_t d,
                          double norm) noexcept nogil:
    cdef double err
    return _value(data,indices,indptr[row],indptr[row+1],center,d,norm,&err)


def point_distances(const floating[::1] data,const int64_t[::1] indices,
                    const int64_t[::1] indptr,Py_ssize_t n_features,
                    const floating[::1] center,int n_threads=1):
    """Float64 squared distances to a single dense center."""
    cdef Py_ssize_t n=indptr.shape[0]-1,d=n_features,i
    if d < 1 or n < 0 or center.shape[0] != d:
        raise ValueError("center must have n_features elements")
    cdef int nt=_threads(n_threads,n,max(1,data.shape[0]))
    cdef object out_np=np.empty(n,dtype=np.float64)
    cdef double[::1] out=out_np
    cdef double norm=_center_norm(&center[0],d)
    cdef const floating* dp=<const floating*>cnp.PyArray_DATA(np.asarray(data))
    cdef const int64_t* ip=<const int64_t*>cnp.PyArray_DATA(np.asarray(indices))
    for i in prange(n,nogil=True,num_threads=nt,schedule='static'):
        out[i]=_one_distance(dp,ip,&indptr[0],i,&center[0],d,norm)
    return out_np


cdef void _distance_row(const floating* data,const int64_t* indices,
                       const int64_t* indptr,Py_ssize_t row,
                       const floating* centers,const double* norms,
                       Py_ssize_t k,Py_ssize_t d,bint squared,
                       double* out) noexcept nogil:
    cdef Py_ssize_t j
    cdef double value,err
    for j in range(k):
        value=_value(data,indices,indptr[row],indptr[row+1],centers+j*d,d,norms[j],&err)
        out[j]=value if squared else sqrt(value)


def distances(const floating[::1] data,const int64_t[::1] indices,
              const int64_t[::1] indptr,Py_ssize_t n_features,
              const floating[:,::1] centers,bint squared=False,int n_threads=1):
    """Float64 distances to all centers; output alone occupies n-by-k memory."""
    cdef Py_ssize_t n=indptr.shape[0]-1,d=n_features,k=centers.shape[0],i
    if n < 0 or d < 1 or k < 1 or centers.shape[1] != d:
        raise ValueError("CSR and centers must have nonempty matching features")
    cdef int nt=_threads(n_threads,n,max(1,data.shape[0])*k)
    cdef object out_np=np.empty((n,k),dtype=np.float64)
    cdef object norms_np=np.empty(k,dtype=np.float64)
    cdef double[:,::1] out=out_np
    cdef double[::1] norms=norms_np
    cdef const floating* dp=<const floating*>cnp.PyArray_DATA(np.asarray(data))
    cdef const int64_t* ip=<const int64_t*>cnp.PyArray_DATA(np.asarray(indices))
    for i in prange(k,nogil=True,num_threads=nt,schedule='static'):
        norms[i]=_center_norm(&centers[i,0],d)
    for i in prange(n,nogil=True,num_threads=nt,schedule='static'):
        _distance_row(dp,ip,&indptr[0],i,&centers[0,0],&norms[0],k,d,squared,&out[i,0])
    return out_np


def moments(const floating[::1] data,const int64_t[::1] indices,
            const int64_t[::1] indptr,Py_ssize_t n_features):
    """Unweighted means and population variances without dense sample storage."""
    cdef Py_ssize_t n=indptr.shape[0]-1,d=n_features,p,j
    if n < 1 or d < 1:
        raise ValueError("CSR must have at least one sample and feature")
    cdef object mean_np=np.zeros(d,dtype=np.float64)
    cdef object var_np=np.zeros(d,dtype=np.float64)
    cdef object count_np=np.zeros(d,dtype=np.int64)
    cdef object anchor_np=np.zeros(d,dtype=np.float64)
    cdef double[::1] mean=mean_np,var=var_np
    cdef double[::1] anchor=anchor_np
    cdef int64_t[::1] count=count_np
    cdef double delta
    # A fully occupied column can have a huge common offset. Center those
    # column sums around their first value; columns with implicit zeros use
    # zero as an anchor. Divide before summing to avoid an unnecessary overflow.
    with nogil:
        for p in range(data.shape[0]):
            j=indices[p]
            if count[j] == 0:
                anchor[j]=<double>data[p]
            count[j] += 1
        for j in range(d):
            if count[j] != n:
                anchor[j]=0.0
        for p in range(data.shape[0]):
            j=indices[p]
            mean[j] += (<double>data[p]-anchor[j])/n
        for j in range(d):
            mean[j] += anchor[j]
        for p in range(data.shape[0]):
            j=indices[p]
            delta=<double>data[p]-mean[j]
            var[j] += delta*delta/n
        for j in range(d):
            var[j] += ((n-count[j])/<double>n)*mean[j]*mean[j]
    return mean_np,var_np


def run(const floating[::1] data,const int64_t[::1] indices,
        const int64_t[::1] indptr,Py_ssize_t n_features,
        const double[::1] weights,const floating[:,::1] centers,
        int max_iter=300,double tol=1e-4,int n_threads=1,
        algorithm='lloyd',bint record_history=False):
    """One independent weighted CSR Lloyd/Elkan run; tol is absolute SSE shift."""
    cdef Py_ssize_t n=indptr.shape[0]-1,d=n_features,k=centers.shape[0]
    if n < 1 or d < 1 or k < 1 or centers.shape[1] != d or weights.shape[0] != n:
        raise ValueError("nonempty compatible CSR, weights and centers are required")
    if max_iter < 1 or tol < 0:
        raise ValueError("max_iter must be positive and tol nonnegative")
    if algorithm not in ('lloyd','elkan'):
        raise ValueError("CSR algorithm must be 'lloyd' or 'elkan'")
    cdef bint use_elkan=algorithm=='elkan',initialized=False,converged=False
    cdef int nt=_threads(n_threads,n,max(1,data.shape[0])*k),relocated
    cdef object centers_np=np.array(centers,order='C',copy=True)
    cdef floating[:,::1] c=centers_np
    cdef object labels_np=np.full(n,-1,dtype=np.int32)
    cdef object distance_np=np.empty(n,dtype=np.float64)
    cdef int32_t[::1] labels=labels_np
    cdef double[::1] distance=distance_np
    cdef object norms_np=np.empty(k,dtype=np.float64)
    cdef double[::1] norms=norms_np
    cdef object sums_np=np.zeros((nt,k,d),dtype=np.float64)
    cdef object masses_np=np.zeros((nt,k),dtype=np.float64)
    cdef object counts_np=np.zeros((nt,k),dtype=np.int64)
    cdef double[:,:,::1] sums=sums_np
    cdef double[:,::1] masses=masses_np
    cdef int64_t[:,::1] counts=counts_np
    cdef object values_np=np.empty((nt,k),dtype=np.float64)
    cdef object errors_np=np.empty((nt,k),dtype=np.float64)
    cdef double[:,::1] values=values_np,errors=errors_np
    cdef object stats_np=np.zeros((nt,2),dtype=np.int64)
    cdef int64_t[:,::1] stats=stats_np
    cdef object upper_np=np.empty(n if use_elkan else 1,dtype=np.float64)
    cdef object lower_np=np.empty((n if use_elkan else 1,k if use_elkan else 1),dtype=np.float64)
    cdef object half_np=np.empty((k if use_elkan else 1,k if use_elkan else 1),dtype=np.float64)
    cdef object separation_np=np.empty(k,dtype=np.float64)
    cdef object shifts_np=np.empty(k,dtype=np.float64)
    cdef double[::1] upper=upper_np,separation=separation_np,shifts=shifts_np
    cdef double[:,::1] lower=lower_np,half=half_np
    cdef const floating* dp=<const floating*>cnp.PyArray_DATA(np.asarray(data))
    cdef const int64_t* ip=<const int64_t*>cnp.PyArray_DATA(np.asarray(indices))
    cdef Py_ssize_t i,j,t,it,largest
    cdef double value,delta,shift,total_shift,inertia,eps=32.0*DBL_EPSILON*(d+1)
    cdef double slack=8.0*sqrt(d*DBL_MIN)
    cdef int64_t changed,evaluations=0,iteration_evaluations
    cdef list history=[]
    cdef object reason='max_iter'
    for i in prange(k,nogil=True,num_threads=nt,schedule='static'):
        norms[i]=_center_norm(&c[i,0],d)
    for it in range(max_iter):
        sums_np.fill(0)
        masses_np.fill(0)
        counts_np.fill(0)
        if use_elkan and initialized:
            for i in prange(k,nogil=True,num_threads=nt,schedule='static'):
                _half_row(&c[0,0],i,k,d,&half[i,0],&separation[i])
            for t in prange(nt,nogil=True,num_threads=nt,schedule='static'):
                _elkan_block(dp,ip,&indptr[0],&c[0,0],&norms[0],&weights[0],
                              t*n//nt,(t+1)*n//nt,k,d,&labels[0],&distance[0],
                              &upper[0],&lower[0,0],&half[0,0],&separation[0],
                              &sums[t,0,0],&masses[t,0],&counts[t,0],&stats[t,0],&stats[t,1])
        elif use_elkan:
            for t in prange(nt,nogil=True,num_threads=nt,schedule='static'):
                _assign_block(dp,ip,&indptr[0],&c[0,0],&norms[0],&weights[0],
                               t*n//nt,(t+1)*n//nt,k,d,&labels[0],&distance[0],
                               &values[t,0],&errors[t,0],&sums[t,0,0],&masses[t,0],
                               &counts[t,0],&stats[t,0],&stats[t,1],True,&upper[0],&lower[0,0])
        else:
            for t in prange(nt,nogil=True,num_threads=nt,schedule='static'):
                _assign_block(dp,ip,&indptr[0],&c[0,0],&norms[0],&weights[0],
                               t*n//nt,(t+1)*n//nt,k,d,&labels[0],&distance[0],
                               &values[t,0],&errors[t,0],&sums[t,0,0],&masses[t,0],
                               &counts[t,0],&stats[t,0],&stats[t,1],True,NULL,NULL)
        changed=0
        iteration_evaluations=0
        with nogil:
            for t in range(nt):
                changed += stats[t,0]
                iteration_evaluations += stats[t,1]
            for t in range(1,nt):
                for i in range(k):
                    masses[0,i] += masses[t,i]
                    counts[0,i] += counts[t,i]
                    for j in range(d):
                        sums[0,i,j] += sums[t,i,j]
            relocated=_relocate(dp,ip,&indptr[0],&c[0,0],&norms[0],&weights[0],
                                 &labels[0],&sums[0,0,0],&masses[0,0],&counts[0,0],n,k,d)
            largest=0
            for i in range(1,k):
                if masses[0,i] > masses[0,largest]:
                    largest=i
            total_shift=0.0
            for i in range(k):
                shift=0.0
                for j in range(d):
                    if masses[0,i] > 0:
                        value=sums[0,i,j]/masses[0,i]
                    else:
                        value=sums[0,largest,j]/masses[0,largest]
                    delta=<double>(<floating>value)-<double>c[i,j]
                    shift += delta*delta
                    c[i,j]=<floating>value
                total_shift += shift
                shifts[i]=sqrt(shift)*(1.0+eps)+slack
        evaluations += iteration_evaluations
        for i in prange(k,nogil=True,num_threads=nt,schedule='static'):
            norms[i]=_center_norm(&c[i,0],d)
        if record_history:
            with nogil:
                inertia=_assigned_inertia(dp,ip,&indptr[0],&c[0,0],&norms[0],
                                           &weights[0],&labels[0],n,d)
            history.append({'iteration':it+1,'objective_fixed_assignment':inertia,
                            'center_shift':sqrt(total_shift),'changed_labels':int(changed+relocated),
                            'distance_evaluations':int(iteration_evaluations)})
        if changed == 0 and relocated == 0:
            converged=True
            reason='labels'
            break
        if total_shift <= tol:
            converged=True
            reason='tolerance'
            break
        if use_elkan and relocated == 0:
            for i in prange(n,nogil=True,num_threads=nt,schedule='static'):
                upper[i]=(upper[i]+shifts[labels[i]])*(1.0+eps)+slack
                for j in range(k):
                    lower[i,j]=fmax(0.0,(lower[i,j]-shifts[j])*(1.0-eps)-slack)
            initialized=True
        else:
            initialized=False
    # Always evaluate the returned centers: early stops may have changed labels.
    for t in prange(nt,nogil=True,num_threads=nt,schedule='static'):
        _assign_block(dp,ip,&indptr[0],&c[0,0],&norms[0],NULL,
                       t*n//nt,(t+1)*n//nt,k,d,&labels[0],&distance[0],
                       &values[t,0],&errors[t,0],NULL,NULL,NULL,
                       &stats[t,0],&stats[t,1],False,NULL,NULL)
    inertia=0.0
    masses_np.fill(0)
    with nogil:
        for t in range(nt):
            evaluations += stats[t,1]
        for i in range(n):
            masses[0,labels[i]] += weights[i]
            if weights[i] > 0:
                inertia += weights[i]*distance[i]
    return {'centers':centers_np,'labels':labels_np,'squared_distances':distance_np,
            'inertia':inertia,'n_iter':int(it+1),'history':history,
            'distance_evaluations':int(evaluations),'converged':bool(converged),
            'convergence_reason':reason,'cluster_weight':masses_np[0].copy(),
            'n_threads':nt,'assignment_engine':'csr-elkan' if use_elkan else 'csr-direct'}
