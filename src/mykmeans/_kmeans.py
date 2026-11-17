"""Validated scikit-learn estimator facade around the Cython engine.

The implementation deliberately separates the Euclidean objective from research
diagnostics. Diagnostics are opt-in and do not add work to each training step.
"""
from __future__ import annotations

from numbers import Integral, Real
import warnings

import numpy as np
from scipy import sparse
from ._compat import (BaseEstimator, ClassNamePrefixFeaturesOutMixin, ClusterMixin,
                      TransformerMixin, ConvergenceWarning, check_random_state,
                      check_array, check_is_fitted, validate_data)
from threadpoolctl import ThreadpoolController

from . import _core

_THREADPOOLS = ThreadpoolController()


def _positive_integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be an integer >= 1; got {value!r}")
    return int(value)


def _weights(sample_weight, n):
    if sample_weight is None:
        return np.ones(n, dtype=np.float64)
    if np.isscalar(sample_weight):
        w = np.full(n, sample_weight, dtype=np.float64)
    else:
        w = np.asarray(sample_weight, dtype=np.float64)
        if w.ndim != 1 or w.shape[0] != n:
            raise ValueError(f"sample_weight must have shape ({n},)")
        w = np.ascontiguousarray(w)
    if not np.all(np.isfinite(w)) or np.any(w < 0):
        raise ValueError("sample_weight must contain finite non-negative values")
    with np.errstate(over="ignore"):
        total = w.sum()
    if not np.isfinite(total):
        raise ValueError("sample_weight must have finite total weight")
    if total <= 0:
        raise ValueError("sample_weight has zero total weight; at least one positive weight is required")
    return w


def _threads(requested, n_samples=0):
    if requested is None or requested == -1:
        available = int(_core.max_threads())
        # Do not launch hundreds of OpenMP workers for a tiny input. Explicit
        # positive n_threads always means exactly that requested upper bound.
        return max(1, min(available, max(1, (n_samples + 2047) // 2048)))
    return _positive_integer(requested, "n_threads") if _core.openmp_enabled() else 1


def _csr_parts(X):
    """Canonical CSR buffers; never allocate a dense observation matrix."""
    X = X.tocsr(copy=False)
    X.check_format(full_check=True)
    if not X.has_canonical_format or not X.has_sorted_indices:
        X = X.copy()
        X.sum_duplicates()
        X.sort_indices()
    return (np.ascontiguousarray(X.data),
            np.ascontiguousarray(X.indices, dtype=np.int64),
            np.ascontiguousarray(X.indptr, dtype=np.int64), X.shape[1])


def _csr_rows(parts, rows):
    data, indices, indptr, d = parts
    result = np.zeros((len(rows), d), dtype=data.dtype)
    for j, i in enumerate(rows):
        start, stop = indptr[i], indptr[i + 1]
        result[j, indices[start:stop]] = data[start:stop]
    return result


def _seed_sparse(parts, n_clusters, weight, rng, n_local_trials, n_threads):
    from . import _sparse
    n, d = len(weight), parts[3]
    trials = 2 + int(np.log(n_clusters)) if n_local_trials is None else n_local_trials
    centers = np.empty((n_clusters, d), dtype=parts[0].dtype)
    indices = np.empty(n_clusters, dtype=np.intp)
    first = int(rng.choice(n, p=weight / weight.sum()))
    centers[0] = _csr_rows(parts, [first])[0]
    indices[0] = first
    closest = _sparse.point_distances(*parts, centers[0], n_threads=n_threads)
    for j in range(1, n_clusters):
        weighted = closest * weight
        potential = float(weighted.sum())
        if not np.isfinite(potential):
            raise ValueError("Squared distances overflowed; rescale X before fitting")
        if potential <= 0:
            chosen = int(rng.choice(n, p=weight / weight.sum()))
            centers[j] = _csr_rows(parts, [chosen])[0]
            indices[j] = chosen
            continue
        cdf = np.cumsum(weighted, dtype=np.float64)
        candidates = np.searchsorted(cdf, rng.uniform(size=trials) * cdf[-1], side="right")
        np.minimum(candidates, n - 1, out=candidates)
        candidate_centers = _csr_rows(parts, candidates)
        best_potential = np.inf
        best_distances = None
        best_index = 0
        for t in range(trials):
            distances = _sparse.point_distances(*parts, candidate_centers[t], n_threads=n_threads)
            np.minimum(closest, distances, out=distances)
            trial_potential = float(np.dot(weight, distances))
            if trial_potential < best_potential:
                best_index, best_potential, best_distances = t, trial_potential, distances
        centers[j] = candidate_centers[best_index]
        indices[j] = candidates[best_index]
        closest = best_distances
    return centers, indices


def _seed_dense(X, n_clusters, weight, rng, n_local_trials, n_threads):
    """Weighted greedy D² seeding; no n-by-k distance matrix is formed.

    The triangle filter only skips candidates that cannot improve a point's
    existing distance. It preserves the greedy trials and sampling rule.
    """
    n, d = X.shape
    trials = 2 + int(np.log(n_clusters)) if n_local_trials is None else n_local_trials
    centers = np.empty((n_clusters, d), dtype=X.dtype)
    indices = np.empty(n_clusters, dtype=np.intp)
    first = int(rng.choice(n, p=weight / weight.sum()))
    centers[0] = X[first]
    indices[0] = first
    closest = _core.point_distances(X, centers[0], n_threads=n_threads)
    use_filter = d >= 8 and n >= 1024 and n_clusters >= 8
    nearest = np.zeros(n, dtype=np.int32) if use_filter else None
    probes = np.linspace(0, n - 1, min(n, 256), dtype=np.intp) if use_filter else None
    filter_fraction = 0.6 if d >= 64 else 0.75
    for j in range(1, n_clusters):
        weighted = closest * weight
        potential = float(weighted.sum())
        if not np.isfinite(potential):
            raise ValueError("Squared distances overflowed; rescale X before fitting")
        if potential <= 0:
            # Identical positive-weight samples need repeated centers.
            candidate = int(rng.choice(n, p=weight / weight.sum()))
            centers[j] = X[candidate]
            indices[j] = candidate
            continue
        cdf = np.cumsum(weighted, dtype=np.float64)
        candidates = np.searchsorted(cdf, rng.uniform(size=trials) * cdf[-1], side="right")
        np.minimum(candidates, n - 1, out=candidates)
        best_potential = np.inf
        best_distances = None
        best_candidate = int(candidates[0])
        for candidate in candidates:
            if use_filter:
                between = _core.point_distances(centers[:j], X[candidate], n_threads=1)
                profitable = np.count_nonzero(
                    0.25 * between[nearest[probes]] > closest[probes]
                ) >= filter_fraction * len(probes)
                if profitable:
                    distance = _core.point_distances_clipped(
                        X, X[candidate], closest, nearest, between, n_threads=n_threads)
                else:
                    distance = _core.point_distances(X, X[candidate], n_threads=n_threads)
                    np.minimum(closest, distance, out=distance)
            else:
                distance = _core.point_distances(X, X[candidate], n_threads=n_threads)
                np.minimum(closest, distance, out=distance)
            trial_potential = float(np.dot(weight, distance))
            if trial_potential < best_potential:
                best_candidate = int(candidate)
                best_potential = trial_potential
                best_distances = distance
        centers[j] = X[best_candidate]
        indices[j] = best_candidate
        if use_filter:
            nearest[best_distances < closest] = j
        closest = best_distances
    return centers, indices


@_THREADPOOLS.wrap(limits=1, user_api="blas")
def kmeans_plusplus(X, n_clusters, *, sample_weight=None, x_squared_norms=None,
                    random_state=None, n_local_trials=None):
    """Select weighted greedy k-means++ seeds with the scikit-learn signature.

    Dense data use direct Cython distances. x_squared_norms is validated for API
    compatibility but is not used in that path, which avoids cancellation in
    the norm-expansion formula. CSR data use the independent sparse Cython kernel.
    """
    n_clusters = _positive_integer(n_clusters, "n_clusters")
    if n_local_trials is not None:
        n_local_trials = _positive_integer(n_local_trials, "n_local_trials")
    X = check_array(X, accept_sparse="csr", dtype=[np.float64, np.float32], order="C")
    if X.shape[0] < n_clusters:
        raise ValueError(f"n_samples={X.shape[0]} should be >= n_clusters={n_clusters}")
    weight = _weights(sample_weight, X.shape[0])
    if x_squared_norms is not None:
        norms = np.asarray(x_squared_norms)
        if norms.shape != (X.shape[0],) or not np.all(np.isfinite(norms)) or np.any(norms < 0):
            raise ValueError("x_squared_norms must be finite, non-negative and have shape (n_samples,)")
    rng = check_random_state(random_state)
    if sparse.issparse(X):
        return _seed_sparse(_csr_parts(X), n_clusters, weight, rng, n_local_trials,
                            _threads(None, X.shape[0]))
    return _seed_dense(X, n_clusters, weight, rng, n_local_trials, _threads(None, len(X)))


class KMeans(ClassNamePrefixFeaturesOutMixin, TransformerMixin, ClusterMixin, BaseEstimator):
    """Cython k-means with a scikit-learn-compatible public interface.

    Standard parameters have the same names/defaults as sklearn.cluster.KMeans.
    Additional parameters are keyword-only:

    n_threads : int or None, default=None
        OpenMP threads. None/-1 honor the runtime maximum with a small-data cap.
        Set an explicit count for reproducible benchmark conditions.
    record_history : bool, default=False
        Save iteration diagnostics for the winning native restart. The trace's
        fixed-assignment objective can differ from final inertia on early stop.

    Native algorithms include Lloyd, Elkan, Hamerly, and an Exponion variant.
    CSR inputs support native Lloyd and Elkan; auto uses CSR Lloyd.
    Neither speed superiority nor bitwise sklearn equivalence is promised.
    """

    def __init__(self, n_clusters=8, *, init="k-means++", n_init="auto",
                 max_iter=300, tol=1e-4, verbose=0, random_state=None,
                 copy_x=True, algorithm="lloyd", n_threads=None,
                 record_history=False):
        self.n_clusters = n_clusters
        self.init = init
        self.n_init = n_init
        self.max_iter = max_iter
        self.tol = tol
        self.verbose = verbose
        self.random_state = random_state
        self.copy_x = copy_x
        self.algorithm = algorithm
        self.n_threads = n_threads
        self.record_history = record_history

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.sparse = True
        if getattr(tags, "transformer_tags", None) is not None:
            tags.transformer_tags.preserves_dtype = ["float64", "float32"]
        return tags

    def _validate_options(self):
        _positive_integer(self.n_clusters, "n_clusters")
        _positive_integer(self.max_iter, "max_iter")
        if self.n_init != "auto":
            _positive_integer(self.n_init, "n_init")
        if isinstance(self.tol, (bool, np.bool_)) or not isinstance(self.tol, Real) or not np.isfinite(self.tol) or self.tol < 0:
            raise ValueError("tol must be finite and non-negative")
        if not isinstance(self.verbose, Integral) or self.verbose < 0:
            raise ValueError("verbose must be a non-negative integer")
        if not isinstance(self.copy_x, (bool, np.bool_)):
            raise ValueError("copy_x must be a boolean")
        if not isinstance(self.record_history, (bool, np.bool_)):
            raise ValueError("record_history must be a boolean")
        if self.algorithm not in ("lloyd", "hamerly", "elkan", "exponion", "auto"):
            raise ValueError("algorithm must be 'lloyd', 'hamerly', 'elkan', 'exponion' or 'auto'")
        if isinstance(self.init, str) and self.init not in ("k-means++", "random"):
            raise ValueError("init must be 'k-means++', 'random', an array, or a callable")
        if self.n_threads is not None and self.n_threads != -1:
            _positive_integer(self.n_threads, "n_threads")
        check_random_state(self.random_state)

    @_THREADPOOLS.wrap(limits=1, user_api="blas")
    def fit(self, X, y=None, sample_weight=None):
        """Minimize the weighted Euclidean within-cluster sum of squares."""
        self._validate_options()
        original_input = X
        X = validate_data(self, X, reset=True, accept_sparse="csr",
                          dtype=[np.float64, np.float32], order="C", copy=False)
        n, d = X.shape
        if n < self.n_clusters:
            raise ValueError(f"n_samples={n} should be >= n_clusters={self.n_clusters}")
        weight = _weights(sample_weight, n)
        self.n_threads_ = _threads(self.n_threads, n)
        self._n_features_out = self.n_clusters
        self.openmp_enabled_ = bool(_core.openmp_enabled())
        resolved = self.algorithm
        if resolved == "auto":
            resolved = "lloyd" if sparse.issparse(X) else ("hamerly" if d <= 64 and self.n_clusters > 1 else "lloyd")
        if sparse.issparse(X):
            return self._fit_sparse(X, weight, resolved)
        # Preserve the exact original coordinates for the final assignment.
        # When copy_x=False really aliases a writable input, the native
        # finalizer reconstructs the same rounded rows the finally block will
        # restore. Detached/converted/read-only inputs retain an original view.
        restore_user_input = (not self.copy_x and isinstance(original_input, np.ndarray)
                              and X.flags.writeable and np.shares_memory(X, original_input))
        if restore_user_input:
            final_X = None
        else:
            final_X = X
            X = X.copy(order="C")
        self.backend_ = "cython"
        self.assignment_engine_ = None
        self.algorithm_ = resolved
        rng = check_random_state(self.random_state)
        array_init = not isinstance(self.init, str) and not callable(self.init)
        if array_init:
            initial_array = check_array(self.init, dtype=X.dtype, order="C", copy=True)
            if initial_array.shape != (self.n_clusters, d):
                raise ValueError(f"init must have shape ({self.n_clusters}, {d})")
        if self.n_init == "auto":
            repeats = 1 if array_init or self.init == "k-means++" else 10
        else:
            repeats = int(self.n_init)
        if array_init and repeats != 1:
            warnings.warn("Explicit initial centers: performing only one init instead of n_init runs",
                          RuntimeWarning, stacklevel=2)
            repeats = 1
        self.n_init_ = repeats
        # The data are centered once per fit. The finally block restores a
        # writable user array even if initialization or a native run fails.
        with np.errstate(over="ignore", invalid="ignore"):
            offset = np.mean(X, axis=0, dtype=np.float64).astype(X.dtype)
            variance = np.var(X, axis=0, dtype=np.float64)
        if not np.all(np.isfinite(offset)) or not np.all(np.isfinite(variance)):
            raise ValueError("Data mean or variance overflowed; rescale X before fitting")
        tolerance = float(np.mean(variance) * self.tol)
        if not np.isfinite(tolerance):
            raise ValueError("Data variance overflowed; rescale X before fitting")
        # Only extreme float32 inputs need a range scan. For ordinary inputs
        # this branch adds no pass through the dataset.
        if X.dtype == np.float32 and (np.max(np.abs(offset)) > 1e30 or np.max(variance) > 1e60):
            lo = np.min(X, axis=0).astype(np.float64) - offset
            hi = np.max(X, axis=0).astype(np.float64) - offset
            if max(np.max(np.abs(lo)), np.max(np.abs(hi))) > np.finfo(np.float32).max:
                raise ValueError("Centering would overflow float32; rescale X or use float64")
        self.tol_ = tolerance
        best = None
        restart_inertias = []
        X -= offset
        try:
            for run_index in range(repeats):
                if array_init:
                    centers = np.ascontiguousarray(initial_array - offset)
                elif callable(self.init):
                    centers = check_array(self.init(X, self.n_clusters, random_state=rng),
                                          dtype=X.dtype, order="C", copy=True)
                    if centers.shape != (self.n_clusters, d):
                        raise ValueError(f"Callable init must return shape ({self.n_clusters}, {d})")
                elif self.init == "random":
                    seeds = rng.choice(n, self.n_clusters, replace=False, p=weight / weight.sum())
                    centers = np.ascontiguousarray(X[seeds])
                else:
                    centers, _ = _seed_dense(X, self.n_clusters, weight, rng, None, self.n_threads_)
                result = _core.run(X, weight, centers, max_iter=int(self.max_iter),
                                   tol=tolerance, n_threads=self.n_threads_,
                                   algorithm=resolved, record_history=self.record_history,
                                   final_X=final_X, center_offset=offset)
                if not np.isfinite(result["inertia"]) or not np.all(np.isfinite(result["centers"])):
                    raise ValueError("Objective overflowed; rescale X or sample_weight before fitting")
                restart_inertias.append(float(result["inertia"]))
                if self.verbose:
                    print(f"[mykmeans] init {run_index + 1}/{repeats}: inertia={result['inertia']:.12g}, iterations={result['n_iter']}")
                if best is None or result["inertia"] < best["inertia"]:
                    best = result
                    self.best_init_ = run_index
            self.cluster_centers_ = np.ascontiguousarray(best["centers"] + offset)
            roundtrip = self.cluster_centers_ - offset
            quantization = float(np.max(np.abs(roundtrip.astype(np.float64) - best["centers"])))
            self.labels_ = np.asarray(best["labels"], dtype=np.int32)
            self.inertia_ = float(best["inertia"])
            self.n_iter_ = int(best["n_iter"])
            self.history_ = best["history"] if self.record_history else []
            self.distance_evaluations_ = int(best["distance_evaluations"])
            self.converged_ = bool(best["converged"])
            self.convergence_reason_ = best["convergence_reason"]
            self.n_threads_ = int(best.get("n_threads", self.n_threads_))
            self.assignment_engine_ = best.get("assignment_engine")
            self.restart_inertias_ = np.asarray(restart_inertias)
            self.cluster_weight_ = np.bincount(self.labels_, weights=weight, minlength=self.n_clusters)
            self.cluster_sizes_ = np.bincount(self.labels_, minlength=self.n_clusters)
        finally:
            if restore_user_input:
                X += offset
        self.center_quantization_ = quantization
        self.center_quantization_corrected_ = bool(quantization > 0)
        distinct = np.unique(self.labels_).size
        if distinct < self.n_clusters:
            warnings.warn(f"Number of distinct clusters ({distinct}) found smaller than n_clusters ({self.n_clusters}). Possibly due to duplicate points in X.",
                          ConvergenceWarning, stacklevel=2)
        return self

    def _fit_sparse(self, X, weight, resolved):
        from . import _sparse
        if resolved not in ("lloyd", "elkan"):
            raise ValueError(f"algorithm={resolved!r} requires dense input; CSR supports 'lloyd', 'elkan', or 'auto'")
        parts = _csr_parts(X)
        n, d = X.shape
        self.backend_ = "cython-csr"
        self.algorithm_ = resolved
        rng = check_random_state(self.random_state)
        array_init = not isinstance(self.init, str) and not callable(self.init)
        initial = None
        if array_init:
            initial = check_array(self.init, dtype=X.dtype, order="C", copy=True)
            if initial.shape != (self.n_clusters, d):
                raise ValueError(f"init must have shape ({self.n_clusters}, {d})")
        repeats = (1 if array_init or self.init == "k-means++" else 10) if self.n_init == "auto" else int(self.n_init)
        if array_init and repeats != 1:
            warnings.warn("Explicit initial centers: performing only one init instead of n_init runs", RuntimeWarning, stacklevel=2)
            repeats = 1
        self.n_init_ = repeats
        _, variance = _sparse.moments(*parts)
        self.tol_ = float(np.mean(variance) * self.tol)
        if not np.isfinite(self.tol_):
            raise ValueError("Data variance overflowed; rescale X before fitting")
        best = None
        restart_inertias = []
        for run_index in range(repeats):
            if array_init:
                centers = initial.copy()
            elif callable(self.init):
                centers = check_array(self.init(X, self.n_clusters, random_state=rng), dtype=X.dtype, order="C", copy=True)
                if centers.shape != (self.n_clusters, d):
                    raise ValueError(f"Callable init must return shape ({self.n_clusters}, {d})")
            elif self.init == "random":
                seeds = rng.choice(n, self.n_clusters, replace=False, p=weight / weight.sum())
                centers = _csr_rows(parts, seeds)
            else:
                centers, _ = _seed_sparse(parts, self.n_clusters, weight, rng, None, self.n_threads_)
            result = _sparse.run(*parts, weight, centers, max_iter=int(self.max_iter),
                                 tol=self.tol_, n_threads=self.n_threads_, algorithm=resolved,
                                 record_history=self.record_history)
            if not np.isfinite(result["inertia"]) or not np.all(np.isfinite(result["centers"])):
                raise ValueError("Objective overflowed; rescale X or sample_weight before fitting")
            restart_inertias.append(float(result["inertia"]))
            if self.verbose:
                print(f"[mykmeans CSR] init {run_index+1}/{repeats}: inertia={result['inertia']:.12g}, iterations={result['n_iter']}")
            if best is None or result["inertia"] < best["inertia"]:
                best = result
                self.best_init_ = run_index
        self.cluster_centers_ = np.ascontiguousarray(best["centers"])
        self.labels_ = np.asarray(best["labels"], dtype=np.int32)
        self.inertia_ = float(best["inertia"])
        self.n_iter_ = int(best["n_iter"])
        self.history_ = best["history"] if self.record_history else []
        self.distance_evaluations_ = int(best["distance_evaluations"])
        self.converged_ = bool(best["converged"])
        self.convergence_reason_ = best["convergence_reason"]
        self.n_threads_ = int(best.get("n_threads", self.n_threads_))
        self.assignment_engine_ = best.get("assignment_engine", "csr-" + resolved)
        self.restart_inertias_ = np.asarray(restart_inertias)
        self.cluster_weight_ = np.bincount(self.labels_, weights=weight, minlength=self.n_clusters)
        self.cluster_sizes_ = np.bincount(self.labels_, minlength=self.n_clusters)
        self.center_quantization_ = 0.0
        self.center_quantization_corrected_ = False
        distinct = np.unique(self.labels_).size
        if distinct < self.n_clusters:
            warnings.warn(f"Number of distinct clusters ({distinct}) found smaller than n_clusters ({self.n_clusters}). Possibly due to duplicate points in X.", ConvergenceWarning, stacklevel=2)
        return self

    def _prediction_data(self, X):
        check_is_fitted(self, "cluster_centers_")
        return validate_data(self, X, reset=False, accept_sparse="csr",
                             dtype=self.cluster_centers_.dtype, order="C")

    def _assign(self, X):
        if sparse.issparse(X):
            from . import _sparse
            return _sparse.assign(*_csr_parts(X), np.ascontiguousarray(self.cluster_centers_), n_threads=self.n_threads_)
        with _THREADPOOLS.limit(limits=1, user_api="blas"):
            return _core.assign(X, np.ascontiguousarray(self.cluster_centers_), n_threads=self.n_threads_)

    def predict(self, X):
        """Return the closest-center index, with the lowest index winning ties."""
        X = self._prediction_data(X)
        labels, _ = self._assign(X)
        return labels

    def predict_distances(self, X):
        """Squared distance to the nearest center for each observation."""
        X = self._prediction_data(X)
        _, distance = self._assign(X)
        return distance

    def transform(self, X):
        """Euclidean distances to every center, as in scikit-learn."""
        original = X
        X = self._prediction_data(X)
        if sparse.issparse(X):
            from . import _sparse
            result = _sparse.distances(*_csr_parts(X), np.ascontiguousarray(self.cluster_centers_),
                                       squared=False, n_threads=self.n_threads_).astype(self.cluster_centers_.dtype, copy=False)
        else:
            result = _core.distances(X, np.ascontiguousarray(self.cluster_centers_), squared=False,
                                     n_threads=self.n_threads_)
        return self._wrap_output(result, original)

    def transform_squared(self, X):
        """Float64 squared Euclidean distances to every center (allocates n-by-k)."""
        X = self._prediction_data(X)
        if sparse.issparse(X):
            from . import _sparse
            return _sparse.distances(*_csr_parts(X), np.ascontiguousarray(self.cluster_centers_),
                                     squared=True, n_threads=self.n_threads_)
        return _core.squared_distances(X, np.ascontiguousarray(self.cluster_centers_),
                                       n_threads=self.n_threads_)

    def fit_predict(self, X, y=None, sample_weight=None):
        return self.fit(X, y=y, sample_weight=sample_weight).labels_

    def fit_transform(self, X, y=None, sample_weight=None):
        return self.fit(X, y=y, sample_weight=sample_weight).transform(X)

    def score(self, X, y=None, sample_weight=None):
        """Negative weighted squared-distance objective."""
        X = self._prediction_data(X)
        weight = _weights(sample_weight, X.shape[0])
        _, distance = self._assign(X)
        return -float(np.dot(weight, distance))

    def diagnostics(self, X, sample_weight=None, covariance="none"):
        """Weighted ANOVA and cluster statistics; expensive work is opt-in."""
        from .analysis import cluster_statistics
        X = self._prediction_data(X)
        if sparse.issparse(X):
            raise TypeError("diagnostics requires dense X; explicitly convert only if it fits memory")
        labels = self._assign(X)[0]
        return cluster_statistics(X, labels, centers=self.cluster_centers_,
                                  sample_weight=sample_weight, covariance=covariance)

    def soft_assign(self, X, temperature=1.0):
        """Gibbs weights derived from squared distances, not fitted probabilities."""
        from .analysis import soft_assign
        return soft_assign(self.transform_squared(X), temperature=temperature)

    def voronoi(self):
        """Closed Euclidean Voronoi halfspaces for each fitted center."""
        from .geometry import voronoi_halfspaces
        check_is_fitted(self, "cluster_centers_")
        return voronoi_halfspaces(self.cluster_centers_)


def k_means(X, n_clusters, *, sample_weight=None, init="k-means++", n_init="auto",
            max_iter=300, verbose=False, tol=1e-4, random_state=None,
            copy_x=True, algorithm="lloyd", return_n_iter=False):
    """Functional API returning (centers, labels, inertia[, n_iter])."""
    estimator = KMeans(n_clusters=n_clusters, init=init, n_init=n_init, max_iter=max_iter,
                       verbose=verbose, tol=tol, random_state=random_state,
                       copy_x=copy_x, algorithm=algorithm).fit(X, sample_weight=sample_weight)
    result = (estimator.cluster_centers_, estimator.labels_, estimator.inertia_)
    return result + (estimator.n_iter_,) if return_n_iter else result
