"""Explicit, opt-in mathematical diagnostics for Euclidean k-means.

These routines are deliberately outside the fitting hot path. Calculations use
float64, population (not unbiased sample) moments, and nonnegative sample
weights. Expensive diagnostics are never run implicitly by ``KMeans.fit``.
"""

from __future__ import annotations

from numbers import Integral, Real

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.sparse import diags
from scipy.spatial.distance import cdist
from ._metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from ._compat import check_random_state, check_array


def _data(X):
    return check_array(X, dtype=np.float64, ensure_2d=True, order="C")


def _integer(value, name, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    if value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}.")
    return int(value)


def _weights(sample_weight, n):
    if sample_weight is None:
        return np.ones(n, dtype=np.float64)
    w = np.asarray(sample_weight, dtype=np.float64)
    if w.ndim == 0:
        w = np.full(n, float(w), dtype=np.float64)
    if w.shape != (n,):
        raise ValueError(f"sample_weight must have shape ({n},).")
    if not np.isfinite(w).all() or np.any(w < 0):
        raise ValueError("sample_weight must be finite and nonnegative.")
    with np.errstate(over="ignore", invalid="ignore"):
        mass = w.sum(dtype=np.float64)
    if not np.isfinite(mass) or mass <= 0:
        raise ValueError("sample_weight must have a positive, finite sum.")
    return w


def _labels(labels, n, n_clusters=None):
    a = np.asarray(labels)
    if a.shape != (n,) or a.dtype.kind not in "iu":
        raise ValueError(f"labels must be an integer array of shape ({n},).")
    if np.any(a < 0):
        raise ValueError("labels must be nonnegative.")
    if n_clusters is not None and np.any(a >= n_clusters):
        raise ValueError("labels must be smaller than n_clusters.")
    return np.asarray(a, dtype=np.intp)


def _centers(centers, n_features):
    C = _data(centers)
    if C.shape[1] != n_features:
        raise ValueError("X and centers must have the same number of features.")
    return C


def _nearest(X, centers, batch_size=8192):
    labels = np.empty(len(X), dtype=np.intp)
    for start in range(0, len(X), batch_size):
        distance = cdist(X[start : start + batch_size], centers, "sqeuclidean")
        if not np.isfinite(distance).all():
            raise ValueError("Squared distances overflowed; rescale X and centers.")
        labels[start : start + batch_size] = distance.argmin(axis=1)
    return labels


def cluster_statistics(
    X,
    labels,
    *,
    centers=None,
    sample_weight=None,
    n_clusters=None,
    covariance="diag",
):
    """Compute weighted moments, objective, and a correct ANOVA decomposition.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
    labels : integer array-like of shape (n_samples,)
    centers : array-like of shape (n_clusters, n_features), optional
        Centers at which to evaluate ``inertia``. If omitted, empirical weighted
        cluster means are used. They need not be converged Lloyd centers.
    sample_weight : nonnegative array-like or scalar, optional
    n_clusters : int, optional
        Include empty clusters; otherwise infer from centers or labels.
    covariance : {None, 'none', 'diag', 'full'}, default='diag'
        Population covariance about empirical cluster means. Full covariance
        costs O(n_samples * n_features**2) work and O(k * n_features**2) output.

    Returns
    -------
    dict
        ``counts``, ``weight_sums``, ``means``, ``covariances``, ``radii``,
        ``inertia_per_cluster``, ``inertia``, ``quantization_error``,
        ``within_per_cluster``, ``within_ss``, ``between_ss``, ``total_ss``,
        ``center_offset_ss``, ``global_mean``, and ``explained_variance_ratio``.
        Empty/zero-mass clusters have NaN means, covariance, and radius, and zero
        inertia. Radii exclude zero-weight observations. The identities are
        total_ss = within_ss + between_ss and
        inertia = within_ss + center_offset_ss (up to roundoff).
    """
    X = _data(X)
    n, d = X.shape
    labels = _labels(labels, n)
    w = _weights(sample_weight, n)
    C = None if centers is None else _centers(centers, d)
    inferred = len(C) if C is not None else int(labels.max()) + 1
    k = inferred if n_clusters is None else _integer(n_clusters, "n_clusters")
    if C is not None and len(C) != k:
        raise ValueError("n_clusters must equal the number of supplied centers.")
    labels = _labels(labels, n, k)
    if covariance == "none":
        covariance = None
    if covariance not in (None, "diag", "full"):
        raise ValueError("covariance must be None, 'none', 'diag', or 'full'.")

    counts = np.bincount(labels, minlength=k)
    masses = np.bincount(labels, weights=w, minlength=k)
    means = np.full((k, d), np.nan)
    within = np.zeros(k)
    objective = np.zeros(k)
    offset = np.zeros(k)
    radii = np.full(k, np.nan)
    shape = (k, d) if covariance == "diag" else (k, d, d)
    covariances = None if covariance is None else np.full(shape, np.nan)

    # Compute in a translated coordinate system to avoid subtracting two huge
    # raw second moments. Zero-weight observations have no statistical effect.
    active = w > 0
    active_X, active_w = X[active], w[active]
    anchor = active_X[0]
    with np.errstate(over="ignore", invalid="ignore"):
        global_delta = np.sum((active_X - anchor) * (active_w / active_w.sum())[:, None], axis=0)
        global_mean = anchor + global_delta
        total_residual = (active_X - anchor) - global_delta
        total = float(np.einsum("i,ij,ij->", active_w, total_residual, total_residual))
    if not np.isfinite(global_mean).all() or not np.isfinite(total):
        raise ValueError("Weighted moments overflowed; rescale X or sample_weight.")
    between = 0.0
    for j in range(k):
        mask = (labels == j) & (w > 0)
        if not mask.any():
            continue
        points, weights = X[mask], w[mask]
        anchor_j = points[0]
        with np.errstate(over="ignore", invalid="ignore"):
            mean_delta = np.sum((points - anchor_j) * (weights / masses[j])[:, None], axis=0)
        mean = anchor_j + mean_delta
        means[j] = mean
        residual = (points - anchor_j) - mean_delta
        squared = np.einsum("ij,ij->i", residual, residual)
        within[j] = np.dot(weights, squared)
        radii[j] = np.sqrt(squared.max())
        delta = (anchor_j - anchor) + mean_delta - global_delta
        between += masses[j] * np.dot(delta, delta)
        if C is None:
            objective[j] = within[j]
        else:
            with np.errstate(over="ignore", invalid="ignore"):
                residual_c = points - C[j]
                objective[j] = np.einsum("i,ij,ij->", weights, residual_c, residual_c)
                center_delta = (C[j] - anchor_j) - mean_delta
                offset[j] = masses[j] * np.dot(center_delta, center_delta)
            if not np.isfinite(objective[j]) or not np.isfinite(offset[j]):
                raise ValueError("Center distances overflowed; rescale X, centers, or sample_weight.")
        if covariance == "diag":
            covariances[j] = np.einsum("i,ij,ij->j", weights, residual, residual) / masses[j]
        elif covariance == "full":
            covariances[j] = (residual.T * weights) @ residual / masses[j]
    inertia = float(objective.sum())
    return {
        "counts": counts,
        "weight_sums": masses,
        "means": means,
        "covariances": covariances,
        "radii": radii,
        "inertia_per_cluster": objective,
        "inertia": inertia,
        "quantization_error": inertia / w.sum(),
        "within_per_cluster": within,
        "within_ss": float(within.sum()),
        "between_ss": float(between),
        "total_ss": total,
        "center_offset_ss": float(offset.sum()),
        "global_mean": global_mean,
        "explained_variance_ratio": float(between / total) if total > 0 else np.nan,
    }


def objective_gradient(X, centers, *, labels=None, sample_weight=None):
    """Return the (k, d) gradient of the selected quadratic objective branch.

    With fixed ``labels``, this is the exact gradient everywhere. With omitted
    labels, nearest centers are selected and the result is a gradient of the
    k-means minimum objective away from Voronoi ties. At a tie, that minimum can
    be nondifferentiable; this returns the branch selected by the smallest
    center index, without asserting differentiability.
    """
    X = _data(X)
    C = _centers(centers, X.shape[1])
    w = _weights(sample_weight, len(X))
    labels = None if labels is None else _labels(labels, len(X), len(C))
    active = w > 0
    if not np.all(active):
        X, w = X[active], w[active]
        if labels is not None:
            labels = labels[active]
    labels = _nearest(X, C) if labels is None else labels
    gradient = np.zeros_like(C)
    with np.errstate(over="ignore", invalid="ignore"):
        for start in range(0, len(X), 8192):
            stop = start + 8192
            block_labels = labels[start:stop]
            residual = C[block_labels] - X[start:stop]
            np.add.at(gradient, block_labels, 2 * w[start:stop, None] * residual)
    if not np.isfinite(gradient).all():
        raise ValueError("Gradient overflowed; rescale X, centers, or sample_weight.")
    return gradient


def assignment_hessian(labels, n_clusters, n_features, *, sample_weight=None, sparse=True):
    """Hessian with assignments fixed, in row-major flattened-center order.

    The j-th d-by-d block is ``2 * weight_sum[j] * I``. A CSR sparse matrix is
    returned by default. This is not a Hessian across Voronoi boundaries, where
    the nearest-center objective may be nondifferentiable.
    """
    k = _integer(n_clusters, "n_clusters")
    d = _integer(n_features, "n_features")
    labels = np.asarray(labels)
    if labels.ndim != 1 or not len(labels):
        raise ValueError("labels must be a nonempty one-dimensional array.")
    labels = _labels(labels, len(labels), k)
    w = _weights(sample_weight, len(labels))
    diagonal = 2 * np.repeat(np.bincount(labels, weights=w, minlength=k), d)
    if sparse:
        return diags(diagonal, format="csr")
    return np.diag(diagonal)


def align_centers(reference, candidate):
    """Match equally many centers by minimum total squared Euclidean distance.

    Return ``(aligned, permutation)`` such that ``aligned = candidate[permutation]``.
    To relabel old candidate labels, use ``np.argsort(permutation)[labels]``.
    This is an optimal bijective correspondence, not an affine registration.
    """
    A, B = _data(reference), _data(candidate)
    if A.shape != B.shape:
        raise ValueError("reference and candidate must have the same shape.")
    cost = cdist(A, B, metric="sqeuclidean")
    if not np.isfinite(cost).all():
        raise ValueError("Squared distances overflowed; rescale centers.")
    _, permutation = linear_sum_assignment(cost)
    return B[permutation], permutation


def align_labels(reference_labels, candidate_labels):
    """Relabel a partition to maximize agreement via a one-to-one assignment.

    The two partitions must contain the same number of observed clusters.
    Reference labels need not be contiguous. The returned values are drawn from
    the reference labels, and adjusted Rand index is unchanged by this operation.
    """
    a, b = np.asarray(reference_labels), np.asarray(candidate_labels)
    if a.ndim != 1 or not len(a):
        raise ValueError("reference_labels must be a nonempty 1D array.")
    a, b = _labels(a, len(a)), _labels(b, len(a))
    va, ia = np.unique(a, return_inverse=True)
    vb, ib = np.unique(b, return_inverse=True)
    if len(va) != len(vb):
        raise ValueError("Both partitions must have the same number of observed clusters.")
    contingency = np.zeros((len(va), len(vb)), dtype=np.int64)
    np.add.at(contingency, (ia, ib), 1)
    rows, cols = linear_sum_assignment(contingency, maximize=True)
    mapping = np.empty(len(vb), dtype=va.dtype)
    mapping[cols] = va[rows]
    return mapping[ib]


def soft_assign(X, centers=None, *, temperature=1.0, batch_size=8192):
    """Return stable Gibbs weights exp(-squared_distance / temperature).

    If centers are given, X is an observation matrix. Otherwise X must already
    contain nonnegative squared distances with shape (n_samples, n_clusters).
    Each row sums to one. These are geometric soft assignments, not calibrated
    posterior probabilities or the result of fitting a Gaussian mixture model.
    ``temperature`` has the units of squared distance. Output memory is O(n*k).
    """
    if not isinstance(temperature, Real) or not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be positive and finite.")
    batch_size = _integer(batch_size, "batch_size")
    X = _data(X)
    C = None if centers is None else _centers(centers, X.shape[1])
    if C is None and np.any(X < 0):
        raise ValueError("Precomputed squared distances must be nonnegative.")
    k = X.shape[1] if C is None else len(C)
    result = np.empty((len(X), k), dtype=np.float64)
    for start in range(0, len(X), batch_size):
        stop = start + batch_size
        D = X[start:stop].copy() if C is None else cdist(X[start:stop], C, "sqeuclidean")
        if not np.isfinite(D).all():
            raise ValueError("Squared distances overflowed; rescale X and centers.")
        D -= D.min(axis=1, keepdims=True)
        with np.errstate(over="ignore", under="ignore"):
            np.divide(D, -float(temperature), out=D)
            np.exp(D, out=D)
        D /= D.sum(axis=1, keepdims=True)
        result[start:stop] = D
    return result


def _candidate_ks(ks, n_samples):
    ks = [_integer(k, "each k") for k in ks]
    if not ks or len(set(ks)) != len(ks):
        raise ValueError("ks must be a nonempty sequence of distinct positive integers.")
    if max(ks) > n_samples:
        raise ValueError("Each k must be no greater than n_samples.")
    return ks


def _model(n_clusters, random_state, parameters):
    from ._kmeans import KMeans

    if "n_clusters" in parameters or "random_state" in parameters:
        raise ValueError("Pass n_clusters/ks and random_state through their named arguments.")
    return KMeans(n_clusters=n_clusters, random_state=random_state, **parameters)


def evaluate_k(
    X,
    ks,
    *,
    sample_weight=None,
    random_state=0,
    silhouette_sample_size=2000,
    return_models=False,
    **kmeans_kwargs,
):
    """Fit candidate k values and return a list of metric dictionaries.

    Inertia uses sample_weight. Silhouette, Calinski-Harabasz, and Davies-Bouldin
    are the usual *unweighted* diagnostics, implemented here independently, even
    when fitting is weighted.
    Silhouette is evaluated on at most ``silhouette_sample_size`` observations;
    pass None for the full quadratic-work computation (distances are blocked). Undefined
    diagnostics (one cluster, one cluster per sample, or a degenerate silhouette
    subsample) are NaN. No metric here is claimed to discover a unique true k.
    """
    X = _data(X)
    ks = _candidate_ks(ks, len(X))
    w = None if sample_weight is None else _weights(sample_weight, len(X))
    if silhouette_sample_size is not None:
        silhouette_sample_size = _integer(silhouette_sample_size, "silhouette_sample_size", 2)
    rng = check_random_state(random_state)
    # Reusing the same seed gives each k a reproducible starting random stream.
    fit_seed = int(rng.randint(np.iinfo(np.int32).max))
    sample_seed = int(rng.randint(np.iinfo(np.int32).max))
    records = []
    for k in ks:
        model = _model(k, fit_seed, kmeans_kwargs).fit(X, sample_weight=w)
        labels = model.labels_
        count = len(np.unique(labels))
        silhouette = calinski = davies = np.nan
        if 1 < count < len(X):
            calinski = float(calinski_harabasz_score(X, labels))
            davies = float(davies_bouldin_score(X, labels))
            size = None if silhouette_sample_size is None else min(len(X), silhouette_sample_size)
            try:
                silhouette = float(silhouette_score(X, labels, sample_size=size, random_state=sample_seed))
            except ValueError:
                # A random subsample can contain only one observed cluster.
                silhouette = np.nan
        record = {
            "k": k,
            "observed_clusters": count,
            "inertia": float(model.inertia_),
            "n_iter": int(model.n_iter_),
            "silhouette": silhouette,
            "calinski_harabasz": calinski,
            "davies_bouldin": davies,
            "diagnostic_weighting": "unweighted",
        }
        if return_models:
            record["model"] = model
        records.append(record)
    return records


def clustering_stability(
    X,
    n_clusters,
    *,
    seeds=(0, 1, 2, 3, 4),
    sample_weight=None,
    return_labels=False,
    **kmeans_kwargs,
):
    """Compare independent initializations on the *same* data using pairwise ARI.

    This measures initialization stability, not bootstrap/sample stability.
    ARI is unweighted and invariant to label permutations. Use n_init=1 to expose
    sensitivity of individual starts; larger n_init measures the entire estimator.
    """
    X = _data(X)
    k = _integer(n_clusters, "n_clusters")
    seeds = tuple(_integer(s, "each seed", 0) for s in seeds)
    if len(seeds) < 2 or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must contain at least two distinct integer seeds.")
    w = None if sample_weight is None else _weights(sample_weight, len(X))
    models = [_model(k, seed, kmeans_kwargs).fit(X, sample_weight=w) for seed in seeds]
    n = len(models)
    ari = np.eye(n)
    for i in range(n):
        for j in range(i):
            ari[i, j] = ari[j, i] = adjusted_rand_score(models[i].labels_, models[j].labels_)
    pairs = ari[np.triu_indices(n, 1)]
    result = {
        "seeds": seeds,
        "pairwise_ari": ari,
        "mean_ari": float(pairs.mean()),
        "min_ari": float(pairs.min()),
        "inertias": np.array([model.inertia_ for model in models]),
    }
    if return_labels:
        result["labels"] = np.stack([model.labels_ for model in models])
    return result


def gap_statistic(
    X,
    ks,
    *,
    n_refs=10,
    reference="pca",
    random_state=0,
    **kmeans_kwargs,
):
    """Estimate Tibshirani-Walther-Hastie's gap statistic using Euclidean inertia.

    ``reference='pca'`` samples a uniform box in empirical principal-component
    coordinates; ``'axis'`` uses the input axes. This is an expensive, explicit
    diagnostic requiring len(ks)*(n_refs+1) fits. It is unweighted. ``chosen_k``
    is the first candidate whose gap is at least the next candidate's gap minus
    that next candidate's standard error; otherwise it is the final candidate.
    Supply consecutive ks to obtain the usual k versus k+1 selection rule.
    Zero inertias make log dispersion undefined and raise ValueError.
    """
    X = _data(X)
    ks = sorted(_candidate_ks(ks, len(X)))
    n_refs = _integer(n_refs, "n_refs", 2)
    if reference not in ("pca", "axis"):
        raise ValueError("reference must be 'pca' or 'axis'.")
    if "sample_weight" in kmeans_kwargs:
        raise ValueError("gap_statistic is unweighted; sample_weight is not supported.")
    rng = check_random_state(random_state)
    anchor = X[0]
    mean = anchor + (X - anchor).mean(axis=0)
    if reference == "pca":
        _, _, axes = np.linalg.svd(X - mean, full_matrices=False)
        coordinates = (X - mean) @ axes.T
    else:
        axes = np.eye(X.shape[1])
        coordinates = X - mean
    lower, upper = coordinates.min(axis=0), coordinates.max(axis=0)
    log_observed = np.empty(len(ks))
    log_reference = np.empty((n_refs, len(ks)))

    def log_inertia(data, k):
        seed = int(rng.randint(np.iinfo(np.int32).max))
        value = float(_model(k, seed, kmeans_kwargs).fit(data).inertia_)
        if not np.isfinite(value) or value <= 0:
            raise ValueError("The gap statistic requires positive finite inertia for every fit.")
        return np.log(value)

    for j, k in enumerate(ks):
        log_observed[j] = log_inertia(X, k)
    # A reference draw is shared across all k values, reducing comparison noise.
    for b in range(n_refs):
        reference_X = rng.uniform(lower, upper, size=coordinates.shape) @ axes + mean
        for j, k in enumerate(ks):
            log_reference[b, j] = log_inertia(reference_X, k)
    gap = log_reference.mean(axis=0) - log_observed
    standard_error = log_reference.std(axis=0, ddof=1) * np.sqrt(1 + 1 / n_refs)
    chosen_k = ks[-1]
    for j in range(len(ks) - 1):
        if gap[j] >= gap[j + 1] - standard_error[j + 1]:
            chosen_k = ks[j]
            break
    return {
        "ks": np.asarray(ks, dtype=int),
        "gap": gap,
        "standard_error": standard_error,
        "log_inertia": log_observed,
        "reference_log_inertia": log_reference,
        "chosen_k": chosen_k,
        "reference": reference,
    }


__all__ = [
    "cluster_statistics", "objective_gradient", "assignment_hessian",
    "align_centers", "align_labels", "soft_assign", "evaluate_k",
    "clustering_stability", "gap_statistic",
]
