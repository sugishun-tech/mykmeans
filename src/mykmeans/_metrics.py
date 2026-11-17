"""Independent clustering diagnostics built from their mathematical definitions.

No scikit-learn numerical routine is imported. NumPy supplies array operations;
SciPy's cdist supplies a general-purpose Euclidean-distance primitive. Pairwise
distance workspaces are blocked instead of materializing an n-by-n matrix.
"""
from __future__ import annotations

from numbers import Integral

import numpy as np
from scipy.spatial.distance import cdist

from ._compat import check_array, check_random_state


def _encode_labels(labels, n_samples=None):
    values = np.asarray(labels)
    if values.ndim != 1:
        raise ValueError("labels must be a one-dimensional array")
    if n_samples is not None and len(values) != n_samples:
        raise ValueError("labels and X must contain the same number of samples")
    if values.dtype.kind in "fc" and not np.isfinite(values).all():
        raise ValueError("labels must be finite")
    try:
        unique, encoded, counts = np.unique(values, return_inverse=True, return_counts=True)
    except (TypeError, ValueError) as error:
        raise ValueError("labels must contain comparable categorical values") from error
    return unique, encoded.astype(np.intp, copy=False), counts


def _score_data(X, labels):
    X = check_array(X, dtype=np.float64, order="C")
    _, encoded, counts = _encode_labels(labels, len(X))
    k = len(counts)
    if not 2 <= k <= len(X) - 1:
        raise ValueError("Number of labels must be between 2 and n_samples - 1")
    return X, encoded, counts


def _block_size(requested, width):
    if isinstance(requested, (bool, np.bool_)) or not isinstance(requested, Integral) or requested < 1:
        raise ValueError("chunk_size must be a positive integer")
    # The primary pairwise array stays under approximately 64 MiB.
    return max(1, min(int(requested), (64 * 1024 * 1024) // max(8 * width, 1)))


def _translated(X):
    with np.errstate(over="ignore", invalid="ignore"):
        translated = X - X[0]
    if not np.isfinite(translated).all():
        raise ValueError("Coordinate differences overflowed; rescale X")
    return translated


def adjusted_rand_score(labels_true, labels_pred):
    """Adjusted Rand index using exact integer combinatorial counts.

    Empty/singleton inputs and equivalent trivial partitions return 1. Labels
    identify categories, so their numerical ordering and naming do not matter.
    Only observed category pairs are counted; no dense k_true-by-k_pred table
    is allocated. This is the usual unweighted adjusted Rand index.
    """
    left, a, counts_a = _encode_labels(labels_true)
    right, b, counts_b = _encode_labels(labels_pred, len(a))
    n = len(a)
    if n < 2:
        return 1.0
    if len(left) * len(right) <= np.iinfo(np.int64).max:
        keys = a.astype(np.int64, copy=False) * len(right) + b
        _, cell_counts = np.unique(keys, return_counts=True)
    else:  # Defensive fallback for extraordinarily large category spaces.
        _, cell_counts = np.unique(np.column_stack((a, b)), axis=0, return_counts=True)

    def pairs(counts):
        # Python integers prevent silent overflow in n^4-scale intermediates.
        return sum(int(value) * (int(value) - 1) // 2 for value in counts)

    observed = pairs(cell_counts)
    same_a, same_b = pairs(counts_a), pairs(counts_b)
    total = n * (n - 1) // 2
    denominator = total * (same_a + same_b) - 2 * same_a * same_b
    if denominator == 0:
        return 1.0
    return float(2 * (total * observed - same_a * same_b) / denominator)


def silhouette_samples(X, labels, *, chunk_size=256):
    """Per-observation Euclidean silhouette with bounded pairwise workspace.

    The coefficient is (b-a)/max(a,b), where a is the mean distance to other
    members of the same cluster and b the smallest mean distance to another
    cluster. Singleton clusters and coincident zero-distance cases receive 0.
    Work is still quadratic in n; blocking bounds memory, not arithmetic cost.
    """
    X, encoded, counts = _score_data(X, labels)
    n = len(X)
    block = _block_size(chunk_size, n)
    order = np.argsort(encoded, kind="stable")
    ordered_X = X[order]
    ordered_labels = encoded[order]
    starts = np.concatenate(([0], np.cumsum(counts[:-1], dtype=np.intp)))
    column_scale = 1.0 / counts[ordered_labels]
    result = np.zeros(n, dtype=np.float64)
    for first in range(0, n, block):
        stop = min(n, first + block)
        distance = cdist(X[first:stop], ordered_X, metric="euclidean")
        if not np.isfinite(distance).all():
            raise ValueError("Pairwise distances overflowed; rescale X")
        # Normalize before reducing so representable means do not overflow
        # merely because a cluster contains many observations.
        distance *= column_scale
        cluster_mean = np.add.reduceat(distance, starts, axis=1)
        current = encoded[first:stop]
        rows = np.arange(stop - first)
        own_count = counts[current]
        own = cluster_mean[rows, current].copy()
        correction = np.ones(stop - first, dtype=np.float64)
        np.divide(own_count, own_count - 1, out=correction, where=own_count > 1)
        own *= correction
        own[own_count == 1] = 0.0
        cluster_mean[rows, current] = np.inf
        other = cluster_mean.min(axis=1)
        denominator = np.maximum(own, other)
        value = np.zeros(stop - first, dtype=np.float64)
        np.divide(other - own, denominator, out=value, where=denominator > 0)
        value[own_count == 1] = 0.0
        result[first:stop] = np.clip(value, -1.0, 1.0)
    return result


def silhouette_score(X, labels, *, sample_size=None, random_state=None, chunk_size=256):
    """Mean Euclidean silhouette, optionally on a sampled sub-dataset.

    Sampling selects observations without replacement and computes all terms
    within that sub-dataset. Use sample_size to limit the quadratic work.
    """
    X = check_array(X, dtype=np.float64, order="C")
    _, encoded, _ = _encode_labels(labels, len(X))
    if sample_size is not None:
        if isinstance(sample_size, (bool, np.bool_)) or not isinstance(sample_size, Integral) or sample_size < 1:
            raise ValueError("sample_size must be a positive integer or None")
        rng = check_random_state(random_state)
        selected = rng.permutation(len(X))[:min(int(sample_size), len(X))]
        X, encoded = X[selected], encoded[selected]
    return float(silhouette_samples(X, encoded, chunk_size=chunk_size).mean())


def calinski_harabasz_score(X, labels):
    """Between/within-cluster variance ratio, computed independently.

    For k observed clusters, the score is B*(n-k)/(W*(k-1)). The conventional
    finite value 1 is returned when W is zero. This score is unweighted.
    """
    X, encoded, counts = _score_data(X, labels)
    n, k = len(X), len(counts)
    X = _translated(X)
    overall = X.mean(axis=0)
    within, between = 0.0, 0.0
    with np.errstate(over="ignore", invalid="ignore"):
        for j, count in enumerate(counts):
            points = X[encoded == j]
            center = points.mean(axis=0)
            residual = points - center
            delta = center - overall
            within += float(np.einsum("ij,ij->", residual, residual))
            between += int(count) * float(np.dot(delta, delta))
    if not np.isfinite(within) or not np.isfinite(between):
        raise ValueError("Cluster moments overflowed; rescale X")
    if within == 0:
        return 1.0
    return float((between / within) * ((n - k) / (k - 1)))


def davies_bouldin_score(X, labels, *, chunk_size=256):
    """Mean worst scatter/separation ratio with blocked center distances.

    Each cluster's scatter is its mean Euclidean distance to its centroid.
    Coincident centroids are excluded from pair ratios; wholly zero scatter or
    numerically coincident centroids return 0, matching the conventional finite
    degenerate result used by scikit-learn's public metric.
    """
    X, encoded, counts = _score_data(X, labels)
    X = _translated(X)
    k, d = len(counts), X.shape[1]
    block = _block_size(chunk_size, k)
    centers = np.empty((k, d), dtype=np.float64)
    scatter = np.empty(k, dtype=np.float64)
    for j in range(k):
        points = X[encoded == j]
        centers[j] = points.mean(axis=0)
        residual = cdist(points, centers[j:j + 1], metric="euclidean").ravel()
        if not np.isfinite(residual).all():
            raise ValueError("Cluster distances overflowed; rescale X")
        scatter[j] = np.sum(residual / len(points))
    if np.allclose(scatter, 0):
        return 0.0
    largest = np.empty(k, dtype=np.float64)
    separated = False
    for first in range(0, k, block):
        stop = min(first + block, k)
        distance = cdist(centers[first:stop], centers, metric="euclidean")
        if not np.isfinite(distance).all():
            raise ValueError("Center distances overflowed; rescale X")
        separated = separated or bool(np.any(distance > 1e-8))
        distance[distance == 0] = np.inf
        # Dividing the summands separately avoids overflow in s_i+s_j.
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            ratio = scatter[first:stop, None] / distance + scatter[None, :] / distance
        largest[first:stop] = ratio.max(axis=1)
    if not separated:
        return 0.0
    return float(np.sum(largest / k))


__all__ = ["adjusted_rand_score", "silhouette_samples", "silhouette_score",
           "calinski_harabasz_score", "davies_bouldin_score"]
