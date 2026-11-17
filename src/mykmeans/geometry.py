"""Linear metrics and explicit Voronoi / power-diagram geometry."""

from __future__ import annotations

from numbers import Integral

import numpy as np
from ._compat import BaseEstimator, TransformerMixin
from ._compat import check_array, check_is_fitted, validate_data


class FeatureMetric(TransformerMixin, BaseEstimator):
    """Embed a positive-semidefinite quadratic metric in Euclidean coordinates.

    Pass either nonnegative ``feature_weights`` or a symmetric PSD ``metric``.
    For row vectors the transformation is ``X @ factor_``, where
    ``metric_ = factor_ @ factor_.T``. Thus squared Euclidean distance after
    transformation equals ``(x-y) @ metric_ @ (x-y).T`` before transformation.
    No centering or standardization is applied. A singular metric is a
    pseudometric; inverse_transform uses a pseudoinverse and cannot restore its
    discarded null-space coordinates.

    Examples
    --------
    >>> import numpy as np
    >>> transform = FeatureMetric(feature_weights=[1, 4]).fit([[0, 0], [1, 1]])
    >>> transform.transform([[1, 2]])
    array([[1., 4.]])
    """

    def __init__(self, *, feature_weights=None, metric=None):
        self.feature_weights = feature_weights
        self.metric = metric

    def fit(self, X, y=None):
        """Validate the metric and compute its factorization."""
        X = validate_data(self, X, dtype=np.float64, reset=True)
        d = X.shape[1]
        if self.feature_weights is not None and self.metric is not None:
            raise ValueError("Specify feature_weights or metric, not both.")
        if self.feature_weights is not None:
            weights = np.asarray(self.feature_weights, dtype=np.float64)
            if weights.shape != (d,) or not np.isfinite(weights).all() or np.any(weights < 0):
                raise ValueError("feature_weights must be finite, nonnegative, and have one entry per feature.")
            self.metric_ = np.diag(weights)
            self.factor_ = np.diag(np.sqrt(weights))
            self.rank_ = int(np.count_nonzero(weights))
        elif self.metric is not None:
            metric = np.asarray(self.metric, dtype=np.float64)
            if metric.shape != (d, d) or not np.isfinite(metric).all():
                raise ValueError("metric must be a finite square matrix with one row per feature.")
            # A unit absolute floor would accept an entirely negative matrix
            # such as [[-1e-20]]. Roundoff tolerance must follow the matrix's
            # own scale, even when its coefficients are very small.
            scale = max(np.finfo(float).tiny, float(np.max(np.abs(metric))))
            tolerance = 100 * np.finfo(float).eps * d * scale
            if not np.allclose(metric, metric.T, rtol=0, atol=tolerance):
                raise ValueError("metric must be symmetric.")
            metric = metric * 0.5 + metric.T * 0.5
            eigenvalues, vectors = np.linalg.eigh(metric)
            if not np.isfinite(eigenvalues).all():
                raise ValueError("Metric eigenvalues overflowed; rescale metric.")
            if eigenvalues.min() < -tolerance:
                raise ValueError("metric must be positive semidefinite.")
            eigenvalues = np.maximum(eigenvalues, 0)
            self.factor_ = vectors * np.sqrt(eigenvalues)
            self.metric_ = self.factor_ @ self.factor_.T
            self.rank_ = int(np.linalg.matrix_rank(self.factor_))
        else:
            self.factor_ = np.eye(d)
            self.metric_ = np.eye(d)
            self.rank_ = d
        self.rank_ = int(np.linalg.matrix_rank(self.factor_))
        self.inverse_factor_ = np.linalg.pinv(self.factor_, rcond=max(self.factor_.shape) * np.finfo(float).eps)
        return self

    def transform(self, X):
        """Return metric coordinates as a dense float64 array."""
        check_is_fitted(self, "factor_")
        original = X
        X = validate_data(self, X, dtype=np.float64, reset=False)
        return self._wrap_output(X @ self.factor_, original)

    def inverse_transform(self, X):
        """Return original coordinates, or their projection for a singular metric."""
        check_is_fitted(self, "factor_")
        X = check_array(X, dtype=np.float64)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("The number of metric coordinates does not match the fitted transformer.")
        return X @ self.inverse_factor_

    def get_feature_names_out(self, input_features=None):
        """Return stable names for transformed metric coordinates."""
        check_is_fitted(self, "factor_")
        if input_features is not None:
            names = np.asarray(input_features, dtype=object)
            if names.shape != (self.n_features_in_,):
                raise ValueError("input_features must contain one name per input feature.")
            if hasattr(self, "feature_names_in_") and not np.array_equal(names, self.feature_names_in_):
                raise ValueError("input_features must match feature_names_in_.")
        return np.asarray([f"featuremetric{i}" for i in range(self.n_features_in_)], dtype=object)


def voronoi_halfspaces(centers, index=None, *, power_weights=None):
    """Return (A, b) for the closed cell ``A @ x <= b`` of one center.

    For ordinary Voronoi cells, the score is ||x-c_j||**2. Optional
    ``power_weights[j]`` changes it to ||x-c_j||**2 - power_weights[j], producing
    a power diagram. These are additive geometric weights, unrelated to k-means
    sample weights. Boundaries belong to both adjacent closed cells; the k-means
    implementation can still resolve label ties by index.

    If index is None, return all cells with A.shape == (k, k-1, d) and
    b.shape == (k, k-1). Otherwise return one cell, with rows following center
    order and ``index`` omitted. Repeated centers can produce
    all-zero redundant inequalities; different power weights can make a cell
    empty. For one center, A has shape (0, d) and b has shape (0,).
    """
    C = check_array(centers, dtype=np.float64)
    k, d = C.shape
    p = np.zeros(k) if power_weights is None else np.asarray(power_weights, dtype=np.float64)
    if p.shape != (k,) or not np.isfinite(p).all():
        raise ValueError("power_weights must be a finite vector with one entry per center.")
    if index is None:
        A, b = np.empty((k, k - 1, d)), np.empty((k, k - 1))
        for j in range(k):
            A[j], b[j] = voronoi_halfspaces(C, j, power_weights=p)
        return A, b
    if isinstance(index, (bool, np.bool_)) or not isinstance(index, Integral) or not 0 <= index < k:
        raise ValueError("index must be an integer selecting one center.")
    other = np.arange(k) != index
    difference = C[other] - C[index]
    A = 2 * difference
    b = np.einsum("ij,ij->i", difference, C[other] + C[index]) + p[index] - p[other]
    if not np.isfinite(A).all() or not np.isfinite(b).all():
        raise ValueError("Halfspace coefficients overflowed; rescale the centers.")
    return A.reshape(k - 1, d), b


__all__ = ["FeatureMetric", "voronoi_halfspaces"]
