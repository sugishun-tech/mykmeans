"""Cython/OpenMP k-means, with scikit-learn-compatible estimators."""
from ._kmeans import KMeans, k_means, kmeans_plusplus
from .analysis import (cluster_statistics, objective_gradient, assignment_hessian,
                       align_centers, align_labels, soft_assign, evaluate_k,
                       clustering_stability, gap_statistic)
from .geometry import FeatureMetric, voronoi_halfspaces
from ._metrics import (adjusted_rand_score, silhouette_samples, silhouette_score,
                       calinski_harabasz_score, davies_bouldin_score)

__version__ = "0.2.0"
__all__ = ["KMeans", "k_means", "kmeans_plusplus", "cluster_statistics",
           "objective_gradient", "assignment_hessian", "align_centers",
           "align_labels", "soft_assign", "evaluate_k", "clustering_stability",
           "gap_statistic", "FeatureMetric", "voronoi_halfspaces",
           "adjusted_rand_score", "silhouette_samples", "silhouette_score",
           "calinski_harabasz_score", "davies_bouldin_score"]
