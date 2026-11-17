"""Opt-in statistics, derivatives, geometry, and model selection."""
import numpy as np
from mykmeans import (KMeans, FeatureMetric, objective_gradient,
                       assignment_hessian, evaluate_k, clustering_stability,
                       gap_statistic)

rng = np.random.RandomState(0)
centers = rng.uniform(-10, 10, size=(3, 3))
X = centers[np.arange(240) % 3] + 0.6 * rng.normal(size=(240, 3))
weights = np.linspace(0.5, 1.5, len(X))
model = KMeans(3, algorithm="hamerly", tol=0, random_state=0,
               n_threads=1, record_history=True).fit(X, sample_weight=weights)
stats = model.diagnostics(X, sample_weight=weights, covariance="full")
np.testing.assert_allclose(stats["total_ss"], stats["within_ss"] + stats["between_ss"])
np.testing.assert_allclose(stats["inertia"], stats["within_ss"] + stats["center_offset_ss"])
gradient = objective_gradient(X, model.cluster_centers_, labels=model.labels_, sample_weight=weights)
hessian = assignment_hessian(model.labels_, 3, 3, sample_weight=weights)
print("gradient norm:", np.linalg.norm(gradient), "Hessian shape:", hessian.shape)
A, b = model.voronoi()
print("Voronoi inequality shapes:", A.shape, b.shape)
np.testing.assert_allclose(model.soft_assign(X[:5]).sum(axis=1), 1)

metric = FeatureMetric(feature_weights=[1, 4, 0.5])
transformed = metric.fit_transform(X)
metric_model = KMeans(3, random_state=0, n_threads=1).fit(transformed)
print("weighted-metric objective:", metric_model.inertia_)
print("k sweep:", evaluate_k(X, [2, 3, 4], n_init=2, n_threads=1))
print("seed stability:", clustering_stability(X, 3, seeds=[0, 1, 2], n_threads=1)["mean_ari"])
print("gap selected k:", gap_statistic(X, [1, 2, 3, 4], n_refs=3,
                                       n_threads=1, n_init=2)["chosen_k"])
