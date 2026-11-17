"""Optional sklearn integration; clustering still uses independent mykmeans."""
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from mykmeans import KMeans, FeatureMetric

rng = np.random.RandomState(17)
centers = rng.normal(size=(3, 4)) * 5
X = centers[np.arange(300) % 3] + rng.normal(size=(300, 4))
pipeline = make_pipeline(FeatureMetric(feature_weights=[1, 2, 1, 0.5]),
                         KMeans(random_state=0, n_threads=1))
search = GridSearchCV(pipeline, {"kmeans__n_clusters": [2, 3, 4]}, cv=3)
search.fit(X)
print("best cross-validation score:", search.best_score_)
print("parameters:", search.best_params_)
# Negative inertia is the estimator's default score. It often prefers larger
# k, so this example demonstrates protocol compatibility, not an optimal-k rule.
