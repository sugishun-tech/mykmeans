"""A small, runnable scikit-learn-compatible example."""
import numpy as np
from mykmeans import KMeans

rng = np.random.RandomState(3)
centers = rng.uniform(-10, 10, size=(5, 4))
X = np.ascontiguousarray(centers[np.arange(2000) % 5] + rng.normal(size=(2000, 4)))
model = KMeans(5, algorithm="exponion", random_state=0, n_threads=2).fit(X)
print("backend:", model.backend_, "algorithm:", model.algorithm_)
print("inertia:", model.inertia_, "iterations:", model.n_iter_)
assert np.array_equal(model.predict(X), model.labels_)
np.testing.assert_allclose(-model.score(X), model.inertia_, rtol=1e-10)
print("distances:", model.transform(X[:2]))
