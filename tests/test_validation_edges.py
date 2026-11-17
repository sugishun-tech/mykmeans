"""Public validation and the accelerated initializer's distribution contract."""
import numpy as np
import pytest
from numpy.testing import assert_array_equal, assert_allclose
from scipy import sparse

from mykmeans import KMeans, kmeans_plusplus, _core


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("dimensions", [16, 128])
def test_filtered_greedy_seeding_preserves_full_distance_trials(monkeypatch, dtype, dimensions):
    rng = np.random.RandomState(71)
    groups = rng.normal(size=(16, dimensions)) * 30
    x = (groups[np.arange(2048) % 16] + rng.normal(size=(2048, dimensions))).astype(dtype)
    weights = rng.uniform(0.1, 3, len(x))
    weights[::17] = 0
    accelerated = kmeans_plusplus(x, 16, sample_weight=weights, random_state=3)
    calls = []

    def exhaustive(x, candidate, closest, nearest, between, n_threads=1):
        calls.append(1)
        return np.minimum(closest, _core.point_distances(x, candidate, n_threads=n_threads))

    monkeypatch.setattr(_core, "point_distances_clipped", exhaustive)
    full = kmeans_plusplus(x, 16, sample_weight=weights, random_state=3)
    assert calls, "the fixture must exercise the triangle-filter path"
    assert_array_equal(accelerated[0], full[0])
    assert_array_equal(accelerated[1], full[1])


@pytest.mark.parametrize("algorithm", ["lloyd", "elkan"])
@pytest.mark.parametrize("bad_index", [-1, 3])
def test_malformed_csr_rejected_before_native_access(algorithm, bad_index):
    x = sparse.csr_matrix(([1., 2.], [0, 1], [0, 1, 2]), shape=(2, 3))
    x.indices[0] = bad_index
    with pytest.raises(ValueError):
        KMeans(1, algorithm=algorithm, n_threads=1).fit(x)
    with pytest.raises(ValueError):
        kmeans_plusplus(x, 1)


def test_noncanonical_csr_and_no_observation_densification(monkeypatch):
    x = sparse.csr_matrix((np.array([1., 2., 3., 4., 2., 8.]),
                           np.array([2, 0, 0, 1, 2, 0]),
                           np.array([0, 3, 4, 6])), shape=(3, 4))
    dense = x.toarray()
    initial = dense[[0, 1]]
    expected = KMeans(2, init=initial, n_init=1, tol=0, n_threads=1).fit(dense)

    def forbidden(*args, **kwargs):
        raise AssertionError("CSR observations must not be densified")

    monkeypatch.setattr(sparse.csr_matrix, "toarray", forbidden)
    actual = KMeans(2, init=initial, n_init=1, tol=0, algorithm="elkan", n_threads=1).fit(x)
    assert_array_equal(actual.labels_, expected.labels_)
    assert_allclose(actual.cluster_centers_, expected.cluster_centers_)
    assert actual.inertia_ == pytest.approx(expected.inertia_)
    assert_array_equal(actual.predict(x), actual.labels_)


@pytest.mark.parametrize("algorithm", ["hamerly", "exponion"])
def test_dense_only_algorithm_is_not_silently_relabelled_on_csr(algorithm):
    with pytest.raises(ValueError, match="requires dense input"):
        KMeans(2, algorithm=algorithm).fit(sparse.eye(4, format="csr"))
