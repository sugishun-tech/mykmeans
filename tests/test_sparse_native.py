"""Independent CSR behavior and native dense/sparse differential checks."""
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy.sparse import csr_matrix

from mykmeans import _core, _sparse


def arrays(x):
    csr = csr_matrix(x)
    csr.sum_duplicates()
    csr.sort_indices()
    return csr.data, csr.indices.astype(np.int64), csr.indptr.astype(np.int64), csr.shape[1]


def direct(x, centers):
    d2 = ((np.asarray(x, dtype=np.float64)[:, None, :] -
           np.asarray(centers, dtype=np.float64)[None, :, :]) ** 2).sum(axis=2)
    return d2.argmin(axis=1), d2


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("algorithm", ["lloyd", "elkan"])
@pytest.mark.parametrize("weighted", [False, True])
def test_sparse_native_matches_independent_dense_objective(dtype, algorithm, weighted):
    rng = np.random.RandomState(180)
    x = rng.normal(size=(137, 29))
    x[rng.random_sample(x.shape) < 0.73] = 0
    x = x.astype(dtype)
    initial = x[[1, 7, 12, 80]].copy()
    weights = rng.uniform(0.2, 2.0, len(x)) if weighted else np.ones(len(x))
    weights[::17] = 0
    result = _sparse.run(*arrays(x), weights, initial, algorithm=algorithm,
                         max_iter=80, tol=0, n_threads=1, record_history=True)
    expected_labels, distances = direct(x, result["centers"])
    assert_array_equal(result["labels"], expected_labels)
    assert_allclose(result["squared_distances"], distances.min(axis=1), rtol=2e-12, atol=1e-13)
    assert_allclose(result["inertia"], weights @ distances.min(axis=1), rtol=2e-12)
    dense = _core.run(x, weights, initial, algorithm="lloyd", max_iter=80,
                      tol=0, n_threads=1)
    assert_allclose(result["inertia"], dense["inertia"], rtol=1e-6 if dtype == np.float32 else 1e-11)
    assert result["assignment_engine"] == ("csr-elkan" if algorithm == "elkan" else "csr-direct")
    assert result["centers"].dtype == dtype
    assert result["history"]


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_csr_assignment_distances_and_moments(dtype):
    rng = np.random.RandomState(31)
    x = rng.normal(size=(70, 113))
    x[rng.random_sample(x.shape) < 0.95] = 0
    x = x.astype(dtype)
    centers = rng.normal(scale=0.2, size=(7, x.shape[1])).astype(dtype)
    args = arrays(x)
    labels, nearest = _sparse.assign(*args, centers, n_threads=2)
    expected_labels, d2 = direct(x, centers)
    assert_array_equal(labels, expected_labels)
    assert_allclose(nearest, d2.min(axis=1), rtol=1e-12)
    assert_allclose(_sparse.distances(*args, centers, squared=True), d2, rtol=1e-12)
    assert_allclose(_sparse.distances(*args, centers), np.sqrt(d2), rtol=1e-12)
    assert_allclose(_sparse.point_distances(*args, centers[2]), d2[:, 2], rtol=1e-12)
    mean, variance = _sparse.moments(*args)
    assert_allclose(mean, x.astype(float).mean(axis=0), atol=2e-15)
    assert_allclose(variance, x.astype(float).var(axis=0), atol=2e-15)


@pytest.mark.parametrize("algorithm", ["lloyd", "elkan"])
def test_empty_rows_empty_clusters_and_duplicate_centers(algorithm):
    x = np.array([[0., 0, 0], [0, 0, 0], [4, 0, 1], [4, 0, 1], [0, 8, 0], [0, 8, 0]])
    result = _sparse.run(*arrays(x), np.ones(len(x)), np.zeros((4, 3)),
                         max_iter=30, tol=0, algorithm=algorithm)
    labels, d2 = direct(x, result["centers"])
    assert_array_equal(labels, result["labels"])
    assert_allclose(result["inertia"], 0, atol=1e-14)
    assert np.isfinite(result["centers"]).all()


@pytest.mark.parametrize("algorithm", ["lloyd", "elkan"])
def test_all_zero_csr_and_zero_weight_samples(algorithm):
    x = np.zeros((13, 7))
    args = arrays(x)
    result = _sparse.run(*args, np.ones(len(x)), np.zeros((3, 7)), algorithm=algorithm)
    assert_array_equal(result["labels"], np.zeros(len(x)))
    assert result["inertia"] == 0
    x[-1, 0] = 50
    weights = np.ones(len(x))
    weights[-1] = 0
    result = _sparse.run(*arrays(x), weights, np.zeros((1, 7)), algorithm=algorithm)
    assert_allclose(result["centers"], 0)
    assert result["inertia"] == 0


@pytest.mark.parametrize("dtype,offset", [(np.float32, 1e7), (np.float64, 1e15)])
def test_sparse_cancellation_and_exact_ties(dtype, offset):
    x = np.array([[offset, 0, offset], [offset+2, 0, offset+2],
                   [offset+4, 0, offset+4]], dtype=dtype)
    centers = x[[0, 2]].copy()
    labels, nearest = _sparse.assign(*arrays(x), centers)
    expected_labels, d2 = direct(x, centers)
    assert_array_equal(labels, expected_labels)
    assert_allclose(nearest, d2.min(axis=1), rtol=0, atol=0)
    assert labels[1] == 0


@pytest.mark.parametrize("algorithm", ["lloyd", "elkan"])
def test_sparse_early_stop_final_assignment(algorithm):
    rng = np.random.RandomState(91)
    x = rng.normal(size=(67, 19))
    x[rng.uniform(size=x.shape) < 0.5] = 0
    result = _sparse.run(*arrays(x), np.ones(len(x)), x[:5].copy(),
                         algorithm=algorithm, max_iter=1, tol=0)
    labels, d2 = direct(x, result["centers"])
    assert_array_equal(labels, result["labels"])
    assert_allclose(result["inertia"], d2.min(axis=1).sum(), rtol=1e-12)


def test_sparse_moments_large_constant_fully_populated_columns():
    x = np.full((121, 3), 1e15)
    mean, variance = _sparse.moments(*arrays(x))
    assert_array_equal(mean, x[0])
    assert_array_equal(variance, np.zeros(3))


def test_readonly_csr_and_large_feature_space():
    # One hundred thousand dimensions, but only a handful of stored values;
    # no dense n-by-d sample matrix is constructed by this test or the kernel.
    n, d = 50, 100_000
    rows = np.arange(n)
    cols = rows % 17
    values = np.arange(n, dtype=float) + 1
    csr = csr_matrix((values, (rows, cols)), shape=(n, d))
    args = (csr.data, csr.indices.astype(np.int64), csr.indptr.astype(np.int64), d)
    for array in args[:3]:
        array.flags.writeable = False
    centers = np.zeros((2, d))
    centers[1, :17] = 2
    labels, distances = _sparse.assign(*args, centers)
    reference0 = values**2
    reference1 = (values-2)**2 + 16*4
    expected = np.stack([reference0, reference1], axis=1)
    assert_array_equal(labels, expected.argmin(axis=1))
    assert_allclose(distances, expected.min(axis=1), rtol=1e-12)


@pytest.mark.parametrize("algorithm", ["lloyd", "elkan"])
def test_sparse_parallel_matches_serial(algorithm):
    rng = np.random.RandomState(49)
    x = rng.normal(size=(4100, 23))
    x[rng.uniform(size=x.shape) < 0.8] = 0
    args = arrays(x)
    initial = x[[3, 21, 103, 140]].copy()
    serial = _sparse.run(*args, np.ones(len(x)), initial, algorithm=algorithm,
                         tol=0, max_iter=60, n_threads=1)
    parallel = _sparse.run(*args, np.ones(len(x)), initial, algorithm=algorithm,
                           tol=0, max_iter=60, n_threads=3)
    assert_allclose(serial["inertia"], parallel["inertia"], rtol=1e-10)
    assert_array_equal(serial["labels"], parallel["labels"])
