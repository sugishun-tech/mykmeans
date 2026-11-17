"""Independent tests of original Elkan and Exponion implementations."""
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from threadpoolctl import threadpool_limits

from mykmeans import _core


def run(x, centers, algorithm, weight=None, **kwargs):
    x = np.ascontiguousarray(x)
    weight = np.ones(len(x)) if weight is None else np.ascontiguousarray(weight, dtype=float)
    with threadpool_limits(limits=1, user_api="blas"):
        return _core.run(x, weight, np.ascontiguousarray(centers, dtype=x.dtype),
                         algorithm=algorithm, **kwargs)


def reference_objective(x, centers, weight):
    delta = x.astype(float)[:, None] - centers.astype(float)[None]
    squared = np.einsum("nkd,nkd->nk", delta, delta)
    labels = squared.argmin(axis=1)
    return labels, float(np.dot(weight, squared[np.arange(len(x)), labels]))


@pytest.mark.parametrize("algorithm", ["elkan", "exponion"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("d,k", [(2, 7), (12, 10)])
@pytest.mark.parametrize("weighted", [False, True])
def test_extended_matches_lloyd(algorithm, dtype, d, k, weighted):
    rng = np.random.RandomState(73)
    x = rng.normal(size=(257, d)).astype(dtype)
    centers = x[rng.choice(len(x), k, replace=False)]
    weights = rng.uniform(0.01, 3.0, len(x)) if weighted else np.ones(len(x))
    if weighted:
        weights[::13] = 0
    options = dict(max_iter=80, tol=0, n_threads=1, final_X=x)
    expected = run(x, centers, "lloyd", weights, **options)
    actual = run(x, centers, algorithm, weights, **options)
    assert actual["assignment_engine"] == algorithm
    assert_array_equal(actual["labels"], expected["labels"])
    assert_allclose(actual["centers"], expected["centers"], rtol=3e-6, atol=1e-7)
    assert_allclose(actual["inertia"], expected["inertia"], rtol=3e-7)
    independently_assigned, objective = reference_objective(x, actual["centers"], weights)
    assert_array_equal(actual["labels"], independently_assigned)
    assert_allclose(actual["inertia"], objective, rtol=1e-12)


@pytest.mark.parametrize("algorithm", ["elkan", "exponion"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("max_iter", [1, 30])
def test_original_coordinate_finalizer(algorithm, dtype, max_iter):
    rng = np.random.RandomState(1)
    offset_value = 1e7 if dtype is np.float32 else 1e15
    original = (offset_value + 3*rng.normal(size=(70, 3))).astype(dtype)
    offset = np.mean(original, axis=0, dtype=float).astype(dtype)
    x = np.ascontiguousarray(original-offset)
    centers = x[[0, 4, 7, 12]].copy()
    result = run(x, centers, algorithm, max_iter=max_iter, tol=0, n_threads=1,
                 final_X=original, center_offset=offset)
    public_centers = result["centers"]+offset
    expected_labels, objective = reference_objective(original, public_centers, np.ones(len(x)))
    assert_array_equal(result["labels"], expected_labels)
    assert_allclose(result["inertia"], objective, rtol=1e-14)


@pytest.mark.parametrize("algorithm", ["elkan", "exponion"])
def test_empty_clusters_duplicates_and_ties(algorithm):
    x = np.repeat([[0., 0.], [1., 1.], [8., 8.]], [5, 7, 8], axis=0)
    centers = np.zeros((4, 2))
    result = run(x, centers, algorithm, max_iter=50, tol=0, n_threads=1, final_X=x)
    labels, objective = reference_objective(x, result["centers"], np.ones(len(x)))
    assert_array_equal(result["labels"], labels)
    assert_allclose(objective, 0, atol=1e-14)
    assert np.isfinite(result["centers"]).all()
    constant = run(np.ones((12, 2)), np.ones((3, 2)), algorithm,
                   max_iter=20, tol=0, final_X=np.ones((12, 2)))
    assert_array_equal(constant["labels"], np.zeros(12, dtype=np.int32))


@pytest.mark.parametrize("algorithm", ["elkan", "exponion"])
def test_parallel_partition_and_readonly(algorithm):
    rng = np.random.RandomState(9)
    x = rng.uniform(-3, 3, size=(4097, 5))
    centers = x[:9].copy()
    x.flags.writeable = False
    serial = run(x, centers, algorithm, max_iter=50, tol=1e-10,
                 n_threads=1, final_X=x)
    parallel = run(x, centers, algorithm, max_iter=50, tol=1e-10,
                   n_threads=4, final_X=x)
    assert_array_equal(parallel["labels"], serial["labels"])
    assert_allclose(parallel["inertia"], serial["inertia"], rtol=1e-12)


@pytest.mark.parametrize("algorithm", ["elkan", "exponion"])
def test_single_cluster_weighted_mean(algorithm):
    x = np.array([[-5., 1.], [0., 7.], [11., -3.]])
    w = np.array([1., 2., 0.])
    result = run(x, x[:1], algorithm, w, tol=0, max_iter=10, final_X=x)
    assert_allclose(result["centers"][0], np.average(x, axis=0, weights=w))
    assert_array_equal(result["labels"], np.zeros(3, dtype=np.int32))


@pytest.mark.parametrize("algorithm", ["elkan", "exponion"])
def test_bounds_reduce_work_in_long_quantization(algorithm):
    x = np.random.RandomState(0).uniform(-1, 1, (2000, 2))
    centers = x[np.random.RandomState(42).choice(len(x), 32, replace=False)]
    lloyd = run(x, centers, "lloyd", max_iter=80, tol=0, n_threads=1, final_X=x)
    result = run(x, centers, algorithm, max_iter=80, tol=0, n_threads=1, final_X=x)
    assert_array_equal(result["labels"], lloyd["labels"])
    assert result["distance_evaluations"] < lloyd["distance_evaluations"]


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("d", [2, 31])
def test_clipped_candidate_matches_exhaustive_and_readonly(dtype, d):
    rng = np.random.RandomState(42)
    centers = (20*rng.normal(size=(12, d))).astype(dtype)
    x = (centers[rng.randint(len(centers),size=4097)] +
         rng.normal(size=(4097,d))).astype(dtype)
    full = _core.squared_distances(x,centers)
    labels = full.argmin(axis=1).astype(np.int32)
    closest = np.ascontiguousarray(full[np.arange(len(x)),labels])
    for candidate in (x[11].copy(),np.zeros(d,dtype=dtype)):
        center_distances = _core.point_distances(centers,candidate)
        expected = np.minimum(closest,_core.point_distances(x,candidate))
        for value in (x,candidate,closest,labels,center_distances):
            value.flags.writeable = False
        result = _core.point_distances_clipped(x,candidate,closest,labels,
                                                center_distances,n_threads=4)
        assert result.dtype == np.float64
        assert_array_equal(result,expected)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("scale", [1., 1e-15, 1e15])
def test_clipped_candidate_at_triangle_boundary(dtype, scale):
    # At x = midpoint, ||old-candidate||**2 == 4*||x-old||**2.
    # Adjacent representable x values exercise both sides and the exact tie.
    midpoint = dtype(scale)
    x = np.array([[np.nextafter(midpoint,dtype(-np.inf))],
                  [midpoint],
                  [np.nextafter(midpoint,dtype(np.inf))],
                  [dtype(0.)]],dtype=dtype)
    center = np.zeros((1,1),dtype=dtype)
    candidate = np.array([dtype(2)*midpoint],dtype=dtype)
    closest = _core.point_distances(x,center[0])
    labels = np.zeros(len(x),dtype=np.int32)
    between = _core.point_distances(center,candidate)
    expected = np.minimum(closest,_core.point_distances(x,candidate))
    actual = _core.point_distances_clipped(x,candidate,closest,labels,between)
    assert_array_equal(actual,expected)


@pytest.mark.parametrize("scale", [1e-160, 1e-155, 1e153, 1e154])
def test_clipped_candidate_underflow_and_large_squared_distances(scale):
    x = np.array([[0.],[.999999999*scale],[1.000000001*scale],[2*scale]])
    centers = np.array([[0.]])
    candidate = np.array([2*scale])
    closest = _core.point_distances(x,centers[0])
    labels = np.zeros(len(x),dtype=np.int32)
    between = _core.point_distances(centers,candidate)
    expected = np.minimum(closest,_core.point_distances(x,candidate))
    actual = _core.point_distances_clipped(x,candidate,closest,labels,between)
    assert_array_equal(actual,expected)
