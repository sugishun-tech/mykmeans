"""Behavioral tests: independent objectives and scikit-learn differential checks."""

import inspect
import pickle
import warnings

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from scipy import sparse
from scipy.optimize import linear_sum_assignment
from sklearn.base import clone
from sklearn.cluster import KMeans as ReferenceKMeans
from sklearn.datasets import make_blobs
from sklearn.exceptions import ConvergenceWarning, NotFittedError
from sklearn.metrics import adjusted_rand_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from mykmeans import KMeans, k_means, kmeans_plusplus


def data(dtype=np.float64, n=191, d=5, k=4):
    x, _ = make_blobs(n_samples=n, n_features=d, centers=k, random_state=17,
                      cluster_std=1.1)
    return np.asarray(x, dtype=dtype, order="C")


def objective(x, centers, weights=None):
    """Small, deliberately independent reference, not a native kernel."""
    x = np.asarray(x, dtype=np.float64)
    centers = np.asarray(centers, dtype=np.float64)
    d2 = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    labels = d2.argmin(axis=1)
    residual = d2[np.arange(len(x)), labels]
    return labels, residual, np.dot(residual, np.ones(len(x)) if weights is None else weights)


@pytest.mark.parametrize("algorithm", ["lloyd", "hamerly"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("weighted", [False, True])
def test_explicit_init_matches_reference(algorithm, dtype, weighted):
    x = data(dtype)
    initial = x[[0, 12, 80, 130]].copy()
    w = np.random.RandomState(81).uniform(0.1, 3.0, len(x)) if weighted else None
    params = dict(n_clusters=4, init=initial, n_init=1, tol=0, max_iter=100)
    model = KMeans(**params, algorithm=algorithm, n_threads=1).fit(x, sample_weight=w)
    ref = ReferenceKMeans(**params, algorithm="lloyd").fit(x, sample_weight=w)
    assert_allclose(model.inertia_, ref.inertia_, rtol=2e-5 if dtype == np.float32 else 1e-10)
    assert adjusted_rand_score(model.labels_, ref.labels_) == 1
    rows, columns = linear_sum_assignment(((model.cluster_centers_[:, None] -
                                            ref.cluster_centers_[None, :]) ** 2).sum(axis=2))
    assert_allclose(model.cluster_centers_[rows], ref.cluster_centers_[columns],
                    rtol=2e-5, atol=2e-6)
    assert model.cluster_centers_.dtype == dtype
    assert model.labels_.dtype.kind in "iu"


@pytest.mark.parametrize("algorithm", ["lloyd", "hamerly", "auto", "elkan"])
def test_prediction_transform_score_consistency(algorithm):
    x = data()
    weights = np.random.RandomState(3).uniform(0.2, 2, len(x))
    model = KMeans(4, algorithm=algorithm, n_init=2, random_state=4, n_threads=1)
    y = model.fit_predict(x, sample_weight=weights)
    expected_labels, d2, inertia = objective(x, model.cluster_centers_, weights)
    assert_array_equal(y, expected_labels)
    assert_array_equal(model.predict(x), expected_labels)
    distances = model.transform(x)
    assert_allclose(distances ** 2,
                    ((x[:, None] - model.cluster_centers_[None, :]) ** 2).sum(axis=2),
                    atol=1e-10)
    assert_allclose(model.inertia_, inertia, rtol=1e-12)
    assert_allclose(model.score(x, sample_weight=weights), -inertia, rtol=1e-12)
    assert_allclose(model.score(x), -d2.sum(), rtol=1e-12)
    assert hasattr(model, "backend_")


@pytest.mark.parametrize("algorithm", ["lloyd", "hamerly"])
def test_early_stop_predict_is_final_assignment(algorithm):
    x = np.random.RandomState(7).normal(size=(101, 4))
    model = KMeans(5, init=x[:5], n_init=1, max_iter=1, tol=0,
                   algorithm=algorithm, n_threads=1).fit(x)
    assert model.n_iter_ == 1
    labels, _, inertia = objective(x, model.cluster_centers_)
    assert_array_equal(model.labels_, labels)
    assert_array_equal(model.labels_, model.predict(x))
    assert_allclose(model.inertia_, inertia)


def test_public_constructor_standard_defaults_match():
    public = inspect.signature(KMeans).parameters
    reference = inspect.signature(ReferenceKMeans).parameters
    for name, value in reference.items():
        assert name in public
        assert public[name].default == value.default
        assert public[name].kind == value.kind


def test_clone_pipeline_grid_search_and_pickle():
    x = data(n=80)
    base = KMeans(4, n_init=1, random_state=6, n_threads=1)
    copied = clone(base)
    assert copied.get_params() == base.get_params()
    assert copied is not base
    pipeline = make_pipeline(StandardScaler(), base)
    grid = GridSearchCV(pipeline, {"kmeans__n_clusters": [3, 4]}, cv=2)
    grid.fit(x)
    assert grid.best_estimator_.predict(x).shape == (80,)
    fit = base.fit(x)
    restored = pickle.loads(pickle.dumps(fit))
    assert_array_equal(restored.predict(x), fit.labels_)
    assert_allclose(restored.transform(x), fit.transform(x))


@pytest.mark.parametrize("method", ["predict", "transform", "score"])
def test_not_fitted(method):
    with pytest.raises(NotFittedError):
        getattr(KMeans(2), method)([[0.0, 1.0]])


def test_fit_transform_and_feature_names():
    pd = pytest.importorskip("pandas")
    x = data()
    frame = pd.DataFrame(x, columns=[f"f{i}" for i in range(x.shape[1])])
    model = KMeans(4, random_state=3, n_threads=1)
    distances = model.fit_transform(frame)
    assert_allclose(distances, model.transform(frame))
    assert_array_equal(model.feature_names_in_, frame.columns)
    assert model.n_features_in_ == 5
    assert len(model.get_feature_names_out()) == 4
    with pytest.raises(ValueError):
        model.predict(frame.iloc[:, ::-1])
    with pytest.warns(UserWarning, match="does not have valid feature names"):
        with pytest.raises(ValueError):
            model.predict(np.zeros((5, 6)))


@pytest.mark.parametrize("copy_x", [True, False])
@pytest.mark.parametrize("layout", ["C", "F", "slice", "readonly"])
def test_input_layout_and_copy_behavior(copy_x, layout):
    x = data()
    if layout == "F":
        x = np.asfortranarray(x)
    elif layout == "slice":
        x = x[::2]
    elif layout == "readonly":
        x.flags.writeable = False
    before = x.copy()
    model = KMeans(4, random_state=11, copy_x=copy_x, n_threads=1).fit(x)
    assert_allclose(x, before, rtol=0, atol=4e-15)
    assert_array_equal(model.labels_, model.predict(x))


@pytest.mark.parametrize("algorithm", ["lloyd", "hamerly"])
def test_integer_weights_match_repetition(algorithm):
    x = np.array([[0, 0], [0, 2], [9, 8], [11, 10]], dtype=float)
    weights = np.array([1, 4, 2, 3])
    init = x[[0, 3]]
    opts = dict(n_clusters=2, init=init, n_init=1, tol=0,
                algorithm=algorithm, n_threads=1)
    weighted = KMeans(**opts).fit(x, sample_weight=weights)
    repeated = KMeans(**opts).fit(np.repeat(x, weights, axis=0))
    assert_allclose(weighted.cluster_centers_, repeated.cluster_centers_)
    assert_allclose(weighted.inertia_, repeated.inertia_)


def test_zero_weight_outlier_does_not_change_centers():
    x = np.array([[0., 0], [0, 2], [10, 10], [10, 12], [1000, -1000]])
    init = x[[0, 2]].copy()
    opts = dict(n_clusters=2, init=init, n_init=1, tol=0, n_threads=1)
    weighted = KMeans(**opts).fit(x, sample_weight=[1, 1, 1, 1, 0])
    baseline = KMeans(**opts).fit(x[:-1])
    assert_allclose(weighted.cluster_centers_, baseline.cluster_centers_, atol=1e-10)
    assert_allclose(weighted.inertia_, baseline.inertia_, atol=1e-10)


@pytest.mark.parametrize("weights", [[-1, 1, 1], [0, 0, 0], [1, np.nan, 1],
                                     [1, np.inf, 1], [1, 2]])
def test_invalid_weights_rejected(weights):
    with pytest.raises(ValueError):
        KMeans(2, n_threads=1).fit([[0., 0], [1, 1], [2, 2]], sample_weight=weights)


@pytest.mark.parametrize("kwargs", [{"n_clusters": 0}, {"n_clusters": -1},
                                    {"n_init": 0}, {"max_iter": 0},
                                    {"tol": -1}, {"algorithm": "bogus"},
                                    {"init": "bogus"}])
def test_invalid_parameters_rejected(kwargs):
    with pytest.raises((ValueError, TypeError)):
        KMeans(**kwargs).fit(data())


@pytest.mark.parametrize("x", [np.array([[np.nan, 1], [2, 3]]),
                               np.array([[np.inf, 1], [2, 3]]),
                               np.empty((0, 2)), np.ones(5), np.empty((5, 0))])
def test_invalid_data_rejected(x):
    with pytest.raises(ValueError):
        KMeans(1).fit(x)


def test_too_many_clusters_rejected():
    with pytest.raises(ValueError):
        KMeans(4).fit([[1., 2], [3, 4], [5, 6]])


@pytest.mark.parametrize("algorithm", ["lloyd", "hamerly"])
def test_duplicates_constant_data_and_empty_relocation(algorithm):
    x = np.repeat([[0., 0.], [1., 1.], [8., 8.]], repeats=[5, 7, 8], axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model = KMeans(4, init=np.zeros((4, 2)), n_init=1, max_iter=30,
                       algorithm=algorithm, n_threads=1).fit(x)
    assert np.isfinite(model.cluster_centers_).all()
    assert model.inertia_ >= 0
    assert_allclose(model.inertia_, 0, atol=1e-15)
    assert_array_equal(model.labels_, model.predict(x))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        constant = KMeans(3, algorithm=algorithm, random_state=3,
                          n_threads=1).fit(np.ones((10, 2)))
    assert_allclose(constant.inertia_, 0)
    assert np.isfinite(constant.cluster_centers_).all()


@pytest.mark.parametrize("algorithm", ["lloyd", "hamerly"])
def test_single_cluster_equals_weighted_mean(algorithm):
    x = data()
    w = np.linspace(0, 2, len(x))
    model = KMeans(1, algorithm=algorithm, n_threads=1).fit(x, sample_weight=w)
    assert_allclose(model.cluster_centers_[0], np.average(x, axis=0, weights=w))
    assert_array_equal(model.labels_, np.zeros(len(x), dtype=int))


def test_ties_choose_first_cluster():
    x = np.array([[-2.], [2.], [0.]])
    model = KMeans(2, init=[[-2.], [2.]], n_init=1, max_iter=1, n_threads=1).fit(x)
    midpoint = model.cluster_centers_.mean(axis=0, keepdims=True)
    assert model.predict(midpoint)[0] == 0


def test_callable_init_and_auto_restarts():
    x = data()
    calls = []

    def initialize(X, n_clusters, random_state):
        calls.append((len(X), n_clusters))
        return X[random_state.choice(len(X), n_clusters, replace=False)]

    model = KMeans(4, init=initialize, n_init="auto", random_state=12,
                   n_threads=1).fit(x)
    assert len(calls) == 10
    assert len(model.restart_inertias_) == 10
    array_model = KMeans(4, init=x[:4], n_init="auto", n_threads=1).fit(x)
    assert len(array_model.restart_inertias_) == 1
    plusplus = KMeans(4, n_init="auto", random_state=2, n_threads=1).fit(x)
    assert len(plusplus.restart_inertias_) == 1


def test_n_init_keeps_best_restart_and_seed_is_repeatable():
    x = np.random.RandomState(7).normal(size=(160, 3))
    opts = dict(n_clusters=5, init="random", n_init=4, random_state=13,
                n_threads=1)
    first, second = KMeans(**opts).fit(x), KMeans(**opts).fit(x)
    assert_allclose(first.inertia_, min(first.restart_inertias_))
    assert_array_equal(first.labels_, second.labels_)
    assert_allclose(first.cluster_centers_, second.cluster_centers_, rtol=0, atol=0)


@pytest.mark.parametrize("format", ["csr", "csc"])
def test_sparse_input_public_contract(format):
    x = data(n=100)
    sx = getattr(sparse, f"{format}_matrix")(x)
    model = KMeans(4, random_state=7, n_threads=1).fit(sx)
    assert_array_equal(model.labels_, model.predict(sx))
    assert_allclose(model.transform(sx), model.transform(x), rtol=1e-12)
    assert_allclose(model.score(sx), -model.inertia_)
    assert model.backend_ == "cython-csr"


def test_elkan_uses_independent_native_engine():
    model = KMeans(4, algorithm="elkan", n_threads=1, random_state=1).fit(data())
    assert model.backend_ == "cython"
    assert model.assignment_engine_ == "elkan"


def test_functional_api_and_kmeans_plusplus():
    x = data()
    centers, indices = kmeans_plusplus(x, 4, random_state=7)
    assert centers.shape == (4, 5)
    assert len(indices) == 4
    assert_allclose(centers, x[indices])
    result = k_means(x, 4, init=centers, n_init=1, return_n_iter=True)
    assert len(result) == 4
    fitted_centers, labels, inertia, n_iter = result
    expected_labels, _, expected_inertia = objective(x, fitted_centers)
    assert_array_equal(labels, expected_labels)
    assert_allclose(inertia, expected_inertia)
    assert n_iter >= 1
    assert len(k_means(x, 4, init=centers, n_init=1)) == 3


def test_parallel_fit_matches_serial_quality():
    x = data(n=5000)
    init = x[[0, 12, 80, 130]].copy()
    opts = dict(n_clusters=4, init=init, n_init=1, algorithm="hamerly", tol=0)
    serial = KMeans(**opts, n_threads=1).fit(x)
    parallel = KMeans(**opts, n_threads=2).fit(x)
    assert_allclose(serial.inertia_, parallel.inertia_, rtol=1e-12)
    assert adjusted_rand_score(serial.labels_, parallel.labels_) == 1


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("seed", range(6))
def test_hamerly_bounds_preserve_lloyd_solution(dtype, seed):
    rng = np.random.RandomState(seed)
    x = rng.normal(size=(137 + seed, 1 + 2 * seed)).astype(dtype)
    weights = rng.uniform(0.05, 3, len(x))
    weights[::7] = 0
    initial = x[rng.choice(len(x), 1 + seed, replace=False)]
    opts = dict(n_clusters=1 + seed, init=initial, n_init=1, max_iter=100,
                tol=0, n_threads=1)
    lloyd = KMeans(**opts, algorithm="lloyd").fit(x, sample_weight=weights)
    hamerly = KMeans(**opts, algorithm="hamerly").fit(x, sample_weight=weights)
    assert_allclose(lloyd.inertia_, hamerly.inertia_, rtol=1e-12)
    assert_array_equal(lloyd.labels_, hamerly.labels_)


@pytest.mark.parametrize("algorithm", ["lloyd", "hamerly"])
def test_large_translation_preserves_clustering(algorithm):
    x = data(n=120, d=3, k=4)
    initial = x[[0, 12, 70, 110]]
    offset = np.array([1e8, -1e8, 1e8])
    opts = dict(n_clusters=4, n_init=1, tol=0, algorithm=algorithm, n_threads=1)
    baseline = KMeans(**opts, init=initial).fit(x)
    translated = KMeans(**opts, init=initial + offset).fit(x + offset)
    assert adjusted_rand_score(baseline.labels_, translated.labels_) == 1
    assert_allclose(baseline.inertia_, translated.inertia_, rtol=1e-8)
    assert_allclose(baseline.cluster_centers_, translated.cluster_centers_ - offset,
                    rtol=1e-8, atol=2e-8)


@pytest.mark.parametrize("dtype,offset,scale", [(np.float32, 1e7, 3.0),
                                              (np.float64, 1e15, 0.5)])
def test_rounded_large_offset_centers_have_consistent_objective(dtype, offset, scale):
    rng = np.random.RandomState(1)
    x = np.asarray(offset + scale * rng.normal(size=(40, 2)), dtype=dtype)
    model = KMeans(3, init=x[:3], n_init=1, n_threads=1, tol=0).fit(x)
    labels, _, inertia = objective(x, model.cluster_centers_)
    assert_array_equal(model.labels_, labels)
    assert_allclose(model.inertia_, inertia, rtol=1e-12)
    assert_allclose(model.score(x), -inertia, rtol=1e-12)


@pytest.mark.parametrize("x", [np.full((8, 2), 1e308),
                               np.array([[-1e308, 1e308], [1e308, -1e308]])])
def test_overflow_rejection_does_not_corrupt_input(x):
    before = x.copy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        with pytest.raises(ValueError):
            KMeans(1, copy_x=False, n_threads=1).fit(x)
    assert_array_equal(x, before)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_squared_distance_helpers(dtype):
    x = data(dtype=dtype, n=83)
    model = KMeans(4, random_state=5, n_threads=1).fit(x)
    _, nearest, _ = objective(x, model.cluster_centers_)
    expected = ((x.astype(float)[:, None] - model.cluster_centers_.astype(float)[None]) ** 2).sum(axis=2)
    assert_allclose(model.transform_squared(x), expected, rtol=2e-7, atol=2e-7)
    assert_allclose(model.predict_distances(x), nearest, rtol=1e-12)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("d,k", [(8, 8), (16, 12), (128, 16)])
def test_gemm_assignment_matches_direct_hamerly(dtype, d, k):
    rng = np.random.RandomState(63)
    x = rng.normal(size=(211, d)).astype(dtype)
    weight = rng.uniform(0.05, 3.0, len(x))
    weight[::11] = 0
    initial = x[rng.choice(len(x), k, replace=False)]
    options = dict(n_clusters=k, init=initial, n_init=1, tol=0, n_threads=1)
    lloyd = KMeans(**options, algorithm="lloyd").fit(x, sample_weight=weight)
    hamerly = KMeans(**options, algorithm="hamerly").fit(x, sample_weight=weight)
    assert_array_equal(lloyd.labels_, hamerly.labels_)
    assert_allclose(lloyd.inertia_, hamerly.inertia_, rtol=2e-12)
    labels, _, inertia = objective(x, lloyd.cluster_centers_, weight)
    assert_array_equal(lloyd.labels_, labels)
    assert_allclose(lloyd.inertia_, inertia, rtol=2e-6 if dtype == np.float32 else 1e-12)


@pytest.mark.parametrize("dtype,offset", [(np.float32, 1e7), (np.float64, 1e15)])
def test_gemm_near_ties_with_large_common_offset(dtype, offset):
    rng = np.random.RandomState(31)
    x = (offset + 3 * rng.normal(size=(64, 16))).astype(dtype)
    model = KMeans(8, init=x[:8], n_init=1, tol=0,
                   max_iter=3, n_threads=1).fit(x)
    labels, _, inertia = objective(x, model.cluster_centers_)
    assert_array_equal(model.labels_, labels)
    assert_array_equal(model.predict(x), labels)
    assert_allclose(model.inertia_, inertia, rtol=1e-12)
    assert_allclose(model.score(x), -inertia, rtol=1e-12)


@pytest.mark.parametrize("x,initial,dtype", [
    ([4.4692063331604, 6.947237491607666, 7.773247718811035, 8.599258422851562],
     [6.121227264404297, 8.599258422851562], np.float32),
    ([-6.525582415492611, -4.358386221741273, -3.6359874904908267, -2.9135887592403806],
     [-5.080784952991719, -2.9135887592403806], np.float64),
])
def test_voronoi_boundary_after_tiny_center_rounding(x, initial, dtype):
    # Even sub-epsilon center round-trips can cross a Voronoi boundary.
    # A heuristic threshold on coordinate magnitude cannot certify labels.
    x = np.array(x, dtype=dtype)[:, None]
    initial = np.array(initial, dtype=dtype)[:, None]
    model = KMeans(2, init=initial, n_init=1, max_iter=1, tol=0,
                   n_threads=1).fit(x)
    labels, _, inertia = objective(x, model.cluster_centers_)
    assert_array_equal(model.labels_, labels)
    assert_array_equal(model.predict(x), labels)
    assert_allclose(model.inertia_, inertia, rtol=1e-12)

