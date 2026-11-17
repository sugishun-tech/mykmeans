import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.pipeline import Pipeline

from mykmeans.geometry import FeatureMetric, voronoi_halfspaces


def test_feature_weight_distances_and_inverse():
    X = np.array([[1., 2., 3.], [3., 1., -2.], [0., 0., 0.]])
    weights = np.array([1., 4., 9.])
    metric = FeatureMetric(feature_weights=weights).fit(X)
    Z = metric.transform(X)
    assert_allclose(Z, X * np.sqrt(weights))
    assert_allclose(np.sum((Z[0] - Z[1]) ** 2), np.dot(weights, (X[0] - X[1]) ** 2))
    assert_allclose(metric.inverse_transform(Z), X)
    assert metric.rank_ == 3
    assert_array_equal(metric.get_feature_names_out(), ["featuremetric0", "featuremetric1", "featuremetric2"])
    assert clone(metric).feature_weights is not metric.feature_weights


def test_mahalanobis_psd_and_singular_projection():
    X = np.array([[1., 2.], [3., 4.]])
    M = np.array([[2., 1.], [1., 3.]])
    transform = FeatureMetric(metric=M).fit(X)
    Z = transform.transform(X)
    difference = X[0] - X[1]
    assert_allclose(np.sum((Z[0] - Z[1]) ** 2), difference @ M @ difference)
    assert_allclose(transform.inverse_transform(Z), X, atol=1e-14)
    singular = FeatureMetric(feature_weights=[0, 1]).fit(X)
    assert_allclose(singular.inverse_transform(singular.transform(X)), [[0, 2], [0, 4]])
    assert singular.rank_ == 1


def test_identity_and_pipeline():
    X = np.arange(20, dtype=float).reshape(10, 2)
    pipeline = Pipeline([("metric", FeatureMetric())])
    assert_allclose(pipeline.fit_transform(X), X)
    with pytest.raises(NotFittedError):
        FeatureMetric().transform(X)
    with pytest.raises(ValueError):
        FeatureMetric().fit(X).transform(np.ones((3, 3)))


@pytest.mark.parametrize("kwargs", [
    {"feature_weights": [1, -1]},
    {"feature_weights": [1]},
    {"feature_weights": [1, np.nan]},
    {"metric": [[1, 1], [0, 1]]},
    {"metric": [[1, 0], [0, -1]]},
    {"metric": [[-1e-20, 0], [0, -1e-20]]},
    {"metric": [[1e-20, 1e-20], [0, 1e-20]]},
    {"metric": [[1, 0, 0], [0, 1, 0]]},
    {"metric": np.eye(2), "feature_weights": [1, 1]},
])
def test_invalid_metric(kwargs):
    with pytest.raises(ValueError):
        FeatureMetric(**kwargs).fit([[0, 0], [1, 1]])


def test_small_valid_metric_preserves_relative_geometry():
    M = np.array([[2., .5], [.5, 1.]]) * 1e-20
    X = np.array([[1., 2.], [4., -1.]])
    transformer = FeatureMetric(metric=M).fit(X)
    Z = transformer.transform(X)
    delta = X[0] - X[1]
    assert_allclose(np.dot(Z[0] - Z[1], Z[0] - Z[1]), delta @ M @ delta, rtol=1e-14, atol=0)
    assert transformer.rank_ == 2
    assert_allclose(transformer.inverse_transform(Z), X, atol=1e-14)


def test_halfspaces_classify_points_correctly():
    centers = np.array([[0., 0.], [4., 0.], [0., 4.]])
    X = np.array([[0., 1.], [3., 1.], [1., 3.], [2., 2.], [-3., -3.]])
    distances = np.sum((X[:, None] - centers[None]) ** 2, axis=2)
    all_A, all_b = voronoi_halfspaces(centers)
    assert all_A.shape == (3, 2, 2)
    assert all_b.shape == (3, 2)
    for j in range(3):
        A, b = voronoi_halfspaces(centers, j)
        assert_allclose(A, all_A[j])
        assert_allclose(b, all_b[j])
        contained = np.all(X @ A.T <= b + 1e-12, axis=1)
        closest = distances[:, j] <= distances.min(axis=1) + 1e-12
        assert_array_equal(contained, closest)


def test_power_cells_have_correct_weight_sign():
    C = np.array([[0.], [4.]])
    # Increasing the first center's power weight expands its cell to x <= 3.
    A, b = voronoi_halfspaces(C, 0, power_weights=[8, 0])
    assert_allclose(A, [[8]])
    assert_allclose(b, [24])
    assert 2.9 * A[0, 0] <= b[0]
    assert not 3.1 * A[0, 0] <= b[0]


def test_one_center_and_duplicates():
    A, b = voronoi_halfspaces([[1., 2.]])
    assert A.shape == (1, 0, 2)
    assert b.shape == (1, 0)
    A, b = voronoi_halfspaces([[1.], [1.]], 0, power_weights=[0, 1])
    assert_array_equal(A, [[0]])
    assert_array_equal(b, [-1])  # empty first cell


def test_feature_metric_pandas_output_and_index():
    pd = pytest.importorskip("pandas")
    X = pd.DataFrame([[1., 2.], [3., 4.]], columns=["x", "y"], index=["a", "b"])
    metric = FeatureMetric(feature_weights=[1, 4]).set_output(transform="pandas")
    result = metric.fit_transform(X)
    assert isinstance(result, pd.DataFrame)
    assert_array_equal(result.index, X.index)
    assert_array_equal(result.columns, ["featuremetric0", "featuremetric1"])
    assert_allclose(result, [[1, 4], [3, 8]])


def test_feature_metric_pandas_output_from_list():
    pd = pytest.importorskip("pandas")
    metric = FeatureMetric().set_output(transform="pandas")
    result = metric.fit_transform([[1., 2.], [3., 4.]])
    assert isinstance(result, pd.DataFrame)
    assert_array_equal(result.index, [0, 1])
