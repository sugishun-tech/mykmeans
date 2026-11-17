"""Independent metric implementations checked against formulas and references."""
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from sklearn import metrics as reference

from mykmeans._metrics import (
    adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score,
    silhouette_samples, silhouette_score,
)


@pytest.mark.parametrize("seed", range(6))
def test_metric_definitions_match_reference(seed):
    rng = np.random.RandomState(seed)
    n, d, k = 47 + seed * 3, 1 + seed, 2 + seed % 4
    labels = np.arange(n) % k
    rng.shuffle(labels)
    X = rng.normal(size=(n, d)) + labels[:, None] * 1.7
    other = rng.randint(0, k + 2, size=n)
    assert_allclose(adjusted_rand_score(labels, other),
                    reference.adjusted_rand_score(labels, other), rtol=1e-14, atol=1e-15)
    assert_allclose(calinski_harabasz_score(X, labels),
                    reference.calinski_harabasz_score(X, labels), rtol=1e-13)
    assert_allclose(davies_bouldin_score(X, labels, chunk_size=2),
                    reference.davies_bouldin_score(X, labels), rtol=1e-13)
    assert_allclose(silhouette_samples(X, labels, chunk_size=3),
                    reference.silhouette_samples(X, labels), rtol=1e-12, atol=1e-13)
    assert_allclose(silhouette_score(X, labels, sample_size=31, random_state=8, chunk_size=2),
                    reference.silhouette_score(X, labels, sample_size=31, random_state=8),
                    rtol=1e-12, atol=1e-13)


def test_silhouette_hand_calculation_with_singleton():
    X = np.array([[0.], [2.], [10.]])
    actual = silhouette_samples(X, [0, 0, 1], chunk_size=1)
    # Point 0: a=2,b=10. Point2: a=2,b=8. Singleton: convention0.
    assert_allclose(actual, [0.8, 0.75, 0])
    assert_allclose(silhouette_score(X, [0, 0, 1]), (0.8 + 0.75) / 3)


def test_constant_and_perfect_cluster_conventions():
    labels = [0, 0, 1, 1]
    constant = np.zeros((4, 2))
    assert calinski_harabasz_score(constant, labels) == 1
    assert davies_bouldin_score(constant, labels) == 0
    assert_array_equal(silhouette_samples(constant, labels), [0, 0, 0, 0])
    perfect = np.array([[0.], [0.], [10.], [10.]])
    assert calinski_harabasz_score(perfect, labels) == 1
    assert davies_bouldin_score(perfect, labels) == 0
    assert_array_equal(silhouette_samples(perfect, labels), [1, 1, 1, 1])
    coincident_centroids = np.array([[0.], [2.], [0.], [2.]])
    assert davies_bouldin_score(coincident_centroids, labels) == 0
    assert calinski_harabasz_score(coincident_centroids, labels) == 0
    assert_allclose(silhouette_samples(coincident_centroids, labels), -0.5)


def test_silhouette_sampling_is_reproducible_and_inputs_unchanged():
    rng = np.random.RandomState(2)
    X = rng.normal(size=(60, 3))
    labels = np.repeat([0, 1, 2], 20)
    before = X.copy()
    a = silhouette_score(X, labels, sample_size=25, random_state=4)
    b = silhouette_score(X, labels, sample_size=25, random_state=4, chunk_size=1)
    assert_allclose(a, b, atol=1e-14)
    assert_array_equal(X, before)


@pytest.mark.parametrize("left,right,expected", [
    ([], [], 1), ([0], [2], 1), ([0, 0], [1, 1], 1),
    ([0, 1], [5, 6], 1), ([0, 0], [0, 1], 0),
    ([0, 0, 1, 1], [0, 1, 0, 1], -0.5),
    (["a", "a", "b", "b"], [8, 8, 2, 2], 1),
])
def test_ari_trivial_and_categorical_cases(left, right, expected):
    assert adjusted_rand_score(left, right) == expected


def test_ari_many_observed_categories_does_not_need_square_table():
    n = 30_000
    assert adjusted_rand_score(np.arange(n), np.arange(n)[::-1]) == 1


@pytest.mark.parametrize("function", [calinski_harabasz_score, davies_bouldin_score, silhouette_score])
@pytest.mark.parametrize("labels", [[0, 0, 0], [0, 1, 2], [0, 1], [[0], [1], [1]]])
def test_invalid_partitions_rejected(function, labels):
    with pytest.raises(ValueError):
        function([[0.], [1.], [2.]], labels)


@pytest.mark.parametrize("function", [calinski_harabasz_score, davies_bouldin_score, silhouette_score])
def test_nonfinite_data_rejected(function):
    with pytest.raises(ValueError):
        function([[0.], [np.nan], [2.]], [0, 0, 1])


def test_metric_translation_invariance_large_common_offset():
    # Powers of two and integral offsets avoid changing the represented input.
    X = np.array([[0., 1.], [2., 4.], [9., 10.], [11., 7.], [7., 9.]])
    labels = [0, 0, 1, 1, 1]
    moved = X + 2**40
    for function in [calinski_harabasz_score, davies_bouldin_score, silhouette_score]:
        assert_allclose(function(X, labels), function(moved, labels), rtol=1e-13)


def test_pairwise_workspace_chunks_do_not_change_results():
    X = np.random.RandomState(91).normal(size=(70, 3))
    labels = np.arange(70) % 4
    assert_allclose(silhouette_samples(X, labels, chunk_size=1),
                    silhouette_samples(X, labels, chunk_size=1024), atol=1e-14)
    assert_allclose(davies_bouldin_score(X, labels, chunk_size=1),
                    davies_bouldin_score(X, labels, chunk_size=1024), atol=1e-14)


def test_partial_coincident_centers_follow_finite_convention():
    X = np.array([[-1.], [1.], [-2.], [2.], [9.], [11.]])
    labels = [0, 0, 1, 1, 2, 2]
    assert_allclose(davies_bouldin_score(X, labels), reference.davies_bouldin_score(X, labels))
