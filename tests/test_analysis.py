"""Independent identities and edge cases for the opt-in mathematical helpers."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score

from mykmeans.analysis import (
    align_centers, align_labels, assignment_hessian, cluster_statistics,
    clustering_stability, evaluate_k, gap_statistic, objective_gradient,
    soft_assign,
)


def finite_difference(function, x, step=1e-5):
    """Independent central differences; no differentiation package required."""
    columns = []
    for j in range(len(x)):
        delta = np.zeros_like(x)
        delta[j] = step
        columns.append((np.atleast_1d(function(x + delta)) - np.atleast_1d(function(x - delta))) / (2 * step))
    return np.column_stack(columns)


@pytest.fixture
def sample():
    X = np.array([[0., 1.], [2., 3.], [8., 2.], [10., 5.], [99., -7.]])
    labels = np.array([0, 0, 1, 1, 2])
    weights = np.array([1., 3., 2., 2., 0.])
    return X, labels, weights


def test_statistics_weighted_anova_at_unconverged_centers(sample):
    X, labels, weights = sample
    centers = np.array([[-2., 0.], [14., 8.], [3., 3.]])
    s = cluster_statistics(X, labels, centers=centers, sample_weight=weights, covariance="full")
    expected = np.sum(weights[:, None] * (X - centers[labels]) ** 2)
    assert_allclose(s["inertia"], expected)
    assert_allclose(s["total_ss"], s["within_ss"] + s["between_ss"])
    assert_allclose(s["inertia"], s["within_ss"] + s["center_offset_ss"])
    assert s["center_offset_ss"] > 0
    assert_allclose(s["means"][:2], [[1.5, 2.5], [9., 3.5]])
    assert_array_equal(s["counts"], [2, 2, 1])
    assert_array_equal(s["weight_sums"], [4, 4, 0])
    assert np.isnan(s["means"][2]).all()
    assert np.isnan(s["covariances"][2]).all()
    assert np.isnan(s["radii"][2])
    assert s["inertia_per_cluster"][2] == 0


def test_statistics_matches_replication(sample):
    X, labels, weights = sample
    weighted = cluster_statistics(X, labels, sample_weight=weights, n_clusters=4, covariance="full")
    replicated = cluster_statistics(
        np.repeat(X, weights.astype(int), axis=0),
        np.repeat(labels, weights.astype(int)), n_clusters=4, covariance="full",
    )
    for key in ["means", "covariances", "inertia_per_cluster", "within_ss", "between_ss", "total_ss"]:
        assert_allclose(weighted[key], replicated[key], equal_nan=True)


def test_statistics_diag_and_none(sample):
    X, labels, w = sample
    full = cluster_statistics(X, labels, sample_weight=w, covariance="full")
    diag = cluster_statistics(X, labels, sample_weight=w, covariance="diag")
    assert_allclose(diag["covariances"], np.diagonal(full["covariances"], axis1=1, axis2=2), equal_nan=True)
    assert cluster_statistics(X, labels, covariance=None)["covariances"] is None
    assert cluster_statistics(X, labels, covariance="none")["covariances"] is None


def test_anova_translation_stability():
    X = 1e12 + np.array([[0., 0.], [1., 2.], [6., 4.], [9., 2.]])
    labels = np.array([0, 0, 1, 1])
    s = cluster_statistics(X, labels, sample_weight=[1, 2, 3, 4])
    assert_allclose(s["total_ss"], s["within_ss"] + s["between_ss"], rtol=1e-12)


def test_constant_data_has_undefined_variance_ratio():
    s = cluster_statistics(np.ones((5, 2)), np.zeros(5, dtype=int))
    assert s["inertia"] == 0
    assert s["total_ss"] == 0
    assert np.isnan(s["explained_variance_ratio"])


def test_zero_weight_extreme_point_has_no_effect():
    # Subtraction overflows for the unused row, so it must be excluded before
    # computing global moments rather than multiplied by zero afterwards.
    X = np.array([[1e308], [-1e308]])
    s = cluster_statistics(X, np.array([0, 1]), sample_weight=[1, 0])
    assert s["inertia"] == 0
    assert s["total_ss"] == 0
    assert_array_equal(s["global_mean"], [1e308])
    assert np.isnan(s["means"][1]).all()
    assert_array_equal(objective_gradient(X, [[1e308]], sample_weight=[1, 0]), [[0]])
    assert_array_equal(objective_gradient(X, [[1e308]], labels=[0, 0], sample_weight=[1, 0]), [[0]])


def test_unrepresentable_moments_raise_clear_error():
    with pytest.raises(ValueError, match="overflowed"):
        cluster_statistics([[1e308], [-1e308]], np.array([0, 1]))
    with pytest.raises(ValueError, match="overflowed"):
        cluster_statistics([[1e308], [1e308]], np.array([0, 0]), centers=[[-1e308]])


def test_fixed_gradient_and_hessian_match_finite_differences(sample):
    X, labels, w = sample
    C = np.array([[1., 2.], [9., 4.], [100., -10.]])

    def objective(flat):
        return np.sum(w[:, None] * (X - flat.reshape(C.shape)[labels]) ** 2)

    expected_gradient = finite_difference(objective, C.ravel()).ravel()
    gradient = objective_gradient(X, C, labels=labels, sample_weight=w)
    assert_allclose(gradient.ravel(), expected_gradient, rtol=1e-7, atol=1e-7)

    def grad(flat):
        return objective_gradient(X, flat.reshape(C.shape), labels=labels, sample_weight=w).ravel()

    expected_hessian = finite_difference(grad, C.ravel())
    hessian = assignment_hessian(labels, 3, 2, sample_weight=w)
    assert_allclose(hessian.toarray(), expected_hessian, atol=1e-7)
    assert_allclose(hessian.toarray(), assignment_hessian(labels, 3, 2, sample_weight=w, sparse=False))


def test_nearest_gradient_and_tie_branch():
    X = np.array([[0.], [2.], [9.]])
    C = np.array([[0.], [10.]])
    assert_allclose(objective_gradient(X, C), [[-4.], [2.]])
    assert_allclose(objective_gradient([[5.]], C), [[-10.], [0.]])


def test_align_centers_and_labels():
    A = np.array([[1, 2], [20, 30], [-3, 6.]])
    B = A[[2, 0, 1]]
    aligned, permutation = align_centers(A, B)
    assert_allclose(aligned, A)
    assert_array_equal(permutation, [1, 2, 0])
    reference = np.array([10, 10, 20, 20, 30, 30])
    candidate = np.array([7, 7, 1, 1, 9, 9])
    result = align_labels(reference, candidate)
    assert_array_equal(result, reference)
    assert adjusted_rand_score(result, candidate) == 1
    with pytest.raises(ValueError, match="same number"):
        align_labels(reference, [1, 1, 1, 1, 1, 1])


def test_soft_assign_stability_and_precomputed_equivalence():
    X = np.array([[0.], [1.], [1e4]])
    C = np.array([[0.], [2.]])
    squared = (X - C.T) ** 2
    saved = squared.copy()
    expected = soft_assign(X, C, temperature=.01, batch_size=1)
    actual = soft_assign(squared, temperature=.01, batch_size=2)
    assert_allclose(actual, expected)
    assert_allclose(actual.sum(axis=1), 1)
    assert_allclose(actual[1], [.5, .5])
    assert_array_equal(squared, saved)
    tiny = soft_assign([[1e300, 0., 0.]], temperature=1e-300)
    assert_allclose(tiny, [[0, .5, .5]])
    with pytest.raises(ValueError, match="nonnegative"):
        soft_assign([[-1, 2]])


@pytest.mark.parametrize("weights", [[0, 0], [-1, 2], [1, np.nan], [1, np.inf], [1]])
def test_invalid_weights(weights):
    with pytest.raises(ValueError):
        cluster_statistics([[1.], [2.]], np.array([0, 1]), sample_weight=weights)


@pytest.mark.parametrize("temperature", [0, -1, np.inf, np.nan, "bad"])
def test_invalid_temperature(temperature):
    with pytest.raises(ValueError, match="temperature"):
        soft_assign([[0., 1.]], temperature=temperature)


def test_evaluate_k_and_stability():
    X, _ = make_blobs(n_samples=100, centers=[[-6, -6], [6, 6]], cluster_std=.3, random_state=0)
    records = evaluate_k(X, [1, 2, 3], silhouette_sample_size=50, n_init=2)
    assert [row["k"] for row in records] == [1, 2, 3]
    assert np.isnan(records[0]["silhouette"])
    assert records[1]["silhouette"] > .9
    assert records[1]["inertia"] < records[0]["inertia"]
    assert records[1]["diagnostic_weighting"] == "unweighted"
    stability = clustering_stability(X, 2, seeds=[0, 1, 2], n_init=1, return_labels=True)
    assert_allclose(stability["pairwise_ari"], 1)
    assert stability["labels"].shape == (3, 100)


def test_gap_reference_and_standard_error():
    X, _ = make_blobs(n_samples=50, centers=2, n_features=2, random_state=2)
    result = gap_statistic(X, [1, 2, 3], n_refs=3, reference="axis", random_state=4, n_init=2)
    assert np.isfinite(result["gap"]).all()
    assert result["chosen_k"] in [1, 2, 3]
    assert_allclose(result["standard_error"], result["reference_log_inertia"].std(axis=0, ddof=1) * np.sqrt(1 + 1 / 3))
    repeated = gap_statistic(X, [1, 2, 3], n_refs=3, reference="axis", random_state=4, n_init=2)
    assert_allclose(result["gap"], repeated["gap"])
    pca = gap_statistic(X, [1, 2], n_refs=2, reference="pca", random_state=4)
    assert np.isfinite(pca["gap"]).all()


def test_gap_zero_dispersion_rejected():
    with pytest.raises(ValueError, match="positive finite inertia"):
        gap_statistic(np.ones((8, 2)), [1], n_refs=2)
