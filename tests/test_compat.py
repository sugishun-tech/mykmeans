"""Integration with sklearn's public estimator protocol, not its algorithms."""
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest
from sklearn import config_context
from sklearn.base import clone, is_clusterer
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.pipeline import Pipeline

from mykmeans import FeatureMetric, KMeans


@pytest.fixture
def observations():
    rng = np.random.RandomState(14)
    X = np.vstack([rng.normal(-2, 0.4, (20, 3)), rng.normal(3, 0.4, (20, 3))])
    return X, np.linspace(0.2, 2.0, len(X))


def test_clone_resets_fit_preserves_constructor_and_protocol_settings(observations):
    X, weight = observations
    model = KMeans(2, random_state=np.random.RandomState(7), n_init=1, n_threads=1)
    model.set_output(transform="pandas").set_fit_request(sample_weight=True)
    model.set_score_request(sample_weight="validation_weight").fit(X, sample_weight=weight)
    copied = clone(model)
    assert not hasattr(copied, "cluster_centers_")
    assert copied.random_state is not model.random_state
    assert_array_equal(copied.random_state.get_state()[1], model.random_state.get_state()[1])
    assert copied.get_metadata_routing().fit.requests == {"sample_weight": True}
    assert copied.get_metadata_routing().score.requests == {"sample_weight": "validation_weight"}
    assert copied._output_transform == "pandas"
    assert is_clusterer(copied)
    assert all(not cls.__module__.startswith("sklearn") for cls in type(copied).__mro__)


def test_pipeline_nested_parameters_and_grid_search(observations):
    X, weight = observations
    pipeline = Pipeline([
        ("metric", FeatureMetric(feature_weights=[1.0, 4.0, 0.5])),
        ("cluster", KMeans(random_state=0, n_init=1, n_threads=1)),
    ])
    pipeline.set_params(cluster__n_clusters=2)
    assert pipeline.get_params()["cluster__n_clusters"] == 2
    copied = clone(pipeline)
    assert copied["metric"].feature_weights is not pipeline["metric"].feature_weights
    search = GridSearchCV(pipeline, {"cluster__n_clusters": [1, 2]},
                          cv=KFold(2, shuffle=True, random_state=0), error_score="raise")
    search.fit(X, cluster__sample_weight=weight)
    assert search.best_params_["cluster__n_clusters"] == 2
    best = search.best_estimator_
    assert_allclose(best.score(X), best["cluster"].score(best["metric"].transform(X)))
    assert best.predict(X).shape == (len(X),)
    assert_array_equal(best.get_feature_names_out(), ["kmeans0", "kmeans1"])


@pytest.mark.parametrize("input_kind", ["array", "list", "frame"])
def test_kmeans_set_output_pandas_preserves_index_and_distance_values(observations, input_kind):
    pd = pytest.importorskip("pandas")
    X, _ = observations
    if input_kind == "list":
        data = X.tolist()
        index = pd.RangeIndex(len(X))
    elif input_kind == "frame":
        index = pd.Index([f"row-{j}" for j in range(len(X))], name="observation")
        data = pd.DataFrame(X, columns=["x", "y", "z"], index=index)
    else:
        data = X
        index = pd.RangeIndex(len(X))
    model = KMeans(2, random_state=0, n_threads=1).set_output(transform="pandas")
    output = model.fit_transform(data)
    assert isinstance(output, pd.DataFrame)
    assert output.index.equals(index)
    assert list(output.columns) == ["kmeans0", "kmeans1"]
    expected = np.linalg.norm(X[:, None, :] - model.cluster_centers_[None, :, :], axis=2)
    assert_allclose(output.to_numpy(), expected)
    model.set_output(transform="default")
    assert isinstance(model.transform(data), np.ndarray)


def test_pipeline_set_output_and_feature_name_validation(observations):
    pd = pytest.importorskip("pandas")
    X, _ = observations
    frame = pd.DataFrame(X, columns=["first", "second", "third"], index=np.arange(len(X)) + 90)
    pipeline = Pipeline([
        ("metric", FeatureMetric(feature_weights=[1, 2, 3])),
        ("cluster", KMeans(2, random_state=0, n_threads=1)),
    ]).set_output(transform="pandas")
    output = pipeline.fit_transform(frame)
    assert output.index.equals(frame.index)
    assert_array_equal(pipeline.feature_names_in_, frame.columns)
    assert_array_equal(pipeline["cluster"].feature_names_in_,
                       ["featuremetric0", "featuremetric1", "featuremetric2"])
    with pytest.raises(ValueError, match="feature names"):
        pipeline.predict(frame[["second", "first", "third"]])
    with pytest.raises(ValueError, match="feature names"):
        pipeline.predict(frame.rename(columns={"first": "other"}))
    with pytest.warns(UserWarning, match="feature names"):
        pipeline.predict(X)
    pipeline["metric"].fit(X)
    assert not hasattr(pipeline["metric"], "feature_names_in_")


def test_metadata_routing_passes_weight_to_pipeline_fit_and_score(observations):
    X, weight = observations
    with config_context(enable_metadata_routing=True):
        model = KMeans(2, n_init=1, random_state=4, n_threads=1)
        model.set_fit_request(sample_weight=True).set_score_request(sample_weight=True)
        pipeline = Pipeline([("metric", FeatureMetric(feature_weights=[1, 4, 2])),
                             ("cluster", model)])
        pipeline.fit(X, sample_weight=weight)
        transformed = pipeline["metric"].transform(X)
        explicit = KMeans(2, n_init=1, random_state=4, n_threads=1).fit(transformed, sample_weight=weight)
        assert_allclose(model.cluster_centers_, explicit.cluster_centers_)
        assert_allclose(pipeline.score(X, sample_weight=weight), -explicit.inertia_)


def test_metadata_alias_and_grid_search_routing(observations):
    X, weight = observations
    with config_context(enable_metadata_routing=True):
        model = KMeans(n_init=1, random_state=0, n_threads=1)
        model.set_fit_request(sample_weight="importance").set_score_request(sample_weight="importance")
        search = GridSearchCV(model, {"n_clusters": [1, 2]},
                              cv=KFold(2, shuffle=True, random_state=0), error_score="raise")
        search.fit(X, importance=weight)
        assert search.best_params_["n_clusters"] == 2
        # Older sklearn releases do not forward aliases from the default
        # GridSearchCV scorer. The alias must still reach every fit and refit.
        explicit = clone(model).set_params(n_clusters=2).fit(X, sample_weight=weight)
        assert_allclose(search.best_estimator_.cluster_centers_, explicit.cluster_centers_)
        assert_allclose(search.best_estimator_.score(X, sample_weight=weight), -explicit.inertia_)


def test_grid_search_pipeline_routes_fit_and_score_weights(observations):
    X, weight = observations
    with config_context(enable_metadata_routing=True):
        model = KMeans(n_init=1, random_state=0, n_threads=1)
        model.set_fit_request(sample_weight=True).set_score_request(sample_weight=True)
        pipeline = Pipeline([("metric", FeatureMetric()), ("cluster", model)])
        pipeline.set_score_request(sample_weight=True)
        search = GridSearchCV(pipeline, {"cluster__n_clusters": [1, 2]},
                              cv=KFold(2, shuffle=True, random_state=0), error_score="raise")
        search.fit(X, sample_weight=weight)
        assert search.best_params_["cluster__n_clusters"] == 2
        assert_allclose(search.score(X, sample_weight=weight), -search.best_estimator_["cluster"].inertia_)


def test_metadata_not_requested_raises_instead_of_silently_dropping(observations):
    from sklearn.exceptions import UnsetMetadataPassedError
    X, weight = observations
    with config_context(enable_metadata_routing=True):
        pipeline = Pipeline([("metric", FeatureMetric()),
                             ("cluster", KMeans(2, random_state=0, n_threads=1))])
        with pytest.raises(UnsetMetadataPassedError):
            pipeline.fit(X, sample_weight=weight)
