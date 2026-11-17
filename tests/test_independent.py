"""The numerical package must remain usable with scikit-learn unavailable."""
import ast
import os
from pathlib import Path
import subprocess
import sys
import textwrap


SOURCE = Path(__file__).resolve().parents[1] / "src"


def test_all_numerical_apis_without_any_sklearn_import():
    # A fresh process prevents a previously imported sklearn module from hiding
    # a dependency. Blocking the entire namespace is stronger than uninstalling
    # a particular sklearn.cluster entry point.
    program = textwrap.dedent(
        """
        import importlib.abc
        import sys

        attempted = []
        class BlockSklearn(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "sklearn" or fullname.startswith("sklearn."):
                    attempted.append(fullname)
                    raise ImportError("sklearn is deliberately unavailable")

        assert not any(name == "sklearn" or name.startswith("sklearn.") for name in sys.modules)
        sys.meta_path.insert(0, BlockSklearn())
        import numpy as np
        from numpy.testing import assert_allclose, assert_array_equal
        from scipy.sparse import csr_matrix
        import mykmeans as mk

        rng = np.random.RandomState(29)
        x = np.vstack([rng.normal(loc=offset, scale=0.22, size=(24, 4))
                       for offset in (-4, 0, 5)])
        weights = rng.uniform(0.2, 2.0, size=len(x))
        weights[3] = 0
        initial = x[[0, 24, 48]].copy()

        for dtype in (np.float32, np.float64):
            data = x.astype(dtype)
            for algorithm in ("lloyd", "hamerly", "elkan", "exponion"):
                model = mk.KMeans(3, init=initial.astype(dtype), n_init=1,
                                 algorithm=algorithm, n_threads=1, record_history=True)
                model.fit(data, sample_weight=weights)
                residual = data.astype(float)[:, None, :] - model.cluster_centers_.astype(float)[None, :, :]
                squared = np.einsum("nkd,nkd->nk", residual, residual)
                assert_array_equal(model.labels_, squared.argmin(axis=1))
                assert_array_equal(model.labels_, model.predict(data))
                assert_allclose(model.inertia_, weights @ squared.min(axis=1), rtol=2e-6)
                assert_allclose(model.score(data, sample_weight=weights), -model.inertia_, rtol=2e-6)
                assert_allclose(model.transform(data), np.sqrt(squared), rtol=2e-6, atol=1e-6)
                assert model.backend_ == "cython"

            sparse_data = csr_matrix(data)
            for algorithm in ("lloyd", "elkan"):
                model = mk.KMeans(3, init=initial.astype(dtype), n_init=1,
                                 algorithm=algorithm, n_threads=1).fit(sparse_data, sample_weight=weights)
                residual = data.astype(float)[:, None, :] - model.cluster_centers_.astype(float)[None, :, :]
                squared = np.einsum("nkd,nkd->nk", residual, residual)
                assert_array_equal(model.labels_, squared.argmin(axis=1))
                assert_array_equal(model.labels_, model.predict(sparse_data))
                assert_allclose(model.inertia_, weights @ squared.min(axis=1), rtol=2e-6)
                assert_allclose(model.transform_squared(sparse_data), squared, rtol=2e-6, atol=1e-6)
                assert model.backend_ == "cython-csr"

            for input_data in (data, sparse_data):
                seeds, indices = mk.kmeans_plusplus(input_data, 3, sample_weight=weights, random_state=7)
                assert_array_equal(seeds, data[indices])
                assert len(set(indices)) == 3
                result = mk.k_means(input_data, 3, init=seeds, n_init=1, return_n_iter=True)
                assert result[0].shape == (3, 4) and result[1].shape == (len(data),)
                assert np.isfinite(result[2]) and result[3] >= 1

        fitted = mk.KMeans(3, n_init=2, random_state=4, n_threads=1).fit(x, sample_weight=weights)
        labels = fitted.labels_
        centers = fitted.cluster_centers_
        assert mk.adjusted_rand_score(labels, labels) == 1.0
        assert mk.silhouette_samples(x, labels, chunk_size=9).shape == (len(x),)
        assert 0 < mk.silhouette_score(x, labels, sample_size=40, random_state=0) <= 1
        assert mk.calinski_harabasz_score(x, labels) > 0
        assert mk.davies_bouldin_score(x, labels) > 0
        stats = mk.cluster_statistics(x, labels, centers=centers, sample_weight=weights, covariance="full")
        assert_allclose(stats["inertia"], fitted.inertia_)
        assert_allclose(stats["within_ss"] + stats["between_ss"], stats["total_ss"])
        assert mk.objective_gradient(x, centers, labels=labels, sample_weight=weights).shape == (3, 4)
        assert mk.assignment_hessian(labels, 3, 4, sample_weight=weights).shape == (12, 12)
        assert_allclose(mk.soft_assign(x, centers).sum(axis=1), 1)
        assert mk.voronoi_halfspaces(centers)[0].shape == (3, 2, 4)
        assert_allclose(mk.align_centers(centers, centers[::-1])[0], centers)
        assert_array_equal(mk.align_labels(labels, 2 - labels), labels)

        metric = mk.FeatureMetric(feature_weights=[1, 4, 2, 0.5])
        transformed = metric.fit_transform(x)
        assert_allclose(transformed, x * np.sqrt([1, 4, 2, 0.5]))
        assert_allclose(metric.inverse_transform(transformed), x)
        assert metric.get_params()["feature_weights"] == [1, 4, 2, 0.5]
        records = mk.evaluate_k(x, [2, 3], random_state=0, silhouette_sample_size=40,
                                n_init=1, n_threads=1)
        assert [record["k"] for record in records] == [2, 3]
        assert all(np.isfinite(record["silhouette"]) for record in records)
        stability = mk.clustering_stability(x, 3, seeds=[0, 1], n_init=1, n_threads=1)
        assert stability["pairwise_ari"].shape == (2, 2)
        gap = mk.gap_statistic(x, [1, 2], n_refs=2, random_state=0, n_init=1, n_threads=1)
        assert np.isfinite(gap["gap"]).all()
        assert not attempted, attempted
        assert not any(name == "sklearn" or name.startswith("sklearn.") for name in sys.modules)
        print("independent numerical APIs passed")
        """
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(filter(None, [str(SOURCE), environment.get("PYTHONPATH")]))
    environment.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    result = subprocess.run([sys.executable, "-c", program], env=environment,
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "independent numerical APIs passed" in result.stdout


def test_source_allows_only_optional_sklearn_protocol_value_objects():
    allowed = {
        "sklearn.utils": {"Tags", "TargetTags", "TransformerTags", "InputTags"},
        "sklearn.utils.metadata_routing": {"MetadataRequest"},
    }
    for path in (SOURCE / "mykmeans").glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
            if isinstance(node, ast.Import):
                assert not any(name.name == "sklearn" or name.name.startswith("sklearn.")
                               for name in node.names), (path.name, node.lineno)
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("sklearn"):
                assert path.name == "_compat.py", (path.name, node.lineno)
                assert node.module in allowed, (path.name, node.lineno)
                assert {name.name for name in node.names} <= allowed[node.module]
