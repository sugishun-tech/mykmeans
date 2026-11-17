# Compatibility and implementation independence

`mykmeans.KMeans` follows the public constructor and estimator interface of
`sklearn.cluster.KMeans`. Clustering, initialization, metrics, validation, and
estimator parameter handling are implemented in this repository. Installing or
importing scikit-learn is unnecessary for numerical use of `mykmeans`.

The standard constructor defaults include `n_init="auto"` and
`algorithm="lloyd"`. Additional options are `n_threads`, `record_history`, and
the algorithm values `"hamerly"`, `"exponion"`, and `"auto"`.

## Public interface

| Interface | Behavior |
| --- | --- |
| `fit(X, y=None, sample_weight=None)` | Minimize the weighted squared Euclidean objective; return the estimator |
| `fit_predict` | Fit and return training labels |
| `predict` | Assign rows to the nearest stored center; the smallest index wins exact ties |
| `transform` / `fit_transform` | Return Euclidean distances to every center |
| `score(X, y=None, sample_weight=None)` | Return negative weighted inertia on the supplied data |
| `get_params` / `set_params` | Read and set constructor parameters using the estimator protocol |
| `get_feature_names_out` | Return one name per center, such as `kmeans0` |
| `set_output(transform="pandas")` | Return distances in a DataFrame and preserve an input DataFrame's index |
| `set_fit_request` / `set_score_request` | Declare `sample_weight` metadata for optional sklearn composition |
| `cluster_centers_`, `labels_`, `inertia_`, `n_iter_` | Standard fitted attributes |
| `n_features_in_`, `feature_names_in_` | Validate feature counts and named-column order |
| `k_means`, `kmeans_plusplus` | Functional fitting and initialization interfaces |

`n_init="auto"` performs ten restarts for random or callable initialization and
one for k-means++ or explicit centers. An explicit center array always uses one
restart. An initializer callable receives `(X, n_clusters, random_state=...)`.
The random-state interface accepts an integer, a NumPy `RandomState`, or `None`.

Extensions such as `transform_squared`, `predict_distances`, `diagnostics`,
`soft_assign`, and `voronoi` have their own documented contracts. In particular,
`transform_squared` returns a float64 NumPy array; `set_output` applies to the
standard `transform`/`fit_transform` methods.

## Native execution paths

| Input / algorithm | Implementation | Fitted `backend_` |
| --- | --- | --- |
| Dense float32 / float64, Lloyd | Independent Cython assignment and centroid update, with blocked BLAS assignment when selected | `"cython"` |
| Dense float32 / float64, Hamerly | Independent Cython triangle-inequality bounds | `"cython"` |
| Dense float32 / float64, Elkan | Independent Cython point-to-center lower bounds | `"cython"` |
| Dense float32 / float64, Exponion | Independent Cython Hamerly bounds plus center-neighborhood pruning | `"cython"` |
| CSR float32 / float64, Lloyd | Independent Cython sparse assignment and centroid update | `"cython-csr"` |
| CSR float32 / float64, Elkan | Independent Cython sparse distances and Elkan bounds | `"cython-csr"` |
| Dense, `algorithm="auto"` | A conservative native algorithm choice, exposed as `algorithm_` | `"cython"` |
| CSR, `algorithm="auto"` | Native sparse Lloyd | `"cython-csr"` |

Other supported SciPy sparse formats are converted to CSR. Sparse Hamerly and
Exponion are rejected with a clear error. Sparse observations remain sparse;
centers are dense `k × d` arrays, and Elkan additionally requires `n × k`
lower-bound storage. These allocation costs still matter for very large
feature or cluster counts. The input's sparse structure is checked before
unchecked native kernels consume its index buffers.

Initialization, prediction, transformation, and scoring also have independent
dense and sparse implementations. There is no sklearn fallback for Elkan,
CSR, k-means++, or a difficult input. `backend_`, `algorithm_`, and
`assignment_engine_` expose the actual path rather than silently changing the
implementation provider.

NumPy, SciPy, and threadpoolctl remain runtime dependencies. They provide array
operations, BLAS, generic distances, eigendecomposition/SVD, linear assignment,
sparse containers, and thread-pool control. None supplies a clustering algorithm
to `mykmeans`.

## Optional sklearn integration

The estimator classes inherit this repository's own protocol classes, not
`sklearn.base.BaseEstimator`. The optional dependency can be installed with:

```bash
python -m pip install ".[sklearn]"
```

Integration supports `sklearn.base.clone`, `Pipeline`, `GridSearchCV`, nested
pipeline parameters, feature names, estimator tags, and sample-weight metadata
routing. `FeatureMetric` is also a compatible transformer. The package lazily
imports sklearn **protocol value objects only** when a caller asks for tags or
metadata routing. It never calls sklearn's clustering, initialization, metrics,
random-state helpers, or input validators. Exception identities are reused when
the host has already loaded the corresponding sklearn exception module.

For weighted pipelines with metadata routing enabled:

```python
from sklearn import config_context
from sklearn.pipeline import Pipeline
from mykmeans import FeatureMetric, KMeans

with config_context(enable_metadata_routing=True):
    cluster = KMeans(3, random_state=0)
    cluster.set_fit_request(sample_weight=True)
    cluster.set_score_request(sample_weight=True)
    pipeline = Pipeline([
        ("metric", FeatureMetric()),
        ("cluster", cluster),
    ])
    # Required by older sklearn versions when GridSearchCV routes to score.
    pipeline.set_score_request(sample_weight=True)
    pipeline.fit(X, sample_weight=weights)
    weighted_score = pipeline.score(X, sample_weight=weights)
```

With metadata routing disabled, use ordinary pipeline arguments such as
`pipeline.fit(X, cluster__sample_weight=weights)`. The compatibility tests also
exercise request aliases when fitting through `GridSearchCV`. Metadata support
in the surrounding sklearn objects varies across releases: in particular, old
default GridSearchCV scorers do not forward every alias. Using the standard
`sample_weight` name and explicitly setting both fit and score requests is the
tested portable path.

DataFrame output requires pandas. Array and Python-list inputs receive an
ordinary row index; DataFrame inputs keep their index. `Pipeline.set_output`
propagates the per-estimator configuration. `set_output(transform="polars")`
is available when polars is installed, but the delivered integration tests
exercise pandas. Global sklearn output configuration and Array API/GPU tensors
are outside the compatibility contract.

## Numerical semantics

- Cluster IDs are arbitrary. Compare partitions with adjusted Rand index, or
  align centers and labels before comparing implementations. Distance-matrix
  columns must be permuted together with their corresponding centers.
- Native kernels accumulate distances and centroid sums in float64 even for
  float32 input. Stored centers retain the input floating dtype. Reductions,
  distance formulas, initialization, and ties can differ from sklearn. Bitwise
  equality is not promised.
- A fixed random state is reproducible for a fixed build and thread count.
  Identical integer seeds do not promise sklearn's k-means++ seed sequence or
  identical floating reductions across thread counts.
- Integer sample weights and repeated observations define the same objective.
  Tests compare that equivalence with identical explicit initial centers.
  Independently randomized starts after repeating, deleting zero-weight rows,
  or reordering rows can choose different center orders or local solutions.
  Exact cluster-ID equality under those changes is outside this contract.
- `tol` or `max_iter` can stop before centers equal the means of final labels.
  Each restart finishes by assigning original-coordinate rows to the actual
  stored centers, including floating rounding of those centers. This keeps
  `labels_` consistent with `predict(X)` and uses the same final objective to
  select the winning restart.
- Zero sample weights are supported. Negative or nonfinite weights and a total
  weight of zero are rejected. Weighted fitting does not turn the standard
  silhouette, Calinski-Harabasz, Davies-Bouldin, or adjusted Rand diagnostics
  into weighted variants; those are explicitly unweighted.
- `copy_x=False` permits temporary centering of writable contiguous dense input
  and restoration afterward. Subtraction/addition can introduce rounding.
  Other layouts or read-only inputs may require copying.
- Squared Euclidean k-means supports positive semidefinite quadratic metrics
  through `FeatureMetric`. Arbitrary distances are not accepted as substitutes
  for the squared Euclidean objective and mean update.

## Verification scope

`tests/test_independent.py` starts a clean interpreter, blocks every import in
the `sklearn` namespace, and exercises all four dense algorithms, both sparse
algorithms, dense/sparse k-means++, functional fitting, independent metrics,
`FeatureMetric`, weighted statistics, derivatives, Voronoi halfspaces,
`evaluate_k`, initialization stability, and the gap statistic. It also audits
package imports so only the optional protocol objects described above are
allowed. These tests verify independence as executed, not merely an optional
dependency declaration in package metadata.

`tests/test_compat.py` checks clone behavior, nested pipeline parameters,
GridSearchCV, DataFrame/list outputs, feature-name validation, and metadata
routing. These targeted integration tests were run against sklearn 1.6.1 and
1.9.1 on Python 3.12. Mathematical metrics are tested independently and
differentially against both versions.

The broader tests cover objective calculations, early stopping, integer weight
replication with fixed initial centers, empty clusters, duplicate observations,
zero-weight outliers, sparse formats, serialization, and thread consistency.
Generic sklearn estimator checks additionally exercise the estimator protocol.
Their randomized sample-weight equivalence checks compare raw cluster IDs;
that stricter seed-and-label convention is not a promised invariant here.
In the two such sklearn 1.9.1 checks (dense and sparse), both fits had inertia
`2.9098625181736644` and adjusted Rand index `1.0`; aligning the center order
made the predicted labels identical. The aligned-center difference was at most
`1.12e-16`. The package does not claim an unconditional pass of every generic
check, even when the underlying partition and objective agree.

Run from a development checkout:

```bash
python -m pip install -e ".[test]"
python -m pytest -q
```

Benchmark reports state their hardware, thread counts, input shapes, measured
iterations, objectives, and actual execution engines. Common initial centers
isolate iteration performance; k-means++ comparisons include initialization and
can converge to different local minima. Always assess objective quality with
runtime. Repository CI workflows specify additional configurations, but their
presence does not establish that remote CI jobs have already executed.

## Primary interface references

- [scikit-learn: KMeans API](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)
- [scikit-learn: Developing estimators](https://scikit-learn.org/stable/developers/develop.html)
- [scikit-learn: Metadata routing](https://scikit-learn.org/stable/metadata_routing.html)
