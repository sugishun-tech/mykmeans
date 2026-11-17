# API reference

Version 0.2.0 implements the numerical algorithms and estimator foundation
independently. NumPy, SciPy, and threadpoolctl are the runtime dependencies.
Scikit-learn is optional for its surrounding ecosystem; no fitting, seeding,
prediction, metric, or validation operation delegates to scikit-learn.

## Constructor and input contract

The standard arguments follow the public scikit-learn KMeans API:

```text
KMeans(
    n_clusters=8, *, init="k-means++", n_init="auto", max_iter=300,
    tol=1e-4, verbose=0, random_state=None, copy_x=True,
    algorithm="lloyd", n_threads=None, record_history=False,
)
```

`fit(X, y=None, sample_weight=None)` accepts dense float32/float64 matrices,
numeric array-likes, and SciPy sparse matrices. Other sparse formats are converted
to CSR; the full input is not densified. Centers are always dense arrays.
X must be finite, two-dimensional, and have at least as many rows as requested
clusters. Sample weights are nonnegative finite scalars or vectors with positive
finite total weight. The objective is weighted squared Euclidean distance.
Changing geometry requires an explicit `FeatureMetric` transform; the ordinary
arithmetic-mean update does not implement an arbitrary non-Euclidean metric.

| `algorithm` | Dense input | CSR input |
| --- | --- | --- |
| `"lloyd"` | Independent direct / blocked-BLAS Lloyd | Independent CSR Lloyd |
| `"hamerly"` | Independent Hamerly bounds | `ValueError` |
| `"elkan"` | Independent per-center Elkan bounds | Independent CSR Elkan bounds |
| `"exponion"` | Independent Exponion ball pruning with fully sorted neighbor rows | `ValueError` |
| `"auto"` | Hamerly when `d <= 64` and `k > 1`; otherwise Lloyd | CSR Lloyd |

`auto` is a fixed heuristic, not an optimizer over implementations. Exponion
uses the candidate-ball result from Newling and Fleuret, but replaces their
partial-sort strategy with full neighbor sorting. See [sota.md](sota.md) for
research references and the exact/approximate distinction.

## Standard operations and optional integration

`fit`, `fit_predict`, `fit_transform`, `predict`, `transform`, `score`,
`get_params`, `set_params`, and `get_feature_names_out` are supported.
Parameter handling, validation, feature-name checks, cloning hooks, and output
wrapping are implemented in this package's independent `_compat` foundation.

`transform` returns Euclidean distances in the fitted center dtype; `score`
returns negative weighted inertia. As in scikit-learn, y is accepted and ignored.
`random_state` accepts None, an integer, or NumPy RandomState; NumPy Generator is
not part of this sklearn-style random-state contract.

To use `sklearn.base.clone`, Pipeline, GridSearchCV, and metadata routing, install
the optional `sklearn` extra. The package provides the public protocol hooks and
creates scikit-learn tag/metadata-request value objects when that integration
is requested. It does not inherit scikit-learn's numerical estimator code.
Enable metadata routing in scikit-learn when using requests such as
`set_fit_request(sample_weight=True)` or `set_score_request(sample_weight=True)`
with routing-aware callers. Direct `fit(..., sample_weight=...)` works without
scikit-learn or routing configuration.

`set_output(transform="default")` returns NumPy arrays.
`set_output(transform="pandas")` and `"polars"` require those respective optional
packages and wrap the standard `transform` / `fit_transform` output. Extra
operations such as `transform_squared` return NumPy arrays directly.

Interface compatibility does not promise bitwise-equivalent results or that
every scikit-learn estimator check passes. In particular, independent seeding
can select a different center order for integer sample weights and explicitly
repeated observations. Consult [compatibility.md](compatibility.md) for observed
label-order differences and the checks used to compare partitions and objectives.

## Initialization and functional APIs

```text
kmeans_plusplus(X, n_clusters, *, sample_weight=None,
                x_squared_norms=None, random_state=None, n_local_trials=None)

k_means(X, n_clusters, *, sample_weight=None, init="k-means++",
        n_init="auto", max_iter=300, verbose=False, tol=1e-4,
        random_state=None, copy_x=True, algorithm="lloyd", return_n_iter=False)
```

`kmeans_plusplus` returns `(centers, indices)` and independently implements
weighted greedy D-squared seeding for both dense and CSR input. The default
number of local trials is `2 + int(log(n_clusters))`; `n_local_trials=1` selects
one candidate per step. Optional `x_squared_norms` is validated but not used as
the authority for distances. Native distance computations determine the result.

For dense seeding, a triangle-inequality filter may skip a candidate distance
that cannot improve the existing nearest-center distance. It preserves the
greedy candidates, potentials, and sampling rule. A small deterministic probe
set selects whether filtering is likely to be worthwhile; it does not subsample
the fitted dataset or approximate any sampling potential. The current gate
requires `d >= 8`, `n >= 1024`, and `k >= 8`, with up to 256 probes. Its estimated
pruning threshold is 60% for `d >= 64`, and 75% otherwise. These are internal
performance heuristics, not additional constructor parameters.

`init` also accepts `"random"`, a dense center array, or a callable with signature
`init(X, n_clusters, random_state=...)`. A callable receives the centered working
dense matrix, or the uncentered sparse matrix. It must return a dense `(k, d)`
array. `n_init="auto"` uses one start for k-means++ or explicit centers and ten
for random/callable initialization. Explicit centers force one start even if a
larger `n_init` is requested. Each restart is ranked by its finalized objective.

`k_means` returns `(centers, labels, inertia)`, with `n_iter` appended when
requested. Use the estimator to control threads or request history. An equal
random seed does not promise identical seeds or floating-point results to
another library's independently implemented initialization.

## Extra operations

| Method | Return / semantics |
| --- | --- |
| `predict_distances(X)` | float64 vector of nearest-center squared distances, dense or CSR input |
| `transform_squared(X)` | float64 dense matrix of all squared distances, dense or CSR input |
| `diagnostics(X, sample_weight=None, covariance="none")` | Weighted statistics; assigns supplied dense X to current centers; rejects sparse X |
| `soft_assign(X, temperature=1.0)` | Dense Gibbs assignment matrix based on squared distances; accepts dense or CSR X |
| `voronoi()` | `(A, b)` for all closed cells; use standalone `voronoi_halfspaces` for one cell |

Research functions are available at the package root and in `analysis` /
`geometry`: `cluster_statistics`, `objective_gradient`, `assignment_hessian`,
`align_centers`, `align_labels`, `soft_assign`, `evaluate_k`,
`clustering_stability`, `gap_statistic`, `FeatureMetric`, and
`voronoi_halfspaces`. See [mathematics.md](mathematics.md) for signatures,
equations, computational costs, and returns.

## Independent evaluation metrics

The following functions are exported from `mykmeans`. They use their own
formulas and aggregation logic, with NumPy operations and SciPy's generic
Euclidean distance primitive. They do not call scikit-learn metrics.

```text
adjusted_rand_score(labels_true, labels_pred)
silhouette_samples(X, labels, *, chunk_size=256)
silhouette_score(X, labels, *, sample_size=None, random_state=None, chunk_size=256)
calinski_harabasz_score(X, labels)
davies_bouldin_score(X, labels, *, chunk_size=256)
```

All five are unweighted; fit weights are not implicitly reused. ARI accepts
categorical label vectors and is invariant to label renaming. The other four
accept dense coordinates and require between two and `n_samples - 1` observed
clusters. Their metric is Euclidean; arbitrary `metric=` arguments and precomputed
distance matrices are outside this API.

Silhouette uses blocked pairwise distances to limit workspace, but full-data
arithmetic remains `O(n²d)`. `sample_size` selects a sub-dataset without replacement
and computes the complete silhouette within that sub-dataset. It does not average
selected points' full-data silhouettes. Single-member clusters and zero-distance
degeneracies receive zero silhouette. `chunk_size` bounds the requested block
size; the implementation further caps its primary distance array to about
64 MiB when one row fits within that budget. This is not a bound on total process
memory. A single row still needs `O(n)` distances for silhouette or `O(k)` for
Davies–Bouldin.

`evaluate_k` uses these independent metrics and reports its nonweighted diagnostic
convention even when fitting is weighted. `clustering_stability` uses independent
ARI. See [mathematics.md](mathematics.md) for the mathematical conventions.

## Fitted attributes

Standard attributes: `cluster_centers_`, `labels_`, `inertia_`, `n_iter_`,
`n_features_in_`, and `feature_names_in_` when all input feature names are strings.

| Additional attribute | Meaning |
| --- | --- |
| `backend_` | `cython` for dense fitting, `cython-csr` for CSR fitting |
| `algorithm_` | Resolved algorithm after dispatch |
| `assignment_engine_` | `direct`, `gemm`, `hamerly`, `elkan`, `exponion`, `csr-direct`, or `csr-elkan` |
| `n_threads_` | Effective native worker count for the winning fit |
| `openmp_enabled_` | Whether the compiled extension supports OpenMP |
| `n_init_` | Number of starts, accounting for explicit-array initialization |
| `best_init_` | Zero-based winning restart |
| `restart_inertias_` | Finalized original-coordinate objective from every restart |
| `cluster_sizes_` | Unweighted number of samples in each final label |
| `cluster_weight_` | Sum of sample weights in each final label |
| `tol_` | Variance-scaled squared center-shift tolerance |
| `converged_` | Native stopping criterion satisfied; does not mean global optimality |
| `convergence_reason_` | `labels`, `tolerance`, or `max_iter` |
| `history_` | Winning run trace when requested; otherwise an empty list |
| `distance_evaluations_` | Assignment-distance counter, excluding seeding and center-center work |
| `center_quantization_` | Largest dense-center coordinate change caused by adding/removing the centering offset; zero for CSR |
| `center_quantization_corrected_` | Whether dense offset round-trip quantization was nonzero; false for CSR |

History and counters are available for all implemented fitting algorithms.
History's `objective_fixed_assignment` is evaluated after each center update
with that iteration's fixed labels; dense fitting evaluates it in centered
coordinates. It can differ from final `inertia_` because of reassignment and
output-coordinate rounding. The trace's `center_shift` is the Frobenius norm,
while the stopping threshold compares its square to `tol_`.

`restart_inertias_` and `inertia_` use the actual stored output centers, and each
restart receives a final assignment before selection. The distance counter
counts point-center evaluations, not wall-clock work: center-center distances,
neighbor sorting, bound maintenance, and optional history aggregation are not
included. Ordinary floating-point roundoff remains.

## Stopping, ties, and empty clusters

The engine stops on unchanged labels or a sufficiently small squared Frobenius
center shift, subject to `max_iter`. A final assignment uses original-coordinate
data and the stored output centers on every restart, including early stops and
near-boundary rounding cases. Exact distance ties select the smallest center
index; numerical near-ties can depend on dtype and build.

An empty positive-mass cluster can receive a positive-weight point with a large
weighted residual, provided its donor retains positive mass. If the data cannot
support k distinct clusters, repeated centers are permitted and a convergence
warning reports fewer observed clusters. The relocation policy is not promised
to match another library on degenerate inputs.

## Memory, threading, and reproducibility

Let n be the number of observations, d the feature count, k the cluster count,
and T the worker count. Input storage and optional input copies are additional
to the following workspace costs:

- Dense Lloyd uses blocked BLAS workspace instead of retaining an n-by-k distance
  matrix. Hamerly additionally stores O(n) bounds.
- Dense and CSR Elkan allocate an n-by-k float64 lower-bound matrix: `8*n*k`
  bytes, plus O(k²+n) geometry/bound storage. Sparse input does not remove this
  Elkan memory cost.
- Exponion uses O(n+k²) bounds and neighbor data, and O(k² log k) full sorting per
  initialized iteration. Sorting overhead is one reason it is not uniformly
  faster than Hamerly or Lloyd.
- Both dense and CSR fitting store dense centers and worker-local weighted
  aggregates with O(Tkd) memory. CSR input rows remain sparse.
- `transform`, `transform_squared`, and `soft_assign` allocate their requested
  dense n-by-k outputs. Diagnostics such as full covariance have their own costs.

`n_threads=None` and `-1` use the current OpenMP limit with a cap for small
datasets. Positive values are upper bounds; the core avoids parallel overhead
for small work units. Use explicit counts and a fixed build when comparing
timings. Reductions can differ in their last bits across thread counts.

For dense input, `copy_x=False` can reduce input-copy memory when the supplied
array is writable, contiguous, and has a supported floating dtype. Its final
assignment restores one row per worker into a bounded buffer and uses direct
distances; this may be slower than the normal final BLAS pass at high dimension.
Converted or read-only inputs may still need a copy. CSR fitting does not center
the input, though format/dtype conversion and canonicalization can allocate
storage.

The engine suppresses nested BLAS parallelism. Threadpool controls operate at
the numerical-library process level; avoid concurrently changing global BLAS
limits from unrelated Python threads. Multiple process workers with one inner
thread are a useful configuration to measure for external parallel searches.

The geometric pruning rules preserve Euclidean assignments in exact arithmetic.
The implementation adds conservative floating-point slack and directly evaluates
ambiguous candidates. This is not an interval-arithmetic proof of every rounding
behavior on every compiler or BLAS implementation.

## Scope and evidence

This package implements CPU batch Euclidean k-means. GPU, distributed training,
streaming/`partial_fit`, arbitrary distance metrics, constrained/fair clustering,
and guarantees of a globally optimal partition are outside this release.
See [compatibility.md](compatibility.md), [sota.md](sota.md), and the
[benchmark report](../benchmarks/README.md) for behavior and measured performance.
Source builds are required; no prebuilt wheel distribution is assumed. CI jobs
are defined for multiple platforms; consult [validation.md](validation.md) for
the environments and commands actually executed.
