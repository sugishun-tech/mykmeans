# Mathematical diagnostics and geometry

The fitting engine minimizes a weighted squared Euclidean objective. The
research helpers below are explicit operations: covariances, model selection,
soft assignments, and geometric constructions do not run during ordinary fit.

All helpers in `mykmeans.analysis` and `mykmeans.geometry` accept dense input.
They calculate in float64, independently of the dtype used by the native fitting
engine. Observation weights must be nonnegative, finite, and have a positive
finite sum. Feature weights and additive power weights follow their separate
definitions below.
Zero-weight observations are excluded before moment arithmetic, so even a very
large ignored coordinate cannot create a spurious zero-times-infinity result.
Most diagnostics need additional arrays and should be used after fitting.

## Objective and weighted moments

For points $x_i$, assignments $z_i$, centers $c_j$, and sample weights $w_i$,

$$
J(C,z)=\sum_i w_i\lVert x_i-c_{z_i}\rVert^2.
$$

Define cluster masses $m_j=\sum_{i:z_i=j}w_i$, empirical weighted cluster means
$\mu_j=m_j^{-1}\sum_{i:z_i=j}w_ix_i$, and the global weighted mean $\mu$.
Means in these formulas are defined only for $m_j>0$. Zero-mass clusters are
omitted from the weighted sums; no division by zero or $0\cdot\mathrm{NaN}$ is
intended.
Expanding the square and using
$\sum_{i:z_i=j}w_i(x_i-\mu_j)=0$ gives two separate identities:

$$
\sum_i w_i\lVert x_i-\mu\rVert^2
=\sum_i w_i\lVert x_i-\mu_{z_i}\rVert^2
+\sum_jm_j\lVert\mu_j-\mu\rVert^2,
$$

$$
J(C,z)
=\sum_i w_i\lVert x_i-\mu_{z_i}\rVert^2
+\sum_jm_j\lVert c_j-\mu_j\rVert^2.
$$

These identities remain valid when fitting stops early and the fitted centers
are not the empirical means of their final assignments. Therefore
`cluster_statistics` reports `within_ss` and `center_offset_ss` separately;
it does not incorrectly identify inertia with within-cluster variation for
unconverged centers.

```python
from mykmeans.analysis import cluster_statistics

statistics = cluster_statistics(
    X,
    model.labels_,
    centers=model.cluster_centers_,
    sample_weight=weights,
    covariance="diag",  # None / "none", "diag", or "full"
)

print(statistics["weight_sums"])
print(statistics["means"])
print(statistics["inertia_per_cluster"])
assert abs(
    statistics["total_ss"]
    - statistics["within_ss"]
    - statistics["between_ss"]
) < 1e-8 * max(1, statistics["total_ss"])
```

The returned dictionary contains:

| Key | Meaning |
| --- | --- |
| `counts` | Number of rows assigned to each cluster, including zero-weight rows |
| `weight_sums` | Sum of weights per cluster |
| `means` | Empirical weighted means |
| `covariances` | Population diagonal/full covariances, or `None` |
| `radii` | Largest distance from an empirical mean among its positive-weight rows |
| `inertia_per_cluster`, `inertia` | Objective about supplied centers, or empirical means if centers are omitted |
| `quantization_error` | Inertia divided by total weight |
| `within_per_cluster`, `within_ss` | Variation about empirical cluster means |
| `between_ss`, `total_ss` | Between-cluster and total variation |
| `center_offset_ss` | Objective caused by displacement of supplied centers from empirical means |
| `global_mean` | Global weighted mean |
| `explained_variance_ratio` | `between_ss / total_ss`, or NaN when total variation is zero |

Population covariance divides by $m_j$ and is a descriptive weighted second
moment. It is not an unbiased covariance estimator under every interpretation
of weights. Empty and zero-mass clusters have zero inertia and NaN means,
covariances, and radii. `n_clusters` can include otherwise absent clusters.
Means and moments are computed after translation to limit large-offset
cancellation; roundoff still applies.

Full covariance requires $O(nd^2)$ arithmetic and $O(kd^2)$ output. The simpler
diagonal form requires $O(nd)$ arithmetic. Per-cluster extraction scans the label
array, so the present diagnostic implementation additionally spends $O(nk)$
masking work. This path is for analysis, not the fitting benchmark.

## Gradient and Hessian

With assignments fixed, the objective is a sum of independent quadratics:

$$
\nabla_{c_j}J=2\sum_{i:z_i=j}w_i(c_j-x_i)
=2m_j(c_j-\mu_j),
\qquad
\nabla^2_{c_j,c_l}J=2m_jI_d\,\mathbf{1}_{j=l}.
$$

```python
from mykmeans.analysis import objective_gradient, assignment_hessian

gradient = objective_gradient(
    X, model.cluster_centers_, labels=model.labels_, sample_weight=weights
)
hessian = assignment_hessian(
    model.labels_, model.n_clusters, X.shape[1], sample_weight=weights
)
# hessian is a CSR matrix of shape (k*d, k*d), with row-major center ordering.
```

Omit `labels` in `objective_gradient` to select nearest centers automatically.
The resulting minimum-of-quadratics objective is generally nondifferentiable
on Voronoi ties. There the helper returns the selected branch's derivative,
using the smallest center index; it does not claim an ordinary gradient exists.
The Hessian likewise describes fixed assignments, or a neighborhood in which
assignments remain unchanged. Empty clusters have zero Hessian blocks.

## Voronoi cells and power diagrams

Expanding $\lVert x-c_j\rVert^2\leq\lVert x-c_l\rVert^2$ gives the halfspace

$$
2(c_l-c_j)^\mathsf{T}x
\leq\lVert c_l\rVert^2-\lVert c_j\rVert^2.
$$

```python
from mykmeans.geometry import voronoi_halfspaces

A, b = voronoi_halfspaces(model.cluster_centers_, index=0)
inside_closed_cell = (X @ A.T <= b).all(axis=1)

# All cells: A.shape == (k, k-1, d), b.shape == (k, k-1).
all_A, all_b = voronoi_halfspaces(model.cluster_centers_)
```

These are closed cells: their shared boundaries overlap. The predictor's
deterministic tie-breaking rule assigns a boundary point to one center.
`power_weights=p` instead constructs cells for scores
$\lVert x-c_j\rVert^2-p_j$, adding $p_j-p_l$ to the right side.
These additive weights are geometric parameters, not observation weights.
Duplicate sites can create redundant inequalities or empty power cells.

## Feature weighting and Mahalanobis geometry

An arbitrary distance cannot be substituted into Euclidean k-means while
retaining its mean update. A positive-semidefinite quadratic metric has an
explicit Euclidean embedding. Factor $M=LL^\mathsf{T}$ and transform rows by
$x\mapsto xL$. Then

$$
\lVert(x-y)L\rVert^2=(x-y)M(x-y)^\mathsf{T}.
$$

```python
import numpy as np
from sklearn.pipeline import make_pipeline
from mykmeans import KMeans
from mykmeans.geometry import FeatureMetric

# X in this example has exactly two features.
pipeline = make_pipeline(
    FeatureMetric(feature_weights=[1.0, 4.0]),
    KMeans(n_clusters=3, random_state=0),
)
labels = pipeline.fit_predict(X)

# Alternatively use a symmetric PSD metric matrix.
transformer = FeatureMetric(metric=np.array([[2.0, 0.5], [0.5, 1.0]]))
Z = transformer.fit_transform(X)
metric_model = KMeans(n_clusters=3, random_state=0).fit(Z)
centers_original = transformer.inverse_transform(metric_model.cluster_centers_)
```

`FeatureMetric` preserves the number of coordinates and returns float64 output.
It performs no standardization or centering. Singular metrics define
pseudometrics; inverse transformation returns a minimum-norm preimage and cannot
recover null-space coordinates. Tiny negative eigenvalues attributable to
roundoff relative to the matrix's own scale are clipped to zero; materially
indefinite or asymmetric matrices are rejected, including matrices whose entire
scale is much smaller than one. `factor_`, `metric_`, `inverse_factor_`, and
`rank_` expose the geometry. The latter is a numerical rank, and the
pseudoinverse also discards singular directions below its floating-point
cutoff. The zero metric is allowed and maps every observation to zero.

## Label symmetry and soft assignment

```python
from mykmeans.analysis import align_centers, align_labels, soft_assign

aligned_centers, permutation = align_centers(reference_centers, candidate_centers)
candidate_labels_in_reference_order = np.argsort(permutation)[candidate_labels]

# Alternatively maximize sample agreement between two partitions.
aligned_labels = align_labels(reference_labels, candidate_labels)

# Either pass X and centers, or pass a precomputed squared-distance matrix.
membership = soft_assign(X, model.cluster_centers_, temperature=0.5)
membership_again = soft_assign(model.transform_squared(X), temperature=0.5)
```

Center matching minimizes total squared distance using an optimal bijection.
Label matching maximizes sample agreement and requires equally many observed
clusters. They solve different correspondence problems.

Soft assignment uses

$$
q_{ij}=\frac{\exp(-\lVert x_i-c_j\rVert^2/T)}
{\sum_l\exp(-\lVert x_i-c_l\rVert^2/T)},\quad T>0.
$$

The implementation subtracts the row's minimum squared distance before
exponentiation. Temperature has units of squared distance. These are Gibbs
weights derived from geometry, not calibrated posterior probabilities or a
fitted Gaussian mixture model. Output memory is $O(nk)$.

## Choosing k and measuring initialization stability

```python
from mykmeans.analysis import evaluate_k, clustering_stability, gap_statistic

records = evaluate_k(
    X, range(2, 9), random_state=0,
    silhouette_sample_size=2000, n_init=5,
)
stability = clustering_stability(
    X, 3, seeds=[0, 1, 2, 3, 4], n_init=1,
)
gap = gap_statistic(
    X, range(1, 9), n_refs=20, reference="pca", random_state=0,
    n_init=5,
)
```

`evaluate_k` returns inertia and iteration count alongside independently
implemented standard
[silhouette](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.silhouette_score.html),
[Calinski-Harabasz](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.calinski_harabasz_score.html),
and [Davies-Bouldin](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.davies_bouldin_score.html)
scores. The linked sklearn pages describe the comparison definitions; no sklearn
metric implementation is called. Inertia honors supplied weights. Those three diagnostics are
unweighted, including zero-weight rows; every result records
`diagnostic_weighting="unweighted"`. Undefined metrics are NaN. Silhouette
sampling caps its pairwise calculation by default; passing None requests the
full quadratic-work computation. Pairwise distances are computed in blocks,
so the full n-by-n distance matrix is not retained. Blocking controls memory,
not the number of pairwise distances. No single score establishes a true k.

`clustering_stability` reports the matrix, mean, and minimum of pairwise
[adjusted Rand indices](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.adjusted_rand_score.html)
across seeds, using this package's own contingency-count implementation. This
is unweighted agreement on the same observations, invariant
to label permutation. It measures sensitivity to initialization, not sampling
uncertainty, bootstrap stability, or agreement with an external truth.

`gap_statistic` implements a Monte Carlo gap diagnostic based on the
[Tibshirani-Walther-Hastie proposal](https://doi.org/10.1111/1467-9868.00293).
It is unweighted. The observed and reference dispersions are Euclidean inertias,
and reference data are uniform within an axis-aligned or PCA-aligned box:

$$
\operatorname{Gap}(k)=B^{-1}\sum_b\log W_{kb}^{*}-\log W_k.
$$

The implementation estimates the simulation standard deviation with NumPy's
sample convention `ddof=1` and multiplies it by $\sqrt{1+1/B}$; the returned
`standard_error` uses that convention explicitly. It selects the first candidate
whose gap is at least the next candidate's gap minus the latter's error bound.
Consecutive candidate values recover the usual k-versus-k+1 selection form.
If no candidate qualifies, the final candidate is returned. It raises for zero
dispersion, because its logarithm is undefined.

This operation needs $|K|(B+1)$ fits and a PCA factorization when requested.
The default ten references is a computationally modest starting point, not a
precision guarantee. Increase it to check Monte Carlo sensitivity; repeat across
seeds and inspect the entire gap curve and error values. The output includes
every reference log inertia, allowing the calculation to be audited.
