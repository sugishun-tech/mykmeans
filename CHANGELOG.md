# Changelog

## 0.2.0 — 2026-11-17

- Removed all scikit-learn numerical delegation. Required runtime dependencies
  are now NumPy, SciPy, and threadpoolctl. Scikit-learn is optional for protocol
  integration and independent comparison tests.
- Added an independent estimator foundation for parameter handling, validation,
  feature names, cloning, output wrappers, and metadata-request configuration.
  Optional sklearn tag and routing objects are created only for integration.
- Added independent dense Elkan and Exponion implementations. Exponion uses its
  geometric candidate-ball pruning with full sorted neighbor rows, rather than
  the original paper's partial-sort strategy.
- Added independent CSR Lloyd and CSR Elkan kernels, CSR initialization,
  prediction, transformations, stable sparse moments, and weighted updates.
  CSR `auto` selects Lloyd; CSR Hamerly/Exponion requests raise `ValueError`.
- Added conservative triangle-inequality pruning to dense greedy k-means++
  distance updates, with a deterministic cost gate. Greedy trials and sampling
  rules are preserved; no observations are removed from the sampling potential.
- Added independent adjusted Rand, silhouette samples/score, Calinski–Harabasz,
  and Davies–Bouldin metrics; model-selection and stability helpers use them.
- Extended restart histories, assignment counters, and finalized objectives to
  every implemented dense and CSR fitting path.
- Documented Elkan's float64 n-by-k lower-bound matrix, Exponion's O(k²) neighbor
  storage and O(k² log k) sorting cost, dense center/aggregate storage for CSR,
  and the nonweighted convention of evaluation metrics.
- Added a primary-source review of recent acceleration methods, algorithm and
  seeding regression checks, standalone operation checks, and benchmark drivers.
  See [research](docs/sota.md), [benchmarks](benchmarks/README.md), and
  [validation](docs/validation.md) for the scope and recorded evidence.

## 0.1.0 — 2026-11-17 (previous prototype)

- Added independent Cython/OpenMP dense weighted Lloyd and Hamerly engines.
- Added blocked GEMM, conservative candidate screening, direct distance checks,
  incremental Hamerly aggregates, and deterministic exact-tie ordering.
- Added scikit-learn-compatible estimator and initialization/functional APIs.
- Retained explicit scikit-learn backends for sparse matrices and Elkan.
- Added weighted cluster statistics, ANOVA, covariance, gradients, Hessians,
  Voronoi/power cells, PSD metric transforms, assignment matching, Gibbs weights,
  k evaluation, seed stability, and gap statistics.
- Added public-coordinate finalization for reliable predictions near floating
  boundaries, weighted edge-case checks, serial build support, and optional
  local CPU tuning.
- Included reproducible comparison code, raw measurements, tests, examples,
  documentation, and a multi-platform CI definition.

The 0.1.0 delegation paths above are historical; they were removed in 0.2.0.
See the validation report for the exact platforms and versions actually
executed. A CI configuration is not evidence that its remote jobs have run.
