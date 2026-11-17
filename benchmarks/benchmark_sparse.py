#!/usr/bin/env python3
"""Reproducible CSR comparisons; sklearn is a benchmark dependency only."""
from __future__ import annotations

import argparse
import datetime as dt
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import time

import numpy as np
import scipy
from scipy import sparse
import sklearn
from sklearn.cluster import KMeans as ReferenceKMeans
from sklearn.metrics import adjusted_rand_score
from threadpoolctl import threadpool_info, threadpool_limits

import mykmeans
from mykmeans import KMeans, _core


def cpu_name():
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def make_case(case):
    rng = np.random.default_rng(42)
    x = sparse.random(case["n"], case["d"], density=case["density"],
                      random_state=rng, data_rvs=rng.standard_normal,
                      format="csr", dtype=np.float64)
    x.sum_duplicates()
    x.sort_indices()
    return x


def independent_inertia(x, centers, block_size=128):
    """Small dense reference blocks only, never the complete observation matrix."""
    total = 0.0
    centers = np.asarray(centers, dtype=np.float64)
    for start in range(0, x.shape[0], block_size):
        block = x[start:start + block_size].toarray().astype(np.float64, copy=False)
        distances = ((block[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        total += float(distances.min(axis=1).sum(dtype=np.float64))
    return total


def timed_fit(factory, x, repeats):
    factory().fit(x)
    times = []
    for _ in range(repeats):
        model = factory()
        gc.collect()
        begin = time.perf_counter()
        model.fit(x)
        times.append(time.perf_counter() - begin)
    return model, times


def run(args):
    cases = [dict(name="csr_1000d_1pct", n=12000, d=1000, k=16, density=0.01),
             dict(name="csr_64d_10pct", n=20000, d=64, k=16, density=0.10)]
    report = {
        "metadata": {
            "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "platform": platform.platform(), "cpu": cpu_name(),
            "logical_cpus": os.cpu_count(),
            "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
            "versions": {"python": platform.python_version(), "numpy": np.__version__,
                         "scipy": scipy.__version__, "sklearn": sklearn.__version__,
                         "mykmeans": mykmeans.__version__},
            "native_build": _core.build_info(),
            "source_sha256": {
                p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted((Path(__file__).resolve().parents[1] / "src/mykmeans").glob("*"))
                if p.suffix in {".py", ".pyx", ".h"}
            },
            "arguments": vars(args),
            "methodology": {
                "timer": "time.perf_counter; fit only",
                "warmup": "one unreported fit per implementation/case/initialization",
                "summary": "median of repeats; raw times retained",
                "data": "canonical CSR, uniform random sparsity, stored values standard normal, seed 42",
                "fixed_initialization": "same nonempty rows sampled using RandomState(2026)",
                "end_to_end": "each estimator uses its own k-means++; random_state=7, seeds can differ",
                "thread_control": "same threadpool_limits and native n_threads upper bound",
                "audit": "independent direct differences in dense blocks of at most 128 rows outside timing",
                "scope": "unstructured signed sparse data; not a universal workload",
            },
        },
        "results": [],
    }
    variants = [("sklearn_lloyd", ReferenceKMeans, "lloyd"),
                ("sklearn_elkan", ReferenceKMeans, "elkan"),
                ("mykmeans_lloyd", KMeans, "lloyd"),
                ("mykmeans_elkan", KMeans, "elkan")]
    with threadpool_limits(limits=args.threads):
        report["metadata"]["effective_threadpools"] = threadpool_info()
        for case in cases:
            x = make_case(case)
            nonempty = np.flatnonzero(np.diff(x.indptr))
            seed_rows = np.random.RandomState(2026).choice(nonempty, case["k"], replace=False)
            # Only k centers are dense; the full sample matrix remains CSR.
            initial = x[seed_rows].toarray()
            for initialization in ("fixed", "k-means++"):
                group, reference_labels = [], None
                for name, cls, algorithm in variants:
                    parameters = dict(n_clusters=case["k"], init=initial if initialization == "fixed" else "k-means++",
                                      algorithm=algorithm, n_init=1, max_iter=args.max_iter,
                                      tol=args.tol, random_state=7)
                    if cls is KMeans:
                        parameters["n_threads"] = args.threads
                    model, timings = timed_fit(lambda: cls(**parameters), x, args.repeats)
                    if reference_labels is None:
                        reference_labels = model.labels_.copy()
                    audited = independent_inertia(x, model.cluster_centers_)
                    record = {
                        "case": case["name"], "data_parameters": case,
                        "n_samples": x.shape[0], "n_features": x.shape[1],
                        "n_clusters": case["k"], "nnz": x.nnz, "density": case["density"],
                        "dtype": x.dtype.name, "initialization": initialization,
                        "implementation": name, "threads": args.threads,
                        "seconds_median": statistics.median(timings), "seconds_min": min(timings),
                        "seconds_max": max(timings), "seconds_all": timings,
                        "inertia": float(model.inertia_), "inertia_independent": audited,
                        "inertia_report_relative_error": abs(float(model.inertia_)-audited) / max(audited,np.finfo(float).tiny),
                        "n_iter": int(model.n_iter_),
                        "ari_vs_sklearn_lloyd": float(adjusted_rand_score(reference_labels, model.labels_)),
                        "backend": getattr(model,"backend_","sklearn"),
                        "resolved_algorithm": getattr(model,"algorithm_",algorithm),
                        "assignment_engine": getattr(model,"assignment_engine_",None),
                        "distance_evaluations": getattr(model,"distance_evaluations_",None),
                    }
                    group.append(record)
                    print(f"{case['name']:18s} {initialization:10s} {name:18s} "
                          f"{record['seconds_median']:.6f}s inertia={record['inertia']:.9g} "
                          f"iters={record['n_iter']} ARI={record['ari_vs_sklearn_lloyd']:.6f}", flush=True)
                baseline = group[0]
                for record in group:
                    record["speedup_vs_sklearn_lloyd"] = baseline["seconds_median"] / record["seconds_median"]
                    record["inertia_relative_delta_vs_sklearn_lloyd"] = (
                        record["inertia_independent"] - baseline["inertia_independent"]) / max(
                            baseline["inertia_independent"], np.finfo(float).tiny)
                report["results"].extend(group)
                Path(args.output).parent.mkdir(parents=True, exist_ok=True)
                Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--output", default="benchmarks/results_csr_sklearn191_1thread.json")
    options = parser.parse_args()
    if options.threads < 1 or options.repeats < 1:
        parser.error("threads and repeats must be positive")
    run(options)
