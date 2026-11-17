#!/usr/bin/env python3
"""Reproducible timing and objective audit against scikit-learn.

Run from an installed checkout::

    python benchmarks/benchmark.py --quick --threads 1 --repeats 3 --output results.json

Each case runs shared explicit centers and ordinary k-means++ initialization.
Data generation, warmup, objective audit, and ARI are outside the timed region.
"""

from __future__ import annotations

import argparse
import datetime as dt
import gc
import hashlib
import importlib.metadata
import json
import os
import platform
from pathlib import Path
import statistics
import time

import numpy as np
import scipy
import sklearn
from sklearn.cluster import KMeans as SklearnKMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import adjusted_rand_score
from threadpoolctl import threadpool_info, threadpool_limits

import mykmeans
from mykmeans import KMeans


def _cpu_name():
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def _metadata(args):
    versions = dict(python=platform.python_version(), numpy=np.__version__,
                    scipy=scipy.__version__, sklearn=sklearn.__version__,
                    mykmeans=getattr(mykmeans, "__version__", "unknown"))
    try:
        versions["cython"] = importlib.metadata.version("Cython")
    except importlib.metadata.PackageNotFoundError:
        versions["cython"] = None
    try:
        affinity = len(os.sched_getaffinity(0))
    except AttributeError:
        affinity = None
    core_info = {}
    try:
        from mykmeans import _core
        if hasattr(_core, "build_info"):
            core_info = _core.build_info()
        else:
            core_info = {"openmp_enabled": _core.openmp_enabled(),
                         "max_threads_before_limit": _core.max_threads()}
    except ImportError:
        pass
    return dict(
        generated_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
        platform=platform.platform(), cpu=_cpu_name(),
        logical_cpus=os.cpu_count(), affinity_cpus=affinity,
        versions=versions, threadpools=threadpool_info(),
        native_build=core_info,
        source_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in sorted((Path(__file__).resolve().parents[1] / "src/mykmeans").glob("*"))
                       if p.suffix in {".py", ".pyx", ".h"}},
        arguments=vars(args),
        methodology={
            "timer": "time.perf_counter; fit only",
            "summary": "median; raw repeated timings retained",
            "warmup": "one unreported fit per algorithm/case/initialization",
            "thread_control": "threadpool_limits for BLAS and OpenMP; mykmeans n_threads identical",
            "fixed_initialization": "identical explicit centers, sampled without replacement with RandomState(2026)",
            "end_to_end": "each estimator performs its own greedy k-means++; same random_state, seeds need not match",
            "timing_exclusions": "data generation, warmup, independent inertia calculation, ARI",
            "quality_note": "speed ratios are meaningful together with objective/iterations; end-to-end seeds may differ",
            "scope": "synthetic dense CPU workloads; no universal performance guarantee",
        },
    )


def _cases(quick):
    # The overlapping case tests weak triangle-bound pruning; high-dimensional
    # data tests the regime where BLAS-backed scikit-learn may be faster.
    scale = 1 if quick else 4
    return [
        dict(name="separated_3d", n=25000 * scale, d=3, k=32, std=0.65),
        dict(name="overlapping_16d", n=12000 * scale, d=16, k=16, std=7.0),
        dict(name="highdim_128d", n=5000 * scale, d=128, k=32, std=3.0),
        dict(name="uniform_2d", n=30000 * scale, d=2, k=64, kind="uniform"),
    ]


def _independent_inertia(x, centers):
    total = 0.0
    centers = centers.astype(np.float64, copy=False)
    for start in range(0, len(x), 1024):
        chunk = x[start:start + 1024].astype(np.float64, copy=False)
        d2 = np.sum((chunk[:, None, :] - centers[None, :, :]) ** 2, axis=2)
        total += d2.min(axis=1).sum(dtype=np.float64)
    return float(total)


def _fit_timing(factory, x, repeats):
    factory().fit(x)
    elapsed = []
    model = None
    for _ in range(repeats):
        model = factory()
        gc.collect()
        start = time.perf_counter()
        model.fit(x)
        elapsed.append(time.perf_counter() - start)
    return model, elapsed


def run(args):
    report = dict(metadata=_metadata(args), results=[])
    dtypes = [np.dtype(name) for name in args.dtypes.split(",")]
    if any(dtype not in (np.dtype("float32"), np.dtype("float64")) for dtype in dtypes):
        raise ValueError("--dtypes must contain float32 and/or float64")
    cases = _cases(args.quick)
    if args.cases:
        selected = set(args.cases.split(","))
        unknown = selected.difference(case["name"] for case in cases)
        if unknown:
            raise ValueError(f"Unknown cases: {sorted(unknown)}")
        cases = [case for case in cases if case["name"] in selected]
    initializations = args.initializations.split(",")
    if not set(initializations).issubset({"fixed", "k-means++"}):
        raise ValueError("--initializations expects fixed and/or k-means++")
    variants = [("sklearn_lloyd", SklearnKMeans, "lloyd"),
                ("sklearn_elkan", SklearnKMeans, "elkan"),
                ("mykmeans_lloyd", KMeans, "lloyd"),
                ("mykmeans_hamerly", KMeans, "hamerly"),
                ("mykmeans_elkan", KMeans, "elkan"),
                ("mykmeans_exponion", KMeans, "exponion")]
    with threadpool_limits(limits=args.threads):
        report["metadata"]["effective_threadpools"] = threadpool_info()
        for case in cases:
            if case.get("kind") == "uniform":
                source = np.random.RandomState(42).uniform(-1, 1, size=(case["n"], case["d"]))
            else:
                source, _ = make_blobs(
                    n_samples=case["n"], n_features=case["d"], centers=case["k"],
                    cluster_std=case["std"], center_box=(-20, 20), random_state=42)
            for dtype in dtypes:
                x = np.asarray(source, dtype=dtype, order="C")
                indices = np.random.RandomState(2026).choice(len(x), case["k"], replace=False)
                fixed = x[indices].copy()
                for initialization in initializations:
                    group = []
                    reference_labels = None
                    for label, cls, algorithm in variants:
                        options = dict(
                            n_clusters=case["k"], algorithm=algorithm,
                            init=fixed if initialization == "fixed" else "k-means++",
                            n_init=1, tol=args.tol, max_iter=args.max_iter, random_state=7)
                        if cls is KMeans:
                            options["n_threads"] = args.threads
                        model, timings = _fit_timing(lambda: cls(**options), x, args.repeats)
                        if reference_labels is None:
                            reference_labels = model.labels_.copy()
                        check_inertia = _independent_inertia(x, model.cluster_centers_)
                        record = dict(
                            case=case["name"], n_samples=case["n"], n_features=case["d"],
                            n_clusters=case["k"], dtype=dtype.name,
                            data_parameters=case,
                            initialization=initialization, implementation=label,
                            threads=args.threads, seconds_median=statistics.median(timings),
                            seconds_min=min(timings), seconds_max=max(timings), seconds_all=timings,
                            inertia=float(model.inertia_), inertia_independent=check_inertia,
                            inertia_report_relative_error=abs(float(model.inertia_) - check_inertia) /
                                                          max(check_inertia, np.finfo(float).tiny),
                            n_iter=int(model.n_iter_),
                            ari_vs_sklearn_lloyd=float(adjusted_rand_score(reference_labels, model.labels_)),
                            backend=getattr(model, "backend_", "sklearn"),
                            resolved_algorithm=getattr(model, "algorithm_", algorithm),
                            assignment_engine=getattr(model, "assignment_engine_", None),
                            distance_evaluations=getattr(model, "distance_evaluations_", None),
                        )
                        group.append(record)
                        print(f"{case['name']:18s} {dtype.name:7s} {initialization:10s} "
                              f"{label:19s}: {record['seconds_median']:.6f} s "
                              f"inertia={record['inertia']:.6g} iterations={record['n_iter']}", flush=True)
                    baseline = group[0]
                    for record in group:
                        record["speedup_vs_sklearn_lloyd"] = baseline["seconds_median"] / record["seconds_median"]
                        record["inertia_relative_delta_vs_sklearn_lloyd"] = (
                            record["inertia_independent"] - baseline["inertia_independent"]) / max(
                                baseline["inertia_independent"], np.finfo(float).tiny)
                    report["results"].extend(group)
                    if args.output:
                        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
                        Path(args.output).write_text(json.dumps(report, indent=2, default=str) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="use smaller, but nontrivial, cases")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--dtypes", default="float64", help="comma-separated float32,float64")
    parser.add_argument("--cases", default=None, help="comma-separated separated_3d,overlapping_16d,highdim_128d,uniform_2d")
    parser.add_argument("--initializations", default="fixed,k-means++")
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--output", default="benchmarks/results.json")
    args = parser.parse_args()
    if args.threads < 1 or args.repeats < 1:
        parser.error("--threads and --repeats must be positive")
    run(args)


if __name__ == "__main__":
    main()
