"""Timing harness.

The assignment asks for queries-per-second measured **one query at a time on a
single thread**, repeated a few times with the median reported.  The details
that actually decide whether those numbers mean anything:

*   **Thread control.**  FAISS parallelises its scans with OpenMP and hnswlib
    takes a ``num_threads`` argument.  Both are forced to one thread, and the
    OpenMP environment variables are set before NumPy/FAISS are imported (see
    ``run_all.sh`` and the notebook's first cell), because setting them after
    import is too late for some BLAS builds.

*   **Warm-up.**  The first query through a fresh index pays page faults and
    cold caches.  One warm-up pass is run and discarded.

*   **Median of repeats, and repeats are interleaved by the caller.**  Taking
    the median of a few repeats removes one-off interference; the driver runs
    the repeats of a configuration back to back but the configurations
    themselves are ordered so that thermal drift does not line up with any one
    index type.

*   **Time budget instead of a fixed query count.**  A configuration at 50 QPS
    would need 100 s per repeat over 5 000 queries.  Each repeat therefore
    stops at ``time_budget_s`` provided at least ``min_queries`` have been
    timed, and the number actually timed is recorded in ``results.csv`` so the
    report can state the policy honestly.  Recall is always computed on the
    full query set; only the *timing* is truncated.

*   **One clock reading per repeat, not per query.**  ``perf_counter`` costs
    tens of nanoseconds, which is below the noise floor here, but reading it
    once per repeat also avoids any question about whether the measurement
    itself perturbs the loop.  The budget check reads the clock every
    ``check_every`` queries instead.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

import numpy as np

__all__ = ["TimingConfig", "TimingResult", "set_single_thread", "time_single_query", "time_batch"]


@dataclass
class TimingConfig:
    repeats: int = 3
    warmup: bool = True
    max_queries: int = 5000
    min_queries: int = 500
    time_budget_s: float = 20.0
    check_every: int = 32


@dataclass
class TimingResult:
    qps_median: float
    qps_runs: list = field(default_factory=list)
    n_queries_timed: int = 0
    seconds_median: float = 0.0


def set_single_thread() -> None:
    """Force every library in the stack down to one thread."""
    try:
        import faiss

        faiss.omp_set_num_threads(1)
    except Exception:  # pragma: no cover - faiss always present in practice
        pass
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(var, "1")


def time_single_query(index, queries: np.ndarray, k: int, cfg: TimingConfig) -> TimingResult:
    """Measure one-query-at-a-time throughput.

    ``index`` must expose ``search_one(vector, k)``.  The returned ids are
    discarded here: correctness is measured separately, on the full query set.
    """
    set_single_thread()
    queries = np.ascontiguousarray(queries, dtype=np.float32)
    nq = min(len(queries), cfg.max_queries)

    if cfg.warmup:
        for i in range(min(16, nq)):
            index.search_one(queries[i], k)

    qps_runs: list[float] = []
    timed_counts: list[int] = []
    for _ in range(max(1, cfg.repeats)):
        t0 = time.perf_counter()
        done = 0
        for i in range(nq):
            index.search_one(queries[i], k)
            done += 1
            if done % cfg.check_every == 0 and done >= cfg.min_queries:
                if time.perf_counter() - t0 >= cfg.time_budget_s:
                    break
        elapsed = time.perf_counter() - t0
        qps_runs.append(done / elapsed if elapsed > 0 else float("inf"))
        timed_counts.append(done)

    return TimingResult(
        qps_median=float(np.median(qps_runs)),
        qps_runs=[float(x) for x in qps_runs],
        n_queries_timed=int(np.median(timed_counts)),
        seconds_median=float(np.median([c / q for c, q in zip(timed_counts, qps_runs)])),
    )


def time_batch(index, queries: np.ndarray, k: int, cfg: TimingConfig, threads: int = 1) -> TimingResult:
    """Measure throughput with the whole query set handed over in one call.

    Used only for the Part 1 single-vs-batch comparison.  ``threads`` is kept
    at 1 by default so that the comparison isolates the BLAS level-2 vs level-3
    effect rather than measuring parallelism.
    """
    import faiss

    faiss.omp_set_num_threads(threads)
    queries = np.ascontiguousarray(queries[: cfg.max_queries], dtype=np.float32)

    if cfg.warmup:
        index.search(queries[: min(256, len(queries))], k)

    qps_runs = []
    for _ in range(max(1, cfg.repeats)):
        t0 = time.perf_counter()
        index.search(queries, k)
        elapsed = time.perf_counter() - t0
        qps_runs.append(len(queries) / elapsed if elapsed > 0 else float("inf"))

    faiss.omp_set_num_threads(1)
    return TimingResult(
        qps_median=float(np.median(qps_runs)),
        qps_runs=[float(x) for x in qps_runs],
        n_queries_timed=len(queries),
        seconds_median=float(len(queries) / np.median(qps_runs)),
    )
