"""The one sweep loop that every experiment reuses.

Parts 2, 3 and 4 differ only in *which* datasets and configurations they feed
in; the measurement protocol has to be identical or the numbers are not
comparable across parts.  So the protocol lives here, once.

Protocol
--------
For each build configuration: build once, record build time and serialized
size, then walk the search parameter from cheap to expensive.  At each value:

1.  **Recall on both query splits.**  Measured with batched, multi-threaded
    search, because recall is a correctness measurement and parallelism cannot
    change it.  Tie resolution uses exact recomputed distances, never the
    distances the index reports (IVF-PQ's are approximate).

2.  **QPS on one split, one query at a time, one thread.**  By default the
    *tuning* split is timed.  QPS is a property of the index and its parameter,
    not of which 5 000 queries are used — the work done per query is bounded by
    ``efSearch``/``nprobe``, not by the query's content — so timing on the
    tuning split keeps the held-out queries completely untouched by anything
    that influences a reported choice.  ``scripts/check_timing_split.py``
    verifies the two splits agree to within run-to-run noise, and the report
    cites that check.

3.  **Early stop.**  Once recall passes ``stop_above_recall`` (default 0.995)
    the remaining, larger parameter values only cost time: the curve is flat
    there and the tables need nothing above 0.95.  The stop is recorded in the
    row's ``notes`` so the sweep's extent is visible in the CSV.
"""

from __future__ import annotations

import json
import sys

import numpy as np

from .frontier import pareto_frontier
from .indexes import index_type_name, make_index
from .recall import recall_at_k, true_distances
from .results import ResultsWriter
from .timing import TimingConfig, time_single_query

__all__ = ["evaluate_recall", "sweep_config", "run_sweep"]


def evaluate_recall(index, ds, k: int, train_for_ties: np.ndarray, threads: int = 0) -> float:
    """Tie-aware recall@k of ``index`` on the queries of ``ds``."""
    ids, _approx = index.search(ds.queries, k, threads=threads)
    exact = true_distances(ds.queries, train_for_ties, ids)
    return recall_at_k(ids, ds.gt_ids, ds.gt_dists, exact, k=k)


def sweep_config(
    *,
    writer: ResultsWriter,
    experiment: str,
    dataset_name: str,
    variant: str,
    index_kind: str,
    build_params: dict,
    search_values,
    train: np.ndarray,
    ds_tune,
    ds_test,
    k: int,
    timing_cfg: TimingConfig,
    timed_on: str = "tune",
    build_threads: int = 0,
    seed: int = 42,
    git_commit: str = "",
    machine_id: str = "",
    stop_above_recall: float = 0.995,
    verbose: bool = True,
) -> int:
    """Build one configuration and sweep its search parameter.

    Returns the number of rows written.
    """
    params_json = json.dumps(build_params, sort_keys=True)
    search_values = list(search_values) or [None]
    # How this configuration is reported (IVF-PQ with re-ranking is its own
    # index type). The build still uses `index_kind`.
    type_name = index_type_name(index_kind, build_params)

    # Skip the (expensive) build entirely when every row already exists.
    todo = [
        v
        for v in search_values
        if not writer.has(
            experiment=experiment,
            dataset=dataset_name,
            variant=variant,
            n_vectors=train.shape[0],
            index_type=type_name,
            params_json=params_json,
            search_param=("" if v is None else v),
        )
    ]
    if not todo:
        if verbose:
            print(f"    [skip] {type_name} {params_json} — already in results.csv")
        return 0

    if verbose:
        print(f"    build {type_name} {params_json} on n={train.shape[0]} ...", flush=True)
    index = make_index(index_kind, train.shape[1], seed=seed, build_threads=build_threads, **build_params)
    build_time = index.build(train)
    size = index.serialized_bytes()
    if verbose:
        print(
            f"      built in {build_time:.1f}s, {size/1e6:.1f} MB "
            f"({size/max(1,index.n_vectors):.1f} B/vec)",
            flush=True,
        )

    timing_queries = (ds_tune if timed_on == "tune" else ds_test).queries
    written = 0
    for v in search_values:
        key = dict(
            experiment=experiment,
            dataset=dataset_name,
            variant=variant,
            n_vectors=train.shape[0],
            index_type=type_name,
            params_json=params_json,
            search_param=("" if v is None else v),
        )
        if writer.has(**key):
            continue
        if v is not None:
            index.set_search_param(v)

        r_tune = evaluate_recall(index, ds_tune, k, train)
        r_test = evaluate_recall(index, ds_test, k, train)
        timing = time_single_query(index, timing_queries, k, timing_cfg)

        note = ""
        if r_tune >= stop_above_recall:
            note = f"sweep stopped: recall_tune >= {stop_above_recall}"

        writer.write(
            experiment=experiment,
            dataset=dataset_name,
            variant=variant,
            n_vectors=train.shape[0],
            dim=train.shape[1],
            index_type=type_name,
            params_json=params_json,
            search_param_name=index.search_param_name,
            search_param=("" if v is None else v),
            k=k,
            recall_tune=round(float(r_tune), 6),
            recall_test=round(float(r_test), 6),
            qps_median=round(timing.qps_median, 3),
            qps_runs_json=json.dumps([round(x, 3) for x in timing.qps_runs]),
            timed_on=timed_on,
            n_queries_timed=timing.n_queries_timed,
            build_time_s=round(build_time, 3),
            build_threads=index.build_threads,
            index_bytes=size,
            bytes_per_vector=round(size / max(1, index.n_vectors), 2),
            git_commit=git_commit,
            machine_id=machine_id,
            notes=note,
        )
        written += 1
        if verbose:
            print(
                f"      {index.search_param_name}={v!s:>6}  "
                f"recall_tune={r_tune:.4f} recall_test={r_test:.4f}  "
                f"qps={timing.qps_median:8.1f}  (timed {timing.n_queries_timed} q)",
                flush=True,
            )
        if r_tune >= stop_above_recall:
            if verbose:
                print("      early stop: recall saturated", flush=True)
            break

    del index
    return written


def run_sweep(specs, **common) -> int:
    """Run a list of ``(index_kind, build_params, search_values)`` specs."""
    total = 0
    for kind, params, values in specs:
        try:
            total += sweep_config(
                index_kind=kind, build_params=params, search_values=values, **common
            )
        except Exception as exc:  # keep a long sweep alive past one bad config
            print(f"    [FAILED] {kind} {params}: {exc}", file=sys.stderr, flush=True)
    return total


def summarise_frontier(df, recall_col="recall_tune"):
    """Convenience for notebooks: frontier per index type."""
    out = {}
    for kind, sub in df.groupby("index_type"):
        out[kind] = pareto_frontier(sub, recall_col=recall_col)
    return out
