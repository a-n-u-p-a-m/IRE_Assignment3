"""Part 1: exact baseline, ground-truth verification, single vs batch timing.

Produces
--------
*   rows in ``results.csv`` for the exact index (one per dataset, carrying both
    the single-query and the batched QPS);
*   ``results/tables/part1_verification.md`` — how our exact search compares to
    the neighbour ids shipped with each dataset, and the evidence that every
    disagreement is a distance tie rather than an error;
*   ``results/tables/part1_timing.md`` — the single-vs-batch comparison.

The two timings differ by an order of magnitude and the report has to explain
why.  Both are measured on one thread so that the comparison isolates the
memory-traffic effect and is not contaminated by parallelism:

*   one query at a time is a matrix-vector product (BLAS level 2).  Every query
    streams the entire database through the CPU once, so throughput is bounded
    by memory bandwidth and each vector is used for a single multiply-add pass
    before being evicted.
*   the whole query block at once is a matrix-matrix product (BLAS level 3).
    A tile of the database is loaded into cache and reused against every query
    in the block, so the same memory traffic serves thousands of queries, and
    per-call overhead is amortised too.
"""

from __future__ import annotations

import json
import os

from ..datasets import verify_ground_truth
from ..indexes import FlatIndex
from ..timing import TimingConfig, time_batch, time_single_query
from .common import base_parser, get_dataset, setup, splits_of


def main(argv=None) -> int:
    ap = base_parser("Part 1: exact baseline and recall-function verification")
    ap.add_argument("--tables", default="results/tables", help="where to write the Markdown tables")
    args = ap.parse_args(argv)
    cfg, writer, env, timing_cfg = setup(args)
    os.makedirs(args.tables, exist_ok=True)
    k = cfg.k

    verif_rows, timing_rows = [], []

    for name in cfg["datasets"]:
        print(f"\n=== Part 1: {name} ===")
        ds = get_dataset(cfg, name, verbose=not args.quiet)
        ds_tune, ds_test = splits_of(ds, cfg)
        print(f"  {ds!r}")

        # --- does our exact search agree with the shipped ground truth? ----
        print("  verifying ground truth against the shipped neighbour ids ...", flush=True)
        v = verify_ground_truth(ds_test, k=k)
        v["note"] = ds.note
        verif_rows.append(v)
        print(
            f"    naive set recall={v['naive_set_recall']:.6f}  "
            f"tie-aware recall={v['tie_aware_recall']:.6f}  "
            f"queries with a differing set={v['queries_with_differing_sets']}"
        )

        # --- exact index, single-query and batched -------------------------
        index = FlatIndex(ds.dim, seed=cfg.seed, build_threads=1)
        build_time = index.build(ds.train)
        size = index.serialized_bytes()

        single_cfg = TimingConfig(
            repeats=timing_cfg.repeats,
            warmup=True,
            # Exact search is slow; a smaller timed sample keeps Part 1 short
            # without changing the measured rate.
            max_queries=min(timing_cfg.max_queries, 512),
            min_queries=min(timing_cfg.min_queries, 64),
            time_budget_s=timing_cfg.time_budget_s,
        )
        batch_cfg = TimingConfig(
            repeats=timing_cfg.repeats, warmup=True,
            max_queries=single_cfg.max_queries, min_queries=single_cfg.min_queries,
            time_budget_s=timing_cfg.time_budget_s,
        )
        timed = ds_tune.queries
        print("  timing exact search, one query at a time (1 thread) ...", flush=True)
        single = time_single_query(index, timed, k, single_cfg)
        print("  timing exact search, all queries in one batch (1 thread) ...", flush=True)
        batch = time_batch(index, timed, k, batch_cfg, threads=1)
        speedup = batch.qps_median / single.qps_median if single.qps_median else float("nan")
        print(
            f"    single={single.qps_median:.1f} QPS   batch={batch.qps_median:.1f} QPS"
            f"   ({speedup:.1f}x)"
        )

        timing_rows.append(
            dict(dataset=name, n=ds.n, dim=ds.dim, single_qps=single.qps_median,
                 batch_qps=batch.qps_median, speedup=speedup,
                 n_timed=single.n_queries_timed, build_time_s=build_time,
                 index_bytes=size, bytes_per_vector=size / ds.n)
        )

        writer.write(
            experiment="part1",
            dataset=name,
            variant="exact-baseline",
            n_vectors=ds.n,
            dim=ds.dim,
            index_type="Flat",
            params_json=json.dumps({}),
            search_param_name="none",
            search_param="",
            k=k,
            recall_tune=1.0,
            recall_test=1.0,
            qps_median=round(single.qps_median, 3),
            qps_runs_json=json.dumps([round(x, 3) for x in single.qps_runs]),
            timed_on="tune",
            n_queries_timed=single.n_queries_timed,
            batch_qps=round(batch.qps_median, 3),
            build_time_s=round(build_time, 3),
            build_threads=1,
            index_bytes=size,
            bytes_per_vector=round(size / ds.n, 2),
            git_commit=env["git_commit"],
            machine_id=env["machine_id"],
            notes=f"batch/single speedup={speedup:.1f}x; both single-threaded",
        )
        del index

    # ------------------------------------------------------------------
    with open(os.path.join(args.tables, "part1_verification.md"), "w", encoding="utf-8") as f:
        f.write("# Part 1: our exact search vs the shipped ground truth\n\n")
        f.write(
            "`naive` is the plain set-intersection recall; `tie-aware` accepts a returned\n"
            "vector whose exact distance equals the k-th ground-truth distance within\n"
            "`rtol=1e-5`. Where the two differ, the disagreement is a tie, not an error.\n\n"
        )
        f.write("| dataset | queries | naive recall@10 | tie-aware recall@10 | queries with a differing top-10 set | queries still below 1.0 |\n")
        f.write("|---|---|---|---|---|---|\n")
        for v in verif_rows:
            f.write(
                f"| {v['dataset']} | {v['n_queries']} | {v['naive_set_recall']:.6f} | "
                f"{v['tie_aware_recall']:.6f} | {v['queries_with_differing_sets']} | "
                f"{v['queries_below_tie_aware_1.0']} |\n"
            )

    with open(os.path.join(args.tables, "part1_timing.md"), "w", encoding="utf-8") as f:
        f.write("# Part 1: exact search, one query at a time vs one batch\n\n")
        f.write("Both measured on a single thread, median of repeated runs.\n\n")
        f.write("| dataset | n | d | single-query QPS | batched QPS | speedup | queries timed |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for r in timing_rows:
            f.write(
                f"| {r['dataset']} | {r['n']:,} | {r['dim']} | {r['single_qps']:.1f} | "
                f"{r['batch_qps']:.1f} | {r['speedup']:.1f}x | {r['n_timed']} |\n"
            )
    print(f"\nwrote {args.tables}/part1_verification.md and part1_timing.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
