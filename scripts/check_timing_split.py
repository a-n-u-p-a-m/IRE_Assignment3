"""Does it matter which query split we time on?

The benchmark measures QPS on the tuning queries and reports recall on the
held-out queries, on the argument that throughput is a property of the index
and its search parameter rather than of which 5 000 queries are used: the work
done per query is bounded by ``efSearch`` / ``nprobe``, not by the query's
content.

That argument is cheap to check, and Part 5 asks for exactly this kind of
self-criticism, so this script measures a handful of configurations on *both*
splits and prints the difference next to the run-to-run spread of the repeats.
If the split-to-split difference is inside the repeat spread, timing on the
tuning queries costs nothing; if it is not, the report has to say so.

    python3 scripts/check_timing_split.py --config configs/full.yaml --dataset sift
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from annbench.config import load_config, timing_config_from  # noqa: E402
from annbench.experiments.common import get_dataset, splits_of  # noqa: E402
from annbench.indexes import make_index  # noqa: E402
from annbench.timing import time_single_query  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/full.yaml")
    ap.add_argument("--dataset", default="sift")
    ap.add_argument("--out", default="results/tables/timing_split_check.md")
    args = ap.parse_args()

    cfg = load_config(args.config)
    tcfg = timing_config_from(cfg)
    ds = get_dataset(cfg, args.dataset)
    tune, test = splits_of(ds, cfg)

    configs = [
        ("HNSW", {"M": 16, "ef_construction": 200}, 64),
        ("HNSW", {"M": 16, "ef_construction": 200}, 256),
        ("IVF-Flat", {"nlist": 4096}, 8),
        ("IVF-Flat", {"nlist": 4096}, 64),
    ]

    rows = []
    for kind, params, value in configs:
        idx = make_index(kind, ds.dim, seed=cfg.seed, build_threads=0, **params)
        idx.build(ds.train)
        idx.set_search_param(value)
        a = time_single_query(idx, tune.queries, cfg.k, tcfg)
        b = time_single_query(idx, test.queries, cfg.k, tcfg)
        spread_a = (max(a.qps_runs) - min(a.qps_runs)) / a.qps_median if len(a.qps_runs) > 1 else float("nan")
        spread_b = (max(b.qps_runs) - min(b.qps_runs)) / b.qps_median if len(b.qps_runs) > 1 else float("nan")
        diff = abs(a.qps_median - b.qps_median) / statistics.mean([a.qps_median, b.qps_median])
        rows.append((f"{kind} {params} @{value}", a.qps_median, b.qps_median, diff,
                     max(spread_a, spread_b)))
        print(f"{rows[-1][0]:45s} tune={a.qps_median:9.1f} test={b.qps_median:9.1f} "
              f"diff={diff*100:5.2f}%  repeat spread={max(spread_a, spread_b)*100:5.2f}%")
        del idx

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("# Does the timing split matter?\n\n")
        f.write(
            "QPS is measured on the tuning queries throughout the benchmark. This table "
            "compares it against the same configuration timed on the held-out queries. "
            "`split difference` is the gap between the two; `repeat spread` is the "
            "run-to-run variation within one split. A split difference no larger than the "
            "repeat spread means the choice of split is not measurable.\n\n"
        )
        f.write("| configuration | QPS (tune) | QPS (test) | split difference | repeat spread |\n")
        f.write("|---|---|---|---|---|\n")
        for name, qa, qb, diff, spread in rows:
            f.write(f"| {name} | {qa:,.1f} | {qb:,.1f} | {diff*100:.2f}% | {spread*100:.2f}% |\n")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
