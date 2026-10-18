"""Part 3 (scaling): how the cost of recall 0.95 grows with database size.

Hypothesis (committed to git before this was run — see ``hypothesis.md``).

What the script does
--------------------
For each configured size N, it draws a seeded random subset of SIFT,
**recomputes exact ground truth against that subset**, and benchmarks HNSW,
IVF-Flat and exact search.

Two details that decide whether the experiment measures what it claims:

*   **Ground truth must be recomputed.**  The neighbour ids in the file index
    into the full 1 M database.  At N = 100 000 most of them are not even in
    the index, so reusing them would report a recall ceiling of roughly N/1M
    and the whole experiment would be measuring the subsetting, not the index.

*   **``nlist`` has to be retuned at every N.**  IVF's asymptotics come from
    the balance between the number of cells and their occupancy; the usual rule
    is ``nlist ~ c * sqrt(N)``.  Holding ``nlist`` fixed across N would measure
    "an index tuned for 1 M, run on 100 K", which degrades for a reason that
    has nothing to do with scaling.  The config gives the multiplier and the
    script derives ``nlist`` per size, rounded to a power of two.

Exact search is included as the reference: its per-query cost is exactly
linear in N, so it is the yardstick against which "sublinear" has meaning.
"""

from __future__ import annotations

import math

from ..bench import run_sweep, sweep_config
from .common import base_parser, get_dataset, setup, splits_of


def _nlist_for(n: int, multiplier: float) -> int:
    """``multiplier * sqrt(n)``, rounded to the nearest power of two."""
    raw = max(16.0, multiplier * math.sqrt(n))
    return int(2 ** round(math.log2(raw)))


def main(argv=None) -> int:
    ap = base_parser("Part 3: scaling behaviour on SIFT at several database sizes")
    args = ap.parse_args(argv)
    cfg, writer, env, timing_cfg = setup(args)

    p3 = cfg["part3"]
    dataset = p3.get("dataset", "sift")
    sizes = [int(n) for n in p3["sizes"]]
    multiplier = float(p3.get("nlist_multiplier", 4.0))
    build_threads = int((cfg.get("build", {}) or {}).get("threads", 0))
    timed_on = (cfg.get("timing", {}) or {}).get("time_on", "tune")
    hnsw = p3.get("hnsw", {"M": 16, "ef_construction": 200})
    ef_search = p3.get("ef_search", [10, 20, 40, 80, 160, 320])
    nprobe = p3.get("nprobe", [1, 2, 4, 8, 16, 32, 64, 128, 256])

    for n in sizes:
        print(f"\n=== Part 3: {dataset} at n={n:,} ===")
        ds = get_dataset(cfg, dataset, n_vectors=n, verbose=not args.quiet)
        ds_tune, ds_test = splits_of(ds, cfg)
        nlist = _nlist_for(ds.n, multiplier)
        print(f"  {ds!r}; nlist for this size = {nlist}")

        specs = [
            ("HNSW", {"M": int(hnsw["M"]), "ef_construction": int(hnsw.get("ef_construction", 200))},
             list(ef_search)),
            ("IVF-Flat", {"nlist": nlist}, [p for p in nprobe if p <= nlist]),
        ]
        common = dict(
            writer=writer,
            experiment="part3",
            dataset_name=dataset,
            variant=f"n={ds.n}",
            train=ds.train,
            ds_tune=ds_tune,
            ds_test=ds_test,
            k=cfg.k,
            timing_cfg=timing_cfg,
            timed_on=timed_on,
            build_threads=build_threads,
            seed=cfg.seed,
            git_commit=env["git_commit"],
            machine_id=env["machine_id"],
            stop_above_recall=float(p3.get("stop_above_recall", 0.995)),
            verbose=not args.quiet,
        )
        run_sweep(specs, **common)

        # Exact search at this size: one point, recall 1.0 by construction.
        sweep_config(index_kind="Flat", build_params={}, search_values=[None], **common)
        del ds, ds_tune, ds_test
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
