"""Part 2: benchmark HNSW, IVF-Flat and IVF-PQ on both datasets.

For each dataset this builds every configured (index, build-parameter)
combination and sweeps its search parameter until recall saturates, writing one
row per measured point.  Figures and tables come later, from the CSV.

PQ sub-vector counts are validated against the dataset dimension before
anything is built: ``m`` must divide ``d``, so the powers of two that work for
SIFT (d = 128) are illegal for GloVe (d = 100).  The config carries a per
dataset list and this script fails loudly if it is wrong, rather than letting a
FAISS assertion fire ninety minutes into a sweep.
"""

from __future__ import annotations

from ..bench import run_sweep
from ..indexes import divisors
from .common import base_parser, get_dataset, setup, splits_of


def specs_for(cfg, dataset: str, dim: int):
    """Build the (kind, build_params, search_values) list from the config."""
    p2 = cfg["part2"]
    specs = []

    h = p2.get("hnsw")
    if h:
        for M in h["M"]:
            specs.append(
                ("HNSW",
                 {"M": int(M), "ef_construction": int(h.get("ef_construction", 200))},
                 list(h["ef_search"]))
            )

    f = p2.get("ivfflat")
    if f:
        for nlist in f["nlist"]:
            specs.append(
                ("IVF-Flat", {"nlist": int(nlist)},
                 [n for n in f["nprobe"] if n <= int(nlist)])
            )

    q = p2.get("ivfpq")
    if q:
        ms = q["m"][dataset] if isinstance(q["m"], dict) else q["m"]
        for m in ms:
            if dim % int(m):
                raise ValueError(
                    f"IVF-PQ m={m} does not divide d={dim} for dataset {dataset!r}. "
                    f"Legal values: {divisors(dim)}"
                )
            for nlist in q["nlist"]:
                params = {"nlist": int(nlist), "m": int(m), "nbits": int(q.get("nbits", 8))}
                specs.append(("IVF-PQ", params,
                              [n for n in q["nprobe"] if n <= int(nlist)]))
                if q.get("refine"):
                    specs.append(
                        ("IVF-PQ",
                         {**params, "refine": True,
                          "refine_k": int(q.get("refine_k", 100)),
                          "refine_at_k": cfg.k},
                         [n for n in q["nprobe"] if n <= int(nlist)])
                    )
    return specs


def main(argv=None) -> int:
    ap = base_parser("Part 2: HNSW / IVF-Flat / IVF-PQ benchmark on both datasets")
    ap.add_argument("--datasets", nargs="*", default=None, help="subset of configured datasets")
    args = ap.parse_args(argv)
    cfg, writer, env, timing_cfg = setup(args)

    names = args.datasets or list(cfg["datasets"])
    build_threads = int((cfg.get("build", {}) or {}).get("threads", 0))
    timed_on = (cfg.get("timing", {}) or {}).get("time_on", "tune")
    stop_above = float(cfg["part2"].get("stop_above_recall", 0.995))

    for name in names:
        print(f"\n=== Part 2: {name} ===")
        ds = get_dataset(cfg, name, verbose=not args.quiet)
        ds_tune, ds_test = splits_of(ds, cfg)
        print(f"  {ds!r}")
        specs = specs_for(cfg, name, ds.dim)
        print(f"  {len(specs)} build configurations to sweep")
        n = run_sweep(
            specs,
            writer=writer,
            experiment="part2",
            dataset_name=name,
            variant="",
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
            stop_above_recall=stop_above,
            verbose=not args.quiet,
        )
        print(f"  {n} new rows for {name}")
        del ds, ds_tune, ds_test
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
