"""Part 4: how intrinsic dimensionality breaks an index, at fixed ambient d.

The experiment is described in ``annbench/synthetic.py`` and the hypothesis is
in ``hypothesis.md``, committed before this was first run.

Experimental discipline required by the brief, and how it is satisfied:

*   **One thing varies.**  Ambient dimension, database size, query count, noise
    level, seed, index parameters and the timing protocol are identical across
    conditions.  Only ``k_intrinsic`` changes.
*   **At least three settings plus an unchanged control.**  The config lists
    k = 2, 8, 32 and 128; at k = 128 = d the generator produces isotropic
    Gaussian data, which is the control.
*   **At least two of the three indexes.**  HNSW and IVF-Flat are always run;
    IVF-PQ can be added from the config to show the quantisation ceiling moving
    as well.
*   **Seeded generator.**  Same seed, byte-identical data on any machine.

Relative contrast is recorded per condition so the report can show that the
generator actually moved the data into the "nothing is distinctly closest"
regime, rather than asserting it.
"""

from __future__ import annotations

import json
import os

import numpy as np

from ..bench import run_sweep
from ..datasets import split_queries
from ..synthetic import make_subspace_dataset, relative_contrast
from .common import base_parser, setup


def main(argv=None) -> int:
    ap = base_parser("Part 4: intrinsic dimensionality at fixed ambient dimension")
    ap.add_argument("--tables", default="results/tables")
    args = ap.parse_args(argv)
    cfg, writer, env, timing_cfg = setup(args)
    os.makedirs(args.tables, exist_ok=True)

    p4 = cfg["part4"]
    n = int(p4["n"])
    n_queries = int(p4["n_queries"])
    dim = int(p4["dim"])
    # Renamed from `noise` when the generator was corrected to hold signal
    # power constant: the value is now a fraction of the signal norm, not a
    # per-coordinate sigma, so an old config must fail loudly rather than be
    # reinterpreted silently.
    if "noise" in p4 and "noise_frac" not in p4:
        raise KeyError(
            "part4.noise was replaced by part4.noise_frac (noise norm as a "
            "fraction of the signal norm). Update the config; the two are not "
            "the same quantity."
        )
    noise_frac = float(p4["noise_frac"])
    ks = [int(x) for x in p4["k_intrinsic"]]
    build_threads = int((cfg.get("build", {}) or {}).get("threads", 0))
    timed_on = (cfg.get("timing", {}) or {}).get("time_on", "tune")

    if n > 500_000:
        raise ValueError("the brief caps Part 4 at 500K synthetic vectors")
    if dim not in ks:
        print(f"  note: no control condition (k_intrinsic == dim == {dim}) in {ks}")

    contrast_rows = []
    for k_int in ks:
        is_control = k_int == dim
        tag = f"k_intrinsic={k_int}" + (" (control)" if is_control else "")
        print(f"\n=== Part 4: {tag} ===")
        ds = make_subspace_dataset(
            n=n, n_queries=n_queries, dim=dim, k_intrinsic=k_int,
            noise_frac=noise_frac, seed=cfg.seed, kgt=max(100, cfg.k),
        )
        print(f"  {ds!r}")

        rc = relative_contrast(ds, sample=int(p4.get("contrast_sample", 10000)), seed=cfg.seed)
        # How far the true neighbour sits above the noise floor. Below ~1 the
        # nearest neighbour is decided by the noise rather than by the
        # subspace, which is a real property of low k at large n and is
        # reported rather than tuned away.
        nn_dist = float(np.sqrt(ds.gt_dists[:, 0]).mean())
        noise_floor = noise_frac * np.sqrt(2.0)
        contrast_rows.append(
            dict(k_intrinsic=k_int, control=is_control,
                 rc_mean=float(np.mean(rc)), rc_median=float(np.median(rc)),
                 rc_p10=float(np.percentile(rc, 10)), rc_p90=float(np.percentile(rc, 90)),
                 nn_dist=nn_dist, noise_floor=float(noise_floor),
                 nn_over_floor=nn_dist / noise_floor if noise_floor else float("inf"))
        )
        msg = (f"  relative contrast: median {np.median(rc):.3f} "
               f"(p10 {np.percentile(rc, 10):.3f}, p90 {np.percentile(rc, 90):.3f})")
        if noise_floor:
            msg += f"; NN distance {nn_dist:.4f} = {nn_dist / noise_floor:.2f}x the noise floor"
        print(msg)

        half = n_queries // 2
        ds_tune = split_queries(ds, "tune", half=half)
        ds_test = split_queries(ds, "test", half=half)

        specs = []
        for M in p4.get("hnsw", {}).get("M", [16]):
            specs.append(("HNSW",
                          {"M": int(M),
                           "ef_construction": int(p4.get("hnsw", {}).get("ef_construction", 200))},
                          list(p4.get("hnsw", {}).get("ef_search", [10, 20, 40, 80, 160, 320]))))
        for nlist in p4.get("ivfflat", {}).get("nlist", [1024]):
            specs.append(("IVF-Flat", {"nlist": int(nlist)},
                          [p for p in p4.get("ivfflat", {}).get(
                              "nprobe", [1, 2, 4, 8, 16, 32, 64, 128]) if p <= int(nlist)]))
        if p4.get("ivfpq"):
            q = p4["ivfpq"]
            for m in q["m"]:
                for nlist in q["nlist"]:
                    specs.append(("IVF-PQ",
                                  {"nlist": int(nlist), "m": int(m), "nbits": int(q.get("nbits", 8))},
                                  [p for p in q["nprobe"] if p <= int(nlist)]))

        run_sweep(
            specs,
            writer=writer,
            experiment="part4",
            dataset_name="synthetic",
            variant=f"k_intrinsic={k_int}",
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
            stop_above_recall=float(p4.get("stop_above_recall", 0.995)),
            verbose=not args.quiet,
        )
        del ds, ds_tune, ds_test

    path = os.path.join(args.tables, "part4_contrast.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Part 4: relative contrast per condition\n\n")
        f.write(
            "Relative contrast is the mean distance from a query to a random sample of the\n"
            "database divided by its distance to the true nearest neighbour. Values near 1\n"
            "mean nothing is distinctly closest, which is the regime where partition-based\n"
            "pruning stops working.\n\n"
        )
        f.write(
            "`NN / noise floor` is the mean true nearest-neighbour distance divided by\n"
            "the expected distance between two noise vectors. Below ~1 the nearest\n"
            "neighbour is determined by the noise rather than by the subspace, which\n"
            "happens at the lowest k because many points packed into few effective\n"
            "dimensions sit closer together than any noise floor that keeps the data\n"
            "full-rank.\n\n"
        )
        f.write(
            "| k_intrinsic | control | median RC | p10 | p90 | NN dist | NN / noise floor |\n"
            "|---|---|---|---|---|---|---|\n"
        )
        for r in contrast_rows:
            f.write(
                f"| {r['k_intrinsic']} | {'yes' if r['control'] else ''} | "
                f"{r['rc_median']:.3f} | {r['rc_p10']:.3f} | {r['rc_p90']:.3f} | "
                f"{r['nn_dist']:.4f} | {r['nn_over_floor']:.2f} |\n"
            )
    with open(os.path.join(args.tables, "part4_contrast.json"), "w", encoding="utf-8") as f:
        json.dump(contrast_rows, f, indent=2)
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
