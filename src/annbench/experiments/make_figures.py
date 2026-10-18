"""Regenerate every figure in the report from ``results/results.csv``.

This script reads nothing but the CSV.  If it can produce the figures, the
numbers in the report are reproducible by definition; if a figure cannot be
produced, the measurement behind it is missing rather than silently stale.
"""

from __future__ import annotations

import argparse
import os

from ..plots import plot_part4, plot_scaling, plot_tradeoff
from ..results import load_results


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Build all figures from results.csv")
    ap.add_argument("--results", default="results/results.csv")
    ap.add_argument("--figures", default="results/figures")
    ap.add_argument("--target", type=float, default=0.95)
    args = ap.parse_args(argv)

    df = load_results(args.results)
    os.makedirs(args.figures, exist_ok=True)
    made = []

    # --- Part 2: one trade-off figure per dataset ----------------------
    p2 = df[df["experiment"] == "part2"]
    for dataset in sorted(p2["dataset"].unique()):
        exact = df[(df["experiment"] == "part1") & (df["dataset"] == dataset)]
        exact_qps = float(exact["qps_median"].iloc[0]) if len(exact) else None
        made.append(
            plot_tradeoff(
                p2, dataset,
                os.path.join(args.figures, f"part2_tradeoff_{dataset}.png"),
                exact_qps=exact_qps,
                title=f"{dataset}: recall@10 vs throughput",
            )
        )

    # --- Part 3: scaling ----------------------------------------------
    if (df["experiment"] == "part3").any():
        made.append(
            plot_scaling(df, os.path.join(args.figures, "part3_scaling.png"),
                         target=args.target,
                         title="Part 3: cost of recall 0.95 as the database grows")
        )

    # --- Part 4: the single main figure --------------------------------
    p4 = df[df["experiment"] == "part4"]
    if not p4.empty:
        dim = int(p4["dim"].dropna().iloc[0])
        n = int(p4["n_vectors"].dropna().iloc[0])
        made.append(
            plot_part4(df, os.path.join(args.figures, "part4_intrinsic_dimension.png"),
                       target=args.target,
                       title=(f"Part 4: intrinsic dimensionality at fixed ambient "
                              f"d={dim}, n={n:,}"))
        )

    print(f"\n{len(made)} figures written to {args.figures}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
