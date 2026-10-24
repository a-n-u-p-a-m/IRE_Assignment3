"""Regenerate every table in the report from ``results/results.csv``.

Three tables are produced, each as Markdown (for reading) and LaTeX (for
pasting into the report):

``part2_qps_at_recall``  queries per second at recall 0.90 and 0.95, per index
                         per dataset.  The frontier is selected on the tuning
                         split; the recall shown is the held-out recall of the
                         selected configuration.  An index that never reaches a
                         target gets "not reached" and its best recall, which is
                         the expected and interesting outcome for IVF-PQ.

``part2_cost_at_recall`` build time, index size and bytes per vector of the
                         configuration chosen at recall 0.95.

``part3_scaling``        QPS at recall 0.95 and build time per database size.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np

from ..frontier import (
    best_rows_at_recall,
    pareto_frontier,
    qps_at_recall,
    target_is_bracketed,
)
from ..results import load_results, one_run_per_dataset


def _fmt(x, nd=1):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return f"{x:,.{nd}f}"


def _write(md_path: str, title: str, header: list[str], rows: list[list[str]], preamble: str = ""):
    os.makedirs(os.path.dirname(os.path.abspath(md_path)) or ".", exist_ok=True)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"# {title}\n\n")
        if preamble:
            f.write(preamble.rstrip() + "\n\n")
        f.write("| " + " | ".join(header) + " |\n")
        f.write("|" + "|".join(["---"] * len(header)) + "|\n")
        for r in rows:
            f.write("| " + " | ".join(str(c) for c in r) + " |\n")

    tex_path = os.path.splitext(md_path)[0] + ".tex"
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write("\\begin{tabular}{" + "l" * len(header) + "}\n\\hline\n")
        f.write(" & ".join(h.replace("_", r"\_") for h in header) + " \\\\\n\\hline\n")
        for r in rows:
            f.write(" & ".join(str(c).replace("_", r"\_").replace("—", "--") for c in r) + " \\\\\n")
        f.write("\\hline\n\\end{tabular}\n")
    print(f"  wrote {md_path} and {tex_path}")


def table_qps_at_recall(df, out_dir: str, targets=(0.90, 0.95)):
    p2 = df[(df["experiment"] == "part2") & (df["index_type"] != "Flat")]
    rows = []
    for dataset in sorted(p2["dataset"].unique()):
        exact = df[(df["experiment"] == "part1") & (df["dataset"] == dataset)]
        exact_qps = float(exact["qps_median"].iloc[0]) if len(exact) else float("nan")
        for kind in sorted(p2[p2["dataset"] == dataset]["index_type"].unique()):
            g = p2[(p2["dataset"] == dataset) & (p2["index_type"] == kind)]
            front = pareto_frontier(g, recall_col="recall_tune")
            cells = []
            for t in targets:
                q = qps_at_recall(front["recall_tune"], front["qps_median"], t)
                if np.isfinite(q):
                    best = best_rows_at_recall(g, t)
                    speed = q / exact_qps if np.isfinite(exact_qps) and exact_qps else float("nan")
                    cfg = json.loads(best["params_json"]) if best is not None else {}
                    cfg_s = ",".join(f"{k}={v}" for k, v in sorted(cfg.items()))
                    # A target met at the cheapest setting in the sweep was
                    # never bracketed, so the figure is a lower bound.
                    mark = "" if target_is_bracketed(g["recall_tune"], t) else " (≥)"
                    cells += [_fmt(q) + mark, f"{_fmt(speed)}x", cfg_s]
                else:
                    cells += ["not reached", "—", f"best recall_tune {g['recall_tune'].max():.4f}"]
            rows.append([dataset, kind] + cells)

    header = ["dataset", "index"]
    for t in targets:
        header += [f"QPS @ r={t:g}", f"vs exact", f"config @ r={t:g}"]
    _write(
        os.path.join(out_dir, "part2_qps_at_recall.md"),
        "Part 2: throughput at fixed recall",
        header,
        rows,
        preamble=(
            "One query at a time, one thread, median of repeated runs. QPS is interpolated "
            "log-linearly between the two Pareto points bracketing the target. The Pareto "
            "frontier is selected on the tuning queries; `config` is the fastest configuration "
            "that reaches the target. `vs exact` compares against single-query exact search "
            "from Part 1. A value marked (≥) is a lower bound: the index already cleared the "
            "target at the cheapest setting in the sweep, so where it actually crosses was "
            "never measured."
        ),
    )


def table_cost_at_recall(df, out_dir: str, target=0.95):
    p2 = df[(df["experiment"] == "part2") & (df["index_type"] != "Flat")]
    rows = []
    for dataset in sorted(p2["dataset"].unique()):
        flat = df[(df["experiment"] == "part1") & (df["dataset"] == dataset)]
        if len(flat):
            r = flat.iloc[0]
            rows.append([dataset, "Flat (exact)", "—", _fmt(r["build_time_s"]),
                         _fmt(r["index_bytes"] / 1e6), _fmt(r["bytes_per_vector"]), "1.000"])
        for kind in sorted(p2[p2["dataset"] == dataset]["index_type"].unique()):
            g = p2[(p2["dataset"] == dataset) & (p2["index_type"] == kind)]
            best = best_rows_at_recall(g, target)
            if best is None:
                rows.append([dataset, kind, f"never reaches {target:g} on the tuning split", "—", "—", "—",
                             f"{g['recall_tune'].max():.4f} (tune)"])
                continue
            cfg = json.loads(best["params_json"])
            cfg_s = ",".join(f"{k}={v}" for k, v in sorted(cfg.items()))
            rows.append([
                dataset, kind, cfg_s, _fmt(best["build_time_s"]),
                _fmt(best["index_bytes"] / 1e6), _fmt(best["bytes_per_vector"]),
                f"{best['recall_test']:.3f}",
            ])
    _write(
        os.path.join(out_dir, "part2_cost_at_recall.md"),
        f"Part 2: build cost and memory at recall {target:g}",
        ["dataset", "index", "config", "build time (s)", "size (MB)", "bytes/vector", "recall@10 (held-out)"],
        rows,
        preamble=(
            "Size is the serialized index on disk, which for these index types is within a "
            "few percent of resident memory. Builds use all cores (search does not); the "
            "thread count is recorded per row in results.csv."
        ),
    )


def table_scaling(df, out_dir: str, target=0.95):
    p3 = df[df["experiment"] == "part3"]
    if p3.empty:
        return
    rows = []
    for n in sorted(p3["n_vectors"].unique()):
        for kind in sorted(p3[p3["n_vectors"] == n]["index_type"].unique()):
            g = p3[(p3["n_vectors"] == n) & (p3["index_type"] == kind)]
            if kind == "Flat":
                q, note = float(g["qps_median"].max()), "recall 1.0 by construction"
            else:
                front = pareto_frontier(g, recall_col="recall_tune")
                q = qps_at_recall(front["recall_tune"], front["qps_median"], target)
                note = "" if np.isfinite(q) else f"best recall {g['recall_test'].max():.3f}"
            rows.append([f"{int(n):,}", kind, _fmt(q), _fmt(float(g['build_time_s'].min())), note])
    _write(
        os.path.join(out_dir, "part3_scaling.md"),
        f"Part 3: throughput at recall {target:g} vs database size",
        ["n vectors", "index", f"QPS @ recall {target:g}", "build time (s)", "note"],
        rows,
        preamble="nlist is retuned per size as round(4*sqrt(n)) to the nearest power of two.",
    )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Build all tables from results.csv")
    ap.add_argument("--results", default="results/results.csv")
    ap.add_argument("--tables", default="results/tables")
    ap.add_argument("--target", type=float, default=0.95)
    args = ap.parse_args(argv)

    df = load_results(args.results)
    df = one_run_per_dataset(df)
    os.makedirs(args.tables, exist_ok=True)
    table_qps_at_recall(df, args.tables)
    table_cost_at_recall(df, args.tables, target=args.target)
    table_scaling(df, args.tables, target=args.target)
    print(f"\ntables written to {args.tables}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
