"""Figures, built only from ``results/results.csv``.

Plotting choices, briefly:

*   **QPS on a log axis.**  The three index types span two to three orders of
    magnitude; on a linear axis everything except the fastest curve is a flat
    line along the bottom.
*   **Recall on a linear axis, clipped to the interesting region.**  The whole
    question is what happens between 0.8 and 1.0.
*   **One marker per measured configuration, lines along the Pareto frontier
    only.**  Dominated points are drawn faintly so a reader can see how much
    of the grid was explored without the frontier being buried in it.
*   **The exact baseline is a horizontal line**, because exact search has a
    single (recall = 1, QPS) point and the line makes the comparison "how much
    faster than exact, and at what cost in recall" readable at a glance.
*   No seaborn, no style sheets: the ground rules ask for the plots to be our
    own, and a default-matplotlib figure is easier for a marker to reproduce.
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .frontier import pareto_frontier  # noqa: E402

__all__ = [
    "plot_tradeoff",
    "plot_scaling",
    "plot_part4",
    "plot_recall_by_bin",
    "INDEX_COLOURS",
]

INDEX_COLOURS = {
    "HNSW": "#1f77b4",
    "IVF-Flat": "#d62728",
    "IVF-PQ": "#2ca02c",
    "IVF-PQ+rerank": "#9467bd",
    "Flat": "#444444",
}
INDEX_MARKERS = {"HNSW": "o", "IVF-Flat": "s", "IVF-PQ": "^", "IVF-PQ+rerank": "v", "Flat": "x"}


def _save(fig, out_path: str) -> str:
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")
    return out_path


def plot_tradeoff(
    df,
    dataset: str,
    out_path: str,
    recall_col: str = "recall_test",
    select_col: str = "recall_tune",
    exact_qps: float | None = None,
    title: str | None = None,
    xlim=None,
):
    """Recall@10 against QPS, one Pareto curve per index type (Part 2)."""
    fig, ax = plt.subplots(figsize=(6.6, 4.5))
    sub = df[(df["dataset"] == dataset) & (df["index_type"] != "Flat")]

    # Fit the x-range to the data rather than assuming everything clears 0.5.
    # An index whose whole curve sits below the default window (IVF-PQ without
    # re-ranking can) would otherwise appear as a legend entry with no line,
    # which reads as a plotting bug instead of as the finding it is.
    if xlim is None:
        lo = float(sub[recall_col].min()) if not sub.empty else 0.0
        xlim = (max(0.0, min(0.5, lo - 0.02)), 1.005)

    for kind, g in sub.groupby("index_type"):
        colour = INDEX_COLOURS.get(kind, None)
        marker = INDEX_MARKERS.get(kind, "o")
        ax.scatter(
            g[recall_col], g["qps_median"], s=10, alpha=0.18, color=colour, linewidths=0
        )
        # Frontier selected on the tuning split, drawn at held-out recall.
        front = pareto_frontier(g, recall_col=select_col).sort_values(recall_col)
        ax.plot(
            front[recall_col],
            front["qps_median"],
            marker=marker,
            ms=4.5,
            lw=1.6,
            color=colour,
            label=kind,
        )

    if exact_qps and np.isfinite(exact_qps):
        ax.axhline(exact_qps, color=INDEX_COLOURS["Flat"], ls="--", lw=1.3)
        ax.text(
            1.0,
            exact_qps * 1.10,
            f"exact (FAISS Flat), {exact_qps:,.0f} QPS",
            fontsize=8,
            ha="right",
            color=INDEX_COLOURS["Flat"],
        )

    for target in (0.90, 0.95):
        ax.axvline(target, color="0.75", lw=0.8, ls=":", zorder=0)

    ax.set_yscale("log")
    ax.set_xlim(*xlim)
    ax.set_xlabel("recall@10 (held-out queries)")
    ax.set_ylabel("queries per second (1 query, 1 thread)")
    ax.set_title(title or f"{dataset}: accuracy / speed trade-off")
    ax.grid(True, which="both", alpha=0.25, lw=0.5)
    ax.legend(frameon=False, fontsize=9, loc="lower left", ncol=2)
    return _save(fig, out_path)


def plot_scaling(df, out_path: str, target: float = 0.95, title: str | None = None):
    """Cost of reaching a recall target as the database grows (Part 3)."""
    from .frontier import qps_at_recall

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.2))
    sub = df[df["experiment"] == "part3"]

    for kind, g in sub.groupby("index_type"):
        sizes, qpss, builds = [], [], []
        for n, gn in g.groupby("n_vectors"):
            if kind == "Flat":
                q = float(gn["qps_median"].max())
            else:
                front = pareto_frontier(gn, recall_col="recall_tune")
                q = qps_at_recall(front["recall_tune"], front["qps_median"], target)
            sizes.append(n)
            qpss.append(q)
            builds.append(float(gn["build_time_s"].min()))
        order = np.argsort(sizes)
        sizes = np.asarray(sizes)[order]
        colour = INDEX_COLOURS.get(kind)
        marker = INDEX_MARKERS.get(kind, "o")
        ax1.plot(sizes, np.asarray(qpss)[order], marker=marker, color=colour, label=kind)
        ax2.plot(sizes, np.asarray(builds)[order], marker=marker, color=colour, label=kind)

    for ax, ylab, ttl in (
        (ax1, f"QPS at recall {target:g}", f"Throughput at recall {target:g}"),
        (ax2, "build time (s)", "Build cost"),
    ):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("database size (vectors)")
        ax.set_ylabel(ylab)
        ax.set_title(ttl)
        ax.grid(True, which="both", alpha=0.25, lw=0.5)
        ax.legend(frameon=False, fontsize=9)

    if title:
        fig.suptitle(title)
    return _save(fig, out_path)


def plot_part4(
    df,
    out_path: str,
    target: float = 0.95,
    variant_label: str = "intrinsic dimension k",
    title: str | None = None,
):
    """The single main figure for Part 4.

    Left: the whole trade-off curve per condition, so the shape of the
    degradation is visible.  Right: the summary statistic — QPS needed to hit
    the recall target, and the recall ceiling actually reached — against the
    property being varied.

    Exact search is pulled out of the per-index loops and drawn as a reference
    on both panels rather than as a fourth competitor, because it is the line
    that decides whether an approximate index is worth using at all: where a
    curve falls below it, the index is slower than brute force *and* less
    accurate, which is the central result of this experiment at high intrinsic
    dimension.
    """
    from .frontier import qps_at_recall

    sub = df[df["experiment"] == "part4"].copy()
    sub["k_intrinsic"] = sub["variant"].str.extract(r"k_intrinsic=(\d+)").astype(float)

    exact = sub[sub["index_type"] == "Flat"]
    sub = sub[sub["index_type"] != "Flat"]
    kinds = sorted(sub["index_type"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    ax1, ax2 = axes

    ks = sorted(sub["k_intrinsic"].dropna().unique())
    shades = np.linspace(0.25, 1.0, len(ks))
    for kind in kinds:
        colour = np.array(matplotlib.colors.to_rgb(INDEX_COLOURS.get(kind, "#333333")))
        for shade, kint in zip(shades, ks):
            g = sub[(sub["index_type"] == kind) & (sub["k_intrinsic"] == kint)]
            if g.empty:
                continue
            front = pareto_frontier(g, recall_col="recall_tune").sort_values("recall_test")
            c = tuple(1 - shade * (1 - colour))
            ax1.plot(
                front["recall_test"],
                front["qps_median"],
                marker=INDEX_MARKERS.get(kind, "o"),
                ms=3.5,
                lw=1.3,
                color=c,
                label=f"{kind}, k={int(kint)}",
            )

    # Exact search: one point per condition, all at recall 1.0. Their spread is
    # pure measurement noise (same n, same d in every condition), so the band
    # they span doubles as a visible noise floor for the whole panel.
    if not exact.empty:
        ax1.scatter(exact["recall_test"], exact["qps_median"], marker="x", s=42,
                    color=INDEX_COLOURS["Flat"], zorder=5, label="exact (brute force)")
        ax1.axhspan(float(exact["qps_median"].min()), float(exact["qps_median"].max()),
                    color=INDEX_COLOURS["Flat"], alpha=0.10, zorder=0)

    ax1.set_xscale("linear")
    ax1.set_yscale("log")
    ax1.set_xlabel("recall@10")
    ax1.set_ylabel("QPS (1 query, 1 thread)")
    ax1.set_title("Trade-off curve per condition")
    ax1.grid(True, which="both", alpha=0.25, lw=0.5)
    ax1.legend(frameon=False, fontsize=7, ncol=2)

    ax3 = ax2.twinx()
    for kind in kinds:
        xs, ys, ceil = [], [], []
        for kint, g in sub[sub["index_type"] == kind].groupby("k_intrinsic"):
            front = pareto_frontier(g, recall_col="recall_tune")
            xs.append(kint)
            ys.append(qps_at_recall(front["recall_tune"], front["qps_median"], target))
            ceil.append(float(g["recall_test"].max()))
        order = np.argsort(xs)
        xs = np.asarray(xs)[order]
        colour = INDEX_COLOURS.get(kind)
        ax2.plot(xs, np.asarray(ys)[order], marker=INDEX_MARKERS.get(kind, "o"),
                 color=colour, label=f"{kind}: QPS @ recall {target:g}")
        # Dashed: the best recall the sweep reached at all.  Where this drops
        # below the target the solid line is necessarily missing — that is the
        # interesting failure, so both belong on one axis pair.
        ax3.plot(xs, np.asarray(ceil)[order], marker=INDEX_MARKERS.get(kind, "o"),
                 ms=3, ls="--", lw=1.0, color=colour, alpha=0.55)
    ax3.set_ylabel("best recall@10 reached (dashed)", fontsize=9)
    ax3.set_ylim(0, 1.02)

    # The decision line: below it an approximate index is both slower than a
    # brute-force scan and less accurate than one, so it has nothing left to
    # offer. Exact search has no parameter to tune, so this is its measured
    # throughput, not an interpolation.
    if not exact.empty:
        ex = exact.groupby("k_intrinsic")["qps_median"].max().sort_index()
        ax2.plot(ex.index, ex.to_numpy(), color=INDEX_COLOURS["Flat"], ls=":", lw=1.8,
                 marker="x", ms=5, label="exact (brute force), recall 1.0")

    ax2.set_xscale("log", base=2)
    ax2.set_yscale("log")
    ax2.set_xlabel(variant_label)
    ax2.set_ylabel(f"QPS at recall {target:g}")
    ax2.set_title(f"Cost of recall {target:g} vs {variant_label}")
    ax2.grid(True, which="both", alpha=0.25, lw=0.5)
    ax2.legend(frameon=False, fontsize=8)

    if title:
        fig.suptitle(title)
    return _save(fig, out_path)


def plot_recall_by_bin(df, out_path: str, bin_col: str = "variant", title: str | None = None):
    """Grouped bars: recall per difficulty bin per index (optional Part 3)."""
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    kinds = sorted(df["index_type"].unique())
    bins = sorted(df[bin_col].unique())
    width = 0.8 / max(1, len(kinds))
    for i, kind in enumerate(kinds):
        vals = [
            float(df[(df["index_type"] == kind) & (df[bin_col] == b)]["recall_test"].max())
            for b in bins
        ]
        ax.bar(np.arange(len(bins)) + i * width, vals, width,
               label=kind, color=INDEX_COLOURS.get(kind))
    ax.set_xticks(np.arange(len(bins)) + 0.4 - width / 2)
    ax.set_xticklabels(bins, fontsize=8)
    ax.set_ylabel("recall@10")
    ax.set_title(title or "Recall by query-difficulty bin")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, axis="y", alpha=0.25, lw=0.5)
    return _save(fig, out_path)
