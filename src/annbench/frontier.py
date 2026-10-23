"""Pareto frontiers and "QPS at a target recall".

Two things the report depends on and that are easy to get subtly wrong:

*   **The trade-off curve is an envelope, not a parameter sweep.**  An index
    type is represented by the best it can do, pooled over every build
    configuration and every search parameter.  A point belongs on the curve if
    no other point of that index type is at least as accurate *and* at least as
    fast, with at least one of the two strictly better.

*   **Selecting that envelope is itself a tuning decision.**  It is therefore
    done on ``recall_tune`` (the first 5 000 queries), and the selected
    configurations are then reported with ``recall_test`` (the held-out 5 000).
    Picking the envelope directly on the held-out recall would be tuning on the
    evaluation set, which is the exact failure the split exists to prevent.

*   **Interpolating to recall 0.90 / 0.95.**  The table needs QPS at a recall
    the sweep almost never lands on exactly.  We interpolate linearly in
    ``log(QPS)`` between the two frontier points bracketing the target, because
    the trade-off is far closer to exponential than linear in that region.
    When no measured point reaches the target we return NaN and say so rather
    than extrapolating off the end of the sweep.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "pareto_mask",
    "pareto_frontier",
    "qps_at_recall",
    "best_rows_at_recall",
    "target_is_bracketed",
]


def target_is_bracketed(recall, target: float) -> bool:
    """True when the sweep has measured points on *both* sides of ``target``.

    Without a point below the target there is nothing to interpolate towards,
    and :func:`qps_at_recall` falls back to the fastest measured point that
    clears it.  That number is a **lower bound**, not a measurement: the index
    might well be faster at a cheaper setting that was never tried.

    This is not hypothetical.  In the Part 4 experiment at the lowest intrinsic
    dimension, HNSW reaches 0.995 recall at ``efSearch = 10`` -- and hnswlib
    will not search with ``ef < k``, so for k = 10 that is the cheapest legal
    setting in existence.  The sweep cannot locate where recall 0.95 sits
    because the index is already past it at minimum effort.  Plotting that
    point like any other would overstate what was measured, so callers use
    this to mark it.
    """
    recall = np.asarray(recall, dtype=float)
    recall = recall[np.isfinite(recall)]
    if recall.size == 0:
        return False
    return bool((recall < target).any() and (recall >= target).any())


def pareto_mask(recall, qps) -> np.ndarray:
    """Boolean mask of the upper-left (max recall, max QPS) envelope."""
    recall = np.asarray(recall, dtype=float)
    qps = np.asarray(qps, dtype=float)
    n = len(recall)
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not np.isfinite(recall[i]) or not np.isfinite(qps[i]):
            keep[i] = False
            continue
        dominated = (
            (recall >= recall[i])
            & (qps >= qps[i])
            & ((recall > recall[i]) | (qps > qps[i]))
        )
        if dominated.any():
            keep[i] = False
    return keep


def pareto_frontier(df, recall_col: str = "recall_tune", qps_col: str = "qps_median"):
    """Rows of ``df`` on the frontier, sorted by increasing recall."""
    sub = df.dropna(subset=[recall_col, qps_col])
    if sub.empty:
        return sub
    mask = pareto_mask(sub[recall_col].to_numpy(), sub[qps_col].to_numpy())
    return sub[mask].sort_values(recall_col)


def qps_at_recall(
    recall,
    qps,
    target: float,
    method: str = "interp",
) -> float:
    """QPS achievable at ``target`` recall, NaN if the sweep never gets there.

    ``method='interp'`` interpolates log-QPS between the bracketing frontier
    points; ``method='floor'`` returns the fastest measured point that already
    meets the target (always a defensible, slightly pessimistic number).
    """
    recall = np.asarray(recall, dtype=float)
    qps = np.asarray(qps, dtype=float)
    ok = np.isfinite(recall) & np.isfinite(qps)
    recall, qps = recall[ok], qps[ok]
    if recall.size == 0 or recall.max() < target:
        return float("nan")

    above = qps[recall >= target]
    floor = float(above.max()) if above.size else float("nan")
    if method == "floor":
        return floor

    below_mask = recall < target
    if not below_mask.any():
        return floor

    # Nearest point on each side of the target.
    i_lo = int(np.argmax(np.where(below_mask, recall, -np.inf)))
    above_mask = recall >= target
    i_hi = int(np.argmin(np.where(above_mask, recall, np.inf)))
    r_lo, r_hi = recall[i_lo], recall[i_hi]
    q_lo, q_hi = qps[i_lo], qps[i_hi]
    if r_hi <= r_lo or q_lo <= 0 or q_hi <= 0:
        return floor
    w = (target - r_lo) / (r_hi - r_lo)
    interp = float(np.exp(np.log(q_lo) + w * (np.log(q_hi) - np.log(q_lo))))
    # The interpolated value can only be trusted to be between the two
    # measured points; clamp it there.
    return float(np.clip(interp, min(q_lo, q_hi), max(q_lo, q_hi)))


def best_rows_at_recall(df, target: float, recall_col: str = "recall_tune"):
    """The single fastest configuration that reaches ``target`` recall.

    Returns ``None`` when nothing in the sweep reaches it (which is the
    expected outcome for IVF-PQ without re-ranking at recall 0.95, and is
    reported as such rather than hidden).
    """
    sub = df[df[recall_col] >= target]
    if sub.empty:
        return None
    return sub.loc[sub["qps_median"].idxmax()]
