"""Tie-aware recall@k.

Why this module is hand-written (assignment ground rule) and why it is not the
one-liner ``len(set(found) & set(gt)) / k``:

1.  **Ties at the k-th neighbour.**  If several database vectors sit at exactly
    the distance of the k-th true neighbour, any of them is an equally correct
    answer.  A naive set intersection punishes an index for returning a
    different-but-equidistant vector, which is not a mistake.  We therefore
    accept a returned id if it is in the ground-truth top-k set *or* if its
    true distance is within a relative tolerance of the k-th ground-truth
    distance.

2.  **The distances used for the tie test must be exact.**  IVF-PQ reports
    *approximate* distances computed from its quantisation codes.  Using those
    for the tie test would let a badly quantised vector sneak in under the
    threshold and inflate recall.  :func:`true_distances` recomputes the
    distance of every returned id against the original vectors, and that is
    what the tie test consumes.

3.  **Degenerate output.**  FAISS pads with ``-1`` when it finds fewer than k
    candidates, and a buggy index can return the same id twice.  Neither may
    count towards recall.

Everything in this project works in **squared Euclidean distance**, which is
what both FAISS (``IndexFlatL2``) and hnswlib (``space='l2'``) return natively.
Keeping one unit end-to-end removes a whole class of silent factor-of-two and
square-root bugs.
"""

from __future__ import annotations

import numpy as np

__all__ = ["true_distances", "recall_at_k", "tie_counts"]


def true_distances(
    queries: np.ndarray,
    train: np.ndarray,
    ids: np.ndarray,
    chunk: int = 1024,
) -> np.ndarray:
    """Exact squared-L2 distance from each query to each of its returned ids.

    Parameters
    ----------
    queries : (nq, d) array of query vectors.
    train   : (n, d) array of database vectors.
    ids     : (nq, k) integer array of returned ids; ``-1`` marks "no result".

    Returns
    -------
    (nq, k) float64 array.  Entries corresponding to ``-1`` ids are ``+inf``
    so that they can never pass a tie test.
    """
    queries = np.asarray(queries)
    train = np.asarray(train)
    ids = np.asarray(ids)
    if ids.ndim == 1:
        ids = ids[None, :]
    nq, k = ids.shape
    out = np.full((nq, k), np.inf, dtype=np.float64)

    for start in range(0, nq, chunk):
        stop = min(start + chunk, nq)
        block = ids[start:stop]
        valid = block >= 0
        safe = np.where(valid, block, 0)
        # (b, k, d) gather.  Cast to float64 before subtracting: float32
        # accumulation of 128 squared terms loses enough precision to make the
        # tie test flaky at rtol=1e-5.
        vecs = train[safe].astype(np.float64, copy=False)
        diff = vecs - queries[start:stop, None, :].astype(np.float64, copy=False)
        d = np.einsum("ijk,ijk->ij", diff, diff)
        out[start:stop] = np.where(valid, d, np.inf)
    return out


def recall_at_k(
    found_ids: np.ndarray,
    gt_ids: np.ndarray,
    gt_dists: np.ndarray,
    found_true_dists: np.ndarray | None = None,
    k: int = 10,
    rtol: float = 1e-5,
    atol: float = 0.0,
    per_query: bool = False,
):
    """Mean recall@k with ties at the k-th neighbour treated as correct.

    Parameters
    ----------
    found_ids        : (nq, >=k) ids returned by the index, ``-1`` allowed.
    gt_ids           : (nq, >=k) ground-truth neighbour ids.
    gt_dists         : (nq, >=k) ground-truth *squared L2* distances, ascending.
    found_true_dists : (nq, >=k) exact squared-L2 distances of ``found_ids``,
                       from :func:`true_distances`.  If ``None`` the tie test is
                       skipped and the result is the plain set-intersection
                       recall (used only by the unit tests, never by the
                       benchmark driver).
    rtol, atol       : a returned vector counts as a tie if its true distance is
                       ``<= gt_dists[:, k-1] * (1 + rtol) + atol``.

    Returns
    -------
    float, or (nq,) array of per-query recall if ``per_query`` is True.
    """
    found_ids = np.atleast_2d(np.asarray(found_ids))
    gt_ids = np.atleast_2d(np.asarray(gt_ids))
    gt_dists = np.atleast_2d(np.asarray(gt_dists, dtype=np.float64))

    nq = found_ids.shape[0]
    if gt_ids.shape[0] != nq or gt_dists.shape[0] != nq:
        raise ValueError(
            f"query-count mismatch: found={found_ids.shape[0]} "
            f"gt_ids={gt_ids.shape[0]} gt_dists={gt_dists.shape[0]}"
        )
    if gt_ids.shape[1] < 1:
        raise ValueError("ground truth has no neighbours")
    if found_true_dists is not None:
        found_true_dists = np.atleast_2d(np.asarray(found_true_dists, dtype=np.float64))
        if found_true_dists.shape != found_ids.shape:
            raise ValueError("found_true_dists must have the same shape as found_ids")

    n_found = min(k, found_ids.shape[1])
    out = np.zeros(nq, dtype=np.float64)

    for i in range(nq):
        gt_row = gt_ids[i, :k]
        gt_valid = gt_row[gt_row >= 0]
        n_gt = int(gt_valid.size)
        if n_gt == 0:
            # No ground truth for this query: nothing can be wrong.
            out[i] = 1.0
            continue
        gt_set = set(int(x) for x in gt_valid)

        # Distance of the last available true neighbour (the k-th when the
        # ground truth is complete).  This is the tie threshold.
        thr_idx = min(n_gt, gt_dists.shape[1]) - 1
        threshold = gt_dists[i, thr_idx] * (1.0 + rtol) + atol

        seen: set[int] = set()
        hits = 0
        for j in range(n_found):
            fid = int(found_ids[i, j])
            if fid < 0 or fid in seen:
                continue
            seen.add(fid)
            if fid in gt_set:
                hits += 1
            elif found_true_dists is not None and found_true_dists[i, j] <= threshold:
                hits += 1
        out[i] = min(hits, n_gt) / n_gt

    return out if per_query else float(out.mean())


def tie_counts(gt_dists: np.ndarray, k: int = 10, rtol: float = 1e-5) -> np.ndarray:
    """Per query, how many of the top-k true neighbours are tied with the k-th.

    Used in the report to show that disagreements between our exact search and
    the ground truth shipped with the dataset are ties rather than errors.
    """
    gt_dists = np.atleast_2d(np.asarray(gt_dists, dtype=np.float64))
    lo = gt_dists[:, k - 1] * (1.0 - rtol)
    return (gt_dists[:, :k] >= lo[:, None]).sum(axis=1)
