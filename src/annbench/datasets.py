"""Dataset loading, normalisation, splitting and ground-truth recomputation.

Design decisions worth knowing before reading the code
------------------------------------------------------

*   **We ignore the ``distances`` array shipped in the ann-benchmarks files and
    recompute ground-truth distances ourselves.*  The files store Euclidean
    distance for SIFT and angular distance for GloVe, while FAISS and hnswlib
    both return *squared* L2.  Rather than juggling three units we recompute
    the ground-truth distances in squared L2 from the stored neighbour **ids**,
    which costs one pass over 5 000 x 100 vectors and makes every comparison in
    the project unit-consistent.

*   **GloVe is L2-normalised and then treated as Euclidean.**  For unit vectors
    ``||a - b||^2 = 2 - 2 cos(a, b)``, which is a strictly decreasing function
    of cosine similarity.  Ranking by squared L2 therefore produces exactly the
    angular ranking, so the neighbour ids shipped with the file remain valid
    ground truth after normalisation.  Only the *distance values* change, which
    is precisely why we recompute them.

*   **Subsampling invalidates the shipped ground truth.**  The neighbour ids in
    the file refer to the full 1 M database.  Any experiment on a subset (the
    Part 3 scaling study, and every ``--quick`` run) must recompute exact
    ground truth against that subset, with ids in the subset's own index space.
    :func:`subsample` does this and refuses to silently reuse stale ids.

*   **Query split.**  The first 5 000 queries are the tuning split and the last
    5 000 are the held-out reporting split, applied to queries, neighbour ids
    and distances with the same slice.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace

import numpy as np

__all__ = [
    "Dataset",
    "load_hdf5",
    "compute_ground_truth",
    "subsample",
    "split_queries",
    "verify_ground_truth",
    "l2_normalize",
]


@dataclass
class Dataset:
    """A database, a query set and exact ground truth, all in squared L2."""

    name: str
    train: np.ndarray  # (n, d) float32, C-contiguous
    queries: np.ndarray  # (nq, d) float32, C-contiguous
    gt_ids: np.ndarray  # (nq, kgt) int64, ascending distance
    gt_dists: np.ndarray  # (nq, kgt) float64, squared L2
    normalized: bool = False
    note: str = ""

    @property
    def n(self) -> int:
        return int(self.train.shape[0])

    @property
    def dim(self) -> int:
        return int(self.train.shape[1])

    @property
    def n_queries(self) -> int:
        return int(self.queries.shape[0])

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"Dataset(name={self.name!r}, n={self.n}, dim={self.dim}, "
            f"n_queries={self.n_queries}, normalized={self.normalized})"
        )


def l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Row-wise L2 normalisation, safe against zero rows."""
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return np.ascontiguousarray(x / norms, dtype=np.float32)


def compute_ground_truth(
    train: np.ndarray,
    queries: np.ndarray,
    k: int = 100,
    threads: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact k nearest neighbours by brute force, in squared L2.

    This is *not* a timed measurement, so it runs multi-threaded and batched.
    Returns ``(ids int64, dists float64)``.
    """
    import faiss

    if threads and threads > 0:
        faiss.omp_set_num_threads(threads)
    else:
        faiss.omp_set_num_threads(os.cpu_count() or 1)

    train = np.ascontiguousarray(train, dtype=np.float32)
    queries = np.ascontiguousarray(queries, dtype=np.float32)
    index = faiss.IndexFlatL2(train.shape[1])
    index.add(train)
    k = min(k, train.shape[0])
    dists, ids = index.search(queries, k)
    return ids.astype(np.int64), dists.astype(np.float64)


def load_hdf5(
    path: str,
    name: str | None = None,
    normalize: bool = False,
    kgt: int = 100,
    recompute_gt_distances: bool = True,
) -> Dataset:
    """Load an ann-benchmarks HDF5 file into a :class:`Dataset`.

    The file provides ``train``, ``test``, ``neighbors`` and ``distances``.
    We keep the first three and recompute the fourth in squared L2 (see the
    module docstring for why).
    """
    import h5py

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Run `bash scripts/download_data.sh` (or see the "
            f"README) to fetch the ann-benchmarks files before benchmarking."
        )

    with h5py.File(path, "r") as f:
        train = np.ascontiguousarray(f["train"][:], dtype=np.float32)
        queries = np.ascontiguousarray(f["test"][:], dtype=np.float32)
        gt_ids = np.asarray(f["neighbors"][:], dtype=np.int64)
        stored_dists = np.asarray(f["distances"][:], dtype=np.float64)

    if normalize:
        train = l2_normalize(train)
        queries = l2_normalize(queries)

    kgt = min(kgt, gt_ids.shape[1])
    gt_ids = np.ascontiguousarray(gt_ids[:, :kgt])

    if recompute_gt_distances:
        from .recall import true_distances

        gt_dists = true_distances(queries, train, gt_ids)
        note = "gt distances recomputed as squared L2 from stored neighbour ids"
    else:  # pragma: no cover - kept for the unit comparison in the report
        gt_dists = stored_dists[:, :kgt]
        note = "gt distances as stored in the file (NOT squared L2)"

    return Dataset(
        name=name or os.path.basename(path),
        train=train,
        queries=queries,
        gt_ids=gt_ids,
        gt_dists=gt_dists,
        normalized=normalize,
        note=note,
    )


def subsample(
    ds: Dataset,
    n: int,
    seed: int = 0,
    kgt: int | None = None,
    threads: int = 0,
) -> Dataset:
    """Take a seeded random subset of the database and recompute ground truth.

    Ids in the returned dataset index into the *subset*.  Reusing the original
    neighbour ids here would be a silent correctness bug, so they are discarded
    rather than remapped.
    """
    if n >= ds.n:
        return ds
    rng = np.random.default_rng(seed)
    idx = rng.choice(ds.n, size=n, replace=False)
    idx.sort()  # keeps memory access roughly sequential; does not bias the draw
    train = np.ascontiguousarray(ds.train[idx])
    kgt = kgt or ds.gt_ids.shape[1]
    gt_ids, gt_dists = compute_ground_truth(train, ds.queries, k=kgt, threads=threads)
    return replace(
        ds,
        name=f"{ds.name}@{n}",
        train=train,
        gt_ids=gt_ids,
        gt_dists=gt_dists,
        note=f"random subset of {ds.n} (seed={seed}); ground truth recomputed exactly",
    )


def split_queries(ds: Dataset, which: str, half: int = 5000) -> Dataset:
    """Return the tuning (first ``half``) or held-out (last ``half``) queries."""
    if which not in {"tune", "test", "all"}:
        raise ValueError("which must be 'tune', 'test' or 'all'")
    if which == "all":
        return ds
    nq = ds.n_queries
    half = min(half, nq // 2)
    sl = slice(0, half) if which == "tune" else slice(nq - half, nq)
    return replace(
        ds,
        queries=np.ascontiguousarray(ds.queries[sl]),
        gt_ids=np.ascontiguousarray(ds.gt_ids[sl]),
        gt_dists=np.ascontiguousarray(ds.gt_dists[sl]),
        note=f"{ds.note} | query split={which} ({half} queries)",
    )


def verify_ground_truth(ds: Dataset, k: int = 10, rtol: float = 1e-5) -> dict:
    """Check our exact search against the neighbour ids shipped with the file.

    Perfect agreement is *not* expected: where several vectors are equidistant
    at the k-th position, any of them is correct and different implementations
    break the tie differently.  The dictionary returned reports both the naive
    agreement and the tie-aware recall, so the report can show that every
    disagreement is a tie.
    """
    from .recall import recall_at_k, true_distances

    ids, _ = compute_ground_truth(ds.train, ds.queries, k=k)
    found_d = true_distances(ds.queries, ds.train, ids)

    naive = float(
        np.mean(
            [
                len(set(ids[i, :k].tolist()) & set(ds.gt_ids[i, :k].tolist())) / k
                for i in range(ds.n_queries)
            ]
        )
    )
    tie_aware = recall_at_k(ids, ds.gt_ids, ds.gt_dists, found_d, k=k, rtol=rtol)
    per_q = recall_at_k(ids, ds.gt_ids, ds.gt_dists, found_d, k=k, rtol=rtol, per_query=True)
    exact_match = int(
        sum(
            1
            for i in range(ds.n_queries)
            if set(ids[i, :k].tolist()) == set(ds.gt_ids[i, :k].tolist())
        )
    )
    return {
        "dataset": ds.name,
        "k": k,
        "n_queries": ds.n_queries,
        "naive_set_recall": naive,
        "tie_aware_recall": tie_aware,
        "queries_with_identical_sets": exact_match,
        "queries_with_differing_sets": ds.n_queries - exact_match,
        "queries_below_tie_aware_1.0": int((per_q < 1.0).sum()),
    }
