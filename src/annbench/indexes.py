"""Uniform wrappers around FAISS and hnswlib indexes.

Every index exposes the same five things, which is what lets the benchmark
driver be a single loop rather than three:

    build(train) -> build_time_s
    set_search_param(value)        # efSearch or nprobe
    search_one(vector, k)          # the timed path, single query
    search(matrix, k)              # the batched path, used for recall
    serialized_bytes()             # index size on disk

Notes on the choices made here
------------------------------

*   **Squared L2 everywhere.**  ``IndexFlatL2`` and hnswlib's ``space='l2'``
    both return squared distances, so no conversion happens anywhere.

*   **HNSW comes from hnswlib, IVF from FAISS.**  Both are permitted.  hnswlib
    is the reference implementation people actually benchmark against and
    exposes ``ef`` cleanly at query time; FAISS owns the IVF family.

*   **IVF training is seeded and uses a bounded subsample.**  k-means on a
    million vectors is wasteful: FAISS itself only needs a few hundred points
    per centroid.  We draw ``min(n, points_per_centroid * nlist)`` vectors with
    a fixed seed, so the quantiser is reproducible across runs.

*   **``m`` must divide ``d`` for IVF-PQ.**  SIFT is 128-dimensional so
    ``m`` in {8, 16, 32, 64} works; GloVe is **100**-dimensional, so the usual
    powers of two are illegal and ``m`` must come from {4, 10, 20, 25, 50}.
    This is checked with a readable error rather than left to a FAISS assert.

*   **Build threading is configurable and recorded.**  Search is always single
    threaded, but building HNSW on 1.18 M vectors single-threaded would take
    hours.  Builds therefore use all cores by default; ``build_threads`` is
    written to ``results.csv`` and the resulting graph is not bit-reproducible,
    which the README and report state explicitly.
"""

from __future__ import annotations

import json
import os
import tempfile
import time

import numpy as np

__all__ = [
    "FlatIndex", "HnswIndex", "IVFFlatIndex", "IVFPQIndex",
    "make_index", "divisors", "index_type_name",
]


def index_type_name(kind: str, params: dict) -> str:
    """The label a configuration is reported under.

    IVF-PQ with exact re-ranking is a different index from plain IVF-PQ: it
    keeps the original vectors in memory, so it has a different memory cost and
    a different recall ceiling. Pooling the two on one Pareto frontier would
    let the re-ranked variant's accuracy be credited to PQ's memory footprint,
    which is exactly the trade-off Part 2 is supposed to expose. They are
    therefore reported as separate index types.
    """
    if kind == "IVF-PQ" and params.get("refine"):
        return "IVF-PQ+rerank"
    return kind


def divisors(d: int, lo: int = 2, hi: int | None = None) -> list[int]:
    """Legal PQ sub-vector counts for dimension ``d``."""
    hi = hi or d
    return [m for m in range(lo, hi + 1) if d % m == 0]


class BaseIndex:
    kind = "base"
    search_param_name = "none"

    def __init__(self, dim: int, seed: int = 42, build_threads: int = 0, **params):
        self.dim = int(dim)
        self.seed = int(seed)
        self.build_threads = int(build_threads) or (os.cpu_count() or 1)
        self.params = dict(params)
        self.build_time_s = float("nan")
        self.n_vectors = 0
        self._search_param = None

    # -- introspection -------------------------------------------------
    @property
    def params_json(self) -> str:
        return json.dumps(self.params, sort_keys=True)

    @property
    def search_param(self):
        return self._search_param

    def label(self) -> str:
        bits = ",".join(f"{k}={v}" for k, v in sorted(self.params.items()))
        return f"{self.kind}({bits})" if bits else self.kind

    # -- interface -----------------------------------------------------
    def build(self, train: np.ndarray) -> float:  # pragma: no cover - abstract
        raise NotImplementedError

    def set_search_param(self, value):  # pragma: no cover - abstract
        raise NotImplementedError

    def search_one(self, q: np.ndarray, k: int):  # pragma: no cover - abstract
        raise NotImplementedError

    def search(self, qs: np.ndarray, k: int, threads: int = 0):  # pragma: no cover
        raise NotImplementedError

    def serialized_bytes(self) -> int:  # pragma: no cover - abstract
        raise NotImplementedError

    def bytes_per_vector(self) -> float:
        return self.serialized_bytes() / max(1, self.n_vectors)


# ----------------------------------------------------------------------
# FAISS-backed indexes
# ----------------------------------------------------------------------
class _FaissIndex(BaseIndex):
    def __init__(self, dim, seed=42, build_threads=0, **params):
        super().__init__(dim, seed=seed, build_threads=build_threads, **params)
        self.index = None

    def search_one(self, q, k):
        q = np.ascontiguousarray(q.reshape(1, -1), dtype=np.float32)
        d, i = self.index.search(q, k)
        return i[0], d[0]

    def search(self, qs, k, threads: int = 0):
        """Batched search.  Used for *correctness* only, never for timing, so
        it may use every core; the timed path is :meth:`search_one`."""
        import faiss

        faiss.omp_set_num_threads(threads if threads > 0 else (os.cpu_count() or 1))
        qs = np.ascontiguousarray(qs, dtype=np.float32)
        d, i = self.index.search(qs, k)
        faiss.omp_set_num_threads(1)
        return i.astype(np.int64), d.astype(np.float64)

    def serialized_bytes(self) -> int:
        import faiss

        return int(faiss.serialize_index(self.index).nbytes)


class FlatIndex(_FaissIndex):
    """Exact brute-force search: the reference point for recall and speed."""

    kind = "Flat"
    search_param_name = "none"

    def build(self, train):
        import faiss

        faiss.omp_set_num_threads(self.build_threads)
        train = np.ascontiguousarray(train, dtype=np.float32)
        t0 = time.perf_counter()
        self.index = faiss.IndexFlatL2(self.dim)
        self.index.add(train)
        self.build_time_s = time.perf_counter() - t0
        self.n_vectors = train.shape[0]
        faiss.omp_set_num_threads(1)
        return self.build_time_s

    def set_search_param(self, value):
        self._search_param = None  # nothing to tune


class _IVFBase(_FaissIndex):
    search_param_name = "nprobe"

    def _training_subset(self, train, nlist, points_per_centroid=256):
        n_train = min(train.shape[0], points_per_centroid * nlist)
        if n_train >= train.shape[0]:
            return train
        rng = np.random.default_rng(self.seed)
        idx = rng.choice(train.shape[0], size=n_train, replace=False)
        idx.sort()
        return np.ascontiguousarray(train[idx])

    def set_search_param(self, value):
        value = int(value)
        self.index.nprobe = min(value, int(self.params["nlist"]))
        self._search_param = value


class IVFFlatIndex(_IVFBase):
    """Inverted file with uncompressed vectors in each cell."""

    kind = "IVF-Flat"

    def build(self, train):
        import faiss

        faiss.omp_set_num_threads(self.build_threads)
        train = np.ascontiguousarray(train, dtype=np.float32)
        nlist = int(self.params["nlist"])
        t0 = time.perf_counter()
        quantizer = faiss.IndexFlatL2(self.dim)
        index = faiss.IndexIVFFlat(quantizer, self.dim, nlist, faiss.METRIC_L2)
        # FAISS does not take ownership of the coarse quantizer, so a Python
        # reference has to outlive this function or the index ends up pointing
        # at freed memory.
        self._quantizer = quantizer
        index.cp.seed = self.seed
        index.train(self._training_subset(train, nlist))
        index.add(train)
        self.build_time_s = time.perf_counter() - t0
        self.index = index
        self.n_vectors = train.shape[0]
        faiss.omp_set_num_threads(1)
        return self.build_time_s


class IVFPQIndex(_IVFBase):
    """Inverted file with product-quantised residuals.

    ``refine`` wraps the index in ``IndexRefineFlat``, which re-ranks the top
    ``refine_k`` candidates using the original vectors.  That fixes the recall
    ceiling caused by quantisation error but needs the full vectors in memory,
    which is the trade-off the Part 3 compression experiment measures.
    """

    kind = "IVF-PQ"

    def build(self, train):
        import faiss

        faiss.omp_set_num_threads(self.build_threads)
        train = np.ascontiguousarray(train, dtype=np.float32)
        nlist = int(self.params["nlist"])
        m = int(self.params["m"])
        nbits = int(self.params.get("nbits", 8))
        if self.dim % m != 0:
            raise ValueError(
                f"IVF-PQ needs m to divide d: d={self.dim}, m={m}. "
                f"Legal values for this dataset: {divisors(self.dim)}"
            )
        t0 = time.perf_counter()
        quantizer = faiss.IndexFlatL2(self.dim)
        base = faiss.IndexIVFPQ(quantizer, self.dim, nlist, m, nbits)
        # Keep Python references alive: FAISS does not own the quantizer, and
        # IndexRefineFlat does not own its base index.
        self._quantizer = quantizer
        self._ivf = base
        base.cp.seed = self.seed
        base.train(self._training_subset(train, nlist))

        if self.params.get("refine"):
            # The vectors must be added *through* the wrapper: it forwards them
            # to both the PQ index and its own flat copy. Adding to the base
            # first and wrapping afterwards leaves the refine index empty and
            # trips an ntotal assertion inside FAISS.
            index = faiss.IndexRefineFlat(base)
            # IndexRefineFlat fetches k * k_factor candidates from the base
            # index and re-ranks them against the original vectors. We express
            # the knob as "re-rank the top refine_k candidates at k=refine_at_k"
            # so the config file reads the way the assignment phrases it.
            index.k_factor = float(self.params.get("refine_k", 100)) / float(
                self.params.get("refine_at_k", 10)
            )
            index.add(train)
        else:
            base.add(train)
            index = base
        self.build_time_s = time.perf_counter() - t0
        self.index = index
        self.n_vectors = train.shape[0]
        faiss.omp_set_num_threads(1)
        return self.build_time_s

    def set_search_param(self, value):
        value = int(value)
        self._ivf.nprobe = min(value, int(self.params["nlist"]))
        self._search_param = value


# ----------------------------------------------------------------------
# hnswlib-backed index
# ----------------------------------------------------------------------
class HnswIndex(BaseIndex):
    """Hierarchical navigable small-world graph (hnswlib)."""

    kind = "HNSW"
    search_param_name = "efSearch"

    def build(self, train):
        import hnswlib

        train = np.ascontiguousarray(train, dtype=np.float32)
        n = train.shape[0]
        M = int(self.params["M"])
        efc = int(self.params.get("ef_construction", 200))
        t0 = time.perf_counter()
        index = hnswlib.Index(space="l2", dim=self.dim)
        index.init_index(max_elements=n, ef_construction=efc, M=M, random_seed=self.seed)
        index.add_items(train, np.arange(n), num_threads=self.build_threads)
        self.build_time_s = time.perf_counter() - t0
        self.index = index
        self.n_vectors = n
        return self.build_time_s

    def set_search_param(self, value):
        self.index.set_ef(max(int(value), 1))
        self._search_param = int(value)

    def search_one(self, q, k):
        q = np.ascontiguousarray(q.reshape(1, -1), dtype=np.float32)
        labels, dists = self.index.knn_query(q, k=k, num_threads=1)
        return labels[0].astype(np.int64), dists[0].astype(np.float64)

    def search(self, qs, k, threads: int = 0):
        qs = np.ascontiguousarray(qs, dtype=np.float32)
        labels, dists = self.index.knn_query(
            qs, k=k, num_threads=threads if threads > 0 else (os.cpu_count() or 1)
        )
        return labels.astype(np.int64), dists.astype(np.float64)

    def serialized_bytes(self) -> int:
        with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as tmp:
            path = tmp.name
        try:
            self.index.save_index(path)
            return int(os.path.getsize(path))
        finally:
            os.unlink(path)


_REGISTRY = {
    "Flat": FlatIndex,
    "HNSW": HnswIndex,
    "IVF-Flat": IVFFlatIndex,
    "IVF-PQ": IVFPQIndex,
}


def make_index(kind: str, dim: int, seed: int = 42, build_threads: int = 0, **params) -> BaseIndex:
    if kind not in _REGISTRY:
        raise KeyError(f"unknown index kind {kind!r}; known: {sorted(_REGISTRY)}")
    return _REGISTRY[kind](dim, seed=seed, build_threads=build_threads, **params)
