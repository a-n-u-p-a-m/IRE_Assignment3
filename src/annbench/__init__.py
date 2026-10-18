"""Benchmarking approximate nearest neighbour search.

CS4.406 Information Retrieval & Extraction, Assignment 3.

Layout
------
``datasets``   loading, normalisation, query splits, ground-truth recomputation
``recall``     tie-aware recall@k (hand-written, as the ground rules require)
``timing``     single-thread, one-query-at-a-time QPS with a time budget
``indexes``    uniform wrappers over FAISS Flat/IVF-Flat/IVF-PQ and hnswlib HNSW
``bench``      the one sweep loop shared by Parts 2, 3 and 4
``frontier``   Pareto envelopes and "QPS at recall 0.90 / 0.95"
``results``    append-only results.csv, the single source of truth for figures
``plots``      figures, built from results.csv alone
``synthetic``  seeded generator for the Part 4 experiment
``env``        machine and library provenance

Nothing in this package reads a figure's data from memory: measurements append
rows to ``results/results.csv`` and every figure and table is regenerated from
that file.
"""

__version__ = "1.0.0"

__all__ = [
    "bench",
    "config",
    "datasets",
    "env",
    "frontier",
    "indexes",
    "plots",
    "recall",
    "results",
    "synthetic",
    "timing",
]
