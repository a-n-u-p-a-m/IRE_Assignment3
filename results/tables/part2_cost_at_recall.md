# Part 2: build cost and memory at recall 0.95

`size` is the serialized index on disk. For HNSW and IVF-Flat that is within a few percent of resident memory, but **not for IVF-PQ**: FAISS auto-enables a precomputed distance table of nlist x m x 2^nbits floats that is held in RAM at search time and is not serialized. The timings already benefit from it, so omitting it would flatter PQ in exactly the dimension PQ is chosen for. `resident B/vec` adds it back. The table does not grow with n, so its per-vector cost falls as the database grows. Builds use all cores; search is single-threaded.

| dataset | index | config | build time (s) | size (MB) | bytes/vector | PQ search table (MB) | resident B/vec | recall@10 (held-out) |
|---|---|---|---|---|---|---|---|---|
| glove | Flat (exact) | — | 0.2 | 473.4 | 400.0 | — | 400.0 | 1.000 |
| glove | HNSW | M=32,ef_construction=200 | 228.1 | 800.3 | 676.2 | — | 676.2 | 0.955 |
| glove | IVF-Flat | nlist=4096 | 206.5 | 484.5 | 409.4 | — | 409.4 | 0.965 |
| glove | IVF-PQ | never reaches 0.95 on the tuning split | — | — | — | — | — | 0.6302 (tune) |
| glove | IVF-PQ+rerank | m=25,nbits=8,nlist=4096,refine=True,refine_at_k=10,refine_k=100 | 214.7 | 514.2 | 434.5 | 104.9 | 523.1 | 0.970 |
| sift | Flat (exact) | — | 0.5 | 512.0 | 512.0 | — | 512.0 | 1.000 |
| sift | HNSW | M=16,ef_construction=200 | 78.7 | 660.5 | 660.5 | — | 660.5 | 0.966 |
| sift | IVF-Flat | nlist=16384 | 864.3 | 528.5 | 528.5 | — | 528.5 | 0.973 |
| sift | IVF-PQ | never reaches 0.95 on the tuning split | — | — | — | — | — | 0.7383 (tune) |
| sift | IVF-PQ+rerank | m=16,nbits=8,nlist=4096,refine=True,refine_at_k=10,refine_k=100 | 226.2 | 538.3 | 538.3 | 67.1 | 605.4 | 0.964 |
