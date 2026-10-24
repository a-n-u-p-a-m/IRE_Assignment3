# Part 2: build cost and memory at recall 0.95

Size is the serialized index on disk, which for these index types is within a few percent of resident memory. Builds use all cores (search does not); the thread count is recorded per row in results.csv.

| dataset | index | config | build time (s) | size (MB) | bytes/vector | recall@10 (held-out) |
|---|---|---|---|---|---|---|
| glove | Flat (exact) | — | 0.2 | 473.4 | 400.0 | 1.000 |
| glove | HNSW | M=32,ef_construction=200 | 234.0 | 800.3 | 676.2 | 0.955 |
| glove | IVF-Flat | nlist=4096 | 279.9 | 484.5 | 409.4 | 0.965 |
| glove | IVF-PQ | never reaches 0.95 | — | — | — | 0.626 |
| glove | IVF-PQ+rerank | never reaches 0.95 | — | — | — | 0.951 |
| sift | Flat (exact) | — | 0.2 | 512.0 | 512.0 | 1.000 |
| sift | HNSW | M=16,ef_construction=200 | 77.0 | 660.5 | 660.5 | 0.966 |
| sift | IVF-Flat | nlist=16384 | 927.1 | 528.5 | 528.5 | 0.973 |
| sift | IVF-PQ | never reaches 0.95 | — | — | — | 0.747 |
| sift | IVF-PQ+rerank | m=16,nbits=8,nlist=4096,refine=True,refine_at_k=10,refine_k=100 | 295.3 | 538.3 | 538.3 | 0.964 |
