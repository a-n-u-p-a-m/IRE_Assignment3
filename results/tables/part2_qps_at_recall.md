# Part 2: throughput at fixed recall

One query at a time, one thread, median of repeated runs. QPS is interpolated log-linearly between the two Pareto points bracketing the target. The Pareto frontier is selected on the tuning queries; `config` is the fastest configuration that reaches the target. `vs exact` compares against single-query exact search from Part 1. A value marked (≥) is a lower bound: the index already cleared the target at the cheapest setting in the sweep, so where it actually crosses was never measured.

| dataset | index | QPS @ r=0.9 | vs exact | config @ r=0.9 | QPS @ r=0.95 | vs exact | config @ r=0.95 |
|---|---|---|---|---|---|---|---|
| glove | HNSW | 1,954.4 | 47.4x | M=32,ef_construction=200 | 907.1 | 22.0x | M=32,ef_construction=200 |
| glove | IVF-Flat | 933.9 | 22.7x | nlist=16384 | 317.7 | 7.7x | nlist=4096 |
| glove | IVF-PQ | not reached | — | best recall_tune 0.6302 | not reached | — | best recall_tune 0.6302 |
| glove | IVF-PQ+rerank | 898.3 | 21.8x | m=25,nbits=8,nlist=4096,refine=True,refine_at_k=10,refine_k=100 | 379.6 | 9.2x | m=25,nbits=8,nlist=4096,refine=True,refine_at_k=10,refine_k=100 |
| sift | HNSW | 12,937.4 | 351.9x | M=16,ef_construction=200 | 9,436.0 | 256.7x | M=16,ef_construction=200 |
| sift | IVF-Flat | 3,149.0 | 85.7x | nlist=4096 | 2,001.8 | 54.4x | nlist=16384 |
| sift | IVF-PQ | not reached | — | best recall_tune 0.7383 | not reached | — | best recall_tune 0.7383 |
| sift | IVF-PQ+rerank | 5,888.7 | 160.2x | m=16,nbits=8,nlist=4096,refine=True,refine_at_k=10,refine_k=100 | 3,866.8 | 105.2x | m=16,nbits=8,nlist=4096,refine=True,refine_at_k=10,refine_k=100 |
