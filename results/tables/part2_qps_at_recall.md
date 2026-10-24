# Part 2: throughput at fixed recall

One query at a time, one thread, median of repeated runs. QPS is interpolated log-linearly between the two Pareto points bracketing the target. The Pareto frontier is selected on the tuning queries; `config` is the fastest configuration that reaches the target. `vs exact` compares against single-query exact search from Part 1. A value marked (≥) is a lower bound: the index already cleared the target at the cheapest setting in the sweep, so where it actually crosses was never measured.

| dataset | index | QPS @ r=0.9 | vs exact | config @ r=0.9 | QPS @ r=0.95 | vs exact | config @ r=0.95 |
|---|---|---|---|---|---|---|---|
| glove | HNSW | 1,769.1 | 41.1x | M=32,ef_construction=200 | 796.8 | 18.5x | M=32,ef_construction=200 |
| glove | IVF-Flat | 943.4 | 21.9x | nlist=16384 | 316.7 | 7.4x | nlist=4096 |
| glove | IVF-PQ | not reached | — | best recall 0.626 | not reached | — | best recall 0.626 |
| glove | IVF-PQ+rerank | 867.7 | 20.1x | m=25,nbits=8,nlist=4096,refine=True,refine_at_k=10,refine_k=100 | not reached | — | best recall 0.951 |
| sift | HNSW | 13,886.6 | 363.7x | M=32,ef_construction=200 | 9,616.8 | 251.8x | M=16,ef_construction=200 |
| sift | IVF-Flat | 3,090.6 | 80.9x | nlist=4096 | 1,799.1 | 47.1x | nlist=16384 |
| sift | IVF-PQ | not reached | — | best recall 0.747 | not reached | — | best recall 0.747 |
| sift | IVF-PQ+rerank | 5,617.4 | 147.1x | m=16,nbits=8,nlist=4096,refine=True,refine_at_k=10,refine_k=100 | 3,793.2 | 99.3x | m=16,nbits=8,nlist=4096,refine=True,refine_at_k=10,refine_k=100 |
