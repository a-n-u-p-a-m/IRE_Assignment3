# Hypotheses

Written and committed **before** the corresponding experiments were run. The
git history shows this commit preceding any results for these experiments.

A hypothesis that turns out wrong is still a good hypothesis if the reasoning
behind it was explicit, so each one below states a mechanism and a prediction
specific enough to be contradicted.

---

## Part 3 — Scaling on SIFT (100K / 300K / 1M)

**Experiment.** Benchmark HNSW, IVF-Flat and exact search on seeded random
subsets of SIFT at N = 100 000, 300 000 and 1 000 000, recomputing exact ground
truth for each subset and retuning `nlist` as ~4·√N. Measure the throughput
each index achieves at recall 0.95, and the build time.

**Hypothesis.**

1. **Exact search is exactly linear.** Its per-query cost is N·d multiply-adds
   with no data-dependent branching, so QPS should fall by almost exactly 10×
   from 100K to 1M. Any deviation should be small and should come from cache
   effects (at 100K the database is ~51 MB and partially resident in L3; at 1M
   it is ~512 MB and purely streamed from RAM), which would make the *small*
   case look slightly better than linear.

2. **HNSW degrades slowest — roughly logarithmically.** A greedy descent visits
   O(log N) layers and, with `efSearch` fixed, examines a candidate set whose
   size does not grow with N. I expect QPS at recall 0.95 to fall by less than
   2× over the 10× size increase, and I expect the `efSearch` needed to reach
   0.95 to grow slowly (perhaps 1.5×), because the graph has more places to get
   stuck in a local minimum as it grows.

3. **IVF-Flat sits in between, around √N.** With `nlist ∝ √N` each cell holds
   ~√N vectors, so scanning a fixed fraction of cells costs O(√N) — plus an
   O(√N) coarse quantisation step to pick the cells. I expect QPS at 0.95 to
   fall by roughly 3× over the 10× size increase, i.e. visibly better than
   exact and visibly worse than HNSW.

4. **Build time will tell the opposite story.** HNSW build is ~O(N log N) with
   a large constant and will dominate; IVF-Flat build is k-means on a
   subsample plus one assignment pass and should stay cheap. I expect HNSW's
   build at 1M to be more than 10× IVF-Flat's.

**What would falsify this.** HNSW losing more than IVF-Flat over the size
range; exact search deviating from linear by more than ~20%; or the ordering
of the three at recall 0.95 changing with N.

---

## Part 4 — Intrinsic dimensionality at fixed ambient dimension

**Experiment.** Generate 200 000 vectors in a fixed ambient dimension d = 128,
lying on a random k-dimensional linear subspace plus a small isotropic noise
floor, for k ∈ {2, 8, 32, 128}. k = 128 is the control (isotropic Gaussian).
Everything else — n, d, noise, seed, index parameters, timing protocol — is
held constant, so the only thing that varies is how many directions the data
actually occupies. Benchmark HNSW, IVF-Flat and IVF-PQ.

**Why this and not "vary the dimension".** Varying the ambient dimension would
simultaneously change bytes per vector, the cost of one distance computation
and the legal PQ layouts, so a difference in speed could not be attributed to
geometry. Holding d fixed and varying the intrinsic dimension isolates the
property under test.

**Hypothesis.**

1. **Relative contrast falls monotonically as k rises.** With few effective
   directions, the nearest neighbour is genuinely much closer than a random
   point; as k approaches d the distance distribution concentrates and
   everything is roughly equally far away. This is the mechanism driving
   everything else, and I expect to measure it directly.

2. **IVF-Flat degrades first and worst.** It prunes by assuming the query's
   true neighbours live in the few cells whose centroids are closest. When
   contrast collapses, cell membership stops being informative: the true
   neighbour is frequently in an unprobed cell, and the `nprobe` needed for
   0.95 should rise sharply — I predict it has to scan a large fraction of all
   cells at k = 128, at which point it is barely faster than exact search.

3. **HNSW degrades more gracefully but not gratis.** Its graph encodes relative
   proximity rather than a fixed partition, so it has no partition to be wrong
   about. I expect it to still reach 0.95 at every k, but at a steadily larger
   `efSearch`, and I predict its advantage over IVF-Flat at recall 0.95 *grows*
   as k rises. I expect the crossover to be visible by k = 32.

4. **IVF-PQ never reaches 0.95 at high k, and its ceiling falls with k.** PQ
   splits the vector into m sub-vectors and quantises each independently; that
   is efficient exactly when the data has structure for the sub-quantisers to
   latch onto. With k = 2 the sub-vectors are highly redundant and 8-bit codes
   should be nearly lossless; at k = 128 each sub-space is full-entropy
   Gaussian noise and the quantisation error should swamp the inter-point
   distances. I predict a recall ceiling well under 0.95 at k = 128 that no
   `nprobe` can lift, because the error is in the distance function rather than
   in which cells are scanned.

5. **Speed at fixed recall is not monotone in the obvious direction.** At very
   low k every index is both fast *and* accurate, so the interesting curve is
   not "slower with higher k" but "the recall target becomes unreachable" — I
   expect the headline failure to show up as a missing point rather than a slow
   one.

**What would falsify this.** IVF-Flat degrading less than HNSW; IVF-PQ's
ceiling being flat in k; or relative contrast not moving, which would mean the
generator is not producing the regime I think it is.

**What I am not testing.** Non-linear manifolds (the subspace here is linear),
clustered data, drift between index and query distributions, update cost, and
anything about multi-threaded or batched throughput.
