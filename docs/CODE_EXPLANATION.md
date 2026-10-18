# Code explanation and design decisions

A walk through the codebase, module by module, with the reasoning behind the
choices. The README says *how to run it*; this says *why it is built this way*
and where the traps are.

---

## 0. The shape of the thing

```
measurement                      reporting
───────────                      ─────────
part1_exact.py   ─┐
part2_benchmark.py├─► results/results.csv ─┬─► make_figures.py ─► results/figures/
part3_scaling.py  │   (append-only)        └─► make_tables.py  ─► results/tables/
part4_intrinsic.py┘
```

**One rule governs the whole design: measurements only ever append rows to
`results.csv`, and figures and tables only ever read it.** No number is carried
in memory from a measurement to a plot.

Three things follow:

- *Reproduction is structural, not aspirational.* The deliverable requires that
  `bash run_all.sh` regenerate every number, figure and table. If the plotting
  code physically cannot see anything but the CSV, a figure that regenerates is
  proof that the number behind it is in the file.
- *A long sweep is resumable.* The full run is hours. Rows are flushed and
  `fsync`ed as they are produced, and a restarted run skips keys already
  present, so a crash costs one configuration.
- *Provenance travels with the number.* Each row carries the git commit, the
  machine id, build and search parameters, thread counts, how many queries were
  actually timed, and the individual repeat timings — enough to interrogate a
  suspicious entry months later.

The second structural rule: **the measurement protocol exists once**, in
`bench.py`. Parts 2, 3 and 4 differ only in which data and configurations they
feed it. If Part 3 timed things slightly differently from Part 2, its numbers
could not be compared against Part 2's, and the comparison is the point.

---

## 1. `datasets.py` — getting the data right before anything else

Three decisions here prevent three classes of silent wrongness.

### Squared L2 everywhere

FAISS `IndexFlatL2` and hnswlib `space='l2'` both return **squared** distances.
The ann-benchmarks files store **plain Euclidean** for SIFT and **angular** for
GloVe. That is three units in one pipeline, and a tie-tolerance comparison
between two of them silently does the wrong thing — it will not crash, it will
just quietly change recall.

So the `distances` array in the file is discarded and ground-truth distances are
recomputed as squared L2 from the stored neighbour **ids**. It costs one pass
over 10 000 × 100 vectors and removes the entire problem.

### GloVe: normalise, then treat as Euclidean

For unit vectors,

```
‖a − b‖² = ‖a‖² + ‖b‖² − 2·a·b = 2 − 2·cos(a, b)
```

which is strictly decreasing in cosine similarity. Ranking by squared L2 is
therefore *identical* to ranking by angular distance, so the neighbour ids in
the file remain valid ground truth after normalisation — only the distance
values change, which is exactly why we recompute them. This is the one-line
justification for the assignment's "normalise GloVe and use Euclidean"
instruction, and the report should state it rather than just complying.

### Subsetting invalidates the shipped ground truth

The neighbour ids index into the full 1 M database. At N = 100 000 most of them
are not in the index at all, so reusing them would cap recall near N/1M — the
experiment would be measuring the subsetting, not the index.

`subsample()` therefore **discards** the original ids rather than trying to
remap them, and recomputes exact ground truth against the subset by brute
force. This affects Part 3 (every size below 1M) and every `--quick` run.
`experiments/common.py` caches the recomputed ids and distances under
`data/cache/` keyed by (dataset, n, seed, k) — ids and distances only, never
vectors, so the cache stays small and the "don't ship datasets" rule is easy to
honour.

### Query split

First 5 000 tune, last 5 000 report, applied to queries, ids and distances with
the same slice. `test_split_queries_takes_first_and_last_halves` asserts the
ground truth is sliced along with the queries — forgetting that is a one-
character bug that produces plausible-looking garbage.

---

## 2. `recall.py` — why recall@10 is not a one-liner

The naive version is `len(set(found) & set(gt)) / k`. It is wrong in three
ways that matter here.

### Ties at the k-th neighbour

If several database vectors sit at exactly the distance of the 10th true
neighbour, any of them is an equally correct 10th neighbour. The naive formula
punishes an index for returning a different-but-equidistant vector, which is not
a mistake — and the assignment calls this out specifically.

A returned id counts if it is in the ground-truth top-k set **or** its exact
distance is `≤ gt_dists[k-1] · (1 + rtol)`. `rtol = 1e-5` is chosen for float32:
accumulating 128 squared terms in single precision jitters the last few bits, so
a tolerance of zero makes the tie test flaky, while a loose tolerance lets
genuine misses through. Both failure directions have a test
(`test_float_noise_within_rtol_counts_as_a_tie`, `test_just_outside_rtol_is_a_miss`).

### The tie test must use *exact* distances

This is the subtle one. IVF-PQ reports distances computed from its quantisation
codes — they are approximations, and they are systematically wrong in the
direction that matters. Feeding them to the tie test would let a badly
quantised vector slip under the threshold and inflate exactly the index whose
accuracy we are trying to measure.

So `true_distances()` recomputes the distance of every returned id against the
original vectors, in float64, and that is what the tie test consumes. When no
exact distances are supplied the function falls back to plain set intersection
rather than trusting whatever the index said — asserted by
`test_tie_test_ignored_when_no_true_distances_given`.

### Degenerate output

FAISS pads with `-1` when it finds fewer than k candidates; a broken wrapper can
return the same id twice. Neither may count. Both have tests.

**Unit tests** (`tests/test_recall.py`, 20 of them) cover: exact match, shuffled
order, disjoint, half-correct, the hand-built tie case that the naive formula
gets wrong, all-distances-tied, float noise inside and outside tolerance, `-1`
padding, duplicate ids, zero-distance duplicates, short result lists, shape
mismatches, and an end-to-end check that exact search scores exactly 1.0 under
our own metric.

---

## 3. `timing.py` — making QPS mean something

The assignment asks for one query at a time, one thread, repeated, median. The
details that decide whether the number is real:

**Thread control happens before import.** `OMP_NUM_THREADS=1` is exported in
`run_all.sh` and set in the notebook's first cell, because for some BLAS builds
setting it after NumPy is imported is too late. `faiss.omp_set_num_threads(1)`
is called again at timing time as a belt-and-braces measure.

**Warm-up discarded.** The first queries through a fresh index pay page faults
and cold caches and are not representative of steady state.

**Time budget, not a fixed query count.** A configuration running at 50 QPS
would need 100 s per repeat over 5 000 queries, times three repeats, times
dozens of configurations. Each repeat therefore stops after
`time_budget_s` (default 20 s) provided at least `min_queries` have been timed.
The count actually timed is written to `results.csv` per row, so the report can
state the policy honestly instead of implying every number came from 5 000
queries. Recall is always computed on the full query set; only the *timing*
loop is truncated.

**One clock reading per repeat.** `perf_counter` overhead is far below the
noise floor, but reading it once per repeat removes the question entirely; the
budget check reads it every 32 queries.

**Batched timing exists only for Part 1**, and runs single-threaded too, so the
single-vs-batch comparison isolates the memory-traffic effect rather than
measuring parallelism (see §8).

---

## 4. `indexes.py` — one interface, four indexes

Every index exposes `build`, `set_search_param`, `search_one`, `search` and
`serialized_bytes`. That uniformity is what lets `bench.py` be a single loop
rather than three, which is what makes the numbers comparable.

**HNSW from hnswlib, IVF from FAISS.** Both are permitted. hnswlib is the
implementation these benchmarks are usually run against and exposes `ef`
cleanly at query time; FAISS owns the IVF family.

**`search_one` is the timed path, `search` is the correctness path.** Batched
search may use every core because parallelism cannot change *which* neighbours
are found — only how fast. Evaluating recall on 5 000 queries single-threaded
at high `efSearch` would double the cost of the sweep for no benefit.
`test_search_one_and_batch_agree` pins the two together.

**IVF training is seeded and subsampled.** k-means on a million vectors is
wasteful — FAISS wants a few hundred points per centroid — so training draws
`min(n, 256·nlist)` vectors with a fixed seed, and `index.cp.seed` is set so the
quantiser is reproducible.

**The `m | d` trap.** IVF-PQ splits each vector into `m` sub-vectors, so `m`
must divide `d`. SIFT is 128-dimensional and the familiar powers of two work.
**GloVe is 100-dimensional**, where 8, 16, 32 and 64 are all illegal and `m`
must come from {4, 10, 20, 25, 50}. This is the kind of thing that fires as a
FAISS assertion ninety minutes into a sweep, so it is checked three times: in
the index wrapper with a readable error listing the legal values, in
`part2_benchmark.py` before anything is built, and in
`test_configs_use_legal_pq_m_values`, which validates the YAML without running
anything.

**Build threading is configurable and recorded.** Search is always
single-threaded, but a single-threaded HNSW build on 1.18 M vectors takes
hours. Builds use all cores by default, `build_threads` goes into every row,
and the README states plainly that multi-threaded HNSW construction is not
bit-reproducible. `build.threads: 1` buys determinism at the cost of time.

**Size** is the serialized index: `faiss.serialize_index(...).nbytes`, or for
hnswlib a save to a temporary file and `stat`. Close to resident memory for
these index types, but not identical — one of the honest limitations for
Part 5 question 5.

---

## 5. `bench.py` — the protocol, written once

For each build configuration: build once, record build time and size, then walk
the search parameter from cheap to expensive. At each value, recall on **both**
splits (cheap, batched) and QPS on one split (expensive, single-threaded).

### Which split gets timed, and why it is defensible

QPS is measured on the **tuning** split; recall is reported on the **held-out**
split. The argument: the work an index does per query is bounded by `efSearch`
or `nprobe`, not by the content of the query, so throughput is a property of
the index and its parameter rather than of which 5 000 queries are used. Timing
on the tuning split means nothing about the held-out queries influences any
reported choice, not even indirectly through a timing measurement.

That is an argument, not a fact, so `scripts/check_timing_split.py` measures
it: it times several configurations on both splits and prints the difference
next to the run-to-run spread of the repeats. If the split-to-split difference
is inside the repeat noise, the choice costs nothing. Part 5 question 5 asks
for two ways the benchmark could mislead *and whether they were tested* — this
is one of them, tested.

### Early stop

Once tuning recall passes `stop_above_recall` (0.995), larger parameter values
only cost time: the curve is flat there and no table needs anything above 0.95.
The stop is written into the row's `notes` so the extent of the sweep is
visible in the CSV rather than being an invisible policy.

---

## 6. `frontier.py` — the envelope, and the tuning trap inside it

An index type is represented by the best it can do, pooled over every build
configuration and search parameter: a point is on the frontier if no other
point of that type is at least as accurate *and* at least as fast, with one
strictly better.

**Selecting that envelope is itself a tuning decision**, and this is the easiest
place in the whole assignment to tune on the test set without noticing. Picking
the best configuration at each recall level *using the held-out recall* is
exactly the sin the split exists to prevent. So the frontier is selected on
`recall_tune` and *drawn and tabulated* at `recall_test`.

**Interpolation.** The table needs QPS at recall 0.90 and 0.95, which a sweep
almost never lands on exactly. We interpolate linearly in `log(QPS)` between the
two bracketing frontier points, because the trade-off is far closer to
exponential than linear in that region, and clamp the result between the two
measured values. When no measured point reaches the target the function returns
NaN and the table prints "not reached" with the best recall achieved — which is
the expected, and interesting, outcome for IVF-PQ without re-ranking, and is
better reporting than an extrapolation off the end of the sweep.

---

## 7. `synthetic.py` — the Part 4 generator

**The question:** how does the *intrinsic* dimensionality of the data — the
number of directions it actually varies in — change what each index can do, at
a fixed ambient dimension?

**Why not just vary the dimension.** The obvious experiment generates uniform
random vectors at d = 8, 32, 128, 512 and watches recall fall as distances
concentrate. It is confounded three ways: changing d also changes bytes per
vector, the cost of one distance computation, and the legal PQ layouts. A speed
difference between d = 8 and d = 512 cannot be attributed to geometry because
the arithmetic changed too.

**The design.** Ambient dimension stays at 128 in every condition. Data is
generated on a random k-dimensional linear subspace plus a small isotropic
noise floor:

```
X = C·Bᵀ + σ·E,    C ~ N(0, I_k),   B a random orthonormal d×k basis
```

Every condition has identical vector size, distance cost, index memory and PQ
layout. The only thing that changes is k. `k = 128 = d` is the **control**: the
subspace is the whole space and the generator degenerates to isotropic Gaussian
data. `test_ambient_dimension_is_constant_across_conditions` asserts the
invariance the design depends on.

The noise floor matters: at σ = 0 and k = 2 the data is a genuine 2-dimensional
sheet and the nearest neighbour is trivially findable. A small σ keeps every
condition full-rank, so the difference is one of *effective* dimensionality
rather than of rank.

The basis uses a QR decomposition with the sign of `diag(R)` folded in, so the
basis is a deterministic function of the seed across LAPACK versions — without
that, QR sign conventions differ between builds and "seeded" would be a lie.

`relative_contrast()` measures mean-distance / nearest-neighbour-distance per
query. Values near 1 mean nothing is distinctly closest, which is the regime
where partition-based pruning stops working. It is recorded per condition so
the report can *show* that the generator moved the data into that regime rather
than asserting it.

---

## 8. `experiments/` — the four drivers

**`part1_exact.py`** does three things: checks our exact search against the
shipped neighbour ids, times exact search one-at-a-time and batched, and writes
the baseline row.

The verification is expected *not* to be 1.000 under naive set comparison, and
the output reports both the naive agreement and the tie-aware recall so the
report can demonstrate that every disagreement is a tie rather than a bug.

The single-vs-batch gap is the part the report has to *explain*, not just
measure:

- One query at a time is a matrix–vector product (BLAS level 2). Each query
  streams the entire database through the CPU once, so throughput is bounded by
  memory bandwidth and each database vector is used for a single multiply-add
  pass before being evicted.
- The whole query block at once is a matrix–matrix product (BLAS level 3). A
  tile of the database is loaded into cache and reused against every query in
  the block, so the same memory traffic serves thousands of queries — and the
  per-call overhead is amortised too.

Both are timed single-threaded precisely so the comparison isolates that effect
instead of measuring thread count.

**`part2_benchmark.py`** builds the configuration list from the YAML and hands
it to the shared sweep. PQ `m` values are validated against the dataset
dimension before anything is built.

**`part3_scaling.py`** — two details decide whether it measures what it claims:
ground truth **must** be recomputed per subset (§1), and `nlist` **must** be
retuned per size. IVF's asymptotics come from the balance between cell count
and cell occupancy, conventionally `nlist ∝ √N`; holding `nlist` fixed across N
would measure "an index tuned for 1M, run on 100K", which degrades for reasons
that have nothing to do with scaling. The script derives `nlist` per size,
rounded to a power of two. Exact search is included as the yardstick: its cost
is exactly linear in N, so "sublinear" has something to be measured against.

**`part4_intrinsic.py`** holds everything constant except `k_intrinsic`, runs
three indexes, and records relative contrast per condition. The brief's rules —
one variable, at least three settings plus a control, at least two indexes,
seeded generator, ≤ 500K vectors — are enforced in code: the script raises if
`n > 500_000` and `test_part4_respects_the_500k_cap` checks the YAML.

**`make_figures.py` / `make_tables.py`** read `results.csv` and nothing else.

---

## 9. `plots.py`

- **QPS on a log axis.** The three index types span two to three orders of
  magnitude; on a linear axis everything but the fastest is a flat line.
- **Markers for every measured point, lines along the frontier only.**
  Dominated points are drawn faintly, so a reader can see how much of the grid
  was explored without burying the frontier in it.
- **Exact search as a horizontal line**, because it is a single
  (recall = 1, QPS) point and the line makes "how much faster, at what cost in
  recall" readable at a glance.
- **Part 4's figure carries two y-axes**: QPS at the recall target (solid) and
  the best recall actually reached (dashed). Where the dashed line drops below
  the target the solid line is necessarily missing — that absence *is* the
  finding, and putting both on one pair of axes stops it reading as a plotting
  bug.
- No seaborn, no style sheets: the plots have to be ours, and plain matplotlib
  is easier for a marker to reproduce.

---

## 10. `config.py` and the two YAMLs

`--quick` is **not a code path**. Full and quick runs execute the same
functions and differ only in a config file: database size, grid density, repeat
count, timing budget. A quick mode implemented as `if quick: ...` branches
tests something other than the thing it is supposed to be a fast version of,
which defeats the purpose of having one.

---

## 11. What this design does *not* protect against

Stated here because Part 5 asks for it, and because a limitations section
written after the fact is usually a list of excuses.

1. **Single-query single-thread QPS favours HNSW.** IVF-Flat's cell scan is a
   GEMM and gains far more from batching than HNSW's pointer-chasing graph walk
   does. The protocol the assignment specifies is therefore not neutral between
   the two. Testable by re-running one configuration batched; the Part 1
   single-vs-batch gap already demonstrates the mechanism.
2. **Index size ≠ resident memory.** Serialized size is close for these types
   but ignores allocator overhead and any transient memory during build.
3. **Frontier selection is tuning.** Handled by selecting on the tuning split —
   but the *grid itself* was chosen by a human who had seen earlier results.
4. **One machine, one run.** Thermal state, background load and cache residency
   are controlled for only by warm-up, repeats and the median; there is no
   cross-machine validation.
5. **Multi-threaded HNSW builds are not bit-reproducible.** Seeded, but not
   deterministic. Everything else is.
6. **recall@10 is one metric.** recall@1 and recall@100 would rank the indexes
   differently, particularly for IVF-PQ, whose quantisation error hurts fine
   ranking more than coarse retrieval.
