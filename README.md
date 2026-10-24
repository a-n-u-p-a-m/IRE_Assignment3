# Benchmarking Approximate Nearest Neighbour Search

CS4.406 Information Retrieval & Extraction — Assignment 3.

Benchmarks HNSW, IVF-Flat and IVF-PQ against exact search on SIFT-1M and
GloVe-100, plus a scaling study on SIFT (Part 3) and a controlled experiment on
generated data (Part 4).

---

## Quick start

```bash
pip install -r requirements.txt
bash scripts/download_data.sh      # ~1 GB into data/, once
bash run_all.sh --quick            # < 15 min: proves the pipeline runs
bash run_all.sh                    # full run: the numbers in report.pdf
```

Everything lands in `results/`:

| path | what |
|---|---|
| `results/results.csv` | one row per benchmark run — the source of every number |
| `results/figures/` | every figure in the report |
| `results/tables/` | every table, as Markdown and LaTeX |
| `results/env.json` | machine and library versions of the run |

A `--quick` run writes to `results/quick/` instead, so it can never overwrite
the full run's deliverables or pool its measurements into them. The two use
different database sizes, query counts and timing budgets, so their rows must
never share a file: a 100K-vector measurement is roughly ten times faster than
the same configuration at 1M, and would win a Pareto frontier it never earned.

### What `--quick` changes

Nothing in the code. `--quick` selects `configs/quick.yaml` instead of
`configs/full.yaml`, which uses 100 000-vector subsets of each dataset, one
build configuration per index, a coarse parameter grid, 1 000 queries per
split, and a single timing repeat with a 3-second budget. The measurement
protocol is identical. **Quick numbers are not report numbers** — they exist to
show the pipeline runs end to end on a laptop.

Because the datasets are subsetted, exact ground truth is recomputed for the
subset (the neighbour ids shipped with the file index into the full database
and would be meaningless otherwise). That recomputation is cached under
`data/cache/`, so a second `--quick` run is faster.

### Other ways to run

```bash
bash run_all.sh --only part2            # one stage only
bash run_all.sh --force                 # recompute rows already in results.csv
bash run_all.sh --config configs/my.yaml
python3 -m pytest tests -q              # unit tests on their own
python3 scripts/check_timing_split.py   # is timing on the tuning split safe?
```

`run_all.sh` is **resumable**. Rows are appended to `results/results.csv` as
each measurement completes and a re-run skips anything already recorded, so a
crash four hours into the full sweep costs one configuration rather than the
sweep. Delete the CSV or pass `--force` to start clean.

### Running from a notebook instead

`notebooks/run_experiments.ipynb` runs the same experiments cell by cell, with
inline plots and a `QUICK` switch in the first cell. It calls the same library
functions as `run_all.sh` — it is a different front end, not a second
implementation.

---

## Layout

```
code/
├── run_all.sh                  one-command reproduction (--quick for the fast path)
├── hypothesis.md               Parts 3 and 4 hypotheses, committed before results
├── requirements.txt
├── configs/
│   ├── full.yaml               the configuration behind report.pdf
│   └── quick.yaml              same structure, small values
├── src/annbench/
│   ├── datasets.py             loading, normalisation, splits, ground truth
│   ├── recall.py               tie-aware recall@k  (hand-written)
│   ├── timing.py               single-thread QPS   (hand-written)
│   ├── indexes.py              uniform wrappers over FAISS and hnswlib
│   ├── bench.py                the one sweep loop shared by Parts 2–4
│   ├── frontier.py             Pareto envelopes, QPS at a recall target
│   ├── results.py              append-only results.csv
│   ├── plots.py                figures                (hand-written)
│   ├── synthetic.py            Part 4 data generator  (seeded)
│   ├── env.py                  machine / library provenance
│   └── experiments/            one driver per part, plus make_figures/make_tables
├── scripts/
│   ├── download_data.sh
│   └── check_timing_split.py
├── tests/                      pytest: recall ties, frontier, indexes, generator
├── notebooks/run_experiments.ipynb
└── docs/CODE_EXPLANATION.md    what each module does and why it is built that way
```

`docs/CODE_EXPLANATION.md` is the companion to this README: it walks the
codebase module by module and records the design decisions behind it.

---

## Measurement protocol

The numbers only mean something if the protocol is fixed, so it lives in one
place (`src/annbench/bench.py`) and every part uses it.

**Distance.** Everything is squared Euclidean, end to end. `IndexFlatL2` and
hnswlib's `space='l2'` both return squared distances natively, so there is no
conversion anywhere. GloVe is L2-normalised and then treated as Euclidean:
for unit vectors ‖a−b‖² = 2 − 2·cos(a,b), so the Euclidean ranking is exactly
the angular ranking and the neighbour ids shipped with the file stay valid.

**Ground truth.** The `distances` array in the HDF5 files is discarded and
recomputed as squared L2 from the stored neighbour **ids** — the files store
plain Euclidean for SIFT and angular for GloVe, and mixing three units is how
silent factor-of-two bugs happen. Any subset of a dataset gets exact ground
truth recomputed against that subset.

**Query split.** First 5 000 queries tune, last 5 000 report. Every tuning
decision — including which points form the Pareto frontier — is made on the
tuning half.

**Recall.** `recall@10`, tie-aware: a returned vector counts if it is in the
true top-10 *or* if its exact distance equals the 10th true distance within
`rtol=1e-5`. The tie test uses distances recomputed against the original
vectors, never the distances the index reports — IVF-PQ's are approximate, and
trusting them would inflate its recall.

**Throughput.** One query at a time, one thread (`OMP_NUM_THREADS=1` exported
before any import, plus `faiss.omp_set_num_threads(1)`), one warm-up pass
discarded, median of 3 repeats. Slow configurations stop after a 20-second
budget per repeat rather than running all 5 000 queries; the number actually
timed is recorded per row. QPS is measured on the tuning queries and recall is
reported on the held-out queries — throughput is a property of the index and
its search parameter, not of which queries are used, and
`scripts/check_timing_split.py` measures that claim rather than assuming it.

**Index building** uses all cores (search does not); the thread count is
recorded per row. HNSW graph construction is not bit-reproducible when
multi-threaded — set `build.threads: 1` in the config for a deterministic
build at the cost of a much longer run.

**Trade-off curves** are Pareto envelopes over all build configurations and
search parameters of an index type, selected on tuning recall and plotted at
held-out recall. QPS at recall 0.90 / 0.95 is interpolated log-linearly
between the two bracketing frontier points; an index that never reaches a
target is reported as "not reached" with its best recall, not extrapolated.

---

## Reproducing a specific number

Every row in `results/results.csv` carries the git commit, machine id, build
parameters, search parameter, thread counts, number of queries timed and the
individual repeat timings. Figures and tables are generated from that file
alone:

```bash
python3 -m annbench.experiments.make_figures   # results/figures/
python3 -m annbench.experiments.make_tables    # results/tables/
```

If those two commands reproduce the report's figures and tables from the CSV,
the report is reproducible by construction.

---

## Notes and known limitations

- HNSW comes from `hnswlib`, the IVF family from FAISS. The recall function,
  timing harness and plots are written here, as the ground rules require.
- Index size is the serialized size on disk, which is within a few percent of
  resident memory for these index types but is not the same thing as RSS.
- Multi-threaded HNSW builds are not bit-reproducible; everything else is
  seeded (`seed: 42` in the config), including the IVF k-means and the Part 4
  generator.
- The `ann-benchmarks` framework is not used anywhere; only its data files are.
