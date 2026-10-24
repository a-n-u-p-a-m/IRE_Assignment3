#!/usr/bin/env bash
#
# One-command reproduction of every number, figure and table in the report.
#
#   bash run_all.sh            full run  (hours; the numbers in report.pdf)
#   bash run_all.sh --quick    small run (<15 min on a laptop, 100K subsets)
#
# Options:
#   --quick          use configs/quick.yaml instead of configs/full.yaml
#   --force          recompute rows already present in results/results.csv
#   --skip-tests     skip the unit tests (not recommended)
#   --only PART      run one stage only: tests|part1|part2|part3|part4|figures
#   --config FILE    use an explicit config file
#
# The script is resumable: every measurement is appended to results/results.csv
# as it completes, and a re-run skips what is already there. Delete the CSV (or
# pass --force) to start clean.

set -euo pipefail
cd "$(dirname "$0")"

# ---------------------------------------------------------------------------
# Single-threaded search is part of the measurement protocol, and these have to
# be set BEFORE numpy/FAISS are imported - setting them in Python is too late
# for some BLAS builds. Index *building* re-raises the thread count internally.
# ---------------------------------------------------------------------------
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

CONFIG="configs/full.yaml"
MODE="full"
FORCE=""
SKIP_TESTS=0
ONLY=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --quick)      CONFIG="configs/quick.yaml"; MODE="quick"; shift ;;
    --force)      FORCE="--force"; shift ;;
    --skip-tests) SKIP_TESTS=1; shift ;;
    --only)       ONLY="$2"; shift 2 ;;
    --config)     CONFIG="$2"; MODE="custom"; shift 2 ;;
    -h|--help)    sed -n '2,25p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

# Quick and full runs NEVER share a results file. They use different database
# sizes, query counts, repeat counts and time budgets, so pooling them silently
# corrupts every frontier, table and figure built from the file -- and because
# the resume key does not include the measurement protocol, a quick row also
# BLOCKS the full run from re-measuring that point.
if [[ "$MODE" == "quick" ]]; then
  RESULTS="results/results_quick.csv"
else
  RESULTS="results/results.csv"
fi
FIGURES="results/figures"
TABLES="results/tables"
mkdir -p "$FIGURES" "$TABLES" data/cache

run_stage() { [[ -z "$ONLY" || "$ONLY" == "$1" ]]; }
banner() { printf '\n\033[1m=== %s ===\033[0m\n' "$1"; }

START=$(date +%s)
banner "ANN benchmark - $MODE run ($CONFIG)"
python3 - <<'PY'
import os, sys
sys.path.insert(0, "src")
from annbench.env import collect_env
for k, v in sorted(collect_env(os.getcwd()).items()):
    print(f"  {k:18s} {v}")
PY

# --- 0. tests ---------------------------------------------------------------
if run_stage tests && [[ $SKIP_TESTS -eq 0 ]]; then
  banner "Unit tests (recall function, frontier, index wrappers, generator)"
  python3 -m pytest tests -q
fi

# --- 1. data ----------------------------------------------------------------
# Part 4 generates its own data and needs no download; Parts 1-3 do.
if [[ -z "$ONLY" || "$ONLY" == part1 || "$ONLY" == part2 || "$ONLY" == part3 ]]; then
  if [[ ! -f data/sift-128-euclidean.hdf5 || ! -f data/glove-100-angular.hdf5 ]]; then
    echo
    echo "Dataset files are missing from data/."
    echo "Fetch them once with:   bash scripts/download_data.sh"
    echo "(They are deliberately not in this repository.)"
    exit 1
  fi
fi

# --- 2. experiments ---------------------------------------------------------
if run_stage part1; then
  banner "Part 1: exact baseline, ground-truth check, single vs batch timing"
  python3 -m annbench.experiments.part1_exact --config "$CONFIG" --results "$RESULTS" --tables "$TABLES" $FORCE
fi

if run_stage part2; then
  banner "Part 2: HNSW / IVF-Flat / IVF-PQ on both datasets"
  python3 -m annbench.experiments.part2_benchmark --config "$CONFIG" --results "$RESULTS" $FORCE
fi

if run_stage part3; then
  banner "Part 3: scaling on SIFT"
  python3 -m annbench.experiments.part3_scaling --config "$CONFIG" --results "$RESULTS" $FORCE
fi

if run_stage part4; then
  banner "Part 4: intrinsic dimensionality on generated data"
  python3 -m annbench.experiments.part4_intrinsic --config "$CONFIG" --results "$RESULTS" --tables "$TABLES" $FORCE
fi

# --- 3. report artefacts ----------------------------------------------------
if run_stage figures; then
  banner "Figures and tables (regenerated from results/results.csv only)"
  python3 -m annbench.experiments.make_figures --results "$RESULTS" --figures "$FIGURES"
  python3 -m annbench.experiments.make_tables  --results "$RESULTS" --tables  "$TABLES"
fi

END=$(date +%s)
banner "Done in $(( (END-START)/60 ))m $(( (END-START)%60 ))s"
echo "  results : $RESULTS   ($(( $(wc -l < "$RESULTS") - 1 )) rows)"
echo "  figures : $FIGURES/"
echo "  tables  : $TABLES/"
echo "  env     : results/env.json"
if [[ "$MODE" == "quick" ]]; then
  echo
  echo "NOTE: --quick uses 100K-vector subsets and a coarse parameter grid."
  echo "      Its numbers are not the numbers in report.pdf; it exists to show"
  echo "      the pipeline runs end to end."
fi
