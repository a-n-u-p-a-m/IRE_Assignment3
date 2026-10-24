#!/usr/bin/env bash
#
# Assemble the submission zip in the layout the brief specifies.
#
#   bash scripts/make_submission.sh <rollnumber> [path/to/report.pdf]
#
# The brief puts results/ and ai_log/ at the ZIP root, beside code/ — not
# inside it. During development everything lives under code/ so the repository
# is self-contained and runnable; this script rearranges it at packaging time.
#
# Resulting zip:
#
#   <rollnumber>_A3.zip
#   ├── report.pdf
#   ├── code/            the git repository, INCLUDING .git
#   │   ├── run_all.sh, README.md, requirements.txt, hypothesis.md
#   │   ├── configs/ src/ tests/ scripts/ notebooks/ docs/
#   │   └── .git/
#   ├── results/
#   │   ├── results.csv
#   │   ├── figures/
#   │   └── tables/, env.json
#   └── ai_log/
#       └── ai_statement.md (+ prompt/response logs, if any)
#
# Datasets, caches and index files are excluded, as required.

set -euo pipefail
cd "$(dirname "$0")/.."
CODE_DIR="$(pwd)"
PARENT="$(dirname "$CODE_DIR")"

ROLL="${1:-}"
REPORT="${2:-$PARENT/report.pdf}"
if [[ -z "$ROLL" ]]; then
  echo "usage: bash scripts/make_submission.sh <rollnumber> [report.pdf]" >&2
  exit 2
fi

STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
OUT="$PARENT/${ROLL}_A3.zip"

# --- checks the marker will make, made here first --------------------------
fail=0
check() { if [[ $2 -eq 0 ]]; then echo "  ok    $1"; else echo "  MISSING  $1"; fail=1; fi; }

echo "Pre-flight:"
check "report.pdf"                 "$([[ -f "$REPORT" ]] && echo 0 || echo 1)"
check "git repository with history" "$([[ -d .git ]] && echo 0 || echo 1)"
check "hypothesis.md"              "$([[ -f hypothesis.md ]] && echo 0 || echo 1)"
check "README.md"                  "$([[ -f README.md ]] && echo 0 || echo 1)"
check "requirements.txt"           "$([[ -f requirements.txt ]] && echo 0 || echo 1)"
check "run_all.sh"                 "$([[ -f run_all.sh ]] && echo 0 || echo 1)"
check "results/results.csv"        "$([[ -s results/results.csv ]] && echo 0 || echo 1)"
check "results/figures/*"          "$(compgen -G 'results/figures/*' >/dev/null && echo 0 || echo 1)"
check "ai_log/ai_statement.md"     "$([[ -f "$PARENT/ai_log/ai_statement.md" ]] && echo 0 || echo 1)"

if [[ -d .git ]]; then
  if git log --oneline -- hypothesis.md >/dev/null 2>&1 && [[ -n "$(git log --oneline -- hypothesis.md)" ]]; then
    H=$(git log --format=%ct -1 --diff-filter=A -- hypothesis.md)
    R=$(git log --format=%ct -1 --diff-filter=A -- results/results.csv 2>/dev/null || echo "")
    if [[ -n "$R" && "$H" -ge "$R" ]]; then
      echo "  WARNING  hypothesis.md was not committed before results.csv"
      fail=1
    else
      echo "  ok    hypothesis.md committed before any results"
    fi
  else
    echo "  MISSING  hypothesis.md is not committed"; fail=1
  fi
  if [[ -n "$(git status --porcelain)" ]]; then
    echo "  WARNING  working tree is dirty — commit before packaging"
  fi
fi
[[ $fail -eq 0 ]] || { echo; echo "Fix the items above, then re-run."; exit 1; }

# --- stage ------------------------------------------------------------------
echo
echo "Staging ..."
mkdir -p "$STAGE/code" "$STAGE/results" "$STAGE/ai_log"

cp "$REPORT" "$STAGE/report.pdf"

# the repo, .git included, datasets and caches excluded
rsync -a \
  --exclude 'data/*.hdf5' --exclude 'data/cache' \
  --exclude '__pycache__' --exclude '.pytest_cache' --exclude '.ipynb_checkpoints' \
  --exclude '*.pyc' --exclude '*.index' --exclude '*.faiss' --exclude '*.bin' \
  --exclude 'results/quick' \
  ./ "$STAGE/code/"

cp results/results.csv "$STAGE/results/"
cp -r results/figures "$STAGE/results/"
[[ -d results/tables ]] && cp -r results/tables "$STAGE/results/"
[[ -f results/env.json ]] && cp results/env.json "$STAGE/results/"

cp -r "$PARENT/ai_log/." "$STAGE/ai_log/"

# The brief says the zip must contain EXACTLY report.pdf, code/, results/ and
# ai_log/ "and nothing else", so results/ is NOT duplicated inside code/.
# run_all.sh regenerates everything into code/results/ on a fresh clone.

rm -f "$OUT"
(cd "$STAGE" && zip -qr "$OUT" .)

echo
echo "Wrote $OUT"
unzip -l "$OUT" | tail -n 1
echo
echo "Top level:"
unzip -l "$OUT" | awk '{print $4}' | grep -E '^[^/]+/?$' | sort -u | sed 's/^/  /'
