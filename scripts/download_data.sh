#!/usr/bin/env bash
#
# Fetch the two ann-benchmarks datasets into data/.
#
# These files are deliberately NOT in the repository: the submission rules say
# not to ship the datasets, and they are ~1.5 GB together.
#
#   SIFT       1,000,000 x 128, Euclidean       ~ 0.5 GB
#   GloVe-100  1,183,514 x 100, angular         ~ 0.5 GB
#
# Each HDF5 file contains: train (the database), test (10,000 queries),
# neighbors (the true 100 nearest neighbours of each query) and distances.

set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data

BASE="http://ann-benchmarks.com"
FILES=("sift-128-euclidean.hdf5" "glove-100-angular.hdf5")

for f in "${FILES[@]}"; do
  if [[ -f "data/$f" ]]; then
    echo "already present: data/$f"
    continue
  fi
  echo "downloading $f ..."
  if command -v curl >/dev/null 2>&1; then
    curl -fL --retry 3 -o "data/$f.part" "$BASE/$f"
  else
    wget -c -O "data/$f.part" "$BASE/$f"
  fi
  mv "data/$f.part" "data/$f"
done

echo
echo "Datasets in data/:"
ls -lh data/*.hdf5

python3 - <<'PY'
import h5py, os
for f in sorted(os.listdir("data")):
    if not f.endswith(".hdf5"):
        continue
    with h5py.File(os.path.join("data", f), "r") as h:
        print(f"\n{f}")
        for key in ("train", "test", "neighbors", "distances"):
            if key in h:
                print(f"  {key:10s} {h[key].shape} {h[key].dtype}")
        print(f"  distance metric as published: {h.attrs.get('distance', 'unknown')}")
PY
