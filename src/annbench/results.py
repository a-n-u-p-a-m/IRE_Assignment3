"""The single source of truth for every number in the report.

Design rule for this project: **benchmarks only ever append rows to
``results/results.csv``; figures and tables only ever read that file.**  No
number is carried in memory from a measurement to a plot.  Two things follow
from that.

*   Reproduction is real rather than aspirational.  ``make_figures`` and
    ``make_tables`` can regenerate every figure and table in the report from
    the CSV alone, which is exactly what the deliverable asks for.
*   A crash six hours into a sweep costs one configuration, not the sweep.
    Rows are flushed as they are produced and :meth:`ResultsWriter.has` lets a
    restarted run skip work that is already recorded.  ``--force`` overrides.

Each row carries enough provenance (git commit, machine id, library versions,
thread counts, how many queries were actually timed) that a number in the
report can be traced back to the conditions that produced it.
"""

from __future__ import annotations

import csv
import json
import os
import uuid
from datetime import datetime, timezone

__all__ = ["COLUMNS", "ResultsWriter", "make_row", "load_results"]

COLUMNS = [
    "run_id",
    "timestamp",
    "experiment",      # part1 | part2 | part3 | part4
    "dataset",         # sift | glove | synthetic
    "variant",         # free-form: the thing being varied in parts 3/4
    "n_vectors",
    "dim",
    "index_type",      # Flat | HNSW | IVF-Flat | IVF-PQ
    "params_json",     # build parameters
    "search_param_name",
    "search_param",
    "k",
    "recall_tune",
    "recall_test",
    "qps_median",
    "qps_runs_json",
    "timed_on",        # which query split the timing loop used
    "n_queries_timed",
    "batch_qps",
    "build_time_s",
    "build_threads",
    "index_bytes",
    "bytes_per_vector",
    "git_commit",
    "machine_id",
    "notes",
]


def make_row(**kwargs) -> dict:
    """Build a row with every column present, so the CSV stays rectangular."""
    row = {c: "" for c in COLUMNS}
    row["run_id"] = uuid.uuid4().hex[:12]
    row["timestamp"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    unknown = set(kwargs) - set(COLUMNS)
    if unknown:
        raise KeyError(f"unknown result columns: {sorted(unknown)}")
    row.update({k: ("" if v is None else v) for k, v in kwargs.items()})
    return row


def _key(row: dict) -> tuple:
    """Identity of a measurement, for resume-and-skip."""
    return (
        str(row.get("experiment", "")),
        str(row.get("dataset", "")),
        str(row.get("variant", "")),
        str(row.get("n_vectors", "")),
        str(row.get("index_type", "")),
        str(row.get("params_json", "")),
        str(row.get("search_param", "")),
    )


class ResultsWriter:
    """Append-only CSV writer with a resume index."""

    def __init__(self, path: str, force: bool = False):
        self.path = path
        self.force = force
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        self._seen: set[tuple] = set()
        if os.path.exists(path) and not force:
            with open(path, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    self._seen.add(_key(row))
        if force or not os.path.exists(path):
            with open(path, "w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=COLUMNS).writeheader()

    def has(self, **kwargs) -> bool:
        """True when this exact measurement is already in the file."""
        return (not self.force) and _key(kwargs) in self._seen

    def append(self, row: dict) -> None:
        missing = set(COLUMNS) - set(row)
        if missing:
            raise KeyError(f"row missing columns: {sorted(missing)}")
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=COLUMNS).writerow(row)
            f.flush()
            os.fsync(f.fileno())
        self._seen.add(_key(row))

    def write(self, **kwargs) -> dict:
        row = make_row(**kwargs)
        self.append(row)
        return row


def load_results(path: str):
    """Read ``results.csv`` into a pandas DataFrame with sane dtypes."""
    import pandas as pd

    df = pd.read_csv(path)
    numeric = [
        "n_vectors",
        "dim",
        "search_param",
        "k",
        "recall_tune",
        "recall_test",
        "qps_median",
        "n_queries_timed",
        "batch_qps",
        "build_time_s",
        "build_threads",
        "index_bytes",
        "bytes_per_vector",
    ]
    for col in numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "qps_runs_json" in df.columns:
        df["qps_runs"] = df["qps_runs_json"].apply(
            lambda s: json.loads(s) if isinstance(s, str) and s else []
        )
    return df
