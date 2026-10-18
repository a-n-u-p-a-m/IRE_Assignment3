"""Shared entry-point plumbing for the experiment scripts.

Every ``part*.py`` script takes the same flags (``--config``, ``--results``,
``--force``, ``--quiet``) and does the same three things before it starts
measuring: load the config, record the environment, open the results file.
That lives here so the scripts themselves contain only their experiment.

Ground-truth caching
--------------------
Subsetting a dataset invalidates the neighbour ids shipped with it, so every
subset needs exact ground truth recomputed.  That is a brute-force pass and
several scripts want the same subsets, so results are cached under
``data/cache/`` keyed by (dataset, n, seed, k).  The cache stores ids and
distances only — never the vectors — so it stays small and the deliverable
rule about not shipping datasets is easy to honour.
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from ..config import load_config, timing_config_from
from ..datasets import Dataset, load_hdf5, split_queries, subsample
from ..env import write_env
from ..results import ResultsWriter

__all__ = ["base_parser", "setup", "get_dataset", "splits_of", "REPO_ROOT"]

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def base_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", default="configs/full.yaml", help="YAML config file")
    p.add_argument("--results", default="results/results.csv", help="results CSV to append to")
    p.add_argument("--force", action="store_true", help="recompute rows already present")
    p.add_argument("--quiet", action="store_true", help="less progress output")
    return p


def setup(args):
    """Load config, capture the environment, open the results writer."""
    cfg = load_config(args.config)
    env = write_env(os.path.join(os.path.dirname(args.results) or "results", "env.json"),
                    repo_dir=REPO_ROOT)
    writer = ResultsWriter(args.results, force=args.force)
    timing_cfg = timing_config_from(cfg)
    if not args.quiet:
        print(f"config: {cfg.name}  ({args.config})")
        print(f"machine: {env['machine_id']} | faiss {env['faiss']} | commit {env['git_commit']}")
    return cfg, writer, env, timing_cfg


# ----------------------------------------------------------------------
# dataset access with on-disk ground-truth cache
# ----------------------------------------------------------------------
def _cache_path(name: str, n: int, seed: int, kgt: int, cache_dir: str) -> str:
    return os.path.join(cache_dir, f"gt_{name}_n{n}_seed{seed}_k{kgt}.npz")


def get_dataset(
    cfg,
    name: str,
    n_vectors: int | None = None,
    kgt: int = 100,
    cache_dir: str = "data/cache",
    verbose: bool = True,
) -> Dataset:
    """Load a configured dataset, optionally subsampled, with exact ground truth."""
    spec = cfg["datasets"][name]
    ds = load_hdf5(
        spec["path"],
        name=name,
        normalize=bool(spec.get("normalize", False)),
        kgt=kgt,
    )
    target = n_vectors if n_vectors is not None else spec.get("n_vectors")
    if target is None or int(target) >= ds.n:
        return ds

    target = int(target)
    os.makedirs(cache_dir, exist_ok=True)
    path = _cache_path(name, target, cfg.seed, kgt, cache_dir)
    rng = np.random.default_rng(cfg.seed)
    idx = rng.choice(ds.n, size=target, replace=False)
    idx.sort()

    if os.path.exists(path):
        if verbose:
            print(f"  ground truth for {name}@{target} from cache")
        z = np.load(path)
        from dataclasses import replace as _replace

        return _replace(
            ds,
            name=f"{name}@{target}",
            train=np.ascontiguousarray(ds.train[idx]),
            gt_ids=z["ids"],
            gt_dists=z["dists"],
            note=f"random subset of {ds.n} (seed={cfg.seed}); ground truth from cache",
        )

    if verbose:
        print(f"  computing exact ground truth for {name}@{target} ...", flush=True)
    sub = subsample(ds, target, seed=cfg.seed, kgt=kgt)
    np.savez_compressed(path, ids=sub.gt_ids, dists=sub.gt_dists)
    return sub


def splits_of(ds: Dataset, cfg):
    """(tuning split, held-out split) as configured."""
    half_tune = int(cfg.get("tune_queries", 5000))
    half_test = int(cfg.get("test_queries", 5000))
    return (
        split_queries(ds, "tune", half=half_tune),
        split_queries(ds, "test", half=half_test),
    )
