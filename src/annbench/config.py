"""Configuration loading.

``--quick`` is not a separate code path.  Full and quick runs execute exactly
the same functions and differ only in a YAML file: database size, how dense the
parameter grids are, how many timing repeats, and the per-repeat time budget.
That is deliberate — a quick mode implemented as ``if quick: ...`` branches
tests something other than the thing it is supposed to be a fast version of.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

__all__ = ["load_config", "Config", "timing_config_from"]


@dataclass
class Config:
    raw: dict

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def get(self, key: str, default=None):
        return self.raw.get(key, default)

    @property
    def name(self) -> str:
        return self.raw.get("name", "unnamed")

    @property
    def seed(self) -> int:
        return int(self.raw.get("seed", 42))

    @property
    def k(self) -> int:
        return int(self.raw.get("k", 10))


def load_config(path: str) -> Config:
    import yaml

    if not os.path.exists(path):
        raise FileNotFoundError(f"config not found: {path}")
    with open(path, encoding="utf-8") as f:
        return Config(yaml.safe_load(f))


def timing_config_from(cfg: Config):
    from .timing import TimingConfig

    t = cfg.get("timing", {}) or {}
    return TimingConfig(
        repeats=int(t.get("repeats", 3)),
        warmup=bool(t.get("warmup", True)),
        max_queries=int(t.get("max_timed_queries", 5000)),
        min_queries=int(t.get("min_timed_queries", 500)),
        time_budget_s=float(t.get("time_budget_s", 20.0)),
        check_every=int(t.get("check_every", 32)),
    )
