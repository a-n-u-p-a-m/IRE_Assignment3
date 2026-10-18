"""Capture the machine and library versions that produced a result.

The ground rules require the report to name the machine and the library
versions.  Writing them by hand invites them to drift away from reality, so
they are collected automatically into ``results/env.json`` at the start of
every run and a short Markdown version is generated for the report appendix.
"""

from __future__ import annotations

import json
import os
import platform
import re
import subprocess
import sys

__all__ = ["collect_env", "write_env", "git_commit", "machine_id"]


def _run(cmd: list[str]) -> str:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        return out.stdout.strip()
    except Exception:
        return ""


def git_commit(repo_dir: str | None = None) -> str:
    cwd = repo_dir or os.getcwd()
    try:
        out = subprocess.run(
            ["git", "-C", cwd, "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        commit = out.stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", cwd, "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
        return f"{commit}{'+dirty' if dirty else ''}" if commit else "no-git"
    except Exception:
        return "no-git"


def _cpu_model() -> str:
    if platform.system() == "Linux":
        try:
            with open("/proc/cpuinfo", encoding="utf-8") as f:
                for line in f:
                    if line.lower().startswith("model name"):
                        return line.split(":", 1)[1].strip()
        except Exception:
            pass
    if platform.system() == "Darwin":
        return _run(["sysctl", "-n", "machdep.cpu.brand_string"])
    return platform.processor() or "unknown"


def _total_ram_gb() -> float:
    try:
        if platform.system() == "Linux":
            with open("/proc/meminfo", encoding="utf-8") as f:
                for line in f:
                    if line.startswith("MemTotal"):
                        kb = int(re.findall(r"\d+", line)[0])
                        return round(kb / 1024 / 1024, 1)
        if platform.system() == "Darwin":
            return round(int(_run(["sysctl", "-n", "hw.memsize"]) or 0) / 1024**3, 1)
    except Exception:
        pass
    return float("nan")


def _version(mod: str) -> str:
    try:
        m = __import__(mod)
        return str(getattr(m, "__version__", "unknown"))
    except Exception:
        return "not installed"


def machine_id() -> str:
    """A short, stable, non-identifying label for the host."""
    return f"{platform.system()}-{platform.machine()}-{(os.cpu_count() or 0)}core"


def collect_env(repo_dir: str | None = None) -> dict:
    return {
        "machine_id": machine_id(),
        "cpu_model": _cpu_model(),
        "cpu_count": os.cpu_count(),
        "total_ram_gb": _total_ram_gb(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "numpy": _version("numpy"),
        "faiss": _version("faiss"),
        "hnswlib": _version("hnswlib"),
        "h5py": _version("h5py"),
        "pandas": _version("pandas"),
        "matplotlib": _version("matplotlib"),
        "git_commit": git_commit(repo_dir),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS", "<unset>"),
    }


def write_env(path: str = "results/env.json", repo_dir: str | None = None) -> dict:
    info = collect_env(repo_dir)
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, sort_keys=True)
    md = os.path.splitext(path)[0] + ".md"
    with open(md, "w", encoding="utf-8") as f:
        f.write("# Measurement environment\n\n")
        f.write("| key | value |\n|---|---|\n")
        for k, v in sorted(info.items()):
            f.write(f"| {k} | {v} |\n")
    return info
