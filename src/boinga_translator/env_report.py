"""Record python, torch/CUDA, GPU, coqui-tts, numpy, git and disk info to reports/env.json."""

from __future__ import annotations

import datetime as dt
import importlib.metadata as md
import json
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Any

from boinga_translator import paths, report_io

OPTIONAL = (("torch", "torch"), ("coqui_tts", "coqui-tts"))  # (env key, package name)


def _version(dist: str) -> str | None:
    try:
        return md.version(dist)
    except md.PackageNotFoundError:
        return None


def _torch_info() -> dict[str, Any]:
    """torch/CUDA/GPU facts; a broken install (DLL, CUDA, RuntimeError) lands in torch_error."""
    info: dict[str, Any] = {
        "torch": None, "torch_cuda": None, "torch_error": None,
        "gpu_name": "none", "gpu_mem_gb": None,
    }
    try:
        import torch
    except ImportError:
        return info
    except Exception as err:  # noqa: BLE001 - broken install must be reported, not raised
        info["torch_error"] = f"{type(err).__name__}: {err}"
        return info
    info["torch"], info["torch_cuda"] = torch.__version__, torch.version.cuda
    try:
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            info["gpu_name"] = props.name
            info["gpu_mem_gb"] = round(props.total_memory / 1024**3, 1)
    except Exception as err:  # noqa: BLE001 - CUDA driver errors must be reported
        info["torch_error"] = f"{type(err).__name__}: {err}"
    return info


def _git_commit() -> str:
    try:
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=paths.REPO,
                           capture_output=True, text=True, timeout=10, check=False)
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return r.stdout.strip() if r.returncode == 0 and r.stdout.strip() else "unknown"


def _free_gb(path: Path) -> float | None:
    """Free GiB on the disk holding `path` (or its nearest existing parent); None on error."""
    try:
        while not path.exists() and path != path.parent:
            path = path.parent
        return round(shutil.disk_usage(path).free / 1024**3, 1)
    except OSError:  # e.g. a stale Drive mount
        return None


def collect() -> dict[str, Any]:
    """Gather the environment facts; never raises for a missing optional package."""
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "is_colab": paths.is_colab(),
        **_torch_info(),
        "coqui_tts": _version("coqui-tts"),
        "numpy": _version("numpy"),
        "git_commit": _git_commit(),
        "disk_free_gb_root": _free_gb(paths.root()),
        "disk_free_gb_fast": _free_gb(paths.fast()),
        "utc_time": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    }


def run(out_rel: str = "reports/env.json") -> dict[str, Any]:
    """Collect, save (repo + Drive copy), print a table and the BEGIN/END block, then PASS."""
    env = collect()
    if env["torch_error"]:
        print(f"WARN: torch import failed: {env['torch_error']}")
    for key, pkg in OPTIONAL:
        if env[key] is None and not (pkg == "torch" and env["torch_error"]):
            print(f"WARN: {pkg} not installed")
    repo_path, _ = report_io.write_report(json.dumps(env, indent=2, sort_keys=True) + "\n",
                                          out_rel)
    width = max(len(k) for k in env)
    for key, value in env.items():
        print(f"{key:<{width}}  {value}")
    report_io.emit(repo_path, out_rel)
    print("PASS: env recorded")
    return env
