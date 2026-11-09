"""Record the environment (reports/env.json) and the reproducibility report (reports/repro.json)."""

from __future__ import annotations

import datetime as dt
import hashlib
import importlib.metadata as md
import json
import os
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Any

from boinga_translator import paths, report_io, settings

OPTIONAL = (("torch", "torch"), ("coqui_tts", "coqui-tts"))  # (env key, package name)
REPRO_REL = "reports/repro.json"
REPRO_PACKAGES = ("torch", "coqui-tts", "numpy", "librosa", "gradio")
REPRO_CONFIGS = ("vits_ctg_finetune.json", "splits_summary.json", "char_map.json")


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
        "cpu_count": os.cpu_count(),
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


def _sha256(path: Path) -> str | None:
    """sha256 of a file, or None (with a WARN) when it is missing."""
    if not path.is_file():
        print(f"WARN: missing configs/{path.name}")
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run_summary(run_dir: Path) -> dict[str, Any]:
    """Steps = highest checkpoint_<step>.pth on disk (not steps trained: a lost checkpoint
    lowers it); hours = sum of the boinga_run.json sessions that have a start and an end."""
    from boinga_translator import train_runner  # lazy: train_runner imports this module

    latest = train_runner.latest_run_folder(run_dir.name)
    summary: dict[str, Any] = {"steps": latest[1] if latest else None, "hours": None,
                               "sessions": 0}
    record_path = run_dir / train_runner.RUN_RECORD
    if not record_path.is_file():
        return summary
    try:
        sessions = json.loads(record_path.read_text(encoding="utf-8"))["sessions"]
        seconds = [
            (dt.datetime.fromisoformat(s["end_utc"]) - dt.datetime.fromisoformat(s["start_utc"]))
            .total_seconds()
            for s in sessions if s.get("start_utc") and s.get("end_utc")
        ]
    except (ValueError, TypeError, KeyError, AttributeError) as err:  # damaged record
        print(f"WARN: unreadable {run_dir.name}/{train_runner.RUN_RECORD}: "
              f"{type(err).__name__}: {err}")
        return summary
    summary["sessions"] = len(sessions)
    summary["hours"] = round(sum(seconds) / 3600, 2) if seconds else None
    return summary


def final(out_rel: str = REPRO_REL) -> dict[str, Any]:
    """Environment, package versions, config hashes, seed, best provenance and per-run
    steps/hours into reports/repro.json (repo + Drive copy); print it, then PASS."""
    provenance_path = paths.reports_dir() / "best_provenance.json"  # train_runner.promote
    provenance = None
    if provenance_path.is_file():
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    else:
        print("WARN: missing reports/best_provenance.json")
    runs_root = paths.runs()
    run_dirs = sorted(d for d in runs_root.iterdir() if d.is_dir()) if runs_root.is_dir() else []
    repro = {
        "env": collect(),
        "packages": {pkg: _version(pkg) for pkg in REPRO_PACKAGES},
        "config_sha256": {f"configs/{name}": _sha256(paths.configs_dir() / name)
                          for name in REPRO_CONFIGS},
        "seed": settings.SEED,
        "best_provenance": provenance,
        "runs": {d.name: _run_summary(d) for d in run_dirs},
    }
    repo_path, _ = report_io.write_report(
        json.dumps(repro, indent=2, sort_keys=True, ensure_ascii=False) + "\n", out_rel)
    report_io.emit(repo_path, out_rel)
    print("PASS: repro recorded")
    return repro
