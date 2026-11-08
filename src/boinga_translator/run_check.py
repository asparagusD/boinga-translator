"""Upload check of a run's checkpoints: sizes and md5s recorded on the VM, compared on Drive."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from tqdm import tqdm

from boinga_translator import milestones, paths, pretrained, train_runner


def manifest_path(run_name: str) -> Path:
    """VM-disk record of a run's files (never on Drive, so a failed upload cannot hide it)."""
    paths.run_dir(run_name)  # validates the name
    return paths.manifests() / f"{run_name}.json"


def record(run_name: str) -> int:
    """Size + md5 of every checkpoint and the config of the run's latest folder, on the VM."""
    latest = train_runner.latest_run_folder(run_name)
    if latest is None:
        raise FileNotFoundError(f"no checkpoint under {paths.run_dir(run_name)}: train first")
    folder = latest[0]
    files = sorted(folder.glob("checkpoint_*.pth"), key=lambda p: train_runner._ckpt_step(p) or 0)
    files.append(folder / pretrained.CONFIG_FILE)
    data: dict[str, Any] = {"run": run_name, "folder": folder.name, "files": {}}
    for f in tqdm(files, desc="md5", unit="file"):  # ~1 GB each, read from the local cache
        data["files"][f.name] = {"bytes": f.stat().st_size, "md5": milestones.md5(f)}
        print(f"{f.name}: {f.stat().st_size / 1e6:.1f} MB")
    out = manifest_path(run_name)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"PASS: recorded {len(files)} files of run {run_name} ({folder.name})")
    return len(files)


def _verdict(want: dict[str, Any], got: dict[str, Any] | None, api: bool) -> str:
    if got is None:
        return "MISSING on Drive"
    if got.get("size") != want["bytes"]:
        return f"{got.get('size')} bytes on Drive, expected {want['bytes']}"
    if api and got.get("md5") != want["md5"]:
        return "md5 differs"
    return "ok"


def check(run_name: str,
          remote: Callable[[Path], milestones.Listing] | None = None) -> int:
    """Every recorded file is in the run folder on Drive with its size (+ md5 via the API).

    remote = drive_api.lister(): what Drive holds. Without it the mount is read, which may
    still show local copies of failed uploads (C8 run 1).
    """
    record_file = manifest_path(run_name)
    if not record_file.is_file():
        raise FileNotFoundError(f"no {record_file}: run `boinga record-run` first, in this VM")
    data = json.loads(record_file.read_text(encoding="utf-8"))
    folder = paths.run_dir(run_name) / data["folder"]
    try:
        found = (remote or milestones.mount_view)(folder)
    except FileNotFoundError as exc:
        raise RuntimeError(f"run folder not on Drive: {exc}") from exc
    bad = []
    for name, want in data["files"].items():
        why = _verdict(want, found.get(name), remote is not None)
        print(f"{name}: {why}")
        bad += [] if why == "ok" else [name]
    if bad:
        raise RuntimeError(f"not on Drive: {bad}")
    print(f"PASS: {len(data['files'])} files of run {run_name} on Drive "
          f"({'Drive API' if remote else 'mount'})")
    return len(data["files"])
