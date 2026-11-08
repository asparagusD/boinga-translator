"""Milestone copies: model weights + config of chosen run checkpoints, kept apart on Drive."""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

from tqdm import tqdm

from boinga_translator import ckpt_sampler, paths, pretrained, train_runner

KEEP_FILE = "keep.json"  # written last: a milestone without it is incomplete
MANIFEST = "kept_milestones.json"  # in the run folder on the VM disk: what check expects
DROP = ("optimizer", "scaler")  # training state only; everything else is kept


def milestone_dir(run_name: str, step: int) -> Path:
    """Drive folder of one kept step: milestones/<run>/step_<N>/."""
    return paths.milestones() / run_name / f"step_{step}"


def default_steps(run_name: str) -> list[int]:
    """The steps the listening form played (4 best by eval loss_mel + the last), sorted."""
    return sorted(ckpt_sampler.listened_steps(run_name))


def _load(path: Path) -> dict[str, Any]:
    import torch

    return torch.load(path, map_location="cpu", weights_only=False)  # our own checkpoint


def _save(obj: dict[str, Any], path: Path) -> None:
    import torch

    torch.save(obj, path)


def _equal(a: Any, b: Any) -> bool:
    return bool(a.equal(b)) if hasattr(a, "equal") else a == b  # torch.Tensor.equal


def _check(model: dict[str, Any], copy: dict[str, Any], name: str) -> None:
    if copy.keys() != model.keys() or not all(_equal(model[k], copy[k]) for k in model):
        raise RuntimeError(f"milestone copy differs from {name}")


def _keep_one(run_name: str, step: int, src: Path) -> float:
    out = milestone_dir(run_name, step)
    out.mkdir(parents=True, exist_ok=True)
    state = _load(src)
    model, target, part = state["model"], out / pretrained.MODEL_FILE, out / "model.pth.part"
    _save({k: v for k, v in state.items() if k not in DROP}, part)
    try:
        _check(model, _load(part)["model"], src.name)
    except Exception as exc:  # also a truncated file (EOFError, unpickling errors)
        part.unlink(missing_ok=True)
        if isinstance(exc, RuntimeError):
            raise
        raise RuntimeError(f"milestone copy of {src.name} unreadable: {exc!r}") from exc
    digest = md5(part)  # hashed before the rename: the local write, not an upload check
    part.replace(target)
    shutil.copy2(src.parent / pretrained.CONFIG_FILE, out / pretrained.CONFIG_FILE)
    mb = target.stat().st_size / 1e6
    meta = out / f"{KEEP_FILE}.part"
    meta.write_text(json.dumps({
        "run": run_name, "step": step, "source": src.name, "source_folder": src.parent.name,
        "n_tensors": len(model), "mb": round(mb, 1), "bytes": target.stat().st_size,
        "md5": digest},
        indent=2), encoding="utf-8")
    meta.replace(out / KEEP_FILE)  # atomic: a half-written keep.json never marks a step done
    return mb


def md5(path: Path) -> str:
    """Hex md5 of a file (Drive reports md5Checksum for uploaded files)."""
    digest = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            digest.update(block)
    return digest.hexdigest()


def _kept(run_name: str, step: int, src: Path) -> bool:
    """Step already kept from this run folder; one kept from another folder is refused."""
    meta = milestone_dir(run_name, step) / KEEP_FILE
    if not meta.is_file():
        return False
    folder = json.loads(meta.read_text(encoding="utf-8")).get("source_folder")
    if folder != src.parent.name:
        raise RuntimeError(f"{meta.parent} holds step {step} of {folder}, not of "
                           f"{src.parent.name}: move it away by hand before keeping this run")
    return True


def keep(run_name: str, steps: list[int] | None = None) -> int:
    """Copy each checkpoint minus optimizer/scaler, + config, of each step to milestones/<run>/step_<N>/."""
    ckpts = ckpt_sampler.checkpoints(run_name)
    steps = sorted(set(steps)) if steps else default_steps(run_name)
    if missing := [s for s in steps if s not in ckpts]:
        raise FileNotFoundError(f"no checkpoint for step {', '.join(map(str, missing))} in "
                                f"run {run_name} (have {sorted(ckpts)})")
    paths.milestones().mkdir(parents=True, exist_ok=True)
    todo = [s for s in steps if not _kept(run_name, s, ckpts[s])]
    need = sum(ckpts[s].stat().st_size for s in todo) / 1e9  # copies are smaller: a margin
    if (free := train_runner.free_gb(paths.milestones())) < need:
        raise RuntimeError(f"only {free:.1f} GB free on Drive for {len(todo)} milestones "
                           f"(need up to {need:.1f} GB): free Drive space and empty its trash")
    for step in tqdm(todo, desc="keep", unit="ckpt"):
        mb = _keep_one(run_name, step, ckpts[step])
        print(f"step {step}: kept {mb:.0f} MB from {ckpts[step].name}")
    manifest = {str(s): json.loads((milestone_dir(run_name, s) / KEEP_FILE).read_text(
        encoding="utf-8")) for s in steps}
    paths.run_dir(run_name).joinpath(MANIFEST).write_text(json.dumps(manifest, indent=2),
                                                          encoding="utf-8")
    free = shutil.disk_usage(paths.milestones()).free / 1e9
    print(f"PASS: kept {len(todo)} milestones ({len(steps) - len(todo)} already kept) in "
          f"{paths.milestones() / run_name}; {free:.1f} GB free")
    return len(todo)


Listing = dict[str, dict[str, Any]]  # file name -> {"size": int, "md5": str | None}


def mount_view(folder: Path) -> Listing:
    """name -> {size, md5 None} of files in a folder as the Drive mount shows it."""
    if not folder.is_dir():
        return {}
    return {p.name: {"size": p.stat().st_size, "md5": None} for p in folder.iterdir()
            if p.is_file()}


def _problems(info: dict[str, Any], files: Listing, api: bool) -> list[str]:
    got = files.get(pretrained.MODEL_FILE, {})
    out = []
    if info.get("bytes") is None:
        out.append("no recorded size: keep again")
    elif got.get("size") != info["bytes"]:
        out.append(f"model {got.get('size')} bytes, expected {info['bytes']}")
    if api and info.get("md5") is None:
        out.append("no recorded md5: delete this step folder on Drive, then keep again")
    elif api and got.get("md5") != info["md5"]:
        out.append("model md5 differs")
    if pretrained.CONFIG_FILE not in files:
        out.append("no config.json")
    return out


def check(run_name: str, remote: Callable[[Path], Listing] | None = None) -> int:
    """Each step `keep` recorded (MANIFEST, VM disk) is on Drive: config + model size (+ md5).

    remote lists a Drive folder through the Drive API (drive_api.lister): what Drive really
    holds. Without it the Drive mount is read, which may still show local copies of uploads
    that failed (C8 run 1), so the notebook uses the API.
    """
    root, record = paths.milestones() / run_name, paths.run_dir(run_name) / MANIFEST
    if not record.is_file():
        raise FileNotFoundError(f"no {record}: run `boinga keep` first, in the VM that trained")
    expected = json.loads(record.read_text(encoding="utf-8"))  # VM side: never lost in upload
    view, bad = remote or mount_view, []
    for step, info in sorted(expected.items(), key=lambda kv: int(kv[0])):
        d = milestone_dir(run_name, int(step))
        try:
            problems = _problems(info, view(d), remote is not None)
        except FileNotFoundError as exc:  # the whole step folder never reached Drive
            problems = [str(exc)]
        print(f"{d.name}: {'ok' if not problems else '; '.join(problems)}")
        bad += [d.name] if problems else []
    if bad:
        raise RuntimeError(f"milestones not on Drive: {bad}")
    print(f"PASS: {len(expected)} milestones on Drive under {root} "
          f"({'Drive API' if remote else 'mount'})")
    return len(expected)
