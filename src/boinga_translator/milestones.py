"""Milestone copies: model weights + config of chosen run checkpoints, kept apart on Drive."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from tqdm import tqdm

from boinga_translator import ckpt_sampler, paths, pretrained

KEEP_FILE = "keep.json"  # written last: a milestone without it is incomplete
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
    part.replace(target)
    shutil.copy2(src.parent / pretrained.CONFIG_FILE, out / pretrained.CONFIG_FILE)
    mb = target.stat().st_size / 1e6
    meta = out / f"{KEEP_FILE}.part"
    meta.write_text(json.dumps({
        "run": run_name, "step": step, "source": src.name, "source_folder": src.parent.name,
        "n_tensors": len(model), "mb": round(mb, 1)}, indent=2), encoding="utf-8")
    meta.replace(out / KEEP_FILE)  # atomic: a half-written keep.json never marks a step done
    return mb


def keep(run_name: str, steps: list[int] | None = None) -> int:
    """Copy each checkpoint minus optimizer/scaler, + config, of each step to milestones/<run>/step_<N>/."""
    ckpts = ckpt_sampler.checkpoints(run_name)
    steps = sorted(set(steps)) if steps else default_steps(run_name)
    if missing := [s for s in steps if s not in ckpts]:
        raise FileNotFoundError(f"no checkpoint for step {', '.join(map(str, missing))} in "
                                f"run {run_name} (have {sorted(ckpts)})")
    paths.milestones().mkdir(parents=True, exist_ok=True)
    todo = [s for s in steps if not (milestone_dir(run_name, s) / KEEP_FILE).is_file()]
    for step in tqdm(todo, desc="keep", unit="ckpt"):
        mb = _keep_one(run_name, step, ckpts[step])
        print(f"step {step}: kept {mb:.0f} MB from {ckpts[step].name}")
    free = shutil.disk_usage(paths.milestones()).free / 1e9
    print(f"PASS: kept {len(todo)} milestones ({len(steps) - len(todo)} already kept) in "
          f"{paths.milestones() / run_name}; {free:.1f} GB free")
    return len(todo)
