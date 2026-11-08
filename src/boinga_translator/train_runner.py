"""Training runner: coqui VITS loader check, fine-tune runs, smoke cleanup, best-model promotion."""

from __future__ import annotations

import contextlib
import datetime as dt
import hashlib
import inspect
import json
import os
import re
import shutil
import time
import types
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from boinga_translator import (
    env_report,
    paths,
    pretrained,
    report_io,
    settings,
    speaker_select,
    synth_basic,
    text_frontend,
    train_config,
)

RUN_RECORD = "boinga_run.json"  # in paths.run_dir(name): one entry per training session
CKPT_GB = 1.0  # one coqui VITS checkpoint with optimizer state (run 1: 997.8 MB)
MARGIN_GB = 2.0  # logs, TensorBoard events, eval audio

# Keyword arguments get_data_loader may take: 0.28 has config, *, is_eval, samples, verbose;
# assets/num_gpus are older coqui names. Only those in the printed signature are passed.
_LOADER_KWARGS = ("is_eval", "samples", "verbose", "assets", "num_gpus")


def _np(value: Any) -> np.ndarray:
    """A torch tensor or array-like as a numpy array (on the CPU)."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def judge_batch(batch: Mapping[str, Any], vocab_size: int,
                sample_rate: int) -> tuple[list[str], list[str]]:
    """(report lines, problems) for one batch: token ids in [0, vocab), floats finite."""
    lines: list[str] = []
    problems: list[str] = []
    if "tokens" not in batch:
        problems.append("batch has no tokens")
    for key, value in batch.items():
        if not hasattr(value, "shape"):
            if isinstance(value, list | tuple):
                lines.append(f"{key}: list of {len(value)}")
            continue
        arr = _np(value)
        head = f"{key}: shape {tuple(arr.shape)} {arr.dtype}"
        if key == "tokens" and arr.size == 0:
            problems.append("empty tokens")
        elif key == "tokens":
            lo, hi = int(arr.min()), int(arr.max())
            lines.append(f"{head}, ids {lo}..{hi} of vocab {vocab_size}")
            if hi >= vocab_size:
                problems.append(f"token id {hi} >= vocab {vocab_size}")
            if lo < 0:
                problems.append(f"negative token id {lo}")
        elif key == "token_lens":
            lines.append(f"{key}: {int(arr.min())}..{int(arr.max())}")
        elif key == "waveform_lens":
            sec = arr.astype(float) / sample_rate
            lines.append(f"{key}: {int(arr.min())}..{int(arr.max())} samples, "
                         f"seconds: {sec.min():.2f}..{sec.max():.2f}")
        elif np.issubdtype(arr.dtype, np.floating):
            n_nan, n_inf = int(np.isnan(arr).sum()), int(np.isinf(arr).sum())
            lines.append(f"{head}, nan {n_nan}, inf {n_inf}")
            if n_nan + n_inf:
                problems.append(f"{key} has {n_nan + n_inf} NaN/inf")
        else:
            lines.append(head)
    return lines, problems


def _loader_kwargs(signature: inspect.Signature, samples: list[Any]) -> dict[str, Any]:
    """The train-loader keyword arguments this get_data_loader signature accepts."""
    wanted = {"is_eval": False, "samples": samples, "verbose": True, "assets": {},
              "num_gpus": 0}
    return {k: wanted[k] for k in _LOADER_KWARGS if k in signature.parameters}


def _vocab_size(model: Any) -> int:
    """Rows of the text embedding: a token id must be below this."""
    return int(model.text_encoder.emb.num_embeddings)


def check_loader(config_path: Path | None = None) -> bool:
    """Load one train batch the way coqui will train on it; print stats and PASS/FAIL."""
    from TTS.tts.configs.vits_config import VitsConfig  # coqui only on Colab
    from TTS.tts.datasets import load_tts_samples
    from TTS.tts.models.vits import Vits

    path = config_path or paths.REPO / train_config.CONFIG_REL
    config = VitsConfig()
    config.load_json(str(path))
    # coqui 0.28: Vits has no init_from_config; the constructor builds the tokenizer.
    model = Vits(config)
    signature = inspect.signature(model.get_data_loader)
    print(f"get_data_loader{signature}")
    train, evals = load_tts_samples(config, eval_split=True)  # 0.28 takes the whole config
    print(f"samples: {len(train)} train / {len(evals)} eval")
    loader = model.get_data_loader(config, **_loader_kwargs(signature, train))
    start = time.perf_counter()
    batch = next(iter(loader))
    try:
        import torch

        no_grad: contextlib.AbstractContextManager[Any] = torch.no_grad()
    except ImportError:  # only in tests with fake coqui; torch always comes with TTS
        no_grad = contextlib.nullcontext()
    with no_grad:
        batch = model.format_batch_on_device(model.format_batch(batch))  # adds spec and mel
    print(f"first batch (incl. worker startup + spec/mel, not a training speed) in "
          f"{time.perf_counter() - start:.2f} s")
    lines, problems = judge_batch(batch, _vocab_size(model), int(config.audio.sample_rate))
    print("\n".join(lines))
    if problems:
        print(f"FAIL: loader {'; '.join(problems)}")
        return False
    print("PASS: loader ok")
    return True


class StopTraining(Exception):
    """Raised by the step callback once max_steps optimiser steps are done."""


def _coqui() -> types.SimpleNamespace:
    """The coqui classes run() needs (imported here: coqui is only on Colab)."""
    from trainer import Trainer, TrainerArgs
    from TTS.tts.configs.vits_config import VitsConfig
    from TTS.tts.datasets import load_tts_samples
    from TTS.tts.models.vits import Vits

    return types.SimpleNamespace(VitsConfig=VitsConfig, Vits=Vits, Trainer=Trainer,
                                 TrainerArgs=TrainerArgs, load_tts_samples=load_tts_samples)


def _peak_memory_gb() -> float | None:
    """torch.cuda.max_memory_allocated in GB, or None without a GPU."""
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    return round(torch.cuda.max_memory_allocated() / 1e9, 3)


def _utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _load_record(run_dir: Path, run_name: str) -> dict[str, Any]:
    """The run's boinga_run.json, or a new one; a corrupt file is refused, not overwritten."""
    path = run_dir / RUN_RECORD
    if not path.is_file():
        return {"run": run_name, "sessions": []}
    record = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(record.get("sessions"), list):
        raise TypeError(f"{path} has no sessions list; move it away to start over")
    return record


def _save_record(run_dir: Path, record: dict[str, Any]) -> None:
    path = run_dir / RUN_RECORD
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


_CKPT = re.compile(r"^checkpoint_(\d+)\.pth$")


def _ckpt_step(path: Path) -> int | None:
    """The step in a coqui checkpoint_<step>.pth name (None for best_model_* and others)."""
    m = _CKPT.match(path.name)
    return int(m.group(1)) if m else None


def latest_run_folder(run_name: str) -> tuple[Path, int] | None:
    """(coqui run folder, step) holding the highest-step checkpoint_*.pth, or None.

    Chosen by step, not mtime: syncs and copies change mtimes. Folders without a
    checkpoint (a failed attempt) are ignored. A step tie (a copied folder) goes to the larger
    folder name; either folder holds the same step.
    """
    run_dir = paths.run_dir(run_name)
    best: tuple[int, str, Path] | None = None
    for ckpt in run_dir.glob("*/checkpoint_*.pth"):
        step = _ckpt_step(ckpt)
        if step is not None and (best is None or (step, ckpt.parent.name) > best[:2]):
            best = (step, ckpt.parent.name, ckpt.parent)
    return (best[2], best[0]) if best else None


def _stopper(max_steps: int) -> Callable[[Any], None]:
    """on_train_step_end callback: coqui increments total_steps_done just before calling it."""
    def stop(trainer: Any) -> None:
        if trainer.total_steps_done >= max_steps:
            raise StopTraining(trainer.total_steps_done)
    return stop


def run(run_name: str, max_steps: int | None = None) -> dict[str, Any]:
    """Train run_name until max_steps total steps: resume its latest checkpoint, else start new.

    max_steps None falls back to train_config.RUN_BUDGETS[run_name] when that is set.

    A resume passes continue_path (coqui reloads the folder's config.json, optimizer, scaler
    and step) and never restore_path; a new run restores the pretrained weights only.
    coqui's Trainer.fit() hard-exits the process (os._exit) on an interrupt and deletes the
    run folder on any other error, so the epoch loop _fit() is called instead and a stop
    raised by the step callback is caught here, followed by a forced checkpoint save.
    """
    if max_steps is None and run_name in train_config.RUN_BUDGETS:
        max_steps = train_config.RUN_BUDGETS[run_name]
        print(f"budget: {max_steps} steps (RUN_BUDGETS)")
    if max_steps is not None and max_steps < 1:
        raise ValueError(f"max_steps must be >= 1, got {max_steps}")
    if re.search(r"checkpoint_\d", run_name):  # coqui matches this anywhere in a file's path
        raise ValueError(f"run name {run_name!r} would confuse coqui's checkpoint search")
    run_dir = paths.run_dir(run_name)
    latest = latest_run_folder(run_name)
    if latest is None or max_steps is None or latest[1] < max_steps:
        existing = len(list(latest[0].glob("checkpoint_*.pth"))) if latest else 0
        _check_space(run_dir, latest[1] if latest else 0, max_steps, existing)
    if latest is not None:
        folder, step = latest
        if max_steps is not None and step >= max_steps:
            print(f"PASS: run {run_name} already at step {step}")
            # nothing trained, so boinga_run.json gets no session on purpose
            return {"kind": "done", "steps_done": step, "output_path": str(folder)}
        config_path = folder / "config.json"  # the run's own config, never rebuilt
        if not config_path.is_file():
            raise FileNotFoundError(f"cannot resume {folder}: no config.json")
        print(f"RESUME from step {step} in {folder}")
        start: dict[str, Any] = {"kind": "resume", "continue_path": str(folder),
                                 "resumed_from_step": step}
        trainer_args: dict[str, str] = {"continue_path": str(folder)}
    else:
        restore = train_config.base_dir() / pretrained.MODEL_FILE
        if not restore.is_file():
            raise FileNotFoundError(f"pretrained {restore} is missing: boinga fetch-pretrained")
        train_config.build(run_name)
        config_path = paths.REPO / train_config.CONFIG_REL
        start = {"kind": "new", "restore_path": str(restore)}
        trainer_args = {"restore_path": str(restore)}
    coqui = _coqui()
    config = coqui.VitsConfig()
    config.load_json(str(config_path))
    env = env_report.collect()
    session: dict[str, Any] = {
        **start, "git_commit": env.get("git_commit"), "env": env,
        "config_path": str(config_path), "max_steps": max_steps, "start_utc": _utc(),
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    record = _load_record(run_dir, run_name)  # keeps failed earlier attempts (no checkpoint)
    record["sessions"].append(session)
    _save_record(run_dir, record)  # before training: a crash still leaves what was tried
    trainer = None
    try:
        model = coqui.Vits(config)  # coqui 0.28: no Vits.init_from_config (see C7.2)
        train, evals = coqui.load_tts_samples(config, eval_split=True)
        trainer = coqui.Trainer(
            coqui.TrainerArgs(**trainer_args), config, output_path=str(run_dir),
            model=model, train_samples=train, eval_samples=evals,
            parse_command_line_args=False,  # never read `boinga train` flags as overrides
            callbacks={"on_train_step_end": _stopper(max_steps)} if max_steps else {})
        _no_best_models(trainer)
        session["output_path"] = str(trainer.output_path)
        try:
            trainer._fit()
            session["stopped_early"] = False
        except StopTraining:
            print(f"STOP at step {trainer.total_steps_done}")
            trainer.save_checkpoint()
            session["stopped_early"] = True
    except BaseException as exc:
        session["error"] = f"{type(exc).__name__}: {exc}"
        if trainer is not None and isinstance(exc, KeyboardInterrupt):
            trainer.save_checkpoint()  # Colab "stop": keep the work since the last save
            session["saved_on_interrupt"] = True
        raise
    finally:
        session.update(steps_done=getattr(trainer, "total_steps_done", None),
                       end_utc=_utc(), peak_gpu_memory_gb=_peak_memory_gb())
        try:
            _save_record(run_dir, record)
        except OSError as exc:  # never hide the training error behind a disk hiccup
            print(f"WARN: could not update {run_dir / RUN_RECORD}: {exc}")
        if trainer is not None:
            trainer.dashboard_logger.finish()  # flush logs on success and on error
    return session


def free_gb(path: Path) -> float:
    """Free GB on the disk holding path (or its nearest existing parent)."""
    while not path.exists() and path != path.parent:
        path = path.parent
    return shutil.disk_usage(path).free / 1e9


def saves_needed(start_step: int, max_steps: int | None, existing: int = 0) -> int:
    """Checkpoints a session writes: one per save_step crossed, plus the one at the stop.

    Refuses a session after which the run folder would hold more than save_n_checkpoints
    (existing ones, e.g. interrupt saves, included): coqui would then delete the oldest on
    Drive. A run without a step budget is refused for the same reason (epochs 1000).
    """
    if max_steps is None:
        raise ValueError("a run on Drive needs a step budget (--max-steps or RUN_BUDGETS): "
                         "without one coqui would rotate checkpoints, deleting on Drive")
    save_step = int(train_config.OVERRIDES["save_step"][0])
    keep = int(train_config.OVERRIDES["save_n_checkpoints"][0])
    n = max_steps // save_step - start_step // save_step + 1
    if existing + n > keep:  # coqui would delete the oldest: a Drive delete
        raise ValueError(f"{existing} checkpoints + {n} new would exceed save_n_checkpoints "
                         f"{keep}: coqui would delete some on Drive")
    return n


def _check_space(run_dir: Path, start_step: int, max_steps: int | None,
                 existing: int = 0) -> None:
    need = saves_needed(start_step, max_steps, existing) * CKPT_GB + MARGIN_GB
    if (free := free_gb(run_dir)) < need:
        raise RuntimeError(f"only {free:.1f} GB free for {run_dir} (need {need:.0f} GB: "
                           f"~{CKPT_GB:.0f} GB per checkpoint): free Drive space, empty its trash")
    print(f"space: {free:.1f} GB free for {run_dir} (need {need:.0f} GB)")


def _no_best_models(trainer: Any) -> None:
    """Stop coqui saving best_model_<N>.pth: each one replaces and deletes the last on Drive."""
    if not callable(getattr(trainer, "save_best_model", None)):
        raise TypeError("coqui Trainer has no save_best_model: cannot turn best saves off")
    trainer.save_best_model = lambda *args, **kwargs: None
    print("best-model saves: off (checkpoints every save_step are the candidates)")


CLEAN_MAX_STEP = 1000  # clean_run refuses runs past this: those are real training, not smoke


def _steps_seen(run_name: str) -> int:
    """Highest step known for a run: numbered checkpoints and boinga_run.json sessions."""
    latest = latest_run_folder(run_name)
    steps = [latest[1]] if latest else []
    record = paths.run_dir(run_name) / RUN_RECORD
    if record.is_file():
        sessions = json.loads(record.read_text(encoding="utf-8")).get("sessions", [])
        steps += [s["steps_done"] for s in sessions if isinstance(s.get("steps_done"), int)]
    return max(steps, default=0)


def clean_run(run_name: str) -> bool:
    """Delete a smoke run's folder for a fresh start; only names "smoke*", at most step 1000."""
    if not run_name.startswith("smoke"):
        raise ValueError(f"clean-run only deletes smoke runs (name 'smoke...'), not {run_name!r}")
    run_dir = paths.run_dir(run_name)
    if not run_dir.exists():
        print(f"PASS: nothing to clean for run {run_name} ({run_dir} absent)")
        return False
    if (steps := _steps_seen(run_name)) > CLEAN_MAX_STEP:
        raise RuntimeError(f"{run_dir} reached step {steps} (> {CLEAN_MAX_STEP}): not a smoke "
                           "run; delete it by hand if you really mean to")
    size = sum(f.stat().st_size for f in run_dir.rglob("*") if f.is_file())
    shutil.rmtree(run_dir)
    if run_dir.exists():  # a FUSE mount can leave files behind without raising
        raise RuntimeError(f"{run_dir} still exists after deleting it: delete it by hand")
    print(f"PASS: cleaned run {run_name}: removed {run_dir} ({size / 1e9:.2f} GB; Drive keeps "
          "deleted files in its trash until emptied)")
    return True


PROVENANCE = "provenance.json"  # in paths.best_ckpt(), written last
PROVENANCE_REL = "reports/best_provenance.json"
PROMOTE_MARGIN_GB = 0.5


def _copy_md5(src: Path, dst: Path) -> str:
    """Copy src to dst through a .part file (tqdm over bytes); return the md5 of what was read."""
    part, digest = dst.with_name(dst.name + ".part"), hashlib.md5()
    with (open(src, "rb") as fin, open(part, "wb") as fout,
          tqdm(total=src.stat().st_size, unit="B", unit_scale=True, desc=f"copy {src.name}") as bar):
        for block in iter(lambda: fin.read(1 << 24), b""):
            fout.write(block)
            digest.update(block)
            bar.update(len(block))
    os.replace(part, dst)
    return digest.hexdigest()


def _eval_near(folder: Path, step: int) -> dict[str, Any] | None:
    """The eval block nearest to step in the run's coqui log (as in reports/candidates_<run>.md)."""
    from boinga_translator import ckpt_sampler  # lazy: ckpt_sampler imports this module

    log = folder / ckpt_sampler.LOG_FILE
    evals = ckpt_sampler.parse_eval(log.read_text(encoding="utf-8")) if log.is_file() else []
    near = min(evals, key=lambda e: abs(e["step"] - step), default=None)
    return {k: round(v, 4) if isinstance(v, float) else v for k, v in near.items()} if near else None


def _provenance(run_name: str, step: int, src: Path, digest: str) -> dict[str, Any]:
    sel = json.loads((paths.REPO / speaker_select.SELECTION_REL).read_text(encoding="utf-8"))
    voice = train_config.base_voice()
    return {
        "run": run_name, "step": step, "run_folder": src.parent.name, "source": src.name,
        "bytes": src.stat().st_size, "md5": digest, "eval": _eval_near(src.parent, step),
        "base_model": {"voice": voice, "id": settings.PRETRAINED_IDS[voice],
                       "license": settings.PRETRAINED_LICENSE,
                       "coqui_tts": settings.COQUI_TTS_VERSION},
        "dataset": {"name": settings.DATASET_NAME, "doi": settings.DATASET_DOI,
                    "license": settings.DATASET_LICENSE},
        "speaker_selection": {k: sel.get(k) for k in ("kind", "id", "csv_sha256")},
        "char_map": text_frontend.map_digest(),
        "seed": settings.SEED, "inference": dict(synth_basic.INFERENCE_SCALES),
        "git_commit": env_report._git_commit(), "promoted_utc": _utc(),
    }


def _old_provenance(best: Path) -> dict[str, Any] | None:
    path = best / PROVENANCE
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def promote(run_name: str, step: int) -> dict[str, Any]:
    """Copy checkpoint_<step>.pth of the run's latest folder + its config.json into
    paths.best_ckpt() as model.pth / config.json, with provenance.json written last.

    Says what it replaces; the same run folder + step + size already there is not copied again.
    Through the Drive mount a replaced model.pth goes to Drive's trash (~1 GB).
    """
    latest = latest_run_folder(run_name)
    if latest is None:
        raise FileNotFoundError(f"no checkpoint under {paths.run_dir(run_name)}: train first")
    folder = latest[0]
    src = folder / f"checkpoint_{step}.pth"
    if not src.is_file():
        have = sorted(s for p in folder.glob("checkpoint_*.pth") if (s := _ckpt_step(p)))
        raise FileNotFoundError(f"no {src.name} in {folder} (have {', '.join(map(str, have))})")
    best, size = paths.best_ckpt(), src.stat().st_size
    model = best / pretrained.MODEL_FILE
    print(f"source: {src} ({size / 1e6:.1f} MB)")
    old = _old_provenance(best)
    if (old and old.get("run_folder") == folder.name and old.get("step") == step
            and model.is_file() and model.stat().st_size == size
            and (best / pretrained.CONFIG_FILE).is_file()):
        print(f"{best} already holds run {run_name} step {step}: not copied again")
        prov = old
    else:
        trash = "; the old model.pth goes to Drive's trash (empty it to free the space)"
        if old:
            print(f"replaces: run {old.get('run')} step {old.get('step')} "
                  f"({old.get('run_folder')}){trash if model.is_file() else ''}")
        elif model.is_file():
            print(f"replaces: a model.pth without provenance "
                  f"({model.stat().st_size / 1e6:.1f} MB){trash}")
        else:
            print(f"replaces: nothing ({best} holds no model)")
        if (free := free_gb(best)) < size / 1e9 + PROMOTE_MARGIN_GB:
            raise RuntimeError(f"only {free:.1f} GB free for {best} (need "
                               f"{size / 1e9 + PROMOTE_MARGIN_GB:.1f} GB): free Drive space")
        best.mkdir(parents=True, exist_ok=True)
        (best / PROVENANCE).unlink(missing_ok=True)  # a half-done promote must not look done
        digest = _copy_md5(src, model)
        shutil.copy2(folder / pretrained.CONFIG_FILE, best / pretrained.CONFIG_FILE)
        prov = _provenance(run_name, step, src, digest)
        tmp = best / f"{PROVENANCE}.part"
        tmp.write_text(json.dumps(prov, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        os.replace(tmp, best / PROVENANCE)
    text = json.dumps(prov, indent=2, ensure_ascii=False) + "\n"
    repo_path, _ = report_io.write_report(text, PROVENANCE_REL)
    report_io.emit(repo_path, PROVENANCE_REL)
    print(f"PASS: promoted {run_name} step {step}")
    return prov
