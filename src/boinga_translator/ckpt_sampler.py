"""Checkpoint sampler: val prompts per training checkpoint, candidates ranked by eval loss."""

from __future__ import annotations

import gc
import json
import re
from pathlib import Path
from typing import Any

from boinga_translator import paths, pretrained, report_io, synth_basic, train_config, train_runner

LOG_FILE = "trainer_0_log.txt"  # same file train_report reads
DONE = "done.json"
PRETRAINED_DIR = "step_0_pretrained"
N_BEST = 4

_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_STEP = re.compile(r"GLOBAL_STEP:\s*(\d+)")
_AVG = re.compile(r"^\s*\|\s*>\s*avg_(loss[\w-]*):\s*(\S+)")
_BEST = re.compile(r"best_model_(\d+)\.pth$")


def parse_eval(text: str) -> list[dict[str, Any]]:
    """Eval blocks of a coqui log: step (last logged train step before it), loss_mel, total.

    loss_total = avg_loss_0 + avg_loss_1, the sum coqui ranks best models by when
    target_loss is null (two optimizers: discriminator and generator).
    """
    evals: list[dict[str, Any]] = []
    step, current = 0, None
    for line in _ANSI.sub("", text).splitlines():
        if "-->" in line:
            if m := _STEP.search(line):
                step = int(m.group(1))
            current = {"step": step, "losses": {}} if "EVAL PERFORMANCE" in line else None
            if current is not None:
                evals.append(current)
        elif current is not None and (m := _AVG.match(line)):
            try:
                current["losses"][m.group(1)] = float(m.group(2))
            except ValueError:
                continue
    for e in evals:
        losses = e.pop("losses")
        totals = [v for k, v in losses.items() if re.fullmatch(r"loss_\d+", k)]
        e["loss_mel"] = losses.get("loss_mel")
        e["loss_total"] = sum(totals) if totals else losses.get("loss")
    return evals


def _folder(run_name: str) -> Path:
    latest = train_runner.latest_run_folder(run_name)
    if latest is None:
        raise FileNotFoundError(f"no checkpoint under {paths.run_dir(run_name)}: train first")
    return latest[0]


def checkpoints(run_name: str) -> dict[int, Path]:
    """step -> file in the run's latest folder: checkpoint_<N>, else best_model_<N>.

    best_model.pth is a copy of the newest best_model_<N>.pth, so it is not sampled twice.
    """
    folder = _folder(run_name)
    found: dict[int, Path] = {}
    for path in sorted(folder.glob("best_model_*.pth")):
        if m := _BEST.search(path.name):
            found[int(m.group(1))] = path
    for path in folder.glob("checkpoint_*.pth"):
        if (step := train_runner._ckpt_step(path)) is not None:
            found[step] = path
    return dict(sorted(found.items()))


def _texts() -> list[str]:
    cfg = json.loads((paths.REPO / train_config.CONFIG_REL).read_text(encoding="utf-8"))
    return [t[0] if isinstance(t, list) else t for t in cfg["test_sentences"]]


def _say_all(model: Path, config: Path, out: Path, texts: list[str]) -> list[float]:
    synth = synth_basic.load_synth(model, config)
    durations = [synth_basic.say(synth, t, out / f"{i}.wav") for i, t in enumerate(texts)]
    del synth  # one model in GPU memory at a time
    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except ImportError:
        pass
    (out / DONE).write_text(json.dumps({"file": model.name, "texts": texts,
                                        "durations": durations}, ensure_ascii=False),
                            encoding="utf-8")
    return durations


def _done(out: Path, texts: list[str]) -> bool:
    """Sampled already with these texts (changed test_sentences resample the step)."""
    marker = out / DONE
    return marker.is_file() and json.loads(marker.read_text(encoding="utf-8")).get(
        "texts") == texts


def sample(run_name: str) -> int:
    """Synthesise the test sentences for every unsampled checkpoint (+ the pretrained voice once)."""
    ckpts, texts, folder = checkpoints(run_name), _texts(), _folder(run_name)
    root = paths.eval_dir() / run_name
    base = train_config.base_dir()
    if not _done(root / PRETRAINED_DIR, texts):
        _say_all(base / pretrained.MODEL_FILE, base / pretrained.CONFIG_FILE,
                 root / PRETRAINED_DIR, texts)
    done = 0
    for step, model in ckpts.items():
        out = root / f"step_{step}"
        if _done(out, texts):
            continue
        durations = _say_all(model, model.parent / pretrained.CONFIG_FILE, out, texts)
        print(f"step {step}: mean {sum(durations) / len(durations):.2f} s")
        done += 1
    print(f"PASS: sampled {done} checkpoints ({len(ckpts)} in {folder.name})")
    return done


def _round(value: float | None) -> float | None:
    return round(value, 3) if value is not None else None


def _rows(run_name: str) -> list[dict[str, Any]]:
    ckpts, folder = checkpoints(run_name), _folder(run_name)
    log = folder / LOG_FILE
    evals = parse_eval(log.read_text(encoding="utf-8")) if log.is_file() else []
    best = max((int(m.group(1)) for p in folder.glob("best_model_*.pth")
                if (m := _BEST.search(p.name))), default=None)
    rows = []
    for step, path in ckpts.items():
        near = min(evals, key=lambda e: abs(e["step"] - step), default=None)
        rows.append({"step": step, "file": path.name,
                     "eval_step": near["step"] if near else None,
                     "loss_mel": _round(near["loss_mel"]) if near else None,
                     "loss_total": _round(near["loss_total"]) if near else None,
                     "is_best_model": step == best})
    return sorted(rows, key=lambda r: (r["loss_mel"] is None, r["loss_mel"] or 0.0, r["step"]))


def candidates(run_name: str) -> list[dict[str, Any]]:
    """reports/candidates_<run>.md: checkpoints ranked by the nearest eval loss_mel (emitted)."""
    rows = _rows(run_name)
    lines = [f"# Candidates for run {run_name}", "",
             "Eval losses: nearest eval block (eval_step = last logged train step before it).",
             "",
             "| step | file | eval_step | loss_mel | loss_total | best_model |",
             "|---|---|---|---|---|---|"]
    lines += [f"| {r['step']} | {r['file']} | {r['eval_step']} | {r['loss_mel']} | "
              f"{r['loss_total']} | {'yes' if r['is_best_model'] else 'no'} |" for r in rows]
    rel = f"reports/candidates_{run_name}.md"
    repo_path, _ = report_io.write_report("\n".join(lines) + "\n", rel)
    report_io.emit(repo_path, rel)
    print(f"PASS: candidates listed ({len(rows)} checkpoints)")
    return rows


def players(run_name: str, steps: list[int] | None = None) -> list[tuple[str, str]]:
    """(wav, label) pairs by step for colab_runtime.play: default pretrained, 4 best, last."""
    root, texts = paths.eval_dir() / run_name, _texts()
    if steps is None:
        rows = _rows(run_name)
        ranked = [r["step"] for r in rows if r["loss_mel"] is not None][:N_BEST]
        last = max((r["step"] for r in rows), default=None)
        groups: list[tuple[Path, str]] = [(root / PRETRAINED_DIR,
                                           f"pretrained {train_config.base_voice()}")]
        groups += [(root / f"step_{s}", f"step {s}")
                   for s in dict.fromkeys([*ranked, *([last] if last is not None else [])])]
    else:
        groups = [(root / f"step_{s}", f"step {s}") for s in steps]
    return [(str(d / f"{i}.wav"), f"{label} #{i}: {t}")
            for d, label in groups if (d / DONE).is_file() for i, t in enumerate(texts)]
