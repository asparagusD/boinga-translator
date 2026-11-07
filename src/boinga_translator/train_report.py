"""Training reports: step time, losses and memory from a run's coqui log; listen to a checkpoint."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

from boinga_translator import (
    paths,
    pretrained,
    report_io,
    synth_basic,
    train_config,
    train_runner,
)

THROUGHPUT_REL = "reports/throughput.json"
LOG_FILE = "trainer_0_log.txt"  # coqui-tts-trainer 0.4: <run folder>/trainer_<rank>_log.txt

_ANSI = re.compile(r"\x1b\[[0-9;]*m")
_STEP = re.compile(r"-->\s*TIME:.*GLOBAL_STEP:\s*(\d+)")
# Keys may hold "-": coqui renames a second optimizer's duplicate key, e.g. "loss-1".
_VALUE = re.compile(r"^\s*\|\s*>\s*([\w-]+):\s*(\S+)(?:\s+\((\S+)\))?")


def _float(text: str | None) -> float | None:
    try:
        return float(text) if text is not None else None
    except ValueError:  # e.g. "tensor(12.3," for grad norms
        return None


def parse_log(text: str) -> list[dict[str, Any]]:
    """Train-step blocks of a coqui log: [{global_step, values{key: v}, avgs{key: avg}}]."""
    entries: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for line in _ANSI.sub("", text).splitlines():
        if "-->" in line:  # a train step header, or eval/epoch blocks that end the step block
            m = _STEP.search(line)
            current = {"global_step": int(m.group(1)), "values": {}, "avgs": {}} if m else None
            if current is not None:
                entries.append(current)
            continue
        m = _VALUE.match(line)
        if current is None or m is None:
            continue
        key, value, avg = m.group(1), _float(m.group(2)), _float(m.group(3))
        if value is not None:
            current["values"][key] = value
        if avg is not None:
            current["avgs"][key] = avg
    return entries


def summarise(entries: list[dict[str, Any]]) -> dict[str, Any]:
    """Mean step time (step 0 = warm-up left out), losses of the last step, NaN anywhere."""
    if not entries:
        raise ValueError("no train step in the log (print_step not reached?)")
    timed = [e for e in entries if "step_time" in e["values"]]
    warm = [e for e in timed if e["global_step"] > 0]
    times = [e["values"]["step_time"] for e in warm or timed]
    mean = sum(times) / len(times) if times else None
    last = entries[-1]
    losses = [v for e in entries for d in (e["values"], e["avgs"])
              for k, v in d.items() if k.startswith("loss")]
    return {
        "sec_per_step_mean": round(mean, 4) if mean is not None else None,
        "n_step_samples": len(times),
        "warmup_only": bool(timed) and not warm,  # only step 0 logged: mean is the cold step
        "steps_per_hour": round(3600 / mean, 1) if mean else None,
        "sec_per_step_coqui_avg": last["avgs"].get("step_time"),
        "any_nan_loss": any(not math.isfinite(v) for v in losses),
        "last_losses": {k: v for k, v in last["values"].items() if k.startswith("loss")},
        "last_logged_step": last["global_step"],
    }


def _latest(run_name: str) -> tuple[Path, int]:
    latest = train_runner.latest_run_folder(run_name)
    if latest is None:
        raise FileNotFoundError(f"no checkpoint under {paths.run_dir(run_name)}: train first")
    return latest


def throughput(run_name: str) -> bool:
    """reports/throughput.json for the run's latest folder (emitted); False on a NaN loss."""
    folder, step = _latest(run_name)
    summary = summarise(parse_log((folder / LOG_FILE).read_text(encoding="utf-8")))
    config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    record_path = paths.run_dir(run_name) / train_runner.RUN_RECORD
    sessions = (json.loads(record_path.read_text(encoding="utf-8"))["sessions"]
                if record_path.is_file() else [])
    peaks = [s["peak_gpu_memory_gb"] for s in sessions if s.get("peak_gpu_memory_gb")]
    report = {
        "run": run_name, "folder": str(folder), "steps_done": step,
        "gpu": next((s["env"].get("gpu_name") for s in reversed(sessions) if s.get("env")),
                    None),
        "batch_size": config.get("batch_size"), "mixed_precision": config.get("mixed_precision"),
        **summary, "peak_mem_gb": max(peaks) if peaks else None,
    }
    text = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    repo_path, _ = report_io.write_report(text, THROUGHPUT_REL)
    report_io.emit(repo_path, THROUGHPUT_REL)
    if report["any_nan_loss"]:
        print(f"FAIL: NaN loss in {folder / LOG_FILE} (fp16? see vits-finetune: switch to fp32)")
        return False
    if report["warmup_only"]:
        print("WARN: only step 0 (warm-up) was logged: sec_per_step_mean is not a steady rate")
    print(f"PASS: throughput recorded: {report['sec_per_step_mean']} s/step "
          f"({report['n_step_samples']} samples), peak {report['peak_mem_gb']} GB, losses "
          f"from logged step {report['last_logged_step']} of {step}")
    return True


def listen_smoke(run_name: str) -> tuple[list[str], list[str]]:
    """The run's first test sentence by the pretrained base voice and its latest checkpoint."""
    folder, step = _latest(run_name)
    config = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    first = config["test_sentences"][0]
    text = first[0] if isinstance(first, list) else first
    base = train_config.base_dir()
    out = paths.eval_dir() / run_name
    voices = [(base / pretrained.MODEL_FILE, base / pretrained.CONFIG_FILE, "pretrained.wav",
               f"pretrained {train_config.base_voice()}"),
              (folder / f"checkpoint_{step}.pth", folder / pretrained.CONFIG_FILE,
               f"step{step}.wav",
               f"{run_name} step {step}")]
    wavs, labels = [], []
    for model, cfg, name, label in voices:
        synth = synth_basic.load_synth(model, cfg)
        synth_basic.say(synth, text, out / name)
        del synth  # drop the first model before loading the second (two small VITS models)
        wavs.append(str(out / name))
        labels.append(f"{label}: {text}")
    return wavs, labels
