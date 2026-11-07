"""Fine-tune config: the base voice's pretrained config.json plus logged, reasoned overrides."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from boinga_translator import paths, prepare, pretrained, report_io, settings, speaker_select

CONFIG_REL = "configs/vits_ctg_finetune.json"
REPORT_REL = "reports/config_overrides.md"
DECISIONS_REL = "docs/DECISIONS.md"
DECISIONS_BEGIN = "<!-- fine-tune-overrides:begin (written by train_config.build) -->"
DECISIONS_END = "<!-- fine-tune-overrides:end -->"
EXTENDED_SUFFIX = "_extended"  # a base voice with an extended text embedding (C7.0, not used)
N_TEST_SENTENCES = 5
LR_FACTOR = 0.5
MAX_AUDIO_S = 15.0
MIN_AUDIO_S = 0.5

# Must stay equal to the pretrained config: changing any of them breaks the restored weights
# or the text the model learned (vits-finetune skill; DECISIONS "Gate G1", "Gate G4").
PROTECTED = ("audio", "characters", "use_phonemes", "text_cleaner", "add_blank", "model_args")

# Static overrides as key: (value, reason); the reasons also fill the DECISIONS table.
OVERRIDES: dict[str, tuple[Any, str]] = {
    "batch_size": (16, "measured best on the RTX PRO 6000: 46.8 clips/s vs 44.1 at batch 32"),
    "eval_batch_size": (8, "half the train batch"),
    "num_loader_workers": (8, "RTX PRO 6000 VM has 48 CPUs (env.json); no speed change seen"),
    "num_eval_loader_workers": (4, "eval is small; half the train workers"),
    "mixed_precision": (True, "fp16: no NaN in the smoke test, less memory; NaN -> fp32"),
    "run_eval": (True, "val loss picks the best checkpoint, with listening (vits-finetune)"),
    "test_delay_epochs": (0, "test sentences from the first epoch: early sanity listening"),
    "print_step": (25, "step logs often enough to read sec/step in the smoke test"),
    "save_step": (500, "a checkpoint every ~500 steps to Drive (vits-finetune)"),
    "save_n_checkpoints": (2, "keep the last 2 (+ best): ~1 GB each, Drive budget"),
    "save_best_after": (1000, "no best checkpoint from the noisy first steps"),
    "save_checkpoints": (True, "resume after a Colab disconnect needs them"),
    # coqui names the training seed "training_seed"; a "seed" key is silently ignored (C7 run 1)
    "training_seed": (settings.SEED, "the project seed (settings.SEED) for reproducible runs"),
    "epochs": (1000, "upper bound only; runs stop by a step budget (RUN_BUDGETS, C8)"),
    "restore_path": (None, "the author's absolute path; the runner passes it (C7.3)"),
}

# Overrides computed from the pretrained config or the run, with their reasons.
COMPUTED_REASONS: dict[str, str] = {
    "lr_gen": f"{LR_FACTOR} x pretrained: fine-tuning should not undo the pretrained weights",
    "lr_disc": f"{LR_FACTOR} x pretrained, like lr_gen, so G and D stay balanced",
    "max_audio_len": f"{MAX_AUDIO_S:g} s x sample_rate: kept from C7; a batch pads to its longest clip",
    "min_audio_len": f"{MIN_AUDIO_S:g} s x sample_rate, on trimmed clips (C6 follow-up)",
    "test_sentences": f"first {N_TEST_SENTENCES} val manifest texts: prepare_text output as "
                      "trained, not re-normalised (Gate G4)",
    "datasets": "our prepared ljspeech manifests on the fast disk replace the author's paths",
    "output_path": "runs on Drive (paths.runs()) survive a Colab disconnect",
    "run_name": "given per run (boinga build-config --run NAME)",
}
REASONS: dict[str, str] = {**{k: r for k, (_, r) in OVERRIDES.items()}, **COMPUTED_REASONS}

BUDGET_HOURS = 3.5  # step time per run (C8.1); eval, checkpoint saves and restarts come on top


def budget_steps(steps_per_hour: float, hours: float = BUDGET_HOURS) -> int:
    """Steps that fit in hours at a measured rate, rounded down to a save_step multiple.

    Uses the global OVERRIDES save_step (no run overrides it yet).
    """
    save_step = int(OVERRIDES["save_step"][0])
    steps = int(steps_per_hour * hours) // save_step * save_step
    if steps < save_step:
        raise ValueError(f"{steps_per_hour} steps/h x {hours} h is under one save_step "
                         f"({save_step}): no checkpoint would be saved")
    return steps


# Run name -> max steps, used by `boinga train` when --max-steps is absent (C8.1).
# run1, from the batch-16 smoke throughput (C7 re-run, RTX PRO 6000, fp16; commit 40a1a25;
# reports/throughput.json now holds the batch-32 timing, C8.1c) and configs/splits_summary.json:
# steps/epoch = 6,415 train clips / 16 = 400.9; 10,533.4 steps/h x 3.5 h = 36,866.9 -> rounded
# down to a multiple of 500 = 36,500 steps (~91 epochs). Equals budget_steps(10533.4).
# Confirmed in C8.1d after batch 32 proved slower per clip.
RUN_BUDGETS: dict[str, int] = {"run1": 36_500}

# Per-run overrides on top of OVERRIDES, as key: (value, reason); keys must be in OVERRIDES.
RUN_OVERRIDES: dict[str, dict[str, tuple[Any, str]]] = {
    "smoke": {"print_step": (5, "smoke run only: ~10 timed steps in 60 instead of 1")},
}


def reasons(run_name: str) -> dict[str, str]:
    """Reason per overridden key for this run (a per-run reason replaces the general one)."""
    return {**REASONS, **{k: r for k, (_, r) in RUN_OVERRIDES.get(run_name, {}).items()}}


def base_voice(selection_path: Path | None = None) -> str:
    """The pretrained voice to fine-tune, from configs/speaker_selection.json."""
    sel = json.loads((selection_path or paths.REPO / speaker_select.SELECTION_REL).read_text(
        encoding="utf-8"))
    voice = sel.get("base_voice")
    if voice not in pretrained.voices():
        raise ValueError(f"speaker_selection.json base_voice must be one of "
                         f"{pretrained.voices()}, got {voice!r}")
    return str(voice)


def base_dir() -> Path:
    """Cached base voice folder: <voice>_extended if C7.0 made one, else <voice>."""
    voice = base_voice()
    extended = paths.pretrained() / f"{voice}{EXTENDED_SUFFIX}"
    return extended if extended.is_dir() else paths.pretrained() / voice


def _scaled_lr(cfg: dict[str, Any], key: str) -> float:
    value = cfg.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float) or value <= 0:
        raise ValueError(f"pretrained config has no positive {key}: {value!r}")
    return float(value) * LR_FACTOR


def val_texts(n: int = N_TEST_SENTENCES) -> list[str]:
    """First n texts of the prepared val manifest (already prepare_text output)."""
    manifest = paths.data_prepared() / prepare.manifest_names()["val"]
    lines = [ln for ln in manifest.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if len(lines) < n:
        raise ValueError(f"{manifest} has {len(lines)} lines; need {n} test sentences")
    cols = [ln.split("|") for ln in lines[:n]]
    if bad := [i for i, c in enumerate(cols, 1) if len(c) != 3 or not c[1].strip()]:
        raise ValueError(f"{manifest}: lines {bad} are not '<id>|<text>|<text>'")
    return [c[1] for c in cols]


def overrides(pretrained_cfg: dict[str, Any], run_name: str,
              texts: list[str]) -> dict[str, Any]:
    """Every top-level key the fine-tune config changes (static + computed)."""
    run_specific = RUN_OVERRIDES.get(run_name, {})
    if unknown := sorted(set(run_specific) - set(OVERRIDES)):
        raise ValueError(f"RUN_OVERRIDES[{run_name!r}] keys {unknown} are not in OVERRIDES")
    sr = int(pretrained_cfg["audio"]["sample_rate"])
    names = prepare.manifest_names()
    return {
        **{k: v for k, (v, _) in OVERRIDES.items()},
        **{k: v for k, (v, _) in run_specific.items()},
        "lr_gen": _scaled_lr(pretrained_cfg, "lr_gen"),
        "lr_disc": _scaled_lr(pretrained_cfg, "lr_disc"),
        "max_audio_len": int(MAX_AUDIO_S * sr),
        "min_audio_len": int(MIN_AUDIO_S * sr),
        "test_sentences": [[t] for t in texts],
        "datasets": [{
            "formatter": prepare.FORMATTER, "dataset_name": "", "path": str(paths.data_prepared()),
            "meta_file_train": names["train"], "meta_file_val": names["val"],
            "ignored_speakers": None, "language": "bn", "phonemizer": "",
            "meta_file_attn_mask": "",
        }],
        "output_path": str(paths.runs()),
        "run_name": run_name,
    }


def author_paths(cfg: dict[str, Any]) -> dict[str, str]:
    """Absolute paths in cfg outside our Drive root and fast disk (e.g. the author's)."""
    ours = (paths.root(), paths.fast())
    return {k: v for k, v in pretrained._absolute_paths(cfg).items()
            if not any(Path(v).is_relative_to(root) for root in ours)}


def _cell(value: Any) -> str:
    text = json.dumps(value, ensure_ascii=False)
    text = text if len(text) <= 60 else text[:57] + "..."
    return text.replace("|", "\\|")


def render_table(pretrained_cfg: dict[str, Any], cfg: dict[str, Any],
                 run_name: str = "") -> str:
    """Markdown table: key | pretrained | fine-tune | reason, for every override."""
    rows = ["| key | pretrained | fine-tune | reason |", "|---|---|---|---|"]
    why = reasons(run_name)
    for key in sorted(why):
        old = _cell(pretrained_cfg[key]) if key in pretrained_cfg else "(missing)"
        rows.append(f"| `{key}` | {old} | {_cell(cfg[key])} | {why[key]} |")
    return "\n".join(rows) + "\n"


def replace_decisions_table(table: str) -> None:
    """Put table between the DECISIONS markers (C7.V: from reports/config_overrides.md)."""
    path = paths.REPO / DECISIONS_REL
    text = path.read_text(encoding="utf-8")
    start, end = text.find(DECISIONS_BEGIN), text.find(DECISIONS_END)
    if start < 0 or end < start:
        raise RuntimeError(f"{DECISIONS_REL} lacks the fine-tune overrides markers "
                           f"({DECISIONS_BEGIN} ... {DECISIONS_END})")
    new = text[:start + len(DECISIONS_BEGIN)] + "\n" + table + text[end:]
    path.write_text(new, encoding="utf-8", newline="\n")


def build(run_name: str) -> dict[str, Any]:
    """Write configs/vits_ctg_finetune.json for run_name; refuse any protected-key change."""
    paths.run_dir(run_name)  # validates the name (one folder component)
    src = base_dir() / pretrained.CONFIG_FILE
    base = json.loads(src.read_text(encoding="utf-8"))
    changes = overrides(base, run_name, val_texts())
    cfg = {**copy.deepcopy(base), **copy.deepcopy(changes)}
    if changed := [k for k in PROTECTED if cfg.get(k) != base.get(k)]:
        raise RuntimeError(f"fine-tune config must keep {changed} equal to the pretrained one")
    if foreign := author_paths(cfg):
        raise RuntimeError(f"fine-tune config still holds foreign absolute paths: {foreign}")
    table = render_table(base, cfg, run_name)
    for text, rel in ((json.dumps(cfg, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
                       CONFIG_REL),
                      (f"# Fine-tune config overrides\n\nBase: `{src}`, run `{run_name}`.\n\n"
                       + table, REPORT_REL)):
        repo_path, _ = report_io.write_report(text, rel)
        report_io.emit(repo_path, rel)
    replace_decisions_table(table)  # last: never describes a config that was not written
    print(f"PASS: config written to {paths.REPO / CONFIG_REL} (base {src}, "
          f"{len(changes)} overrides)")
    return cfg
