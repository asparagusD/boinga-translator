"""Pretrained Bangla VITS voices: cache model.pth + config.json on Drive and report their facts."""

from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
import unicodedata
import urllib.request
import zipfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from boinga_translator import paths, report_io, settings

# A downloader gets (voice, empty scratch dir) and returns a folder holding the model files.
Downloader = Callable[[str, Path], Path]

MODEL_FILE = "model.pth"
CONFIG_FILE = "config.json"
_MODEL_NAMES = (MODEL_FILE, "model_file.pth")  # preferred when a download holds several *.pth


def voices() -> list[str]:
    """The pretrained voice names ("female", "male")."""
    return list(settings.PRETRAINED_IDS)


def _check_voice(voice: str) -> None:
    if voice not in settings.PRETRAINED_IDS:
        raise ValueError(f"unknown voice {voice!r}; expected one of {voices()}")


def _require_drive(target: Path) -> None:
    """On Colab, refuse to write under /content/drive before Drive is mounted."""
    drive_parent = paths.COLAB_DRIVE.parent
    if paths.is_colab() and target.is_relative_to(drive_parent) and not paths.COLAB_DRIVE.is_dir():
        raise RuntimeError("Google Drive is not mounted: run the bootstrap cells first")


def _download_with_manager(voice: str, workdir: Path) -> Path:
    """Coqui's own model registry download (needs the [train] extra)."""
    from TTS.utils.manage import ModelManager

    manager = ModelManager(output_prefix=str(workdir), progress_bar=False)
    model_path, _config, _item = manager.download_model(settings.PRETRAINED_IDS[voice])
    model = Path(model_path)
    return model if model.is_dir() else model.parent


def safe_extract(zip_path: Path, out_dir: Path) -> None:
    """Unzip, refusing any member that would land outside out_dir."""
    root = out_dir.resolve()
    with zipfile.ZipFile(zip_path) as zf:
        for name in zf.namelist():
            if not (root / name).resolve().is_relative_to(root):
                raise RuntimeError(f"unsafe path in zip {zip_path.name}: {name!r}")
        zf.extractall(root)


def _download_release(voice: str, workdir: Path) -> Path:
    """Fallback: the pinned GitHub release zip."""
    zip_path = workdir / "release.zip"
    url = settings.PRETRAINED_RELEASE_URLS[voice]
    with urllib.request.urlopen(url, timeout=120) as resp, zip_path.open("wb") as out:
        shutil.copyfileobj(resp, out, 1 << 20)
    safe_extract(zip_path, workdir / "release")
    return workdir / "release"


DEFAULT_DOWNLOADERS: tuple[Downloader, ...] = (_download_with_manager, _download_release)


def _find_files(src: Path) -> tuple[Path, Path]:
    """The model checkpoint (model.pth, model_file.pth or the single *.pth) and config.json."""
    pths = sorted(src.rglob("*.pth"))
    named = [p for p in pths if p.name in _MODEL_NAMES]
    pick = named if len(named) == 1 else pths
    configs = sorted(src.rglob(CONFIG_FILE))
    if len(pick) != 1:
        raise RuntimeError(f"expected one *.pth in the download, found {[p.name for p in pths]}")
    if len(configs) != 1:
        raise RuntimeError(f"expected one {CONFIG_FILE} in the download, found {len(configs)}")
    return pick[0], configs[0]


def _install(src: Path, dest: Path) -> None:
    """Copy via a .part file so an interrupted copy never looks cached."""
    part = dest.with_name(dest.name + ".part")
    shutil.copyfile(src, part)
    os.replace(part, dest)


def _cached(target: Path) -> bool:
    return all((target / n).is_file() and (target / n).stat().st_size > 0
               for n in (MODEL_FILE, CONFIG_FILE))


def _mb(path: Path) -> str:
    return f"{path.stat().st_size / 1e6:.2f} MB"


def fetch(voice: str, downloaders: Sequence[Downloader] | None = None) -> dict[str, Path]:
    """Return {"model", "config"} under paths.pretrained()/<voice>, downloading only if missing."""
    _check_voice(voice)
    target = paths.pretrained() / voice
    if not _cached(target):
        _require_drive(target)
        errors: list[str] = []
        for download in downloaders if downloaders is not None else DEFAULT_DOWNLOADERS:
            with tempfile.TemporaryDirectory(prefix=f"boinga-{voice}-") as tmp:
                try:
                    model, config = _find_files(download(voice, Path(tmp)))
                except Exception as exc:  # noqa: BLE001 - coqui/network errors vary; try the next
                    errors.append(f"{getattr(download, '__name__', download)}: {exc}")
                    print(f"WARN: {voice}: {errors[-1]}")
                    continue
                target.mkdir(parents=True, exist_ok=True)
                _install(model, target / MODEL_FILE)
                _install(config, target / CONFIG_FILE)
                break
        else:
            raise RuntimeError(f"could not download {voice}: {'; '.join(errors)}")
    files = {"model": target / MODEL_FILE, "config": target / CONFIG_FILE}
    print(f"{voice}: {MODEL_FILE} {_mb(files['model'])}, {CONFIG_FILE} {_mb(files['config'])}")
    print(f"PASS: {voice} cached ({target})")
    return files


# --- inspect ---------------------------------------------------------------------------------

CHAR_KEYS = ("characters", "punctuations", "pad", "eos", "bos", "blank", "characters_class",
             "phonemes")
TEXT_KEYS = ("use_phonemes", "phonemizer", "phoneme_language", "text_cleaner", "add_blank",
             "enable_eos_bos_chars")
FACTS_REL = "configs/pretrained_facts.json"
INSPECT_REL = "reports/pretrained_inspect.md"


def _absolute_paths(node: object, prefix: str = "") -> dict[str, str]:
    """Config values that are absolute paths (coqui may rewrite them to its download folder)."""
    found: dict[str, str] = {}
    if isinstance(node, Mapping):
        for key, value in node.items():
            if key != "characters":  # symbols, not paths
                found.update(_absolute_paths(value, f"{prefix}{key}."))
        return found
    if isinstance(node, list):
        for i, value in enumerate(node):
            found.update(_absolute_paths(value, f"{prefix}{i}."))
        return found
    if isinstance(node, str) and (node.startswith("/") or node[1:3] in (":\\", ":/")):
        return {prefix.rstrip("."): node}
    return {}


def config_facts(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Audio, characters, text front-end and speaker facts from a parsed config.json."""
    chars = cfg.get("characters") or {}
    args = cfg.get("model_args") or {}
    return {
        "model": cfg.get("model"),
        "audio": dict(cfg.get("audio") or {}),
        "characters": {k: chars.get(k) for k in CHAR_KEYS},
        **{k: cfg.get(k) for k in TEXT_KEYS},
        "num_speakers": args.get("num_speakers"),
        "use_speaker_embedding": args.get("use_speaker_embedding"),
        "absolute_paths": _absolute_paths(cfg),
    }


def checkpoint_facts(ckpt: Mapping[str, Any]) -> dict[str, Any]:
    """Top-level keys, parameter counts and discriminator (`disc.`) keys of a loaded checkpoint."""
    state = ckpt["model"] if isinstance(ckpt.get("model"), Mapping) else ckpt
    sizes = {k: math.prod(v.shape) for k, v in state.items() if hasattr(v, "shape")}
    disc = [n for k, n in sizes.items() if k.startswith("disc.")]
    step = ckpt.get("step", ckpt.get("global_step"))
    return {
        "top_level_keys": sorted(map(str, ckpt)),
        "n_params": sum(sizes.values()),
        "n_disc_keys": len(disc),
        "n_disc_params": sum(disc),
        "has_discriminator": bool(disc),
        "step": None if step is None else int(step),  # numpy/torch scalars -> JSON int
    }


def charset_lines(chars: Mapping[str, Any]) -> str:
    """One line per distinct character/punctuation: char, U+XXXX, Unicode name (tab-separated)."""
    seen = dict.fromkeys((chars.get("characters") or "") + (chars.get("punctuations") or ""))
    rows = []
    for c in seen:
        shown = c if c.isprintable() and not c.isspace() else f"\\u{ord(c):04x}"
        rows.append(f"{shown}\tU+{ord(c):04X}\t{unicodedata.name(c, '(unnamed)')}")
    return "".join(f"{r}\n" for r in rows)


def _load_checkpoint(path: Path) -> Mapping[str, Any]:
    """torch.load on CPU (needs the [train] extra)."""
    import torch

    # weights_only=False: coqui checkpoints hold config objects; the file is the pinned release
    return torch.load(path, map_location="cpu", weights_only=False)


def inspect(voice: str) -> dict[str, Any]:
    """Facts of one cached voice; writes + emits its charset table and prints a summary."""
    _check_voice(voice)
    target = paths.pretrained() / voice
    if not _cached(target):
        raise RuntimeError(f"{voice} is not cached in {target}: run `boinga fetch-pretrained`")
    cfg = config_facts(json.loads((target / CONFIG_FILE).read_text(encoding="utf-8")))
    ckpt = checkpoint_facts(_load_checkpoint(target / MODEL_FILE))
    rel = f"reports/pretrained_charset_{voice}.txt"
    report_io.emit(report_io.write_report(charset_lines(cfg["characters"]), rel)[0], rel)
    print(f"{voice}: sample_rate={cfg['audio'].get('sample_rate')} "
          f"use_phonemes={cfg['use_phonemes']} cleaner={cfg['text_cleaner']} "
          f"add_blank={cfg['add_blank']} params={ckpt['n_params']:,} "
          f"disc_params={ckpt['n_disc_params']:,} step={ckpt['step']}")
    for key, value in cfg["absolute_paths"].items():
        print(f"WARN: {voice}: config {key} is an absolute path: {value}")
    print(f"PASS: inspected {voice}")
    return {"config": cfg, "checkpoint": ckpt}


def _facts_entry(result: Mapping[str, Any]) -> dict[str, Any]:
    cfg, ckpt = result["config"], result["checkpoint"]
    chars = {k: v for k, v in cfg["characters"].items() if k != "phonemes"}
    return {"sample_rate": cfg["audio"].get("sample_rate"), **chars,
            "use_phonemes": cfg["use_phonemes"], "text_cleaner": cfg["text_cleaner"],
            "add_blank": cfg["add_blank"], "has_discriminator": ckpt["has_discriminator"],
            "n_params": ckpt["n_params"], "n_disc_params": ckpt["n_disc_params"]}


def _render_md(results: Mapping[str, Mapping[str, Any]]) -> str:
    lines = ["# Pretrained voices", "", "Generated by `boinga inspect-pretrained`.", ""]
    for voice, result in results.items():
        lines += [f"## {voice}", "", "```json",
                  json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True), "```", ""]
    return "\n".join(lines)


def write_inspect_reports(results: Mapping[str, Mapping[str, Any]]) -> None:
    """Merge the voices into configs/pretrained_facts.json, write the .md; emit each once."""
    facts_path = paths.REPO / FACTS_REL
    try:
        facts = json.loads(facts_path.read_text(encoding="utf-8")) if facts_path.is_file() else {}
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{FACTS_REL} is not valid JSON ({exc}); fix or delete it") from exc
    facts.update({voice: _facts_entry(r) for voice, r in results.items()})
    for text, rel in ((json.dumps(facts, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
                       FACTS_REL), (_render_md(results), INSPECT_REL)):
        report_io.emit(report_io.write_report(text, rel)[0], rel)
