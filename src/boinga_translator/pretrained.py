"""Download the pretrained Bangla VITS voices once and cache model.pth + config.json on Drive."""

from __future__ import annotations

import os
import shutil
import tempfile
import urllib.request
import zipfile
from collections.abc import Callable, Sequence
from pathlib import Path

from boinga_translator import paths, settings

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
