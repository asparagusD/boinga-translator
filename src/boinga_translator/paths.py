"""Single source of truth for paths: Drive + fast disk on Colab, ./local elsewhere."""

# Use `paths.ROOT` / `paths.root()`, never `from boinga_translator.paths import ROOT`:
# a from-import freezes the value and ignores later BOINGA_ROOT / BOINGA_FAST overrides.
# is_colab() is also True on any machine with /content; set BOINGA_ROOT there.

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    ROOT: Path
    FAST: Path

REPO = Path(__file__).resolve().parents[2]
COLAB_CONTENT = Path("/content")
COLAB_DRIVE = COLAB_CONTENT / "drive" / "MyDrive"
COLAB_ROOT = COLAB_DRIVE / "boinga-translator"
COLAB_FAST = COLAB_CONTENT / "boinga-fast"


def is_colab() -> bool:
    """True on Google Colab: `google.colab` is importable or /content exists."""
    try:
        if importlib.util.find_spec("google.colab") is not None:
            return True
    except (ImportError, ValueError):  # parent package `google` missing
        pass
    return COLAB_CONTENT.is_dir()


def _env_path(name: str) -> Path | None:
    value = os.environ.get(name, "").strip()
    return Path(value).expanduser() if value else None


def root() -> Path:
    """Persistent root: $BOINGA_ROOT, else Drive on Colab, else <repo>/local."""
    return _env_path("BOINGA_ROOT") or (COLAB_ROOT if is_colab() else REPO / "local")


def fast() -> Path:
    """Fast disk for training reads: $BOINGA_FAST, else /content/boinga-fast, else ROOT/fast."""
    return _env_path("BOINGA_FAST") or (COLAB_FAST if is_colab() else root() / "fast")


def __getattr__(name: str) -> Path:
    """Evaluate ROOT and FAST on every access so env overrides always apply."""
    if name == "ROOT":
        return root()
    if name == "FAST":
        return fast()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def data_raw() -> Path:
    return root() / "data" / "raw"


def data_unzipped() -> Path:
    return root() / "data" / "unzipped"


def data_fast() -> Path:
    return fast() / "data"


def data_prepared() -> Path:
    return fast() / "prepared"


def prepared_backup() -> Path:
    return root() / "data" / "prepared"


def pretrained() -> Path:
    return root() / "pretrained"


def runs() -> Path:
    return root() / "runs"


def run_dir(name: str) -> Path:
    """Folder of one training run; `name` must be a single path component."""
    if name in ("", ".", "..") or any(sep in name for sep in ("/", "\\")):
        raise ValueError(f"run name must be a simple folder name, got {name!r}")
    return runs() / name


def speaker_emb() -> Path:
    """Per-clip speaker embeddings (<clip_id>.npy) on Drive, made by speaker_select.embed."""
    return root() / "speaker_emb"


def speaker_clusters() -> Path:
    """Cluster label per clip (speaker_select.cluster), ranked 1..K by clean minutes."""
    return root() / "speaker_clusters.csv"


def best_ckpt() -> Path:
    return root() / "best"


def eval_dir() -> Path:
    return root() / "eval"


def drive_reports() -> Path:
    return root() / "reports"


def reports_dir() -> Path:
    return REPO / "reports"


def configs_dir() -> Path:
    return REPO / "configs"


def all_dirs() -> list[Path]:
    """Every directory ensure_dirs() creates."""
    return [
        data_raw(), data_unzipped(), data_fast(), data_prepared(), prepared_backup(),
        pretrained(), runs(), best_ckpt(), eval_dir(), drive_reports(),
        reports_dir(), configs_dir(),
    ]


def ensure_dirs() -> list[Path]:
    """mkdir -p every project directory; refuse on Colab if Drive is not mounted yet."""
    if is_colab() and root().is_relative_to(COLAB_DRIVE.parent) and not COLAB_DRIVE.is_dir():
        raise RuntimeError(
            "Google Drive is not mounted: run drive.mount('/content/drive') first, "
            "otherwise files would land on the VM disk and be lost"
        )
    dirs = all_dirs()
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    return dirs
