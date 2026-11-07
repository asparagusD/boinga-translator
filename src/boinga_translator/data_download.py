"""Corpus download: stream the pinned zip to Drive, check size + sha256, unzip once, copy to fast."""

# Each step leaves a marker holding the pinned sha256, so a repeated "Run all" skips finished
# steps. Drive keeps the zip and the unzipped tree; the fast-disk copy is redone on a fresh VM.

from __future__ import annotations

import hashlib
import os
import shutil
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import IO

from tqdm import tqdm

from boinga_translator import __version__, paths, settings
from boinga_translator.pretrained import safe_extract

# Mendeley's Cloudflare front answers 403 to urllib's default "Python-urllib/3.x" agent.
USER_AGENT = f"boinga-translator/{__version__} (research dataset download)"
ZIP_NAME = "Dataset.zip"
SHA_MARKER = ZIP_NAME + ".sha256_ok"
UNZIP_MARKER = ".unzipped_ok"
COPY_MARKER = ".copied_ok"
_CHUNK = 1 << 20


def _open(req: urllib.request.Request) -> IO[bytes]:
    return urllib.request.urlopen(req, timeout=120)


def _marked(marker: Path) -> bool:
    return marker.is_file() and marker.read_text(encoding="utf-8").strip() == settings.DATASET_SHA256


def _mark(marker: Path) -> None:
    marker.write_text(settings.DATASET_SHA256 + "\n", encoding="utf-8")


def _step(name: str, done: bool, run: Callable[[], None]) -> None:
    if done:
        print(f"SKIP {name} (done)")
        return
    run()
    print(f"DONE {name}")


def _download(dest: Path) -> None:
    """Stream the zip to dest via dest.part, resuming a partial file with an HTTP Range header."""
    part = dest.with_name(dest.name + ".part")
    total = settings.DATASET_ZIP_BYTES
    have = part.stat().st_size if part.is_file() else 0
    if have > total:
        print(f"WARN: {part.name} is larger than expected ({have} > {total}); restarting")
        have = 0
    if have < total:
        headers = {"User-Agent": USER_AGENT, **({"Range": f"bytes={have}-"} if have else {})}
        try:
            resp = _open(urllib.request.Request(settings.DATASET_ZIP_URL, headers=headers))
        except urllib.error.HTTPError as exc:
            if exc.code != 416:
                raise
            part.unlink()  # the server's file is shorter than our partial: start over next run
            raise RuntimeError(f"server refused bytes={have}- (416); deleted {part.name}, "
                               "re-run to download from 0") from exc
        with resp:
            if have and getattr(resp, "status", 200) != 206:
                print(f"WARN: server ignored the Range header; restarting from 0 (had {have} B)")
                have = 0
            with part.open("ab" if have else "wb") as out, tqdm(
                    total=total, initial=have, unit="B", unit_scale=True, mininterval=10,
                    desc=dest.name) as bar:
                while chunk := resp.read(_CHUNK):
                    out.write(chunk)
                    bar.update(len(chunk))
    got = part.stat().st_size
    if got < total:  # stream ended early: keep the partial file so a re-run resumes it
        raise RuntimeError(f"download stopped at {got} of {total} B; re-run to resume")
    os.replace(part, dest)


def _verify(zip_path: Path, marker: Path) -> None:
    """Check size and sha256 against settings; delete a bad zip so the next run re-downloads."""
    size = zip_path.stat().st_size
    if size != settings.DATASET_ZIP_BYTES:
        zip_path.unlink()
        raise RuntimeError(f"{ZIP_NAME} size mismatch: expected {settings.DATASET_ZIP_BYTES} B, "
                           f"got {size} B (deleted; re-run to download again)")
    digest = hashlib.sha256()
    with zip_path.open("rb") as f:
        while chunk := f.read(_CHUNK):
            digest.update(chunk)
    if digest.hexdigest() != settings.DATASET_SHA256:
        zip_path.unlink()
        raise RuntimeError(f"{ZIP_NAME} sha256 mismatch: expected {settings.DATASET_SHA256}, "
                           f"got {digest.hexdigest()} (deleted; re-run to download again)")
    _mark(marker)


def _fresh_dir(path: Path) -> None:
    """Empty a folder this module owns (left half-written by an interrupted step)."""
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)


def _unzip(zip_path: Path, out: Path) -> None:
    _fresh_dir(out)
    safe_extract(zip_path, out)
    _mark(out / UNZIP_MARKER)


def _copy(src: Path, dest: Path) -> None:
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(src, dest, ignore=shutil.ignore_patterns(UNZIP_MARKER))
    _mark(dest / COPY_MARKER)


def _count_files(root: Path) -> int:
    return sum(1 for p in root.rglob("*") if p.is_file() and p.name != COPY_MARKER)


def fetch() -> Path:
    """Make the corpus ready on the fast disk; return paths.data_fast()."""
    paths.ensure_dirs()  # refuses on Colab before Drive is mounted
    zip_path = paths.data_raw() / ZIP_NAME
    sha_marker = paths.data_raw() / SHA_MARKER
    unzipped, fast = paths.data_unzipped(), paths.data_fast()
    # A step redone invalidates every later step (done = all earlier steps skipped + own marker).
    done = zip_path.is_file()
    _step("download", done, lambda: _download(zip_path))
    done = done and _marked(sha_marker)
    _step("verify", done, lambda: _verify(zip_path, sha_marker))
    done = done and _marked(unzipped / UNZIP_MARKER)
    _step("unzip", done, lambda: _unzip(zip_path, unzipped))
    done = done and _marked(fast / COPY_MARKER)
    _step("copy", done, lambda: _copy(unzipped, fast))
    print(f"PASS: dataset ready at {fast}, {_count_files(fast)} files")
    return fast
