"""Move Colab reports into the repo: print files between BEGIN/END markers, parse pasted output."""

# Round-trip limits: leading/trailing blank lines of a file are dropped, CRLF comes back as LF,
# and a file containing its own "=== END <path> ===" line is cut short. Pasted blocks may only
# write under ALLOWED_TOP (never .git/, .claude/ or code), since a paste is untrusted input.

from __future__ import annotations

import hashlib
import re
from pathlib import Path, PurePosixPath

from boinga_translator import paths

MAX_EMIT_BYTES = 200 * 1024
ALLOWED_TOP = ("reports", "configs")
_BEGIN = re.compile(r"^=== BEGIN (\S.*?) ===$")
_END = re.compile(r"^=== END (\S.*?) ===$")
_WIN_RESERVED = re.compile(r"^(CON|PRN|AUX|NUL|COM\d|LPT\d)(\..*)?$", re.IGNORECASE)


def check_relative(repo_relative: str) -> PurePosixPath:
    """Validate a report path: relative POSIX path to a file under reports/ or configs/."""
    rel = PurePosixPath(repo_relative)
    parts = rel.parts
    if (
        not repo_relative
        or "\\" in repo_relative
        or ":" in repo_relative  # drive letters and NTFS alternate data streams
        or rel.is_absolute()
        or ".." in parts
        or len(parts) < 2
        or parts[0] not in ALLOWED_TOP
        or any(_WIN_RESERVED.match(p) or p.endswith((".", " ")) for p in parts)
    ):
        allowed = ", ".join(f"{top}/" for top in ALLOWED_TOP)
        raise ValueError(f"unsafe report path {repo_relative!r}: must be a file under {allowed}")
    return rel


def _target(root: Path, repo_relative: str) -> Path:
    base = root.resolve()
    target = (base / check_relative(repo_relative)).resolve()
    if not target.is_relative_to(base):  # e.g. a symlink pointing outside
        raise ValueError(f"unsafe report path {repo_relative!r}: resolves outside {base}")
    return target


def emit(path: Path | str, repo_relative: str) -> None:
    """Print a file between BEGIN/END markers, or a SKIPPED line with its sha256 if > 200 KB."""
    check_relative(repo_relative)
    data = Path(path).read_bytes()
    sha = hashlib.sha256(data).hexdigest()
    if len(data) > MAX_EMIT_BYTES:
        print(f"SKIPPED (too big): {repo_relative}, sha256={sha}")
        return
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        print(f"SKIPPED (not utf-8): {repo_relative}, sha256={sha}")
        return
    print(f"=== BEGIN {repo_relative} ===")
    print(text, end="" if text.endswith("\n") else "\n")
    print(f"=== END {repo_relative} ===")


def write_report(text: str, repo_relative: str) -> tuple[Path, Path]:
    """Write text to REPO/<rel> and a copy to drive_reports()/<rel>; return both paths.

    The Drive copy keeps the full repo-relative path, e.g. ROOT/reports/reports/env.json.
    """
    targets = (_target(paths.REPO, repo_relative), _target(paths.drive_reports(), repo_relative))
    paths.ensure_dirs()  # refuses on Colab if Drive is not mounted
    for target in targets:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8", newline="\n")
    return targets


def parse_blocks(pasted_text: str) -> dict[str, str]:
    """Map repo_relative -> content for every BEGIN/END block in pasted Colab output.

    Marker lines may be indented; blank lines right after BEGIN and right before END are
    dropped; CRLF becomes LF; content always ends with one newline (empty stays empty).
    """
    blocks: dict[str, str] = {}
    current: str | None = None
    lines: list[str] = []
    for raw in pasted_text.replace("\r\n", "\n").split("\n"):
        marker = raw.strip()
        begin = _BEGIN.match(marker)
        if current is None:
            if begin:
                current, lines = begin.group(1), []
            continue
        end = _END.match(marker)
        if begin:
            raise ValueError(f"BEGIN {begin.group(1)!r} inside unterminated block {current!r}")
        if not end:
            lines.append(raw)
            continue
        if end.group(1) != current:
            raise ValueError(f"END {end.group(1)!r} does not match BEGIN {current!r}")
        while lines and not lines[0].strip():
            lines.pop(0)
        while lines and not lines[-1].strip():
            lines.pop()
        if current in blocks:
            raise ValueError(f"block {current!r} appears twice in the paste")
        blocks[current] = "\n".join(lines) + "\n" if lines else ""
        current = None
    if current is not None:
        raise ValueError(f"block {current!r} has no END marker (truncated paste?)")
    return blocks


def save_blocks(blocks: dict[str, str], root: Path | None = None) -> list[Path]:
    """Write each block under root (default REPO); validates every path before writing any."""
    base = paths.REPO if root is None else root
    targets = [(_target(base, rel), content) for rel, content in blocks.items()]
    for target, content in targets:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8", newline="\n")
    return [t for t, _ in targets]
