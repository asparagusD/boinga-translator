"""Save BEGIN/END report blocks pasted from Colab into the repo (from a file or stdin)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from boinga_translator import paths, report_io


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("file", nargs="?", help="file with the pasted Colab output (default: stdin)")
    args = ap.parse_args(argv)
    if args.file:
        text = Path(args.file).read_text(encoding="utf-8-sig")
    else:
        text = sys.stdin.buffer.read().decode("utf-8-sig")
    try:
        saved = report_io.save_blocks(report_io.parse_blocks(text), paths.REPO)
    except ValueError as err:
        print(f"error: {err}", file=sys.stderr)
        return 1
    if not saved:
        print("error: no '=== BEGIN <path> ===' blocks found", file=sys.stderr)
        return 1
    base = paths.REPO.resolve()
    for path in saved:
        print(f"saved {path.relative_to(base).as_posix()} ({path.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
