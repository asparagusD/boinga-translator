"""Pre-commit guard: block binaries/large files, stale repo graph, ruff errors and failing tests."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BLOCKED_SUFFIXES = (".wav", ".flac", ".mp3", ".pth", ".pt", ".zip", ".ckpt")
MAX_BYTES = 2 * 1024 * 1024
LARGE_COMMIT_LINES = 200
MAP_FILES = ("docs/repo_graph.json", "docs/REPO_GRAPH.md")


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True,
                          encoding="utf-8", check=True).stdout


def _run(cmd: list[str]) -> int:
    return subprocess.run(cmd, cwd=ROOT, check=False).returncode


def _is_docs(path: str) -> bool:
    return path.startswith("docs/") or path.endswith(".md")


def staged_files() -> list[str]:
    out = _git("diff", "--cached", "--name-only", "--diff-filter=ACMR", "-z")
    return [f for f in out.split("\0") if f]


def blocked_files(files: list[str]) -> list[str]:
    reasons = []
    for f in files:
        if f.lower().endswith(BLOCKED_SUFFIXES):
            reasons.append(f"{f}: audio/checkpoint/archive files must not be committed")
        elif int(_git("cat-file", "-s", f":{f}")) > MAX_BYTES:
            reasons.append(f"{f}: staged file is larger than 2 MB")
    return reasons


def added_code_lines() -> int:
    total = 0
    for row in _git("diff", "--cached", "--numstat").splitlines():
        added, _, path = row.split("\t", 2)
        if added != "-" and not path.startswith("tests/") and not _is_docs(path):
            total += int(added)
    return total


def main() -> int:
    files, ok = staged_files(), True
    for reason in blocked_files(files):
        print(f"BLOCKED: {reason}", file=sys.stderr)
        ok = False

    map_unstaged = _git("diff", "--name-only", "--", *MAP_FILES).strip()
    if _run([sys.executable, "tools/repo_graph.py", "--check"]) != 0 or map_unstaged:
        print("Repo graph is stale: run `python tools/repo_graph.py` and stage docs/",
              file=sys.stderr)
        ok = False

    py_files = [f for f in files if f.endswith(".py")]
    if py_files and _run([sys.executable, "-m", "ruff", "check", *py_files]) != 0:
        ok = False

    if os.environ.get("BOINGA_SKIP_TESTS") == "1":
        if all(_is_docs(f) for f in files):
            print("WARNING: BOINGA_SKIP_TESTS=1, tests skipped (docs-only commit)",
                  file=sys.stderr)
        else:
            print("BLOCKED: BOINGA_SKIP_TESTS=1 is only allowed for docs-only commits",
                  file=sys.stderr)
            ok = False
    elif _run([sys.executable, "-m", "pytest", "-q", "-x"]) != 0:
        ok = False

    added = added_code_lines()
    if added > LARGE_COMMIT_LINES:
        print(f"WARNING: Large commit: consider splitting (atomic-commit skill) "
              f"[{added} non-test, non-docs lines added]", file=sys.stderr)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
