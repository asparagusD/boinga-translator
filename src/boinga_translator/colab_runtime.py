"""Helpers called inside generated Colab cells: run header, cell markers, notes, file checks."""

# Output lines are machine-read by tools/read_colab_run.py: "RUN-HEADER k=v ...",
# "CELL-OK <id>", "NOTES-MISSING <field>", "PASS: ..." / "FAIL: ..." and report blocks.
# Printing FAIL: alone does not stop Colab's Run all: cells call fail() (or let library code
# raise) so the run stops; the reader marks the cell FAIL/ERROR either way.
# Header values never contain spaces (a GPU name like "Tesla T4" prints as Tesla_T4).

from __future__ import annotations

import json
import subprocess
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import NoReturn

from boinga_translator import env_report, paths, report_io


def _git(*args: str) -> str:
    """Output of a git command in the repo, or "unknown" if git is missing or fails."""
    try:
        r = subprocess.run(["git", *args], cwd=paths.REPO, capture_output=True, text=True,
                           timeout=10, check=False)
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return r.stdout.strip() if r.returncode == 0 and r.stdout.strip() else "unknown"


def _token(value: object) -> str:
    """Header value with whitespace runs joined by "_" (a detached HEAD shows branch=HEAD)."""
    return "_".join(str(value).split()) or "unknown"


def run_header(chunk: str, run: int) -> dict[str, object]:
    """Record the env (reports/env.json block), then print the RUN-HEADER line; WARN if no GPU."""
    env = env_report.run()
    gpu = env.get("gpu_name") or "none"
    if gpu == "none":
        print("WARN: no GPU (gpu=none): Runtime > Change runtime type > T4 GPU if this run "
              "needs one")
    fields = {"chunk": chunk, "run": run, "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
              "commit": env.get("git_commit") or "unknown", "gpu": gpu,
              "utc": env.get("utc_time") or "unknown"}
    print("RUN-HEADER " + " ".join(f"{k}={_token(v)}" for k, v in fields.items()))
    return env


def cell_ok(cell_id: str) -> None:
    """Last line of every generated cell: proves the cell ran to the end."""
    print(f"CELL-OK {cell_id}")


def record_notes(note_id: str, values: Mapping[str, object]) -> Path:
    """Save form values to reports/notes/<id>.json (repo + Drive), emit it, flag empty strings."""
    if not note_id.strip() or "/" in note_id or "\\" in note_id:
        raise ValueError(f"note id must be a single non-empty name, got {note_id!r}")
    rel = f"reports/notes/{note_id}.json"
    text = json.dumps(dict(values), indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    repo_path, _ = report_io.write_report(text, rel)
    report_io.emit(repo_path, rel)
    for field, value in values.items():
        if isinstance(value, str) and not value.strip():
            print(f"NOTES-MISSING {field}")
    return repo_path


def fail(message: str) -> NoReturn:
    """Print "FAIL: <message>" and stop the cell (and Colab's Run all) with SystemExit(1)."""
    print(f"FAIL: {message}")
    raise SystemExit(1)


def expect_files(paths: Iterable[Path | str], min_bytes: int = 1, *, stop: bool = True) -> bool:
    """Print PASS/FAIL per file (exists, >= min_bytes); stop the run if any fail, else return
    whether all passed (with stop=True it only ever returns True)."""
    ok = True
    for raw in paths:
        path = Path(raw)
        if not path.is_file():
            print(f"FAIL: {path} missing")
            ok = False
            continue
        size = path.stat().st_size
        if size < min_bytes:
            print(f"FAIL: {path} too small ({size} < {min_bytes} bytes)")
            ok = False
        else:
            print(f"PASS: {path} ({size} bytes)")
    if not ok and stop:
        raise SystemExit(1)
    return ok


def play(wav_paths: Sequence[Path | str], labels: Sequence[str]) -> None:
    """Show a labelled audio player per wav in IPython; elsewhere print "AUDIO <label>: <path>"."""
    if len(wav_paths) != len(labels):
        raise ValueError(f"{len(wav_paths)} wav paths but {len(labels)} labels")
    try:
        import IPython

        shell = IPython.get_ipython()
    except (ImportError, AttributeError):
        shell = None
    if shell is None:
        for wav, label in zip(wav_paths, labels, strict=True):
            print(f"AUDIO {label}: {wav}")
        return
    from IPython.display import Audio, display

    for wav, label in zip(wav_paths, labels, strict=True):
        if not Path(wav).is_file():
            print(f"WARN: missing audio for {label}: {wav}")
            continue
        print(f"AUDIO {label}:")
        display(Audio(filename=str(wav)))
