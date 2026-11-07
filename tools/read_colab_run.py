"""Read an executed Colab notebook (colab_runs/*.ipynb) and check every cell against its spec."""

# Cells are matched by their "# [boinga <id>] <title>" first line, never by metadata (Colab may
# drop it). Only text survives: stream text, text/plain and error tracebacks; images, audio,
# HTML are dropped; GitHub tokens are redacted everywhere. Long base64 runs are dropped only
# from the digest text (CellResult.text); CellResult.raw keeps them for saving report blocks.

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path

import build_notebook
import nbformat
import nbkit

MAX_LINES, HEAD, TAIL, TB_LINES = 150, 60, 60, 30
_MARKER = re.compile(r"^# \[boinga (\S+)\] ")
_KEEP = re.compile(r"PASS:|FAIL:|WARN|CELL-OK|RUN-HEADER|NOTES-MISSING|RESUME|=== (BEGIN|END) ")
_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
_BASE64 = re.compile(r"(?=[A-Za-z0-9+/=]*[A-Za-z0-9])[A-Za-z0-9+/=]{100,}")  # not ==== banners
_TOKEN = re.compile(r"\b(ghp|gho|ghs|ghu|github_pat)_[A-Za-z0-9_]+")
_PARAM = re.compile(r"^(\w+)\s*=\s*(.*?)\s*#\s*@param\b")


@dataclass(frozen=True)
class Spec:
    """A loaded spec: the Notebook and every cell of its built notebook, in order."""

    notebook: nbkit.Notebook
    path: str
    cells: tuple[nbkit.Cell, ...]


@dataclass(frozen=True)
class CellResult:
    """Outcome of one spec cell in the executed notebook."""

    id: str
    title: str
    status: str  # PASS | FAIL | ERROR | NOT RUN
    reasons: list[str]
    text: str  # trimmed, for the digest
    raw: str  # full text, for report blocks
    form_values: dict[str, object] | None


def load_spec(spec_path: Path) -> Spec:
    """Load a spec file and expand it to the full cell list the builder writes."""
    nb = build_notebook.load_spec(spec_path)
    rel = spec_path.resolve().relative_to(build_notebook.ROOT.resolve()).as_posix()
    return Spec(nb, rel, tuple(build_notebook.all_cells(nb, rel)))


def load_executed(path: Path) -> nbformat.NotebookNode:
    """Read an executed .ipynb (as downloaded from Colab) as nbformat v4."""
    return nbformat.read(path, as_version=4)


def _clean(text: str) -> str:
    """Strip ANSI codes and redact GitHub tokens (raw text keeps base64 for report blocks)."""
    return _TOKEN.sub("***", _ANSI.sub("", text))


def output_text(cell: nbformat.NotebookNode) -> tuple[str, str | None]:
    """(text of all outputs, error summary or None); binary outputs are dropped."""
    parts, error = [], None
    for out in cell.get("outputs", []):
        kind = out.get("output_type")
        if kind == "stream":
            parts.append(out.get("text", ""))
        elif kind in ("execute_result", "display_data"):
            plain = out.get("data", {}).get("text/plain")
            if plain:
                parts.append(plain if plain.endswith("\n") else plain + "\n")
        elif kind == "error":
            error = f"{out.get('ename', '')}: {out.get('evalue', '')}"
            tb = _ANSI.sub("", "\n".join(out.get("traceback", []))).splitlines()
            parts.append("\n".join(tb[-TB_LINES:]) + "\n")
    return _clean("".join(parts)), error and _clean(error)


def trim(text: str) -> str:
    """Keep the first 60 and last 60 lines plus marker lines when there are more than 150."""
    lines = text.splitlines()
    if len(lines) <= MAX_LINES:
        return text
    middle = [ln for ln in lines[HEAD:-TAIL] if _KEEP.search(ln)]
    dropped = len(lines) - HEAD - TAIL - len(middle)
    return "\n".join([*lines[:HEAD], f"... [{dropped} lines trimmed] ...", *middle,
                      *lines[-TAIL:]])


def digest_text(raw: str) -> str:
    """Text for the digest: long base64 runs dropped, then trimmed."""
    return trim(_BASE64.sub("[base64 dropped]", raw))


def form_values(source: str) -> dict[str, object]:
    """Current #@param values in a form cell's source (Colab rewrites these lines)."""
    values: dict[str, object] = {}
    for line in source.splitlines():
        m = _PARAM.match(line.strip())
        if m:
            try:
                value = ast.literal_eval(m.group(2))
            except (ValueError, SyntaxError):
                value = m.group(2)
            values[m.group(1)] = _clean(value) if isinstance(value, str) else value
    return values


def _status(spec_cell: nbkit.Cell, cell: nbformat.NotebookNode) -> tuple[str, list[str], str]:
    raw, error = output_text(cell)
    if not cell.get("outputs"):
        return "NOT RUN", [], raw
    if error:
        return "ERROR", [error], raw
    reasons = [f"expected /{p}/ not found" for p in spec_cell.expect
               if not re.search(p, raw, re.MULTILINE)]
    reasons += [f"forbidden /{p}/ found" for p in spec_cell.forbid
                if re.search(p, raw, re.MULTILINE)]
    if f"cell_ok('{spec_cell.id}')" in spec_cell.source and not re.search(
            rf"^CELL-OK {re.escape(str(spec_cell.id))}$", raw, re.MULTILINE):
        reasons.append("no CELL-OK line (cell did not reach its end)")
    return ("FAIL" if reasons else "PASS"), reasons, raw


def check_run(executed: nbformat.NotebookNode, spec: Spec) -> list[CellResult]:
    """One result per spec code/form cell, in spec order."""
    found: dict[str, nbformat.NotebookNode] = {}
    for cell in executed.cells:
        if cell.cell_type == "code":
            m = _MARKER.match("".join(cell.source))
            if m:
                found[m.group(1)] = cell  # last copy wins (a re-run duplicate is the newer one)
    results = []
    for sc in spec.cells:
        if sc.kind == "markdown" or sc.id is None:
            continue
        cell = found.get(sc.id)
        if cell is None:
            results.append(CellResult(sc.id, sc.title, "NOT RUN",
                                      ["cell not found in the notebook"], "", "", None))
            continue
        status, reasons, raw = _status(sc, cell)
        values = form_values("".join(cell.source)) if sc.kind == "form" else None
        results.append(CellResult(sc.id, sc.title, status, reasons, digest_text(raw), raw,
                                  values))
    return results
