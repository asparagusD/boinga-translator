"""Read an executed Colab notebook (colab_runs/*.ipynb) and check every cell against its spec."""

# Cells are matched by their "# [boinga <id>] <title>" first line, never by metadata (Colab may
# drop it). Only text survives: stream text, text/plain and error tracebacks; images, audio,
# HTML are dropped; GitHub tokens are redacted everywhere. Long base64 runs are dropped only
# from the digest text (CellResult.text); CellResult.raw keeps them for saving report blocks.

from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import build_notebook
import nbformat
import nbkit

from boinga_translator import paths, report_io

MAX_LINES, HEAD, TAIL, TB_LINES = 150, 60, 60, 30
_MARKER = re.compile(r"^# \[boinga (\S+)\] ")
_KEEP = re.compile(r"PASS:|FAIL:|WARN|CELL-OK|RUN-HEADER|NOTES-MISSING|RESUME|=== (BEGIN|END) ")
_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
_BASE64 = re.compile(r"(?=[A-Za-z0-9+/=]*[A-Za-z0-9])[A-Za-z0-9+/=]{100,}")  # not ==== banners
_TOKEN = re.compile(r"\b(ghp|gho|ghs|ghu|github_pat)_[A-Za-z0-9_]+")
_PARAM = re.compile(r"^(\w+)\s*=\s*(.*?)\s*#\s*@param\b")
_HEADER = re.compile(r"^RUN-HEADER (.*)$", re.MULTILINE)


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
        ran = cell.get("execution_count") is not None
        return "NOT RUN", ["executed but printed nothing"] if ran else [], raw
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


# ---- CLI: spec lookup, staleness, digest, --save-reports, exit code ----


def _git(*args: str) -> str:
    """git output in the repo ("ok" if it printed nothing), or "" if git fails."""
    try:
        r = subprocess.run(["git", *args], cwd=paths.REPO, capture_output=True,
                           text=True, timeout=20, check=False)
    except (OSError, subprocess.SubprocessError):
        return ""
    return (r.stdout.strip() or "ok") if r.returncode == 0 else ""


def run_header(executed: nbformat.NotebookNode) -> dict[str, str]:
    """Fields of the last RUN-HEADER line ({} if none); values never contain spaces (colab_runtime)."""
    found: dict[str, str] = {}
    for cell in executed.cells:
        if cell.cell_type == "code":
            for m in _HEADER.finditer(output_text(cell)[0]):
                found = dict(kv.split("=", 1) for kv in m.group(1).split() if "=" in kv)
    return found


def find_spec(executed: nbformat.NotebookNode, nb_name: str = "") -> Spec:
    """Match RUN-HEADER chunk/run; if several specs match, decide by metadata spec, then file name."""
    header = run_header(executed)
    meta = executed.metadata.get("boinga", {}).get("spec")
    if header.get("chunk") and header.get("run"):
        found = []
        specs = (build_notebook.ROOT / build_notebook.SPECS).glob("*.py")
        for path in sorted(p for p in specs if not p.name.startswith("_")):
            nb = build_notebook.load_spec(path)
            if nb.chunk == header["chunk"] and str(nb.run) == header["run"]:
                found.append((path, nb))
        if len(found) > 1:  # e.g. C1 bootstrap and C1 pilot are both run 1
            rel = {p: p.relative_to(build_notebook.ROOT).as_posix() for p, _ in found}
            by_meta = [(p, nb) for p, nb in found if rel[p] == meta]
            stem = Path(nb_name).stem
            by_name = [(p, nb) for p, nb in found
                       if stem and stem.startswith(Path(build_notebook.filename(nb)).stem)]
            found = by_meta or by_name
            if len(found) != 1:
                raise ValueError(f"chunk={header['chunk']} run={header['run']} matches "
                                 f"{sorted(rel.values())}: use --spec")
        if not found:
            raise ValueError(f"no spec for chunk={header['chunk']} run={header['run']}: "
                             "use --spec")
        return load_spec(found[0][0])
    if meta:
        return load_spec(build_notebook.ROOT / meta)
    raise ValueError("no RUN-HEADER output and no spec in metadata: use --spec")


def staleness(run_commit: str, nb_rel: str) -> str:
    """OK / STALE / UNKNOWN: did the run use a commit that contains the notebook build?"""
    if run_commit in ("", "unknown"):
        return "UNKNOWN: the run's commit is not recorded (no RUN-HEADER)"
    built = _git("log", "-1", "--format=%h", "--", nb_rel)
    if not built:
        return f"UNKNOWN: {nb_rel} has no commit yet"
    if run_commit != built:
        if not _git("cat-file", "-e", f"{run_commit}^{{commit}}"):
            return f"UNKNOWN: the run's commit {run_commit} is not in this clone (git fetch?)"
        if not _git("merge-base", "--is-ancestor", built, run_commit):
            return (f"STALE: the run used {run_commit}, which does not contain the notebook "
                    f"build {built}")
    head = _git("rev-parse", "--short", "HEAD")
    note = f"; HEAD is {head}" if head and head != run_commit else ""
    return f"OK: the run used {run_commit}, notebook built in {built}{note}"


def table(results: list[CellResult]) -> list[str]:
    """Markdown status table rows (also printed to stdout)."""
    rows = ["| Cell | Title | Status | Reason |", "|---|---|---|---|"]
    for r in results:
        reason = "; ".join(r.reasons).replace("|", r"\|")
        rows.append(f"| {r.id} | {r.title} | {r.status} | {reason} |")
    return rows


def _fence(text: str) -> str:
    ticks = "`" * max([3, *(len(m) + 1 for m in re.findall(r"`+", text))])
    return f"{ticks}text\n{text.rstrip()}\n{ticks}"


def _value(v: object) -> str:
    return json.dumps(v, ensure_ascii=False) if isinstance(v, str) else str(v)


def digest(nb_file: str, spec: Spec, header: dict[str, str], stale: str,
           results: list[CellResult]) -> str:
    """Markdown digest: run header, status table, form values, trimmed output per cell."""
    nb, passed = spec.notebook, sum(r.status == "PASS" for r in results)
    lines = [f"# Colab run {nb.chunk} run {nb.run} — {nb.title}", "",
             f"- File: `{nb_file}` · spec `{spec.path}`",
             (f"- Run: commit `{header.get('commit', 'unknown')}` · branch "
              f"`{header.get('branch', 'unknown')}` · GPU `{header.get('gpu', 'unknown')}` · "
              f"time `{header.get('utc', 'unknown')}`"),
             f"- Staleness: {stale}", f"- Result: {passed}/{len(results)} cells PASS", "",
             "## Cells", "", *table(results)]
    forms = [r for r in results if r.form_values is not None]
    if forms:
        lines += ["", "## Form values"]
        for r in forms:
            lines += ["", f"### {r.id} — {r.title}", ""]
            lines += [f"- `{k}` = {_value(v)}" for k, v in (r.form_values or {}).items()]
    lines += ["", "## Output"]
    for r in results:
        lines += ["", f"### {r.id} — {r.title}", "", _fence(r.text) if r.text else "(no output)"]
    return "\n".join(lines) + "\n"


def save_reports(results: list[CellResult]) -> list[Path]:
    """Save every BEGIN/END block (per cell; a later cell wins) with report_io's safety rules."""
    blocks: dict[str, str] = {}
    for r in results:
        try:
            blocks.update(report_io.parse_blocks(r.raw))
        except ValueError as err:
            print(f"WARN: {r.id}: {err}")
    return report_io.save_blocks(blocks, paths.REPO)


def exit_code(results: list[CellResult], stale: str = "") -> int:
    """0 if every spec cell passed and the run is not STALE, else 1 (UNKNOWN does not fail)."""
    if stale.startswith("STALE"):
        return 1
    return 0 if results and all(r.status == "PASS" for r in results) else 1


def main(argv: list[str] | None = None) -> int:
    """Check an executed notebook, write its digest, optionally save its report blocks."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("notebook", help="executed notebook, e.g. colab_runs/C1_pilot_run1.ipynb")
    ap.add_argument("--spec", help="spec file (default: found from the RUN-HEADER)")
    ap.add_argument("--save-reports", action="store_true", help="write BEGIN/END blocks to repo")
    args = ap.parse_args(argv)
    executed = load_executed(Path(args.notebook))
    try:
        spec = load_spec(Path(args.spec)) if args.spec else find_spec(executed, Path(args.notebook).name)
    except ValueError as err:
        print(f"error: {err}", file=sys.stderr)
        return 1
    results = check_run(executed, spec)
    header = run_header(executed)
    stale = staleness(header.get("commit", ""),
                      f"{build_notebook.RUNS}/{build_notebook.filename(spec.notebook)}")
    out = paths.reports_dir() / "colab_runs" / f"{Path(build_notebook.filename(spec.notebook)).stem}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(digest(Path(args.notebook).name, spec, header, stale, results),
                   encoding="utf-8", newline="\n")
    print("\n".join(table(results)))
    print(f"{stale}\ndigest: {out.relative_to(paths.REPO).as_posix()}")
    if args.save_reports:
        base = paths.REPO.resolve()
        try:
            saved_files = save_reports(results)
        except ValueError as err:  # one unsafe block path -> nothing is written
            print(f"error: {err} (no report files written)", file=sys.stderr)
            return 1
        if not saved_files:
            print("no '=== BEGIN <path> ===' blocks found")
        for saved in saved_files:
            print(f"saved {saved.relative_to(base).as_posix()} ({saved.stat().st_size} bytes)")
    return exit_code(results, stale)


if __name__ == "__main__":
    raise SystemExit(main())
