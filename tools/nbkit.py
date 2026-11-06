"""Notebook spec API (stdlib only): md / code / form cells and the Notebook a spec defines."""

# A spec file (notebooks/specs/<name>.py) sets NOTEBOOK = Notebook(...). tools/build_notebook.py
# turns it into a Colab .ipynb; tools/read_colab_run.py checks the executed copy against the
# spec's expect/forbid regexes (re.MULTILINE), not against the cell metadata Colab may drop.

from __future__ import annotations

import json
import keyword
import re
import textwrap
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

ALWAYS_FORBID: tuple[str, ...] = (r"^FAIL:", "Traceback")
RESERVED_IDS = re.compile(r"^(BOOT\.\d+|HDR)$")  # bootstrap and run-header cells (builder)
_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_SLUG = re.compile(r"^[a-z0-9][a-z0-9-]*$")
_RUNTIME = "from boinga_translator.colab_runtime import"


@dataclass(frozen=True)
class Cell:
    """One notebook cell; code/form cells carry an id and their output checks."""

    kind: str  # "markdown" | "code" | "form"
    source: str
    id: str | None = None
    title: str = ""
    expect: tuple[str, ...] = ()
    forbid: tuple[str, ...] = ()
    gpu: bool = False
    timeout_min: int | None = None

    @property
    def metadata(self) -> dict[str, object]:
        """Cell metadata for the .ipynb (a copy of the spec's checks, for humans)."""
        if self.kind == "markdown":
            return {}
        return {"boinga": {"id": self.id, "kind": self.kind, "expect": list(self.expect),
                           "forbid": list(self.forbid), "gpu": self.gpu,
                           "timeout_min": self.timeout_min}}


def _check_id(cell_id: str) -> None:
    if not _ID.match(cell_id) or RESERVED_IDS.match(cell_id):
        raise ValueError(f"bad or reserved cell id {cell_id!r}: use letters, digits, . _ -")


def _check_title(title: str) -> None:
    if not title.strip() or "\n" in title or "\r" in title:
        raise ValueError(f"cell title must be one non-empty line, got {title!r}")


def _patterns(items: Iterable[str]) -> tuple[str, ...]:
    out = tuple(dict.fromkeys(items))  # de-duplicate, keep order
    for pattern in out:
        try:
            re.compile(pattern, re.MULTILINE)
        except re.error as exc:
            raise ValueError(f"bad regex {pattern!r}: {exc}") from exc
    return out


def _body(src: str) -> str:
    return textwrap.dedent(src).strip("\n")


def _checked_cell(kind: str, cell_id: str, title: str, lines: list[str], expect: Iterable[str],
                  forbid: Iterable[str], gpu: bool, timeout_min: int | None) -> Cell:
    _check_id(cell_id)
    _check_title(title)
    if timeout_min is not None and timeout_min < 1:
        raise ValueError(f"timeout_min must be >= 1 minute, got {timeout_min}")
    expect_t = _patterns(expect)
    body = "\n".join(lines)
    if not expect_t and any(ln.lstrip().startswith("!") for ln in body.split("\n")):
        raise ValueError(f"cell {cell_id!r} runs a shell command (!...) but has no expect: "
                         "a failing shell command does not stop Colab")
    source = "\n".join([f"# [boinga {cell_id}] {title}", *([body] if body else []),
                        f"{_RUNTIME} cell_ok; cell_ok('{cell_id}')"])
    return Cell(kind, source, cell_id, title, expect_t, _patterns([*ALWAYS_FORBID, *forbid]),
                gpu, timeout_min)


def md(text: str) -> Cell:
    """Markdown cell (dedented, surrounding blank lines dropped)."""
    return Cell("markdown", _body(text))


def code(cell_id: str, title: str, src: str, expect: Iterable[str] = (),
         forbid: Iterable[str] = (), gpu: bool = False, timeout_min: int | None = None) -> Cell:
    """Code cell: marker line, dedented src, then cell_ok('<id>')."""
    return _checked_cell("code", cell_id, title, [_body(src)], expect, forbid, gpu, timeout_min)


def _literal(value: object) -> str:
    """JSON text that is also a one-line Python literal (U+2028/9 escaped for splitlines)."""
    text = json.dumps(value, ensure_ascii=False)
    return text.replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")


def _param_line(name: str, value: object) -> str:
    if not name.isidentifier() or keyword.iskeyword(name):
        raise ValueError(f"form field name {name!r} is not a Python identifier")
    if isinstance(value, bool):
        return f'{name} = {value} #@param {{type:"boolean"}}'
    if isinstance(value, int):
        return f'{name} = {value} #@param {{type:"integer"}}'
    if isinstance(value, str):
        return f'{name} = {_literal(value)} #@param {{type:"string"}}'
    if (isinstance(value, Sequence) and value
            and all(isinstance(v, str) for v in value)):  # dropdown, first option is default
        return f"{name} = {_literal(value[0])} #@param {_literal(list(value))}"
    raise ValueError(f"form field {name!r}: use str, int, bool or a non-empty list of str")


def form(cell_id: str, title: str, fields: Mapping[str, object], after_src: str = "",
         expect: Iterable[str] = ()) -> Cell:
    """Colab form cell: one #@param line per field, record_notes, after_src, cell_ok."""
    if not fields:
        raise ValueError(f"form {cell_id!r} needs at least one field")
    params = [_param_line(name, value) for name, value in fields.items()]
    notes = ", ".join(f'"{name}": {name}' for name in fields)
    lines = [f"#@title {title}", *params,
             f"{_RUNTIME} record_notes; record_notes('{cell_id}', {{{notes}}})"]
    if after_src.strip():
        lines.append(_body(after_src))
    return _checked_cell("form", cell_id, title, lines, expect, (), False, None)


@dataclass(frozen=True)
class Notebook:
    """What a spec defines: one Colab run of one chunk."""

    chunk: str
    run: int
    slug: str
    title: str
    branch: str
    cells: Sequence[Cell]
    gpu: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "cells", tuple(self.cells))
        if self.run < 1:
            raise ValueError(f"run must be >= 1, got {self.run}")
        if not _SLUG.match(self.slug):
            raise ValueError(f"slug must be lowercase letters, digits and -, got {self.slug!r}")
        for name in ("chunk", "title", "branch"):
            _check_title(getattr(self, name))
        ids = [str(c.id) for c in self.checked_cells]
        dupes = sorted({i for i in ids if ids.count(i) > 1})
        if dupes:
            raise ValueError(f"duplicate cell ids: {dupes}")

    @property
    def checked_cells(self) -> tuple[Cell, ...]:
        """Code and form cells, in order (the ones the reader checks)."""
        return tuple(c for c in self.cells if c.kind != "markdown")

    @property
    def forms(self) -> tuple[Cell, ...]:
        """Form cells, in order (the builder puts them under "Your observations")."""
        return tuple(c for c in self.cells if c.kind == "form")

    def cell(self, cell_id: str) -> Cell:
        """The code/form cell with this id (KeyError if none)."""
        for c in self.checked_cells:
            if c.id == cell_id:
                return c
        raise KeyError(cell_id)
