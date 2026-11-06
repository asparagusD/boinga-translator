"""Generate the repo map (docs/REPO_GRAPH.md + docs/repo_graph.json): file purposes and edges."""

from __future__ import annotations

import argparse
import ast
import fnmatch
import json
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
JSON_OUT = "docs/repo_graph.json"
MD_OUT = "docs/REPO_GRAPH.md"
CHANGED_HEADING = "## Changed since HEAD"
SKIP_DIRS = {
    ".git", ".venv", "venv", "__pycache__", "local", "outputs", ".ipynb_checkpoints",
    ".pytest_cache", ".ruff_cache", "wandb", "build", "dist",
}
SUFFIXES = {".py", ".json", ".toml", ".md", ".ipynb", ".yml", ".yaml", ".sh"}


def _ignore_patterns(root: Path) -> list[str]:
    gi = root / ".gitignore"
    if not gi.is_file():
        return []
    lines = (ln.strip() for ln in gi.read_text(encoding="utf-8").splitlines())
    return [ln for ln in lines if ln and not ln.startswith(("#", "!"))]


def _ignored(rel: str, is_dir: bool, patterns: list[str]) -> bool:
    name = rel.rsplit("/", 1)[-1]
    for pat in patterns:
        if pat.endswith("/"):
            if not is_dir:
                continue
            pat = pat.rstrip("/")
        pat = pat.lstrip("/")
        if fnmatch.fnmatch(name, pat) or fnmatch.fnmatch(rel, pat):
            return True
    return False


def iter_files(root: Path) -> list[str]:
    """Return repo-relative POSIX paths of every mapped file, sorted."""
    patterns, out = _ignore_patterns(root), []

    def walk(d: Path) -> None:
        for p in sorted(d.iterdir()):
            rel = p.relative_to(root).as_posix()
            if p.is_dir():
                skip = p.name in SKIP_DIRS or p.name.endswith(".egg-info")
                if not skip and not _ignored(rel, True, patterns):
                    walk(p)
            elif (
                rel not in (JSON_OUT, MD_OUT)
                and not _ignored(rel, False, patterns)
                and (p.suffix in SUFFIXES or (rel.startswith(".githooks/") and not p.suffix))
            ):
                out.append(rel)

    walk(root)
    return sorted(out)  # plain str sort: case-sensitive and identical on Windows and Linux


def kind_of(rel: str) -> str:
    if rel.startswith(".githooks/") or rel.endswith(".sh"):
        return "hook"
    if rel.startswith(".claude/skills/"):
        return "skill"
    if rel.startswith(".claude/agents/"):
        return "agent"
    if rel.endswith(".ipynb"):
        return "notebook"
    if rel.endswith(".py"):
        return {"tests": "test", "tools": "tool"}.get(rel.split("/")[0], "module")
    return "doc" if rel.endswith(".md") else "config"


def _nb_cells(text: str, cell_type: str) -> list[str]:
    try:
        cells = json.loads(text).get("cells", [])
    except (ValueError, AttributeError):
        return []
    if not isinstance(cells, list):
        return []
    return [
        "".join(c.get("source", ""))
        for c in cells
        if isinstance(c, dict) and c.get("cell_type") == cell_type
    ]


def purpose_of(rel: str, text: str, kind: str) -> str:
    if rel.endswith(".py"):
        try:
            doc = ast.get_docstring(ast.parse(text))
        except SyntaxError:
            doc = None
        return doc.strip().splitlines()[0] if doc and doc.strip() else "(no docstring)"
    if kind in ("skill", "agent") and text.startswith("---"):
        m = re.search(r"^description:\s*(.+)$", text.split("---", 2)[1], re.MULTILINE)
        if m:
            first = re.match(r"(.+?\.)(\s|$)", m.group(1).strip())
            return first.group(1) if first else m.group(1).strip()
    if kind == "notebook":
        text = "\n".join(_nb_cells(text, "markdown"))
    if kind == "hook":  # first comment line after the shebang
        heads = [ln for ln in text.splitlines() if ln.startswith("#") and not ln.startswith("#!")]
    elif rel.endswith(".md") or kind == "notebook":  # first markdown heading
        heads = [ln for ln in text.splitlines() if re.match(r"#+ ", ln)]
    else:
        heads = []
    return heads[0].lstrip("#").strip() if heads else ""


def module_name(rel: str) -> str:
    parts = rel[:-3].split("/")
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts[1:] if parts[0] == "src" else parts)


def module_index(files: list[str]) -> dict[str, str]:
    """Map importable dotted names to repo files (src/ stripped; tools/ is on pytest's path)."""
    idx: dict[str, str] = {}
    for rel in (f for f in files if f.endswith(".py")):
        idx.setdefault(module_name(rel), rel)
        parts = rel[:-3].split("/")
        if parts[0] == "tools" and len(parts) > 1:
            idx.setdefault(".".join(parts[1:]), rel)
    return idx


def imported_names(tree: ast.AST, modname: str, is_pkg: bool) -> set[str]:
    pkg = modname.split(".") if is_pkg else modname.split(".")[:-1]
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                if node.level - 1 > len(pkg):
                    continue
                base_parts = pkg[: len(pkg) - (node.level - 1)]
                base = ".".join(base_parts + ([node.module] if node.module else []))
            else:
                base = node.module or ""
            names.update(f"{base}.{a.name}" if a.name != "*" else base for a in node.names)
    return names


def resolve(name: str, idx: dict[str, str]) -> str | None:
    parts = name.strip(".").split(".")
    while parts:
        if ".".join(parts) in idx:
            return idx[".".join(parts)]
        parts.pop()
    return None


def _parse(src: str) -> ast.AST | None:
    try:
        return ast.parse(src)
    except SyntaxError:
        return None


def build_graph(root: Path) -> dict:
    files = iter_files(root)
    texts = {f: (root / f).read_text(encoding="utf-8", errors="replace") for f in files}
    nodes = []
    for f in files:
        kind = kind_of(f)
        nodes.append({"path": f, "kind": kind, "lines": len(texts[f].splitlines()),
                      "purpose": purpose_of(f, texts[f], kind)})
    idx, configs = module_index(files), [f for f in files if f.startswith("configs/")]
    edges: set[tuple[str, str, str]] = set()
    for f in files:
        if f.endswith(".py"):
            ekind = "test" if kind_of(f) == "test" else "import"
            tree = _parse(texts[f])
            names = imported_names(tree, module_name(f), f.endswith("__init__.py")) if tree else ()
            targets = {resolve(n, idx) for n in names}
            edges.update((f, t, ekind) for t in targets if t and t != f)
            edges.update(
                (f, c, "config") for c in configs
                if re.search(rf"(?<![\w/.]){re.escape(c)}(?![\w.])", texts[f])
            )
        elif f.endswith(".ipynb"):
            for cell in _nb_cells(texts[f], "code"):  # per cell: one bad cell can't hide others
                code = "\n".join(ln for ln in cell.splitlines()
                                 if not ln.lstrip().startswith(("%", "!")))
                tree = _parse(code)
                names = imported_names(tree, "", False) if tree else set()
                targets = {resolve(n, idx) for n in names if n.startswith("boinga_translator")}
                edges.update((f, t, "notebook") for t in targets if t)
    return {
        "nodes": nodes,
        "edges": [{"source": s, "target": t, "kind": k} for s, t, k in sorted(edges)],
    }


def _tree_key(path: str) -> list[tuple[int, str]]:
    parts = path.split("/")
    return [(0, p) for p in parts[:-1]] + [(1, parts[-1])]


def _nid(prefix: str, path: str) -> str:
    return prefix + re.sub(r"\W", "_", path)


def render_md(graph: dict, changed: list[str]) -> str:
    out = ["# Repo graph", "", "Generated by `tools/repo_graph.py`. Do not edit by hand.", "",
           "## Tree", ""]
    seen: set[str] = set()
    for n in sorted(graph["nodes"], key=lambda n: _tree_key(n["path"])):
        parts = n["path"].split("/")
        for i in range(len(parts) - 1):
            d = "/".join(parts[: i + 1])
            if d not in seen:
                seen.add(d)
                out.append("  " * i + f"- `{parts[i]}/`")
        purpose = f" — {n['purpose']}" if n["purpose"] else ""
        out.append("  " * (len(parts) - 1) + f"- `{parts[-1]}`{purpose} ({n['lines']} lines)")

    edges = graph["edges"]
    out += ["", "## Dependency graph", "", "```mermaid", "graph LR"]
    used = sorted({e["source"] for e in edges} | {e["target"] for e in edges})
    groups: dict[str, list[str]] = {}
    for p in used:
        groups.setdefault(p.split("/")[0] if "/" in p else "root", []).append(p)
    for top in sorted(groups):
        out.append(f'  subgraph {_nid("sg_", top)}["{top}"]')
        out += [f'    {_nid("n_", p)}["{p}"]' for p in groups[top]]
        out.append("  end")
    out += [f"  {_nid('n_', e['source'])} -->|{e['kind']}| {_nid('n_', e['target'])}"
            for e in edges]
    if not edges:
        out.append("  empty[no edges yet]")
    out += ["```", "", "## Hotspots", ""]

    deps: dict[str, list[str]] = {}
    for e in edges:
        deps.setdefault(e["target"], []).append(e["source"])
    py = [n["path"] for n in graph["nodes"] if n["kind"] in ("module", "tool") and n["path"] in deps]
    counts = Counter({p: len(deps[p]) for p in py})
    hot = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:5]
    out += [f"{i}. `{p}` — {c} dependent{'s' * (c != 1)} ({', '.join(f'`{s}`' for s in sorted(deps[p]))})"
            for i, (p, c) in enumerate(hot, 1)] or ["(no module has dependents yet)"]
    return "\n".join(out + ["", CHANGED_HEADING, "", *changed]) + "\n"


def changed_since_head(root: Path, graph: dict) -> list[str]:
    try:
        r = subprocess.run(["git", "show", f"HEAD:{JSON_OUT}"], cwd=root, check=False,
                           capture_output=True, text=True, encoding="utf-8")
        old = {n["path"] for n in json.loads(r.stdout)["nodes"]} if r.returncode == 0 else None
    except (OSError, ValueError, KeyError, TypeError):
        old = None
    if old is None:
        return [f"(no {JSON_OUT} in HEAD yet)"]
    new = {n["path"] for n in graph["nodes"]}
    lines = [f"- added: `{p}`" for p in sorted(new - old)]
    lines += [f"- removed: `{p}`" for p in sorted(old - new)]
    return lines or ["No files added or removed."]


def _stable(md: str | None) -> str | None:
    return md.split(CHANGED_HEADING)[0] if md is not None else None


def _read(p: Path) -> str | None:
    return p.read_text(encoding="utf-8") if p.is_file() else None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="exit 1 if the map on disk is stale (ignores 'Changed since HEAD')")
    ap.add_argument("--root", type=Path, default=ROOT, help="repo root (default: this repo)")
    args = ap.parse_args(argv)
    root = args.root.resolve()
    graph = build_graph(root)
    js = json.dumps(graph, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    md = render_md(graph, changed_since_head(root, graph))
    jp, mp = root / JSON_OUT, root / MD_OUT
    if args.check:
        stale = [JSON_OUT] if _read(jp) != js else []
        stale += [MD_OUT] if _stable(_read(mp)) != _stable(md) else []
        if stale:
            print(f"repo graph is stale: {', '.join(stale)}. "
                  "Run `python tools/repo_graph.py` and stage both files.", file=sys.stderr)
            return 1
        print("repo graph is up to date")
        return 0
    jp.parent.mkdir(parents=True, exist_ok=True)
    jp.write_text(js, encoding="utf-8", newline="\n")
    mp.write_text(md, encoding="utf-8", newline="\n")
    print(f"wrote {JSON_OUT} and {MD_OUT} ({len(graph['nodes'])} nodes, "
          f"{len(graph['edges'])} edges)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
