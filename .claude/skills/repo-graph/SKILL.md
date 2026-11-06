---
name: repo-graph
description: Keep and use the repo map of boinga-translator. Use before planning any change, when asked where something lives, what depends on a module, or after adding, moving, renaming or deleting files.
---
# repo-graph

The repo map lives in:
- docs/REPO_GRAPH.md  — human view: folder tree with each file's purpose,
  a Mermaid dependency graph, and a "changes since last commit" list
- docs/repo_graph.json — machine view: nodes (files) and edges (imports,
  test->module, notebook->module, config->module)

## Use it
- Before planning: read docs/REPO_GRAPH.md. Name the nodes you will touch and
  their dependents (incoming edges). Mention dependents in your plan.
- To answer "what uses X?": search repo_graph.json edges where target == X.

## Keep it current
- After any file is added/removed/renamed or imports change:
  `python tools/repo_graph.py`
- The Claude Code PostToolUse hook runs it automatically after Write/Edit, and
  .githooks/pre-commit runs `python tools/repo_graph.py --check`, which fails
  if the map is stale. If it fails: run the tool, `git add docs/REPO_GRAPH.md
  docs/repo_graph.json`, commit again.
- If a file shows "(no docstring)" in the map, add a one-line module docstring.

## Never
- Never edit docs/REPO_GRAPH.md or repo_graph.json by hand.
