# Progress

Tick a subchunk in the same commit that completes it (boinga-workflow step 6),
with its short commit hash. Chunks C1–C13 get their subchunk list when the
promptbook for that chunk is handed over.

## C0 — Working system (no ML code)
- [x] C0.1 Repo skeleton, .gitignore, .gitattributes, pyproject.toml (`31e9666`)
- [x] C0.2 CLAUDE.md project rules (`444d614`)
- [x] C0.3a Skills: boinga-workflow, atomic-commit (`97acf15`)
- [x] C0.3b Skills: repo-graph, debug-protocol (`104814d`)
- [x] C0.3c Skills: colab-handoff, dialect-data-rules, vits-finetune (`e369d76`)
- [x] C0.4 Subagents: code-reviewer, debugger (`12a8f5c`)
- [x] C0.5a Repo graph tool: tools/repo_graph.py + generated map (`57b91a7`)
- [x] C0.5b Repo graph tests (`73bed9f`)
- [x] C0.5c Claude Code PostToolUse hook + permissions allow list (`c0fb112`)
- [x] C0.6 Git hooks: commit-msg format, binary guard, fresh repo graph (`8f56f81`)
- [x] C0.7 PROGRESS, DECISIONS and DEBUG_LOG docs (`28ea218`)

## C1 — Paths, settings, report I/O, env report, Colab bootstrap
- [x] C1.1 paths.py: single source of truth for paths (Drive on Colab, ./local elsewhere) (`8834a02`)
- [x] C1.2 settings.py: pinned model ids, dataset URL and sha256, seed (`93f6feb`)
- [x] C1.3 report_io.py: print files between BEGIN/END markers and import pasted output (`63f4c1b`)
- [x] C1.4 env_report.py + `boinga env` CLI (`4d3399f`)
- [ ] C1.5 Colab bootstrap cells (notebooks/colab_bootstrap.py) + first Colab env report;
  two commits: A code + local tests, B pasted reports/env.json

## C2
- [ ] C2.x (subchunks not yet defined)

## C3
- [ ] C3.x (subchunks not yet defined)

## C4
- [ ] C4.x (subchunks not yet defined)

## C5
- [ ] C5.x (subchunks not yet defined)

## C6
- [ ] C6.x (subchunks not yet defined)

## C7
- [ ] C7.x (subchunks not yet defined)

## C8
- [ ] C8.x (subchunks not yet defined)

## C9
- [ ] C9.x (subchunks not yet defined)

## C10
- [ ] C10.x (subchunks not yet defined)

## C11
- [ ] C11.x (subchunks not yet defined)

## C12
- [ ] C12.x (subchunks not yet defined)

## C13
- [ ] C13.x (subchunks not yet defined)

## F — Fallback: MMS stack (only if the VITS fine-tune path is abandoned)
- [ ] F.x (subchunks not yet defined; trigger rules in the debug-protocol skill)
