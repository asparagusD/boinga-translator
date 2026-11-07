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
- [x] C1.5 Colab bootstrap cells (notebooks/colab_bootstrap.py) + first Colab env report;
  two commits: A code + local tests (`cc31783`), B pasted reports/env.json (`9f760cf`);
  fix: drop untracked Colab reports before pulling (`29c7184`)
- [x] C1.6 Colab notebook notebooks/c1_colab_bootstrap.ipynb generated from the bootstrap
  cells by tools/build_notebooks.py (sync enforced by tests/test_notebooks.py)

### C1 follow-ups (from the whole-chunk review; not blocking)
- [ ] Bootstrap cell 2: warn loudly when BRANCH == "main" and newer chunk/* branches exist
- [ ] Drive budget: 13.0 GiB free after cleanup (was 5.9); keep unzipped data on /content,
  last 2 + best checkpoints, empty Drive trash after deletes (decide in the data/train chunks)
- [ ] report_io: Drive copy lands at ROOT/reports/reports/...; mirror repo paths as root()/rel
- [ ] paths.require_drive(): reusable Drive-mount guard instead of ensure_dirs() in write_report
- [ ] paths.COLAB_REPO constant + test that bootstrap literals match paths constants
- [ ] paths.REPO assumes an editable/source install: assert REPO/pyproject.toml exists
- [ ] DECISIONS: bootstrap cells may hold logic (they run before the package is importable)
- [ ] Optional: enforce ruff E501 (100 chars) and wrap the existing long lines

## C1N — notebook run pipeline
Branch `chunk/C1N-notebook-pipeline`. Per chunk: Claude Code writes code + a notebook spec;
the generated notebook is run once on Colab; the executed .ipynb goes to `colab_runs/`
(git-ignored); `tools/read_colab_run.py` verifies it and saves its reports per subchunk.
- [x] C1.6a tools/nbkit.py: spec API (md / code / form cells, cell ids, expectations)
- [x] C1.6b tools/build_notebook.py: spec -> .ipynb with the bootstrap cells and branch
  baked in, spec hash in metadata (the commit is recorded at run time by C1.7), `--check` for stale notebooks
- [x] C1.6c retire tools/build_notebooks.py: C1 bootstrap notebook is now
  notebooks/runs/C1_bootstrap_run1.ipynb from notebooks/specs/c1_bootstrap.py
- [x] C1.7 src/boinga_translator/colab_runtime.py: run_header (RUN-HEADER line + env
  report), cell_ok (CELL-OK line), record_notes, expect_files, play
  Merge gate: C1N must not merge to main without C1.7 (every built notebook's HDR cell
  calls run_header; BOOT.2 checks out the spec's branch, main for the C1 bootstrap)
- [x] C1.8a tools/read_colab_run.py core: load executed .ipynb, drop binary outputs, trim text,
  identify cells, per-cell status vs the spec (PASS / FAIL / ERROR / NOT RUN), form values,
  token redaction
- [x] C1.8b read_colab_run CLI: spec auto-found from RUN-HEADER, staleness, digest
  reports/colab_runs/<chunk>_run<k>.md, stdout table, --save-reports, exit code
- [x] C1.9 colab-handoff skill update, new colab-verify skill, CLAUDE.md, .gitignore
  (`colab_runs/`), DECISIONS entry; colab-verify must say a STALE run exits 1 (re-run it)
- [x] C1.10 pilot: env-check spec + notebook, run on Colab, verified with the reader
  (A = spec + notebook, B = reports from the verified run)

### C1N follow-ups (before the C1N merge)
- [ ] reports/env.json is emitted twice per run (BOOT.4 and HDR's run_header);
  report_io.parse_blocks rejects duplicates, so save_pasted_reports fails on a full paste.
  Make HDR the only emitter in built notebooks; the C1.8 reader parses per cell anyway
- [ ] Stop-on-FAIL: cell_ok only prints CELL-OK and expect_files only returns False, so a
  FAIL line does not stop Run all; decide raise-in-cell vs reader-only (C1.8 / C1.9)

## C2
- [ ] C2.x (subchunks not yet defined; draft plan parked in `git stash` on
  chunk/C2-pretrained-baseline)

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
