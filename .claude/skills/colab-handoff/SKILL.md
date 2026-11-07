---
name: colab-handoff
description: "How to put work on Colab in boinga-translator: write cells into the chunk's notebook spec and hand the whole notebook to the user. Use for anything that needs a GPU, Drive, the dataset or the pretrained checkpoint."
---
# colab-handoff

You cannot see Colab. The user runs one whole notebook per chunk run and
downloads the executed copy into `colab_runs/`; the `colab-verify` skill takes
it from there.

## Never
- Never ask the user to paste single cells or copy cell output into chat.
- Never put logic in a cell. Never edit a built `.ipynb` by hand.
- Never change the bootstrap cells here: they come from
  `notebooks/colab_bootstrap.py` (BOOT.1-4) and the builder bakes the branch in.

## Write the spec
Colab work for chunk N goes into `notebooks/specs/cN_<slug>.py`, which defines
`NOTEBOOK = Notebook(chunk=..., run=..., slug=..., title=..., branch=..., cells=[...], gpu=True)`
(`gpu=` on the Notebook sets the runtime type; `code(gpu=...)` only marks a cell)
using `tools/nbkit.py`:
- `md(text)`: short explanation for the user.
- `code(id, title, src, expect=[...], forbid=[...], gpu=..., timeout_min=...)`
- `form(id, title, fields, after_src="", expect=[...])`

Each `code()` cell:
- is **preceded by an `md()` cell** that says what the cell does, why, and what
  output to expect (form cells too);
- shows **tqdm progress** when it may run longer than about a minute: the
  library function it calls uses tqdm (or the library's own progress output,
  e.g. coqui's step logs), so a slow cell is never silent;
- makes **one call** into `boinga_translator` (logic lives in src/);
- is **idempotent**: skips work already done (files present, hashes match) and
  resumes training from Drive, so "Run all" can always be repeated;
- prints clear `PASS: ...` / `FAIL: ...` lines, and **stops on failure**:
  call `colab_runtime.fail(msg)` (prints `FAIL:` and raises SystemExit) or let
  library code raise; `expect_files()` stops by default. A printed `FAIL:`
  alone does not stop "Run all" (the reader still marks the cell FAIL; a stopped
  cell shows as ERROR `SystemExit: 1`);
- has `expect=` regexes that prove success, written **before** the run (a
  `!shell` line without `expect=` is refused by nbkit, because a failing shell
  command does not stop Colab);
- prints anything that must reach the repo as a BEGIN/END block
  (`report_io.emit(path, "reports/<name>")`; only `reports/` and `configs/`).

Human judgement (listening, choosing a voice, rating) goes in `form()` cells,
which the builder puts under "Your observations": explicit field names, allowed
values as a dropdown list where possible. Required free-text fields default to
`""` (flagged as `NOTES-MISSING`); required dropdowns start with a `"(choose)"`
option, which colab-verify treats as missing. Never ask in chat for something a
form can capture. Form values
are saved by `record_notes` to `reports/notes/<id>.json`.

Bump `run=` for a new run of the same chunk; the old built notebook must then be
deleted (`build_notebook.py --all --check` flags it).

## Hand off
1. `python tools/build_notebook.py notebooks/specs/cN_<slug>.py` (or `--all`).
2. `pytest -q` and `ruff check .`; commit the spec + built notebook (with the
   code it calls, per atomic-commit); push the branch.
3. Give the user:
   - the Colab link
     `https://colab.research.google.com/github/<owner>/boinga-translator/blob/<branch>/<notebook path>`
     (owner from `git remote get-url origin`; branch = the chunk branch;
     path like `notebooks/runs/C2_baseline_run1.ipynb`); the repo is private,
     so if the link fails: Colab > File > Open notebook > GitHub, tick
     "Include private repos", pick the branch and the notebook;
   - the runtime type (T4 GPU, or CPU when `Notebook(gpu=False)`);
   - the expected duration (measured, or say it is not measured yet);
   - which form fields to fill after "Run all", and their allowed values;
   - the file name to save it as: `colab_runs/<notebook file name>`.
4. Wait. When the user says it is done, use the `colab-verify` skill.

## Colab facts to respect
- Free tier: GPU not guaranteed; the session can drop any time; ~12 h max.
- Read data from /content (fast); write checkpoints/logs to Drive (safe).
- Drive root: /content/drive/MyDrive/boinga-translator/.
- The bootstrap needs the `GH_TOKEN` Colab secret (BOOT.2 explains if missing).

## Fallback
If the user pastes BEGIN/END blocks anyway, save the paste to
`local/pasted/<timestamp>.txt` and run `python tools/save_pasted_reports.py
<file>`; then ask for the downloaded notebook so it can be verified properly.
