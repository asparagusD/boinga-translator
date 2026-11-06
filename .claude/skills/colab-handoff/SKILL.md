---
name: colab-handoff
description: How to hand work to Google Colab and take results back. Use when a step needs a GPU, the real dataset, Google Drive, or the pretrained checkpoint, or when the user pastes Colab output.
---
# colab-handoff

You cannot see Colab. The user is your hands there.

## Sending a step to Colab
1. Make sure the code is committed and pushed (Colab pulls from GitHub).
2. Give the user ONE code block to paste as a new cell, always starting with:
```python
   %cd /content/boinga-translator
   !git pull -q && git log -1 --oneline
```
   then one call into boinga_translator (no logic in the cell), and printed
   checks that end in a clear `PASS:` or `FAIL:` line.
3. Say what output you expect (shapes, counts, file paths, roughly how long).
4. Ask the user to paste the full output, including warnings.

## Standard cell header
Every Colab cell you give the user after the bootstrap (notebooks/colab_bootstrap.py,
cells 1-4) must start with these lines. Replace the example branch `chunk/C1-paths-env`
with the real current chunk branch (`git branch --show-current`); a cell you hand
the user never contains a placeholder:
```python
import os; assert os.path.exists("/content/boinga-translator"), "FAIL: run the bootstrap cells first"
%cd /content/boinga-translator
!git checkout -q -- reports/ && git fetch -q && git switch chunk/C1-paths-env && git pull -q && git log -1 --oneline
```
- `git checkout -- reports/` drops reports re-generated on Colab so switch/pull
  can't fail on "local changes"; the pasted block and the Drive copy are the record.
- If the cell re-imports `boinga_translator` after a pull, restart the session
  first (Runtime > Restart session) or the old code stays cached.
- Add `!pip install -q -e ".[train]"` right after these lines whenever
  pyproject.toml changed on this branch since the last Colab run
  (`git diff <last Colab commit> -- pyproject.toml`).
- The cell must end with a `PASS:` or `FAIL:` line.
- Fetch/pull work without the token in any URL because bootstrap cell 2 stored
  it in git's credential store on the VM. After a VM reset, run the bootstrap again.

## Receiving output
- Compare with what you expected. Quote the lines that matter.
- If there's any traceback or FAIL line, switch to `debug-protocol`.
- Only commit "How verified" lines that came from the pasted output.

## Saving pasted reports
- On Colab, reports are printed with `report_io.emit(path, "reports/<name>")`,
  which wraps each file in `=== BEGIN <path> ===` / `=== END <path> ===`
  (files > 200 KB print a `SKIPPED (too big)` line with their sha256 instead).
- When the user pastes Colab output that contains BEGIN/END blocks:
  1. Save the paste verbatim to `local/pasted/<timestamp>.txt`
     (timestamp like `20261107-153000`; `local/` is git-ignored).
  2. Run `python tools/save_pasted_reports.py local/pasted/<timestamp>.txt`.
     It writes each block to its repo-relative path, and only under
     `reports/` or `configs/`. It rejects absolute paths, `..`, `.git/`,
     `.claude/`, code paths and Windows tricks (`:`, `NUL`, trailing dots).
  3. Check `git diff` on the written files, then commit them with the
     subchunk that needed them (atomic-commit).
- If the tool reports a missing END marker, the paste was truncated: ask the
  user to copy the full cell output again.

## Colab facts to respect
- Free tier: GPU not guaranteed; session can drop anytime; ~12 h max.
- Read data from /content (fast); write checkpoints/logs to Drive (safe).
- Drive root: /content/drive/MyDrive/boinga-translator/
