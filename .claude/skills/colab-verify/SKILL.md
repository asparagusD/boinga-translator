---
name: colab-verify
description: Verify an executed Colab notebook the user put in colab_runs/. Use whenever the user says a run is done, uploads or mentions a .ipynb, or pastes a Verify prompt.
---
# colab-verify

## 1. Read the run (never open the .ipynb)
- Find the newest file in `colab_runs/` matching the expected
  `<chunk>_<slug>_run<k>.ipynb` (ask the user only if none matches).
- Run `python tools/read_colab_run.py colab_runs/<file> --save-reports`.
  Never open the notebook directly: it can hold large outputs and secrets; the
  reader drops binary output and redacts tokens.
- Read `reports/colab_runs/<chunk>_<slug>_run<k>.md` (the digest): header, staleness,
  status table, form values, trimmed output per cell.

## 2. Stop if STALE
The reader exits 1 when the run is STALE (the run's commit does not contain the
latest build of the notebook), even if every cell passed (DECISIONS
2026-11-07). Tell the user to open the current notebook link, "Run all" and
download again. Commit nothing from a STALE run: discard the digest and any
reports it saved (`git checkout -- reports/ configs/`, delete new files).
UNKNOWN staleness: say so and continue.

## 3. Failed cells
For each FAIL / ERROR / NOT RUN cell:
- follow `debug-protocol` (reasons and traceback tail are in the digest);
- each fix is its own `fix(...)` commit; rebuild the notebook
  (`tools/build_notebook.py`), commit and push it;
- tell the user to "Run all" again (cells are idempotent) and download again.
- Don't commit results for a subchunk whose cells did not all PASS.
- Revert report files that `--save-reports` wrote for failed subchunks
  (`git checkout -- <file>` / delete new ones) so they don't leak into commits.

## 4. Form values
Read them from the digest's "Form values" section. A required field is missing
when it is empty (`NOTES-MISSING <field>` in the output) or still `"(choose)"`
(dropdown placeholder). Ask the user in chat for that field only, and record
the answer in the results commit's Notes.

## 5. Commit results
For each subchunk whose cells all PASS, one results commit (type `data`,
`docs`, `train` or `eval`, per atomic-commit) containing its saved
`reports/...` and `reports/notes/...` files. The digest
`reports/colab_runs/<chunk>_<slug>_run<k>.md` goes into the first results commit of
the run. "How verified" quotes the PASS / key lines from the digest with their
cell ids, e.g. `[env] PASS: env recorded`, and the reader command and exit code.

## 6. Report
Finish with a table: subchunk | status (PASS / FAIL / STALE / waiting) |
commit hash (or "-").
