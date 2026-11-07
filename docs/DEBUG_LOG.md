# Debug log

Every bug handled with the debug-protocol skill gets an entry here (step 7),
newest last, committed together with its `fix(<scope>): ...` commit.

Template:

```
## YYYY-MM-DD C<n>.<m> <one-line symptom>
**Symptom:** <exact command, error / traceback excerpt, where it ran (local / Colab)>
**Root cause:** <the first wrong value and why it was wrong>
**Fix:** <smallest change that fixed the root cause>
**Test added:** <tests/test_x.py::test_name, or the Colab cell re-run and its output>
**Commit:** <short hash>
```

## Entries

## 2026-11-07 C1.5 Colab cell 4: ImportError: cannot import name 'env_report'
**Symptom:** First Colab bootstrap run. Cell 4 raised `ImportError: cannot import
name 'env_report' from 'boinga_translator'`; cell 2 had printed
`e5c1a7b merge(C0): ...` and `PASS: /content/boinga-translator on main`.
**Root cause:** `BRANCH` in cell 2 was left at its default `"main"`, so Colab
checked out main (C0 only). `env_report.py` exists only on `chunk/C1-paths-env`.
Operator step, not a code bug.
**Fix:** Set `BRANCH = "chunk/C1-paths-env"` and re-ran cells 2 and 4 (cell 3's
`[train]` deps are identical on both branches). Cell 2 then printed `cc31783`, and
cell 4 printed `PASS: env recorded`.
**Test added:** none (no code change). Follow-up idea: make cell 2 warn loudly when
`BRANCH == "main"` and newer `chunk/*` branches exist.
**Commit:** chore(colab): record first Colab environment report (C1.5 commit B)

## 2026-11-07 C1.5 Colab cell 2 pull aborts on an untracked reports/env.json
**Symptom:** Found in the whole-chunk C1 review: on a reused Colab VM where cell 4
ran before reports/env.json was committed, re-running cell 2 to pull a commit that
tracks it fails with `error: The following untracked working tree files would be
overwritten by merge: reports/env.json` -> `FAIL: git pull failed`.
**Root cause:** cell 2 (and the skill's standard cell header) only ran
`git checkout -- reports/`, which restores tracked files; reports created on Colab
stay untracked until they are committed locally and pushed.
**Fix:** also run `git clean -fdq -- reports/` before fetch/switch/pull in cell 2 and
in the colab-handoff standard header (header pull aligned to `--ff-only`). The pasted
block and the Drive copy are the record.
**Test added:** tests/test_colab_bootstrap.py::test_guards_and_branch asserts
checkout -> clean -> switch order; scratch-repo repro: old sequence errors, fixed
sequence pulls `track env`.
**Commit:** fix(colab): drop untracked Colab reports before pulling

## 2026-11-07 C3.1 build_notebook: slug must be lowercase letters, digits and -
**Symptom:** Local `python tools/build_notebook.py notebooks/specs/c3_data_audit_run1.py`
raised `ValueError: slug must be lowercase letters, digits and -, got 'data_audit'`.
**Root cause:** The C3.1 prompt gave `slug="data_audit"`; nbkit.Notebook only allows
`[a-z0-9-]` in slugs (they become notebook file names).
**Fix:** `slug="data-audit"` (matches the branch name and the C2 `pretrained-baseline`
style); the notebook is notebooks/runs/C3_data-audit_run1.ipynb.
**Test added:** none (nbkit already validates it; tests/test_nbkit.py covers the rule).
**Commit:** data(download): fetch corpus zip with resume, sha256 check and one-time unzip
