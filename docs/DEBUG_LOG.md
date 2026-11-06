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
