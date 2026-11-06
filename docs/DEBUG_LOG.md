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

(none yet)
