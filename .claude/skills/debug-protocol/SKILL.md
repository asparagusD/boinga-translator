---
name: debug-protocol
description: Structured debugging for boinga-translator. Use whenever a test fails, a command errors, Colab output shows a traceback, NaN loss, bad audio, an out-of-memory error, or the user says something is broken.
---
# debug-protocol

Work through these steps in order. Do not jump to a fix.

1. **Capture** — get the full error: exact command, full traceback, versions
   (reports/env.json), and the commit hash (`git rev-parse --short HEAD`).
   If it happened on Colab and you don't have the full output, ask for it.
2. **Reproduce** — reproduce at the smallest scale possible: one pytest, one
   file, one batch, or `max_steps=5`. Write the repro as a failing test if
   it's in our code.
3. **Isolate** — find the first wrong value, not the last symptom. Bisect:
   inputs → our function → library call. Print shapes, dtypes, min/max,
   NaN counts, sample rates, token IDs.
4. **Hypothesise** — write up to 3 ranked hypotheses, each with one cheap
   test that would confirm or kill it. Run them in order.
5. **Fix** — the smallest change that fixes the root cause. No unrelated
   refactors. Don't weaken tests to make them pass.
6. **Prove** — the repro test now passes, the full `pytest -q` passes, and
   for Colab issues the user re-runs the cell and pastes the output.
7. **Log** — append to docs/DEBUG_LOG.md:
   `## <date> <chunk> <one-line symptom>` then Symptom / Root cause / Fix /
   Test added / Commit.
8. **Commit** the fix on its own: `fix(<scope>): <what was wrong>`.

## Timeboxes and escalation
- 30 minutes on one bug in a subchunk without progress → stop, summarise what
  you know, and offer 2 options (workaround vs. deeper dig).
- 2 hours stuck on one chunk → tell the user to apply the fallback from the
  plan's risk register (e.g. switch to chunk F, the MMS stack).
- Never `pip install` random versions to "see if it helps". Version changes
  are a hypothesis and must be logged.

## Known symptom → first check
| Symptom | First check |
|---|---|
| CUDA OOM | batch size, max_audio_len, fp16 setting; print peak memory |
| NaN loss | fp16 on? LR too high? empty/zero audio in batch? text with only unknown chars? |
| Robotic / babbling output | LR too high or too many steps → earlier checkpoint |
| KeyError / size mismatch on restore | character set changed → embedding size differs from checkpoint |
| Silent or very short wavs | sample-rate mismatch, over-aggressive silence trim |
| Drive "Transport endpoint not connected" | remount Drive; resume from latest checkpoint |
| Very slow epochs | data read from Drive instead of /content; num_loader_workers |
