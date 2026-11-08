"""Spec: C8 timing run, smoke and resume at the RTX PRO 6000 settings (batch 32) to re-measure speed."""

from nbkit import Notebook, code, md

NOTEBOOK = Notebook(
    chunk="C8",
    run=1,
    slug="timing",
    title="C8 — timing at batch 32 (RTX PRO 6000)",
    branch="chunk/C8-finetune",
    gpu=True,
    cells=[
        md("""
            **Prepared data on this VM.** `fetch-data` makes sure the corpus is here; `prepare`
            restores the prepared dataset from its Drive tar (progress bars) instead of
            recomputing it. Expect `PASS: dataset ready` and `PASS: prepared <n> clips`.
        """),
        code("C8T.pre", "Restore prepared data on this VM", "!boinga fetch-data\n!boinga prepare",
             expect=[r"PASS: dataset ready", r"PASS: prepared \d+ clips"]),
        md("""
            **Config at the new settings.** Builds the fine-tune config for run `smoke` with the
            RTX PRO 6000 overrides (batch 32, eval batch 16, 8/4 loader workers, fp16) and
            checks that coqui's `VitsConfig` loads it. Expect `PASS: config written`, a line
            with `batch_size 32`, and `PASS: coqui accepts config`.
        """),
        code("C8T.1", "Build and validate config",
             "!boinga build-config --run smoke\n!python tools/validate_config.py",
             expect=[r"PASS: config written", r"batch_size 32", r"PASS: coqui accepts config"]),
        md("""
            **One-batch loader check.** Pulls one batch of 32 through the same loader training
            uses (a few seconds) and checks token ids and finite audio; the printed shapes start
            with the batch size (32). Expect `PASS: loader ok`.
        """),
        code("C8T.2", "One-batch loader check", "!boinga check-loader",
             expect=[r"PASS: loader ok"]),
        md("""
            **Fresh smoke run.** Deletes the old batch-16 smoke run on Drive (about 2 GB, two
            checkpoints; they go to the Drive trash, so empty it afterwards) so the timing
            starts from the pretrained checkpoint at the new settings. Only runs named
            `smoke...` under step 1000 can be deleted. Expect `PASS: cleaned run smoke` or
            `PASS: nothing to clean`.
        """),
        code("C8T.3", "Fresh smoke run", """
            from boinga_translator import cli, colab_runtime

            if cli.main(["clean-run", "--run", "smoke"]) != 0:
                colab_runtime.fail("clean-run failed (see the FAIL line above)")
        """, expect=[r"PASS: (cleaned run smoke|nothing to clean for run smoke)"]),
        md("""
            **Smoke test: 50 steps at batch 32.** Starts run `smoke` from the pretrained male
            checkpoint and stops after 50 steps with a checkpoint on Drive; coqui logs every 5
            steps (its progress output). The cell before deletes the smoke run, so "Run all"
            always re-times from step 0. Expect `STOP at step 50`; "Partial model initialization" would mean a bad restore and fails the
            cell.
        """),
        code("C8T.4", "Smoke test: 50 steps", "!boinga train --run smoke --max-steps 50",
             expect=[r"STOP at step 50|already at step (50|60)"],
             forbid=[r"Partial model initialization"], gpu=True, timeout_min=60),
        md("""
            **Resume test: to 60 steps.** Same command with a higher limit: it must continue
            from the step-50 checkpoint (never the pretrained weights again). Expect `RESUME
            from step 50` and `STOP at step 60`.
        """),
        code("C8T.5", "Resume test: to 60 steps", "!boinga train --run smoke --max-steps 60",
             expect=[r"RESUME from step 50|already at step 60",
                     r"STOP at step 60|already at step 60"],
             forbid=[r"Partial model initialization"], gpu=True, timeout_min=30),
        md("""
            **Throughput at batch 32.** Seconds per step, steps per hour, peak GPU memory, last
            losses and any NaN loss into `reports/throughput.json` (replaces the batch-16
            numbers). C8.1d derives run1's step budget from this. Expect `PASS: throughput
            recorded` and the report block with `"batch_size": 32`; a NaN loss prints
            `FAIL: NaN loss`.
        """),
        code("C8T.6", "Throughput", "!boinga throughput --run smoke",
             expect=[r"PASS: throughput recorded", r'"batch_size":\s*32']),
    ],
)
