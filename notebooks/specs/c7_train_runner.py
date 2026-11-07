"""Spec: C7 run 1, training runner: prepared data, fine-tune config, loader, smoke and resume (T4)."""

from nbkit import Notebook, code, md

# Cells C7.pre and C7.1 are shell lines (C7.1 promptbook): a non-zero exit marks the cell FAIL
# through expect= but does not stop "Run all". The slug uses a hyphen: nbkit slugs allow
# [a-z0-9-] only (the promptbook's "train_runner" is not a valid slug).

NOTEBOOK = Notebook(
    chunk="C7",
    run=1,
    slug="train-runner",
    title="C7 — training runner and smoke test",
    branch="chunk/C7-train-runner",
    gpu=True,
    cells=[
        md("""
            **Prepared data on this VM.** Training reads wavs and manifests from the fast disk.
            `fetch-data` makes sure the corpus is here (download, unzip and copy show progress
            bars); `prepare` restores the prepared dataset from its Drive tar (progress bar)
            instead of recomputing it, so expect `restored ...` and `0 clip(s) to do`. Expect
            `PASS: dataset ready` and `PASS: prepared <n> clips`.
        """),
        code("C7.pre", "Restore prepared data on this VM", "!boinga fetch-data\n!boinga prepare",
             expect=[r"PASS: dataset ready", r"PASS: prepared \d+ clips"]),
        md("""
            **Fine-tune config.** Builds `configs/vits_ctg_finetune.json` from the base voice's
            pretrained config.json plus the logged overrides (halved learning rates, batch 16,
            fp16, 0.5-15 s clips, our manifests, runs on Drive) for run `smoke`, and refuses any
            change to the audio, characters or model settings. Then coqui's own `VitsConfig`
            loads it. Expect `PASS: config written`, the config and overrides blocks, the key
            fields, and `PASS: coqui accepts config`.
        """),
        code("C7.1", "Build and validate config",
             "!boinga build-config --run smoke\n!python tools/validate_config.py",
             expect=[r"PASS: config written", r"PASS: coqui accepts config"]),
        md("""
            **One-batch loader check.** Loads the fine-tune config into coqui, builds the VITS
            model and its train data loader exactly as training will, and pulls one batch
            (a few seconds, no progress bar needed). It prints the loader's signature first,
            then per tensor the shape, token id range against the text-embedding size, clip
            seconds and NaN/inf counts (spectrograms included). Expect `PASS: loader ok`; a
            token id outside the vocabulary or a NaN/inf prints `FAIL: loader <reason>`.
        """),
        code("C7.2", "One-batch loader check", "!boinga check-loader",
             expect=[r"PASS: loader ok"]),
        md("""
            **Smoke test: 50 steps (not timed yet; model load plus 50 steps).** Starts run
            `smoke` from the pretrained male checkpoint (fp16, batch 16) and stops after 50
            steps with a saved checkpoint on Drive. coqui prints a step log every 5 steps (smoke
            run only; 25 otherwise), its progress output for this cell. Re-running "Run all" is safe: a run already at
            or past step 50 prints `already at step N` and trains nothing. Expect `STOP at step
            50` (or `already at step N` on a re-run); coqui's
            "Partial model initialization" line would mean the restore did not match the model
            and fails the cell.
        """),
        code("C7.4.a", "Smoke test: 50 steps", "!boinga train --run smoke --max-steps 50",
             expect=[r"STOP at step 50|already at step (50|60)"],
             forbid=[r"Partial model initialization"],
             gpu=True, timeout_min=60),
        md("""
            **Resume test: to 60 steps.** Runs the same command with a higher limit: it must
            pick up the step-50 checkpoint from Drive (optimizer and step restored, never the
            pretrained weights again) and stop at step 60. This is what happens after a Colab
            disconnect. Expect `RESUME from step 50` and `STOP at step 60` (a re-run prints
            `already at step 60`).
        """),
        code("C7.4.b", "Resume test: to 60 steps", "!boinga train --run smoke --max-steps 60",
             expect=[r"RESUME from step 50|already at step 60",
                     r"STOP at step 60|already at step 60"],
             forbid=[r"Partial model initialization"], gpu=True, timeout_min=30),
        md("""
            **Throughput.** Reads the run's coqui log and `boinga_run.json`: seconds per step
            (the smoke run logs every 5 steps; step 0 is warm-up and left out, so steps 5-45
            and 55 are timed; coqui's running average is reported too), steps per hour, peak
            GPU memory, last losses and any NaN loss, into `reports/throughput.json`. Expect
            `PASS: throughput recorded` and the report block; a NaN loss prints `FAIL: NaN
            loss` (then switch to fp32).
        """),
        code("C7.4.c", "Throughput", "!boinga throughput --run smoke",
             expect=[r"PASS: throughput recorded", r"=== BEGIN reports/throughput.json ==="]),
        md("""
            **Listen: smoke checkpoint vs pretrained.** The run's first test sentence (a val
            line), read by the pretrained male voice and by the step-60 smoke checkpoint, with
            the fixed seed and noise scales. After 60 steps the voice should still sound like
            the pretrained one; this only checks that a training checkpoint synthesises (no
            silence, no noise burst). Nothing is checked automatically.
        """),
        code("C7.4.d", "Listen: smoke checkpoint vs pretrained", """
            from boinga_translator import colab_runtime, train_report

            colab_runtime.play(*train_report.listen_smoke("smoke"))
        """, expect=[], gpu=True),
    ],
)
