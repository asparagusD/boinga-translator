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
    ],
)
