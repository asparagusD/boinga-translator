"""Spec: C13 run 1, repro and release: the reproducibility report (CPU)."""

from nbkit import Notebook, code, md

NOTEBOOK = Notebook(
    chunk="C13",
    run=1,
    slug="release",
    title="C13 — repro and release",
    branch="chunk/C13-release",
    gpu=False,
    cells=[
        md("""
            **Reproducibility report.** `boinga repro` records what this release rests on
            into `reports/repro.json` (and a Drive copy): the environment (Python, GPU or none,
            git commit), the versions of torch, coqui-tts, numpy, librosa and gradio, the
            sha256 of `configs/vits_ctg_finetune.json`, `configs/splits_summary.json` and
            `configs/char_map.json`, the seed, the provenance of the best checkpoint
            (`reports/best_provenance.json`), and the steps (highest checkpoint) and training
            hours (sum of the recorded sessions) of each run folder on Drive. It takes
            seconds and needs no GPU. A missing input prints a `WARN:` line instead of
            failing. Expect the file between `=== BEGIN reports/repro.json ===` / `=== END
            ... ===` markers, then `PASS: repro recorded`.
        """),
        code("C13.1", "Reproducibility report", "!boinga repro",
             expect=[r"PASS: repro recorded", r"=== BEGIN reports/repro.json ==="]),
    ],
)
