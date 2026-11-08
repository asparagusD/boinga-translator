"""Spec: C9 run 1, promote run1 step 36500 (the user's BEST_STEP) to best/ with provenance."""

# BEST_STEP 36500 from reports/notes/C8-notes.json (newest C8 notes); run name run1 from the
# "Training runs" table in DECISIONS "Run 1 retrained: checkpoints on Drive, promote step 36500".

from nbkit import Notebook, code, md

NOTEBOOK = Notebook(
    chunk="C9",
    run=1,
    slug="synth-best",
    title="C9 — best checkpoint",
    branch="chunk/C9-synth-best",
    gpu=True,
    cells=[
        md("""
            **Promote the chosen checkpoint.** Copies `checkpoint_36500.pth` of run1 (the step
            you chose by ear in C8, verdict "good: promote BEST_STEP") and the run's
            `config.json` into Drive `best/` as `model.pth` / `config.json`, then writes
            `provenance.json` last: run, step, eval losses, base model id + licence, dataset
            DOI + licence, speaker selection, char map, git commit, seed and UTC date. The ~1 GB
            copy shows a tqdm bar. It first says what `best/` held (`replaces: ...`); if `best/`
            already holds run1 step 36500 it copies nothing. Expect `PASS: promoted run1 step
            36500` and the provenance printed between BEGIN/END markers for the repo.
        """),
        code("C9.2", "Promote the chosen checkpoint",
             "!boinga promote --run run1 --step 36500",
             expect=[r"PASS: promoted", r"=== BEGIN reports/best_provenance.json ==="]),
    ],
)
