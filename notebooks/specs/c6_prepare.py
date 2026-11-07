"""Spec: C6 run 1, prepared dataset: resampled clips, splits and coqui manifests (CPU)."""

from nbkit import Notebook, code

# Cells C6.0, C6.4 and C6.3 are shell lines (C6.3/C6.4 promptbooks): a non-zero exit marks
# the cell FAIL through expect= but does not stop "Run all", so a failed prepare also fails
# the parse check after it. `boinga prepare` restores from the Drive tar on a new VM.

NOTEBOOK = Notebook(
    chunk="C6",
    run=1,
    slug="prepare",
    title="C6 — prepared dataset",
    branch="chunk/C6-prepare",
    gpu=False,
    cells=[
        code("C6.0", "Ensure dataset on this VM", "!boinga fetch-data",
             expect=[r"PASS: dataset ready"]),
        code("C6.4", "Prepare dataset", "!boinga prepare",
             expect=[r"PASS: prepared \d+ clips", r"=== BEGIN configs/splits_summary.json ==="]),
        code("C6.3", "Coqui parses our manifests",
             "!python tools/check_manifest_parse.py",
             expect=[r"PASS: coqui parsed \d+ train / \d+ val samples"]),
        code("C6.4.play", "Listen: 5 prepared clips", """
            from boinga_translator import colab_runtime, prepare

            colab_runtime.play(*prepare.pick_prepared(5))
        """, expect=[]),
    ],
)
