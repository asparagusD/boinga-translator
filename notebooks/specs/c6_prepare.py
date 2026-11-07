"""Spec: C6 run 1, prepared dataset: resampled clips, splits and coqui manifests (CPU)."""

from nbkit import Notebook, code

# C6.4 adds the fetch-data and prepare cells before the parse check. The parse check is a
# shell line (C6.3 promptbook): a non-zero exit marks the cell FAIL through expect= but
# does not stop "Run all", so it stays the last code cell (or moves into cli + colab_runtime).

NOTEBOOK = Notebook(
    chunk="C6",
    run=1,
    slug="prepare",
    title="C6 — prepared dataset",
    branch="chunk/C6-prepare",
    gpu=False,
    cells=[
        code("C6.3", "Coqui parses our manifests",
             "!python tools/check_manifest_parse.py",
             expect=[r"PASS: coqui parsed \d+ train / \d+ val samples"]),
    ],
)
