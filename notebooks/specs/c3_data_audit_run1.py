"""Spec: C3 run 1, Chittagonian corpus: download, verify and unpack it, then record its layout."""

from nbkit import Notebook, code

NOTEBOOK = Notebook(
    chunk="C3",
    run=1,
    slug="data-audit",
    title="Download and layout",
    branch="chunk/C3-data-audit",
    gpu=False,
    cells=[
        code("C3.1", "Download and unpack corpus", "!boinga fetch-data",
             expect=[r"PASS: dataset ready at .+, \d+ files"]),
    ],
)
