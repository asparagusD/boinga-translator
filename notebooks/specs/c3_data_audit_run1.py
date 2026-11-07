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
        # Called from Python, not `!boinga fetch-data`: a failing shell line does not stop
        # Run all, and C3.2 would then inspect an empty folder (C3 run 1, DEBUG_LOG).
        code("C3.1", "Download and unpack corpus", """
            from boinga_translator import cli, colab_runtime

            if cli.main(["fetch-data"]) != 0:
                colab_runtime.fail("fetch-data failed (see the FAIL line above)")
        """, expect=[r"PASS: dataset ready at .+, \d+ files"]),
        code("C3.2", "Inspect corpus layout",
             "!python tools/inspect_layout.py \"$(python -c 'from boinga_translator import paths;"
             " print(paths.data_fast())')\"",
             expect=[r"PASS: layout inspected", r"=== BEGIN reports/corpus_layout.txt ==="]),
    ],
)
