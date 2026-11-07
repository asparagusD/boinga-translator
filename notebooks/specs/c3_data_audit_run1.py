"""Spec: C3 run 1, Chittagonian corpus: download, verify and unpack it, then record its layout."""

from nbkit import Notebook, code, md

NOTEBOOK = Notebook(
    chunk="C3",
    run=1,
    slug="data-audit",
    title="Download and layout",
    branch="chunk/C3-data-audit",
    gpu=False,
    cells=[
        md("""
            **Download and unpack the corpus.** Streams the pinned Mendeley zip to Drive
            (resumes if interrupted), checks its size and sha256, unzips it once on Drive and
            copies it to the fast VM disk, with progress bars for the download, unzip and copy.
            Expect `PASS: dataset ready at <path>, <n> files`.
        """),
        # Called from Python, not `!boinga fetch-data`: a failing shell line does not stop
        # Run all, and C3.2 would then inspect an empty folder (C3 run 1, DEBUG_LOG).
        code("C3.1", "Download and unpack corpus", """
            from boinga_translator import cli, colab_runtime

            if cli.main(["fetch-data"]) != 0:
                colab_runtime.fail("fetch-data failed (see the FAIL line above)")
        """, expect=[r"PASS: dataset ready at .+, \d+ files"]),
        md("""
            **Inspect the corpus layout.** Describes the unpacked tree (folders, tables, wav
            headers, metadata values) into `reports/corpus_layout.txt`, so later code is written
            against the real layout. Expect `PASS: layout inspected` and the report block.
        """),
        code("C3.2", "Inspect corpus layout",
             "!python tools/inspect_layout.py \"$(python -c 'from boinga_translator import paths;"
             " print(paths.data_fast())')\"",
             expect=[r"PASS: layout inspected", r"=== BEGIN reports/corpus_layout.txt ==="]),
    ],
)
