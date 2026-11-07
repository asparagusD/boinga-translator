"""Spec: C3 run 2, Chittagonian corpus: index, audit and summary (CPU)."""

from nbkit import Notebook, code

# Cells call the CLI from Python and stop Run all on a non-zero exit: a failing `!` shell
# line does not stop the run (C3 run 1, DEBUG_LOG).

NOTEBOOK = Notebook(
    chunk="C3",
    run=2,
    slug="data-audit",
    title="Index, audit, summary",
    branch="chunk/C3-data-audit",
    gpu=False,
    cells=[
        code("C3.2.fetch", "Ensure dataset on this VM", """
            from boinga_translator import cli, colab_runtime

            if cli.main(["fetch-data"]) != 0:
                colab_runtime.fail("fetch-data failed (see the FAIL line above)")
        """, expect=[r"PASS: dataset ready"]),
        code("C3.3", "Index corpus", """
            from boinga_translator import cli, colab_runtime

            if cli.main(["index"]) != 0:
                colab_runtime.fail("index failed (see the FAIL line above)")
        """, expect=[r"PASS: index has 8009 rows \(8004 with audio\)"]),
        code("C3.4c", "Audit all clips (10–30 min)", """
            from boinga_translator import cli, colab_runtime

            if cli.main(["audit"]) != 0:
                colab_runtime.fail("audit failed (see the FAIL line above)")
        """, expect=[r"PASS: audited \d+ clips"], timeout_min=45),
    ],
)
