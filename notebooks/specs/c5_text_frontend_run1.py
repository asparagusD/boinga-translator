"""Spec: C5 run 1, OOV report: selected speaker's model text vs the base voice's charset (CPU)."""

from nbkit import Notebook, code, md

# Cells call the CLI from Python and stop Run all on a non-zero exit: a failing `!` shell
# line does not stop the run (DECISIONS "Colab cells stop the run themselves on failure").
# The slug uses a hyphen: nbkit slugs allow [a-z0-9-] only.

NOTEBOOK = Notebook(
    chunk="C5",
    run=1,
    slug="text-frontend",
    title="C5 run 1 — OOV report",
    branch="chunk/C5-text-frontend",
    gpu=False,
    cells=[
        md("""
            **Dataset on this VM.** Makes sure the corpus is on the fast disk: reuses the
            verified Drive copy or downloads it; download, unzip and copy show progress bars.
            Expect `PASS: dataset ready`.
        """),
        code("C5.0", "Ensure dataset on this VM", """
            from boinga_translator import cli, colab_runtime

            if cli.main(["fetch-data"]) != 0:
                colab_runtime.fail("fetch-data failed (see the FAIL line above)")
        """, expect=[r"PASS: dataset ready"]),
        md("""
            **OOV report.** Lists characters in the selected clips' model text that the base
            voice's charset lacks, into `reports/oov_report.md`. Expect `PASS: oov report` and
            the report block.
        """),
        code("C5.2", "OOV report", """
            from boinga_translator import cli, colab_runtime

            if cli.main(["oov"]) != 0:
                colab_runtime.fail("oov failed (see the FAIL line above)")
        """, expect=[r"PASS: oov report", r"=== BEGIN reports/oov_report.md ==="]),
    ],
)
