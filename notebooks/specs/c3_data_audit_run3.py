"""Spec: C3 run 3, re-audit the corpus on the char-mapped text and re-summarise (CPU)."""

from nbkit import Notebook, code, md

# C3.8c: the audit now judges text_frontend.prepare() output (DECISIONS "Character map").
# Run-2 audit parts on Drive are stale (new charset_id) and are redone, not reused.
# Cells call the CLI from Python and stop Run all on a non-zero exit (C3 run 1, DEBUG_LOG).

NOTEBOOK = Notebook(
    chunk="C3",
    run=3,
    slug="data-audit",
    title="Char-mapped re-audit",
    branch="chunk/C3-data-audit",
    gpu=False,
    cells=[
        md("""
            **Dataset on this VM.** Makes sure the corpus is on the fast disk: reuses the
            verified Drive copy or downloads it; download, unzip and copy show progress bars.
            Expect `PASS: dataset ready`.
        """),
        code("C3.8c.fetch", "Ensure dataset on this VM", """
            from boinga_translator import cli, colab_runtime

            if cli.main(["fetch-data"]) != 0:
                colab_runtime.fail("fetch-data failed (see the FAIL line above)")
        """, expect=[r"PASS: dataset ready"]),
        md("""
            **Re-audit on the char-mapped text (~1 min).** Same audit as run 2, but the text is
            judged after `configs/char_map.json`; stale run-2 parts are redone. A progress bar
            counts clips. Expect `PASS: audited 8009 clips`.
        """),
        code("C3.8c.audit", "Re-audit all clips on the mapped text (~1 min)", """
            from boinga_translator import cli, colab_runtime

            if cli.main(["audit"]) != 0:
                colab_runtime.fail("audit failed (see the FAIL line above)")
        """, expect=[r"PASS: audited 8009 clips"], timeout_min=30),
        md("""
            **Audit summary with the char map.** Rewrites `reports/audit_summary.md`, now with
            the char_map exclusion rule. Expect `PASS: summary written`, the report block and an
            `excluded character (char_map` row.
        """),
        code("C3.8c.summary", "Audit summary with the char map", """
            from boinga_translator import cli, colab_runtime

            if cli.main(["audit-summary"]) != 0:
                colab_runtime.fail("audit-summary failed (see the FAIL line above)")
        """, expect=[r"PASS: summary written", r"=== BEGIN reports/audit_summary.md ===",
                     r"excluded character \(char_map"]),
    ],
)
