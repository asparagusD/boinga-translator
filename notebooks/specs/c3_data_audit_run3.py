"""Spec: C3 run 3, re-audit the corpus on the char-mapped text and re-summarise (CPU)."""

from nbkit import Notebook, code

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
        code("C3.8c.fetch", "Ensure dataset on this VM", """
            from boinga_translator import cli, colab_runtime

            if cli.main(["fetch-data"]) != 0:
                colab_runtime.fail("fetch-data failed (see the FAIL line above)")
        """, expect=[r"PASS: dataset ready"]),
        code("C3.8c.audit", "Re-audit all clips on the mapped text (~1 min)", """
            from boinga_translator import cli, colab_runtime

            if cli.main(["audit"]) != 0:
                colab_runtime.fail("audit failed (see the FAIL line above)")
        """, expect=[r"PASS: audited 8009 clips"], timeout_min=30),
        code("C3.8c.summary", "Audit summary with the char map", """
            from boinga_translator import cli, colab_runtime

            if cli.main(["audit-summary"]) != 0:
                colab_runtime.fail("audit-summary failed (see the FAIL line above)")
        """, expect=[r"PASS: summary written", r"=== BEGIN reports/audit_summary.md ===",
                     r"excluded character \(char_map"]),
    ],
)
