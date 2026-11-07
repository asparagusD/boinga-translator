"""Spec: C5 run 2, coqui tokenizer round trip of the selected speaker's model text (CPU)."""

from nbkit import Notebook, code

# Cells call the CLI from Python and stop Run all on a non-zero exit: a failing `!` shell
# line does not stop the run (DECISIONS "Colab cells stop the run themselves on failure").
# The slug uses a hyphen: nbkit slugs allow [a-z0-9-] only. check-tokenizer reads the base
# voice's config.json from the Drive cache (pretrained.fetch; downloads only if missing).

NOTEBOOK = Notebook(
    chunk="C5",
    run=2,
    slug="text-frontend",
    title="C5 run 2 — tokenizer check",
    branch="chunk/C5-text-frontend",
    gpu=False,
    cells=[
        code("C5.0", "Ensure dataset on this VM", """
            from boinga_translator import cli, colab_runtime

            if cli.main(["fetch-data"]) != 0:
                colab_runtime.fail("fetch-data failed (see the FAIL line above)")
        """, expect=[r"PASS: dataset ready"]),
        code("C5.4", "Tokenizer round trip", """
            from boinga_translator import cli, colab_runtime

            if cli.main(["check-tokenizer"]) != 0:
                colab_runtime.fail("check-tokenizer failed (see the FAIL line above)")
        """, expect=[r"PASS: tokenizer check unknown_rate=",
                     r"=== BEGIN reports/tokenizer_check.md ==="]),
    ],
)
