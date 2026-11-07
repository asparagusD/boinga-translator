"""Spec: C4 run 1, training speaker: embed every clean clip's speaker, then cluster and pick (T4)."""

from nbkit import Notebook, code

# Cells call the CLI from Python and stop Run all on a non-zero exit: a failing `!` shell
# line does not stop the run (C3 run 1, DEBUG_LOG; DECISIONS "Colab cells stop the run").

NOTEBOOK = Notebook(
    chunk="C4",
    run=1,
    slug="speaker",
    title="C4 — training speaker",
    branch="chunk/C4-speaker",
    gpu=True,  # T4 for the speaker encoder
    cells=[
        code("C4.0", "Ensure dataset on this VM", """
            from boinga_translator import cli, colab_runtime

            if cli.main(["fetch-data"]) != 0:
                colab_runtime.fail("fetch-data failed (see the FAIL line above)")
        """, expect=[r"PASS: dataset ready"]),
        code("C4.2a", "Speaker embeddings (20–40 min, resumable)", """
            from boinga_translator import cli, colab_runtime

            if cli.main(["embed-speakers"]) != 0:
                colab_runtime.fail("embed-speakers failed (see the FAIL line above)")
        """, expect=[r"PASS: \d+ embeddings"], gpu=True, timeout_min=60),
        code("C4.2b", "Cluster voices", """
            from boinga_translator import cli, colab_runtime

            if cli.main(["cluster-speakers"]) != 0:
                colab_runtime.fail("cluster-speakers failed (see the FAIL line above)")
        """, expect=[r"PASS: ranked \d+ clusters",
                     r"=== BEGIN reports/speaker_ranking.md ==="]),
    ],
)
