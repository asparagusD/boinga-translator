"""Spec: C4 run 1, training speaker: embed, cluster and rank voices, listen, save the choice (T4)."""

from nbkit import Notebook, code, form

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
        code("C4.3.play", "Listen: top 3 candidates, 10 clips each", """
            from boinga_translator import speaker_select

            speaker_select.play_candidates(3)
        """, expect=[]),
        # Saves only "yes" + an id + a gender; the base voice follows the speaker's gender.
        form("C4.3-notes", "Your choice — training speaker", fields={
            "CHOSEN_ID": "",
            "ONE_CONSISTENT_VOICE": ["", "yes", "no"],
            "GENDER": ["", "female", "male"],
            "AUDIO_QUALITY": ["", "clean", "some noise", "noisy"],
            "NOTES": "",
        }, after_src="""
            from boinga_translator import speaker_select

            speaker_select.save_from_form(CHOSEN_ID, ONE_CONSISTENT_VOICE, GENDER)
        """, expect=[r"PASS: selection saved"]),
    ],
)
