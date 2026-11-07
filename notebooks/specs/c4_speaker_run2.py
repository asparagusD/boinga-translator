"""Spec: C4 run 2, save the mixed-voice selection (cluster 1, base voice male) from run 1 (CPU)."""

from nbkit import Notebook, code

# User decision after run 1 (C4.V): train on mixed voices = cluster 1 of
# reports/speaker_ranking.md (7,506 clips), base voice male. Reads Drive speaker_clusters.csv
# from run 1; no audio, GPU or listening needed.

NOTEBOOK = Notebook(
    chunk="C4",
    run=2,
    slug="speaker",
    title="C4 — save the mixed-voice selection",
    branch="chunk/C4-speaker",
    gpu=False,
    cells=[
        code("C4.3b", "Save cluster 1 (mixed voices) with base voice male", """
            from boinga_translator import cli, colab_runtime

            if cli.main(["save-speaker", "--id", "1", "--base-voice", "male"]) != 0:
                colab_runtime.fail("save-speaker failed (see the FAIL line above)")
        """, expect=[r"PASS: selection saved",
                     r"=== BEGIN configs/speaker_selection.json ===",
                     r'"id": 1,', r'"base_voice": "male"']),
    ],
)
