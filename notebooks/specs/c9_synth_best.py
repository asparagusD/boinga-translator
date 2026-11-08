"""Spec: C9 run 1, promote run1 step 36500 (the user's BEST_STEP) to best/, reload it, listen."""

# BEST_STEP 36500 from reports/notes/C8-notes.json (newest C8 notes); run name run1 from the
# "Training runs" table in DECISIONS "Run 1 retrained: checkpoints on Drive, promote step 36500".

from nbkit import Notebook, code, form, md

NOTEBOOK = Notebook(
    chunk="C9",
    run=1,
    slug="synth-best",
    title="C9 — best checkpoint",
    branch="chunk/C9-synth-best",
    gpu=True,
    cells=[
        md("""
            **Prepared data on this VM.** The reload check (C9.3) reads the first rows of the
            test manifest `metadata_test.csv`, which lives on this VM's fast disk. `fetch-data`
            makes sure the corpus is here; `prepare` restores the prepared dataset from its
            Drive tar (progress bars) instead of recomputing it. Expect `PASS: dataset ready`
            and `PASS: prepared <n> clips`.
        """),
        code("C9.pre", "Restore prepared data", "!boinga fetch-data\n!boinga prepare",
             expect=[r"PASS: dataset ready", r"PASS: prepared \d+ clips"]),
        md("""
            **Promote the chosen checkpoint.** Copies `checkpoint_36500.pth` of run1 (the step
            you chose by ear in C8, verdict "good: promote BEST_STEP") and the run's
            `config.json` into Drive `best/` as `model.pth` / `config.json`, then writes
            `provenance.json` last: run, step, eval losses, base model id + licence, dataset
            DOI + licence, speaker selection, char map, git commit, seed and UTC date. The ~1 GB
            copy shows a tqdm bar. It first says what `best/` held (`replaces: ...`); if `best/`
            already holds run1 step 36500 it copies nothing. Expect `PASS: promoted run1 step
            36500` and the provenance printed between BEGIN/END markers for the repo.
        """),
        code("C9.2", "Promote the chosen checkpoint",
             "!boinga promote --run run1 --step 36500",
             expect=[r"PASS: promoted", r"=== BEGIN reports/best_provenance.json ==="]),
        md("""
            **Reload the best checkpoint in a fresh process.** `tools/reload_check.py` runs as
            its own Python process, so nothing from the promote cell is in memory: it loads
            `best/model.pth` + `config.json` through the voice bank ("boinga" voice, fixed seed
            and inference scales) and speaks the first 3 rows of `metadata_test.csv` into Drive
            `eval/reload_check/` (tqdm bar; loading the model takes a little while). Each wav
            must be longer than 0.5 s with an RMS above -50 dBFS. Expect one line per wav and
            `PASS: reload check (3 wavs)`.
        """),
        code("C9.3", "Reload best checkpoint in a fresh process",
             "!python tools/reload_check.py", expect=[r"PASS: reload check"]),
        md("""
            **Listen: reload check.** Plays the 3 wavs the fresh process just made, each
            labelled with its clip id and the text it read. Nothing is checked automatically:
            judge them in the form below.
        """),
        code("C9.3.play", "Listen: reload check", """
            from boinga_translator import colab_runtime, synth

            colab_runtime.play(*synth.reload_players())
        """, expect=[]),
        md("""
            **Your observations — best checkpoint.** `INTELLIGIBLE`: can you follow the words
            of the 3 reload-check clips (yes / partly / no)? `NOTES`: anything else you heard
            (artefacts, wrong words, speed); write `none` if there is nothing to say. Both
            fields are required.
        """),
        form("C9-notes", "Your observations — best checkpoint", fields={
            "INTELLIGIBLE": ["", "yes", "partly", "no"],
            "NOTES": "",
        }),
    ],
)
