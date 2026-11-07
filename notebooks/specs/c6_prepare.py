"""Spec: C6 run 1, prepared dataset: resampled clips, splits and coqui manifests (CPU)."""

from nbkit import Notebook, code, md

# Cells C6.0, C6.4 and C6.3 are shell lines (C6.3/C6.4 promptbooks): a non-zero exit marks
# the cell FAIL through expect= but does not stop "Run all", so a failed prepare also fails
# the parse check after it. `boinga prepare` restores from the Drive tar on a new VM.

NOTEBOOK = Notebook(
    chunk="C6",
    run=1,
    slug="prepare",
    title="C6 — prepared dataset",
    branch="chunk/C6-prepare",
    gpu=False,
    cells=[
        md("""
            **Dataset on this VM.** Makes sure the corpus is on the fast disk: reuses the
            verified Drive copy or downloads it; download, unzip and copy show progress bars.
            Expect `PASS: dataset ready`.
        """),
        code("C6.0", "Ensure dataset on this VM", "!boinga fetch-data",
             expect=[r"PASS: dataset ready"]),
        md("""
            **Prepare the dataset (resumable).** Splits, 22,050 Hz wavs and coqui manifests on
            the fast disk, plus a Drive tar backup; a fresh VM restores the tar instead of
            recomputing. Progress bars for the restore, the transform and the backup. Expect
            `PASS: prepared <n> clips` and the `configs/splits_summary.json` block.
        """),
        code("C6.4", "Prepare dataset", "!boinga prepare",
             expect=[r"PASS: prepared \d+ clips", r"=== BEGIN configs/splits_summary.json ==="]),
        md("""
            **Coqui parses our manifests.** Loads the train/val/test manifests with coqui's own
            ljspeech loader and checks counts, audio files and text. Expect `PASS: coqui parsed
            <n> train / <n> val samples`.
        """),
        code("C6.3", "Coqui parses our manifests",
             "!python tools/check_manifest_parse.py",
             expect=[r"PASS: coqui parsed \d+ train / \d+ val samples"]),
        md("""
            **Listen.** 5 random prepared training clips with their manifest text, to hear the
            resampling, trimming and levels.
        """),
        code("C6.4.play", "Listen: 5 prepared clips", """
            from boinga_translator import colab_runtime, prepare

            colab_runtime.play(*prepare.pick_prepared(5))
        """, expect=[]),
    ],
)
