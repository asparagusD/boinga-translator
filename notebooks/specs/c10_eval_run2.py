"""Spec: C10 run 2, blind A/B evaluation: baseline, fine-tuned and real audio for the 20 prompts."""

# Prompts: configs/eval_prompts.json from C10 run 1 (97e9cee); the review (C10.1-notes)
# replaced none, so no `pick-eval-prompts --exclude` cell.

from nbkit import Notebook, code, md

NOTEBOOK = Notebook(
    chunk="C10",
    run=2,
    slug="eval",
    title="evaluation audio",
    branch="chunk/C10-eval",
    gpu=True,
    cells=[
        md("""
            **Corpus on this VM.** The reference clips (R) are the original corpus recordings
            of the 20 prompts, read from this VM's fast disk. `fetch-data` makes sure the corpus
            is here (it skips the download and unzip when they are done). Expect
            `PASS: dataset ready`.
        """),
        code("C10.pre", "Restore data", "!boinga fetch-data",
             expect=[r"PASS: dataset ready"]),
        md("""
            **Generate the A/B/R audio.** For each of the 20 reviewed prompts in
            `configs/eval_prompts.json`: the pretrained Bangla voice speaks it into Drive
            `eval/final/A_baseline/<id>.wav`, the fine-tuned voice (Drive `best/`, run1 step
            36500) into `B_boinga/`, both with the fixed seed and inference scales; the real
            corpus clip goes through the training audio transform (mono, model sample rate,
            edge silence trimmed) into `R_real/`. Loading the two models takes a little while,
            then a tqdm bar counts prompts. Every wav must be longer than 0.5 s and louder than
            -50 dBFS. A wav already listed in `eval/final/index.csv` with the same text is kept,
            so a re-run only fills gaps. Expect `PASS: generated A=20 B=20 R=20` and
            `reports/eval_generation.md` (count, durations and level per set) between
            BEGIN/END markers.
        """),
        code("C10.2", "Generate A/B/R audio", "!boinga eval-generate",
             expect=[r"PASS: generated A=20 B=20 R=20",
                     r"=== BEGIN reports/eval_generation\.md ==="]),
        md("""
            **Listen: 3 prompts side by side.** Plays P01-P03, each as A (baseline), B
            (fine-tuned) and R (real recording), labelled with the prompt text. This is a
            sanity check that the files are right, not the blind rating: the labels show which
            voice is which. The blind sheet for raters comes next (C10.3). Nothing is checked
            automatically.
        """),
        code("C10.2.play", "Listen: 3 prompts side by side", """
            from boinga_translator import colab_runtime, evaluate

            colab_runtime.play(*evaluate.players(("P01", "P02", "P03")))
        """, expect=[]),
        md("""
            **Build the blind rating sheet.** For each prompt, the baseline and fine-tuned
            wavs become `<id>_clip1.wav` / `<id>_clip2.wav` in Drive `eval/final/blind/`, in a
            fixed-seed random order that puts each voice first for 10 of the 20 prompts. Both
            clips are scaled to the same loudness (-23 dBFS RMS), so volume cannot give a voice
            away. The key (which clip is which voice) is saved as `eval/final/key.csv`,
            outside `blind/`, and never printed: only its sha256 is, so you stay blind too.
            `blind/` also gets `rating_sheet.csv` (40 empty rows: rater, prompt, clip, text, four
            1-5 scores, comments) and `INSTRUCTIONS.md` (English and Bangla), printed between
            BEGIN/END markers as `reports/rater_instructions.md`. A sheet that raters already
            edited is kept. Share the `blind/` folder with your raters. Expect
            `PASS: rating sheet 40 rows`.
        """),
        code("C10.3", "Build blind rating sheet", "!boinga eval-sheet",
             expect=[r"PASS: rating sheet 40 rows",
                     r"=== BEGIN reports/rater_instructions\.md ==="],
             forbid=[r"A_baseline", r"B_boinga"]),
    ],
)
